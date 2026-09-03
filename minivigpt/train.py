from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .config import MiniViGPTConfig
from .data import BinaryTokenDataset, iter_hf_texts, write_token_binary
from .model import MiniViGPT




def load_checkpoint_file(path: str | Path, map_location):
    """Load our full-state checkpoint. Only load checkpoints you trust.

    MiniViGPT checkpoints contain optimizer/RNG/tokenizer metadata, so PyTorch
    weights-only loading is insufficient on newer PyTorch releases.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_perplexity(loss: float) -> float:
    """exp(loss) without raising OverflowError on a diverged/very early loss."""
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def learning_rate(step: int, cfg: dict) -> float:
    base_lr = float(cfg["learning_rate"])
    min_lr = float(cfg["min_lr"])
    warmup = int(cfg["warmup_steps"])
    max_steps = int(cfg["max_steps"])
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


def resolve_precision(device: torch.device, requested: str) -> tuple[str, torch.dtype | None]:
    requested = requested.lower()
    if requested not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision must be one of: auto, fp32, fp16, bf16")
    if device.type != "cuda":
        return "fp32", None
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but this CUDA device does not support bf16")
    if requested == "fp32":
        return "fp32", None
    return requested, torch.bfloat16 if requested == "bf16" else torch.float16


def make_grad_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def configure_optimizer(
    model: MiniViGPT,
    cfg: dict,
    device: torch.device,
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Matmul/embedding matrices get AdamW decay; RMSNorm vectors do not.
        (decay if param.ndim >= 2 else no_decay).append(param)
    groups = [
        {"params": decay, "weight_decay": float(cfg["weight_decay"])},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    kwargs = {}
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(
        groups,
        lr=float(cfg["learning_rate"]),
        betas=(float(cfg["beta1"]), float(cfg["beta2"])),
        eps=float(cfg.get("adam_eps", 1e-8)),
        **kwargs,
    )


def _stream_kwargs(config: dict, split_seed: int) -> dict:
    ds = config["dataset"]
    return {
        "revision": ds.get("revision"),
        "min_quality_score": ds.get("min_quality_score"),
        "quality_column": ds.get("quality_column", "quality_score"),
        "category_column": ds.get("category_column", "main_category"),
        "exclude_categories_contains": ds.get("exclude_categories_contains", []),
        "shuffle_buffer": int(ds.get("shuffle_buffer", 0)),
        "seed": split_seed,
        "deduplicate_exact": bool(ds.get("deduplicate_exact", True)),
    }


def _build_split(
    config: dict,
    tokenizer,
    split_name: str,
    max_articles: int,
    output_path: Path,
    seed: int,
) -> dict:
    ds = config["dataset"]
    texts = iter_hf_texts(
        ds["name"],
        split_name,
        ds["text_column"],
        max_articles,
        int(ds["min_chars"]),
        **_stream_kwargs(config, seed),
    )
    return write_token_binary(
        tokenizer,
        texts,
        output_path,
        bool(ds.get("append_eos", True)),
    )


def _restore_tokenizer_from_checkpoint(resume: str | None, tokenizer_path: Path) -> None:
    if not resume or tokenizer_path.exists():
        return
    resume_path = Path(resume)
    if not resume_path.exists():
        return
    checkpoint = load_checkpoint_file(resume_path, map_location="cpu")
    tokenizer_json = checkpoint.get("tokenizer_json")
    if isinstance(tokenizer_json, str) and tokenizer_json.strip():
        tokenizer_path.write_text(tokenizer_json, encoding="utf-8")
        print(f"[resume] restored tokenizer from {resume_path.name}")


def _adopt_prepared_data(config: dict, output_dir: Path) -> None:
    """Reuse an already-tokenized corpus instead of rebuilding it.

    `dataset.prepared_dir` (or a directory under /kaggle/input holding train.bin)
    lets a Kaggle run skip the ~30-minute stream+tokenize step. Files are copied
    into output_dir so the rest of the pipeline is unchanged and the run stays
    self-contained; a split is only adopted when its sidecar .json is present too.
    """
    ds = config.get("dataset", {})
    candidates: list[Path] = []
    explicit = ds.get("prepared_dir")
    if explicit:
        candidates.append(Path(explicit))
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(sorted(p.parent for p in input_root.glob("*/train.bin")))

    splits = ("train.bin", "validation.bin", "test.bin")

    def _pairs(base: Path) -> list[tuple[Path, Path]]:
        """Every file a complete corpus needs, as (source, destination)."""
        items = [(base / "tokenizer.json", output_dir / "tokenizer.json")]
        for name in splits:
            items.append((base / name, output_dir / name))
            meta = name + ".json"
            items.append((base / meta, output_dir / meta))
        return items

    for source in candidates:
        if not source.is_dir() or source.resolve() == output_dir.resolve():
            continue
        pairs = _pairs(source)
        # Adopt a source only if it can supply a complete corpus. A partial copy
        # would leave a .bin without its sidecar .json and silently fall through
        # to re-streaming the dataset.
        if not all(src.exists() for src, _ in pairs):
            continue
        if all(dst.exists() for _, dst in pairs):
            print(f"[data] prepared corpus already present in {output_dir}")
            return
        for src, dst in pairs:
            if not dst.exists():
                shutil.copy2(src, dst)
        print(f"[data] reusing prepared corpus from {source}")
        return


def prepare_data(config: dict, output_dir: Path, resume: str | None = None) -> tuple[Path, Path, Path, Path]:
    from .tokenizer import load_tokenizer, train_bpe_tokenizer
    ds = config["dataset"]
    tok_cfg = config["tokenizer"]
    base_seed = int(config.get("seed", 42))
    tokenizer_path = output_dir / "tokenizer.json"
    train_bin = output_dir / "train.bin"
    val_bin = output_dir / "validation.bin"
    test_bin = output_dir / "test.bin"

    _adopt_prepared_data(config, output_dir)
    _restore_tokenizer_from_checkpoint(resume, tokenizer_path)

    if not tokenizer_path.exists():
        print("[data] Training Vietnamese byte-level BPE tokenizer from scratch...")
        texts = iter_hf_texts(
            ds["name"],
            ds["train_split"],
            ds["text_column"],
            int(ds["tokenizer_articles"]),
            int(ds["min_chars"]),
            **_stream_kwargs(config, base_seed + 101),
        )
        train_bpe_tokenizer(
            texts=texts,
            output_path=tokenizer_path,
            vocab_size=int(tok_cfg["vocab_size"]),
            min_frequency=int(tok_cfg["min_frequency"]),
            special_tokens=list(tok_cfg["special_tokens"]),
        )

    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.get_vocab_size() != int(config["model"]["vocab_size"]):
        raise ValueError(
            f"Tokenizer vocab ({tokenizer.get_vocab_size()}) does not match model vocab "
            f"({config['model']['vocab_size']})."
        )

    if not train_bin.exists():
        print("[data] Building train token stream...")
        meta = _build_split(
            config, tokenizer, ds["train_split"], int(ds["train_articles"]),
            train_bin, base_seed + 202,
        )
        print("[data] train:", meta)

    if not val_bin.exists():
        print("[data] Building validation token stream...")
        meta = _build_split(
            config, tokenizer, ds["validation_split"], int(ds["validation_articles"]),
            val_bin, base_seed + 303,
        )
        print("[data] validation:", meta)

    if not test_bin.exists():
        print("[data] Building test token stream...")
        meta = _build_split(
            config, tokenizer, ds["test_split"], int(ds["test_articles"]),
            test_bin, base_seed + 404,
        )
        print("[data] test:", meta)

    manifest = {
        "dataset": ds,
        "tokenizer": tok_cfg,
        "files": {
            name: json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
            for name, path in {"train": train_bin, "validation": val_bin, "test": test_bin}.items()
        },
    }
    (output_dir / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return tokenizer_path, train_bin, val_bin, test_bin


@torch.no_grad()
def estimate_loss(
    model: MiniViGPT,
    dataset: BinaryTokenDataset,
    batch_size: int,
    eval_batches: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    *,
    eval_seed: int,
) -> float:
    """Deterministic fixed-sample validation/test loss."""
    was_training = model.training
    model.eval()
    losses = []
    generator = torch.Generator(device="cpu").manual_seed(eval_seed)
    for _ in range(eval_batches):
        x, y = dataset.get_batch(batch_size, device, generator=generator)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_dtype is not None,
        ):
            _, loss = model(x, y)
        assert loss is not None
        losses.append(float(loss.item()))
    model.train(was_training)
    return sum(losses) / len(losses)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: MiniViGPT,
    optimizer: torch.optim.Optimizer,
    scaler,
    step: int,
    best_val: float,
    raw_config: dict,
    tokenizer_path: Path,
) -> None:
    state = {
        "format_version": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_val": best_val,
        "config": raw_config,
        "rng_state": capture_rng_state(),
        # Makes a checkpoint self-contained with respect to the token-ID mapping.
        "tokenizer_json": tokenizer_path.read_text(encoding="utf-8"),
    }
    torch.save(state, path)


def _validate_resume_config(checkpoint: dict, model_cfg: MiniViGPTConfig) -> None:
    old_cfg = MiniViGPTConfig.from_dict(checkpoint["config"]["model"])
    if old_cfg != model_cfg:
        raise ValueError(
            "Resume checkpoint model config does not match current model config. "
            "Use the checkpoint's original architecture."
        )


def train(config_path: str | Path, resume: str | None = None) -> Path:
    raw_config = load_yaml(config_path)
    seed = int(raw_config.get("seed", 42))
    set_seed(seed)

    output_dir = Path(raw_config["output_dir"])
    if str(output_dir).startswith("/kaggle/working") and not Path("/kaggle/working").exists():
        output_dir = Path("artifacts/local")
    output_dir.mkdir(parents=True, exist_ok=True)
    # The Kaggle bootstrap materializes config.yaml inside the working dir, so the
    # archive copy can be the very same file; copying it onto itself would fail.
    config_copy = output_dir / "config.yaml"
    if Path(config_path).resolve() != config_copy.resolve():
        shutil.copy2(config_path, config_copy)

    tokenizer_path, train_bin, val_bin, test_bin = prepare_data(raw_config, output_dir, resume=resume)
    model_cfg = MiniViGPTConfig.from_dict(raw_config["model"])
    train_cfg = raw_config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision, amp_dtype = resolve_precision(device, str(train_cfg.get("precision", "auto")))

    model = MiniViGPT(model_cfg).to(device)
    n_params = model.num_parameters()
    print(f"[model] parameters: {n_params:,}")
    print(f"[device] {device}; precision={precision}")

    optimizer = configure_optimizer(model, train_cfg, device)
    scaler = make_grad_scaler(device, precision)
    start_step = 0
    best_val = float("inf")

    if resume:
        resume_path = Path(resume)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        print(f"[resume] loading {resume_path}")
        checkpoint = load_checkpoint_file(resume_path, map_location=device)
        _validate_resume_config(checkpoint, model_cfg)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_step = int(checkpoint["step"]) + 1
        best_val = float(checkpoint.get("best_val", best_val))
        # Restore after data/model construction so resumed sampling matches the saved trajectory.
        restore_rng_state(checkpoint.get("rng_state"))

    if bool(train_cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    train_data = BinaryTokenDataset(train_bin, model_cfg.max_seq_len)
    val_data = BinaryTokenDataset(val_bin, model_cfg.max_seq_len)
    test_data = BinaryTokenDataset(test_bin, model_cfg.max_seq_len)
    batch_size = int(train_cfg["batch_size"])
    grad_accum = int(train_cfg["grad_accum_steps"])
    max_steps = int(train_cfg["max_steps"])
    tokens_per_step = batch_size * grad_accum * model_cfg.max_seq_len
    metrics_path = output_dir / "metrics.jsonl"

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    last_grad_norm = float("nan")

    for step in range(start_step, max_steps):
        lr = learning_rate(step, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        running_loss = 0.0
        for _ in range(grad_accum):
            x, y = train_data.get_batch(batch_size, device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                _, loss = model(x, y)
                assert loss is not None
                scaled_loss = loss / grad_accum
            scaler.scale(scaled_loss).backward()
            running_loss += float(loss.item()) / grad_accum

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(train_cfg["grad_clip"])
        )
        last_grad_norm = float(grad_norm.item())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % int(train_cfg["log_interval"]) == 0:
            elapsed = time.time() - started
            tokens_seen = (step + 1) * tokens_per_step
            record = {
                "step": step,
                "train_loss": running_loss,
                "lr": lr,
                "grad_norm_preclip": last_grad_norm,
                "tokens_seen": tokens_seen,
                "tokens_per_parameter": tokens_seen / n_params,
                "elapsed_sec": elapsed,
            }
            print(json.dumps(record, ensure_ascii=False))
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        do_eval = step % int(train_cfg["eval_interval"]) == 0 or step == max_steps - 1
        if do_eval:
            val_loss = estimate_loss(
                model,
                val_data,
                batch_size,
                int(train_cfg["eval_batches"]),
                device,
                amp_dtype,
                eval_seed=int(train_cfg.get("eval_seed", 12345)),
            )
            ppl = safe_perplexity(val_loss)
            record = {"step": step, "val_loss": val_loss, "perplexity": ppl}
            print("[eval]", json.dumps(record, ensure_ascii=False))
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    output_dir / "checkpoint_best.pt",
                    target_model,
                    optimizer,
                    scaler,
                    step,
                    best_val,
                    raw_config,
                    tokenizer_path,
                )

        if step % int(train_cfg["save_interval"]) == 0 or step == max_steps - 1:
            target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(
                output_dir / "checkpoint_latest.pt",
                target_model,
                optimizer,
                scaler,
                step,
                best_val,
                raw_config,
                tokenizer_path,
            )

    # Report test exactly once from the best validation checkpoint.
    best_path = output_dir / "checkpoint_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"No best checkpoint at {best_path}: no training/eval step ran "
            f"(start_step={start_step}, max_steps={max_steps})."
        )
    best_checkpoint = load_checkpoint_file(best_path, map_location=device)
    target_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    target_model.load_state_dict(best_checkpoint["model"])
    test_loss = estimate_loss(
        target_model,
        test_data,
        batch_size,
        int(train_cfg.get("test_batches", train_cfg["eval_batches"])),
        device,
        amp_dtype,
        eval_seed=int(train_cfg.get("test_seed", 54321)),
    )
    summary = {
        "best_step": int(best_checkpoint["step"]),
        "best_val_loss": float(best_checkpoint["best_val"]),
        "best_val_perplexity": safe_perplexity(float(best_checkpoint["best_val"])),
        "test_loss": test_loss,
        "test_perplexity": safe_perplexity(test_loss),
        "parameters": n_params,
        "tokens_per_step": tokens_per_step,
        "planned_training_tokens": max_steps * tokens_per_step,
        "planned_tokens_per_parameter": (max_steps * tokens_per_step) / n_params,
        "precision": precision,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[test]", json.dumps(summary, ensure_ascii=False))
    print(f"[done] outputs: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    train(args.config, args.resume)


if __name__ == "__main__":
    main()
