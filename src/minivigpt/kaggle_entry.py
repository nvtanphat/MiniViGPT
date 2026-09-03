from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def ensure_dependencies() -> None:
    packages = {
        "yaml": "PyYAML>=6.0",
        "datasets": "datasets>=3.0",
        "tokenizers": "tokenizers>=0.20",
        "tqdm": "tqdm>=4.66",
    }
    missing = [pkg for module, pkg in packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def find_resume_checkpoint(config: dict) -> Path | None:
    resume_cfg = config.get("resume", {})
    if not bool(resume_cfg.get("auto_from_kaggle_input", False)):
        return None
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return None
    pattern = str(resume_cfg.get("checkpoint_glob", "**/checkpoint_latest.pt"))
    matches = sorted(input_root.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        print("[resume] multiple checkpoints found; selecting lexicographically last:")
        for path in matches:
            print("  -", path)
    return matches[-1]


def main() -> None:
    ensure_dependencies()
    import yaml

    sys.path.insert(0, str(Path(__file__).parent.resolve()))
    from minivigpt.generate import generate_text
    from minivigpt.train import train

    # A Kaggle script kernel ships only the single code file, so the bundler
    # embeds config.yaml and writes it next to the unpacked package. Fall back
    # to a sibling file for plain local runs of this entry point.
    config_path = Path(globals().get("_config_path", Path(__file__).with_name("config.yaml")))
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resume = find_resume_checkpoint(config)
    if resume:
        print(f"[resume] auto-resuming from {resume}")

    output_dir = train(config_path, resume=str(resume) if resume else None)
    checkpoint = output_dir / "checkpoint_best.pt"
    tokenizer = output_dir / "tokenizer.json"

    if checkpoint.exists() and tokenizer.exists():
        sampling = config["sampling"]
        sample = generate_text(
            checkpoint=checkpoint,
            tokenizer_path=tokenizer,
            prompt=str(sampling["prompt"]),
            max_new_tokens=int(sampling["max_new_tokens"]),
            temperature=float(sampling["temperature"]),
            top_k=int(sampling["top_k"]),
            top_p=float(sampling.get("top_p", 0.95)),
        )
        (output_dir / "sample.txt").write_text(sample, encoding="utf-8")
        print("\n=== SAMPLE ===\n")
        print(sample)


if __name__ == "__main__":
    main()
