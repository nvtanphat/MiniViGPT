from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running this script directly from a source checkout (no editable install).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml

from minivigpt.config import MiniViGPTConfig
from minivigpt.model import MiniViGPT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/minivigpt_20m.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = MiniViGPTConfig.from_dict(cfg["model"])
    model = MiniViGPT(model_cfg)
    n_params = model.num_parameters()
    train = cfg["training"]
    tokens_per_step = int(train["batch_size"]) * int(train["grad_accum_steps"]) * model_cfg.max_seq_len
    total_tokens = tokens_per_step * int(train["max_steps"])
    result = {
        "parameters": n_params,
        "parameters_millions": n_params / 1e6,
        "effective_sequences_per_update": int(train["batch_size"]) * int(train["grad_accum_steps"]),
        "tokens_per_update": tokens_per_step,
        "planned_updates": int(train["max_steps"]),
        "planned_training_token_presentations": total_tokens,
        "planned_tokens_per_parameter": total_tokens / n_params,
        "note": "Token presentations count repeated/random windows; this is not unique-corpus token count.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
