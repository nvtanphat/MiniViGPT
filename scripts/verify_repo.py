from __future__ import annotations

import argparse
import subprocess
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
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((repo / args.config).read_text(encoding="utf-8"))
    model_cfg = MiniViGPTConfig.from_dict(cfg["model"])
    model = MiniViGPT(model_cfg)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    print(f"[ok] config valid; parameters={model.num_parameters():,}")
    print(f"[ok] hidden_dim={model_cfg.hidden_dim}; head_dim={model_cfg.dim // model_cfg.n_heads}")

    if not args.skip_tests:
        result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=repo)
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
