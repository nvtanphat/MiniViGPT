from __future__ import annotations

import argparse
from pathlib import Path

# Allow running this script directly from a source checkout (no editable install).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
import sys

import yaml

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minivigpt.train import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dataset and prepare token streams.")
    parser.add_argument("--config", default="configs/minivigpt_20m.yaml", help="Path to config file")
    parser.add_argument("--output-dir", default="artifacts/local", help="Directory to save output binaries and tokenizer")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg_path = repo / args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] Preparing dataset using config: {args.config}")
    print(f"[data] Output directory: {output_dir.resolve()}")

    tokenizer_path, train_bin, val_bin, test_bin = prepare_data(config, output_dir)
    print("\n[data] Dataset preparation completed successfully!")
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Train binary: {train_bin}")
    print(f"  Validation binary: {val_bin}")
    print(f"  Test binary: {test_bin}")


if __name__ == "__main__":
    main()
