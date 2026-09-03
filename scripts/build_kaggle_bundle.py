from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--slug", default="minivigpt-from-scratch")
    parser.add_argument("--title", default=None)
    parser.add_argument("--machine-shape", default="NvidiaTeslaT4")
    parser.add_argument("--config", default="configs/minivigpt_20m.yaml")
    parser.add_argument(
        "--dataset-source",
        action="append",
        default=[],
        help="Optional Kaggle dataset source, repeatable. Useful for attaching resume checkpoints.",
    )
    args = parser.parse_args()

    title = args.title or args.slug.replace("-", " ").replace("_", " ").title()

    repo = Path(__file__).resolve().parents[1]
    dist = repo / "dist" / "kaggle"
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True)

    config_source = repo / args.config
    if not config_source.exists():
        raise FileNotFoundError(config_source)
    config_b64 = base64.b64encode(
        config_source.read_bytes()
    ).decode("ascii")

    minivigpt_dir = repo / "src" / "minivigpt"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in minivigpt_dir.rglob("*.py"):
            rel = p.relative_to(minivigpt_dir.parent)
            zf.write(p, rel)
    b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

    bootstrap_code = f'''import base64
import io
import sys
import tempfile
import zipfile
from pathlib import Path

_EMBEDDED_MINIVIGPT_ZIP = """{b64_str}"""


def _minivigpt_pkg_dir() -> Path:
    """Pick a writable place to unpack into.

    Kaggle runs the script from /kaggle/src, which is read-only, so unpacking
    next to __file__ fails with EROFS. Prefer /kaggle/working, then the script
    directory (plain local runs), then a temp dir.
    """
    candidates = []
    kaggle_working = Path("/kaggle/working")
    # Only trust the Kaggle path when we are actually on Kaggle; on Windows
    # "/kaggle/working" resolves to a drive-relative path that would happily
    # be created, scattering junk outside the project.
    if kaggle_working.is_dir():
        candidates.append(kaggle_working)
    candidates.append(Path(__file__).parent.resolve())
    for base in candidates:
        try:
            base.mkdir(parents=True, exist_ok=True)
            probe = base / ".minivigpt_write_test"
            probe.touch()
            probe.unlink()
            return base / "_pkg"
        except OSError:
            continue
    return Path(tempfile.mkdtemp(prefix="minivigpt_")) / "_pkg"


_pkg_dir = _minivigpt_pkg_dir()
if not (_pkg_dir / "minivigpt").exists():
    _pkg_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(_EMBEDDED_MINIVIGPT_ZIP))) as _zf:
        _zf.extractall(_pkg_dir)
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))
print(f"[bootstrap] minivigpt unpacked to {{_pkg_dir}}")

# A Kaggle script kernel ships only code_file, so config.yaml cannot sit beside
# the script. Embed it and materialize it somewhere writable instead.
_EMBEDDED_CONFIG_YAML = """{config_b64}"""
_config_path = _pkg_dir.parent / "config.yaml"
_config_path.write_bytes(base64.b64decode(_EMBEDDED_CONFIG_YAML))
print(f"[bootstrap] config written to {{_config_path}}")

'''
    entry_content = (repo / "src" / "minivigpt" / "kaggle_entry.py").read_text(encoding="utf-8")
    # `from __future__` must stay the first statement of the generated script, so hoist
    # those lines above the bootstrap prelude instead of letting it push them down.
    future_lines, body_lines = [], []
    for line in entry_content.splitlines(keepends=True):
        (future_lines if line.startswith("from __future__ import") else body_lines).append(line)
    script = "".join(future_lines) + bootstrap_code + "".join(body_lines)
    (dist / "train.py").write_text(script, encoding="utf-8")

    shutil.copy2(config_source, dist / "config.yaml")
    shutil.copytree(minivigpt_dir, dist / "minivigpt")

    metadata = {
        "id": f"{args.username}/{args.slug}",
        "title": title,
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": args.machine_shape,
        "dataset_sources": args.dataset_source,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (dist / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Kaggle bundle created at: {dist}")
    print(f"Kernel: {metadata['id']}")
    if args.dataset_source:
        print("Attached dataset sources:", ", ".join(args.dataset_source))


if __name__ == "__main__":
    main()
