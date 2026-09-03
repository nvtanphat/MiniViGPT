"""Guard the generated Kaggle script: it must actually parse and bootstrap.

Kernel v2 failed in production with "from __future__ imports must occur at the
beginning of the file" because the bootstrap prelude was prepended ahead of the
entry module's __future__ import. Grepping the bundle for symbols did not catch
it -- only compiling the emitted file does.
"""

import base64
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "build_kaggle_bundle.py"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    dist = REPO / "dist" / "kaggle"
    if not (dist / "train.py").exists():
        pytest.skip("no built bundle at dist/kaggle")
    return dist


def test_generated_script_compiles(bundle):
    source = (bundle / "train.py").read_text(encoding="utf-8")
    compile(source, "train.py", "exec")


def test_future_import_is_first_statement(bundle):
    lines = (bundle / "train.py").read_text(encoding="utf-8").splitlines()
    code = [l for l in lines if l.strip() and not l.lstrip().startswith("#")]
    future = [i for i, l in enumerate(code) if l.startswith("from __future__ import")]
    if future:
        assert future[0] == 0, "__future__ import must be the first statement"


def test_embedded_package_modules_compile(bundle):
    source = (bundle / "train.py").read_text(encoding="utf-8")
    b64 = re.search(r'_EMBEDDED_MINIVIGPT_ZIP = """(.*?)"""', source, re.S)
    assert b64, "bootstrap payload missing from generated script"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(b64.group(1)))) as zf:
        names = [n for n in zf.namelist() if n.endswith(".py")]
        assert "minivigpt/train.py" in names
        for name in names:
            compile(zf.read(name).decode("utf-8"), name, "exec")


def test_bootstrap_unpacks_and_imports(bundle, tmp_path):
    """Execute the real prelude in a clean dir and import the package it unpacks."""
    script = tmp_path / "train.py"
    script.write_text((bundle / "train.py").read_text(encoding="utf-8"), encoding="utf-8")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import pathlib\n"
        "src = pathlib.Path('train.py').read_text(encoding='utf-8')\n"
        "cut = src.index('def ensure_dependencies')\n"
        "exec(compile(src[:cut], 'train.py', 'exec'),"
        " {'__file__': str(pathlib.Path('train.py').resolve())})\n"
        "from minivigpt.train import train\n"
        "from minivigpt.model import MiniViGPT\n"
        "print('OK')\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "probe.py"], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
