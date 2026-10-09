"""The vendored ``src/`` and fonts: what resolves where after ``bootstrap()``,
and (until the scale line moves to ``finished/``) the copy is scale's bar the
path plumbing."""

import json
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
SRC = HOME / "src"
SCALE_SRC = HOME.parent / "cjk_anima_scale" / "src"
# the files the vendoring edits (proposal_refactor.md § 2's path plumbing)
PLUMBING = {"common/paths.py", "common/prompts.py", "eval/__init__.py"}


def _vendored() -> list:
    return sorted(str(f.relative_to(SRC)) for f in SRC.rglob("*.py"))


def test_byte_equal_to_scale():
    # one-time: deleted when the scale line moves (proposal_refactor.md § 4 step 5)
    for rel in _vendored():
        if rel in PLUMBING:
            continue
        assert (SRC / rel).read_bytes() == (SCALE_SRC / rel).read_bytes(), rel


def test_resolution():
    code = f"""
import json, sys
sys.path.insert(0, {str(HOME)!r})
from reseed import bootstrap
bootstrap()
import common.paths, common.render.flat, data.stage, eval.enref, train.stage
print(json.dumps({{
    m: sys.modules[m].__file__
    for m in ("train", "train.stage", "common", "data.stage", "eval.enref")
}} | {{"OUT": str(common.paths.OUT), "FONT_DIR": str(common.render.flat.FONT_DIR)}}))
"""
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    got = json.loads(r.stdout.strip().splitlines()[-1])
    for m in ("train", "train.stage", "common", "data.stage", "eval.enref"):
        assert Path(got[m]).is_relative_to(SRC), (m, got[m])
    repo = HOME.parents[1]
    assert Path(got["OUT"]) == repo / "output" / "cjk_anima_scale"
    assert Path(got["FONT_DIR"]) == HOME / "assets" / "fonts"


def test_fonts():
    fonts = HOME / "assets" / "fonts"
    faces = [*fonts.glob("*.ttf"), *fonts.glob("*.otf")]
    kozh = [*(fonts / "kozh").glob("*.ttf"), *(fonts / "kozh").glob("*.otf")]
    assert faces, f"no faces in {fonts} (binaries are gitignored: copy them)"
    assert kozh, f"no faces in {fonts / 'kozh'}"
