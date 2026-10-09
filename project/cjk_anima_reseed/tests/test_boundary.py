"""reseed stands alone: no live file imports the scale line's ``cjk_scale``
or names its line dir (its outputs, ``output/cjk_anima_scale``, are read), and
the table is whole."""

import ast
import re
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))

# a mention of the scale line that is not its output root
SCALE_DIR = re.compile(r'(?<!output/)(?<!"output" / ")cjk_anima_scale')


def _live() -> list:
    return [
        *HOME.glob("*.py"),
        *(HOME / "reseed").glob("*.py"),
        *(HOME / "probes").glob("*.py"),
        *(HOME / "src").rglob("*.py"),
    ]


def _imports(f: Path) -> set:
    out = set()
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
    return out


def test_no_scale_imports():
    for f in _live():
        hit = {m for m in _imports(f) if m.split(".")[0] == "cjk_scale"}
        assert not hit, f"{f.relative_to(HOME)} imports {sorted(hit)}"


def test_no_scale_dir():
    for f in _live():
        for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            assert not SCALE_DIR.search(ln), f"{f.relative_to(HOME)}:{i}: {ln.strip()}"


def test_table():
    from reseed.recipes import RECIPES
    from reseed.table import TABLE

    names = [t.name for t in TABLE]
    assert len(set(names)) == len(names)
    assert abs(sum(t.share for t in TABLE) - 1.5) < 1e-9
    for t in TABLE:
        assert t.recipe in RECIPES, t.name
        lo, hi = t.band
        assert 0 <= lo < hi <= 0.9, t.name
