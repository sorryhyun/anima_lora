"""bake — a run's table → a vocab pack pair, through the repo's baker
(``scripts/toolkits/bake_vocab_pack.py``), which folds ``trained.pt`` into
the pack the DiT trained against and stamps the provenance. Not a
``scale.py`` verb: ``python -c "from cjk_scale.bake import bake; bake('<run>')"``
from the line dir."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .paths import REPO, run_dir, table_path


def bake(run: str, out: str | None = None) -> Path:
    assert table_path(run).exists(), f"no table at {table_path(run)}"
    dest = (
        Path(out)
        if out
        else REPO / "models" / "vocab_packs" / f"anima_cjk_vocab_pack_{run}"
    )
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "toolkits" / "bake_vocab_pack.py"),
        str(run_dir(run)),
        "--out",
        str(dest),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)
    return dest
