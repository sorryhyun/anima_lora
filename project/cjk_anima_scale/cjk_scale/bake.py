"""bake — a stage table → a vocab pack pair, through the repo's baker
(``scripts/toolkits/bake_vocab_pack.py``), which folds ``trained.pt`` into
the pack the DiT trained against and stamps the provenance."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .paths import REPO, arm_dir


def bake(stage: str, tag: str, out: str | None = None) -> Path:
    arm = arm_dir(stage, tag)
    assert (arm / "trained.pt").exists(), f"no table at {arm / 'trained.pt'}"
    dest = (
        Path(out)
        if out
        else REPO / "models" / "vocab_packs" / f"anima_cjk_vocab_pack_{stage}_{tag}"
    )
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "toolkits" / "bake_vocab_pack.py"),
        str(arm),
        "--out",
        str(dest),
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)
    return dest
