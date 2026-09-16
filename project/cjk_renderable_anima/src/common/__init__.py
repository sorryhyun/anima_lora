"""common — the plumbing three or more stages share (stage code lives in
``data/`` ``train/`` ``eval/`` ``scenes/``; the registry is ``stages.py``).

  paths        REPO, OUT, the corpus dirs, FONT_DIR, data_dir / arm_dir
  text         kana inventories, regexes, norm / lev / cer
  prompts      TPL_* templates, EVAL_GROUPS, NATIVE_PROMPTS / NATIVE_CLAUSES, EN_WORDS
  shapes       wh, parse_shape, parse_shapes
  models       checkpoints, generation requests, VAE, text encoding, DiT forward
  hooks        ExtDelta (the ext-row delta), AdapterLoRA, OutVec (the Q shift)
  readers      detector + two OCR readers, contact sheets
  bubble       bubble flood-mask geometry (mask / interior / bbox)
  render/flat  font layouts: find_fonts, sample_layout, render_string, crop_bubble
  render/scene the S-line compositor: JA text drawn into a generated scene's bubble

Importing any ``common.*`` module runs the bootstrap below: the repo root
(``library``) and the frozen line's ``ocr/`` dir (``pseudo_label``, read by
``readers``) go on ``sys.path`` right after ``src/``. After, not at position 0:
``src/`` holds generic top-level names (``train`` ``eval`` ``bench`` ``cli``)
and the repo root has ``train.py`` and ``bench/``.

Torch is imported only by the modules that need it, so ``--stage data``
stays CPU-only.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
_REPO = Path(__file__).resolve().parents[4]
_at = next(
    (i + 1 for i, p in enumerate(sys.path) if p and Path(p).resolve() == _SRC), 0
)
for _p in (_REPO, _REPO / "project" / "cjk_aware_anima_dit" / "ocr"):
    if str(_p) not in sys.path:
        sys.path.insert(_at, str(_p))
