#!/usr/bin/env python3
"""Write a DiT + vocab pack as one merged checkpoint for ComfyUI's
``AnimaMergedLoader``.

Thin CLI shell over ``library.anima.vocab_pack.merge_pack_into_dit`` — see it
for the file layout. anima_lora itself keeps loading the pack through
``vocab_pack`` and ignores the rows in a merged file.

    .venv/bin/python scripts/toolkits/merge_vocab_pack.py \\
        --pack models/vocab_packs/anima_cjk_vocab_pack_jp_v1 \\
        --out output/anima_jp_extended.safetensors
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from library import downloads as DL  # noqa: E402
from library.anima.vocab_pack import merge_pack_into_dit, read_merged_pack  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge a vocab pack into a DiT checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dit",
        type=Path,
        default=DL.models_dir() / "diffusion_models" / DL.ANIMA_DIT_FILE,
        help="Base DiT safetensors.",
    )
    parser.add_argument(
        "--pack", required=True, help="Vocab pack prefix (or either file / its dir)."
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = resolve_under_home(args.out)
    if out.exists():
        parser.error(f"{out} exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    digest = merge_pack_into_dit(resolve_under_home(args.dit), args.pack, out)
    _, _, read_back = read_merged_pack(out)
    if read_back != digest:
        logger.error("read-back digest %s… != %s…", read_back[:12], digest[:12])
        return 1
    logger.info("read back OK (sha %s…)", digest[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
