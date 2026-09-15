#!/usr/bin/env python
"""wake_probe — can an *address* alone wake JA glyph rendering in the frozen DiT?

Hypothesis (2026-09-13): Anima saw JA/KO/ZH glyph pixels in pretraining but the
T5 side collapsed every CJK caption to ``<unk>``, so the DiT learned the glyph
*texture* under a presence tag and never a character-level address. If discrete
glyph units exist in the weights, a trained ext row (the address) should steer a
frozen DiT to draw a specific kana. Nothing in the line has run a pixel loss on
the rows with the DiT frozen — JA-BODY had a target-stream LoRA and a whole-crop
inpaint loss; JA-SHIP had no trainable weight on the row → pixel path.

Stages (each one daemon job; ``--stage all`` = salad data train eval):

  salad   Probe 0 — base model, EN prompts asking for manga speech bubbles / signs;
          detector + two readers over the output: does the salad contain real,
          reader-agreed kana/kanji units?
  data    build the glyph set: font renders (Noto Sans/Serif CJK weights) of 1–3
          kana strings + kana-only corpus bubble crops; eval prompt sets.
          ``--balanced N`` renders N distinct strings per shared layout (W2a).
  train   frozen DiT, frozen Qwen; trainable = a delta on the ext rows the
          training captions touch (arm ``rows``), that + a LoRA on every
          Linear of ``llm_adapter.blocks`` (arm ``rows_adapter``), or a glyph
          encoder g(render of the piece) → row delta shared across every row
          (arm ``encoder``, W2d; ``--held_out N`` for the generalisation
          test). Plain rectified-flow loss on the glyph crops.
  classify  same-noise N-way diffusion classifier over the trained single kana
          (delta on vs off): which σ carries identity, do the rows discriminate.
  classify_str  the same over 2–3 kana strings: order / count / identity by σ.
  native  scene prompts (the blind-pairs set) + a kana clause, delta 0/1, read:
          does the address survive an ordinary prompt outside the template?
  scenes  S line: base model draws tag-prompted scenes with one EN-anchored
          speech bubble; detector + readers keep exactly-one-box images that
          read the anchor → scenes_<tag>/scenes.jsonl for the composite data.
  eval    T2I the eval set with the delta scaled 0 (floor) and 1 (trained),
          same seeds; read; CER vs the floor. An EN string set is the pipeline
          control (no training needed; the base reads Latin).

The stages live in ``stage/`` (one module each) and the shared plumbing in
``wake/`` (models, hooks, readers, renders, inventories, encoder); each
package's ``__init__.py`` has the map. Run record: ``../reports/README.md``.

    make daemon-run ARGS="--label wake-salad --stall-timeout 0 \
        project/cjk_renderable_anima/probes/wake_probe.py --stage salad"
    make daemon-run ARGS="--label wake-train-rows --stall-timeout 0 \
        project/cjk_renderable_anima/probes/wake_probe.py --stage data train eval --arm rows"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage import ALL, STAGES, run  # noqa: E402
from wake.cli import build_parser  # noqa: E402
from wake.common import OUT  # noqa: E402


def main():
    a = build_parser(STAGES, __doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in ALL if "all" in a.stage else a.stage:
        print(f"===== stage {s} ({a.arm})", flush=True)
        run(s, a)


if __name__ == "__main__":
    main()
