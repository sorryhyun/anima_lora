# clause_hw — position clauses on RoPE's h / w axes

On by default in the t2i LoRA recipe (`clause_hw = 1.0`, `configs/moss/lora.toml`). Training only:
generation runs without it.

MOSS puts every text token at RoPE `(h, w) = (0, 0)`. With `clause_hw`, the tokens of each position
clause (`On the left, black hair, …`) take the centre of the clause's 3×3 cell instead, times
`clause_hw` (1 = a third of the grid side per cell). Every other token and the `f` axis are unchanged.
After SpaG-DiT's CDPE; design in [../proposal/cdpe_text_grounding.md](../proposal/cdpe_text_grounding.md).
Code: `library/moss/training/grounding.py`; the value is stamped in the LoRA metadata.

## When it acts

- Only on captions that carry position clauses (`anime_tools.captions.position_clauses`). A `tags` or
  `text` cache with clauses gets it on its full captions. On a `layout_json` cache the full captions are
  JSON, so only the plain-caption steps (`plain_caption_rate`) can move.
- A cache with no clause in any full caption prints a notice and trains as before.
- It refuses `distill`, `preserve_weight`, `trigger_registers` and a `token_split` cache. The GUI's
  self-distill and trigger-register recipes reset it to 0.

## Evidence (2026-11-16)

| Set | Arms | Read |
|---|---|---|
| ama mitsuki, `tags` cache, 8 epochs | `amathin` vs its same-config retrain `amathinr` vs `amahw` (clause_hw 1); 8 seeds × 4 general tag prompts + 4 clause prompts | hw judged better on colour, tone, shape and material (a small margin); left / right placement already correct in every arm |
| caststation casbox, 480 steps | control, retrain, clause_hw 0.25 / 1.0, clause_hw 1 + `text_box_hw` 1 | general renders not different from the retrain; one Miku text-logo effect at 1.0 that did not survive adding text boxes |

The ama read set the default. Renders: `output/tests/ama_clause/`; configs: `_archive/configs/pe_lora/`.
