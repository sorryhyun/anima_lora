# lora_blocks — precise MOSS i2i edits from reusable parts

Opened 2026-11-07 out of `project/moss_te_sae` (the copy-head and reference-edge probes).
An i2i edit splits into **copy** (a trained protocol, reused across tasks), **gate** (where
the copy holds: a bias on the main-query → reference-key edges, set from a mask, no
parameters) and **draw** (a trained block per kind of content). Tasks are assembled from
these and fine-tuned briefly, instead of one curated pair set and one long run per task.

- [proposal.md](proposal.md) — the blocks, composition, phases P1–P5 and their gates, the code
  it needs.
- Evidence: [`../moss_te_sae/reports/copy_protocol.md`](../moss_te_sae/reports/copy_protocol.md)
  (the copy is a protocol, not a head; the edge gate), [`../moss_te_sae/reports/i2i_copy_head.md`](../moss_te_sae/reports/i2i_copy_head.md).
- [plan.md](plan.md) — the order of work: Korean strokes first (per-glyph read, colorize copy in the text boxes, C, L).
- [opinion.md](opinion.md) — a reading of the idea against the literature; open design points.
- [findings/](findings/) — conclusions and conjectures above the reports:
  [copy.md](findings/copy.md) (a copy block saves pairs and the 200–300-step phase change, not steps of a long run),
  [gate.md](findings/gate.md) (G fixes "where" on shipped LoRAs; `diag` inside for removal, `boost` outside for lettering, `hard` invents).
- [reports/copy_vs_step.md](reports/copy_vs_step.md) — the copy head forms at steps ~175–225 of any pair set, after the pixels already copy; training past it buys breadth.
- [reports/p0_gate.md](reports/p0_gate.md) — P0: the gate fixes both shipped LoRAs' "where" failures with no training; `diag1` inside AnimeText boxes removes the watermark sanitize v3 keeps, `boost` outside the layout boxes stops swap_text rewriting unasked regions; `hard` invents content.
- [reports/colorize_inboost.md](reports/colorize_inboost.md) — a copy boost inside colorize v5's text boxes makes strokes heavier and leaves every deformed glyph as it was; the gate cannot fix lettering.
- [reports/p1_c_fill.md](reports/p1_c_fill.md) — P1: C (400 steps, clean illustrations, text-layout holes under `ref_gate cut`) removes the sanitize eval pages' stamps and watermark under `cut` with no sanitize pair; the copy head forms by step 200.
- Probes: `project/moss_te_sae/src/{copy_head_ext,copy_transplant,ref_edge_mask,ref_gate,copy_vs_step}.py`; data `src/build_c_data.py`.

State: P0 and P1 done; C (`output/ckpt_moss_i2i/c_fill_s400`) + `cut` is a zero-shot sanitize. Next: P2's fine-tune from C, or the Hangul block on C (plan.md).
