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
- [reports/](reports/) — one report per run or probe, indexed in its `README.md`.
- Probes: `project/moss_te_sae/src/{copy_head_ext,copy_transplant,ref_edge_mask,ref_gate,copy_vs_step}.py`; data `src/build_c_data.py`.
- [request.md](request.md) — request: isolate the text stream as the reference is (cross / intra read), soft tokens for what it loses. Only the render half of its first step is done, on t2i (`reports/t2i_rows_ablate.md`).

State: P0, P1 and P2's first pass done; C (`output/ckpt_moss_i2i/c_fill_s400`) + `cut` is a zero-shot sanitize, and C as init gives v3-like removal at 300 steps but not v3's tone. The Hangul block: v1 font data negative; v2 data (edition layout, balloons) gives a working Korean letterer from C in 500 steps. On the held-out ko10 pages v2 alone misspells far more than e2 and exact concat fails, so the block is not yet a swap_text part; e2's weak spot is long dense narration. The Hangul block is shelved (its data deleted 2026-11-08) until swap_text's Korean loss is fixed. **The current copy block is `output/ckpt_moss_i2i/c_fill_bw10_mod16_s800`** (2026-11-09, `reports/c_fill_mod16_bw10.md`): `mod_rank` 16, no tone term, B&W pairs at 10 %. Fine-tunes from it run with `mod_rank 16` (`init_lora` refuses its modulation keys otherwise). As a sanitize init it removes as C does with less residue; under `cut` alone it still letters large masks and lets masked text back.
