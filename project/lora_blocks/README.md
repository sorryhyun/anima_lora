# lora_blocks — precise MOSS i2i edits from reusable parts

Opened 2026-11-07 out of `project/moss_te_sae` (the copy-head and reference-edge probes).
An i2i edit splits into **copy** (a trained protocol, reused across tasks), **gate** (where
the copy holds: a bias on the main-query → reference-key edges, set from a mask, no
parameters) and **draw** (a trained block per kind of content). Tasks are assembled from
these and fine-tuned briefly, instead of one curated pair set and one long run per task.

- [proposal.md](proposal.md) — the blocks, composition, phases P0–P5 and their gates, the code
  it needs.
- Evidence: [`../moss_te_sae/reports/copy_protocol.md`](../moss_te_sae/reports/copy_protocol.md)
  (the copy is a protocol, not a head; the edge gate), [`../moss_te_sae/reports/i2i_copy_head.md`](../moss_te_sae/reports/i2i_copy_head.md).
- [opinion.md](opinion.md) — a reading of the idea against the literature; the checkpoint read.
- [reports/copy_vs_step.md](reports/copy_vs_step.md) — the copy head forms at steps ~175–225 of any pair set, after the pixels already copy; training past it buys breadth.
- [reports/p0_gate.md](reports/p0_gate.md) — P0: the gate fixes both shipped LoRAs' "where" failures with no training; `diag1` inside AnimeText boxes removes the watermark sanitize v3 keeps, `boost` outside the layout boxes stops swap_text rewriting unasked regions; `hard` invents content.
- Probes: `project/moss_te_sae/src/{copy_head_ext,copy_transplant,ref_edge_mask,ref_gate}.py`.

State: P0 done (p0_gate.md); whether `boost1` keeps `boost2`'s correction, and an own-cell boost, are the open reads.
