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
- Probes: `project/moss_te_sae/src/{copy_head_ext,copy_transplant,ref_edge_mask}.py`.

State: proposal; P0 (gate on our own LoRAs, inference only) is next.
