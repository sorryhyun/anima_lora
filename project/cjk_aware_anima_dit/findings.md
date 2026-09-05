# CJK-aware Anima, DiT side — findings

Settled verdicts of this line, one entry per phase, evidence pointer beside
each. The encoder-side verdicts it builds on are in
[`../cjk_aware_anima/findings.md`](../cjk_aware_anima/findings.md) (read-only).

## D0 — ISO1 vs C9 direct blind set: flat (2026-09-05)

`reports/blind_s13_ISO1_vs_C9.md` (in the old line's `reports/`): 48 pairs,
16 v2 rows × seeds 6/7/8, both arms fresh to the grader. **ISO1 23 – C9 20,
tie 5; rows 6-6 (tie 4); p 0.76.** The isotropic table and the trained
r256 pack are indistinguishable for unmask training on this grid.

- The transitivity claim ISO1 ≈ HOT > C9 (s12 + s11) does **not** survive
  the direct test; transitivity has now failed twice in this protocol
  (s03/s04, s12/s13). Do not chain blind sets — pair the arms you want to
  compare.
- Pooled s01–s13: a content-free table is never worse than the trained pack
  for the OCR route, and rows must exist (C9 > P). The isotropic block is
  therefore the OCR-route default on **cost** grounds (seed-generated,
  deterministic, no distill), not on quality grounds. The hypothesis doc's
  "structured low-rank spread hurts" mechanism is weakened, not confirmed.
- Mechanism note carried from `bench/frontload_text_boost`: `k_norm`
  strips row scale on the K path, which is why HOT (norm ×5) ≈ ISO1 (s12).

Gate outcome: **proceed to D1** with the isotropic block for 「…」 spans;
bare CJK tags keep the trained rows (plan principle 2).
