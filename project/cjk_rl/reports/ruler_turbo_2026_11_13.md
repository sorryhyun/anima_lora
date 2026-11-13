# Ruler on the turbo policy (2026-11-13)

The evidence the line opened on: does the soup stack keep the asked string, and is the
4-step turbo a fair proxy of its 28-step teacher? Proposal: [`../proposal.md`](../proposal.md).

Dialogue ruler (`../../cjk_anima_reseed/criteria.md`), 64 of its 96 strings
(21 / 21 / 22 by bin), seed 0, jp_v1 rows (`seed_1008`, punct pack, routed) on all
three generators. The ruler gained `X#<model>` arms for this
(`src/eval/ruler/__init__.py` `MODELS`). Result:
`project/cjk_anima_reseed/results/20261113-1417-ruler-sensitive-turbo_m14v3_only64/`.

| | cer ↓ | exact | ≤ 2 edits | drawn | g_p | a_p | en_match |
|---|---|---|---|---|---|---|---|
| base, 28 st, cfg 4 ¹ | 0.667 | 4 | 8 | 35.5 | 0.23 | 0.29 | 0.57 |
| merged14, 28 st, cfg 4 | 0.811 | 3 | 4 | 58.1 | 0.12 | 0.07 | 0.64 |
| merged14 + v3, 4 st, cfg 1 | 0.849 | 1 | 3 | 45.9 | 0.10 | 0.02 | 0.72 |

¹ cached 10-08 renders on older code. The gaps below are far outside that session's
noise (mean |Δpx| 3.8 / 255).

- **base → m14 loses the string**: cer +0.14 (38 / 7, p 3e-6), drawn +23 (p 1e-6),
  a_p 0.29 → 0.07. On the sheets m14 letters several small bubbles of fluent
  invented dialogue. The asked string almost never fills one bubble; the best read
  is a 1–3 glyph fragment.
- **m14 → v3 is a fair proxy of its teacher**: cer n.s. (+0.04, p 0.14), 12 fewer
  invented glyphs (p 3e-4), kana recall −0.12 (p 4e-4). Its page sits closer to its
  own EN ref (en_match +0.08, p 0.03).
- So the rows, fitted on the base, lose the string on the soup's text path. RL on
  the soup policy re-fits them where they are used.
- exact and ≤ 2 edits are near 0 on every arm. A pass / fail reward gives a zero
  advantage in nearly every group, so the reward must be dense (proposal § 3.3).

Rendering took 0.3 s per 768²-area image for the policy against about 6.5 s for a
28-step render. Reads ran about 1.3 s per render (detector + OCR reader + VL16 + PE
page tokens).
