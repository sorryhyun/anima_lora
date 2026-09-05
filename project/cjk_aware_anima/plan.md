# CJK-aware Anima — plan (archived 2026-09-05)

The encoder-side line is **frozen**. Every plan file this directory carried
(`plan.md`, `plan_ko.md`, `plan_ko2.md`, `plan_ko3.md`, `plan_zh.md`,
`plan_zh2.md`, `plan_zh3.md`, `temp_plan.md`) moved verbatim to
`_archive/cjk_aware_anima/plans/` (gitignored tree, preserved in the private
mirror; the pre-move versions are in git history except `plan_zh3.md`, which
was never committed). Links to them from [`findings.md`](findings.md) and the
dated reports resolve against that archive.

What stays live here: [`findings.md`](findings.md) (every settled verdict,
§1–§14), [`deliverables.md`](deliverables.md) (code, data builders, packs,
ship contract), the dated `reports/`, `datasets/`, `probes/`, `gates/`,
`assets/`. The shipped tag-tier pack (`synthja_v4` at
`sorryhyun/anima-vocab-pack-ja`, `AnimaVocabPackLoader`) and the JA+KO+ZH
`synthjakozh1sym_r256` pack are kept as they are — no further distill arms.

Why frozen (findings §4, §12–§14, `reports/0905_blind_g0g1_readout.md`,
`reports/0905_isotropic_ext_hypothesis.md`): names never compose at any
corpus; the coverage refinements are inert; twelve blind sets found no table
property that beats the trained pack, and content-free tables (HOT / ISO1 /
COLLAPSE) tie or beat it for the unmask goal. The only confirmed result is
that rows must exist (C9 > P).

**The line continues at [`../cjk_aware_anima_dit/`](../cjk_aware_anima_dit/plan.md)**
— ext rows as content-free addresses, CJK semantics learned on the DiT side.
The first item there (s13, ISO1 vs C9 direct blind set) was queued from this
directory's `probes/regrid_set.py` on 2026-09-05.
