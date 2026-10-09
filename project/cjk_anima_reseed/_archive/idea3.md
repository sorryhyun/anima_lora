# idea3 — the twin difference: glyph identity as a subspace (2026-10-07)

From the review of idea2 (10-07), after its probe 0
(`reports/probe_jl_2026_10_07.md`). Run 10-08 — `probes/probe_twin.py`, see the end.

Probe 0's M_in splits the region, not the glyph: in the training bands
60–90 % of a row's in-box sensitivity does not depend on which glyph is
drawn (`grad_identity`), so its top directions are layout. The identity
part is a second moment of a *difference*: one item drawn twice (A with
glyph u in slot k, B with v there, `render_into_scene(ref_text=…)`, the
same scene, font, layout and rng), the same caption (row u), the same ε,
σ and probes v;

    D = g_A − g_B = (J_A − J_B)ᵀ v,   M_D = E[D Dᵀ] = E[(J_A − J_B)ᵀ (J_A − J_B)]
    S = (g_A + g_B) / 2,             M_S = E[S Sᵀ]

M_D's top directions are where row u's effect depends on the glyph drawn;
M_D u = λ M_S u ranks identity against layout (`grad_identity`'s f,
1 − cos, as a subspace). The same backward reads the item's other rows
(`cross`): `grad_identity` found f_cross ≈ f_own, so M_D(own) against
M_D(cross) says whether the identity directions are the row's own or the
window's.

On f0's `sent` items (70 % of its data; 8–14-glyph dialogue lines in 2–3
columns), u kana or kanji (half each), v the same family, not a dakuten /
small-kana sibling, absent from the caption; a twin kept iff A and B
differ in one glyph's box. σ uniform on 0.45–0.8 (sent's band and the
step above it). fp32 as probe 0; the probes on the dilated box (in) and
off it (out). Two fits on disjoint items.

1. f_own and f_cross by σ, in-box, against `grad_identity`'s (0.68 /
   0.52 at half plateau for 36 / 19 px windows).
2. Stability of M_D's top 16 between the fits; the spectrum.
3. Cross-fit λ of M_D u = λ M_S u against tr(M_D) / tr(M_S).
4. Own against cross: the overlap of M_D(own)'s and M_D(cross)'s top 16
   against the fits' floor.
5. The moves: f0's, Δ(pres − f0), the stick — xᵀM_D x / xᵀM_S x; and
   probe 0's `check`: f0's move as M_D sees it against the glyph's recall,
   beside its plain size.

**Stop if** M_D's two fits overlap under 0.6 at k 16 (pair-weighted; probe
0's M_in reached 0.77–0.80 on kana at a like budget), or no cross-fit λ
clears 2 × the trace ratio (the identity part has no directions of its
own), or own and cross overlap as much as own with own (identity is the
window's, not the row's).

**Ran 10-08 — stopped** (`reports/probe_twin_2026_10_08.md`; 400 twins,
100 per family per fit, rows at f0's start): all three rules at both
weightings. M_D's own in-box A / B overlap 0.25–0.27 at k 16 (pair), flat
from 10 to 100 twins; cross-fit λ / r 0.96–1.28 (the identity part has no
directions of its own: r 0.6–0.8 everywhere, high-λ A / B at the isotropic
0.017); own–cross 0.31–0.36 above own–own. A third of the in-box Jacobian
turns with the glyph (f 0.22–0.44), the neighbours' rows 0.6–0.9 as much;
off the box the row's sensitivity near-decorrelates (f 0.67–0.98).
