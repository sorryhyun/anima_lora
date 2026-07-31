# Review triage (2026-07-28) — verdict on the external review

Triage of the 2026-07-28 external review (ChatGPT), verified against the
actual instrument (`bench/run_sigma_probe.py`), the report
(`record/report.md`), and `paper/main.tex`. This is the origin document
for E1–E4 and E6; each finding's discharge lives in the corresponding
`experiments/<eN>/README.md`.

## Verified correct (checked in code/repo, not taken on faith)

- **R1 — estimator variance confound is real and unaddressed.** In
  `run_sigma_probe.py` the floor is `cos(g_native_a, g_native_b)` (two
  independent draw sets, `seeds(0)`/`seeds(1)`); every demote arm gets
  **one** estimate and `gap = floor − ½[cos(a,d)+cos(b,d)]`. There is no
  demoted/demoted self-floor anywhere in the codebase. If per-draw
  gradient variance grows as token count falls (MSE averages over fewer
  elements), an iso-direction null still produces a positive gap that
  grows as the target grid shrinks — the same signature as "absolute
  token count sets the floor." Back-of-envelope with the measured
  endpoint floor ≈ 0.85 (run `20260727-2225`) and noise ∝ 1/tokens: a
  pure-variance null predicts spurious endpoint gaps ≈ 0.02 / 0.05 /
  0.15 for 896 / 768 / 512 vs measured −0.01 / 0.13 / 0.33. So the
  confound plausibly explains **~40% of the 768/512 floors**, not zero
  and not all. This is the one critique that genuinely gated the paper.
  Note the review under-claims in one spot: the x-zero probe is subject
  to the **same** confound (single demoted estimate), so x-zero is not a
  clean rescue of the graph term either — E1's debiasing had to be
  applied to x-zero too. → **[E1](experiments/e1/)**
- **R2 — σ=1 is not graph-only.** `target = noise − lat`: at σ=1 the
  input is pure ε but the target still carries x per arm. The paper's own
  Table (floor table) showed it: 768 endpoint 0.127 vs x-zero 0.064 —
  **half** the 768 endpoint gap looked like target-content, yet the text
  said "any gap *is* the floor by construction" and only highlighted the
  512 route (where endpoint ≈ x-zero and the claim holds). *Resolved by
  [E1](experiments/e1/)(c): the apparent target-content share was
  estimator bias.*
- **R3 — "never safe" is aggregation-dependent.** Confirmed in
  `record/report.md` (pool4 addendum): pooled gap_768 ≈ 0 at σ ≥ 0.875,
  pooled gap_896 ≈ 0 at σ ≥ 0.625. The per-image and batch-SGD objects
  genuinely disagree at high σ; the safety map must state which object it
  is a map *of*, and the trainer claim should be conditioned on the real
  batch/accumulation size. → **[E3](experiments/e3/), still open.**
- **R4 — 14% is a projected ceiling.** main.tex derived 0.86 from token
  ratios; the CMMD A/B is explicitly pending. Abstract/conclusion stated
  it as an outcome. *(Wording fixed in the Branch A rewrite; the A/B
  itself is [E4](experiments/e4/) — done 2026-07-30, now measured.)*
- **R5 — hygiene, all confirmed:** `.gitignore:35` ignores `results`
  globally (so the repro claim was false for the public repo;
  `paper_bench/runs/` is now gitignore-exempt and in-repo); pending
  markers in the manuscript; SwD bib listed a nonexistent author
  ("Khoroshikh", missing Drobyshevskiy/Kuznedelev) — **fixed 2026-07-28**
  against arXiv:2503.16397.

## Partly right / softened rather than rerun

- **Eq. 3 "derivation."** The G(σ) renormalization was already flagged
  post-hoc in the paper, but the abstract's "we first derive" and the
  "ratio sets amplitude / token count sets floor" language outran the
  evidence (2 ratio-matched pairs, 1 crossed pair). Now presented as a
  first-order *account* whose terms are individually evidenced; held-out
  prediction is [E5](experiments/e5/).
- **Framing vs SPD/SwD.** The paper already conceded "the null's error is
  not its governor but its scope"; intro/related work now make explicit
  that neither SPD nor SwD *claims* naive gradient equivalence — we test
  a tempting extension, not their methods.

## Overstated / optional

- The 2-model × 2-adapter × 2-domain generalization matrix is the right
  ask for a strong venue but is not what makes the current claims true or
  false. One extra DiT + one full-FT probe arm is the 80/20 →
  **[E6](experiments/e6/)**.
