# E3 — aggregation-conditioned safety map

| | |
|---|---|
| **Status** | **OPEN** — one GPU run + a framing edit |
| **Why it exists** | Review finding R3: "never safe" is aggregation-dependent. `bench/report.md` (pool4 addendum) has pooled gap_768 ≈ 0 at σ ≥ 0.875 and pooled gap_896 ≈ 0 at σ ≥ 0.625 — the per-image and batch-SGD objects genuinely disagree at high σ. |
| **Depends on** | [E1](../e1/) (paired debiased object + self-floors), [E8.1](../e8/) (ε\* is now the definition of "safe") |
| **In the paper** | §3.1 (aggregation operator is part of the estimand, not an application detail), §5 two-maps table — **[pending]** marker still in `main.tex` |

Mostly already measured (`--pool`); what's missing is the **framing** and
one run at the real operating point:

- One verdict-grid run with `--pool <actual train batch × accum>` (read
  from the shipped LoRA config) + `--self_floor` so pooled arms get
  self-floors too.
- Paper: publish **two maps** — per-example (worst case, what a
  batch-1 user sees) and batch-aggregate (what the shipped trainer
  consumes) — and define "safe" as the pre-specified non-inferiority
  test (now the ε\* definition, [E8.1](../e8/)). The CI itself is free
  (per-bin mean + 1.645·SEM from existing `per_image.jsonl` rows); the
  debiasing inside it is E1's paired object. This resolves the
  report-vs-paper 768 contradiction the review found instead of hiding
  it.

Also owed alongside this run: the a/(b/B) batch-size fit (the estimand
fix registered in `paper/action.md`).
