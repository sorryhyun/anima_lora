# record/ — the frozen exploration record

The pre-paper phase of the sigma_lowres line, moved here 2026-07-31.
These files are **historical and no longer maintained**, but they are
kept tracked (not archived out of the repo) because the manuscript's
pre-registration claims cite them: freeze dates, G-numbers, and the
original hypothesis text are the evidence that a prediction was
committed before its run.

| file | what it is | last live |
|---|---|---|
| `initial_proposal.md` | Phase 0 design, frozen before any run; H1–H3 and the kill criteria. Verdict: spectral mechanism REFUTED. | 2026-07-24 |
| `hypothesis.md` | the living mechanism account, v2.1 — the normalized two-term form that became the paper's Eq. 3 | 2026-07-27 |
| `groundings.md` | the evidence ledger G1–G9: one entry per measurement, with pre-registration status and what it grounds | 2026-07-27 |
| `questions.md` | the line's open-questions list Q1–Q7 (most now answered) | 2026-07-27 |

**Where the live equivalents are now:**

- Experiment designs, results, and verdicts → `../paper_bench/experiments/<eN>/README.md`
  (index at `../paper_bench/README.md`).
- The account as the paper states it → `../paper/main.tex` §3, with the
  manuscript plan in `../paper_bench/paper_plan.md`.
- Open questions → `../paper/action.md` (open items) and the
  correspondence in `../paper/review/`.
- Raw bench verdicts and tables → `../bench/report.md`.
- Remaining phases for the *line* (not the paper) → `../roadmap.md`;
  implementation → `../methods.md`.
