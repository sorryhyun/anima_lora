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
| `report.md` | the probe log — every Phase 0 / 1a / 1b verdict with its full table, in run order. Moved from `../bench/` 2026-07-31 | 2026-07-27 |
| `yarnsig_report.md` | the yarnsig sub-line's probe log, extracted from `report.md` 2026-07-27; its closing "open probes" were never run. Moved from `../bench/` 2026-07-31 | 2026-07-27 |

Run IDs quoted in the two probe logs as `results/<id>/` are directories
under `../bench/results/` — the **instruments** stay in `../bench/`
(`run_sigma_probe.py` and friends are live, and paper_bench E1–E12 call
them); only the write-ups froze.

**Where the live equivalents are now:**

- Experiment designs, results, and verdicts → `../paper_bench/experiments/<eN>/README.md`
  (index at `../paper_bench/README.md`).
- The account as the paper states it → `../paper/main.tex` §3, with the
  manuscript plan in `../paper_bench/paper_plan.md`.
- Open questions → `../paper/action.md` (open items) and the
  correspondence in `../paper/review/`.
- Raw verdicts and tables for anything measured *since* the paper reorg →
  the per-experiment records under `../paper_bench/experiments/`; the
  instruments that produced them → `../bench/`.
- Remaining phases for the *line* (not the paper) → `../roadmap.md`;
  implementation → `../methods.md`.
