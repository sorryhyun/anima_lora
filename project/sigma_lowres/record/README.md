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
| `roadmap.md` | the line's phase plan, gates and kill criteria, plus the 2026-07-27 pre-registrations for the yarnsig 768-rescue and 1280→1024 gate probes (**never run**). Frozen 2026-08-01: its Phase-1b gate resolved with E4 and the line moved to paper mode | 2026-07-27 |
| `report.md` | the probe log — every Phase 0 / 1a / 1b verdict with its full table, in run order. Moved from `../bench/` 2026-07-31 | 2026-07-27 |
| `yarnsig_report.md` | the yarnsig sub-line's probe log, extracted from `report.md` 2026-07-27; its closing "open probes" were never run. Moved from `../bench/` 2026-07-31 | 2026-07-27 |
| `review_triage_20260728.md` | R1–R5 triage of the 2026-07-28 external review — the origin document for E1–E4 and E6; each finding's discharge lives in `../paper_bench/experiments/<eN>/`. Moved from `../paper_bench/` 2026-08-09 | 2026-07-28 |
| `paper_plan.md` | the manuscript restructure plan (theory → evidence → application) + manuscript status §9; its steps are discharged into `../paper_bench/experiments/` and `../paper/main.tex`. Moved from `../paper_bench/` 2026-08-09 | 2026-08-04 |
| `e12/` | experiment record E12 (posterior-budget probes) — **REFUTED on both probes**; the lossy-code hypothesis and its "native-only pre-screening" payoff are void. Kept whole (README + instrument gotchas). Moved from `../paper_bench/experiments/` 2026-08-09; its status row stays in the paper_bench index | 2026-07-31 |
| `e15/` | experiment record E15 (two-level unbiased demotion / MLMC roulette) — **priced out**, pre-registered kill at 15.0; yield = the aggregate-coherence finding. Kept whole (README + `e15_price.py`). Moved from `../paper_bench/experiments/` 2026-08-09; its status row stays in the paper_bench index | 2026-07-31 |

Run IDs quoted in the two probe logs as `results/<id>/` are directories
under `../bench/results/` — the **instruments** stay in `../bench/`
(`run_sigma_probe.py` and friends are live, and the paper_bench
experiments call them); only the write-ups froze. The same goes for the
`e12/`/`e15/` records above: their instruments
(`bench/run_posterior_budget.py`, `e15/e15_price.py`) are unchanged.

**Where the live equivalents are now:**

- Experiment designs, results, and verdicts → `../paper_bench/experiments/<eN>/README.md`
  (index at `../paper_bench/README.md`).
- The account as the paper states it → `../paper/main.tex` §3, with the
  manuscript plan in `paper_plan.md` (here).
- Open questions → `../paper/action.md` (open items) and the
  correspondence in `../paper/review/`.
- Raw verdicts and tables for anything measured *since* the paper reorg →
  the per-experiment records under `../paper_bench/experiments/`; the
  instruments that produced them → `../bench/`.
- Remaining work is now the *paper's* open work → `../paper_bench/README.md`
  §"Open work"; there is no separate live line roadmap (`roadmap.md` above
  froze with the move to paper mode).
- Implementation → `../methods.md`, which stays **live**: it documents
  shipped code (`--sigma_lowres`, `--sigma_lowres_yarnsig`,
  `--deterministic`, `make preprocess-demote`) and is cited as current by
  the E15/E16 records (E14/E15 until the 2026-08-01 renumbering).
