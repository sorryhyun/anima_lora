# sigma_lowres — σ-conditional low-res gradient routing

Does training on a downscaled image give the same gradient as training
on the full-resolution one? Answer: sometimes, and the map of *when* is
the deliverable. Spectral sufficiency of the noisy input does **not**
guarantee gradient equivalence under resolution substitution; one route
is measured-safe at our operating point (1024→896 at σ > 0.5), worth a
measured −14.6% wall.

The line is currently in **paper mode** — most activity is in
`paper_bench/` and `paper_v2/`.

| where | what |
|---|---|
| [`paper_bench/`](paper_bench/) | **the experiment tree** — `README.md` indexes E1–E16, one dir per experiment under `experiments/<eN>/` with its record and scripts; `runs/` holds the committed run artifacts |
| [`paper_v2/`](paper_v2/) | **the live manuscript (copied from `paper_suggestion/` 2026-08-09)** — being reframed around the E19–E25.0 cancellation geometry (B–C near-cancellation, σ-indexed axis field, angle-borne residual); plan and order of work in [`paper_v2/revision_plan.md`](paper_v2/revision_plan.md) |
| [`paper_suggestion/`](paper_suggestion/) | the frozen 2026-07-31 draft-of-record — same evidence as the archived v1, restructured as "two accounts, scored head to head": the spectral account is stated first as co-protagonist (§2.2), a side-by-side disagreement table (§2.5) sites every experiment, and the `accounts_headtohead` figure + RMSE table lead the experiments. Known open items at freeze: repo URL in the repro statement de-anonymizes, abstract dropped the open-CMMD-gate caveat. The pre-rewrite v1 tree (incl. `action.md` + `review/` correspondence) is archived at `_archive/paper/sigma_lowres_v1/` |
| [`bench/`](bench/) | the instruments — `run_sigma_probe.py` (the σ-binned gradient probe; internals in `bench/sigma_probe/`), `run_prior_distance.py`, `run_posterior_budget.py`, … — plus their raw run dirs in `bench/results/` |
| [`methods.md`](methods.md) | the implementation: what code exists, where it lives, how to run it — the one **live** doc outside the paper tree |
| [`record/`](record/) | the frozen pre-paper record — proposal, hypothesis, groundings ledger, open-questions list, the line roadmap (frozen 2026-08-01), and the two probe logs `report.md` / `yarnsig_report.md` (moved out of `bench/` 2026-07-31). Historical, still cited for pre-registration provenance |

Start at [`paper_bench/README.md`](paper_bench/README.md) for what has
been measured and what is still open.

Build the manuscript with `tectonic main.tex` from `paper_v2/`
(~6 s). `main.tex` is just the shell (preamble, abstract, bibliography);
the body lives in `sec_intro.tex` / `sec_theory.tex` /
`sec_experiments.tex` / `sec_discussion.tex` + `appendix.tex` — **edit
those, not `main.tex`**. The remaining warnings — `main.bbl`
consistency, "rerun needed, stopping at 6 passes", underfull boxes —
are long-standing and benign.
