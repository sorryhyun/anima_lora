# sigma_lowres — σ-conditional low-res gradient routing

Does training on a downscaled image give the same gradient as training
on the full-resolution one? Answer: sometimes, and the map of *when* is
the deliverable. Spectral sufficiency of the noisy input does **not**
guarantee gradient equivalence under resolution substitution; one route
is measured-safe at our operating point (1024→896 at σ > 0.5), worth a
measured −14.6% wall.

The line is currently in **paper mode** — most activity is in
`paper_bench/` and `paper/`.

| where | what |
|---|---|
| [`paper_bench/`](paper_bench/) | **the experiment tree** — `README.md` indexes E1–E12, one dir per experiment under `experiments/<eN>/` with its record and scripts; `runs/` holds the committed run artifacts |
| [`paper/`](paper/) | the manuscript — `main.tex` is now just the shell (preamble, abstract, bibliography); the body lives in `sec_intro.tex` / `sec_theory.tex` / `sec_experiments.tex` / `sec_discussion.tex` + `appendix.tex`, **edit those, not `main.tex`**. Plus `figs/`, `action.md` (open items), `review/` (the review correspondence) |
| [`bench/`](bench/) | the instruments — `run_sigma_probe.py` (the σ-binned gradient probe; internals in `bench/sigma_probe/`), `run_prior_distance.py`, `run_posterior_budget.py`, … — plus their raw run dirs in `bench/results/` |
| [`methods.md`](methods.md) | the implementation: what code exists, where it lives, how to run it |
| [`roadmap.md`](roadmap.md) | remaining phases, gates, and kill criteria for the *line* (the paper's own open work is in `paper_bench/README.md`) |
| [`record/`](record/) | the frozen pre-paper record — proposal, hypothesis, groundings ledger, open-questions list, and the two probe logs `report.md` / `yarnsig_report.md` (moved out of `bench/` 2026-07-31). Historical, still cited for pre-registration provenance |

Start at [`paper_bench/README.md`](paper_bench/README.md) for what has
been measured and what is still open.

Build the manuscript with `tectonic main.tex` from `paper/` (~6 s,
29 pp.). The remaining warnings — `main.bbl` consistency, "rerun needed,
stopping at 6 passes", underfull boxes — are long-standing and benign.
