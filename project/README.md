# project/ — active promoted lines

One subdir per research line that has graduated past "proposal + bench report"
into an ongoing project with open phases. Each subdir is the line's home page:

| File | Contents |
|---|---|
| `methods.md` | The implementation — what code exists, where it lives, how to run it |
| `bench.md` | Digest of measured results — omitted when the line's bench lives in-tree (its own `report.md` serves directly) |
| `questions.md` | Open questions the line has not answered |
| `roadmap.md` | Remaining phases, gates, and kill criteria |
| `outcomes.md` | Shippable/practical artifacts the line produced (optional — appears once something is ship-shaped) |

Canonical sources these digest (never duplicated wholesale):
the line's proposal(s) (frozen designs) and its bench (`report.md` = raw
verdicts + full tables, `results/` = run envelopes). A promoted line may
adopt these into its home — `project/<line>/bench/` for the bench and e.g.
`initial_proposal.md` for the founding proposal (directedit_ec and
sigma_lowres do both); lines that haven't keep them in `bench/<line>/`
and `docs/proposal/<line>*.md`.

Active projects:

- [`sigma_lowres/`](sigma_lowres/) — σ-conditional low-res gradient routing.
  Spectral mechanism refuted; one measured-safe route (1024→896 @ σ>0.5),
  measured −14.6% wall. Now in paper mode: the experiment tree is
  [`paper_bench/`](sigma_lowres/paper_bench/) (E1–E15, one dir each) and
  the pre-paper docs — including this table's `questions.md` and
  `roadmap.md` (frozen 2026-08-01; open work now lives in
  `paper_bench/README.md`) — are frozen under
  [`record/`](sigma_lowres/record/). `methods.md` stays live. Line home:
  [`sigma_lowres/README.md`](sigma_lowres/README.md).
- [`directedit_ec/`](directedit_ec/) — EasyControl cond stream as a learned
  preservation prior for DirectEdit. Phases 0–1b passed zero-training;
  Phase 2.5 (delta-caption instruction editor) probe PASSED at the trained
  point → EasyEdit ship proposal + paper prep are the owed write-ups
  (neither the line's `outcomes.md` nor an `easyedit_comfy_node` proposal
  exists yet).
