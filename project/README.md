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
`initial_proposal.md` for the founding proposal (the archived directedit_ec
and sigma_lowres lines did both); lines that haven't keep them in
`bench/<line>/` and `docs/proposal/<line>*.md`.

A line leaves the active set one of two ways:

- Finished — it ran to a successful conclusion (goal reached or measured
  ceiling hit). Its digest home moves to the tracked
  [`finished/`](finished/) tier so the verdicts stay visible in the repo;
  any still-operational working tree (code, make targets) stays where it is.
- Retired — killed, superseded, or shelved. It moves to the gitignored
  `_archive/` tree (local + preserved in the private mirror).

Retired lines so far:

- `sigma_lowres` — archived 2026-08-19 → `_archive/sigma_lowres/`. The research
  branches + paper drafts were already mirrored to the private repo
  (2026-08-15) and deleted from public origin; the shipped `--sigma_lowres`
  feature stays live (`docs/optimizations/sigma_lowres.md`).
- `directedit_ec` — archived 2026-08-19 → `_archive/directedit_ec/`. Private
  mirroring still pending; the state is snapshot in the mirror's `main` and
  in origin history. EasyEdit ship proposal + paper prep remain the owed
  write-ups if the line reopens.

Finished lines are listed in [`finished/README.md`](finished/README.md)
(the ResShift SR sidecar, 2026-08-22; mod guidance, 2026-08-24; the
encoder-side CJK line `cjk_aware_anima` and its DiT-side successor
`cjk_aware_anima_dit`, both 2026-09-24).

Active projects:

- [`cjk_renderable_anima/`](cjk_renderable_anima/) — promoted 2026-09-14 from
  the wake line of `finished/cjk_aware_anima_dit`: a frozen DiT renders a requested JA
  glyph, or a whole common word, from a delta on the vocab pack's ext rows
  (all 92 kana 34/36; word rows are units, します/してる from one row), and a
  static table trained on strings carries order and count through the frozen
  adapter (unseen 3-kana 3/16, order read 28/48 vs 5/48) at the cost of a
  unit-count prior in the rows (singles 34 → 5). Open phase: the mixed
  distribution (P1), then the repeat mode. Home: `README.md`, `plan.md`,
  `findings.md`.

- [`cjk_anima_scale/`](cjk_anima_scale/) — opened 2026-09-23 out of
  `cjk_renderable_anima`: the production line for the JA pack. Home:
  `README.md`; its `band_experiment_results.md` is the **vocab band law**
  (band keyed on glyph count, px sets the floor, nothing above 0.9 — all
  measured, not theory). `design.md` is the stage-schedule sketch under
  review; nothing has trained there yet.
