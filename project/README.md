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

- **Finished** — it ran to a successful conclusion (goal reached or measured
  ceiling hit). Its digest home moves to the tracked
  [`finished/`](finished/) tier so the verdicts stay visible in the repo;
  any still-operational working tree (code, make targets) stays where it is.
- **Retired** — killed, superseded, or shelved. It moves to the gitignored
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
(the ResShift SR sidecar, 2026-08-22; mod guidance, 2026-08-24).

Active projects:

- [`cjk_aware_anima/`](cjk_aware_anima/) — native JA/CJK prompt conditioning
  via an extended T5-side vocab distilled against the EN-translation teacher.
  Probe + zero-shot ext vocab measured (`bench/cjk_adapter/`); the Phase 2a
  data assets MT cannot produce are built ([`datasets/`](cjk_aware_anima/datasets/)
  — Wikidata proper-noun lexicon, native-register manga eval set); Phase 2b is
  closed (loop in `scripts/distill_cjk/`, one-off gate drivers in
  [`gates/`](cjk_aware_anima/gates/)) and Phase 2c is running. Split four ways
  instead of a single founding proposal: [`motivation.md`](cjk_aware_anima/motivation.md) (why,
  incl. the directions already ruled out),
  [`done.md`](cjk_aware_anima/done.md) (completed-item checklist),
  [`plan.md`](cjk_aware_anima/plan.md) (what remains — live phase, deployment,
  risks), and one **dated per-phase report** carrying the measured verdicts:
  [`report_0816_phase2.md`](cjk_aware_anima/report_0816_phase2.md) (Phase 2b/2c,
  incl. the two withdrawn metrics). Measured numbers stay with the code:
  [`datasets/README.md`](cjk_aware_anima/datasets/README.md) and
  `bench/cjk_adapter/results/`.
