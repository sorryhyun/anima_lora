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

- [`cjk_aware_anima_dit/`](cjk_aware_anima_dit/) — successor to the encoder
  line (2026-09-05): ext rows as content-free, deterministic addresses; CJK
  semantics learned on the DiT side. Goals: manga trains healthily unmasked
  at corpus scale (873 text-masked images), and a LoRA learns CJK tag
  meaning for isotropic rows. Home: [`plan.md`](cjk_aware_anima_dit/plan.md)
  (D0 ISO1-vs-C9 blind set → deterministic table + route partition →
  OCR-diverse corpus → corpus-scale unmask arms → tag-semantics arm).
- [`cjk_aware_anima/`](cjk_aware_anima/) — the encoder-side line, **frozen
  2026-09-05**: native JA prompt conditioning via an extended T5-side vocab
  distilled against the EN-translation teacher. Rare kanji names fail under
  every lever; coverage and geometry refinements are inert; content-free
  tables tie or beat the trained pack for unmask training. `synthja_v4`
  ships as the zero-shot tag tier (`sorryhyun/anima-vocab-pack-ja`).
  Home: [`findings.md`](cjk_aware_anima/findings.md) (§1–§14, read-only),
  [`deliverables.md`](cjk_aware_anima/deliverables.md); plans archived to
  `_archive/cjk_aware_anima/plans/`. Measured tables stay in the dated
  `reports/`, [`datasets/README.md`](cjk_aware_anima/datasets/README.md)
  and `bench/cjk_{adapter,distill}/results/`.
