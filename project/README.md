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

Active projects: **none** — both CJK lines are frozen (below), and no other
line has open phases.

Frozen lines (kept here rather than in `finished/` or `_archive/`: their
`findings.md` is the anti-re-proposal record and is read often, but neither
reached its own top-line goal, so neither is a "finished" line):

- [`cjk_aware_anima_dit/`](cjk_aware_anima_dit/) — the DiT-side successor,
  **frozen 2026-09-08**: ext rows as content-free, deterministic addresses;
  CJK semantics to be learned on the DiT side. **Its OCR half shipped and its
  two DiT goals were never tested at scale.** Shipped: the AnimeText detector
  + a PaddleOCR-VL-1.6 SFX reader (`sorryhyun/paddleocr-vl-1.6-manga-lora`,
  wired as `anime_tools.ocr.sfx`), the caption clause rules, and the D1
  quote-partitioned pack. Open: G-A (unmasked training at corpus scale) was
  measured only on one 351-image shard, where the caption clauses tie; G-B (a
  LoRA learning CJK tag meaning) was never run. Ceiling found on the reader:
  five arms decoupled in-domain COO from the doujin gate, so the headroom is
  ♡ / small-kana **labels**, not representation.
  Home: [`findings.md`](cjk_aware_anima_dit/findings.md),
  [`plan.md`](cjk_aware_anima_dit/plan.md) (freeze note); plans and dated
  reports archived to `_archive/cjk_aware_anima_dit/{plans,reports}/`.
- [`cjk_aware_anima/`](cjk_aware_anima/) — the encoder-side line, **frozen
  2026-09-05**: native JA prompt conditioning via an extended T5-side vocab
  distilled against the EN-translation teacher. Rare kanji names fail under
  every lever; coverage and geometry refinements are inert; content-free
  tables tie or beat the trained pack for unmask training. `synthja_v4`
  ships as the zero-shot tag tier (`sorryhyun/anima-vocab-pack-ja`).
  Home: [`findings.md`](cjk_aware_anima/findings.md) (§1–§14, read-only),
  [`deliverables.md`](cjk_aware_anima/deliverables.md); plans **and the dated
  reports** archived to `_archive/cjk_aware_anima/{plans,reports}/`.
  Dataset-side numbers stay live in
  [`datasets/README.md`](cjk_aware_anima/datasets/README.md) and
  `bench/cjk_{adapter,distill}/results/`.
