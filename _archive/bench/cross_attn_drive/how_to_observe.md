# How to observe `tag_influence.py` (tag-headroom Phase-0)

Practical guide for reading a `tag_influence.py` run — **what to look at** and
**how to read the images**. Proposal: `docs/proposal/tag_headroom_commitment_sigma.md`.
Method memory: `[[project_tag_headroom_phase0_impl]]`.

## The question this bench answers

Which prompt tags are **real adapter/conditioning headroom** vs base-owned dead
ends? Detector-free — we never *classify* whether a tag rendered; we *intervene*
(drop the tag, gate it by σ) and read the model's own causal response. The target
output is a ranked, deduplicated list of tags that are **used ∧ rendered-wrong ∧
commit-late ∧ common** — the headroom signature.

## Where the results are

```
bench/cross_attn_drive/results/<YYYYMMDD-HHMM>-<label>/
  result.json      # verdict, medians, flagged set, top-by-headroom-score (READ FIRST)
  per_tag.csv      # one row per tag — the ranking
  per_caption.csv  # one row per (tag, caption) — the raw spread behind each tag
  montages/        # <tag>__capN.png — visual, one per (tag, caption)
  shortlist/       # montages for the flagged (or top-N) tags, copied out for the eye
  commitment.csv   # 0b: rel(σ-cutoff) curve per (tag, caption, seed)
```

`per_tag.csv` / `result.json` are written **only at the end of phase 0a** — a
killed run leaves montages but no CSV. Montages stream to disk as each (tag,
caption) finishes, so they're the way to watch a run in flight.

## What to observe — the metrics (per_tag.csv)

Columns, and how to read them. "High/low" is **relative to this run's medians**
(in `result.json → phase0a.medians`), not absolute — the bench classifies against
its own measured set.

| column | meaning | what you want |
|---|---|---|
| `influence_local` | image distance C vs C∖T inside the tag's affected region | **high** = the model uses the tag |
| `concentration` | region-mean / whole-image-mean of the diff | **high** = localized feature (addressable); low = diffuse global shift |
| `instability_rel` | seed-variance in the tag's region ÷ whole-image seed-variance | **high** = "trying but rendering inconsistently" = failure proxy; ~1 = no excess wobble |
| `train_freq` | # corpus captions carrying the tag | **high** = plausibly adapter-addressable (low-freq = data-limited, not architecture) |
| `cell` | classification (below) | hunt the `headroom` cell |
| `headroom_score` | rank-norm product of influence × instability × freq | ranking key; `per_tag.csv` is sorted by it |

### The cells (proposal interpretation table)

- **`headroom`** — high influence + high instability + high freq. *The cell we
  hunt.* Used, late-committing-candidate, rendered inconsistently, common ⇒
  plausibly adapter/conditioning-addressable.
- `headroom_lowfreq` — same but rare → data/capability-limited, not a lever.
- `ignored` — high-freq + ~zero influence → the tag isn't even attempted (often
  redundant with a co-occurring tag, e.g. `holding` vs `holding phone`).
- `solved_or_memorized` — high influence + **low** instability → it works (or is
  memorized). Disambiguate memorization with `bench/memorization/probe.py`
  (low instability **and** high nearest-training PE-Spatial similarity = memorized).
- `weak` — low influence, low instability → nothing to see.

### Verdicts

- **0a** PASS = a high-influence ∧ high-instability population exists (`result.json
  → phase0a.verdict`). KILL = every suspect tag is either solved or ignored ⇒ no
  trying-but-failing population, the line stops.
- **0b** PASS = the flagged tags **commit late** — `commitment_sigma ∈ [0.6, 0.8]`
  (`result.json → phase0b`). That co-location (used ∧ late ∧ unstable ∧ common) is
  the green light for Phase-1.
- **0c** PASS = boosting a tag's cross-attn drive below the cutoff adds **localized
  structure** for ≥1 tag (`result.json → phase0c`). KILL = every tag's boost is a
  diffuse tone shift / incoherent — the magnitude-rescale trap the front-loaded
  finding predicts at low σ, so a "sustain attention at later σ" LoRA wouldn't help.

### commitment-σ (0b)

Per (tag, caption, seed) `commitment.csv` has `rel = dist(full, gated) /
dist(full, full-drop)` at each σ cutoff. The tag is presented only **above** the
cutoff and dropped below it. `rel→0` means presenting the tag only above that σ
already reproduces the full caption (the tag locked early); `rel→1` means it
hadn't committed yet. `commitment_sigma` = the largest cutoff still at `rel ≤
--commit-thresh` (default 0.2) — the σ at which the tag locks. Late = it keeps
needing the [0.6, 0.8] band.

### boost probe (0c) — the no-train falsification gate

`run_phase0c` is the cheap test of the *"train a LoRA to keep cross-attn driving at
later σ"* idea **before building it**. For each tag it amplifies that tag's
embedding-space delta `embed_alt + scale·(embed − embed_alt)` **below** `--boost-cutoff`
(default σ 0.85 — the front-loaded collinear threshold where text supposedly has no
authority) and reads `boost.csv` per (tag, scale):

- **delta_local** — does cranking the scale move the tag's region at all (rising = the
  lever bites).
- **concentration** — region/whole-frame ratio of the boost's effect. **High (≥
  `--concentration-min`, default 1.5) = localized structure**; ~1 = a diffuse,
  whole-image shift (the magnitude-rescale tone trap).
- **instability_rel vs base_instability_rel** — boosted seed-wobble vs the
  full-caption baseline in the region. Climbing past baseline·(1+`--instab-tol`) =
  off-manifold incoherence, not new structure.

A tag is `structure: true` when the top scale is **rising ∧ localized ∧ coherent**.
PASS (any tag) green-lights the LoRA on that lever; KILL says don't build it. Read
`boost_montages/<tag>.png` (`full | boosted@top-scale | boost-diff heatmap`) — a
sharpened in-region feature = structure; a washed/tinted whole frame = tone.

## How to read the montage images

Each `montages/<tag>__capN.png` is one row, **thumbnailed**:

```
[ full@seed0 | full@seed1 | full@seed2 | full@seed3 ]  ‖  [ drop@seed0 | diff ]
└─────────── instability strip ───────────┘            └──── influence ────┘
```

- **Instability strip** (the 4 full-caption seeds): read *across* them. Does the
  tag's feature render the **same** each seed, or does it wobble? Consistent =
  solved/memorized; wobbling in the tag's region = struggling. This is the
  detector-free stand-in for "is it correct" — we measure *consistent* instead.
  *(Runs before 2026-06-25 evening saved only seed0 + drop + diff — no strip.)*
- **drop@seed0**: the same seed with the tag removed. Compare to `full@seed0` —
  what disappeared/changed is the tag's contribution.
- **diff**: normalized heatmap of |full − drop| averaged over seeds. **Bright +
  tight** = localized influence (addressable). **Bright + spread over the whole
  frame** = diffuse/global trajectory shift (not a clean localized feature).
  **Dark** = the tag barely mattered (ignored/redundant).

### Patterns seen in the first run (qualitative, seed-0)

- **Rendered text** (`english text`): tight diff on the text region **and** the
  rendered glyphs are gibberish → used ∧ failing = **prime headroom**.
- **Logos / counts** (`logo`, `2girls`): high influence but **diffuse** — dropping
  them re-renders much of the frame. Global structure, low concentration.
- **Hands** (`holding`, `interlocked fingers`): **low** influence — the action is
  carried by co-occurring tags, so the bare tag drop barely moves the image.
  Looks **ignored/redundant**, a different failure mode than text.

## Reading discipline (don't over-read)

- A single montage is **one seed, one caption**. The verdict aggregates 4 seeds ×
  6 captions; trust `per_tag.csv`, use montages to *understand* a row, not rank.
- **Instability confounds legit variation** (a "smile" genuinely varies). That's
  why it's normalized to whole-image variance (`instability_rel`) — and why the
  final shortlist still needs the **human eye**, not a variance threshold alone.
- **Base-owned ceiling**: x̂₀-wander is ~90% base-owned;
  the `train_freq` split is the guard — only high-freq failures are plausibly
  adapter-addressable.
- **Per-σ-reweight graveyard**: any Phase-1 lever must be *localized* — a global per-σ guidance reweight is a
  settled no-win.

## Re-running

```bash
# full Phase-0 (one GPU run at a time — 16 GB card; verify nvidia-smi is clear first)
python bench/cross_attn_drive/tag_influence.py --phase both --compile \
  --captions-per-tag 6 --seeds 4 --infer-steps 28 --size 1024 --cfg 4.0 --label phase0

# 0b only, on an explicit tag set
python bench/cross_attn_drive/tag_influence.py --phase 0b --tags "english text" "logo" ...
```

Knobs worth knowing: `--suspect-tags` (override the a-priori set), `--region-frac`
(localization mask, default top 5 %), `--commit-thresh` (0b rejoin threshold),
`--cutoffs` (0b σ grid). **Ops note**: this box is a 16 GB GPU — run one process at
a time and kill by PID (never `pkill -f tag_influence.py` — it matches its own
shell). See `[[project_tag_headroom_phase0_impl]]`.
