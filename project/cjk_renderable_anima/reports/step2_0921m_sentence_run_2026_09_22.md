# Sentence run on the merged inventory: `step2_0921m` 30 k (2026-09-22)

> The plan § 2 run: `data_step2_0921m` (merged `step1_0921` + `ja_cold_0001_1900`
> inventory, 40 000 items, 24 762 unique strings) trained 30 k steps from the
> merged table `rows_step1_0921m_merge` at μ 0.3. **Sentences moved**: sub-exact
> pooled lift **+0.233** (held groups lead, +0.37 / +0.14), above every prior
> sentence arm's +0.09 – +0.15, and the multi-glyph native ruler read its first
> non-zero ever (はい 3/16, CER 0.97 → 0.83). **Single-glyph native collapsed**
> (en 34 → 12 both-hit, swap 28 → 0) — but the glyph is right and the *length*
> is wrong: あ renders as `ああああ。`, はい as `はいいい`. One defect, in the data
> mix (10 % single, nothing shorter than a 2-piece short), not in μ. The loss
> curve says nothing after ~5 k steps earned its cost.

## Runs

| job | what | wall |
|---|---|---|
| `20260922-074720-b20063` | `--stage train eval native`, arm `rows_step2_0921m_plain_bs05c25_30k` | 265.7 m (compile done at +23 m; 2.31 it/s) |
| `20260922-122314-7c3dfe` | `--stage native --eval_tag sent` (はい, おしい, やったネ, ちょっと来い × en) | 4.6 m |

Train argv = plan § 2 *Train* with `--train_steps 30000 --save_every 5000`,
`--init_rows output/wake_probe/rows_step1_0921m_merge/trained.pt --init_anchor 0.3`,
plus `single_extra` in `--eval_groups`. Data = plan § 2 *Data* with
`--n_items 40000` (`data_step2_0921m/args.json`; on `/media/sorryhyun/new`).
Train set: 20 000 short / 15 999 sentence / 4 000 single; 30 k × 4 = 3 epochs.

Warm start: 2 271 / 2 271 rows, `row_scale 196.07 → 205.73` (× 0.953). Loss
0.122 → 0.097 by 5 k and flat after (0.093 at 26 k); `in_box` 0.175 → 0.145 by
5 k, flat after; `out_box` 0.090 → 0.085; `warm_cos` 0.989 at 5 k **rising** to
0.998 at 26 k, drift 0.032 → 0.026 — the cosine tail + anchor pulls the rows
back onto the warm start for the last 20 k.

## Reads

### Sentence groups — `src/probe/sub_exact.py`

| group | n | recall | control | lift |
|---|---|---|---|---|
| short | 16 | 0.431 | 0.116 | +0.315 |
| short_held | 16 | 0.495 | 0.128 | **+0.367** |
| phrase | 16 | 0.300 | 0.191 | +0.109 |
| phrase_held | 16 | 0.252 | 0.111 | +0.141 |
| **pooled** | 64 | | | **+0.233** |

Exact 0/16 everywhere except short_held 1/16 (floor-saturated, as always).
The control roughly doubled against the 6 k arms (0.05 – 0.07 → 0.11 – 0.19):
reads carry more JA glyphs per item — the same habit the native sheets show.
`0920b_seed0921` reads +0.246 on *its* strings (boot diff +0.013, CI
[−0.05, +0.08]) — different eval set, does not order the two.

### Singles (exact, sfx) vs the warm source `rows_step1_0921_s30k`

| single | single_ext | single_small | single_kanji |
|---|---|---|---|
| 28 → **19** / 36 | 20 → **24** / 36 | 10 → **5** / 36 | 19 → **16** / 36 |

### Single-glyph native (あ か す 日 × en / swap, 8 prompts × 2 seeds)

| clause | step1 hit sfx / both | this run hit sfx / both | any CJK read |
|---|---|---|---|
| en | 37 / 34 | 16 / 12 | 64 |
| swap | 35 / 28 | 5 / **0** | 40 |

`kept` 0/16 for every kana. Sheets (`native/sheet_*.png`): the requested glyph
is on the canvas in nearly every cell, **as a run** — `ああ` `あああ` `ああああ。`
`すすで` `実よたんすす` — or a bubble/chalkboard of pseudo-sentence. Two prompts
(p00, p01 seed 0) lose the scene to a text panel. す/en is the only kana with
hits (8/16 sfx); か and あ read 0 on both clauses because every render is ≥ 2
glyphs.

### Multi-glyph native — `native_sent/`

| arm | CER sfx | CER vl16 | hit sfx | hit vl | both |
|---|---|---|---|---|---|
| `step2_0920` 6 k | 0.966 | 0.962 | 0 | 0 | 0 |
| `step2_0920` boost8 6 k | 0.970 | 0.979 | 0 | 0 | 0 |
| **`step2_0921m` 30 k** | **0.826** | **0.801** | **3** | **4** | **3** |

All hits are はい; the three longer words 0/16 each. Sheet: は and い present
in most cells — `は い` `はい` `はいい。` `はいいい` `いはかい` — placed in a
speech bubble (p07), a caption strip (p05), a small bubble (p02): manga-native
placement the 6 k arms never produced. Same length failure as the singles.

## Verdict

- **The merged-inventory lever works**: +0.233 pooled with held groups leading,
  first multi-glyph native hits, natural placement. This is what section 2 set
  out to test.
- **It broke length control, not glyph identity.** Singles and 2-glyph words
  pad to a bubble-sized line. The training set has 10 % singles and no short
  under 2 pieces / no sentence under 20 glyphs; the rows learned "text is a
  line". μ is not the knob for this (the plan's μ trade was identity, not
  length, and this happened at μ 0.3).
- **30 k bought nothing over ~5 k** by every training scalar; size the next
  arm at 6 k.
- Owed: `row_dose.py` needs a seed read of `rows_step1_0921m_merge` on these
  eval strings (only the old-table `rows_step2_0920_seedread` exists) — an
  eval-only GPU job.

Next arm, one variable: same data recipe with `--scene_mix single=0.35,short=0.45,sentence=0.2`
(and a 1-piece short if the builder allows it), 6 k steps, same μ; read native
singles + `native_sent` beside sub-exact.
