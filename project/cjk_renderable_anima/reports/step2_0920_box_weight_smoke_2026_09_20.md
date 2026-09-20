# step 2 on `step1_0920`: row-cos reads and the in-box weight smoke (2026-09-20)

> Two things. **Row geometry** (CPU): `step1_0920`'s rows share one direction
> m̂ (pair cos 0.177, 0 once m̂ is removed) and what is left is glyph-shape
> neighbours (へ·ヘ, ピ·ビ, 体·休, 後·彼); `step2_0919` is its seed to cos
> 0.994 — that sentence pass did not move the table. **In-box weight smoke**
> (4 arms × 2 000 steps, plain, seed `step1_0920`, `data_step2_0919`'s build):
> weight moves the rows monotonically (drift 0.021 → 0.084 from share 2 % →
> 48 %) and buys **nothing on sentences** — sub-exact pooled lift +0.114 /
> +0.077 / +0.079 / +0.109, no order, exact 0/16 everywhere — while singles
> (6 → 4 of 8) and native (4 → 2 of 8) pay and en cos holds. The anchor pulls
> about half the drift back as the lr decays, in every arm. n = 8 per group,
> one seed: directions, not numbers. Owed: `ρ_g 0.05` + cap 0.25 (needs a
> `--box_share_cap` flag), then close the axis.

## 1. Row-cos reads (no GPU)

`src/probe/table_geometry.py` on `trained.pt` (rows = `raw × row_scale`), plus
two scratch scripts for top pairs and per-row movement (not in the repo).

| table | rows | ‖row‖ | m̂ energy | PR | pair cos mean / p95 | resid mean / p95 | neighbour / ctrl (resid) |
|---|---|---|---|---|---|---|---|
| `step1_0920` | 374 | 160.2 | 0.199 | 21.9 | 0.177 / 0.310 | −0.002 / 0.131 | 0.108 / 0.058 |
| `step1_0919` | 374 | 75.6 | 0.175 | 27.0 | 0.151 / 0.284 | −0.002 / 0.117 | 0.139 / 0.073 |
| `step2_0919` | 369 | 75.7 | 0.172 | 27.8 | 0.149 / 0.282 | −0.002 / 0.117 | 0.137 / 0.072 |
| plain `src53k` | 434 | 113.7 | 0.273 | 12.4 | 0.154 / 0.351 | 0.004 / 0.118 | 0.085 / 0.084 |

m̂·m̂ across tables: `step1_0920` ↔ `src53k` 0.826, ↔ `step1_0919` 0.783;
per-row cos on shared ids 0.321 (`src53k`), 0.375 (`step1_0919`). The
"cos to pack row" column is against the default (preview) pack and means
nothing for a raw-pack arm.

- **The mean cos is m̂.** Remove it and the mean is 0. The ΔFM tables are less
  bunched on it than plain (energy 0.20 vs 0.27, PR 22 vs 12).
- **The residual's top pairs are shape neighbours**, `step1_0920`: へ·ヘ 0.48,
  ピ·ビ 0.38, 、·」 0.37, ソ·ゾ 0.36, ラ·ヲ 0.35, 二·コ 0.34, ン·シ 0.33,
  体·休 0.33, ー·一 0.33, ば·ぱ 0.33; kanji by shared component (後·彼, 使·便,
  流·激, 開·離). Same class in `step1_0919` (使·便, ブ·プ, ユ·ヨ) at a lower
  max (0.38). Plain `src53k` carries no neighbour signal (0.085 vs ctrl 0.084).
- **Script blocks are weak**: resid CJK↔CJK 0.019, KATA↔KATA 0.047, CJK↔kana
  −0.02…−0.03; the 10 punct rows 0.095.
- **Small kana are the negative tail and the smallest rows** in both ΔFM
  tables (`step1_0920` ゅ 53, ャ 73, ぅ 74, ッ 79 against a mean of 160;
  `step1_0919` ェ 16, ゅ 17) — they are also 0/36 on `single_small`.
- The max-norm row in both (435 / 237) has no text from `row_texts`.

**`step2_0919` vs its seed** (369 shared rows): ‖Δ‖/‖row‖ mean 0.054, p50
0.026 (= the log's `warm_drift` 0.054), row cos 0.994, norm ratio 1.007, no
common direction in the move (pair cos among Δ rows 0.014, radial 0.001).
What moved is sentence-frequent rows — ！！ 1.42, っ 1.07, ・・・・ 0.97,
だ 0.40, ！ 0.37, ま 0.31 — and kanji sit at 0.000. 確 思 初 濃 考 are in the
step-1 table and not in `data_step2_0919`'s inventory. Its loss is flat
(0.1010 → 0.1006). It ran `--box_weight 4` (`trained.pt` args), the
whole-canvas-normalised form.

## 2. The smoke (2026-09-20 11:12 → 12:37)

One variable, the in-box weight. Everything else is `step2_0919`'s train argv
(`synth.md`): plain (`--pair_loss 0`), `--init_anchor 0.3 --lr_warmup 500`,
`--lr_rows 1e-3` cosine, σ 0.5–0.9, batch 4, `--free_residual 1e-3`,
`--c_flat 0`, raw pack (`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`).
Seed `output/wake_probe/rows_step1_0920_s53k/trained.pt`; data
`data_step2_0920` → symlink to `data_step2_0919`; **`--train_steps 2000`**.
Eval cut down for a smoke: `--eval_groups single,short,short_held,en
--eval_limit 8 --seeds 1`, native in the same job (`--stage train eval
native`, `--native_chars あ,か --native_clauses en --native_limit 4`, 8
renders). ≈ 20 min per arm.

| arm (`rows_step2_0920_…`) | job | weight |
|---|---|---|
| `smoke_plain_w1_s2k` | `20260920-111243-1fbf84` | `--box_weight 1` |
| `smoke_plain_w4_s2k` | `20260920-111244-64b00e` | `--box_weight 4` |
| `smoke_plain_w12_s2k` | `20260920-111244-a4402f` | `--box_weight 12` |
| `smoke_plain_bs10_s2k` | `20260920-121638-29bb3f` | `--box_share 0.1` |

**What the weights are on this data** (`train.jsonl`, 10 000 scene items:
short 5 000, sentence 4 000, single 1 000; box area mean 1.9 % of the canvas,
p90 2.9 %; glyphs per item 1–18, mean ≈ 5.5):

| setting | mean in-box share | items at the 0.75 cap |
|---|---|---|
| w 1 | 0.02 | – |
| w 4 | 0.07 | – |
| w 12 | 0.18 | – |
| ρ_g 0.05 | 0.29 | 2 % |
| ρ_g 0.1 | 0.48 | 31 % (≥ 8 glyphs) |
| ρ_g 0.25 | 0.67 | 77 % (≥ 3 glyphs) |

`BOX_SHARE_CAP = 0.75` is a constant in `src/train/stage.py`, per item, no
flag. Box area is ∝ glyph count, so `--box_weight` is already close to a
per-glyph share here (w 12 ≈ ρ_g 0.035 with a soft saturation).

### Reads

| | seed | w 1 | w 4 | w 12 | ρ_g 0.1 |
|---|---|---|---|---|---|
| `warm_drift` step 1000 → 2000 | – | 0.045 → 0.021 | 0.063 → 0.033 | 0.090 → 0.054 | 0.123 → 0.084 |
| row move mean / max | – | 0.021 / 0.18 | 0.033 / 0.48 | 0.054 / 0.69 | 0.084 / 0.85 |
| loss, step 1 → 2000 | – | 0.0995 → 0.0914 | 0.1026 → 0.0939 | 0.1088 → 0.0968 | 0.1261 → 0.1047 |
| `single` exact /8 | 6 | 6 | 5 | 5 | 4 |
| sub-exact lift `short` | – | +0.136 | +0.041 | +0.092 | +0.174 |
| sub-exact lift `short_held` | – | +0.092 | +0.113 | +0.065 | +0.045 |
| **sub-exact pooled (n = 16)** | – | **+0.114** | **+0.077** | **+0.079** | **+0.109** |
| native hit sfx / vl, /8 | 4 / 4 | 4 / 4 | 3 / 3 | 2 / 1 | 2 / 1 |
| native en cos / out | 0.920 / 0.923 | 0.919 / 0.923 | 0.918 / 0.921 | 0.924 / 0.930 | 0.925 / 0.929 |
| native box IoU | 0.06 | 0.07 | 0.17 | 0.27 | 0.28 |

Seed column = `rows_step1_0920_s53k`'s own reads on the same 8 singles and
the same 8 native renders. Pooled-lift differences: w 1 − w 12 = +0.035, CI
[−0.095, +0.149]; w 12 − ρ_g 0.1 = −0.031, CI [−0.178, +0.122]. Exact is
0/8 on `short` and `short_held` in every arm; `en` 8/8 in every arm.

- **Weight moves the table, monotonically**, and the rows that move are the
  ones `step2_0919` moved: ！！ ・・・・ っ だ ね ！ ？ が ー. At ρ_g 0.1 the
  move spreads across the frequent kana at ≈ 0.3 each (よ い が お す た あ
  え は て).
- **The anchor takes about half of it back.** Drift peaks near step 1 000
  and falls as the cosine lr decays, in all four arms: the end point is the
  balance between μ 0.3 and the in-box gradient, not the step count. A 6 k run
  on the same argv should land near the same place.
- **No sentence ruler orders the arms.** Pooled lift is flat inside its CI
  and `short_held` does not rise with weight. An ad-hoc recall pooled over
  every reader and box *did* rise on `short_held` (0.12 / 0.19 / 0.39 / 0.35)
  — that is read length: heavier arms emit longer reads, and the perm control
  removes it. It was reported mid-session as a trend and is withdrawn.
- **The failure shape does not change with weight**: several pieces fuse into
  one large pseudo-kanji (`次は` → 勲 / 畏爻木, `仕事も` → 香). A few w 12
  items separate a piece (`実に・・・・` → 東に…, `おおお玉や` → お); n is
  too small to call.
- **Singles and native pay.** `は` breaks at w ≥ 4 (ぱーぼよ → at ρ_g 0.1 a
  sentence fragment, だけわぃはよ; the seed already read には), `む` breaks at
  ρ_g 0.1; native 4 → 3 → 2 → 2 of 8, one render per step, monotone. The rows
  that move are the sentence-frequent kana, and they lose their lone-glyph
  function.
- **The scene is held** (en cos 0.92 in every arm), and the native box IoU
  rises 0.06 → 0.28: the glyph lands where the EN word sits, wrong glyph.

**Verdict (smoke-grade):** the in-box weight is not step 2's bottleneck. In
step 1 the in-box signal *is* the row's identity (one glyph per box), which is
why the share doubled it; in a sentence box the signal is split across rows
and what is missing is composition, which no weight buys. Ceiling for a 6 k
run is about w 12 / mean share ≈ 0.2.

## 3. Owed

- **`ρ_g 0.05` + cap 0.25** — same smoke frame. Cap reached at 5 glyphs, mean
  share ≈ 0.18 = w 12, with 1–2-glyph items (22 %) weighted above w 12. The
  read is whether singles pay less at w 12's sentence level. Needs
  `--box_share_cap` (default 0.75 keeps every run on record reproducible) in
  `src/cli/train.py` + `src/train/stage.py`. Same as w 12 → the weight axis
  is closed and 6 k runs on `--box_weight 12`.
- Not tested here, separate levers: the anchor μ (0.3 → 0.1 was worse on
  09-18, on the weak seed and the old weighting), a sentence-length
  curriculum, and the open question in `plan.md` — which loss trains a
  multi-glyph piece as a unit.
