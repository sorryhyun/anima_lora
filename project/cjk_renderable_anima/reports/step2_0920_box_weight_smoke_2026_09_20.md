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

### The owed arm: ρ_g 0.05 + cap 0.25 (2026-09-20 14:34, job `20260920-143430-d8d588`)

`rows_step2_0920_smoke_plain_bs05c25_s2k` — the bs10 arm's argv with
`--box_share 0.05 --box_share_cap 0.25` (the flag is new, default 0.75).

| | w 12 | ρ_g 0.05 cap 0.25 |
|---|---|---|
| `warm_drift` step 1000 → 2000 | 0.090 → 0.054 | 0.083 → 0.051 |
| loss, step 1 → 2000 | 0.1088 → 0.0968 | 0.1076 → 0.0962 |
| `single` exact /8 | 5 | 5 |
| sub-exact lift `short` / `short_held` | +0.092 / +0.065 | +0.151 / +0.152 |
| **sub-exact pooled (n = 16)** | **+0.079** | **+0.151** |
| native hit sfx / vl, /8 | 2 / 1 | 3 / 3 |
| native en cos / out | 0.924 / 0.930 | 0.920 / 0.925 |
| native box IoU | 0.27 | 0.16 |

Pooled lift difference +0.073, CI [−0.066, +0.207], P(>0) 0.86. It lands on
w 12's drift, as designed. Singles break on the same item (`は` → a sentence
fragment; `む` holds, which ρ_g 0.1 lost; `の` → あの and `み` → ん are the
seed's own misses in every arm). Native is one render above w 12 (あ 2/4,
か 1/4). Per-item sheets show the same failure shape as every other arm —
fused pseudo-kanji (`次は` → 悶災木罵は, `彼を・・・・` → 複), `実に・・・・` →
東に… as at w 12 — and exact stays 0/16.

**Read: the same as w 12 inside this frame's noise** — nowhere worse, the
pooled lift nominally the highest of the five arms but inside the CI, and
w 1 (+0.114) already showed the lift does not follow weight. The weight axis
is closed: no in-box weight orders a sentence ruler at n = 8.

### ρ_g 0.1 + cap 0.33, and the pick for the 6 k run (2026-09-20 14:59, job `20260920-145948-d6527f`)

`rows_step2_0920_smoke_plain_bs10c33_s2k` — user's arm: ρ_g 0.1 with the cap
at 0.33 (reached at 4 glyphs). The three `--box_share` arms side by side:

| | ρ_g 0.05 cap 0.25 | ρ_g 0.1 cap 0.33 | ρ_g 0.1 cap 0.75 |
|---|---|---|---|
| `warm_drift` step 1000 → 2000 | 0.083 → 0.051 | 0.100 → 0.065 | 0.123 → 0.084 |
| `single` exact /8 | 5 | 6 | 4 |
| sub-exact lift `short` | +0.151 | +0.157 | +0.174 |
| sub-exact lift `short_held` | +0.152 | +0.097 | +0.045 |
| **sub-exact pooled (n = 16)** | **+0.151** | **+0.127** | **+0.109** |
| native hit sfx / vl, /8 | 3 / 3 | 2 / 1 | 2 / 1 |
| native en cos / out | 0.920 / 0.925 | 0.925 / 0.930 | 0.925 / 0.929 |
| native box IoU | 0.16 | 0.28 | 0.28 |

`short` is flat across the three; what falls as the long-item share rises is
**`short_held`** (+0.152 → +0.097 → +0.045), and native with it. All inside
n = 8 noise, but the order is the same on both. Pick (rule fixed before the
read: pooled lift first, singles and native as the constraint):
**ρ_g 0.05 cap 0.25** — highest pooled and held lift, best native, singles one
item under cap 0.33. The 6 k Round 2 run is on it: job
`20260920-152026-39bf6f`, arm `rows_step2_0920_plain_bs05c25_6k`
(`step2_0919`'s train argv, seed `step1_0920`, `--box_share 0.05
--box_share_cap 0.25`).

### Round 2 at 6 k: `rows_step2_0920_plain_bs05c25_6k` (job `20260920-152026-39bf6f`, 62.5 min)

| | `step2_0919` (weak seed, w 4) | Round 2 (`step1_0920` seed, ρ_g 0.05 cap 0.25) |
|---|---|---|
| `single` / `_ext` / `_kanji` / `_small` of 36 | 10 / 4 / 2 / 0 (seed 10 / 5 / 3) | **15 / 7 / 6 / 0** (seed 20 / 8 / 8) |
| lift `short` / `short_held` | +0.125 / +0.141 | +0.182 / +0.108 |
| lift `phrase` / `phrase_held` | +0.031 / +0.049 | +0.071 / +0.033 |
| **pooled lift (n = 64)** | +0.086 | **+0.099** |
| exact, multi-glyph groups | 0/64 | 0/64 |
| `warm_drift` 1000 / 3000 / 6000 | – / – / 0.054 | 0.102 / 0.080 / 0.042 |

Pooled difference +0.012, CI [−0.050, +0.075], P(>0) 0.65. **The gate fails on
both clauses**: the lift is not above `step2_0919`'s, the `_held` groups fall
(the gain is on the trained groups only — `short` +0.057, `phrase` +0.040,
`short_held` −0.033, `phrase_held` −0.016), and singles drop a quarter below the
seed (20 → 15 on `single`) where `step2_0919` held its seed flat. A seed table
twice as good on singles buys the sentence step nothing measurable; the
sentence pass still costs singles in proportion to how far it moves the rows
(end drift 0.042, row cos 0.997).

**Native on the 6 k arm** (job `20260920-162359-446936`, `step1_0920`'s native
argv: あ か す 日 × `en,swap` × 2 seeds, both readers of 64):

| | seed `step1_0920` | Round 2 6 k |
|---|---|---|
| `en` both / `swap` both | 19 / 8 | **9 / 4** |
| あ `en` / `swap` of 16 | 7 / 6 | 3 / 2 |
| か | 7 / 0 | 3 / 0 |
| す | 5 / 1 | 1 / 1 |
| 日 | 0 / 1 | 2 / 1 |
| en cos `en` / `swap` | 0.934 / 0.967 | 0.932 / 0.971 |

Native halves (kana 19 → 7 on `en`; 日 0 → 2 is inside noise) with the scene
held. The sentence pass gives back about what `--box_share` had bought step 1
on native — the 6 k arm reads like `step1_0919` (8 / 4), not like its seed.

**Multi-glyph native, seed vs Round 2** (jobs `20260920-163649-b806df` /
`…-43e836`, `native_sent/`: はい おしい やったネ ちょっと来い × 8 scene prompts ×
2 seeds, clause `en`, the four shortest of `step2_0919`'s six strings):

| | seed `step1_0920` | Round 2 6 k | `step2_0919` |
|---|---|---|---|
| はい both / sfx, of 16 | 1 / 2 | 0 / 0 | 2 |
| おしい · やったネ · ちょっと来い | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 |
| sub-exact lift (n = 64) | +0.093 | +0.084 | – |
| en cos | 0.925 | 0.928 | – |

No difference (lift CI [−0.068, +0.065]). On the はい sheets the seed answers
with one large glyph (は alone 4×, one clean はい at p04 s1); Round 2 answers
with smaller multi-glyph lines that carry the pieces in the wrong company
(はかい, ばい, ぱばい、, は婿れないい) and loses the clean hit. The sentence pass
changes the *habit* — one glyph → a line — not the content.

### μ 0.1 at 6 k, with the in-box / out-of-box split log (job `20260920-164206-43ab2d`, 71 min)

`rows_step2_0920_plain_bs05c25_mu01_6k` — the Round 2 argv with
`--init_anchor 0.1`, eval + native in the same job. `BoxSplit`
(`src/train/stage.py`, new) logs the plain residual's per-item in-box and
out-of-box mean squares averaged between log rows, plus the in-box mean at
σ ≥ / < 0.7.

| | μ 0.3 | **μ 0.1** |
|---|---|---|
| `single` / `_ext` / `_kanji` / `_small` of 36 (seed 20 / 8 / 8 / 0) | 15 / 7 / 6 / 0 | 14 / 7 / 6 / 0 |
| lift `short` / `short_held` | +0.182 / +0.108 | +0.220 / +0.165 |
| lift `phrase` / `phrase_held` | +0.071 / +0.033 | +0.159 / +0.067 |
| **pooled lift (n = 64)** | +0.099 | **+0.153** |
| exact, multi-glyph groups | 0/64 | 0/64 |
| native `en` / `swap` both of 64 (seed 19 / 8) | 9 / 4 | **5 / 2** (あ 0/32) |
| native en cos | 0.932 | 0.929 |
| end `warm_drift` / row cos | 0.042 / 0.997 | 0.089 / 0.990 |

Pooled lift vs μ 0.3: +0.054, CI [−0.009, +0.119], P(>0) 0.95; vs
`step2_0919`: **+0.066, CI [+0.003, +0.129]** — the first sentence arm whose
lift clears the old run's CI, and all four groups rise, the `_held` ones
included (the reward-hack check passes). Eval singles are unchanged against
μ 0.3; **native single glyphs halve again** (19 → 9 → 5, あ 7+6 → 0).

Split log (1 k-step means): `in_box` 0.1526 → 0.1488 → 0.1467 → 0.1446 →
0.1442 → **0.1421** (−7 %, still falling at the end, no plateau);
`out_box` 0.0874 → 0.085 by step 1 000 and flat after. Both σ halves fall by
the same relative amount (σ ≥ 0.7 −5.5 %, < 0.7 −5.8 % at step 3 k) — no sign
that σ 0.5–0.7 is wasted. The mixed `loss` shows none of this (out-of-box is
≈ 80 % of it, batch noise ±0.013): "the sentence loss is flat" was a statement
about the log, not the rows. μ 0.3 has no split log, so whether the looser
anchor learns *faster* in-box is not read.

The trade is now measured on both sides: the anchor is what holds the
single-glyph function *and* what holds the sentence lift down. The run is not
converged in-box at 6 k.

**Multi-glyph native on μ 0.1** (job `20260920-175915-c4615a`, same four
strings): exact 0/64, sub-exact lift **+0.084** — the same as the seed (+0.093)
and μ 0.3 (+0.084). The eval-frame lift (+0.153) does not reach a scene prompt.
Whole-canvas reads, はい: the bigram appears in 1 / 0 / 2 of 16 renders (seed /
μ 0.3 / μ 0.1), and in μ 0.1 it is the *tail of a longer invented line*
(`笺どおありはい`, `笺こたをりはい`); first glyph present 8 / 4 / 4 of 16. やったネ
and ちょっと来い: no first glyph in any table (0–1 of 16).

### The boost gate: `rows_step2_0920_plain_bs05c25_boost8_6k` (jobs `20260920-203327-b24831`, `…-5521aa`)

Round 2's μ 0.3 argv + `--row_boost 1101,192,831,145,237,672,521,220`
(ろ 事 カ そ ら め 知 も, each lifted from 259–787 to 1 007–1 612 expected
draws). **Gate failed: the boosted rows gained nothing.**

`row_dose.py --rows …` against `rows_step2_0920_seedread`, paired on the same
64 items:

| bin | partner (Round 2 μ 0.3) | μ 0.1 | **boost8** |
|---|---|---|---|
| the 8 rows, all (44 pieces) | −0.023 [−0.10, +0.05] | −0.023 [−0.11, +0.04] | **−0.023 [−0.07, +0.00]** |
| the 8 rows, held (24) | −0.042 | −0.042 | **−0.042** |
| occ 400–1 000, other rows | +0.148 | +0.227 | +0.148 [+0.05, +0.24] |
| occ ≥ 1 000 | +0.031 | +0.172 | −0.016 |

Hit 3/44 (seed) → 2/44 in all three tables. The pre-registered pass was
≈ +0.15 with the interval off 0.

The boost did reach the rows. Drift `|Δ|/|seed|` of the eight: partner
0.034–0.139 → boost **0.093–0.149** (table median unchanged, 0.022 → 0.021),
in the partner's direction (cos(Δ_boost, Δ_partner) 0.53–0.76) and against the
seed like every heavy row (cos(Δ, seed) −0.09 … −0.30). μ 0.1 moves the same
rows as far or further (0.089–0.336) for the same −0.023. **Two levers that
each double these rows' travel — draws, anchor — buy no content.**

Everything else is the partner's, inside noise: `sub_exact.py` pooled
**+0.095** vs +0.099 (Δ −0.004 [−0.069, +0.064]); eval singles 16 / 7 / 7
(partner 15 / 7 / 6); native あ か す 日 both-readers **9 / 4** of 64
(partner 9 / 4); multi-glyph native exact 0/64, sub-exact +0.088 (partner
+0.084). The × 0.78 share cut on unboosted items cost nothing measurable.

What the arm does not separate: the boost repeats a row's *own* strings. In
`data_step2_0919` the eight rows ride **13–51 distinct strings** (median 30);
the rows that gained at 400–1 000 items ride 22–123 (median 60), the ≥ 1 000
rows a median of 106. Draw count was matched to the gaining bin; string
variety was not. So item 8's slope is not draws per row; "distinct contexts per
row" is the reading still standing, and it is the pool's question, not the
sampler's.

## 3. Owed

- ~~`ρ_g 0.05` + cap 0.25~~ — ran, above; the axis is closed.
- Not tested here, separate levers: the anchor μ (0.3 → 0.1 was worse on
  09-18, on the weak seed and the old weighting), a sentence-length
  curriculum, and the open question in `plan.md` — which loss trains a
  multi-glyph piece as a unit.
- ~~The boost gate~~ — ran, above: draws per row are not the lever.
