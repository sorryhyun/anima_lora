# probe_scene: how much of a row's gradient the scene sets (2026-10-06)

From the user (10-06, after `probe_cf_2026_10_06.md`): the row gradient
scatters across items — how much of that is scene diversity?
`probe_scene.py`: the rows at `seed_fixed_1005_stick080` (f0's start), 8
kana + 8 kanji rows; per row a crossed block of **6 dialogue lines** (8–10
cells, the glyph once, from `sent_kanji`'s `sent` texts) × **6 scenes**
(448×640, the scale pool's whole bubbles, every line fitted by `_sent_plan`
at 30 px) × **4 noise draws** (σ 0.50 / 0.55 / 0.60 / 0.65, one ε each,
shared by every cell). "Scene" = what it is at training: the image, its tag
caption (`scene_caption`) and the layout the fit gives there. One font per
row. Batch-1 forward per draw, the trainer's box-share loss, the target
row's gradient read; no step. 2 304 draws, 5.6 min on the daemon →
`output/cjk_anima_reseed/probe_scene/sc1/`.

Read: a three-way random-effects ANOVA on the gradient vectors (variance
summed over the 1 024 dims), the row's common direction ("signal", ‖μ̂‖² less
its sampling share) beside it; and the mean cos of two draws by what they
share.

| share of the variance (median, 16 rows) | |
|---|---|
| signal (the row's common direction) | 0.026 |
| line | 0.097 |
| **scene** | **0.001** |
| noise (the same σ, ε across cells) | 0.002 |
| line × scene | 0.017 |
| line × noise | 0.017 |
| scene × noise | 0.029 |
| line × scene × noise (each draw's own) | **0.823** |

| two draws share | nothing | scene | noise | line | line + noise | line + scene (same image) |
|---|---|---|---|---|---|---|
| mean cos | 0.033 | 0.038 | 0.034 | 0.141 | 0.157 | 0.159 |

Kana and kanji alike; every row of the 16 has scene ≤ 0.021 and the
three-way term 0.73–0.88.

- **The scene sets nothing on its own.** Main effect 0.001; two draws on the
  same scene with a different line agree as two unrelated draws do (0.038
  vs 0.033). With its interactions the scene accounts for ~5 %.
- **The line sets ~10 %**: the same line on another scene, at another
  noise, agrees at cos 0.14.
- **Most of a draw is its own (σ, ε) on its own image.** The same image
  under another noise draw agrees at 0.16 — barely over the same line on
  another scene. And a noise draw has no common effect across images
  (0.002): the same ε does not push the row the same way on two canvases.
- **The signal share reproduces** `probe_geom`'s per-draw ρ ≈ 0.025 on a
  third, independent design.

## Box-share against plain MSE (user, 10-06)

The scene's pixels sit outside the box, and the box-share loss gives the
box s = 0.50 at these lines against an area share of 0.04 — the scene
could be null only because the loss barely sees it. Re-run with the two
summands' gradients kept apart (`grad` stores `g_in` / `g_out`; same
renders, same σ / ε; 8.5 min), read four ways: the in-box term, the
out-box term, the box-share sum, and **plain MSE** rebuilt from the same
two (∇mean_in · a + ∇mean_out · (1 − a)). Under plain MSE the out-box part
is 0.31 of the in-box part's norm per draw (box-share: 0.013).

| view (median, 16 rows) | signal | line | scene | scene × noise | three-way | cos same image | cos same scene |
|---|---|---|---|---|---|---|---|
| box-share (trained) | 0.027 | 0.094 | 0.001 | 0.031 | 0.827 | 0.159 | 0.038 |
| in-box term | 0.027 | 0.095 | 0.001 | 0.031 | 0.827 | 0.160 | 0.038 |
| out-box term | 0.002 | 0.001 | 0.002 | 0.030 | 0.962 | 0.038 | 0.025 |
| plain MSE | 0.021 | 0.067 | 0.001 | 0.029 | 0.859 | 0.135 | 0.036 |

- **Not the weighting.** Under plain MSE the scene's main effect is still
  0.001. What the out-box share adds is noise: signal 0.027 → 0.021, line
  0.094 → 0.067.
- **The out-box gradient has no direction at all** — signal 0.002, 96 % its
  own draw. What structure it has is scene × noise (two draws sharing scene
  and ε: cos 0.148): the scene's pixels under that one ε, never a scene
  direction across ε.

## Read

Scene diversity is not where the scatter comes from, and not a lever on it:
fewer scenes would not quiet the gradient and more would not add signal.
f0's `sent_34` (53 920 items) sits on 202 scenes, ~270 items a scene; by
this read that costs the row gradient nothing. The scatter is the noise draw
on the image — the per-draw part `probe_geom`'s Open asked about — and the
line context is the one item-level factor that carries direction.

Not read: the scene's image apart from its caption (they move together
here); other canvas shapes and px; whether the scene matters for *what the
picture looks like* (layout, the base's co-text) — this read is the rows'
gradient only.
