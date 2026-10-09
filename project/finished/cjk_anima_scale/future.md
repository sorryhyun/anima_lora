# future.md — directions past the retrain (not planned, not scheduled)

Ideas the line may need once the retrain seed (`retrain_kana` + `retrain_kanji_b1..b3`)
is baked. Nothing here has a run, a budget or a gate yet; each entry says what
motivated it and what it would answer first.

## 1. Real images do not train rows directly (2026-09-28)

`experiments/real_kana` (smoke, `native_sig12`): the 174 kana rows warm from
`retrain_kana`, trained on the 291 kanji-free OCR images of the training set
(native size, batch 1, the LoRA trainer's sigmoid σ, captions verbatim, loss
box = the quoted lines' union). Every read fell to ≈ 0 against `retrain_kana`
(words official 16 → 0 / 104, singles contained 73 → 14 / 112, target はい
5 → 0 / 8): renders keep "Japanese text here" and the bubble, and lose which
glyphs and how many. Rows moved mostly per row (84 % of the change; centred
row cos 0.82), the rest a shared shrink (norm 249 → 169, the shared direction
of `reports/row_geometry_2026_09_28.md`). It reproduces `cjk_aware_anima`
§ 14 ("row content is inert" for real manga text) on a rows-only trainer.
The plain-MSE arm (`native_sig_mse12`, no in-box weight; stopped at 1 700 /
2 088, unread, daemon job `20260928-170449-bd552c`) moved the rows **further**,
not less: warm_cos 0.52 vs 0.74 at 1 700 (drift 0.75 vs 0.55).

Reading (user, 2026-09-28): a real image is not a self-generated one, so its
FM loss is high everywhere — the frozen DiT cannot reach it — and the
trainable rows are the only place that residual can go. They absorb the
image's whole mismatch, not the text's; taking the in-box weight away hands
them more of the scene's. On a render the DiT made itself the loss is low
outside the text, and what is left for the rows is the text.

So a row learns identity from renders it can be sure of (one string, one
box, known glyphs, in-domain for the DiT), and real pages give it none of
that. Two ways past that:

## 2. OCR reward (RL) on the rows

If real, in-domain native text cannot supervise a row through the FM loss,
the signal may have to come from reading the render instead: generate with
the rows, read with the OCR readers (sfx / VL), reward the read against the
caption's string, and update the rows only. The readers already score every
ruler; what is missing is the gradient path (a policy-gradient / reward-
weighted update over σ-trajectories, or a differentiable-reader proxy) and
a guard against the known reward hacks (`product_criteria.md`: a white-box
glyph pasted over the scene reads as a hit).

First question: does a reward-weighted update on a handful of rows (`p1_mix`'s
donors) move their official reads without the pasted-box hack, at a render
cost the local GPU can pay?

## 3. Token scaling with a self-generated set, as the last stage

The line trains at ≈ 1 k tokens (512² shapes); users render at 1024-tier
(≈ 4 k tokens) and above. A final stage would scale the rows across
resolutions — 1 024 → 2 048 → 4 096 tokens — on a **self-generated** set:
the model's own renders with the current rows, kept only where the readers
confirm the string (one string, one box, read exact), so every item is
in-domain for the DiT and certain for the row. Each resolution step would
retrain the rows on that step's own confirmed renders.

Piloted at 4 k on the kana rows in `plan_polish.md` (archived 2026-10-01;
read in `reports/polish_seed_2026_09_30.md`).
First question: at 4 k tokens, do the retrain seed's rows read as they do at
1 k (the `single` / `target` rulers rendered at 1024-tier), and if not, which
glyph sizes lose them?
