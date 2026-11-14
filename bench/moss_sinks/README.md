# bench/moss_sinks

Attention sinks / registers on MOSS-Image: does a small set of keys take most of the
attention, and where is it? Tier 1 §1 of `_archive/proposals/moss_anima_negatives_revisit.md`.
Summary and hypotheses: `docs/moss/findings/chat_template.md`. This file is
the run log.

```bash
make daemon-run ARGS="bench/moss_sinks/probe_attention.py --label base"
```

## Probe

`probe_attention.py` (docstring has the details). The shipped sampler (Euler, CFG 3, uncond
skip 22, reference per-sample forward), Base, 2 prompts (`anime_sunset`, `ja_manga`) × 4
seeds × 30 steps at 1024², read on 8 of the 30 steps. On those steps every joint block's
attention is recomputed with an explicit softmax (the block output still comes from SDPA).
Keys are split into image / `tok0` (`<|im_start|>`) / `prefix` / `caption` / `suffix`; cond
and uncond are kept apart. Per block and head:

- the mass each query class (image / text) puts on each key class;
- the top-1 % key share: the head's mean attention per key over the image (or text)
  queries, summed over its top `ceil(0.01·N)` keys (uniform = 0.01);
- a per-token norm census on the block output with the 18 channels of
  `library/moss/calibration/massive_channels.npz` left out (with them in, DC 0.99 caps
  image max/median near 1 on blocks 0–18 by construction); raw norms are kept too.

**Gate**: on most blocks, ≥ 25 % of heads put ≥ 30 % of their image-query mass on ≤ 1 % of
the keys; or massive-channel-excluded image norm max/median ≥ 10 on most blocks. If neither
holds, close.

Output: `results/<stamp>-<label>/result.json` (gate, per-block head fraction, top-1 % share,
mass by key class, norms; cond and uncond), `per_block.npz` (per seed × step × block ×
head), `top_keys.json` (per block, the 8 keys receiving the most image-query mass, with
their class) and `probe.png`.

## Runs (2026-10-18)

**smoke** (`results/20261018-1553-smoke`: 512², 4 steps, 1 prompt × 2 seeds). Plumbing only:
at 512² the 10 template tokens are ~1 % of the keys, so the top-1 % share passes trivially.

**base** (`results/20261018-1601-base`, job `20261018-155337-1cf9d5`; steps 0 3 7 11 15 19
23 29). **Gate passes on attention (35 / 42 blocks), fails on norm (0 / 42).** The sink is
the 10 chat-template tokens. Raw per-key masses behind the ratios in the findings doc:
`<|im_start|>` 0.06–0.07, each prefix token 0.044–0.048, each suffix token 0.016–0.017, an
image token 0.0002, a caption token 0.0005–0.0006. The head-mean top-8 keys are template
tokens on nearly every block (`tok0` and both prefix tokens in all 42); image position 0
(top-left) reaches the top 8 on a few early blocks only. Template mass by step: step 0
0.31–0.32, mid 0.24–0.25, step 29 0.31–0.34. Raw image norm max/median 1.06–1.6.

**base-write** (`results/20261018-1616-base-write`, same shape, with write stats). Values
are centred on v̄_img first: on raw values every write is near-constant (image write
coherence 0.92–0.98 in the smoke), so raw shares cannot separate the two. **The template is
not a null sink: it writes a per-head bias.** Template write coherence is 0.34 on block 41;
`<|im_start|>` centred value norm 0.43–0.9 holds on blocks 0–37. Uncond's template takes a
little more mass and writes a little more (0.16–0.62 against 0.15–0.51).

**LoRA arms** (`results/20261018-1632-lora-lr1e4`, `results/20261018-1645-lora-lr5e5dc`;
same shape, `--lora` at ×1.0). `lr1e4_svd` (no channel scaling, no `dc_isolate`) and
`lr5e5_svd_cs05_dc` (the current defaults). Paired-by-seed step-0 shifts are resolved
(|t| > 4) on 35 / 42 and 33 / 42 blocks.

| | lr1e4_svd | lr5e5_svd_cs05_dc |
|---|---|---|
| step-0 template mass, mean \|Δ\| / base | 5.4 % | 3.0 % |
| largest block shifts | −20 % (6), −19 % (5, 17), +18 % (23), −12 % (1) | −11 % (8), −9 % (40), +8 % (7) |
| pooled mass / write, mean \|Δ\| / base | 7.1 % / 4.8 % | 4.2 % / 2.5 % |
| template write vs base, cosine (mean, min) | 0.905, 0.825 (41) | 0.948, 0.852 (41) |
| template write norm ratio | 1.06 (0.78–1.33) | 1.04 (0.92–1.14) |

The pooled cosines include the LoRA's change of trajectory (different images from step 1
on), so they are a lower bound on how well the write direction is kept.

## Next

1. **Does the prompt move it**: caption length and format (short tag bag, long layout JSON)
   against the template mass and the per-block head counts.
2. **Only then** anything that edits the template (a shortened or altered template, Null-TTA
   on the template span); see the findings doc's hypotheses.
3. **LoRA and the template rows**: `library/moss/lora.py` wraps whole linears, so the 10
   template rows get the LoRA delta in every block's q / k / v / o. Masked renders
   (`_archive/bench/moss_sinks/render_template_mask.py`) are done; open: whether masking is
   better or worse, read on layout-JSON prompts and more seeds.

Dropped: test-time registers (the model already ships fixed registers).
