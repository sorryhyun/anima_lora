# Qwen-Image-2.1 LoRA — first end-to-end run (2026-09-21)

`channel_(caststation)`, 30 images, 8 epochs on a 16 GB RTX 5070 Ti. The pipeline runs and
the adapter is live; the **eval prompt set turned out not to be a usable ruler**, so the
style verdict is inconclusive rather than negative. State: `README.md`. Scope rules that
differ from the rest of the repo: `CLAUDE.md`.

## What exists

| File | Purpose |
|---|---|
| `library/qwen21/loader.py` | 33 GB checkpoint on a 16 GB card — phase split, `place()` |
| `library/qwen21/blockswap.py` | `ModelOffloader` on stock diffusers/transformers block lists |
| `library/qwen21/accel.py` | attention backend + per-block `torch.compile` |
| `library/qwen21/lora.py` | adapters held **outside** the swapped blocks; `set_multiplier`, `load_network` |
| `src/backward_smoke.py` | the fit gate — one training step on random tensors |
| `library/qwen21/cache.py` | Qwen3-VL embeddings + VAE latents, two files per image |
| `library/qwen21/train.py` | flow matching, batch 1, `report_fit` after step 1 |
| `src/generate.py` | A/B eval, `--multipliers 1.0,0.0` from one model |

## Measured envelope

| | 512² / 1088 tok | 1024² / ~4100 tok |
|---|---|---|
| swap 12–14 | peak 9.59 GB, 1.18 s/step | peak 10.32 GB, 4.00 s/step |
| swap 5 | — | peak 13.96 GB, 3.93 s/step |
| bound by | **PCIe** (12 × 0.41 GB × 2 ≈ 10 GB/step) | **compute** (100 % util, 275 W/300 W) |
| block compile | ±0 (7 s to pay for it) | not retried; off, token count moves per sample |

Swap count is a VRAM lever at both sizes and a *speed* lever only at 512². Inference is a
separate regime: 20 steps at 896×1184 ran 1.18 it/s ≈ 17 s/image.

## Runs

**Cache** — 30 pairs. Text: 112–346 tokens, 20.2 s, peak 12.39 GB (TE block-swapped, 12 of
36). Latents: 12 distinct shapes, 3996–4216 tokens, 18.7 s, peak 0.78 GB.

**Train** — rank 16 / alpha 16, 224 linears, 41.9M params **bf16**, lr 1e-4 constant after
24 warmup steps, 240 steps, swap 5, activation checkpointing, compile off. 15.7 min, peak
13.96 GB.

| e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8 |
|---|---|---|---|---|---|---|---|
| 0.477 | 0.283 | 0.272 | 0.250 | 0.232 | 0.223 | 0.226 | 0.209 |

**Eval** — 12 prompts × {1.0, 0.0}, 896×1184 (4144 tokens, mid-band), 20 steps, CFG off,
seed fixed per prompt. 7.1 min denoise + 26.9 s decode, peak 12.54 GB.

## What the eval showed

1. **The adapter is live.** All 12 pairs differ: mean |Δ| 6.0–28.6 per channel, 9–48 % of
   pixels past a 16/255 threshold.
2. **Style transfer is modest.** On the classroom prompt the *base* already renders a
   competent anime illustration; the adapter sharpens linework and moves eye style and
   palette, but does not read as a decisive artist transfer at 8 epochs.
3. **The prompt set is not a ruler.** It was `bench/grad_init/e1_general_prompts.txt` with
   the artist tag substituted — an Anima danbooru tag bag with a `safe`/`sensitive`/`nsfw`
   rating prefix. On this model:
   - the rating prefix is **inert**. Qwen3-VL is a natural-language encoder and does not
     read it as a rating, so `safe` constrains nothing.
   - a prompt with no clothing tags (`1girl, solo, upper body, portrait, simple
     background`) renders nude **in the base arm too** — that is the base model filling an
     underspecified prompt, not the adapter.
   - style is unstable across the set: one prompt went anime, another photoreal.

   Any eval prompt here has to specify clothing and setting explicitly, in natural
   language, or it measures the base model's defaults rather than the adapter.

## Open

- **Trigger isolation** (cheapest, most informative): same prompts at multiplier 1.0 with
  and without `@channel (caststation)`. Says whether the adapter is bound to the trigger or
  pushing style globally. 12 images.
- **A Qwen-shaped eval set**: natural language, clothing and setting specified. Replaces
  the ruler without touching the caption format.
- **The caption format itself**: the 30 cached embeddings are the same danbooru tag bags.
  Whether that is the right input for an NL encoder is untested — re-caching text is 20 s
  and a retrain is 16 min, so it is affordable, but only worth doing once trigger
  isolation says what the adapter currently keys on.

## What cost time, for next time

Every wrong turn was an Anima assumption carried into a different model, which is why
`CLAUDE.md` in this directory now lists them as a delta table:

- centre-cropped a portrait folder to 1:1 — Anima's own native-aspect rule was the correct
  one and was not applied.
- pushed block compile as the OOM lever; activation checkpointing is what fits 32 blocks at
  4096 tokens here.
- set `true_cfg_scale` 4.0 on a model whose pipeline default is 1.0 with no guidance
  embedding anywhere in the transformer.
- sized the swap from a guessed `activation_reserve_gb` and reported peak only at the end
  of an epoch, so a wrong guess cost minutes instead of seconds. `report_fit` now prints
  after step 1, against `mem_get_info` free — not `total - max_allocated`, which ignores
  the allocator reserve (~0.7 GB) and the desktop (~0.5 GB) and recommends a swap that OOMs.
