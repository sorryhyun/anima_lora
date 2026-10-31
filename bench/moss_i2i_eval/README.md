# bench/moss_i2i_eval

Held-out evals for MOSS i2i LoRAs. Moved from `project/finished/moss_lora/src/`
(2026-10-28). Recipes and results: `docs/moss/methods/i2i.md`, `docs/moss/methods/tone_loss.md`,
`project/finished/moss_lora/{README,colorize_ocr}.md`.

| Script | What it measures |
|---|---|
| `colorize_eval.py` | per held-out page of a `colorize_cache.py` cache, one arm per caption kind the LoRA trained on (`base`, `lora_ocr`, `lora_sparse`, `lora_bare`, …): luma r (128 px) and chroma error against the target, chroma |
| `colorize_wild.py` | colorize LoRAs on real B&W pages (no target, no caption): luma r against the page, chroma as mean CbCr distance from grey |
| `colorize_run.py` | one daemon job: `scripts/moss/colorize_cache.py` → `scripts/moss/train.py` → `colorize_eval.py` on the final LoRA |
| `tone_map.py` | tone drift of i2i renders (sanitize, swap_text) on the pixels no compared run changed: global ΔL / Δb / sat×, per-128 px cell spread, ΔL by input L, median ΔE |

```bash
make daemon-run ARGS="bench/moss_i2i_eval/colorize_eval.py --lora output/ckpt_moss_i2i/<name>.safetensors --cache post_image_dataset/moss/<name>"
make daemon-run ARGS="bench/moss_i2i_eval/colorize_wild.py --pages a.png b.webp --loras <lora> ..."
make daemon-run ARGS="--queue --stall-timeout 1800 bench/moss_i2i_eval/colorize_run.py --name <name>"
.venv/bin/python bench/moss_i2i_eval/tone_map.py --runs a=output/tests/<dir a> b=output/tests/<dir b>
```

Output goes to `output/tests/colorize_<lora>/` (base-arm latents shared under
`output/tests/colorize_base/<cache>/`) and `output/tests/colorize_wild/`, not a `results/`
envelope; `index.json` holds the scores. Renders resume from latents keyed by the LoRA and
settings. `tone_map.py` reads the render dirs and needs no GPU.

Last baseline (colorize v4 `moss_colorize_3k_textref1_iso_muon_ema`, `lora_ocr`, seed 1,
ER-SDE + Spectrum + RefCache, 30 steps): luma r 0.995, chroma error 11.2 (8 illustrations
11.5 / 4 manga 10.5). Sanitize v3's tone numbers: `docs/moss/methods/tone_loss.md`.
