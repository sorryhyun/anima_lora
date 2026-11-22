# bench/moss_i2i_eval

Held-out evals for MOSS i2i LoRAs. Recipes and results: `docs/moss/methods/i2i.md`, `docs/moss/methods/tone_loss.md`,
`project/finished/moss_lora/{README,colorize_ocr}.md`.

| Script | What it measures |
|---|---|
| `colorize_eval.py` | per held-out page of a `colorize_cache.py` cache, one arm per caption kind the LoRA trained on (`base`, `lora_ocr`, `lora_sparse`, `lora_bare`, …): luma r (128 px) and chroma error against the target, chroma |
| `colorize_wild.py` | colorize LoRAs on real B&W pages (no target, no caption): luma r against the page, chroma as mean CbCr distance from grey |
| `colorize_run.py` | one daemon job: `scripts/moss/colorize_cache.py` → `scripts/moss/train.py` → `colorize_eval.py` on the final LoRA |
| `tone_map.py` | tone drift of i2i renders (sanitize, swap_text) on the pixels no compared run changed: global ΔL / Δb / sat×, per-128 px cell spread, ΔL by input L, median ΔE |
| `vision_detect_probe.py` | TE only: a linear probe on Qwen3.5's image rows of a texted sanitize page predicts, per 32 px cell, where the clean twin differs (folds by artist); arms image alone / caption → image / neutral caption → image / a quarter of the pixels, against the cached reference latent; heat maps on the five eval pages |

```bash
make daemon-run ARGS="bench/moss_i2i_eval/colorize_eval.py --lora output/ckpt_moss_i2i/<name>.safetensors --cache post_image_dataset/moss/<name>"
make daemon-run ARGS="bench/moss_i2i_eval/colorize_wild.py --pages a.png b.webp --loras <lora> ..."
make daemon-run ARGS="--queue --stall-timeout 1800 bench/moss_i2i_eval/colorize_run.py --name <name>"
make daemon-run ARGS="--queue --stall-timeout 600 bench/moss_i2i_eval/vision_detect_probe.py"
.venv/bin/python bench/moss_i2i_eval/tone_map.py --runs a=output/tests/<dir a> b=output/tests/<dir b>
```

Output goes to `output/tests/colorize_<lora>/` (base-arm latents shared under
`output/tests/colorize_base/<cache>/`) and `output/tests/colorize_wild/`, not a `results/`
envelope; `index.json` holds the scores. Renders resume from latents keyed by the LoRA and
settings. `tone_map.py` reads the render dirs and needs no GPU.

Last baseline (colorize v4 `moss_colorize_3k_textref1_iso_muon_ema`, `lora_ocr`, seed 1,
ER-SDE + Spectrum + RefCache, 30 steps): luma r 0.995, chroma error 11.2 (8 illustrations
11.5 / 4 manga 10.5). Sanitize v3's tone numbers: `docs/moss/methods/tone_loss.md`.

`vision_detect_probe.py`, 2026-11-14 (`output/tests/vision_detect_probe/20261114-1411`, 189 pairs,
int8 TE, +1,007 image rows per page): held-out AP 0.669 for the image rows against 0.708 for the
reference latent read linearly, recall of faint cells at 1 % false positives 0.094 against 0.229.
Caption → image 0.665, neutral caption → image 0.661, a quarter of the pixels 0.496. `ref_vision`
was not trained on sanitize.
