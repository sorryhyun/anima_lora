# bench/moss_ab

LoRA and sampler A/B grids on MOSS-Image: same noise per (prompt, seed), one column per
arm, so a row differs by the arm only. Moved from `project/finished/moss_lora/src/`
(2026-10-28); findings that used them: `docs/moss/findings/lr_clip_sweep.md`,
`docs/moss/findings/spectrum_schedule.md`, `project/finished/moss_lora/done/{sampler,swift}.md`.

| Script | What it renders |
|---|---|
| `lora_ab.py` | each prompt × seed over `--arms` (`<sampler>:<multiplier>`, sampler `euler` / `er_sde` / `cns`), Base and SFT; `--stream fp32` runs the residual stream, modulation and norms in fp32 (out dir `_fp32s`; `docs/moss/findings/bf16_residual_stream.md`) |
| `ab_cols.py` | one sheet per cell from several `lora_ab.py` grid dirs, side by side (stacked when wide) |
| `schedule_ab.py` | the pipeline's snapped σ grid against ComfyUI's `normal` / `sgm_uniform` / `simple`, base DiT, no LoRA; `--ref` adds a fine-grid solve and per-arm rel-L2 |
| `_render.py` | the shared render loop (DiT + LoRA + block compile, ER-SDE + Spectrum, latent cache keyed by its settings, labelled grid); `bench.moss_i2i_eval` and `correction_ab.py` import it |

```bash
make daemon-run ARGS="bench/moss_ab/lora_ab.py --lora output/ckpt_moss/<name>.safetensors"
.venv/bin/python bench/moss_ab/ab_cols.py <out_dir> <glob> a=<grid dir> b=<grid dir>
make daemon-run ARGS="--stall-timeout 900 bench/moss_ab/schedule_ab.py"
```

Output goes to `output/tests/ab_<lora>_<steps>st/` (`lora_ab.py`) and
`output/tests/schedule_ab/`, not a `results/` envelope (`schedule_ab.py --ref` also writes
`result.json` there). Latents are cached under a digest of the LoRA (path + mtime), prompts
and sampler settings, so a rerun renders only what changed. Compare compiled arms only
against compiled arms.

`space/`: the two HF Space gallery prompts (`004`, `017`) replayed with
`scripts/moss/generate.py --example` (see `space/README.md`).
