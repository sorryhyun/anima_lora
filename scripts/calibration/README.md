# scripts/calibration

Scripts that regenerate the shipped artifacts under `networks/calibration/`.
They import the `anima_lora` façade + `library` / `networks` directly; the `moss_*`
scripts also use `bench/_common.py` (run envelopes) and `bench/_moss.py` (prompts) and
write runs under `output/calibration/`. Probe/analysis history stays in `_archive/bench/`.

## CNS — colored-noise sampling γ matrix

Produces `networks/calibration/cns_gamma.npz` (consumed by `--cns auto`).

```bash
python scripts/calibration/cns_calibrate.py --cfg 4.0 --n_aspects 3   # compiled
```

- `cns_calibrate.py` — Phase-1 deploy calibrator (cfg=4.0, top-N aspect buckets);
  writes the npz (per-aspect σ50 staircase summary prints to stdout).
- `gamma_probe.py` — read-only Phase-0 staircase check + the shared γ/FFT helpers
  `cns_calibrate` imports. `--out_dir` for its standalone npz/heatmaps.

Phase log / precondition / composition tensions: `_archive/bench/cns/plan.md`.

MOSS-Image has its own γ, one file per (variant, cfg) —
`networks/calibration/moss_cns_gamma*.npz`, listed in `library.moss.cns.GAMMA_PATHS` —
from `moss_cns_calibrate.py` (`--model`, `--cfg`, `--out`; see § MOSS-Image). The ComfyUI
Spectrum node bundles copies (`moss_cns_gamma*.npz` in its repo root); copy regenerated
ones over them before publishing the node.

Consumer + math: `library/inference/corrections/cns.py`. User doc:
`docs/cns.md`.

## channel_stats — per-channel LoRA gradient rebalance (SmoothQuant-style)

Produces the calibrations the `channel_scaling_alpha > 0` LoRA path absorbs:

```bash
# main stream → networks/calibration/channel_stats.safetensors
python scripts/calibration/analyze_lora_input_channels.py --per_artist \
    --dump_channel_stats networks/calibration/channel_stats.safetensors

# EasyControl cond stream → networks/calibration/cond_channel_stats.safetensors
python scripts/calibration/cond_stream_profile.py --per_artist \
    --dump_cond_stats networks/calibration/cond_channel_stats.safetensors
```

- `analyze_lora_input_channels.py` — per-input-channel `mean|x|` over real samples
  × 5 sigmas → dominance report + dumpable calibration.
- `cond_stream_profile.py` — the cond-stream counterpart (reuses the collector's
  dataset/dump helpers); cond calib does **not** transfer from the main file.

The DC-bias-vs-attention-sink decomposition: `_archive/bench/channel_stats/channel_dominance_analysis.md`
(its GraLoRA claim is wrong: `docs/anima/optimizations/channel_scaling.md`
§ "What it does not touch").

Consumer: `networks/lora_anima/factory.py` (`_CHANNEL_STATS_PATH`) and
`networks/methods/easycontrol.py`. User doc:
`docs/anima/optimizations/channel_scaling.md`.

## MOSS-Image (`library/moss/calibration/`, `networks/calibration/moss_cns_gamma*.npz`)

GPU scripts, run through the daemon (`make daemon-run ARGS="scripts/calibration/<script> …"`).

| Script | Produces |
|---|---|
| `moss_cns_calibrate.py` | `networks/calibration/moss_cns_gamma*.npz` (CNS γ per variant × cfg) |
| `moss_channel_stats.py` | `channel_stats.safetensors` + `channel_mean.safetensors` in its run dir; copy into `library/moss/calibration/` (`svd_drop_loud`, `down_init weight_svd_nodc`) |
| `moss_dc_probe.py` | `channels.npz` (per-block channel ranking), the input of the next row |
| `moss_massive_channels.py` | `library/moss/calibration/massive_channels.npz` (CPU; `--probe <run>/channels.npz`) |
