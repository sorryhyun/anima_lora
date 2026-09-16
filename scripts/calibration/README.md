# scripts/calibration

Scripts that regenerate the shipped artifacts under `networks/calibration/`.
They import the `anima_lora` façade + `library` / `networks` directly (no `bench/`
dependency). Probe/analysis history stays in `_archive/bench/`.

## CNS — colored-noise sampling γ matrix

Produces `networks/calibration/cns_gamma.npz` (consumed by `--cns auto`).

```bash
python scripts/calibration/cns_calibrate.py --cfg 4.0 --n_aspects 3   # compiled
```

- `cns_calibrate.py` — Phase-1 deploy calibrator (cfg=4.0, top-N aspect buckets);
  writes the npz (per-aspect σ50 staircase summary prints to stdout).
- `gamma_probe.py` — read-only Phase-0 staircase check + the shared γ/FFT helpers
  `cns_calibrate` imports. `--out_dir` for its standalone npz/heatmaps.

Phase log / precondition / composition tensions: `_archive/bench/cns/plan.md`
(premise corroborated by `project_sigma_signal_resolves_by_045`).

Consumer + math: `library/inference/corrections/cns.py`. User doc:
`docs/inference/cns.md`.

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

The DC-bias-vs-attention-sink decomposition and the GraLoRA alternative weighed
against: `_archive/bench/channel_stats/channel_dominance_analysis.md`.

Consumer: `networks/lora_anima/factory.py` (`_CHANNEL_STATS_PATH`) and
`networks/methods/easycontrol.py`. Regime analysis: memory
`project_per_channel_scaling_audit`. User doc:
`docs/optimizations/channel_scaling.md`.
