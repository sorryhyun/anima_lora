# Mod guidance — `pooled_text_proj` distillation loop

**Finished line** (2026-08-24) — verdicts and the open remainder are digested in
[`STATUS.md`](STATUS.md). This README is the ops surface.

The tree moved here from its old `distill_mod` home under `scripts/` when the
line finished, and the
`distill-prep` / `distill-mod` targets were removed with it: the head
is a **one-shot artifact** (re-distilled only when the base DiT changes), and
the shipped `pooled_text_proj_0413.safetensors` is a release asset, so the
everyday surface is inference-side (`make test MOD=1`, the ComfyUI node), not
training-side. The loop still runs, as module invocations from the repo root.

Feature docs stay where they were — [`docs/inference/mod-guidance.md`](../../../docs/inference/mod-guidance.md)
is canonical for the architecture, the inference profiles, and the **full flag
tables** for both commands below.

## Layout

| Module | Role |
|---|---|
| `prep.py` | Phase 1 (`T5("")` uncond sidecar) + Phase 2 (teacher-synthesized clean latents) |
| `synth.py` | Phase 2 teacher-driven synthesis |
| `distill.py` | The training loop — student (uncond crossattn + pooled inject) vs frozen teacher, MSE |
| `config.py` | argparser → frozen dataclass (CLI-first, no TOML layer; the template `scripts/distill_cjk/` was cloned from) |
| `teacher_cache.py` | Train + val teacher-prediction caches (K pre-sampled σ bins per sample) |
| `validation.py` | Fixed-σ teacher↔student MSE pass |

Single-GPU bespoke loop — it bypasses `train.py` and the config merge chain
entirely, so every knob is a CLI flag.

## Running it

From the repo root:

```bash
# Step 1 — pre-stage. Phase 1 = T5("") sidecar (make preprocess-te already
# produces it for free); Phase 2 = teacher-synthesized clean latents.
uv run python -m project.finished.mod_guidance.prep
uv run python -m project.finished.mod_guidance.prep --skip_synth     # Phase 1 only
uv run python -m project.finished.mod_guidance.prep --max_samples 16 # smoke-test Phase 2

# Step 2 — distill (paper-faithful: fit on the teacher's manifold)
uv run python -m project.finished.mod_guidance.distill \
    --data_dir post_image_dataset/lora \
    --dit_path models/diffusion_models/anima-base-v1.0.safetensors \
    --output_path output/ckpt/pooled_text_proj.safetensors \
    --synth_data_dir post_image_dataset/distill_mod_synth \
    --attn_mode flash --no_grad_ckpt
```

`output/ckpt/pooled_text_proj.safetensors` is where `make test MOD=1` looks, so
a fresh head is picked up with no extra flag.

**GPU jobs go through the daemon** (an agent-launched GPU process from a
background shell gets SIGKILLed after ~1 min):

```bash
make daemon-run ARGS="-m project.finished.mod_guidance.distill --synth_data_dir post_image_dataset/distill_mod_synth"
```

### Preset translation (what `PRESET=… distill-mod` used to do)

The removed targets appended `configs/presets.toml[<preset>]` as flags. Pass
them by hand now — only three keys were ever honored:

| presets.toml key | flag |
|---|---|
| `blocks_to_swap` | `--blocks_to_swap N` |
| `gradient_checkpointing` | `--grad_ckpt` / `--no_grad_ckpt` (**default `--no_grad_ckpt`** — this footprint is tiny, so ckpt is a pure perf loss unless VRAM is tight) |
| `sample_ratio` | `--sample_ratio R` |

So `PRESET=low_vram` ≈ `--grad_ckpt` (+ that preset's `blocks_to_swap`); the
default preset ≈ `--no_grad_ckpt`.

**VRAM**: the teacher runs under `no_grad` and holds almost nothing — the
student forward dominates (~12 GB on the default config). Leave `--no_grad_ckpt`
on if that fits; it's faster.

**Teacher cache RAM** scales as `dataset_size × K × latent_bytes` (K =
`--teacher_cache_K`, default 6); shrink K if RAM is tight.
