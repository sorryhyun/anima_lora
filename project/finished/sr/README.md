# SR sidecar — ResShift ×4/×2 super-resolution

Finished line (2026-08-22) — verdicts and the open remainder are digested in
[`STATUS.md`](STATUS.md). The `make sr-*` targets were removed when the line
finished (training-only surface, essentially unused); everything below runs as
direct script invocations from the repo root.

Standalone super-resolution for our art, deliberately outside the Anima adapter
system (it has its own VQGAN, latent space, and training loop). Design rationale,
phasing, and the ×2-vs-×4 / degradation decisions live in
[`_archive/proposals/resshift_sr_sidecar.md`](../../../_archive/proposals/resshift_sr_sidecar.md) (retired — shipped).
This README is the ops surface.

## Self-contained: vendored source, no external clone

ResShift's source is vendored under [`sr/resshift/`](resshift/) (committed —
`models/`, `ldm/`, `utils/`, `configs/`, `LICENSE`; ~530 KB, basicsr-free). There is
no external `ResShift/` clone to fetch or patch. Weights live separately under
`sr/weights/` (gitignored, ~1.6 GB; auto-downloaded by `sr_infer.py` from the v2.0
release if missing).

The sidecar runs in the root Anima venv — there is no separate per-sidecar virtualenv anymore.
Its deps are an opt-in dependency group in the root `pyproject.toml`, installed with
`uv sync --group sr --inexact` (`--inexact` keeps hand-installed extras). Keeping them in a group rather than
core `dependencies` keeps the heavy metrics closure (`pyiqa` → `opencv-python-headless`,
`bitsandbytes`, `facexlib`, `datasets`, plus dev-tools as runtime deps) out of every
`uv sync`. The old isolated venv was for torch-conflict reasons that no longer exist —
the SR env long ago converged on root's torch (see below), and ResShift is vendored
basicsr-free, added to `sys.path` at import time (venv-independent).

- Same Blackwell torch as root (`torch 2.12 + cu132`, cu132 index). Python 3.13
  to match root and kill a version-drift axis.
- No xformers — and not worth building one. Its only candidate use here is the
  VQGAN single-head `head_dim=512` mid-attention, which xformers can't accelerate
  anyway; the vendored VQGAN ships query-chunked exact SDPA (`ldm/modules/
  diffusionmodules/model.py`, bit-faithful to the trained single-head math, `O(Bq·N)`
  memory) that fixes the OOM, and the UNet's Swin attention gets SDPA-flash for free.
- No basicsr. The vendored tree drops it (and the data/inference paths that needed
  it); `sr_infer.py` reimplements released-model inference over the vendored core.

All from the repo root (`SR=project/finished/sr` for brevity):

```bash
uv sync --group sr --inexact && $SR/scripts/setup_env.sh   # one-time: deps + verify
uv run python $SR/scripts/build_eval_set.py                # frozen synthetic-LR eval set (--n 30)
uv run python $SR/scripts/run_phase0.py                    # released x4 (v3) on eval set + metrics + montages
uv run python $SR/scripts/sr_infer.py -i foo.png --version v3 --chop_size 512   # tiled SR on any image/dir
```

The setup is idempotent (`uv sync` is). The vendored VQGAN attention patch lives
in the source (committed), so there's nothing to re-apply.

## Invariant: `lq` is the LR **pixels**, not the latent

The UNet takes two LR-derived inputs and they are NOT interchangeable:

- `z_y` — VQ latent of the bicubic-upsampled LR. The residual-shift base
  (`q_sample` / `prior_sample` / `p_sample`). Always the latent.
- `lq` (`model_kwargs`) — a separate 3-channel map concatenated onto the latent
  inside the first conv. The released realsr checkpoints were trained with the LR
  pixels resampled to latent resolution ([-1,1]); `lq_size == image_size == 64` makes
  `feature_extractor = nn.Identity`, so whatever you hand it goes straight into the concat.

Passing `z_y` as `lq` is off-manifold for those weights: the teacher's x0 prediction
degrades from img-MSE 0.004 to 1.50 (t=14) and 15-step inference comes out neon green.
That bug was live in `sr_infer.py` (2026-06-30 → 2026-07-11) and in the RSD distill loop
from the start — the shipped x4 student was distilled from a mis-conditioned teacher.

Build the conditioning with `rsd_models.make_cond_lq(lq_img, z_y, mode)` and pass it
as `predict_x0`'s `c_y`. Default mode is `pixel` for every version except the legacy
x2 line, whose teacher and 1-step student were both trained under `latent` and are
therefore self-consistent. Student checkpoints record `cond_lq` in their meta; ckpts
without it predate the fix and run as `latent`. Override anywhere with `--cond_lq`.

## Finetuning (`train_sr/train.py`)

One loop, `train_sr/train.py`, parametrized by `--version`. GPU runs should go
through the daemon (agent-launched ones must — see the `daemon` skill):

```bash
make daemon-run ARGS="project/finished/sr/train_sr/train.py --version x2 --iters 30000 --bs 8"            # 2x line (shipped)
make daemon-run ARGS="project/finished/sr/train_sr/train.py --version x4 --iters 30000 --bs 8 --compile"  # 15-step x4 teacher
make daemon-run ARGS="project/finished/sr/train_sr/train.py --version x4s4 --iters 30000 --bs 8 --compile" # 4-step (v3) x4 teacher
```

Each version picks a config + released warm-start ckpt + output dir (see `VERSIONS` in
the file); the x4 configs are the released x4 config with only the trainer/data sections
dropped, so the warm-start is a strict 564/564 load. The x4 finetune feeds the distiller
as `distill_rsd/train.py --version x4ft` (an s4 finetune additionally needs
`--config project/finished/sr/configs/realsr_x4_s4_art.yaml`). The distiller and its
1-step-student inference (`distill_rsd/infer.py`) run the same way; note the old
`sr-rsd-train` wrapper defaulted `--src` to `data/rsd_hr_cap4096` when present — pass
it explicitly now (never the downsized 1024 pool, see `distill_rsd/DESIGN.md`).

## Phase 0 — verdict (2026-06-29): released model transfers well to our art

Ran released ResShift ×4 v3 (4-step) on 30 art images (synthetic LR = bicubic
÷4 of a 1024-long-edge HR), scored vs HR and vs a bicubic baseline:

| metric | ResShift | bicubic |
|---|---|---|
| PSNR ↑ | 27.80 | 25.82 |
| SSIM ↑ | 0.875 | 0.835 |
| LPIPS ↓ | 0.116 | 0.352 |
| MUSIQ ↑ | 73.8 | 41.0 |

ResShift wins every axis, hugely on perceptual (LPIPS/MUSIQ). Eyeball
(`sr/data/montage/`, panels = bicubic \| ResShift \| HR): sharp lineart + recovered
hair strands, no texture hallucination and no color shift on flat-shaded
regions. The feared photo-prior-vs-art domain gap did not materialize — a more
positive result than the proposal expected.

Implication: the released model is already usable for clean upscaling. Phase 1
finetune is still worth it but for narrower reasons — (a) a true ×2 (1024→2048)
model, (b) matching our pipeline's actual degradation (the `bicsr`/light-degradation
ablation, proposal §1b), (c) pushing line-edge sharpness — not for closing a large
domain gap. Caveat: this eval used *clean bicubic* LR; the realsr model handling it
well is encouraging, but a `bicsr` checkpoint may do even better on clean input.

Artifacts: `sr/data/{hr_eval,lr_eval,results,montage}/`, `sr/data/phase0_summary.json`.

## Next (Phase 1, gated on the above)

Finetune the ×2 config (`sr/resshift/configs/realsr_realesrgan256_x2.yaml`) from the released
checkpoint on our HR pool; first ablation = `bicsr` vs light-Real-ESRGAN degradation.
The realsr inference path is ×4-only (`assert scale==4`); wiring ×2 inference is a
Phase-1 task. See proposal §3 Phase 1.
