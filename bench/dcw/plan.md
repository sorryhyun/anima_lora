# DCW Integration Plan

**Paper:** *Elucidating the SNR-t Bias of Diffusion Probabilistic Models* (Yu et al., CVPR 2026, arXiv 2604.16044).
**Upstream code:** cloned to `DCW/` — FLUX reference in `DCW/FlowMatchEulerDiscreteScheduler.py`.

DCW is a training-free, plug-and-play **post-solver-step correction** that nudges `x_{t-1}` along the differential `x_{t-1} - x^0_pred` to counter per-step SNR drift. Overhead is negligible (≤0.5 % in the paper). It's already validated on FLUX and Qwen-Image — same flow-matching DiT family as Anima — so the integration surface is well-defined.

---

## 0. Translating the paper to Anima's flow-matching solver

Anima's step function (`library/inference/sampling.py:35`):

```python
def step(latents, noise_pred, sigmas, step_i):
    return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()
```

`noise_pred` is the model's velocity `v` (the FLUX convention). For flow-matching with `σ ∈ [0, 1]`:

| quantity | formula |
|---|---|
| `x_0_pred` | `latents - σ_i · v` |
| `prev_sample` (Euler) | `latents + (σ_{i+1} - σ_i) · v` |
| DCW correction (pixel) | `prev_sample + scaler · (prev_sample - x_0_pred)` |

With `scaler = λ · σ_i` (time-scaled per Eq. 20). The wavelet variants apply DWT, correct per-subband, then iDWT.

**One subtlety:** the paper's Fig. 4 hyperparameter ranges (λ_l ∈ [0.02, 0.08], λ_h ∈ [0.001, 0.013]) are tuned for DDPM-parameterised ADM / A-DPM / EDM on pixel-space CIFAR/CelebA. FLUX uses a *single scalar* `scaler` (upstream README says "pass in the parameter scaler by yourself"). For Anima (DiT on Qwen-VAE latents, 16-ch, latent resolution ~H/8), we should re-tune from scratch and not assume those numbers transfer. Start with a single `λ · σ_i` schedule in pixel mode, add wavelet mode behind the same flag.

---

## 1. Deliverables (in priority order)

| # | Deliverable | Size | Priority |
|---|---|---|---|
| 1 | `networks/dcw.py` — DWT helpers + `apply_dcw()` | ~120 lines | P0 |
| 2 | CLI flags in `inference.py` | ~40 lines | P0 |
| 3 | Plumb `apply_dcw` into `generation.py` non-spectrum loop | ~15 lines | P0 |
| 4 | Plumb into `ERSDESampler.step` path | ~10 lines | P0 |
| 5 | Plumb into `spectrum_denoise` | ~10 lines | P1 |
| 6 | `docs/methods/dcw.md` | ~80 lines | P1 |
| 7 | Optional dep in `pyproject.toml` + graceful import | ~10 lines | P1 |
| 8 | `make test-dcw` + `make test-spectrum-dcw` targets | ~5 lines | P1 |
| 9 | ComfyUI sampler-wrapper node | separate repo | P2 |
| 10 | Small hyperparam sweep → record best λ in doc | ablation | P2 |

---

## 2. File-by-file changes

### 2.1 New: `networks/dcw.py`

Self-contained module. No cross-imports with spectrum / LoRA. Mirrors the upstream `dcw_low` / `dcw_high` / `dcw_pix` helpers but generalised over the flow-matching step.

```python
"""DCW: Differential Correction in Wavelet domain.

Training-free, post-solver-step correction for SNR-t bias in flow-matching DiTs.
Paper: Yu et al., "Elucidating the SNR-t Bias of Diffusion Probabilistic Models"
(CVPR 2026, arXiv:2604.16044). Upstream repo: DCW/.

Call site: apply_dcw(prev_sample, x0_pred, sigma_i, mode, lambda_l, lambda_h)
after the Euler / ER-SDE step and before casting prev_sample back to the model dtype.
"""

from typing import Literal, Optional
import torch

_DWT = None
_IDWT = None


def _ensure_wavelets(device, wave: str = "haar"):
    """Lazy import + lazy CUDA instantiation. pytorch_wavelets is optional."""
    global _DWT, _IDWT
    if _DWT is None:
        from pytorch_wavelets import DWTForward, DWTInverse  # lazy
        _DWT = DWTForward(J=1, mode="zero", wave=wave).to(device)
        _IDWT = DWTInverse(mode="zero", wave=wave).to(device)
    return _DWT, _IDWT


def _as_2d_batched(x: torch.Tensor) -> tuple[torch.Tensor, torch.Size]:
    """Anima latents are (B, C, 1, H, W). DWT expects (B, C, H, W). Squeeze T."""
    orig = x.shape
    if x.dim() == 5:
        assert x.size(2) == 1, "DCW expects single-frame latents (T=1)"
        x = x.squeeze(2)
    return x, orig


def _restore(x: torch.Tensor, orig: torch.Size) -> torch.Tensor:
    if len(orig) == 5:
        x = x.unsqueeze(2)
    return x


def apply_dcw(
    prev_sample: torch.Tensor,
    x0_pred: torch.Tensor,
    sigma_i: float,
    *,
    mode: Literal["pixel", "low", "high", "dual"] = "pixel",
    lambda_l: float = 0.04,
    lambda_h: float = 0.005,
    wave: str = "haar",
) -> torch.Tensor:
    """Apply differential correction to prev_sample.

    pixel: prev += λ_l·σ_i · (prev - x0)   -- paper Eq. 17, pixel space
    low:   correct only LL subband with λ_l·σ_i
    high:  correct only {LH,HL,HH} subbands with (1-λ_h)·σ_i  (paper Eq. 21)
    dual:  low AND high simultaneously (Eq. 18 full form; default recommendation
           from DCW paper §5.3)
    """
    if mode == "pixel":
        return prev_sample + (lambda_l * sigma_i) * (prev_sample - x0_pred)

    dev = prev_sample.device
    dwt, idwt = _ensure_wavelets(dev, wave=wave)

    x, orig = _as_2d_batched(prev_sample.float())
    y, _    = _as_2d_batched(x0_pred.float())
    xl, xh  = dwt(x)
    yl, yh  = dwt(y)

    if mode in ("low", "dual"):
        xl = xl + (lambda_l * sigma_i) * (xl - yl)
    if mode in ("high", "dual"):
        s = (1.0 - lambda_h) * sigma_i  # per Eq. 21
        xh = [xh_b + s * (xh_b - yh_b) for xh_b, yh_b in zip(xh, yh)]

    out = idwt((xl, xh)).to(prev_sample.dtype)
    return _restore(out, orig)
```

Notes:
- Anima latents are `(B, C, 1, H, W)`; the 5-D squeeze keeps Qwen-VAE conventions intact (`generation.py:187`, `generation.py:386`).
- `dtype` is preserved at the boundary. DWT requires f32 internally (same as upstream).
- `pytorch_wavelets` is instantiated once and cached module-global — the upstream scheduler does the same on `__init__`.

### 2.2 `inference.py` — CLI flags

Insert after the Spectrum flags (around `inference.py:365`):

```python
# DCW: SNR-t bias correction (arXiv:2604.16044)
parser.add_argument("--dcw", action="store_true",
    help="Enable Differential Correction in Wavelet domain. Training-free, "
         "post-step correction for SNR-t bias. Composes with --spectrum and --sampler.")
parser.add_argument("--dcw_mode", type=str, default="pixel",
    choices=["pixel", "low", "high", "dual"],
    help="DCW correction domain (default: pixel; dual = paper-recommended low+high).")
parser.add_argument("--dcw_lambda_l", type=float, default=0.04,
    help="Low-frequency / pixel scaler λ_l (default 0.04 — retune per model).")
parser.add_argument("--dcw_lambda_h", type=float, default=0.005,
    help="High-frequency scaler λ_h (default 0.005; only used in high/dual modes).")
parser.add_argument("--dcw_wave", type=str, default="haar",
    help="Wavelet basis: haar (default), db4, sym8, etc.")
```

### 2.3 `library/inference/generation.py` — non-spectrum path

Inside `generate_body()`, replace the step block (lines 614–621) with:

```python
# ensure latents dtype is consistent
if er_sde is not None:
    denoised = latents.float() - sigmas[i] * noise_pred.float()
    new_latents = er_sde.step(latents, denoised, i)
else:
    denoised = latents.float() - sigmas[i] * noise_pred.float()  # NEW: compute x0 explicitly
    new_latents = inference_utils.step(latents, noise_pred, sigmas, i)

if getattr(args, "dcw", False) and sigmas[i + 1] > 0:  # skip final step (σ=0)
    from networks.dcw import apply_dcw
    new_latents = apply_dcw(
        new_latents.float(), denoised, float(sigmas[i]),
        mode=args.dcw_mode,
        lambda_l=args.dcw_lambda_l,
        lambda_h=args.dcw_lambda_h,
        wave=args.dcw_wave,
    )

latents = new_latents.to(latents.dtype)
```

Two small shape points:
- In the current ER-SDE branch, `er_sde.step` *returns* `denoised` on the last step (σ=0), so gating on `sigmas[i+1] > 0` is mandatory — otherwise DCW would be fed `(x0, x0)` and noop anyway, but we'd pay a pointless DWT.
- Apply DCW **before** re-casting to `bfloat16`. The f32 cast inside `apply_dcw` handles precision; casting back to bf16 happens once at the end.

Repeat the same change in `generate_body_tiled` (lines 315–321) — identical structure, applies to the post-merge latents, not per-tile.

### 2.4 `networks/spectrum.py` — spectrum path

Spectrum's step block (lines 423–430) is the same shape. Pipe `args` into `spectrum_denoise` via the existing call-site kwargs and add the same three lines. Alternative (cleaner): have `generate_body` pass a closure `apply_step(latents, noise_pred, i)` into both branches. For a first cut, duplicate the 4 lines — the interface is stable.

**Important composition note:** Spectrum *injects* a new error source (forecasted-feature error) on cached steps. DCW operates on `prev_sample - x0_pred`, where `x0_pred = latents - σ_i · noise_pred`. On cached Spectrum steps `noise_pred` was produced from a predicted feature, so DCW will see the Spectrum-biased `x0_pred` and correct against *that*. Empirically this should still help (DCW is bias-agnostic), but it's worth an explicit ablation row in the doc: `{spectrum, dcw, spectrum+dcw, baseline}` × FID.

### 2.5 `pyproject.toml` — optional dep

Add to dependencies (not optional — it's small, ~200 KB wheel, and a hard import gate would confuse users):

```toml
"pytorch-wavelets>=1.3.0",
"PyWavelets>=1.5.0",
```

(The project uses `uv`; per memory, run `uv add pytorch-wavelets PyWavelets` — **not** `uv sync`.)

### 2.6 `docs/methods/dcw.md`

Short doc in the existing style (like `spectrum.md`):

1. What & paper link
2. Quick start: `make test-dcw`, flags table
3. How it works — one paragraph on SNR-t bias + Eq. 17 in pixel space; one paragraph on wavelet variant
4. Flow-matching adaptation note (our version of the paper's Eq. 17)
5. Composition table: works with `--spectrum`, `--sampler er_sde`, `--tiled_diffusion`; does not interact with LoRA / Hydra / ReFT / postfix (purely sampler-level)
6. Hyperparameter tuning recipe (paper's two-stage search)
7. Overhead: one DWT + one iDWT per step on `(B, 16, H/8, W/8)` — negligible

### 2.7 `Makefile` / `tasks.py` — convenience targets

```makefile
test-dcw:
	$(PY) inference.py --dcw --dcw_mode dual  $(INFER_ARGS)

test-spectrum-dcw:
	$(PY) inference.py --spectrum --dcw --dcw_mode dual  $(INFER_ARGS)
```

Same for `tasks.py` (cross-platform). Wire into `test-*` via the existing pattern.

### 2.8 ComfyUI node (P2, separate repo)

Same shape as the Spectrum KSampler node that already lives at `sorryhyun/ComfyUI-Spectrum-KSampler`. Two plausible forms:

- **Sampler wrapper:** override `KSampler`'s inner-step callback to apply DCW. Clean but couples to ComfyUI internals.
- **Model patch:** `ModelPatcher.add_object_patch` on the sigmas/cfg function to splice DCW in. Matches how the Anima Adapter Loader in `custom_nodes/comfyui-hydralora/` already hooks things.

I'd sibling this into the Spectrum KSampler repo as a second node (`DCWKSampler` or a `DCW` wrapper node), so users can stack Spectrum + DCW in one workflow.

---

## 3. Validation plan

Keep small — avoid the full 50 K-sample FID pipeline of the paper. We're eyeballing whether the method helps on our data, then tuning.

### 3.1 Qualitative

Pick 8 prompts from existing sample prompt files, same seed across runs. Compare:

| run | flags |
|---|---|
| baseline | — |
| DCW pixel | `--dcw --dcw_mode pixel` |
| DCW low | `--dcw --dcw_mode low` |
| DCW dual | `--dcw --dcw_mode dual` |
| Spectrum | `--spectrum` |
| Spectrum + DCW | `--spectrum --dcw --dcw_mode dual` |
| APEX | `--apex` (4 NFE) |
| APEX + DCW | `--apex --dcw --dcw_mode pixel` |

Save to `test_output/dcw_ablation/`. Eyeball for the paper's reported failure modes (over-smoothing, overexposure in FLUX Fig. 3) — these are visible in Anima outputs too.

### 3.2 Quantitative (small)

If qualitative looks promising:
- 256 samples on held-out captions, CLIPScore + a quick FID against training set (not 50 K — ~1 K is enough for method-vs-method ordering).
- Sweep `λ_l ∈ {0.02, 0.04, 0.06, 0.08}` × `λ_h ∈ {0.001, 0.005, 0.01}` → pick a default for the doc.

### 3.3 Unit test

`tests/test_dcw.py`:
- `apply_dcw` with `λ_l=0` and `λ_h=0` is a no-op (bit-equivalent).
- Pixel-mode output shape == input shape for 5-D `(B, C, 1, H, W)` latents.
- DWT round-trip: `iDWT(DWT(x)) ≈ x` at f32 tolerance.
- `wave="haar"` and `wave="db4"` both run.

---

## 4. Rollout order

1. Land `networks/dcw.py` + CLI flags + `generation.py` non-spectrum path. Run `make test-dcw` on 3 prompts, eyeball. **(~1 hour)**
2. If step 1 looks reasonable, add Spectrum + ER-SDE paths. **(~30 min)**
3. Hyperparam sweep → doc default. **(~2 hours, mostly GPU time)**
4. `docs/methods/dcw.md` + `make test-*` targets. **(~30 min)**
5. ComfyUI node. **(separate session)**

Everything through step 4 fits in a single ~half-day of focused work, with negligible merge-conflict surface: the only file edits outside the new module are `inference.py` (arg block), `generation.py` (two small blocks), `spectrum.py` (one small block), `pyproject.toml`, `Makefile`, `tasks.py`, and one doc. No changes to training, networks, or any LoRA code.

---

## 5. Open questions / risks

1. **FLUX uses a single `scaler`, not `λ · σ_i`.** The paper's Eq. 20/21 schedule is derived for DDPM where `σ_t` means something specific. In flow-matching, `σ_i ∈ [0, 1]` is the interpolation ratio. Multiplying by it gives the "large correction early, small correction late" shape the paper argues for in low-freq, but the exact form may under- or over-correct. Worth comparing fixed `scaler` vs `σ_i · scaler` in the sweep.
2. **Latent space ≠ pixel space for the wavelet interpretation.** The paper's "low-freq = shape, high-freq = detail" story is about pixel images. On Qwen-VAE latents, low vs high frequency in *latent* DWT decomposes something different — probably still useful (structural vs textural modes of the latent code) but the mechanism description in the doc should be hedged.
3. **APEX + DCW may be double-correcting.** APEX is trained to produce good 1–4-NFE samples; the trained velocity already compensates for some per-step drift. Empirical ablation will tell us whether DCW adds value or fights APEX.
4. **Time-conditioned postfix (cond-timestep mode).** Orthogonal to DCW — postfix lives in the cross-attention embedding, DCW lives in the sampler. No interaction expected.
