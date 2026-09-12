"""Gradient-SVD ``lora_down`` init — seed ``A`` from the task gradient's row space.

``down_init="weight_svd"`` seeds ``A`` with W₀'s top-r right singular vectors —
"where W is big", not "where the task pushes". This module builds the other
basis: a one-pass randomized sketch of the flow-matching gradient per
LoRA-target Linear, whose top-r right singular vectors are the subspace a
LoRA-GA / LoRA-One init would use (with ``B = 0`` the first optimizer step is
the rank-r truncated full-FT step).

    S = Ωᵀ G = Σ_tokens (Ωᵀ δ) xᵀ      # q × in, fp32, Ω ~ N(0,1) out×q
    V_r = top-r right singular vectors of S
    A_0 = V_rᵀ / sqrt(3)               # row-norm matched to the Kaiming default

Two consumers, one code path:

* ``down_init="grad_svd"`` — ``train.py`` sketches over the run's own cached
  dataset with the frozen DiT before the network is built, writes the basis
  beside the checkpoint, and hands the path to the factory.
* ``down_init="basis_file"`` — a basis built once over many artists
  (``bench/grad_init/build_universal_basis.py``) is read from disk; no per-run
  backward pass. E0 measured 0.633 held-out capture for a 20-artist pool vs
  0.709 for the artist's own gradient and 0.206 for ``weight_svd``.

A basis is **depth-baked**: module names carry the block index, so a 28-block
basis must not seed a 40-block DiT (see ``docs/methods/anima-2.9b.md``).
``load_basis`` refuses the mismatch on the block count read back from the keys.

Measurement and gates: ``bench/grad_init/README.md``,
``docs/proposal/grad_basis_init.md``.
"""

from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)

# Stored layout: one (in_features × r) fp16 tensor per lora_name. `in x r` so a
# consumer taking the leading `network_dim` columns is a contiguous slice.
BASIS_LAYOUT = "in x r"
BASIS_SUFFIX = ".grad_basis.safetensors"

DEFAULT_OVERSAMPLE = 32
DEFAULT_MAX_TOKENS = 4608  # skip larger latents (the sketch + backward is VRAM-bound)

_BLOCK_KEY_RE = re.compile(r"^lora_unet_blocks_(\d+)_")
_BLOCK_NAME_RE = re.compile(r"^blocks\.(\d+)\.")


# --------------------------------------------------------------------------- #
# target enumeration
# --------------------------------------------------------------------------- #
def enumerate_linear_targets(
    dit: torch.nn.Module,
) -> list[tuple[str, str, torch.nn.Linear]]:
    """→ ``[(lora_name, original_name, module)]`` for every LoRA-target Linear.

    Mirrors ``LoRANetwork.create_modules`` (Linear children of the replaced
    parent module classes), so the keys a basis carries are exactly the keys
    ``LoRAModule`` instances will be created under. Imported lazily: the
    network package imports this module.
    """
    from networks.lora_anima.network import LoRANetwork

    wanted = set(LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
    out: list[tuple[str, str, torch.nn.Linear]] = []
    seen: set[int] = set()
    for name, module in dit.named_modules():
        if module.__class__.__name__ not in wanted:
            continue
        for child_name, child in module.named_modules():
            if not isinstance(child, torch.nn.Linear) or id(child) in seen:
                continue
            seen.add(id(child))
            original = (name + "." if name else "") + child_name
            original = original.replace("_orig_mod.", "")
            lora_name = f"{LoRANetwork.LORA_PREFIX_ANIMA}.{original}".replace(".", "_")
            out.append((lora_name, original, child))
    return out


def count_blocks(names: Sequence[str]) -> int:
    """Top-level block count implied by a set of lora_names (0 if none)."""
    idx = [int(m.group(1)) for m in (_BLOCK_KEY_RE.match(n) for n in names) if m]
    return max(idx) + 1 if idx else 0


def dit_num_blocks(dit: torch.nn.Module) -> int:
    blocks = getattr(dit, "blocks", None)
    if blocks is not None:
        return len(blocks)
    idx = [
        int(m.group(1))
        for m in (
            _BLOCK_NAME_RE.match(n.replace("_orig_mod.", ""))
            for n, _ in dit.named_modules()
        )
        if m
    ]
    return max(idx) + 1 if idx else 0


# --------------------------------------------------------------------------- #
# sketch accumulator
# --------------------------------------------------------------------------- #
class GradientSketch:
    """One ``(Ω, S)`` pair per target Linear; forward/backward hooks feed it.

    ``Ω`` is drawn from a CPU generator seeded with ``seed``, so two sketches
    built with the same seed are additive (their ``S`` live in the same sketch
    space) — that is what lets ``build_universal_basis.py`` pool per-artist
    sketches, and what makes a per-run sketch comparable to a shipped one.
    """

    def __init__(
        self,
        targets: Sequence[tuple[str, str, torch.nn.Linear]],
        q: int,
        device: torch.device,
        seed: int,
    ):
        self.q = q
        self.device = device
        self.omega: dict[str, torch.Tensor] = {}
        self.sketch: dict[str, torch.Tensor] = {}
        self.n_tokens: dict[str, int] = {}
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for lora_name, _orig, mod in targets:
            om = torch.randn(mod.out_features, q, generator=gen, dtype=torch.float32)
            self.omega[lora_name] = om.to(device)
            self.sketch[lora_name] = torch.zeros(
                q, mod.in_features, dtype=torch.float32, device=device
            )
            self.n_tokens[lora_name] = 0
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    # -- hooks ------------------------------------------------------------- #
    def attach(self, targets: Sequence[tuple[str, str, torch.nn.Linear]]) -> None:
        for lora_name, _orig, mod in targets:
            self._handles.append(mod.register_forward_pre_hook(self._pre_hook))
            self._handles.append(
                mod.register_forward_hook(self._make_fwd_hook(lora_name))
            )

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def release(self) -> None:
        self.detach()
        self.sketch.clear()
        self.omega.clear()

    @staticmethod
    def _pre_hook(_mod, inputs):
        # Frozen DiT: a Linear whose input carries no grad (the t-embedding
        # path, the cached cross-attn context, …) would never see a δ. Give it a
        # leaf that requires grad — nothing upstream was receiving grad through
        # it anyway, so no other layer's δ changes.
        x = inputs[0]
        if torch.is_grad_enabled() and not x.requires_grad:
            x = x.detach().requires_grad_(True)
            return (x,) + tuple(inputs[1:])
        return None

    def _make_fwd_hook(self, lora_name: str):
        def hook(_mod, inputs, out):
            if not out.requires_grad:  # the no-grad pass of the checkpointer
                return
            x = inputs[0]
            om = self.omega[lora_name]

            def on_grad(delta: torch.Tensor):
                d = delta.reshape(-1, delta.shape[-1]).float()  # (T, out)
                xx = x.reshape(-1, x.shape[-1]).float()  # (T, in)
                self.sketch[lora_name].add_((d @ om).T @ xx)
                self.n_tokens[lora_name] += d.shape[0]

            out.register_hook(on_grad)

        return hook


# --------------------------------------------------------------------------- #
# subspace math
# --------------------------------------------------------------------------- #
def top_right_basis(S: torch.Tensor, r: int) -> torch.Tensor:
    """``S`` (q×in) → the top-r right singular vectors as ``(in, r)``."""
    _u, _s, vh = torch.linalg.svd(S.float(), full_matrices=False)
    return vh[:r].T.contiguous()


def stratified_logit_normal(n: int, gen: torch.Generator) -> torch.Tensor:
    """``n`` σ values covering the logit-normal(0,1) marginal by quantile strata.

    The trainer's default density (``timestep_sampling="sigmoid"``, scale 1,
    bias 0) drawn stratified rather than i.i.d., so a 32-image artist covers the
    σ marginal evenly instead of by luck — per-sample gradients are heavy-tailed
    in σ on this model (``bench/grad_init/README.md`` §gradient noise scale).
    """
    u = (torch.randperm(n, generator=gen).float() + torch.rand(n, generator=gen)) / n
    u = u.clamp(1e-4, 1 - 1e-4)
    return torch.sigmoid(torch.special.ndtri(u))


# --------------------------------------------------------------------------- #
# the sketch pass
# --------------------------------------------------------------------------- #
def _latent_tokens(latents: torch.Tensor) -> int:
    h, w = latents.shape[-2:]
    return (h // 2) * (w // 2)


def sketch_dataset(
    dit: torch.nn.Module,
    pairs: Sequence[tuple[str, str]],
    *,
    rank: int,
    device: torch.device,
    oversample: int = DEFAULT_OVERSAMPLE,
    passes: int = 1,
    seed: int = 42,
    max_samples: int = 0,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    log_every: int = 16,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Accumulate ``S = Ωᵀ G`` over ``pairs`` with a frozen ``dit``.

    ``pairs`` are ``(latent_npz_path, te_safetensors_path)`` — the same cached
    artifacts training reads, so the sketch sees the run's own data with no
    VAE/TE load. Returns ``({lora_name: S (q, in) cpu fp32}, metadata)``.

    The caller owns the DiT's placement and grad-checkpointing state; this only
    runs forwards/backwards through it. ``dit`` must be frozen
    (``requires_grad_(False)``) — δ is taken from activation hooks, never from
    parameter grads, so no optimizer state is touched either way.
    """
    from library.io.cache import load_cached_latents, load_cached_text_features

    targets = enumerate_linear_targets(dit)
    if not targets:
        raise RuntimeError("grad_basis: no LoRA-target Linear found on the DiT")
    q = rank + oversample

    gen = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(pairs), generator=gen).tolist()
    pairs = [pairs[i] for i in order]
    if max_samples:
        pairs = pairs[:max_samples]
    noise_gen = torch.Generator(device=device).manual_seed(seed + 1)

    acc = GradientSketch(targets, q, device, seed=seed)
    acc.attach(targets)

    schedule: list[tuple[int, torch.Tensor]] = []
    for _p in range(max(1, passes)):
        sig = stratified_logit_normal(len(pairs), gen)
        schedule += [(j, sig[j]) for j in range(len(pairs))]

    used = skipped_tokens = skipped_te = 0
    losses: list[float] = []
    t0 = time.perf_counter()
    try:
        for j, sigma_cpu in schedule:
            npz_path, te_path = pairs[j]
            latents = load_cached_latents(npz_path)[0].unsqueeze(0)
            if _latent_tokens(latents) > max_tokens:
                skipped_tokens += 1
                continue
            crossattn, _pooled = load_cached_text_features(te_path, variant=0)
            if crossattn is None:
                skipped_te += 1
                continue
            latents = latents.to(device)
            crossattn = crossattn.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
            noise = torch.randn(
                latents.shape, generator=noise_gen, device=device, dtype=latents.dtype
            )
            sigma = sigma_cpu.view(1).to(device)
            s4 = sigma.view(-1, 1, 1, 1)
            noisy = (1.0 - s4) * latents + s4 * noise
            target = noise - latents  # rectified-flow target, as train.py
            # 4D around the DiT, 5D inside it — the singleton frame axis is
            # dim 2 and nothing else (CLAUDE.md §5D latents).
            noisy_5d = noisy.unsqueeze(2).to(torch.bfloat16).requires_grad_(True)
            padding_mask = torch.zeros(
                1,
                1,
                latents.shape[-2],
                latents.shape[-1],
                dtype=torch.bfloat16,
                device=device,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = dit(noisy_5d, sigma, crossattn, padding_mask=padding_mask)
            pred = pred.squeeze(2).float()
            loss = torch.nn.functional.mse_loss(pred, target)  # uniform weighting
            loss.backward()
            losses.append(loss.item())
            used += 1
            del pred, loss, noisy_5d, noisy, latents, crossattn, noise
            if log_every and used % log_every == 0:
                logger.info(
                    "grad_basis sketch %d/%d  loss=%.4f  %.2fs/img",
                    used,
                    len(schedule),
                    sum(losses[-log_every:]) / log_every,
                    (time.perf_counter() - t0) / used,
                )
        if used == 0:
            raise RuntimeError(
                "grad_basis: sketch pass used 0 samples "
                f"(skipped {skipped_tokens} over max_tokens={max_tokens}, "
                f"{skipped_te} without a cached crossattn)"
            )
        out = {name: acc.sketch[name].cpu() for name, _o, _m in targets}
    finally:
        acc.release()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta = {
        "n_pairs": len(pairs),
        "n_used": used,
        "passes": max(1, passes),
        "rank": rank,
        "oversample": oversample,
        "seed": seed,
        "skipped_tokens": skipped_tokens,
        "skipped_te": skipped_te,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return out, meta


def basis_from_sketches(
    sketches: dict[str, torch.Tensor], rank: int
) -> dict[str, torch.Tensor]:
    """``{name: S (q,in)}`` → ``{name: V (in, r)}`` fp32, skipping empty sketches."""
    basis: dict[str, torch.Tensor] = {}
    for name, S in sketches.items():
        if S.abs().sum() == 0:
            logger.warning("grad_basis: %s got no gradient; left Kaiming", name)
            continue
        basis[name] = top_right_basis(S, rank)
    return basis


# --------------------------------------------------------------------------- #
# on-disk artifact
# --------------------------------------------------------------------------- #
def save_basis(
    path: str | Path,
    basis: dict[str, torch.Tensor],
    *,
    num_blocks: int,
    extra_metadata: Optional[dict[str, str]] = None,
) -> Path:
    """Write ``{name: V (in, r)}`` as fp16, stamped with the DiT block count."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rank = max((v.shape[1] for v in basis.values()), default=0)
    meta = {
        "layout": BASIS_LAYOUT,
        "rank": str(rank),
        "ss_num_blocks": str(num_blocks),
        "n_layers": str(len(basis)),
    }
    meta.update({k: str(v) for k, v in (extra_metadata or {}).items()})
    save_file(
        {k: v.to(torch.float16).contiguous() for k, v in basis.items()},
        str(path),
        metadata=meta,
    )
    return path


def load_basis(
    path: str | Path,
    *,
    num_blocks: Optional[int] = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Read a basis artifact → ``({name: V (in, r) fp32}, metadata)``.

    Refuses a depth mismatch: a LoRA's module names carry the block index, so a
    28-block basis seeding a 40-block DiT would silently leave the tail blocks
    Kaiming (the same failure mode as merging a 40-block adapter onto the base
    — CLAUDE.md §DiT depth). The stamped ``ss_num_blocks`` is preferred; older
    artifacts are checked against the block count implied by their keys.
    """
    from safetensors import safe_open

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"grad_basis file not found: {path}. Build one with "
            f"`bench/grad_init/build_universal_basis.py` or use "
            f"down_init='grad_svd' to sketch this run's own data."
        )
    basis: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        meta = dict(f.metadata() or {})
        for key in f.keys():
            basis[key] = f.get_tensor(key).float()

    stamped = meta.get("ss_num_blocks")
    file_blocks = int(stamped) if stamped is not None else count_blocks(list(basis))
    if num_blocks is not None and file_blocks and file_blocks != num_blocks:
        raise ValueError(
            f"grad_basis {path.name} is depth-baked for {file_blocks} blocks but "
            f"this DiT has {num_blocks}. A basis is per checkpoint arch — "
            f"rebuild it against this checkpoint (docs/methods/anima-2.9b.md)."
        )
    if meta.get("layout", BASIS_LAYOUT) != BASIS_LAYOUT:
        raise ValueError(
            f"grad_basis {path.name}: layout={meta.get('layout')!r}, expected "
            f"{BASIS_LAYOUT!r}."
        )
    logger.info(
        "grad_basis: loaded %d layers from %s (rank=%s, blocks=%s)",
        len(basis),
        path.name,
        meta.get("rank", "?"),
        file_blocks,
    )
    return basis, meta


def init_down_from_basis(
    weight: torch.Tensor, basis: torch.Tensor, *, lora_name: str = ""
) -> int:
    """Copy ``V_rᵀ / sqrt(3)`` into ``weight`` (r, in); → columns actually seeded.

    ``basis`` is ``(in, r_store)``. When ``r_store < r`` the leading
    ``r_store`` rows of ``weight`` are seeded and the rest keep whatever init
    the caller already wrote (Kaiming), so the seeded directions are never
    diluted by zeros. The ``1/sqrt(3)`` matches the expected row-norm of the
    Kaiming default exactly as ``weight_svd`` does, so "better direction" is not
    confounded with "larger effective step".
    """
    r, in_features = weight.shape
    if basis.shape[0] != in_features:
        raise ValueError(
            f"grad_basis[{lora_name}]: basis in_features={basis.shape[0]} but "
            f"lora_down expects {in_features}"
        )
    take = min(r, basis.shape[1])
    with torch.no_grad():
        weight[:take].copy_(
            (basis[:, :take].T / math.sqrt(3)).to(weight.dtype, copy=False)
        )
    return take
