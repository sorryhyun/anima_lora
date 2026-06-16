"""EasyControl network module for Anima — two-stream rewrite.

Architecture (adapter-only — DiT frozen):

  reference image (clean VAE latent, 4D [B, C, H, W])
      -> DiT.x_embedder (frozen, reused)             [B, T_c, H_c, W_c, D]
      -> flatten to native token count                [B, S_c, D]
      -> cond_rope = DiT.pos_embedder at cond's native shape
      -> cond_temb = DiT.t_embedder(zeros) (cond is "clean", t=0)

  Per Anima Block (patched ``Block.forward``):

    target stream (frozen DiT)            cond stream (frozen DiT + cond LoRA)
    ───────────────────────────           ────────────────────────────────────
    AdaLN_self(t_emb)                     AdaLN_self(cond_temb)
    self_attn.compute_qkv(                self_attn.qkv_proj(cond_normed)
        target_normed, rope=target_rope)    + cond_lora_qkv(cond_normed)·scale
                                          q,k,v unbind → q_norm,k_norm,v_norm
                                          apply_rotary_pos_emb_qk(cond_rope)
              │                                     │
              ▼  ◄── target attends to ──┐          ▼
    target_out = LSE-extended attn       │   cond_out = SDPA(cond_q,
       (target_q vs [target_k ; cond_k], │                 cond_k, cond_v)
        with b_cond bias on cond rows)   │   (own self-attn, S_c × S_c)
              │                          │          │
              ▼                          │          ▼
    output_proj(target_out)              │   output_proj(cond_out)
                                         │   + cond_lora_o(cond_out)·scale
    + gate · residual                    │   + cond_gate · residual
              │                          │          │
              ▼                          │   (cross_attn skipped on cond — official
    AdaLN_cross(t_emb) + cross_attn(text)│    drops it for the simple two-stream variant)
    + gate · residual                    │          │
              │                          │          ▼
              ▼                          │   AdaLN_mlp(cond_temb)
    AdaLN_mlp(t_emb) + mlp               │   + mlp + cond_lora_ffn{1,2}·scale
    + gate · residual                    │   + cond_gate · residual
              │                          │          │
              └─►  next block            └─►  next block (cond_x flows
                                              block-by-block via per-block
                                              side channel; autograd is
                                              preserved through the patched
                                              forward's explicit arg/return)

Key properties (vs. the Phase 1.5 cond pre-pass):

  - No cross-block ``K_c/V_c`` cache. Each block produces its own cond_k/cond_v
    fresh in the same scope where the LSE-extended target attention consumes
    them; nothing pinned across blocks.
  - No deferred-backward dance. cond_x flows as an explicit checkpoint
    input/output of each patched ``Block.forward``, so unsloth / cpu_offload
    per-block backward sees a normal sequential graph and recomputes the cond
    stream alongside target on backward. ``backward_cond_path()`` is gone.
  - Cond gets its OWN RoPE at its own native (smaller) shape — same code path
    target uses (``Attention.compute_qkv`` consumes ``rope_cos_sin``). Matches
    the official EasyControl reference's intent. (Positional alignment with
    target — the official's ``resize_position_encoding`` for spatial control —
    is a separate follow-up; this revision uses cond's native positions, which
    matches the official's "subject" mode.)

Step-0 baseline equivalence (still ``b_cond=-10``):

  exp(-10) ≈ 4.5e-5, so cond softmax mass on target rows is negligible at
  init → α ≈ 1 → target_out ≈ baseline DiT regardless of cond evolution.
  Verified by ``bench/easycontrol/step0_equivalence.py`` Section B
  under the new layout (separate cond Q/K/V, cond RoPE, smaller S_c).

Train-time contract:

  Caller invokes ``network.set_cond(clean_vae_latent)`` ONCE per batch before
  the DiT forward. Pass ``None`` (or call ``clear_cond``) for unconditional /
  CFG-dropout passes — patched ``Block.forward`` then falls through to the
  baseline. After ``accelerator.backward(loss)``, **no extra call is needed**
  — autograd handles the cond chain via the per-block checkpoint outputs.
"""

from __future__ import annotations

import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch._dynamo  # noqa: F401  (mark_dynamic for compile_dynamic_seq)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library.log import setup_logging
from library.training.method_adapter import MethodAdapter, SetupCtx, StepCtx
from networks.lora_modules.base import _absorb_channel_scale
from networks.methods.base import AdapterNetworkBase
from networks.methods.easycontrol_attention import _extended_target_attention

setup_logging()
logger = logging.getLogger(__name__)


# Anima DiT defaults — see library/anima/models.py:Anima.__init__
DEFAULT_NUM_BLOCKS = 28
DEFAULT_HIDDEN_SIZE = 2048  # query_dim
DEFAULT_NUM_HEADS = 16
DEFAULT_HEAD_DIM = DEFAULT_HIDDEN_SIZE // DEFAULT_NUM_HEADS  # 128
DEFAULT_MLP_RATIO = 4.0
DEFAULT_LORA_DIM = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_B_COND_INIT = -10.0
DEFAULT_COND_RES_SCALE = 1.0  # 1.0 = native cond res (bit-exact to pre-PAI path)


# ---- cond-stream channel scaling (SmoothQuant-style per-input rebalance) ----
# Absorbs a per-input-channel scale into each cond LoRA down-projection (rebalances
# per-column gradient magnitudes; output-preserving). Uses a COND-SPECIFIC
# calibration — the LoRA-family main-stream file does NOT transfer to the cond
# stream (post-GELU mlp.layer2 inputs diverge; xfer_eff ~0.06). Measured by
# bench/channel_stats/cond_stream_profile.py; keyed by the same lora_unet_*
# convention as networks/calibration/channel_stats.safetensors.
_COND_CHANNEL_STATS_PATH = (
    Path(__file__).resolve().parent.parent
    / "calibration"
    / "cond_channel_stats.safetensors"
)

# kind -> (state_dict ModuleList attr, in_dim selector). The four cond LoRA
# down-projections channel scaling rebalances.
_COND_LORA_KINDS = {
    "qkv": ("cond_lora_qkv", "hidden"),
    "o": ("cond_lora_o", "hidden"),
    "ffn1": ("cond_lora_ffn1", "hidden"),
    "ffn2": ("cond_lora_ffn2", "ffn"),
}


def _cond_lora_calib_key(kind: str, idx: int) -> str:
    """Calibration key for a cond LoRA down-proj — names the DiT Linear it
    shadows, in the lora_unet_* convention the calibration file is keyed by."""
    suffix = {
        "qkv": "self_attn_qkv_proj",
        "o": "self_attn_output_proj",
        "ffn1": "mlp_layer1",
        "ffn2": "mlp_layer2",
    }[kind]
    return f"lora_unet_blocks_{idx}_{suffix}"


def _load_cond_channel_scales(alpha: float) -> Optional[dict]:
    """mean|x| calibration -> per-channel scale ``s``, replicating
    ``lora_anima/factory._load_channel_scales`` exactly (``s = clamp_min(1e-6)^alpha``
    normalized to mean 1). Returns None when scaling is off (``alpha <= 0``)."""
    if alpha <= 0.0:
        return None
    if not _COND_CHANNEL_STATS_PATH.is_file():
        raise FileNotFoundError(
            f"cond channel calibration missing at {_COND_CHANNEL_STATS_PATH}. "
            "Regenerate via `bench/channel_stats/cond_stream_profile.py "
            "--dump_cond_stats ...`, or set channel_scaling_alpha=0 to disable."
        )
    from safetensors.torch import load_file

    raw = load_file(str(_COND_CHANNEL_STATS_PATH))
    out = {}
    for name, mean_abs in raw.items():
        s = mean_abs.float().clamp_min(1e-6).pow(alpha)
        out[name] = s / s.mean().clamp_min(1e-12)
    return out


class _LoRAProj(nn.Module):
    """Plain LoRA-style D->r->out_dim projection with up zero-init.

    Standalone (not a wrapper around an org_module) — used by EasyControl to
    add a delta to a frozen DiT projection only on the cond stream. Output
    added by the caller; this module just produces the delta.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        r: int,
        alpha: float,
        channel_scale: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if r > 0 else 1.0
        self.lora_down = nn.Linear(in_dim, r, bias=False)
        self.lora_up = nn.Linear(r, out_dim, bias=False)
        # Standard LoRA init: Kaiming uniform on down, zeros on up so the delta
        # is exactly zero at step 0.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        # Per-channel input scaling (SmoothQuant-style). Absorbs s into lora_down
        # (W[:,c] *= s_norm[c]) and stores inv_scale = 1/s_norm; forward applies
        # ``x * inv_scale`` so the output is unchanged but per-column gradients are
        # rebalanced. Persistent buffer — the absorbed weight and inv_scale are
        # saved/loaded together, so train/resume/inference stay self-consistent.
        # See _absorb_channel_scale + bench/channel_stats/cond_stream_profile.py.
        self._has_channel_scale = False
        if channel_scale is not None:
            inv_scale = _absorb_channel_scale(self.lora_down.weight.data, channel_scale)
            self.register_buffer("inv_scale", inv_scale, persistent=True)
            self._has_channel_scale = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Activation-dtype GEMMs: bit-identical under the trainer's
            # autocast(bf16) to the retired fp32-bottleneck path (autocast
            # re-cast its ``.float()`` inputs back to bf16 before every GEMM
            # — see bench/lora_fp32_bottleneck), minus the dead cast traffic.
            # Channel-scale rebalance follows the LoRA-family convention
            # (``base._rebalance``): applied to x in the activation dtype.
            x_lora = x
            if self._has_channel_scale:
                x_lora = x * self.inv_scale.to(device=x.device, dtype=x.dtype)
            h = F.linear(x_lora, self.lora_down.weight.to(x.dtype))
            h = F.linear(h, self.lora_up.weight.to(x.dtype))
            return (h * self.scale).to(x.dtype)
        # Inference (the KV-cache prefill runs without autocast): keep the
        # historical fp32 compute so cond-stream outputs are unchanged.
        x_lora = x.float()
        if self._has_channel_scale:
            x_lora = x_lora * self.inv_scale.to(device=x.device, dtype=torch.float32)
        h = F.linear(x_lora, self.lora_down.weight.float())
        h = F.linear(h, self.lora_up.weight.float())
        return (h * self.scale).to(x.dtype)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    del vae, text_encoders, neuron_dropout
    cond_lora_dim = network_dim if network_dim is not None else DEFAULT_LORA_DIM
    cond_lora_alpha = (
        network_alpha if network_alpha is not None else float(cond_lora_dim)
    )

    b_cond_init = float(kwargs.get("b_cond_init", DEFAULT_B_COND_INIT))
    cond_scale = float(kwargs.get("cond_scale", 1.0))
    apply_ffn_lora = bool(int(kwargs.get("apply_ffn_lora", 1)))
    cond_res_scale = float(kwargs.get("cond_res_scale", DEFAULT_COND_RES_SCALE))

    # Deprecated 2026-06-10 (accepted so old snapshot TOMLs replay): the
    # fp32-bottleneck down-projection autograd was removed — training GEMMs
    # run in the activation dtype, which is what the trainer's autocast(bf16)
    # already produced. See bench/lora_fp32_bottleneck.
    if str(kwargs.get("use_custom_down_autograd", "false")).strip().lower() in (
        "true",
        "1",
    ):
        logger.info(
            "EasyControl: use_custom_down_autograd is deprecated and ignored "
            "(fp32-bottleneck path removed; activation-dtype GEMMs are "
            "bit-identical under the trainer's autocast)"
        )

    # Cond-stream channel scaling. Honors the same `channel_scaling_alpha` knob as
    # the LoRA family (inherited from base.toml), but loads the COND-SPECIFIC
    # calibration — the main-stream file does not transfer (see helper). alpha<=0
    # (or a missing knob) disables it. The cond-LoRA down-projections then absorb
    # the per-channel rebalance at build time.
    channel_scaling_alpha = float(kwargs.get("channel_scaling_alpha", 0.0) or 0.0)
    channel_scales = _load_cond_channel_scales(channel_scaling_alpha)

    num_blocks = (
        getattr(unet, "num_blocks", DEFAULT_NUM_BLOCKS)
        if unet is not None
        else DEFAULT_NUM_BLOCKS
    )
    hidden_size = (
        getattr(unet, "model_channels", DEFAULT_HIDDEN_SIZE)
        if unet is not None
        else DEFAULT_HIDDEN_SIZE
    )
    num_heads = (
        getattr(unet, "num_heads", DEFAULT_NUM_HEADS)
        if unet is not None
        else DEFAULT_NUM_HEADS
    )
    mlp_ratio = DEFAULT_MLP_RATIO  # Anima default; not exposed on the unet attr

    network = EasyControlNetwork(
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        cond_lora_dim=int(cond_lora_dim),
        cond_lora_alpha=float(cond_lora_alpha),
        b_cond_init=b_cond_init,
        cond_scale=cond_scale,
        apply_ffn_lora=apply_ffn_lora,
        cond_res_scale=cond_res_scale,
        multiplier=multiplier,
        channel_scaling_alpha=channel_scaling_alpha,
        channel_scales=channel_scales,
    )

    # REPA v2 auxiliary alignment loss, mirroring networks.lora_anima.factory.
    # Stash the config on the network: REPAMethodAdapter / losses._repa_loss /
    # build_method_adapters all key off network._repa_weight, so no args
    # plumbing is needed. The DiT is frozen here, so the alignment gradient
    # reaches the cond LoRA only through the extended self-attention in blocks
    # <= repa_layer — a conditioning-utilization pressure rather than the LoRA
    # family's representation shaping. The block hook captures patched_forward's
    # return value, which is the target stream alone (cond_x rides side
    # channels), so REPAMethodAdapter works unchanged.
    from networks.lora_anima.config import _as_bool

    if _as_bool(kwargs.get("use_repa")):
        repa_mode = str(kwargs.get("repa_mode", "relational")).lower()
        if repa_mode != "relational":
            raise ValueError(
                "EasyControl supports repa_mode='relational' only (the absolute "
                "arm needs a repa_head, which EasyControlNetwork does not carry)."
            )
        network._repa_mode = repa_mode
        network._repa_weight = float(kwargs.get("repa_weight", 0.05) or 0.0)
        network._repa_layer = int(kwargs.get("repa_layer", 8))
        network._repa_encoder = str(kwargs.get("repa_encoder", "pe_spatial"))
        network._repa_anneal_steps = float(kwargs.get("repa_anneal_steps", 0.0) or 0.0)
        network._repa_spatial_norm = _as_bool(kwargs.get("repa_spatial_norm"))
        network._repa_timestep_weighting = float(
            kwargs.get("repa_timestep_weighting", 0.0) or 0.0
        )
        network._repa_grad_heatmap = float(kwargs.get("repa_grad_heatmap", 0) or 0)
        # REPA-DoG target band-pass (docs/proposal/repa_dog_target.md): off by
        # default; when on it replaces the spatial_norm block in the relational
        # target preprocess. Pure target-side, no head — mirrors the LoRA factory.
        network._repa_target_dog = _as_bool(kwargs.get("repa_target_dog"))
        network._repa_dog_sigma1_div = float(
            kwargs.get("repa_dog_sigma1_div", 16.0) or 16.0
        )
        network._repa_dog_sigma2_div = float(
            kwargs.get("repa_dog_sigma2_div", 0.0) or 0.0
        )
        network._repa_dog_norm_std = float(kwargs.get("repa_dog_norm_std", 0.0) or 0.0)
        logger.info(
            f"EasyControl REPA[{repa_mode}]: weight={network._repa_weight}, "
            f"layer={network._repa_layer}, encoder={network._repa_encoder}, "
            f"anneal_steps={network._repa_anneal_steps:g}, "
            f"spatial_norm={network._repa_spatial_norm}, "
            f"target_dog={network._repa_target_dog}"
        )
    else:
        network._repa_weight = 0.0

    return network


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    del ae, text_encoders, for_inference
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    metadata = {}
    if file is not None and os.path.splitext(file)[1] == ".safetensors":
        from safetensors import safe_open

        with safe_open(file, framework="pt") as f:
            metadata = f.metadata() or {}

    num_blocks = int(metadata.get("ss_num_blocks", DEFAULT_NUM_BLOCKS))
    hidden_size = int(metadata.get("ss_hidden_size", DEFAULT_HIDDEN_SIZE))
    num_heads = int(metadata.get("ss_num_heads", DEFAULT_NUM_HEADS))
    mlp_ratio = float(metadata.get("ss_mlp_ratio", DEFAULT_MLP_RATIO))
    cond_lora_dim = int(metadata.get("ss_cond_lora_dim", DEFAULT_LORA_DIM))
    cond_lora_alpha = float(metadata.get("ss_cond_lora_alpha", float(cond_lora_dim)))
    b_cond_init = float(metadata.get("ss_b_cond_init", DEFAULT_B_COND_INIT))
    cond_scale = float(kwargs.get("cond_scale") or metadata.get("ss_cond_scale", 1.0))
    apply_ffn_lora = bool(int(metadata.get("ss_apply_ffn_lora", 1)))
    cond_res_scale = float(
        kwargs.get("cond_res_scale")
        or metadata.get("ss_cond_res_scale", DEFAULT_COND_RES_SCALE)
    )
    channel_scaling_alpha = float(metadata.get("ss_channel_scaling_alpha", 0.0))

    # If the checkpoint was trained with channel scaling, every absorbed
    # lora_down rides with a persistent ``inv_scale`` buffer. We must allocate a
    # matching buffer on each such module BEFORE load (load is strict=False — an
    # unallocated inv_scale would be silently dropped, leaving the absorbed
    # W·s loaded without the 1/s rebalance → wrong output). We don't need the
    # calibration file here: pass placeholder ones (identity absorb) for exactly
    # the modules whose inv_scale is in the checkpoint; load overwrites both the
    # absorbed weight and inv_scale with the real values.
    present_inv = {k for k in (weights_sd or {}) if k.endswith(".inv_scale")}
    channel_scales = None
    if present_inv:
        ffn_dim = int(hidden_size * mlp_ratio)
        channel_scales = {}
        for kind, (mlname, dim_sel) in _COND_LORA_KINDS.items():
            in_dim = hidden_size if dim_sel == "hidden" else ffn_dim
            for idx in range(num_blocks):
                if f"{mlname}.{idx}.inv_scale" in present_inv:
                    channel_scales[_cond_lora_calib_key(kind, idx)] = torch.ones(in_dim)

    network = EasyControlNetwork(
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        cond_lora_dim=cond_lora_dim,
        cond_lora_alpha=cond_lora_alpha,
        b_cond_init=b_cond_init,
        cond_scale=cond_scale,
        apply_ffn_lora=apply_ffn_lora,
        cond_res_scale=cond_res_scale,
        multiplier=multiplier,
        channel_scaling_alpha=channel_scaling_alpha,
        channel_scales=channel_scales,
    )
    return network, weights_sd


class EasyControlNetwork(AdapterNetworkBase):
    network_module = "networks.methods.easycontrol"
    network_spec = "easycontrol"

    def __init__(
        self,
        *,
        num_blocks: int,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        cond_lora_dim: int,
        cond_lora_alpha: float,
        b_cond_init: float,
        cond_scale: float,
        apply_ffn_lora: bool,
        cond_res_scale: float = DEFAULT_COND_RES_SCALE,
        multiplier: float = 1.0,
        channel_scaling_alpha: float = 0.0,
        channel_scales: Optional[dict] = None,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
            )
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.mlp_ratio = mlp_ratio
        self.ffn_dim = int(hidden_size * mlp_ratio)
        self.cond_lora_dim = cond_lora_dim
        self.cond_lora_alpha = cond_lora_alpha
        self.b_cond_init = b_cond_init
        self.cond_scale = cond_scale
        self.apply_ffn_lora = apply_ffn_lora
        # Position-Aware Interpolation downscale factor for the cond stream.
        # 1.0 = native (bit-exact to the pre-PAI path). 0 < s < 1 downsamples
        # the cond latent to ~s× per axis (fewer tokens → faster/cheaper) and
        # rescales cond's RoPE positions back onto the target grid so spatial
        # alignment survives. Values outside (0, 1] are clamped to 1.0.
        if not (0.0 < cond_res_scale <= 1.0):
            logger.warning(
                f"EasyControl: cond_res_scale={cond_res_scale} outside (0, 1]; "
                f"resetting to 1.0 (native cond resolution)."
            )
            cond_res_scale = 1.0
        self.cond_res_scale = cond_res_scale
        self.multiplier = multiplier

        D = hidden_size
        r = cond_lora_dim
        a = cond_lora_alpha

        # Per-channel input rebalance per cond LoRA down-proj. None when off, or
        # for any module whose calibration key is absent (e.g. the last block's
        # o/ffn — dead compute, never measured): that module trains unscaled.
        self.channel_scaling_alpha = float(channel_scaling_alpha)

        def _cs(kind: str, idx: int):
            if channel_scales is None:
                return None
            return channel_scales.get(_cond_lora_calib_key(kind, idx))

        # Per-block cond LoRA on self_attn:
        # qkv: fused D -> 3D delta (matches frozen Attention.qkv_proj layout).
        # o:   D -> D delta on the output projection.
        self.cond_lora_qkv = nn.ModuleList(
            [
                _LoRAProj(D, 3 * D, r, a, channel_scale=_cs("qkv", i))
                for i in range(num_blocks)
            ]
        )
        self.cond_lora_o = nn.ModuleList(
            [
                _LoRAProj(D, D, r, a, channel_scale=_cs("o", i))
                for i in range(num_blocks)
            ]
        )

        # Per-block cond LoRA on FFN (GPT2FeedForward layer1: D -> 4D, layer2: 4D -> D).
        if apply_ffn_lora:
            self.cond_lora_ffn1 = nn.ModuleList(
                [
                    _LoRAProj(D, self.ffn_dim, r, a, channel_scale=_cs("ffn1", i))
                    for i in range(num_blocks)
                ]
            )
            self.cond_lora_ffn2 = nn.ModuleList(
                [
                    _LoRAProj(self.ffn_dim, D, r, a, channel_scale=_cs("ffn2", i))
                    for i in range(num_blocks)
                ]
            )
        else:
            self.cond_lora_ffn1 = None
            self.cond_lora_ffn2 = None

        # Per-block scalar additive logit bias on cond keys. Init -10 → cond
        # softmax mass ≈ 4.5e-5 at step 0 → α ≈ 1 → target_out ≈ baseline DiT.
        # Stored as a ParameterList of 0-d Parameters (not a single
        # [num_blocks] Parameter) so each block's patched forward closure can
        # capture its bias as a *Parameter object*, not a Python int index —
        # dynamo specializes on int closure cells (treating them as static
        # nn.Module attributes), which used to cause one recompile per block.
        # Capturing a Parameter is fine: dynamo lifts it as a graph input.
        self.b_cond = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(b_cond_init, dtype=torch.float32))
                for _ in range(num_blocks)
            ]
        )

        # Populated by apply_to() — references to the DiT and its blocks. Plain
        # lists (NOT nn.ModuleList) so PyTorch doesn't re-parent the DiT into
        # this network's parameter tree.
        self._dit: Optional[nn.Module] = None
        self._block_modules: list[nn.Module] = []
        self._original_block_forwards: list = []
        self._patched: bool = False

        # Per-step cond state. None = no cond / CFG-dropped → patched block
        # forward falls through to the baseline DiT path.
        # When set, contains:
        #   "cond_emb"         : (B, 1, D) RMSNormed t_embedder(zeros)
        #   "cond_adaln_lora"  : (B, 1, 3*D_adaln) or None (matches DiT's
        #                        use_adaln_lora flag)
        #   "cond_rope"        : (cos, sin) RoPE tables for cond at S_c
        #                        (matches the shape DiT.pos_embedder produces,
        #                        at cond's native token count)
        # cond_x_init for block 0 lives on block_modules[0]._easycontrol_cond_x_in.
        self._cond_state: Optional[dict] = None

        # Inference KV cache: per-block (cond_k, cond_v) post-RoPE-and-norm,
        # i.e. the exact tensors `_extended_target_attention` consumes from the
        # cond stream. Populated by `precompute_cond_kv()`. When non-None, the
        # patched Block.forward bypasses the cond stream entirely and feeds
        # these tensors into target's extended self-attention. Training keeps
        # this None — every step needs the cond LoRA's gradient.
        self._cond_kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None

        # compile_dynamic_seq: when True (set by compile_cond_stream), the patched
        # block forward marks the target/cond seq axes dynamic via mark_dynamic so
        # the two-stream inner compiles one graph instead of one per
        # (target × cond) token-count pair. _dynamic_seq_range bounds the marks.
        # Mirrors the DiT-side mechanism in library/anima/models.py::_run_blocks.
        self._dynamic_seq: bool = False
        self._dynamic_seq_range: Optional[tuple] = None

        n_scaled = sum(
            1
            for m in self.modules()
            if isinstance(m, _LoRAProj) and m._has_channel_scale
        )
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"EasyControlNetwork: blocks={num_blocks}, hidden={hidden_size}/{num_heads}h, "
            f"r={cond_lora_dim} alpha={cond_lora_alpha}, ffn_lora={apply_ffn_lora}, "
            f"b_cond_init={b_cond_init}, cond_scale={cond_scale}, "
            f"cond_res_scale={self.cond_res_scale}, "
            f"channel_scaling_alpha={self.channel_scaling_alpha} "
            f"({n_scaled} cond projections rebalanced), "
            f"params={total / 1e6:.1f}M"
        )

    # ------------------------------------------------------------ apply / hook

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        del text_encoders, apply_text_encoder
        if not apply_unet:
            return
        if self._patched:
            logger.warning("EasyControlNetwork.apply_to called twice — skipping")
            return
        if unet is None or not hasattr(unet, "blocks"):
            raise ValueError("apply_to requires the Anima DiT (unet) with .blocks")
        if len(unet.blocks) != self.num_blocks:
            raise ValueError(
                f"DiT has {len(unet.blocks)} blocks, EasyControl expects {self.num_blocks}. "
                "Re-create the network with matching num_blocks."
            )

        # Bypass nn.Module.__setattr__'s auto-registration — otherwise
        # ``self._dit = unet`` would silently register the DiT as a submodule
        # and inflate ``self.parameters()`` with the entire frozen DiT.
        object.__setattr__(self, "_dit", unet)
        for idx, block in enumerate(unet.blocks):
            attn = block.self_attn
            if not attn.is_selfattn:
                raise RuntimeError(
                    f"block[{idx}].self_attn is unexpectedly cross-attention"
                )
            if attn.n_heads != self.num_heads or attn.head_dim != self.head_dim:
                raise ValueError(
                    f"block[{idx}].self_attn heads/head_dim mismatch: "
                    f"({attn.n_heads}, {attn.head_dim}) vs ({self.num_heads}, {self.head_dim})"
                )
            self._block_modules.append(block)
            self._original_block_forwards.append(block.forward)
            block._easycontrol_cond_x_in = None
            block.forward = _make_patched_block_forward(block, idx, self)

        self._patched = True
        logger.info(
            f"EasyControl: patched Block.forward on {len(self._block_modules)} blocks"
        )

    def compile_cond_stream(
        self,
        backend: str = "inductor",
        mode: Optional[str] = None,
        n_token_families: Optional[int] = None,
        dynamic_seq: bool = False,
        seq_range: Optional[tuple] = None,
    ):
        """torch.compile each block's two-stream cond forward.

        compile_blocks() only reaches the DiT's own ``block._forward``; the
        active (cond-on) training path routes through ``_two_stream_inner``
        instead (see _make_patched_block_forward), so without this the entire
        cond stream — every cond LoRA projection — runs eager and
        ``torch_compile`` is a no-op for EasyControl training. So this is what
        makes both torch_compile AND the lever earn their keep here.

        Mirrors compile_blocks: ``backend=inductor``, ``dynamic=False``, same
        ``mode``. Flash attention (``_ExtendedSelfAttnLSEFunc``) may graph-break
        — that's fine: the cond LoRA projections sit in their own compiled
        subgraphs, which is exactly where the lever must be live. Call AFTER
        apply_to (the compile-after-apply invariant).

        ``dynamic_seq`` collapses the per-(target × cond) token-count graph
        cascade to one, the same way compile_blocks does for the DiT — but here
        BOTH seq axes vary (target self-attn seq AND the cond stream's own seq),
        so the patched forward wraps the compiled inner in an eager mark_dynamic
        prologue (marks x dim 2 + cond_x dim 1 + both RoPE tables) that becomes
        the checkpointed callable — so the marks re-apply on the grad-checkpoint
        backward RECOMPUTE too, not just the forward (else detach_variable strips
        the latent marks but keeps the RoPE-tuple marks and dynamo raises a
        ConstraintViolationError). We keep ``dynamic=False`` and scope the
        symbolic axes via ``mark_dynamic`` rather than blanket ``dynamic=True``
        (mirrors the DiT path, library/anima/models.py::_run_blocks).
        ``seq_range`` bounds the marks; ``None`` falls back to the canonical 1024
        table (4032/4200). The flash graph-break around ``_ExtendedSelfAttnLSEFunc``
        splits the inner into pre/post subgraphs, each symbolic in the seq axes —
        both collapse.
        """
        if not self._patched:
            raise RuntimeError("compile_cond_stream requires apply_to() first")

        from library.runtime.dynamo import pin_dynamo_limit

        # The two-stream inner needs MANY more graphs than the target-only
        # block._forward compile_blocks() handles:
        #   - the (target × cond) token-count product (up to n² where the
        #     target-only path is just n),
        #   - times grad-on / grad-off GLOBAL_STATE (the inner is recompute'd
        #     under non-reentrant checkpoint → dynamo guards on grad_mode),
        #   - plus requires_grad/stride/num_threads specializations and the
        #     flash graph-break segments around _ExtendedSelfAttnLSEFunc.
        # That blows past the dynamo recompile_limit *default of 8* — and a
        # plain `config.cache_size_limit = …` (== recompile_limit alias) is a
        # context-local ContextVar override that REVERTS to 8 in the backward
        # compile context where the grad-bearing inner is actually traced (see
        # pin_dynamo_limit). So the budget was silently 8 here and the cond
        # stream spilled to eager mid-warmup — the recompile storm. Pin the
        # canonical `.default` so the raise survives every context. The
        # target-only block._forward never hit this only because n full-res
        # families (2) sits under 8; the cond product does not.
        n = n_token_families if n_token_families is not None else 2
        per_obj = 4 * n + 16
        pin_dynamo_limit("recompile_limit", per_obj)
        # accumulated_recompile_limit is the cross-code-object ceiling; every
        # block owns its own compiled inner, so budget for all of them.
        pin_dynamo_limit(
            "accumulated_recompile_limit", len(self._block_modules) * per_obj
        )

        # dynamic_seq does NOT use torch.compile(dynamic=True); compile static and
        # let the patched forward mark the target/cond seq axes (see the inner
        # dispatch). Derive the (min, max) seq bound for those marks.
        self._dynamic_seq = dynamic_seq
        if dynamic_seq:
            if seq_range is not None:
                self._dynamic_seq_range = (int(seq_range[0]), int(seq_range[1]))
            else:
                from library.datasets.buckets import token_count_range

                self._dynamic_seq_range = token_count_range([1024])

        compile_kwargs = {"backend": backend, "dynamic": False}
        if mode is not None:
            compile_kwargs["mode"] = mode
        for block in self._block_modules:
            block._easycontrol_two_stream_inner = torch.compile(
                block._easycontrol_two_stream_inner, **compile_kwargs
            )
        logger.info(
            f"EasyControl: compiled two-stream cond forward on "
            f"{len(self._block_modules)} blocks (backend={backend}, mode={mode}, "
            f"dynamic_seq={dynamic_seq} seq∈{self._dynamic_seq_range}, "
            f"recompile_limit pinned to {per_obj})"
        )

    def remove_from(self):
        for block, orig in zip(self._block_modules, self._original_block_forwards):
            block.forward = orig
            if hasattr(block, "_easycontrol_cond_x_in"):
                del block._easycontrol_cond_x_in
            if hasattr(block, "_easycontrol_two_stream_inner"):
                del block._easycontrol_two_stream_inner
        self._block_modules.clear()
        self._original_block_forwards.clear()
        object.__setattr__(self, "_dit", None)
        self._patched = False
        self._cond_kv_cache = None

    # ------------------------------------------------------------ runtime API

    def encode_cond_latent(
        self,
        cond_latent: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Patch-embed the clean VAE latent into ``[B, S_c, D]`` cond tokens
        plus the matching RoPE table at cond's native shape.

        Reuses the DiT's (frozen) ``x_embedder`` and ``pos_embedder``. Both
        outputs are kept at cond's native token count — no static padding.
        Anima's native-shape bucketing makes every forward run at its real
        token count (one bucket per batch → uniform S_c within a batch), and
        for the common ref==target setup cond's shape equals the target's, so
        S_c lands on one of the two bucket families (4032 / 4200). Padding
        would only leak zero tokens into the cond stream's self-attention and
        into target's LSE-extended attention (the same padding-leak the
        static-pad path was removed to avoid).

        Args:
            cond_latent: ``[B, C, H, W]`` (image) or ``[B, C, T, H, W]`` (video).
            padding_mask: optional ``[B, 1, H, W]``. If None and the DiT
                requires it, a default all-ones mask is synthesized.
        Returns:
            ``(cond_x, cond_rope)``:
              - ``cond_x``:    ``[B, S_c, D]``,    S_c = cond's native token count
              - ``cond_rope``: ``(cos, sin)`` each ``[S_c, 1, 1, D_head]``
        """
        if self._dit is None:
            raise RuntimeError("encode_cond_latent called before apply_to")

        if cond_latent.ndim == 4:
            cond_latent = cond_latent.unsqueeze(2)  # [B, C, 1, H, W]
        if cond_latent.ndim != 5:
            raise ValueError(
                f"cond_latent must be [B, C, T, H, W] or [B, C, H, W], got {tuple(cond_latent.shape)}"
            )

        B, _, _, H, W = cond_latent.shape

        # ---- Position-Aware Interpolation (cond-only downscale) ----
        # When cond_res_scale < 1, downsample the cond latent (fewer tokens →
        # cheaper cond self-attn + smaller KV cache) and rescale its RoPE
        # positions back onto the *target* grid so spatial alignment survives.
        # The target grid equals cond's full-resolution grid (cond is the same
        # spatial size as the target for both ref==target and colorize), so we
        # derive the rescale from the pre/post patch-grid sizes — no caller
        # plumbing. At cond_res_scale == 1 this whole block is skipped and the
        # path is bit-exact to the native-resolution behavior.
        h_scale = w_scale = 1.0
        if self.cond_res_scale < 1.0:
            p = self._dit.patch_spatial
            full_gh, full_gw = H // p, W // p  # target patch grid
            new_H = max(p, int(round(H * self.cond_res_scale / p)) * p)
            new_W = max(p, int(round(W * self.cond_res_scale / p)) * p)
            if (new_H, new_W) != (H, W):
                # area resampling = anti-aliased average pooling, the right
                # filter for downsampling. Operate on the 4D spatial form.
                cond_latent = F.interpolate(
                    cond_latent.squeeze(2), size=(new_H, new_W), mode="area"
                ).unsqueeze(2)
                H, W = new_H, new_W
                small_gh, small_gw = H // p, W // p
                h_scale = full_gh / small_gh
                w_scale = full_gw / small_gw

        if self._dit.concat_padding_mask and padding_mask is None:
            padding_mask = torch.ones(
                B, 1, H, W, device=cond_latent.device, dtype=cond_latent.dtype
            )

        # prepare_embedded_sequence handles padding-mask concat + patch embed,
        # AND returns the RoPE (cos, sin) for cond at its native (T_c, H_c, W_c)
        # shape. We keep the RoPE this time (Phase 1.5 discarded it).
        cond_x_5d, cond_rope = self._dit.prepare_embedded_sequence(
            cond_latent,
            fps=None,
            padding_mask=padding_mask,
        )
        # PAI: replace the native-position RoPE with positions rescaled onto the
        # target grid (cond patch i → i * H_t/H_c). Reduces to the native rope
        # at scale 1.0, but we only pay for it when actually downscaling.
        if h_scale != 1.0 or w_scale != 1.0:
            cond_rope = self._dit.pos_embedder.generate_embeddings_scaled(
                cond_x_5d.shape,
                h_scale=h_scale,
                w_scale=w_scale,
                fps=None,
            )

        # Flatten cond_x to [B, S_c, D] at cond's (possibly reduced) token count.
        # Pin to cond_latent's dtype (== weight_dtype, set by the trainer at
        # prime time). The patch-embed runs in the prime hook *outside* the
        # forward's autocast scope, so without this the dtype of cond_x tracks
        # the surrounding context (fp32 in the train forward vs bf16 under the
        # no_grad val/sample autocast) and flip-flops — which makes the compiled
        # two-stream inner specialize a *second* copy of every (token-count ×
        # is_last) graph keyed on cond_x's dtype. Pinning collapses that axis.
        # Numerically safe: cond_q/k/v are re-cast to the target stream's dtype
        # for flash downstream, and the whole forward runs under bf16 autocast.
        cond_x = cond_x_5d.flatten(1, 3).to(cond_latent.dtype)

        return cond_x, cond_rope

    def set_cond(
        self,
        cond_latent: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Prime per-step cond state on the network and on block 0's slot.

        Pass ``None`` (or call ``clear_cond``) for unconditional / CFG-dropout
        passes — patched ``Block.forward`` will fall through to the baseline
        DiT path.
        """
        if not self._patched:
            raise RuntimeError("set_cond called before apply_to")
        if cond_latent is None:
            self.clear_cond()
            return

        # New reference: any prior cache is stale until precompute_cond_kv runs
        # again. The two-stream path is the safe default in the meantime.
        self._cond_kv_cache = None

        cond_x, cond_rope = self.encode_cond_latent(
            cond_latent, padding_mask=padding_mask
        )

        # Build cond_temb at t=0 through the same t_embedder as target. The
        # AdaLN-LoRA branch is mirrored: t_embedder returns
        # (emb_B_T_D, adaln_lora_B_T_3D) when use_adaln_lora=True. We follow
        # forward_mini_train_dit and apply t_embedding_norm on emb_B_T_D.
        # Pooled-text projection is intentionally NOT applied: cond is the
        # reference image at t=0, with no text channel.
        B = cond_latent.shape[0]
        device = cond_x.device
        # Match the dtype t_embedder expects — its Timesteps layer handles
        # float32 internally and casts back to input dtype. Use the cond_x
        # dtype to avoid a needless promotion on the AdaLN inputs downstream.
        zeros = torch.zeros(B, 1, device=device, dtype=cond_x.dtype)
        cond_emb_B_T_D, cond_adaln_lora_B_T_3D = self._dit.t_embedder(zeros)
        cond_emb_B_T_D = self._dit.t_embedding_norm(cond_emb_B_T_D)

        self._cond_state = {
            "cond_emb": cond_emb_B_T_D,
            "cond_adaln_lora": cond_adaln_lora_B_T_3D,
            "cond_rope": cond_rope,
        }
        # Block 0's input. Subsequent blocks' slots are written by the
        # previous block's patched forward.
        self._block_modules[0]._easycontrol_cond_x_in = cond_x

    def clear_cond(self) -> None:
        self._cond_state = None
        for block in self._block_modules:
            block._easycontrol_cond_x_in = None
        # Stale cache after clear: a different reference would need a re-prime.
        self._cond_kv_cache = None

    def clear_cond_kv_cache(self) -> None:
        """Drop the per-block KV cache. Cond stream will be recomputed on the
        next forward (or until ``precompute_cond_kv`` is called again).
        """
        self._cond_kv_cache = None

    @torch.no_grad()
    def precompute_cond_kv(self) -> None:
        """Walk the cond stream once and cache (cond_k, cond_v) per block.

        Inference-only optimization. The cond stream is deterministic across
        denoising steps (cond_temb = t_embedder(zeros), no dependence on the
        noisy target, frozen DiT + frozen LoRA), so the per-block post-RoPE
        post-norm K/V tensors that target's extended self-attention consumes
        can be computed once and reused across every step and every CFG branch.

        After this call, the patched ``Block.forward`` skips all cond work
        (AdaLN, qkv_proj+LoRA, cond's own SDPA, MLP, residuals) and feeds the
        cached (cond_k, cond_v) directly into ``_extended_target_attention``.

        Caller contract: ``set_cond(reference_latent)`` must have run first.
        Changing ``multiplier``/``cond_scale`` after caching makes the cache
        stale — call ``clear_cond_kv_cache`` and re-prime if you change them.
        """
        if not self._patched:
            raise RuntimeError("precompute_cond_kv called before apply_to")
        if self._cond_state is None:
            raise RuntimeError(
                "precompute_cond_kv called before set_cond — set_cond must "
                "run first to populate cond_emb / cond_rope / block 0 cond_x"
            )

        from library.anima.models import apply_rotary_pos_emb_qk
        from networks import attention_dispatch as anima_attention

        cond_x = self._block_modules[0]._easycontrol_cond_x_in
        if cond_x is None:
            raise RuntimeError(
                "block 0 has no _easycontrol_cond_x_in — set_cond did not run "
                "or was followed by clear_cond"
            )

        cond_emb = self._cond_state["cond_emb"]
        cond_adaln_lora = self._cond_state["cond_adaln_lora"]
        cond_rope = self._cond_state["cond_rope"]
        eff_scale = self.cond_scale * self.multiplier

        # Run cond's own self-attention through the SAME dispatched backend the
        # two-stream training path uses, so the cached cond stream matches what
        # training built (bare SDPA here vs dispatched flash there would diverge
        # at the bf16-ulp level over 28 blocks) AND honors a non-default
        # attn_softmax_scale (which a bare SDPA call silently ignored).
        attn_params = anima_attention.AttentionParams.create_attention_params(
            self._dit.attn_mode, self._dit.attn_softmax_scale
        )
        last_idx = self.num_blocks - 1

        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, block in enumerate(self._block_modules):
            attn = block.self_attn
            cond_lora_qkv = self.cond_lora_qkv[idx]
            cond_lora_o = self.cond_lora_o[idx]
            cond_lora_ffn1 = self.cond_lora_ffn1[idx] if self.apply_ffn_lora else None
            cond_lora_ffn2 = self.cond_lora_ffn2[idx] if self.apply_ffn_lora else None

            # ---- AdaLN modulation (cond stream only — no cross-attn) ----
            (
                (cond_shift_self, cond_scale_self, cond_gate_self),
                (cond_shift_mlp, cond_scale_mlp, cond_gate_mlp),
            ) = _adaln_self_mlp(block, cond_emb, cond_adaln_lora)

            # ---- cond Q/K/V with LoRA + RoPE — this is what we cache ----
            cond_normed = (
                block.layer_norm_self_attn(cond_x) * (1 + cond_scale_self)
                + cond_shift_self
            )
            cond_qkv = attn.qkv_proj(cond_normed) + eff_scale * cond_lora_qkv(
                cond_normed
            )
            cond_q, cond_k, cond_v = cond_qkv.unflatten(
                -1, (3, attn.n_heads, attn.head_dim)
            ).unbind(dim=-3)
            cond_q = attn.q_norm(cond_q)
            cond_k = attn.k_norm(cond_k)
            cond_v = attn.v_norm(cond_v)
            if cond_rope is not None:
                cond_q, cond_k = apply_rotary_pos_emb_qk(
                    cond_q, cond_k, cond_rope, tensor_format=attn.qkv_format
                )
            cache.append((cond_k.detach(), cond_v.detach()))

            # ---- evolve cond_x to feed the next block ----
            # The last block's evolved cond_x is never consumed (only its cached
            # K/V feed target's extended attention), so skip its dead self-attn
            # + output proj + MLP — mirrors `is_last` in the two-stream path.
            if idx == last_idx:
                continue

            # cond_q/k/v are BLHD; cast to the cond compute dtype (flash rejects
            # fp32, and the fp32 cond-LoRA delta + q/k/v norms can promote them).
            # Mirrors _two_stream_inner's cast before dispatch_attention.
            compute_dtype = cond_x.dtype
            cond_attn_out = anima_attention.dispatch_attention(
                [
                    cond_q.to(compute_dtype),
                    cond_k.to(compute_dtype),
                    cond_v.to(compute_dtype),
                ],
                attn_params=attn_params,
            )
            cond_attn_proj = attn.output_proj(cond_attn_out) + eff_scale * cond_lora_o(
                cond_attn_out
            )
            cond_attn_proj = attn.output_dropout(cond_attn_proj)
            cond_x = cond_x + cond_gate_self * cond_attn_proj

            cond_mlp_normed = (
                block.layer_norm_mlp(cond_x) * (1 + cond_scale_mlp) + cond_shift_mlp
            )
            cond_mlp_h = block.mlp.layer1(cond_mlp_normed)
            if cond_lora_ffn1 is not None:
                cond_mlp_h = cond_mlp_h + eff_scale * cond_lora_ffn1(cond_mlp_normed)
            cond_mlp_h = block.mlp.activation(cond_mlp_h)
            cond_mlp_out = block.mlp.layer2(cond_mlp_h)
            if cond_lora_ffn2 is not None:
                cond_mlp_out = cond_mlp_out + eff_scale * cond_lora_ffn2(cond_mlp_h)
            cond_x = cond_x + cond_gate_mlp * cond_mlp_out

        self._cond_kv_cache = cache
        # Cache replaces the side-channel — drop slots so a stale write can't
        # confuse the patched forward if the user toggles cache off later.
        for block in self._block_modules:
            block._easycontrol_cond_x_in = None

        kv_bytes = (
            sum(k.numel() + v.numel() for k, v in cache) * cache[0][0].element_size()
        )
        logger.info(
            f"EasyControl: precomputed cond KV cache "
            f"({len(cache)} blocks × 2 tensors, {kv_bytes / 1e6:.0f} MB)"
        )

    def get_effective_scale(self) -> float:
        return self.cond_scale * self.multiplier

    # ------------------------------------------------------------ trainer hooks

    def get_trainable_params(self):
        return list(self.parameters())

    # ------------------------------------------------------------ I/O

    def metadata_fields(self) -> dict[str, str]:
        return {
            "ss_num_blocks": str(self.num_blocks),
            "ss_hidden_size": str(self.hidden_size),
            "ss_num_heads": str(self.num_heads),
            "ss_mlp_ratio": str(self.mlp_ratio),
            "ss_cond_lora_dim": str(self.cond_lora_dim),
            "ss_cond_lora_alpha": str(self.cond_lora_alpha),
            "ss_b_cond_init": str(self.b_cond_init),
            "ss_cond_scale": str(self.cond_scale),
            "ss_apply_ffn_lora": str(int(self.apply_ffn_lora)),
            "ss_cond_res_scale": str(self.cond_res_scale),
            "ss_channel_scaling_alpha": str(self.channel_scaling_alpha),
        }

    def state_dict_for_save(self, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().to(dtype) for k, v in self.state_dict().items()}

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            sd = load_file(file)
        else:
            sd = torch.load(file, map_location="cpu")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logger.warning(
                f"EasyControlNetwork.load_state_dict: missing={missing}, unexpected={unexpected}"
            )
        else:
            logger.info(f"Loaded EasyControl weights from {file} ({len(sd)} tensors)")


# ----------------------------------------------------------------- AdaLN modulation helpers


def _adaln_self_cross_mlp(block: nn.Module, emb, adaln_lora):
    """``(shift, scale, gate)`` triples for self-attn, cross-attn, and mlp.

    Mirrors Anima ``Block._forward``'s modulation computation exactly; factored
    out so the EasyControl target path (two-stream + cached-cond-KV) shares one
    copy instead of inlining the chunk-dance. Returns three 3-tuples.
    """
    if block.use_adaln_lora:
        down_self, down_cross, down_mlp = block.adaln_fused_down(emb).chunk(3, dim=-1)
        self_p = (block.adaln_up_self_attn(down_self) + adaln_lora).chunk(3, dim=-1)
        cross_p = (block.adaln_up_cross_attn(down_cross) + adaln_lora).chunk(3, dim=-1)
        mlp_p = (block.adaln_up_mlp(down_mlp) + adaln_lora).chunk(3, dim=-1)
    else:
        self_p = block.adaln_modulation_self_attn(emb).chunk(3, dim=-1)
        cross_p = block.adaln_modulation_cross_attn(emb).chunk(3, dim=-1)
        mlp_p = block.adaln_modulation_mlp(emb).chunk(3, dim=-1)
    return self_p, cross_p, mlp_p


def _adaln_self_mlp(block: nn.Module, emb, adaln_lora):
    """``(shift, scale, gate)`` triples for self-attn and mlp only.

    The EasyControl cond stream does no cross-attention, so its modulation skips
    the cross third entirely (the fused-down cross slice is computed-and-dropped
    in the ``use_adaln_lora`` path, matching the original inline code). Returns
    two 3-tuples.
    """
    if block.use_adaln_lora:
        down_self, _down_cross, down_mlp = block.adaln_fused_down(emb).chunk(3, dim=-1)
        self_p = (block.adaln_up_self_attn(down_self) + adaln_lora).chunk(3, dim=-1)
        mlp_p = (block.adaln_up_mlp(down_mlp) + adaln_lora).chunk(3, dim=-1)
    else:
        self_p = block.adaln_modulation_self_attn(emb).chunk(3, dim=-1)
        mlp_p = block.adaln_modulation_mlp(emb).chunk(3, dim=-1)
    return self_p, mlp_p


# ----------------------------------------------------------------- target-only path (cached cond KV)


def _target_only_with_cached_cond_kv(
    block: nn.Module,
    x_B_T_H_W_D: torch.Tensor,
    emb_B_T_D: torch.Tensor,
    crossattn_emb: torch.Tensor,
    attn_params,
    rope_cos_sin,
    adaln_lora_B_T_3D,
    cond_k_cached: torch.Tensor,
    cond_v_cached: torch.Tensor,
    b_param: torch.Tensor,
) -> torch.Tensor:
    """Block.forward equivalent for inference when cond KV is cached.

    Identical to baseline ``Block._forward`` except self-attention uses
    ``_extended_target_attention`` over ``[K_t; cond_k_cached]`` /
    ``[V_t; cond_v_cached]`` with the per-block ``b_cond`` logit bias. Cross-attn
    and MLP run baseline. No cond stream — the cache is the cond stream's
    cumulative effect on KV.
    """
    attn = block.self_attn
    T_dim, H_dim, W_dim = x_B_T_H_W_D.shape[1:4]
    scale_attn = attn_params.softmax_scale

    (
        (shift_self_attn, scale_self_attn, gate_self_attn),
        (shift_cross_attn, scale_cross_attn, gate_cross_attn),
        (shift_mlp, scale_mlp, gate_mlp),
    ) = _adaln_self_cross_mlp(block, emb_B_T_D, adaln_lora_B_T_3D)

    sh_self_5 = shift_self_attn[:, :, None, None, :]
    sc_self_5 = scale_self_attn[:, :, None, None, :]
    ga_self_5 = gate_self_attn[:, :, None, None, :]
    sh_cross_5 = shift_cross_attn[:, :, None, None, :]
    sc_cross_5 = scale_cross_attn[:, :, None, None, :]
    ga_cross_5 = gate_cross_attn[:, :, None, None, :]
    sh_mlp_5 = shift_mlp[:, :, None, None, :]
    sc_mlp_5 = scale_mlp[:, :, None, None, :]
    ga_mlp_5 = gate_mlp[:, :, None, None, :]

    # ---- Self-attention (extended over [target; cached cond]) ----
    target_normed = (
        block.layer_norm_self_attn(x_B_T_H_W_D) * (1 + sc_self_5) + sh_self_5
    )
    target_flat = target_normed.flatten(1, 3)
    target_q, target_k, target_v = attn.compute_qkv(
        target_flat, target_flat, rope_cos_sin=rope_cos_sin
    )
    # If the cache was primed at B=1 and we're running at a larger batch
    # (e.g. CFG-batched), broadcast K_c/V_c on the batch dim.
    B_t = target_q.shape[0]
    if cond_k_cached.shape[0] != B_t:
        if cond_k_cached.shape[0] == 1:
            cond_k_cached = cond_k_cached.expand(B_t, -1, -1, -1)
            cond_v_cached = cond_v_cached.expand(B_t, -1, -1, -1)
        else:
            raise RuntimeError(
                f"cond KV cache batch ({cond_k_cached.shape[0]}) "
                f"does not match target batch ({B_t}) and is not 1 to broadcast"
            )
    target_attn_out = _extended_target_attention(
        target_q,
        target_k,
        target_v,
        cond_k_cached,
        cond_v_cached,
        b_param=b_param,
        scale=scale_attn,
        attn_params=attn_params,
    )
    target_attn_proj = attn.output_proj(target_attn_out)
    target_attn_proj = attn.output_dropout(target_attn_proj)
    target_attn_5d = target_attn_proj.unflatten(1, (T_dim, H_dim, W_dim))
    x_B_T_H_W_D = x_B_T_H_W_D + ga_self_5 * target_attn_5d

    # ---- Cross-attention (baseline) ----
    target_cross_normed = (
        block.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + sc_cross_5) + sh_cross_5
    )
    target_cross_out = block.cross_attn(
        target_cross_normed.flatten(1, 3),
        attn_params,
        crossattn_emb,
        rope_cos_sin=rope_cos_sin,
    ).unflatten(1, (T_dim, H_dim, W_dim))
    x_B_T_H_W_D = x_B_T_H_W_D + ga_cross_5 * target_cross_out

    # ---- MLP (baseline) ----
    target_mlp_normed = block.layer_norm_mlp(x_B_T_H_W_D) * (1 + sc_mlp_5) + sh_mlp_5
    target_mlp_out = block.mlp(target_mlp_normed)
    x_B_T_H_W_D = x_B_T_H_W_D + ga_mlp_5 * target_mlp_out

    return x_B_T_H_W_D


# ----------------------------------------------------------------- patched Block.forward


def _make_patched_block_forward(
    block: nn.Module, block_idx: int, ec_net: EasyControlNetwork
):
    """Build a closure that replaces ``Block.forward`` for one DiT block.

    The closure mirrors Anima's ``Block.forward`` checkpoint dispatch — three
    paths (unsloth / cpu_offload / plain torch_checkpoint / no-ckpt) — but
    routes to the two-stream inner instead of the original ``_forward`` when
    cond is active. When no cond is set on the network, falls through to the
    original baseline forward unchanged.

    cond_x flows block-by-block via per-block side channels:
      - ``block._easycontrol_cond_x_in`` is set by the previous block's
        patched forward (or by ``set_cond`` for block 0).
      - The two-stream inner takes ``cond_x_in`` as an explicit arg and
        returns ``cond_x_out`` as an explicit return value, so the per-block
        checkpoint preserves the autograd connection across blocks.
    """
    # Capture once.
    original_forward = block.forward
    b_param = ec_net.b_cond[block_idx]
    cond_lora_qkv = ec_net.cond_lora_qkv[block_idx]
    cond_lora_o = ec_net.cond_lora_o[block_idx]
    cond_lora_ffn1 = ec_net.cond_lora_ffn1[block_idx] if ec_net.apply_ffn_lora else None
    cond_lora_ffn2 = ec_net.cond_lora_ffn2[block_idx] if ec_net.apply_ffn_lora else None

    # The last block's cond_x_out is discarded (no block consumes it — see the
    # next-block write below), so its cond self-attn output, output proj, and
    # MLP are dead compute: cond_lora_o / cond_lora_ffn on the final block never
    # reach the loss. Only its cond K/V (fed to the target's extended attention)
    # are live. Skip the discarded cond-stream evolution on the last block.
    is_last = block_idx == ec_net.num_blocks - 1

    # Lazy imports to avoid a circular at module load.
    from library.anima.models import apply_rotary_pos_emb_qk
    from networks import attention_dispatch as anima_attention

    def _two_stream_inner(
        x_B_T_H_W_D,
        emb_B_T_D,
        crossattn_emb,
        attn_params,
        rope_cos_sin,
        adaln_lora_B_T_3D,
        cond_x_B_S_D,
        cond_emb_B_T_D,
        cond_adaln_lora_B_T_3D,
        cond_rope_cos_sin,
    ):
        """Two-stream block: (target, cond) → (target_out, cond_out)."""
        attn = block.self_attn
        T_dim, H_dim, W_dim = x_B_T_H_W_D.shape[1:4]
        scale_attn = attn_params.softmax_scale

        # ---- AdaLN modulation params for both streams ----
        # Target gets the full self/cross/mlp triples; cond skips cross (it does
        # no cross-attention).
        (
            (shift_self_attn, scale_self_attn, gate_self_attn),
            (shift_cross_attn, scale_cross_attn, gate_cross_attn),
            (shift_mlp, scale_mlp, gate_mlp),
        ) = _adaln_self_cross_mlp(block, emb_B_T_D, adaln_lora_B_T_3D)
        (
            (cond_shift_self_attn, cond_scale_self_attn, cond_gate_self_attn),
            (cond_shift_mlp, cond_scale_mlp, cond_gate_mlp),
        ) = _adaln_self_mlp(block, cond_emb_B_T_D, cond_adaln_lora_B_T_3D)

        # Reshape target shifts/scales/gates for 5D broadcasting.
        # Cond shifts/scales/gates are (B, 1, D); broadcast over (B, S_c, D)
        # naturally — no reshape needed.
        sh_self_5 = shift_self_attn[:, :, None, None, :]
        sc_self_5 = scale_self_attn[:, :, None, None, :]
        ga_self_5 = gate_self_attn[:, :, None, None, :]
        sh_cross_5 = shift_cross_attn[:, :, None, None, :]
        sc_cross_5 = scale_cross_attn[:, :, None, None, :]
        ga_cross_5 = gate_cross_attn[:, :, None, None, :]
        sh_mlp_5 = shift_mlp[:, :, None, None, :]
        sc_mlp_5 = scale_mlp[:, :, None, None, :]
        ga_mlp_5 = gate_mlp[:, :, None, None, :]

        # ============ 1. SELF-ATTENTION (extended target + cond's own) ============
        # Target normalized → flat sequence
        target_normed = (
            block.layer_norm_self_attn(x_B_T_H_W_D) * (1 + sc_self_5) + sh_self_5
        )
        target_flat = target_normed.flatten(1, 3)

        # Target Q/K/V with target RoPE — reuse Attention.compute_qkv (it
        # handles q_norm, k_norm, v_norm + apply_rotary_pos_emb_qk for us).
        target_q, target_k, target_v = attn.compute_qkv(
            target_flat, target_flat, rope_cos_sin=rope_cos_sin
        )

        # Cond normalized
        cond_normed = (
            block.layer_norm_self_attn(cond_x_B_S_D) * (1 + cond_scale_self_attn)
            + cond_shift_self_attn
        )

        # Cond Q/K/V — base + LoRA delta inserted between qkv_proj and the
        # q/k/v norms. We re-implement compute_qkv inline so the LoRA delta
        # lands at the same point in the projection chain that Phase 1.5 used.
        eff_scale = ec_net.cond_scale * ec_net.multiplier
        cond_qkv_base = attn.qkv_proj(cond_normed)
        cond_qkv_delta = cond_lora_qkv(cond_normed)
        cond_qkv = cond_qkv_base + eff_scale * cond_qkv_delta
        cond_q, cond_k, cond_v = cond_qkv.unflatten(
            -1, (3, attn.n_heads, attn.head_dim)
        ).unbind(dim=-3)
        cond_q = attn.q_norm(cond_q)
        cond_k = attn.k_norm(cond_k)
        cond_v = attn.v_norm(cond_v)
        if cond_rope_cos_sin is not None:
            cond_q, cond_k = apply_rotary_pos_emb_qk(
                cond_q, cond_k, cond_rope_cos_sin, tensor_format=attn.qkv_format
            )

        # Target extended attention over [target_k; cond_k].
        target_attn_out = _extended_target_attention(
            target_q,
            target_k,
            target_v,
            cond_k,
            cond_v,
            b_param=b_param,
            scale=scale_attn,
            attn_params=attn_params,
        )

        # Target output projection + gated residual (always runs).
        target_attn_proj = attn.output_dropout(attn.output_proj(target_attn_out))
        target_attn_5d = target_attn_proj.unflatten(1, (T_dim, H_dim, W_dim))
        x_B_T_H_W_D = x_B_T_H_W_D + ga_self_5 * target_attn_5d

        # Cond stream's own self-attention + output proj + residual. Only feeds
        # the next block, so it's dead on the last block (cond K/V already
        # consumed by the target extended attention above).
        if not is_last:
            # Route through the same dispatched backend (flash when available)
            # and softmax_scale as the target, instead of a bare SDPA. cond_q/k/v
            # are already BLHD (B, S_c, n_h, d_h) — dispatch_attention's expected
            # layout — and it returns [B, S_c, n_h*d_h].
            #
            # cond_q/k/v can be fp32 here: the trainable cond LoRA delta (fp32)
            # promotes cond_qkv to fp32 and the q/k/v norms keep it there. Flash
            # only accepts fp16/bf16, so cast all three to the target attention's
            # compute dtype (bf16 under autocast; no-op in pure-fp32 training).
            # Mirrors _extended_target_attention, which casts cond_k/cond_v to
            # target_k.dtype before its own flash call.
            cond_q = cond_q.to(target_v.dtype)
            cond_k = cond_k.to(target_v.dtype)
            cond_v = cond_v.to(target_v.dtype)
            cond_attn_out = anima_attention.dispatch_attention(
                [cond_q, cond_k, cond_v], attn_params=attn_params
            )
            cond_attn_proj = attn.output_dropout(
                attn.output_proj(cond_attn_out) + eff_scale * cond_lora_o(cond_attn_out)
            )
            cond_x_B_S_D = cond_x_B_S_D + cond_gate_self_attn * cond_attn_proj

        # ============ 2. CROSS-ATTENTION (target only) ============
        target_cross_normed = (
            block.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + sc_cross_5) + sh_cross_5
        )
        target_cross_out = block.cross_attn(
            target_cross_normed.flatten(1, 3),
            attn_params,
            crossattn_emb,
            rope_cos_sin=rope_cos_sin,
        ).unflatten(1, (T_dim, H_dim, W_dim))
        x_B_T_H_W_D = x_B_T_H_W_D + ga_cross_5 * target_cross_out

        # ============ 3. MLP ============
        # Target MLP (existing path).
        target_mlp_normed = (
            block.layer_norm_mlp(x_B_T_H_W_D) * (1 + sc_mlp_5) + sh_mlp_5
        )
        target_mlp_out = block.mlp(target_mlp_normed)
        x_B_T_H_W_D = x_B_T_H_W_D + ga_mlp_5 * target_mlp_out

        # Cond MLP — re-implement layer1/act/layer2 inline so we can splice
        # FFN LoRA at layer1 and layer2 outputs (matches Phase 1.5). Discarded
        # on the last block (cond_x_out unused), so skip the FFN entirely there.
        if not is_last:
            cond_mlp_normed = (
                block.layer_norm_mlp(cond_x_B_S_D) * (1 + cond_scale_mlp)
                + cond_shift_mlp
            )
            cond_mlp_h = block.mlp.layer1(cond_mlp_normed)
            if cond_lora_ffn1 is not None:
                cond_mlp_h = cond_mlp_h + eff_scale * cond_lora_ffn1(cond_mlp_normed)
            cond_mlp_h = block.mlp.activation(cond_mlp_h)
            cond_mlp_out = block.mlp.layer2(cond_mlp_h)
            if cond_lora_ffn2 is not None:
                cond_mlp_out = cond_mlp_out + eff_scale * cond_lora_ffn2(cond_mlp_h)
            cond_x_B_S_D = cond_x_B_S_D + cond_gate_mlp * cond_mlp_out

        return x_B_T_H_W_D, cond_x_B_S_D

    # Expose the inner on the block so EasyControlNetwork.compile_cond_stream()
    # can swap in a torch.compile'd version. compile_blocks() only reaches the
    # DiT's own block._forward, which the active (cond-on) training path
    # bypasses — patched_forward calls this inner directly. Without the swap the
    # whole cond stream (incl. every cond LoRA projection) runs eager and
    # torch_compile is a no-op for EasyControl training. patched_forward reads
    # the attribute per call so the compiled version takes effect immediately
    # once swapped.
    block._easycontrol_two_stream_inner = _two_stream_inner

    def patched_forward(
        x_B_T_H_W_D,
        emb_B_T_D,
        crossattn_emb,
        attn_params,
        rope_cos_sin=None,
        adaln_lora_B_T_3D=None,
    ):
        # Inference fast path: cond KV cached → skip the cond stream entirely.
        kv_cache = ec_net._cond_kv_cache
        if kv_cache is not None:
            cond_k_cached, cond_v_cached = kv_cache[block_idx]
            return _target_only_with_cached_cond_kv(
                block,
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                rope_cos_sin,
                adaln_lora_B_T_3D,
                cond_k_cached,
                cond_v_cached,
                b_param,
            )

        cond_state = ec_net._cond_state
        if cond_state is None:
            # No cond — exact baseline DiT behavior.
            return original_forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                rope_cos_sin=rope_cos_sin,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            )

        cond_x_in = block._easycontrol_cond_x_in
        if cond_x_in is None:
            raise RuntimeError(
                f"EasyControl: block[{block_idx}] has cond_state set but no "
                f"_easycontrol_cond_x_in. Did set_cond run before the DiT forward? "
                f"Did the previous block fail to write its cond_x_out?"
            )

        cond_emb = cond_state["cond_emb"]
        cond_adaln_lora = cond_state["cond_adaln_lora"]
        cond_rope = cond_state["cond_rope"]

        # Match cond's dtype to the target stream — under autocast the AdaLN
        # outputs are bf16 while cond_emb / cond_adaln_lora landed in cond_x's
        # dtype upstream. We let the multiplications cast naturally by relying
        # on PyTorch's type promotion rules; nothing to do here.

        # Dispatch the two-stream inner through the SAME checkpoint path that
        # Block.forward uses, with the extra cond args appended to the arg
        # tuple so the checkpoint preserves them as inputs. `inner` is the
        # torch.compile'd two-stream forward once compile_cond_stream() ran
        # (else the eager closure). The checkpoint dispatch itself stays eager —
        # mirrors compile_blocks, which compiles _forward (the inner), never the
        # checkpoint wrapper (unsloth_checkpoint is @torch._disable_dynamo).
        inner = block._easycontrol_two_stream_inner

        # compile_dynamic_seq: mark the two varying seq axes dynamic. This MUST be
        # the checkpointed callable itself (an eager wrapper around the compiled
        # inner), not a one-shot mark before dispatch: the unsloth/torch
        # checkpoint recomputes the inner in BACKWARD via detach_variable, which
        # detaches the top-level tensor args (x / cond_x) into fresh tensors that
        # LOSE the dynamic mark — while the RoPE *tuples* are passed through
        # unchanged (detach_variable skips non-tensors) and KEEP it. That
        # asymmetry (rope hard-dynamic, q specialized constant) is the
        # ConstraintViolationError. Marking inside the recomputed callable
        # re-applies the marks to the detached inputs each pass, so forward and
        # backward agree. Two independent symbols: target self-attn seq (x dim 2,
        # fake-5D (B,1,seq,1,D) under native_flatten — guaranteed on when
        # _dynamic_seq is set) and the cond stream's own seq (cond_x dim 1); each
        # RoPE table rides dim 0. Marking is idempotent across blocks.
        if ec_net._dynamic_seq:
            _compiled_inner = inner
            _lo, _hi = ec_net._dynamic_seq_range

            def inner(
                x_,
                emb_,
                crossattn_,
                attn_params_,
                rope_,
                adaln_,
                cond_x_,
                cond_emb_,
                cond_adaln_,
                cond_rope_,
                _ci=_compiled_inner,
                _lo=_lo,
                _hi=_hi,
            ):
                torch._dynamo.mark_dynamic(x_, 2, min=_lo, max=_hi)
                torch._dynamo.mark_dynamic(cond_x_, 1, min=_lo, max=_hi)
                for _r in (rope_, cond_rope_):
                    if _r is not None:
                        torch._dynamo.mark_dynamic(_r[0], 0, min=_lo, max=_hi)
                        torch._dynamo.mark_dynamic(_r[1], 0, min=_lo, max=_hi)
                return _ci(
                    x_,
                    emb_,
                    crossattn_,
                    attn_params_,
                    rope_,
                    adaln_,
                    cond_x_,
                    cond_emb_,
                    cond_adaln_,
                    cond_rope_,
                )

        if block.training and block.gradient_checkpointing:
            if block.unsloth_offload_checkpointing:
                from library.anima.models import unsloth_checkpoint

                target_x_out, cond_x_out = unsloth_checkpoint(
                    inner,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    rope_cos_sin,
                    adaln_lora_B_T_3D,
                    cond_x_in,
                    cond_emb,
                    cond_adaln_lora,
                    cond_rope,
                )
            elif block.cpu_offload_checkpointing:
                # cpu_offload variant moves activations to CPU on save and
                # back on recompute. Mirrors Block.forward.
                from library.anima.models import to_device, to_cpu

                def _custom_forward(*inputs):
                    device = next(
                        t.device for t in inputs if isinstance(t, torch.Tensor)
                    )
                    device_inputs = to_device(inputs, device)
                    outputs = inner(*device_inputs)
                    return to_cpu(outputs)

                target_x_out, cond_x_out = torch_checkpoint(
                    _custom_forward,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    rope_cos_sin,
                    adaln_lora_B_T_3D,
                    cond_x_in,
                    cond_emb,
                    cond_adaln_lora,
                    cond_rope,
                    use_reentrant=False,
                )
            else:
                target_x_out, cond_x_out = torch_checkpoint(
                    inner,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    rope_cos_sin,
                    adaln_lora_B_T_3D,
                    cond_x_in,
                    cond_emb,
                    cond_adaln_lora,
                    cond_rope,
                    use_reentrant=False,
                )
        else:
            target_x_out, cond_x_out = inner(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                rope_cos_sin,
                adaln_lora_B_T_3D,
                cond_x_in,
                cond_emb,
                cond_adaln_lora,
                cond_rope,
            )

        # Pass cond_x_out to the next block via its side channel. The tensor
        # carries its autograd connection to *this* block's checkpoint output,
        # so backward through the next block flows back here correctly.
        next_idx = block_idx + 1
        if next_idx < ec_net.num_blocks:
            ec_net._block_modules[next_idx]._easycontrol_cond_x_in = cond_x_out
        # else: last block's cond_x_out is unused (cond evolution stops).

        return target_x_out

    return patched_forward


# ----------------------------------------------------------------- trainer integration


class EasyControlMethodAdapter(MethodAdapter):
    """Bridges EasyControl into AnimaTrainer's adapter dispatch.

    Setup: validate the network module exposes set_cond / encode_cond_latent.
    Step: encode the per-step cond latent and prime it on the network before
    the DiT forward, with whole-batch CFG dropout and optional Gaussian
    perturbation in train mode (sigma=0 keeps clean-cond inference valid)."""

    name = "easycontrol"

    def on_network_built(self, ctx: SetupCtx) -> None:
        net = ctx.network
        if not (hasattr(net, "set_cond") and hasattr(net, "encode_cond_latent")):
            raise ValueError(
                "--use_easycontrol requires a network module with set_cond / "
                "encode_cond_latent (e.g. networks.methods.easycontrol)."
            )
        ctx.accelerator.print(
            f"EasyControl: two-stream cond enabled "
            f"(drop_p={getattr(ctx.args, 'easycontrol_drop_p', 0.1)}, "
            f"cond_noise_max={getattr(ctx.args, 'easycontrol_cond_noise_max', 0.0)})"
        )

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        args = ctx.args
        network = ctx.network
        if not hasattr(network, "set_cond"):
            return

        drop_p = float(getattr(args, "easycontrol_drop_p", 0.1) or 0.0)
        if is_train and drop_p > 0.0 and random.random() < drop_p:
            network.set_cond(None)
            return

        # Condition source: prefer a distinct cond latent from the batch
        # (cond≠target tasks like colorization, where the manga-style latent is
        # cached in a parallel cond_cache_dir). Falls back to the target latent
        # for ref==target setups (default EasyControl) → unchanged behavior.
        cond_src = batch.get("cond_latents") if isinstance(batch, dict) else None
        if cond_src is None:
            cond_src = latents
        elif cond_src.ndim == 5:  # 5D fallback (old cache), mirror train.py:761
            cond_src = cond_src.squeeze(2)
        cond_latent = cond_src.to(ctx.accelerator.device, dtype=ctx.weight_dtype)

        sigma_max = float(getattr(args, "easycontrol_cond_noise_max", 0.0) or 0.0)
        if is_train and sigma_max > 0.0:
            sigma = (
                torch.rand(
                    cond_latent.shape[0],
                    *([1] * (cond_latent.ndim - 1)),
                    device=cond_latent.device,
                    dtype=cond_latent.dtype,
                )
                * sigma_max
            )
            cond_latent = cond_latent + sigma * torch.randn_like(cond_latent)

        network.set_cond(cond_latent)
