"""EasyControl network module for Anima — two-stream rewrite (adapter-only, DiT frozen).

Per block, a cond stream (reference latent, patch-embedded + own RoPE at its
native token count) runs alongside the target stream: cond gets its own
self-attn + cond LoRA (qkv/o/ffn) + AdaLN(t=0), while target's self-attention
is LSE-extended over [target_k; cond_k] with a per-block ``b_cond`` logit bias
gating how much cond it attends to. Cross-attn and MLP on the target stream
are baseline; cond skips cross-attn entirely. cond_x flows block-to-block via
a per-block side channel (``block._easycontrol_cond_x_in``), threaded as an
explicit checkpoint arg/return so autograd stays intact under grad-checkpoint.
Step-0 equivalence: ``b_cond`` inits to -10 (exp(-10)~4.5e-5 softmax mass) so
target_out ~ baseline DiT regardless of cond, verified by
``bench/easycontrol/step0_equivalence.py``. See docs/experimental/easycontrol.md.

Train-time contract: caller calls ``network.set_cond(clean_vae_latent)`` once
per batch before the DiT forward (``None``/``clear_cond`` for CFG-dropout —
patched ``Block.forward`` then falls through to the baseline, unpatched path).
No extra call after backward — autograd handles the cond chain via the
per-block checkpoint outputs.

GOTCHA: the cond path bypasses ``block._forward`` entirely (routes through
``_two_stream_inner`` instead), so ``compile_blocks()`` (which only compiles
``block._forward``) never reaches it — training runs the whole cond stream
eager unless ``compile_cond_stream()`` is called separately, after apply_to.
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
DEFAULT_ADALN_IN_DIM = 256  # AdaLN-LoRA bottleneck width (adaln_up_* in_features)
DEFAULT_ADALN_RANK = 8
DEFAULT_TARGET_RANK = 32
DEFAULT_CROSSATTN_DIM = 1024  # crossattn_emb width (T5-target space)

# Body LoRA (plan_render S6): target-stream deltas, the llm_adapter block LoRA
# and the ext-row delta. State-dict prefixes select their lr group (target_lr).
_TARGET_LORA_KINDS = ("qkv", "o", "xq", "xkv", "ffn1", "ffn2")
_BODY_PREFIXES = ("target_lora_", "adapter_lora.", "ext_lora_")


# Cond-stream channel scaling uses a COND-SPECIFIC calibration — the LoRA-family
# main-stream file does NOT transfer to the cond stream (post-GELU mlp.layer2
# inputs diverge; xfer_eff ~0.06). See scripts/calibration/cond_stream_profile.py.
_COND_CHANNEL_STATS_PATH = (
    Path(__file__).resolve().parent.parent
    / "calibration"
    / "cond_channel_stats.safetensors"
)

# kind -> (state_dict ModuleList attr, in_dim selector).
_COND_LORA_KINDS = {
    "qkv": ("cond_lora_qkv", "hidden"),
    "o": ("cond_lora_o", "hidden"),
    "ffn1": ("cond_lora_ffn1", "hidden"),
    "ffn2": ("cond_lora_ffn2", "ffn"),
}


def _cond_lora_calib_key(kind: str, idx: int) -> str:
    """Calibration key naming the DiT Linear a cond LoRA down-proj shadows."""
    suffix = {
        "qkv": "self_attn_qkv_proj",
        "o": "self_attn_output_proj",
        "ffn1": "mlp_layer1",
        "ffn2": "mlp_layer2",
    }[kind]
    return f"lora_unet_blocks_{idx}_{suffix}"


def _load_cond_channel_scales(alpha: float) -> Optional[dict]:
    """mean|x| calibration -> per-channel scale, mirroring
    ``lora_anima/factory._load_channel_scales``. None when ``alpha <= 0``."""
    if alpha <= 0.0:
        return None
    if not _COND_CHANNEL_STATS_PATH.is_file():
        raise FileNotFoundError(
            f"cond channel calibration missing at {_COND_CHANNEL_STATS_PATH}. "
            "Regenerate via `scripts/calibration/cond_stream_profile.py "
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
    """Plain LoRA-style D->r->out_dim delta (up zero-init), standalone (not
    a wrapper around an org_module) — adds a delta to a frozen DiT projection
    on the cond stream only; the caller adds the output."""

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
        # Kaiming uniform on down, zeros on up so the delta is exactly zero at step 0.
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        # SmoothQuant-style per-channel scaling: absorb s into lora_down, store
        # inv_scale=1/s_norm; forward applies x*inv_scale (output-preserving,
        # rebalances per-column grad). See scripts/calibration/cond_stream_profile.py.
        self._has_channel_scale = False
        if channel_scale is not None:
            inv_scale = _absorb_channel_scale(self.lora_down.weight.data, channel_scale)
            self.register_buffer("inv_scale", inv_scale, persistent=True)
            self._has_channel_scale = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Activation-dtype GEMMs: bit-identical under autocast(bf16) to the
            # retired fp32-bottleneck path.
            x_lora = x
            if self._has_channel_scale:
                x_lora = x * self.inv_scale.to(device=x.device, dtype=x.dtype)
            h = F.linear(x_lora, self.lora_down.weight.to(x.dtype))
            h = F.linear(h, self.lora_up.weight.to(x.dtype))
            return (h * self.scale).to(x.dtype)
        # Inference (KV-cache prefill runs without autocast): keep fp32 compute.
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

    # Target-stream adaln LoRA (opt-in; docs/methods/adaln.md §EasyControl).
    # base.toml sets train_adaln=true for the LoRA family, so
    # configs/easycontrol/*.toml MUST pin train_adaln=false to keep this opt-in.
    from networks.lora_anima.config import _as_bool

    train_adaln = _as_bool(kwargs.get("train_adaln"))
    adaln_rank = int(kwargs.get("adaln_rank", DEFAULT_ADALN_RANK) or DEFAULT_ADALN_RANK)
    adaln_alpha = float(kwargs.get("adaln_alpha", 0.0) or 0.0)  # <=0 → √r law

    # Body LoRA (opt-in, plan_render S6) — the reading path the adapter-only
    # form leaves frozen. train_llm_adapter needs the adapter live
    # (cache_llm_adapter_outputs=false + a prompt_embeds TE cache).
    train_target = _as_bool(kwargs.get("train_target"))
    target_rank = int(
        kwargs.get("target_rank", DEFAULT_TARGET_RANK) or DEFAULT_TARGET_RANK
    )
    train_llm_adapter = _as_bool(kwargs.get("train_llm_adapter"))
    train_ext_rows = _as_bool(kwargs.get("train_ext_rows"))
    target_lr = kwargs.get("target_lr")
    target_lr = float(target_lr) if target_lr not in (None, "", "None") else None

    # Deprecated 2026-06-10, accepted so old snapshot TOMLs replay (fp32-bottleneck
    # autograd removed).
    if str(kwargs.get("use_custom_down_autograd", "false")).strip().lower() in (
        "true",
        "1",
    ):
        logger.info(
            "EasyControl: use_custom_down_autograd is deprecated and ignored "
            "(fp32-bottleneck path removed; activation-dtype GEMMs are "
            "bit-identical under the trainer's autocast)"
        )

    # Loads the COND-SPECIFIC calibration (see helper); alpha<=0 disables.
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
    mlp_ratio = DEFAULT_MLP_RATIO  # not exposed on the unet attr

    adaln_in_dim = DEFAULT_ADALN_IN_DIM
    if unet is not None and getattr(unet, "blocks", None):
        up = getattr(unet.blocks[0], "adaln_up_self_attn", None)
        if up is not None:
            adaln_in_dim = up.in_features

    # Body LoRA shapes come off the live DiT.
    crossattn_dim = DEFAULT_CROSSATTN_DIM
    if unet is not None and getattr(unet, "blocks", None):
        crossattn_dim = int(
            getattr(unet.blocks[0].cross_attn, "context_dim", crossattn_dim)
        )
    adapter_linears: dict[str, tuple[int, int]] = {}
    ext_rows = ext_dim = 0
    if train_llm_adapter or train_ext_rows:
        adapter = getattr(unet, "llm_adapter", None) if unet is not None else None
        if adapter is None:
            raise ValueError(
                "train_llm_adapter / train_ext_rows need the Anima DiT with its "
                "llm_adapter at network build time."
            )
        if train_llm_adapter:
            adapter_linears = _adapter_linear_shapes(adapter)
        if train_ext_rows:
            from library.anima.vocab_pack import attached_pack_rows

            ext_rows = attached_pack_rows(unet) or 0
            if ext_rows <= 0:
                raise ValueError(
                    "train_ext_rows needs a vocab pack attached to the DiT "
                    "(set vocab_pack in the config)."
                )
            ext_dim = int(adapter.embed.embedding_dim)

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
        train_adaln=train_adaln,
        adaln_rank=adaln_rank,
        adaln_alpha=adaln_alpha,
        adaln_in_dim=adaln_in_dim,
        train_target=train_target,
        target_rank=target_rank,
        crossattn_dim=crossattn_dim,
        adapter_linears=adapter_linears,
        ext_rows=ext_rows,
        ext_dim=ext_dim,
        target_lr=target_lr,
    )

    # REPA v2 alignment, mirroring networks.lora_anima.factory. DiT is frozen, so
    # the alignment gradient reaches the cond LoRA only through the extended
    # self-attention in blocks <= repa_layer.
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
        # REPA-DoG target band-pass: default-on under use_repa; opt out with
        # repa_target_dog=false.
        network._repa_target_dog = _as_bool(kwargs.get("repa_target_dog"), default=True)
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

    # GOTCHA: channel-scaled checkpoints carry a persistent inv_scale per absorbed
    # lora_down; the buffer MUST be allocated before load (strict=False would
    # silently drop an unallocated inv_scale -> absorbed W*s without the 1/s
    # rebalance -> wrong output). Pass placeholder ones for exactly the modules
    # whose inv_scale is present; load overwrites weight and inv_scale.
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

    # Adaln LoRA presence/sizing from the weights themselves (shapes authoritative).
    adaln_w = (weights_sd or {}).get("adaln_lora_self_attn.0.lora_down.weight")
    train_adaln = adaln_w is not None
    adaln_rank = int(adaln_w.shape[0]) if train_adaln else DEFAULT_ADALN_RANK
    adaln_in_dim = int(adaln_w.shape[1]) if train_adaln else DEFAULT_ADALN_IN_DIM
    adaln_alpha = float(metadata.get("ss_adaln_alpha", 0.0))

    # Body LoRA presence/sizing from the weights too (one rank for all three).
    sd = weights_sd or {}
    tgt_w = sd.get("target_lora_qkv.0.lora_down.weight")
    xkv_w = sd.get("target_lora_xkv.0.lora_down.weight")
    body_rank = int(tgt_w.shape[0]) if tgt_w is not None else None
    adapter_linears = {}
    for k, w in sd.items():
        if k.startswith("adapter_lora.") and k.endswith(".lora_down.weight"):
            name = k[len("adapter_lora.") : -len(".lora_down.weight")]
            up = sd[f"adapter_lora.{name}.lora_up.weight"]
            adapter_linears[name] = (int(w.shape[1]), int(up.shape[0]))
            body_rank = body_rank or int(w.shape[0])
    ext_b = sd.get("ext_lora_b")
    if ext_b is not None:
        body_rank = body_rank or int(ext_b.shape[0])

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
        train_adaln=train_adaln,
        adaln_rank=adaln_rank,
        adaln_alpha=adaln_alpha,
        adaln_in_dim=adaln_in_dim,
        train_target=tgt_w is not None,
        target_rank=body_rank or DEFAULT_TARGET_RANK,
        crossattn_dim=(
            int(xkv_w.shape[1]) if xkv_w is not None else DEFAULT_CROSSATTN_DIM
        ),
        adapter_linears=adapter_linears,
        ext_rows=int(sd["ext_lora_a"].shape[0]) if ext_b is not None else 0,
        ext_dim=int(ext_b.shape[1]) if ext_b is not None else 0,
    )
    return network, weights_sd


def _adapter_lora_key(name: str) -> str:
    """ModuleDict key (dot-free) for an ``llm_adapter.blocks`` Linear."""
    return "blocks_" + name.replace(".", "_")


def _adapter_linear_shapes(adapter: nn.Module) -> dict[str, tuple[int, int]]:
    """``{key: (in, out)}`` for every Linear inside ``llm_adapter.blocks``."""
    return {
        _adapter_lora_key(name): (m.in_features, m.out_features)
        for name, m in adapter.blocks.named_modules()
        if isinstance(m, nn.Linear)
    }


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
        train_adaln: bool = False,
        adaln_rank: int = DEFAULT_ADALN_RANK,
        adaln_alpha: float = 0.0,
        adaln_in_dim: int = DEFAULT_ADALN_IN_DIM,
        train_target: bool = False,
        target_rank: int = DEFAULT_TARGET_RANK,
        crossattn_dim: int = DEFAULT_CROSSATTN_DIM,
        adapter_linears: Optional[dict] = None,
        ext_rows: int = 0,
        ext_dim: int = 0,
        target_lr: Optional[float] = None,
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
        # Position-Aware Interpolation downscale for the cond stream: 1.0 = native
        # (bit-exact to pre-PAI). 0<s<1 downsamples the cond latent (~s^2 tokens)
        # and rescales cond's RoPE positions back onto the target grid.
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

        # Per-block cond LoRA on self_attn (qkv: fused D->3D; o: D->D).
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

        # Per-block cond LoRA on FFN (layer1: D->4D, layer2: 4D->D).
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

        # Target-stream adaln LoRA (opt-in): per-block, per-branch delta on the
        # AdaLN-LoRA up-projections. Applied ONLY on the two-stream / cached-KV
        # target paths — the no-cond fallback runs original Block.forward, so
        # the delta is cond-gated by construction ("no reference = exact
        # baseline DiT" survives). Zero-init up keeps step-0 equivalence.
        # Rationale + sizing: docs/methods/adaln.md.
        self.train_adaln = bool(train_adaln)
        self.adaln_rank = int(adaln_rank)
        self.adaln_in_dim = int(adaln_in_dim)
        if adaln_alpha <= 0.0:
            # √r law (docs/methods/adaln.md): keep α/√r consistent with the
            # cond LoRA rather than linearly scaling α/r.
            adaln_alpha = cond_lora_alpha * math.sqrt(
                self.adaln_rank / max(cond_lora_dim, 1)
            )
        self.adaln_alpha = float(adaln_alpha)
        if self.train_adaln:
            self.adaln_lora_self_attn = nn.ModuleList(
                [
                    _LoRAProj(
                        self.adaln_in_dim, 3 * D, self.adaln_rank, self.adaln_alpha
                    )
                    for _ in range(num_blocks)
                ]
            )
            self.adaln_lora_cross_attn = nn.ModuleList(
                [
                    _LoRAProj(
                        self.adaln_in_dim, 3 * D, self.adaln_rank, self.adaln_alpha
                    )
                    for _ in range(num_blocks)
                ]
            )
            self.adaln_lora_mlp = nn.ModuleList(
                [
                    _LoRAProj(
                        self.adaln_in_dim, 3 * D, self.adaln_rank, self.adaln_alpha
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self.adaln_lora_self_attn = None
            self.adaln_lora_cross_attn = None
            self.adaln_lora_mlp = None

        # Body LoRA (opt-in, plan_render S6): the ext row → pixel path that
        # the adapter-only form leaves with zero trainable weights.
        # (1) target-stream deltas on self-attn qkv/out, cross-attn q/kv and
        #     mlp ffn1/ffn2 — applied only on the cond-active paths (the
        #     no-cond fallback runs the original Block.forward), so they are
        #     cond-gated by construction like the adaln deltas;
        # (2) LoRA on every Linear of the llm_adapter blocks (only fires when
        #     the adapter runs live: cache_llm_adapter_outputs=false);
        # (3) a low-rank delta on the vocab pack's ext rows (LoRA-embedding
        #     layout: A gathered per ext id, B zero-init).
        # Zero-init up/B sides keep step-0 equivalence; all three ride their
        # own lr group when target_lr is set. alpha = rank (scale 1).
        self.train_target = bool(train_target)
        self.target_rank = int(target_rank)
        self.target_alpha = float(self.target_rank)
        self.crossattn_dim = int(crossattn_dim)
        self.target_lr = None if target_lr is None else float(target_lr)
        tr, ta = self.target_rank, self.target_alpha
        target_dims = {
            "qkv": (D, 3 * D),
            "o": (D, D),
            "xq": (D, D),
            "xkv": (self.crossattn_dim, 2 * D),
            "ffn1": (D, self.ffn_dim),
            "ffn2": (self.ffn_dim, D),
        }
        for kind in _TARGET_LORA_KINDS:
            i_dim, o_dim = target_dims[kind]
            setattr(
                self,
                f"target_lora_{kind}",
                nn.ModuleList(
                    [_LoRAProj(i_dim, o_dim, tr, ta) for _ in range(num_blocks)]
                )
                if self.train_target
                else None,
            )

        adapter_linears = dict(adapter_linears or {})
        self.train_llm_adapter = bool(adapter_linears)
        self.adapter_lora = (
            nn.ModuleDict(
                {
                    k: _LoRAProj(i_dim, o_dim, tr, ta)
                    for k, (i_dim, o_dim) in sorted(adapter_linears.items())
                }
            )
            if adapter_linears
            else None
        )

        self.ext_rows = int(ext_rows)
        if self.ext_rows > 0:
            self.ext_lora_a = nn.Parameter(torch.randn(self.ext_rows, tr))
            self.ext_lora_b = nn.Parameter(torch.zeros(tr, int(ext_dim)))
        else:
            self.ext_lora_a = None
            self.ext_lora_b = None

        # apply_to state for (2)/(3), plus the "adapter LoRA never ran" guard
        # EasyControlMethodAdapter checks (a cached crossattn_emb bypasses it).
        self._adapter_patches: list = []
        self._ext_hook_handles: list = []
        self._adapter_lora_calls = 0
        self._train_primes = 0

        # Per-block scalar additive logit bias on cond keys. Init -10 → cond
        # softmax mass ≈ 4.5e-5 at step 0 → target_out ≈ baseline DiT.
        # GOTCHA: 0-d Parameters (not one [num_blocks] Parameter) so each block's
        # closure captures a Parameter object, not a Python int index — dynamo
        # specializes on int closure cells (one recompile per block).
        self.b_cond = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(b_cond_init, dtype=torch.float32))
                for _ in range(num_blocks)
            ]
        )

        # Populated by apply_to(). Plain lists (NOT nn.ModuleList) so PyTorch
        # doesn't re-parent the frozen DiT into this network's parameter tree.
        self._dit: Optional[nn.Module] = None
        self._block_modules: list[nn.Module] = []
        self._original_block_forwards: list = []
        self._patched: bool = False

        # Per-step cond state: None = no cond / CFG-dropped → patched block
        # forward falls through to baseline. When set: "cond_emb" (B,1,D)
        # RMSNormed t_embedder(zeros), "cond_adaln_lora" (B,1,3*D_adaln) or None,
        # "cond_rope" (cos,sin) at cond's native token count. cond_x_init for
        # block 0 lives on block_modules[0]._easycontrol_cond_x_in.
        self._cond_state: Optional[dict] = None

        # Inference KV cache: per-block (cond_k, cond_v) post-RoPE-and-norm.
        # Populated by precompute_cond_kv(); when non-None the patched
        # Block.forward bypasses the cond stream. Training keeps it None (every
        # step needs the LoRA grad).
        self._cond_kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None

        # compile_dynamic_seq: when set by compile_cond_stream, marks the
        # target/cond seq axes dynamic so the two-stream inner compiles one graph
        # instead of one per (target × cond) token-count pair. GOTCHA: the cond
        # axis gets its OWN bound — at cond_res_scale<1 the cond stream runs a
        # downscaled grid (~scale² tokens), so reusing the target band raises
        # ConstraintViolationError on the cond mark.
        self._dynamic_seq: bool = False
        self._dynamic_seq_range: Optional[tuple] = None
        self._dynamic_seq_cond_range: Optional[tuple] = None

        n_scaled = sum(
            1
            for m in self.modules()
            if isinstance(m, _LoRAProj) and m._has_channel_scale
        )
        total = sum(p.numel() for p in self.parameters())
        adaln_desc = (
            f"r={self.adaln_rank} alpha={self.adaln_alpha:g}"
            if self.train_adaln
            else "off"
        )
        logger.info(
            f"EasyControlNetwork: blocks={num_blocks}, hidden={hidden_size}/{num_heads}h, "
            f"r={cond_lora_dim} alpha={cond_lora_alpha}, ffn_lora={apply_ffn_lora}, "
            f"adaln_lora={adaln_desc}, "
            f"body[target={'r%d' % tr if self.train_target else 'off'}, "
            f"llm_adapter={len(adapter_linears) or 'off'}, "
            f"ext_rows={self.ext_rows or 'off'}, target_lr={self.target_lr}], "
            f"b_cond_init={b_cond_init}, cond_scale={cond_scale}, "
            f"cond_res_scale={self.cond_res_scale}, "
            f"channel_scaling_alpha={self.channel_scaling_alpha} "
            f"({n_scaled} cond projections rebalanced), "
            f"params={total / 1e6:.1f}M"
        )

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
        if self.train_adaln:
            b0 = unet.blocks[0]
            if not getattr(b0, "use_adaln_lora", False):
                raise ValueError(
                    "train_adaln requires the AdaLN-LoRA bottleneck form "
                    "(use_adaln_lora=True); this DiT uses the vanilla adaln MLPs."
                )
            if b0.adaln_up_self_attn.in_features != self.adaln_in_dim:
                raise ValueError(
                    f"adaln in-dim mismatch: network built for {self.adaln_in_dim}, "
                    f"DiT has {b0.adaln_up_self_attn.in_features}."
                )
        if self.train_target:
            ctx_dim = getattr(unet.blocks[0].cross_attn, "context_dim", None)
            if ctx_dim is not None and ctx_dim != self.crossattn_dim:
                raise ValueError(
                    f"cross-attn context dim mismatch: network built for "
                    f"{self.crossattn_dim}, DiT has {ctx_dim}."
                )

        # Bypass nn.Module.__setattr__ — a plain assignment would register the
        # frozen DiT as a submodule, inflating parameters().
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

        if self.adapter_lora is not None:
            self._patch_llm_adapter(unet)
        if self.ext_rows > 0:
            self._hook_ext_rows(unet)

        self._patched = True
        logger.info(
            f"EasyControl: patched Block.forward on {len(self._block_modules)} blocks"
        )

    def _patch_llm_adapter(self, unet) -> None:
        """Wrap each ``llm_adapter.blocks`` Linear's forward with its LoRA delta."""
        adapter = getattr(unet, "llm_adapter", None)
        if adapter is None:
            raise ValueError("train_llm_adapter needs a DiT with an llm_adapter")
        found = {
            _adapter_lora_key(name): m
            for name, m in adapter.blocks.named_modules()
            if isinstance(m, nn.Linear)
        }
        missing = sorted(set(self.adapter_lora.keys()) - set(found))
        if missing:
            raise ValueError(
                f"llm_adapter has no Linear for {len(missing)} adapter LoRA "
                f"keys (first: {missing[:3]})"
            )
        for key, lora in self.adapter_lora.items():
            lin = found[key]
            self._adapter_patches.append((lin, lin.forward))
            lin.forward = _make_adapter_lora_forward(lin.forward, lora, self)
        logger.info(
            f"EasyControl: LoRA on {len(self._adapter_patches)} llm_adapter Linears"
        )

    def _hook_ext_rows(self, unet) -> None:
        """Add ``A[ext] @ B`` at the ext-id positions of ``llm_adapter.embed``.

        The pack's own pre-hook clamps ext ids to ``<unk>``, so ours is
        prepended to see the raw ids; our forward hook runs after the pack's
        (which wrote the pack rows), so the delta lands on top of them.
        GOTCHA: re-attaching a pack after apply_to re-registers its forward
        hook after ours and silently overwrites the delta.
        """
        from library.anima.ext_vocab import T5_TABLE_SIZE
        from library.anima.vocab_pack import attached_pack_rows

        rows = attached_pack_rows(unet)
        if rows != self.ext_rows:
            raise ValueError(
                f"train_ext_rows: network built for {self.ext_rows} ext rows, "
                f"the DiT has {rows or 'no'} pack rows attached."
            )
        embed = unet.llm_adapter.embed
        state: dict = {}

        def _ids_pre_hook(module, args):
            state.pop("mask", None)
            if args and torch.is_tensor(args[0]):
                mask = args[0] >= T5_TABLE_SIZE
                if bool(mask.any()):
                    state["mask"] = mask
                    state["ext"] = args[0][mask] - T5_TABLE_SIZE
            return None

        def _delta_hook(module, args, output):
            mask = state.pop("mask", None)
            if mask is None:
                return None
            ext = state.pop("ext").to(self.ext_lora_a.device)
            delta = (self.ext_lora_a[ext] @ self.ext_lora_b) * self.multiplier
            out = output.clone()
            out[mask] = out[mask] + delta.to(out.dtype)
            return out

        self._ext_hook_handles = [
            embed.register_forward_pre_hook(_ids_pre_hook, prepend=True),
            embed.register_forward_hook(_delta_hook),
        ]

    def compile_cond_stream(
        self,
        backend: str = "inductor",
        mode: Optional[str] = None,
        n_token_families: Optional[int] = None,
        dynamic_seq: bool = False,
        seq_range: Optional[tuple] = None,
    ):
        """torch.compile each block's two-stream cond forward.

        GOTCHA: ``compile_blocks()`` only reaches the DiT's own ``block._forward``;
        the cond-on training path routes through ``_two_stream_inner`` instead
        (see ``_make_patched_block_forward``), so without this call the entire
        cond stream runs eager and ``torch_compile`` is a no-op for EasyControl.
        Mirrors compile_blocks (``backend=inductor``, ``dynamic=False``, same
        ``mode``); flash attention may graph-break, which is fine. Call AFTER
        apply_to (compile-after-apply invariant).

        ``dynamic_seq`` collapses the per-(target × cond) token-count graph
        cascade to one graph — BOTH seq axes vary here, so the patched forward
        wraps the compiled inner in an eager ``mark_dynamic`` prologue that must
        re-apply on the grad-checkpoint backward RECOMPUTE too (else
        ``detach_variable`` strips the latent marks but keeps the RoPE-tuple
        marks and dynamo raises ``ConstraintViolationError``). ``seq_range``
        bounds the marks; ``None`` falls back to the canonical 1024 table.
        """
        if not self._patched:
            raise RuntimeError("compile_cond_stream requires apply_to() first")

        from library.runtime.dynamo import pin_dynamo_limit

        # GOTCHA: the two-stream inner needs far more graphs than block._forward
        # (target × cond token-count product, × grad-on/off state, × flash
        # graph-break segments) — exceeds dynamo's recompile_limit default of 8,
        # and a plain config write REVERTS to 8 in the backward compile context
        # (see pin_dynamo_limit), so pin the canonical .default.
        n = n_token_families if n_token_families is not None else 2
        per_obj = 4 * n + 16
        pin_dynamo_limit("recompile_limit", per_obj)
        # accumulated_recompile_limit is the cross-code-object ceiling.
        pin_dynamo_limit(
            "accumulated_recompile_limit", len(self._block_modules) * per_obj
        )

        # With train_adaln the target shift/scale/gate carry grad, triggering a
        # mix-order-reduction fusion hazard that contradicts strict dynamic-seq
        # marks — same as the DiT path (docs/optimizations/for_compile.md §2.6).
        # compile_blocks usually pins this already; repeat here for standalone use.
        if dynamic_seq and self.train_adaln:
            import torch._inductor.config as _inductor_config

            if _inductor_config.triton.mix_order_reduction:
                from library.runtime.dynamo import pin_inductor_flag

                pin_inductor_flag("triton.mix_order_reduction", False)
                logger.info(
                    "EasyControl: inductor mix_order_reduction disabled "
                    "(train_adaln under dynamic-seq marks)"
                )

        # dynamic_seq compiles static and lets the patched forward mark the seq
        # axes (not torch.compile(dynamic=True)). Derive their (min, max) bound.
        self._dynamic_seq = dynamic_seq
        if dynamic_seq:
            if seq_range is not None:
                self._dynamic_seq_range = (int(seq_range[0]), int(seq_range[1]))
            else:
                from library.datasets.buckets import token_count_range

                self._dynamic_seq_range = token_count_range([1024])
            self._dynamic_seq_cond_range = self._cond_seq_range(self._dynamic_seq_range)

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
            f"dynamic_seq={dynamic_seq} seq∈{self._dynamic_seq_range} "
            f"cond_seq∈{self._dynamic_seq_cond_range}, "
            f"recompile_limit pinned to {per_obj})"
        )

    def _cond_seq_range(self, seq_range: tuple) -> tuple:
        """Token-count band for the COND stream, given the target's band.

        PAI (``encode_cond_latent``) downscales the cond latent by
        ``cond_res_scale`` per axis, landing its token count near ``s²`` of the
        target's — marking the cond seq axis with the target band would raise
        ``ConstraintViolationError`` once ``cond_res_scale < 1``. Per-axis
        rounding means the exact count isn't ``s²·n``, so pad generously; these
        are only bounds on a symbolic axis (too wide costs a slightly more
        general kernel, too narrow is a hard error).
        """
        lo, hi = int(seq_range[0]), int(seq_range[1])
        s = float(self.cond_res_scale)
        if s >= 1.0:
            return (lo, hi)
        lo_c = max(1, int(math.floor(lo * s * s * 0.7)))
        hi_c = max(lo_c + 1, int(math.ceil(hi * s * s * 1.4)))
        return (lo_c, hi_c)

    def _fit_cond_seq_range(self, n: int) -> None:
        """Widen the cond band in place if ``n`` tokens fall outside it.

        ``_cond_seq_range`` is only an estimate — on a cond≠target subset the
        cond image can free-fit to another shape/tier entirely. Rather than
        crash deep in dynamo, widen with headroom and let dynamo recompile once.
        """
        rng = self._dynamic_seq_cond_range
        if rng is None or rng[0] <= n <= rng[1]:
            return
        lo = min(rng[0], max(1, int(n * 0.8)))
        hi = max(rng[1], int(math.ceil(n * 1.25)))
        self._dynamic_seq_cond_range = (lo, hi)
        logger.info(
            f"EasyControl: cond seq {n} outside the marked band {rng} — widened to "
            f"{self._dynamic_seq_cond_range} (one dynamo recompile of the two-stream "
            f"inner; expected on cond≠target subsets where the cond image free-fits "
            f"to its own shape)"
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
        for lin, orig in self._adapter_patches:
            lin.forward = orig
        self._adapter_patches.clear()
        for h in self._ext_hook_handles:
            h.remove()
        self._ext_hook_handles.clear()
        object.__setattr__(self, "_dit", None)
        self._patched = False
        self._cond_kv_cache = None

    def encode_cond_latent(
        self,
        cond_latent: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Patch-embed the clean VAE latent into ``[B, S_c, D]`` cond tokens
        plus the matching RoPE table at cond's native shape.

        Reuses the DiT's (frozen) ``x_embedder``/``pos_embedder``. Both outputs
        are kept at cond's native token count — no static padding, which would
        leak zero tokens into the cond self-attn and target's LSE-extended attn.

        Returns ``(cond_x [B, S_c, D], cond_rope (cos, sin) each [S_c,1,1,D_head])``.
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

        # PAI (cond-only downscale): at cond_res_scale<1, downsample the cond
        # latent and rescale its RoPE positions back onto the target grid.
        # cond_res_scale==1 skips this block → bit-exact to native resolution.
        h_scale = w_scale = 1.0
        if self.cond_res_scale < 1.0:
            p = self._dit.patch_spatial
            full_gh, full_gw = H // p, W // p  # target patch grid
            new_H = max(p, int(round(H * self.cond_res_scale / p)) * p)
            new_W = max(p, int(round(W * self.cond_res_scale / p)) * p)
            if (new_H, new_W) != (H, W):
                # area resampling = anti-aliased avg pooling (correct for downscale).
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

        cond_x_5d, cond_rope = self._dit.prepare_embedded_sequence(
            cond_latent,
            fps=None,
            padding_mask=padding_mask,
        )
        # PAI: rescale cond RoPE positions onto the target grid (only when downscaling)
        if h_scale != 1.0 or w_scale != 1.0:
            cond_rope = self._dit.pos_embedder.generate_embeddings_scaled(
                cond_x_5d.shape,
                h_scale=h_scale,
                w_scale=w_scale,
                fps=None,
            )

        # GOTCHA: pin cond_x to cond_latent's dtype — patch-embed runs outside the
        # forward's autocast scope, so without the pin cond_x's dtype flip-flops
        # (fp32 train vs bf16 no_grad val/sample), doubling the compiled
        # two-stream graphs keyed on dtype. Safe: cond_q/k/v are re-cast downstream.
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

        # New reference invalidates any prior cache (two-stream path until reprimed)
        self._cond_kv_cache = None

        cond_x, cond_rope = self.encode_cond_latent(
            cond_latent, padding_mask=padding_mask
        )

        # cond_temb at t=0 through the same t_embedder as target. Pooled-text
        # projection intentionally NOT applied — cond is a reference image at
        # t=0 with no text channel.
        B = cond_latent.shape[0]
        device = cond_x.device
        zeros = torch.zeros(B, 1, device=device, dtype=cond_x.dtype)
        cond_emb_B_T_D, cond_adaln_lora_B_T_3D = self._dit.t_embedder(zeros)
        cond_emb_B_T_D = self._dit.t_embedding_norm(cond_emb_B_T_D)

        self._cond_state = {
            "cond_emb": cond_emb_B_T_D,
            "cond_adaln_lora": cond_adaln_lora_B_T_3D,
            "cond_rope": cond_rope,
        }
        # Block 0's input; later blocks' slots are written by the prior block's forward
        self._block_modules[0]._easycontrol_cond_x_in = cond_x

    def clear_cond(self) -> None:
        self._cond_state = None
        for block in self._block_modules:
            block._easycontrol_cond_x_in = None
        self._cond_kv_cache = None

    def clear_cond_kv_cache(self) -> None:
        """Drop the per-block KV cache; recomputed on the next forward (or
        until ``precompute_cond_kv`` is called again)."""
        self._cond_kv_cache = None

    @torch.no_grad()
    def precompute_cond_kv(self) -> None:
        """Walk the cond stream once and cache (cond_k, cond_v) per block.

        Inference-only: the cond stream is deterministic across denoising steps
        (cond_temb = t_embedder(zeros), no dependence on the noisy target), so
        the K/V tensors target's extended self-attn consumes can be computed
        once and reused across every step and CFG branch. After this call, the
        patched ``Block.forward`` skips all cond work and feeds the cached
        (cond_k, cond_v) directly into ``_extended_target_attention``.

        Caller contract: ``set_cond`` must have run first. Changing
        ``multiplier``/``cond_scale`` after caching makes the cache stale —
        call ``clear_cond_kv_cache`` and re-prime.
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

        # Run cond self-attn through the SAME dispatched backend training uses
        # (bare SDPA vs dispatched flash diverges at bf16-ulp over 28 blocks).
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

            (
                (cond_shift_self, cond_scale_self, cond_gate_self),
                (cond_shift_mlp, cond_scale_mlp, cond_gate_mlp),
            ) = _adaln_self_mlp(block, cond_emb, cond_adaln_lora)

            # cond Q/K/V with LoRA + RoPE — this is what we cache.
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

            # Last block's evolved cond_x is never consumed — skip its dead
            # self-attn + output proj + MLP (mirrors is_last).
            if idx == last_idx:
                continue

            # Cast cond_q/k/v to the compute dtype — flash rejects fp32 and the
            # fp32 cond-LoRA delta + q/k/v norms can promote them.
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
        # Cache replaces the side-channel — drop stale slots
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

    def get_trainable_params(self):
        return list(self.parameters())

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        """One group, or two when body LoRA is on and ``target_lr`` is set:
        the cond adapter at the run lr, the body params at ``target_lr``."""
        body = [
            p
            for n, p in self.named_parameters()
            if p.requires_grad and n.startswith(_BODY_PREFIXES)
        ]
        if not body or self.target_lr is None:
            return super().prepare_optimizer_params_with_multiple_te_lrs(
                text_encoder_lr, unet_lr, default_lr
            )
        body_ids = {id(p) for p in body}
        rest = [p for p in self.get_trainable_params() if id(p) not in body_ids]
        params = [
            {"params": rest, "lr": unet_lr or default_lr},
            {"params": body, "lr": self.target_lr},
        ]
        return params, [self.network_spec, f"{self.network_spec}_body"]

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
            "ss_train_adaln": str(int(self.train_adaln)),
            "ss_adaln_rank": str(self.adaln_rank),
            "ss_adaln_alpha": str(self.adaln_alpha),
            "ss_train_target": str(int(self.train_target)),
            "ss_target_rank": str(self.target_rank),
            "ss_train_llm_adapter": str(int(self.train_llm_adapter)),
            "ss_ext_rows": str(self.ext_rows),
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


def _adaln_self_cross_mlp(
    block: nn.Module, emb, adaln_lora, adaln_deltas=None, delta_scale: float = 1.0
):
    """``(shift, scale, gate)`` triples for self-attn, cross-attn, and mlp.

    Mirrors Anima ``Block._forward``'s modulation computation exactly, factored
    out so the EasyControl target path (two-stream + cached-cond-KV) shares one
    copy. ``adaln_deltas``: optional ``(self, cross, mlp)`` triple of
    ``_LoRAProj`` modules (target-stream adaln LoRA), each adding a delta on
    top of the frozen ``adaln_up_*`` up-projection. Requires the AdaLN-LoRA
    bottleneck form (guarded in ``apply_to``).
    """
    if block.use_adaln_lora:
        down_self, down_cross, down_mlp = block.adaln_fused_down(emb).chunk(3, dim=-1)
        up_self = block.adaln_up_self_attn(down_self)
        up_cross = block.adaln_up_cross_attn(down_cross)
        up_mlp = block.adaln_up_mlp(down_mlp)
        if adaln_deltas is not None:
            d_self, d_cross, d_mlp = adaln_deltas
            up_self = up_self + delta_scale * d_self(down_self)
            up_cross = up_cross + delta_scale * d_cross(down_cross)
            up_mlp = up_mlp + delta_scale * d_mlp(down_mlp)
        self_p = (up_self + adaln_lora).chunk(3, dim=-1)
        cross_p = (up_cross + adaln_lora).chunk(3, dim=-1)
        mlp_p = (up_mlp + adaln_lora).chunk(3, dim=-1)
    else:
        self_p = block.adaln_modulation_self_attn(emb).chunk(3, dim=-1)
        cross_p = block.adaln_modulation_cross_attn(emb).chunk(3, dim=-1)
        mlp_p = block.adaln_modulation_mlp(emb).chunk(3, dim=-1)
    return self_p, cross_p, mlp_p


def _adaln_self_mlp(block: nn.Module, emb, adaln_lora):
    """``(shift, scale, gate)`` triples for self-attn and mlp only — the cond
    stream does no cross-attention, so its modulation skips the cross third."""
    if block.use_adaln_lora:
        down_self, _down_cross, down_mlp = block.adaln_fused_down(emb).chunk(3, dim=-1)
        self_p = (block.adaln_up_self_attn(down_self) + adaln_lora).chunk(3, dim=-1)
        mlp_p = (block.adaln_up_mlp(down_mlp) + adaln_lora).chunk(3, dim=-1)
    else:
        self_p = block.adaln_modulation_self_attn(emb).chunk(3, dim=-1)
        mlp_p = block.adaln_modulation_mlp(emb).chunk(3, dim=-1)
    return self_p, mlp_p


def _lora_add(base: torch.Tensor, lora, x: torch.Tensor, s: float) -> torch.Tensor:
    """``base + s·lora(x)`` cast to the base dtype (target dtypes stay baseline)."""
    return base + (s * lora(x)).to(base.dtype)


def _make_adapter_lora_forward(orig_forward, lora, ec_net):
    """Linear.forward replacement on an llm_adapter block: base + LoRA delta."""

    def forward(x):
        ec_net._adapter_lora_calls += 1
        out = orig_forward(x)
        return _lora_add(out, lora, x, ec_net.multiplier)

    return forward


# Target-stream sublayers with the body LoRA spliced in. ``loras`` is the
# per-block ``(qkv, o, xq, xkv, ffn1, ffn2)`` tuple or None — None runs the
# frozen module exactly as Block._forward does (bit-exact baseline).


def _target_self_qkv(attn, x_flat, rope_cos_sin, loras, s: float):
    if loras is None:
        return attn.compute_qkv(x_flat, x_flat, rope_cos_sin=rope_cos_sin)
    from library.anima.models import apply_rotary_pos_emb_qk

    qkv = _lora_add(attn.qkv_proj(x_flat), loras[0], x_flat, s)
    q, k, v = qkv.unflatten(-1, (3, attn.n_heads, attn.head_dim)).unbind(dim=-3)
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    v = attn.v_norm(v)
    if rope_cos_sin is not None:
        q, k = apply_rotary_pos_emb_qk(
            q, k, rope_cos_sin, tensor_format=attn.qkv_format
        )
    return q, k, v


def _target_out_proj(attn, attn_out, loras, s: float):
    out = attn.output_proj(attn_out)
    if loras is not None:
        out = _lora_add(out, loras[1], attn_out, s)
    return attn.output_dropout(out)


def _target_cross_attn(attn, x_flat, attn_params, context, rope_cos_sin, loras, s):
    """``Attention.forward`` (cross) with q / kv deltas."""
    if loras is None:
        return attn(x_flat, attn_params, context, rope_cos_sin=rope_cos_sin)
    if getattr(attn, "_ctx_k_bias", None) is not None:
        raise NotImplementedError(
            "cross-attn key bias is not wired into the EasyControl body LoRA path"
        )
    from networks import attention_dispatch as anima_attention

    q = _lora_add(attn.q_proj(x_flat), loras[2], x_flat, s)
    q = q.unflatten(-1, (attn.n_heads, attn.head_dim))
    kv = _lora_add(attn.kv_proj(context), loras[3], context, s)
    k, v = kv.unflatten(-1, (2, attn.n_heads, attn.head_dim)).unbind(dim=-3)
    q = attn.q_norm(q)
    k = attn.k_norm(k)
    v = attn.v_norm(v)
    if q.dtype != v.dtype:
        if not attn_params.supports_fp32 and torch.is_autocast_enabled():
            q = q.to(v.dtype)
            k = k.to(v.dtype)
    out = anima_attention.dispatch_attention([q, k, v], attn_params=attn_params)
    return attn.output_dropout(attn.output_proj(out))


def _target_mlp(mlp, x, loras, s: float):
    if loras is None:
        return mlp(x)
    h = _lora_add(mlp.layer1(x), loras[4], x, s)
    h = mlp.activation(h)
    return _lora_add(mlp.layer2(h), loras[5], h, s)


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
    adaln_deltas=None,
    adaln_delta_scale: float = 1.0,
    target_loras=None,
) -> torch.Tensor:
    """Block.forward equivalent for inference when cond KV is cached.

    Identical to baseline ``Block._forward`` except self-attention uses
    ``_extended_target_attention`` over ``[K_t; cond_k_cached]`` /
    ``[V_t; cond_v_cached]`` with the per-block ``b_cond`` logit bias.
    Cross-attn and MLP run baseline; no cond stream.
    """
    attn = block.self_attn
    T_dim, H_dim, W_dim = x_B_T_H_W_D.shape[1:4]
    scale_attn = attn_params.softmax_scale

    (
        (shift_self_attn, scale_self_attn, gate_self_attn),
        (shift_cross_attn, scale_cross_attn, gate_cross_attn),
        (shift_mlp, scale_mlp, gate_mlp),
    ) = _adaln_self_cross_mlp(
        block,
        emb_B_T_D,
        adaln_lora_B_T_3D,
        adaln_deltas=adaln_deltas,
        delta_scale=adaln_delta_scale,
    )

    sh_self_5 = shift_self_attn[:, :, None, None, :]
    sc_self_5 = scale_self_attn[:, :, None, None, :]
    ga_self_5 = gate_self_attn[:, :, None, None, :]
    sh_cross_5 = shift_cross_attn[:, :, None, None, :]
    sc_cross_5 = scale_cross_attn[:, :, None, None, :]
    ga_cross_5 = gate_cross_attn[:, :, None, None, :]
    sh_mlp_5 = shift_mlp[:, :, None, None, :]
    sc_mlp_5 = scale_mlp[:, :, None, None, :]
    ga_mlp_5 = gate_mlp[:, :, None, None, :]

    # Self-attention extended over [target; cached cond].
    target_normed = (
        block.layer_norm_self_attn(x_B_T_H_W_D) * (1 + sc_self_5) + sh_self_5
    )
    target_flat = target_normed.flatten(1, 3)
    target_q, target_k, target_v = _target_self_qkv(
        attn, target_flat, rope_cos_sin, target_loras, adaln_delta_scale
    )
    # Broadcast a B=1-primed cache onto a larger (CFG-batched) target batch
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
    target_attn_proj = _target_out_proj(
        attn, target_attn_out, target_loras, adaln_delta_scale
    )
    target_attn_5d = target_attn_proj.unflatten(1, (T_dim, H_dim, W_dim))
    x_B_T_H_W_D = x_B_T_H_W_D + ga_self_5 * target_attn_5d

    # Cross-attention (baseline)
    target_cross_normed = (
        block.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + sc_cross_5) + sh_cross_5
    )
    target_cross_out = _target_cross_attn(
        block.cross_attn,
        target_cross_normed.flatten(1, 3),
        attn_params,
        crossattn_emb,
        rope_cos_sin,
        target_loras,
        adaln_delta_scale,
    ).unflatten(1, (T_dim, H_dim, W_dim))
    x_B_T_H_W_D = x_B_T_H_W_D + ga_cross_5 * target_cross_out

    # MLP (baseline)
    target_mlp_normed = block.layer_norm_mlp(x_B_T_H_W_D) * (1 + sc_mlp_5) + sh_mlp_5
    target_mlp_out = _target_mlp(
        block.mlp, target_mlp_normed, target_loras, adaln_delta_scale
    )
    x_B_T_H_W_D = x_B_T_H_W_D + ga_mlp_5 * target_mlp_out

    return x_B_T_H_W_D


def _make_patched_block_forward(
    block: nn.Module, block_idx: int, ec_net: EasyControlNetwork
):
    """Build a closure that replaces ``Block.forward`` for one DiT block.

    Mirrors Anima's ``Block.forward`` checkpoint dispatch (unsloth / cpu_offload
    / plain torch_checkpoint / no-ckpt) but routes to the two-stream inner
    instead of the original ``_forward`` when cond is active; falls through to
    the original baseline forward when no cond is set.

    cond_x flows block-by-block via per-block side channels:
    ``block._easycontrol_cond_x_in`` is set by the previous block's patched
    forward (or by ``set_cond`` for block 0); the two-stream inner takes
    ``cond_x_in`` as an explicit arg and returns ``cond_x_out``, so the
    per-block checkpoint preserves autograd across blocks.
    """
    original_forward = block.forward
    b_param = ec_net.b_cond[block_idx]
    cond_lora_qkv = ec_net.cond_lora_qkv[block_idx]
    cond_lora_o = ec_net.cond_lora_o[block_idx]
    cond_lora_ffn1 = ec_net.cond_lora_ffn1[block_idx] if ec_net.apply_ffn_lora else None
    cond_lora_ffn2 = ec_net.cond_lora_ffn2[block_idx] if ec_net.apply_ffn_lora else None
    # Target-stream adaln LoRA — lives only on the cond-active paths below (the
    # no-cond fallback runs original_forward) so it is cond-gated by construction.
    adaln_deltas = (
        (
            ec_net.adaln_lora_self_attn[block_idx],
            ec_net.adaln_lora_cross_attn[block_idx],
            ec_net.adaln_lora_mlp[block_idx],
        )
        if ec_net.train_adaln
        else None
    )
    # Target-stream body LoRA — same cond-gating as the adaln deltas.
    target_loras = (
        tuple(
            getattr(ec_net, f"target_lora_{kind}")[block_idx]
            for kind in _TARGET_LORA_KINDS
        )
        if ec_net.train_target
        else None
    )

    # Last block's cond_x_out is discarded (only its cond K/V are live) — skip
    # that cond-stream evolution.
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

        # Target gets full self/cross/mlp triples; cond skips cross
        (
            (shift_self_attn, scale_self_attn, gate_self_attn),
            (shift_cross_attn, scale_cross_attn, gate_cross_attn),
            (shift_mlp, scale_mlp, gate_mlp),
        ) = _adaln_self_cross_mlp(
            block,
            emb_B_T_D,
            adaln_lora_B_T_3D,
            adaln_deltas=adaln_deltas,
            delta_scale=ec_net.multiplier,
        )
        (
            (cond_shift_self_attn, cond_scale_self_attn, cond_gate_self_attn),
            (cond_shift_mlp, cond_scale_mlp, cond_gate_mlp),
        ) = _adaln_self_mlp(block, cond_emb_B_T_D, cond_adaln_lora_B_T_3D)

        # Reshape target shifts/scales/gates for 5D broadcasting; cond's (B, 1, D)
        # broadcast over (B, S_c, D) naturally.
        sh_self_5 = shift_self_attn[:, :, None, None, :]
        sc_self_5 = scale_self_attn[:, :, None, None, :]
        ga_self_5 = gate_self_attn[:, :, None, None, :]
        sh_cross_5 = shift_cross_attn[:, :, None, None, :]
        sc_cross_5 = scale_cross_attn[:, :, None, None, :]
        ga_cross_5 = gate_cross_attn[:, :, None, None, :]
        sh_mlp_5 = shift_mlp[:, :, None, None, :]
        sc_mlp_5 = scale_mlp[:, :, None, None, :]
        ga_mlp_5 = gate_mlp[:, :, None, None, :]

        # Self-attention (extended target + cond's own)
        target_normed = (
            block.layer_norm_self_attn(x_B_T_H_W_D) * (1 + sc_self_5) + sh_self_5
        )
        target_flat = target_normed.flatten(1, 3)
        target_q, target_k, target_v = _target_self_qkv(
            attn, target_flat, rope_cos_sin, target_loras, ec_net.multiplier
        )

        cond_normed = (
            block.layer_norm_self_attn(cond_x_B_S_D) * (1 + cond_scale_self_attn)
            + cond_shift_self_attn
        )

        # Cond Q/K/V — compute_qkv re-implemented inline so the LoRA delta lands
        # between qkv_proj and the q/k/v norms
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

        target_attn_proj = _target_out_proj(
            attn, target_attn_out, target_loras, ec_net.multiplier
        )
        target_attn_5d = target_attn_proj.unflatten(1, (T_dim, H_dim, W_dim))
        x_B_T_H_W_D = x_B_T_H_W_D + ga_self_5 * target_attn_5d

        # Cond's own self-attn + proj + residual — feeds the next block only, so
        # dead on the last block (its K/V are already consumed above).
        if not is_last:
            # Cast to target compute dtype: fp32 cond LoRA delta + norms promote
            # cond_q/k/v, but flash only accepts fp16/bf16.
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

        # Cross-attention (target only)
        target_cross_normed = (
            block.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + sc_cross_5) + sh_cross_5
        )
        target_cross_out = _target_cross_attn(
            block.cross_attn,
            target_cross_normed.flatten(1, 3),
            attn_params,
            crossattn_emb,
            rope_cos_sin,
            target_loras,
            ec_net.multiplier,
        ).unflatten(1, (T_dim, H_dim, W_dim))
        x_B_T_H_W_D = x_B_T_H_W_D + ga_cross_5 * target_cross_out

        # MLP
        target_mlp_normed = (
            block.layer_norm_mlp(x_B_T_H_W_D) * (1 + sc_mlp_5) + sh_mlp_5
        )
        target_mlp_out = _target_mlp(
            block.mlp, target_mlp_normed, target_loras, ec_net.multiplier
        )
        x_B_T_H_W_D = x_B_T_H_W_D + ga_mlp_5 * target_mlp_out

        # Cond MLP — re-implement layer1/act/layer2 inline to splice FFN LoRA at
        # layer1/layer2 outputs. Discarded on the last block (cond_x_out unused).
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

    # Expose the inner on the block so compile_cond_stream() can swap a compiled
    # version in — compile_blocks() never reaches it (see module docstring
    # GOTCHA). patched_forward reads the attribute per call so the swap takes
    # effect at once.
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
                adaln_deltas=adaln_deltas,
                adaln_delta_scale=ec_net.multiplier,
                target_loras=target_loras,
            )

        cond_state = ec_net._cond_state
        if cond_state is None:
            # No cond — exact baseline DiT behavior
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

        # Dispatch the two-stream inner through the SAME checkpoint path
        # Block.forward uses, with the cond args appended so the checkpoint
        # preserves them as inputs. `inner` is the compiled forward once
        # compile_cond_stream() ran; the checkpoint dispatch itself stays eager.
        inner = block._easycontrol_two_stream_inner

        # GOTCHA: mark the varying seq axes dynamic INSIDE the checkpointed
        # callable. The checkpoint recomputes in BACKWARD via detach_variable,
        # which detaches tensor args (x / cond_x) into fresh tensors that LOSE
        # the mark while the RoPE tuples keep it — that asymmetry is the
        # ConstraintViolationError. Marking inside re-applies on each recompute.
        # Two symbols: target seq (x dim 2) and cond seq (cond_x dim 1); each
        # RoPE table rides dim 0.
        if ec_net._dynamic_seq:
            _compiled_inner = inner
            _lo, _hi = ec_net._dynamic_seq_range
            # The cond stream carries its own band (see _cond_seq_range)
            ec_net._fit_cond_seq_range(cond_x_in.shape[1])
            _clo, _chi = ec_net._dynamic_seq_cond_range or (_lo, _hi)

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
                _clo=_clo,
                _chi=_chi,
            ):
                torch._dynamo.mark_dynamic(x_, 2, min=_lo, max=_hi)
                torch._dynamo.mark_dynamic(cond_x_, 1, min=_clo, max=_chi)
                for _r, _rlo, _rhi in ((rope_, _lo, _hi), (cond_rope_, _clo, _chi)):
                    if _r is not None:
                        torch._dynamo.mark_dynamic(_r[0], 0, min=_rlo, max=_rhi)
                        torch._dynamo.mark_dynamic(_r[1], 0, min=_rlo, max=_rhi)
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

        # Pass cond_x_out to the next block's side channel — carries the
        # autograd link so backward flows here.
        next_idx = block_idx + 1
        if next_idx < ec_net.num_blocks:
            ec_net._block_modules[next_idx]._easycontrol_cond_x_in = cond_x_out
        # else: last block's cond_x_out is unused (cond evolution stops).

        return target_x_out

    return patched_forward


class EasyControlMethodAdapter(MethodAdapter):
    """Bridges EasyControl into AnimaTrainer's adapter dispatch: validates the
    network exposes set_cond/encode_cond_latent, and primes per-step cond
    (with CFG dropout + optional Gaussian perturbation) before the DiT forward."""

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

        # train_llm_adapter guard: two train forwards in, the adapter LoRA must
        # have fired — a cached crossattn_emb skips the llm_adapter entirely
        # and would train the rest while the adapter LoRA sits at zero.
        if is_train and getattr(network, "adapter_lora", None) is not None:
            network._train_primes += 1
            if network._train_primes == 3 and network._adapter_lora_calls == 0:
                raise RuntimeError(
                    "EasyControl train_llm_adapter: the llm_adapter LoRA never "
                    "ran in two train steps — crossattn_emb is arriving "
                    "precomputed. Train with cache_llm_adapter_outputs=false on "
                    "a prompt_embeds TE cache (prep_render.py text "
                    "--text_layout prompt)."
                )

        drop_p = float(getattr(args, "easycontrol_drop_p", 0.1) or 0.0)
        if is_train and drop_p > 0.0 and random.random() < drop_p:
            network.set_cond(None)
            return

        # Prefer a distinct cond latent from the batch (cond≠target tasks like
        # colorization), else fall back to the target latent (ref==target default).
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
