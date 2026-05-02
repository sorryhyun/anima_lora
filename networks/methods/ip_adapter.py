"""IP-Adapter network module for Anima (decoupled cross-attention).

Architecture (adapter-only — DiT frozen):
  reference image
      -> PE-Core vision encoder (live, eval, no_grad)        [B, T_pe, 1024]
      -> small Perceiver resampler                            [B, K, 1024]
      -> per-block (28) to_k_ip / to_v_ip Linear(1024, 2048)  [B, K, n_h, d_h]
      -> SDPA(text_q, ip_k, ip_v)
      -> add scale * ip_attn_out to existing text cross-attention output

Init:
  - to_k_ip / to_v_ip near-zero (std 1e-4) — at step 0, the IP path contributes
    ~zero so training starts from baseline DiT behavior. Standard IP-Adapter
    trick (mirrors postfix's zero-init last layer).
  - Resampler queries init at N(0, 0.15) (img2emb's PerceiverResampler default).

Hooking:
  apply_to() monkey-patches each Block.cross_attn.forward. The patched closure
  captures (orig_attn, block_idx, network), runs the existing text cross-attn
  via attention_dispatch.dispatch_attention(), then computes a decoupled SDPA over IP K/V and
  adds scale * ip_out before the output projection. Lives on the instance, so
  gradient-checkpointing rerolls inside Block._forward see the same patch.

Train-time contract:
  Caller invokes network.set_ip_tokens(ip_tokens) ONCE per batch before the
  DiT forward. set_ip_tokens precomputes per-block (K, V) post-RMSNorm so the
  patched cross-attn forward is a single SDPA call (no redundant projection
  on each step). Pass ``None`` (or call clear_ip_tokens) for unconditional
  passes / CFG dropout.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.log import setup_logging
from library.training.method_adapter import MethodAdapter, SetupCtx, StepCtx
from library.vision import (
    VisionEncoderBundle,
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)
from library.vision.resampler import PerceiverResampler

setup_logging()
logger = logging.getLogger(__name__)


# Anima DiT defaults — see library/anima/models.py:Anima.__init__
DEFAULT_NUM_BLOCKS = 28
DEFAULT_HIDDEN_SIZE = 2048  # query_dim
DEFAULT_NUM_HEADS = 16
DEFAULT_HEAD_DIM = DEFAULT_HIDDEN_SIZE // DEFAULT_NUM_HEADS  # 128
DEFAULT_CONTEXT_DIM = 1024  # crossattn_emb_channels — also matches PE-Core / TIPSv2 d_enc
DEFAULT_NUM_IP_TOKENS = 16
DEFAULT_RESAMPLER_HEADS = 8
DEFAULT_RESAMPLER_LAYERS = 2
DEFAULT_ENCODER_NAME = "pe"
DEFAULT_IP_INIT_STD = 1e-4


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
    del vae, text_encoders, neuron_dropout, network_alpha
    num_ip_tokens = network_dim if network_dim is not None else DEFAULT_NUM_IP_TOKENS

    encoder_name = kwargs.get("encoder", DEFAULT_ENCODER_NAME)
    encoder_dim = int(kwargs.get("encoder_dim", DEFAULT_CONTEXT_DIM))
    resampler_layers = int(kwargs.get("resampler_layers", DEFAULT_RESAMPLER_LAYERS))
    resampler_heads = int(kwargs.get("resampler_heads", DEFAULT_RESAMPLER_HEADS))
    ip_init_std = float(kwargs.get("ip_init_std", DEFAULT_IP_INIT_STD))
    ip_scale = float(kwargs.get("ip_scale", 1.0))

    num_blocks = getattr(unet, "num_blocks", DEFAULT_NUM_BLOCKS) if unet is not None else DEFAULT_NUM_BLOCKS
    hidden_size = getattr(unet, "model_channels", DEFAULT_HIDDEN_SIZE) if unet is not None else DEFAULT_HIDDEN_SIZE
    num_heads = getattr(unet, "num_heads", DEFAULT_NUM_HEADS) if unet is not None else DEFAULT_NUM_HEADS
    context_dim = DEFAULT_CONTEXT_DIM

    return IPAdapterNetwork(
        num_ip_tokens=num_ip_tokens,
        encoder_name=encoder_name,
        encoder_dim=encoder_dim,
        context_dim=context_dim,
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        num_heads=num_heads,
        resampler_layers=resampler_layers,
        resampler_heads=resampler_heads,
        ip_init_std=ip_init_std,
        ip_scale=ip_scale,
        multiplier=multiplier,
    )


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

    encoder_name = kwargs.get("encoder") or metadata.get("ss_encoder", DEFAULT_ENCODER_NAME)
    encoder_dim = int(metadata.get("ss_encoder_dim", DEFAULT_CONTEXT_DIM))
    context_dim = int(metadata.get("ss_context_dim", DEFAULT_CONTEXT_DIM))
    num_ip_tokens = int(metadata.get("ss_num_ip_tokens", DEFAULT_NUM_IP_TOKENS))
    num_blocks = int(metadata.get("ss_num_blocks", DEFAULT_NUM_BLOCKS))
    hidden_size = int(metadata.get("ss_hidden_size", DEFAULT_HIDDEN_SIZE))
    num_heads = int(metadata.get("ss_num_heads", DEFAULT_NUM_HEADS))
    resampler_layers = int(metadata.get("ss_resampler_layers", DEFAULT_RESAMPLER_LAYERS))
    resampler_heads = int(metadata.get("ss_resampler_heads", DEFAULT_RESAMPLER_HEADS))
    ip_scale = float(kwargs.get("ip_scale") or metadata.get("ss_ip_scale", 1.0))

    network = IPAdapterNetwork(
        num_ip_tokens=num_ip_tokens,
        encoder_name=encoder_name,
        encoder_dim=encoder_dim,
        context_dim=context_dim,
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        num_heads=num_heads,
        resampler_layers=resampler_layers,
        resampler_heads=resampler_heads,
        ip_init_std=DEFAULT_IP_INIT_STD,
        ip_scale=ip_scale,
        multiplier=multiplier,
    )
    return network, weights_sd


class IPAdapterNetwork(nn.Module):
    def __init__(
        self,
        *,
        num_ip_tokens: int,
        encoder_name: str,
        encoder_dim: int,
        context_dim: int,
        num_blocks: int,
        hidden_size: int,
        num_heads: int,
        resampler_layers: int,
        resampler_heads: int,
        ip_init_std: float,
        ip_scale: float,
        multiplier: float = 1.0,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size {hidden_size} not divisible by num_heads {num_heads}"
            )
        self.num_ip_tokens = num_ip_tokens
        self.encoder_name = encoder_name
        self.encoder_dim = encoder_dim
        self.context_dim = context_dim
        self.num_blocks = num_blocks
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.resampler_layers = resampler_layers
        self.resampler_heads = resampler_heads
        self.ip_init_std = ip_init_std
        self.ip_scale = ip_scale
        self.multiplier = multiplier

        # Resampler: PE patch tokens -> K compact image tokens in context-dim space.
        # d_out=context_dim so to_k_ip/to_v_ip can read directly without another proj.
        self.resampler = PerceiverResampler(
            d_enc=encoder_dim,
            d_model=context_dim,
            n_heads=resampler_heads,
            n_slots=num_ip_tokens,
            n_layers=resampler_layers,
            d_out=context_dim,
        )

        # Per-block IP projections. Bias=False matches the existing kv_proj.
        self.to_k_ip = nn.ModuleList(
            [nn.Linear(context_dim, hidden_size, bias=False) for _ in range(num_blocks)]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(context_dim, hidden_size, bias=False) for _ in range(num_blocks)]
        )
        for proj in list(self.to_k_ip) + list(self.to_v_ip):
            nn.init.normal_(proj.weight, std=ip_init_std)

        # Step 0 = baseline DiT is enforced purely by ip_init_std on K/V; with
        # ip_init_std=1e-4 the SDPA output is ~1e-4 in magnitude, well below
        # the text path. This matches reference IP-Adapter (Tencent), which
        # also has no learned per-block gate.

        # Populated by apply_to() — references to the DiT cross-attn Attention
        # modules. Held as a plain list (NOT nn.ModuleList) so PyTorch doesn't
        # treat the DiT as a child of this network.
        self._cross_attn_modules: list[nn.Module] = []
        self._original_forwards: list = []
        self._patched: bool = False

        # Diagnostic accumulators: per-block running sum of ‖ip_out‖/‖text_result‖
        # plus call count. Stored as 0-d tensors ON THE cross_attn MODULES (not
        # here) so the patched forward can read them via ``orig_attn`` and
        # never has to read ``block_idx`` as a Python int — that would force
        # torch.compile to specialize a separate graph per block (28 of them)
        # and blow past recompile_limit. Allocated lazily on first
        # ``set_diagnostics_enabled(True)`` call. Read out via
        # ``diagnostic_summary()``.
        self._diag_enabled: bool = False

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"IPAdapterNetwork: K={num_ip_tokens}, encoder={encoder_name}, "
            f"context_dim={context_dim}, num_blocks={num_blocks}, "
            f"hidden={hidden_size}/{num_heads}h, ip_init_std={ip_init_std}, "
            f"ip_scale={ip_scale}, params={total / 1e6:.1f}M"
        )

    # ------------------------------------------------------------ apply / hook

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        del text_encoders, apply_text_encoder
        if not apply_unet:
            return
        if self._patched:
            logger.warning("IPAdapterNetwork.apply_to called twice — skipping")
            return
        if unet is None or not hasattr(unet, "blocks"):
            raise ValueError("apply_to requires the Anima DiT (unet) with .blocks")
        if len(unet.blocks) != self.num_blocks:
            raise ValueError(
                f"DiT has {len(unet.blocks)} blocks, IP-Adapter expects {self.num_blocks}. "
                "Re-create the network with matching num_blocks."
            )

        from networks import attention_dispatch as anima_attention  # local: avoid import cycle at file load

        for idx, block in enumerate(unet.blocks):
            cross_attn = block.cross_attn
            if cross_attn.is_selfattn:
                raise RuntimeError(f"block[{idx}].cross_attn unexpectedly self-attention")
            if cross_attn.context_dim != self.context_dim:
                raise ValueError(
                    f"block[{idx}].cross_attn context_dim {cross_attn.context_dim} "
                    f"!= IP context_dim {self.context_dim}"
                )
            if cross_attn.n_heads != self.num_heads or cross_attn.head_dim != self.head_dim:
                raise ValueError(
                    f"block[{idx}].cross_attn heads/head_dim mismatch: "
                    f"({cross_attn.n_heads}, {cross_attn.head_dim}) vs "
                    f"({self.num_heads}, {self.head_dim})"
                )
            self._cross_attn_modules.append(cross_attn)
            self._original_forwards.append(cross_attn.forward)
            cross_attn._ip_k_cached = None
            cross_attn._ip_v_cached = None
            cross_attn._ip_diag_ratio_sum = None  # 0-d float32, set by set_diagnostics_enabled
            cross_attn._ip_diag_count = None  # 0-d int64
            cross_attn.forward = _make_patched_forward(cross_attn, self, idx, anima_attention)

        self._patched = True
        logger.info(
            f"IP-Adapter: patched cross-attn forward on {len(self._cross_attn_modules)} blocks"
        )

    def remove_from(self):
        for cross_attn, orig in zip(self._cross_attn_modules, self._original_forwards):
            cross_attn.forward = orig
            cross_attn._ip_k_cached = None
            cross_attn._ip_v_cached = None
            cross_attn._ip_diag_ratio_sum = None
            cross_attn._ip_diag_count = None
        self._cross_attn_modules.clear()
        self._original_forwards.clear()
        self._patched = False

    # ------------------------------------------------------------ runtime API

    def encode_ip_tokens(self, image_features: torch.Tensor) -> torch.Tensor:
        """Run the resampler on raw vision-encoder features.

        Args:
            image_features: [B, T_pe, encoder_dim] from PE-Core.
        Returns:
            [B, K, context_dim]
        """
        return self.resampler(image_features)

    def set_ip_tokens(self, ip_tokens: Optional[torch.Tensor]) -> None:
        """Pre-compute per-block (K, V) post-norm and cache on each cross_attn.

        Pass ``None`` (or call ``clear_ip_tokens``) for unconditional / CFG
        dropout passes.
        """
        if not self._patched:
            raise RuntimeError("set_ip_tokens called before apply_to")
        if ip_tokens is None:
            self.clear_ip_tokens()
            return
        if ip_tokens.dim() != 3 or ip_tokens.shape[-1] != self.context_dim:
            raise ValueError(
                f"ip_tokens must be [B, K, {self.context_dim}], got {tuple(ip_tokens.shape)}"
            )

        B, K, _ = ip_tokens.shape
        for idx, cross_attn in enumerate(self._cross_attn_modules):
            proj_dtype = self.to_k_ip[idx].weight.dtype
            ip_in = ip_tokens.to(proj_dtype)
            k = self.to_k_ip[idx](ip_in)  # [B, K, hidden]
            v = self.to_v_ip[idx](ip_in)
            k = k.unflatten(-1, (self.num_heads, self.head_dim))  # [B, K, n_h, d_h]
            v = v.unflatten(-1, (self.num_heads, self.head_dim))
            # Match the text-side normalization so the dot-product scale is
            # consistent with the existing (post-RMSNorm) text Q.
            k = cross_attn.k_norm(k)
            v = cross_attn.v_norm(v)
            cross_attn._ip_k_cached = k
            cross_attn._ip_v_cached = v

    def clear_ip_tokens(self) -> None:
        for cross_attn in self._cross_attn_modules:
            cross_attn._ip_k_cached = None
            cross_attn._ip_v_cached = None

    def get_effective_scale(self) -> float:
        return self.ip_scale * self.multiplier

    # ------------------------------------------------------------ diagnostics

    def set_diagnostics_enabled(self, enabled: bool, device: Optional[torch.device] = None) -> None:
        """Toggle per-step accumulation of ‖scale·ip_out‖/‖text_result‖.

        Buffers are lazy-allocated on first enable, as 0-d tensors on each
        cross_attn module (so the patched forward never indexes by block_idx
        and torch.compile doesn't specialize per-block). They are NOT
        registered as nn.Buffers — this is pure runtime state, not part of
        ``state_dict``.
        """
        if not self._patched:
            raise RuntimeError("set_diagnostics_enabled called before apply_to")
        if enabled:
            if device is None:
                device = next(self.to_k_ip[0].parameters()).device
            for cross_attn in self._cross_attn_modules:
                if cross_attn._ip_diag_ratio_sum is None:
                    cross_attn._ip_diag_ratio_sum = torch.zeros((), dtype=torch.float32, device=device)
                    cross_attn._ip_diag_count = torch.zeros((), dtype=torch.int64, device=device)
        self._diag_enabled = bool(enabled)

    def reset_diagnostics(self) -> None:
        for cross_attn in self._cross_attn_modules:
            if cross_attn._ip_diag_ratio_sum is not None:
                cross_attn._ip_diag_ratio_sum.zero_()
                cross_attn._ip_diag_count.zero_()

    def diagnostic_summary(self, *, reset: bool = True, log: bool = True) -> dict:
        """Return per-block ‖to_k_ip‖, ‖to_v_ip‖, mean ip/text ratio.

        Heavy logging is gated on ``log=True``: prints aggregate stats
        (min/mean/max across blocks) plus a few representative per-block lines
        (first / middle / last block). The full per-block tensors are returned
        in the dict for further inspection.
        """
        with torch.no_grad():
            k_norms = torch.tensor(
                [self.to_k_ip[i].weight.detach().float().norm().item() for i in range(self.num_blocks)]
            )
            v_norms = torch.tensor(
                [self.to_v_ip[i].weight.detach().float().norm().item() for i in range(self.num_blocks)]
            )
            sums_list = []
            counts_list = []
            for cross_attn in self._cross_attn_modules:
                if cross_attn._ip_diag_ratio_sum is not None:
                    sums_list.append(cross_attn._ip_diag_ratio_sum.detach().float().cpu())
                    counts_list.append(cross_attn._ip_diag_count.detach().float().cpu())
                else:
                    sums_list.append(torch.zeros(()))
                    counts_list.append(torch.zeros(()))
            if sums_list:
                sums = torch.stack(sums_list)
                counts = torch.stack(counts_list)
                ratios = torch.where(counts > 0, sums / counts.clamp_min(1), torch.zeros_like(sums))
                total_calls = int(counts.sum().item())
            else:
                ratios = torch.zeros(self.num_blocks)
                total_calls = 0

        if log:
            def _stats(t: torch.Tensor) -> str:
                return f"min={t.min().item():.3e} mean={t.mean().item():.3e} max={t.max().item():.3e}"

            logger.info(
                f"[IP-Adapter diag] params: ‖to_k_ip‖ {_stats(k_norms)} | ‖to_v_ip‖ {_stats(v_norms)}"
            )
            logger.info(
                f"[IP-Adapter diag] runtime: ‖scale·ip_out‖/‖text_result‖ {_stats(ratios)} "
                f"(N={total_calls} calls across {self.num_blocks} blocks)"
            )
            sample_idx = [0, self.num_blocks // 2, self.num_blocks - 1]
            for i in sample_idx:
                logger.info(
                    f"[IP-Adapter diag]   block[{i}]: ‖k‖={k_norms[i].item():.3e} "
                    f"‖v‖={v_norms[i].item():.3e} ratio={ratios[i].item():.3e}"
                )

        if reset:
            self.reset_diagnostics()

        return {
            "to_k_ip_norm": k_norms,
            "to_v_ip_norm": v_norms,
            "ip_text_ratio": ratios,
            "num_calls": total_calls,
        }

    # ------------------------------------------------------------ trainer hooks

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def is_mergeable(self):
        return False

    def enable_gradient_checkpointing(self):
        # Resampler is shallow; no-op. (Block-level grad checkpointing on the
        # DiT is handled by Anima's Block.enable_gradient_checkpointing.)
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return list(self.parameters())

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        params = [
            {"params": list(self.resampler.parameters()), "lr": lr},
            {"params": list(self.to_k_ip.parameters()) + list(self.to_v_ip.parameters()), "lr": lr},
        ]
        descriptions = ["ip_resampler", "ip_kv_proj"]
        return params, descriptions

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr=None):
        params, _ = self.prepare_optimizer_params_with_multiple_te_lrs(
            text_encoder_lr, unet_lr, default_lr
        )
        return params

    # ------------------------------------------------------------ I/O

    def save_weights(self, file, dtype, metadata):
        dtype = dtype or torch.bfloat16
        sd = {k: v.detach().cpu().to(dtype) for k, v in self.state_dict().items()}

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library.training.hashing import precalculate_safetensors_hashes

            if metadata is None:
                metadata = {}
            metadata["ss_network_module"] = "networks.methods.ip_adapter"
            metadata["ss_network_spec"] = "ip_adapter"
            metadata["ss_num_ip_tokens"] = str(self.num_ip_tokens)
            metadata["ss_encoder"] = self.encoder_name
            metadata["ss_encoder_dim"] = str(self.encoder_dim)
            metadata["ss_context_dim"] = str(self.context_dim)
            metadata["ss_num_blocks"] = str(self.num_blocks)
            metadata["ss_hidden_size"] = str(self.hidden_size)
            metadata["ss_num_heads"] = str(self.num_heads)
            metadata["ss_resampler_layers"] = str(self.resampler_layers)
            metadata["ss_resampler_heads"] = str(self.resampler_heads)
            metadata["ss_ip_scale"] = str(self.ip_scale)

            model_hash, legacy_hash = precalculate_safetensors_hashes(sd, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(sd, file, metadata)
        else:
            torch.save(sd, file)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            sd = load_file(file)
        else:
            sd = torch.load(file, map_location="cpu")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logger.warning(
                f"IPAdapterNetwork.load_state_dict: missing={missing}, unexpected={unexpected}"
            )
        else:
            logger.info(f"Loaded IP-Adapter weights from {file} ({len(sd)} tensors)")


# ----------------------------------------------------------------- patched forward


def _make_patched_forward(orig_attn, ip_net: "IPAdapterNetwork", block_idx: int, anima_attention):
    """Build a closure that replaces ``Attention.forward`` for one block's cross-attn.

    Mirrors the original ``Attention.forward`` logic (library/anima/models.py:446)
    but additionally computes a decoupled SDPA over the cached IP K/V and adds
    ``scale * ip_attn_out`` to the text attention output before ``output_proj``.
    """

    def patched_forward(x, attn_params, context, rope_cos_sin=None):
        q, k, v = orig_attn.compute_qkv(x, context, rope_cos_sin)
        if q.dtype != v.dtype:
            if (
                not attn_params.supports_fp32 or attn_params.requires_same_dtype
            ) and torch.is_autocast_enabled():
                target_dtype = v.dtype
                q = q.to(target_dtype)
                k = k.to(target_dtype)
        text_qkv = [q, k, v]
        text_result = anima_attention.dispatch_attention(text_qkv, attn_params=attn_params)

        ip_k = getattr(orig_attn, "_ip_k_cached", None)
        ip_v = getattr(orig_attn, "_ip_v_cached", None)
        if ip_k is not None and ip_v is not None:
            # q: [B, S, n_h, d_h] (bshd from compute_qkv).
            # ip_k, ip_v: [B_ip, K, n_h, d_h] (cached post-norm by set_ip_tokens).
            # Broadcast B_ip=1 -> B (free view) for inference where the same
            # reference image conditions the whole CFG batch.
            B = q.shape[0]
            if ip_k.shape[0] == 1 and B > 1:
                ip_k = ip_k.expand(B, *ip_k.shape[1:])
                ip_v = ip_v.expand(B, *ip_v.shape[1:])
            elif ip_k.shape[0] != B:
                raise RuntimeError(
                    f"IP-Adapter K/V batch {ip_k.shape[0]} does not match q batch {B}"
                )
            q_sdpa = q.transpose(1, 2)  # [B, n_h, S, d_h]
            k_sdpa = ip_k.to(q_sdpa.dtype).transpose(1, 2)
            v_sdpa = ip_v.to(q_sdpa.dtype).transpose(1, 2)
            ip_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
            # Back to [B, S, n_h*d_h] to match text_result's flattened layout.
            ip_out = ip_out.transpose(1, 2).reshape(text_result.shape)
            scale = ip_net.get_effective_scale()
            diag_sum = orig_attn._ip_diag_ratio_sum
            if ip_net._diag_enabled and diag_sum is not None:
                # Detached, on-device scalar update on tensors that live on
                # ``orig_attn`` itself — read via the captured closure object,
                # never via ``block_idx`` as a Python int (which would specialize
                # one compiled graph per block and break recompile_limit).
                with torch.no_grad():
                    ip_norm = (scale * ip_out).float().norm()
                    txt_norm = text_result.float().norm().clamp_min(1e-12)
                    diag_sum.add_(ip_norm / txt_norm)
                    orig_attn._ip_diag_count.add_(1)
            text_result = text_result + scale * ip_out

        return orig_attn.output_dropout(orig_attn.output_proj(text_result))

    return patched_forward


# ----------------------------------------------------------------- trainer integration


class IPAdapterMethodAdapter(MethodAdapter):
    """Bridges IP-Adapter into AnimaTrainer's adapter dispatch.

    Owns the live PE-Core vision encoder (when not running off pre-cached
    features) so the trainer no longer needs an ``_vision_bundle`` field.

    Setup: validate the network exposes set_ip_tokens / encode_ip_tokens,
    enforce the cache_latents/ip_features_cache_to_disk constraint, load the
    live PE encoder when needed, enable runtime diagnostics.
    Step: encode the reference image (cached features or live PE) and prime
    per-block ``ip_k`` / ``ip_v`` on the network with whole-batch CFG dropout.
    Epoch end: dump ‖to_k_ip‖ / ‖to_v_ip‖ / ‖ip_out‖ / ‖text_result‖ ratios."""

    name = "ip_adapter"

    def __init__(self) -> None:
        self._vision_bundle: Optional[VisionEncoderBundle] = None
        self._epochs_completed: int = 0

    def on_network_built(self, ctx: SetupCtx) -> None:
        args = ctx.args
        accelerator = ctx.accelerator
        net = ctx.network
        if not (hasattr(net, "set_ip_tokens") and hasattr(net, "encode_ip_tokens")):
            raise ValueError(
                "--use_ip_adapter requires a network module with set_ip_tokens / "
                "encode_ip_tokens (e.g. networks.methods.ip_adapter)."
            )
        cache_features = getattr(args, "ip_features_cache_to_disk", False)
        if not cache_features and getattr(args, "cache_latents", False):
            raise ValueError(
                "--use_ip_adapter without --ip_features_cache_to_disk requires "
                "--cache_latents=false so batch['images'] carries the raw reference "
                "image for live PE encoding. Either set ip_features_cache_to_disk=true "
                "(after `make preprocess-pe`) or cache_latents=false."
            )
        if cache_features:
            accelerator.print(
                f"IP-Adapter: reading cached vision features "
                f"(encoder={getattr(args, 'ip_encoder', 'pe')}, "
                f"image_drop_p={getattr(args, 'ip_image_drop_p', 0.1)}) — "
                "vision encoder NOT loaded."
            )
        else:
            self._vision_bundle = load_pe_encoder(
                accelerator.device,
                name=getattr(args, "ip_encoder", "pe"),
                dtype=torch.bfloat16,
            )
            accelerator.print(
                f"IP-Adapter: loaded vision encoder {self._vision_bundle.name} "
                f"(d_enc={self._vision_bundle.d_enc}, "
                f"image_drop_p={getattr(args, 'ip_image_drop_p', 0.1)})"
            )
        diag_epochs = int(getattr(args, "ip_diagnostics_epochs", 1) or 0)
        if hasattr(net, "set_diagnostics_enabled") and diag_epochs > 0:
            net.set_diagnostics_enabled(True, device=accelerator.device)
            if accelerator.is_main_process:
                net.diagnostic_summary(reset=True, log=True)

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        args = ctx.args
        accelerator = ctx.accelerator
        network = ctx.network
        if not hasattr(network, "set_ip_tokens"):
            return

        drop_p = float(getattr(args, "ip_image_drop_p", 0.1) or 0.0)
        if is_train and drop_p > 0.0 and random.random() < drop_p:
            network.set_ip_tokens(None)
            return

        cached = batch.get("ip_features") if isinstance(batch, dict) else None
        if cached is not None:
            ip_features = cached.to(accelerator.device, dtype=ctx.weight_dtype)
        else:
            if self._vision_bundle is None:
                raise RuntimeError(
                    "IP-Adapter has no feature source: --ip_features_cache_to_disk "
                    "is off and the live vision encoder isn't loaded."
                )
            images = batch.get("images") if isinstance(batch, dict) else None
            if images is None:
                raise RuntimeError(
                    "IP-Adapter expected batch['images'] but got None — re-check "
                    "cache_latents=false in the IP-Adapter config, or set "
                    "ip_features_cache_to_disk=true with `make preprocess-pe`."
                )
            with torch.no_grad():
                feats_list = encode_pe_from_imageminus1to1(
                    self._vision_bundle,
                    images.to(accelerator.device),
                    same_bucket=True,  # dataloader bucketing guarantees per-batch shape
                )
                ip_features = torch.stack(feats_list, dim=0).to(
                    ctx.weight_dtype
                )  # [B, T_pe, d_enc]
        # Resampler runs in network-param dtype (bf16 typically); gradient
        # flows from here.
        ip_tokens = network.encode_ip_tokens(ip_features)
        network.set_ip_tokens(ip_tokens)

    def on_epoch_end(self, ctx: StepCtx) -> None:
        # Dump per-block param norms + ‖ip_out‖/‖text_result‖ ratio averaged
        # over the just-finished epoch, then reset. Main process only — the
        # caller already gates on is_main_process.
        net = ctx.accelerator.unwrap_model(ctx.network)
        if hasattr(net, "diagnostic_summary"):
            net.diagnostic_summary(reset=True, log=True)
        # Diagnostics add 56 fp32 norm reductions per step (2 per block × 28
        # blocks). Keep them on only for the warm-up window — once we've
        # confirmed the IP path is contributing, the per-block norms become
        # pure overhead on the forward critical path.
        self._epochs_completed += 1
        diag_epochs = int(getattr(ctx.args, "ip_diagnostics_epochs", 1) or 0)
        if (
            self._epochs_completed >= diag_epochs
            and hasattr(net, "set_diagnostics_enabled")
            and getattr(net, "_diag_enabled", False)
        ):
            net.set_diagnostics_enabled(False)
            if ctx.accelerator.is_main_process:
                logger.info(
                    f"[IP-Adapter diag] disabled after {self._epochs_completed} epoch(s) "
                    f"(--ip_diagnostics_epochs={diag_epochs}); per-block norm overhead removed"
                )
