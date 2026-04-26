"""EasyControl network module for Anima — Phase 1.

Architecture (adapter-only — DiT frozen):
  reference image (clean VAE latent, 4D [B, C, H, W])
      -> DiT.x_embedder (frozen, reused)                 [B, T, H', W', D]
      -> flatten to S_c = H'*W' tokens                   [B, S_c, D]
      -> for each Block:
           cond_x -> layer_norm_self_attn(cond_x)
           cond_x -> self_attn projections + cond_lora_qkv  (cache K_c, V_c on the block)
           cond_x = cond_x + output_proj(SDPA(Q_c, K_c, V_c)) + cond_lora_o(SDPA_out)
           # skip cross_attn entirely
           cond_x -> layer_norm_mlp -> mlp + cond_lora_ffn (residual)
      -> target's self_attn.forward (patched) reads cached (K_c, V_c) and runs
         SDPA over EXTENDED keys [K_t; K_c]/[V_t; V_c] with a learnable additive
         per-block scalar logit bias `b_cond` on the cond positions (init -10).

Step-0 baseline equivalence:
  `b_cond[i]` initialized to -10. exp(-10) ≈ 4.5e-5, so cond positions hold
  negligible softmax mass at init — α ≈ 1.0000, output ≈ baseline DiT.
  Verified by bench/active/easycontrol/step0_equivalence.py.

Train-time contract:
  Caller invokes network.set_cond_tokens(clean_vae_latent) ONCE per batch before
  the DiT forward. set_cond_tokens runs the full cond path and primes per-block
  (K_c, V_c) on each block.self_attn. Pass ``None`` (or call clear_cond_tokens)
  for unconditional / CFG-dropout passes.

Phase 1 limitations:
  - No KV cache at inference: cond path is recomputed every step. Phase 2 lifts.
  - Cond pass uses plain LayerNorm (no AdaLN modulation). The full DiT-with-AdaLN
    cond pass is deferred — adds complexity for marginal Phase 1 benefit since
    `b_cond=-10` zeroes the cond's contribution at init regardless.
  - No RoPE on cond Q/K — different positions from target. Step 0 is unaffected
    by this choice (gated to ~0 by b_cond); learning may want to revisit later.
  - blocks_to_swap=0 expected. Block-swap interleave with the cond pre-pass is
    a known integration risk: the cond pre-pass walks every block in sequence
    before the DiT main forward, so blocks parked on CPU by the offloader are
    not on-device when the cond pre-pass tries to use them. Use
    --gradient_checkpointing instead — see enable_gradient_checkpointing.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library.log import setup_logging

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


class _LoRAProj(nn.Module):
    """Plain LoRA-style D->r->out_dim projection with up zero-init.

    Standalone (not a wrapper around an org_module) — used by EasyControl to
    add a delta to a frozen DiT projection only on the cond pass. Output added
    by the caller; this module just produces the delta.
    """

    def __init__(self, in_dim: int, out_dim: int, r: int, alpha: float):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 bottleneck for bf16 numerical stability (matches LoRAModule policy).
        h = F.linear(x.float(), self.lora_down.weight.float())
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

    return EasyControlNetwork(
        num_blocks=num_blocks,
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        cond_lora_dim=int(cond_lora_dim),
        cond_lora_alpha=float(cond_lora_alpha),
        b_cond_init=b_cond_init,
        cond_scale=cond_scale,
        apply_ffn_lora=apply_ffn_lora,
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

    num_blocks = int(metadata.get("ss_num_blocks", DEFAULT_NUM_BLOCKS))
    hidden_size = int(metadata.get("ss_hidden_size", DEFAULT_HIDDEN_SIZE))
    num_heads = int(metadata.get("ss_num_heads", DEFAULT_NUM_HEADS))
    mlp_ratio = float(metadata.get("ss_mlp_ratio", DEFAULT_MLP_RATIO))
    cond_lora_dim = int(metadata.get("ss_cond_lora_dim", DEFAULT_LORA_DIM))
    cond_lora_alpha = float(metadata.get("ss_cond_lora_alpha", float(cond_lora_dim)))
    b_cond_init = float(metadata.get("ss_b_cond_init", DEFAULT_B_COND_INIT))
    cond_scale = float(kwargs.get("cond_scale") or metadata.get("ss_cond_scale", 1.0))
    apply_ffn_lora = bool(int(metadata.get("ss_apply_ffn_lora", 1)))

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
        multiplier=multiplier,
    )
    return network, weights_sd


class EasyControlNetwork(nn.Module):
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
        multiplier: float = 1.0,
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
        self.multiplier = multiplier

        D = hidden_size
        r = cond_lora_dim
        a = cond_lora_alpha

        # Per-block cond LoRA on self_attn:
        # qkv: fused D -> 3D delta (matches frozen Attention.qkv_proj layout).
        # o:   D -> D delta on the output projection.
        self.cond_lora_qkv = nn.ModuleList(
            [_LoRAProj(D, 3 * D, r, a) for _ in range(num_blocks)]
        )
        self.cond_lora_o = nn.ModuleList(
            [_LoRAProj(D, D, r, a) for _ in range(num_blocks)]
        )

        # Per-block cond LoRA on FFN (GPT2FeedForward layer1: D -> 4D, layer2: 4D -> D).
        if apply_ffn_lora:
            self.cond_lora_ffn1 = nn.ModuleList(
                [_LoRAProj(D, self.ffn_dim, r, a) for _ in range(num_blocks)]
            )
            self.cond_lora_ffn2 = nn.ModuleList(
                [_LoRAProj(self.ffn_dim, D, r, a) for _ in range(num_blocks)]
            )
        else:
            self.cond_lora_ffn1 = None
            self.cond_lora_ffn2 = None

        # Per-block scalar additive logit bias on cond keys. Init -10 → cond
        # softmax mass ≈ 4.5e-5 at step 0 → α ≈ 1 → out ≈ baseline DiT.
        # Verified by bench/active/easycontrol/step0_equivalence.py.
        #
        # Stored as a ParameterList of 0-d Parameters (not a single
        # [num_blocks] Parameter) so each block's patched self-attn closure
        # can capture its bias as a *Parameter object*, not a Python int
        # index. dynamo specializes on int closure cells (it treats them like
        # integer attributes of an nn.Module), so capturing block_idx blew
        # past recompile_limit at block 8/28. Capturing a Parameter is fine
        # — dynamo lifts it as a graph input.
        self.b_cond = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(b_cond_init, dtype=torch.float32))
                for _ in range(num_blocks)
            ]
        )

        # Populated by apply_to() — references to the DiT.blocks. Plain lists
        # (NOT nn.ModuleList) so PyTorch doesn't re-parent the DiT into this
        # network's parameter tree.
        self._dit: Optional[nn.Module] = None
        self._block_modules: list[nn.Module] = []
        self._self_attn_modules: list[nn.Module] = []
        self._original_self_attn_forwards: list = []
        self._patched: bool = False

        # Toggled by enable_gradient_checkpointing(); applied per-block in the
        # cond pre-pass so activations for all 28 blocks aren't held at once.
        self._gradient_checkpointing: bool = False

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"EasyControlNetwork: blocks={num_blocks}, hidden={hidden_size}/{num_heads}h, "
            f"r={cond_lora_dim} alpha={cond_lora_alpha}, ffn_lora={apply_ffn_lora}, "
            f"b_cond_init={b_cond_init}, cond_scale={cond_scale}, "
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

        from networks import attention as anima_attention  # local: avoid import cycle

        self._dit = unet
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
            self._self_attn_modules.append(attn)
            self._original_self_attn_forwards.append(attn.forward)
            attn._cond_k_cached = None
            attn._cond_v_cached = None
            attn.forward = _make_patched_self_attn_forward(
                attn, self.b_cond[idx], anima_attention
            )

        self._patched = True
        logger.info(
            f"EasyControl: patched self-attn forward on {len(self._self_attn_modules)} blocks"
        )

    def remove_from(self):
        for attn, orig in zip(
            self._self_attn_modules, self._original_self_attn_forwards
        ):
            attn.forward = orig
            attn._cond_k_cached = None
            attn._cond_v_cached = None
        self._block_modules.clear()
        self._self_attn_modules.clear()
        self._original_self_attn_forwards.clear()
        self._dit = None
        self._patched = False

    # ------------------------------------------------------------ runtime API

    def encode_cond_latent(
        self,
        cond_latent: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Patch-embed the clean VAE latent into [B, S_c, D] cond tokens.

        Reuses the DiT's (frozen) ``x_embedder`` so we don't introduce a new
        learnable embedder. The padding-mask channel is concatenated when the
        DiT was built with ``concat_padding_mask=True`` (Anima default).

        Args:
            cond_latent: [B, C, H, W] (image) or [B, C, T, H, W] (video). For
                Phase 1 we expect images; T is unsqueezed to 1 if missing.
            padding_mask: optional [B, 1, H, W] mask. If None and the DiT
                requires it, we synthesize a default all-ones mask.
        Returns:
            [B, S_c, D] cond tokens.
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
        if self._dit.concat_padding_mask and padding_mask is None:
            padding_mask = torch.ones(
                B, 1, H, W, device=cond_latent.device, dtype=cond_latent.dtype
            )

        # prepare_embedded_sequence handles padding-mask concat + patch embed.
        # It also returns rope_cos_sin, which we discard (cond uses no RoPE).
        cond_x_5d, _ = self._dit.prepare_embedded_sequence(
            cond_latent,
            fps=None,
            padding_mask=padding_mask,
        )
        # Flatten to [B, S_c, D]
        cond_x = cond_x_5d.flatten(1, 3)

        # Static-shape padding: target tokens are padded to static_token_count
        # for torch.compile stability (see Anima.forward_mini_train_dit). Pad
        # cond the same way so the extended K length is fixed at 2*static and
        # compile holds across buckets.
        target_static = getattr(self._dit, "static_token_count", None)
        if target_static is not None and cond_x.shape[1] < target_static:
            cond_x = F.pad(cond_x, (0, 0, 0, target_static - cond_x.shape[1]))

        return cond_x

    def set_cond_tokens(
        self,
        cond_latent: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Run the cond pre-pass and prime per-block (K_c, V_c) on each block.

        Pass ``None`` (or call ``clear_cond_tokens``) for unconditional /
        CFG-dropout passes.
        """
        if not self._patched:
            raise RuntimeError("set_cond_tokens called before apply_to")
        if cond_latent is None:
            self.clear_cond_tokens()
            return

        cond_x = self.encode_cond_latent(cond_latent, padding_mask=padding_mask)
        self._run_cond_path(cond_x)

    def clear_cond_tokens(self) -> None:
        for attn in self._self_attn_modules:
            attn._cond_k_cached = None
            attn._cond_v_cached = None

    def get_effective_scale(self) -> float:
        return self.cond_scale * self.multiplier

    # ------------------------------------------------------------ cond pre-pass

    def _run_cond_path(self, cond_x: torch.Tensor) -> None:
        """Run cond tokens through every block, caching (K_c, V_c) per block.

        Phase 1 simplifications:
          - Plain LayerNorm (no AdaLN scale/shift/gate). The cond stream has
            no per-token timestep embedding to drive AdaLN; using plain norm
            is the simplest well-defined alternative. Step-0 baseline
            equivalence is preserved by ``b_cond=-10`` regardless.
          - No RoPE on cond Q/K (different positions from target).
          - Plain torch SDPA for the cond's own self-attention (no anima
            attention dispatch).

        When gradient checkpointing is on (and we're in training mode), each
        per-block cond forward is wrapped in ``torch.utils.checkpoint``. K_c
        and V_c are returned from the checkpointed function (so the target's
        loss can backprop through them into the cond LoRAs); everything else
        is recomputed on backward.
        """
        if self._dit is None:
            raise RuntimeError("_run_cond_path called before apply_to")
        scale = self.get_effective_scale()

        use_ckpt = self._gradient_checkpointing and self.training

        for idx, attn in enumerate(self._self_attn_modules):
            if use_ckpt:
                # Bind idx and scale via default args so the saved closure
                # doesn't see the latest loop value during recompute.
                def _fn(cx, _i=idx, _s=scale):
                    return self._cond_block_forward(cx, _i, _s)

                cond_x, k, v = torch_checkpoint(_fn, cond_x, use_reentrant=False)
            else:
                cond_x, k, v = self._cond_block_forward(cond_x, idx, scale)

            # Cache K, V for the target's extended self-attn. Assigned outside
            # the checkpoint so they remain reachable as autograd-graph tensors
            # — target's backward flows from these into the cond LoRA params.
            attn._cond_k_cached = k
            attn._cond_v_cached = v

    def _cond_block_forward(
        self, cond_x: torch.Tensor, idx: int, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One block's cond forward. Returns ``(cond_x_out, k, v)``.

        Pulled out of ``_run_cond_path`` so it can be wrapped per-block in
        ``torch.utils.checkpoint``. K and V are *returned* (not assigned via
        side effect) so they sit in the autograd graph as checkpoint outputs;
        the caller is responsible for stashing them on ``attn._cond_*_cached``.
        """
        block = self._block_modules[idx]
        attn = self._self_attn_modules[idx]

        # 1. Self-attention block (cond LoRA on qkv + o)
        normed = block.layer_norm_self_attn(cond_x)

        # qkv = frozen qkv_proj + cond LoRA delta (zero at init)
        qkv_base = attn.qkv_proj(normed)  # [B, S_c, 3D]
        qkv_delta = self.cond_lora_qkv[idx](normed)
        qkv = qkv_base + scale * qkv_delta
        q, k, v = qkv.unflatten(-1, (3, self.num_heads, self.head_dim)).unbind(
            dim=-3
        )
        # Apply DiT's q/k/v RMSNorms (shared, frozen — same as target).
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        v = attn.v_norm(v)

        # Cond's own self-attention evolution (no RoPE, no mask).
        # q,k,v: [B, S_c, n_h, d_h] -> SDPA wants [B, n_h, S_c, d_h]
        q_s = q.transpose(1, 2).contiguous()
        k_s = k.transpose(1, 2).contiguous()
        v_s = v.transpose(1, 2).contiguous()
        attn_out = F.scaled_dot_product_attention(q_s, k_s, v_s)
        attn_out = attn_out.transpose(1, 2).reshape(
            cond_x.shape[0], cond_x.shape[1], self.num_heads * self.head_dim
        )

        # output_proj + cond_lora_o (residual back to cond_x)
        o_base = attn.output_proj(attn_out)
        o_delta = self.cond_lora_o[idx](attn_out)
        cond_x = cond_x + (o_base + scale * o_delta)

        # 2. Cross-attn skipped entirely.

        # 3. MLP (with optional cond LoRA on layer1 and layer2)
        normed = block.layer_norm_mlp(cond_x)
        h = block.mlp.layer1(normed)
        if self.apply_ffn_lora and self.cond_lora_ffn1 is not None:
            h = h + scale * self.cond_lora_ffn1[idx](normed)
        h = block.mlp.activation(h)
        o = block.mlp.layer2(h)
        if self.apply_ffn_lora and self.cond_lora_ffn2 is not None:
            o = o + scale * self.cond_lora_ffn2[idx](h)
        cond_x = cond_x + o

        return cond_x, k, v

    # ------------------------------------------------------------ trainer hooks

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def is_mergeable(self):
        return False

    def enable_gradient_checkpointing(self):
        # Per-block checkpoint of the cond pre-pass (see _run_cond_path). K_c
        # and V_c are returned from each checkpointed block so target-loss
        # backward can flow through them into the cond LoRAs; everything else
        # (norm, qkv, attn_out, FFN intermediates) is recomputed on backward.
        self._gradient_checkpointing = True

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
        # Single param group keeps configuration simple. b_cond is included so
        # it learns alongside the LoRA branches.
        params = [{"params": list(self.parameters()), "lr": lr}]
        descriptions = ["easycontrol"]
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
            metadata["ss_network_module"] = "networks.easycontrol_anima"
            metadata["ss_network_spec"] = "easycontrol"
            metadata["ss_num_blocks"] = str(self.num_blocks)
            metadata["ss_hidden_size"] = str(self.hidden_size)
            metadata["ss_num_heads"] = str(self.num_heads)
            metadata["ss_mlp_ratio"] = str(self.mlp_ratio)
            metadata["ss_cond_lora_dim"] = str(self.cond_lora_dim)
            metadata["ss_cond_lora_alpha"] = str(self.cond_lora_alpha)
            metadata["ss_b_cond_init"] = str(self.b_cond_init)
            metadata["ss_cond_scale"] = str(self.cond_scale)
            metadata["ss_apply_ffn_lora"] = str(int(self.apply_ffn_lora))

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
                f"EasyControlNetwork.load_state_dict: missing={missing}, unexpected={unexpected}"
            )
        else:
            logger.info(f"Loaded EasyControl weights from {file} ({len(sd)} tensors)")


# ----------------------------------------------------------------- patched forward


class _ExtendedSelfAttnLSEFunc(torch.autograd.Function):
    """LSE-decomposed extended self-attention with a per-block scalar logit bias.

    Mathematically equivalent to::

        joint_out = softmax([Q@K_t^T·s ; Q@K_c^T·s + b]) @ [V_t; V_c]

    but never materializes the ``[B, H, S_q, S_t+S_c]`` attention matrix. Two
    memory-efficient FA2 forwards on the disjoint key tiles, then a Python
    LSE-arithmetic combine::

        α = exp(lse_t  - joint_lse)
        β = exp(lse_c+b - joint_lse)        joint_lse = logaddexp(lse_t, lse_c + b)
        joint_out = α · out_t + β · out_c

    Forward correctness (vs. masked SDPA) is identity, modulo fp32 ulp.

    Backward correctness is more subtle. FA2's stock ``FlashAttnFunc.backward``
    only consumes ``dout`` and silently discards the upstream gradient on
    ``softmax_lse``. A plain "two FA + Python combine via flash_attn_func"
    therefore drops the *path-2* gradient that flows from the loss back through
    α/β into ``q``/``k_t``/``k_c`` (the contribution scales as α·β·(out_c−out_t)
    in dout-space; negligible at init when β≈4.5e-5 from b_cond=-10, but grows
    as b_cond rises during training).

    To recover the joint-softmax gradient exactly, this Function bypasses the
    stock autograd and calls ``_wrapped_flash_attn_forward / _backward``
    directly. The trick: feeding ``softmax_lse = joint_lse`` (target) and
    ``softmax_lse = joint_lse - b`` (cond) into the per-tile FA backward causes
    FA to compute joint-softmax probabilities ``exp(L_t·s)/Z`` and
    ``exp(L_c·s + b)/Z`` respectively, so per-tile contributions sum to the
    correct joint gradient on q/k/v. ``b_cond``'s gradient is computed
    analytically from α, β, out_t, out_c, dout.
    """

    @staticmethod
    def forward(ctx, q, k_t, v_t, k_c, v_c, b_cond, softmax_scale):
        from networks import attention as anima_attention

        if anima_attention._wrapped_flash_attn_forward is None:
            raise RuntimeError(
                "_ExtendedSelfAttnLSEFunc requires flash-attn to be installed"
            )
        fa_fwd = anima_attention._wrapped_flash_attn_forward

        # Two FA forwards (no dropout, no causal, no window).
        out_t, lse_t, _, rng_state_t = fa_fwd(
            q, k_t, v_t, 0.0, softmax_scale,
            causal=False, window_size_left=-1, window_size_right=-1,
            softcap=0.0, alibi_slopes=None, return_softmax=False,
        )
        out_c, lse_c, _, rng_state_c = fa_fwd(
            q, k_c, v_c, 0.0, softmax_scale,
            causal=False, window_size_left=-1, window_size_right=-1,
            softcap=0.0, alibi_slopes=None, return_softmax=False,
        )

        # LSE arithmetic combine. (FA returns lse in fp32 regardless of
        # input dtype, so b_cond — also fp32 — adds without promotion.)
        b_fp32 = b_cond.to(lse_c.dtype)
        lse_c_adj = lse_c + b_fp32
        joint_lse = torch.logaddexp(lse_t, lse_c_adj)
        alpha = (lse_t - joint_lse).exp()        # [B, H, S_q] fp32
        beta = (lse_c_adj - joint_lse).exp()     # [B, H, S_q] fp32

        # out_t, out_c are [B, S_q, H, D] (BLHD). Broadcast α/β over D.
        alpha_bd = alpha.transpose(1, 2).unsqueeze(-1).to(out_t.dtype)
        beta_bd = beta.transpose(1, 2).unsqueeze(-1).to(out_c.dtype)
        joint_out = alpha_bd * out_t + beta_bd * out_c

        ctx.save_for_backward(
            q, k_t, v_t, k_c, v_c,
            joint_out, joint_lse, alpha, beta, out_t, out_c,
            b_fp32, rng_state_t, rng_state_c,
        )
        ctx.softmax_scale = softmax_scale
        ctx.b_cond_orig_dtype = b_cond.dtype
        return joint_out

    @staticmethod
    def backward(ctx, dout):
        from networks import attention as anima_attention

        fa_bwd = anima_attention._wrapped_flash_attn_backward
        (q, k_t, v_t, k_c, v_c,
         joint_out, joint_lse, alpha, beta, out_t, out_c,
         b_fp32, rng_state_t, rng_state_c) = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale

        dout = dout.contiguous()

        # Tile 1 (target) — feed JOINT lse and JOINT out so that FA computes
        # per-key softmax mass = exp(L_t·s - joint_lse) = exp(L_t·s) / Z, which
        # is the joint-softmax probability on target keys; and uses joint_out
        # as the "softmax output" reference (V_t - joint_out is the correct
        # second term).
        dq_t = torch.empty_like(q)
        dk_t = torch.empty_like(k_t)
        dv_t = torch.empty_like(v_t)
        fa_bwd(
            dout, q, k_t, v_t, joint_out, joint_lse,
            dq_t, dk_t, dv_t,
            0.0, softmax_scale, False, -1, -1, 0.0, None, False,
            rng_state=rng_state_t,
        )

        # Tile 2 (cond) — feed (joint_lse - b) so per-key mass becomes
        # exp(L_c·s - (joint_lse - b)) = exp(L_c·s + b) / Z, the joint-softmax
        # probability on cond keys (with the bias).
        effective_lse_c = joint_lse - b_fp32
        dq_c = torch.empty_like(q)
        dk_c = torch.empty_like(k_c)
        dv_c = torch.empty_like(v_c)
        fa_bwd(
            dout, q, k_c, v_c, joint_out, effective_lse_c,
            dq_c, dk_c, dv_c,
            0.0, softmax_scale, False, -1, -1, 0.0, None, False,
            rng_state=rng_state_c,
        )

        dq = dq_t + dq_c

        # b_cond gradient — analytical from the LSE arithmetic.
        #   ∂joint_out/∂b = α · β · (out_c − out_t)            [B, S_q, H, D]
        #   ∂L/∂b         = sum (α · β · ⟨out_c − out_t, dout⟩_D)
        # Reduction in fp32 for stability (α, β are fp32; bf16 inner can lose
        # ulps on long S_q reductions).
        inner_bsh = ((out_c.float() - out_t.float()) * dout.float()).sum(dim=-1)  # [B, S_q, H]
        inner_bhq = inner_bsh.transpose(1, 2)                                     # [B, H, S_q]
        db_scalar = (alpha * beta * inner_bhq).sum()
        db_cond = db_scalar.to(ctx.b_cond_orig_dtype)
        # Match b_cond's original 0-d shape.
        if b_fp32.dim() == 0:
            db_cond = db_cond.reshape(())

        return dq, dk_t, dv_t, dk_c, dv_c, db_cond, None


_LSE_FALLBACK_WARNED = False


def _warn_lse_fallback_once(reason: str) -> None:
    """One-shot warning when we can't use the LSE-decomposed path."""
    global _LSE_FALLBACK_WARNED
    if _LSE_FALLBACK_WARNED:
        return
    _LSE_FALLBACK_WARNED = True
    logger.warning(
        f"EasyControl: falling back to masked-SDPA path ({reason}). The math "
        f"kernel materializes a [B, H, S_t, S_t+S_c] attention matrix per "
        f"block (~1 GB / block at bf16), which can OOM on real hardware. "
        f"Install flash-attn and use attn_mode='flash' for the LSE-decomposed "
        f"path."
    )


def _make_patched_self_attn_forward(orig_attn, b_param: nn.Parameter, anima_attention):
    """Build a closure that replaces ``Attention.forward`` for one block's self-attn.

    Mirrors the original ``Attention.forward`` (library/anima/models.py:446) but
    extends self-attn keys/values with cached cond K/V when present, with an
    additive per-block scalar logit bias on the cond positions. When no cond
    is set, falls back to the original anima attention dispatch.

    The extended self-attn uses the LSE-decomposition path
    (``_ExtendedSelfAttnLSEFunc``) when flash-attn is available and
    ``attn_params.attn_mode == 'flash'`` — two memory-efficient FA forwards
    plus a Python LSE-arithmetic combine. Otherwise it falls back to the
    masked-SDPA path (math kernel; OOM risk on real hardware) with a one-shot
    warning.

    ``b_param`` is the 0-d ``nn.Parameter`` from this block's slot in
    ``ec_net.b_cond`` (a ``ParameterList``). Captured as a tensor object — NOT
    via ``ec_net.b_cond[block_idx]`` — because dynamo specializes per-call on
    int closure cells (treating them as static nn.Module attributes), which
    used to cause one recompile per block until recompile_limit blew up.
    """

    def patched_forward(x, attn_params, context, rope_cos_sin=None):
        q, k, v = orig_attn.compute_qkv(x, context, rope_cos_sin)

        cond_k = getattr(orig_attn, "_cond_k_cached", None)
        cond_v = getattr(orig_attn, "_cond_v_cached", None)

        if cond_k is None or cond_v is None:
            # No cond — exact baseline DiT behavior (anima dispatch).
            if q.dtype != v.dtype:
                if (
                    not attn_params.supports_fp32 or attn_params.requires_same_dtype
                ) and torch.is_autocast_enabled():
                    target_dtype = v.dtype
                    q = q.to(target_dtype)
                    k = k.to(target_dtype)
            qkv = [q, k, v]
            result = anima_attention.attention(qkv, attn_params=attn_params)
            return orig_attn.output_dropout(orig_attn.output_proj(result))

        # Extended self-attn path.
        B = q.shape[0]
        if cond_k.shape[0] == 1 and B > 1:
            cond_k = cond_k.expand(B, *cond_k.shape[1:]).contiguous()
            cond_v = cond_v.expand(B, *cond_v.shape[1:]).contiguous()
        elif cond_k.shape[0] != B:
            raise RuntimeError(
                f"EasyControl cond K/V batch {cond_k.shape[0]} does not match q batch {B}"
            )

        # Match dtypes.
        if q.dtype != v.dtype:
            if (
                not attn_params.supports_fp32 or attn_params.requires_same_dtype
            ) and torch.is_autocast_enabled():
                target_dtype = v.dtype
                q = q.to(target_dtype)
                k = k.to(target_dtype)
        cond_k = cond_k.to(k.dtype)
        cond_v = cond_v.to(v.dtype)

        scale = attn_params.softmax_scale  # may be None → default 1/sqrt(d)
        if scale is None:
            scale = q.shape[-1] ** -0.5

        # Pick implementation: LSE-decomposed (memory-efficient) when FA is
        # available, masked SDPA (math kernel; OOM risk) otherwise.
        use_lse = (
            anima_attention._wrapped_flash_attn_forward is not None
            and attn_params.attn_mode == "flash"
        )
        if use_lse:
            S_t = q.shape[1]
            # Inputs are BLHD; FA wants BLHD; pass through directly.
            out = _ExtendedSelfAttnLSEFunc.apply(
                q.contiguous(), k.contiguous(), v.contiguous(),
                cond_k, cond_v,
                b_param, scale,
            )
            out = out.reshape(B, S_t, -1)
            return orig_attn.output_dropout(orig_attn.output_proj(out))

        # Fallback: masked extended SDPA. Materializes the full attention
        # matrix in the math kernel — this is the OOM cliff that Phase 1.5
        # was meant to fix. Use only when FA is unavailable.
        if attn_params.attn_mode == "flash":
            _warn_lse_fallback_once("flash-attn import failed at module load")
        else:
            _warn_lse_fallback_once(f"attn_mode={attn_params.attn_mode!r} unsupported by LSE path")

        S_t = k.shape[1]
        S_c = cond_k.shape[1]
        k_ext = torch.cat([k, cond_k], dim=1)
        v_ext = torch.cat([v, cond_v], dim=1)
        q_s = q.transpose(1, 2)
        k_s = k_ext.transpose(1, 2)
        v_s = v_ext.transpose(1, 2)
        b = b_param.to(q_s.dtype)
        target_zeros = torch.zeros(S_t, device=q.device, dtype=q_s.dtype)
        cond_b = b.expand(S_c)
        attn_bias = torch.cat([target_zeros, cond_b], dim=0)
        attn_mask = attn_bias.view(1, 1, 1, S_t + S_c)
        out = F.scaled_dot_product_attention(
            q_s, k_s, v_s, attn_mask=attn_mask, scale=scale
        )
        out = out.transpose(1, 2).reshape(B, S_t, -1)
        return orig_attn.output_dropout(orig_attn.output_proj(out))

    return patched_forward
