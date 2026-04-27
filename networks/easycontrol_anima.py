"""EasyControl network module for Anima — two-stream rewrite.

Architecture (adapter-only — DiT frozen):

  reference image (clean VAE latent, 4D [B, C, H, W])
      -> DiT.x_embedder (frozen, reused)             [B, T_c, H_c, W_c, D]
      -> flatten -> static-pad to cond_token_count    [B, S_c, D]
      -> cond_rope = DiT.pos_embedder at cond shape, padded to S_c
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
# Cond is static-padded to this token count. Default 4096 matches Anima's
# constant-token bucketing (target is also static-padded to 4096), so for the
# common ref==target setup the cond latent's native tokens (≤ 4096 by bucket
# design) just pad to 4096 with no downsample required. Lower it (e.g. 1024)
# to match the official EasyControl's 32×32 reference image if memory is
# tight; ``encode_cond_latent`` will then refuse cond latents that would
# exceed the budget — the caller must downsample explicitly.
DEFAULT_COND_TOKEN_COUNT = 4096


class _LoRAProj(nn.Module):
    """Plain LoRA-style D->r->out_dim projection with up zero-init.

    Standalone (not a wrapper around an org_module) — used by EasyControl to
    add a delta to a frozen DiT projection only on the cond stream. Output
    added by the caller; this module just produces the delta.
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
    cond_token_count = int(kwargs.get("cond_token_count", DEFAULT_COND_TOKEN_COUNT))

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
        cond_token_count=cond_token_count,
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
    cond_token_count = int(
        metadata.get("ss_cond_token_count", DEFAULT_COND_TOKEN_COUNT)
    )

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
        cond_token_count=cond_token_count,
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
        cond_token_count: int,
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
        self.cond_token_count = cond_token_count
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
        #                        padded to cond_token_count)
        # cond_x_init for block 0 lives on block_modules[0]._easycontrol_cond_x_in.
        self._cond_state: Optional[dict] = None

        # Inference KV cache: per-block (cond_k, cond_v) post-RoPE-and-norm,
        # i.e. the exact tensors `_extended_target_attention` consumes from the
        # cond stream. Populated by `precompute_cond_kv()`. When non-None, the
        # patched Block.forward bypasses the cond stream entirely and feeds
        # these tensors into target's extended self-attention. Training keeps
        # this None — every step needs the cond LoRA's gradient.
        self._cond_kv_cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"EasyControlNetwork: blocks={num_blocks}, hidden={hidden_size}/{num_heads}h, "
            f"r={cond_lora_dim} alpha={cond_lora_alpha}, ffn_lora={apply_ffn_lora}, "
            f"b_cond_init={b_cond_init}, cond_scale={cond_scale}, "
            f"cond_token_count={cond_token_count}, params={total / 1e6:.1f}M"
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

    def remove_from(self):
        for block, orig in zip(self._block_modules, self._original_block_forwards):
            block.forward = orig
            if hasattr(block, "_easycontrol_cond_x_in"):
                del block._easycontrol_cond_x_in
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
        plus the matching RoPE table at cond's native (smaller) shape.

        Reuses the DiT's (frozen) ``x_embedder`` and ``pos_embedder``. Both
        outputs are static-padded to ``cond_token_count`` so block compute
        sees a single S_c across all batches / buckets.

        Args:
            cond_latent: ``[B, C, H, W]`` (image) or ``[B, C, T, H, W]`` (video).
            padding_mask: optional ``[B, 1, H, W]``. If None and the DiT
                requires it, a default all-ones mask is synthesized.
        Returns:
            ``(cond_x, cond_rope)``:
              - ``cond_x``:    ``[B, S_c, D]``,    S_c = cond_token_count
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
        # Flatten cond_x to [B, S_c_native, D].
        cond_x = cond_x_5d.flatten(1, 3)
        S_c_native = cond_x.shape[1]
        S_c_static = self.cond_token_count

        if S_c_native > S_c_static:
            raise ValueError(
                f"cond latent produces {S_c_native} tokens > cond_token_count={S_c_static}. "
                f"Either lower the reference resolution or raise cond_token_count."
            )
        if S_c_native < S_c_static:
            cond_x = F.pad(cond_x, (0, 0, 0, S_c_static - S_c_native))

        # Pad RoPE (cos, sin) to S_c_static. Each is [S_c_native, 1, 1, D_head].
        if cond_rope is not None:
            cos, sin = cond_rope
            if cos.shape[0] < S_c_static:
                pad = (0, 0, 0, 0, 0, 0, 0, S_c_static - cos.shape[0])
                cos = F.pad(cos, pad)
                sin = F.pad(sin, pad)
            cond_rope = (cos, sin)

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

        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        for idx, block in enumerate(self._block_modules):
            attn = block.self_attn
            cond_lora_qkv = self.cond_lora_qkv[idx]
            cond_lora_o = self.cond_lora_o[idx]
            cond_lora_ffn1 = (
                self.cond_lora_ffn1[idx] if self.apply_ffn_lora else None
            )
            cond_lora_ffn2 = (
                self.cond_lora_ffn2[idx] if self.apply_ffn_lora else None
            )

            # ---- AdaLN modulation (cond stream only) ----
            if block.use_adaln_lora:
                cond_fused_down = block.adaln_fused_down(cond_emb)
                cond_down_self, _cd_cross, cond_down_mlp = cond_fused_down.chunk(
                    3, dim=-1
                )
                cond_shift_self, cond_scale_self, cond_gate_self = (
                    block.adaln_up_self_attn(cond_down_self) + cond_adaln_lora
                ).chunk(3, dim=-1)
                cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = (
                    block.adaln_up_mlp(cond_down_mlp) + cond_adaln_lora
                ).chunk(3, dim=-1)
            else:
                cond_shift_self, cond_scale_self, cond_gate_self = (
                    block.adaln_modulation_self_attn(cond_emb).chunk(3, dim=-1)
                )
                cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = (
                    block.adaln_modulation_mlp(cond_emb).chunk(3, dim=-1)
                )

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
            B_c = cond_x.shape[0]
            S_c = cond_x.shape[1]
            cq = cond_q.transpose(1, 2)
            ck = cond_k.transpose(1, 2)
            cv = cond_v.transpose(1, 2)
            cond_attn_out = F.scaled_dot_product_attention(cq, ck, cv)
            cond_attn_out = cond_attn_out.transpose(1, 2).reshape(B_c, S_c, -1)
            cond_attn_proj = attn.output_proj(
                cond_attn_out
            ) + eff_scale * cond_lora_o(cond_attn_out)
            cond_attn_proj = attn.output_dropout(cond_attn_proj)
            cond_x = cond_x + cond_gate_self * cond_attn_proj

            cond_mlp_normed = (
                block.layer_norm_mlp(cond_x) * (1 + cond_scale_mlp)
                + cond_shift_mlp
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

        kv_bytes = sum(k.numel() + v.numel() for k, v in cache) * cache[0][0].element_size()
        logger.info(
            f"EasyControl: precomputed cond KV cache "
            f"({len(cache)} blocks × 2 tensors, {kv_bytes / 1e6:.0f} MB)"
        )

    def get_effective_scale(self) -> float:
        return self.cond_scale * self.multiplier

    # ------------------------------------------------------------ trainer hooks

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def is_mergeable(self):
        return False

    def enable_gradient_checkpointing(self):
        # The two-stream design relies on the existing per-Block checkpoint
        # wrappers (configured by ``Block.enable_gradient_checkpointing``).
        # The patched ``Block.forward`` re-implements the same dispatch with
        # the two-stream inner function as its target. No EasyControl-side
        # state is needed here — kept as a no-op for trainer-side parity.
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
            metadata["ss_cond_token_count"] = str(self.cond_token_count)

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


# ----------------------------------------------------------------- LSE-extended attention


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
            q,
            k_t,
            v_t,
            0.0,
            softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
        out_c, lse_c, _, rng_state_c = fa_fwd(
            q,
            k_c,
            v_c,
            0.0,
            softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )

        # LSE arithmetic combine. (FA returns lse in fp32 regardless of
        # input dtype, so b_cond — also fp32 — adds without promotion.)
        b_fp32 = b_cond.to(lse_c.dtype)
        lse_c_adj = lse_c + b_fp32
        joint_lse = torch.logaddexp(lse_t, lse_c_adj)
        alpha = (lse_t - joint_lse).exp()  # [B, H, S_q] fp32
        beta = (lse_c_adj - joint_lse).exp()  # [B, H, S_q] fp32

        # out_t, out_c are [B, S_q, H, D] (BLHD). Broadcast α/β over D.
        alpha_bd = alpha.transpose(1, 2).unsqueeze(-1).to(out_t.dtype)
        beta_bd = beta.transpose(1, 2).unsqueeze(-1).to(out_c.dtype)
        joint_out = alpha_bd * out_t + beta_bd * out_c

        ctx.save_for_backward(
            q,
            k_t,
            v_t,
            k_c,
            v_c,
            joint_out,
            joint_lse,
            alpha,
            beta,
            out_t,
            out_c,
            b_fp32,
            rng_state_t,
            rng_state_c,
        )
        ctx.softmax_scale = softmax_scale
        ctx.b_cond_orig_dtype = b_cond.dtype
        return joint_out

    @staticmethod
    def backward(ctx, dout):
        from networks import attention as anima_attention

        fa_bwd = anima_attention._wrapped_flash_attn_backward
        (
            q,
            k_t,
            v_t,
            k_c,
            v_c,
            joint_out,
            joint_lse,
            alpha,
            beta,
            out_t,
            out_c,
            b_fp32,
            rng_state_t,
            rng_state_c,
        ) = ctx.saved_tensors
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
            dout,
            q,
            k_t,
            v_t,
            joint_out,
            joint_lse,
            dq_t,
            dk_t,
            dv_t,
            0.0,
            softmax_scale,
            False,
            -1,
            -1,
            0.0,
            None,
            False,
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
            dout,
            q,
            k_c,
            v_c,
            joint_out,
            effective_lse_c,
            dq_c,
            dk_c,
            dv_c,
            0.0,
            softmax_scale,
            False,
            -1,
            -1,
            0.0,
            None,
            False,
            rng_state=rng_state_c,
        )

        dq = dq_t + dq_c

        # b_cond gradient — analytical from the LSE arithmetic.
        #   ∂joint_out/∂b = α · β · (out_c − out_t)            [B, S_q, H, D]
        #   ∂L/∂b         = sum (α · β · ⟨out_c − out_t, dout⟩_D)
        # Reduction in fp32 for stability (α, β are fp32; bf16 inner can lose
        # ulps on long S_q reductions).
        inner_bsh = ((out_c.float() - out_t.float()) * dout.float()).sum(
            dim=-1
        )  # [B, S_q, H]
        inner_bhq = inner_bsh.transpose(1, 2)  # [B, H, S_q]
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


def _extended_target_attention(
    target_q,
    target_k,
    target_v,
    cond_k,
    cond_v,
    *,
    b_param,
    scale,
    attn_params,
):
    """Run target's extended self-attention over [target_k; cond_k].

    Inputs are BSHD: target_q/k/v ``[B, S_t, H, D]``, cond_k/v ``[B, S_c, H, D]``.
    Returns ``[B, S_t, H*D]`` ready for output_proj. Uses
    ``_ExtendedSelfAttnLSEFunc`` (memory-efficient) when flash-attn + flash
    mode is available; falls back to masked-SDPA (math kernel; OOM risk) with
    a one-shot warning otherwise.
    """
    from networks import attention as anima_attention

    # dtype matching mirrors the original Attention.forward casting policy.
    if target_q.dtype != target_v.dtype:
        if (
            not attn_params.supports_fp32 or attn_params.requires_same_dtype
        ) and torch.is_autocast_enabled():
            target_q = target_q.to(target_v.dtype)
            target_k = target_k.to(target_v.dtype)
    cond_k = cond_k.to(target_k.dtype)
    cond_v = cond_v.to(target_v.dtype)

    if scale is None:
        scale = target_q.shape[-1] ** -0.5

    use_lse = (
        anima_attention._wrapped_flash_attn_forward is not None
        and attn_params.attn_mode == "flash"
    )
    if use_lse:
        out = _ExtendedSelfAttnLSEFunc.apply(
            target_q.contiguous(),
            target_k.contiguous(),
            target_v.contiguous(),
            cond_k.contiguous(),
            cond_v.contiguous(),
            b_param,
            scale,
        )
        # out: [B, S_t, H, D] → [B, S_t, H*D]
        B, S_t = out.shape[0], out.shape[1]
        return out.reshape(B, S_t, -1)

    # Fallback: masked extended SDPA. Materializes the full attention matrix
    # in the math kernel — only used when FA is unavailable.
    if attn_params.attn_mode == "flash":
        _warn_lse_fallback_once("flash-attn import failed at module load")
    else:
        _warn_lse_fallback_once(
            f"attn_mode={attn_params.attn_mode!r} unsupported by LSE path"
        )

    B, S_t = target_q.shape[0], target_q.shape[1]
    S_c = cond_k.shape[1]
    k_ext = torch.cat([target_k, cond_k], dim=1)
    v_ext = torch.cat([target_v, cond_v], dim=1)
    q_s = target_q.transpose(1, 2)
    k_s = k_ext.transpose(1, 2)
    v_s = v_ext.transpose(1, 2)
    b = b_param.to(q_s.dtype)
    target_zeros = torch.zeros(S_t, device=target_q.device, dtype=q_s.dtype)
    cond_b = b.expand(S_c)
    attn_bias = torch.cat([target_zeros, cond_b], dim=0).view(1, 1, 1, S_t + S_c)
    out = F.scaled_dot_product_attention(
        q_s, k_s, v_s, attn_mask=attn_bias, scale=scale
    )
    return out.transpose(1, 2).reshape(B, S_t, -1)


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

    if block.use_adaln_lora:
        fused_down = block.adaln_fused_down(emb_B_T_D)
        down_self, down_cross, down_mlp = fused_down.chunk(3, dim=-1)
        shift_self_attn, scale_self_attn, gate_self_attn = (
            block.adaln_up_self_attn(down_self) + adaln_lora_B_T_3D
        ).chunk(3, dim=-1)
        shift_cross_attn, scale_cross_attn, gate_cross_attn = (
            block.adaln_up_cross_attn(down_cross) + adaln_lora_B_T_3D
        ).chunk(3, dim=-1)
        shift_mlp, scale_mlp, gate_mlp = (
            block.adaln_up_mlp(down_mlp) + adaln_lora_B_T_3D
        ).chunk(3, dim=-1)
    else:
        shift_self_attn, scale_self_attn, gate_self_attn = (
            block.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
        )
        shift_cross_attn, scale_cross_attn, gate_cross_attn = (
            block.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
        )
        shift_mlp, scale_mlp, gate_mlp = block.adaln_modulation_mlp(
            emb_B_T_D
        ).chunk(3, dim=-1)

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
    target_mlp_normed = (
        block.layer_norm_mlp(x_B_T_H_W_D) * (1 + sc_mlp_5) + sh_mlp_5
    )
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

    # Lazy import to avoid a circular at module load.
    from library.anima.models import apply_rotary_pos_emb_qk

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
        if block.use_adaln_lora:
            fused_down = block.adaln_fused_down(emb_B_T_D)
            down_self, down_cross, down_mlp = fused_down.chunk(3, dim=-1)
            shift_self_attn, scale_self_attn, gate_self_attn = (
                block.adaln_up_self_attn(down_self) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn, scale_cross_attn, gate_cross_attn = (
                block.adaln_up_cross_attn(down_cross) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (
                block.adaln_up_mlp(down_mlp) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)

            cond_fused_down = block.adaln_fused_down(cond_emb_B_T_D)
            cond_down_self, _cond_down_cross, cond_down_mlp = cond_fused_down.chunk(
                3, dim=-1
            )
            cond_shift_self_attn, cond_scale_self_attn, cond_gate_self_attn = (
                block.adaln_up_self_attn(cond_down_self) + cond_adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = (
                block.adaln_up_mlp(cond_down_mlp) + cond_adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn, scale_self_attn, gate_self_attn = (
                block.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_cross_attn, scale_cross_attn, gate_cross_attn = (
                block.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_mlp, scale_mlp, gate_mlp = block.adaln_modulation_mlp(
                emb_B_T_D
            ).chunk(3, dim=-1)

            cond_shift_self_attn, cond_scale_self_attn, cond_gate_self_attn = (
                block.adaln_modulation_self_attn(cond_emb_B_T_D).chunk(3, dim=-1)
            )
            cond_shift_mlp, cond_scale_mlp, cond_gate_mlp = block.adaln_modulation_mlp(
                cond_emb_B_T_D
            ).chunk(3, dim=-1)

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

        # Cond's own self-attention — small (S_c × S_c), plain torch SDPA.
        # cond_q/k/v are BSHD: (B, S_c, n_h, d_h). SDPA expects (B, n_h, S, d_h).
        cq = cond_q.transpose(1, 2)
        ck = cond_k.transpose(1, 2)
        cv = cond_v.transpose(1, 2)
        cond_attn_out = F.scaled_dot_product_attention(cq, ck, cv)
        # Back to (B, S_c, n_h*d_h) for output_proj.
        B_c = cond_x_B_S_D.shape[0]
        S_c = cond_x_B_S_D.shape[1]
        cond_attn_out = cond_attn_out.transpose(1, 2).reshape(B_c, S_c, -1)

        # Output projections + LoRA on cond + gated residuals.
        target_attn_proj = attn.output_proj(target_attn_out)
        cond_attn_proj = attn.output_proj(cond_attn_out) + eff_scale * cond_lora_o(
            cond_attn_out
        )
        # Output dropout (Anima has nn.Identity by default; harmless when on).
        target_attn_proj = attn.output_dropout(target_attn_proj)
        cond_attn_proj = attn.output_dropout(cond_attn_proj)

        target_attn_5d = target_attn_proj.unflatten(1, (T_dim, H_dim, W_dim))
        x_B_T_H_W_D = x_B_T_H_W_D + ga_self_5 * target_attn_5d
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
        # FFN LoRA at layer1 and layer2 outputs (matches Phase 1.5).
        cond_mlp_normed = (
            block.layer_norm_mlp(cond_x_B_S_D) * (1 + cond_scale_mlp) + cond_shift_mlp
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
        # tuple so the checkpoint preserves them as inputs.
        if block.training and block.gradient_checkpointing:
            if block.unsloth_offload_checkpointing:
                from library.anima.models import unsloth_checkpoint

                target_x_out, cond_x_out = unsloth_checkpoint(
                    _two_stream_inner,
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
                    outputs = _two_stream_inner(*device_inputs)
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
                    _two_stream_inner,
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
            target_x_out, cond_x_out = _two_stream_inner(
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
