# Original code: NVIDIA CORPORATION & AFFILIATES, licensed under Apache-2.0

import math
from typing import Any, Optional, Tuple

import torch
from einops import repeat
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library.runtime import offloading as custom_offloading_utils
from library.runtime.device import weighs_to_device
from library.inference.corrections.mod_guidance_core import project_pooled
from networks import attention_dispatch


# Based on Unsloth Zoo by Daniel Han-Chen & the Unsloth team
try:
    from deepspeed.runtime.activation_checkpointing.checkpointing import detach_variable
except ImportError:

    def detach_variable(inputs, device=None):
        """Detach tensors from computation graph, optionally moving to a device.

        Reimplementation of deepspeed.runtime.activation_checkpointing.checkpointing.detach_variable
        for environments without DeepSpeed installed.
        """
        if isinstance(inputs, tuple):
            out = []
            for inp in inputs:
                if not isinstance(inp, torch.Tensor):
                    out.append(inp)
                    continue
                requires_grad = inp.requires_grad
                if device is not None:
                    x = inp.to(device=device)
                else:
                    x = inp
                x = x.detach()
                x.requires_grad = requires_grad
                out.append(x)
            return tuple(out)
        else:
            raise RuntimeError(
                "Only tuple of tensors is supported. Got Unsupported input type: ",
                type(inputs).__name__,
            )


class UnslothOffloadedGradientCheckpointer(torch.autograd.Function):
    """Saves VRAM by offloading activations to CPU RAM using non-blocking transfers.

    Uses non_blocking=True to hide CPU<->GPU transfer latency behind compute.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, forward_function, hidden_states, *args):
        # Remember the original device for backward pass (multi-GPU support)
        ctx.input_device = hidden_states.device
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        # args stored on ctx (not save_for_backward): the training loop already
        # holds references to these tensors, so GC isn't a concern.
        ctx.args = args
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grads):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to(ctx.input_device, non_blocking=True).detach()
        hidden_states.requires_grad_(True)
        args = detach_variable(ctx.args)
        inputs = (hidden_states,) + args
        with torch.enable_grad():
            outputs = ctx.forward_function(*inputs)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(
            outputs if isinstance(outputs, tuple) else (outputs,),
            grads if isinstance(grads, tuple) else (grads,),
        ):
            if isinstance(out, torch.Tensor) and out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None,) + tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs
        )


@torch._disable_dynamo
def unsloth_checkpoint(function, *args):
    """Wrapper for UnslothOffloadedGradientCheckpointer."""
    return UnslothOffloadedGradientCheckpointer.apply(function, *args)


def _make_dynamic_seq_forward(compiled_inner, lo, hi):
    """Wrap a compiled ``Block._forward`` in an eager ``mark_dynamic`` prologue.

    compile_dynamic_seq collapses the per-token-count block graphs to one by
    marking the seq axis dynamic. The marks MUST live inside the checkpointed
    callable, not in a one-shot prologue before ``Block.forward`` dispatches:
    under grad checkpointing the inner is recomputed in BACKWARD via
    ``detach_variable``, which detaches the tensor args (``x``) into fresh
    tensors that LOSE the dynamic mark — while the ``rope_cos_sin`` *tuple* is
    passed through unchanged (``detach_variable`` skips non-tensors) and KEEPS
    it. That asymmetry (rope hard-dynamic, x specialized to a constant token
    count) is the ``ConstraintViolationError``. Marking inside the recomputed
    callable re-applies the marks to the detached inputs every pass, so forward
    and backward agree. Mirrors the EasyControl two-stream fix
    (networks/methods/easycontrol.py). ``x`` is fake-5D ``(B,1,seq,1,D)`` under
    native_flatten (guaranteed on when compile_blocks set dynamic_seq), so the
    seq axis is dim 2; each RoPE table rides dim 0. Marking is idempotent.
    """

    def marked_forward(
        x_B_T_H_W_D,
        emb_B_T_D,
        crossattn_emb,
        attn_params,
        rope_cos_sin=None,
        adaln_lora_B_T_3D=None,
    ):
        torch._dynamo.mark_dynamic(x_B_T_H_W_D, 2, min=lo, max=hi)
        if rope_cos_sin is not None:
            torch._dynamo.mark_dynamic(rope_cos_sin[0], 0, min=lo, max=hi)
            torch._dynamo.mark_dynamic(rope_cos_sin[1], 0, min=lo, max=hi)
        return compiled_inner(
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            attn_params,
            rope_cos_sin,
            adaln_lora_B_T_3D,
        )

    return marked_forward


@torch.compiler.disable(recursive=True)
def _unflatten_native_shape(x, flatten_info):
    """Restore the fake-5D flattened sequence back to (B, T, H, W, D).

    Disabled from dynamo tracing on purpose: flatten_info is a 4-tuple of Python
    ints (T_s, H_s, W_s, seq_len) computed from the input's pre-flatten shape, so
    if this ran inside the compiled frame each bucket would specialize
    ``flatten_info[1] == H_s`` (per-value guard) and narrow the symbolic range
    on ``flatten_info[3]`` (per-bucket seq_len guard). Running it eagerly keeps
    the returned tensor's shape as the only signal crossing back into the
    compile zone — downstream ops (final_layer, unpatchify) then pick up
    symbolic T/H/W from the tensor itself, not from Python ints.
    """
    T_s, H_s, W_s, seq_len = flatten_info
    x = x.squeeze(3).squeeze(1)
    x = x[:, :seq_len, :]
    x = x.unflatten(1, (T_s, H_s, W_s))
    return x


from library.log import setup_logging  # noqa: E402

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def apply_rotary_pos_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_cos_sin: tuple[torch.Tensor, torch.Tensor],
    tensor_format: str = "sbhd",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k using precomputed (cos, sin) tensors.

    RoPE is always generated from the same shape as the ``q`` it is applied to
    (``pos_embedder`` builds cos/sin at the exact ``T*H*W`` of the current latent;
    the EasyControl cond path builds its cond rope at the cond's native shape),
    so ``cos_.shape[0] == q``'s seq length in every path — no length trim needed.
    Dropping the old ``cos_[:cur_seq_len]`` slice also keeps the seq axis a clean
    single symbol under ``compile_dynamic_seq``'s ``mark_dynamic`` (no symbolic
    indexing op in the compiled block).
    """
    cos_, sin_ = rope_cos_sin

    if tensor_format == "bshd":
        cos_ = cos_.transpose(0, 1)
        sin_ = sin_.transpose(0, 1)

    rot_dim = cos_.shape[-1]

    cos_q = cos_.to(q.dtype)
    sin_q = sin_.to(q.dtype)
    # For Anima rot_dim == head_dim (dims sum by construction), so the pass-through
    # slice is empty; skip the torch.cat to avoid a full per-Q/K-per-block alloc.
    # `rot_dim == q.shape[-1]` is a compile-time constant, so the branch resolves
    # once under torch.compile (no per-bucket recompile).
    q_rot = q[..., :rot_dim]
    q_emb = (q_rot * cos_q) + (_rotate_half(q_rot, False) * sin_q)
    q = (
        q_emb
        if rot_dim == q.shape[-1]
        else torch.cat((q_emb, q[..., rot_dim:]), dim=-1)
    )

    cos_k = cos_q if k.dtype == q.dtype else cos_.to(k.dtype)
    sin_k = sin_q if k.dtype == q.dtype else sin_.to(k.dtype)
    k_rot = k[..., :rot_dim]
    k_emb = (k_rot * cos_k) + (_rotate_half(k_rot, False) * sin_k)
    k = (
        k_emb
        if rot_dim == k.shape[-1]
        else torch.cat((k_emb, k[..., rot_dim:]), dim=-1)
    )

    return q, k


class RMSNorm(torch.nn.Module):
    """RMS Normalization for DiT blocks."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float())
        return (output * self.weight).to(x.dtype)


class GPT2FeedForward(nn.Module):
    """GELU feedforward network."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

        self._layer_id = None
        self._dim = d_model
        self._hidden_dim = d_ff
        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self._dim)
        torch.nn.init.trunc_normal_(self.layer1.weight, std=std, a=-3 * std, b=3 * std)

        std = 1.0 / math.sqrt(self._hidden_dim)
        if self._layer_id is not None:
            std = std / math.sqrt(2 * (self._layer_id + 1))
        torch.nn.init.trunc_normal_(self.layer2.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x


class Attention(nn.Module):
    """Multi-head attention supporting both self-attention and cross-attention.

    Uses QK-norm (RMSNorm on q/k) and optional RoPE (only for self-attention).
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        n_heads: int = 16,
        head_dim: int = 128,
        dropout: float = 0.0,
        qkv_format: str = "bshd",
    ) -> None:
        super().__init__()
        self.is_selfattn = context_dim is None

        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = head_dim * n_heads

        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv_format = qkv_format
        self.query_dim = query_dim
        self.context_dim = context_dim

        if self.is_selfattn:
            self.qkv_proj = nn.Linear(query_dim, 3 * inner_dim, bias=False)
        else:
            self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
            self.kv_proj = nn.Linear(context_dim, 2 * inner_dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.v_norm = nn.Identity()

        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        if not self.is_selfattn:
            # Inference-side per-key logit bias on the cross-attn QK^T rows
            # (frontload_text_boost arm (d) — the allocation probe; embedding
            # -row scaling can't reach allocation because k_norm is scale-
            # invariant per (token, head)). None = off, exact identity. A
            # (L_ctx,) float tensor routes this call through SDPA's additive
            # attn_mask. Non-persistent, set/cleared via
            # library.inference.adapters.set_xattn_kbias.
            self.register_buffer("_ctx_k_bias", None, persistent=False)

        self._query_dim = query_dim
        self._context_dim = context_dim
        self._inner_dim = inner_dim
        self.init_weights()

    def init_weights(self) -> None:
        if self.is_selfattn:
            # Self-attention: query_dim == context_dim, single std for fused QKV
            std = 1.0 / math.sqrt(self._query_dim)
            torch.nn.init.trunc_normal_(
                self.qkv_proj.weight, std=std, a=-3 * std, b=3 * std
            )
        else:
            std = 1.0 / math.sqrt(self._query_dim)
            torch.nn.init.trunc_normal_(
                self.q_proj.weight, std=std, a=-3 * std, b=3 * std
            )
            std = 1.0 / math.sqrt(self._context_dim)
            torch.nn.init.trunc_normal_(
                self.kv_proj.weight, std=std, a=-3 * std, b=3 * std
            )

        std = 1.0 / math.sqrt(self._inner_dim)
        torch.nn.init.trunc_normal_(
            self.output_proj.weight, std=std, a=-3 * std, b=3 * std
        )

        for layer in self.q_norm, self.k_norm, self.v_norm:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple:
        if self.is_selfattn:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.unflatten(-1, (3, self.n_heads, self.head_dim)).unbind(dim=-3)
        else:
            q = self.q_proj(x).unflatten(-1, (self.n_heads, self.head_dim))
            kv = self.kv_proj(context)
            k, v = kv.unflatten(-1, (2, self.n_heads, self.head_dim)).unbind(dim=-3)

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        if self.is_selfattn and rope_cos_sin is not None:
            q, k = apply_rotary_pos_emb_qk(
                q, k, rope_cos_sin, tensor_format=self.qkv_format
            )

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        attn_params: attention_dispatch.AttentionParams,
        context: torch.Tensor,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        q, k, v = self.compute_qkv(x, context, rope_cos_sin=rope_cos_sin)
        if q.dtype != v.dtype:
            if not attn_params.supports_fp32 and torch.is_autocast_enabled():
                # FlashAttention requires fp16/bf16; only cast when autocast is active.
                target_dtype = v.dtype
                q = q.to(target_dtype)
                k = k.to(target_dtype)
        ctx_k_bias = None if self.is_selfattn else self._ctx_k_bias
        if ctx_k_bias is not None:
            # Per-key logit bias needs SDPA's additive float mask; flash/sage
            # take none, so this call drops to attn_mode="torch" while set.
            attn_params = attention_dispatch.AttentionParams(
                attn_mode="torch",
                attention_mask=ctx_k_bias.to(q.dtype)[None, None, None, :],
                softmax_scale=(
                    attn_params.softmax_scale if attn_params is not None else None
                ),
            )
        qkv = [q, k, v]
        del q, k, v
        result = attention_dispatch.dispatch_attention(qkv, attn_params=attn_params)
        return self.output_dropout(self.output_proj(result))


class VideoPositionEmb(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @property
    def seq_dim(self) -> int:
        return 1

    def forward(
        self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B_T_H_W_C = x_B_T_H_W_C.shape
        return self.generate_embeddings(B_T_H_W_C, fps=fps)

    def generate_embeddings(
        self, B_T_H_W_C: torch.Size, fps: Optional[torch.Tensor]
    ) -> Any:
        raise NotImplementedError


class VideoRopePosition3DEmb(VideoPositionEmb):
    """3D Rotary Position Embedding for video (T, H, W) dimensions."""

    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 24,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        self.register_buffer(
            "seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float)
        )
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, (
            f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        )
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self._cos_sin_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = (
            torch.arange(max(self.max_h, self.max_w, self.max_t))
            .float()
            .to(self.dim_spatial_range.device)
        )
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)]
            .float()
            .to(self.dim_spatial_range.device)
            / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)]
            .float()
            .to(self.dim_spatial_range.device)
            / dim_t
        )

    def _cache_key(
        self,
        T: int,
        H: int,
        W: int,
        fps: Optional[torch.Tensor],
        h_offset: int,
        w_offset: int,
    ) -> tuple:
        fps_val = None if fps is None else fps[:1].item()
        return (T, H, W, fps_val, h_offset, w_offset)

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, W, _ = B_T_H_W_C

        # Skip the Python dict cache inside compiled code — dict mutations cause
        # dynamo guard failures/recompiles; the RoPE math traces cleanly without it.
        _compiling = torch.compiler.is_compiling()

        if not _compiling:
            if h_ntk_factor is None and w_ntk_factor is None and t_ntk_factor is None:
                key = self._cache_key(T, H, W, fps, 0, 0)
                cached = self._cos_sin_cache.get(key)
                if cached is not None:
                    return cached

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        assert H <= self.max_h and W <= self.max_w, (
            f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        )
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert uniform_fps or B == 1 or T == 1, (
                "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
            )

            if fps is None:
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(
                    self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs
                )
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        freqs = em_T_H_W_D.flatten(0, 2).unsqueeze(1).unsqueeze(1).float()
        result = (torch.cos(freqs), torch.sin(freqs))

        if not _compiling:
            if (
                h_ntk_factor == self.h_ntk_factor
                and w_ntk_factor == self.w_ntk_factor
                and t_ntk_factor == self.t_ntk_factor
            ):
                key = self._cache_key(T, H, W, fps, 0, 0)
                self._cos_sin_cache[key] = result

        return result

    def generate_embeddings_with_offset(
        self,
        B_T_H_W_C: torch.Size,
        h_offset: int = 0,
        w_offset: int = 0,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate RoPE (cos, sin) with spatial offsets for tiled diffusion."""
        B, T, H, W, _ = B_T_H_W_C

        _compiling = torch.compiler.is_compiling()

        if not _compiling:
            if h_ntk_factor is None and w_ntk_factor is None and t_ntk_factor is None:
                key = self._cache_key(T, H, W, fps, h_offset, w_offset)
                cached = self._cos_sin_cache.get(key)
                if cached is not None:
                    return cached

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        assert h_offset + H <= self.max_h, (
            f"h_offset + H ({h_offset + H}) exceeds max_h ({self.max_h})"
        )
        assert w_offset + W <= self.max_w, (
            f"w_offset + W ({w_offset + W}) exceeds max_w ({self.max_w})"
        )
        half_emb_h = torch.outer(self.seq[h_offset : h_offset + H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[w_offset : w_offset + W], w_spatial_freqs)

        # Temporal dimension always starts at 0
        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert uniform_fps or B == 1 or T == 1, (
                "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
            )

            if fps is None:
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(
                    self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs
                )
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        freqs = em_T_H_W_D.flatten(0, 2).unsqueeze(1).unsqueeze(1).float()
        result = (torch.cos(freqs), torch.sin(freqs))

        if not _compiling:
            if (
                h_ntk_factor == self.h_ntk_factor
                and w_ntk_factor == self.w_ntk_factor
                and t_ntk_factor == self.t_ntk_factor
            ):
                key = self._cache_key(T, H, W, fps, h_offset, w_offset)
                self._cos_sin_cache[key] = result

        return result

    def generate_embeddings_scaled(
        self,
        B_T_H_W_C: torch.Size,
        h_scale: float = 1.0,
        w_scale: float = 1.0,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE (cos, sin) at *fractional* spatial positions (Position-Aware
        Interpolation).

        Identical to :meth:`generate_embeddings` except each spatial patch ``i``
        sits at position ``i * h_scale`` (height) / ``i * w_scale`` (width)
        instead of the integer index ``i``. Used by EasyControl to place a
        *downscaled* condition's tokens back onto the full-resolution target's
        coordinate grid: a cond grid of ``H_c`` patches representing a target
        grid of ``H_t`` patches uses ``h_scale = H_t / H_c`` so cond patch ``i``
        lands at ``i * H_t / H_c`` — spanning ``[0, H_t)`` aligned with target.

        The frequencies are computed analytically (no table lookup on the
        position value), so fractional positions are exact — this is the same
        mechanism Anima already uses for fractional temporal positions under FPS
        modulation. At ``h_scale == w_scale == 1.0`` this reduces bit-exactly to
        :meth:`generate_embeddings` (``self.seq[:H] * 1.0 == self.seq[:H]``).

        See EasyControl §3.3 (Position-Aware Interpolation), Eq. 11-12.
        """
        B, T, H, W, _ = B_T_H_W_C

        _compiling = torch.compiler.is_compiling()
        # Distinct cache key (scaled positions) so the integer-position cache is
        # never aliased.
        scaled_key = None
        if not _compiling:
            if h_ntk_factor is None and w_ntk_factor is None and t_ntk_factor is None:
                fps_val = None if fps is None else fps[:1].item()
                scaled_key = (
                    "scaled",
                    T,
                    H,
                    W,
                    fps_val,
                    float(h_scale),
                    float(w_scale),
                )
                cached = self._cos_sin_cache.get(scaled_key)
                if cached is not None:
                    return cached

        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta**self.dim_spatial_range)
        w_spatial_freqs = 1.0 / (w_theta**self.dim_spatial_range)
        temporal_freqs = 1.0 / (t_theta**self.dim_temporal_range)

        assert H <= self.max_h and W <= self.max_w, (
            f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions "
            f"(max_h={self.max_h}, max_w={self.max_w})"
        )
        # Fractional spatial positions; value range bounded by the trained grid
        # (max scaled position ≈ H_t-1 ≤ max_h), not by the slice.
        h_pos = self.seq[:H] * float(h_scale)
        w_pos = self.seq[:W] * float(w_scale)
        half_emb_h = torch.outer(h_pos, h_spatial_freqs)
        half_emb_w = torch.outer(w_pos, w_spatial_freqs)

        # Temporal positions are unscaled (cond is a single frame, T=1).
        if self.enable_fps_modulation and fps is not None:
            half_emb_t = torch.outer(
                self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs
            )
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        freqs = em_T_H_W_D.flatten(0, 2).unsqueeze(1).unsqueeze(1).float()
        result = (torch.cos(freqs), torch.sin(freqs))

        if scaled_key is not None and (
            h_ntk_factor == self.h_ntk_factor
            and w_ntk_factor == self.w_ntk_factor
            and t_ntk_factor == self.t_ntk_factor
        ):
            self._cos_sin_cache[scaled_key] = result

        return result

    def generate_embeddings_yarn(
        self,
        B_T_H_W_C: torch.Size,
        h_scale: float,
        w_scale: float,
        alpha: float,
        beta: float,
        mu: float,
        fps: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE with frequency-banded position alignment (YaRN/NTK-by-parts),
        σ-gated by SigMa-style boundary scaling.

        Per spatial frequency ``f_d`` the rotation count across the extent is
        ``r_d = N·f_d/2π``. Bands with ``r_d < alpha·mu`` (global-extent
        carriers) get the full PI stretch toward the native coordinate span
        (position ``i·scale``, as :meth:`generate_embeddings_scaled`); bands
        with ``r_d > beta·mu`` (local content-precision carriers) keep native
        integer spacing; linear ramp between. ``mu ∈ [0, 1]`` shrinks both
        thresholds (SigMa Eq. 21 boundary gating): at ``mu → 0`` every band
        clears ``beta·mu`` and the result reduces bit-exactly to native
        integer-position embeddings; at ``mu = 1`` this is the static YaRN
        alignment. Not cached — ``mu`` is continuous per training step, so a
        cache would only accrete; the build is a handful of small outer
        products.
        """
        B, T, H, W, _ = B_T_H_W_C

        h_spatial_freqs = 1.0 / (
            (10000.0 * self.h_ntk_factor) ** self.dim_spatial_range
        )
        w_spatial_freqs = 1.0 / (
            (10000.0 * self.w_ntk_factor) ** self.dim_spatial_range
        )
        temporal_freqs = 1.0 / (
            (10000.0 * self.t_ntk_factor) ** self.dim_temporal_range
        )

        def band_scale(freqs: torch.Tensor, n: int, scale: float) -> torch.Tensor:
            if mu < 1e-9:
                return torch.ones_like(freqs)
            r = n * freqs / (2 * math.pi)
            g = ((r - alpha * mu) / ((beta - alpha) * mu)).clamp(0.0, 1.0)
            return (1.0 - g) * float(scale) + g

        half_emb_h = torch.outer(
            self.seq[:H], h_spatial_freqs * band_scale(h_spatial_freqs, H, h_scale)
        )
        half_emb_w = torch.outer(
            self.seq[:W], w_spatial_freqs * band_scale(w_spatial_freqs, W, w_scale)
        )
        if self.enable_fps_modulation and fps is not None:
            half_emb_t = torch.outer(
                self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs
            )
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        freqs = em_T_H_W_D.flatten(0, 2).unsqueeze(1).unsqueeze(1).float()
        return (torch.cos(freqs), torch.sin(freqs))

    @property
    def seq_dim(self) -> int:
        return 0


def sigma_lowres_res_cond_delta(
    proj_w: torch.Tensor, s: float, timesteps_B_T: torch.Tensor
) -> torch.Tensor:
    """E25b explicit-resolution-conditioning delta on the t-embedding.

    ``s = log2(step_edge / native_edge)`` (0 on native-grid forwards) goes
    through the model's own sinusoidal embedding (``Timesteps``, dim =
    ``proj_w.shape[1]``) and the zero-init projection ``proj_w``
    ``(t_emb_dim, sinusoid_dim)``; returns ``(B, T, t_emb_dim)``. Zero-init
    proj ⇒ exactly-zero delta (the E25b identity invariant, pinned in
    tests/test_sigma_lowres.py)."""
    s_B_T = torch.full(
        timesteps_B_T.shape[:2],
        float(s),
        dtype=proj_w.dtype,
        device=proj_w.device,
    )
    return Timesteps(proj_w.shape[1])(s_B_T) @ proj_w.t()


class Timesteps(nn.Module):
    """Sinusoidal timestep features."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        assert timesteps_B_T.ndim == 2, f"Expected 2D input, got {timesteps_B_T.ndim}"
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 0.0)

        emb = torch.exp(exponent)
        emb = timesteps[:, None] * emb[None, :]

        sin_emb = torch.sin(emb)
        cos_emb = torch.cos(emb)
        emb = torch.cat([cos_emb, sin_emb], dim=-1)

        return emb.to(dtype=in_dtype).reshape(
            timesteps_B_T.shape[0], timesteps_B_T.shape[1], -1
        )


class TimestepEmbedding(nn.Module):
    """Projects timestep features to model dimension, with optional AdaLN-LoRA."""

    def __init__(
        self, in_features: int, out_features: int, use_adaln_lora: bool = False
    ):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.linear_1 = nn.Linear(in_features, out_features, bias=not use_adaln_lora)
        self.activation = nn.SiLU()
        self.use_adaln_lora = use_adaln_lora
        if use_adaln_lora:
            self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)
        else:
            self.linear_2 = nn.Linear(out_features, out_features, bias=False)

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.in_dim)
        torch.nn.init.trunc_normal_(
            self.linear_1.weight, std=std, a=-3 * std, b=3 * std
        )
        std = 1.0 / math.sqrt(self.out_dim)
        torch.nn.init.trunc_normal_(
            self.linear_2.weight, std=std, a=-3 * std, b=3 * std
        )

    def forward(
        self, sample: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        emb = self.linear_1(sample)
        emb = self.activation(emb)
        emb = self.linear_2(emb)

        if self.use_adaln_lora:
            adaln_lora_B_T_3D = emb
            emb_B_T_D = sample
        else:
            adaln_lora_B_T_3D = None
            emb_B_T_D = emb

        return emb_B_T_D, adaln_lora_B_T_3D


# Commented out Fourier Features (not used in Anima). Kept for reference.
# class FourierFeatures(nn.Module):
#     """Fourier feature transform: [B] -> [B, D]."""

#     def __init__(self, num_channels: int, bandwidth: int = 1, normalize: bool = False):
#         super().__init__()
#         self.register_buffer("freqs", 2 * np.pi * bandwidth * torch.randn(num_channels), persistent=True)
#         self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels), persistent=True)
#         self.gain = np.sqrt(2) if normalize else 1
#         self.bandwidth = bandwidth
#         self.num_channels = num_channels
#         self.reset_parameters()

#     def reset_parameters(self) -> None:
#         generator = torch.Generator()
#         generator.manual_seed(0)
#         self.freqs = 2 * np.pi * self.bandwidth * torch.randn(self.num_channels, generator=generator).to(self.freqs.device)
#         self.phases = 2 * np.pi * torch.rand(self.num_channels, generator=generator).to(self.freqs.device)

#     def forward(self, x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
#         in_dtype = x.dtype
#         x = x.to(torch.float32).ger(self.freqs.to(torch.float32)).add(self.phases.to(torch.float32))
#         x = x.cos().mul(self.gain * gain).to(in_dtype)
#         return x


class PatchEmbed(nn.Module):
    """Patch embedding: (B, C, T, H, W) -> (B, T', H', W', D)"""

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 17,
        out_channels: int = 2048,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels
                * spatial_patch_size
                * spatial_patch_size
                * temporal_patch_size,
                out_channels,
                bias=False,
            ),
        )
        self.dim = (
            in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size
        )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0, (
            f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        )
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x


class FinalLayer(nn.Module):
    """Final layer with AdaLN modulation + unpatchify."""

    def __init__(
        self,
        hidden_size: int,
        spatial_patch_size: int,
        temporal_patch_size: int,
        out_channels: int,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size,
            spatial_patch_size
            * spatial_patch_size
            * temporal_patch_size
            * out_channels,
            bias=False,
        )
        self.hidden_size = hidden_size
        self.n_adaln_chunks = 2
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        if use_adaln_lora:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=False),
                nn.Linear(
                    adaln_lora_dim, self.n_adaln_chunks * hidden_size, bias=False
                ),
            )
        else:
            self.adaln_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, self.n_adaln_chunks * hidden_size, bias=False),
            )

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        torch.nn.init.trunc_normal_(self.linear.weight, std=std, a=-3 * std, b=3 * std)
        if self.use_adaln_lora:
            torch.nn.init.trunc_normal_(
                self.adaln_modulation[1].weight, std=std, a=-3 * std, b=3 * std
            )
            torch.nn.init.zeros_(self.adaln_modulation[2].weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation[1].weight)

        self.layer_norm.reset_parameters()

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ):
        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_B_T_D, scale_B_T_D = (
                self.adaln_modulation(emb_B_T_D)
                + adaln_lora_B_T_3D[:, :, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
        else:
            shift_B_T_D, scale_B_T_D = self.adaln_modulation(emb_B_T_D).chunk(2, dim=-1)

        shift_B_T_1_1_D = shift_B_T_D[:, :, None, None, :]
        scale_B_T_1_1_D = scale_B_T_D[:, :, None, None, :]

        x_B_T_H_W_D = (
            self.layer_norm(x_B_T_H_W_D) * (1 + scale_B_T_1_1_D) + shift_B_T_1_1_D
        )
        x_B_T_H_W_O = self.linear(x_B_T_H_W_D)
        return x_B_T_H_W_O


class Block(nn.Module):
    """Transformer block with self-attention + cross-attention + MLP, each modulated by AdaLN.

    Each sublayer: x = x + gate * sublayer(norm(x) * (1 + scale) + shift)
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(
            x_dim, elementwise_affine=False, eps=1e-6
        )
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
        )

        self.layer_norm_cross_attn = nn.LayerNorm(
            x_dim, elementwise_affine=False, eps=1e-6
        )
        self.cross_attn = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
        )

        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_fused_down = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, 3 * adaln_lora_dim, bias=False),
            )
            self.adaln_up_self_attn = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
            self.adaln_up_cross_attn = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
            self.adaln_up_mlp = nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False)
        else:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False)
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False)
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False)
            )

        self.gradient_checkpointing = False
        self.unsloth_offload_checkpointing = False

        # Inference-side cross-attn residual gain (frontload_text_boost arm b).
        # Non-persistent buffer read inside the compiled _forward so the sampler
        # can retune it per step via fill_() without a recompile — same pattern
        # as the _mod_guidance_* buffers. 1.0 = exact identity.
        self.register_buffer("_xattn_gain", torch.ones(()), persistent=False)
        # Norm-matched variant of the gain (frontload_text_boost arm g): when
        # True, the post-cross-attn hidden state is rescaled back toward the
        # norm it would have had at gain 1.0 — the boost rotates the state
        # toward the cross-attn residual without leaving the norm shell the
        # next block was trained on. pertoken=False matches the per-image
        # MEAN token norm instead (keeps the token-norm distribution — its
        # peaks carry local contrast; full per-token matching flattens them
        # to a grey tone). frac ρ applies scale**ρ (partial correction,
        # ρ=1 full shell, ρ=0 raw boost). Plain Python attrs (static dynamo
        # guards, one graph variant per combo). False = exact identity.
        self._xattn_renorm = False
        self._xattn_renorm_pertoken = True
        self._xattn_renorm_frac = 1.0

    def enable_gradient_checkpointing(self, unsloth_offload: bool = False):
        self.gradient_checkpointing = True
        self.unsloth_offload_checkpointing = unsloth_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def reset_parameters(self) -> None:
        self.layer_norm_self_attn.reset_parameters()
        self.layer_norm_cross_attn.reset_parameters()
        self.layer_norm_mlp.reset_parameters()

        if self.use_adaln_lora:
            std = 1.0 / math.sqrt(self.x_dim)
            torch.nn.init.trunc_normal_(
                self.adaln_fused_down[1].weight,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
            torch.nn.init.zeros_(self.adaln_up_self_attn.weight)
            torch.nn.init.zeros_(self.adaln_up_cross_attn.weight)
            torch.nn.init.zeros_(self.adaln_up_mlp.weight)
        else:
            torch.nn.init.zeros_(self.adaln_modulation_self_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_cross_attn[1].weight)
            torch.nn.init.zeros_(self.adaln_modulation_mlp[1].weight)

    def init_weights(self) -> None:
        self.reset_parameters()
        self.self_attn.init_weights()
        self.cross_attn.init_weights()
        self.mlp.init_weights()

    def _forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention_dispatch.AttentionParams,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_adaln_lora:
            fused_down = self.adaln_fused_down(emb_B_T_D)
            down_self, down_cross, down_mlp = fused_down.chunk(3, dim=-1)
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_up_self_attn(down_self) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_up_cross_attn(down_cross) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_up_mlp(down_mlp) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)
            )

        shift_self_attn_B_T_1_1_D = shift_self_attn_B_T_D[:, :, None, None, :]
        scale_self_attn_B_T_1_1_D = scale_self_attn_B_T_D[:, :, None, None, :]
        gate_self_attn_B_T_1_1_D = gate_self_attn_B_T_D[:, :, None, None, :]

        shift_cross_attn_B_T_1_1_D = shift_cross_attn_B_T_D[:, :, None, None, :]
        scale_cross_attn_B_T_1_1_D = scale_cross_attn_B_T_D[:, :, None, None, :]
        gate_cross_attn_B_T_1_1_D = gate_cross_attn_B_T_D[:, :, None, None, :]

        shift_mlp_B_T_1_1_D = shift_mlp_B_T_D[:, :, None, None, :]
        scale_mlp_B_T_1_1_D = scale_mlp_B_T_D[:, :, None, None, :]
        gate_mlp_B_T_1_1_D = gate_mlp_B_T_D[:, :, None, None, :]

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _adaln_fn(_x, _norm_layer, _scale, _shift):
            return _norm_layer(_x) * (1 + _scale) + _shift

        normalized_x = _adaln_fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )
        x_flat = normalized_x.flatten(1, 3)
        result = self.self_attn(
            x_flat,
            attn_params,
            x_flat,
            rope_cos_sin=rope_cos_sin,
        ).unflatten(1, (T, H, W))
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result

        normalized_x = _adaln_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
        )
        result = self.cross_attn(
            normalized_x.flatten(1, 3),
            attn_params,
            crossattn_emb,
            rope_cos_sin=rope_cos_sin,
        ).unflatten(1, (T, H, W))
        x_new = result * (gate_cross_attn_B_T_1_1_D * self._xattn_gain) + x_B_T_H_W_D
        if self._xattn_renorm:
            plain = result * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D
            norm_plain = plain.float().norm(dim=-1, keepdim=True)
            norm_new = x_new.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            if not self._xattn_renorm_pertoken:
                norm_plain = norm_plain.mean(dim=(1, 2, 3), keepdim=True)
                norm_new = norm_new.mean(dim=(1, 2, 3), keepdim=True)
            scale = norm_plain / norm_new
            if self._xattn_renorm_frac != 1.0:
                scale = scale**self._xattn_renorm_frac
            x_new = x_new * scale.to(x_new.dtype)
        x_B_T_H_W_D = x_new

        normalized_x = _adaln_fn(
            x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp_B_T_1_1_D, shift_mlp_B_T_1_1_D
        )
        result = self.mlp(normalized_x)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result

        return x_B_T_H_W_D

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention_dispatch.AttentionParams,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if torch.is_grad_enabled() and self.training and self.gradient_checkpointing:
            if self.unsloth_offload_checkpointing:
                return unsloth_checkpoint(
                    self._forward,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    rope_cos_sin,
                    adaln_lora_B_T_3D,
                )
            else:
                return torch_checkpoint(
                    self._forward,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    rope_cos_sin,
                    adaln_lora_B_T_3D,
                    use_reentrant=False,
                )
        else:
            return self._forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                rope_cos_sin,
                adaln_lora_B_T_3D,
            )


class Anima(nn.Module):
    """Cosmos-Predict2 DiT model for image/video generation.

    28 transformer blocks with AdaLN-LoRA modulation, 3D RoPE, and optional LLM Adapter.
    """

    LATENT_CHANNELS = 16

    def __init__(
        self,
        max_img_h: int,
        max_img_w: int,
        max_frames: int,
        in_channels: int,
        out_channels: int,
        patch_spatial: int,
        patch_temporal: int,
        concat_padding_mask: bool = True,
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        crossattn_emb_channels: int = 1024,
        pos_emb_learnable: bool = True,
        min_fps: int = 1,
        max_fps: int = 30,
        use_adaln_lora: bool = True,
        adaln_lora_dim: int = 256,
        rope_h_extrapolation_ratio: float = 4.0,
        rope_w_extrapolation_ratio: float = 4.0,
        rope_t_extrapolation_ratio: float = 1.0,
        rope_enable_fps_modulation: bool = False,
        use_llm_adapter: bool = True,
        attn_mode: str = "torch",
        attn_softmax_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.max_img_h = max_img_h
        self.max_img_w = max_img_w
        self.max_frames = max_frames
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_spatial = patch_spatial
        self.patch_temporal = patch_temporal
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.model_channels = model_channels
        self.concat_padding_mask = concat_padding_mask
        self.pos_emb_learnable = pos_emb_learnable
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.rope_h_extrapolation_ratio = rope_h_extrapolation_ratio
        self.rope_w_extrapolation_ratio = rope_w_extrapolation_ratio
        self.rope_t_extrapolation_ratio = rope_t_extrapolation_ratio
        self.rope_enable_fps_modulation = rope_enable_fps_modulation
        self.use_llm_adapter = use_llm_adapter

        self.attn_mode = attn_mode
        self.attn_softmax_scale = attn_softmax_scale

        self.blocks_to_swap = None
        self.offloader: Optional[custom_offloading_utils.ModelOffloader] = None
        # Stashed blocks_to_swap while paused (e.g. during eval). None = not paused.
        self._paused_blocks_to_swap: Optional[int] = None

        # Native-shape flattening for torch.compile, flipped True by compile_blocks():
        # the forward flattens each bucket to fake-5D (B,1,seq_len,1,D) so dynamo keys
        # the block graph on token count alone, not H/W separately. Eager forwards
        # leave it False and skip the reshape (bit-exact to the flattened path).
        self._native_flatten: bool = False

        # Dynamic-seq compile: when True, marks the seq-length axis dynamic to
        # collapse the per-token-count graphs to one; _dynamic_seq_range is the
        # (min, max) token-count bound. Both inert on the static/eager paths.
        self._dynamic_seq: bool = False
        self._dynamic_seq_range: Optional[tuple] = None

        self.build_patch_embed()
        self.build_pos_embed()
        self.use_adaln_lora = use_adaln_lora
        self.adaln_lora_dim = adaln_lora_dim
        self.t_embedder = nn.Sequential(
            Timesteps(model_channels),
            TimestepEmbedding(
                model_channels, model_channels, use_adaln_lora=use_adaln_lora
            ),
        )

        if self.use_llm_adapter:
            self.llm_adapter = LLMAdapter(
                source_dim=1024,
                target_dim=1024,
                model_dim=1024,
                num_layers=6,
                self_attn=True,
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_layer = FinalLayer(
            hidden_size=self.model_channels,
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            out_channels=self.out_channels,
            use_adaln_lora=self.use_adaln_lora,
            adaln_lora_dim=self.adaln_lora_dim,
        )

        self.t_embedding_norm = RMSNorm(model_channels, eps=1e-6)

        # Mod-guidance: project pooled crossattn_emb into modulation space.
        # Zero-init → no-op before distillation training.
        self.pooled_text_proj = nn.Sequential(
            nn.Linear(crossattn_emb_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        # σ-FiLM (experimental): timestep-condition the mod head's hidden so the
        # text push scales per σ (the plain head is σ-flat, ‖ΔS‖/‖ΔT‖ collapses at
        # high σ). Sibling of pooled_text_proj (not folded in) to keep the [...]
        # indexing valid. Gated by enable_pooled_text_sigma_film — off ⇒ bit-exact
        # to the plain head; zero-init ⇒ identity FiLM. See _archive/bench/mod_guidance.
        self.pooled_text_sigma_film = nn.Linear(model_channels, 2 * model_channels)
        self.enable_pooled_text_sigma_film = False

        # Whether the per-forward pooled_text_proj path runs. Default off: the base
        # ckpt re-zeroes these weights, so the proj is a no-op + pure overhead.
        # Flipped True only where active (load_pooled_text_proj / distill-mod). A
        # plain bool set once at load, so it guards once under compile (no churn).
        self.enable_pooled_text_modulation = False

        # Mod-guidance runtime state as non-persistent buffers (zeros = off).
        # Registered unconditionally so the forward does unconditional arithmetic
        # without a Python branch (branches guard-fire under torch.compile per
        # bucket/block). Setters in library/inference/mod_guidance.py.
        self.register_buffer(
            "_mod_guidance_delta",
            torch.zeros(1, model_channels),
            persistent=False,
        )
        self.register_buffer(
            "_mod_guidance_schedule",
            torch.zeros(num_blocks),
            persistent=False,
        )
        self.register_buffer(
            "_mod_guidance_final_w",
            torch.zeros(()),
            persistent=False,
        )

        # DAVE — DC Attenuation for diVersity Enhancement (training-free). Per-block
        # edit `ĥ = α·μ + (h−μ)` via post-forward hooks (library/inference/corrections/dave.py);
        # these buffers carry the hooks' runtime state. _dave_atten[l] = (1−α_l)
        # (zeros ⇒ no-op). The edit is σ-gated to [lo, hi]; _dave_cur_sigma is
        # restamped from the timestep every forward so the block hooks (which never
        # see the timestep) can gate on σ. enable_dave: plain bool, set by setup_dave.
        self.enable_dave = False
        self.register_buffer("_dave_atten", torch.zeros(num_blocks), persistent=False)
        self.register_buffer("_dave_sigma_lo", torch.zeros(()), persistent=False)
        self.register_buffer("_dave_sigma_hi", torch.ones(()), persistent=False)
        self.register_buffer("_dave_cur_sigma", torch.zeros(()), persistent=False)

        self.init_weights()

    def _pooled_text_delta(
        self,
        pooled_text: torch.Tensor,
        t_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mod-guidance delta (B, D) added to the AdaLN time embedding.

        With σ-FiLM enabled, the head's hidden activation is FiLM-modulated by
        the (normed) time embedding so the text push becomes σ-dependent — this
        is what lets ΔS track the teacher's σ-growing text response instead of
        staying σ-flat. ``t_embedding`` is required for that path; callers
        without one (the legacy single-delta inference bake) fall back to the
        σ-flat projection.
        """
        lin_in, _act, lin_out = self.pooled_text_proj
        use_film = self.enable_pooled_text_sigma_film and t_embedding is not None
        t = None
        if use_film:
            t = t_embedding[:, 0, :] if t_embedding.ndim == 3 else t_embedding
        # Single source of truth for the projection math (shared verbatim with the
        # ComfyUI node via library/inference/corrections/mod_guidance_core.py).
        return project_pooled(
            pooled_text,
            lin_in.weight,
            lin_in.bias,
            lin_out.weight,
            lin_out.bias,
            film_w=self.pooled_text_sigma_film.weight if use_film else None,
            film_b=self.pooled_text_sigma_film.bias if use_film else None,
            t_emb=t,
        )

    def reset_mod_guidance(self) -> None:
        """Disable modulation guidance by zeroing the runtime buffers."""
        self._mod_guidance_delta.zero_()
        self._mod_guidance_schedule.zero_()
        self._mod_guidance_final_w.zero_()

    def reset_dave(self) -> None:
        """Disable DAVE: zero the attenuation factors and the enable flag.

        Hooks are removed by their owner (``setup_dave`` returns a handle); this
        just neutralizes the buffers so any still-attached hook is a no-op.
        """
        self.enable_dave = False
        self._dave_atten.zero_()
        self._dave_cur_sigma.zero_()

    def init_weights(self) -> None:
        self.x_embedder.init_weights()
        self.pos_embedder.reset_parameters()
        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()
        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()
        # Zero-init pooled_text_proj output layer so it's a no-op at init
        nn.init.zeros_(self.pooled_text_proj[-1].weight)
        nn.init.zeros_(self.pooled_text_proj[-1].bias)
        # Zero-init σ-FiLM generator → identity (scale=shift=0) at init.
        nn.init.zeros_(self.pooled_text_sigma_film.weight)
        nn.init.zeros_(self.pooled_text_sigma_film.bias)

    def enable_gradient_checkpointing(self, unsloth_offload: bool = False):
        if not self.training:
            # Block.forward gates checkpointing on `self.training` (see models.py
            # ~1208), so enabling it on a module in eval mode — e.g. one built via
            # the inference loader `load_dit_model` — is silently inert: you still
            # OOM with no signal. Warn once. (issues.md DX1)
            logger.warning(
                "enable_gradient_checkpointing() called but module is in eval mode "
                "— checkpointing is inert until you call .train() (Block.forward "
                "gates on self.training)."
            )
        for block in self.blocks:
            block.enable_gradient_checkpointing(unsloth_offload=unsloth_offload)

    def disable_gradient_checkpointing(self):
        for block in self.blocks:
            block.disable_gradient_checkpointing()

    def compile_blocks(
        self,
        backend: str = "inductor",
        mode: Optional[str] = None,
        n_token_families: Optional[int] = None,
        dynamic_seq: bool = False,
        seq_range: Optional[tuple] = None,
    ):
        """Enable native-shape flattening and torch.compile each block's _forward.

        Two coupled effects, both owned by this one call:

        1. Sets ``self._native_flatten = True`` so the forward flattens every
           bucket's patch sequence to a fake-5D ``(B, 1, seq_len, 1, D)`` shape.
           This is what keys the block graph on token count alone — the shipped
           ``CONSTANT_TOKEN_BUCKETS`` collapses to 2 token-count families (the
           4032 and 4200 groups) — instead of guarding H and W separately (one
           graph per resolution, ~24 buckets). Native shapes, no padding, so
           flash self-attention sees no padded tokens (bit-exact to the eager
           5D path; the gap=0 control of the retired pad-leak probe verified it).

        2. Compiles ``_forward`` (the actual attention/MLP computation) rather
           than ``forward`` (the checkpointing wrapper). This is critical because
           unsloth_checkpoint has @torch._disable_dynamo, which causes an
           immediate graph break if forward itself is compiled — dynamo compiles
           nothing useful but still checks shape guards, causing recompile storms.

        Also raises the dynamo cache-size budget to fit those token-count
        families. ``2 * n + 8``: the ``2 *`` covers fwd+bwd sharing the one
        ``_forward`` bytecode, the ``+ 8`` covers requires_grad / stride
        specializations (the live path traces ~5 graphs, not 2). ``max()`` is
        load-bearing — a caller that knows it has *more* distinct shapes (e.g.
        the multi-resolution SPD distill) raises the limit higher beforehand and
        this must not clobber it back down. This call's own budget only ever
        covers the two full-res families.

        ``mode`` maps to torch.compile's inductor preset (e.g. ``reduce-overhead``
        to enable per-block CUDAGraphs). ``None`` leaves it unset (inductor default).

        ``dynamic_seq`` collapses the N-graph compile cascade (each graph loads
        its own inductor kernel module + flash/cuBLAS workspaces into the CUDA
        context — the ``nvidia-smi``-visible cold-compile VRAM transient) down to
        a single block graph. Mechanism: keep ``dynamic=False`` (force static
        specialization by default) and let ``_run_blocks`` annotate *only* the
        seq-length axis via ``torch._dynamo.mark_dynamic``. Under
        ``native_flatten`` the in-block latent is ``(B, 1, seq_len, 1, D)`` (T=1,
        W=1), so ``seq_len`` is the *only* varying axis; B / D / head-dim / text-len
        stay statically specialized. This is deliberately tighter than blanket
        ``dynamic=True`` (which marks every dim of every input symbolic and can
        over-generalize into worse kernels). ``seq_range`` bounds the symbolic
        axis (min/max token count over the active tiers) so inductor guards
        against a real range, not ``[2, ∞)``; ``None`` derives it from the
        canonical 1024 table (4032/4200). Off by default; the static path stays
        the trusted one until benched (graph count via ``TORCH_LOGS=recompiles``,
        peak via ``mem_get_info``, step time, bit-exactness vs eager). See
        [[project_compile_context_vram_climb]].
        """
        self._native_flatten = True

        # Local import avoids a circular import (buckets does not import models).
        from library.datasets.buckets import token_count_families
        from library.runtime.dynamo import pin_dynamo_limit

        # Number of distinct token-count families (== compiled block graphs).
        # Defaults to the canonical 1024 tier (2: 4032/4200); callers pass the count
        # derived from the buckets the dataset actually populated
        # (train.py::_derive_token_budget).
        if n_token_families is not None:
            n = n_token_families
        else:
            n = token_count_families((1024,))
        # pin_dynamo_limit (not a plain config.recompile_limit=…): the budget is a
        # ContextVar that reverts to the default 8 in the backward compile context;
        # a wide multi-scale run would silently spill to eager without pinning .default.
        limit = pin_dynamo_limit("recompile_limit", 2 * n + 8)

        # dynamic_seq compiles static and marks only the seq axis dynamic (not
        # torch.compile(dynamic=True)). Derive the (min,max) seq bound: passed-in
        # seq_range (multi-tier) or the canonical 1024 tier's band (4032, 4200).
        self._dynamic_seq = dynamic_seq
        if dynamic_seq:
            if seq_range is not None:
                self._dynamic_seq_range = (int(seq_range[0]), int(seq_range[1]))
            else:
                from library.datasets.buckets import token_count_range

                self._dynamic_seq_range = token_count_range((1024,))
            # Inductor's mix-order-reduction fusion (torch 2.12, default-on) is
            # incompatible with the strict seq marks: its profitability check
            # calls guard_or_true(Ge(nrow, 4096)) where nrow is the symbolic seq
            # axis (it fires on backward graphs that pair a seq-axis reduction
            # with an elementwise grad — e.g. any LoRA on a broadcast-consumed
            # Linear like adaln_up, whose shift/scale/gate grads reduce over
            # seq). The recorded guard (either branch: Ge(seq, 4096) or its
            # negation seq <= 4095, per the first-traced hint) contradicts any
            # mark range straddling 4096 → ConstraintViolationError at guard
            # build. MUST be pinned via pin_inductor_flag, not plain assignment:
            # inductor config overrides are thread-local ContextVars, and the
            # grad-enabled step-0 compile (grad-ckpt recompute / AOT backward
            # path) schedules in a context where a plain override is absent and
            # the read falls back to the env-derived default True — the exact
            # regression that hit v1.14.0 (adaln default-on) users.
            import torch._inductor.config as _inductor_config

            if _inductor_config.triton.mix_order_reduction:
                from library.runtime.dynamo import pin_inductor_flag

                pin_inductor_flag("triton.mix_order_reduction", False)
                print(
                    "Anima: inductor mix_order_reduction disabled — default pinned "
                    "(hint-derived 4096-boundary guard breaks strict dynamic-seq marks)"
                )

        compile_kwargs = {"backend": backend, "dynamic": False}
        if mode is not None:
            compile_kwargs["mode"] = mode
        for block in self.blocks:
            compiled_inner = torch.compile(block._forward, **compile_kwargs)
            if dynamic_seq:
                # Mark the seq axis dynamic INSIDE the checkpointed callable so the
                # marks re-apply on the grad-checkpoint backward recompute, not just
                # forward. See _make_dynamic_seq_forward.
                lo, hi = self._dynamic_seq_range
                block._forward = _make_dynamic_seq_forward(compiled_inner, lo, hi)
            else:
                block._forward = compiled_inner
        graph_mode = (
            f"dynamic-seq mark_dynamic seq∈{self._dynamic_seq_range} (1 graph)"
            if dynamic_seq
            else f"static ({n} graphs)"
        )
        print(
            f"Anima: native_flatten on, {n} token-count families, {graph_mode} "
            f"(recompile_limit={limit}); compiled "
            f"{len(self.blocks)} block._forward with backend={backend}, mode={mode}"
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def build_patch_embed(self) -> None:
        in_channels = (
            self.in_channels + 1 if self.concat_padding_mask else self.in_channels
        )
        self.x_embedder = PatchEmbed(
            spatial_patch_size=self.patch_spatial,
            temporal_patch_size=self.patch_temporal,
            in_channels=in_channels,
            out_channels=self.model_channels,
        )

    def build_pos_embed(self) -> None:
        self.pos_embedder = VideoRopePosition3DEmb(
            model_channels=self.model_channels,
            len_h=self.max_img_h // self.patch_spatial,
            len_w=self.max_img_w // self.patch_spatial,
            len_t=self.max_frames // self.patch_temporal,
            max_fps=self.max_fps,
            min_fps=self.min_fps,
            is_learnable=self.pos_emb_learnable,
            head_dim=self.model_channels // self.num_heads,
            h_extrapolation_ratio=self.rope_h_extrapolation_ratio,
            w_extrapolation_ratio=self.rope_w_extrapolation_ratio,
            t_extrapolation_ratio=self.rope_t_extrapolation_ratio,
            enable_fps_modulation=self.rope_enable_fps_modulation,
        )

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        h_offset: int = 0,
        w_offset: int = 0,
    ) -> Tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.concat_padding_mask:
            if padding_mask is None:
                raise ValueError(
                    "padding_mask must be provided when concat_padding_mask is True"
                )
            if padding_mask.ndim != 4:
                raise ValueError(
                    f"padding_mask must be 4D (B, 1, H, W), got shape {tuple(padding_mask.shape)}"
                )
            if padding_mask.shape[-2:] != x_B_C_T_H_W.shape[-2:]:
                from torchvision import transforms

                padding_mask = transforms.functional.resize(
                    padding_mask,
                    list(x_B_C_T_H_W.shape[-2:]),
                    interpolation=transforms.InterpolationMode.NEAREST,
                )

            # (B, 1, H, W) -> (B, 1, T, H, W) without materializing a repeated tensor
            padding_mask_B_1_T_H_W = padding_mask.unsqueeze(2).expand(
                -1, -1, x_B_C_T_H_W.shape[2], -1, -1
            )
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, padding_mask_B_1_T_H_W], dim=1)
        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        # sigma_lowres yarnsig: a demoted train step sets this 5-tuple
        # (h_scale, w_scale, alpha, beta, mu) for the duration of its forward
        # (train.py clears it in a finally); rope is built OUTSIDE the compiled
        # block graph, so the blocks just see different cos/sin inputs at the
        # same token count.
        yarn = getattr(self, "_sigma_lowres_yarn", None)
        if yarn is not None:
            rope_cos_sin = self.pos_embedder.generate_embeddings_yarn(
                x_B_T_H_W_D.shape, *yarn, fps=fps
            )
        elif h_offset != 0 or w_offset != 0:
            rope_cos_sin = self.pos_embedder.generate_embeddings_with_offset(
                x_B_T_H_W_D.shape, h_offset=h_offset, w_offset=w_offset, fps=fps
            )
        else:
            rope_cos_sin = self.pos_embedder(x_B_T_H_W_D, fps=fps)
        return x_B_T_H_W_D, rope_cos_sin

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        B, T, H, W, M = x_B_T_H_W_M.shape
        p1 = self.patch_spatial
        p2 = self.patch_spatial
        pt = self.patch_temporal
        C = M // (p1 * p2 * pt)
        # (B,T,H,W, p1*p2*pt*C) → (B,T,H,W, p1,p2,pt,C) → (B,C, T,pt, H,p1, W,p2)
        #                                                    → (B,C, T*pt, H*p1, W*p2)
        x_B_C_Tt_Hp_Wp = (
            x_B_T_H_W_M.unflatten(-1, (p1, p2, pt, C))
            .permute(0, 7, 1, 6, 2, 4, 3, 5)
            .reshape(B, C, T * pt, H * p1, W * p2)
        )
        return x_B_C_Tt_Hp_Wp

    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap = num_blocks

        assert self.blocks_to_swap <= self.num_blocks - 2, (
            f"Cannot swap more than {self.num_blocks - 2} blocks. Requested: {self.blocks_to_swap} blocks."
        )

        self.offloader = custom_offloading_utils.ModelOffloader(
            self.blocks, self.blocks_to_swap, device
        )
        logger.info(
            f"Anima: Block swap enabled. Swapping {num_blocks} blocks, total blocks: {self.num_blocks}, device: {device}."
        )

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None  # Use None to skip .to() on blocks (consistent with flux_models.py)

        self.to(device)

        if self.blocks_to_swap:
            self.blocks = save_blocks

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.set_forward_only(True)
        self.prepare_block_swap_before_forward()
        print("Anima: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.set_forward_only(False)
        self.prepare_block_swap_before_forward()
        print("Anima: Block swap set to forward and backward.")

    def pause_block_swap(self) -> bool:
        # Drains the offloader, pulls parked blocks back onto the forward device,
        # and zeroes blocks_to_swap so the _run_blocks swap path short-circuits.
        # For no_grad eval where the full DiT fits on GPU and streaming is overhead.
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return False
        if self._paused_blocks_to_swap is not None:
            return False
        for block_idx in list(self.offloader.futures.keys()):
            self.offloader._wait_blocks_move(block_idx)
        for b in self.blocks:
            weighs_to_device(b, self.offloader.device)
        if self.offloader.cuda_available:
            torch.cuda.synchronize()
        self._paused_blocks_to_swap = self.blocks_to_swap
        self.blocks_to_swap = 0
        return True

    def resume_block_swap(self) -> bool:
        # Inverse of pause_block_swap: restores blocks_to_swap and re-parks the
        # tail blocks' weights on CPU.
        if self._paused_blocks_to_swap is None:
            return False
        self.blocks_to_swap = self._paused_blocks_to_swap
        self._paused_blocks_to_swap = None
        self.prepare_block_swap_before_forward()
        return True

    def prepare_block_swap_before_forward(self, free_cache: bool = True):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(
            self.blocks, free_cache=free_cache
        )

    def _run_blocks(
        self,
        x_padded: torch.Tensor,
        t_embedding_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params,
        capture_blocks: Optional[set] = None,
        feature_sink: Optional[dict] = None,
        stop_after_block: Optional[int] = None,
        **block_kwargs,
    ) -> torch.Tensor:
        """The block loop — the per-block compiled hot path (see compile_blocks).

        Inputs from the eager pre-blocks region:
        - ``x_padded``: ``(B, 1, seq_len, 1, D)`` when native-flattened, else the
          plain ``(B, T, H, W, D)`` grid (eager forwards skip the flatten)
        - ``t_embedding_B_T_D``: ``(B, 1, D)``
        - ``crossattn_emb``: ``(B, max_text_len, D)`` (padded to max_length)
        - ``attn_params``: attention params (no self-attn mask in native mode)
        - ``block_kwargs["rope_cos_sin"]``: each ``(seq_len, 1, 1, D_head)``
        - ``block_kwargs["adaln_lora_B_T_3D"]``: ``(B, 1, 3, D)``

        Mod-guidance is applied via buffers on ``self`` (zero = off) so the
        per-block ``t_emb`` arithmetic is unconditional. No Python branches.

        Feature tap (opt-in, all defaults off → bit-exact no-op): when
        ``capture_blocks`` is given, each listed block's output is stored into
        ``feature_sink`` (keyed by block index). ``stop_after_block`` breaks the
        loop right after that index — so a feature-only forward that taps block
        ``k`` only runs ``blocks[0..k]`` and retains just their activations for
        backward (the memory win that makes the Turbo GAN gen-forward affordable;
        see ``forward_mini_train_dit``'s ``return_features_early``). The capture
        sits at block ``__call__`` granularity — eager, OUTSIDE the compiled
        ``_forward`` — so it is compile-safe.
        """
        # Normalize requires_grad once at stack entry (block 0 frozen patch_embed
        # output is False, blocks 1+ are True); a mismatch would fragment guards if
        # the loop were traced per-block. No-op under torch.no_grad().
        x = x_padded.requires_grad_()

        # compile_dynamic_seq marks the seq axis dynamic INSIDE each compiled
        # block._forward (via _make_dynamic_seq_forward), not here, so the marks
        # re-apply on the grad-checkpoint backward recompute — else x's mark is
        # stripped while the RoPE tuple's survives → ConstraintViolationError.

        for block_idx, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(block_idx)

            # Unconditional: zero buffers collapse to identity when guidance is off;
            # avoids a data-dependent branch inside the compiled frame.
            t_emb_block = t_embedding_B_T_D + (
                self._mod_guidance_schedule[block_idx] * self._mod_guidance_delta
            ).unsqueeze(1)

            x = block(
                x,
                t_emb_block,
                crossattn_emb,
                attn_params,
                **block_kwargs,
            )

            if capture_blocks is not None and block_idx in capture_blocks:
                feature_sink[block_idx] = x

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.blocks, block_idx)

            if stop_after_block is not None and block_idx == stop_after_block:
                break
        return x

    def forward_mini_train_dit(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        t5_input_ids: Optional[torch.Tensor] = None,
        t5_attn_mask: Optional[torch.Tensor] = None,
        crossattn_seqlens: Optional[torch.Tensor] = None,
        h_offset: int = 0,
        w_offset: int = 0,
        pooled_text_override: Optional[torch.Tensor] = None,
        skip_pooled_text_proj: bool = False,
        return_block_features: Optional[set] = None,
        return_features_early: bool = False,
        token_merger=None,
    ) -> torch.Tensor:
        """
        Args:
            x_B_C_T_H_W: (B, C, T, H, W) noisy latents
            timesteps_B_T: (B,) or (B, T) timesteps
            crossattn_emb: (B, N, D) cross-attention embeddings (or raw Qwen3 prompt_embeds if t5_input_ids provided)
            fps: Optional frames per second
            padding_mask: Optional padding mask
            source_attention_mask: Optional attention mask for Qwen3 embeddings (used with LLM adapter)
            t5_input_ids: Optional T5 token IDs (triggers LLM adapter when provided)
            t5_attn_mask: Optional T5 attention mask
            crossattn_seqlens: Optional per-sample text token counts [B] for flex cross-attention masking
            h_offset: Height offset in patched space for tiled diffusion RoPE
            w_offset: Width offset in patched space for tiled diffusion RoPE
            pooled_text_override: Optional pre-computed pooled text (B, 1024) for modulation guidance.
                Use to decouple modulation from prefix/postfix tokens in crossattn_emb.
            return_block_features: Optional set of block indices to tap. When given,
                each listed block's raw output (the post-block hidden state, in the
                native-flatten ``(B, 1, L, 1, D)`` layout under ``compile_blocks`` or
                the eager ``(B, T, H, W, D)`` grid otherwise) is captured into a dict.
            return_features_early: When True (requires ``return_block_features``),
                the block loop stops right after the deepest tapped block and the
                method returns the captured-feature dict directly — skipping the
                remaining blocks, ``final_layer`` and ``unpatchify``. This is the
                feature-tap fast path: a forward that only needs a mid-stack feature
                runs (and, when grad-bearing, retains activations for) just the
                blocks up to the tap. With ``return_block_features`` but NOT early,
                the method returns ``(velocity, feature_dict)``. Both default off →
                bit-exact no-op (plain velocity return). Unsupported with block swap.
            token_merger: Optional foveated token merger
                (``networks.foveated.FoveatedTokenMerge``): the block stack runs
                on a reduced sequence (fovea tokens 1:1, periphery cells
                averaged) with the merged rope, broadcast back to the full grid
                before ``final_layer``. Rides the same fake-5D
                ``(B, 1, L_red, 1, D)`` layout as native flatten, so Block code
                is unaffected. ``None`` (default) → bit-exact no-op.
        """
        if return_features_early and not return_block_features:
            raise ValueError(
                "return_features_early=True requires a non-empty return_block_features"
            )
        if return_block_features is not None and self.blocks_to_swap:
            # Early-exit would leave tail-block offloader moves un-submitted,
            # desyncing swap state. Turbo keeps the teacher resident so this never
            # fires; fails loud rather than corrupting silently if that changes.
            raise RuntimeError(
                "feature tap (return_block_features) is unsupported with block swap "
                f"(blocks_to_swap={self.blocks_to_swap}); keep the tapped DiT resident"
            )
        if (
            t5_input_ids is not None
            and self.use_llm_adapter
            and hasattr(self, "llm_adapter")
        ):
            crossattn_emb = self.llm_adapter(
                source_hidden_states=crossattn_emb,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=source_attention_mask,
            )
            if t5_attn_mask is not None:
                crossattn_emb[~t5_attn_mask.bool()] = 0

        x_B_T_H_W_D, rope_cos_sin = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            h_offset=h_offset,
            w_offset=w_offset,
        )

        # Native-shape flattening (compile_blocks): flatten 5D → fake-5D
        # (B,1,seq_len,1,D) so the block graph keys on token count (2 families),
        # not H/W separately (per-resolution recompiles). t=1,w=1 gives the same
        # flat token order, so Block code is unaffected. No padding → native flash,
        # bit-exact to the eager 5D path; eager forwards skip the reshape.
        _native_flatten_info = None
        _merge_shape = None
        if token_merger is not None:
            # Foveated token merge: the merged sequence already IS the fake-5D
            # (B, 1, L_red, 1, D) native-flatten layout, so this branch replaces
            # the flatten (never both). Rope is reduced to match (fovea rows
            # pass through; periphery cells get the exact mean-position rope).
            B_s = x_B_T_H_W_D.shape[0]
            _merge_shape = (B_s, x_B_T_H_W_D.shape[-1])
            x_B_T_H_W_D = token_merger.merge(x_B_T_H_W_D)
            if rope_cos_sin is not None:
                rope_cos_sin = token_merger.merge_rope(rope_cos_sin)
        elif self._native_flatten:
            B_s, T_s, H_s, W_s, D_s = x_B_T_H_W_D.shape
            seq_len = T_s * H_s * W_s
            _native_flatten_info = (T_s, H_s, W_s, seq_len)

            x_B_T_H_W_D = x_B_T_H_W_D.flatten(1, 3)
            x_B_T_H_W_D = x_B_T_H_W_D.unsqueeze(1).unsqueeze(3)

        # Cast RoPE cos/sin to the block compute dtype once. Without this, every
        # block's apply_rotary_pos_emb_qk re-materializes the fp32 cache in bf16.
        if rope_cos_sin is not None:
            compute_dtype = x_B_T_H_W_D.dtype
            rope_cos_sin = (
                rope_cos_sin[0].to(compute_dtype),
                rope_cos_sin[1].to(compute_dtype),
            )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)

        # DAVE: restamp current σ (timesteps is the DiT time arg on the σ∈[0,1]
        # scale) so per-block hooks can σ-gate the DC edit. Eager region, so this
        # scalar copy never enters the compiled graph.
        if self.enable_dave:
            self._dave_cur_sigma.copy_(timesteps_B_T.detach().float().reshape(-1)[0])

        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Mod-guidance: inject pooled text into the modulation path.
        # pooled_text_override decouples it from prefix/postfix tokens;
        # skip_pooled_text_proj disables it (distillation teacher forward). The
        # enable flag short-circuits the max/proj path when no trained
        # pooled_text_proj is loaded — bit-exact (output layer zero-init there).
        if self.enable_pooled_text_modulation and not skip_pooled_text_proj:
            if pooled_text_override is not None:
                pooled_text = pooled_text_override
            elif crossattn_emb is not None:
                pooled_text = crossattn_emb.max(dim=1).values  # (B, 1024)
            else:
                pooled_text = None
            if pooled_text is not None:
                t_embedding_B_T_D = t_embedding_B_T_D + self._pooled_text_delta(
                    pooled_text, t_embedding_B_T_D
                ).unsqueeze(1)

        # The steering delta is NOT baked into the shared t_embedding here — it is
        # applied per-block below via _mod_guidance_schedule so early tonal-DC blocks
        # and the final compensation layer can be skipped. Zero buffers ⇒ identity
        # when off. See docs/inference/mod-guidance.md.

        # E25b res-cond: explicit resolution conditioning on the modulation
        # trunk. (proj_weight, s) is attached per-forward by the trainer/probe
        # (the _sigma_lowres_yarn idiom — try/finally scoped, absent ⇒ branch
        # never taken; eager region, outside the compiled block graph). Like
        # the pooled-text delta above, this modulates the trunk input only —
        # adaln_lora_B_T_3D is computed inside t_embedder from the unmodified
        # sinusoid.
        res_cond = getattr(self, "_sigma_lowres_res_cond", None)
        if res_cond is not None:
            proj_w, s = res_cond
            t_embedding_B_T_D = t_embedding_B_T_D + sigma_lowres_res_cond_delta(
                proj_w, s, timesteps_B_T
            ).to(t_embedding_B_T_D.dtype)

        block_kwargs = {
            "rope_cos_sin": rope_cos_sin,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        }

        attn_params = attention_dispatch.AttentionParams.create_attention_params(
            self.attn_mode, self.attn_softmax_scale
        )

        # Pre-compute cross-attention BlockMask once for all blocks (flex mode only)
        if (
            self.attn_mode == "flex"
            and crossattn_seqlens is not None
            and attention_dispatch.create_block_mask is not None
        ):
            B, T, H, W, _D = x_B_T_H_W_D.shape
            q_len = T * H * W
            kv_len = crossattn_emb.shape[1]
            seqlens = crossattn_seqlens

            def _crossattn_mask_mod(b, h, q_idx, kv_idx):
                return kv_idx < seqlens[b]

            attn_params.crossattn_block_mask = attention_dispatch.create_block_mask(
                _crossattn_mask_mod,
                B,
                None,
                q_len,
                kv_len,
                device=x_B_T_H_W_D.device,
            )

        # No self-attention pad-mask: native shapes never have padded KV positions,
        # so selfattn_block_mask stays None (the legacy pad-to-static path is gone).

        # Feature tap: when requested, capture listed block outputs and — if early —
        # stop after the deepest tap so only blocks[0..k] run.
        feature_sink = {} if return_block_features is not None else None
        stop_after_block = (
            max(return_block_features)
            if (return_block_features and return_features_early)
            else None
        )

        # Block stack runs in _run_blocks — a split point so pre/post-block regions
        # stay eager while the block loop is the compiled hot path.
        x_B_T_H_W_D = self._run_blocks(
            x_B_T_H_W_D,
            t_embedding_B_T_D,
            crossattn_emb,
            attn_params,
            capture_blocks=return_block_features,
            feature_sink=feature_sink,
            stop_after_block=stop_after_block,
            **block_kwargs,
        )

        # Early feature-only return: skip the rest of the head. Captured features
        # stay in the block-output layout (native-flatten or eager grid); consumers
        # pool over the spatial/token axes, which is shape-agnostic across both.
        if return_features_early:
            return feature_sink

        # Foveated merge: broadcast the reduced sequence back to the full grid
        # (group-shared periphery rows) before final_layer/unpatchify.
        if _merge_shape is not None:
            x_B_T_H_W_D = token_merger.unmerge(x_B_T_H_W_D, *_merge_shape)

        # Native flatten: restore the original 5D shape. Delegated to a
        # @torch.compiler.disable'd helper so the bucket-dependent tuple never
        # enters the compile zone. See _unflatten_native_shape.
        if _native_flatten_info is not None:
            x_B_T_H_W_D = _unflatten_native_shape(x_B_T_H_W_D, _native_flatten_info)

        # Unconditional: zero buffers collapse to identity when guidance is off.
        t_emb_final = t_embedding_B_T_D + (
            self._mod_guidance_final_w * self._mod_guidance_delta
        ).unsqueeze(1)
        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D, t_emb_final, adaln_lora_B_T_3D=adaln_lora_B_T_3D
        )
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if return_block_features is not None:
            return x_B_C_Tt_Hp_Wp, feature_sink
        return x_B_C_Tt_Hp_Wp

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        target_input_ids: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
        crossattn_seqlens: Optional[torch.Tensor] = None,
        h_offset: int = 0,
        w_offset: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        if crossattn_seqlens is None:
            context, crossattn_seqlens = self._preprocess_text_embeds(
                context, target_input_ids, target_attention_mask, source_attention_mask
            )
        return self.forward_mini_train_dit(
            x,
            timesteps,
            context,
            fps=fps,
            padding_mask=padding_mask,
            crossattn_seqlens=crossattn_seqlens,
            h_offset=h_offset,
            w_offset=w_offset,
            **kwargs,
        )

    def _preprocess_text_embeds(
        self,
        source_hidden_states,
        target_input_ids,
        target_attention_mask=None,
        source_attention_mask=None,
    ):
        if target_input_ids is not None and target_input_ids.shape[-1] > 0:
            context = self.llm_adapter(
                source_hidden_states,
                target_input_ids,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
            )
            crossattn_mask = target_attention_mask
            # Adapter may have appended T5-side postfix tokens (dual mode) — extend mask to match
            if (
                crossattn_mask is not None
                and context.shape[1] > crossattn_mask.shape[-1]
            ):
                num_extra = context.shape[1] - crossattn_mask.shape[-1]
                extra_mask = torch.ones(
                    crossattn_mask.shape[0],
                    num_extra,
                    device=crossattn_mask.device,
                    dtype=crossattn_mask.dtype,
                )
                crossattn_mask = torch.cat([crossattn_mask, extra_mask], dim=-1)
            context[~crossattn_mask.bool()] = 0
        else:
            # Adapter skipped (pre-cached output or no adapter) — use source mask
            context = source_hidden_states
            crossattn_mask = source_attention_mask

        # Per-sample text token counts. Used only by the flex-attention BlockMask
        # path; the default sink-padded modes ignore it and treat zero keys as
        # attention sinks, which is what the pretrained model expects.
        crossattn_seqlens = None
        if crossattn_mask is not None:
            crossattn_seqlens = crossattn_mask.sum(dim=-1).to(torch.int32)
        return context, crossattn_seqlens


class LLMAdapterRMSNorm(nn.Module):
    """RMSNorm specifically for the LLM Adapter (T5-style, no mean subtraction)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


def _adapter_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _adapter_apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    x_embed = (x * cos) + (_adapter_rotate_half(x) * sin)
    return x_embed


class AdapterRotaryEmbedding(nn.Module):
    """Rotary embedding for LLM Adapter."""

    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float)
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # inv_freq is registered as fp32 but a parent .to(bf16) casts it too —
        # force fp32 here so the matmul matches position_ids_expanded.
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LLMAdapterAttention(nn.Module):
    """Attention module for LLM Adapter with QK-norm and separate RoPE for query/key."""

    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()

        inner_dim = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = LLMAdapterRMSNorm(self.head_dim)

        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = LLMAdapterRMSNorm(self.head_dim)

        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)

        self.o_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def forward(
        self,
        x,
        q_mask=None,
        kv_mask=None,
        context=None,
        position_embeddings=None,
        position_embeddings_context=None,
    ):
        """
        Args:
            x: Query input [B, L_q, D].
            q_mask: Optional 2-D bool mask [B, L_q] — True = valid token.
            kv_mask: Optional 2-D bool mask [B, L_kv] — True = valid token.
            context: Key/Value input [B, L_kv, D]. Defaults to x (self-attention).
            position_embeddings: (cos, sin) for query RoPE.
            position_embeddings_context: (cos, sin) for key RoPE.
        """
        context = x if context is None else context
        input_shape = x.shape[:-1]
        q_shape = (*input_shape, self.n_heads, self.head_dim)
        context_shape = context.shape[:-1]
        kv_shape = (*context_shape, self.n_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(x).view(q_shape))
        key_states = self.k_norm(self.k_proj(context).view(kv_shape))
        value_states = self.v_proj(context).view(kv_shape)

        if position_embeddings is not None:
            assert position_embeddings_context is not None
            cos, sin = position_embeddings
            # RoPE expects [B, H, L, D] layout
            query_states = _adapter_apply_rotary_pos_emb(
                query_states.transpose(1, 2), cos, sin
            ).transpose(1, 2)
            cos, sin = position_embeddings_context
            key_states = _adapter_apply_rotary_pos_emb(
                key_states.transpose(1, 2), cos, sin
            ).transpose(1, 2)

        can_use_flash = (
            attention_dispatch.flash_attn_varlen_func is not None
            and query_states.dtype in (torch.float16, torch.bfloat16)
        )

        if can_use_flash and q_mask is None and kv_mask is None:
            # No masking — simple flash attention, [B, L, H, D] layout
            attn_output = attention_dispatch.flash_attn_func(
                query_states, key_states, value_states
            )
        elif can_use_flash:
            # Varlen flash attention: pack valid tokens, attend, unpack
            B, L_q = query_states.shape[:2]
            L_kv = key_states.shape[1]

            eff_q_mask = (
                q_mask
                if q_mask is not None
                else query_states.new_ones(B, L_q, dtype=torch.bool)
            )
            eff_kv_mask = (
                kv_mask
                if kv_mask is not None
                else key_states.new_ones(B, L_kv, dtype=torch.bool)
            )

            q_seqlens = eff_q_mask.sum(dim=1, dtype=torch.int32)
            kv_seqlens = eff_kv_mask.sum(dim=1, dtype=torch.int32)

            cu_seqlens_q = F.pad(q_seqlens.cumsum(0, dtype=torch.int32), (1, 0))
            cu_seqlens_kv = F.pad(kv_seqlens.cumsum(0, dtype=torch.int32), (1, 0))

            q_packed = query_states[eff_q_mask]
            k_packed = key_states[eff_kv_mask]
            v_packed = value_states[eff_kv_mask]

            # Pass the padded lengths as max_seqlen_q/k. Slightly over-sizes the
            # flash kernel's metadata vs the true batch maxima but avoids a
            # host-device sync from .item() on every adapter layer.
            out_packed = attention_dispatch.flash_attn_varlen_func(
                q_packed,
                k_packed,
                v_packed,
                cu_seqlens_q,
                cu_seqlens_kv,
                L_q,
                L_kv,
            )

            attn_output = query_states.new_zeros(B, L_q, self.n_heads, self.head_dim)
            attn_output[eff_q_mask] = out_packed
        else:
            # Fallback to PyTorch SDPA: needs [B, H, L, D] layout
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            sdpa_mask = kv_mask[:, None, None, :] if kv_mask is not None else None
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=sdpa_mask
            )
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class LLMAdapterTransformerBlock(nn.Module):
    """Transformer block for LLM Adapter: optional self-attn + cross-attn + MLP."""

    def __init__(
        self,
        source_dim,
        model_dim,
        num_heads=16,
        mlp_ratio=4.0,
        self_attn=False,
        layer_norm=False,
    ):
        super().__init__()
        self.has_self_attn = self_attn

        if self.has_self_attn:
            self.norm_self_attn = (
                nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
            )
            self.self_attn = LLMAdapterAttention(
                query_dim=model_dim,
                context_dim=model_dim,
                n_heads=num_heads,
                head_dim=model_dim // num_heads,
            )

        self.norm_cross_attn = (
            nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        )
        self.cross_attn = LLMAdapterAttention(
            query_dim=model_dim,
            context_dim=source_dim,
            n_heads=num_heads,
            head_dim=model_dim // num_heads,
        )

        self.norm_mlp = (
            nn.LayerNorm(model_dim) if layer_norm else LLMAdapterRMSNorm(model_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )

    def forward(
        self,
        x,
        context,
        target_attention_mask=None,
        source_attention_mask=None,
        position_embeddings=None,
        position_embeddings_context=None,
    ):
        if self.has_self_attn:
            # Self-attention: target_attention_mask is not expected to be all zeros
            normed = self.norm_self_attn(x)
            attn_out = self.self_attn(
                normed,
                q_mask=target_attention_mask,
                kv_mask=target_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings,
            )
            x = x + attn_out

        normed = self.norm_cross_attn(x)
        attn_out = self.cross_attn(
            normed,
            q_mask=target_attention_mask,
            kv_mask=source_attention_mask,
            context=context,
            position_embeddings=position_embeddings,
            position_embeddings_context=position_embeddings_context,
        )
        x = x + attn_out

        x = x + self.mlp(self.norm_mlp(x))
        return x

    def init_weights(self):
        torch.nn.init.zeros_(self.mlp[2].weight)


class LLMAdapter(nn.Module):
    """Bridge module: Qwen3 embeddings (source) → T5-compatible space (target).

    Uses T5 token IDs as target input, embeds them, and cross-attends to Qwen3 hidden states.
    """

    def __init__(
        self,
        source_dim,
        target_dim,
        model_dim,
        num_layers=6,
        num_heads=16,
        embed=None,
        self_attn=False,
        layer_norm=False,
    ):
        super().__init__()
        if embed is not None:
            self.embed = nn.Embedding.from_pretrained(embed.weight)
        else:
            self.embed = nn.Embedding(32128, target_dim)
        if model_dim != target_dim:
            self.in_proj = nn.Linear(target_dim, model_dim)
        else:
            self.in_proj = nn.Identity()
        self.rotary_emb = AdapterRotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList(
            [
                LLMAdapterTransformerBlock(
                    source_dim,
                    model_dim,
                    num_heads=num_heads,
                    self_attn=self_attn,
                    layer_norm=layer_norm,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = LLMAdapterRMSNorm(target_dim)

    def forward(
        self,
        source_hidden_states,
        target_input_ids,
        target_attention_mask=None,
        source_attention_mask=None,
    ):
        # Keep masks as 2D [B, L] bool tensors — the attention layer handles
        # expansion to 4D for SDPA or packing for flash_attn_varlen_func.
        if target_attention_mask is not None:
            target_attention_mask = target_attention_mask.to(torch.bool)
            if target_attention_mask.ndim == 4:
                target_attention_mask = target_attention_mask.squeeze(1).squeeze(1)

        if source_attention_mask is not None:
            source_attention_mask = source_attention_mask.to(torch.bool)
            if source_attention_mask.ndim == 4:
                source_attention_mask = source_attention_mask.squeeze(1).squeeze(1)

        x = self.in_proj(self.embed(target_input_ids))

        context = source_hidden_states
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_ids_context = torch.arange(
            context.shape[1], device=x.device
        ).unsqueeze(0)
        position_embeddings = self.rotary_emb(x, position_ids)
        position_embeddings_context = self.rotary_emb(x, position_ids_context)
        for block in self.blocks:
            x = block(
                x,
                context,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
                position_embeddings=position_embeddings,
                position_embeddings_context=position_embeddings_context,
            )
        return self.norm(self.out_proj(x))


# Not used currently, but kept for reference
