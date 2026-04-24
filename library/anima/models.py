# Anima Model Architecture
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
from networks import attention

# KV length buckets for cross-attention trimming. Captions trimmed to the smallest
# bucket >= max(real_token_lengths). Keeps torch.compile shapes stable (max 4 variants).
_KV_BUCKETS = (128, 192, 256, 512)


class _FP8LinearFunc(torch.autograd.Function):
    """Custom autograd for fp8 linear: saves compact fp8 weight instead of transient bf16 copy."""

    @staticmethod
    def forward(
        input: torch.Tensor, weight_fp8: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        return F.linear(input, weight_fp8.to(input.dtype), bias)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, weight_fp8, _ = inputs
        ctx.save_for_backward(weight_fp8)

    @staticmethod
    def backward(ctx, grad_output):
        (weight_fp8,) = ctx.saved_tensors
        grad_input = grad_output @ weight_fp8.to(grad_output.dtype)
        return grad_input, None, None


class FP8Linear(nn.Linear):
    """Drop-in nn.Linear replacement that stores weights in float8_e4m3fn.

    Subclasses nn.Linear so that LoRA's isinstance checks still match.
    Uses a custom autograd function so that only the compact fp8 weight
    (not a transient bf16 copy) is saved for backward.
    """

    def __init__(self, original: nn.Linear):
        nn.Module.__init__(self)
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = nn.Parameter(
            original.weight.data.to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        self.bias = original.bias

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse)
        if self.weight.dtype != torch.float8_e4m3fn:
            self.weight = nn.Parameter(
                self.weight.data.to(torch.float8_e4m3fn),
                requires_grad=False,
            )
        return result

    def forward(self, input):
        return _FP8LinearFunc.apply(input, self.weight, self.bias)


# Class names whose children should NOT be quantized to fp8.
_FP8_SKIP_CLASS_NAMES = {"RMSNorm", "TimestepEmbedding", "FinalLayer", "LLMAdapter"}


def quantize_to_fp8(model: nn.Module) -> int:
    """Replace frozen nn.Linear modules with FP8Linear, skipping sensitive layers.

    Skips: RMSNorm, TimestepEmbedding, FinalLayer, LLMAdapter,
    and any module with requires_grad=True parameters.

    Returns the number of modules replaced.
    """
    skip_modules: set[int] = set()
    for mod in model.modules():
        if type(mod).__name__ in _FP8_SKIP_CLASS_NAMES:
            skip_modules.add(id(mod))
            for child in mod.modules():
                skip_modules.add(id(child))

    count = 0
    for parent in model.modules():
        if id(parent) in skip_modules:
            continue
        for name, child in parent.named_children():
            if not isinstance(child, nn.Linear):
                continue
            if id(child) in skip_modules:
                continue
            setattr(parent, name, FP8Linear(child))
            count += 1

    return count


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, (list, tuple)):
        return type(x)(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, (list, tuple)):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


# Unsloth Offloaded Gradient Checkpointing
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

    Compared to standard cpu_offload_checkpointing which uses blocking transfers,
    this uses non_blocking=True to hide CPU<->GPU transfer latency behind compute.
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
        # NOTE: args stored directly on ctx (not via save_for_backward) because
        # the training loop holds references to these tensors, preventing GC.
        # Using save_for_backward for all args would add complexity for no benefit.
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


@torch.compiler.disable(recursive=True)
def _unpad_static_shape(x, pad_info):
    """Strip the static-shape padding back to (B, T, H, W, D).

    Disabled from dynamo tracing on purpose: pad_info is a 4-tuple of Python
    ints (T_s, H_s, W_s, seq_len) computed from the input's pre-pad shape, so
    if this ran inside the compiled frame each bucket would specialize
    ``pad_info[1] == H_s`` (per-value guard) and narrow the symbolic range
    on ``pad_info[3]`` (per-bucket seq_len guard). Running it eagerly keeps
    the returned tensor's shape as the only signal crossing back into the
    compile zone — downstream ops (final_layer, unpatchify) then pick up
    symbolic T/H/W from the tensor itself, not from Python ints.
    """
    T_s, H_s, W_s, seq_len = pad_info
    x = x.squeeze(3).squeeze(1)
    x = x[:, :seq_len, :]
    x = x.unflatten(1, (T_s, H_s, W_s))
    return x


from library.log import setup_logging  # noqa: E402

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


# Utility functions: RoPE for DiT
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
    """Apply RoPE to q and k using precomputed (cos, sin) tensors."""
    cos_, sin_ = rope_cos_sin
    max_seq_len = cos_.shape[0]
    cur_seq_len = q.shape[1] if tensor_format == "bshd" else q.shape[0]

    assert cur_seq_len <= max_seq_len, (
        f"Rotary Embeddings only supported up to {max_seq_len} sequence length!"
    )
    cos_ = cos_[:cur_seq_len]
    sin_ = sin_[:cur_seq_len]

    if tensor_format == "bshd":
        cos_ = cos_.transpose(0, 1)
        sin_ = sin_.transpose(0, 1)

    rot_dim = cos_.shape[-1]

    cos_q = cos_.to(q.dtype)
    sin_q = sin_.to(q.dtype)
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    q = torch.cat(
        ((q_rot * cos_q) + (_rotate_half(q_rot, False) * sin_q), q_pass), dim=-1
    )

    cos_k = cos_q if k.dtype == q.dtype else cos_.to(k.dtype)
    sin_k = sin_q if k.dtype == q.dtype else sin_.to(k.dtype)
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    k = torch.cat(
        ((k_rot * cos_k) + (_rotate_half(k_rot, False) * sin_k), k_pass), dim=-1
    )

    return q, k


# Basic building blocks
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


# Attention module for DiT
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
        attn_params: attention.AttentionParams,
        context: torch.Tensor,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        q, k, v = self.compute_qkv(x, context, rope_cos_sin=rope_cos_sin)
        if q.dtype != v.dtype:
            if (
                not attn_params.supports_fp32 or attn_params.requires_same_dtype
            ) and torch.is_autocast_enabled():
                # FlashAttention requires fp16/bf16, xformers require same dtype; only cast when autocast is active.
                target_dtype = v.dtype  # v has fp16/bf16 dtype
                q = q.to(target_dtype)
                k = k.to(target_dtype)
        # return self.compute_attention(q, k, v)
        qkv = [q, k, v]
        del q, k, v
        result = attention.attention(qkv, attn_params=attn_params)
        return self.output_dropout(self.output_proj(result))


# Positional Embeddings
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

        # Skip Python dict cache inside compiled code — dict mutations cause dynamo
        # guard failures and recompilation.  The RoPE computation is pure tensor math
        # that dynamo traces cleanly; with static_token_count the shapes are constant
        # so there is no recompilation from shape guards.
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

    @property
    def seq_dim(self) -> int:
        return 0


# Timestep Embedding
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
        emb = timesteps[:, None].float() * emb[None, :]

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


# Patch Embedding
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


# Final Layer
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


# Transformer Block (DiT Block)
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
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False

    def enable_gradient_checkpointing(
        self, cpu_offload: bool = False, unsloth_offload: bool = False
    ):
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload if not unsloth_offload else False
        self.unsloth_offload_checkpointing = unsloth_offload

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
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
        attn_params: attention.AttentionParams,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Compute AdaLN modulation parameters
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

        # Reshape for broadcasting: (B, T, D) -> (B, T, 1, 1, D)
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

        # 1. Self-attention
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

        # 2. Cross-attention
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
        x_B_T_H_W_D = result * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        # 3. MLP
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
        attn_params: attention.AttentionParams,
        rope_cos_sin: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            if self.unsloth_offload_checkpointing:
                # Unsloth: async non-blocking CPU RAM offload (fastest offload method)
                return unsloth_checkpoint(
                    self._forward,
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    rope_cos_sin,
                    adaln_lora_B_T_3D,
                )
            elif self.cpu_offload_checkpointing:
                # Standard cpu offload: blocking transfers
                def create_custom_forward(func):
                    def custom_forward(*inputs):
                        # Determine original device from first tensor input
                        device = next(
                            t.device for t in inputs if isinstance(t, torch.Tensor)
                        )
                        device_inputs = to_device(inputs, device)
                        outputs = func(*device_inputs)
                        return to_cpu(outputs)

                    return custom_forward

                return torch_checkpoint(
                    create_custom_forward(self._forward),
                    x_B_T_H_W_D,
                    emb_B_T_D,
                    crossattn_emb,
                    attn_params,
                    rope_cos_sin,
                    adaln_lora_B_T_3D,
                    use_reentrant=False,
                )
            else:
                # Standard gradient checkpointing (no offload)
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


# Main DiT Model: MiniTrainDIT (renamed to Anima)
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
        split_attn: bool = False,
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
        self.split_attn = split_attn
        self.attn_softmax_scale = attn_softmax_scale

        # Block swap support
        self.blocks_to_swap = None
        self.offloader: Optional[custom_offloading_utils.ModelOffloader] = None

        # Static-shape training: pad all token sequences to this count to eliminate
        # torch.compile recompilation across different bucket resolutions.
        # Set via set_static_token_count(). None = disabled (original behavior).
        self.static_token_count: Optional[int] = None

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

        # Modulation guidance: project pooled crossattn_emb into modulation space.
        # Zero-initialized so it's a no-op before distillation training.
        self.pooled_text_proj = nn.Sequential(
            nn.Linear(crossattn_emb_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        # Modulation guidance runtime state as non-persistent buffers (zeros = off).
        # Registered unconditionally so the forward can do unconditional arithmetic
        # (``t_emb + schedule[l] * delta``) without a Python-level None/zero branch —
        # branches here guard-fire under torch.compile per bucket/block combination.
        # Setters live on library/inference/mod_guidance.py; reset via reset_mod_guidance().
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

        self.init_weights()

    def reset_mod_guidance(self) -> None:
        """Disable modulation guidance by zeroing the runtime buffers."""
        self._mod_guidance_delta.zero_()
        self._mod_guidance_schedule.zero_()
        self._mod_guidance_final_w.zero_()

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

    def enable_gradient_checkpointing(
        self, cpu_offload: bool = False, unsloth_offload: bool = False
    ):
        for block in self.blocks:
            block.enable_gradient_checkpointing(
                cpu_offload=cpu_offload, unsloth_offload=unsloth_offload
            )

    def disable_gradient_checkpointing(self):
        for block in self.blocks:
            block.disable_gradient_checkpointing()

    def set_static_token_count(self, count: Optional[int]):
        """Enable static-shape training by padding all token sequences to `count`.

        All bucket resolutions must produce <= `count` spatial tokens after
        patchification.  Passing None disables static-shape mode.
        """
        self.static_token_count = count

    def compile_blocks(self, backend: str = "inductor", mode: Optional[str] = None):
        """torch.compile each block's _forward individually.

        Compiles _forward (the actual attention/MLP computation) rather than
        forward (the checkpointing wrapper).  This is critical because
        unsloth_checkpoint has @torch._disable_dynamo, which causes an
        immediate graph break if forward itself is compiled — dynamo compiles
        nothing useful but still checks shape guards, causing recompile storms.

        ``mode`` maps to torch.compile's inductor preset (e.g. ``reduce-overhead``
        to enable per-block CUDAGraphs). ``None`` leaves it unset (inductor default).
        """
        compile_kwargs = {"backend": backend, "dynamic": False}
        if mode is not None:
            compile_kwargs["mode"] = mode
        for i, block in enumerate(self.blocks):
            block._forward = torch.compile(block._forward, **compile_kwargs)
        print(
            f"Anima: compiled {len(self.blocks)} block._forward with "
            f"backend={backend}, mode={mode}"
        )

    def compile_core(self, backend: str = "inductor", mode: Optional[str] = None):
        """torch.compile the constant-shape block stack (``_run_blocks``).

        Works with ``set_static_token_count``: the pre-blocks eager region
        (patch/embed/static-pad/RoPE-pad/t_embedder/BlockMask construction)
        hands off a shape-invariant bundle, so ``_run_blocks`` traces once
        and a single CUDAGraph serves every bucket in CONSTANT_TOKEN_BUCKETS.
        Post-blocks (unpad/final_layer/unpatchify) stay eager.

        Requires ``static_token_count`` to be set, ``gradient_checkpointing``
        off, and ``blocks_to_swap`` unset — the caller asserts these.
        ``dynamic=False`` is safe because every input to ``_run_blocks`` is
        shape-pinned by the pre-blocks eager region.
        """
        assert self.static_token_count is not None, (
            "compile_core requires set_static_token_count() to be called first"
        )
        compile_kwargs = {"backend": backend, "dynamic": False}
        if mode is not None:
            compile_kwargs["mode"] = mode
        self._run_blocks = torch.compile(self._run_blocks, **compile_kwargs)
        print(f"Anima: compiled _run_blocks with backend={backend}, mode={mode}")

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

        if h_offset != 0 or w_offset != 0:
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
        # Move all modules to device except blocks (which are managed by offloader)
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

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def _run_blocks(
        self,
        x_padded: torch.Tensor,
        t_embedding_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params,
        **block_kwargs,
    ) -> torch.Tensor:
        """Constant-shape block stack — the compile target for compile_core.

        Every input is shape-pinned by the eager pre-blocks region:
        - ``x_padded``: ``(B, 1, static_token_count, 1, D)`` via flatten+pad
        - ``t_embedding_B_T_D``: ``(B, 1, D)``
        - ``crossattn_emb``: ``(B, max_text_len, D)`` (padded to max_length)
        - ``attn_params``: BlockMasks built with tensor-valued seq lens so
          no per-bucket guards fire on the mask_mod closure
        - ``block_kwargs["rope_cos_sin"]``: each ``(static_token_count, 1, 1, D_head)``
        - ``block_kwargs["adaln_lora_B_T_3D"]``: ``(B, 1, 3, D)``

        Mod-guidance is applied via buffers on ``self`` (zero = off) so the
        per-block ``t_emb`` arithmetic is unconditional. No Python branches.
        """
        # Normalize requires_grad once at the stack entry. Block 0 receives
        # requires_grad=False (frozen patch_embed output) while blocks 1+
        # receive True (LoRA-enhanced); a mismatch would fragment guards if
        # the loop were ever traced per-block. requires_grad_(True) is a no-op
        # under torch.no_grad().
        x = x_padded.requires_grad_()
        for block_idx, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(block_idx)

            # Unconditional: zero buffers collapse to identity when guidance
            # is off; avoids a data-dependent branch inside the compiled frame.
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

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.blocks, block_idx)
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
        max_crossattn_seqlen: Optional[int] = None,
        h_offset: int = 0,
        w_offset: int = 0,
        pooled_text_override: Optional[torch.Tensor] = None,
        skip_pooled_text_proj: bool = False,
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
        """
        # Run LLM adapter inside forward for correct DDP gradient synchronization
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

        # --- Static-shape padding: flatten, pad to fixed token count, reshape to fake-5D ---
        # This makes ALL block inputs shape-identical across buckets, eliminating
        # torch.compile recompilation.  The fake-5D shape (B, 1, target, 1, D) is
        # compatible with existing Block code because rearrange("b t h w d -> b (t h w) d")
        # with t=1, w=1 produces the same flat sequential order as the original.
        _static_pad_info = None
        if self.static_token_count is not None:
            target = self.static_token_count
            B_s, T_s, H_s, W_s, D_s = x_B_T_H_W_D.shape
            seq_len = T_s * H_s * W_s
            _static_pad_info = (T_s, H_s, W_s, seq_len)

            # Flatten 5D → 2D and pad sequence to target length.
            # Always pad (even when seq_len == target) to avoid a data-dependent
            # branch that causes torch.compile recompilation across bucket shapes.
            x_B_T_H_W_D = x_B_T_H_W_D.flatten(1, 3)
            x_B_T_H_W_D = torch.nn.functional.pad(
                x_B_T_H_W_D, (0, 0, 0, target - seq_len)
            )
            # Reshape to fake-5D: (B, 1, target, 1, D)
            x_B_T_H_W_D = x_B_T_H_W_D.unsqueeze(1).unsqueeze(3)

            # Pad RoPE cos/sin: each (L, 1, 1, D_head) → (target, 1, 1, D_head)
            if rope_cos_sin is not None:
                pad = (0, 0, 0, 0, 0, 0, 0, target - rope_cos_sin[0].shape[0])
                rope_cos_sin = (
                    torch.nn.functional.pad(rope_cos_sin[0], pad),
                    torch.nn.functional.pad(rope_cos_sin[1], pad),
                )

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Modulation guidance: inject pooled text embedding into modulation path.
        # - pooled_text_override: use this tensor instead of computing from crossattn_emb
        #   (used to decouple modulation from prefix/postfix tokens)
        # - skip_pooled_text_proj: disable entirely (for distillation teacher forward)
        if not skip_pooled_text_proj:
            if pooled_text_override is not None:
                pooled_text = pooled_text_override
            elif crossattn_emb is not None:
                pooled_text = crossattn_emb.max(dim=1).values  # (B, 1024)
            else:
                pooled_text = None
            if pooled_text is not None:
                t_embedding_B_T_D = t_embedding_B_T_D + self.pooled_text_proj(
                    pooled_text
                ).unsqueeze(1)

        # Phase 2: modulation guidance delta.
        # The steering delta (proj_pos - proj_neg) is NOT baked into the shared
        # t_embedding here — it is applied per-block below via _mod_guidance_schedule,
        # so early tonal-DC blocks and the final compensation layer can be skipped.
        # Buffers are zeros when guidance is off (see __init__), so the arithmetic
        # below is an unconditional identity in that case — no Python branch.
        # See docs/methods/mod-guidance.md for the rationale.

        block_kwargs = {
            "rope_cos_sin": rope_cos_sin,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        }

        attn_params = attention.AttentionParams.create_attention_params(
            self.attn_mode, self.split_attn, self.attn_softmax_scale
        )

        # Bucketed KV trimming for cross-attention requires flash4 (LSE correction),
        # which is not supported yet (flash-attention-sm120 disabled).
        # if (
        #     crossattn_seqlens is not None
        #     and getattr(self, "trim_crossattn_kv", False)
        #     and self.attn_mode == "flash4"
        #     and not self.split_attn
        # ):
        #     full_len = crossattn_emb.shape[1]
        #     max_real_len = (
        #         max_crossattn_seqlen
        #         if max_crossattn_seqlen is not None
        #         else int(crossattn_seqlens.max())
        #     )
        #     trim_len = next((b for b in _KV_BUCKETS if b >= max_real_len), full_len)
        #     if trim_len < full_len:
        #         crossattn_emb = crossattn_emb[:, :trim_len].contiguous()
        #         attn_params.crossattn_full_len = full_len

        # Pre-compute cross-attention BlockMask once for all blocks (flex mode only)
        if (
            self.attn_mode == "flex"
            and crossattn_seqlens is not None
            and attention.create_block_mask is not None
        ):
            B, T, H, W, _D = x_B_T_H_W_D.shape
            q_len = T * H * W
            kv_len = crossattn_emb.shape[1]
            seqlens = crossattn_seqlens

            def _crossattn_mask_mod(b, h, q_idx, kv_idx):
                return kv_idx < seqlens[b]

            attn_params.crossattn_block_mask = attention.create_block_mask(
                _crossattn_mask_mod,
                B,
                None,
                q_len,
                kv_len,
                device=x_B_T_H_W_D.device,
            )

        # Pre-compute self-attention BlockMask for static-shape mode (flex only).
        # IMPORTANT: always create the mask (even when seq_len == target, i.e. no
        # actual padding) so that the compiled _forward always takes the same
        # code path.  A None-vs-BlockMask control-flow difference triggers dynamo
        # recompilation; with 5 bucket token-counts × 2 requires_grad states the
        # shared code cache (all blocks use the same _forward bytecode) exceeds
        # the recompile limit and falls back to eager, losing flex_attention fusion.
        if (
            _static_pad_info is not None
            and self.attn_mode == "flex"
            and attention.create_block_mask is not None
        ):
            # Use a tensor instead of a Python int so dynamo tracks it
            # symbolically rather than guarding on the exact value.  A plain
            # int in the mask_mod closure causes a recompile per bucket size.
            _sa_seq_len = torch.tensor(
                _static_pad_info[3], dtype=torch.int64, device=x_B_T_H_W_D.device
            )
            _sa_target = self.static_token_count
            _sa_B = x_B_T_H_W_D.shape[0]

            def _selfattn_mask_mod(b, h, q_idx, kv_idx):
                return kv_idx < _sa_seq_len

            attn_params.selfattn_block_mask = attention.create_block_mask(
                _selfattn_mask_mod,
                _sa_B,
                None,
                _sa_target,
                _sa_target,
                device=x_B_T_H_W_D.device,
            )

        # Block stack runs in _run_blocks — a split point so `compile_core`
        # can wrap just the shape-invariant region while pre/post stay eager.
        x_B_T_H_W_D = self._run_blocks(
            x_B_T_H_W_D,
            t_embedding_B_T_D,
            crossattn_emb,
            attn_params,
            **block_kwargs,
        )

        # --- Static-shape: strip padding and restore original 5D shape ---
        # Delegated to a @torch.compiler.disable'd helper so the bucket-
        # dependent tuple (T_s, H_s, W_s, seq_len) never enters the compile
        # zone. See _unpad_static_shape for rationale.
        if _static_pad_info is not None:
            x_B_T_H_W_D = _unpad_static_shape(x_B_T_H_W_D, _static_pad_info)

        # Unconditional: zero buffers collapse to identity when guidance is off.
        t_emb_final = t_embedding_B_T_D + (
            self._mod_guidance_final_w * self._mod_guidance_delta
        ).unsqueeze(1)
        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D, t_emb_final, adaln_lora_B_T_3D=adaln_lora_B_T_3D
        )
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
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
            # Compute seqlens from mask inside _preprocess_text_embeds
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
            context[~crossattn_mask.bool()] = 0  # zero out padding tokens
        else:
            # Adapter skipped (pre-cached output or no adapter) — use source mask
            context = source_hidden_states
            crossattn_mask = source_attention_mask

        # Compute seqlens from mask for bucketed KV trimming with LSE correction.
        # Pretrained model expects padding as attention sinks (zero keys contribute
        # exp(0)=1 to softmax denominator); the attention function accounts for
        # removed sinks via an exact sigmoid correction on the logsumexp.
        crossattn_seqlens = None
        if crossattn_mask is not None:
            crossattn_seqlens = crossattn_mask.sum(dim=-1).to(torch.int32)
        return context, crossattn_seqlens


# LLM Adapter: Bridges Qwen3 embeddings to T5-compatible space
class LLMAdapterRMSNorm(nn.Module):
    """RMSNorm specifically for the LLM Adapter (T5-style, no mean subtraction)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
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
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
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
            attention.flash_attn_varlen_func is not None
            and query_states.dtype in (torch.float16, torch.bfloat16)
        )

        if can_use_flash and q_mask is None and kv_mask is None:
            # No masking — simple flash attention, [B, L, H, D] layout
            attn_output = attention.flash_attn_func(
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

            # Pack by removing padding: [B, L, H, D] → [total_valid, H, D]
            q_packed = query_states[eff_q_mask]
            k_packed = key_states[eff_kv_mask]
            v_packed = value_states[eff_kv_mask]

            out_packed = attention.flash_attn_varlen_func(
                q_packed,
                k_packed,
                v_packed,
                cu_seqlens_q,
                cu_seqlens_kv,
                q_seqlens.max().item(),
                kv_seqlens.max().item(),
            )

            # Unpack: [total_valid_q, H, D] → [B, L_q, H, D]
            attn_output = query_states.new_zeros(B, L_q, self.n_heads, self.head_dim)
            attn_output[eff_q_mask] = out_packed
        else:
            # Fallback to PyTorch SDPA: needs [B, H, L, D] layout
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            # Expand kv_mask to 4D for SDPA broadcasting: [B, L] → [B, 1, 1, L]
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
