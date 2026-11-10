"""MOSS-Image DiT + scheduler, vendored from ``pipeline_moss_image.py``.

Source: OpenMOSS-Team/MOSS-Image-{Base,SFT} (``pipeline_moss_image.py``,
https://github.com/OpenMOSS/MOSS-Image), Apache License 2.0, Copyright the
OpenMOSS team. Vendored so nothing loads with ``trust_remote_code``. State-dict
names are unchanged, so ``from_pretrained`` reads the published ``transformer/``
folder as-is.

Changes from the reference, all additive:

- ``forward`` is the reference per-sample loop, op for op. It is split into
  ``time_conditioning`` / ``embed_sample`` / ``run_layers`` / ``head`` so the
  sampler (Spectrum) and the trainer can reach the pieces.
- ``forward_batched``: every sample in one forward, text padded to the longest
  and excluded by a key mask; ``skip_layers`` per sample, so cond and the
  skip-22 uncond share a batch.
- ``layer_hook``: optional ``(index, x, n_img) -> x`` run on each block's output
  (rows that ran the block only), outside the compiled block. DAVE
  (``library.moss.dave``) and its probe use it; ``None`` is the reference.
- ``ref`` (i2i): clean reference latents on RoPE's ``r = 1`` axis, last in
  the sequence (image, text, reference), so one int ``ref_start`` bounds them.
  ``ref_timestep`` (None = the image's timestep) gives the reference tokens their
  own block modulation, e.g. ``0`` for a clean latent. ``ref_isolate`` restricts
  the reference queries to the reference keys; image and text still read it.
  Under ``ref_isolate`` a skipped block (``skip_layers``, e.g. the skip-22 uncond)
  still runs the reference tokens, so every branch reads the same reference;
  ``ref_follows_skip`` skips them with the image and text instead.
  ``library.moss.lora.load_for_inference`` sets both from the LoRA's metadata.
  Isolated at a fixed ``ref_timestep``, the reference stream depends on the
  reference alone; ``ref_cache`` (a ``RefCache``, set by ``generate.denoise``)
  runs it once and later forwards read its per-block keys and values.
  ``ref_scope`` (``LoRANetwork.reference`` at ``ref_adapters = all``): the reference rows
  have an adapter bank of their own and Base modulation. A forward with a reference then
  goes through ``ref_cache`` or under ``set_row_mask(main, ref)``.
  ``ref_gate`` (``(rows, keep)``, None by default): the image queries ``rows`` read
  only the reference keys ``keep``; ``TrainRequest.ref_gate`` sets it per step
  (``project/lora_blocks``).
- Attention: unmasked CUDA bf16 / fp16 calls run FA2 (``flash_attn_func``) on
  ``(B, L, H, D)``; masked (``forward_batched``), CPU and fp32 calls keep the
  reference's SDPA. Same math, a different kernel: FA2 issues no ``% 8``
  query-length guard under compile.
- ``pack_latents`` / ``unpack_latents``: the FLUX.2 VAE ↔ DiT latent boundary
  (2×2 ``patchify`` + the VAE's BatchNorm statistics).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from torch.nn import functional as F

from library.moss.requests import DEFAULT_SHIFT

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
try:
    # the sm_120 build's fp16-input path: fp16 copies made here, fused into RoPE under compile
    from flash_attn import flash_attn_f16in_func, flash_attn_f16in_takes
except ImportError:
    flash_attn_f16in_func = flash_attn_f16in_takes = None


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
        25.0 * torch.tanh(hidden / 25.0)
    )


def _validate_skip_layers(layers, depth, name="skip_layers"):
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f"{name} must contain integer block indices in [0, {depth}).")
    return frozenset(layers)


def rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


def _flash_ok(x):
    return (
        flash_attn_func is not None
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
    )


def _sdpa(q, k, v, attn_mask=None):
    """``(B, L, H, D)`` in and out."""
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).transpose(1, 2)


def _attention(
    q,
    k,
    v,
    rope,
    heads,
    key_mask=None,
    ref_start=None,
    ref_isolate=False,
    ref_kv=None,
    ref_gate=None,
):
    """``key_mask``: bool ``(B, L)``, True = attend. None = the reference path.
    ``ref_isolate``: the reference queries (``ref_start`` on) attend only to the
    reference keys. ``ref_kv = (k, v, rope)``: cached reference keys (before RoPE)
    and values appended to the key set. ``ref_gate = (rows, keep)``: the image
    queries ``rows`` read only the reference keys ``keep`` (``_gate_rows``). Unmasked
    CUDA half-precision calls run FA2, the rest SDPA (the ref-only call is unmasked:
    its keys are all reference tokens)."""
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = rotate(q.reshape(shape), rope).to(v.dtype)
    k = rotate(k.reshape(shape), rope).to(v.dtype)
    v = v.reshape(shape)
    if ref_kv is not None:
        ref_k, ref_v, ref_rope = ref_kv
        ref_shape = (*ref_k.shape[:2], heads, ref_k.shape[-1] // heads)
        ref_k = rotate(ref_k.reshape(ref_shape), ref_rope).to(v.dtype)
        k = torch.cat((k, ref_k), 1)
        v = torch.cat((v, ref_v.reshape(ref_shape)), 1)
        if key_mask is not None:
            key_mask = torch.cat((key_mask, key_mask.new_ones(ref_shape[:2])), 1)
    flash = _flash_ok(v)
    mask = None if key_mask is None else key_mask[:, None, None, :]

    def attend(q, k, v, mask=None):
        if mask is None and flash:
            if flash_attn_f16in_func is not None and flash_attn_f16in_takes(q, k, v):
                return flash_attn_f16in_func(q, k, v)
            return flash_attn_func(q, k, v)
        return _sdpa(q, k, v, mask)

    if not ref_isolate or ref_start is None:
        out = attend(q, k, v, mask)
        if ref_gate is not None and (ref_kv is not None or ref_start is not None):
            s = ref_start if ref_kv is None else k.shape[1] - ref_k.shape[1]
            out = _gate_rows(out, q, k, v, mask, s, ref_gate, attend)
    else:
        s = ref_start
        main = attend(q[:, :s], k, v, mask)
        if ref_gate is not None:
            main = _gate_rows(main, q[:, :s], k, v, mask, s, ref_gate, attend)
        out = torch.cat((main, attend(q[:, s:], k[:, s:], v[:, s:])), 1)
    return out.flatten(2)


def _gate_rows(out, q, k, v, mask, s, ref_gate, attend):
    """``out`` with the query rows ``rows`` attended again over the keys before ``s``
    (image and text) and the reference keys ``keep`` alone. Gathers instead of a
    bias, so the call stays on FA2 with gradients."""
    rows, keep = ref_gate
    k = torch.cat((k[:, :s], k[:, s:].index_select(1, keep)), 1)
    v = torch.cat((v[:, :s], v[:, s:].index_select(1, keep)), 1)
    if mask is not None:
        mask = torch.cat((mask[..., :s], mask[..., s:].index_select(-1, keep)), -1)
    return out.index_copy(1, rows, attend(q.index_select(1, rows), k, v, mask))


class _TextBlock(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope, key_mask=None):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, key_mask))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, key_mask=None):
        x = self.proj(x.to(self.proj.weight.dtype))
        for block in self.blocks:
            x = block(x, rope, key_mask)
        return x


def _affine(y, scale, shift, ref_start):
    """``y·(1 + scale) + shift``. ``scale``/``shift`` are ``(B, D)``, or ``(B, 2, D)``
    with the second row applied to the tokens from ``ref_start`` on."""
    if scale.ndim == 2:
        return y * (1 + scale[:, None]) + shift[:, None]
    s = ref_start
    scale, shift = (1 + scale).unbind(1), shift.unbind(1)

    def seg(part, i):
        return part * scale[i][:, None] + shift[i][:, None]

    return torch.cat((seg(y[:, :s], 0), seg(y[:, s:], 1)), 1)


def _gated(y, gate, ref_start):
    if gate.ndim == 2:
        return y * gate[:, None]
    s = ref_start
    main, ref = gate.unbind(1)
    return torch.cat((y[:, :s] * main[:, None], y[:, s:] * ref[:, None]), 1)


class _Block(nn.Module):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)
        # On an int8 base, one linear holding the q, k and v rows (dit_int8.base).
        self.qkv = None
        # (k, v) export adapters, set by LoRANetwork.apply_to with ref_rank. A tuple,
        # so they stay outside the block's modules (the swapper moves every .weight).
        self.ref_bank = None

    def forward(
        self,
        x,
        rope,
        modulation,
        key_mask=None,
        ref_start=None,
        ref_isolate=False,
        ref_kv=None,
        kv_out=False,
        ref_gate=None,
    ):
        """``ref_start``: where the reference tokens (the sequence tail) begin, if any. ``modulation``
        is ``(B,6,D)``, or ``(B,6,2,D)`` with the second set modulating the
        reference tokens. ``ref_isolate``, ``ref_kv``: see ``_attention``. ``kv_out``
        also returns this block's keys (before RoPE) and values. With ``ref_bank`` the
        reference rows leave for ``ref_rows`` (``kv_out`` then is reference-only)."""
        if self.ref_bank is None or not (kv_out or ref_start is not None):
            return self.main_rows(
                x,
                rope,
                modulation,
                key_mask,
                ref_start,
                ref_isolate,
                ref_kv,
                kv_out,
                ref_gate=ref_gate,
            )
        s = 0 if kv_out else ref_start
        with torch.no_grad():
            ref, y, raw_k, raw_v = self.ref_rows(
                x[:, s:], rope[:, s:], modulation if kv_out else modulation[:, :, 1]
            )
        if kv_out:
            return (ref, *self._ref_export(y, raw_k, raw_v))
        main = self.main_rows(
            x[:, :s],
            rope[:, :s],
            modulation[:, :, 0],
            None if key_mask is None else key_mask[:, :s],
            ref_export=(y, raw_k, raw_v, rope[:, s:]),
            ref_gate=ref_gate,
        )
        return torch.cat((main, ref), 1)

    def _ref_export(self, y, raw_k, raw_v):
        """The keys and values the image rows read, ``ref_bank``'s deltas on. In the
        main graph while training: eager, k_norm's fp32 residuals ran a card out."""
        bank_k, bank_v = self.ref_bank
        k = self.k_norm(bank_k(y, raw_k).unflatten(-1, (self.heads, -1))).flatten(2)
        return k, bank_v(y, raw_v)

    def _qkv(self, y):
        """``proj.q`` / ``k`` / ``v`` on ``y``, adapters included."""
        if self.qkv is None:
            return self.proj["q"](y), self.proj["k"](y), self.proj["v"](y)
        rows = self.qkv(y).chunk(3, -1)
        return tuple(self.proj[name](y, row) for name, row in zip("qkv", rows))

    def main_rows(
        self,
        x,
        rope,
        modulation,
        key_mask=None,
        ref_start=None,
        ref_isolate=False,
        ref_kv=None,
        kv_out=False,
        ref_export=None,
        ref_gate=None,
    ):
        if ref_export is not None:
            ref_kv = (*self._ref_export(*ref_export[:3]), ref_export[3])
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = _affine(self.norm1(x), scale_a, shift_a, ref_start)
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q, k, v = self._qkv(y)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        # By keyword and only when set: probes that patch ``_attention`` keep the
        # nine-argument call.
        gate = {} if ref_gate is None else {"ref_gate": ref_gate}
        attn = _attention(
            q, k, v, rope, self.heads, key_mask, ref_start, ref_isolate, ref_kv, **gate
        )
        x = x + _gated(self.proj["o"](attn), gate_a, ref_start)
        y = _affine(self.norm2(x), scale_f, shift_f, ref_start)
        hidden, gate = self.proj["ffn_up_gate"](y.to(dtype)).chunk(2, -1)
        x = x + _gated(self.proj["ffn_down"](_situ(hidden, gate)), gate_f, ref_start)
        return (x, k, v) if kv_out else x

    def ref_rows(self, x, rope, modulation):
        """Isolated reference rows on the Base weights, past every patched adapter:
        ``(x, y, k, v)``, ``y`` the input of ``proj.k`` / ``proj.v`` and ``k`` / ``v``
        their outputs, which ``forward`` turns into the rows the image reads. Its own
        graph (``compile_blocks``): in one graph with the image rows a reference length
        ``n - ref_start`` raises Inductor's CantSplit."""

        def base(name, y):
            linear = self.proj[name]
            return type(linear).forward(linear, y)

        def heads(t):
            return t.unflatten(-1, (self.heads, -1))

        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        dtype = self.proj["q"].weight.dtype
        y = _affine(self.norm1(x), scale_a, shift_a, None).to(dtype)
        if self.qkv is None:
            q, raw_k, v = base("q", y), base("k", y), base("v", y)
        else:
            q, raw_k, v = self.qkv(y).chunk(3, -1)
        q = self.q_norm(heads(q)).flatten(2)
        k = self.k_norm(heads(raw_k)).flatten(2)
        attn = _attention(q, k, v, rope, self.heads)
        x = x + _gated(base("o", attn), gate_a, None)
        hidden = _affine(self.norm2(x), scale_f, shift_f, None).to(dtype)
        hidden, gate = base("ffn_up_gate", hidden).chunk(2, -1)
        x = x + _gated(base("ffn_down", _situ(hidden, gate)), gate_f, None)
        return x, y, raw_k, v


class RefCache:
    """The isolated reference stream at a fixed ``ref_timestep``, run once: it
    depends on the reference alone, not on the noisy latent, the text, the
    timestep or the CFG branch. Holds each block's reference keys (before RoPE:
    the reference's frame coordinate is the text length, which differs between
    branches) and values, rebuilt when the reference tokens change. It does not
    see a LoRA multiplier change, so it lives for one sampling loop."""

    def __init__(self):
        self.tokens = None
        self.kv = None
        self.builds = 0

    def lookup(self, layers, tokens, rope, mod):
        """Per-block ``(k, v)`` for ``tokens (B, n_ref, D)`` with their ``rope`` and
        reference modulation ``mod (B, 6, D)``."""
        if (
            self.tokens is None
            or self.tokens.shape != tokens.shape
            or not torch.equal(self.tokens, tokens)
        ):
            self.tokens, self.kv, x = tokens, [], tokens
            for layer in layers:
                x, k, v = layer(x, rope, mod, kv_out=True)
                self.kv.append((k, v))
            self.builds += 1
        return self.kv


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = (
            torch.arange(self.num_bands, device="cpu", dtype=torch.float32)
            / self.num_bands
        )
        return (1 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # .float() cannot recover frequencies already rounded by .to(bf16).
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Frequencies are derived constants; retain the key for checkpoint compatibility.
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device, ref_hw=None):
        """``(1, H·W + text_length [+ h·w], head_dim)``, fp32. Image tokens first.

        ``ref_hw = (h, w)`` adds a reference grid after the text tokens on the
        reserved ``r = 1`` axis, its ``(h, w)`` coordinates stretched onto the
        image's grid so each reference token sits over the image region it covers."""
        # Diffusers/Accelerate can install tensors directly, bypassing load hooks.
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image, text, reference.
        h, w = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        image = (
            torch.stack(
                (torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1
            )
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        parts = [image, text]
        if ref_hw is not None:
            rh, rw = ref_hw
            h, w = torch.meshgrid(
                (torch.arange(rh, device=device) + 0.5) * height / rh - 0.5,
                (torch.arange(rw, device=device) + 0.5) * width / rw - 0.5,
                indexing="ij",
            )
            ref = torch.stack(
                (torch.ones_like(h), torch.full_like(h, text_length), h, w), -1
            ).flatten(0, 1)
            ref[:, 2] -= (height - 1) / 2
            ref[:, 3] -= (width - 1) / 2
            parts.append(ref)
        phase = torch.cat(parts, 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(ModelMixin, ConfigMixin):
    """MOSS-Image DiT with training-compatible state-dict names and SDPA attention.

    The sequence is ``[image (H·W) | text (S)]`` through every block; only the
    image tokens reach the head.
    """

    @register_to_config
    def __init__(
        self,
        dim=3072,
        ffn_dim=8192,
        num_heads=24,
        num_layers=42,
        text_dim=4096,
        freq_dim=256,
        latent_channels=128,
        rope_base=10000.0,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError(
                "Require dim divisible by heads, head_dim divisible by 8, and even freq_dim."
            )
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList(
            [_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)]
        )
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.layer_hook = None
        self.ref_timestep = None
        self.ref_isolate = False
        self.ref_follows_skip = False
        self.ref_cache = None
        self.ref_scope = None
        # (rows, keep) index tensors for every block's main-row attention (``_attention``),
        # set per forward by the caller; None = the stock path.
        self.ref_gate = None

    def load_training_state_dict(self, state_dict, *, prefix=""):
        """Load a consolidated raw/EMA model mapping, strictly; no silent key drops."""
        if prefix:
            if any(not key.startswith(prefix) for key in state_dict):
                raise ValueError("All keys must start with the explicit prefix.")
            state_dict = {
                key[len(prefix) :]: value for key, value in state_dict.items()
            }
        return self.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------ pieces

    def time_conditioning(self, timestep, batch, base=False):
        """``timestep`` is ``1000·σ`` (scalar or ``(B,)``) → ``mod (B,6,D)``, ``head (B,2,D)``.
        ``base``: ``mod`` past the adapter patched on ``shared_block_modulation``."""
        times = torch.as_tensor(
            timestep, device=self.time_embedding[0].weight.device, dtype=torch.float32
        ).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.config.freq_dim // 2
        freq = 10000 ** (
            -torch.arange(half, device=times.device, dtype=torch.float64) / half
        )
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(
            embedding.to(self.time_embedding[0].weight.dtype)
        ).float()
        linear = self.shared_block_modulation
        time_mod = time.to(linear.weight.dtype)
        mod = type(linear).forward(linear, time_mod) if base else linear(time_mod)
        head = self.video_head_modulation(
            time.to(self.video_head_modulation.weight.dtype)
        )
        return mod.view(batch, 6, -1), head.view(batch, 2, -1)

    def ref_modulation(self, mod, ref_hw):
        """``(mod, n_ref)`` for ``run_layers``: ``n_ref`` the reference token count
        (0 without a reference); ``mod`` unchanged when the reference shares the
        image's timestep, else ``(B,6,2,D)`` with the ``ref_timestep`` set second."""
        if ref_hw is None:
            return mod, 0
        n_ref = ref_hw[0] * ref_hw[1]
        if self.ref_timestep is None:
            return mod, n_ref
        base = self.ref_scope is not None or self.layers[0].ref_bank is not None
        ref_mod, _ = self.time_conditioning(self.ref_timestep, mod.shape[0], base=base)
        return torch.stack((mod, ref_mod.to(mod.dtype)), 2), n_ref

    def embed_sample(self, latent, context, ref=None):
        """One sample: ``latent (1,C,H,W)``, ``context (1,S,text_dim)`` at its true
        length, optional clean reference latent ``ref (1,C,h,w)`` → ``x (1, H·W
        + S [+ h·w], D)``, ``rope`` to match."""
        _, _, height, width = latent.shape
        video = self._embed_latent(latent)
        ref_hw = None if ref is None else tuple(ref.shape[-2:])
        # Match the reference activation dtype under fp32 weights + autocast.
        rope = self.rope(height, width, context.shape[1], latent.device, ref_hw).to(
            video.dtype
        )
        n_img, n_text = video.shape[1], context.shape[1]
        parts = [video, self.text_embedder(context, rope[:, n_img : n_img + n_text])]
        if ref is not None:
            parts.append(self._embed_latent(ref))
        return torch.cat(parts, 1), rope

    def _embed_latent(self, latent):
        tokens = latent.flatten(2).transpose(1, 2)
        return self.video_embedder(
            tokens.to(self.video_embedder.weight.dtype).contiguous()
        )

    def run_layers(
        self,
        x,
        rope,
        mod,
        skip_layers=frozenset(),
        key_mask=None,
        n_img=None,
        n_ref=0,
    ):
        """The 42 blocks. ``skip_layers``: one frozenset for the whole batch, or a
        sequence of frozensets, one per sample (rows skipping a block keep ``x``).
        ``n_img`` (leading image tokens) is what ``layer_hook`` sees. ``n_ref``
        (the trailing reference tokens) and ``mod`` come from ``ref_modulation``;
        ``ref_isolate`` applies to those tokens, and then a skipped block still runs
        them (``_ref_only``) unless ``ref_follows_skip``. The blocks get the
        reference's start as one int, ``ref_start``. With ``ref_cache`` set and the
        reference isolated at a fixed ``ref_timestep``, the reference tokens leave
        ``x`` and every block reads them from the cache (the output then ends at
        the text)."""
        per_sample = not isinstance(skip_layers, (frozenset, set))
        ref_start = x.shape[1] - n_ref if n_ref else None
        iso = self.ref_isolate and ref_start is not None
        ref_runs = iso and not self.ref_follows_skip
        cached = None
        if ref_runs and self.ref_cache is not None and self.ref_timestep is not None:
            s = ref_start
            ref_rope = rope[:, s:]
            with (self.ref_scope or nullcontext)():
                cached = self.ref_cache.lookup(
                    self.layers, x[:, s:], ref_rope, mod[:, :, 1]
                )
            x, rope, mod = x[:, :s], rope[:, :s], mod[:, :, 0]
            key_mask = None if key_mask is None else key_mask[:, :s]
            ref_start, iso, ref_runs = None, False, False

        gate = {} if self.ref_gate is None else {"ref_gate": self.ref_gate}

        def kv(index, rows=None):
            if cached is None:
                return gate
            k, v = cached[index]
            if rows is None:
                return {"ref_kv": (k, v, ref_rope), **gate}
            return {"ref_kv": (k[rows], v[rows], ref_rope[rows]), **gate}

        hook = self.layer_hook
        for index, layer in enumerate(self.layers):
            if per_sample:
                active = [
                    b for b, skips in enumerate(skip_layers) if index not in skips
                ]
                if len(active) < x.shape[0]:
                    if ref_runs:
                        idle = [b for b in range(x.shape[0]) if b not in active]
                        x = self._ref_only(
                            layer,
                            x,
                            rope,
                            mod,
                            ref_start,
                            torch.tensor(idle, device=x.device),
                        )
                    if not active:
                        continue
                    rows = torch.tensor(active, device=x.device)
                    sub = layer(
                        x[rows],
                        rope[rows],
                        mod[rows],
                        None if key_mask is None else key_mask[rows],
                        ref_start,
                        iso,
                        **kv(index, rows),
                    )
                    if hook is not None:
                        sub = hook(index, sub, n_img)
                    x = x.index_copy(0, rows, sub)
                    continue
            elif index in skip_layers:
                if ref_runs:
                    x = self._ref_only(layer, x, rope, mod, ref_start)
                continue
            x = layer(x, rope, mod, key_mask, ref_start, iso, **kv(index))
            if hook is not None:
                x = hook(index, x, n_img)
        return x

    @staticmethod
    def _ref_only(layer, x, rope, mod, ref_start, rows=None):
        """``x`` with the reference tokens of ``rows`` (None = all) through ``layer``
        on their own: under ``ref_isolate`` what the whole block gives them."""
        s = ref_start
        if mod.ndim == 4:
            mod = mod[:, :, 1]
        sub, sub_rope, sub_mod = (
            (x, rope, mod) if rows is None else (x[rows], rope[rows], mod[rows])
        )
        sub = torch.cat((sub[:, :s], layer(sub[:, s:], sub_rope[:, s:], sub_mod)), 1)
        return sub if rows is None else x.index_copy(0, rows, sub)

    def head(self, features, head_mod, height, width):
        """Image-token hidden state after the last block ``(B, H·W, D)`` →
        velocity ``(B, C, H, W)``. Spectrum forecasts ``features`` and calls this."""
        x = self.final_norm_video(features.to(self.final_norm_video.weight.dtype))
        x = x * (1 + head_mod[:, 1, None]) + head_mod[:, 0, None]
        x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
        return x.transpose(1, 2).reshape(x.shape[0], -1, height, width)

    # ---------------------------------------------------------------- forwards

    def _check_inputs(self, hidden_states, encoder_hidden_states, mask):
        if (
            hidden_states.ndim != 4
            or hidden_states.shape[1] != self.config.latent_channels
        ):
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch = hidden_states.shape[0]
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        if mask is None:
            mask = torch.ones(
                encoder_hidden_states.shape[:2],
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all(
            (mask == 0) | (mask == 1)
        ):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        if not torch.all(mask.bool().any(1)):
            raise ValueError("Each sample needs at least one valid text token.")
        return mask.bool()

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        return_dict=True,
        skip_layers=None,
        return_features=False,
        ref=None,
    ):
        """The reference per-sample loop. ``return_features`` returns the
        image-token hidden state ``(B, H·W, D)`` before ``head`` instead.
        ``ref (B,C,h,w)``: clean reference latents (``embed_sample``)."""
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        mask = self._check_inputs(
            hidden_states, encoder_hidden_states, encoder_attention_mask
        )
        batch, _, height, width = hidden_states.shape
        mod, head = self.time_conditioning(timestep, batch)
        ref_hw = None if ref is None else tuple(ref.shape[-2:])
        mod, n_ref = self.ref_modulation(mod, ref_hw)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1, mask[index]]
            x, rope = self.embed_sample(
                hidden_states[index : index + 1],
                context,
                None if ref is None else ref[index : index + 1],
            )
            x = self.run_layers(
                x,
                rope,
                mod[index : index + 1],
                skip_layers,
                n_img=height * width,
                n_ref=n_ref,
            )
            features = x[:, : height * width]
            if return_features:
                outputs.append(features)
            else:
                outputs.append(
                    self.head(features, head[index : index + 1], height, width)
                )
        sample = torch.cat(outputs)
        if return_features:
            return sample
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)

    def forward_batched(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        skip_layers=None,
        return_features=False,
        ref=None,
    ):
        """All samples in one forward. Text is right-padded to the longest valid
        length and masked out of every key set (text refiner and joint blocks), so
        padding never reaches a valid token. Matches ``forward`` within bf16 noise
        (the masked SDPA picks a different kernel).

        ``skip_layers``: None, one iterable for the whole batch, or a list with one
        iterable per sample (e.g. ``[(), (22,)]`` for cond + skip-22 uncond).
        Returns the velocity ``(B, C, H, W)`` (or features ``(B, H·W, D)``).
        """
        depth = len(self.layers)
        mask = self._check_inputs(
            hidden_states, encoder_hidden_states, encoder_attention_mask
        )
        batch, _, height, width = hidden_states.shape
        if (
            skip_layers is not None
            and len(skip_layers)
            and not isinstance(skip_layers[0], int)
        ):
            if len(skip_layers) != batch:
                raise ValueError("Per-sample skip_layers needs one entry per sample.")
            skips = [_validate_skip_layers(s, depth) for s in skip_layers]
            if len(set(skips)) == 1:
                skips = skips[0]
        else:
            skips = _validate_skip_layers(skip_layers, depth)
        mod, head = self.time_conditioning(timestep, batch)
        n_img = height * width
        ref_hw = None if ref is None else tuple(ref.shape[-2:])
        mod, n_ref = self.ref_modulation(mod, ref_hw)
        lengths = mask.sum(1)
        s_max = int(lengths.max())
        device = hidden_states.device

        context = encoder_hidden_states.new_zeros(
            batch, s_max, encoder_hidden_states.shape[-1]
        )
        text_mask = torch.zeros(batch, s_max, dtype=torch.bool, device=device)
        ropes = []
        for b in range(batch):
            n = int(lengths[b])
            context[b, :n] = encoder_hidden_states[b, mask[b]]
            text_mask[b, :n] = True
            rope = self.rope(height, width, n, device, ref_hw)
            if n < s_max:
                # Pad rows sit between the text and the reference.
                pad = rope.new_zeros(1, s_max - n, rope.shape[-1])
                rope = torch.cat((rope[:, : n_img + n], pad, rope[:, n_img + n :]), 1)
            ropes.append(rope)
        video = self._embed_latent(hidden_states)
        rope = torch.cat(ropes).to(video.dtype)
        text = self.text_embedder(context, rope[:, n_img : n_img + s_max], text_mask)
        parts = [video, text] if ref is None else [video, text, self._embed_latent(ref)]
        x = torch.cat(parts, 1)
        ones = text_mask.new_ones
        key_mask = torch.cat((ones(batch, n_img), text_mask, ones(batch, n_ref)), 1)
        if bool(text_mask.all()):
            key_mask = None
        x = self.run_layers(x, rope, mod, skips, key_mask, n_img, n_ref)
        features = x[:, :n_img]
        if return_features:
            return features
        return self.head(features, head, height, width)


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """Euler integration using the training project's V2, nearest-grid schedule."""

    @register_to_config
    def __init__(self, num_train_timesteps=1000, shift=5.0, training_shift=1.0):
        super().__init__(num_train_timesteps=num_train_timesteps, shift=1.0)
        self.register_to_config(shift=shift, training_shift=training_shift)

    def set_timesteps(self, num_inference_steps=50, device=None):
        if (
            num_inference_steps < 1
            or self.config.shift <= 0
            or self.config.training_shift <= 0
        ):
            raise ValueError("Steps and shifts must be positive.")
        sigmas = snapped_sigmas(
            num_inference_steps,
            self.config.shift,
            self.config.training_shift,
            self.config.num_train_timesteps,
        )
        # Our sigmas have already been shifted and snapped. Avoid a second shift.
        self._shift = 1.0
        super().set_timesteps(device=device, sigmas=sigmas.tolist())


def snapped_sigmas(num_steps, shift=DEFAULT_SHIFT, training_shift=1.0, total=1000):
    """The inference σ grid (descending, no terminal 0): ``shift``-warped linspace
    snapped to the nearest σ of the training grid, duplicates collapsed, so a
    high step count can return fewer than ``num_steps`` values."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(num_steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(),
        lower,
        upper,
    )
    return torch.unique_consecutive(training[nearest])


# ------------------------------------------------------------- VAE boundary


def _bn_stats(vae, like):
    bn, eps = vae.bn, vae.config.batch_norm_eps
    mean = bn.running_mean.to(like).view(1, -1, 1, 1)
    std = (bn.running_var + eps).sqrt().to(like).view(1, -1, 1, 1)
    return mean, std


def patchify(z):
    """``(B,32,H,W)`` → ``(B,128,H/2,W/2)``, FLUX.2's ``_patchify_latents``."""
    b, c, h, w = z.shape
    return (
        z.view(b, c, h // 2, 2, w // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, c * 4, h // 2, w // 2)
    )


def unpatchify(z):
    """Inverse of :func:`patchify`: ``(B,128,h,w)`` → ``(B,32,2h,2w)``."""
    b, c, h, w = z.shape
    return (
        z.reshape(b, c // 4, 2, 2, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(b, c // 4, 2 * h, 2 * w)
    )


def unpack_latents(z, vae):
    """DiT latent ``(B,128,h,w)`` → FLUX.2 VAE latent ``(B,32,2h,2w)``, in the VAE's
    dtype: BN de-normalise, then 2×2 un-patchify (the reference decode)."""
    z = z.to(dtype=vae.dtype)
    mean, std = _bn_stats(vae, z)
    return unpatchify(z * std + mean)


def pack_latents(z, vae):
    """FLUX.2 VAE latent ``(B,32,H,W)`` → DiT latent ``(B,128,H/2,W/2)``: the exact
    inverse of ``unpack_latents`` (``patchify`` + BN normalise)."""
    z = patchify(z)
    mean, std = _bn_stats(vae, z)
    return (z - mean) / std


@torch.no_grad()
def decode(z, vae):
    """DiT latent → image tensor in ``[-1, 1]``, ``(B,3,16h,16w)``."""
    return vae.decode(unpack_latents(z, vae), return_dict=False)[0].clamp(-1, 1)


@torch.no_grad()
def encode(pixels, vae):
    """Image tensor in ``[-1, 1]`` ``(B,3,H,W)``, sides multiples of 16 → DiT latent
    ``(B,128,H/16,W/16)``, fp32. Distribution mode, as FLUX.2 encodes references."""
    dist = vae.encode(pixels.to(device=vae.device, dtype=vae.dtype)).latent_dist
    return pack_latents(dist.mode(), vae).float()
