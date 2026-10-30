"""int8 ConvRot base linears for LoRA training on a frozen DiT.

A converted ``nn.Linear`` runs its forward ``Y = X Wᵀ`` through
``comfy_kitchen.int8_linear`` (per-256-column Hadamard rotation, per-token int8
activations, per-row int8 weight). Its input gradient ``dX = dY W`` runs either in
bf16 against the kept weight (mode ``fwd``) or through the same kernel against a
second int8 copy of ``Wᵀ`` rotated along N (mode ``full``, the weight is dropped).
The base weight gradient is never computed. Mode ``infer`` keeps only the forward copy,
for generation on the int8 base.

``convert_(module, mode)`` swaps the class in place, so LoRA modules that hold a
reference to the original module object keep working. The forward copy is round to
nearest unless ``convert_dit_`` is given a calibrated file (``dit_gptq.py``); the
backward copy ``wq_t`` is always round to nearest.
"""

from __future__ import annotations

import hashlib
import json
import re

import comfy_kitchen
import torch
import torch.nn as nn
from safetensors import safe_open

from library.anima.weights import _SELF_ATTN_QKV_ORDER
from library.moss.int8 import INT8_MAX, rotate, row_scale

GROUP = 256
MODES = ("fwd", "full", "infer")

# per block: self qkv/out, cross q/out, MLP 1/2. Cross kv (M = context length) and
# adaLN (M = batch) stay bf16; phase 0 measured int8 slower there.
CONVERT_SUFFIXES = (
    "self_attn.qkv_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.output_proj",
    "mlp.layer1",
    "mlp.layer2",
)
_DIT_BLOCK = re.compile(r"^blocks\.\d+\.")


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(int8 (N, K) rotated along K, fp32 (N, 1) row scale)`` on ``w``'s device."""
    w_rot = rotate(w.float())
    scale = row_scale(w_rot)
    q = (w_rot / scale[:, None]).round().clamp(-INT8_MAX - 1, INT8_MAX)
    return q.to(torch.int8), scale[:, None].contiguous()


@torch.library.custom_op("int8_train::mm", mutates_args=())
def int8_mm(x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """``x (…, K) @ dequant(wq)ᵀ`` → bf16 ``(…, N)``."""
    y = comfy_kitchen.int8_linear(
        x.reshape(-1, x.shape[-1]).to(torch.bfloat16),
        wq,
        s,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=GROUP,
    )
    return y.reshape(*x.shape[:-1], wq.shape[0])


@int8_mm.register_fake
def _(x, wq, s):
    return x.new_empty((*x.shape[:-1], wq.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("int8_train::linear_full", mutates_args=())
def linear_full(
    x: torch.Tensor,
    wq: torch.Tensor,
    s: torch.Tensor,
    wq_t: torch.Tensor,
    s_t: torch.Tensor,
) -> torch.Tensor:
    return int8_mm(x, wq, s)


@linear_full.register_fake
def _(x, wq, s, wq_t, s_t):
    return x.new_empty((*x.shape[:-1], wq.shape[0]), dtype=torch.bfloat16)


def _full_setup(ctx, inputs, output):
    x, _wq, _s, wq_t, s_t = inputs
    ctx.save_for_backward(wq_t, s_t)
    ctx.x_dtype = x.dtype


def _full_backward(ctx, dy):
    wq_t, s_t = ctx.saved_tensors
    dx = None
    if ctx.needs_input_grad[0]:
        dx = int8_mm(dy, wq_t, s_t).to(ctx.x_dtype)
    return dx, None, None, None, None


linear_full.register_autograd(_full_backward, setup_context=_full_setup)


@torch.library.custom_op("int8_train::linear_fwd", mutates_args=())
def linear_fwd(
    x: torch.Tensor, wq: torch.Tensor, s: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    return int8_mm(x, wq, s)


@linear_fwd.register_fake
def _(x, wq, s, w):
    return x.new_empty((*x.shape[:-1], wq.shape[0]), dtype=torch.bfloat16)


def _fwd_setup(ctx, inputs, output):
    x, _wq, _s, w = inputs
    ctx.save_for_backward(w)
    ctx.x_dtype = x.dtype


def _fwd_backward(ctx, dy):
    (w,) = ctx.saved_tensors
    dx = None
    if ctx.needs_input_grad[0]:
        dx = (dy.to(w.dtype) @ w).to(ctx.x_dtype)
    return dx, None, None, None


linear_fwd.register_autograd(_fwd_backward, setup_context=_fwd_setup)


class Int8TrainLinear(nn.Linear):
    """Only ever made by :func:`convert_`; never constructed directly."""

    int8_mode: str

    def _apply(self, fn, recurse=True):
        # the trainer's unet.to(dtype=bf16) would round the fp32 row scales; keep
        # them fp32 and follow wq (int8, so dtype casts skip it) to its device
        scales = {k: v for k, v in self._buffers.items() if k in ("s", "s_t")}
        super()._apply(fn, recurse)
        for k, v in scales.items():
            self._buffers[k] = v.to(self.wq.device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.int8_mode == "infer":
            return int8_mm(x, self.wq, self.s)
        if self.int8_mode == "full":
            return linear_full(x, self.wq, self.s, self.wq_t, self.s_t)
        return linear_fwd(x, self.wq, self.s, self.weight)


def ckpt_keys(name: str) -> list[str]:
    """A converted module's weight keys in the checkpoint (``net.`` prefix, the fused
    ``qkv_proj`` split back into q / k / v)."""
    if name.endswith("self_attn.qkv_proj"):
        base = name.removesuffix("qkv_proj")
        return [f"net.{base}{p}.weight" for p in _SELF_ATTN_QKV_ORDER]
    return [f"net.{name}.weight"]


def load_int8_file(path: str) -> tuple[dict, dict]:
    """``({module name: (wq, s)}, metadata)`` from a ``dit_gptq.py`` file, for the
    converted module names (q / k / v rows concatenated into ``qkv_proj``)."""
    out = {}
    with safe_open(path, framework="pt") as f:
        meta = dict(f.metadata() or {})
        keys = set(f.keys())
        names = json.loads(meta["int8_modules"])
        for name in names:
            ks = ckpt_keys(name)
            missing = [k for k in ks if k not in keys]
            if missing:
                raise ValueError(f"{path}: no {missing[0]}")
            wq = torch.cat([f.get_tensor(k) for k in ks])
            s = torch.cat(
                [f.get_tensor(k.removesuffix("weight") + "weight_scale") for k in ks]
            )
            out[name] = (wq, s.float().reshape(-1, 1))
    return out, meta


def convert_(module: nn.Linear, mode: str, forward_copy=None) -> None:
    """``forward_copy``: a calibrated ``(wq, s)`` in place of round to nearest."""
    if mode not in MODES:
        raise ValueError(f"int8 mode {mode!r}, expected one of {MODES}")
    if module.bias is not None:
        raise ValueError("int8 base linears are bias-free")
    w = module.weight.detach()
    if forward_copy is None:
        wq, s = quantize(w)
    else:
        wq, s = (t.to(w.device) for t in forward_copy)
    module.register_buffer("wq", wq, persistent=False)
    module.register_buffer("s", s, persistent=False)
    if mode == "full":
        wq_t, s_t = quantize(w.t().contiguous())
        module.register_buffer("wq_t", wq_t, persistent=False)
        module.register_buffer("s_t", s_t, persistent=False)
    if mode != "fwd":
        del module.weight
    module.int8_mode = mode
    module.__class__ = Int8TrainLinear


def convert_targets(dit: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Every DiT block's large linears (not the llm_adapter's)."""
    return [
        (name, mod)
        for name, mod in dit.named_modules()
        if isinstance(mod, nn.Linear)
        and _DIT_BLOCK.match(name)
        and name.endswith(CONVERT_SUFFIXES)
    ]


def base_fingerprint(dit: nn.Module) -> str:
    """sha256 over the bf16 bytes of the first and last target weights, before any
    conversion: an int8 file applies only to the checkpoint it was made from."""
    h = hashlib.sha256()
    targets = convert_targets(dit)
    for name, mod in (targets[0], targets[-1]):
        h.update(name.encode())
        h.update(
            mod.weight.detach()
            .to(torch.bfloat16)
            .cpu()
            .view(torch.int16)
            .numpy()
            .tobytes()
        )
    return h.hexdigest()


def convert_dit_(dit: nn.Module, mode: str, int8_file: str | None = None) -> list[str]:
    """Convert the DiT's large linears; returns the converted names. ``int8_file``
    supplies the forward copies, and must cover every target and match the loaded
    checkpoint's ``base_fingerprint``."""
    copies = {}
    if int8_file:
        copies, meta = load_int8_file(int8_file)
        if meta.get("base_fingerprint") != base_fingerprint(dit):
            raise ValueError(
                f"{int8_file} was made from {meta.get('source')}, not the loaded DiT"
            )
    names = []
    for name, mod in convert_targets(dit):
        if int8_file and name not in copies:
            raise ValueError(f"{int8_file} has no forward copy for {name}")
        convert_(mod, mode, copies.get(name))
        names.append(name)
    return names
