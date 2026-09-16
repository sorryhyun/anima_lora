"""Ext-gated LoRA on the LLM Adapter's per-block Linears (plan3, rung 2-ii).

The vocab-pack line (plan.md) trains ext *rows* only and leaves the adapter
frozen. §8/§9 of ``_archive/cjk_aware_anima/reports/0827_names_synth.md`` showed that a rare kanji name whose whole
neighbourhood is new rows never composes — the adapter's self-attention was
pretrained on EN pieces and cannot be asked to compose new rows through frozen
weights. This module adds the capacity: a LoRA on the neighbour-mixing path
(self-attn q/k/v/o) plus the query side of cross-attn.

The EN bit-exactness guarantee is kept **by construction**: the delta is
multiplied by a per-sequence gate ``g ∈ {0, 1}`` = "this sequence contains at
least one ext id (>= 32128)". A pure-EN prompt has ``g = 0`` on every row and
the LoRA branch is never even evaluated for it.

Wiring is hook-based, never a ``forward`` override, so the same module shape
ships through the ComfyUI Anima Adapter Loader (``forward_hook``-not-override
invariant, see its CLAUDE.md):

* ``forward_pre_hook`` on the adapter reads ``target_input_ids`` and stashes
  the gate ``[B, 1, 1]``;
* ``forward_hook`` on every targeted ``nn.Linear`` adds ``g · (B A x) · α/r``.

State-dict keys follow the repo's kohya convention so the existing key parser
consumes them unchanged::

    lora_unet_llm_adapter_blocks_{i}_self_attn_q_proj.lora_down.weight
    lora_unet_llm_adapter_blocks_{i}_self_attn_q_proj.lora_up.weight
    lora_unet_llm_adapter_blocks_{i}_self_attn_q_proj.alpha
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from scripts.distill_cjk.ext_table import T5_TABLE_SIZE

KEY_PREFIX = "lora_unet_llm_adapter_"

# --adapter_lora_targets tokens → per-block Linear paths.
TARGET_GROUPS: dict[str, tuple[str, ...]] = {
    "self_q": ("self_attn.q_proj",),
    "self_k": ("self_attn.k_proj",),
    "self_v": ("self_attn.v_proj",),
    "self_o": ("self_attn.o_proj",),
    "self_qkvo": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ),
    "cross_q": ("cross_attn.q_proj",),
    "cross_k": ("cross_attn.k_proj",),
    "cross_v": ("cross_attn.v_proj",),
    "cross_o": ("cross_attn.o_proj",),
    "cross_kv": ("cross_attn.k_proj", "cross_attn.v_proj"),
    "mlp": ("mlp.0", "mlp.2"),
}
DEFAULT_TARGETS = "self_qkvo,cross_q"


def parse_targets(spec: str) -> tuple[str, ...]:
    """``"self_qkvo,cross_q"`` → ordered, de-duplicated per-block Linear paths."""
    out: list[str] = []
    for tok in str(spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in TARGET_GROUPS:
            raise ValueError(
                f"unknown adapter LoRA target {tok!r} (known: {', '.join(TARGET_GROUPS)})"
            )
        for path in TARGET_GROUPS[tok]:
            if path not in out:
                out.append(path)
    if not out:
        raise ValueError("--adapter_lora_targets selected nothing")
    return tuple(out)


def lora_name(module_path: str) -> str:
    """``blocks.0.self_attn.q_proj`` (adapter-relative) → kohya lora_name."""
    return KEY_PREFIX + module_path.replace(".", "_")


def module_path(name: str) -> str:
    """Inverse of :func:`lora_name` for the adapter's own naming.

    ``blocks_{i}_self_attn_q_proj`` is unambiguous because every adapter
    Linear path is ``blocks.<int>.<self_attn|cross_attn>.<x>_proj`` or
    ``blocks.<int>.mlp.<0|2>`` — the only ``_`` that is *not* a ``.`` is the
    one inside ``self_attn`` / ``cross_attn`` / ``{q,k,v,o}_proj``.
    """
    if not name.startswith(KEY_PREFIX):
        raise ValueError(f"not an adapter LoRA key: {name!r}")
    rest = name[len(KEY_PREFIX) :]
    for keep in ("self_attn", "cross_attn", "q_proj", "k_proj", "v_proj", "o_proj"):
        rest = rest.replace(keep, keep.replace("_", "\x00"))
    return rest.replace("_", ".").replace("\x00", "_")


class _LoRALinear(nn.Module):
    """One LoRA pair. ``down`` is Kaiming-uniform, ``up`` is zero ⇒ delta = 0 at init."""

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.rank = rank
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    @property
    def scale(self) -> float:
        return float(self.alpha) / self.rank

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        w_dtype = self.lora_down.weight.dtype
        return self.lora_up(self.lora_down(x.to(w_dtype))) * self.scale


class AdapterLoRA(nn.Module):
    """Gated LoRA over an :class:`LLMAdapter`'s targeted Linears.

    Construct, then :meth:`attach` to the adapter (hooks only — the adapter's
    modules and forward are untouched). ``parameters()`` yields exactly the
    LoRA leaves, ``state_dict()`` is loader-shaped (see module docstring).
    """

    def __init__(
        self,
        adapter: nn.Module,
        *,
        rank: int = 16,
        alpha: float | None = None,
        targets: Iterable[str] | str = DEFAULT_TARGETS,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if isinstance(targets, str):
            targets = parse_targets(targets)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        self.targets = tuple(targets)
        self.multiplier = float(multiplier)
        self.loras = nn.ModuleDict()
        self._paths: dict[str, str] = {}  # lora_name → adapter-relative path
        for i, block in enumerate(adapter.blocks):
            for path in self.targets:
                lin = block.get_submodule(path)
                if not isinstance(lin, nn.Linear):
                    raise TypeError(
                        f"blocks.{i}.{path} is {type(lin).__name__}, not Linear"
                    )
                full = f"blocks.{i}.{path}"
                key = lora_name(full)
                self.loras[key] = _LoRALinear(
                    lin.in_features, lin.out_features, self.rank, self.alpha
                )
                self._paths[key] = full
        # Per-sequence gate, [B, 1, 1]; None = not inside an adapter forward.
        self._gate: torch.Tensor | None = None
        self._handles: list = []
        # Plain attribute, NOT a registered submodule: assigning an nn.Module
        # via normal setattr would make the whole adapter (and its ext table)
        # part of ``self.parameters()`` and land it in the LoRA param group.
        object.__setattr__(self, "_adapter", None)

    # ---- gate ---------------------------------------------------------------

    @staticmethod
    def gate_from_ids(ids: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` ids → ``[B, 1, 1]`` float gate: 1 if any id is an ext row."""
        return (ids >= T5_TABLE_SIZE).any(dim=-1).to(torch.float32)[:, None, None]

    def _pre_hook(self, module, args, kwargs):
        ids = kwargs.get("target_input_ids")
        if ids is None and len(args) >= 2:
            ids = args[1]
        self._gate = None if ids is None else self.gate_from_ids(ids)

    def _post_hook(self, module, args, output):
        self._gate = None

    def _make_linear_hook(self, key: str):
        lora = self.loras[key]

        def hook(module, args, output):
            g = self._gate
            if g is None or self.multiplier == 0.0:
                return output
            if not bool(g.any()):
                return output  # pure-EN batch: the LoRA branch never runs
            x = args[0]
            d = lora.delta(x) * self.multiplier
            return output + (g.to(output.dtype) * d.to(output.dtype))

        return hook

    # ---- wiring -------------------------------------------------------------

    def attach(self, adapter: nn.Module) -> "AdapterLoRA":
        self.detach()
        object.__setattr__(self, "_adapter", adapter)
        self._handles.append(
            adapter.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
        )
        self._handles.append(adapter.register_forward_hook(self._post_hook))
        for key, path in self._paths.items():
            lin = adapter.get_submodule(path)
            self._handles.append(lin.register_forward_hook(self._make_linear_hook(key)))
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        object.__setattr__(self, "_adapter", None)
        self._gate = None

    # ---- io -----------------------------------------------------------------

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        """Flat, loader-shaped ``{lora_name.lora_down.weight, …}`` tensors (CPU)."""
        out = {}
        for key, lora in self.loras.items():
            out[f"{key}.lora_down.weight"] = (
                lora.lora_down.weight.detach().cpu().contiguous()
            )
            out[f"{key}.lora_up.weight"] = (
                lora.lora_up.weight.detach().cpu().contiguous()
            )
            out[f"{key}.alpha"] = lora.alpha.detach().cpu().clone()
        return out

    def metadata(self) -> dict[str, str]:
        return {
            "ss_network_module": "scripts.distill_cjk.adapter_lora",
            "ss_network_dim": str(self.rank),
            "ss_network_alpha": str(self.alpha),
            "ss_adapter_lora_targets": ",".join(self.targets),
            "ss_adapter_lora_gate": f"any(ids >= {T5_TABLE_SIZE})",
        }

    def save(self, path) -> None:
        from safetensors.torch import save_file

        save_file(self.export_state_dict(), str(path), metadata=self.metadata())

    @classmethod
    def load(
        cls, adapter: nn.Module, path, *, multiplier: float = 1.0
    ) -> "AdapterLoRA":
        """Rebuild from a sidecar (rank/alpha/targets read off the tensors)."""
        from safetensors import safe_open

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        names = sorted(
            {
                (k.rsplit(".", 2)[0] if k.endswith(".weight") else k.rsplit(".", 1)[0])
                for k in tensors
                if k.startswith(KEY_PREFIX)
            }
        )
        if not names:
            raise ValueError(f"{path}: no {KEY_PREFIX}* keys")
        paths = [module_path(n) for n in names]
        # Per-block Linear paths (strip "blocks.<i>.")
        targets: list[str] = []
        for p in paths:
            rel = p.split(".", 2)[2]
            if rel not in targets:
                targets.append(rel)
        if meta.get("ss_adapter_lora_targets"):
            stamped = meta["ss_adapter_lora_targets"].split(",")
            if set(stamped) == set(targets):
                targets = stamped  # keep the training-time order
        first = names[0]
        rank = int(tensors[f"{first}.lora_down.weight"].shape[0])
        alpha = float(tensors.get(f"{first}.alpha", torch.tensor(float(rank))))
        obj = cls(
            adapter, rank=rank, alpha=alpha, targets=targets, multiplier=multiplier
        )
        for key, lora in obj.loras.items():
            lora.lora_down.weight.data.copy_(tensors[f"{key}.lora_down.weight"])
            lora.lora_up.weight.data.copy_(tensors[f"{key}.lora_up.weight"])
            if f"{key}.alpha" in tensors:
                lora.alpha.copy_(tensors[f"{key}.alpha"])
        ref = next(adapter.parameters())
        obj.to(device=ref.device)
        return obj.attach(adapter)
