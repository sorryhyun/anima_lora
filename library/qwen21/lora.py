"""A LoRA whose parameters live outside the blocks being swapped.

``peft``'s ``add_adapter`` nests ``lora_A``/``lora_B`` inside the module they
adapt, so on this model they would land inside ``transformer_blocks`` — and
``ModelOffloader`` swaps a block by walking its ``named_modules()`` and moving
every ``.weight`` it finds. The trainable weights would ride to the CPU while
their ``.grad`` stayed on the card, and the optimizer would see the two on
different devices.

The trainer's own adapters avoid this by construction (``networks/`` builds a
separate ``LoRANetwork`` and monkey-patches the target's ``forward``), and that
is what this is: the down/up pairs are held by :class:`LoRANetwork`, which stays
resident, and each target ``nn.Linear`` gets an instance-level ``forward`` that
adds their output. Patching ``forward`` rather than replacing the module also
keeps the swapper's name→shape matching intact, and survives the swap because
only ``weight.data`` moves.
"""

from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.qwen21.requests import DEFAULT_TARGETS


class LoRAAdapter(nn.Module):
    """``scale * up(down(x))``, added to the base linear's output.

    Training builds this fp32 (the master weights AdamW updates — bf16's
    8-bit mantissa rounds away updates below ~0.4 % of a weight, and at
    lr 1e-4 the kaiming-scaled ``down`` sits right at that edge) and saves in
    ``TrainRequest.lora_dtype`` (bf16 by default); ``load_network`` rebuilds
    in the saved dtype, which is all inference needs. The rank
    GEMMs run in the *model's* dtype: ``x`` and both weights are cast to
    ``base_out.dtype`` first, so an fp32 adapter never lifts a ``(T, 4096)``
    activation to fp32. Same policy as ``networks/lora_modules/base.py``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.multiplier = 1.0

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0:
            return base_out
        work = base_out.dtype
        lx = F.linear(x.to(work), self.down.weight.to(work))
        delta = F.linear(lx, self.up.weight.to(work))
        return base_out + delta * (self.scale * self.multiplier)


class LoRANetwork(nn.Module):
    """The adapters for one model, held outside it.

    ``apply_to`` patches the targets and must run before any ``torch.compile``
    of the blocks, the same ordering ``build_anima`` encodes for Anima.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: float | None = None,
        targets: str = DEFAULT_TARGETS,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        pattern = re.compile(targets)
        self.rank = rank
        self.alpha = rank if alpha is None else alpha
        self.adapters = nn.ModuleDict()
        self._targets: list[tuple[str, nn.Linear]] = []

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or not pattern.fullmatch(name):
                continue
            self.adapters[name.replace(".", "_")] = LoRAAdapter(
                module.in_features, module.out_features, rank, self.alpha, dtype
            )
            self._targets.append((name, module))

        if not self._targets:
            raise ValueError(f"no nn.Linear matched {targets!r}")
        self._applied = False

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def apply_to(self) -> int:
        """Patch every target's ``forward``. Returns the count."""
        if self._applied:
            raise RuntimeError("already applied")
        for name, module in self._targets:
            adapter = self.adapters[name.replace(".", "_")]
            module.forward = _patched_forward(module, adapter)
        self._applied = True
        return len(self._targets)

    def set_multiplier(self, multiplier: float) -> None:
        """Scale every adapter. ``0.0`` is the base model exactly — the adapters
        short-circuit rather than adding a zeroed delta, so an A/B needs no
        reload and no second model."""
        for adapter in self.adapters.values():
            adapter.multiplier = multiplier

    def restore(self) -> None:
        for _name, module in self._targets:
            module.__dict__.pop("forward", None)
        self._applied = False


def load_network(model: nn.Module, path, dtype=None) -> tuple[LoRANetwork, dict]:
    """Rebuild the network a ``train.py`` checkpoint describes, and load it.

    The rank, alpha and target pattern come from the file's own metadata, so a
    checkpoint trained against a different surface still loads correctly.
    """
    from safetensors import safe_open

    with safe_open(path, framework="pt") as handle:
        meta = handle.metadata() or {}
        state = {key: handle.get_tensor(key) for key in handle.keys()}

    missing = [k for k in ("rank", "alpha", "targets") if k not in meta]
    if missing:
        raise ValueError(f"{path}: checkpoint metadata lacks {missing}")
    saved_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    network = LoRANetwork(
        model,
        rank=int(meta["rank"]),
        alpha=float(meta["alpha"]),
        targets=meta["targets"],
        dtype=dtype or saved_dtype.get(meta.get("lora_dtype", ""), torch.float32),
    )
    network.load_state_dict(state)
    return network, meta


def _patched_forward(module: nn.Linear, adapter: LoRAAdapter):
    base = module.__class__.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        return adapter(x, base(module, x))

    return forward
