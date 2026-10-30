"""Adapters for the non-Anima lines, with their parameters outside the blocks.

``peft``'s ``add_adapter`` nests ``lora_A``/``lora_B`` inside the module they
adapt, so they would land inside the block list — and ``ModelOffloader`` swaps a
block by walking its ``named_modules()`` and moving every ``.weight`` it finds.
The trainable weights would ride to the CPU while their ``.grad`` stayed on the
card. Here the adapters are held by :class:`LoRANetwork`, which stays resident,
and each target ``nn.Linear`` gets an instance-level ``forward`` that adds their
output (Anima's ``networks/`` does the same). Patching ``forward`` rather than
replacing the module keeps the swapper's name→shape matching intact.

A line subclasses :class:`LoRANetwork` with its ``BASE_MODEL`` name, its block
list attribute (``BLOCKS``) and its default target regex
(``library/{moss,qwen21}/lora.py``). Files are saved in the kohya layout
ComfyUI's LoRA loader reads — ``lora_unet_<module_name>.{lora_down,lora_up}.weight``
plus a ``.alpha`` tensor (LyCORIS keys keep their own names under the same
prefix) — so a file is tied to its line's module names. ``from_file`` also
reads the earlier ``adapters.<module_name>.{down,up}.weight`` layout.

``down_init="weight_svd"`` seeds ``down`` with ``V_rᵀ/√3`` from the base weight's
SVD, as Anima's SVD-Down (``docs/anima/methods/svd-down-lora.md``). The Kaiming draw
still runs first, so the global RNG stream (and the training noise drawn after
it) is the same under either init. ``svd_slice=k`` moves the window to singular
vectors ``[k·r, (k+1)·r)`` (``svd_slice_count=N``: the interleaved comb
``k + N·j``), Anima's ``svd_slice``: slices of one basis are mutually orthogonal,
so LoRAs trained on different slices own disjoint input subspaces.
``"weight_svd_nodc"`` takes the basis of ``W(I − uuᵀ)`` instead, with
``u = x̄ / ‖x̄‖`` the calibrated input DC direction (``channel_mean``), so every row
starts orthogonal to the token mean and ``down · x̄ = 0`` at init; nothing holds it
there during training (``docs/moss/proposal/lora_dcfree_svd_init.md``).
``svd_drop_loud=k`` also projects out the ``k`` input channels with the largest
calibrated ``mean|x|`` (``channel_stats``; on MOSS ch 42 into ``ffn_up_gate``), so
their ``down`` columns start at zero.

``algo`` picks the adapter: ``lora``, or the LyCORIS ``lokr`` / ``loha``. Their
init, scale and ΔW math come from ``lycoris.functional`` (pinned ``backend="torch"``
so dynamo traces them inside the compiled blocks), and their tensors carry
LyCORIS's key names (``lokr_w1``, ``hada_w1_a``, …, plus an ``alpha`` buffer).
LoKr applies in bypass (grouped GEMMs on the tokens; ΔW is never built). LoHa
rebuilds ΔW each forward and runs one GEMM against ``W + ΔW``, LyCORIS's own
LoHa path, so the delta is rounded to the base weight's bf16 grid in the forward.

``rank_pattern`` (``{regex: (rank, alpha)}``) adds the linears each regex fullmatches,
at their own rank and alpha (MOSS's ``mod_rank``: ``shared_block_modulation``). They
are built after the ``targets`` pass under a forked RNG, so the ``targets`` adapters'
init and the training noise drawn after it match a run without them. Stamped as JSON
(``rank_pattern``) and read back by ``from_file``.

``LoRANetwork.set_row_mask(mask)`` (``(1, L, 1)`` over the joint token axis, or
None) confines every adapter's delta to the rows the mask keeps: a token-shaped
input ``(B, L, in)`` gets ``base + mask · delta``, a 2-D input (``mod_rank``'s
modulation) is left untouched. The mask is a tensor attribute, so a compiled
block takes it as an input and a new mask per step is not a recompile.
"""

from __future__ import annotations

import json
import math
import re
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

SAVED_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DOWN_INITS = ("kaiming", "weight_svd", "weight_svd_nodc")
_SVD_INITS = ("weight_svd", "weight_svd_nodc")
ALGOS = ("lora", "lokr", "loha")
FILE_PREFIX = "lora_unet_"
_LORA_FILE_KEYS = {"down.weight": "lora_down.weight", "up.weight": "lora_up.weight"}
_LORA_STATE_KEYS = {v: k for k, v in _LORA_FILE_KEYS.items()}


def _masked(delta: torch.Tensor, row_mask: torch.Tensor | None) -> torch.Tensor:
    """``delta`` confined to the rows ``row_mask`` keeps (``set_row_mask``)."""
    if row_mask is None or delta.ndim != 3:
        return delta
    return delta * row_mask.to(delta.dtype)


def _drop_directions(
    mean: torch.Tensor | None, abs_mean: torch.Tensor | None, loud: int
) -> torch.Tensor | None:
    """``(in, k)`` directions a ``weight_svd`` basis is made orthogonal to: the
    input DC ``x̄`` and the ``loud`` channels with the largest ``mean|x|``."""
    cols = []
    if mean is not None:
        cols.append(mean.float() / mean.float().norm().clamp_min(1e-12))
    if loud:
        eye = torch.zeros(abs_mean.numel(), loud)
        eye[torch.topk(abs_mean.float(), loud).indices, torch.arange(loud)] = 1.0
        cols.extend(eye.T)
    return torch.stack(cols, 1) if cols else None


class LoRAAdapter(nn.Module):
    """``scale · multiplier · up(down(x))``, added to the base linear's output.

    Built fp32 for training (the master weights AdamW updates — bf16's 8-bit
    mantissa rounds away updates below ~0.4 % of a weight). The rank GEMMs run
    in the base output's dtype, so an fp32 adapter never lifts a ``(T, hidden)``
    activation to fp32; same policy as ``networks/lora_modules/base.py``.
    ``multiplier == 0`` returns the base output untouched — the base model
    exactly, not a zeroed delta.
    """

    def __init__(self, in_features, out_features, rank, alpha, dtype):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, dtype=dtype)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.multiplier = 1.0
        self.row_mask = None

    @torch.no_grad()
    def init_down_weight_svd(
        self,
        base: nn.Linear,
        svd_slice: int = 0,
        svd_slice_count: int = 0,
        drop: torch.Tensor | None = None,
    ) -> None:
        """``down = V_rᵀ / √3``: the base weight's top-r right singular vectors,
        at the Kaiming default's expected row norm. ``svd_slice`` /
        ``svd_slice_count`` pick another window of the same basis (module
        docstring). ``drop`` (``(in, k)``): the basis is that of ``W(I − P)``,
        ``P`` the projector onto its columns' span, every row ⊥ them. Computed on the card when there is one (an fp32 eigh of the
        smaller Gram per linear)."""
        from networks.lora_modules.lora import _top_right_singular_vectors

        rank = self.down.out_features
        if svd_slice_count:
            idx = svd_slice + svd_slice_count * torch.arange(rank)
        else:
            idx = svd_slice * rank + torch.arange(rank)
        end = int(idx[-1]) + 1
        # Each dropped direction costs one vector: past them the basis would
        # hand the dropped span back.
        spectrum = min(base.weight.shape) - (0 if drop is None else drop.shape[1])
        if end > spectrum:
            raise ValueError(
                f"svd_slice={svd_slice}: window ends at {end}, past the "
                f"{spectrum}-vector spectrum of a "
                f"{tuple(base.weight.shape)} linear; lower the slice or the rank"
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight = base.weight.detach().to(device, torch.float32)
        if drop is not None:
            q, _ = torch.linalg.qr(drop.to(device, torch.float32))
            weight = weight - (weight @ q) @ q.T
        basis = _top_right_singular_vectors(weight, end)
        self.down.weight.copy_(basis[:, idx.to(basis.device)].T / math.sqrt(3.0))

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0:
            return base_out
        work = base_out.dtype
        lx = F.linear(x.to(work), self.down.weight.to(work))
        delta = F.linear(lx, self.up.weight.to(work))
        return base_out + _masked(delta, self.row_mask) * (self.scale * self.multiplier)

    def patched(self, module: nn.Linear, base, x: torch.Tensor) -> torch.Tensor:
        return self(x, base(module, x))


class LoKrAdapter(nn.Module):
    """LyCORIS LoKr: ``ΔW = scale · kron(w1, w2)``, ``out = a·b``, ``in = c·d``.

    ``w1`` (a, c) is the small full factor; ``w2`` (b, d) is ``w2_a @ w2_b`` at
    ``rank`` when the rank is under half its larger side, else a full matrix
    (LyCORIS's huge-dim convention), and with both halves full the scale is 1
    whatever alpha says. ``factor`` caps ``a`` / ``c`` (-1 = near-square).
    """

    def __init__(self, base: nn.Linear, rank, alpha, factor, dtype):
        super().__init__()
        from lycoris.functional import lokr

        # tucker=False: weight_gen's default inits a conv-only t2 that a
        # Linear never allocates (None).
        w1, _w1a, _w1b, w2, w2a, w2b, _t = lokr.weight_gen(
            base.weight, rank, tucker=False, factor=factor
        )
        self.lokr_w1 = nn.Parameter(w1.to(dtype))
        if w2 is not None:
            self.lokr_w2 = nn.Parameter(w2.to(dtype))
            alpha = rank
        else:
            self.lokr_w2_a = nn.Parameter(w2a.to(dtype))
            self.lokr_w2_b = nn.Parameter(w2b.to(dtype))
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.scale = alpha / rank
        self.multiplier = 1.0
        self.row_mask = None

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0:
            return base_out
        from lycoris.functional import lokr

        work = base_out.dtype
        full = hasattr(self, "lokr_w2")
        delta = lokr.kron_bypass(
            x.to(work),
            self.lokr_w1.to(work),
            None,
            None,
            self.lokr_w2.to(work) if full else None,
            None if full else self.lokr_w2_a.to(work),
            None if full else self.lokr_w2_b.to(work),
            None,
            scale=self.scale * self.multiplier,
            backend="torch",
        )
        return base_out + _masked(delta, self.row_mask)

    def patched(self, module: nn.Linear, base, x: torch.Tensor) -> torch.Tensor:
        return self(x, base(module, x))


class LoHaAdapter(nn.Module):
    """LyCORIS LoHa: ``ΔW = scale · (w1_a @ w1_b) ⊙ (w2_a @ w2_b)``, each pair at
    ``rank`` (an effective rank up to rank²). ``hada_w1_a`` starts at zero, so
    ΔW does too."""

    def __init__(self, base: nn.Linear, rank, alpha, dtype):
        super().__init__()
        from lycoris.functional import loha

        w1d, w1u, w2d, w2u, _t1, _t2 = loha.weight_gen(base.weight, rank)
        self.hada_w1_a = nn.Parameter(w1u.to(dtype))
        self.hada_w1_b = nn.Parameter(w1d.to(dtype))
        self.hada_w2_a = nn.Parameter(w2u.to(dtype))
        self.hada_w2_b = nn.Parameter(w2d.to(dtype))
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        # HadaWeight saves its gamma for backward, so it must be a tensor; a
        # device-resident 1 keeps the Python-float scale outside it.
        self.register_buffer("_unit", torch.ones(()), persistent=False)
        self.scale = alpha / rank
        self.multiplier = 1.0
        self.row_mask = None

    def delta_weight(self, dtype: torch.dtype) -> torch.Tensor:
        from lycoris.functional import loha

        hada = loha.diff_weight(
            self.hada_w1_b.to(dtype),
            self.hada_w1_a.to(dtype),
            self.hada_w2_b.to(dtype),
            self.hada_w2_a.to(dtype),
            None,
            None,
            gamma=self._unit.to(dtype),
            backend="torch",
        )
        return hada * (self.scale * self.multiplier)

    def patched(self, module: nn.Linear, base, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0:
            return base(module, x)
        weight = module.weight
        if self.row_mask is not None:
            delta = F.linear(x.to(weight.dtype), self.delta_weight(weight.dtype))
            return base(module, x) + _masked(delta, self.row_mask)
        merged = weight + self.delta_weight(weight.dtype)
        return F.linear(x.to(weight.dtype), merged, module.bias)


class LoRANetwork(nn.Module):
    """The adapters for one transformer, held outside it.

    ``apply_to`` patches the targets and must run before any ``torch.compile``
    of the blocks (compile traces the patched forward), the same ordering
    ``build_anima`` encodes for Anima.
    """

    BASE_MODEL: ClassVar[str] = ""
    BLOCKS: ClassVar[str] = ""
    DEFAULT_TARGETS: ClassVar[str] = ""

    def __init__(
        self,
        model: nn.Module,
        rank: int = 16,
        alpha: float | None = None,
        targets: str | None = None,
        dtype: torch.dtype = torch.float32,
        down_init: str = "kaiming",
        algo: str = "lora",
        factor: int = -1,
        channel_stats: dict[str, torch.Tensor] | None = None,
        channel_mean: dict[str, torch.Tensor] | None = None,
        svd_slice: int = 0,
        svd_slice_count: int = 0,
        svd_drop_loud: int = 0,
        rank_pattern: dict[str, tuple[int, float]] | None = None,
    ):
        super().__init__()
        if down_init not in DOWN_INITS:
            raise ValueError(
                f"down_init must be one of {DOWN_INITS}, not {down_init!r}"
            )
        if algo not in ALGOS:
            raise ValueError(f"algo must be one of {ALGOS}, not {algo!r}")
        if down_init == "weight_svd_nodc" and channel_mean is None:
            raise ValueError("down_init='weight_svd_nodc' needs channel_mean")
        if svd_slice < 0 or svd_slice_count < 0:
            raise ValueError("svd_slice / svd_slice_count must be non-negative")
        if (svd_slice or svd_slice_count) and (
            algo != "lora" or down_init not in _SVD_INITS
        ):
            raise ValueError(
                "svd_slice / svd_slice_count apply to algo='lora' with "
                "down_init='weight_svd' / 'weight_svd_nodc' only"
            )
        if svd_drop_loud < 0:
            raise ValueError("svd_drop_loud must be non-negative")
        if svd_drop_loud and (algo != "lora" or down_init not in _SVD_INITS):
            raise ValueError(
                "svd_drop_loud applies to algo='lora' with "
                "down_init='weight_svd' / 'weight_svd_nodc' only"
            )
        if svd_drop_loud and channel_stats is None:
            raise ValueError("svd_drop_loud needs channel_stats")
        if svd_slice_count and svd_slice >= svd_slice_count:
            raise ValueError(
                f"svd_slice={svd_slice} must be < svd_slice_count={svd_slice_count}"
            )
        targets = targets or self.DEFAULT_TARGETS
        pattern = re.compile(targets)
        self.algo = algo
        self.factor = factor
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.targets = targets
        self.down_init = down_init
        self.svd_slice = int(svd_slice)
        self.svd_slice_count = int(svd_slice_count)
        self.svd_drop_loud = int(svd_drop_loud)
        self.rank_pattern = {
            p: (int(r), float(a)) for p, (r, a) in (rank_pattern or {}).items()
        }
        extra = [(re.compile(p), r, a) for p, (r, a) in self.rank_pattern.items()]
        self.num_blocks = self._num_blocks(model)
        self.adapters = nn.ModuleDict()
        self._targets: list[tuple[str, nn.Linear]] = []
        linears = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
        ]
        for name, module in linears:
            if not pattern.fullmatch(name) or any(
                p.fullmatch(name) for p, _, _ in extra
            ):
                continue
            self._add(
                name, module, rank, self.alpha, dtype, channel_stats, channel_mean
            )
        with torch.random.fork_rng(devices=[]):
            for name, module in linears:
                for p, r, a in extra:
                    if p.fullmatch(name):
                        self._add(
                            name, module, r, a, dtype, channel_stats, channel_mean
                        )
                        break
        if not self._targets:
            raise ValueError(f"no nn.Linear matched {targets!r}")
        self._applied = False

    def _add(self, name, module, rank, alpha, dtype, channel_stats, channel_mean):
        algo, down_init = self.algo, self.down_init
        if algo == "lokr":
            adapter = LoKrAdapter(module, rank, alpha, self.factor, dtype)
        elif algo == "loha":
            adapter = LoHaAdapter(module, rank, alpha, dtype)
        else:
            adapter = LoRAAdapter(
                module.in_features, module.out_features, rank, alpha, dtype
            )
            if down_init == "weight_svd_nodc" and name not in channel_mean:
                raise ValueError(f"no channel mean for {name}")
            if self.svd_drop_loud and name not in channel_stats:
                raise ValueError(f"no channel stats for {name}")
            if down_init in _SVD_INITS:
                adapter.init_down_weight_svd(
                    module,
                    self.svd_slice,
                    self.svd_slice_count,
                    drop=_drop_directions(
                        channel_mean[name] if down_init == "weight_svd_nodc" else None,
                        channel_stats[name] if self.svd_drop_loud else None,
                        self.svd_drop_loud,
                    ),
                )
        self.adapters[name.replace(".", "_")] = adapter
        self._targets.append((name, module))

    @classmethod
    def _num_blocks(cls, model: nn.Module) -> int | None:
        blocks = getattr(model, cls.BLOCKS, None) if cls.BLOCKS else None
        return None if blocks is None else len(blocks)

    @property
    def target_names(self) -> list[str]:
        return [name for name, _ in self._targets]

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def apply_to(self) -> int:
        """Patch every target's ``forward``. Returns the count."""
        if self._applied:
            raise RuntimeError("already applied")
        for name, module in self._targets:
            module.forward = _patched_forward(
                module, self.adapters[name.replace(".", "_")]
            )
        self._applied = True
        return len(self._targets)

    def set_multiplier(self, multiplier: float) -> None:
        """Scale every adapter; ``0.0`` is the base model exactly, so an A/B
        needs neither a reload nor a second model."""
        for adapter in self.adapters.values():
            adapter.multiplier = float(multiplier)

    def set_row_mask(self, mask: torch.Tensor | None) -> None:
        """Confine every adapter's delta to the rows ``mask`` ``(1, L, 1)`` keeps
        (module docstring); None lifts it."""
        for adapter in self.adapters.values():
            adapter.row_mask = mask

    def restore(self) -> None:
        """Un-patch the targets (the adapters stay built)."""
        for _name, module in self._targets:
            module.__dict__.pop("forward", None)
        self._applied = False

    def metadata(self, **extra) -> dict[str, str]:
        meta = {
            "base_model": self.BASE_MODEL,
            "algo": self.algo,
            "rank": str(self.rank),
            "alpha": str(self.alpha),
            "targets": self.targets,
        }
        if self.rank_pattern:
            meta["rank_pattern"] = json.dumps(
                {p: [r, a] for p, (r, a) in self.rank_pattern.items()}
            )
        if self.num_blocks is not None:
            # A LoRA is depth-specific: module names carry the block index.
            meta["ss_num_blocks"] = str(self.num_blocks)
        if self.algo == "lora":
            meta["down_init"] = self.down_init
            # Same keys as Anima's: which window of W0's basis seeded down.
            if self.svd_slice:
                meta["ss_svd_slice"] = str(self.svd_slice)
            if self.svd_slice_count:
                meta["ss_svd_slice_count"] = str(self.svd_slice_count)
            if self.svd_drop_loud:
                meta["svd_drop_loud"] = str(self.svd_drop_loud)
        if self.algo == "lokr":
            meta["factor"] = str(self.factor)
        meta.update({k: str(v) for k, v in extra.items()})
        return meta

    def save(self, path, dtype: str = "bf16", **extra_meta) -> None:
        from safetensors.torch import save_file

        state = {}
        for key, value in self.state_dict().items():
            name, rest = key.removeprefix("adapters.").split(".", 1)
            stem = FILE_PREFIX + name
            if self.algo == "lora" and rest == "down.weight":
                state[f"{stem}.alpha"] = torch.tensor(self.alpha)
            state[f"{stem}.{_LORA_FILE_KEYS.get(rest, rest)}"] = value
        state = {
            k: v.detach().to("cpu", SAVED_DTYPES[dtype]).contiguous()
            for k, v in state.items()
        }
        save_file(
            state, str(path), metadata=self.metadata(lora_dtype=dtype, **extra_meta)
        )

    @classmethod
    def from_file(cls, model: nn.Module, path, dtype=None) -> tuple[LoRANetwork, dict]:
        """Rebuild the network a trainer checkpoint describes, and load it.
        Algo, rank, alpha and targets come from the file's metadata (no ``algo``
        = a LoRA from before LyCORIS); a depth mismatch (``ss_num_blocks``)
        raises instead of silently dropping blocks."""
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            meta = handle.metadata() or {}
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        missing = [k for k in ("rank", "alpha", "targets") if k not in meta]
        if missing:
            raise ValueError(f"{path}: checkpoint metadata lacks {missing}")
        depth, have = meta.get("ss_num_blocks"), cls._num_blocks(model)
        if depth is not None and have is not None and int(depth) != have:
            raise ValueError(
                f"{path}: trained on a {depth}-block DiT, this one has {have}"
            )
        network = cls(
            model,
            rank=int(meta["rank"]),
            alpha=float(meta["alpha"]),
            targets=meta["targets"],
            dtype=dtype or SAVED_DTYPES.get(meta.get("lora_dtype", ""), torch.float32),
            algo=meta.get("algo", "lora"),
            factor=int(meta.get("factor", -1)),
            rank_pattern=json.loads(meta.get("rank_pattern", "{}")),
        )
        network.load_state_dict(_state_from_file(state, network.algo))
        return network, meta

    @classmethod
    def for_inference(
        cls, transformer: nn.Module, path, multiplier: float = 1.0
    ) -> LoRANetwork:
        """Load a LoRA onto ``transformer`` (hooked, not merged), on its device, at
        ``multiplier``. Call before compiling the blocks. ``set_multiplier``
        switches strength later; ``0.0`` is the base model exactly, ``restore()``
        takes the hooks off."""
        network, _meta = cls.from_file(transformer, path)
        network.to(next(transformer.parameters()).device)
        network.requires_grad_(False)
        network.apply_to()
        network.set_multiplier(multiplier)
        return network


def load_channel_stats(path) -> dict[str, torch.Tensor]:
    """Per-target input vectors keyed by qualified module name: ``mean|x|``
    (``svd_drop_loud``) or the signed token mean ``x̄`` (``weight_svd_nodc``)."""
    from safetensors.torch import load_file

    return load_file(str(path))


def _state_from_file(state: dict, algo: str) -> dict:
    """File keys → ``state_dict`` keys; the pre-kohya layout passes through."""
    out = {}
    for key, value in state.items():
        if not key.startswith(FILE_PREFIX):
            out[key] = value
            continue
        name, rest = key.removeprefix(FILE_PREFIX).split(".", 1)
        if algo == "lora":
            if rest == "alpha":
                continue
            rest = _LORA_STATE_KEYS[rest]
        out[f"adapters.{name}.{rest}"] = value
    return out


def _patched_forward(module: nn.Linear, adapter: nn.Module):
    base = module.__class__.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        return adapter.patched(module, base, x)

    return forward
