"""Register-token adapter for the Anima DiT — full method wiring.

Driven by `train.py --method register`. DSR mechanism (arXiv:2605.05206) on a
frozen pretrained DiT:

* K non-decoded **register tokens** injected mid-stack at ``insert_block``
  (default 8 — DSR Tab. 9's sweet spot; 0 = stack entry). Mechanism lives in
  ``networks/register_injection.py``.
* A trained self-attn QKV surface on the target blocks — either a low-rank
  LoRA (`qkv_mode="lora"`) or a full-rank ΔW (`qkv_mode="unfrozen"`, the DSR
  sweet-spot reachability arm, proposal `_archive/proposals/headroom_register_tokens.md`).

Two arms (proposal §Arms; `bench/headroom/README.md`):

* **arm B** (default, shippable) — K *learnable* register embeddings.
* **arm A** — K *fixed-zero* registers (a lesion; ablation only, not for
  shipping). `arm=L` shorthand (`num_registers=0`) is the LoRA-only drift
  control.

Design notes that make this both train-side and ComfyUI-loadable:

* **Split-by-construction QKV surface.** The training DiT fuses self-attn into
  one `qkv_proj` (`Linear(D, 3·inner)`, output ordered ``[q; k; v]`` in
  contiguous ``inner``-wide blocks — `library/anima/models.py` compute_qkv), but
  ComfyUI's cosmos backbone keeps split `q_proj/k_proj/v_proj`. So the surface
  holds **separate q/k/v deltas** and patches the fused `qkv_proj.forward` by
  concatenating outputs — bit-identical to a fused `(3·inner, D)` delta, and the
  saved state dict is already split, so the ComfyUI node loads each delta onto
  its own projection with zero fusion logic.
* **Frozen DiT, not a submodule.** The DiT is stashed via `object.__setattr__`
  so `self.parameters()` stays adapter-only (the base freeze trick).
* **Kept-live, non-mergeable.** Register tokens cannot be baked into DiT
  weights; inference keeps the adapter live (mirrors EasyControl).

Compile is supported: the constant +K seq growth is threaded through
``compile_dynamic_seq`` by widening the mark_dynamic bound's *max* by K
(``train.py`` reads ``network.extra_seq_tokens``; the min stays — blocks before
``insert_block`` still run at the bare seq). Block swap stays forced off in the
method config (mid-stack token surgery is unaudited against the offloader).
"""

from __future__ import annotations

import json
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.methods.base import AdapterNetworkBase
from networks.register_injection import RegisterInjector


def _parse_target_blocks(spec, n_blocks: int) -> list[int]:
    """Accept ``"all"``, an ``"a-b"`` inclusive range, a comma list, or a JSON
    list of block indices."""
    if spec is None or spec == "all":
        return list(range(n_blocks))
    if isinstance(spec, (list, tuple)):
        return [int(x) for x in spec]
    s = str(spec).strip()
    if s.startswith("["):
        return [int(x) for x in json.loads(s)]
    if "-" in s and "," not in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",") if x != ""]


class _QKVSurface(nn.Module):
    """Per-block trained self-attn QKV surface, held split into q/k/v.

    ``forward(x)`` returns the delta to add to the fused ``qkv_proj(x)`` output,
    concatenated as ``[Δq; Δk; Δv]`` to match its ``[q; k; v]`` layout.
    """

    def __init__(
        self,
        in_dim: int,
        inner_dim: int,
        qkv_mode: str,
        rank: int,
        down_init: str = "kaiming",
        base_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.qkv_mode = qkv_mode
        if qkv_mode == "lora":
            # Shared down, per-component up. Up zero-init → no-op at step 0
            # regardless of the down init.
            if down_init == "weight_svd":
                # OrthoInit-style warm start: seed the shared down with the
                # top-r RIGHT-singular directions of this block's frozen fused
                # qkv weight (3·inner, D) — the principal input subspace across
                # q/k/v jointly, which is exactly what a shared down reads.
                # Init-only: the basis stays trainable, nothing is frozen.
                w = base_weight.detach().float()
                q = min(max(4 * rank, rank + 8), min(w.shape))
                _, _, v = torch.svd_lowrank(w, q=q, niter=4)
                self.down = nn.Parameter(v[:, :rank].T.contiguous())
            elif down_init == "kaiming":
                self.down = nn.Parameter(torch.empty(rank, in_dim))
                nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
            else:
                raise ValueError(
                    f"down_init must be 'kaiming' or 'weight_svd', got {down_init!r}"
                )
            self.up_q = nn.Parameter(torch.zeros(inner_dim, rank))
            self.up_k = nn.Parameter(torch.zeros(inner_dim, rank))
            self.up_v = nn.Parameter(torch.zeros(inner_dim, rank))
        else:  # unfrozen: full-rank ΔW per component, zero-init (no-op at step 0)
            self.q = nn.Parameter(torch.zeros(inner_dim, in_dim))
            self.k = nn.Parameter(torch.zeros(inner_dim, in_dim))
            self.v = nn.Parameter(torch.zeros(inner_dim, in_dim))

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        if self.qkv_mode == "lora":
            d = F.linear(x, self.down)
            dq, dk, dv = (
                F.linear(d, self.up_q),
                F.linear(d, self.up_k),
                F.linear(d, self.up_v),
            )
            return torch.cat([dq, dk, dv], dim=-1) * scale
        return torch.cat(
            [F.linear(x, self.q), F.linear(x, self.k), F.linear(x, self.v)], dim=-1
        )


class RegisterNetwork(AdapterNetworkBase):
    """Register-token + self-attn QKV adapter on a frozen Anima DiT."""

    network_module = "networks.methods.register"
    network_spec = "register"
    mergeable = False

    def __init__(
        self,
        unet,
        num_registers: int = 36,
        arm: str = "B",
        qkv_mode: str = "unfrozen",
        lora_rank: int = 8,
        lora_alpha: Optional[float] = None,
        target_blocks=None,
        insert_block: int = 8,
        down_init: str = "kaiming",
        init_std: float = 0.02,
        register_lr_scale: float = 100.0,
        multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if arm not in ("A", "B"):
            raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")
        if qkv_mode not in ("lora", "unfrozen"):
            raise ValueError(f"qkv_mode must be 'lora' or 'unfrozen', got {qkv_mode!r}")
        if down_init != "kaiming" and qkv_mode != "lora":
            raise ValueError(
                f"down_init={down_init!r} only applies to qkv_mode='lora' "
                f"(got qkv_mode={qkv_mode!r})"
            )
        # Hide the frozen DiT from nn.Module registration so self.parameters()
        # stays adapter-only (the frozen-base trick, cf. EasyControl).
        object.__setattr__(self, "_dit", unet)
        self.multiplier = multiplier
        self.K = int(num_registers)
        self.arm = arm
        self.qkv_mode = qkv_mode
        self.D = int(unet.model_channels)
        self.register_lr_scale = float(register_lr_scale)

        if self.K > 0 and arm == "B":
            self.register = nn.Parameter(torch.randn(self.K, self.D) * init_std)
        else:  # arm A (content-free sink slot) or K=0 LoRA-only control
            self.register_buffer("register", torch.zeros(max(self.K, 0), self.D))

        n_blocks = len(unet.blocks)
        self.insert_block = int(insert_block)
        if not (0 <= self.insert_block < n_blocks):
            raise ValueError(
                f"insert_block must be in [0, {n_blocks}), got {self.insert_block}"
            )
        self.target_blocks = _parse_target_blocks(target_blocks, n_blocks)
        self.lora_rank = int(lora_rank)
        alpha = lora_alpha if lora_alpha is not None else lora_rank
        self.scale = alpha / lora_rank if self.lora_rank else 1.0

        inner_dim = (
            unet.blocks[self.target_blocks[0]].self_attn.qkv_proj.out_features // 3
        )
        self.down_init = down_init
        self.qkv = nn.ModuleDict(
            {
                str(bi): _QKVSurface(
                    self.D,
                    inner_dim,
                    qkv_mode,
                    self.lora_rank,
                    down_init=down_init,
                    base_weight=unet.blocks[bi].self_attn.qkv_proj.weight,
                )
                for bi in self.target_blocks
            }
        )

        # Every forward appends K tokens to the self-attn sequence, so the
        # compile dynamic-seq mark_dynamic bound must be widened by K
        # (train.py::_widen_seq_range_for_network reads this). 0 for arm L (K=0).
        self.extra_seq_tokens = self.K

        # Shared injection machinery (run_blocks wrap + mid-stack pre-hooks +
        # adoption metrics) — also driven by the LoRA family's num_registers.
        self._injector = RegisterInjector(
            num_registers=self.K,
            insert_block=self.insert_block,
            get_scaled_tokens=lambda: self.register * self.multiplier,
        )

        self._applied = False
        self._orig_qkv_fwd: dict[int, object] = {}

    # -- trainer-facing protocol --------------------------------------------
    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def prepare_optimizer_params_with_multiple_te_lrs(
        self, text_encoder_lr, unet_lr, default_lr
    ):
        del text_encoder_lr
        lr = unet_lr or default_lr
        # QKV surface at the base lr; register embeddings get their own (higher)
        # lr group — they compete with a ~20× baked-in attractor and a shared
        # LoRA-scale lr rarely lets them grow into sinks (proposal §metrics).
        qkv_params = [p for m in self.qkv.values() for p in m.parameters()]
        groups = [{"params": qkv_params, "lr": lr}]
        descriptions = ["register.qkv"]
        if self.arm == "B" and self.K > 0:
            groups.append(
                {"params": [self.register], "lr": lr * self.register_lr_scale}
            )
            descriptions.append("register.tokens")
        return groups, descriptions

    def metadata_fields(self) -> dict[str, str]:
        return {
            "ss_num_registers": str(self.K),
            "ss_arm": self.arm,
            "ss_qkv_mode": self.qkv_mode,
            "ss_lora_rank": str(self.lora_rank),
            "ss_scale": str(self.scale),
            "ss_insert_block": str(self.insert_block),
            "ss_down_init": self.down_init,
            "ss_target_blocks": json.dumps(self.target_blocks),
            "ss_model_channels": str(self.D),
            "ss_num_blocks": str(len(self._dit.blocks)),
        }

    # Relocation-crossover readouts (training-time adoption gate; no-grad),
    # computed by the shared injector each forward.
    @property
    def last_reg_ratio(self) -> Optional[float]:
        return self._injector.last_reg_ratio

    @property
    def last_patch_sink_ratio(self) -> Optional[float]:
        return self._injector.last_patch_sink_ratio

    # -- apply / remove ------------------------------------------------------
    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if self._applied:
            return
        anima = unet
        net = self

        # Register-token injection (run_blocks wrap + mid-stack pre-hooks) —
        # shared with the LoRA family via RegisterInjector.
        self._injector.apply(anima)

        # Additive split QKV surface on the target blocks. The delta is fp32;
        # under the training autocast F.linear runs it in the block dtype.
        for bi in self.target_blocks:
            qkv = anima.blocks[bi].self_attn.qkv_proj
            self._orig_qkv_fwd[bi] = qkv.forward
            surface = self.qkv[str(bi)]

            def make_fwd(orig, surface):
                def fwd(x):
                    return orig(x) + surface(x, net.scale) * net.multiplier

                return fwd

            qkv.forward = make_fwd(qkv.forward, surface)

        self._applied = True

    def remove(self) -> None:
        if not self._applied:
            return
        anima = self._dit
        self._injector.remove()
        for bi, orig in self._orig_qkv_fwd.items():
            anima.blocks[bi].self_attn.qkv_proj.forward = orig
        self._orig_qkv_fwd = {}
        self._applied = False


# --------------------------------------------------------------------------
# Factories (train.py contract — mirrors networks/methods/easycontrol.py)
# --------------------------------------------------------------------------
def _build(unet, network_dim, network_alpha, kwargs) -> RegisterNetwork:
    return RegisterNetwork(
        unet,
        num_registers=int(kwargs.get("num_registers", 36)),
        arm=str(kwargs.get("arm", "B")),
        qkv_mode=str(kwargs.get("qkv_mode", "unfrozen")),
        lora_rank=int(network_dim) if network_dim else 8,
        lora_alpha=float(network_alpha) if network_alpha else None,
        target_blocks=kwargs.get("target_blocks", "all"),
        insert_block=int(kwargs.get("insert_block", 8)),
        down_init=str(kwargs.get("down_init", "kaiming")),
        init_std=float(kwargs.get("init_std", 0.02)),
        register_lr_scale=float(kwargs.get("register_lr_scale", 100.0)),
    )


def create_network(
    multiplier,
    network_dim,
    network_alpha,
    vae,
    text_encoders,
    unet,
    neuron_dropout=None,
    **kwargs,
):
    del vae, text_encoders, neuron_dropout
    net = _build(unet, network_dim, network_alpha, kwargs)
    net.set_multiplier(float(multiplier))
    return net


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
    """Rebuild the network shape from a checkpoint's ``ss_*`` metadata, then load."""
    del ae, text_encoders, for_inference
    meta: dict[str, str] = {}
    if weights_sd is None:
        from safetensors import safe_open

        weights_sd = {}
        with safe_open(file, framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            for k in f.keys():
                weights_sd[k] = f.get_tensor(k)
    rank = int(meta.get("ss_lora_rank", kwargs.get("lora_rank", 8)))
    # scale = alpha/rank was stamped at save; rebuild alpha so lora-mode
    # checkpoints trained with alpha != rank reload at the trained strength.
    alpha = float(meta["ss_scale"]) * rank if "ss_scale" in meta else None
    net = RegisterNetwork(
        unet,
        num_registers=int(
            meta.get("ss_num_registers", kwargs.get("num_registers", 36))
        ),
        arm=str(meta.get("ss_arm", kwargs.get("arm", "B"))),
        qkv_mode=str(meta.get("ss_qkv_mode", kwargs.get("qkv_mode", "unfrozen"))),
        lora_rank=rank,
        lora_alpha=alpha,
        target_blocks=json.loads(meta["ss_target_blocks"])
        if "ss_target_blocks" in meta
        else kwargs.get("target_blocks", "all"),
        # pre-insert_block checkpoints trained with entry insertion — default 0.
        insert_block=int(meta.get("ss_insert_block", kwargs.get("insert_block", 0))),
        # down_init intentionally not restored — load_state_dict overwrites the
        # init, and re-running the SVD at load time would be wasted work.
    )
    net.load_state_dict(weights_sd, strict=False)
    net.set_multiplier(float(multiplier))
    return net
