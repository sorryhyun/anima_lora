"""Turbo Anima — DP-DMD distillation harness.

Owns two plain ``LoRANetwork`` instances (student + fake) on one frozen Anima
DiT — three with the τ-split critic (``fake_tau_banks=2`` adds ``fake_hi``, a
second fake bank owning the high-τ band). Each calls ``apply_to(unet)`` which
chains them onto every targeted Linear's forward:
``linear(x) -> [fake_hi ->] fake -> student -> original_linear``.

Each LoRA module short-circuits at ``not self.enabled``, so view-toggling is
just ``set_enabled(bool)`` on each network. The fake view drives exactly ONE
bank (``set_fake_bank``); inactive banks stay disabled.

Used by ``scripts/distill_turbo/distill.py``. Inference loads the saved
``anima_turbo.safetensors`` through the standard LoRA path — the student LoRA
is just a normal LoRA with CFG=4 baked in.

Docs: ``docs/structure/turbo.md`` (structure), ``docs/methods/turbo.md`` (ops).
Paper: Wu, Li, Zhang, Ma, "Diversity-Preserved Distribution Matching
Distillation" (arXiv:2602.03139).
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.attn_fuse import match_fused_spec
from networks.lora_anima.factory import create_network
from networks.lora_anima.network import LoRANetwork

logger = logging.getLogger(__name__)

View = Literal["teacher", "student", "fake"]


class TeacherFeatureDiscriminator(nn.Module):
    """GAN head over frozen-teacher block features (FastGen idea 1).

    FastGen's ``Discriminator_ImageDiT`` un-flattens each tap's tokens back to
    ``(B, D, H_p, W_p)`` for a conv head — fragile under Anima's per-bucket
    native-shape patch grid + compiled fake-5D ``(B, 1, L, 1, D)`` layout. Both
    granularities here sidestep it by never reshaping to a grid:

    - ``granularity="pooled"``: mean-pool each tap over every axis between
      batch and channel -> ``(B, D)``, then a 2-layer MLP per tap -> ``(B, num_taps)``.
    - ``granularity="token"`` (LADD-style dense head): same per-tap MLP applied
      to every token (no reshape needed) -> ``(B, num_taps·N)``.

    Both run identically on the eager and compiled layouts, in fp32 for
    GAN-loss stability. Parameters are IDENTICAL across granularities, so
    resume bundles load across a head switch.

    Tiny by design (~inner_dim²/2 params/tap, ~2M at D=2048) and discarded at
    save — pure training scaffolding, like the fake/critic LoRA.
    """

    def __init__(
        self,
        *,
        inner_dim: int,
        num_taps: int,
        hidden_dim: int | None = None,
        granularity: Literal["pooled", "token"] = "pooled",
    ) -> None:
        super().__init__()
        if granularity not in ("pooled", "token"):
            raise ValueError(f"unknown disc granularity: {granularity!r}")
        self.granularity = granularity
        h = hidden_dim if hidden_dim is not None else inner_dim // 2
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(inner_dim),
                nn.Linear(inner_dim, h),
                nn.LeakyReLU(0.2),
                nn.Linear(h, 1),
            )
            for _ in range(num_taps)
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        if len(feats) != len(self.heads):
            raise ValueError(
                f"TeacherFeatureDiscriminator expected {len(self.heads)} feature "
                f"tensors, got {len(feats)}"
            )
        logits = []
        for head, f in zip(self.heads, feats):
            if self.granularity == "token":
                # LayerNorm/Linear act on the last dim, so this runs unchanged on
                # eager (B,T,H,W,D) and compiled (B,1,L,1,D) alike.
                logits.append(head(f.float()).flatten(1))
            else:
                # Pool over every axis between batch (0) and channel (-1).
                pooled = f.float().mean(dim=tuple(range(1, f.ndim - 1)))  # (B, D)
                logits.append(head(pooled))
        return torch.cat(logits, dim=1)  # (B, num_taps) | (B, num_taps·N)


def gan_loss_generator(fake_logits: torch.Tensor) -> torch.Tensor:
    """Softplus hinge — generator wants the disc to score its samples as real."""
    return F.softplus(-fake_logits).mean()


def gan_loss_discriminator(
    real_logits: torch.Tensor, fake_logits: torch.Tensor
) -> torch.Tensor:
    """Softplus hinge — push real logits up, fake logits down (FastGen common_loss)."""
    return F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()


def _plain_lora_module_delta(
    weights_sd: dict[str, torch.Tensor], lora_name: str
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """One runtime module's ΔW from a defused plain-LoRA file, kept factored.

    Checkpoints store attention LoRAs per-component (q/k/v split); the runtime
    module is the fused Linear, whose ΔW is the components' deltas concatenated
    along the output dim. Returns fp32 factors ``(A [out,k], B [k,in])`` with
    ``ΔW = A @ B`` — never the materialized ``[out, in]`` matrix, so the
    downstream SVD stays a small-core problem. None when any component absent.
    """
    spec = match_fused_spec(lora_name)
    if spec is None:
        prefixes = [lora_name]
    else:
        base = lora_name.removesuffix(spec.fused_frag)
        prefixes = [base + spec.component_frag(c) for c in spec.component_letters]
    ups, downs = [], []
    for p in prefixes:
        down = weights_sd.get(f"{p}.lora_down.weight")
        up = weights_sd.get(f"{p}.lora_up.weight")
        if down is None or up is None:
            return None
        rank = down.shape[0]
        alpha = weights_sd.get(f"{p}.alpha")
        scale = (float(alpha) / rank) if alpha is not None else 1.0
        ups.append(scale * up.float())
        downs.append(down.float())
    if len(ups) == 1:
        return ups[0], downs[0]
    return torch.block_diag(*ups), torch.cat(downs, dim=0)


@torch.no_grad()
def _warm_start_module(module, A: torch.Tensor, B: torch.Tensor) -> float:
    """SVD-truncate ``ΔW = A @ B`` into a plain LoRAModule's down/up; return
    captured energy fraction. Works in the k-dim factor space (QR of each
    factor + SVD of the k×k core) instead of the full ``[out, in]`` matrix; a
    196-module warm start is seconds, not minutes. Folds the module's fixed
    ``scale`` so the applied delta equals the truncated ΔW exactly.
    """
    dev = module.lora_down.weight.device
    QA, RA = torch.linalg.qr(A.to(dev))
    QB, RB = torch.linalg.qr(B.to(dev).T)
    Uc, S, Vhc = torch.linalg.svd(RA @ RB.T)
    r = min(module.lora_dim, S.shape[0])
    e = S.pow(2)
    energy = (e[:r].sum() / e.sum().clamp(min=1e-30)).item()
    sq = (S[:r] / module.scale).sqrt()
    up = torch.zeros_like(module.lora_up.weight, dtype=torch.float32)
    down = torch.zeros_like(module.lora_down.weight, dtype=torch.float32)
    up[:, :r] = QA @ Uc[:, :r] * sq.unsqueeze(0)
    down[:r] = sq.unsqueeze(1) * (Vhc[:r] @ QB.T)
    module.lora_up.weight.copy_(up.to(module.lora_up.weight.dtype))
    module.lora_down.weight.copy_(down.to(module.lora_down.weight.dtype))
    return energy


def warm_start_plain_lora(network: LoRANetwork, weights_path: str, label: str) -> None:
    """Initialize a plain-LoRA network's ΔW from a standard LoRA checkpoint.

    Per runtime module: reconstruct the file's exact ΔW (re-fusing defused
    q/k/v components), SVD-truncate to the module's rank, write down/up. File
    rank may differ from the network's — truncation/zero-pad-by-SVD handles
    both directions. Modules absent from the file keep their constructed init.

    Intended for the DP-DMD student/fake warm start from an extracted
    full-model delta; accepts any plain LoRA file. Must run before
    ``compile_dit_blocks`` traces the forwards.
    """
    from safetensors.torch import load_file

    weights_sd = load_file(weights_path)
    # GOTCHA: a ComfyUI-native adaln extract names its adaln keys
    # adaln_modulation_{br}_2, but the modules to seed are named adaln_up_{br}.
    # Rename so the per-module lookup below matches (else the 84 adaln modules
    # silently keep default init). Presence-gated, no-op otherwise.
    from networks.lora_utils import (
        has_comfy_adaln_keys,
        relayout_adaln_comfy_to_runtime,
    )

    if has_comfy_adaln_keys(weights_sd):
        weights_sd = relayout_adaln_comfy_to_runtime(weights_sd)
        logger.info(
            f"{label}: renamed ComfyUI-layout adaln keys → runtime names for warm start."
        )
    if any(".lora_ups." in k or ".lora_up_weight" in k for k in weights_sd):
        raise RuntimeError(
            f"{weights_path}: MoE layout — warm start needs a "
            "plain single-head LoRA checkpoint."
        )
    logger.info(
        f"{label}: warm-starting {len(network.unet_loras)} modules from "
        f"{weights_path}..."
    )
    warmed, missing, energies = 0, [], []
    for module in network.unet_loras:
        if not hasattr(module, "lora_down") or module.lora_down.weight.dim() != 2:
            raise RuntimeError(
                f"warm start supports plain Linear LoRA modules only — "
                f"{module.lora_name} is {type(module).__name__}."
            )
        factors = _plain_lora_module_delta(weights_sd, module.lora_name)
        if factors is None:
            missing.append(module.lora_name)
            continue
        energies.append(_warm_start_module(module, *factors))
        warmed += 1
    if not warmed:
        raise RuntimeError(
            f"{weights_path}: no module matched the {label} network — wrong "
            "architecture or non-standard key layout?"
        )
    logger.info(
        f"{label}: warm-started {warmed}/{len(network.unet_loras)} modules from "
        f"{weights_path} (ΔW energy kept: mean "
        f"{sum(energies) / len(energies):.3f}, min {min(energies):.3f})"
        + (
            f"; {len(missing)} kept default init (first: {missing[:3]})"
            if missing
            else ""
        )
    )


class TurboDMDNetwork:
    """Two LoRA stacks on one frozen DiT, view-toggleable per forward.

    Not a ``nn.Module`` — it's a thin coordinator that holds two real
    ``LoRANetwork`` instances. The DiT itself is owned by the caller and
    stays frozen.
    """

    def __init__(
        self,
        unet,
        *,
        student_rank: int,
        fake_rank: int,
        student_alpha: float | None = None,
        fake_alpha: float | None = None,
        channel_scaling_alpha: float = 0.0,
        student_down_init: str = "kaiming",
        fake_down_init: str = "kaiming",
        fake_tau_banks: int = 1,
        train_adaln: bool = False,
        fake_adaln: bool | None = None,
        adaln_rank: int = 0,
        adaln_alpha: float = 0.0,
        gan_feature_indices: set[int] | None = None,
        gan_disc_hidden: int | None = None,
        gan_disc_head: Literal["pooled", "token"] = "pooled",
        build_fake: bool = True,
    ) -> None:
        self.unet = unet
        self.student_rank = int(student_rank)
        self.fake_rank = int(fake_rank)
        # Target the t-conditioned AdaLN modulation up-projections
        # (adaln_up_{branch}) in addition to the default attn+MLP set — adaln is
        # the largest mover in the official turbo delta (docs/methods/adaln.md).
        # Both student and fake carry the extra 84 modules (3 branches × 28
        # blocks); the saved student is renamed to the ComfyUI adaln layout in
        # save_student. ``fake_adaln`` (default: mirror train_adaln) lets a
        # VRAM-tight run drop the critic's 84 adaln modules at the cost of a
        # score estimate blind to the student's adaln subspace (unbenched).
        self.train_adaln = bool(train_adaln)
        self.fake_adaln = self.train_adaln if fake_adaln is None else bool(fake_adaln)
        # 0 = adaln modules share network_dim; >0 = their own rank (see
        # network_reg_dims below). Applies to student and (when fake_adaln)
        # the fake banks alike.
        self.adaln_rank = int(adaln_rank)
        if self.adaln_rank < 0:
            raise ValueError(f"adaln_rank must be >= 0, got {adaln_rank}")
        if self.adaln_rank > 0 and not self.train_adaln:
            raise ValueError("adaln_rank > 0 requires train_adaln=true")
        # 0 = adaln modules keep the network alpha (hotter alpha/rank scale
        # when adaln_rank is set); >0 = their own alpha. Shared by student and
        # (when fake_adaln) the fake banks, like adaln_rank.
        self.adaln_alpha = float(adaln_alpha)
        if self.adaln_alpha < 0:
            raise ValueError(f"adaln_alpha must be >= 0, got {adaln_alpha}")
        if self.adaln_alpha > 0 and not self.train_adaln:
            raise ValueError("adaln_alpha > 0 requires train_adaln=true")
        # τ-split critic: 2 = a second fake stack (fake_hi) owning the high-τ
        # band. Which bank answers/trains is the CALLER's choice via
        # set_fake_bank. 1 = the second stack is never constructed.
        self.fake_tau_banks = int(fake_tau_banks)
        if self.fake_tau_banks not in (1, 2):
            raise ValueError(f"fake_tau_banks={self.fake_tau_banks}: expected 1 or 2.")
        # build_fake=False (turbo DMAD, where a discriminator replaces the
        # critic): no fake stack exists — fake_banks is empty and the "fake"
        # view raises. make_aux_stack still builds fake-shaped stacks.
        if not build_fake and self.fake_tau_banks != 1:
            raise ValueError("build_fake=False requires fake_tau_banks=1.")
        # SmoothQuant-style per-input-channel rebalance absorbed into each
        # lora_down (bit-equivalent at init, merges out cleanly). 0.0 = off,
        # 0.5 = sqrt-balance. Applied to both student and fake — conditions the
        # LoRA gradient on the DiT's outlier-DC input channels only, so it
        # leaves the DP-DMD objective untouched.
        self.channel_scaling_alpha = float(channel_scaling_alpha)

        # SVD-Down init (down_init="weight_svd"): seed the student's lora_down
        # from W0's top-r right singular vectors (scale-matched).
        self.student_down_init = str(student_down_init)
        if self.student_down_init not in ("kaiming", "weight_svd"):
            raise ValueError(
                f"student_down_init={self.student_down_init!r}: "
                "expected 'kaiming' or 'weight_svd'."
            )
        # Same lever on the fake/critic.
        self.fake_down_init = str(fake_down_init)
        if self.fake_down_init not in ("kaiming", "weight_svd"):
            raise ValueError(
                f"fake_down_init={self.fake_down_init!r}: "
                "expected 'kaiming' or 'weight_svd'."
            )

        # Plain LoRA on both (LoRANetworkCfg defaults: no MoE/T-LoRA); alpha =
        # rank by default. adaln_rank > 0 builds the adaln modules at their own (lower) rank via
        # network_reg_dims: the official turbo's adaln ΔW is near-lossless well
        # below attn rank, so uniform rank there is wasted VRAM. NB the reg_dims
        # path keeps the NETWORK alpha, so adaln runs a higher alpha/rank scale
        # — warm start folds the scale so the init is exact either way.
        _adaln_reg_dims = (
            f".*adaln_up_.*={self.adaln_rank}" if self.adaln_rank > 0 else None
        )
        _adaln_reg_alphas = (
            f".*adaln_up_.*={self.adaln_alpha}" if self.adaln_alpha > 0 else None
        )
        _student_kwargs: dict = {}
        if self.student_down_init != "kaiming":
            _student_kwargs["down_init"] = self.student_down_init
        if self.train_adaln:
            _student_kwargs["include_patterns"] = [".*adaln_up_.*"]
            if _adaln_reg_dims is not None:
                _student_kwargs["network_reg_dims"] = _adaln_reg_dims
            if _adaln_reg_alphas is not None:
                _student_kwargs["network_reg_alphas"] = _adaln_reg_alphas
        self.student: LoRANetwork = create_network(
            multiplier=1.0,
            network_dim=self.student_rank,
            network_alpha=student_alpha
            if student_alpha is not None
            else self.student_rank,
            vae=None,
            text_encoders=[],
            unet=unet,
            channel_scaling_alpha=self.channel_scaling_alpha,
            **_student_kwargs,
        )

        _fake_kwargs: dict = {}
        if self.fake_down_init != "kaiming":
            _fake_kwargs["down_init"] = self.fake_down_init
        if self.fake_adaln:
            _fake_kwargs["include_patterns"] = [".*adaln_up_.*"]
            if _adaln_reg_dims is not None:
                _fake_kwargs["network_reg_dims"] = _adaln_reg_dims
            if _adaln_reg_alphas is not None:
                _fake_kwargs["network_reg_alphas"] = _adaln_reg_alphas

        def _make_fake() -> LoRANetwork:
            return create_network(
                multiplier=1.0,
                network_dim=self.fake_rank,
                network_alpha=fake_alpha if fake_alpha is not None else self.fake_rank,
                vae=None,
                text_encoders=[],
                unet=unet,
                channel_scaling_alpha=self.channel_scaling_alpha,
                **_fake_kwargs,
            )

        self._make_fake = _make_fake  # make_aux_stack builds fake-shaped stacks
        # bank 0 (banks=2: the low-τ bank)
        self.fake: LoRANetwork | None = _make_fake() if build_fake else None
        self.fake_hi: LoRANetwork | None = (
            _make_fake() if self.fake_tau_banks == 2 else None
        )

        # Apply student-first so the runtime chain is
        # linear -> [fake_hi ->] fake -> student -> original (fake_hi outermost)
        self.student.apply_to(
            text_encoders=[],
            unet=unet,
            apply_text_encoder=False,
            apply_unet=True,
        )
        if self.fake is not None:
            self.fake.apply_to(
                text_encoders=[],
                unet=unet,
                apply_text_encoder=False,
                apply_unet=True,
            )
        if self.fake_hi is not None:
            self.fake_hi.apply_to(
                text_encoders=[],
                unet=unet,
                apply_text_encoder=False,
                apply_unet=True,
            )

        if self.fake is None:
            fake_desc = "no fake stack"
        else:
            banks = (
                f", x{self.fake_tau_banks} τ-banks" if self.fake_tau_banks > 1 else ""
            )
            fake_desc = (
                f"fake rank={self.fake_rank} "
                f"({len(self.fake.unet_loras)} modules{banks})"
            )
        logger.info(
            f"TurboDMDNetwork: student rank={self.student_rank} "
            f"({len(self.student.unet_loras)} modules), {fake_desc}"
        )

        # GOTCHA: LoRAModule defaults enabled=True, and set_view short-circuits
        # when view==self._view, so we MUST disable every stack explicitly here
        # — else the set_view invariant (cur state matches _VIEW_FLAGS[_view])
        # breaks once any stack carries nonzero weights.
        self.student.set_enabled(False)
        if self.fake is not None:
            self.fake.set_enabled(False)
        if self.fake_hi is not None:
            self.fake_hi.set_enabled(False)
        self._view: View = "teacher"
        self._fake_bank = 0

        # GAN feature tap: off unless gan_feature_indices is given. The disc reads
        # frozen-teacher block activations via the DiT's first-class feature-tap
        # (return_block_features/return_features_early), NOT a forward hook; the
        # caller hands the feature dict to self.disc via features_in_order.
        self.disc: TeacherFeatureDiscriminator | None = None
        self.gan_feature_indices: list[int] = []
        if gan_feature_indices:
            self.gan_feature_indices = sorted(gan_feature_indices)
            self.disc = TeacherFeatureDiscriminator(
                inner_dim=unet.model_channels,
                num_taps=len(self.gan_feature_indices),
                hidden_dim=gan_disc_hidden,
                granularity=gan_disc_head,
            )
            logger.info(
                f"TurboDMDNetwork: GAN disc attached (taps={self.gan_feature_indices}, "
                f"head={gan_disc_head}, inner_dim={unet.model_channels}, "
                f"{sum(p.numel() for p in self.disc.parameters()):,} params)"
            )

    @property
    def gan_feature_set(self) -> set[int] | None:
        """Tap indices as a set for ``return_block_features`` (None when GAN off)."""
        return set(self.gan_feature_indices) if self.gan_feature_indices else None

    def features_in_order(self, feats: dict[int, torch.Tensor]) -> list[torch.Tensor]:
        """Reorder a model feature-dict to tap-index order for the disc.

        ``feats`` is the dict returned by the teacher feature-tap forward
        (``return_block_features``); raises if a tap is missing (mis-wired call).
        """
        try:
            return [feats[i] for i in self.gan_feature_indices]
        except KeyError as e:
            raise RuntimeError(
                f"GAN feature tap {e} missing from teacher forward output — "
                "the forward was not run with return_block_features=gan_feature_set."
            ) from e

    def disc_params(self):
        """Trainable params for the discriminator optimizer."""
        if self.disc is None:
            return []
        return [p for p in self.disc.parameters() if p.requires_grad]

    def set_disc_requires_grad(self, flag: bool) -> None:
        """Toggle disc grad: False during the student (gen) step so the GAN-gen
        gradient only reaches x_pred; True during the disc update."""
        if self.disc is not None:
            self.disc.requires_grad_(flag)

    # Per-view (student_on, fake_on) target states; lookup makes the
    # "flip only what changed" diff explicit.
    _VIEW_FLAGS: dict[str, tuple[bool, bool]] = {
        "teacher": (False, False),
        "student": (True, False),
        "fake": (False, True),
    }

    def set_view(self, view: View) -> None:
        """Flip per-network enabled flags so the next DiT forward acts as
        the named view.

        - ``teacher``: both LoRA stacks off, DiT delivers base velocity.
        - ``student``: student on, fake off — produces v_student for x_pred.
        - ``fake``: fake on, student off — fake's score estimate at τ_DM.

        Short-circuits when already in the target view (avoids the
        O(num_modules) attribute-write loop and an avoidable dynamo guard
        re-validation).
        """
        if view == self._view:
            return
        try:
            want_student, want_fake = self._VIEW_FLAGS[view]
        except KeyError as e:
            raise ValueError(
                f"Unknown view {view!r}; expected teacher/student/fake"
            ) from e
        if want_fake and self.fake is None:
            raise RuntimeError("set_view('fake'): built with build_fake=False.")
        cur_student, cur_fake = self._VIEW_FLAGS[self._view]
        if want_student != cur_student:
            self.student.set_enabled(want_student)
        if want_fake != cur_fake:
            # Only the ACTIVE bank ever toggles — inactive banks stay disabled
            # from __init__ on.
            self.fake_banks[self._fake_bank].set_enabled(want_fake)
        self._view = view

    @property
    def view(self) -> View:
        return self._view

    @property
    def fake_banks(self) -> list[LoRANetwork]:
        """The fake stacks in bank order (bank 0 = low-τ; length 1 unless split,
        0 under build_fake=False)."""
        if self.fake is None:
            return []
        return [self.fake] if self.fake_hi is None else [self.fake, self.fake_hi]

    @property
    def fake_bank(self) -> int:
        """Index of the bank that answers/trains the next fake-view forward."""
        return self._fake_bank

    def set_fake_bank(self, i: int) -> None:
        """Select which fake bank the fake view drives (τ-split critic routing).

        The caller routes by the batch's drawn τ vs its configured boundary —
        BEFORE the fake forward. No-op at fake_tau_banks=1 (i must be 0).
        """
        banks = self.fake_banks
        if not 0 <= i < len(banks):
            raise ValueError(
                f"set_fake_bank({i}): only {len(banks)} fake bank(s) exist."
            )
        if i == self._fake_bank:
            return
        if self._view == "fake":
            banks[self._fake_bank].set_enabled(False)
            banks[i].set_enabled(True)
        self._fake_bank = i

    def student_params(self):
        """Trainable params for the student optimizer."""
        return [p for p in self.student.parameters() if p.requires_grad]

    def fake_params(self):
        """Trainable params for the fake optimizer — all banks, bank order.

        One optimizer spans every bank: with ``zero_grad(set_to_none=True)`` the
        inactive bank's grads are None and AdamW skips them.
        """
        return [
            p for bank in self.fake_banks for p in bank.parameters() if p.requires_grad
        ]

    def make_aux_stack(self) -> LoRANetwork:
        """A fake-shaped LoRA stack applied outermost and left disabled.

        Not part of any view — the caller enables it around its own forwards
        (the DMAD probe's disc backbone). Call after ``freeze_dit`` and before
        ``compile_dit_blocks``; its params keep ``requires_grad=True``.
        """
        stack = self._make_fake()
        stack.apply_to(
            text_encoders=[],
            unet=self.unet,
            apply_text_encoder=False,
            apply_unet=True,
        )
        stack.set_enabled(False)
        return stack

    def freeze_dit(self) -> None:
        """Set ``requires_grad=False`` on every base DiT param.

        GOTCHA: must be called AFTER both ``apply_to``'s — the LoRA networks
        add sub-modules to ``unet``, so a wholesale ``unet.requires_grad_(False)``
        after apply would zero the LoRA params too. Selectively walks only
        ``unet`` params whose name doesn't start with a LoRA prefix.
        """
        lora_prefixes = tuple(
            {m.lora_name for m in self.student.unet_loras}
            | {m.lora_name for bank in self.fake_banks for m in bank.unet_loras}
        )
        n_frozen = 0
        for name, param in self.unet.named_parameters():
            if name.startswith(lora_prefixes):
                continue
            param.requires_grad_(False)
            n_frozen += 1
        logger.info(f"freeze_dit: {n_frozen} base params frozen")

    def save_student(
        self,
        file: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Serialize only the student LoRA in the standard plain-LoRA layout.

        Output is loadable by ``inference.py --lora_weight <file>`` — the
        fake network is training scaffolding and never shipped.
        """
        self.student.save_weights(file, dtype, metadata)
        logger.info(f"saved student LoRA → {file}")
