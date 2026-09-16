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


def load_step_expert_student(
    unet,
    weights_sd: dict[str, torch.Tensor],
    metadata: dict[str, str],
    *,
    multiplier: float = 1.0,
) -> LoRANetwork:
    """Rebuild a per-step-expert turbo student as a router-free, kept-live net.

    Per-step-expert checkpoints can't go through ``merge_to_dit`` (K up-heads
    don't fold into one DiT weight) nor the shared key-sniff (the
    ``.lora_ups.{k}.weight`` layout collides with Hydra's). Builds a fresh
    ``StepExpertLoRAModule`` network on the same fused DiT and
    ``load_state_dict`` matches directly — no split→re-fuse.

    This applies + loads + freezes, returning a net ready for
    ``set_step_index`` per denoise step.
    """
    K = int(metadata.get("ss_turbo_step_expert_K", "0") or "0")
    if K <= 1:
        raise RuntimeError(
            "load_step_expert_student called on a checkpoint without "
            "ss_turbo_step_expert_K > 1 — not a per-step-expert turbo file."
        )
    rank = int(metadata.get("ss_turbo_student_rank", "0") or "0")
    if rank <= 0:
        raise RuntimeError(
            "per-step-expert turbo checkpoint missing ss_turbo_student_rank "
            "metadata — cannot rebuild the student at the right rank."
        )
    # GOTCHA: scale = alpha/rank is fixed at module construction (load_state_dict
    # only overwrites the alpha buffer, not the cached scale), so build with the
    # trained alpha; fall back to rank (scale 1.0) when the stamp is absent.
    alpha = float(metadata.get("ss_turbo_student_alpha", str(rank)) or rank)

    network = create_network(
        multiplier=multiplier,
        network_dim=rank,
        network_alpha=alpha,
        vae=None,
        text_encoders=[],
        unet=unet,
        step_expert_K=K,
    )
    network.apply_to([], unet, apply_text_encoder=False, apply_unet=True)
    info = network.load_state_dict(weights_sd, strict=False)
    if info.unexpected_keys:
        logger.warning(
            f"step-expert turbo: unexpected keys in state dict: "
            f"{info.unexpected_keys[:5]}..."
        )
    if info.missing_keys:
        # Surface the count so a rank/target mismatch is loud, not silent.
        logger.warning(
            f"step-expert turbo: {len(info.missing_keys)} missing keys "
            f"(first: {info.missing_keys[:5]})"
        )
    network.set_step_index(0)
    logger.info(
        f"step-expert turbo: router-free kept-live attached "
        f"({len(network.unet_loras)} modules, K={K} heads, rank={rank})"
    )
    return network


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
            f"{weights_path}: MoE/step-expert layout — warm start needs a "
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
        use_custom_down_autograd: bool = False,
        channel_scaling_alpha: float = 0.0,
        student_step_expert_K: int = 0,
        dual_pool: bool = False,
        div_pool_rank: int = 16,
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
        # SmoothQuant-style per-input-channel rebalance absorbed into each
        # lora_down (bit-equivalent at init, merges out cleanly). 0.0 = off,
        # 0.5 = sqrt-balance. Applied to both student and fake — conditions the
        # LoRA gradient on the DiT's outlier-DC input channels only, so it
        # leaves the DP-DMD objective untouched.
        self.channel_scaling_alpha = float(channel_scaling_alpha)
        # Per-step expert: when > 1 the student's adapted Linears become
        # StepExpertLoRAModule (shared down + K step-indexed up-heads); head k
        # is trained only by step-k's gradient (set_student_step + the detach
        # in distill.py). The fake/critic stays a plain single-head LoRA.
        self.student_step_expert_K = int(student_step_expert_K)

        # SVD-Down init (down_init="weight_svd"): seed the plain-LoRA student's
        # lora_down from W0's top-r right singular vectors (scale-matched).
        # Targets the plain LoRAModule only, so mutually exclusive with the
        # per-step-expert student.
        self.student_down_init = str(student_down_init)
        if self.student_down_init not in ("kaiming", "weight_svd"):
            raise ValueError(
                f"student_down_init={self.student_down_init!r}: "
                "expected 'kaiming' or 'weight_svd'."
            )
        if self.student_down_init == "weight_svd" and self.student_step_expert_K > 1:
            raise ValueError(
                "student_down_init='weight_svd' is incompatible with the "
                "per-step-expert student (step_expert_K>1): StepExpertLoRAModule "
                "is not a plain LoRAModule. Disable per_step_expert."
            )
        # Same lever on the fake/critic. It is always a plain single-head LoRA.
        self.fake_down_init = str(fake_down_init)
        if self.fake_down_init not in ("kaiming", "weight_svd"):
            raise ValueError(
                f"fake_down_init={self.fake_down_init!r}: "
                "expected 'kaiming' or 'weight_svd'."
            )

        # Plain LoRA on both (LoRANetworkCfg defaults: no MoE/T-LoRA); alpha =
        # rank by default. use_custom_down_autograd is forwarded for config
        # compat but a deprecated no-op. step_expert_K rides **kwargs; >1 flips
        # resolve_network_spec to StepExpertLoRAModule for the student only.
        # adaln_rank > 0 builds the adaln modules at their own (lower) rank via
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
        if self.student_step_expert_K > 1:
            _student_kwargs["step_expert_K"] = self.student_step_expert_K
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
            use_custom_down_autograd=use_custom_down_autograd,
            channel_scaling_alpha=self.channel_scaling_alpha,
            **_student_kwargs,
        )

        # Dual-pool gradient routing: a SECOND always-on plain-LoRA pool. Pool A
        # (this one) receives only the step-0 diversity gradient; pool B
        # (self.student) only the DMD/GAN/CDM refinement gradients (routing is
        # `requires_grad` gating in distill.py — both pools stay *enabled*, so
        # activations always flow through A+B). Pool A is deliberately MINIMAL:
        # plain LoRA on attn/MLP only, no adaln/per-step-expert/channel scaling.
        # Zero-init up means at step 0 the merged student IS the warm-started
        # pool B, and A grows a pure de-collapse correction from zero.
        self.dual_pool = bool(dual_pool)
        self.div_pool_rank = int(div_pool_rank)
        self.student_div: LoRANetwork | None = None
        if self.dual_pool:
            if self.student_step_expert_K > 1:
                raise ValueError(
                    "dual_pool is incompatible with the per-step-expert student "
                    "(step_expert_K>1): both restructure the student stack."
                )
            if self.div_pool_rank < 1:
                raise ValueError(f"div_pool_rank must be >= 1, got {div_pool_rank}")
            self.student_div = create_network(
                multiplier=1.0,
                network_dim=self.div_pool_rank,
                network_alpha=self.div_pool_rank,  # scale 1.0; zero-init up
                vae=None,
                text_encoders=[],
                unet=unet,
                use_custom_down_autograd=use_custom_down_autograd,
                channel_scaling_alpha=0.0,
            )
        # Both pools train under the student optimizer and toggle together in the
        # "student" view. Bank order mirrors fake_banks: pool B first (the warm
        # start / save anchor), pool A second.
        self.student_pools: list[LoRANetwork] = [self.student]
        if self.student_div is not None:
            self.student_pools.append(self.student_div)
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
                use_custom_down_autograd=use_custom_down_autograd,
                channel_scaling_alpha=self.channel_scaling_alpha,
                **_fake_kwargs,
            )

        self.fake: LoRANetwork = _make_fake()  # bank 0 (banks=2: the low-τ bank)
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
        if self.student_div is not None:
            self.student_div.apply_to(
                text_encoders=[],
                unet=unet,
                apply_text_encoder=False,
                apply_unet=True,
            )
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

        logger.info(
            f"TurboDMDNetwork: student rank={self.student_rank} "
            f"({len(self.student.unet_loras)} modules), "
            f"fake rank={self.fake_rank} "
            f"({len(self.fake.unet_loras)} modules"
            + (f", x{self.fake_tau_banks} τ-banks" if self.fake_tau_banks > 1 else "")
            + ")"
        )

        # GOTCHA: LoRAModule defaults enabled=True, and set_view short-circuits
        # when view==self._view, so we MUST disable every stack explicitly here
        # — else the set_view invariant (cur state matches _VIEW_FLAGS[_view])
        # breaks once any stack carries nonzero weights.
        self.student.set_enabled(False)
        if self.student_div is not None:
            self.student_div.set_enabled(False)
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
        cur_student, cur_fake = self._VIEW_FLAGS[self._view]
        if want_student != cur_student:
            # Both dual pools toggle together — "student" view is always A+B
            # (grad routing is `requires_grad`, not enablement; see route_*).
            for pool in self.student_pools:
                pool.set_enabled(want_student)
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
        """The fake stacks in bank order (bank 0 = low-τ; length 1 unless split)."""
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

    def set_student_step(self, i: int) -> None:
        """Select the student's step-``i`` up-head before its forward.

        No-op when per-step-expert is off. The detach between the diversity
        step-0 backward and the DMD steps-1..N chain (distill.py) means head 0
        only ever sees the diversity gradient and head k only step-k's DMD
        gradient — the shared lora_down is trained by both. Kept as a
        coordinator method so the training loop never reaches into the student
        network directly.
        """
        if self.student_step_expert_K > 1:
            self.student.set_step_index(i)

    def student_params(self):
        """Trainable params for the student optimizer (all pools under dual_pool).

        GOTCHA: filtered by ``requires_grad`` — the caller must restore both
        pools to grad-on (``route_all_on``) BEFORE grad-clip/step, else a
        transiently-gated pool's grads escape clipping.
        """
        return [
            p
            for pool in self.student_pools
            for p in pool.parameters()
            if p.requires_grad
        ]

    def student_param_groups(self, student_lr: float, div_pool_lr: float = 0.0):
        """AdamW param groups: pool B at ``student_lr``, pool A at ``div_pool_lr``.

        ``div_pool_lr=0`` inherits ``student_lr``. Under single-pool this is one
        group.
        """
        groups = [{"params": list(self.student.parameters()), "lr": student_lr}]
        if self.student_div is not None:
            groups.append(
                {
                    "params": list(self.student_div.parameters()),
                    "lr": div_pool_lr if div_pool_lr > 0.0 else student_lr,
                }
            )
        return groups

    def _set_pool_requires_grad(self, pool: "LoRANetwork | None", flag: bool) -> None:
        if pool is None:
            return
        for p in pool.parameters():
            p.requires_grad_(flag)

    def route_div(self) -> None:
        """Gate grads to pool A only (the step-0 diversity backward).

        No-op under single-pool. Both pools stay ENABLED (activations flow); only
        pool B's grad accumulation is switched off so the diversity/soft-rank
        backward lands on pool A alone.
        """
        if self.student_div is None:
            return
        self._set_pool_requires_grad(self.student_div, True)
        self._set_pool_requires_grad(self.student, False)

    def route_quality(self) -> None:
        """Gate grads to pool B only (DMD / GAN / CDM refinement backwards)."""
        if self.student_div is None:
            return
        self._set_pool_requires_grad(self.student, True)
        self._set_pool_requires_grad(self.student_div, False)

    def route_all_on(self) -> None:
        """Restore grad-on for both pools (call before grad-clip / optimizer step)."""
        if self.student_div is None:
            return
        self._set_pool_requires_grad(self.student, True)
        self._set_pool_requires_grad(self.student_div, True)

    def fake_params(self):
        """Trainable params for the fake optimizer — all banks, bank order.

        One optimizer spans every bank: with ``zero_grad(set_to_none=True)`` the
        inactive bank's grads are None and AdamW skips them.
        """
        return [
            p for bank in self.fake_banks for p in bank.parameters() if p.requires_grad
        ]

    def freeze_dit(self) -> None:
        """Set ``requires_grad=False`` on every base DiT param.

        GOTCHA: must be called AFTER both ``apply_to``'s — the LoRA networks
        add sub-modules to ``unet``, so a wholesale ``unet.requires_grad_(False)``
        after apply would zero the LoRA params too. Selectively walks only
        ``unet`` params whose name doesn't start with a LoRA prefix.
        """
        lora_prefixes = tuple(
            {m.lora_name for pool in self.student_pools for m in pool.unet_loras}
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
        div_scale: float = 1.0,
    ) -> None:
        """Serialize only the student LoRA in the standard plain-LoRA layout.

        Output is loadable by ``inference.py --lora_weight <file>`` — the
        fake network is training scaffolding and never shipped.

        Under dual_pool the two pools are merged EXACTLY into one plain LoRA of
        rank ``student_rank + div_pool_rank`` via factor concatenation (no SVD).
        ``div_scale`` dials pool A at merge time (0 = the warm-start-ish quality
        map, 1 = full anchor correction) — the free diversity knob.
        """
        if self.student_div is not None:
            self._save_dual_pool(file, dtype, metadata, div_scale)
            return
        if self.student_step_expert_K > 1:
            sd = self.student.state_dict()
            # Strip any non-LoRA keys defensively (per-step-expert layout is
            # written verbatim).
            sd = {k: v for k, v in sd.items() if ".lora_" in k or ".alpha" in k}
            self._save_student_step_expert(sd, file, dtype, metadata)
            return

        # Delegate to the network's own save so the full distill chain runs.
        # GOTCHA: an OrthoInit/OrthoLoRA student stores P_init/Q_init/lambda_layer
        # (or Cayley/SVD bases), NOT lora_down/lora_up — save_weights distills
        # those into the standard factorization. A naive pre-filter here would
        # strip them before the distill step ever sees them.
        self.student.save_weights(file, dtype, metadata)
        logger.info(f"saved student LoRA → {file}")

    @staticmethod
    def _finalize_pool_state_dict(
        pool: "LoRANetwork", metadata: dict[str, str] | None
    ) -> tuple[dict[str, torch.Tensor], dict[str, str] | None]:
        """One plain-LoRA pool → its on-disk (defused, adaln-relayed) fp32 dict.

        dtype=None keeps fp32 so the concat/scale-bake is exact; the caller casts
        the merged dict once at the end.
        """
        from networks import lora_save

        sd = pool.state_dict()
        for prefix in getattr(pool, "_training_only_prefixes", ()):
            for key in [k for k in sd if k.startswith(prefix)]:
                del sd[key]
        return lora_save.build_standard_state_dict(sd, None, metadata)

    def _save_dual_pool(
        self,
        file: str,
        dtype: torch.dtype,
        metadata: dict[str, str] | None,
        div_scale: float,
    ) -> None:
        """Merge pool B (quality) + pool A (diversity) into one exact plain LoRA.

        Two rank-r LoRAs on the same Linear fold to rank r_B+r_A by factor
        concatenation: ``down = concat(rows)``, ``up = concat(cols)`` reproduces
        ``ΔW_B + ΔW_A`` identically. Per-pool scale (alpha/rank) is baked into
        ``up`` and the merged alpha set to the merged rank (scale 1). ``div_scale``
        multiplies pool A's up — the merge-time diversity dial. Pool A's module
        set ⊆ pool B's; adaln-only keys pass through from B untouched.
        """
        from safetensors.torch import save_file
        from library.training.hashing import precalculate_safetensors_hashes

        sd_b, meta = self._finalize_pool_state_dict(self.student, dict(metadata or {}))
        sd_a, _ = self._finalize_pool_state_dict(self.student_div, None)
        meta = meta or {}

        merged: dict[str, torch.Tensor] = dict(sd_b)  # adaln-only keys ride through
        down_suffix = ".lora_down.weight"
        for a_down_key in [k for k in sd_a if k.endswith(down_suffix)]:
            prefix = a_down_key[: -len(down_suffix)]
            up_key = prefix + ".lora_up.weight"
            alpha_key = prefix + ".alpha"
            if prefix + down_suffix not in sd_b or up_key not in sd_b:
                raise ValueError(
                    f"dual-pool merge: pool A module {prefix!r} absent from pool B "
                    "(pool A must be a subset of pool B's module set)."
                )
            down_a, up_a = sd_a[a_down_key].float(), sd_a[up_key].float()
            down_b, up_b = sd_b[a_down_key].float(), sd_b[up_key].float()
            r_a, r_b = down_a.shape[0], down_b.shape[0]
            scale_a = (float(sd_a[alpha_key]) / r_a) if alpha_key in sd_a else 1.0
            scale_b = (float(sd_b[alpha_key]) / r_b) if alpha_key in sd_b else 1.0
            # Bake scale into up; div_scale dials pool A. B first, A second.
            up_merged = torch.cat([up_b * scale_b, up_a * (scale_a * div_scale)], dim=1)
            down_merged = torch.cat([down_b, down_a], dim=0)
            merged[prefix + down_suffix] = down_merged
            merged[up_key] = up_merged
            r_merged = r_a + r_b
            if alpha_key in sd_b:
                merged[alpha_key] = torch.tensor(
                    float(r_merged), dtype=sd_b[alpha_key].dtype
                )
            else:
                merged[alpha_key] = torch.tensor(float(r_merged))

        # Provenance: this IS a plain LoRA (loads with no special path); the
        # dual-pool stamps are for arm verification only — never trust the TOML.
        spec = getattr(self.student, "_network_spec", None)
        if spec is not None:
            meta["ss_network_spec"] = spec.name
        meta["ss_turbo_dual_pool"] = "true"
        meta["ss_turbo_quality_pool_rank"] = str(int(self.student_rank))
        meta["ss_turbo_div_pool_rank"] = str(int(self.div_pool_rank))
        if div_scale != 1.0:
            meta["ss_turbo_div_scale"] = str(float(div_scale))

        if dtype is not None:
            merged = {
                k: v.detach().clone().to("cpu").to(dtype) for k, v in merged.items()
            }
        model_hash, legacy_hash = precalculate_safetensors_hashes(merged, meta)
        meta["sshs_model_hash"] = model_hash
        meta["sshs_legacy_hash"] = legacy_hash
        save_file(merged, file, meta)
        logger.info(
            f"saved dual-pool student LoRA → {file}  (merged rank "
            f"{self.student_rank}+{self.div_pool_rank}={self.student_rank + self.div_pool_rank}, "
            f"div_scale={div_scale}; exact plain-LoRA concat)"
        )

    def _save_student_step_expert(
        self,
        sd: dict[str, torch.Tensor],
        file: str,
        dtype: torch.dtype,
        metadata: dict[str, str] | None,
    ) -> None:
        """Write the per-step-expert student in its bespoke layout.

        Multi-head keys (``...lora_ups.{k}.weight``) are NOT plain-LoRA loadable,
        so the standard defuse-qkv save pipeline can't run. Keeps the fused-qkv
        key layout the training-runtime DiT uses verbatim — CLI/ComfyUI
        step-expert loaders rebuild the network on the same fused DiT and
        ``load_state_dict`` matches directly. The ``.lora_ups.`` keys
        deliberately make a stock loader fail loudly rather than silently
        mis-merging only head 0.
        """
        from safetensors.torch import save_file
        from library.training.hashing import precalculate_safetensors_hashes
        from networks.lora_modules.lora import bake_inv_scale

        # Fold any per-channel scaling into lora_down (no-op when absent) so the
        # on-disk delta acts on raw inputs.
        bake_inv_scale(sd)

        if dtype is not None:
            sd = {k: v.detach().clone().to("cpu").to(dtype) for k, v in sd.items()}

        meta = dict(metadata or {})
        model_hash, legacy_hash = precalculate_safetensors_hashes(sd, meta)
        meta["sshs_model_hash"] = model_hash
        meta["sshs_legacy_hash"] = legacy_hash

        save_file(sd, file, meta)
        n_heads = self.student_step_expert_K
        logger.info(
            f"saved step-expert student LoRA → {file}  ({len(sd)} keys, "
            f"K={n_heads} up-heads/Linear; kept-live only — not plain-LoRA mergeable)"
        )
