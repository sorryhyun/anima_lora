"""Turbo Anima DP-DMD harness: student + fake ``LoRANetwork`` stacks on one frozen DiT.

The saved student is a plain LoRA. Docs: ``docs/anima/structure/turbo.md`` (structure),
``docs/anima/methods/turbo.md`` (ops). Paper: arXiv:2602.03139.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.attn_fuse import match_fused_spec
from networks.lora_anima.factory import create_network
from networks.lora_anima.network import LoRANetwork

logger = logging.getLogger(__name__)

View = Literal["teacher", "student", "fake"]


class FakeTransitionEmbed(nn.Module):
    """σ of the student transition a fake sample came from → t-embedding delta.

    VA-TDM's transition-aware field (arXiv:2611.06796, ``project/anima_turbo/vatdm.md``
    idea 1): the fake's ``x_pred`` samples come from grad steps at different σ_g, each
    with its own distribution. The output is added to the DiT's normalised t-embedding under the
    fake view only. ``fc2`` is zero-initialised, so the fake starts as the plain one.
    fp32; the fake is discarded at save.
    """

    def __init__(self, dim: int, hidden: int = 256, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = int(freq_dim)
        self.fc1 = nn.Linear(self.freq_dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """``sigma`` (B,) in [0, 1] → (B, dim)."""
        half = self.freq_dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=sigma.device))
            * torch.arange(half, dtype=torch.float32, device=sigma.device)
            / half
        )
        arg = (1000.0 * sigma.float())[:, None] * freqs[None, :]
        emb = torch.cat([torch.cos(arg), torch.sin(arg)], dim=-1)
        return self.fc2(F.silu(self.fc1(emb)))


class TeacherFeatureDiscriminator(nn.Module):
    """GAN head over frozen-teacher block features (FastGen idea 1).

    - ``granularity="pooled"``: mean-pool each tap over every axis between
      batch and channel -> ``(B, D)``, then a 2-layer MLP per tap -> ``(B, num_taps)``.
    - ``granularity="token"`` (LADD-style dense head): the same per-tap MLP on
      every token -> ``(B, num_taps·N)``.

    Neither reshapes to a patch grid, so both run unchanged on the eager
    ``(B,T,H,W,D)`` and compiled ``(B,1,L,1,D)`` layouts. fp32. The parameters are
    identical across granularities, so resume bundles load across a head switch.

    ``pool`` k > 1 (token head): the tokens are folded to the ``grid`` (h, w) given
    to ``forward`` and mean-pooled k×k before the MLP, one logit per cell; rows and
    columns past a multiple of k are dropped.
    ~inner_dim²/2 params per tap (~2M at D=2048); discarded at save.
    """

    def __init__(
        self,
        *,
        inner_dim: int,
        num_taps: int,
        hidden_dim: int | None = None,
        granularity: Literal["pooled", "token"] = "pooled",
        pool: int = 1,
    ) -> None:
        super().__init__()
        if granularity not in ("pooled", "token"):
            raise ValueError(f"unknown disc granularity: {granularity!r}")
        if pool < 1 or (pool > 1 and granularity != "token"):
            raise ValueError(f"disc pool={pool} needs pool >= 1 and the token head")
        self.granularity = granularity
        self.pool = pool
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

    def _pool_tokens(self, f: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        """(B, ..., D) tokens in row-major order → (B, h//k · w//k, D) cell means."""
        h, w = grid
        k, B, D = self.pool, f.shape[0], f.shape[-1]
        f = f.reshape(B, -1, D)
        if f.shape[1] < h * w:
            raise ValueError(f"{f.shape[1]} tokens for a {h}x{w} grid")
        f = f[:, : h * w].reshape(B, h, w, D)[:, : h // k * k, : w // k * k]
        return f.reshape(B, h // k, k, w // k, k, D).mean((2, 4)).flatten(1, 2)

    def forward(
        self, feats: list[torch.Tensor], grid: tuple[int, int] | None = None
    ) -> torch.Tensor:
        if len(feats) != len(self.heads):
            raise ValueError(
                f"TeacherFeatureDiscriminator expected {len(self.heads)} feature "
                f"tensors, got {len(feats)}"
            )
        if self.pool > 1 and grid is None:
            raise ValueError(f"disc pool={self.pool} needs the token grid (h, w)")
        logits = []
        for head, f in zip(self.heads, feats):
            if self.granularity == "token":
                f = f.float()
                if self.pool > 1:
                    f = self._pool_tokens(f, grid)
                logits.append(head(f).flatten(1))
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
    ``StepExpertLoRAModule`` network on the fused DiT and loads the state dict
    directly (no split→re-fuse). Returns it applied and loaded, ready for
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
    factor + SVD of the k×k core), not the full ``[out, in]`` matrix: a
    196-module warm start takes seconds. Folds the module's fixed
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
    rank may differ from the network's in either direction. Modules absent from
    the file keep their constructed init. Must run before ``compile_dit_blocks``
    traces the forwards.
    """
    from safetensors.torch import load_file

    weights_sd = load_file(weights_path)
    # A ComfyUI-native adaln extract names its keys adaln_modulation_{br}_2; the
    # modules are adaln_up_{br}. Without the rename the 84 adaln modules silently
    # keep their default init.
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
    """Two LoRA stacks on one frozen DiT, view-toggled per forward.

    A coordinator, not an ``nn.Module``; the caller owns the DiT.
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
        train_adaln: bool = False,
        fake_adaln: bool | None = None,
        adaln_rank: int = 0,
        adaln_alpha: float = 0.0,
        gan_feature_indices: set[int] | None = None,
        gan_disc_hidden: int | None = None,
        gan_disc_head: Literal["pooled", "token"] = "pooled",
        gan_disc_pool: int = 1,
        fake_transition: bool = False,
        before_apply: Callable[[], None] | None = None,
    ) -> None:
        self.unet = unet
        self.student_rank = int(student_rank)
        self.fake_rank = int(fake_rank)
        # adaln_up_{branch} targets on top of the attn + MLP set: adaln is the largest
        # mover in the official turbo delta (docs/anima/methods/adaln.md). The saved
        # student is renamed to the ComfyUI adaln layout in save_student.
        self.train_adaln = bool(train_adaln)
        self.fake_adaln = self.train_adaln if fake_adaln is None else bool(fake_adaln)
        # Own rank / alpha for the adaln modules (0 = the network's), on both stacks.
        self.adaln_rank = int(adaln_rank)
        self.adaln_alpha = float(adaln_alpha)
        if self.adaln_rank < 0 or self.adaln_alpha < 0:
            raise ValueError("adaln_rank / adaln_alpha must be >= 0")
        if (self.adaln_rank > 0 or self.adaln_alpha > 0) and not self.train_adaln:
            raise ValueError("adaln_rank / adaln_alpha > 0 require train_adaln=true")
        # Per-input-channel rebalance absorbed into each lora_down (exact at init,
        # merges out at save); 0 off, 0.5 sqrt-balance.
        self.channel_scaling_alpha = float(channel_scaling_alpha)
        self.student_down_init = str(student_down_init)
        self.fake_down_init = str(fake_down_init)
        for name, down_init in (
            ("student", self.student_down_init),
            ("fake", self.fake_down_init),
        ):
            if down_init not in ("kaiming", "weight_svd"):
                raise ValueError(
                    f"{name}_down_init={down_init!r}: expected 'kaiming' or 'weight_svd'."
                )

        self.student: LoRANetwork = self._create_stack(
            self.student_rank, student_alpha, self.student_down_init, self.train_adaln
        )
        self.fake: LoRANetwork = self._create_stack(
            self.fake_rank, fake_alpha, self.fake_down_init, self.fake_adaln
        )

        # The stacks are built and still hold the DiT's own forward: an int8 base
        # converts here, so weight_svd reads bf16 and org_forward is the int8 forward.
        if before_apply is not None:
            before_apply()

        # Apply student-first so the runtime chain is linear -> fake -> student ->
        # original.
        for net in (self.student, self.fake):
            net.apply_to(
                text_encoders=[],
                unet=unet,
                apply_text_encoder=False,
                apply_unet=True,
            )
        logger.info(
            f"TurboDMDNetwork: student rank={self.student_rank} "
            f"({len(self.student.unet_loras)} modules), fake rank={self.fake_rank} "
            f"({len(self.fake.unet_loras)} modules)"
        )

        # LoRAModule defaults to enabled=True and set_view diffs against
        # _VIEW_FLAGS[self._view], so both stacks must start disabled to match
        # the initial "teacher" view.
        self.student.set_enabled(False)
        self.fake.set_enabled(False)
        self._view: View = "teacher"

        # Transition-aware fake: a forward hook on the DiT's t_embedding_norm adds
        # FakeTransitionEmbed(σ_src) under the fake view. The norm runs in the
        # uncompiled top-level forward, so the hook fires under compile_blocks.
        # Every fake-view forward needs a source σ (set_fake_source).
        self.fake_transition: FakeTransitionEmbed | None = None
        self._fake_src: float | torch.Tensor | None = None
        if fake_transition:
            self.fake_transition = FakeTransitionEmbed(unet.model_channels)
            unet.t_embedding_norm.register_forward_hook(self._t_embedding_hook)
            logger.info(
                "TurboDMDNetwork: transition-aware fake "
                f"({sum(p.numel() for p in self.fake_transition.parameters()):,} params, "
                "zero-init)"
            )

        # GAN: the disc reads frozen-teacher block activations through the DiT's
        # feature tap (return_block_features / return_features_early); the caller
        # passes the dict through features_in_order.
        self.disc: TeacherFeatureDiscriminator | None = None
        self.gan_feature_indices: list[int] = []
        if gan_feature_indices:
            self.gan_feature_indices = sorted(gan_feature_indices)
            self.disc = TeacherFeatureDiscriminator(
                inner_dim=unet.model_channels,
                num_taps=len(self.gan_feature_indices),
                hidden_dim=gan_disc_hidden,
                granularity=gan_disc_head,
                pool=gan_disc_pool,
            )
            logger.info(
                f"TurboDMDNetwork: GAN disc attached (taps={self.gan_feature_indices}, "
                f"head={gan_disc_head}, pool={gan_disc_pool}, "
                f"inner_dim={unet.model_channels}, "
                f"{sum(p.numel() for p in self.disc.parameters()):,} params)"
            )

    def _create_stack(
        self, rank: int, alpha: float | None, down_init: str, adaln: bool
    ) -> LoRANetwork:
        """One plain LoRA stack (alpha = rank unless given)."""
        kwargs: dict = {}
        if down_init != "kaiming":
            kwargs["down_init"] = down_init
        if adaln:
            kwargs["include_patterns"] = [".*adaln_up_.*"]
            if self.adaln_rank > 0:
                kwargs["network_reg_dims"] = f".*adaln_up_.*={self.adaln_rank}"
            if self.adaln_alpha > 0:
                kwargs["network_reg_alphas"] = f".*adaln_up_.*={self.adaln_alpha}"
        return create_network(
            multiplier=1.0,
            network_dim=rank,
            network_alpha=alpha if alpha is not None else rank,
            vae=None,
            text_encoders=[],
            unet=self.unet,
            channel_scaling_alpha=self.channel_scaling_alpha,
            **kwargs,
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
        """False during the student step (the GAN gen grad reaches only x_pred),
        True during the disc update."""
        if self.disc is not None:
            self.disc.requires_grad_(flag)

    # Per-view (student_on, fake_on).
    _VIEW_FLAGS: dict[str, tuple[bool, bool]] = {
        "teacher": (False, False),
        "student": (True, False),
        "fake": (False, True),
    }

    def set_view(self, view: View) -> None:
        """Set the stacks' enabled flags so the next DiT forward runs as ``view``
        (teacher: both off; student: student on; fake: fake on). Flips only the
        stack that changes, and nothing when already in ``view`` (no dynamo guard
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
            self.student.set_enabled(want_student)
        if want_fake != cur_fake:
            self.fake.set_enabled(want_fake)
        self._view = view

    @property
    def view(self) -> View:
        return self._view

    def set_fake_source(self, sigma: float | torch.Tensor | None) -> None:
        """σ of the student transition the fake's next inputs came from (a float, or
        a (B,) tensor). No-op without ``fake_transition``."""
        self._fake_src = sigma

    def _t_embedding_hook(self, module, inputs, output):
        if self._view != "fake":
            return None
        src = self._fake_src
        if src is None:
            raise RuntimeError(
                "transition-aware fake: fake-view forward with no source σ "
                "(call set_fake_source first)"
            )
        B = output.shape[0]
        if isinstance(src, torch.Tensor):
            sigma = src.detach().to(output.device, torch.float32).reshape(-1).expand(B)
        else:
            sigma = torch.full((B,), float(src), device=output.device)
        with torch.autocast(output.device.type, enabled=False):
            delta = self.fake_transition(sigma)
        return output + delta.to(output.dtype).unsqueeze(1)

    def student_params(self):
        """Trainable params for the student optimizer."""
        return [p for p in self.student.parameters() if p.requires_grad]

    def fake_params(self):
        """Trainable params for the fake optimizer (with the transition embed)."""
        params = [p for p in self.fake.parameters() if p.requires_grad]
        if self.fake_transition is not None:
            params += list(self.fake_transition.parameters())
        return params

    def freeze_dit(self) -> None:
        """Set ``requires_grad=False`` on every base DiT param.

        Call after both ``apply_to``: the LoRA modules are registered under
        ``unet``, so this skips names with a LoRA prefix (a wholesale
        ``unet.requires_grad_(False)`` would freeze the LoRAs too).
        """
        lora_prefixes = tuple(
            {m.lora_name for net in (self.student, self.fake) for m in net.unet_loras}
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
        """Write only the student, in the standard plain-LoRA layout
        (``inference.py --lora_weight <file>``)."""
        self.student.save_weights(file, dtype, metadata)
        logger.info(f"saved student LoRA → {file}")
