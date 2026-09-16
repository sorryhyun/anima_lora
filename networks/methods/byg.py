"""BYG — Bootstrap Your Generator: unpaired instruction editing for Anima.

Paper: *Bootstrap Your Generator: Unpaired Visual Editing with Flow Matching*
(Tewel, Atzmon, Chechik, Wolf; arXiv 2606.03911). Proposal:
``bench/byg/README.md``.

BYG trains a **plain rank-64 LoRA** to follow edit instructions with *no paired
(source, edited) data and no reward model* — only unpaired images plus
``(src_caption, tgt_caption, instruction, reverse_instruction)`` tuples. The
trainable weights are an ordinary Anima LoRA; everything BYG-specific lives in
this module's ``BYGMethodAdapter``, which (a) installs a **parameter-free
source-latent concat** conditioning patch on the DiT blocks and (b) **owns the
whole training step** (the standard ``flow_match`` path doesn't fit — BYG has no
``target = noise - latents``).

Conditioning — Kontext-native single-concat semantics (gate-free, source shares
the target's positions). On Anima a *literal* pre-PatchEmbed sequence concat
breaks the eager 5D bucketing (source+target form no rectangular grid), so the
concat is realized at the **attention level**: each block's target tokens attend
to ``[target_k ; source_k]`` while the source stream evolves through the same
LoRA'd block linears. This reuses EasyControl's extended-self-attention LSE
helper but **strips the ``b_cond`` gate (fixed at 0) and the cond-LoRA branch** —
the only trainable weights are the standard LoRA, so the checkpoint is a plain
LoRA loadable by the normal loader. Source/target are always same-resolution, so
the DiT's native RoPE already gives matching positions (paper App. B.2).

Objective (paper Alg. 1), per step with ``t~U(0,1)``, ``ε~N(0,I)``, source ``x``:
  1. Bootstrap (no_grad, snapshot weights, concat=x, instr=c): n-step Euler
     rollout ``ỹ_1=ε → ỹ_0`` capturing ``ỹ_t`` mid-rollout — yields BOTH the
     pseudo-noisy-target ``ỹ_t`` and the clean ``ỹ_0``.
  2. Forward (grad, concat=x, instr=c): ``v_fwd=G(ỹ_t,t,c,x)``; ``ŷ=ỹ_t−t·v_fwd``.
  3. Prior (no_grad base, multiplier=0, no concat): ``v_src=G_t2i(ỹ_t,t,p_src)``,
     ``v_tgt=G_t2i(ỹ_t,t,p_tgt)``; ``L_dir=1−cos(v_fwd−v_src,v_tgt−v_src)``,
     ``L_MSE=‖v_fwd−v_tgt‖²``, ``L_prior=L_dir+α·L_MSE``.
  4. Cycle (grad, concat=ŷ_hyb, instr=c̄): ``x_t=(1−t)x+tε``,
     ``ŷ_hyb=sg(ỹ_0)+(ŷ−sg(ŷ))``, ``v_rev=G(x_t,t,c̄,ŷ_hyb)``,
     ``L_cycle=‖v_rev−(ε−x)‖²``.
  5. Identity (grad, concat=x, instr=c̄): ``L_id=‖G(x_t,t,c̄,x)−(ε−x)‖²`` —
     staged on an independent graph (anti-collapse anchor + VRAM win).

Deviations from the paper: snapshot bootstrap instead of EMA
(``byg_ema_decay`` toggles EMA); t discretized to the rollout grid for exact
``ỹ_t`` capture. The symmetric prior (paper Eq. 5, ``L_prior^fwd +
L_prior^rev``) is on by default; ``byg_prior_symmetric = false`` uses the
fwd-only prior (two fewer frozen-base forwards/step).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library.log import setup_logging
from library.training.method_adapter import ComputeLossCtx, MethodAdapter, SetupCtx
from library.training.forward.ste import ste_clean_blend
from networks.methods.easycontrol_attention import _extended_target_attention

setup_logging()
logger = logging.getLogger(__name__)


def directional_prior_loss(
    v_fwd: torch.Tensor,
    v_src: torch.Tensor,
    v_tgt: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """DDS-style prior: align the *edit direction* with the T2I prior (Eq. 2)
    plus a magnitude anchor (paper §4.2).

        L_dir = 1 − cos(v_fwd − v_src, v_tgt − v_src)
        L_MSE = ‖v_fwd − v_tgt‖²
        L_prior = L_dir + alpha·L_MSE

    ``v_src`` / ``v_tgt`` are the frozen base's velocities (detached upstream).
    fp32 throughout; cosine denominator eps-clamped. Returns a scalar.
    """
    vf = v_fwd.float()
    vs = v_src.float()
    vt = v_tgt.float()
    edit = vf - vs
    tgt_dir = vt - vs
    # Per-sample cosine over all non-batch dims.
    dims = list(range(1, vf.ndim))
    dot = (edit * tgt_dir).sum(dim=dims)
    edit_norm = edit.pow(2).sum(dim=dims).clamp_min(1e-12).sqrt()
    tgt_norm = tgt_dir.pow(2).sum(dim=dims).clamp_min(1e-12).sqrt()
    cos = dot / (edit_norm * tgt_norm)
    l_dir = (1.0 - cos).mean()
    l_mse = (vf - vt).pow(2).mean()
    return l_dir + alpha * l_mse


def _make_byg_block_forward(block, block_idx, byg: "BYGConditioning"):
    """Stripped two-stream Block.forward for BYG.

    Identical structure to EasyControl's two-stream block, but with **no
    cond-LoRA delta and a fixed b_cond=0 gate** (source attended on equal
    footing). The source stream evolves through the block's own (standard-LoRA
    patched) linears, so the trainable LoRA adapts the unified model. Falls
    through to the original forward when no source is set (prior / baseline).
    """
    original_forward = block.forward
    is_last = block_idx == byg.num_blocks - 1

    from networks import attention_dispatch as anima_attention

    def _two_stream_inner(
        x_B_T_H_W_D,
        emb_B_T_D,
        crossattn_emb,
        attn_params,
        rope_cos_sin,
        adaln_lora_B_T_3D,
        src_x_B_S_D,
        src_emb_B_T_D,
        src_adaln_lora_B_T_3D,
        src_rope_cos_sin,
    ):
        attn = block.self_attn
        T_dim, H_dim, W_dim = x_B_T_H_W_D.shape[1:4]
        scale_attn = attn_params.softmax_scale
        b_param = byg.zero_b(x_B_T_H_W_D.device)

        # AdaLN modulation for both streams.
        if block.use_adaln_lora:
            fused_down = block.adaln_fused_down(emb_B_T_D)
            down_self, down_cross, down_mlp = fused_down.chunk(3, dim=-1)
            shift_self_attn, scale_self_attn, gate_self_attn = (
                block.adaln_up_self_attn(down_self) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn, scale_cross_attn, gate_cross_attn = (
                block.adaln_up_cross_attn(down_cross) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = (
                block.adaln_up_mlp(down_mlp) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)

            src_fused_down = block.adaln_fused_down(src_emb_B_T_D)
            src_down_self, _src_down_cross, src_down_mlp = src_fused_down.chunk(
                3, dim=-1
            )
            src_shift_self_attn, src_scale_self_attn, src_gate_self_attn = (
                block.adaln_up_self_attn(src_down_self) + src_adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            src_shift_mlp, src_scale_mlp, src_gate_mlp = (
                block.adaln_up_mlp(src_down_mlp) + src_adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn, scale_self_attn, gate_self_attn = (
                block.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_cross_attn, scale_cross_attn, gate_cross_attn = (
                block.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            )
            shift_mlp, scale_mlp, gate_mlp = block.adaln_modulation_mlp(
                emb_B_T_D
            ).chunk(3, dim=-1)

            src_shift_self_attn, src_scale_self_attn, src_gate_self_attn = (
                block.adaln_modulation_self_attn(src_emb_B_T_D).chunk(3, dim=-1)
            )
            src_shift_mlp, src_scale_mlp, src_gate_mlp = block.adaln_modulation_mlp(
                src_emb_B_T_D
            ).chunk(3, dim=-1)

        sh_self_5 = shift_self_attn[:, :, None, None, :]
        sc_self_5 = scale_self_attn[:, :, None, None, :]
        ga_self_5 = gate_self_attn[:, :, None, None, :]
        sh_cross_5 = shift_cross_attn[:, :, None, None, :]
        sc_cross_5 = scale_cross_attn[:, :, None, None, :]
        ga_cross_5 = gate_cross_attn[:, :, None, None, :]
        sh_mlp_5 = shift_mlp[:, :, None, None, :]
        sc_mlp_5 = scale_mlp[:, :, None, None, :]
        ga_mlp_5 = gate_mlp[:, :, None, None, :]

        # SELF-ATTENTION (extended target + source's own).
        target_normed = (
            block.layer_norm_self_attn(x_B_T_H_W_D) * (1 + sc_self_5) + sh_self_5
        )
        target_flat = target_normed.flatten(1, 3)
        target_q, target_k, target_v = attn.compute_qkv(
            target_flat, target_flat, rope_cos_sin=rope_cos_sin
        )

        # Source Q/K/V — NO cond-LoRA; the block's qkv_proj already carries the
        # standard LoRA delta (it monkey-patches the Linear's forward).
        src_normed = (
            block.layer_norm_self_attn(src_x_B_S_D) * (1 + src_scale_self_attn)
            + src_shift_self_attn
        )
        src_q, src_k, src_v = attn.compute_qkv(
            src_normed, src_normed, rope_cos_sin=src_rope_cos_sin
        )

        # Target extended attention over [target_k; src_k] (b_cond = 0 → no gate).
        target_attn_out = _extended_target_attention(
            target_q,
            target_k,
            target_v,
            src_k,
            src_v,
            b_param=b_param,
            scale=scale_attn,
            attn_params=attn_params,
        )
        target_attn_proj = attn.output_dropout(attn.output_proj(target_attn_out))
        target_attn_5d = target_attn_proj.unflatten(1, (T_dim, H_dim, W_dim))
        x_B_T_H_W_D = x_B_T_H_W_D + ga_self_5 * target_attn_5d

        # Source's own self-attn — feeds the next block only → dead on last block.
        if not is_last:
            src_q = src_q.to(target_v.dtype)
            src_k = src_k.to(target_v.dtype)
            src_v = src_v.to(target_v.dtype)
            src_attn_out = anima_attention.dispatch_attention(
                [src_q, src_k, src_v], attn_params=attn_params
            )
            src_attn_proj = attn.output_dropout(attn.output_proj(src_attn_out))
            src_x_B_S_D = src_x_B_S_D + src_gate_self_attn * src_attn_proj

        # CROSS-ATTENTION (target only).
        target_cross_normed = (
            block.layer_norm_cross_attn(x_B_T_H_W_D) * (1 + sc_cross_5) + sh_cross_5
        )
        target_cross_out = block.cross_attn(
            target_cross_normed.flatten(1, 3),
            attn_params,
            crossattn_emb,
            rope_cos_sin=rope_cos_sin,
        ).unflatten(1, (T_dim, H_dim, W_dim))
        x_B_T_H_W_D = x_B_T_H_W_D + ga_cross_5 * target_cross_out

        # MLP.
        target_mlp_normed = (
            block.layer_norm_mlp(x_B_T_H_W_D) * (1 + sc_mlp_5) + sh_mlp_5
        )
        target_mlp_out = block.mlp(target_mlp_normed)
        x_B_T_H_W_D = x_B_T_H_W_D + ga_mlp_5 * target_mlp_out

        if not is_last:
            src_mlp_normed = (
                block.layer_norm_mlp(src_x_B_S_D) * (1 + src_scale_mlp) + src_shift_mlp
            )
            src_mlp_out = block.mlp(src_mlp_normed)
            src_x_B_S_D = src_x_B_S_D + src_gate_mlp * src_mlp_out

        return x_B_T_H_W_D, src_x_B_S_D

    def patched_forward(
        x_B_T_H_W_D,
        emb_B_T_D,
        crossattn_emb,
        attn_params,
        rope_cos_sin=None,
        adaln_lora_B_T_3D=None,
    ):
        state = byg.source_state
        if state is None:
            # No source set — exact baseline DiT (used by the prior forwards).
            return original_forward(
                x_B_T_H_W_D,
                emb_B_T_D,
                crossattn_emb,
                attn_params,
                rope_cos_sin=rope_cos_sin,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            )

        src_x_in = block._byg_source_x_in
        if src_x_in is None:
            raise RuntimeError(
                f"BYG: block[{block_idx}] has source_state but no _byg_source_x_in; "
                "did set_source run before the DiT forward?"
            )
        src_emb = state["src_emb"]
        src_adaln = state["src_adaln_lora"]
        src_rope = state["src_rope"]

        inner_args = (
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            attn_params,
            rope_cos_sin,
            adaln_lora_B_T_3D,
            src_x_in,
            src_emb,
            src_adaln,
            src_rope,
        )
        if block.training and block.gradient_checkpointing:
            if block.unsloth_offload_checkpointing:
                from library.anima.models import unsloth_checkpoint

                target_x_out, src_x_out = unsloth_checkpoint(
                    _two_stream_inner, *inner_args
                )
            else:
                target_x_out, src_x_out = torch_checkpoint(
                    _two_stream_inner, *inner_args, use_reentrant=False
                )
        else:
            target_x_out, src_x_out = _two_stream_inner(*inner_args)

        next_idx = block_idx + 1
        if next_idx < byg.num_blocks:
            byg.block_modules[next_idx]._byg_source_x_in = src_x_out
        return target_x_out

    return patched_forward


class BYGConditioning:
    """Param-free source-latent concat conditioning installed on the DiT.

    Holds no trainable parameters (it is NOT an nn.Module — keeping it off
    ``network.parameters()``). Patches every ``Block.forward`` to run the
    stripped two-stream when a source latent is primed via ``set_source`` and
    fall through to the baseline otherwise. The editing capacity is the
    separately-applied standard LoRA, which patches the block linears the
    two-stream calls.
    """

    def __init__(self, dit):
        self._dit = dit
        self.num_blocks = len(dit.blocks)
        self.block_modules = list(dit.blocks)
        self._original_forwards = []
        self.source_state: Optional[dict] = None
        self._patched = False
        self._zero_b_cache: dict = {}
        self._zero_emb_cache: dict = {}

    def zero_b(self, device) -> torch.Tensor:
        b = self._zero_b_cache.get(device)
        if b is None:
            b = torch.zeros((), device=device, dtype=torch.float32)
            self._zero_b_cache[device] = b
        return b

    def _zero_emb(self, B: int, device, dtype):
        """Cached source timestep embedding (always the zeros-timestep, so it is
        invariant per (B, device, dtype))."""
        key = (B, device, dtype)
        cached = self._zero_emb_cache.get(key)
        if cached is None:
            zeros = torch.zeros(B, 1, device=device, dtype=dtype)
            src_emb, src_adaln = self._dit.t_embedder(zeros)
            src_emb = self._dit.t_embedding_norm(src_emb)
            cached = (src_emb, src_adaln)
            self._zero_emb_cache[key] = cached
        return cached

    def apply(self):
        if self._patched:
            return
        for idx, block in enumerate(self.block_modules):
            self._original_forwards.append(block.forward)
            block._byg_source_x_in = None
            block.forward = _make_byg_block_forward(block, idx, self)
        self._patched = True
        logger.info(f"BYG: patched Block.forward on {self.num_blocks} blocks")

    def remove(self):
        for block, orig in zip(self.block_modules, self._original_forwards):
            block.forward = orig
            if hasattr(block, "_byg_source_x_in"):
                del block._byg_source_x_in
        self._original_forwards.clear()
        self.source_state = None
        self._patched = False

    def encode_source(
        self, source_latent: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ):
        """Patch-embed the source latent into ``[B, S, D]`` tokens + native RoPE.

        Differentiable (the cycle pass routes gradient through ŷ_hyb), reusing
        the DiT's frozen x_embedder / pos_embedder.
        """
        if source_latent.ndim == 4:
            source_latent = source_latent.unsqueeze(2)  # [B, C, 1, H, W]
        B, _, _, H, W = source_latent.shape
        if self._dit.concat_padding_mask and padding_mask is None:
            # zeros = "no padding / full real image" — matches the DiT's training
            # distribution and the target-stream forwards (train.py builds a
            # zeros padding_mask; ones would feed the source's concat'd pad
            # channel an off-distribution constant the model never saw).
            padding_mask = torch.zeros(
                B, 1, H, W, device=source_latent.device, dtype=source_latent.dtype
            )
        src_x_5d, src_rope = self._dit.prepare_embedded_sequence(
            source_latent, fps=None, padding_mask=padding_mask
        )
        return src_x_5d.flatten(1, 3), src_rope

    def set_source(
        self, source_latent: Optional[torch.Tensor], padding_mask=None
    ) -> None:
        """Prime per-forward source state (None → clear → baseline path)."""
        if source_latent is None:
            self.source_state = None
            return
        src_x, src_rope = self.encode_source(source_latent, padding_mask=padding_mask)
        self.set_source_precomputed(src_x, src_rope)

    def set_source_precomputed(self, src_x: torch.Tensor, src_rope) -> None:
        """Prime per-forward source state from an already-encoded source.

        Lets callers patch-embed an invariant source latent once and re-prime it
        across several DiT forwards in the same step (the frozen embedder makes
        the encoding identical regardless of the shadow swap / LoRA state).
        """
        B = src_x.shape[0]
        src_emb, src_adaln = self._zero_emb(B, src_x.device, src_x.dtype)
        self.source_state = {
            "src_emb": src_emb,
            "src_adaln_lora": src_adaln,
            "src_rope": src_rope,
        }
        self.block_modules[0]._byg_source_x_in = src_x

    def clear_source(self) -> None:
        self.source_state = None


def _crossattn_seqlens(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return mask.sum(dim=-1).to(torch.int32)


def _resolve_rollout_sigmas(args) -> Optional[list]:
    """Parse ``byg_rollout_sigmas`` into a validated descending node list, or None.

    Accepts a Python list/tuple (TOML array) or a comma-separated string (CLI),
    and returns ``[1.0, ..., 0.0]`` (n+1 nodes → n Euler steps). None ⇒ caller
    falls back to the uniform ``byg_rollout_steps`` grid (bit-identical path).
    The grid must start at 1.0, end at 0.0, and be strictly descending so every
    step has a positive dsigma and every interior node is a valid training t∈(0,1).
    """
    raw = getattr(args, "byg_rollout_sigmas", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        vals = [float(x) for x in raw.replace(" ", "").split(",") if x]
    else:
        vals = [float(x) for x in raw]
    if len(vals) < 3:
        raise ValueError(f"byg_rollout_sigmas needs >=3 nodes (>=2 steps); got {vals}")
    if abs(vals[0] - 1.0) > 1e-6 or abs(vals[-1]) > 1e-6:
        raise ValueError(
            f"byg_rollout_sigmas must start at 1.0 and end at 0.0; got {vals}"
        )
    if any(a <= b for a, b in zip(vals, vals[1:])):
        raise ValueError(f"byg_rollout_sigmas must be strictly descending; got {vals}")
    return vals


class BYGMethodAdapter(MethodAdapter):
    """Owns the BYG training step (paper Alg. 1)."""

    name = "byg"

    def __init__(self):
        self._cond: Optional[BYGConditioning] = None
        self._network = None
        self._dit = None
        self._args = None
        self._shadow: Optional[dict] = None  # snapshot/EMA of LoRA params
        self._lora_param_map: Optional[dict] = None  # cached trainable params
        self._rollout_sigmas: Optional[list] = None  # explicit grid, or None=uniform
        self._step = 0
        self._metrics: dict = {}

    def owns_training_step(self, args) -> bool:
        return bool(getattr(args, "use_byg", False))

    def on_network_built(self, ctx: SetupCtx) -> None:
        args = ctx.args
        if int(getattr(args, "blocks_to_swap", 0) or 0) != 0:
            raise ValueError(
                "BYG runs many DiT forwards per step; the block-swap offloader "
                "desyncs on a second forward (project_blockswap_extra_forwards_"
                "gradcache). Set blocks_to_swap=0 for BYG."
            )
        self._args = args
        self._network = ctx.network
        self._dit = ctx.unet
        self._cond = BYGConditioning(ctx.unet)
        self._cond.apply()
        self._rollout_sigmas = _resolve_rollout_sigmas(args)
        if self._rollout_sigmas is not None:
            logger.info(
                "BYG rollout: explicit %d-step sigma grid %s (non-uniform; t sampled "
                "from interior nodes)",
                len(self._rollout_sigmas) - 1,
                [round(s, 3) for s in self._rollout_sigmas],
            )
        # Shadow is populated lazily on the first _update_shadow (when the LoRA
        # params are guaranteed on-device) — cloning here would capture CPU
        # tensors before accelerate's device move and corrupt the rollout swap.
        self._shadow = {}
        logger.info(
            "BYG adapter ready (snapshot bootstrap, prior=%s)",
            "symmetric"
            if bool(getattr(args, "byg_prior_symmetric", False))
            else "fwd-only",
        )

    def _lora_params(self):
        """Trainable LoRA tensors keyed by name (frozen DiT excluded).

        The trainable set is fixed after the network is built, so it is cached
        on first use (lazily — after accelerate's device move, the same Parameter
        objects are reused, only their ``.data`` is migrated) instead of
        re-scanning the whole network on every snapshot / shadow-swap call.
        """
        if self._lora_param_map is None:
            self._lora_param_map = {
                n: p for n, p in self._network.named_parameters() if p.requires_grad
            }
        return self._lora_param_map

    def _update_shadow(self) -> None:
        decay = float(getattr(self._args, "byg_ema_decay", 0.0) or 0.0)
        snap_every = int(getattr(self._args, "byg_snapshot_every", 200) or 200)
        with torch.no_grad():
            if decay > 0.0:
                for n, p in self._lora_params().items():
                    s = self._shadow.get(n)
                    if s is None:
                        self._shadow[n] = p.detach().clone()
                    else:
                        s.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
            elif not self._shadow or (snap_every > 0 and self._step % snap_every == 0):
                # Populate on first call (params now on-device) or hard-refresh.
                for n, p in self._lora_params().items():
                    self._shadow[n] = p.detach().clone()

    class _SwapShadow:
        """Context manager: swap live LoRA .data ↔ shadow for the rollout."""

        def __init__(self, adapter):
            self.a = adapter
            self._backup = None

        def __enter__(self):
            self._backup = []
            if not self.a._shadow:
                # No snapshot yet (first full step) — roll out against the live
                # weights; the shadow is populated at the end of this step.
                return self
            for n, p in self.a._lora_params().items():
                s = self.a._shadow.get(n)
                if s is None:
                    continue
                self._backup.append((p, p.data))
                p.data = s.to(p.data.device)
            return self

        def __exit__(self, *exc):
            # Restore via the cached Parameter refs (no full network re-scan).
            for p, data in self._backup:
                p.data = data
            self._backup = None

    class _BaseMode:
        """Context manager: frozen-base T2I forward (source off, LoRA zeroed)."""

        def __init__(self, adapter):
            self.a = adapter
            self._mult = None

        def __enter__(self):
            self.a._cond.clear_source()
            self._mult = float(getattr(self.a._network, "multiplier", 1.0) or 1.0)
            self.a._network.set_multiplier(0.0)
            return self

        def __exit__(self, *exc):
            self.a._network.set_multiplier(self._mult)

    def _dit_velocity(self, x_4d, t_B, crossattn_emb, attn_mask, padding_mask):
        """One DiT velocity forward in the 4D→5D→4D boundary layout."""
        x_5d = x_4d.unsqueeze(2)
        v = self._dit(
            x_5d,
            t_B,
            crossattn_emb,
            padding_mask=padding_mask,
            crossattn_seqlens=_crossattn_seqlens(attn_mask),
        )
        return v.squeeze(2)

    def compute_loss(self, ctx: ComputeLossCtx) -> torch.Tensor:
        args = ctx.args
        self._step += 1
        device = ctx.latents.device

        x = ctx.latents  # source image latent, 4D [B,C,H,W] (shift-scaled)
        B = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]
        padding_mask = torch.zeros(B, 1, H, W, dtype=ctx.weight_dtype, device=device)

        c_emb, c_mask = self._get_cond(ctx, "instruction")
        rc_emb, rc_mask = self._get_cond(ctx, "reverse_instruction")

        eps = torch.randn_like(x)

        # t discretized to the rollout grid (exact ỹ_t capture). sigmas[j] is the σ
        # at node j (1.0..0.0); uniform path is the 1/n grid, an
        # explicit byg_rollout_sigmas grid warps the nodes (Anima-aware NFE).
        if self._rollout_sigmas is not None:
            sigmas = self._rollout_sigmas
            n = len(sigmas) - 1
        else:
            n = int(getattr(args, "byg_rollout_steps", 10) or 10)
            sigmas = [1.0 - j / n for j in range(n + 1)]
        sig_t = torch.tensor(sigmas, device=device, dtype=x.dtype)  # [n+1]
        k = torch.randint(1, n, (B,), device=device)  # interior node in [1, n-1]
        t = sig_t[k]  # σ per sample at node k, in (0,1)
        t_5 = t.view(B, 1, 1, 1)

        # Identity-only schedule (warmup + random identity steps).
        warmup = int(getattr(args, "byg_identity_warmup_steps", 200) or 0)
        id_prob = float(getattr(args, "byg_identity_prob", 0.0) or 0.0)
        identity_only = (self._step <= warmup) or (torch.rand(()).item() < id_prob)

        lam_prior = float(getattr(args, "byg_lambda_prior", 1.0) or 0.0)
        lam_id = float(getattr(args, "byg_lambda_id", 0.2) or 0.0)
        lam_cycle = float(getattr(args, "byg_lambda_cycle", 1.0) or 0.0)
        alpha = float(getattr(args, "byg_alpha", 0.1) or 0.0)

        # Source = x is patch-embedded once and reused across the identity,
        # bootstrap, and forward passes — the frozen embedder makes the encoding
        # invariant to the shadow swap / LoRA state. (The cycle pass encodes the
        # differentiable ŷ_hyb separately.)
        with ctx.accelerator.autocast():
            src_x_cached, src_rope_cached = self._cond.encode_source(x)

        # Identity loss (independent graph → staged backward):
        # x_t = (1-t)x + t·eps ; G(x_t, t, c̄, source=x) must predict (eps - x).
        x_t = (1.0 - t_5) * x + t_5 * eps
        target_src = eps - x
        with ctx.accelerator.autocast():
            self._cond.set_source_precomputed(src_x_cached, src_rope_cached)
            v_id = self._dit_velocity(x_t, t, rc_emb, rc_mask, padding_mask)
            l_id = (v_id.float() - target_src.float()).pow(2).mean()
        self._metrics["byg/L_id"] = float(l_id.detach())

        if identity_only:
            # No coupled graph this step — let the training loop backward the
            # identity loss (warmup / random-identity regularizer steps).
            self._update_shadow()
            return lam_id * l_id

        # Full step: stage the identity backward on its independent graph NOW so
        # it frees before the coupled forward builds (lower peak VRAM).
        if lam_id > 0.0:
            ctx.accelerator.backward(lam_id * l_id)

        # Bootstrap rollout (no_grad, snapshot weights, source=x).
        with torch.no_grad(), self._SwapShadow(self), ctx.accelerator.autocast():
            self._cond.set_source_precomputed(src_x_cached, src_rope_cached)
            y = eps.clone()
            y_t = torch.empty_like(eps)
            captured = torch.zeros(B, dtype=torch.bool, device=device)
            for j in range(n + 1):
                sel = k == j
                if sel.any():
                    y_t[sel] = y[sel]
                    captured |= sel
                if j < n:
                    s = sigmas[j]
                    dsigma = sigmas[j] - sigmas[j + 1]  # >0; =1/n on the uniform grid
                    s_B = torch.full((B,), s, device=device, dtype=x.dtype)
                    v = self._dit_velocity(y, s_B, c_emb, c_mask, padding_mask)
                    y = y - dsigma * v
            if not bool(captured.all()):
                raise RuntimeError("BYG rollout failed to capture ỹ_t for all samples")
            y0 = y  # clean multi-step estimate
        y_t = y_t.detach()
        y0 = y0.detach()

        with ctx.accelerator.autocast():
            # Forward pass (grad, source=x, instr=c).
            self._cond.set_source_precomputed(src_x_cached, src_rope_cached)
            v_fwd = self._dit_velocity(y_t, t, c_emb, c_mask, padding_mask)
            y_hat = y_t - t_5 * v_fwd  # one-step clean-edit prediction

            # Prior loss (no_grad base, multiplier=0, no source).
            l_prior = x.new_zeros(())
            if lam_prior > 0.0:
                sp_emb, sp_mask = self._get_cond(ctx, "src_caption")
                tp_emb, tp_mask = self._get_cond(ctx, "tgt_caption")
                with torch.no_grad(), self._BaseMode(self):
                    v_src = self._dit_velocity(y_t, t, sp_emb, sp_mask, padding_mask)
                    v_tgt = self._dit_velocity(y_t, t, tp_emb, tp_mask, padding_mask)
                l_prior_fwd = directional_prior_loss(v_fwd, v_src, v_tgt, alpha)
                l_prior = l_prior_fwd
                self._metrics["byg/L_prior_fwd"] = float(l_prior_fwd.detach())

            # Cycle loss (grad, source=ŷ_hyb, instr=c̄).
            l_cycle = x.new_zeros(())
            if lam_cycle > 0.0:
                y_hyb = ste_clean_blend(y0.unsqueeze(2), y_hat.unsqueeze(2)).squeeze(2)
                self._cond.set_source(y_hyb)
                v_rev = self._dit_velocity(x_t, t, rc_emb, rc_mask, padding_mask)
                l_cycle = (v_rev.float() - target_src.float()).pow(2).mean()
                self._metrics["byg/L_cycle"] = float(l_cycle.detach())

                if getattr(args, "byg_prior_symmetric", False) and lam_prior > 0.0:
                    # Reverse prior (paper Eq. 5): anchor the reverse-instruction
                    # edit velocity v_rev (computed at x_t above) to the frozen
                    # base's edited→source direction — i.e. the DDS prior with the
                    # captions swapped (p_tgt is the reverse edit's "source",
                    # p_src its "target"). sp_*/tp_* were already fetched in the
                    # forward-prior block above (symmetric ⇒ lam_prior > 0).
                    with torch.no_grad(), self._BaseMode(self):
                        v_rsrc = self._dit_velocity(
                            x_t, t, tp_emb, tp_mask, padding_mask
                        )
                        v_rtgt = self._dit_velocity(
                            x_t, t, sp_emb, sp_mask, padding_mask
                        )
                    l_prior_rev = directional_prior_loss(v_rev, v_rsrc, v_rtgt, alpha)
                    l_prior = l_prior + l_prior_rev
                    self._metrics["byg/L_prior_rev"] = float(l_prior_rev.detach())

        self._cond.clear_source()
        self._update_shadow()

        if lam_prior > 0.0:
            self._metrics["byg/L_prior"] = float(
                l_prior.detach()
            )  # fwd (+ rev if symmetric)

        loss = lam_cycle * l_cycle + lam_prior * l_prior
        return loss

    def _get_cond(self, ctx: ComputeLossCtx, role: str):
        """Return (crossattn_emb [B,S,D], attn_mask [B,S]) for a role."""
        emb = ctx.batch.get(f"byg_{role}_emb")
        if emb is None:
            raise KeyError(
                f"BYG batch missing 'byg_{role}_emb'; run build_edit_tuples + "
                "the BYG text-embedding cache pass first."
            )
        mask = ctx.batch.get(f"byg_{role}_mask")
        emb = emb.to(ctx.latents.device, dtype=ctx.weight_dtype)
        if mask is not None:
            mask = mask.to(ctx.latents.device)
        return emb, mask

    def metrics(self, ctx) -> dict:
        return dict(self._metrics)
