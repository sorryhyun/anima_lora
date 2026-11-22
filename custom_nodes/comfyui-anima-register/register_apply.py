"""Comfy-native register-token adapter for the Anima DiT (cosmos MiniTrainDIT).

Register checkpoints were trained against the DiT's **fused**
``self_attn.qkv_proj`` and its ``_run_blocks`` seam. ComfyUI's Anima runs
``comfy.ldm.cosmos.predict2.MiniTrainDIT``, which has **split**
``q_proj``/``k_proj``/``v_proj`` and an inline block loop (no ``_run_blocks``), so
this file reimplements the mechanism against the comfy backbone. No vendored
training code.

What it does, faithfully mirroring the training mechanism:

* **Split QKV surface.** The checkpoint stores the surface already split into
  q/k/v (``qkv.{bi}.q/k/v`` for unfrozen, or ``down`` + ``up_q/up_k/up_v`` for
  lora). We fold any lora into a single ``(inner, D)`` ΔW per component and add
  it onto ``q_proj``/``k_proj``/``v_proj`` — a direct match, zero fusion logic.
* **Register tokens.** ``K`` learned tokens are concatenated onto the self-attn
  sequence at block ``insert_block`` (DSR's "starting block"; 0 = stack entry —
  the default for pre-``ss_insert_block`` checkpoints), carried through the
  remaining blocks (they attend and are attended to, and get a residual each
  block), and stripped before the final
  layer. Rope-exempt: the cosmos rope is a per-position ``(head_dim//2, 2, 2)``
  stack of 2×2 rotation matrices, so register rows get the **2×2 identity**
  (``[[1,0],[0,1]]``) — no rotation. Comfy keeps the ``(B,T,H,W,D)`` patch grid,
  so we replicate the training DiT's native-flatten to ``(B,1,L+K,1,D)`` around
  the block loop (valid because Anima is image-only, ``T==1``, so adaLN
  modulation broadcasts uniformly).

Robustness: bound via ``set_model_unet_function_wrapper`` and patched onto
whatever ``diffusion_model`` is resident **around each forward call**, restored
in a ``finally`` before returning. The shared module is never left mutated, so
it survives ComfyUI's clone/reload/recompile for free (cf. the AnimaBlockCompile
rebuild), a strength change on re-queue always takes effect, workflows that skip
the node see the pristine model, and no stranded references trip ComfyUI's
"memory leak with model" detector (refcount-based — permanent bound-method
attributes on the module are self-cycles it can't free without a full gc).
The register forward runs eager.
"""

from __future__ import annotations

import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F


class RegisterComfyAdapter:
    """Loaded register adapter, applied transiently per forward via ``applied()``."""

    def __init__(self, state_dict: dict, config: dict, strength: float = 1.0):
        self.cfg = config
        self.K = int(config["num_registers"])
        self.arm = config.get("arm", "B")
        self.qkv_mode = config.get("qkv_mode", "unfrozen")
        self.target_blocks = list(config["target_blocks"])
        self.scale = float(config.get("scale", 1.0))
        self.insert_block = int(config.get("insert_block", 0))
        self.strength = float(strength)
        self._device_cache = {}  # (device, dtype) -> (register, deltas) on-device

        # Register embedding (arm B). Arm A / K=0 → no token injection.
        reg = state_dict.get("register")
        self.register = (
            reg if (reg is not None and self.K > 0 and self.arm == "B") else None
        )

        # Fold the QKV surface into one (inner, D) ΔW per component per block.
        self.deltas: dict[int, dict[str, torch.Tensor]] = {}
        for bi in self.target_blocks:
            p = f"qkv.{bi}."
            if self.qkv_mode == "lora":
                down = state_dict[f"{p}down"]  # (r, D)
                comp = {
                    c: (state_dict[f"{p}up_{c}"] @ down) * self.scale  # (inner, D)
                    for c in ("q", "k", "v")
                }
            else:  # unfrozen: already (inner, D)
                comp = {c: state_dict[f"{p}{c}"] for c in ("q", "k", "v")}
            self.deltas[bi] = comp

    # -- apply --------------------------------------------------------------
    def _materialize(self, dev, dt):
        """Strength-scaled register + ΔW tensors on the model's device/dtype."""
        key = (dev, dt)
        cached = self._device_cache.get(key)
        if cached is None:
            reg = None
            if self.register is not None:
                reg = self.register.to(device=dev, dtype=dt) * self.strength
            deltas = {
                bi: {
                    c: t.to(device=dev, dtype=dt) * self.strength
                    for c, t in comp.items()
                }
                for bi, comp in self.deltas.items()
            }
            cached = (reg, deltas)
            self._device_cache[key] = cached
        return cached

    @contextmanager
    def applied(self, dm):
        """Patch ``dm`` for one forward call, restoring it on exit.

        Installs the additive ΔW on each target block's split q/k/v projections
        and (when the checkpoint carries register tokens) the register-aware
        ``_forward``, then puts every touched attribute back — the shared
        ``diffusion_model`` leaves this context as it entered.
        """
        ref = next(dm.parameters())
        reg, deltas = self._materialize(ref.device, ref.dtype)

        # Instance-attr snapshots: (obj, name, was_instance_attr, old_value).
        saved = []

        def patch(obj, name, new):
            saved.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
            setattr(obj, name, new)

        def make_fwd(orig, d):
            def fwd(x):
                return orig(x) + F.linear(x, d)

            return fwd

        try:
            # 1) Additive ΔW on each target block's split q/k/v projections.
            for bi in self.target_blocks:
                attn = dm.blocks[bi].self_attn
                for c in ("q", "k", "v"):
                    proj = getattr(attn, f"{c}_proj")
                    patch(proj, "forward", make_fwd(proj.forward, deltas[bi][c]))

            # 2) Register-aware _forward (native-flatten + concat + strip). Bound
            #    as a MethodType so `self._forward` stays a bound method (the
            #    public `forward` → WrapperExecutor → `self._forward` seam is
            #    preserved). Skipped entirely when there are no register tokens
            #    (arm A / K=0) — the ΔW patches alone suffice.
            if reg is not None:
                patch(
                    dm,
                    "_anima_register_state",
                    {"register": reg, "K": self.K, "insert_block": self.insert_block},
                )
                patch(dm, "_forward", types.MethodType(_register_forward, dm))
            yield
        finally:
            for obj, name, had, old in reversed(saved):
                if had:
                    setattr(obj, name, old)
                else:
                    try:
                        delattr(obj, name)
                    except AttributeError:
                        pass


def _identity_rope_rows(rope_blk: torch.Tensor, k: int) -> torch.Tensor:
    """K rope rows that are the 2×2 identity, matching ``rope_blk`` layout.

    ``rope_blk`` is ``(1, L, 1, head_dim//2, 2, 2)`` — the block-level cosmos rope
    (``VideoRopePosition3DEmb`` → ``(L, D_half, 2, 2)`` then ``.unsqueeze(1).unsqueeze(0)``).
    Identity rotation = ``[[1,0],[0,1]]`` per frequency pair, so registers are
    position-exempt.
    """
    d_half = rope_blk.shape[-3]
    eye = torch.eye(2, dtype=rope_blk.dtype, device=rope_blk.device)
    return eye.view(1, 1, 1, 1, 2, 2).expand(1, k, 1, d_half, 2, 2)


def _register_forward(
    self, x, timesteps, context, fps=None, padding_mask=None, **kwargs
):
    """MiniTrainDIT._forward with K register tokens injected around the block loop.

    Copies the comfy predict2 forward verbatim except for the three register
    edits (flatten→concat→strip + rope identity). Kept close to the original so
    it's obvious what changed.
    """
    import comfy.ldm.common_dit

    st = self._anima_register_state
    K = st["K"]
    reg = st["register"]
    insert_block = st["insert_block"]

    orig_shape = list(x.shape)
    x = comfy.ldm.common_dit.pad_to_patch_size(
        x, (self.patch_temporal, self.patch_spatial, self.patch_spatial)
    )
    x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
        x, fps=fps, padding_mask=padding_mask
    )
    if extra_pos_emb is not None:
        # Anima uses extra_per_block_abs_pos_emb=False; the register-flatten path
        # doesn't carry a per-block abs pos emb. Bail loudly rather than silently
        # producing wrong images.
        raise NotImplementedError(
            "register adapter: extra_per_block_abs_pos_emb is not supported"
        )

    timesteps_B_T = timesteps
    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)
    t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](
        self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype)
    )
    t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

    B, T, H, W, D = x_B_T_H_W_D.shape
    if T != 1:
        raise NotImplementedError(f"register adapter assumes image T==1, got T={T}")
    L = T * H * W

    block_kwargs = {
        "rope_emb_L_1_1_D": rope_emb_L_1_1_D.unsqueeze(1).unsqueeze(0),
        "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
        "extra_per_block_pos_emb": None,
        "transformer_options": kwargs.get("transformer_options", {}),
    }

    if x_B_T_H_W_D.dtype == torch.float16:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    # --- register injection (native-flatten so appending K on the seq axis is
    #     clean; valid for T==1 since adaLN modulation broadcasts over H,W).
    #     Registers enter at block `insert_block` (DSR's "starting block");
    #     blocks before it run at the bare L. ---
    x_flat = x_B_T_H_W_D.reshape(B, 1, L, 1, D)
    r = rope_ext = None
    if K > 0:
        r = reg.to(dtype=x_flat.dtype).view(1, 1, K, 1, D).expand(B, -1, -1, -1, -1)
        rope = block_kwargs["rope_emb_L_1_1_D"]  # (1,L,1,D_half,2,2)
        rope_ext = torch.cat([rope, _identity_rope_rows(rope, K)], dim=1)
        if insert_block == 0:
            x_flat = torch.cat([x_flat, r], dim=2)  # (B,1,L+K,1,D)
            block_kwargs["rope_emb_L_1_1_D"] = rope_ext

    for bi, block in enumerate(self.blocks):
        if K > 0 and insert_block > 0 and bi == insert_block:
            x_flat = torch.cat([x_flat, r], dim=2)  # (B,1,L+K,1,D)
            block_kwargs["rope_emb_L_1_1_D"] = rope_ext
        x_flat = block(x_flat, t_embedding_B_T_D, context, **block_kwargs)

    x_flat = x_flat[:, :, :L, :, :]  # strip registers
    x_B_T_H_W_D = x_flat.reshape(B, T, H, W, D)

    x_B_T_H_W_O = self.final_layer(
        x_B_T_H_W_D.to(context.dtype),
        t_embedding_B_T_D,
        adaln_lora_B_T_3D=adaln_lora_B_T_3D,
    )
    x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)[
        :, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]
    ]
    return x_B_C_Tt_Hp_Wp
