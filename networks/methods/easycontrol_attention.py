"""LSE-decomposed extended self-attention for EasyControl's two-stream forward.

Self-contained attention math shared by the EasyControl network
(``networks/methods/easycontrol.py``) and BYG (``networks/methods/byg.py``):
the target stream attends over the concatenation ``[target_k ; cond_k]`` with a
per-block scalar logit bias (``b_cond``) on the cond rows, without ever
materializing the ``[B, H, S_t, S_t + S_c]`` attention matrix.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class _ExtendedSelfAttnLSEFunc(torch.autograd.Function):
    """LSE-decomposed extended self-attention with a per-block scalar logit bias.

    Mathematically equivalent to::

        joint_out = softmax([Q@K_t^T·s ; Q@K_c^T·s + b]) @ [V_t; V_c]

    but never materializes the ``[B, H, S_q, S_t+S_c]`` attention matrix. Two
    memory-efficient FA2 forwards on the disjoint key tiles, then a Python
    LSE-arithmetic combine::

        α = exp(lse_t  - joint_lse)
        β = exp(lse_c+b - joint_lse)        joint_lse = logaddexp(lse_t, lse_c + b)
        joint_out = α · out_t + β · out_c

    Forward correctness (vs. masked SDPA) is identity, modulo fp32 ulp.

    Backward correctness is more subtle. FA2's stock ``FlashAttnFunc.backward``
    only consumes ``dout`` and silently discards the upstream gradient on
    ``softmax_lse``. A plain "two FA + Python combine via flash_attn_func"
    therefore drops the *path-2* gradient that flows from the loss back through
    α/β into ``q``/``k_t``/``k_c`` (the contribution scales as α·β·(out_c−out_t)
    in dout-space; negligible at init when β≈4.5e-5 from b_cond=-10, but grows
    as b_cond rises during training).

    To recover the joint-softmax gradient exactly, this Function bypasses the
    stock autograd and calls ``_wrapped_flash_attn_forward / _backward``
    directly. The trick: feeding ``softmax_lse = joint_lse`` (target) and
    ``softmax_lse = joint_lse - b`` (cond) into the per-tile FA backward causes
    FA to compute joint-softmax probabilities ``exp(L_t·s)/Z`` and
    ``exp(L_c·s + b)/Z`` respectively, so per-tile contributions sum to the
    correct joint gradient on q/k/v. ``b_cond``'s gradient is computed
    analytically from α, β, out_t, out_c, dout.
    """

    @staticmethod
    def forward(ctx, q, k_t, v_t, k_c, v_c, b_cond, softmax_scale):
        from networks import attention_dispatch as anima_attention

        if anima_attention._wrapped_flash_attn_forward is None:
            raise RuntimeError(
                "_ExtendedSelfAttnLSEFunc requires flash-attn to be installed"
            )
        fa_fwd = anima_attention._wrapped_flash_attn_forward

        out_t, lse_t, _, rng_state_t = fa_fwd(
            q,
            k_t,
            v_t,
            0.0,
            softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
        out_c, lse_c, _, rng_state_c = fa_fwd(
            q,
            k_c,
            v_c,
            0.0,
            softmax_scale,
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )

        # LSE arithmetic combine. (FA returns lse in fp32 regardless of
        # input dtype, so b_cond — also fp32 — adds without promotion.)
        b_fp32 = b_cond.to(lse_c.dtype)
        lse_c_adj = lse_c + b_fp32
        joint_lse = torch.logaddexp(lse_t, lse_c_adj)
        alpha = (lse_t - joint_lse).exp()  # [B, H, S_q] fp32
        beta = (lse_c_adj - joint_lse).exp()  # [B, H, S_q] fp32

        # out_t/out_c are [B, S_q, H, D] (BLHD); broadcast α/β over D.
        alpha_bd = alpha.transpose(1, 2).unsqueeze(-1).to(out_t.dtype)
        beta_bd = beta.transpose(1, 2).unsqueeze(-1).to(out_c.dtype)
        joint_out = alpha_bd * out_t + beta_bd * out_c

        ctx.save_for_backward(
            q,
            k_t,
            v_t,
            k_c,
            v_c,
            joint_out,
            joint_lse,
            alpha,
            beta,
            out_t,
            out_c,
            b_fp32,
            rng_state_t,
            rng_state_c,
        )
        ctx.softmax_scale = softmax_scale
        ctx.b_cond_orig_dtype = b_cond.dtype
        return joint_out

    @staticmethod
    def backward(ctx, dout):
        from networks import attention_dispatch as anima_attention

        fa_bwd = anima_attention._wrapped_flash_attn_backward
        (
            q,
            k_t,
            v_t,
            k_c,
            v_c,
            joint_out,
            joint_lse,
            alpha,
            beta,
            out_t,
            out_c,
            b_fp32,
            rng_state_t,
            rng_state_c,
        ) = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale

        dout = dout.contiguous()

        # Tile 1 (target) — feed JOINT lse and JOINT out so that FA computes
        # per-key softmax mass = exp(L_t·s - joint_lse) = exp(L_t·s) / Z, which
        # is the joint-softmax probability on target keys; and uses joint_out
        # as the "softmax output" reference (V_t - joint_out is the correct
        # second term).
        dq_t = torch.empty_like(q)
        dk_t = torch.empty_like(k_t)
        dv_t = torch.empty_like(v_t)
        fa_bwd(
            dout,
            q,
            k_t,
            v_t,
            joint_out,
            joint_lse,
            dq_t,
            dk_t,
            dv_t,
            0.0,
            softmax_scale,
            False,
            -1,
            -1,
            0.0,
            None,
            False,
            rng_state=rng_state_t,
        )

        # Tile 2 (cond) — feed (joint_lse - b) so per-key mass becomes
        # exp(L_c·s - (joint_lse - b)) = exp(L_c·s + b) / Z, the joint-softmax
        # probability on cond keys (with the bias).
        effective_lse_c = joint_lse - b_fp32
        dq_c = torch.empty_like(q)
        dk_c = torch.empty_like(k_c)
        dv_c = torch.empty_like(v_c)
        fa_bwd(
            dout,
            q,
            k_c,
            v_c,
            joint_out,
            effective_lse_c,
            dq_c,
            dk_c,
            dv_c,
            0.0,
            softmax_scale,
            False,
            -1,
            -1,
            0.0,
            None,
            False,
            rng_state=rng_state_c,
        )

        dq = dq_t + dq_c

        # b_cond gradient — analytical from the LSE arithmetic.
        #   ∂joint_out/∂b = α · β · (out_c − out_t)            [B, S_q, H, D]
        #   ∂L/∂b         = sum (α · β · ⟨out_c − out_t, dout⟩_D)
        # Reduction in fp32 for stability (α, β are fp32; bf16 inner can lose
        # ulps on long S_q reductions).
        inner_bsh = ((out_c.float() - out_t.float()) * dout.float()).sum(
            dim=-1
        )  # [B, S_q, H]
        inner_bhq = inner_bsh.transpose(1, 2)  # [B, H, S_q]
        db_scalar = (alpha * beta * inner_bhq).sum()
        db_cond = db_scalar.to(ctx.b_cond_orig_dtype)
        # Match b_cond's original 0-d shape.
        if b_fp32.dim() == 0:
            db_cond = db_cond.reshape(())

        return dq, dk_t, dv_t, dk_c, dv_c, db_cond, None


_LSE_FALLBACK_WARNED = False


def _warn_lse_fallback_once(reason: str) -> None:
    """One-shot warning when we can't use the LSE-decomposed path."""
    global _LSE_FALLBACK_WARNED
    if _LSE_FALLBACK_WARNED:
        return
    _LSE_FALLBACK_WARNED = True
    logger.warning(
        f"EasyControl: falling back to masked-SDPA path ({reason}). The math "
        f"kernel materializes a [B, H, S_t, S_t+S_c] attention matrix per "
        f"block (~1 GB / block at bf16), which can OOM on real hardware. "
        f"Install flash-attn and use attn_mode='flash' for the LSE-decomposed "
        f"path."
    )


def _extended_target_attention(
    target_q,
    target_k,
    target_v,
    cond_k,
    cond_v,
    *,
    b_param,
    scale,
    attn_params,
):
    """Run target's extended self-attention over [target_k; cond_k].

    Inputs are BSHD: target_q/k/v ``[B, S_t, H, D]``, cond_k/v ``[B, S_c, H, D]``.
    Returns ``[B, S_t, H*D]`` ready for output_proj. Uses
    ``_ExtendedSelfAttnLSEFunc`` (memory-efficient) when flash-attn + flash
    mode is available; falls back to masked-SDPA (math kernel; OOM risk) with
    a one-shot warning otherwise.
    """
    from networks import attention_dispatch as anima_attention

    # dtype matching mirrors Attention.forward's casting policy.
    if target_q.dtype != target_v.dtype:
        if (
            not attn_params.supports_fp32 or attn_params.requires_same_dtype
        ) and torch.is_autocast_enabled():
            target_q = target_q.to(target_v.dtype)
            target_k = target_k.to(target_v.dtype)
    cond_k = cond_k.to(target_k.dtype)
    cond_v = cond_v.to(target_v.dtype)

    if scale is None:
        scale = target_q.shape[-1] ** -0.5

    use_lse = (
        anima_attention._wrapped_flash_attn_forward is not None
        and attn_params.attn_mode == "flash"
    )
    if use_lse:
        out = _ExtendedSelfAttnLSEFunc.apply(
            target_q.contiguous(),
            target_k.contiguous(),
            target_v.contiguous(),
            cond_k.contiguous(),
            cond_v.contiguous(),
            b_param,
            scale,
        )
        # out: [B, S_t, H, D] → [B, S_t, H*D]
        B, S_t = out.shape[0], out.shape[1]
        return out.reshape(B, S_t, -1)

    # Fallback: masked extended SDPA. Materializes the full attention matrix
    # in the math kernel — only used when FA is unavailable.
    if attn_params.attn_mode == "flash":
        _warn_lse_fallback_once("flash-attn import failed at module load")
    else:
        _warn_lse_fallback_once(
            f"attn_mode={attn_params.attn_mode!r} unsupported by LSE path"
        )

    B, S_t = target_q.shape[0], target_q.shape[1]
    S_c = cond_k.shape[1]
    k_ext = torch.cat([target_k, cond_k], dim=1)
    v_ext = torch.cat([target_v, cond_v], dim=1)
    q_s = target_q.transpose(1, 2)
    k_s = k_ext.transpose(1, 2)
    v_s = v_ext.transpose(1, 2)
    b = b_param.to(q_s.dtype)
    target_zeros = torch.zeros(S_t, device=target_q.device, dtype=q_s.dtype)
    cond_b = b.expand(S_c)
    attn_bias = torch.cat([target_zeros, cond_b], dim=0).view(1, 1, 1, S_t + S_c)
    out = F.scaled_dot_product_attention(
        q_s, k_s, v_s, attn_mask=attn_bias, scale=scale
    )
    return out.transpose(1, 2).reshape(B, S_t, -1)
