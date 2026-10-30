"""Triton attention with int8 QKᵀ, for the phase 5 gates (reports.md).

The forward is below; the 5b backward is under ``# --- backward``.

One kernel, FA2's loop: a program owns a ``BLOCK_M`` slice of Q for one (batch, head),
walks K/V in ``BLOCK_N`` blocks with an online softmax in the log2 domain, and writes
O and the log2 logsumexp (the backward's input). Two compile-time switches:

- ``qk``: ``bf16``, ``fp16acc`` (fp16 inputs and accumulation, no quantization) or
  ``int8``. int8 quantizes Q in the kernel (``sm_scale·log2 e``
  folded into its scale) and reads a K copy made by :func:`quant_k`: K minus its token
  mean (per batch, head). Subtracting a vector from every key adds a per-row constant
  to QKᵀ, which the softmax ignores. ``gran`` sets the scale granularity of both:
  ``block`` (one per Q block, one per 64 keys) or ``token`` (one per query row and per
  key, applied as an outer product on the int32 tile). ``rot`` multiplies Q and K by
  the same ±1 Hadamard matrix along D before rounding (``QHᵀ·HKᵀ = 128·QKᵀ``, the
  128 folded into the scale), which spreads outlier channels over the row.
- ``pv``: ``bf16`` (fp32 accumulate), ``fp16`` (fp32 accumulate), ``fp16acc`` (fp16
  accumulate inside a block, Sage v1), or ``int8``: P rounded to 1/127 steps, V
  quantized per channel and stored transposed by :func:`quant_vt` so the PV
  reduction axis is contiguous. O is normalised by the sum of the rounded P, so it
  matches what was multiplied; the stored logsumexp uses the exact sum.

Inputs are FA2's ``[B, L, H, D]`` layout; outputs ``o [B, L, H, D]`` bf16 and
``lse [B, H, L]`` fp32 (log2 domain).
"""

from __future__ import annotations

import functools
import os

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
KBLK = 64  # tokens per K scale; any BLOCK_N gathers its columns' scales
PV_MODES = {"bf16": 0, "fp16": 1, "fp16acc": 2, "int8": 3}


@functools.cache
def hadamard(d: int, device: torch.device) -> torch.Tensor:
    """Unnormalised ±1 Sylvester Hadamard ``[d, d]`` bf16 (exact), ``HᵀH = d·I``."""
    h = torch.ones(1, 1)
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device, torch.bfloat16)


@triton.jit
def _quant_k_kernel(
    K,
    Mean,
    Hm,
    KQ,
    KS,
    skb,
    skl,
    skh,
    H,
    L,
    LP,
    D: tl.constexpr,
    BLK: tl.constexpr,
    TOKEN: tl.constexpr,
    ROT: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    valid = rows[:, None] < L
    x = tl.load(
        K + b * skb + h * skh + rows[:, None] * skl + offs_d[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    x = tl.where(valid, x - tl.load(Mean + bh * D + offs_d)[None, :], 0.0)
    if ROT:
        hm = tl.load(Hm + offs_d[:, None] * D + offs_d[None, :])
        x = tl.dot(x.to(tl.bfloat16), hm)
    if TOKEN:
        s = tl.maximum(tl.max(tl.abs(x), 1), 1e-8) / 127.0
        x = x / s[:, None]
        tl.store(KS + bh * LP + rows, s)
    else:
        s = tl.maximum(tl.max(tl.abs(x)), 1e-8) / 127.0
        x = x / s
        tl.store(KS + bh * (LP // BLK) + pid, s)
    q = (x + tl.where(x >= 0, 0.5, -0.5)).to(tl.int8)
    tl.store(KQ + bh * LP * D + rows[:, None] * D + offs_d[None, :], q)


@triton.jit
def _quant_vt_kernel(
    V,
    VS,
    VQ,
    svb,
    svl,
    svh,
    H,
    L,
    LP,
    D: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    valid = rows[:, None] < L
    x = tl.load(
        V + b * svb + h * svh + rows[:, None] * svl + offs_d[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    x = x / tl.load(VS + bh * D + offs_d)[None, :]
    q = (x + tl.where(x >= 0, 0.5, -0.5)).to(tl.int8)
    tl.store(VQ + bh * D * LP + offs_d[None, :] * LP + rows[:, None], q)


def _pad(n: int, blk: int = 128) -> int:
    return (n + blk - 1) // blk * blk


def k_center(k: torch.Tensor, center: str = "mean") -> torch.Tensor | None:
    """The vector removed from every key, ``[B, H, D]`` fp32 (``None`` for ``none``)."""
    if center == "last":  # max-padded context: the repeated pad key becomes exactly 0
        return k[:, -1].float().contiguous()
    if center == "mean":
        return k.mean(1, dtype=torch.float32).contiguous()
    return None


def quant_k(
    k: torch.Tensor, gran: str = "block", rot: bool = False, center: str = "mean"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``k [B, L, H, D]`` → int8 ``[B, H, Lp, D]`` (centered) and fp32 scales,
    ``[B·H, Lp/64]`` (``block``) or ``[B·H, Lp]`` (``token``)."""
    return quant_tok(k, k_center(k, center), gran, rot)


def quant_tok(
    x: torch.Tensor, mean: torch.Tensor | None, gran: str = "token", rot: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`quant_k` with an explicit center; ``mean=None`` quantizes ``x`` as is
    (Q, and V and dO for an int8 dO·Vᵀ)."""
    B, L, H, D = x.shape
    lp = _pad(L)
    if mean is None:
        mean = torch.zeros((B, H, D), dtype=torch.float32, device=x.device)
    kq = torch.empty((B, H, lp, D), dtype=torch.int8, device=x.device)
    n_s = lp if gran == "token" else lp // KBLK
    ks = torch.empty((B * H, n_s), dtype=torch.float32, device=x.device)
    hm = hadamard(D, x.device)
    _quant_k_kernel[(lp // KBLK, B * H)](
        x,
        mean,
        hm,
        kq,
        ks,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        H,
        L,
        lp,
        D=D,
        BLK=KBLK,
        TOKEN=gran == "token",
        ROT=rot,
    )
    return kq, ks


def quant_vt(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``v [B, L, H, D]`` → int8 ``[B, H, D, Lp]``, per-channel scales ``[B·H, D]``."""
    B, L, H, D = v.shape
    lp = _pad(L)
    vs = (v.abs().amax(1).float().clamp_min(1e-8) / 127.0).contiguous()  # [B, H, D]
    vq = torch.zeros((B, H, D, lp), dtype=torch.int8, device=v.device)
    _quant_vt_kernel[(lp // KBLK, B * H)](
        v,
        vs,
        vq,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        H,
        L,
        lp,
        D=D,
        BLK=KBLK,
    )
    return vq, vs


_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
    for bm in (64, 128)
    for bn in (32, 64, 128)
    for w in (4, 8)
    for s in (2, 3)
    if not (bm == 64 and w == 8)
]
if os.environ.get("INT8_ATTN_FIXED"):  # tests: one config, no autotune sweep
    _CONFIGS = [
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2)
    ]


@triton.autotune(
    configs=_CONFIGS,
    key=["LQ", "LK", "QK_INT8", "QK_F16ACC", "PV_MODE", "TOKEN", "ROT", "Q_PRE"],
)
@triton.jit
def _fwd_kernel(
    Q,
    QS,
    K,
    KS,
    V,
    VS,
    Hm,
    Out,
    LSE,
    sqb,
    sql,
    sqh,
    skb,
    skh,
    skl,
    svb,
    svh,
    svl,
    svd,
    sob,
    sol,
    soh,
    H,
    LQ,
    LK,
    NKS,
    NQS,
    qk_scale,
    D: tl.constexpr,
    QK_INT8: tl.constexpr,
    QK_F16ACC: tl.constexpr,
    PV_MODE: tl.constexpr,
    KBLK: tl.constexpr,
    TOKEN: tl.constexpr,
    ROT: tl.constexpr,
    Q_PRE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    q = tl.load(
        Q + b * sqb + h * sqh + offs_m[:, None] * sql + offs_d[None, :],
        mask=offs_m[:, None] < LQ,
        other=0.0,
    )
    if Q_PRE:  # int8 Q copy from quant_tok (token + rot), kept for the backward
        q_s = (tl.load(QS + bh * NQS + offs_m, mask=offs_m < LQ, other=0.0) * qk_scale)[
            :, None
        ]
    elif QK_INT8:
        if ROT:
            qf = tl.dot(q, tl.load(Hm + offs_d[:, None] * D + offs_d[None, :]))
        else:
            qf = q.to(tl.float32)
        if TOKEN:
            q_s = tl.maximum(tl.max(tl.abs(qf), 1), 1e-8) / 127.0
            qf = qf / q_s[:, None]
            q_s = (q_s * qk_scale)[:, None]
        else:
            q_s = tl.maximum(tl.max(tl.abs(qf)), 1e-8) / 127.0
            qf = qf / q_s
            q_s = q_s * qk_scale
        q = (qf + tl.where(qf >= 0, 0.5, -0.5)).to(tl.int8)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    lq_i = tl.zeros([BLOCK_M], tl.float32)  # sum of the rounded P (pv int8)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    k_base = K + b * skb + h * skh
    v_base = V + b * svb + h * svh

    for start_n in range(0, LK, BLOCK_N):
        cols = start_n + offs_n
        kmask = cols < LK
        k = tl.load(
            k_base + cols[None, :] * skl + offs_d[:, None], mask=kmask[None, :], other=0
        )
        if QK_INT8:
            if TOKEN:
                ks = tl.load(KS + bh * NKS + cols, mask=kmask, other=0.0)
            else:
                ks = tl.load(KS + bh * NKS + cols // KBLK, mask=kmask, other=0.0)
            s = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32) * (q_s * ks[None, :])
        elif QK_F16ACC:
            s = (
                tl.dot(q.to(tl.float16), k.to(tl.float16), out_dtype=tl.float16).to(
                    tl.float32
                )
                * qk_scale
            )
        else:
            s = tl.dot(q, k) * qk_scale
        s = tl.where(kmask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(s - m_new[:, None])
        acc = acc * alpha[:, None]
        v = tl.load(
            v_base + cols[:, None] * svl + offs_d[None, :] * svd,
            mask=kmask[:, None],
            other=0,
        )
        if PV_MODE == 3:
            pq = (p * 127.0 + 0.5).to(tl.int8)
            l_i = l_i * alpha + tl.sum(p, 1)
            lq_i = lq_i * alpha + tl.sum(pq.to(tl.float32), 1) * (1.0 / 127.0)
            acc += tl.dot(pq, v, out_dtype=tl.int32).to(tl.float32)
        elif PV_MODE == 2:
            l_i = l_i * alpha + tl.sum(p, 1)
            acc += tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float16).to(
                tl.float32
            )
        elif PV_MODE == 1:
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = tl.dot(p.to(tl.float16), v.to(tl.float16), acc)
        else:
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
        m_i = m_new

    if PV_MODE == 3:
        acc = acc * (tl.load(VS + bh * D + offs_d) * (1.0 / 127.0))[None, :]
        acc = acc / lq_i[:, None]
    else:
        acc = acc / l_i[:, None]
    qmask = offs_m < LQ
    tl.store(
        Out + b * sob + h * soh + offs_m[:, None] * sol + offs_d[None, :],
        acc.to(tl.bfloat16),
        mask=qmask[:, None],
    )
    tl.store(LSE + bh * LQ + offs_m, m_i + tl.math.log2(l_i), mask=qmask)


def attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk: str = "int8",
    pv: str = "fp16",
    sm_scale: float | None = None,
    gran: str = "block",
    rot: bool = False,
    center: str = "mean",
    pre: tuple | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``q [B, Lq, H, D]``, ``k, v [B, Lk, H, D]`` bf16 → ``(o, lse)``. ``pre``:
    ``(qr, qs, kr, ks)`` int8 copies from :func:`quant_tok` (token + rot), see
    :func:`attn_fwd_train`."""
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    B, LQ, H, D = q.shape
    LK = k.shape[1]
    scale = (sm_scale if sm_scale is not None else D**-0.5) * LOG2E
    if rot:
        scale /= D
    hm = hadamard(D, q.device)
    qq, qs, q_strides = q, q.new_empty(1, dtype=torch.float32), q.stride()[:3]
    if pre is not None:
        assert qk == "int8" and gran == "token" and rot
        qq, qs, kk, ks = pre
        q_strides = qq.stride(0), qq.stride(2), qq.stride(1)
        k_strides = kk.stride(0), kk.stride(1), kk.stride(2)
    elif qk == "int8":
        kk, ks = quant_k(k, gran, rot, center)
        k_strides = kk.stride(0), kk.stride(1), kk.stride(2)
    else:
        kk, ks = k, k.new_empty(1, dtype=torch.float32)
        k_strides = k.stride(0), k.stride(2), k.stride(1)
    if pv == "int8":
        vv, vs = quant_vt(v)
        v_strides = vv.stride(0), vv.stride(1), 1, vv.stride(2)
    else:
        vv, vs = v, v.new_empty(1, dtype=torch.float32)
        v_strides = v.stride(0), v.stride(2), v.stride(1), 1
    o = torch.empty_like(q)
    lse = torch.empty((B, H, LQ), dtype=torch.float32, device=q.device)

    def grid(meta):
        return (triton.cdiv(LQ, meta["BLOCK_M"]), B * H)

    _fwd_kernel[grid](
        qq,
        qs,
        kk,
        ks,
        vv,
        vs,
        hm,
        o,
        lse,
        *q_strides,
        *k_strides,
        *v_strides,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        H,
        LQ,
        LK,
        ks.shape[-1] if qk == "int8" else 0,
        qs.shape[-1],
        scale,
        D=D,
        QK_INT8=qk == "int8",
        QK_F16ACC=qk == "fp16acc",
        PV_MODE=PV_MODES[pv],
        KBLK=KBLK,
        TOKEN=gran == "token",
        ROT=rot,
        Q_PRE=pre is not None,
    )
    return o, lse


# --- backward (phase 5b) -------------------------------------------------------------
#
# FA2's backward split into two kernels, as in Triton's fused-attention tutorial: one
# program per K/V block accumulates dK and dV over all Q blocks, one per Q block
# accumulates dQ over all K blocks; both recompute P from the forward's int8 Q and K
# copies and stored logsumexp. dO is divided by its max per (batch, head) and held in
# fp16 (captured dO is ~1e-7, below fp16's normal range); every result is multiplied
# back at the store. Per-GEMM modes:
#
# - ``dp`` (dO·Vᵀ, reduction over D): ``fp16`` (fp32 accumulate), ``fp16acc``, or
#   ``int8`` (dO and V per token, Hadamard-rotated, copies from :func:`quant_tok`).
# - ``dv`` (Pᵀ·dO) and ``dqk`` (dSᵀ·Q, dS·K), reduction over tokens: ``fp16``
#   (``dv`` also ``fp16acc``), or ``int8``. The left operand (P or dS, in registers) is
#   rounded per row of the tile; the right one is a transposed int8 copy from
#   :func:`quant_t`, tokens contiguous, one scale per 64 tokens and channel. A loop
#   tile therefore spans at most one scale block.

BWD_MODES = {"fp16": 0, "fp16acc": 1, "int8": 2}


@triton.jit
def _quant_t_kernel(
    X,
    Mean,
    XT,
    XS,
    sxb,
    sxl,
    sxh,
    H,
    L,
    LP,
    D: tl.constexpr,
    BLK: tl.constexpr,
    CENTER: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    valid = rows[:, None] < L
    x = tl.load(
        X + b * sxb + h * sxh + rows[:, None] * sxl + offs_d[None, :],
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    if CENTER:
        x = tl.where(valid, x - tl.load(Mean + bh * D + offs_d)[None, :], 0.0)
    s = tl.maximum(tl.max(tl.abs(x), 0), 1e-30) / 127.0
    x = x / s[None, :]
    q = (x + tl.where(x >= 0, 0.5, -0.5)).to(tl.int8)
    tl.store(XT + bh * D * LP + offs_d[None, :] * LP + rows[:, None], q)
    tl.store(XS + (bh * (LP // BLK) + pid) * D + offs_d, s)


def quant_t(
    x: torch.Tensor, mean: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x [B, L, H, D]`` (minus ``mean [B, H, D]``) → int8 ``[B, H, D, Lp]`` and fp32
    scales ``[B·H, Lp/64, D]``, one per 64 tokens and channel."""
    B, L, H, D = x.shape
    lp = _pad(L)
    xt = torch.empty((B, H, D, lp), dtype=torch.int8, device=x.device)
    xs = torch.empty((B * H, lp // KBLK, D), dtype=torch.float32, device=x.device)
    _quant_t_kernel[(lp // KBLK, B * H)](
        x,
        mean if mean is not None else xs,
        xt,
        xs,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        H,
        L,
        lp,
        D=D,
        BLK=KBLK,
        CENTER=mean is not None,
    )
    return xt, xs


@triton.jit
def _bwd_pre_kernel(
    DO, Out, SDO, DOH, DELTA, sb, sl, sh, H, L, D: tl.constexpr, BLK: tl.constexpr
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    rows = pid * BLK + tl.arange(0, BLK)
    offs_d = tl.arange(0, D)
    valid = rows[:, None] < L
    off = b * sb + h * sh + rows[:, None] * sl + offs_d[None, :]
    x = (
        tl.load(DO + off, mask=valid, other=0.0).to(tl.float32) / tl.load(SDO + bh)
    ).to(tl.float16)
    tl.store(DOH + off, x, mask=valid)
    o = tl.load(Out + off, mask=valid, other=0.0).to(tl.float32)
    tl.store(DELTA + bh * L + rows, tl.sum(x.to(tl.float32) * o, 1), mask=rows < L)


_BWD_CONFIGS = [
    triton.Config({"BLOCK_O": bo, "BLOCK_L": bl}, num_warps=w, num_stages=s)
    for bo in (64, 128)
    for bl in (32, 64)
    for w in (4, 8)
    for s in (2, 3)
    if not (bo == 128 and w == 4)
]
if os.environ.get("INT8_ATTN_FIXED"):
    _BWD_CONFIGS = [
        triton.Config({"BLOCK_O": 64, "BLOCK_L": 32}, num_warps=4, num_stages=2)
    ]
_BWD_KEY = ["LQ", "LK", "QK_INT8", "DP_MODE", "DV_MODE", "DQK_MODE"]


@triton.jit
def _round_rows(x):
    """int8 copy of a tile rounded per row, and the row scales."""
    s = tl.maximum(tl.max(tl.abs(x), 1), 1e-30) / 127.0
    x = x / s[:, None]
    return (x + tl.where(x >= 0, 0.5, -0.5)).to(tl.int8), s


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY)
@triton.jit
def _bwd_dkdv_kernel(
    Q,
    K,
    V,
    DOH,
    QR,
    QS,
    KR,
    KS,
    VR,
    VRS,
    DOR,
    DORS,
    QT,
    QTS,
    DOT,
    DOTS,
    LSE,
    DELTA,
    SDO,
    DK,
    DV,
    sqb,
    sql,
    sqh,
    skb,
    skl,
    skh,
    svb,
    svl,
    svh,
    sdb,
    sdl,
    sdh,
    sob,
    sol,
    soh,
    H,
    LQ,
    LK,
    LQP,
    LKP,
    qk_scale,
    sm_scale,
    D: tl.constexpr,
    QK_INT8: tl.constexpr,
    DP_MODE: tl.constexpr,
    DV_MODE: tl.constexpr,
    DQK_MODE: tl.constexpr,
    KBLK: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_n = pid_n * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_l = tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    nmask = offs_n < LK

    if QK_INT8:
        kr = tl.load(KR + bh * LKP * D + offs_n[:, None] * D + offs_d[None, :])
        ksn = tl.load(KS + bh * LKP + offs_n)
    else:
        kr = tl.load(
            K + b * skb + h * skh + offs_n[:, None] * skl + offs_d[None, :],
            mask=nmask[:, None],
            other=0.0,
        )
    if DP_MODE == 2:
        vr = tl.load(VR + bh * LKP * D + offs_n[:, None] * D + offs_d[None, :])
        vsn = tl.load(VRS + bh * LKP + offs_n)
    else:
        vr = tl.load(
            V + b * svb + h * svh + offs_n[:, None] * svl + offs_d[None, :],
            mask=nmask[:, None],
            other=0.0,
        ).to(tl.float16)

    dv = tl.zeros([BLOCK_O, D], tl.float32)
    dk = tl.zeros([BLOCK_O, D], tl.float32)
    nqb = LQP // KBLK
    for start_m in range(0, LQ, BLOCK_L):
        offs_m = start_m + offs_l
        mmask = offs_m < LQ
        if not QK_INT8 or DQK_MODE != 2:
            qf = tl.load(
                Q + b * sqb + h * sqh + offs_m[:, None] * sql + offs_d[None, :],
                mask=mmask[:, None],
                other=0.0,
            )
        if DP_MODE != 2 or DV_MODE != 2:
            doh = tl.load(
                DOH + b * sdb + h * sdh + offs_m[:, None] * sdl + offs_d[None, :],
                mask=mmask[:, None],
                other=0.0,
            )
        # Sᵀ and Pᵀ, [BLOCK_O keys, BLOCK_L queries]
        if QK_INT8:
            qr_t = tl.load(QR + bh * LQP * D + offs_m[None, :] * D + offs_d[:, None])
            qsm = tl.load(QS + bh * LQP + offs_m)
            s_t = tl.dot(kr, qr_t, out_dtype=tl.int32).to(tl.float32) * (
                ksn[:, None] * (qsm * qk_scale)[None, :]
            )
        else:
            s_t = tl.dot(kr, tl.trans(qf)) * qk_scale
        lse_m = tl.load(LSE + bh * LQ + offs_m, mask=mmask, other=0.0)
        p_t = tl.where(mmask[None, :], tl.math.exp2(s_t - lse_m[None, :]), 0.0)
        # dPᵀ = V·dOᵀ
        if DP_MODE == 2:
            dor_t = tl.load(DOR + bh * LQP * D + offs_m[None, :] * D + offs_d[:, None])
            dosm = tl.load(DORS + bh * LQP + offs_m)
            dp_t = tl.dot(vr, dor_t, out_dtype=tl.int32).to(tl.float32) * (
                vsn[:, None] * (dosm * (1.0 / D))[None, :]
            )
        else:
            doh_t = tl.trans(doh)
            if DP_MODE == 1:
                dp_t = tl.dot(vr, doh_t, out_dtype=tl.float16).to(tl.float32)
            else:
                dp_t = tl.dot(vr, doh_t)
        delta_m = tl.load(DELTA + bh * LQ + offs_m, mask=mmask, other=0.0)
        ds_t = p_t * (dp_t - delta_m[None, :])
        sblk = (bh * nqb + start_m // KBLK) * D + offs_d
        # dV += Pᵀ·dO
        if DV_MODE == 2:
            pq, ps = _round_rows(p_t)
            do_q = tl.load(DOT + bh * D * LQP + offs_d[None, :] * LQP + offs_m[:, None])
            dv += tl.dot(pq, do_q, out_dtype=tl.int32).to(tl.float32) * (
                ps[:, None] * tl.load(DOTS + sblk)[None, :]
            )
        else:
            if DV_MODE == 1:
                dv += tl.dot(p_t.to(tl.float16), doh, out_dtype=tl.float16).to(
                    tl.float32
                )
            else:
                dv = tl.dot(p_t.to(tl.float16), doh, dv)
        # dK += dSᵀ·Q
        if DQK_MODE == 2:
            dsq, dss = _round_rows(ds_t)
            q_q = tl.load(QT + bh * D * LQP + offs_d[None, :] * LQP + offs_m[:, None])
            dk += tl.dot(dsq, q_q, out_dtype=tl.int32).to(tl.float32) * (
                dss[:, None] * tl.load(QTS + sblk)[None, :]
            )
        else:
            dk = tl.dot(ds_t.to(tl.float16), qf.to(tl.float16), dk)

    sdo = tl.load(SDO + bh)
    out = b * sob + h * soh + offs_n[:, None] * sol + offs_d[None, :]
    tl.store(DV + out, (dv * sdo).to(tl.bfloat16), mask=nmask[:, None])
    tl.store(DK + out, (dk * (sdo * sm_scale)).to(tl.bfloat16), mask=nmask[:, None])


@triton.autotune(configs=_BWD_CONFIGS, key=_BWD_KEY)
@triton.jit
def _bwd_dq_kernel(
    Q,
    K,
    V,
    DOH,
    QR,
    QS,
    KR,
    KS,
    VR,
    VRS,
    DOR,
    DORS,
    KT,
    KTS,
    LSE,
    DELTA,
    SDO,
    DQ,
    sqb,
    sql,
    sqh,
    skb,
    skl,
    skh,
    svb,
    svl,
    svh,
    sdb,
    sdl,
    sdh,
    sob,
    sol,
    soh,
    H,
    LQ,
    LK,
    LQP,
    LKP,
    qk_scale,
    sm_scale,
    D: tl.constexpr,
    QK_INT8: tl.constexpr,
    DP_MODE: tl.constexpr,
    DV_MODE: tl.constexpr,
    DQK_MODE: tl.constexpr,
    KBLK: tl.constexpr,
    BLOCK_O: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = pid_m * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_l = tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)
    mmask = offs_m < LQ

    if QK_INT8:
        qr = tl.load(QR + bh * LQP * D + offs_m[:, None] * D + offs_d[None, :])
        qsm = tl.load(QS + bh * LQP + offs_m) * qk_scale
    else:
        qr = tl.load(
            Q + b * sqb + h * sqh + offs_m[:, None] * sql + offs_d[None, :],
            mask=mmask[:, None],
            other=0.0,
        )
    if DP_MODE == 2:
        dor = tl.load(DOR + bh * LQP * D + offs_m[:, None] * D + offs_d[None, :])
        dosm = tl.load(DORS + bh * LQP + offs_m) * (1.0 / D)
    else:
        dor = tl.load(
            DOH + b * sdb + h * sdh + offs_m[:, None] * sdl + offs_d[None, :],
            mask=mmask[:, None],
            other=0.0,
        )
    lse_m = tl.load(LSE + bh * LQ + offs_m, mask=mmask, other=0.0)
    delta_m = tl.load(DELTA + bh * LQ + offs_m, mask=mmask, other=0.0)

    dq = tl.zeros([BLOCK_O, D], tl.float32)
    nkb = LKP // KBLK
    for start_n in range(0, LK, BLOCK_L):
        offs_n = start_n + offs_l
        nmask = offs_n < LK
        if not QK_INT8 or DQK_MODE != 2:
            kf = tl.load(
                K + b * skb + h * skh + offs_n[:, None] * skl + offs_d[None, :],
                mask=nmask[:, None],
                other=0.0,
            )
        if QK_INT8:
            kr_t = tl.load(KR + bh * LKP * D + offs_n[None, :] * D + offs_d[:, None])
            ksn = tl.load(KS + bh * LKP + offs_n)
            s = tl.dot(qr, kr_t, out_dtype=tl.int32).to(tl.float32) * (
                qsm[:, None] * ksn[None, :]
            )
        else:
            s = tl.dot(qr, tl.trans(kf)) * qk_scale
        p = tl.where(nmask[None, :], tl.math.exp2(s - lse_m[:, None]), 0.0)
        if DP_MODE == 2:
            vr_t = tl.load(VR + bh * LKP * D + offs_n[None, :] * D + offs_d[:, None])
            vsn = tl.load(VRS + bh * LKP + offs_n)
            dp = tl.dot(dor, vr_t, out_dtype=tl.int32).to(tl.float32) * (
                dosm[:, None] * vsn[None, :]
            )
        else:
            v_t = tl.load(
                V + b * svb + h * svh + offs_n[None, :] * svl + offs_d[:, None],
                mask=nmask[None, :],
                other=0.0,
            ).to(tl.float16)
            if DP_MODE == 1:
                dp = tl.dot(dor, v_t, out_dtype=tl.float16).to(tl.float32)
            else:
                dp = tl.dot(dor, v_t)
        ds = p * (dp - delta_m[:, None])
        # dQ += dS·K
        if DQK_MODE == 2:
            dsq, dss = _round_rows(ds)
            k_q = tl.load(KT + bh * D * LKP + offs_d[None, :] * LKP + offs_n[:, None])
            dq += tl.dot(dsq, k_q, out_dtype=tl.int32).to(tl.float32) * (
                dss[:, None]
                * tl.load(KTS + (bh * nkb + start_n // KBLK) * D + offs_d)[None, :]
            )
        else:
            dq = tl.dot(ds.to(tl.float16), kf.to(tl.float16), dq)

    sdo = tl.load(SDO + bh)
    tl.store(
        DQ + b * sob + h * soh + offs_m[:, None] * sol + offs_d[None, :],
        (dq * (sdo * sm_scale)).to(tl.bfloat16),
        mask=mmask[:, None],
    )


def attn_fwd_train(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float | None = None
) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """The 5a forward (token + rot int8 QKᵀ, fp16acc PV) with Q quantized by
    :func:`quant_tok` instead of in the kernel, so the backward recomputes P from the
    same int8 copies. Returns ``(o, lse, saved)``; ``saved`` goes to :func:`attn_bwd`."""
    qr, qs = quant_tok(q, None)
    mean = k_center(k)
    kr, ks = quant_tok(k, mean)
    o, lse = attn_fwd(
        q, k, v, "int8", "fp16acc", sm_scale, "token", True, pre=(qr, qs, kr, ks)
    )
    return o, lse, (qr, qs, kr, ks, mean)


def attn_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    saved: tuple | None = None,
    dp: str = "fp16acc",
    dv: str = "int8",
    dqk: str = "int8",
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dq, dk, dv)`` bf16 in ``[B, L, H, D]``. ``saved`` from :func:`attn_fwd_train`
    recomputes P with int8 QKᵀ; ``None`` recomputes it in bf16 (pair with the bf16
    ``attn_fwd`` arm)."""
    B, LQ, H, D = q.shape
    LK = k.shape[1]
    sm = sm_scale if sm_scale is not None else D**-0.5
    qk_int8 = saved is not None
    qk_scale = sm * LOG2E / (D if qk_int8 else 1)
    dev = q.device
    lqp, lkp = _pad(LQ), _pad(LK)

    do, o = do.contiguous(), o.contiguous()
    sdo = do.abs().amax((1, 3)).float().clamp_min(1e-30).contiguous()  # [B, H]
    doh = torch.empty_like(do, dtype=torch.float16)
    delta = torch.empty((B, H, LQ), dtype=torch.float32, device=dev)
    _bwd_pre_kernel[(triton.cdiv(LQ, KBLK), B * H)](
        do, o, sdo, doh, delta, *do.stride()[:3], H, LQ, D=D, BLK=KBLK
    )

    nil = torch.empty(1, dtype=torch.float32, device=dev)
    qr, qs, kr, ks, mean = saved if qk_int8 else (nil, nil, nil, nil, None)
    vr = vrs = dor = dors = nil
    if dp == "int8":
        vr, vrs = quant_tok(v, None)
        dor, dors = quant_tok(doh, None)
    dot_ = dots = nil
    if dv == "int8":
        dot_, dots = quant_t(doh)
    qt = qts = kt = kts = nil
    if dqk == "int8":
        qt, qts = quant_t(q)
        kt, kts = quant_t(k, mean)

    dq = torch.empty((B, LQ, H, D), dtype=torch.bfloat16, device=dev)
    dk = torch.empty((B, LK, H, D), dtype=torch.bfloat16, device=dev)
    dvv = torch.empty_like(dk)
    strides = (*q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *doh.stride()[:3])
    modes = dict(
        D=D,
        QK_INT8=qk_int8,
        DP_MODE=BWD_MODES[dp],
        DV_MODE=BWD_MODES[dv],
        DQK_MODE=BWD_MODES[dqk],
        KBLK=KBLK,
    )
    tail = (H, LQ, LK, lqp, lkp, qk_scale, sm)

    def grid_k(meta):
        return (triton.cdiv(LK, meta["BLOCK_O"]), B * H)

    def grid_q(meta):
        return (triton.cdiv(LQ, meta["BLOCK_O"]), B * H)

    _bwd_dkdv_kernel[grid_k](
        q, k, v, doh, qr, qs, kr, ks, vr, vrs, dor, dors, qt, qts, dot_, dots,
        lse, delta, sdo, dk, dvv, *strides, *dk.stride()[:3], *tail, **modes,
    )  # fmt: skip
    _bwd_dq_kernel[grid_q](
        q, k, v, doh, qr, qs, kr, ks, vr, vrs, dor, dors, kt, kts,
        lse, delta, sdo, dq, *strides, *dq.stride()[:3], *tail, **modes,
    )  # fmt: skip
    return dq, dk, dvv


def lse_natural(
    q: torch.Tensor,
    lse: torch.Tensor,
    mean: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """The forward's log2 logsumexp over centered keys → FA2's natural-log one over the
    raw keys, ``[B, H, L]``: add back ``sm_scale·q·mean`` per row."""
    sm = sm_scale if sm_scale is not None else q.shape[-1] ** -0.5
    qm = torch.einsum("blhd,bhd->bhl", q.float(), mean)
    return lse / LOG2E + sm * qm
