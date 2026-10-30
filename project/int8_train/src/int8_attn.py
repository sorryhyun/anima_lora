"""Triton attention forward with int8 QKᵀ, for the phase 5 gate (proposal.md §Phase 5).

One kernel, FA2's loop: a program owns a ``BLOCK_M`` slice of Q for one (batch, head),
walks K/V in ``BLOCK_N`` blocks with an online softmax in the log2 domain, and writes
O and the log2 logsumexp (the backward's input). Two compile-time switches:

- ``qk``: ``bf16`` or ``int8``. int8 quantizes Q in the kernel (``sm_scale·log2 e``
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


def quant_k(
    k: torch.Tensor, gran: str = "block", rot: bool = False, center: str = "mean"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``k [B, L, H, D]`` → int8 ``[B, H, Lp, D]`` (mean removed) and fp32 scales,
    ``[B·H, Lp/64]`` (``block``) or ``[B·H, Lp]`` (``token``)."""
    B, L, H, D = k.shape
    lp = _pad(L)
    if center == "last":  # max-padded context: the repeated pad key becomes exactly 0
        mean = k[:, -1].float().contiguous()  # [B, H, D]
    else:
        mean = k.mean(1, dtype=torch.float32).contiguous()
    kq = torch.empty((B, H, lp, D), dtype=torch.int8, device=k.device)
    n_s = lp if gran == "token" else lp // KBLK
    ks = torch.empty((B * H, n_s), dtype=torch.float32, device=k.device)
    hm = hadamard(D, k.device)
    _quant_k_kernel[(lp // KBLK, B * H)](
        k,
        mean,
        hm,
        kq,
        ks,
        k.stride(0),
        k.stride(1),
        k.stride(2),
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
    configs=_CONFIGS, key=["LQ", "LK", "QK_INT8", "PV_MODE", "TOKEN", "ROT"]
)
@triton.jit
def _fwd_kernel(
    Q,
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
    qk_scale,
    D: tl.constexpr,
    QK_INT8: tl.constexpr,
    PV_MODE: tl.constexpr,
    KBLK: tl.constexpr,
    TOKEN: tl.constexpr,
    ROT: tl.constexpr,
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
    if QK_INT8:
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """``q [B, Lq, H, D]``, ``k, v [B, Lk, H, D]`` bf16 → ``(o, lse)``."""
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    B, LQ, H, D = q.shape
    LK = k.shape[1]
    scale = (sm_scale if sm_scale is not None else D**-0.5) * LOG2E
    if rot:
        scale /= D
    hm = hadamard(D, q.device)
    if qk == "int8":
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
        q,
        kk,
        ks,
        vv,
        vs,
        hm,
        o,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        *k_strides,
        *v_strides,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        H,
        LQ,
        LK,
        ks.shape[-1] if qk == "int8" else 0,
        scale,
        D=D,
        QK_INT8=qk == "int8",
        PV_MODE=PV_MODES[pv],
        KBLK=KBLK,
        TOKEN=gran == "token",
        ROT=rot,
    )
    return o, lse
