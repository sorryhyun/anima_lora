"""Probe bank for the set-level attention loss — real DiT K/V, no DiT load.

Two verified facts about how the DiT consumes the adapter output fix the shape
of this loss (``library/anima/models.py``):

1. The DiT's cross-attention applies **no RoPE to the context** (rope is gated
   on ``is_selfattn`` in ``Attention.compute_qkv``), so the text side is consumed
   **permutation-invariantly** — position-wise matching asks for more than the
   model can observe.
2. Padded positions are zeroed after the adapter, never masked out
   (``crossattn_emb[~mask] = 0``, ``library/inference/text.py``) — the CLAUDE.md
   attention-sink invariant. With ``k_norm(0) = 0`` those ``S - N`` rows are
   sink mass at logit 0 carrying zero value, so the **number of real tokens is
   itself part of the conditioning**, and teacher and student do not have the
   same number of them.

So the faithful objective is: match the attention *readout* over
``{N real vectors} ∪ {S - N zeros}``. The sink term is folded into the softmax
denominator analytically rather than materializing 512-row tensors.

The K/V projections and the q/k RMSNorm gains are read straight out of the DiT
safetensors (``net.blocks.<i>.cross_attn.*``) — no model instantiation. The
**queries cannot be**: they are ``q_proj(image tokens)`` and only exist during a
forward, so they come from a pre-built pool
(``build_query_bank.py`` → ``bench/cjk_distill/assets/query_bank.safetensors``).

That pool is not optional. Seeded random directions were used once and the
readout space collapsed — a random query attends almost uniformly, so the
readout degenerates to a near-mean over the sequence, which is blind to wording.
Measured in ``_archive/cjk_aware_anima/reports/0816_phase2.md``: unrelated prompts at 0.374
*rising to 0.738 during training*, two captions of one image at exactly 1.000.
It broke ``L_attn`` as an objective as well as the metric, so ``--allow_random_queries``
exists only to reproduce that withdrawn run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

RMS_EPS = 1e-6  # matches models.py::RMSNorm(eps=1e-6) on q_norm/k_norm
DEFAULT_QUERY_BANK = (
    Path(__file__).resolve().parents[2]
    / "bench"
    / "cjk_distill"
    / "assets"
    / "query_bank.safetensors"
)


@dataclass
class BlockProbe:
    block: int
    k_weight: torch.Tensor  # [inner, ctx_dim]
    v_weight: torch.Tensor  # [inner, ctx_dim]
    k_gain: torch.Tensor  # [head_dim]
    queries: torch.Tensor  # [heads, n_query, head_dim] — already q_norm'ed
    center: torch.Tensor | None = None  # [heads, n_query, head_dim], see fit_centers


def _rms_norm(x: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + RMS_EPS) * gain


def _load_query_pool(path: Path, blocks: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """``{block: [pool, heads, head_dim]}`` from a built query bank."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        missing = [b for b in blocks if f"queries_block{b}" not in keys]
        if missing:
            raise KeyError(
                f"{path}: no queries for block(s) {missing} — the bank was built "
                f"for {sorted(keys)}. Rebuild with "
                f"`--blocks {','.join(str(b) for b in blocks)}`."
            )
        return {b: f.get_tensor(f"queries_block{b}") for b in blocks}


def build_bank(
    dit_path: str,
    blocks: tuple[int, ...],
    *,
    n_query: int = 64,
    seed: int = 0,
    device=None,
    dtype=torch.float32,
    query_bank: str | Path | None = None,
    allow_random_queries: bool = False,
) -> list[BlockProbe]:
    from safetensors import safe_open

    bank_path = Path(query_bank) if query_bank else DEFAULT_QUERY_BANK
    pool: dict[int, torch.Tensor] | None = None
    if bank_path.exists():
        pool = _load_query_pool(bank_path, blocks)
        logger.info("attn bank: real queries from %s", bank_path)
    elif not allow_random_queries:
        raise FileNotFoundError(
            f"no query bank at {bank_path}. The attn loss needs REAL cross-attn "
            "queries — random directions collapse the readout space (see "
            "_archive/cjk_aware_anima/reports/0816_phase2.md). Build one with:\n"
            "    make daemon-run ARGS='scripts/distill_cjk/build_query_bank.py "
            f"--blocks {','.join(str(b) for b in blocks)}'\n"
            "or pass --allow_random_queries to reproduce the withdrawn G2 run."
        )
    else:
        logger.warning(
            "attn bank: RANDOM probe queries — the readout space is near-blind to "
            "wording in this mode; metrics from it are not comparable to a real-bank run"
        )

    probes: list[BlockProbe] = []
    g = torch.Generator().manual_seed(seed)
    with safe_open(dit_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        prefix = "net." if any(k.startswith("net.blocks.") for k in keys) else ""
        for b in blocks:
            base = f"{prefix}blocks.{b}.cross_attn."
            missing = [
                s
                for s in (
                    "k_proj.weight",
                    "v_proj.weight",
                    "k_norm.weight",
                    "q_norm.weight",
                )
                if base + s not in keys
            ]
            if missing:
                raise KeyError(f"block {b}: missing {missing} in {dit_path}")
            kw = f.get_tensor(base + "k_proj.weight").to(dtype)
            vw = f.get_tensor(base + "v_proj.weight").to(dtype)
            k_gain = f.get_tensor(base + "k_norm.weight").to(dtype)
            q_gain = f.get_tensor(base + "q_norm.weight").to(dtype)
            head_dim = k_gain.shape[0]
            heads = kw.shape[0] // head_dim
            if pool is not None:
                # [pool, heads, head_dim] — real, already q_norm'ed token queries.
                qp = pool[b].float()
                if qp.shape[1:] != (heads, head_dim):
                    raise ValueError(
                        f"block {b}: bank queries are {tuple(qp.shape[1:])}, this DiT "
                        f"wants ({heads}, {head_dim}) — bank built against another model"
                    )
                if qp.shape[0] < n_query:
                    raise ValueError(
                        f"block {b}: bank holds {qp.shape[0]} queries < --attn_queries "
                        f"{n_query} — rebuild with a larger --n_images/--per_forward"
                    )
                pick = torch.randperm(qp.shape[0], generator=g)[:n_query]
                q = qp[pick].permute(1, 0, 2).contiguous().to(dtype)
            else:
                q = torch.randn(
                    heads, n_query, head_dim, generator=g, dtype=torch.float32
                )
                q = _rms_norm(q, q_gain.float()).to(dtype)
            probes.append(
                BlockProbe(
                    block=b,
                    k_weight=kw.to(device),
                    v_weight=vw.to(device),
                    k_gain=k_gain.to(device),
                    queries=q.to(device),
                )
            )
            logger.info(
                "attn bank: block %d — %d heads × %d dim, %d probe queries",
                b,
                heads,
                head_dim,
                n_query,
            )
    return probes


def fit_centers(
    probes: list[BlockProbe],
    samples,
    *,
    seq_total: int = 512,
) -> list[BlockProbe]:
    """Set each probe's ``center`` to the mean readout over ``samples``.

    ``samples`` is an iterable of ``(x, mask)`` — use a fixed reference set (the
    cached *teacher* outputs) so the center is identical across G2 arms and
    independent of the batch being scored.

    Why this exists — measured, and it is not what ``report_0816_phase2.md`` blamed. The
    readout is *not* a washed-out sequence mean: attention is sharp (3-9
    effective tokens of ~99, with 80-87% of the softmax mass on the zero-pad
    sink) under both random and real queries. The defect is that every readout
    carries a large **common offset** — ``||mean|| / ||vec||`` is 0.73 with
    random queries and 1.02 with real ones — and an *uncentered* cosine over
    vectors sharing an offset that big saturates at 1 for everything. That is
    what made two unrelated prompts read 0.997 alike and what left ``L_attn``
    with no gradient about wording.

    The offset is real conditioning, but it is *shared* — teacher and student
    both have it, so matching it is free and carries no information. Projecting
    it out restores the space: measured on 128 held-out pairs with real queries,
    ``cos(teacher, en)`` 0.9993 → 0.9092, ``cos(native, en)`` 0.9967 → −0.1180,
    and the far control ``cos(t, t')`` 0.9973 → 0.0427.
    """
    if not probes:
        return probes
    sums = [None] * len(probes)
    n = 0
    for x, mask in samples:
        for i, p in enumerate(probes):
            r = readout(x, mask, p, seq_total=seq_total, center=False)
            s = r.sum(dim=0)
            sums[i] = s if sums[i] is None else sums[i] + s
        n += x.shape[0]
    if n == 0:
        raise ValueError("fit_centers: no samples")
    for p, s in zip(probes, sums):
        p.center = (s / n).to(p.queries.dtype)
        logger.info(
            "attn bank: block %d center ||mu||=%.4f", p.block, float(p.center.norm())
        )
    return probes


def readout(
    x: torch.Tensor,
    mask: torch.Tensor,
    probe: BlockProbe,
    *,
    seq_total: int = 512,
    center: bool = True,
) -> torch.Tensor:
    """Cross-attention readout of ``x`` under ``probe``: ``[B, heads, n_query, head_dim]``.

    ``mask`` marks the real (non-pad) rows of ``x``; the remaining
    ``seq_total - mask.sum()`` rows the DiT would see are zeros, folded into
    the denominator as ``n_sink * exp(0)``.

    With ``center=True`` (the default) the probe's fitted common offset is
    subtracted — see ``fit_centers`` for why the uncentered readout is a
    near-constant vector that no cosine can tell apart. Pass ``center=False``
    for the raw quantity the DiT actually consumes.
    """
    head_dim = probe.k_gain.shape[0]
    B, L, _ = x.shape
    xf = x.float()
    k = (xf @ probe.k_weight.float().T).view(B, L, -1, head_dim)
    v = (xf @ probe.v_weight.float().T).view(B, L, -1, head_dim)
    k = _rms_norm(k, probe.k_gain.float())

    q = probe.queries.float()  # [H, M, hd]
    # [B, H, M, L]
    logits = torch.einsum("hmd,blhd->bhml", q, k) / (head_dim**0.5)
    valid = mask.bool()[:, None, None, :]
    logits = logits.masked_fill(~valid, float("-inf"))

    n_sink = (seq_total - mask.sum(dim=1)).clamp(min=0).float()  # [B]
    # Stable softmax including the zero-logit sink block.
    m = torch.maximum(
        logits.masked_fill(~valid, float("-inf")).amax(dim=-1),
        torch.zeros(1, device=x.device),
    )  # [B, H, M]
    w = torch.exp(logits - m[..., None]).masked_fill(~valid, 0.0)
    den = w.sum(dim=-1) + n_sink[:, None, None] * torch.exp(-m)
    out = torch.einsum("bhml,blhd->bhmd", w, v)
    out = out / den[..., None].clamp_min(1e-12)
    if center and probe.center is not None:
        out = out - probe.center.to(out.dtype)
    return out
