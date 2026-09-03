"""Per-row bookkeeping shared by the distill loop and the coverage probes.

Everything here is about *rows* of the ext table rather than pairs: which
script a row's surface is in, how often the training pool looks it up, and
the two plan_zh2 refinements that key on that count —

``choose_holdout_rows``  U4: a seeded, script-stratified sample of *visited*
                         rows whose spans leave the training pool so the map
                         can be scored on rows it never trained.
``span_row_hits``        U1: per-span "touches a row in this set" — the
                         vectorized test both the row holdout and the span
                         visit floor need.

CPU only, tokenizer-light (the Qwen tokenizer is needed once to decode the
``qwen``/``sym`` block surfaces; the ``char`` blocks carry their surface in
the mapping).
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import torch

T5_TABLE_SIZE = 32128

BANDS = ("0", "1-4", "5-49", "50-499", "500+")


def band(v: int) -> str:
    if v == 0:
        return "0"
    if v < 5:
        return "1-4"
    if v < 50:
        return "5-49"
    if v < 500:
        return "50-499"
    return "500+"


def script_of(s: str) -> str:
    """Coarse script label of a row surface (``han`` / ``kana`` / ``hangul`` / …)."""
    kinds = set()
    for ch in s.strip():
        o = ord(ch)
        if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
            kinds.add("hangul")
        elif 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:
            kinds.add("kana")
        elif 0x4E00 <= o <= 0x9FFF:
            kinds.add("han")
        elif 0x3400 <= o <= 0x4DBF:
            kinds.add("han_extA")
        elif 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF:
            kinds.add("punct_fw")
        elif ch == " ":
            pass
        else:
            kinds.add("symbol")
    if not kinds:
        return "empty"
    return "+".join(sorted(kinds)) if len(kinds) > 1 else next(iter(kinds))


def row_surfaces(mapping: dict, qwen_tok) -> dict[int, tuple[str, str]]:
    """row → (block, surface). Blocks: qwen / char / sym / sym_char."""
    out: dict[int, tuple[str, str]] = {}
    for block in ("qwen", "sym"):
        qmap = {int(k): v for k, v in (mapping.get(block) or {}).items()}
        ids = sorted(qmap)
        for qid, s in zip(ids, qwen_tok.batch_decode([[i] for i in ids])):
            out[qmap[qid]] = (block, s)
    for block in ("char", "sym_char"):
        for ch, r in (mapping.get(block) or {}).items():
            out[r] = (block, ch)
    return out


def row_scripts(mapping: dict, qwen_tok, n_rows: int) -> list[str]:
    """``script_of`` per row, in row order (``empty`` for rows the mapping lacks)."""
    surf = row_surfaces(mapping, qwen_tok)
    return [script_of(surf[r][1]) if r in surf else "empty" for r in range(n_rows)]


def visits_from_caches(
    cache_dirs, n_rows: int, registers=(), split: str = "train"
) -> torch.Tensor:
    """Ext-row visit histogram straight off the staged shards (``.sids`` only).

    Same pool filter as ``distill.make_pool`` (register allow-list), but reads
    nothing except the id tensors — a few seconds per cache instead of the
    4 GB hidden-state load the training loop pays.
    """
    from safetensors import safe_open

    regs = {r for r in registers if r}
    visits = torch.zeros(n_rows, dtype=torch.long)
    for cd in cache_dirs:
        d = Path(cd) / split
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        by_shard: dict[str, list[str]] = collections.defaultdict(list)
        for rec in meta["pairs"]:
            if not regs or rec["register"] in regs:
                by_shard[rec["shard"]].append(rec["key"])
        for sh, keys in by_shard.items():
            with safe_open(str(d / sh), "pt") as f:
                for k in keys:
                    ids = f.get_tensor(f"{k}.sids").long()
                    e = ids[ids >= T5_TABLE_SIZE] - T5_TABLE_SIZE
                    e = e[e < n_rows]
                    if e.numel():
                        visits.index_add_(0, e, torch.ones_like(e))
    return visits


# ---------------------------------------------------------------------------
# U4 — row-disjoint holdout
# ---------------------------------------------------------------------------


def choose_holdout_rows(
    visits: torch.Tensor,
    scripts: list[str],
    frac: float,
    *,
    min_visits: int = 5,
    max_visits: int = 0,
    seed: int = 0,
    strata: tuple[str, ...] = ("han", "hangul", "kana"),
) -> torch.Tensor:
    """Seeded, script-stratified sample of ``frac`` of the eligible rows.

    Eligible = visited at least ``min_visits`` times (a row seen once cannot be
    held out meaningfully — there is one occurrence to score) and, when
    ``max_visits`` > 0, fewer than that: the 500+ band is ~140 rows carrying
    a third of all visits, so holding 5 % of it out strips ~5 % of every
    span's tokens from the pool (measured 2026-09-03: 4.8 % of visits without
    the cap, 0.17 % with cap 500), and a row that common is not the "row the
    map never saw" the holdout is asking about. Stratified by script so KO,
    which has few rows but many visits, does not dominate by row count;
    scripts outside ``strata`` (mixed, symbol, fullwidth) pool into one
    ``other`` bucket. Returns sorted row indices.
    """
    if frac <= 0:
        return torch.zeros(0, dtype=torch.long)
    ok = visits >= min_visits
    if max_visits > 0:
        ok &= visits < max_visits
    eligible = ok.nonzero(as_tuple=True)[0].tolist()
    by_stratum: dict[str, list[int]] = collections.defaultdict(list)
    for r in eligible:
        s = scripts[r] if r < len(scripts) else "empty"
        by_stratum[s if s in strata else "other"].append(r)
    rng = random.Random(seed)
    chosen: list[int] = []
    for s in sorted(by_stratum):
        rows = sorted(by_stratum[s])
        k = int(round(len(rows) * frac))
        chosen.extend(rng.sample(rows, k))
    return torch.tensor(sorted(chosen), dtype=torch.long)


def span_row_hits(
    s_ids_flat: torch.Tensor,
    s_flat: torch.Tensor,
    s_seg: torch.Tensor,
    n_spans: int,
    row_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-span count of student tokens that are ext rows flagged in ``row_mask``.

    ``s_ids_flat`` is the batch's ``s_ids.reshape(-1)``; ``s_flat`` / ``s_seg``
    / ``n_spans`` are the collated span pack. One gather + one index_add — the
    same shape as ``focus_spans`` so both filters stay launch-cheap.
    """
    rows = s_ids_flat[s_flat]
    is_ext = rows >= T5_TABLE_SIZE
    idx = (rows - T5_TABLE_SIZE).clamp(min=0, max=row_mask.numel() - 1)
    hit = (is_ext & row_mask[idx]).to(torch.float32)
    out = torch.zeros(n_spans, device=hit.device, dtype=torch.float32)
    out.index_add_(0, s_seg, hit)
    return out


def apply_row_holdout(
    train_cache, pool: list[int], held_rows: torch.Tensor
) -> dict[int, list[list]]:
    """Strip every span touching a held-out row from the *records* (in place).

    Spans, not pairs: a pair with one held-out row keeps its other spans. The
    stripped spans are returned per pair index so the eval can score exactly
    those. The pair's tokens are unchanged — the held-out row is still looked
    up in the forward, so a residual signal reaches the shared map through
    context (its embedding shapes the neighbours' outputs); what the holdout
    removes is every span that *supervises* the row directly.
    """
    held: dict[int, list[list]] = {}
    if held_rows.numel() == 0:
        return held
    mask = torch.zeros(int(held_rows.max()) + 1, dtype=torch.bool)
    mask[held_rows] = True
    n_mask = mask.numel()
    for i in pool:
        rec = train_cache.records[i]
        spans = rec.get("spans")
        if not spans:
            continue
        s_ids = train_cache.get(i).s_ids
        keep, drop = [], []
        for s in spans:
            rows = s_ids[torch.as_tensor(s[1], dtype=torch.long)] - T5_TABLE_SIZE
            rows = rows[(rows >= 0) & (rows < n_mask)]
            (drop if rows.numel() and bool(mask[rows].any()) else keep).append(s)
        if drop:
            rec["spans"] = keep
            held[i] = drop
    train_cache._span_cache.clear()
    return held
