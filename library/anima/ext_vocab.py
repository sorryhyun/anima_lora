"""CJK vocab extension for the LLM Adapter's T5-side query stream.

Promoted from ``bench/cjk_adapter/ext_vocab.py`` (which now re-exports from
here) — the runtime surface (``segment_runs`` / ``HybridT5Encoder`` /
``load_ext_assets``) is what the vocab-pack loaders (in-repo shim, ComfyUI
node vendor tree) consume; the ``build_*`` / ``fit_anchor_map`` helpers are
build-time only (``bench/cjk_adapter/build_ext.py``).

The adapter's target vocab is the old t5-v1_1-xxl spiece (32128 rows, no CJK).
Instead of training a new SentencePiece, we borrow the CJK subset of Qwen3's
own tokenizer — every borrowed piece is an existing Qwen token, so its
extended-row init is an exact anchor-mapped embedding (``W @ qwen_embed[id]``,
where ``W`` is a ridge least-squares fit Qwen-embed → T5-embed on tokens whose
surface form exists in both vocabs), and the Qwen source stream and T5 query
stream segment CJK spans identically (token-aligned cross-attention).

Byte-fragment fallback: Qwen is byte-BPE, so some chars tokenize as UTF-8
fragments. Those get per-character supplementary rows, initialised from the
mapped mean of their fragment embeddings. Plain mean is order-invariant, so
two chars whose UTF-8 bytes are permutations of each other would collide
bit-identically (527 such pairs, e.g. 鯰/鰯 — the Phase 0.2 separability
finding); exactly those colliding rows use a position-weighted mean instead,
which breaks the tie while leaving every non-colliding row at the plain mean.

Symbol routing (2026-09-03): the stock spiece also has no row for a long
tail of non-CJK symbols (``^`` ``<`` ``~`` ``·`` ``×`` ``☆``, emoji …) that
danbooru tags and zh names use — T5 folds ``^^^`` into a single ``<unk>``, so
``^^^`` / ``☆`` / ``\\`` were the same token. Those chars are routed to the
Qwen side exactly like CJK, with their rows appended *after* the CJK blocks
(``mapping["sym"]`` / ``mapping["sym_char"]``) so every pre-existing row id,
distill cache and trained pack stays valid. The routing rule ships **inside
the pack json** (``mapping["route"]``); a pack without it routes the legacy
CJK ranges only, bit-identical to before.

Quote partition (2026-09-05, DiT line D1): a pack may carry a second,
**content-free isotropic block** (``mapping["iso"]``: i.i.d. Gaussian rows
regenerated from ``(seed, n_rows, dim, norm)`` — :func:`iso_block`) that
mirrors the trained blocks row-for-row at an offset. Routed spans *inside a
quote pair* (``route.quotes``: ``「…」`` / ``『…』`` / ``"…"``) resolve to the
isotropic block; bare CJK keeps the trained rows; the delimiters themselves
stay on their usual path (``「」`` → trained row, ``"`` → spiece). A pack
without ``iso`` encodes bit-identically to before. :func:`pack_digest` is
the hash a LoRA trained through the pack stamps (``ss_ext_pack_sha``).

Pure-CPU module — no model load; consumers pass embedding tensors in.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

T5_PAD_ID = 0
T5_EOS_ID = 1
T5_UNK_ID = 2
T5_TABLE_SIZE = 32128  # embedding rows in the checkpoint; ext ids start here

# Character ranges routed to the Qwen fallback (and eligible for rows).
_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Ext A
    (0x3040, 0x309F),  # hiragana
    (0x30A0, 0x30FF),  # katakana
    (0x31F0, 0x31FF),  # katakana phonetic ext
    (0xAC00, 0xD7AF),  # hangul syllables
    (0x1100, 0x11FF),  # hangul jamo
    (0x3130, 0x318F),  # hangul compat jamo
    (0x3000, 0x303F),  # CJK symbols & punctuation (「」、。々 etc.)
    (0xFF00, 0xFFEF),  # fullwidth forms
)


def is_cjk_char(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK_RANGES)


def is_hangul_char(ch: str) -> bool:
    o = ord(ch)
    return 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F


@dataclass(frozen=True)
class Route:
    """Which characters leave the T5 spiece path for the Qwen-side ext rows.

    ``ranges`` are inclusive codepoint intervals (the legacy CJK blocks);
    ``chars`` is an explicit set (the symbol tail, chosen at build time as
    "T5 emits ``<unk>`` for it"). Serialised into the pack json so the
    ComfyUI node and the trainer segment a prompt identically without a code
    release per rule change; a pack that carries no ``route`` gets
    :meth:`default`, which is exactly :func:`is_cjk_char`.
    """

    ranges: tuple[tuple[int, int], ...] = _CJK_RANGES
    chars: frozenset[str] = frozenset()
    # Delimiter pairs whose *content* routes to the isotropic block (only
    # meaningful when the pack also carries ``iso``; see ``quote_spans``).
    quotes: tuple[tuple[str, str], ...] = ()

    def __call__(self, ch: str) -> bool:
        if ch in self.chars:
            return True
        o = ord(ch)
        return any(lo <= o <= hi for lo, hi in self.ranges)

    def any(self, text: str) -> bool:
        return any(self(c) for c in text)

    @classmethod
    def default(cls) -> "Route":
        return cls()

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> "Route":
        spec = (mapping or {}).get("route")
        if not spec:
            return cls.default()
        ranges = tuple((int(lo), int(hi)) for lo, hi in spec.get("ranges", _CJK_RANGES))
        quotes = tuple((str(o), str(c)) for o, c in spec.get("quotes", ()))
        return cls(ranges=ranges, chars=frozenset(spec.get("chars", "")), quotes=quotes)

    def to_json(self) -> dict:
        out = {
            "ranges": [list(r) for r in self.ranges],
            "chars": "".join(sorted(self.chars)),
        }
        if self.quotes:
            out["quotes"] = [list(q) for q in self.quotes]
        return out

    def quote_spans(self, text: str) -> list[tuple[int, int]]:
        """``[start, end)`` of the *content* of every closed quote pair.

        One non-greedy regex over the whole caption, alternation over the
        pack's pairs, no nesting: an opener without its closer matches
        nothing (so a stray ``"`` never swallows the rest of the prompt),
        and pairs are consumed left to right. Delimiters fall outside the
        returned spans.
        """
        if not self.quotes:
            return []
        pat = "|".join(
            f"{re.escape(o)}([^{re.escape(o)}{re.escape(c)}]*?){re.escape(c)}"
            for o, c in self.quotes
        )
        spans: list[tuple[int, int]] = []
        for m in re.finditer(pat, text):
            g = next(i for i in range(1, len(m.groups()) + 1) if m.group(i) is not None)
            if m.end(g) > m.start(g):
                spans.append((m.start(g), m.end(g)))
        return spans


# The quote pairs the D1 span rule recognises (principle 8 of the DiT plan):
# CJK corner brackets (both weights) and the ASCII double quote new caption
# builders emit. Script-neutral by design.
DEFAULT_QUOTES: tuple[tuple[str, str], ...] = (("「", "」"), ("『", "』"), ('"', '"'))


# ---------------------------------------------------------------------------
# Isotropic block — content-free rows regenerated from a seed
# ---------------------------------------------------------------------------

ISO_RECIPE = "gauss_rows_v1"


def iso_block(
    seed: int, n_rows: int, dim: int, norm: float, chunk: int = 4096
) -> torch.Tensor:
    """``(n_rows, dim)`` float32 rows, i.i.d. Gaussian directions at ``norm``.

    Byte-reproducible across machines: NumPy's legacy ``RandomState`` stream
    (its bit-stability is a NumPy compatibility guarantee — NEP 19, unlike
    ``Generator``/torch RNGs), drawn in float64 row-major so chunking never
    changes the values, each row normalised in float64 and cast once. No
    BLAS, no threads, no device — the pack json only needs
    ``(seed, n_rows, dim, norm)`` to get the identical table back.
    """
    import numpy as np

    rs = np.random.RandomState(int(seed))
    out = torch.empty(int(n_rows), int(dim), dtype=torch.float32)
    done = 0
    while done < n_rows:
        n = min(chunk, n_rows - done)
        z = rs.standard_normal((n, int(dim)))  # float64, sequential stream
        z *= float(norm) / np.sqrt((z * z).sum(axis=1, keepdims=True))
        out[done : done + n] = torch.from_numpy(z.astype(np.float32))
        done += n
    return out


@dataclass(frozen=True)
class IsoSpec:
    """The ``mapping["iso"]`` record: how to regenerate the isotropic block
    and where it sits in the table (``rows = [start, end)``)."""

    seed: int
    n_rows: int
    dim: int
    norm: float
    start: int
    recipe: str = ISO_RECIPE

    @property
    def end(self) -> int:
        return self.start + self.n_rows

    @classmethod
    def from_mapping(cls, mapping: dict | None) -> "IsoSpec | None":
        spec = (mapping or {}).get("iso")
        if not spec:
            return None
        start, end = spec["rows"]
        return cls(
            seed=int(spec["seed"]),
            n_rows=int(end) - int(start),
            dim=int(spec["dim"]),
            norm=float(spec["norm"]),
            start=int(start),
            recipe=str(spec.get("recipe", ISO_RECIPE)),
        )

    def to_json(self) -> dict:
        return {
            "recipe": self.recipe,
            "seed": self.seed,
            "dim": self.dim,
            "norm": self.norm,
            "rows": [self.start, self.end],
        }

    def build(self) -> torch.Tensor:
        if self.recipe != ISO_RECIPE:
            raise ValueError(f"unknown iso recipe {self.recipe!r} (have {ISO_RECIPE})")
        return iso_block(self.seed, self.n_rows, self.dim, self.norm)


def materialize_iso(table: torch.Tensor, mapping: dict) -> torch.Tensor:
    """Append the isotropic block when the pack shipped without its rows.

    A pack may carry only the ``iso`` record (seed-only, ~0 bytes) or the
    rows themselves; either way the table handed to the model is the same.
    Raises when the table is neither ``[0, start)`` nor ``[0, end)`` rows —
    the json is from a different pack.
    """
    spec = IsoSpec.from_mapping(mapping)
    if spec is None:
        return table
    if table.shape[0] == spec.end:
        return table
    if table.shape[0] != spec.start:
        raise ValueError(
            f"vocab pack mismatch: iso block starts at row {spec.start} but the "
            f"table has {table.shape[0]} rows"
        )
    return torch.cat([table, spec.build().to(table.dtype)])


# Mapping keys that describe the *rows and routing* — what a LoRA trained
# through the pack is coupled to. Provenance (``training`` / ``stats``) is
# excluded so a re-annotated json keeps its digest.
_DIGEST_KEYS = ("qwen", "char", "sym", "sym_char", "word", "word_sub", "route", "iso")


def pack_digest(table: torch.Tensor, mapping: dict) -> str:
    """sha256 over the (materialised, float32) table bytes + the id/route maps.

    Stamped by ``save_weights`` as ``ss_ext_pack_sha``; the trainer and the
    ComfyUI node compute it the same way so a LoRA meeting a different pack
    (rows, ids or quote rule) is detectable, never silent.
    """
    table = materialize_iso(table, mapping)
    h = hashlib.sha256()
    h.update(table.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    sub = {k: mapping[k] for k in _DIGEST_KEYS if mapping.get(k)}
    h.update(json.dumps(sub, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


# Candidate blocks scanned by ``symbol_route_chars`` — ASCII/Latin-1 symbols,
# general punctuation through dingbats, enclosed/compat CJK, variation
# selectors, emoji. Letters in these blocks that T5 *can* spell are filtered
# out by the ``<unk>`` test, so only genuine spiece gaps get routed.
SYMBOL_CANDIDATE_RANGES = (
    (0x0021, 0x007E),  # ASCII printable (^ < > ~ \ ` { } | …)
    (0x00A1, 0x00BF),  # Latin-1 punctuation & symbols (¡ · ¿ …)
    (0x00D7, 0x00D7),  # ×
    (0x00F7, 0x00F7),  # ÷
    # Kaomoji alphabet — (˘ω˘) ٩(◕‿◕｡)۶ (꒳) ˗ˏˋ ˎˊ˗ — spelled with IPA /
    # spacing modifiers, combining marks, Greek, Arabic and Yi radicals; the
    # <unk> test keeps only what T5 cannot spell.
    (0x0250, 0x036F),  # IPA extensions, spacing modifier letters, combining marks
    (0x0370, 0x03FF),  # Greek (ω ρ δ …)
    (0x0600, 0x06FF),  # Arabic (٩ ۶ و …)
    (0x1D00, 0x1DBF),  # phonetic extensions (ᵕ ᴗ …)
    (0x2000, 0x2BFF),  # general punct, arrows, math, shapes, misc symbols, dingbats
    (0x2E00, 0x2E7F),  # supplemental punctuation
    (0x3200, 0x33FF),  # enclosed CJK letters/months, CJK compatibility (㊙ ㎝ …)
    (0xA490, 0xA4CF),  # Yi radicals (꒳)
    (0xFE00, 0xFE0F),  # variation selectors (emoji presentation U+FE0F)
    (0x1F000, 0x1FAFF),  # emoji & pictographs
)


def symbol_route_chars(t5_tok, qwen_tok, candidates=SYMBOL_CANDIDATE_RANGES) -> str:
    """Chars T5 spiece cannot encode (emits ``<unk>``) but Qwen round-trips.

    Deterministic given the two tokenizers, so the result is written into the
    pack json once (``route.chars``) and never recomputed at load time. Chars
    already inside the legacy CJK ranges are excluded (they route anyway).
    """
    base = Route.default()
    out: list[str] = []
    for lo, hi in candidates:
        for o in range(lo, hi + 1):
            ch = chr(o)
            if base(ch) or ch.isspace():
                continue
            t5_ids = t5_tok(ch, add_special_tokens=False)["input_ids"]
            if T5_UNK_ID not in t5_ids:
                continue
            q_ids = qwen_tok(ch, add_special_tokens=False)["input_ids"]
            if not q_ids or qwen_tok.decode(q_ids) != ch:
                continue
            out.append(ch)
    return "".join(out)


def segment_runs(text: str, route: "Route | None" = None) -> list[tuple[str, str]]:
    """Split text into ("t5"|"cjk", span) runs.

    ``route`` decides which chars leave the spiece path (default: the legacy
    CJK ranges). The "cjk" kind name is kept for every routed run — symbol
    runs included — because the row lookup downstream is the same.

    Whitespace immediately preceding a routed run is folded into it (Qwen's
    byte-BPE handles leading spaces natively; T5 spiece would just emit a
    bare ``▁``).
    """
    route = route or is_cjk_char
    runs: list[tuple[str, str]] = []
    buf, kind = [], None
    for ch in text:
        k = "cjk" if route(ch) else "t5"
        if k != kind and buf:
            runs.append((kind, "".join(buf)))
            buf = []
        buf.append(ch)
        kind = k
    if buf:
        runs.append((kind, "".join(buf)))

    merged: list[tuple[str, str]] = []
    for kind, span in runs:
        if (
            kind == "cjk"
            and merged
            and merged[-1][0] == "t5"
            and merged[-1][1].isspace()
        ):
            span = merged.pop()[1] + span
        if merged and merged[-1][0] == kind:
            span = merged.pop()[1] + span
        merged.append((kind, span))
    return merged


def collect_clean_qwen_tokens(qwen_tok, chars: str | None = None) -> dict[int, str]:
    """Qwen token ids whose decoded surface is pure CJK (spaces allowed).

    With ``chars`` given, the surface must instead be made only of those
    chars (plus spaces) — the symbol block. The two calls partition the
    vocab: a token mixing CJK and symbol chars lands in neither and resolves
    per-char through the fragment path at encode time.
    """
    n = qwen_tok.vocab_size
    surfaces = qwen_tok.batch_decode([[i] for i in range(n)])
    if chars is None:
        ok = is_cjk_char
    else:
        allowed = frozenset(chars)
        ok = allowed.__contains__
    clean = {}
    for i, s in enumerate(surfaces):
        core = s.strip()
        if core and "�" not in s and all(ok(c) or c == " " for c in s):
            clean[i] = s
    return clean


def build_anchor_pairs(t5_tok, qwen_tok) -> tuple[list[int], list[int]]:
    """(t5_ids, qwen_ids) of tokens with identical decoded surfaces."""
    qwen_surface_to_id: dict[str, int] = {}
    n = qwen_tok.vocab_size
    for i, s in enumerate(qwen_tok.batch_decode([[i] for i in range(n)])):
        if s and "�" not in s:
            qwen_surface_to_id.setdefault(s, i)

    t5_ids, qwen_ids = [], []
    for piece, tid in t5_tok.get_vocab().items():
        if tid >= T5_TABLE_SIZE or piece.startswith("<"):
            continue
        surface = piece.replace("▁", " ")
        qid = qwen_surface_to_id.get(surface)
        if qid is not None:
            t5_ids.append(tid)
            qwen_ids.append(qid)
    return t5_ids, qwen_ids


MAP_METHODS = ("ridge", "procrustes-mix")


def fit_anchor_map(
    qwen_embed: torch.Tensor,
    t5_embed: torch.Tensor,
    t5_ids: list[int],
    qwen_ids: list[int],
    ridge: float = 1e-2,
    holdout: int = 1000,
    method: str = "ridge",
    mix: float = 0.6,
) -> tuple[torch.Tensor, float]:
    """Anchor map W (d_qwen × d_t5) with a held-out cosine sanity.

    ``method="ridge"`` — plain ridge least squares (the v1 asset). Ridge
    shrinks toward the directions the anchors share, so the mapped ext keys
    collapse onto a thin subspace (PR 236 of 1024, 16 % of random row pairs
    above cos 0.5; ``probes/map_probe.py``, 2026-09-02).

    ``method="procrustes-mix"`` — ridge plus ``mix`` × the scaled orthogonal
    Procrustes rotation fitted on the same anchors (centered fit, applied as
    a plain linear map like the ridge term). The rotation carries the
    Qwen spectrum through untouched, so the mix keeps the ext keys spread
    (PR 373, 1 % collisions) for a small held-out cost (cos 0.75 → 0.70).
    ``mix=0.6`` is the probe's measured point; the v2 asset default.

    Returns (W, mean held-out cosine of ``qwen_embed @ W`` vs the true T5 row).
    """
    if method not in MAP_METHODS:
        raise ValueError(f"unknown anchor-map method {method!r} (want {MAP_METHODS})")
    A = qwen_embed[qwen_ids].float()
    B = t5_embed[t5_ids].float()
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(t5_ids), generator=g)
    hold, train = perm[:holdout], perm[holdout:]
    At, Bt = A[train], B[train]
    d = At.shape[1]
    lam = ridge * At.pow(2).mean() * len(train)
    W = torch.linalg.solve(At.T @ At + lam * torch.eye(d), At.T @ Bt)
    if method == "procrustes-mix":
        ma, mb = At.mean(0), Bt.mean(0)
        U, _, Vt = torch.linalg.svd((At - ma).T @ (Bt - mb))
        R = U @ Vt
        scale = float((Bt - mb).norm() / ((At - ma) @ R).norm())
        W = W + mix * (R * scale)
    cos = torch.nn.functional.cosine_similarity(A[hold] @ W, B[hold], dim=-1)
    return W, float(cos.mean())


def char_row_surfaces(
    qwen_tok, clean: dict[int, str], chars: str | None = None
) -> dict[str, list[int]]:
    """Chars that get a supplementary row → their Qwen byte-fragment ids.

    Every standard-range char that is not itself a clean single Qwen token
    and round-trips through the tokenizer. Shared by the table builder and
    the contextual-init path (``build_ext.py --char-init contextual``), so
    both see the identical char inventory and row order. ``chars`` swaps the
    legacy CJK ranges for an explicit inventory (the symbol block).
    """
    single_clean = {s: qid for qid, s in clean.items() if len(s) == 1}
    char_ids: dict[str, list[int]] = {}
    inventory = (
        (chr(o) for lo, hi in _CJK_RANGES for o in range(lo, hi + 1))
        if chars is None
        else iter(chars)
    )
    for ch in inventory:
        if ch in single_clean:
            continue
        ids = qwen_tok(ch, add_special_tokens=False)["input_ids"]
        if not ids or qwen_tok.decode(ids) != ch:
            continue
        char_ids[ch] = ids
    return char_ids


def build_ext_table(
    qwen_tok,
    qwen_embed: torch.Tensor,
    W: torch.Tensor,
    clean: dict[int, str],
    char_init: dict[str, torch.Tensor] | None = None,
    *,
    symbols: str | None = None,
    sym_clean: dict[int, str] | None = None,
) -> tuple[torch.Tensor, dict]:
    """Extended rows + id mapping.

    Rows = every clean Qwen CJK token, plus per-character supplementary rows
    for standard-range chars whose Qwen tokenization is byte-fragments only
    (init = mapped mean of fragment embeddings; chars sharing a fragment-id
    multiset with another char get a position-weighted mean so permuted byte
    orders cannot land on bit-identical rows).

    ``char_init`` — optional per-char vectors *already in T5 space* that
    replace the fragment-mean init for the chars they cover (the v2 asset's
    contextual init: the fragment mean is near-degenerate because the lead
    byte is shared by thousands of chars — 60 % of random char-row pairs
    above cos 0.5, ``probes/char_probe.py``). Chars not covered keep the
    fragment mean. The id mapping is identical either way.

    ``symbols`` (the string :func:`symbol_route_chars` returns) + ``sym_clean``
    (``collect_clean_qwen_tokens(qwen_tok, chars=symbols)``) append the symbol
    block **after** the two CJK blocks: symbol tokens sorted by Qwen id, then
    per-char rows for symbol chars that are byte-fragments. Row ids of the
    CJK blocks are untouched, so a table built with symbols is a strict
    superset of one built without.
    """
    qwen_map: dict[int, int] = {}
    vecs: list[torch.Tensor] = []
    for qid in sorted(clean):
        qwen_map[qid] = len(vecs)
        vecs.append(qwen_embed[qid].float() @ W)

    def _char_rows(char_ids: dict[str, list[int]]) -> tuple[dict[str, int], int]:
        by_multiset: dict[tuple[int, ...], int] = {}
        for ids in char_ids.values():
            key = tuple(sorted(ids))
            by_multiset[key] = by_multiset.get(key, 0) + 1
        cmap: dict[str, int] = {}
        n_ctx = 0
        for ch, ids in char_ids.items():
            if char_init is not None and ch in char_init:
                cmap[ch] = len(vecs)
                vecs.append(char_init[ch].float().reshape(-1))
                n_ctx += 1
                continue
            emb = qwen_embed[ids].float()
            if by_multiset[tuple(sorted(ids))] > 1:
                # Same byte multiset as another char — order is the only
                # distinguishing signal, so pool with position weights 1..n.
                w = torch.arange(1, len(ids) + 1, dtype=torch.float32)
                pooled = (emb * w.unsqueeze(1)).sum(dim=0) / w.sum()
            else:
                pooled = emb.mean(dim=0)
            cmap[ch] = len(vecs)
            vecs.append(pooled @ W)
        return cmap, n_ctx

    char_map, n_contextual = _char_rows(char_row_surfaces(qwen_tok, clean))
    n_cjk = len(vecs)

    sym_map: dict[int, int] = {}
    sym_char_map: dict[str, int] = {}
    route = Route.default()
    if symbols:
        sym_clean = sym_clean or {}
        for qid in sorted(sym_clean):
            sym_map[qid] = len(vecs)
            vecs.append(qwen_embed[qid].float() @ W)
        sym_char_map, n_sym_ctx = _char_rows(
            char_row_surfaces(qwen_tok, sym_clean, chars=symbols)
        )
        n_contextual += n_sym_ctx
        route = Route(chars=frozenset(symbols))

    table = torch.stack(vecs) if vecs else torch.zeros(0, W.shape[1])
    mapping = {
        "qwen": {str(k): v for k, v in qwen_map.items()},
        "char": char_map,
        "rows": table.shape[0],
        "char_rows_contextual": n_contextual,
    }
    if symbols:
        mapping["sym"] = {str(k): v for k, v in sym_map.items()}
        mapping["sym_char"] = sym_char_map
        mapping["sym_rows"] = [n_cjk, table.shape[0]]
        mapping["route"] = route.to_json()
    return table, mapping


@dataclass
class HybridT5Encoder:
    """T5-side encoder: old spiece for covered text, extended rows for CJK.

    Non-CJK spans tokenize exactly as before (bit-identical for pure-English
    prompts). CJK spans tokenize with the Qwen tokenizer: clean tokens map to
    ``T5_TABLE_SIZE + row``; byte-fragment runs are regrouped into characters
    and looked up per-char (unknown chars degrade to the old ``<unk>``).
    """

    t5_tok: object
    qwen_tok: object
    qwen_map: dict[int, int]
    char_map: dict[str, int]
    word_map: dict[str, int] | None = None
    # Which chars leave the spiece path. ``None`` = the legacy CJK ranges.
    route: Route | None = None
    # surface → base-vocab t5 id sequence (the C fallback: substitute a known
    # CJK surface with the EN tag's stock spiece tokens at encode time — the
    # pretrained rows carry the identity no minted row learns; zero training).
    word_sub: dict[str, list[int]] | None = None
    # Row offset of the isotropic block (``mapping["iso"]["rows"][0]``).
    # With ``route.quotes`` set, routed spans inside a quote pair land on
    # ``T5_TABLE_SIZE + iso_offset + row`` instead of the trained row.
    iso_offset: int | None = None

    @classmethod
    def from_mapping(cls, t5_tok, qwen_tok, mapping: dict) -> "HybridT5Encoder":
        qwen_map = {int(k): v for k, v in mapping["qwen"].items()}
        # The symbol block (if the pack has one) is a plain extension of the
        # same two lookups — kept separate in the json only so the CJK row
        # ids stay contiguous for everything that indexes them.
        qwen_map.update({int(k): v for k, v in (mapping.get("sym") or {}).items()})
        # Chars that are clean single tokens were excluded from char_map (no
        # separate row needed) — but they can still arrive as byte-fragments
        # mid-word, so the char lookup must cover them via their token row.
        char_map = dict(mapping["char"])
        char_map.update(mapping.get("sym_char") or {})
        ids = sorted(qwen_map)
        for qid, s in zip(ids, qwen_tok.batch_decode([[i] for i in ids])):
            core = s.strip()
            if len(core) == 1:
                char_map.setdefault(core, qwen_map[qid])
        return cls(
            t5_tok=t5_tok,
            qwen_tok=qwen_tok,
            qwen_map=qwen_map,
            char_map=char_map,
            word_map=mapping.get("word") or None,
            word_sub=mapping.get("word_sub") or None,
            route=Route.from_mapping(mapping),
            iso_offset=(iso.start if (iso := IsoSpec.from_mapping(mapping)) else None),
        )

    @property
    def quote_routing(self) -> bool:
        """Both halves of the partition present: a quote rule and a block."""
        return bool(self.iso_offset is not None and self.route and self.route.quotes)

    def routes(self, text: str) -> bool:
        """Does any char of ``text`` leave the spiece path under this pack?"""
        return (self.route or Route.default()).any(text)

    def _encode_cjk(
        self, span: str, offset: int = 0
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """(ids, char offsets into ``span``) for one CJK run.

        ``offset`` shifts every ext row id (never ``<unk>``) — the isotropic
        block is a row-for-row mirror of the trained blocks, so quoted
        content uses the same lookups at ``iso_offset``.
        """
        base = T5_TABLE_SIZE + int(offset)
        out: list[int] = []
        offs: list[tuple[int, int]] = []
        frag: list[int] = []
        frag_off: list[tuple[int, int]] = []

        def flush_frag():
            if not frag:
                return
            decoded = self.qwen_tok.decode(frag)
            # A byte-fragment group resolves to characters jointly, so every
            # character it yields inherits the whole group's span.
            group = (frag_off[0][0], frag_off[-1][1])
            for ch in decoded:
                if ch in self.char_map:
                    out.append(base + self.char_map[ch])
                elif not ch.isspace():
                    out.append(T5_UNK_ID)
                else:
                    continue
                offs.append(group)
            frag.clear()
            frag_off.clear()

        enc = self.qwen_tok(span, add_special_tokens=False, return_offsets_mapping=True)
        for qid, off in zip(enc["input_ids"], enc["offset_mapping"]):
            off = (int(off[0]), int(off[1]))
            if qid in self.qwen_map:
                # A clean token can never complete a byte sequence — any
                # pending fragments are unresolvable, degrade them now.
                flush_frag()
                out.append(base + self.qwen_map[qid])
                offs.append(off)
                continue
            frag.append(qid)
            frag_off.append(off)
            if "�" not in self.qwen_tok.decode(frag):
                flush_frag()
        flush_frag()
        return out, offs

    def _encode_cjk_words(self, span: str) -> tuple[list[int], list[tuple[int, int]]]:
        """``_encode_cjk`` with a greedy longest-match pass over ``word_map``.

        Minted word rows (``mapping["word"]``) take one slot for a surface the
        base vocab would spell out char-by-char; everything between matches
        goes through the ordinary Qwen path unchanged.

        Eojeol boundary guard (plan_ko3 risk 1): a hangul surface may only
        match where an eojeol starts — BOS, or after a non-hangul char
        (space/punct/other script). Particles attach at the *end* of an
        eojeol, so a boundary-anchored prefix match still fires on 레이무가;
        what the guard kills is a surface waking up mid-word (…아레이무…).
        JA has no spaces — minting JA words is deferred until it gets its own
        boundary design (plan_ko3 M3).
        """
        surfaces: set[str] = set(self.word_map or ()) | set(self.word_sub or ())
        if not surfaces:
            return self._encode_cjk(span)
        lengths = sorted({len(w) for w in surfaces}, reverse=True)
        out: list[int] = []
        offs: list[tuple[int, int]] = []
        i, rest_start = 0, 0

        def flush_rest(end: int):
            if end > rest_start:
                r_ids, r_offs = self._encode_cjk(span[rest_start:end])
                out.extend(r_ids)
                offs.extend((rest_start + a, rest_start + b) for a, b in r_offs)

        while i < len(span):
            if is_hangul_char(span[i]) and i > 0 and is_hangul_char(span[i - 1]):
                i += 1
                continue
            hit = next(
                (
                    span[i : i + n]
                    for n in lengths
                    if i + n <= len(span) and span[i : i + n] in surfaces
                ),
                None,
            )
            if hit is None:
                i += 1
                continue
            flush_rest(i)
            group = (i, i + len(hit))
            if self.word_sub and hit in self.word_sub:
                # C fallback wins over a minted row for the same surface: the
                # substituted stock tokens carry pretrained identity.
                sub = self.word_sub[hit]
                out.extend(int(t) for t in sub)
                offs.extend(group for _ in sub)
            else:
                out.append(T5_TABLE_SIZE + self.word_map[hit])
                offs.append(group)
            i += len(hit)
            rest_start = i
        flush_rest(len(span))
        return out, offs

    def encode_aligned(
        self, text: str, max_length: int = 512
    ) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        """``encode`` plus per-token ``(start, end)`` char offsets into ``text``.

        The offsets are what makes span-level supervision possible: a caption
        composed tag-by-tag (``build_pairs.py``) knows which EN tag each JA tag
        came from, and these offsets turn that into token index sets on both
        sides. Offsets cover the real tokens only — the trailing EOS gets a
        zero-width span at ``len(text)`` and padding gets none.
        """
        ids: list[int] = []
        offs: list[tuple[int, int]] = []
        base = 0
        quotes = self.quote_spans(text)
        for kind, span in segment_runs(text, self.route):
            if kind == "cjk":
                s_ids, s_offs = self.encode_cjk_run(span, base, quotes)
            else:
                enc = self.t5_tok(
                    span, add_special_tokens=False, return_offsets_mapping=True
                )
                s_ids = list(enc["input_ids"])
                s_offs = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
            ids.extend(s_ids)
            offs.extend((base + a, base + b) for a, b in s_offs)
            base += len(span)

        keep = max_length - 1
        ids, offs = ids[:keep], offs[:keep]
        ids.append(T5_EOS_ID)
        offs.append((len(text), len(text)))
        mask = [1] * len(ids)
        pad = max_length - len(ids)
        return ids + [T5_PAD_ID] * pad, mask + [0] * pad, offs

    def quote_spans(self, text: str) -> list[tuple[int, int]]:
        """Quote-content intervals of ``text`` under this pack (``[]`` unless
        both halves of the partition — rule and block — are present)."""
        return self.route.quote_spans(text) if self.quote_routing else []

    @staticmethod
    def _cut_run(
        span: str, start: int, quotes: list[tuple[int, int]]
    ) -> list[tuple[str, bool, int]]:
        """Cut one routed run ``text[start:start+len(span)]`` at the quote
        boundaries → ``(piece, inside_quote, offset_in_span)``."""
        end = start + len(span)
        pieces: list[tuple[str, bool, int]] = []
        pos = start
        for a, b in quotes:
            a, b = max(a, start), min(b, end)
            if b <= a:
                continue
            if a > pos:
                pieces.append((span[pos - start : a - start], False, pos - start))
            pieces.append((span[a - start : b - start], True, a - start))
            pos = b
        if pos < end:
            pieces.append((span[pos - start :], False, pos - start))
        return pieces

    def encode_cjk_run(
        self, span: str, start: int, quotes: list[tuple[int, int]]
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Ids + offsets (into ``span``) for one routed run of a text whose
        quote intervals are ``quotes`` (absolute; from :meth:`quote_spans`).

        The span rule: the regex ran once over the whole text *before*
        ``segment_runs``, so the spiece side is tokenised exactly as without
        the partition (EN stays bit-identical) and only routed runs are cut at
        the delimiters. Quoted pieces go to the isotropic mirror with no
        minted-word / C-fallback substitutions (those are trained content);
        unquoted pieces take the ordinary path.
        """
        if not quotes:
            return self._encode_cjk_words(span)
        ids: list[int] = []
        offs: list[tuple[int, int]] = []
        for piece, quoted, rel in self._cut_run(span, start, quotes):
            if quoted:
                p_ids, p_offs = self._encode_cjk(piece, self.iso_offset)
            else:
                p_ids, p_offs = self._encode_cjk_words(piece)
            ids.extend(p_ids)
            offs.extend((rel + a, rel + b) for a, b in p_offs)
        return ids, offs

    def encode(self, text: str, max_length: int = 512) -> tuple[list[int], list[int]]:
        """Return (ids, attention_mask), eos-terminated and padded to max_length."""
        ids, mask, _ = self.encode_aligned(text, max_length)
        return ids, mask


def load_ext_assets(prefix: Path) -> tuple[torch.Tensor, dict]:
    """Load (table, mapping) written by build_ext.py from a path prefix.

    A pack that ships its ``iso`` record without the rows gets the block
    regenerated here (:func:`materialize_iso`), so every consumer sees the
    full table.
    """
    from safetensors.torch import load_file

    prefix = Path(prefix)
    table = load_file(str(prefix.with_suffix(".safetensors")))["ext_embed"]
    mapping = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    return materialize_iso(table, mapping), mapping
