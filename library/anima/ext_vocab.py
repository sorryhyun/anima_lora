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

Pure-CPU module — no model load; consumers pass embedding tensors in.
"""

from __future__ import annotations

import json
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


def segment_runs(text: str) -> list[tuple[str, str]]:
    """Split text into ("t5"|"cjk", span) runs.

    Whitespace immediately preceding a CJK run is folded into it (Qwen's
    byte-BPE handles leading spaces natively; T5 spiece would just emit a
    bare ``▁``).
    """
    runs: list[tuple[str, str]] = []
    buf, kind = [], None
    for ch in text:
        k = "cjk" if is_cjk_char(ch) else "t5"
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


def collect_clean_qwen_tokens(qwen_tok) -> dict[int, str]:
    """Qwen token ids whose decoded surface is pure CJK (spaces allowed)."""
    n = qwen_tok.vocab_size
    surfaces = qwen_tok.batch_decode([[i] for i in range(n)])
    clean = {}
    for i, s in enumerate(surfaces):
        core = s.strip()
        if core and "�" not in s and all(is_cjk_char(c) or c == " " for c in s):
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


def fit_anchor_map(
    qwen_embed: torch.Tensor,
    t5_embed: torch.Tensor,
    t5_ids: list[int],
    qwen_ids: list[int],
    ridge: float = 1e-2,
    holdout: int = 1000,
) -> tuple[torch.Tensor, float]:
    """Ridge least-squares W (d_qwen × d_t5) with a held-out cosine sanity.

    Returns (W, mean held-out cosine of ``qwen_embed @ W`` vs the true T5 row).
    """
    A = qwen_embed[qwen_ids].float()
    B = t5_embed[t5_ids].float()
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(t5_ids), generator=g)
    hold, train = perm[:holdout], perm[holdout:]
    At, Bt = A[train], B[train]
    d = At.shape[1]
    lam = ridge * At.pow(2).mean() * len(train)
    W = torch.linalg.solve(At.T @ At + lam * torch.eye(d), At.T @ Bt)
    cos = torch.nn.functional.cosine_similarity(A[hold] @ W, B[hold], dim=-1)
    return W, float(cos.mean())


def build_ext_table(
    qwen_tok,
    qwen_embed: torch.Tensor,
    W: torch.Tensor,
    clean: dict[int, str],
) -> tuple[torch.Tensor, dict]:
    """Extended rows + id mapping.

    Rows = every clean Qwen CJK token, plus per-character supplementary rows
    for standard-range chars whose Qwen tokenization is byte-fragments only
    (init = mapped mean of fragment embeddings; chars sharing a fragment-id
    multiset with another char get a position-weighted mean so permuted byte
    orders cannot land on bit-identical rows).
    """
    qwen_map: dict[int, int] = {}
    vecs: list[torch.Tensor] = []
    for qid in sorted(clean):
        qwen_map[qid] = len(vecs)
        vecs.append(qwen_embed[qid].float() @ W)

    single_clean = {s: qid for qid, s in clean.items() if len(s) == 1}
    char_ids: dict[str, list[int]] = {}
    for lo, hi in _CJK_RANGES:
        for o in range(lo, hi + 1):
            ch = chr(o)
            if ch in single_clean:
                continue
            ids = qwen_tok(ch, add_special_tokens=False)["input_ids"]
            if not ids or qwen_tok.decode(ids) != ch:
                continue
            char_ids[ch] = ids

    by_multiset: dict[tuple[int, ...], int] = {}
    for ids in char_ids.values():
        key = tuple(sorted(ids))
        by_multiset[key] = by_multiset.get(key, 0) + 1

    char_map: dict[str, int] = {}
    for ch, ids in char_ids.items():
        emb = qwen_embed[ids].float()
        if by_multiset[tuple(sorted(ids))] > 1:
            # Same byte multiset as another char — order is the only
            # distinguishing signal, so pool with position weights 1..n.
            w = torch.arange(1, len(ids) + 1, dtype=torch.float32)
            pooled = (emb * w.unsqueeze(1)).sum(dim=0) / w.sum()
        else:
            pooled = emb.mean(dim=0)
        char_map[ch] = len(vecs)
        vecs.append(pooled @ W)

    table = torch.stack(vecs) if vecs else torch.zeros(0, W.shape[1])
    mapping = {
        "qwen": {str(k): v for k, v in qwen_map.items()},
        "char": char_map,
        "rows": table.shape[0],
    }
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
    # surface → base-vocab t5 id sequence (the C fallback: substitute a known
    # CJK surface with the EN tag's stock spiece tokens at encode time — the
    # pretrained rows carry the identity no minted row learns; zero training).
    word_sub: dict[str, list[int]] | None = None

    @classmethod
    def from_mapping(cls, t5_tok, qwen_tok, mapping: dict) -> "HybridT5Encoder":
        qwen_map = {int(k): v for k, v in mapping["qwen"].items()}
        # Chars that are clean single tokens were excluded from char_map (no
        # separate row needed) — but they can still arrive as byte-fragments
        # mid-word, so the char lookup must cover them via their token row.
        char_map = dict(mapping["char"])
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
        )

    def _encode_cjk(self, span: str) -> tuple[list[int], list[tuple[int, int]]]:
        """(ids, char offsets into ``span``) for one CJK run."""
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
                    out.append(T5_TABLE_SIZE + self.char_map[ch])
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
                out.append(T5_TABLE_SIZE + self.qwen_map[qid])
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
        for kind, span in segment_runs(text):
            if kind == "cjk":
                s_ids, s_offs = self._encode_cjk_words(span)
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

    def encode(self, text: str, max_length: int = 512) -> tuple[list[int], list[int]]:
        """Return (ids, attention_mask), eos-terminated and padded to max_length."""
        ids, mask, _ = self.encode_aligned(text, max_length)
        return ids, mask


def load_ext_assets(prefix: Path) -> tuple[torch.Tensor, dict]:
    """Load (table, mapping) written by build_ext.py from a path prefix."""
    from safetensors.torch import load_file

    table = load_file(str(prefix.with_suffix(".safetensors")))["ext_embed"]
    mapping = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    return table, mapping
