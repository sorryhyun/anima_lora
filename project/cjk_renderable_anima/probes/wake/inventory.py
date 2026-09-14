"""What the ext rows stand for: Qwen pieces → pack rows, corpus inventories.

CPU-only (tokenizer + pack mapping, no model weights).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from itertools import permutations
from pathlib import Path

from .common import CORPUS_TRAIN, KANA, KANA_RE, KANJI_RE, WORD_RE
from .models import checkpoints


def corpus_lines(boxes_jsonl: Path, max_len: int):
    """Kana-only bubble lines of 1..max_len chars → ``[(line, rel, box)]``."""
    out = []
    for ln in boxes_jsonl.read_text().splitlines():
        r = json.loads(ln)
        for b in r["bubbles"]:
            t = b["line"]
            if (
                KANA_RE.match(t)
                and 1 <= len(t) <= max_len
                and any(c in KANA for c in t)
            ):
                out.append((t, r["rel"], b["box"]))
    return out


def qwen_pieces():
    """(Qwen3 tokenizer, qwen id → pack ext row). The pack's ext rows are
    Qwen pieces, many of them whole words (ありがとう / 行く / 明日 are one
    piece → one row), so a "word address" is an existing row."""
    from library.anima.vocab_pack import VocabPack, resolve_pack_prefix
    from library.anima.weights import load_qwen3_tokenizer

    ck = checkpoints()
    tok = load_qwen3_tokenizer(ck.text_encoder)
    pack = VocabPack.load(resolve_pack_prefix(ck.vocab_pack))
    q = {int(k): int(v) for k, v in pack.mapping["qwen"].items()}
    return tok, q


def pieces(tok, q, text: str):
    """text → [(piece text, ext row or None)] on the Qwen side."""
    out = []
    for i in tok.encode(text, add_special_tokens=False):
        out.append((tok.decode([i]), q.get(int(i))))
    return out


def word_inventory(tok, q, n: int, min_len: int = 2):
    """The ``n`` most frequent multi-char single-piece words in the training
    corpus bubbles (piece frequency over every line, not the length-capped
    subset) plus their counts."""
    cnt: Counter = Counter()
    for ln in (CORPUS_TRAIN / "boxes.jsonl").read_text().splitlines():
        r = json.loads(ln)
        for b in r["bubbles"]:
            for p, row in pieces(tok, q, b["line"]):
                if row is not None and len(p) >= min_len and WORD_RE.match(p):
                    cnt[p] += 1
    return cnt.most_common(n)


def kanji_inventory(tok, q, n: int):
    """P0b (2026-09-14): the ``n`` most frequent kanji in the training corpus
    bubbles that are each exactly one Qwen piece with its own pack row, plus
    their character counts (frequency, not school grade: the rows the corpus
    actually uses, and the widest base for kanji-bearing words later)."""
    cnt: Counter = Counter()
    for ln in (CORPUS_TRAIN / "boxes.jsonl").read_text().splitlines():
        for b in json.loads(ln)["bubbles"]:
            cnt.update(c for c in b["line"] if KANJI_RE.match(c))
    out = []
    for c, k in cnt.most_common():
        ps = pieces(tok, q, c)
        if len(ps) == 1 and ps[0][0] == c and ps[0][1] is not None:
            out.append((c, k))
            if len(out) >= n:
                break
    return out


def tokenizes_clean(tok, q, s: str, rows: dict | None = None) -> bool:
    """Every permutation of ``s`` tokenizes to exactly its own single-char
    pieces, each with a pack row (``rows`` given: that char's row) — so an
    order flip is a pure order contrast, never a merged word piece."""
    for perm in permutations(s):
        ps = pieces(tok, q, "".join(perm))
        if len(ps) != len(s):
            return False
        for (p, r), c in zip(ps, perm):
            if p != c or (r is None if rows is None else r != rows[c]):
                return False
    return True


def clean_kana_strings(
    tok,
    qmap,
    kana: list,
    rng: random.Random,
    n: int,
    k: int,
    excl=(),
    rows: dict | None = None,
    skip_reversed: bool = True,
):
    """``n`` clean (see ``tokenizes_clean``) strings of ``k`` distinct kana,
    none in ``excl`` (nor, with ``skip_reversed``, the reverse of one)."""
    out, seen = [], set(excl)
    tries = 0
    while len(out) < n and tries < 20_000:
        tries += 1
        s = "".join(rng.sample(kana, k))
        if s in seen or (skip_reversed and s[::-1] in seen):
            continue
        if tokenizes_clean(tok, qmap, s, rows):
            out.append(s)
            seen.add(s)
    return out
