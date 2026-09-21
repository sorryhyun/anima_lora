"""What the ext rows stand for: Qwen pieces → pack rows, corpus inventories.

CPU-only (tokenizer + pack mapping, no model weights).
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from itertools import permutations
from pathlib import Path

from common.models import checkpoints
from common.paths import CORPUS_TRAIN
from common.text import KANA, KANA_RE, KANA_SMALL, KANJI_RE, WORD_RE


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


_ELLIPSIS_RE = re.compile(r"[・･]{2,}|…+|‥+|\.{3,}")
_BANG_RE = re.compile(r"[！!]{2,}")


def norm_phrase(text: str) -> str:
    """``--phrase_norm``: every ellipsis spelling (・・ / ・・・・ / ･･･ / … / ‥ / ...)
    → ``・・・`` and every run of bangs → ``！！`` — one warm row each instead of
    a spelling per book (user, 2026-09-20: the top sentence strings were all
    ・・・・ lines)."""
    return _BANG_RE.sub("！！", _ELLIPSIS_RE.sub("・・・", text))


def phrase_file_lines(
    path: Path, min_pieces: int, max_pieces: int, norm: bool = False
) -> list:
    """``[(line, book, n_pieces)]`` from a phrase TSV (``line[\\tbook[\\tn]]``,
    one per row; a bare line gets book ``""`` and its count is taken from the
    tokenizer lazily by the caller when absent). Lines outside
    ``[min_pieces, max_pieces]`` are dropped when the count column is present."""
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        cols = ln.rstrip("\n").split("\t")
        text = cols[0].strip()
        book = cols[1].strip() if len(cols) > 1 else ""
        n = int(cols[2]) if len(cols) > 2 and cols[2].strip() else None
        if n is not None and not (min_pieces <= n <= max_pieces):
            continue
        if norm and norm_phrase(text) != text:
            # the file's piece count is the raw spelling's: the caller recounts
            text, n = norm_phrase(text), None
        out.append((text, book, n))
    if norm:
        # spellings that merged: keep the first (its book decides held / train)
        seen, uniq = set(), []
        for t in out:
            if t[0] not in seen:
                seen.add(t[0])
                uniq.append(t)
        out = uniq
    return out


def phrase_pieces(tok, q, lines: list, covered: set, n: int) -> list:
    """The ``n`` most frequent pieces (each with a pack row) over ``lines``
    that are not already in ``covered`` — the rows a phrase set needs beyond
    the singles inventory. Returns ``[(piece, count)]``."""
    cnt: Counter = Counter()
    for text, _book, _n in lines:
        for p, row in pieces(tok, q, text):
            if row is not None and p not in covered:
                cnt[p] += 1
    return cnt.most_common(n)


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


# standard yōon / gairaigo digraphs, tried after the corpus-attested ones
_YOON = [h + s for h in "きしちにひみりぎじびぴ" for s in "ゃゅょ"] + [
    h + s for h in "キシチニヒミリギジビピ" for s in "ャュョ"
]
_GAIRAIGO = (
    "ファ フィ フェ フォ ウィ ウェ ウォ ティ ディ トゥ ドゥ チェ シェ ジェ デュ テュ "
    "フュ ツァ ツィ ツェ ツォ クァ クィ クェ クォ グァ イェ"
).split()


def small_digraphs(tok, q, hosts, per: int) -> dict:
    """``--units small``: small kana → ``per`` two-glyph digraphs (host + small;
    っ / ッ also small + host) whose Qwen pieces are exactly the two glyphs,
    each with an ext row, the host a trained single. Corpus-attested digraphs
    (>= 3 bubbles) first, by count, then the yōon / gairaigo tables, then the
    rarer corpus ones; a small kana with
    fewer than ``per`` cycles its list so every one carries ``per`` entries.
    Most frequent uses (って ちゃ った じゃ) are one Qwen piece — word rows, not
    these."""
    hosts = set(hosts) - set(KANA_SMALL)
    cnt: Counter = Counter()
    for ln in (CORPUS_TRAIN / "boxes.jsonl").read_text().splitlines():
        for b in json.loads(ln)["bubbles"]:
            t = b["line"]
            for i in range(len(t) - 1):
                a, c = t[i], t[i + 1]
                if (a in hosts and c in KANA_SMALL) or (a in "っッ" and c in hosts):
                    cnt[a + c] += 1
    # one- and two-bubble digraphs are mostly OCR noise (おゃ うゅ ナュ): they
    # rank after the tables
    seen = cnt.most_common()
    ranked = (
        [d for d, n in seen if n >= 3]
        + _YOON
        + _GAIRAIGO
        + [d for d, n in seen if n < 3]
    )
    out: dict = {s: [] for s in KANA_SMALL}
    for d in dict.fromkeys(ranked):
        ps = pieces(tok, q, d)
        if [p for p, _ in ps] != list(d) or any(r is None for _, r in ps):
            continue
        host, small = (d[1], d[0]) if d[0] in "っッ" and d[1] in hosts else (d[0], d[1])
        if host in hosts and small in out and len(out[small]) < per:
            out[small].append(d)
    missing = [s for s, ds in out.items() if not ds]
    assert not missing, f"--units small: no splitting digraph for {missing}"
    return {s: [ds[i % len(ds)] for i in range(per)] for s, ds in out.items()}


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
