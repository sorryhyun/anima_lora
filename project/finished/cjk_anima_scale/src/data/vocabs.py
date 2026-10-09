"""Vocab specs: the one token-list surface for what the ext table trains on
(a run's ``vocabs``, ``cjk_scale/config.py::RunConfig.vocab_specs``).

One source per spec::

    kana                  92 basic kana                     → single
    kana_ext              68 voiced / handakuten / small    → single_ext, ×2
    small                 the 18 small kana, inside digraphs → single_small
    kanji:200             top-200 single-row corpus kanji   → single_kanji, ×2
    words:100/held=8      top-100 single-piece corpus words → word / word_held
    chars:あかす出人日      a literal vocab list              → single
    list:、,。,！！         literal multi-char vocabs         → single_extra, ×2
    list:、,@ja_cold.txt   … and / or one vocab per line of  → single_extra, ×2
                          assets/vocabs/ja_cold.txt (# = comment)

``*W`` overrides the source's weight in the S-line singles pool and ``/held=K``
holds K of its vocabs out of every training item. No spec at all means
``kana``. ``small`` draws each small kana inside ``SMALL_PER`` two-glyph
digraphs (あっ きゃ ニャ — a lone ゃ renders full-size, so it cannot be a single)
whose Qwen pieces are the host row + the small row; every small kana gets the
pool mass of one vocab of its weight. ``kana`` and ``chars:`` both feed the base
inventory (combos, random strings and corpus-line filtering draw from it); a
``chars:`` base with no ``kana`` alongside it is the textual-inversion regime —
few vocabs, many exposures — and scores its own vocabs as the ``single`` group.

**Spec order is not data order.** The singles pool is built in the canonical
order below (base → ext → kanji → words → list), so a recipe rebuilds its data
dir byte for byte whatever order its sources were typed in — the bit-identity
contract in ``data/stage.py`` depends on that pool order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from common.text import KANA_SMALL

# canonical pool order (NOT the typed order). `kana` / `chars` are both the
# base inventory — combos, strings and corpus-line filtering draw from it.
KINDS = ("kana", "chars", "kana_ext", "small", "kanji", "words", "list")
# kinds resolved against the Qwen tokenizer + the training corpus (they take a
# count, not a literal vocab list)
CORPUS_KINDS = ("kanji", "words")
NEEDS_ARG = ("kanji", "words", "chars", "list")

# `list:@<file>` inventories (a path with a separator is taken as typed)
VOCABS_DIR = Path(__file__).resolve().parents[2] / "assets" / "vocabs"

# default draws per vocab in the S-line singles pool (`*W` overrides)
WEIGHT = {
    "kana": 1,
    "chars": 1,
    "kana_ext": 2,
    "small": 1,
    "kanji": 2,
    "words": 1,
    "list": 2,
}
# digraphs per small kana in the singles pool (the `small` spec)
SMALL_PER = 6
# the kinds that take ``/held=K`` (→ their held eval group)
HELD_GROUP = {"words": "word_held"}

_SPEC_RE = re.compile(
    r"^(?P<kind>[a-z_]+)"
    r"(?::(?P<arg>.*?))?"
    r"(?:\*(?P<weight>\d+))?"
    r"(?:/held=(?P<held>\d+))?$"
)


@dataclass
class VocabSource:
    """One vocab spec, and the vocabs it resolved to."""

    kind: str
    arg: str = ""
    weight: int = 1
    held: int = 0
    vocabs: list = field(default_factory=list)
    held_vocabs: list = field(default_factory=list)
    # (vocab, corpus count) for the corpus-derived kinds; written to words.json
    # / kanji.json so a table's provenance is readable off the data dir
    freq: list = field(default_factory=list)

    @property
    def n(self) -> int:
        """The numeric argument (``kanji:200`` → 200); 0 when absent."""
        return int(self.arg) if self.arg else 0

    def spec(self) -> str:
        """The spec as typed, for the run record."""
        s = self.kind + (f":{self.arg}" if self.arg else "")
        if self.weight != WEIGHT[self.kind]:
            s += f"*{self.weight}"
        return s + (f"/held={self.held}" if self.held else "")


def parse_vocabs(specs) -> list[VocabSource]:
    """Vocab specs → sources in canonical order (stable within a kind)."""
    out = []
    for spec in specs or ["kana"]:
        out.append(_parse_one(spec))
    order = {k: i for i, k in enumerate(KINDS)}
    out.sort(key=lambda s: order[s.kind])
    return out


def _parse_one(spec: str) -> VocabSource:
    s = spec.strip()
    assert s, "vocab spec: empty"
    m = _SPEC_RE.match(s)
    assert m, f"vocab spec {spec!r}: expected kind[:arg][*weight][/held=K]"
    kind = m.group("kind")
    assert kind in KINDS, (
        f"vocab spec {spec!r}: unknown kind {kind!r} (kinds: {', '.join(KINDS)})"
    )
    arg = (m.group("arg") or "").strip()
    assert arg or kind not in NEEDS_ARG, f"vocab spec {spec!r}: {kind} needs an argument"
    assert not arg or kind in NEEDS_ARG, f"vocab spec {spec!r}: {kind} takes no argument"
    if kind in CORPUS_KINDS:
        assert arg.isdigit(), f"vocab spec {spec!r}: {kind} takes a count, not {arg!r}"
    src = VocabSource(
        kind=kind,
        arg=arg,
        weight=int(m.group("weight")) if m.group("weight") else WEIGHT[kind],
        held=int(m.group("held") or 0),
    )
    assert not src.held or kind in HELD_GROUP, (
        f"vocab spec {spec!r}: /held= is only defined for {', '.join(HELD_GROUP)}"
    )
    if kind == "chars":
        src.vocabs = list(arg)
    elif kind == "list":
        toks = [u for u in arg.split(",") if u]
        if any(t.startswith("@") for t in toks):  # literals and files, in order
            us = [u for t in toks for u in (_list_file(t[1:]) if t[0] == "@" else [t])]
            src.vocabs = list(dict.fromkeys(us))
        else:
            src.vocabs = toks
        assert src.vocabs, f"vocab spec {spec!r}: empty vocab list"
    return src


def _list_file(name: str) -> list:
    """``list:@<file>``: one vocab per line, first tab-separated column; blank
    lines and ``#`` lines skipped."""
    path = Path(name) if "/" in name else VOCABS_DIR / name
    assert path.is_file(), f"vocab spec list:@{name}: no such file {path}"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.split("\t")[0].strip() for ln in lines if ln.strip() and ln[0] != "#"]


@dataclass
class Inventory:
    """Everything a training item may contain, and the eval strings drawn.

    The named fields are the canonical pool order: a source's vocabs land in the
    field its kind feeds, and ``pool()`` reads them back in that order.
    """

    sources: list = field(default_factory=list)
    # base inventory (`kana` / `chars`): combos, strings and corpus lines draw
    # from it; `restricted` is the old --only_chars (a hand-picked base)
    kana: list = field(default_factory=list)
    restricted: bool = False
    kana_ext: list = field(default_factory=list)
    # `small`: small kana → its SMALL_PER digraphs (cycled when fewer split)
    small_of: dict = field(default_factory=dict)
    kanji: list = field(default_factory=list)
    words: list = field(default_factory=list)
    words_held: list = field(default_factory=list)
    words_train: list = field(default_factory=list)
    # rows a phrase file needs beyond the singles inventory (--phrase_pieces):
    # trained through the phrases only, never drawn as singles / evals
    phrase_pieces: list = field(default_factory=list)
    # `list:` vocabs — drawn as singles like kana (punctuation arm, 2026-09-16)
    extra: list = field(default_factory=list)
    # word mode: a string is usable only when every piece is a trained row
    piece_ok: Callable[[str], bool] | None = None
    evals: dict = field(default_factory=dict)  # group → [text]

    def source(self, kind: str) -> VocabSource | None:
        for s in self.sources:
            if s.kind == kind:
                return s
        return None

    def has(self, kind: str) -> bool:
        return self.source(kind) is not None

    def weight(self, kind: str) -> int:
        s = self.source(kind)
        return s.weight if s else WEIGHT[kind]

    def pool(self) -> list:
        """The S-line singles pool — every trained vocab repeated by its source
        weight, in canonical order.

        Small kana are dropped from the ``kana_ext`` draw: a lone ゃ renders
        full-size, so it is not a singles concept (P0b); the S line shows them
        inside words / phrases only — or, with a ``small`` source, inside its
        digraphs: a small kana's ``SMALL_PER`` digraphs share the mass of one
        vocab, so every other vocab is repeated ``SMALL_PER`` times over.
        """
        ext_ns = [c for c in self.kana_ext if c not in KANA_SMALL]
        m = SMALL_PER if self.small_of else 1
        small = [d for ds in self.small_of.values() for d in ds]
        return (
            self.kana * (m * self.weight("kana" if self.has("kana") else "chars"))
            + ext_ns * (m * self.weight("kana_ext"))
            + small * self.weight("small")
            + self.kanji * (m * self.weight("kanji"))
            + self.words_train * (m * self.weight("words"))
            + self.extra * (m * self.weight("list"))
        )

    def describe(self) -> str:
        parts = [s.spec() for s in self.sources]
        return " ".join(parts) if parts else "(none)"
