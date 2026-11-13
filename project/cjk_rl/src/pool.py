"""P1's prompt pool (CPU, proposal § 3.5): the training set's captions with
their quoted OCR strings swapped for other OCR text, nothing else changed.

A template is any caption under ``post_image_dataset/resized`` with a
``Japanese text reads as`` (track ``dialogue``) or only a ``Japanese SFX reads
as`` clause (track ``sfx``, for § 5). The caption stays as it is (tags,
rating, position clauses, each text clause's position and string count); each
quoted string is replaced by one of about its length:

- dialogue: a Manga109-s text (``LINES["m109_pack"]``), or with probability
  ``--self_frac`` a dialogue string of *another* training caption; same
  length bin (``BINS``);
- SFX: a Manga109-s SFX crop's text (``manifest.parquet``, ``kind sfx``), or
  with ``--self_frac`` another caption's SFX string; same glyph count, else
  the nearest.

Strings are taken as written. Held out: the ruler's source images (as
templates) and every string sharing a 5-gram with a ruler string, so the P1
read stays held out. Captions with a tag in ``$CHARACTER_REF_EXCLUDE``
(``.env``, the character_ref miner's set) are left out too.

Writes ``output/cjk_rl/pool/pool.json``: per item the caption, ``texts`` (the
dialogue targets, in clause order), ``sfx``, their sources and the render
shape. The originals' text is not kept.

    .venv/bin/python project/cjk_rl/src/pool.py build [--variants 2]
    .venv/bin/python project/cjk_rl/src/pool.py show [-n 8]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_rl
sys.path.insert(0, str(LINE.parent / "cjk_anima_reseed"))

from reseed import REPO, bootstrap  # noqa: E402

bootstrap()

from eval.ruler import RULER  # noqa: E402

OUT = REPO / "output" / "cjk_rl" / "pool"
SEED = 1114
VARIANTS = 2
SELF_FRAC = 0.3
BINS = {"short": (0, 4), "mid": (5, 9), "long": (10, 20), "xlong": (21, 10**9)}
SFX_MANIFEST = "$MANGA109S/derived/manifest.parquet"


def bin_of(n: int) -> str:
    return next(b for b, (lo, hi) in BINS.items() if lo <= n <= hi)


def unquote(t: str) -> str:
    return t.strip().rstrip(".").strip('"').strip()


# ---------------------------------------------------------------- sources


def m109_text() -> list:
    from reseed.config import LINES, _expand

    src = Path(_expand(LINES["m109_pack"]))
    lines = (ln.split("\t")[0].strip() for ln in src.read_text("utf-8").splitlines())
    return [s for s in dict.fromkeys(lines) if s]


def m109_sfx() -> list:
    import pandas as pd

    from reseed.config import _expand

    src = Path(_expand(SFX_MANIFEST))
    d = pd.read_parquet(src, columns=["kind", "text"])
    return [s for s in dict.fromkeys(d.loc[d["kind"] == "sfx", "text"].dropna()) if s]


class Draw:
    """Cycles each key's list in a shuffled order, so the pool spreads over
    the strings before it repeats one."""

    def __init__(self, rng: random.Random):
        self.lists, self.rng, self.at = defaultdict(list), rng, {}

    def add(self, key, s: str, stems=frozenset()) -> None:
        self.lists[key].append((s, stems))

    def __call__(self, key, stem: str, taken=()) -> str | None:
        xs = self.lists.get(key) or []
        for _ in range(len(xs)):
            if self.at.get(key, len(xs)) >= len(xs):
                self.rng.shuffle(xs)
                self.at[key] = 0
            s, stems = xs[self.at[key]]
            self.at[key] += 1
            if s not in taken and stem not in stems:
                return s
        return None


# ---------------------------------------------------------------- templates


def scan() -> tuple[list, dict]:
    """(templates, skip counts). A template: (stem, caption file, parsed,
    {"text": [...], "sfx": [...]} its own strings)."""
    from anime_tools.captions.position_clauses import TEXT_PREFIXES, parse_caption

    from i2i_tasks.tools.character_ref_pairs import exclude_tags
    from library.env import load_dotenv

    from eval.ruler.build import CAPTIONS

    load_dotenv()

    ruler = json.loads((RULER / "ruler.json").read_text("utf-8"))["items"]
    held = {m["stem"] for m in ruler}
    excluded = exclude_tags()
    kind = {TEXT_PREFIXES[0]: "text", TEXT_PREFIXES[1]: "sfx"}
    out, skip = [], Counter()
    for f in sorted(CAPTIONS.rglob("*.txt")):
        if f.name.endswith(".variants.txt"):
            continue
        p = parse_caption(f.read_text(encoding="utf-8"))
        own: dict = {"text": [], "sfx": []}
        for cl in p.clauses:
            if cl.prefix in kind:
                own[kind[cl.prefix]] += [s for t in cl.tags if (s := unquote(t))]
        if not own["text"] and not own["sfx"]:
            continue
        if excluded.intersection(p.flat_tags):
            skip["excluded"] += 1
        elif f.stem in held:
            skip["ruler"] += 1
        else:
            out.append((f.stem, f, p, own))
    return out, skip


def rewrite(p, draw_text, draw_sfx) -> dict | None:
    """The caption with each quoted string redrawn; ``None`` if a slot finds
    no string."""
    from anime_tools.captions.position_clauses import (
        TEXT_PREFIXES,
        PositionClause,
        compose_caption,
        quote_text,
    )

    texts, srcs, sfx, sfx_srcs, clauses = [], [], [], [], []
    for cl in p.clauses:
        if cl.prefix not in TEXT_PREFIXES:
            clauses.append(cl)
            continue
        is_sfx = cl.prefix == TEXT_PREFIXES[1]
        out, out_src, draw = (sfx, sfx_srcs, draw_sfx) if is_sfx else (texts, srcs, draw_text)
        new = []
        for t in cl.tags:
            got = draw(unquote(t), taken=out)
            if got is None:
                return None
            new.append(got[0])
            out.append(got[0])
            out_src.append(got[1])
        tags = tuple(f'"{quote_text(s)}"' for s in new)
        clauses.append(PositionClause(position=cl.position, tags=tags, prefix=cl.prefix))
    return {
        "caption": compose_caption(list(p.flat_tags), clauses),
        "texts": texts,
        "src": srcs,
        "sfx": sfx,
        "sfx_src": sfx_srcs,
    }


# ---------------------------------------------------------------- build


def build(variants: int, self_frac: float) -> None:
    from eval.ruler.build import length, shape_of

    ruler = json.loads((RULER / "ruler.json").read_text("utf-8"))["items"]
    r5 = {m["text"][i : i + 5] for m in ruler for i in range(len(m["text"]) - 4)}

    def held(s: str) -> bool:
        return any(s[i : i + 5] in r5 for i in range(len(s) - 4))

    tmpls, skip = scan()
    print(f"{len(tmpls)} templates; held out {dict(skip)}", flush=True)

    rng = random.Random(SEED)
    pick = Draw(rng)
    own: dict = defaultdict(set)
    for st, _, _, strs in tmpls:
        for k, xs in strs.items():
            for s in xs:
                own[(k, s)].add(st)
    for (k, s), sts in own.items():
        if not held(s):
            key = bin_of(length(s)) if k == "text" else length(s)
            pick.add(("self", k, key), s, frozenset(sts))
    for s in m109_text():
        if not held(s):
            pick.add(("m109", "text", bin_of(length(s))), s)
    for s in m109_sfx():
        if not held(s):
            pick.add(("m109", "sfx", length(s)), s)
    for k, xs in sorted(pick.lists.items(), key=str):
        print(f"  {k[0]:<5} {k[1]:<4} {k[2]!s:<5} {len(xs):6d}")
    sfx_lens = sorted({k[2] for k in pick.lists if k[1] == "sfx"})

    items, short = [], Counter()
    for st, f, p, strs in tmpls:
        shape = shape_of(f)
        for v in range(variants):

            def order():
                return ("self", "m109") if rng.random() < self_frac else ("m109", "self")

            def draw_text(old, taken=()):
                for src in order():
                    if (s := pick((src, "text", bin_of(length(old))), st, taken)):
                        return s, src
                return None

            def draw_sfx(old, taken=()):
                n = length(old)
                for src in order():
                    for m in sorted(sfx_lens, key=lambda m: (abs(m - n), m)):
                        if (s := pick((src, "sfx", m), st, taken)):
                            return s, src
                return None

            it = rewrite(p, draw_text, draw_sfx)
            if it is None:
                short[st] += 1
                continue
            items.append(
                {
                    "i": len(items),
                    "stem": st,
                    "variant": v,
                    "track": "dialogue" if strs["text"] else "sfx",
                    "shape": shape,
                    **it,
                }
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pool.json").write_text(
        json.dumps(
            {"seed": SEED, "variants": variants, "self_frac": self_frac, "items": items},
            ensure_ascii=False,
            indent=1,
        ),
        "utf-8",
    )
    summary(items)
    if short:
        print(f"{len(short)} templates lost a variant (no string left)")
    print(f"→ {OUT / 'pool.json'}")


def summary(items: list) -> None:
    from eval.ruler.build import length

    print("items by track:", sorted(Counter(m["track"] for m in items).items()))
    nt = Counter(min(len(m["texts"]), 6) for m in items if m["track"] == "dialogue")
    print("dialogue strings an item (6 = 6+):", sorted(nt.items()))
    bins = Counter(bin_of(length(s)) for m in items for s in m["texts"])
    srcs = Counter(s for m in items for s in m["src"])
    print("dialogue strings by bin:", dict(bins), "by source:", dict(srcs))
    print(
        "sfx strings:",
        sum(len(m["sfx"]) for m in items),
        "by source:",
        dict(Counter(s for m in items for s in m["sfx_src"])),
    )


def show(n: int) -> None:
    items = json.loads((OUT / "pool.json").read_text("utf-8"))["items"]
    print(f"{len(items)} items; first {min(n, len(items))}:")
    for m in items[:n]:
        print(f"\n[{m['i']}] {m['stem']} v{m['variant']} {m['track']} {m['shape']}")
        print(f"  texts={m['texts']} src={m['src']} sfx={m['sfx']} sfx_src={m['sfx_src']}")
        print(f"  {m['caption']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--variants", type=int, default=VARIANTS)
    b.add_argument("--self_frac", type=float, default=SELF_FRAC)
    s = sub.add_parser("show")
    s.add_argument("-n", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.variants, a.self_frac)
    else:
        show(a.n)


if __name__ == "__main__":
    main()
