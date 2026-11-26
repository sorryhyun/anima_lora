#!/usr/bin/env python
"""Sentence strings — the draw of kana strings from the OCR reader's labels (CPU).

For ``src/s1.py``. Source: the reader's main labels (``OCR:`` rows of
``project/ocr_reader/core/crop_dataset.py::load_mainsub``, metric ``main``), as the
reader's rule normalises them (``normalize_target``, ``TARGET_NORM = 11``: a ``・`` run
arrives as ``…`` per three dots, ``・・・・・`` → ``……``). ``sent_strings_s1126`` was built
on norm 10, ``・`` runs as labelled; a draw on norm 11 cannot rebuild it. A label
becomes a string when, with its line breaks taken out,

- it holds no space, at least one kana and ``--lengths`` glyphs (a dot run one glyph,
  ``s1.units``);
- every glyph is a kana of the config's ``hira`` / ``kata``, a mark with a row (``ー``
  ``―`` ``。`` ``、`` ``~`` a dot run ``♥`` ``「`` ``」``) or ``!`` / ``?``; a blank heart is
  written as the filled one (``♡`` → ``♥``);
- it shares no 5-gram with a ruler string (``src/pool.py``'s rule);
- the quote's rows on the T5 side are the rows its glyphs name (the tokenizer run on the
  framed caption; a string that fails is dropped and counted).

``train``: the distinct strings of the train split, shuffled (``--seed``). ``val``: the
val split's strings whose glyph keys (``s1.keys``) no train label has, by length bin,
shuffled. Counted once here: strings a bin, the marks, the rows the first n train
strings hold.

Writes ``output/cjk_rl/pool/<name>.json`` (local: Manga109-s text is not redistributed).

    .venv/bin/python project/cjk_rl/probes/sent_strings.py --config project/cjk_rl/configs/<name>.toml --name sent_strings_s1126
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import tomllib
from collections import Counter
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LINE / "src"))
sys.path.insert(0, str(LINE / "probes"))

import glyph_g0 as G  # noqa: E402
import turbo_read as T  # noqa: E402  (the pack and its routing, the reseed path)

import s1  # noqa: E402

OUT = T.REPO / "output" / "cjk_rl" / "pool"
FREE = "!?"  # T5's own tokens, no row


def labels(split: str) -> list:
    from library.env import load_dotenv

    load_dotenv()
    if not os.environ.get("ANIMA_MANGA109S_ROOT") and os.environ.get("MANGA109S"):
        (root,) = glob.glob(os.path.expandvars("$MANGA109S/Manga109s_released_*"))
        os.environ["ANIMA_MANGA109S_ROOT"] = root
    sys.path.insert(0, str(T.REPO))
    from project.ocr_reader.core import crop_dataset as C

    df = C.load_mainsub(split)
    return [t for t in df[df.metric == "main"].target if isinstance(t, str)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="a config with hira and kata")
    ap.add_argument("--name", required=True, help="the file's stem")
    ap.add_argument("--seed", type=int, default=1126)
    ap.add_argument("--lengths", default="2-12")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    lo, hi = (int(x) for x in a.lengths.split("-"))
    kana = set(cfg["hira"]) | set(cfg["kata"])
    ok = kana | {"ー", "―", "。", "、", "~", "♥", "「", "」"} | set(FREE)

    from eval.ruler import RULER

    ruler = json.loads((RULER / "ruler.json").read_text("utf-8"))["items"]
    r5 = {m["text"][i : i + 5] for m in ruler for i in range(len(m["text"]) - 4)}

    def quote(t: str) -> str | None:
        lines = t.split("\n")
        if any(" " in x for x in lines):
            return None
        s = "".join(lines).replace("♡", "♥")
        us = s1.units(s)
        if not lo <= len(us) <= hi or not any(u in kana for u in us):
            return None
        if not all(u in ok or s1.DOT_RUN.fullmatch(u) for u in us):
            return None
        if any(s[i : i + 5] in r5 for i in range(len(s) - 4)):
            return None
        return s

    tr_raw, va_raw = labels("train"), labels("val")
    tr_keys = {tuple(s1.keys(t)) for t in tr_raw}
    train = sorted({q for t in tr_raw if (q := quote(t))})
    val = sorted(
        {q for t in va_raw if (q := quote(t)) and tuple(s1.keys(q)) not in tr_keys}
    )

    # the quote's rows on the T5 side against the rows its glyphs name
    from common.models import checkpoints
    from library.anima.ext_vocab import T5_TABLE_SIZE
    from library.inference.text import ensure_text_strategies, process_escape

    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)

    def ext(text: str) -> list:
        o = tok.tokenize([process_escape(G.framed("1girl, speech bubble", text))])
        ids = [int(v) for v, m in zip(o[2][0].tolist(), o[3][0].tolist()) if m]
        return [i - T5_TABLE_SIZE for i in ids if i >= T5_TABLE_SIZE]

    at = {}
    for c in sorted(kana):
        (at[s1.key(c)],) = ext(c)
    for m in s1.MARKS:
        (at[s1.key(m)],) = ext("あ" + m)[1:]
    assert len(set(at.values())) == len(at), "two glyphs on one row"

    def routed(s: str) -> bool:
        return set(ext(s)) == {at[k] for k in s1.keys(s) if k in at}

    bad = [s for s in train + val if not routed(s)]
    print(f"dropped, rows not as the glyphs name them: {len(bad)} {bad[:12]}")
    train = [s for s in train if s not in set(bad)]
    val = [s for s in val if s not in set(bad)]

    rng = random.Random(a.seed)
    rng.shuffle(train)
    by = {f"{x}-{y}": [] for x, y in s1.BINS}
    for s in val:
        if (b := s1.bin_of(len(s1.units(s)))) in by:
            by[b].append(s)
    for b in by.values():
        rng.shuffle(b)

    def count(xs: list) -> dict:
        c = Counter(s1.bin_of(len(s1.units(s))) for s in xs)
        return {k: c[k] for k in by}

    print(f"train labels {len(tr_raw)}, val labels {len(va_raw)}")
    print(
        f"train strings {len(train)} {count(train)}; val-only {len(val)} {count(val)}"
    )
    marks = Counter(k for s in train for k in set(s1.keys(s)) if k not in kana)
    print("train strings holding a mark:", dict(marks.most_common()))
    names = {v: k for k, v in at.items()}
    for n in (100, 400, 1000, 4000, len(train)):
        c = Counter(names[i] for s in train[:n] for i in ext(s))
        rows = [c.get(k, 0) for k in at]
        none = [k for k in at if not c.get(k)]
        print(
            f"first {n}: glyphs a string {sum(rows) / n:.2f}; rows held {len(at) - len(none)} "
            f"of {len(at)}; a row's renders ×4, median {4 * sorted(rows)[len(rows) // 2]}, "
            f"most {[(k, 4 * v) for k, v in c.most_common(6)]}; never held {''.join(none)}"
        )
    f = OUT / f"{a.name}.json"
    f.write_text(
        json.dumps(
            {"seed": a.seed, "lengths": [lo, hi], "train": train, "val": by},
            ensure_ascii=False,
            indent=0,
        ),
        "utf-8",
    )
    print(f"→ {f}")


if __name__ == "__main__":
    main()
