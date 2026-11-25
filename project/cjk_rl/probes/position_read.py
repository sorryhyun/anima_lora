#!/usr/bin/env python
"""position_read — where in the asked string the ruler's renders lose it (CPU).

Reads the dialogue ruler's reads of record (the ``renders.json`` of the reseed line's
ruler results: each record holds its string, bin and box reads; nothing is rendered or
read again) and aligns each asked string to the best read of one bubble. The strings
come from the records, not from ``ruler.json``: the string set was rebuilt on 11-13
(19 of 96 strings kept), so the sensitive reads and the captioned reads are two sets.

    candidates  per bubble (``reward.bubbles``): each box, and the bubble's boxes
                joined in column and in line order; no whole-page read
    pick        the candidate with the largest LCS / max(asked, read) (``r_seq``'s
                rule), the smaller edit distance on a tie
    align       the Levenshtein backtrace: each asked glyph is a hit, a substitution
                or a deletion; read glyphs with no asked glyph are insertions
    cer         apart from the pick: the smallest edit distance over the candidates

Tables, per arm and reader key (``sfx``: the tuned reader of the read's date; ``vl``:
stock VL16, on the reads that kept it):

    position    hit rate by third of the string, at the first / last glyph and by
                index; per string, the head third against the tail third (sign test)
    runs        hit rate after a hit against after a miss
    length      read length over asked length, deletions and insertions a string,
                insertions that repeat a neighbour
    class       hit rate by glyph class (hiragana, voiced, small, katakana, ー, kanji)
    subs        substitution pairs by kind (same kana in the other script, voicing,
                size, kanji → kanji / kana, …) and the commonest pairs
    glyph       hit rate per glyph over the jp_v1 arms
    en          the EN references' own text read against their line, by T5 token
                count, beside the JA strings at the same count

The null for every rate: the same string aligned to the page of another string of its
bin (three shifts), so a rate is read against what the pick finds on a page that was
not asked for it.

    .venv/bin/python project/cjk_rl/probes/position_read.py
"""

from __future__ import annotations

import json
import math
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_rl
sys.path.insert(0, str(LINE.parent / "cjk_anima_reseed" / "src"))
sys.path.insert(0, str(LINE / "src"))

from common.text import lev, norm  # noqa: E402
from reward import bubbles, lcs  # noqa: E402

from library.env import resolve_under_home  # noqa: E402

RULER = resolve_under_home("output/cjk_anima_reseed/ruler")
RESULTS = LINE.parent / "cjk_anima_reseed" / "results"
R1006 = "20261006-1502-ruler-sensitive-glyph_f1"
R1008 = "20261008-2025-ruler-sensitive-seed_1008"
R1113 = "20261113-1417-ruler-sensitive-turbo_m14v3_only64"
R1113C = "20261113-2057-ruler-captioned-rl_p1_only64"
# (label, result dir, arm). The tables of record on the base (sensitive prompts, 96
# strings, seed 0), oldest first; then jp_v1's rows on the soup and the turbo (64 of
# the 96); then the captioned set (64 strings of the rebuilt ruler) on the turbo.
TABLES = (
    ("retrain_kana", R1008, "retrain_kana"),
    ("seed_retrain_0930", R1008, "seed_retrain_0930"),
    ("preview51", R1008, "seed_fixed_1005_stick080@punct"),
    ("sent_whole", R1006, "sent_whole"),
    ("sent_kanji_f0", R1008, "sent_kanji_f0"),
    ("sent_kanji_pres", R1008, "sent_kanji_pres"),
    ("jp_v1", R1008, "seed_1008"),
)
SOUP = (
    ("jp_v1 (the 64)", R1113, "seed_1008"),
    ("jp_v1#m14", R1113, "seed_1008#m14"),
    ("jp_v1#m14v3", R1113, "seed_1008#m14v3"),
)
CAPTIONED = (
    ("cap jp_v1#m14v3", R1113C, "seed_1008#m14v3"),
    ("cap rl_p1#m14v3", R1113C, "rl_p1#m14v3"),
)
GLYPH_ARMS = ("jp_v1", "jp_v1#m14", "jp_v1#m14v3", "cap jp_v1#m14v3")
# the EN references: (label, the record that names the strings, the reads' dir)
EN_ARMS = (
    ("en", "jp_v1", "sensitive/en"),
    ("en#m14", "jp_v1#m14", "sensitive/en#m14"),
    ("en#m14v3", "jp_v1#m14v3", "sensitive/en#m14v3"),
)
NULL_SHIFTS = (1, 2, 3)
TOK_BINS = ((1, 2), (3, 4), (5, 9), (10, 20), (21, 99))


def sign_p(up: int, dn: int) -> float:
    n, k = up + dn, min(up, dn)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def cands(reads: list, key: str) -> list:
    boxes = [r for r in reads if not r.get("whole") and r.get(key)]
    cx = lambda r: (r["box"][0] + r["box"][2]) / 2  # noqa: E731
    out = []
    for g in bubbles(boxes):
        out += [norm(r[key]) for r in g]
        if len(g) > 1:
            for k in (
                lambda r: (-cx(r), r["box"][1]),
                lambda r: (r["box"][1], r["box"][0]),
            ):
                out.append("".join(norm(r[key]) for r in sorted(g, key=k)))
    return [c for c in out if c]


def align(t: str, c: str) -> tuple[list, list]:
    """``ops[j]`` per asked glyph: ``("M",)``, ``("S", read)`` or ``("D",)``;
    ``ins``: ``(j, read)``, a read glyph sitting before asked index ``j``."""
    n, m = len(t), len(c)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j - 1] + (t[i - 1] != c[j - 1]),
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
            )
    ops, ins = [None] * n, []
    i, j = n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (t[i - 1] != c[j - 1]):
            ops[i - 1] = ("M",) if t[i - 1] == c[j - 1] else ("S", c[j - 1])
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops[i - 1] = ("D",)
            i -= 1
        else:
            ins.append((i, c[j - 1]))
            j -= 1
    return ops, ins


def read_string(text: str, reads: list, key: str) -> dict:
    t = norm(text)
    cs = cands(reads, key)
    best = max(
        cs, key=lambda c: (lcs(t, c) / max(len(t), len(c), 1), -lev(c, t)), default=""
    )
    ops, ins = align(t, best)
    d = min((lev(c, t) for c in cs), default=len(t))
    return {"t": t, "best": best, "ops": ops, "ins": ins, "lev": d}


def third(j: int, n: int) -> int:
    return min(2, int(3 * (j + 0.5) / n))


def base_of(c: str) -> str:
    return unicodedata.normalize("NFD", c)[0]


SMALL = dict(
    zip(
        "ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮ",
        "あいうえおつやゆよわアイウエオツヤユヨワ",
    )
)


def hira(c: str) -> str:
    return chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c


def cls(c: str) -> str:
    if c == "ー":
        return "ー"
    if "一" <= c <= "鿿":
        return "kanji"
    if c in SMALL:
        return "small kana"
    if "ぁ" <= c <= "ゟ":
        return "hira voiced" if base_of(c) != c else "hiragana"
    if "ァ" <= c <= "ヿ":
        return "kata voiced" if base_of(c) != c else "katakana"
    return "other"


def sub_kind(a: str, b: str) -> str:
    ka, kb = cls(a), cls(b)
    kana = lambda k: k not in ("kanji", "other", "ー")  # noqa: E731
    if kana(ka) and kana(kb):
        ha, hb = hira(a), hira(b)
        if ha == hb:
            return "same kana, other script"
        if base_of(ha) == base_of(hb):
            return "voicing"
        if SMALL.get(ha, ha) == SMALL.get(hb, hb):
            return "size"
        return "kana → other kana"
    if ka == "kanji":
        return "kanji → kanji" if kb == "kanji" else "kanji → kana / other"
    if kb == "kanji":
        return "kana → kanji"
    return "other"


_RECORDS: dict = {}


def records(run: str, arm: str) -> list:
    """One arm's records of a ruler result: ``i``, ``bin``, ``text``, ``reads``."""
    if run not in _RECORDS:
        _RECORDS[run] = json.loads((RESULTS / run / "renders.json").read_text("utf-8"))
    return [
        {"i": r["i"], "bin": r["bin"], "text": r["text"], "reads": r["reads"]}
        for r in _RECORDS[run][arm]
    ]


def tables(recs: list) -> dict:
    """``recs``: ``read_string`` outputs with ``bin``. Every rate with its n."""
    pos = defaultdict(lambda: [0, 0])  # (bin, slot) -> [hit, n]
    idx = defaultdict(lambda: [0, 0])
    run = defaultdict(lambda: [0, 0])
    kind = defaultdict(lambda: [0, 0])
    head_tail = defaultdict(lambda: [0, 0])  # bin -> [head > tail, tail > head]
    length = defaultdict(lambda: Counter())
    subs, glyph = Counter(), defaultdict(lambda: [0, 0])
    for r in recs:
        t, ops, b = r["t"], r["ops"], r["bin"]
        n = len(t)
        hit = [o[0] == "M" for o in ops]
        thirds = [[], [], []]
        for j, h in enumerate(hit):
            for key in ((b, f"third{third(j, n)}"), ("all", f"third{third(j, n)}")):
                pos[key][0] += h
                pos[key][1] += 1
            thirds[third(j, n)].append(h)
            if n >= 5:
                idx[min(j, 14)][0] += h
                idx[min(j, 14)][1] += 1
            if j:
                run["after hit" if hit[j - 1] else "after miss"][0] += h
                run["after hit" if hit[j - 1] else "after miss"][1] += 1
            kind[cls(t[j])][0] += h
            kind[cls(t[j])][1] += 1
            glyph[t[j]][0] += h
            glyph[t[j]][1] += 1
            if ops[j][0] == "S":
                subs[(t[j], ops[j][1])] += 1
        for key, j in (("first", 0), ("last", n - 1)):
            for g in (b, "all"):
                pos[(g, key)][0] += hit[j]
                pos[(g, key)][1] += 1
        if thirds[0] and thirds[2]:
            h, tl = sum(thirds[0]) / len(thirds[0]), sum(thirds[2]) / len(thirds[2])
            for g in (b, "all"):
                head_tail[g][0] += h > tl
                head_tail[g][1] += tl > h
        rep = sum(
            1 for j, c in r["ins"] if (j < n and t[j] == c) or (j and t[j - 1] == c)
        )
        for g in (b, "all"):
            L = length[g]
            L["strings"] += 1
            L["asked"] += n
            L["read"] += len(r["best"])
            L["del"] += sum(o[0] == "D" for o in ops)
            L["sub"] += sum(o[0] == "S" for o in ops)
            L["ins"] += len(r["ins"])
            L["ins_repeat"] += rep
            L["no_read"] += not r["best"]
    rate = lambda d: {  # noqa: E731
        (" | ".join(k) if isinstance(k, tuple) else str(k)): [
            round(v[0] / v[1], 3) if v[1] else None,
            v[1],
        ]
        for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))
    }
    kinds = Counter()
    for (a, b), c in subs.items():
        kinds[sub_kind(a, b)] += c
    return {
        "position": rate(pos),
        "index": rate(idx),
        "runs": rate(run),
        "class": rate(kind),
        "head_vs_tail": {
            g: {"head": v[0], "tail": v[1], "p": round(sign_p(*v), 4)}
            for g, v in head_tail.items()
        },
        "length": {g: dict(c) for g, c in length.items()},
        "sub_kinds": dict(kinds.most_common()),
        "subs": [[a, b, c] for (a, b), c in subs.most_common(40)],
        "glyph": {g: v for g, v in glyph.items()},
    }


def arm_recs(recs: list, key: str, shift: int = 0) -> list:
    """Each string against its own page (``shift`` 0) or the page of the string
    ``shift`` places on in its bin."""
    by_bin = defaultdict(list)
    for m in recs:
        by_bin[m["bin"]].append(m)
    out = []
    for b, ms in by_bin.items():
        for k, m in enumerate(ms):
            page = ms[(k + shift) % len(ms)]["reads"]
            out.append(
                read_string(m["text"], page, key)
                | {"bin": b, "i": m["i"], "ntok": m.get("ntok", len(m["text"].strip()))}
            )
    return out


def run_arm(recs: list) -> dict:
    keys = [
        k
        for k in ("sfx", "vl")
        if any(r.get(k) for m in recs for r in m["reads"] if not r.get("whole"))
    ]
    out = {"n": len(recs)}
    for k in keys:
        real = arm_recs(recs, k)
        null = [r for s in NULL_SHIFTS for r in arm_recs(recs, k, s)]
        out[k] = {
            "real": tables(real),
            "null": tables(null),
            "recs": real,
            "nrecs": null,
        }
    if len(keys) == 2:  # the two readers on the same glyphs
        a = {r["i"]: r for r in out["sfx"]["recs"]}
        b = {r["i"]: r for r in out["vl"]["recs"]}
        c = Counter()
        for i in a:
            for x, y in zip(a[i]["ops"], b[i]["ops"]):
                c[(x[0] == "M", y[0] == "M")] += 1
        n = sum(c.values())
        out["agree"] = {
            "both_hit": c[(True, True)] / n,
            "sfx_only": c[(True, False)] / n,
            "vl_only": c[(False, True)] / n,
            "neither": c[(False, False)] / n,
            "n": n,
        }
    return out


def by_tokens(real: list, null: list) -> dict:
    """cer / hit rate / exact by the string's T5 token count (JA: one row a glyph
    or mark, so the raw string's length), own page and null."""
    rows = defaultdict(lambda: Counter())
    for w, recs in (("real", real), ("null", null)):
        for r in recs:
            t = r["t"]
            if not t:
                continue
            tb = next(f"{a}-{b}" for a, b in TOK_BINS if a <= r["ntok"] <= b)
            for g in (tb, "all"):
                R = rows[(g, w)]
                R["n"] += 1
                R["chars"] += len(t)
                R["cer"] += min(1.0, r["lev"] / len(t))
                R["hit"] += sum(o[0] == "M" for o in r["ops"])
                R["exact"] += r["lev"] == 0
                R["le2"] += r["lev"] <= 2
    return {
        f"{g} | {w}": {
            "n": R["n"],
            "cer": round(R["cer"] / R["n"], 3),
            "hit": round(R["hit"] / R["chars"], 3),
            "exact": R["exact"],
            "le2": R["le2"],
            "chars": round(R["chars"] / R["n"], 1),
        }
        for (g, w), R in sorted(rows.items())
    }


def en_recs(src: list, reads_dir: str, en: dict, t5) -> list:
    """The EN references as records: the EN line as the asked string, the EN
    render's reads, its T5 token count."""
    reads = json.loads((RULER / reads_dir / "reads.json").read_text("utf-8"))
    out = []
    for m in src:
        line = en.get(m["text"])
        if line and str(m["i"]) in reads:
            n = len(t5(line, add_special_tokens=False)["input_ids"])
            out.append(m | {"text": line, "reads": reads[str(m["i"])], "ntok": n})
    return out


def main() -> None:
    from bench._common import make_run_dir, write_result

    from library.anima.weights import load_t5_tokenizer

    src = {lab: records(run, arm) for lab, run, arm in (*TABLES, *SOUP, *CAPTIONED)}
    res = {lab: run_arm(recs) for lab, recs in src.items()}
    # the per-glyph table: jp_v1's rows on every generator, the tuned reader
    glyph = defaultdict(lambda: [0, 0, 0, 0])
    for lab in GLYPH_ARMS:
        for w, o in (("real", 0), ("null", 2)):
            for g, v in res[lab]["sfx"][w]["glyph"].items():
                glyph[g][o] += v[0]
                glyph[g][o + 1] += v[1]
    tok = {
        f"{lab} | {k}": by_tokens(v[k]["recs"], v[k]["nrecs"])
        for lab, v in res.items()
        for k in ("sfx", "vl")
        if k in v
    }
    t5 = load_t5_tokenizer()
    en = json.loads((RULER / "en.json").read_text("utf-8"))
    en_res = {}
    for lab, of, reads_dir in EN_ARMS:
        recs = en_recs(src[of], reads_dir, en, t5)
        v = run_arm(recs)
        for k in ("sfx", "vl"):
            if k in v:
                tok[f"{lab} | {k}"] = by_tokens(v[k]["recs"], v[k]["nrecs"])
                en_res[f"{lab} | {k}"] = {
                    "n": len(recs),
                    "position": v[k]["real"]["position"],
                    "null_position": v[k]["null"]["position"],
                    "length": v[k]["real"]["length"],
                    "recs": [
                        {"i": r["i"], "t": r["t"], "best": r["best"], "ntok": r["ntok"]}
                        for r in v[k]["recs"]
                    ],
                }
    align = {}
    for lab, v in res.items():
        align[lab] = {}
        for k in ("sfx", "vl"):
            if k in v:
                v[k]["real"].pop("glyph")
                v[k]["null"].pop("glyph")
                v[k].pop("nrecs")
                align[lab][k] = v[k].pop("recs")
    align["en"] = {k: v.pop("recs") for k, v in en_res.items()}
    run_dir = make_run_dir("cjk_rl", label="position_read", root=LINE / "results")
    (run_dir / "alignments.json").write_text(
        json.dumps(align, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/position_read.py",
        args={
            "tables": TABLES,
            "soup": SOUP,
            "captioned": CAPTIONED,
            "en": EN_ARMS,
            "null_shifts": NULL_SHIFTS,
        },
        label="position_read",
        metrics={
            "arms": res,
            "glyph_jpv1_sfx": {
                g: {"hit": v[0], "n": v[1], "null_hit": v[2], "null_n": v[3]}
                for g, v in sorted(glyph.items(), key=lambda kv: -kv[1][1])
            },
            "by_tokens": tok,
            "en": en_res,
        },
        artifacts=[str(RESULTS / r) for r in (R1006, R1008, R1113, R1113C)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
