#!/usr/bin/env python
"""c3_kanji — retrain_experiments.md § 4 C3: does the P1b recipe carry to kanji? (2026-09-28)

P1b (``p1_mix``: 36 donor kana, cold, in-word + lone) held the floor's
singles and composed. C3 runs the same recipe on 36 kanji, with the two
changes the retrain makes:

- **Windows, not whole lines** (retrain_experiments § 3): an in-word item is a
  substring of 2–4 glyphs of a dialogue line, every glyph a donor kanji or a
  kana of the context, none repeated, holding at least one donor kanji. A
  draw picks a donor kanji uniformly, then one of its windows. Windows
  holding a 3-glyph substring of a read word are dropped.
- **Routed captions**: the window is unspaced on the image and in the
  caption, and per-glyph routing (``ANIMA_VOCAB_GLYPH_ROUTE``, set by this
  script for every leg) sends each glyph to its single row. C0 found the
  routed caption renders at least as well as the spelled one, and spelling a
  kanji hands T5 its space-prefixed token's row (`` 名`` ≠ 名), so the
  retrain's kanji need the routed form. Every window must encode to its
  glyphs' single ids and nothing else, or it is dropped.

The context is ``p1_mix``'s merged rows (plan_retrain § 2 order: the kanji
run sits on the kana run's rows), so the windows' kana are cold-trained
rows. Donors (``KANJI``): stage_i's 12 (no seed row) + dense_a0's 12 (seed
rows), whose floor singles are cached; 日本人大丈夫何時 (seed rows) for the
read words; 死父誰名 (frequent, no row yet). All start cold from the pack
rows. Budget (plan_retrain § 2): the cold single row (150) × 1.5 = 225
steps / row; items at the cold single factor (150 / 90), so the in-word
items keep ``p1_mix``'s epochs (5.4).

Table: ``p1_mix``'s (Stage B's in-word groups + the production ``b0709``
at 0.5) with ``scene_spelled`` → ``scene_window``.

Legs:
  data   (CPU) windows, the encoding check, the data dir
  train  (GPU) ``c3_kanji``, cold, on ``p1_mix``'s rows
  read   (GPU) the 24 cached kanji singles (vs the floor's ``native_stagei`` /
         ``native_densea0``) and ``READ_WORDS`` routed on the floor,
         ``p1_mix`` (the kana context alone) and ``c3_kanji``, into each
         dir's ``native_route/``

``--dry_run`` prints the windows per donor and the encodings.

``--steps`` (plan_retrain § 1): the same data at another steps / row; any
value but 225 trains into ``c3_kanji_<steps>`` and reads the 225 arm
(cached) beside it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # every leg is routed (docstring)

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, pin_old_seed, floor_dir, load_experiment  # noqa: E402

bootstrap()
pin_old_seed()  # the retrain reads against the old seed's floor of record
from bench._common import make_run_dir, write_result  # noqa: E402
from cjk_scale import reads as scoring  # noqa: E402


SB = load_experiment("stage_b")
P1 = load_experiment("p1_cap")
P2 = load_experiment("p2_route")

EXP = OUT / "experiments"
NAME = "c3_kanji"
DATA = "run0928_c3_kanji"
CONTEXT = EXP / "p1_mix" / "trained.pt"
KANA = SB.DONOR  # p1_mix's 36 donor kana: the windows' kana, cold-trained
SI = "山田野郎太道場星空天地小"  # stage_i: no seed row; floor in native_stagei/
DA = "精俺感間奥愛様無最輩帰飲"  # dense_a0: seed rows; floor in native_densea0/
KANJI = SI + DA + "日本人大丈夫何時" + "死父誰名"
READ_WORDS = ("山田太郎", "小山田", "日本人", "大丈夫", "何時間", "愛してる")
WIN_LEN = (2, 4)
ITEM_FACTOR = 150 / 90  # budget.py's cold single row
STEPS = 225  # 150 × 1.5 (plan_retrain § 2)
TAG = "route"
FLOOR_SINGLES = {"stagei": SI, "densea0": DA}

_BY_GLYPH: dict = {}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs", nargs="+", default=["data"], choices=["data", "train", "read"]
    )
    p.add_argument("--workers", type=int, help="data: render processes")
    p.add_argument("--steps", type=int, default=STEPS, help="steps / row")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def rc_of(name: str):
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=name, path=Path(__file__), vocabs=(f"chars:{KANJI}",), read=()
    )


# ----------------------------------------------------------------------------
# windows


def windows() -> list[str]:
    from cjk_scale.config import phrase_file

    assert len(set(KANJI)) == len(KANJI) == 36, len(KANJI)
    allowed = set(KANJI) | set(KANA)
    tri = {w[i : i + 3] for w in READ_WORDS for i in range(len(w) - 2)}
    out = set()
    for ln in Path(phrase_file()).read_text(encoding="utf-8").splitlines():
        for run in re.findall(r"[぀-ヿ一-鿿]+", ln.split("\t")[0]):
            for i in range(len(run)):
                for n in range(WIN_LEN[0], WIN_LEN[1] + 1):
                    w = run[i : i + n]
                    if len(w) < n:
                        break
                    if (
                        set(w) <= allowed
                        and len(set(w)) == n
                        and set(w) & set(KANJI)
                        and not any(w[k : k + 3] in tri for k in range(n - 2))
                    ):
                        out.add(w)
    return sorted(out)


def set_windows(ws: list[str]) -> dict:
    _BY_GLYPH.clear()
    for g in KANJI:
        _BY_GLYPH[g] = [w for w in ws if g in w]
    cov = {g: len(v) for g, v in _BY_GLYPH.items()}
    assert min(cov.values()) > 0, cov
    return cov


def check_encoding(ws: list[str]) -> tuple[list[str], dict]:
    """Each glyph is one single id (routing on = off for a lone glyph); a
    window / read word routed encodes to its glyphs' ids and nothing else.
    Returns the windows that do, and the glyph ids."""
    from transformers import AutoTokenizer

    from library.anima import ext_vocab
    from library.anima.ext_vocab import T5_TABLE_SIZE, HybridT5Encoder
    from library.anima.vocab_pack import resolve_pack_prefix
    from library.env import resolve_under_home

    t5 = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/t5_old")
    )
    qw = AutoTokenizer.from_pretrained(
        resolve_under_home("library/anima/configs/qwen3_06b")
    )
    _, mapping = ext_vocab.load_ext_assets(
        resolve_pack_prefix(os.environ["ANIMA_VOCAB_PACK"])
    )
    encs = {
        r: HybridT5Encoder.from_mapping(t5, qw, mapping, glyph_route=r)
        for r in (False, True)
    }

    def ext(route: bool, text: str) -> list[int]:
        ids, mask = encs[route].encode(f'Japanese text reads as "{text}".', 512)
        return [
            i - T5_TABLE_SIZE for i, m in zip(ids, mask) if m and i >= T5_TABLE_SIZE
        ]

    ids = {}
    for c in KANJI + KANA:
        a, b = ext(False, c), ext(True, c)
        assert len(a) == 1 and a == b, (c, a, b)
        ids[c] = a[0]
    for w in READ_WORDS:
        assert ext(True, w) == [ids[c] for c in w], (w, ext(True, w))
    ok = [w for w in ws if ext(True, w) == [ids[c] for c in w]]
    return ok, ids


# ----------------------------------------------------------------------------
# recipe + table


def scene_window(pools, rng, p: dict):
    """A donor window in one bubble, unspaced on the image and in the
    caption (routed at encode); drawn kanji-first. Stage B's
    ``scene_spelled`` without the spacing."""
    from cjk_scale.recipes import _draw_scene, _target

    word = rng.choice(_BY_GLYPH[rng.choice(KANJI)])
    f = p.get("fill", [0.7, 1.0])
    lo, hi = f if isinstance(f, list) else (f, f)
    fill = rng.uniform(float(lo), float(hi))
    item = _draw_scene(
        pools,
        rng,
        word,
        min_glyph=int(p.get("min_glyph", 28)),
        fill=fill,
        max_lines=1,
        target_px=_target(rng, p),
        fill_max=fill,
        fill_min=float(p.get("fill_min", 0)),
    )
    if item is None:
        return None
    assert item.caption.count(f'"{word}"') == 1, item.caption
    return item


def table() -> tuple:
    """``p1_mix``'s table with its in-word tier drawing windows (the tiers
    that became ``bubbleN_34`` / ``bubbleN_18``, on this arm's own draw)."""
    import dataclasses

    name = {(0.5, 0.7): "bubbleN_34", (0.3, 0.5): "bubbleN_18"}
    return tuple(
        dataclasses.replace(
            g,
            tiers=tuple(
                dataclasses.replace(t, name=name[g.band], recipe="bubbleN")
                if t.recipe == "scene_spelled"
                else t
                for t in g.tiers
            ),
        )
        for g in P1.mix_table()
    )


def fixed_budget() -> None:
    """The donors mix seed rows and none (budget.py would call them two
    budgets); all train cold, so all take the cold single factor."""
    from cjk_scale import budget

    budget.run_factor = lambda *a, **k: ITEM_FACTOR
    budget.run_budget = lambda vocabs, *a, **k: dict.fromkeys(vocabs, ITEM_FACTOR)


# ----------------------------------------------------------------------------
# reads


def read_singles(arm: Path | None) -> dict:
    """The 24 cached kanji alone (en): the floor from its caches of record,
    an arm from its ``native_route/``."""
    keys = list(SI + DA)
    if arm is None:
        h = {}
        for tag, chars in FLOOR_SINGLES.items():
            f = floor_dir() / f"native_{tag}" / "native_reads.json"
            h.update(scoring.hits(f, list(chars), "en"))
        return h
    return scoring.hits(P2.ensure_reads(arm, TAG, keys), keys, "en")


def read_words(arm: Path | None) -> dict:
    words = list(READ_WORDS)
    rp = P2.ensure_reads(arm, TAG, words)
    for m in json.loads(rp.read_text("utf-8")):
        if m["text"] in words:
            assert m["ext_rows"] == len(m["text"]), (m["file"], m["ext_rows"])
    return scoring.hits(rp, words, "en")


# ----------------------------------------------------------------------------


def main():
    args = parse_args()
    name = NAME if args.steps == STEPS else f"{NAME}_{args.steps}"
    from cjk_scale import recipes

    recipes.RECIPES["bubbleN"] = scene_window  # this arm's draw, kanji-first
    recipes.RECIPES["scene_single_small"] = SB.scene_single_small
    fixed_budget()
    ws = windows()
    ok, ids = check_encoding(ws)
    print(
        f"windows {len(ws)} ({len(ws) - len(ok)} dropped by the encoding check); "
        f"by length {dict(sorted(Counter(map(len, ok)).items()))}",
        flush=True,
    )
    cov = set_windows(ok)
    print(
        "windows per donor: " + " ".join(f"{g}{n}" for g, n in cov.items()),
        flush=True,
    )
    metrics: dict = {
        "kanji": KANJI,
        "read_words": list(READ_WORDS),
        "n_windows": len(ok),
        "dropped": len(ws) - len(ok),
        "per_glyph_windows": cov,
        "steps_per_row": args.steps,
        "item_factor": ITEM_FACTOR,
        "context": str(CONTEXT),
    }
    if args.dry_run:
        print(" ".join(ok[:80]), flush=True)
        for g in P1.mix_table():
            print(g.label, g.band, g.share, [t.recipe for t in g.tiers], flush=True)
        return
    run_dir = make_run_dir(
        "c3_kanji", label=args.label, root=LINE / "experiments" / "c3_kanji" / "results"
    )
    (OUT / DATA).mkdir(parents=True, exist_ok=True)
    (OUT / DATA / "windows.json").write_text(
        json.dumps(ok, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    if "data" in args.legs:
        from cjk_scale.builder import build

        build(rc_of(DATA), workers=args.workers, table=table())
        items = Counter(
            json.loads(ln)["tier"]
            for ln in (OUT / DATA / "data" / "train.jsonl").open(encoding="utf-8")
        )
        metrics["items"] = dict(items)
        print(f"items {dict(items)}", flush=True)
    if "train" in args.legs:
        from cjk_scale.train import train

        assert CONTEXT.exists(), CONTEXT
        train(
            rc_of(name),
            data=OUT / DATA / "data",
            out=EXP / name,
            cold=True,
            steps_per_row=args.steps,
            context=CONTEXT,
        )
    if "read" in args.legs:
        arms = {"floor": None, "p1_mix": EXP / "p1_mix", NAME: EXP / NAME}
        arms[name] = EXP / name
        sh = {"floor": read_singles(None), NAME: read_singles(EXP / NAME)}
        sh[name] = read_singles(EXP / name)
        wh = {a: read_words(p) for a, p in arms.items()}
        out = metrics.setdefault("reads", {})
        for a, h in sh.items():
            print(f"{a} · singles:", flush=True)
            out.setdefault(a, {})["singles"] = scoring.tally(h)
        pr = scoring.paired(sh[name], sh["floor"])
        out[name]["singles_paired_vs_floor"] = pr
        print(f"  singles {name} vs floor {pr}", flush=True)
        for grp, chars in (("stagei", SI), ("densea0", DA)):
            sub = {k: v for k, v in sh[name].items() if k[0] in chars}
            fsub = {k: v for k, v in sh["floor"].items() if k[0] in chars}
            pr = scoring.paired(sub, fsub)
            out[name][f"singles_{grp}_paired_vs_floor"] = pr
            print(f"  singles ({grp}) {name} vs floor {pr}", flush=True)
        for a, h in wh.items():
            print(f"{a} · words (routed):", flush=True)
            out.setdefault(a, {})["words"] = scoring.tally(h)
        pairs = [(name, "floor"), (name, "p1_mix"), ("p1_mix", "floor")]
        if name != NAME:
            pairs.append((name, NAME))
            for grp, chars in (("stagei", SI), ("densea0", DA)):
                sub = {k: v for k, v in sh[name].items() if k[0] in chars}
                rsub = {k: v for k, v in sh[NAME].items() if k[0] in chars}
                pr = scoring.paired(sub, rsub)
                out[name][f"singles_{grp}_paired_vs_{NAME}"] = pr
                print(f"  singles ({grp}) {name} vs {NAME} {pr}", flush=True)
        for a, ref in pairs:
            pr = scoring.paired(wh[a], wh[ref])
            out[a][f"words_paired_vs_{ref}"] = pr
            print(f"  words {a} vs {ref} {pr}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / DATA), str(EXP / name)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
