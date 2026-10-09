#!/usr/bin/env python
"""p2_route — retrain_experiments.md § 4 C0: does a routed caption render what the spelled one does? (2026-09-28)

Per-glyph routing (``ANIMA_VOCAB_GLYPH_ROUTE``, ``HybridT5Encoder.glyph_split``)
sends every JA Qwen token to its glyphs' single rows on the T5 side; the Qwen
text is untouched. So an unspaced ``"こんにちは"`` gives T5 the same five single
ids as the spelled ``"こ ん に ち は"`` (the ``--dry_run`` asserts it), and the
only difference between the two captions is Qwen's: the whole word vs spaced
glyphs (ctx_trigger c1's ``ja_hybrid`` vs ``ja_spaced``, 0.965 = 0.965 at the
adapter, never rendered).

Three conditions on ``WORDS`` (en clause), same 8 prompts × 2 seeds:

    routed     unspaced, routing on           rendered here → ``native_route/``
    spelled    spaced                         cached: ``native_spell/``
    unrouted   unspaced, routing off          cached: the floor's ``native_piece/``
               (the seed's piece row; frozen at the seed in every arm)

on two arms: ``p1_mix`` (hypothesis.md P1b: 36 donor kana, cold, in-word +
lone) and the floor (the seed rows). ``native_route/`` holds routed renders
only, in either dir: this script sets the env var itself (never the submit
shell — the daemon would carry it into every later job) and the read asserts
every routed render carried one ext row per glyph.

Decision (retrain_experiments § 4): ``p1_mix`` routed ≈ spelled (≤ 1 edit within
noise of 11 / 16) → pieces stay out (§ 2 holds). Routed ≪ spelled → Qwen's
word context interferes at render; the piece question reopens.

**Read (``results/20260928-0100-c0/``, job ``20260928-010053-5dd1b7``)**:
``p1_mix`` routed ≤ 1 edit 14 / 16 vs spelled 11 (paired 3 / 0, p 0.25),
official 8 vs 5, contained 11 vs 6; the piece row 0; the seed rows 0 on all
three. Routing holds; pieces stay out.

C2 (retrain_experiments § 4) widens it: ``C2_WORDS`` = こんにちは + 7 held-in words
of donor glyphs, none repeated, sharing no 3-glyph substring with any donor
word (Stage B's ``donor_words.json``), spelled **and** routed, on the floor,
Stage B (warm), ``p1_cold``, ``p1_mix``, ``p1_lone`` (C1). Only the keys an
arm's cache lacks render (a scratch tag folded in, so no cached read is
overwritten). Routing is on for the whole process; the encoding check
asserts every spelled word encodes the same with it on and off, so the
spelled renders are the unrouted spelled captions of record.

Legs:
  c0   (GPU) the routed renders the caches lack (``p1_mix`` + floor), then the
       three-way read, paired
  c2   (GPU) ``C2_WORDS`` spelled + routed on ``C2_ARMS``; per arm the words
       total, routed vs spelled, and each arm vs the floor and vs ``p1_mix``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, bootstrap, pin_old_seed, floor_dir, load_experiment  # noqa: E402

bootstrap()
pin_old_seed()  # the retrain reads against the old seed's floor of record
from bench._common import make_run_dir, write_result  # noqa: E402
from cjk_scale import reads as scoring  # noqa: E402

SB = load_experiment("stage_b")

EXP = OUT / "experiments"
ENV = "ANIMA_VOCAB_GLYPH_ROUTE"
TAG = "route"  # native_route/: routed renders only
WORDS = (SB.HELD_IN,)  # C0: こんにちは, the one donor word the floor holds
# C2: + 7 donor-glyph words, trigram-held-out from the donor words (checked)
C2_WORDS = (
    SB.HELD_IN,
    "たいせつ",
    "かなしい",
    "かんがえ",
    "たすけて",
    "こうえん",
    "てつだう",
    "ことば",
)
CLAUSE = "en"
ARMS = {"floor": None, "p1_mix": EXP / "p1_mix"}  # None = the seed rows' dir
C2_ARMS = {
    "floor": None,
    "stage_b": OUT / SB.NAME,
    "p1_cold": EXP / "p1_cold",
    "p1_mix": EXP / "p1_mix",
    "p1_lone": EXP / "p1_lone",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--legs", nargs="+", default=["c0"], choices=["c0", "c2"])
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def check_encoding() -> dict:
    """Routed unspaced == spelled on the T5 side, for every word."""
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

    donor = json.loads((OUT / SB.NAME / "donor_words.json").read_text("utf-8"))
    tri = {d[i : i + 3] for d in donor for i in range(len(d) - 2)}
    out = {}
    for w in dict.fromkeys((*WORDS, *C2_WORDS)):
        assert set(w) <= set(SB.DONOR) and len(set(w)) == len(w), w
        assert not any(w[i : i + 3] in tri for i in range(len(w) - 2)), w
        rec = {
            "unrouted": ext(False, w),
            "spelled": ext(False, SB.spell(w)),
            "routed": ext(True, w),
        }
        assert rec["routed"] == rec["spelled"] == ext(True, SB.spell(w)), (w, rec)
        assert len(rec["routed"]) == len(w), (w, rec)
        out[w] = rec
        print(f"{w}: {json.dumps(rec)}", flush=True)
    return out


def routed_read(arm: str) -> Path:
    """The routed renders of ``WORDS`` in ``arm``'s ``native_route/`` (only
    the keys it lacks render)."""
    assert os.environ.get(ENV) == "1"
    if ARMS[arm] is None:
        from cjk_scale.eval import ensure_native_floor

        n = ensure_native_floor(SB.arm_rc(), f"native_{TAG}", list(WORDS), CLAUSE)
        print(f"floor: {n} keys rendered into native_{TAG}/", flush=True)
        return floor_dir() / f"native_{TAG}" / "native_reads.json"
    from cjk_scale.config import RunConfig
    from cjk_scale.eval import TRAINED_ARM, probe_args
    from stages import run as run_stage

    path = ARMS[arm]
    reads = path / f"native_{TAG}" / "native_reads.json"
    if reads.exists():
        held = {(m["text"], m["clause"]) for m in json.loads(reads.read_text("utf-8"))}
        if all((w, CLAUSE) in held for w in WORDS):
            print(f"  (read from disk: {reads})", flush=True)
            return reads
    rc = RunConfig(name=SB.NAME, path=Path(__file__), vocabs=(), read=())
    a = probe_args(
        rc,
        TRAINED_ARM,
        ["native"],
        [
            "--eval_tag",
            TAG,
            "--native_chars",
            ",".join(WORDS),
            "--native_clauses",
            CLAUSE,
        ],
    )
    a.arm_path, a.data_path = str(path), str(path / "data")
    run_stage("native", a)
    return reads


def ensure_reads(path: Path | None, sub: str, keys: list[str]) -> Path:
    """``keys`` (en) in ``path``'s ``native_<sub>/`` (``None`` = the floor):
    the missing ones render into a scratch ``native_add_<sub>/`` and are
    folded in, so a cached read is never overwritten."""
    import shutil

    from cjk_scale.config import RunConfig
    from cjk_scale.eval import TRAINED_ARM, _fold, ensure_native_floor, probe_args
    from stages import run as run_stage

    if path is None:
        n = ensure_native_floor(SB.arm_rc(), f"native_{sub}", keys, CLAUSE)
        print(f"  floor: {n} keys rendered into native_{sub}/", flush=True)
        return floor_dir() / f"native_{sub}" / "native_reads.json"
    dst = path / f"native_{sub}" / "native_reads.json"
    held = (
        {(m["text"], m["clause"]) for m in json.loads(dst.read_text("utf-8"))}
        if dst.exists()
        else set()
    )
    miss = [k for k in keys if (k, CLAUSE) not in held]
    if miss:
        rc = RunConfig(name=SB.NAME, path=Path(__file__), vocabs=(), read=())
        a = probe_args(
            rc,
            TRAINED_ARM,
            ["native"],
            [
                "--eval_tag",
                f"add_{sub}",
                "--native_chars",
                ",".join(miss),
                "--native_clauses",
                CLAUSE,
            ],
        )
        a.arm_path, a.data_path = str(path), str(path / "data")
        run_stage("native", a)
        scratch = path / f"native_add_{sub}"
        recs = json.loads((scratch / "native_reads.json").read_text("utf-8"))
        n = _fold("native", recs, dst, move=True)
        shutil.rmtree(scratch)
        print(f"  {path.name}: {n} keys rendered into native_{sub}/", flush=True)
    return dst


def check_routed(path: Path) -> None:
    """Every routed render of a word carried one ext row per glyph (a render
    with routing off would carry the piece row: 1)."""
    for m in json.loads(path.read_text("utf-8")):
        if m["text"] in WORDS:
            assert m["ext_rows"] == len(m["text"]), (m["file"], m["ext_rows"])


def keyed(h: dict) -> dict:
    """Stage B's per-render hits, keyed by the unspaced word (so a spelled and
    a routed render of one prompt × seed pair up)."""
    return {(t.replace(" ", ""), c, pi, s): v for (t, c, pi, s), v in h.items()}


def read(path: Path, spaced: bool, words=WORDS) -> dict:
    keys = [SB.spell(w) if spaced else w for w in words]
    return keyed(scoring.hits(path, keys, CLAUSE))


def c2(metrics: dict) -> None:
    missing = [a for a, p in C2_ARMS.items() if p and not (p / "trained.pt").exists()]
    assert not missing, f"no trained.pt for {missing}"
    words = list(C2_WORDS)
    conds: dict = {}
    for arm, path in C2_ARMS.items():
        print(f"{arm}:", flush=True)
        rp = ensure_reads(path, TAG, words)
        check_routed(rp)
        sp = ensure_reads(path, "spell", [SB.spell(w) for w in words])
        conds[arm] = {
            "routed": read(rp, False, words),
            "spelled": read(sp, True, words),
        }
    out = metrics.setdefault("c2", {})
    for arm, cs in conds.items():
        out[arm] = {}
        for cond, h in cs.items():
            print(f"{arm} · {cond}:", flush=True)
            out[arm][cond] = scoring.tally(h)
        pr = scoring.paired(cs["routed"], cs["spelled"])
        out[arm]["paired_routed_vs_spelled"] = pr
        print(f"  {arm} routed vs spelled {pr}", flush=True)
        for ref in ("floor", "p1_mix"):
            if arm == ref:
                continue
            for cond in ("routed", "spelled"):
                pr = scoring.paired(cs[cond], conds[ref][cond])
                out[arm][f"paired_{cond}_vs_{ref}"] = pr
                print(f"  {arm} {cond} vs {ref} {pr}", flush=True)


def main():
    args = parse_args()
    enc = check_encoding()
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "p2_route", label=args.label, root=LINE / "experiments" / "p2_route" / "results"
    )
    metrics: dict = {"words": list(WORDS), "clause": CLAUSE, "encoding": enc}
    if "c2" in args.legs:
        os.environ[ENV] = "1"  # before the native stage installs the strategy
        metrics["c2_words"] = list(C2_WORDS)
        c2(metrics)
    if "c0" in args.legs:
        os.environ[ENV] = "1"  # before the native stage installs the strategy
        floor = floor_dir()
        cached = {
            "floor": {
                "spelled": floor / "native_spell" / "native_reads.json",
                "unrouted": floor / "native_piece" / "native_reads.json",
            },
            "p1_mix": {
                "spelled": ARMS["p1_mix"] / "native_spell" / "native_reads.json",
                # the piece row is frozen at the seed in p1_mix: the floor's render
                "unrouted": floor / "native_piece" / "native_reads.json",
            },
        }
        conds: dict = {}
        for arm in ARMS:
            rp = routed_read(arm)
            check_routed(rp)
            conds[arm] = {
                "routed": read(rp, False),
                "spelled": read(cached[arm]["spelled"], True),
                "unrouted": read(cached[arm]["unrouted"], False),
            }
        out = metrics.setdefault("reads", {})
        for arm, cs in conds.items():
            out[arm] = {}
            for cond, h in cs.items():
                print(f"{arm} · {cond}:", flush=True)
                out[arm][cond] = scoring.tally(h)
            for a, b in (
                ("routed", "spelled"),
                ("routed", "unrouted"),
                ("spelled", "unrouted"),
            ):
                pr = scoring.paired(cs[a], cs[b])
                out[arm][f"paired_{a}_vs_{b}"] = pr
                print(f"  {arm} {a} vs {b} {pr}", flush=True)
        pr = scoring.paired(conds["p1_mix"]["routed"], conds["floor"]["routed"])
        out["p1_mix_routed_vs_floor_routed"] = pr
        print(f"  p1_mix routed vs floor routed {pr}", flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(floor_dir() / f"native_{TAG}"), str(ARMS["p1_mix"])],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
