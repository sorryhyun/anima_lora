#!/usr/bin/env python
"""transplant_line — a shared "in-line" Δ direction added to untrained singles (2026-09-26)

spell_b's five single rows, trained in lines, share a Δ component (pairwise
cos +0.18, radial part ≤ 11 %); the same direction sits in the Δ of every
run trained at the low bands / in lines (stage0305 micro +0.34, stage0507
+0.22, run0926_300f_sp +0.30) and not in the single-canvas one (stage0709
+0.02, random 95 % ≈ 0.06). Question: added training-free to the seed rows
of あ り が と う (u is estimated from pieces, so it is held out from these
singles), does it buy the spelled string spell_b's training bought (≤ 2
edits 13 / 32 vs the seed's 1 / 32) — and does it bring spell_b's doubling?

Direction ``u`` = the normalized mean of run0926_300f_sp's per-row Δ vs the
seed, each Δ with its component along its own seed row removed (300 rows:
the less noisy estimate; its cos to spell_b's is printed). Step = the mean
projection of spell_b's five Δ on ``u`` (how far a single moved along it
when trained in lines: 14.4, ≈ 5 % of a row; the pieces moved 94.7), ×
``--alphas``. Arms: ``u`` × α, and a random unit direction ⟂ ``u`` at the
largest α (the control: row norm alone is a hit lever). Only the target glyphs'
rows move; every other row is the seed's.

Read: native (en + swap, 8 prompts × 2 seeds) — the spelled string and its
glyphs alone — against the seed floor's ``native_spell/`` cache, the same
floor spell_b read. The eval renders no floor: a key the cache lacks is an
error (the floor of record is not re-rendered per experiment).

``--mode strip`` (spell_2026_09_26.md § 7, "the shared Δ"): spell_b's five
rows with their shared Δ component removed — ``uB`` = the normalized mean
of the five Δ (own-row component removed), each row minus its Δ's
projection on ``uB``; the rest of its Δ kept. Does composition stay
without it (then the line signal is per row) or go (the shared part is
needed)? ``--mode shared`` is the complement: the seed rows plus only that
projection — is the shared part alone enough? Read on spell_b's own keys (held-out spelled ありがとう, the three
spelled training words, the singles), against the floor and spell_b's
reads on disk.

legs: build (CPU: the arms' ``trained.pt`` under
``output/cjk_anima_scale/<arm>/``), eval (GPU, through ``make daemon-run``).
``--dry_run`` prints the direction, the step and the encodings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap  # noqa: E402

bootstrap()

from bench._common import make_run_dir, write_result  # noqa: E402

SRC_RUN = "run0926_300f_sp"  # u
MAG_RUN = "run0926_spell_b"  # the step along u
MAG_GLYPHS = "ありがとう"
TARGET = "ありがとう"  # spell_b's string: its floor is in the cache


def spell(s: str) -> str:
    return " ".join(s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--legs", nargs="+", default=["build"], choices=["build", "eval"])
    p.add_argument("--alphas", type=float, nargs="+", default=[1.0])
    p.add_argument("--no_random", action="store_true", help="skip the control arm")
    p.add_argument(
        "--mode", default="transplant", choices=["transplant", "strip", "shared"]
    )
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def encoder():
    import os

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
    enc = HybridT5Encoder.from_mapping(t5, qw, mapping)

    def ext(text):
        ids, mask = enc.encode(f'Japanese text reads as "{text}".', 512)
        return [
            i - T5_TABLE_SIZE for i, m in zip(ids, mask) if m and i >= T5_TABLE_SIZE
        ]

    return ext


def single_ids(ext, glyphs: str) -> list[int]:
    """Each glyph's single ext id; asserts the spelled string is exactly them."""
    single = {c: ext(c) for c in glyphs}
    assert all(len(v) == 1 for v in single.values()), single
    ids = [single[c][0] for c in glyphs]
    got = ext(spell(glyphs))
    assert got == ids, (glyphs, got, ids)
    return ids


def rows(path: Path) -> tuple[dict, dict]:
    """``{ext id: effective row}`` (raw × row_scale) and the file itself."""
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)
    d = sd["delta"]
    s = float(d["row_scale"])
    return {int(e): d["raw"][i].float() * s for i, e in enumerate(d["ext_ids"])}, sd


def tangential(run_rows: dict, seed: dict, ids=None):
    """Δ vs the seed for the rows the run moved (or ``ids``), each with its
    component along its own seed row removed."""
    import torch
    import torch.nn.functional as F

    if ids is None:
        ids = [
            e
            for e in run_rows
            if e in seed and (run_rows[e] - seed[e]).norm() > 1e-4 * seed[e].norm()
        ]
    D = torch.stack([run_rows[e] - seed[e] for e in ids])
    S = F.normalize(torch.stack([seed[e] for e in ids]), dim=1)
    return ids, D - (D * S).sum(1, keepdim=True) * S


def direction(seed: dict, mag_ids: list[int]) -> dict:
    import torch
    import torch.nn.functional as F

    _, T = tangential(rows(OUT / SRC_RUN / "trained.pt")[0], seed)
    u = F.normalize(T.mean(0), dim=0)
    _, TB = tangential(rows(OUT / MAG_RUN / "trained.pt")[0], seed, mag_ids)
    uB = F.normalize(TB.mean(0), dim=0)
    step = float((TB @ u).mean())
    g = torch.Generator().manual_seed(0)
    r = torch.randn(u.shape[0], generator=g)
    r = F.normalize(r - (r @ u) * u, dim=0)
    return {
        "u": u,
        "r": r,
        "step": step,
        "src_rows": len(T),
        "cos_u_uB": float(u @ uB),
        "src_mean_proj": float((T @ u).mean()),
        "mag_proj": [round(float(x), 3) for x in TB @ u],
        "mag_dnorm": [round(float(x), 3) for x in TB.norm(dim=1)],
    }


def arm_names(label: str, alphas, random: bool) -> dict:
    out = {f"tl_{label}_u{a:g}": ("u", a) for a in alphas}
    if random:
        out[f"tl_{label}_rand{max(alphas):g}"] = ("r", max(alphas))
    return out


def build_arm(name: str, vec, step: float, tgt_ids: list[int], seed_path: Path):
    """The seed's trained.pt with ``step · vec`` added to the target rows."""
    import torch

    sd = torch.load(seed_path, map_location="cpu", weights_only=False)
    d = sd["delta"]
    s = float(d["row_scale"])
    idx = {int(e): i for i, e in enumerate(d["ext_ids"])}
    raw = d["raw"].clone()
    moved = {}
    for e in tgt_ids:
        i = idx[e]
        before = raw[i].float() * s
        raw[i] = ((before + step * vec) / s).to(raw.dtype)
        moved[e] = round(float(step * vec.norm() / before.norm()), 3)
    out = {
        **sd,
        "delta": {**d, "raw": raw},
        "arm": "rows",
        "seed_merged": str(seed_path),
        "transplant": {"step": step, "rows": tgt_ids, "rel": moved},
    }
    dst = OUT / name
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst / "trained.pt")
    return moved


def native_read(name: str, chars: list, tag: str) -> Path:
    from cjk_scale.config import RunConfig
    from cjk_scale.eval import TRAINED_ARM, arm_out, probe_args
    from stages import run as run_stage

    rc = RunConfig(name=name, path=Path(__file__), vocabs=(), read=())
    reads = arm_out(rc, TRAINED_ARM) / f"native_{tag}" / "native_reads.json"
    if reads.exists():
        held = {(m["text"], m["clause"]) for m in json.loads(reads.read_text("utf-8"))}
        if all((k, c) in held for k in chars for c in ("en", "swap")):
            print(f"  (read from disk: {reads})", flush=True)
            return reads
    a = probe_args(
        rc,
        TRAINED_ARM,
        ["native"],
        ["--eval_tag", tag, "--native_chars", ",".join(chars)],
    )
    run_stage("native", a)
    return arm_out(rc, TRAINED_ARM) / f"native_{tag}" / "native_reads.json"


def tally(path: Path, chars, target: str) -> dict:
    """official / loose / contained per string × clause, plus, for the
    spelled target, best-read edit distance ≤ 1 / ≤ 2 and repeated-glyph
    reads for the singles."""
    from common.readers import norm

    def lev(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    out: dict = {}
    for m in json.loads(path.read_text("utf-8")):
        if m["text"] not in chars:
            continue
        k = f"{m['text']}|{m['clause']}"
        t = norm(m["text"])
        c = out.setdefault(
            k,
            {
                "n": 0,
                "official": 0,
                "loose": 0,
                "contained": 0,
                "le1": 0,
                "le2": 0,
                "repeat": 0,
            },
        )
        reads = [
            norm(r.get(x) or "") for r in m.get("reads", []) for x in ("sfx", "vl")
        ]
        c["n"] += 1
        c["official"] += bool(m.get("hit_sfx")) and bool(m.get("hit_vl"))
        c["loose"] += bool(m.get("hit_sfx")) or bool(m.get("hit_vl"))
        c["contained"] += any(t in r for r in reads)
        if len(t) > 2:  # a 2-glyph read is always ≤ 2 edits
            best = min((lev(r, t) for r in reads if r), default=len(t))
            c["le1"] += best <= 1
            c["le2"] += best <= 2
        else:
            c["repeat"] += any(r.count(t) >= 2 for r in reads)
    for k, c in out.items():
        extra = (
            f"≤1 {c['le1']:>2}  ≤2 {c['le2']:>2}"
            if len(norm(k.split("|")[0])) > 2
            else f"repeat {c['repeat']:>2}"
        )
        print(
            f"  {k:<14} official {c['official']:>2}  loose {c['loose']:>2}  "
            f"contained {c['contained']:>2}  {extra} / {c['n']}",
            flush=True,
        )
    return out


READ_WORDS = ("あり", "とう", "あがり")  # spell_b's spelled training words


def build_strip(name: str, mag_ids: list[int], seed: dict, mode: str) -> dict:
    """spell_b's trained.pt with each of its five rows minus its Δ's
    projection on their shared direction ``uB`` (``strip``), or the seed row
    plus only that projection (``shared``)."""
    import torch
    import torch.nn.functional as F

    src = OUT / MAG_RUN / "trained.pt"
    brows, sd = rows(src)
    _, TB = tangential(brows, seed, mag_ids)
    uB = F.normalize(TB.mean(0), dim=0)
    d = sd["delta"]
    s = float(d["row_scale"])
    idx = {int(e): i for i, e in enumerate(d["ext_ids"])}
    raw = d["raw"].clone()
    info = {}
    for e in mag_ids:
        delta = brows[e] - seed[e]
        c = float(delta @ uB)
        row = brows[e] - c * uB if mode == "strip" else seed[e] + c * uB
        raw[idx[e]] = (row / s).to(raw.dtype)
        info[e] = {
            "proj": round(c, 2),
            "dnorm": round(float(delta.norm()), 2),
            "removed_energy": round(c * c / float(delta.norm()) ** 2, 3),
        }
    dst = OUT / name
    dst.mkdir(parents=True, exist_ok=True)
    torch.save({**sd, "delta": {**d, "raw": raw}, mode: info}, dst / "trained.pt")
    return info


def check_floor(chars) -> Path:
    from cjk_scale.paths import floor_dir

    floor = floor_dir() / "native_spell" / "native_reads.json"
    held = {(m["text"], m["clause"]) for m in json.loads(floor.read_text("utf-8"))}
    miss = [(k, c) for k in chars for c in ("en", "swap") if (k, c) not in held]
    assert not miss, f"floor cache lacks {miss} — no floor renders here"
    return floor


def main_strip(args, ext):
    mag_ids = single_ids(ext, MAG_GLYPHS)
    seed, _ = rows(SEED_ROWS)
    name = f"tl_{args.label}_b{args.mode}"
    chars = [spell(MAG_GLYPHS), *(spell(w) for w in READ_WORDS), *MAG_GLYPHS]
    floor = check_floor(chars)
    print(f"arm {name}; keys {chars}", flush=True)
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "transplant_line",
        label=args.label,
        root=LINE / "experiments" / "transplant_line" / "results",
    )
    metrics: dict = {"mode": args.mode, "arm": name}
    if "build" in args.legs:
        metrics["proj"] = build_strip(name, mag_ids, seed, args.mode)
        print(f"built {name}: {json.dumps(metrics['proj'])}", flush=True)
    if "eval" in args.legs:
        print("floor (the seed dir's native_spell/ cache):", flush=True)
        metrics["floor"] = tally(floor, chars, MAG_GLYPHS)
        print(f"{MAG_RUN} (on disk):", flush=True)
        metrics["spell_b"] = tally(
            OUT / MAG_RUN / "native_spell" / "native_reads.json", chars, MAG_GLYPHS
        )
        print(f"{name}:", flush=True)
        metrics["reads"] = tally(native_read(name, chars, "spell"), chars, MAG_GLYPHS)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / name)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


def main():
    args = parse_args()
    ext = encoder()
    if args.mode in ("strip", "shared"):
        return main_strip(args, ext)
    mag_ids = single_ids(ext, MAG_GLYPHS)
    tgt_ids = single_ids(ext, TARGET)
    seed, _ = rows(SEED_ROWS)
    assert all(e in seed for e in tgt_ids), "a target glyph has no seed row"
    dr = direction(seed, mag_ids)
    info = {k: v for k, v in dr.items() if k not in ("u", "r")}
    info["target_ids"] = tgt_ids
    info["target_row_norm"] = [round(float(seed[e].norm()), 2) for e in tgt_ids]
    print(json.dumps(info, ensure_ascii=False), flush=True)
    arms = arm_names(args.label, args.alphas, not args.no_random)
    print(f"arms: {arms}", flush=True)
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "transplant_line",
        label=args.label,
        root=LINE / "experiments" / "transplant_line" / "results",
    )
    metrics: dict = {"direction": info, "arms": {}}
    if "build" in args.legs:
        for name, (which, a) in arms.items():
            rel = build_arm(name, dr[which], a * dr["step"], tgt_ids, SEED_ROWS)
            print(f"built {name}: rel |Δ|/|row| {rel}", flush=True)
            metrics["arms"][name] = {"vec": which, "alpha": a, "rel": rel}
    if "eval" in args.legs:
        chars = [spell(TARGET), *TARGET]
        floor = check_floor(chars)
        print("floor (the seed dir's native_spell/ cache):", flush=True)
        metrics["floor"] = tally(floor, chars, TARGET)
        for name in arms:
            print(f"{name}:", flush=True)
            metrics["arms"].setdefault(name, {})["reads"] = tally(
                native_read(name, chars, "spell"), chars, TARGET
            )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / n) for n in arms],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
