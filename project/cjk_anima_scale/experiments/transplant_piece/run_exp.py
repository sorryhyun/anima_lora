#!/usr/bin/env python
"""transplant_piece — proposal.md Stage A: a piece leave-out transplant (2026-09-26)

Every read of the shared "line" Δ so far is in-sample (transplant_line § 4:
``uB`` and each row's coefficient fit on the same five rows). This one is
out of sample. The 8 pieces the piece ruler reads
(``あと きて こう こと こんにちは しい ちょっと った``) are among
run0926_300f_sp's 300 trained pieces, and the floor cache holds their
``native_piece`` keys.

Direction ``u_P`` = the normalized mean of the **other 292** pieces' Δ vs
the seed, each Δ with its component along its own seed row removed (as in
transplant_line § 1). Step = their mean projection on ``u_P`` — one
coefficient for every row, no per-row fit. Arms: the seed rows of the 8
held-out pieces + α · step · ``u_P`` (``--alphas``, default 0.5 1), and a
random unit direction ⟂ ``u_P`` at the largest α (the norm control). Every
other row is the seed's.

Read: the piece ruler (native, en + swap, the 8 alone, 8 prompts × 2 seeds)
against the floor cache's ``native_piece/`` and 300f_sp's own reads on disk
(floor 3 / 256 official, 300f_sp 16 / 256, contained 27 → 76). Decision
(proposal § 1): ≥ half of 300f_sp's gain over the floor and above the random
arm → a generic piece direction exists; at the floor → the line mode is per
row. Leak: the 292 trained on items that sometimes carried the 8 next to
their own vocab (``--dry_run`` counts them) — a positive read is an upper
bound.

legs: build (CPU: the arms' ``trained.pt`` under
``output/cjk_anima_scale/<arm>/``), eval (GPU, through ``make daemon-run``).
``--dry_run`` prints the direction, the step, the leak and the encodings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap, floor_dir  # noqa: E402

bootstrap()
EXP = OUT / "experiments"  # the row arms (a trained.pt each)

from bench._common import make_run_dir, write_result  # noqa: E402

SRC_RUN = "run0926_300f_sp"
HELD = ("あと", "きて", "こう", "こと", "こんにちは", "しい", "ちょっと", "った")
CLAUSES = ("en", "swap")
TAG = "piece"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--legs", nargs="+", default=["build"], choices=["build", "eval"])
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0])
    p.add_argument("--no_random", action="store_true", help="skip the control arm")
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


def rows(path: Path) -> tuple[dict, dict]:
    """``{ext id: effective row}`` (raw × row_scale) and the file itself."""
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)
    d = sd["delta"]
    s = float(d["row_scale"])
    return {int(e): d["raw"][i].float() * s for i, e in enumerate(d["ext_ids"])}, sd


def tangential(run_rows: dict, seed: dict, ids: list[int]):
    """Δ vs the seed for ``ids``, each with its component along its own seed
    row removed."""
    import torch
    import torch.nn.functional as F

    D = torch.stack([run_rows[e] - seed[e] for e in ids])
    S = F.normalize(torch.stack([seed[e] for e in ids]), dim=1)
    return D - (D * S).sum(1, keepdim=True) * S


def piece_ids(ext, run_vocabs: list[str]) -> dict[str, int]:
    """Each run vocab's single ext id (a piece is one Qwen token → one row)."""
    out = {}
    for v in run_vocabs:
        got = ext(v)
        assert len(got) == 1, (v, got)
        out[v] = got[0]
    return out


def direction(seed: dict, src: dict, held_ids: list[int], donor_ids: list[int]):
    import torch
    import torch.nn.functional as F

    T = tangential(src, seed, donor_ids)
    u = F.normalize(T.mean(0), dim=0)
    TH = tangential(src, seed, held_ids)
    T1, T2 = T[0::2], T[1::2]
    g = torch.Generator().manual_seed(0)
    r = torch.randn(u.shape[0], generator=g)
    r = F.normalize(r - (r @ u) * u, dim=0)
    uH = F.normalize(TH.mean(0), dim=0)
    return {
        "u": u,
        "r": r,
        "step": float((T @ u).mean()),
        "donor_rows": len(T),
        "donor_proj_sd": round(float((T @ u).std()), 3),
        "donor_energy_frac": round(float(((T @ u) ** 2).sum() / (T**2).sum()), 4),
        "split_half_cos": round(
            float(F.normalize(T1.mean(0), dim=0) @ F.normalize(T2.mean(0), dim=0)), 4
        ),
        # the held-out 8 as 300f_sp trained them, seen from u_P (not used by the arms)
        "held_own_proj": [round(float(x), 2) for x in TH @ u],
        "held_own_dnorm": [round(float(x), 2) for x in TH.norm(dim=1)],
        "held_own_energy_frac": round(float(((TH @ u) ** 2).sum() / (TH**2).sum()), 4),
        "cos_u_uHeld": round(float(u @ uH), 4),
    }


def leak(held: tuple[str, ...]) -> dict:
    """300f_sp's training items that carry a held-out piece next to another
    vocab (the 292's gradient saw the 8 there)."""
    f = OUT / SRC_RUN / "data" / "train.jsonl"
    n = multi = with_held = 0
    for line in f.read_text(encoding="utf-8").splitlines():
        it = json.loads(line)
        units = it.get("units") or [it["text"]]
        n += 1
        if len(units) > 1:
            multi += 1
            with_held += any(u in held for u in units) and any(
                u not in held for u in units
            )
    return {"items": n, "multi_unit": multi, "held_beside_donor": with_held}


def arm_names(label: str, alphas, random: bool) -> dict:
    out = {f"tp_{label}_u{a:g}": ("u", a) for a in alphas}
    if random:
        out[f"tp_{label}_rand{max(alphas):g}"] = ("r", max(alphas))
    return out


def build_arm(name: str, vec, step: float, tgt_ids: list[int]) -> dict:
    """The seed's trained.pt with ``step · vec`` added to the target rows."""
    import torch

    sd = torch.load(SEED_ROWS, map_location="cpu", weights_only=False)
    d = sd["delta"]
    s = float(d["row_scale"])
    idx = {int(e): i for i, e in enumerate(d["ext_ids"])}
    raw = d["raw"].clone()
    moved = {}
    for e in tgt_ids:
        i = idx[e]
        before = raw[i].float() * s
        after = before + step * vec
        raw[i] = (after / s).to(raw.dtype)
        moved[e] = {
            "rel": round(float(step * vec.norm() / before.norm()), 3),
            "norm_ratio": round(float(after.norm() / before.norm()), 3),
        }
    out = {
        **sd,
        "delta": {**d, "raw": raw},
        "arm": "rows",
        "seed_merged": str(SEED_ROWS),
        "transplant": {"step": step, "rows": tgt_ids, "moved": moved},
    }
    dst = EXP / name
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst / "trained.pt")
    return moved


def native_read(name: str) -> Path:
    from cjk_scale.config import RunConfig
    from cjk_scale.eval import TRAINED_ARM, probe_args
    from stages import run as run_stage

    rc = RunConfig(name=name, path=Path(__file__), vocabs=(), read=())
    reads = EXP / name / f"native_{TAG}" / "native_reads.json"
    if reads.exists():
        held = {(m["text"], m["clause"]) for m in json.loads(reads.read_text("utf-8"))}
        if all((k, c) in held for k in HELD for c in CLAUSES):
            print(f"  (read from disk: {reads})", flush=True)
            return reads
    a = probe_args(
        rc,
        TRAINED_ARM,
        ["native"],
        [
            "--eval_tag",
            TAG,
            "--native_chars",
            ",".join(HELD),
            "--native_clauses",
            ",".join(CLAUSES),
        ],
    )
    a.arm_path, a.data_path = str(EXP / name), str(EXP / name / "data")
    run_stage("native", a)
    return reads


def tally(path: Path) -> dict:
    """Per piece × clause and summed: official (sfx ∧ VL hit), loose,
    contained (the piece inside any box read), repeat (inside a read ≥ 2×)."""
    from common.readers import norm

    out: dict = {}
    tot = {"n": 0, "official": 0, "loose": 0, "contained": 0, "repeat": 0}
    for m in json.loads(path.read_text("utf-8")):
        if m["text"] not in HELD or m["clause"] not in CLAUSES:
            continue
        t = norm(m["text"])
        reads = [
            norm(r.get(x) or "") for r in m.get("reads", []) for x in ("sfx", "vl")
        ]
        hit = {
            "n": 1,
            "official": bool(m.get("hit_sfx")) and bool(m.get("hit_vl")),
            "loose": bool(m.get("hit_sfx")) or bool(m.get("hit_vl")),
            "contained": any(t in r for r in reads),
            "repeat": any(r.count(t) >= 2 for r in reads),
        }
        c = out.setdefault(f"{m['text']}|{m['clause']}", dict.fromkeys(tot, 0))
        for k, v in hit.items():
            c[k] += v
            tot[k] += v
    for k, c in sorted(out.items()):
        print(
            f"  {k:<14} official {c['official']:>2}  loose {c['loose']:>2}  "
            f"contained {c['contained']:>2}  repeat {c['repeat']:>2} / {c['n']}",
            flush=True,
        )
    print(
        f"  {'TOTAL':<14} official {tot['official']:>3}  loose {tot['loose']:>3}  "
        f"contained {tot['contained']:>3}  repeat {tot['repeat']:>3} / {tot['n']}",
        flush=True,
    )
    return {"per_key": out, "total": tot}


def check_floor() -> Path:
    floor = floor_dir() / f"native_{TAG}" / "native_reads.json"
    held = {(m["text"], m["clause"]) for m in json.loads(floor.read_text("utf-8"))}
    miss = [(k, c) for k in HELD for c in CLAUSES if (k, c) not in held]
    assert not miss, f"floor cache lacks {miss} — no floor renders here"
    return floor


def main():
    args = parse_args()
    ext = encoder()
    src, _ = rows(OUT / SRC_RUN / "trained.pt")
    seed, _ = rows(SEED_ROWS)
    vocabs = json.loads((OUT / SRC_RUN / "data" / "vocabs.json").read_text("utf-8"))
    assert all(h in vocabs for h in HELD), [h for h in HELD if h not in vocabs]
    ids = piece_ids(ext, vocabs)
    held_ids = [ids[h] for h in HELD]
    donor_ids = [ids[v] for v in vocabs if v not in HELD]
    assert all(e in seed and e in src for e in ids.values()), "a vocab lacks a row"
    dr = direction(seed, src, held_ids, donor_ids)
    info = {k: v for k, v in dr.items() if k not in ("u", "r")}
    info["held"] = dict(zip(HELD, held_ids))
    info["held_seed_norm"] = [round(float(seed[e].norm()), 2) for e in held_ids]
    info["leak"] = leak(HELD)
    print(json.dumps(info, ensure_ascii=False), flush=True)
    arms = arm_names(args.label, args.alphas, not args.no_random)
    print(f"arms: {arms}", flush=True)
    floor = check_floor()
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "transplant_piece",
        label=args.label,
        root=LINE / "experiments" / "transplant_piece" / "results",
    )
    metrics: dict = {"direction": info, "arms": {}}
    if "build" in args.legs:
        for name, (which, a) in arms.items():
            moved = build_arm(name, dr[which], a * dr["step"], held_ids)
            print(f"built {name}: {json.dumps(moved)}", flush=True)
            metrics["arms"][name] = {"vec": which, "alpha": a, "moved": moved}
    if "eval" in args.legs:
        print("floor (the seed dir's native_piece/ cache):", flush=True)
        metrics["floor"] = tally(floor)
        print(f"{SRC_RUN} (on disk):", flush=True)
        metrics["src"] = tally(OUT / SRC_RUN / f"native_{TAG}" / "native_reads.json")
        for name in arms:
            print(f"{name}:", flush=True)
            metrics["arms"].setdefault(name, {})["reads"] = tally(native_read(name))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(EXP / n) for n in arms],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
