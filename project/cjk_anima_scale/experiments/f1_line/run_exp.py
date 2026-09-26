#!/usr/bin/env python
"""f1_line — proposal_factorizedrows.md § 3 (Stage F1): a factorized donor on
Stage B's data (2026-09-26)

F0 found the adapter blind to a row's neighbours, so a line mode baked into a
row reaches the DiT alone too (doubling A), and a gate has to act at the
hook. Here the Stage B donor is retrained with the rows ``r_i`` **plus one
shared ``v_line``** that the hook adds only to pack rows in a run of ≥ 2
(``ExtDelta.line``). Same data dir (36 donors, 568 words, count tier 0.3),
same seed rows, same trainer and 90 steps / row. The count tier's singles
are the gate-off items that should push the line behaviour out of ``r_i``.

Legs:
  train     (GPU) the factorized donor → ``run0926_f1_line/trained.pt``
  build     (CPU) ``v_line`` vs Stage B's ``u_S``, how much ``u_S`` stays in
            the rows; arms ``tf_<label>_ronly`` (the rows, no ``v_line``) and
            ``tf_<label>_line`` (the held-out 10's seed rows + ``v_line``,
            gated — the transplant without extraction)
  read      (GPU) the donor keys (en): F1 as trained (gate live) and
            ``ronly``, vs the floor and the plain Stage B donor (on disk)
  transplant (GPU) the held-out keys (en + swap) on ``tf_<label>_line`` vs
            the floor and Stage B's ``tb_t1_u1`` (on disk)
  dose      (GPU) ``tf_<label>_line<d>`` = the seed rows + d · ``v_line``
            (``--doses``, from ``run0926_f1_line``), built and read on the
            held-out keys like ``transplant`` — does in-word ``dup`` fall
            back toward u1's while composition holds? (report § 4)

Reads of record to beat (proposal § 3): donor singles alone official ·
repeat / 144 floor 91 · 29, plain donor 43 · 50; こんにちは ≤ 2 edits plain
13 / 16, ``dup`` 13 / 16; held-out words ≤ 1 edit u1 66 / 160.

``--dry_run`` prints the plan and checks the encodings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage_b_exp", LINE / "experiments" / "stage_b" / "run_exp.py"
)
SB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SB)

EXP = OUT / "experiments"
NAME = "run0926_f1_line"
DONOR_DATA = OUT / SB.NAME / "data"
PLAIN = OUT / SB.NAME  # the plain Stage B donor (its native_spell/ reads)
U1 = EXP / "tb_t1_u1"  # Stage B's post-hoc transplant (its native_spell/ reads)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["train"],
        choices=["train", "build", "read", "transplant", "dose"],
    )
    p.add_argument("--doses", type=float, nargs="+", default=[0.5])
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def rc():
    from cjk_scale.config import RunConfig

    return RunConfig(
        name=NAME, path=Path(__file__), vocabs=(f"chars:{SB.DONOR}",), read=()
    )


def arm_names(label: str) -> dict:
    return {"ronly": f"tf_{label}_ronly", "line": f"tf_{label}_line"}


def build(ids: dict, label: str) -> dict:
    """``v_line`` against ``u_S``, what of ``u_S`` the rows kept, and the two
    arms."""
    import torch
    import torch.nn.functional as F

    sd = torch.load(OUT / NAME / "trained.pt", map_location="cpu", weights_only=False)
    d = sd["delta"]
    assert d.get("line") is not None, "the F1 rows carry no v_line"
    rs = float(d["row_scale"])
    v = d["line"].float() * rs  # effective units
    seed, seed_sd = SB.rows(SEED_ROWS)
    f1, _ = SB.rows(OUT / NAME / "trained.pt")
    plain, _ = SB.rows(PLAIN / "trained.pt")
    dr = SB.direction(ids)  # u_S, from the plain donor
    u = dr["u"]
    d_ids = [ids[c] for c in SB.DONOR]
    info = {
        "v_line_norm": round(float(v.norm()), 2),
        "v_line_cos_uS": round(float(F.normalize(v, dim=0) @ u), 4),
        "uS_step": round(dr["step"], 2),
        "v_line_proj_uS": round(float(v @ u), 2),
    }
    for name, R in (("f1", f1), ("plain", plain)):
        T = SB.tangential(R, seed, d_ids)
        P = T @ u
        info[f"{name}_rows"] = {
            "rel_drift": round(
                sum(float((R[e] - seed[e]).norm() / seed[e].norm()) for e in d_ids)
                / len(d_ids),
                3,
            ),
            "uS_proj_mean": round(float(P.mean()), 2),
            "uS_energy_frac": round(float((P**2).sum() / (T**2).sum()), 4),
            "mean_dir_cos_uS": round(float(F.normalize(T.mean(0), dim=0) @ u), 4),
        }
        if name == "f1":
            info["f1_rows"]["mean_dir_cos_v"] = round(
                float(F.normalize(T.mean(0), dim=0) @ F.normalize(v, dim=0)), 4
            )
    names = arm_names(label)
    # ronly: the F1 rows, the mode dropped
    dst = EXP / names["ronly"]
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(
        {**sd, "delta": {k: x for k, x in d.items() if k != "line"}},
        dst / "trained.pt",
    )
    line_arm(names["line"], 1.0)
    return info


def line_arm(name: str, dose: float) -> None:
    """The seed rows (the held-out 10 untouched) + ``dose`` · ``v_line``, in
    the seed's row units."""
    import torch

    d = torch.load(OUT / NAME / "trained.pt", map_location="cpu", weights_only=False)[
        "delta"
    ]
    seed_sd = torch.load(SEED_ROWS, map_location="cpu", weights_only=False)
    sdd = seed_sd["delta"]
    k = float(d["row_scale"]) / float(sdd["row_scale"])
    dst = EXP / name
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **seed_sd,
            "delta": {**sdd, "line": d["line"].float() * k * dose},
            "arm": "rows",
            "seed_merged": str(SEED_ROWS),
            "transplant": {"line_from": NAME, "rescale": k, "dose": dose},
        },
        dst / "trained.pt",
    )


def read_held(metrics: dict, key: str, path: Path, fh: dict, uh: dict) -> None:
    chars = SB.held_keys()
    print(f"{path.name}, held-out keys:", flush=True)
    h = SB.hits(SB.native_read(path, path / "data", chars, "en,swap"), chars, "en,swap")
    metrics[key] = SB.tally(h)
    metrics[key]["paired_vs_floor"] = SB.paired(h, fh)
    metrics[key]["paired_vs_u1"] = SB.paired(h, uh)
    print(
        f"  vs floor {metrics[key]['paired_vs_floor']}\n"
        f"  vs u1 {metrics[key]['paired_vs_u1']}",
        flush=True,
    )


def main():
    args = parse_args()
    words = json.loads((OUT / SB.NAME / "donor_words.json").read_text("utf-8"))
    SB.set_words(words)
    ids = SB.check_spelling(SB.encoder(), words)
    names = arm_names(args.label)
    print(
        f"{NAME}: data {DONOR_DATA} ({len(words)} donor words), arms {names}",
        flush=True,
    )
    metrics: dict = {"name": NAME, "data": str(DONOR_DATA), "arms": names}
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "f1_line", label=args.label, root=LINE / "experiments" / "f1_line" / "results"
    )
    if "train" in args.legs:
        from cjk_scale.train import train

        train(rc(), data=DONOR_DATA, out=OUT / NAME, line_mode=True)
    if "build" in args.legs:
        info = build(ids, args.label)
        print(json.dumps(info, ensure_ascii=False), flush=True)
        metrics["build"] = info
    if "read" in args.legs:
        chars = SB.donor_keys()
        floor = SB.check_floor(chars, "en")
        fh = SB.hits(floor, chars, "en")
        print("floor, donor keys:", flush=True)
        metrics["donor_floor"] = SB.tally(fh)
        ph = SB.hits(PLAIN / f"native_{SB.TAG}" / "native_reads.json", chars, "en")
        print("plain Stage B donor, donor keys:", flush=True)
        metrics["donor_plain"] = SB.tally(ph)
        for arm, path in (("f1", OUT / NAME), ("ronly", EXP / names["ronly"])):
            print(f"{arm}, donor keys:", flush=True)
            h = SB.hits(SB.native_read(path, path / "data", chars, "en"), chars, "en")
            metrics[f"donor_{arm}"] = SB.tally(h)
            metrics[f"donor_{arm}"]["paired_vs_floor"] = SB.paired(h, fh)
            metrics[f"donor_{arm}"]["paired_vs_plain"] = SB.paired(h, ph)
            print(
                f"  vs floor {metrics[f'donor_{arm}']['paired_vs_floor']}\n"
                f"  vs plain {metrics[f'donor_{arm}']['paired_vs_plain']}",
                flush=True,
            )
    if {"transplant", "dose"} & set(args.legs):
        chars = SB.held_keys()
        floor = SB.check_floor(chars, "en,swap")
        fh = SB.hits(floor, chars, "en,swap")
        print("floor, held-out keys:", flush=True)
        metrics["held_floor"] = SB.tally(fh)
        uh = SB.hits(U1 / f"native_{SB.TAG}" / "native_reads.json", chars, "en,swap")
        print("tb_t1_u1 (post-hoc u_S), held-out keys:", flush=True)
        metrics["held_u1"] = SB.tally(uh)
    if "transplant" in args.legs:
        read_held(metrics, "held_line", EXP / names["line"], fh, uh)
    if "dose" in args.legs:
        for dose in args.doses:
            name = f"{names['line']}{dose:g}"
            line_arm(name, dose)
            names[f"line{dose:g}"] = name
            read_held(metrics, f"held_line{dose:g}", EXP / name, fh, uh)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / NAME), *(str(EXP / n) for n in names.values())],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
