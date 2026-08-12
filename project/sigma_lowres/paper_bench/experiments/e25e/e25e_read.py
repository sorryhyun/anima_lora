"""E25e read — gates + 25e-1 (mechanism) + 25e-2 (ship) + descriptives.

Frozen registration (README 2026-08-12). Sources (all on disk):
- yardstick: runs/20260812-e25e-yardstick/yardstick.json (renders one boot)
- wall + token_step_hist: daemon job progress streams (ids in e25e_jobs.json)
- dW cos vs native twin: stage2_dw/dw_<corpus>_<seed>.json
  (compare_ckpt_dw.py, lora_up/down modules only — the res-cond projection
  is not a dW and is excluded by construction)
- learned differential ||W.(phi(s) - phi(0))||: read from the rescond_c
  checkpoints directly (the registered descriptive the 25b read failed to
  emit — emitted this time)
- CMMD (descriptive, no verdict): the per-seed eval result.json files
- determinism control: recomputed here (ctrl2 vs combo key bit-identity)

25e-1 (judgment constants 0.05 / 0.15, frozen): per corpus, median over
seeds of in-batch dW cos(arm ~ native twin).
  median(rescond_c) >= median(combo) - 0.05  -> COMMON-MODE-CONFIRMED
  median(rescond_c) <= median(combo) - 0.15  -> COLLAPSE-PERSISTS
  otherwise / corpora disagree               -> NULL-MIXED

25e-2 (the 25b-2 bar verbatim): rescond_c mean within-seed render cos vs
native twin at-or-inside the in-batch yardstick on BOTH corpora; near-miss
= miss. Throughput measured, not gated.

Emits e25e_read.json next to this file.

Usage::

    uv run python project/sigma_lowres/paper_bench/experiments/e25e/e25e_read.py
"""

from __future__ import annotations

import glob
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
RUNS = REPO / "project/sigma_lowres/paper_bench/runs"
YARD = RUNS / "20260812-e25e-yardstick/yardstick.json"
EVALS = {s: RUNS / f"20260812-e25e-eval-sfw-s{s}" for s in (1001, 1002, 1003)}
JOBS = json.loads((HERE / "e25e_jobs.json").read_text())

ARMS = ("native", "combo", "rescond_c")
CORPORA = {"hews": "hews", "channel": "channel_(caststation)"}
SEEDS = (1001, 1002, 1003)
YARD_RECORDED = {"hews": 0.9578, "channel": 0.9541}
DELTA_CONFIRM, DELTA_COLLAPSE = 0.05, 0.15
S896 = math.log2(896 / 1024)
S768 = math.log2(768 / 1024)


def runend(name: str) -> dict:
    evs = [
        json.loads(line)
        for line in (REPO / f"output/daemon/jobs/{JOBS[name]}/progress.jsonl")
        .read_text()
        .splitlines()
    ]
    return next(e for e in reversed(evs) if e.get("ev") == "run_end")


def step2_loss(run: str) -> float:
    d = sorted(glob.glob(str(REPO / f"output/logs/{run}_2026*")))[-1]
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts/export_logs_json.py"), d, "--stdout"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = json.loads(out.stdout)["tags"]["loss/current"]
    return next(r[2] for r in rows if r[0] == 2)


def main() -> None:
    out: dict = {
        "doc": "E25e read — frozen 2026-08-12 registration",
        "sources": {"yardstick": str(YARD), "jobs": str(HERE / "e25e_jobs.json")},
    }

    # --- gate 1: twin-start identity (earliest in-grid TB record = step 2;
    # step-1 identity is the Stage-0 structural invariant, pinned in
    # tests/test_sigma_lowres.py::TestResCondCentered, not logged in-grid) ---
    g1 = {}
    for c in CORPORA:
        for s in SEEDS:
            a = step2_loss(f"e25e_{c}_combo_s{s}")
            b = step2_loss(f"e25e_{c}_rescond_c_s{s}")
            g1[f"{c}_s{s}"] = {"combo": a, "rescond_c": b, "identical": a == b}
    out["gate1_twin_start_identity"] = {
        "record": "TB loss/current at step 2 (earliest logged step)",
        "cells": g1,
        "pass": all(v["identical"] for v in g1.values()),
    }

    # --- gate 2: determinism control ----------------------------------------
    import torch
    from safetensors.torch import load_file

    a = load_file(str(REPO / "output/ckpt/e25e_hews_combo_s1001.safetensors"))
    b = load_file(str(REPO / "output/ckpt/e25e_hews_combo_s1001_ctrl2.safetensors"))
    out["gate2_determinism_control"] = {
        "keys_equal": set(a) == set(b),
        "differing_keys": sum(0 if torch.equal(a[k], b[k]) else 1 for k in a),
        "pass": set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a),
    }

    # --- gate 3: demote-mass pair-identity ----------------------------------
    walls, hists = {}, {}
    for c in CORPORA:
        for s in SEEDS:
            for arm in ARMS:
                e = runend(f"e25e_{c}_{arm}_s{s}")
                walls[(c, arm, s)] = e["ts"]
                hists[(c, arm, s)] = e["token_step_hist"]
    out["gate3_demote_mass_pair_identity"] = {
        "pass": all(
            hists[(c, "combo", s)] == hists[(c, "rescond_c", s)]
            for c in CORPORA
            for s in SEEDS
        )
    }

    # --- yardstick + gate 4 -------------------------------------------------
    y = json.loads(YARD.read_text())
    quality: dict = {}
    for short, cell_name in CORPORA.items():
        cell = y["cells"][cell_name]
        cross = cell["cross_seed_same_arm"]
        native_lottery = [v for k, v in cross.items() if k.startswith("native|")]
        bar = statistics.mean(native_lottery)
        within = cell["within_seed_arm_pairs"]
        arms = {}
        for arm in ARMS[1:]:
            per_seed = {s: within[f"native~{arm}|s{s}"] for s in SEEDS}
            arms[arm] = {
                "per_seed": per_seed,
                "mean": statistics.mean(per_seed.values()),
                "at_or_inside": statistics.mean(per_seed.values()) >= bar,
                "margin": statistics.mean(per_seed.values()) - bar,
            }
        quality[short] = {
            "in_batch_yardstick": bar,
            "native_lottery_pairs": native_lottery,
            "arms": arms,
        }
    out["quality"] = quality
    out["gate4_yardstick_sanity"] = {
        s: {
            "in_batch": quality[s]["in_batch_yardstick"],
            "recorded": YARD_RECORDED[s],
            "within_0.02": abs(quality[s]["in_batch_yardstick"] - YARD_RECORDED[s])
            <= 0.02,
        }
        for s in CORPORA
    }
    gates_pass = (
        out["gate1_twin_start_identity"]["pass"]
        and out["gate2_determinism_control"]["pass"]
        and out["gate3_demote_mass_pair_identity"]["pass"]
        and all(v["within_0.02"] for v in out["gate4_yardstick_sanity"].values())
    )
    out["gates_all_pass"] = gates_pass

    # --- 25e-1: mechanism read (primary) ------------------------------------
    dw: dict = {}
    for c in CORPORA:
        per_arm: dict = {}
        for s in SEEDS:
            j = json.loads((HERE / f"stage2_dw/dw_{c}_{s}.json").read_text())
            for name, pair in j["pairs"].items():
                if "_native_" not in name.split("~")[0]:
                    continue
                arm = next(
                    (a for a in ("rescond_c", "combo") if f"_{a}_s" in name), None
                )
                if arm:
                    per_arm.setdefault(arm, {})[s] = pair["cos_global"]
        dw[c] = {
            a: {"per_seed": v, "median": statistics.median(v.values())}
            for a, v in per_arm.items()
        }
    per_corpus_verdict = {}
    for c in CORPORA:
        d = dw[c]["rescond_c"]["median"] - dw[c]["combo"]["median"]
        if d >= -DELTA_CONFIRM:
            v = "COMMON-MODE-CONFIRMED"
        elif d <= -DELTA_COLLAPSE:
            v = "COLLAPSE-PERSISTS"
        else:
            v = "NULL"
        per_corpus_verdict[c] = {"median_delta": d, "verdict": v}
    vs = {v["verdict"] for v in per_corpus_verdict.values()}
    verdict_25e1 = vs.pop() if len(vs) == 1 else "NULL-MIXED"
    out["verdict_25e1"] = {
        "per_corpus_dw": dw,
        "per_corpus": per_corpus_verdict,
        "judgment_constants": {"confirm": -DELTA_CONFIRM, "collapse": -DELTA_COLLAPSE},
        "context_25b_not_compared": "25b batch: combo 0.74-0.77, rescond 0.41-0.47",
        "verdict": verdict_25e1,
        "note": "standing limit: dW closeness != render closeness (measured, "
        "twice); this row reads the mechanism, not the ship question.",
    }

    # --- 25e-2: ship read (the 25b-2 bar verbatim) --------------------------
    q_pass = all(quality[s]["arms"]["rescond_c"]["at_or_inside"] for s in CORPORA)
    r1 = [
        walls[(c, "rescond_c", s)] / walls[(c, "combo", s)]
        for c in CORPORA
        for s in SEEDS
    ]
    out["throughput_measured_not_gated"] = {
        "paired_wall_rescond_c_over_combo": {
            "mean": statistics.mean(r1),
            "per_pair": r1,
        },
        "wall_vs_native_pct": {
            f"{c}_{arm}": statistics.mean(
                walls[(c, arm, s)] / walls[(c, "native", s)] - 1 for s in SEEDS
            )
            for c in CORPORA
            for arm in ARMS[1:]
        },
    }
    out["verdict_25e2"] = {
        "quality_gate_pass": q_pass,
        "verdict": "PASS (paper-method update, no product ship)"
        if q_pass
        else "FAIL (quality)",
        "note": "frozen wording: rescond_c mean within-seed render cos "
        "at-or-inside the in-batch yardstick on BOTH corpora; near-miss = miss.",
    }

    # --- descriptives (no verdict weight) -----------------------------------
    from safetensors import safe_open

    from library.anima.models import sigma_lowres_res_cond_delta

    ts = torch.zeros(1, 1)
    diff_norms = {}
    for c in CORPORA:
        for s in SEEDS:
            name = f"e25e_{c}_rescond_c_s{s}"
            with safe_open(
                str(REPO / f"output/ckpt/{name}.safetensors"), framework="pt"
            ) as f:
                proj = f.get_tensor("sigma_lowres_res_cond_proj").float()
            row = {
                "proj_norm": proj.norm().item(),
                "norm_delta_896": sigma_lowres_res_cond_delta(
                    proj, S896, ts, centered=True
                )
                .norm()
                .item(),
                "norm_delta_768": sigma_lowres_res_cond_delta(
                    proj, S768, ts, centered=True
                )
                .norm()
                .item(),
            }
            diff_norms[name] = {k: round(v, 5) for k, v in row.items()}
    cmmd: dict = {}
    for s, d in EVALS.items():
        r = json.loads((d / "result.json").read_text())
        for artist, cell in r["cells"].items():
            for arm, v in cell["arms"].items():
                cmmd.setdefault(artist, {}).setdefault(arm, {})[s] = v["cmmd_holdout"]
    out["descriptive"] = {
        "learned_differential_norms": {
            "note": "||W.(phi(s)-phi(0))|| per rescond_c checkpoint — the "
            "trained lever when the differential is the ONLY channel "
            "(25b prior: differential 0.005-0.024 while common-mode ate ~90%)",
            "per_ckpt": diff_norms,
        },
        "cmmd_holdout_no_verdict": cmmd,
        "cross_grid_determinism": "e25e native/combo checkpoints bit-identical "
        "to the e25b2 grid's (0/1092 keys differ, hews+combo s1001 checked) — "
        "same boot, same argv; the shared arms replicate exactly across grids.",
    }

    (HERE / "e25e_read.json").write_text(json.dumps(out, indent=1, default=str))
    print("gates_all_pass:", gates_pass)
    print(json.dumps(out["verdict_25e1"]["per_corpus"], indent=1))
    print("25e-1 verdict:", out["verdict_25e1"]["verdict"])
    print(json.dumps(out["verdict_25e2"], indent=1))
    for c in CORPORA:
        bar = quality[c]["in_batch_yardstick"]
        row = {a: round(quality[c]["arms"][a]["mean"], 4) for a in ARMS[1:]}
        print(c, "yardstick", round(bar, 4), row)
    print("wrote", HERE / "e25e_read.json")


if __name__ == "__main__":
    main()
