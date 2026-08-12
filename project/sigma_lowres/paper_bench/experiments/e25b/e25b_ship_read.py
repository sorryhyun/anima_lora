"""E25b Stage-2 ship read — computes the frozen gates + verdict rows.

Sources (all already on disk):
- yardstick: runs/20260812-e25b2-yardstick/yardstick.json (renders one boot)
- wall + token_step_hist: the daemon job progress streams (ids in
  e25b_stage2_jobs.json)
- dW cos vs native twin: stage2_dw/dw_<corpus>_<seed>.json
  (compare_ckpt_dw.py outputs, lora_up/down modules only — the res-cond
  projection is not a dW and is excluded by construction)
- CMMD (descriptive, no verdict — the E4 lesson): the per-seed eval
  result.json files
- determinism control: recomputed here (ctrl2 vs combo key bit-identity)

Emits e25b_ship_read.json next to this file. Verdict wording is the frozen
Stage-2 registration's (README), incl. the 2026-08-12 pre-submission
amendment (throughput measured-not-gated; dW = registered secondary that can
corroborate but never carry/veto; PASS = paper-method update, no product
ship).
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
RUNS = REPO / "project/sigma_lowres/paper_bench/runs"
YARD = RUNS / "20260812-e25b2-yardstick/yardstick.json"
EVALS = {s: RUNS / f"20260812-e25b2-eval-sfw-s{s}" for s in (1001, 1002, 1003)}
JOBS = json.loads((HERE / "e25b_stage2_jobs.json").read_text())

ARMS = ("native", "combo", "rescond", "sigma768", "rescond768")
CORPORA = {
    "hews": "hews",
    "channel": "channel_(caststation)",
}  # short -> yardstick cell
SEEDS = (1001, 1002, 1003)


def runend(name: str) -> dict:
    evs = [
        json.loads(line)
        for line in (REPO / f"output/daemon/jobs/{JOBS[name]}/progress.jsonl")
        .read_text()
        .splitlines()
    ]
    return next(e for e in reversed(evs) if e.get("ev") == "run_end")


def main() -> None:
    y = json.loads(YARD.read_text())
    out: dict = {
        "sources": {"yardstick": str(YARD), "jobs": str(HERE / "e25b_stage2_jobs.json")}
    }

    # --- in-batch yardstick + within-seed arm means -------------------------
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

    # gate 4 — yardstick sanity vs recorded 0.9547/0.9541 (±0.02)
    recorded = {"hews": 0.9547, "channel": 0.9541}
    out["gate_yardstick_sanity"] = {
        s: {
            "in_batch": quality[s]["in_batch_yardstick"],
            "recorded": recorded[s],
            "within_0.02": abs(quality[s]["in_batch_yardstick"] - recorded[s]) <= 0.02,
        }
        for s in CORPORA
    }

    # --- 25b-2: quality gate (the only ship gate) + measured throughput -----
    q_pass = all(quality[s]["arms"]["rescond"]["at_or_inside"] for s in CORPORA)
    walls, hists = {}, {}
    for short in CORPORA:
        for seed in SEEDS:
            for arm in ARMS:
                e = runend(f"e25b2_{short}_{arm}_s{seed}")
                walls[(short, arm, seed)] = e["ts"]
                hists[(short, arm, seed)] = e["token_step_hist"]
    r1 = [
        walls[(c, "rescond", s)] / walls[(c, "combo", s)]
        for c in CORPORA
        for s in SEEDS
    ]
    r2 = [
        walls[(c, "rescond768", s)] / walls[(c, "sigma768", s)]
        for c in CORPORA
        for s in SEEDS
    ]
    out["throughput_measured_not_gated"] = {
        "paired_wall_rescond_over_combo": {"mean": statistics.mean(r1), "per_pair": r1},
        "paired_wall_rescond768_over_sigma768": {
            "mean": statistics.mean(r2),
            "per_pair": r2,
        },
        "wall_vs_native_pct": {
            f"{c}_{arm}": statistics.mean(
                walls[(c, arm, s)] / walls[(c, "native", s)] - 1 for s in SEEDS
            )
            for c in CORPORA
            for arm in ARMS[1:]
        },
    }
    out["gate_demote_mass_pair_identity"] = all(
        hists[(c, "combo", s)] == hists[(c, "rescond", s)]
        and hists[(c, "sigma768", s)] == hists[(c, "rescond768", s)]
        for c in CORPORA
        for s in SEEDS
    )
    out["demote_mass_note"] = (
        "realized sigma>0.5 mass 180/480 (37.5%) vs the E16.1-era recorded "
        "244/480 (50.8%) at the same seeds — the sigma draw distribution "
        "shifted between 2026-08-02 and 2026-08-12; in-batch comparisons "
        "unaffected (CRN pair-identity holds), wall deltas vs native are "
        "correspondingly smaller than the recorded -18.3%."
    )
    out["verdict_25b2"] = {
        "quality_gate_pass": q_pass,
        "verdict": "PASS (paper-method update, no product ship)"
        if q_pass
        else "FAIL (quality)",
        "note": "frozen wording: rescond mean within-seed render cos at-or-inside "
        "the in-batch yardstick on BOTH corpora; a near-miss is a miss.",
    }

    # --- 25b-3: rescue read (never gates the ship read) ----------------------
    rescue: dict = {}
    for short, cell_name in CORPORA.items():
        within = y["cells"][cell_name]["within_seed_arm_pairs"]
        diffs = {
            s: within[f"native~rescond768|s{s}"] - within[f"native~sigma768|s{s}"]
            for s in SEEDS
        }
        rescue[short] = {
            "paired_dcos_per_seed": diffs,
            "median": statistics.median(diffs.values()),
        }
    direction = all(rescue[s]["median"] > 0 for s in CORPORA)
    s768_below = [
        s for s in CORPORA if not quality[s]["arms"]["sigma768"]["at_or_inside"]
    ]
    r768_inside = all(quality[s]["arms"]["rescond768"]["at_or_inside"] for s in CORPORA)
    band = (
        "NOT-TESTABLE-AT-THIS-BATCH (sigma768 inside both bands)"
        if not s768_below
        else (
            "RESCUED" if r768_inside and statistics.mean(r2) <= 1.01 else "not rescued"
        )
    )
    out["verdict_25b3"] = {
        "paired": rescue,
        "direction": "RESCUE-DIRECTION" if direction else "NULL",
        "band_claim": band,
        "sigma768_below_band_on": s768_below,
    }

    # --- registered secondary: dW toward native (corroborate-only) ----------
    dw: dict = {}
    for short in CORPORA:
        per_arm: dict = {}
        for seed in SEEDS:
            j = json.loads((HERE / f"stage2_dw/dw_{short}_{seed}.json").read_text())
            for name, pair in j["pairs"].items():
                if "_native_" not in name:
                    continue
                arm = next(
                    (
                        a
                        for a in ("rescond768", "rescond", "sigma768", "combo")
                        if f"_{a}_s" in name
                    ),
                    None,
                )
                if arm:
                    per_arm.setdefault(arm, {})[seed] = pair["cos_global"]
        dw[short] = {
            a: {"per_seed": v, "median": statistics.median(v.values())}
            for a, v in per_arm.items()
        }
    dw_direction = all(
        dw[s]["rescond"]["median"] > dw[s]["combo"]["median"] for s in CORPORA
    )
    out["secondary_dw_toward_native"] = {
        "per_corpus": dw,
        "registered_direction_rescond_gt_combo": dw_direction,
        "note": "corroborate-only; blind to the projection (not a dW); "
        "cross-batch comparison forbidden (combo itself sits ~0.74-0.77 this "
        "batch vs 0.365/0.434 recorded in E16.1 — same sigma-draw shift).",
    }

    # --- CMMD (descriptive, NO verdict — the E4 lesson) ----------------------
    cmmd: dict = {}
    for seed, d in EVALS.items():
        r = json.loads((d / "result.json").read_text())
        for artist, cell in r["cells"].items():
            for arm, v in cell["arms"].items():
                cmmd.setdefault(artist, {}).setdefault(arm, {})[seed] = v[
                    "cmmd_holdout"
                ]
    out["cmmd_descriptive_no_verdict"] = cmmd

    # --- determinism control (gate 2), recomputed ---------------------------
    import torch
    from safetensors.torch import load_file

    a = load_file(str(REPO / "output/ckpt/e25b2_hews_combo_s1001.safetensors"))
    b = load_file(str(REPO / "output/ckpt/e25b2_hews_combo_s1001_ctrl2.safetensors"))
    out["gate_determinism_control"] = {
        "keys_equal": set(a) == set(b),
        "differing_keys": sum(0 if torch.equal(a[k], b[k]) else 1 for k in a),
    }

    (HERE / "e25b_ship_read.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["verdict_25b2"], indent=1))
    print(json.dumps(out["verdict_25b3"], indent=1))
    for short in CORPORA:
        bar = out["quality"][short]["in_batch_yardstick"]
        row = {a: round(out["quality"][short]["arms"][a]["mean"], 4) for a in ARMS[1:]}
        print(short, "yardstick", round(bar, 4), row)
    sys.exit(0)


if __name__ == "__main__":
    main()
