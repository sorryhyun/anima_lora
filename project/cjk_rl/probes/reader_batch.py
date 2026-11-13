#!/usr/bin/env python
"""reader_batch — the GRPO reward reader before P1 (``project/cjk_rl/proposal.md``; report ``reports/reward_reader_2026_11_13.md``).

Re-reads P0's 640 renders (EN ref + four samplers, 16 strings × 8 seeds) with
``Readers.read_images``: a group's 8 renders read in shared batches, no whole-page
read. Against P0's reads of record (one page per call, whole page included):

    agreement  detector boxes equal; per box the read strings equal
    gate       paired r_seq and g_p, P0's reads with the whole page stripped
               against the batched reads: batching qualifies when neither shifts
               (sign test p ≥ 0.05), fixed before the run
    whole      r_seq with P0's whole page against without it (a reward definition
               change, reported, not gated)
    time       s a group (8 renders) per stage, batched against one-page-a-call
               with and without the whole page (the first 64 euler renders)
    vram       allocated after load, peak while reading

Its run (20261113-1657) read stock VL16 beside the tuned reader; ``Readers`` now
reads the tuned one alone, so a re-run compares that reader only.

int8 / fp8 copies of the two PaddleOCR-VL-1.6 readers were dropped before this
probe: the tower's K (1152, 4304) is not a multiple of ConvRot's 256-column group,
and at decode M every int8 / fp8 linear measured slower than bf16 (11-13).

    make daemon-run ARGS="--stall-timeout 600 project/cjk_rl/probes/reader_batch.py"
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_rl
sys.path.insert(0, str(LINE.parent / "cjk_anima_reseed"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grpo_p0 import P0, SAMPLERS, SEEDS, load_pool, r_seq, render_file  # noqa: E402

KINDS = ("en", *SAMPLERS)
SERIAL_N = 64


def strip_whole(reads: list) -> list:
    return [r for r in reads if not r.get("whole")] + [
        r | {"sfx": None, "sfx_conf": None, "vl": None} for r in reads if r.get("whole")
    ]


def paired(a: list, b: list) -> dict:
    from eval.ruler.stats import sign_p

    d = [y - x for x, y in zip(a, b, strict=True)]
    up, dn = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
    return {
        "a": round(sum(a) / len(a), 4),
        "b": round(sum(b) / len(b), 4),
        "mean_diff": round(sum(d) / len(d), 4),
        "up": up,
        "down": dn,
        "p": sign_p(up, dn),
    }


def main():
    import torch

    from bench._common import make_run_dir, write_result
    from common.readers import Readers, load_bgr
    from eval.ruler.score import score_page

    its = load_pool()
    torch.cuda.reset_peak_memory_stats()
    rd = Readers("cuda")
    vram_load = torch.cuda.memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()

    # one page a call, the first SERIAL_N euler renders, with and without the whole page
    serial = {}
    files = [render_file("euler", m["i"], s) for m in its for s in SEEDS][:SERIAL_N]
    for whole in (True, False):
        times: dict = {}
        t0 = time.time()
        for f in files:
            rd.read_images([load_bgr(f)], whole=whole, times=times)
        serial["whole" if whole else "boxes"] = {
            "s_per_group": round((time.time() - t0) / len(files) * 8, 2),
            **{k: round(v / len(files) * 8, 2) for k, v in times.items()},
        }
        print(
            f"serial whole={whole}: {serial['whole' if whole else 'boxes']}", flush=True
        )

    # a group's 8 renders a call, no whole page
    new, times, t0, n = {}, {}, time.time(), 0
    for kind in KINDS:
        new[kind] = {}
        for m in its:
            keys = [f"{m['i']}_{s}" for s in SEEDS]
            bgrs = [load_bgr(render_file(kind, m["i"], s)) for s in SEEDS]
            for k, r in zip(keys, rd.read_images(bgrs, whole=False, times=times)):
                new[kind][k] = r
            n += 1
        (P0 / kind / "reads_batch.json").write_text(
            json.dumps(new[kind], ensure_ascii=False, indent=1), "utf-8"
        )
        print(f"batch {kind}: {time.time() - t0:.0f} s", flush=True)
    batch = {
        "s_per_group": round((time.time() - t0) / n, 2),
        **{k: round(v / n, 2) for k, v in times.items()},
        "groups": n,
    }
    vram_peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"batch: {batch}; vram load {vram_load:.2f} peak {vram_peak:.2f}", flush=True)

    # agreement and the gate against P0's reads of record
    old = {k: json.loads((P0 / k / "reads.json").read_text("utf-8")) for k in KINDS}
    agree = {"pages": 0, "boxes_equal": 0, "boxes": 0, "sfx_equal": 0}
    diffs = []
    rs = {"p0": [], "p0_boxes": [], "batch": []}
    gp = {"p0_boxes": [], "batch": []}
    for kind in KINDS:
        for m in its:
            for s in SEEDS:
                k = f"{m['i']}_{s}"
                o, b = old[kind][k], new[kind][k]
                ob = [r for r in o if not r["whole"]]
                bb = [r for r in b if not r["whole"]]
                agree["pages"] += 1
                if [r["box"] for r in ob] == [r["box"] for r in bb]:
                    agree["boxes_equal"] += 1
                    for x, y in zip(ob, bb):
                        agree["boxes"] += 1
                        agree["sfx_equal"] += x["sfx"] == y["sfx"]
                        if x["sfx"] != y["sfx"]:
                            diffs.append(
                                {"kind": kind, "page": k, "box": x["box"]}
                                | {"sfx_p0": x["sfx"], "sfx_batch": y["sfx"]}
                            )
                if kind == "en":
                    continue
                rs["p0"].append(r_seq(m["text"], o))
                rs["p0_boxes"].append(r_seq(m["text"], strip_whole(o)))
                rs["batch"].append(r_seq(m["text"], b))
                gp["p0_boxes"].append(score_page(m["text"], o, o)["g_p"])
                gp["batch"].append(score_page(m["text"], b, b)["g_p"])
    gate = {
        "r_seq": paired(rs["p0_boxes"], rs["batch"]),
        "g_p": paired(gp["p0_boxes"], gp["batch"]),
    }
    gate["pass"] = all(gate[t]["p"] >= 0.05 for t in ("r_seq", "g_p"))
    whole = paired(rs["p0"], rs["p0_boxes"])
    print(f"agreement: {agree}", flush=True)
    print(f"gate: {gate}", flush=True)
    print(f"whole page dropped (P0 reads): {whole}", flush=True)

    run_dir = make_run_dir("cjk_rl", label="reader_batch", root=LINE / "results")
    (run_dir / "diffs.json").write_text(
        json.dumps(diffs, ensure_ascii=False, indent=1), "utf-8"
    )
    write_result(
        run_dir,
        script=__file__,
        args={"serial_n": SERIAL_N, "sfx_batch": rd.sfx.batch_size},
        label="reader_batch",
        metrics={
            "time": {"serial": serial, "batch": batch},
            "vram_gib": {"load": round(vram_load, 2), "peak": round(vram_peak, 2)},
            "agreement": agree,
            "gate": gate,
            "whole_dropped": whole,
        },
        artifacts=[str(P0)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
