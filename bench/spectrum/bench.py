#!/usr/bin/env python3
"""Spectrum sampler/scheduler sweep via ComfyUI HTTP API.

For each (prompt, sampler, scheduler) cell this submits two runs:
  - reference: SpectrumKSamplerAdvanced with warmup_steps = steps  (no caching,
               full actual forwards — equivalent to stock KSampler but through
               the same node so the rest of the graph is byte-identical)
  - spectrum: SpectrumKSamplerAdvanced at the configured caching settings

Same seed within a cell, so reference and spectrum outputs are directly
comparable. Results land in a timestamped folder as PNGs plus a results.csv
indexed by (prompt_idx, sampler, scheduler, spectrum_on).

Requires ComfyUI running with the SpectrumKSampler custom node loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path


def _post_json(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def queue_prompt(host: str, workflow: dict, client_id: str) -> str:
    resp = _post_json(
        f"{host}/prompt", {"prompt": workflow, "client_id": client_id}
    )
    return resp["prompt_id"]


def wait_for_completion(host: str, prompt_id: str, timeout: float) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        h = _get_json(f"{host}/history/{prompt_id}")
        if prompt_id in h:
            return h[prompt_id]
        time.sleep(0.5)
    raise TimeoutError(f"prompt {prompt_id} did not finish in {timeout}s")


def fetch_image(host: str, filename: str, subfolder: str, typ: str) -> bytes:
    q = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": typ}
    )
    with urllib.request.urlopen(f"{host}/view?{q}") as r:
        return r.read()


def patch_workflow(
    template: dict,
    *,
    sampler_node: str,
    pos_node: str,
    neg_node: str,
    sampler: str,
    scheduler: str,
    positive: str,
    negative: str,
    seed: int,
    steps: int,
    use_spectrum: bool,
    warmup_steps: int,
    window_size: float,
    flex_window: float,
    stop_caching_step: int,
) -> dict:
    wf = json.loads(json.dumps(template))  # deep copy
    wf[pos_node]["inputs"]["text"] = positive
    wf[neg_node]["inputs"]["text"] = negative
    sn = wf[sampler_node]["inputs"]
    sn["sampler_name"] = sampler
    sn["scheduler"] = scheduler
    sn["seed"] = seed
    sn["steps"] = steps
    sn["window_size"] = window_size
    sn["flex_window"] = flex_window
    sn["stop_caching_step"] = stop_caching_step
    # Reference = warmup==steps → every step runs actual forward. Spectrum =
    # normal warmup_steps. Same node, same graph — only the caching behavior
    # differs between the two runs.
    sn["warmup_steps"] = steps if not use_spectrum else warmup_steps
    return wf


def extract_output_image(
    host: str, history_entry: dict, out_path: Path
) -> str | None:
    outputs = history_entry.get("outputs", {})
    for _nid, o in outputs.items():
        images = o.get("images") or []
        if not images:
            continue
        im = images[0]
        data = fetch_image(
            host, im["filename"], im.get("subfolder", ""), im.get("type", "output")
        )
        out_path.write_bytes(data)
        return im["filename"]
    return None


def parse_cells(spec: str) -> list[tuple[str, str]]:
    """'euler:simple,er_sde:karras' -> [('euler','simple'),('er_sde','karras')]"""
    pairs = []
    for cell in spec.split(","):
        cell = cell.strip()
        if not cell:
            continue
        sampler, _, scheduler = cell.partition(":")
        pairs.append((sampler.strip(), (scheduler or "simple").strip()))
    return pairs


def read_prompts(path: Path) -> list[tuple[str, str]]:
    """Reads `positive|||negative` per line; negative is optional."""
    out = []
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        pos, _, neg = s.partition("|||")
        out.append((pos.strip(), neg.strip() or "worst quality, bad anatomy"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://localhost:8188")
    ap.add_argument(
        "--template",
        default="workflows/modonly.json",
        help="ComfyUI workflow JSON (API format). Must contain one "
        "SpectrumKSamplerAdvanced node.",
    )
    ap.add_argument(
        "--prompts",
        default="bench/spectrum/prompts.example.txt",
        help="One prompt per line. Optional negative after '|||'.",
    )
    ap.add_argument(
        "--cells",
        default="euler:simple,euler_a:simple,er_sde:simple,er_sde:karras,"
        "er_sde:exponential,er_sde:kl_optimal,dpmpp_2m_sde_gpu:simple",
        help="Comma-separated sampler:scheduler pairs.",
    )
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_steps", type=int, default=7)
    ap.add_argument("--window_size", type=float, default=2.0)
    ap.add_argument("--flex_window", type=float, default=0.25)
    ap.add_argument("--stop_caching_step", type=int, default=-1)
    ap.add_argument("--out", default="bench/spectrum/results")
    ap.add_argument(
        "--sampler_node", default="19", help="Node ID of SpectrumKSamplerAdvanced."
    )
    ap.add_argument("--pos_node", default="11")
    ap.add_argument("--neg_node", default="12")
    ap.add_argument("--skip_reference", action="store_true")
    ap.add_argument("--skip_spectrum", action="store_true")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    template = json.loads(Path(args.template).read_text())
    if args.sampler_node not in template:
        raise SystemExit(
            f"sampler_node={args.sampler_node!r} not in template. "
            f"Available: {sorted(template.keys())[:20]}"
        )

    cells = parse_cells(args.cells)
    prompts = read_prompts(Path(args.prompts))
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    # Stash run config so you can reconstruct what produced each grid.
    (out_dir / "config.json").write_text(
        json.dumps(
            {"argv": vars(args), "cells": cells, "n_prompts": len(prompts)},
            indent=2,
        )
    )
    print(f"writing results → {out_dir}")

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_idx", "sampler", "scheduler", "spectrum",
                "wall_s", "filename", "prompt",
            ],
        )
        writer.writeheader()

        modes = []
        if not args.skip_reference:
            modes.append(False)
        if not args.skip_spectrum:
            modes.append(True)
        if not modes:
            raise SystemExit("both --skip_reference and --skip_spectrum set")

        for pi, (pos, neg) in enumerate(prompts):
            for sampler, scheduler in cells:
                for use_spec in modes:
                    tag = "spec" if use_spec else "ref"
                    fname = f"p{pi:02d}_{sampler}_{scheduler}_{tag}.png"
                    wf = patch_workflow(
                        template,
                        sampler_node=args.sampler_node,
                        pos_node=args.pos_node,
                        neg_node=args.neg_node,
                        sampler=sampler,
                        scheduler=scheduler,
                        positive=pos,
                        negative=neg,
                        seed=args.seed,
                        steps=args.steps,
                        use_spectrum=use_spec,
                        warmup_steps=args.warmup_steps,
                        window_size=args.window_size,
                        flex_window=args.flex_window,
                        stop_caching_step=args.stop_caching_step,
                    )
                    client_id = str(uuid.uuid4())
                    start = time.time()
                    pid = queue_prompt(args.host, wf, client_id)
                    entry = wait_for_completion(args.host, pid, args.timeout)
                    wall = time.time() - start
                    saved = extract_output_image(args.host, entry, out_dir / fname)
                    writer.writerow(
                        {
                            "prompt_idx": pi,
                            "sampler": sampler,
                            "scheduler": scheduler,
                            "spectrum": use_spec,
                            "wall_s": f"{wall:.2f}",
                            "filename": fname if saved else "",
                            "prompt": pos[:80],
                        }
                    )
                    f.flush()
                    mark = "✓" if saved else "⚠"
                    print(
                        f"{mark} [{pi}] {sampler:22s} {scheduler:12s} "
                        f"{tag:4s} {wall:5.1f}s → {fname}"
                    )

    print(f"\nDone. Results → {out_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
