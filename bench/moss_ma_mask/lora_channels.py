#!/usr/bin/env python3
"""How much of a MOSS LoRA sits on the massive channels (weights only, CPU).

For each checkpoint and block range, over the plain-LoRA adapters
(``ΔW = alpha/rank · up · down``):

- read side (``q`` / ``k`` / ``v`` / ``ffn_up_gate``, input = the stream after the
  block's norm and modulation): the share of ``ΔW``'s column energy on a channel
  set; the same weighted by the calibrated input ``mean|x|²`` (``channel_stats``),
  next to that set's share of the input itself; ``‖ΔW x̄‖ / (‖ΔW‖_F ‖x̄‖)`` for the
  calibrated input mean ``x̄`` (``channel_mean``) and the set's share of ``ΔW x̄``.
- write side (``o`` / ``ffn_down``, output = the stream): the share of ``ΔW``'s row
  energy on the set.
- the modulation adapter (``mod_rank``): per chunk and timestep, the set's share of
  ``‖Δmod(t)‖²`` and of the Base modulation's, and ``Δmod / mod`` on each channel of
  ``--single``.

Channel sets: the 18 of ``massive_channels.npz`` (uniform share 18 / 3072 = 0.59 %)
and ``--single`` (default 1356, 42; 0.065 %).

    .venv/bin/python bench/moss_ma_mask/lora_channels.py --label i2i <a.safetensors> ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from bench._common import make_run_dir, write_result

READ = ("q", "k", "v", "ffn_up_gate")
WRITE = ("o", "ffn_down")
RANGES = ((0, 13), (14, 25), (26, 35), (36, 41))
CHUNKS = ("shift_a", "scale_a", "gate_a", "shift_f", "scale_f", "gate_f")


def delta_parts(sd, key):
    down = sd[f"{key}.lora_down.weight"].double()
    up = sd[f"{key}.lora_up.weight"].double()
    return up, down, float(sd[f"{key}.alpha"]) / down.shape[0]


def base_tensors(model_dir, names):
    out = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), "pt") as s:
            for n in names:
                if n in s.keys():
                    out[n] = s.get_tensor(n).double()
    return out


def time_features(base, t, freq_dim=256):
    half = freq_dim // 2
    freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = torch.tensor(float(t), dtype=torch.float64) * freq
    e = torch.cat((phase.cos(), phase.sin())).float().double()
    h = base["time_embedding.0.weight"] @ e + base["time_embedding.0.bias"]
    h = h * torch.sigmoid(h)
    return base["time_embedding.2.weight"] @ h + base["time_embedding.2.bias"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument(
        "--massive", default="library/moss/calibration/massive_channels.npz"
    )
    ap.add_argument("--single", default="1356,42")
    ap.add_argument(
        "--stats", default="library/moss/calibration/channel_stats.safetensors"
    )
    ap.add_argument(
        "--mean", default="library/moss/calibration/channel_mean.safetensors"
    )
    ap.add_argument("--timesteps", default="0,300,900")
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    from library.env import resolve_under_home
    from library.moss.loader import variant_dir
    from library.moss.requests import resolve_model_dir

    m18 = torch.from_numpy(np.load(resolve_under_home(args.massive))["channels"])
    single = torch.tensor([int(c) for c in args.single.split(",")])
    stats = load_file(str(resolve_under_home(args.stats)))
    mean = load_file(str(resolve_under_home(args.mean)))
    times = [float(t) for t in args.timesteps.split(",")]
    base = base_tensors(
        variant_dir(resolve_model_dir(args.model_dir), "base") / "transformer",
        [
            "time_embedding.0.weight",
            "time_embedding.0.bias",
            "time_embedding.2.weight",
            "time_embedding.2.bias",
            "shared_block_modulation.weight",
            "shared_block_modulation.bias",
        ],
    )
    feats = {t: time_features(base, t) for t in times}
    base_mod = {
        t: (
            base["shared_block_modulation.weight"] @ f
            + base["shared_block_modulation.bias"]
        ).view(6, -1)
        for t, f in feats.items()
    }

    def share(e, idx):
        return float(e[idx].sum() / e.sum())

    out = {}
    for path in args.ckpts:
        sd = load_file(str(resolve_under_home(path)))
        name = Path(path).stem
        rows = {}
        for lo, hi in RANGES:
            acc = {}
            for block in range(lo, hi + 1):
                for proj in READ + WRITE:
                    up, down, scale = delta_parts(
                        sd, f"lora_unet_layers_{block}_proj_{proj}"
                    )
                    r = acc.setdefault(proj, {})
                    if proj in WRITE:
                        e = torch.einsum("or,rs,os->o", up, down @ down.T, up)
                        vals = {"w18": share(e, m18), "w2": share(e, single)}
                    else:
                        e = torch.einsum("rc,rs,sc->c", down, up.T @ up, down)
                        key = f"layers.{block}.proj.{proj}"
                        a2 = stats[key].double() ** 2
                        xbar = mean[key].double()
                        only18 = torch.zeros_like(xbar)
                        only18[m18] = xbar[m18]
                        only2 = torch.zeros_like(xbar)
                        only2[single] = xbar[single]
                        full = (up @ (down @ xbar)).norm()
                        vals = {
                            "w18": share(e, m18),
                            "w2": share(e, single),
                            "in18": share(a2, m18),
                            "in2": share(a2, single),
                            "wx18": share(e * a2, m18),
                            "wx2": share(e * a2, single),
                            # 1 / √3072 = 0.018 for a ΔW blind to the mean direction
                            "dc_align": float(full / (e.sum().sqrt() * xbar.norm())),
                            "dc18": float((up @ (down @ only18)).norm() / full),
                            "dc2": float((up @ (down @ only2)).norm() / full),
                        }
                    for k, v in vals.items():
                        r.setdefault(k, []).append(v)
            rows[f"{lo}-{hi}"] = {
                p: {k: round(float(np.mean(v)), 5) for k, v in r.items()}
                for p, r in acc.items()
            }
        entry = {"blocks": rows}
        if "lora_unet_shared_block_modulation.lora_up.weight" in sd:
            up, down, scale = delta_parts(sd, "lora_unet_shared_block_modulation")
            mod = {}
            for t in times:
                d = (scale * (up @ (down @ feats[t]))).view(6, -1)
                mod[str(int(t))] = {
                    c: {
                        "rel": round(float(d[i].norm() / base_mod[t][i].norm()), 5),
                        "d18": round(share(d[i] ** 2, m18), 5),
                        "d2": round(share(d[i] ** 2, single), 5),
                        "base18": round(share(base_mod[t][i] ** 2, m18), 5),
                        "base2": round(share(base_mod[t][i] ** 2, single), 5),
                        "ratio": {
                            int(ch): round(float(d[i, ch] / base_mod[t][i, ch]), 4)
                            for ch in single
                        },
                    }
                    for i, c in enumerate(CHUNKS)
                }
            entry["mod"] = mod
        out[name] = entry

        print(f"\n== {name}")
        print(
            "  read side, blocks: w18 / w2 column share (%), wx18 (in18) input-weighted (%), dc_align, dc18"
        )
        for rng, r in rows.items():
            cells = []
            for p in READ:
                v = r[p]
                cells.append(
                    f"{p}: {100 * v['w18']:.2f}/{100 * v['w2']:.3f} "
                    f"wx {100 * v['wx18']:.0f}({100 * v['in18']:.0f}) "
                    f"dc {v['dc_align']:.3f}/{v['dc18']:.2f}"
                )
            print(f"  {rng:>6}  " + " | ".join(cells))
        print("  write side, blocks: w18 / w2 row share (%)")
        for rng, r in rows.items():
            print(
                f"  {rng:>6}  "
                + " | ".join(
                    f"{p}: {100 * r[p]['w18']:.2f}/{100 * r[p]['w2']:.3f}"
                    for p in WRITE
                )
            )
        if "mod" in entry:
            for t, chunks in entry["mod"].items():
                print(
                    f"  mod t={t}: "
                    + " | ".join(
                        f"{c} rel {100 * v['rel']:.1f}% d18 {100 * v['d18']:.1f}(base {100 * v['base18']:.1f})"
                        for c, v in chunks.items()
                    )
                )
                print(
                    "      Δ/base on "
                    + ", ".join(
                        f"{c} "
                        + "/".join(f"{v['ratio'][int(ch)]:+.3f}" for ch in single)
                        for c, v in chunks.items()
                    )
                )

    run_dir = make_run_dir("moss_ma_mask", args.label)
    (run_dir / "lora_channels.json").write_text(json.dumps(out, indent=1))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "checkpoints": list(out),
            "uniform18": 18 / 3072,
            "uniform2": 2 / 3072,
        },
        label=args.label,
        artifacts=["lora_channels.json"],
    )
    print(f"\n[done] {run_dir}")


if __name__ == "__main__":
    main()
