"""ORT run (CPU, or CUDA with ``--ep cuda``) of a browser DiT set against torch fp32.

Per σ of the dump: the relative L2 of ``noise_pred`` against the torch DiT (mean over
images). ORT CPU's MatMulNBits rounds activations its own way at ``accuracy_level`` 4
and CUDA's dequantises to fp16 (the browser's non-DP4A path), so neither is the DP4A
number; an fp16 rewrite (no ``--gptq``) isolates the graph and weight placement.

    uv run --no-project --with onnx --with onnxruntime --with numpy python \
        project/webgpu_int8/src/dit_onnx_eval.py --ref output/webgpu_int8/dit_ref_52x78.npz \
        --manifest <segments.json> --sidecar <onnx.data> [--manifest ... --sidecar ...]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from dit_onnx_dyn_eval import Segmented, rel


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", required=True, nargs="+", help="dit_ref_dump.py npz files")
    ap.add_argument("--manifest", required=True, action="append")
    ap.add_argument("--sidecar", required=True, action="append")
    ap.add_argument("--out", default="")
    ap.add_argument("--ep", default="cpu", choices=("cpu", "cuda"))
    a = ap.parse_args()
    report = {}
    for man, side in zip(a.manifest, a.sidecar):
        eps = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if a.ep == "cuda"
            else ["CPUExecutionProvider"]
        )
        m = Segmented(man, os.path.dirname(man), side, eps)
        name = os.path.basename(man).removesuffix(".segments.json")
        print(name, "providers", m.sess[0][0].get_providers(), flush=True)
        for ref_path in a.ref:
            ref = np.load(ref_path)
            per = defaultdict(list)
            for i in range(len(ref["timestep"])):
                feeds = {
                    "latent": ref["latent"][i : i + 1],
                    "timestep": ref["timestep"][i : i + 1],
                    "encoder_hidden_states": ref["encoder_hidden_states"][i : i + 1],
                }
                y = m.run(feeds)["noise_pred"]
                per[float(ref["timestep"][i])].append(
                    rel(y, ref["noise_pred"][i : i + 1])
                )
            grid = "x".join(map(str, ref["latent"].shape[3:]))
            r = {
                f"{s:g}": float(np.mean(v))
                for s, v in sorted(per.items(), reverse=True)
            }
            r["mean"] = float(np.mean([x for v in per.values() for x in v]))
            report[f"{name} @ {grid}"] = r
            print(name, grid, r, flush=True)
        del m
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
