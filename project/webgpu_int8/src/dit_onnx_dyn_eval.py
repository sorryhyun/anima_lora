"""ORT CPU check of the resolution-free DiT segments against the per-bucket graphs.

Runs a segment manifest end to end (initializers read from the shared sidecar through
the manifest's byte ranges and inlined, as the browser's range loader does) on one
seeded input, for each requested bucket, and compares the dynamic set's ``noise_pred``
and every intermediate with the bucket's own graphs. ``--extra HxW`` runs latent grids
no bucket has, checked for finiteness only.

    uv run --no-project --with onnx --with onnxruntime --with numpy python \
        project/webgpu_int8/src/dit_onnx_dyn_eval.py --hf_dir <dir> --dyn <manifest>
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto


class Segmented:
    def __init__(
        self,
        manifest,
        graph_dir,
        sidecar,
        providers=("CPUExecutionProvider",),
        opt_level=None,
    ):
        self.man = json.load(open(manifest))
        self.dir = graph_dir
        self.data = np.memmap(sidecar, dtype=np.uint8, mode="r")
        self.sess = []
        so = ort.SessionOptions()
        so.log_severity_level = 3
        if opt_level is not None:
            so.graph_optimization_level = opt_level
        for s in self.man["segments"]:
            m = onnx.load(
                os.path.join(self.dir, os.path.basename(s["graph"])),
                load_external_data=False,
            )
            rng = {d["name"]: d["range"][0] for d in s["data"]}
            for t in m.graph.initializer:
                if t.data_location != TensorProto.EXTERNAL:
                    continue
                e = {x.key: x.value for x in t.external_data}
                lo = rng[e["location"]] + int(e.get("offset", 0))
                t.raw_data = self.data[lo : lo + int(e["length"])].tobytes()
                t.data_location = TensorProto.DEFAULT
                del t.external_data[:]
            self.sess.append(
                (
                    ort.InferenceSession(
                        m.SerializeToString(), so, providers=list(providers)
                    ),
                    s,
                )
            )
            del m

    def run(self, feeds):
        env = dict(feeds)
        for sess, s in self.sess:
            names = [i.name for i in sess.get_inputs()]
            outs = sess.run(s["outputs"], {k: env[k] for k in names})
            env.update(zip(s["outputs"], outs))
        return env


def feeds_for(H, W, seed=0):
    r = np.random.default_rng(seed)
    return {
        "latent": r.standard_normal((1, 16, 1, H, W), dtype=np.float32),
        "timestep": np.array([0.75], np.float32),
        "encoder_hidden_states": (
            r.standard_normal((1, 512, 1024), dtype=np.float32) * 0.5
        ),
    }


def rel(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--hf_dir", required=True, help="dir holding the bucket graphs + sidecar"
    )
    ap.add_argument(
        "--dyn", required=True, help="dynamic segments.json (graphs beside it)"
    )
    ap.add_argument(
        "--sidecar", default="", help="default: <hf_dir>/<manifest sidecar>"
    )
    ap.add_argument("--buckets", default="624x416,384x688,512")
    ap.add_argument("--extra", default="", help="comma list of latent HxW, e.g. 64x96")
    a = ap.parse_args()

    dyn_man = json.load(open(a.dyn))
    sidecar = a.sidecar or os.path.join(a.hf_dir, os.path.basename(dyn_man["sidecar"]))
    t0 = time.time()
    dyn = Segmented(a.dyn, os.path.dirname(a.dyn), sidecar)
    print(f"dynamic set loaded in {time.time() - t0:.0f}s", flush=True)
    report = {}
    for b in [x for x in a.buckets.split(",") if x]:
        ref = Segmented(
            os.path.join(a.hf_dir, f"dit_{b}_fp16_r16.segments.json"), a.hf_dir, sidecar
        )
        lat = ref.sess[0][0].get_inputs()[0].shape
        f = feeds_for(lat[3], lat[4])
        t0 = time.time()
        er = ref.run(f)
        t1 = time.time()
        ed = dyn.run(f)
        t2 = time.time()
        common = [k for k in er if k not in f and k in ed]
        # intermediates are only comparable when both sets share names (non-square buckets)
        worst = max(
            ((rel(ed[k], er[k]), k) for k in common if er[k].shape == ed[k].shape),
            default=(None, None),
        )
        out = {
            "latent": [lat[3], lat[4]],
            "noise_pred_exact": bool(
                np.array_equal(ed["noise_pred"], er["noise_pred"])
            ),
            "noise_pred_rel": rel(ed["noise_pred"], er["noise_pred"]),
            "noise_pred_finite": bool(np.isfinite(ed["noise_pred"]).all()),
            "noise_pred_std": float(ed["noise_pred"].astype(np.float64).std()),
            "worst_intermediate": worst,
            "sec_ref_dyn": [round(t1 - t0, 1), round(t2 - t1, 1)],
        }
        report[b] = out
        print(b, out, flush=True)
        del ref
    for hw in [x for x in a.extra.split(",") if x]:
        H, W = map(int, hw.split("x"))
        t0 = time.time()
        ed = dyn.run(feeds_for(H, W))
        npd = ed["noise_pred"]
        out = {
            "shape": list(npd.shape),
            "finite": bool(np.isfinite(npd).all()),
            "dtype": str(npd.dtype),
            "std": float(npd.astype(np.float64).std()),
            "absmax": float(np.abs(npd.astype(np.float64)).max()),
            "sec": round(time.time() - t0, 1),
        }
        report[f"latent {hw}"] = out
        print(f"latent {hw}", out, flush=True)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
