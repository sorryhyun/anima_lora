"""CPU emulation of ORT-web 1.30.0's DP4A MatMulNBits arithmetic on a DiT segment set.

Runs the set on ORT CPU at ``accuracy_level`` 0 (float activations) on one sample of a
dit_ref_dump.py file, taps every MatMulNBits input, and replays the DP4A shaders
(``dp4a_quantize`` / ``dp4a_matmul`` templates, 8-bit) per linear in the op's own float
type T (fp16, or fp32 for a ``--fp32_matmul`` set):

    s       = max|a| per token per 128-wide K block           scale_a = T(s / 127)
    qa      = round(clamp(T(a / s), -1, 1) * 127)
    per 32-wide K chunk:  scale = T(scale_a * scale_b)
                          y    += T(f32(dot(qa, qb)) * f32(scale))

Reported per linear type: the relative L2 against the same int8 weights on float
activations, for the replay and for W8A8 in exact arithmetic, and the share of
non-zero scale pairs whose product rounds to zero in T.

    uv run --no-project --with onnx --with onnxruntime --with numpy --with safetensors \
        --with torch python project/webgpu_int8/src/dp4a_emu.py --manifest <segments.json>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper


def dp4a(a, qb, sb, blk, T):
    """a (M, K) and sb (N, K/blk) in T, qb (N, K) int8 -> y (M, N) in T, zeroed-pair share."""
    M, K = a.shape
    N = qb.shape[0]
    ab = a.float().reshape(M, K // 128, 128)
    s = ab.abs().amax(-1, keepdim=True)
    qa = torch.nan_to_num((ab / s).to(T).float()).clamp(-1, 1).mul(127).round()
    qa = qa.reshape(M, K // 32, 32)
    sa = (s / 127.0).to(T).reshape(M, K // 128).repeat_interleave(4, dim=1)
    sbc = sb.repeat_interleave(blk // 32, dim=1)
    qb = qb.float().reshape(N, K // 32, 32)
    y = torch.zeros(M, N, dtype=T)
    zero = tot = 0
    for c in range(K // 32):
        sc = (sa[:, c, None].float() * sbc[None, :, c].float()).to(T)
        nz = (sa[:, c, None] != 0) & (sbc[None, :, c] != 0)
        tot += nz.sum().item()
        zero += ((sc == 0) & nz).sum().item()
        part = ((qa[:, c] @ qb[:, c].T) * sc.float()).to(T)
        y = (y.float() + part.float()).to(T)
    return y, zero / max(tot, 1)


def rel(y, r):
    return ((y.double() - r).norm() / r.norm()).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True, help="segments.json, graphs beside it")
    ap.add_argument("--sidecar", default="", help="default: beside the manifest")
    ap.add_argument("--ref", default="output/webgpu_int8/dit_ref_52x78.npz")
    ap.add_argument("--index", type=int, default=0, help="sample of the ref file")
    ap.add_argument("--rows", type=int, default=128, help="tokens replayed per linear")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(8)
    man = json.load(open(a.manifest))
    gdir = os.path.dirname(a.manifest)
    side = a.sidecar or os.path.join(gdir, os.path.basename(man["sidecar"]))
    data = np.memmap(side, dtype=np.uint8, mode="r")
    ref = np.load(a.ref)
    env = {
        k: ref[k][a.index : a.index + 1]
        for k in ("latent", "timestep", "encoder_hidden_states")
    }
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 8
    rng = np.random.default_rng(0)
    rows = []
    for s in man["segments"]:
        m = onnx.load(
            os.path.join(gdir, os.path.basename(s["graph"])), load_external_data=False
        )
        off = {d["name"]: d["range"][0] for d in s["data"]}
        for t in m.graph.initializer:
            if t.data_location != TensorProto.EXTERNAL:
                continue
            e = {x.key: x.value for x in t.external_data}
            lo = off[e["location"]] + int(e.get("offset", 0))
            t.raw_data = data[lo : lo + int(e["length"])].tobytes()
            t.data_location = TensorProto.DEFAULT
            del t.external_data[:]
        init = {t.name: t for t in m.graph.initializer}
        lin = {}
        for n in m.graph.node:
            if n.op_type != "MatMulNBits":
                continue
            at = {x.name: x for x in n.attribute}
            at["accuracy_level"].i = 0
            sb = numpy_helper.to_array(init[n.input[2]])
            qb = numpy_helper.to_array(init[n.input[1]])
            N, K = at["N"].i, at["K"].i
            lin[n.name.removesuffix(".matmulnbits")] = (
                n.input[0],
                (qb.reshape(N, K).astype(np.int16) - 128).astype(np.int8),
                sb.reshape(N, -1),
            )
        taps = sorted({v[0] for v in lin.values()})
        etype = {np.dtype("float16"): TensorProto.FLOAT16}
        for x in taps:
            dt = next(v[2].dtype for v in lin.values() if v[0] == x)
            m.graph.output.append(
                helper.make_tensor_value_info(x, etype.get(dt, TensorProto.FLOAT), None)
            )
        sess = ort.InferenceSession(
            m.SerializeToString(), so, providers=["CPUExecutionProvider"]
        )
        del m, init
        feeds = {i.name: env[i.name] for i in sess.get_inputs()}
        outs = sess.run(s["outputs"] + taps, feeds)
        env.update(zip(s["outputs"], outs[: len(s["outputs"])]))
        got = dict(zip(taps, outs[len(s["outputs"]) :]))
        del sess
        for key, (x, qb, sb) in lin.items():
            A = got[x].reshape(-1, got[x].shape[-1])
            sel = np.sort(rng.choice(len(A), min(a.rows, len(A)), replace=False))
            A, qb, sb = (torch.from_numpy(np.array(v)) for v in (A[sel], qb, sb))
            N, K = qb.shape
            blk = K // sb.shape[1]
            w = (qb.double().reshape(N, -1, blk) * sb.double()[..., None]).reshape(N, K)
            r8 = A.double() @ w.T
            ab = A.double().reshape(len(A), K // 128, 128)
            sa = ab.abs().amax(-1, keepdim=True).clamp_min(1e-30)
            r88 = ((ab / sa * 127).round() * sa / 127).reshape(len(A), K) @ w.T
            y, zero = dp4a(A, qb, sb, blk, A.dtype)
            rows.append(
                dict(
                    key=key,
                    dtype=str(A.dtype),
                    exact=rel(r88, r8),
                    dp4a=rel(y, r8),
                    zero=zero,
                )
            )
        print(f"{os.path.basename(s['graph'])}: {len(lin)} linears", flush=True)
    r = ref["noise_pred"][a.index : a.index + 1].astype(np.float64)
    e2e = float(
        np.linalg.norm(env["noise_pred"].astype(np.float64) - r) / np.linalg.norm(r)
    )
    print(f"noise_pred vs torch fp32, float activations (acc 0): {e2e:.4f}")
    by = defaultdict(list)
    for x in rows:
        by[x["key"].split(".", 2)[2]].append(x)
    print(
        f"{'linear':24s} {'T':14s} exact W8A8   DP4A in T: median / max    zeroed pairs"
    )
    for t, v in by.items():
        d = np.array([x["dp4a"] for x in v])
        print(
            f"{t:24s} {v[0]['dtype']:14s} {np.median([x['exact'] for x in v]):.4f}       "
            f"{np.median(d):.4f} / {d.max():.4f}            "
            f"{np.median([x['zero'] for x in v]):.3f}"
        )
    if a.out:
        json.dump(dict(noise_pred_acc0=e2e, linears=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
