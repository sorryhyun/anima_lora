"""Single-op MatMulNBits probe for ORT-web's DP4A path (accuracy_level 4) on the DiT shapes.

Builds one graph per (linear, acc) from the DiT GPTQ file: input A fp16 (1, M, K) with M
symbolic, MatMulNBits bits 8 / block 32 / zero point 128, output fp16 (1, M, N). Inputs are
synthetic rotated activations at M 512 / 1014 / 1024 and two magnitudes; references are
fp32 numpy: W8 (A @ deq(W).T) and W8A8 (A rounded per token per 128-wide K block,
absmax / 127, as the DP4A path does). probe.html runs every graph on the WebGPU EP and
reports rel L2 against both references plus which rows are off.

    uv run --no-project --with onnx --with numpy --with safetensors --with torch \
        python project/webgpu_int8/src/dp4a_probe.py
    python project/webgpu_int8/src/dp4a_serve.py   # then open / launch Chrome on it
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open

LINEARS = (
    "blocks.4.self_attn.q_proj",
    "blocks.4.self_attn.output_proj",
    "blocks.4.cross_attn.q_proj",
    "blocks.4.mlp.layer1",
    "blocks.4.mlp.layer2",
)
MS = (512, 1014, 1024)
MAGS = (1.0, 32.0)
ROT = 256


def hadamard(n):
    h = np.ones((1, 1), np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def graph(name, qi, sc, acc):
    N, K = qi.shape
    blk = K // sc.shape[1]
    bq = (qi.astype(np.int16) + 128).astype(np.uint8).reshape(N, K // blk, blk)
    node = helper.make_node(
        "MatMulNBits",
        ["A", "qweight", "scales"],
        ["Y"],
        domain="com.microsoft",
        K=K,
        N=N,
        bits=8,
        block_size=blk,
        accuracy_level=acc,
    )
    g = helper.make_graph(
        [node],
        name,
        [helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, "M", K])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, "M", N])],
        [
            numpy_helper.from_array(bq, "qweight"),
            numpy_helper.from_array(sc.astype(np.float16).reshape(-1), "scales"),
        ],
    )
    m = helper.make_model(
        g,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    m.ir_version = 9
    return m


def activations(M, K, mag, rng):
    # LayerNorm-like rows with a few heavy channels, then the in-graph ConvRot rotation
    x = rng.standard_normal((M, K)).astype(np.float64)
    x *= np.exp(0.5 * rng.standard_normal(K))[None]
    heavy = rng.choice(K, 8, replace=False)
    x[:, heavy] *= 20
    x *= mag / np.sqrt((x**2).mean())
    H = hadamard(ROT)
    x = (x.reshape(M, K // ROT, ROT) @ H).reshape(M, K)
    return x.astype(np.float16)


def dp4a_round(a):
    M, K = a.shape
    b = a.astype(np.float32).reshape(M, K // 128, 128)
    s = np.abs(b).max(-1, keepdims=True) / 127
    s = np.where(s == 0, 1, s)
    return (np.clip(np.round(b / s), -127, 127) * s).reshape(M, K)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--gptq",
        default="output/webgpu_int8/dit_int8.anima_turbo_4step_v4_preview.safetensors",
    )
    ap.add_argument("--out_dir", default="output/webgpu_int8/dp4a_probe")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    f = safe_open(a.gptq, "np")
    cases, inputs = [], {}
    for key in LINEARS:
        qi, sc = f.get_tensor(f"{key}.q"), f.get_tensor(f"{key}.scale")
        N, K = qi.shape
        w = (
            qi.astype(np.float32).reshape(N, sc.shape[1], -1)
            * sc.astype(np.float32)[..., None]
        ).reshape(N, K)
        for acc in (4, 0):
            onnx.save(graph(key, qi, sc, acc), f"{a.out_dir}/{key}.acc{acc}.onnx")
        for M in MS:
            for mag in MAGS:
                iname = f"A_K{K}_M{M}_mag{mag:g}"
                if iname not in inputs:
                    x = activations(M, K, mag, rng)
                    x.tofile(f"{a.out_dir}/{iname}.bin")
                    inputs[iname] = x
                x = inputs[iname]
                tag = f"{key}_M{M}_mag{mag:g}"
                (x.astype(np.float32) @ w.T).astype(np.float32).tofile(
                    f"{a.out_dir}/{tag}.w8.bin"
                )
                (dp4a_round(x) @ w.T).astype(np.float32).tofile(
                    f"{a.out_dir}/{tag}.w8a8.bin"
                )
                cases.append(dict(key=key, K=K, N=N, M=M, mag=mag, input=iname, tag=tag))
    json.dump(cases, open(f"{a.out_dir}/cases.json", "w"), indent=1)
    print(f"{len(cases)} cases, {2 * len(LINEARS)} graphs -> {a.out_dir}")


if __name__ == "__main__":
    main()
