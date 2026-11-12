"""ORT CPU run of a TE graph on te_ref_dump's padded prompts; per-token rel L2 vs fp32.

    uv run --no-project --with onnxruntime --with numpy python \
        project/webgpu_int8/src/te_onnx_eval.py GRAPH.onnx output/webgpu_int8/te_ref.npz [N]

CPU MatMulNBits at accuracy_level 4 rounds activations its own way (not WebGPU's DP4A);
rebuild with --acc 0 to check the graph against te_convrot_gptq's w8 numbers.
"""

import sys
import time

import numpy as np
import onnxruntime as ort

d = np.load(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 32
t0 = time.time()
s = ort.InferenceSession(sys.argv[1], providers=["CPUExecutionProvider"])
print(f"load {time.time() - t0:.1f}s")
rel, rel1, r0, cos, padz = [], [], [], [], []
for i in range(n):
    o = s.run(
        None, {"input_ids": d["ids"][i : i + 1], "attention_mask": d["mask"][i : i + 1]}
    )[0][0]
    o = o.astype(np.float32)
    assert np.isfinite(o).all()
    L = int(d["mask"][i].sum())
    r, oo = d["ref"][i, :L], o[:L]
    e = np.linalg.norm(oo - r, axis=-1) / np.linalg.norm(r, axis=-1)
    c = (oo * r).sum(-1) / np.linalg.norm(oo, axis=-1) / np.linalg.norm(r, axis=-1)
    rel.append(e.mean())
    rel1.append(e[1:].mean())
    r0.append(e[0])
    cos.append(c[1:].mean())
    padz.append(np.abs(o[L:]).max() if L < 512 else 0)


def f(x):
    return round(float(np.mean(x)), 5)


print(
    {
        "rel": f(rel),
        "rel_tok1+": f(rel1),
        "rel_tok0": f(r0),
        "cos_tok1+": f(cos),
        "pad_absmax": float(np.max(padz)),
    }
)
