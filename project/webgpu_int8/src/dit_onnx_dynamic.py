"""One resolution-free DiT segment set from the published per-bucket graphs.

The browser DiT (HF ``onnx/dit_<bucket>_fp16_r16``) was exported per resolution: the
token count, the patch grid and the RoPE angle tables constant-fold into each bucket's
seven segment graphs. The four non-square buckets are the same graph (same nodes, same
names) and differ only in those constants:

- int64 shape vectors (Reshape targets) holding N = ph·pw, ph, pw, H, W, where (H, W) is
  the latent grid and (ph, pw) = (H/2, W/2) the patch grid. Each element is fitted to a
  symbol across the four buckets and rebuilt in-graph from ``Shape(latent)``.
- ``repeat``: the zero padding-mask channel, (1, 1, 1, H, W) -> ConstantOfShape.
- ``repeat_1`` / ``repeat_2``: Cosmos RoPE angles (1, ph, pw, 21) along rows / columns,
  fp16(outer(arange, f)) with f = 1 / θ^(arange(0, 42, 2) / 42), θ = 1e4·4^(42/40)
  (rope_scale 4). Rebuilt with fp32 Range·Mul then Cast; the script checks bit equality
  against every bucket's table.
- ``repeat_3``: the T-axis angles, all zero at T = 1 -> ConstantOfShape.

Every segment takes ``latent`` (the model input, already in the segmented runner's
environment) as the source of H and W; the manifest lists it. Weights stay where they
are: the external-data ranges of the template bucket's manifest into the shared
``dit_512_fp16_r16.onnx.data``, so the output is seven graphs plus a manifest.

    uv run --no-project --with onnx --with numpy --with torch python \
        project/webgpu_int8/src/dit_onnx_dynamic.py --hf_dir <dir with onnx/> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

FIT_BUCKETS = ("624x416", "416x624", "688x384", "384x688")
TEMPLATE = FIT_BUCKETS[0]
NSEG = 7
ROPE_DIM_HW = 42
ROPE_SCALE = 4.0
SYMS = ("H", "W", "ph", "pw", "N")


def bucket_syms(g) -> dict[str, int]:
    lat = next(i for i in g.input if i.name == "latent")
    d = [x.dim_value for x in lat.type.tensor_type.shape.dim]
    H, W = d[3], d[4]
    return {"H": H, "W": W, "ph": H // 2, "pw": W // 2, "N": (H // 2) * (W // 2)}


def rope_freqs() -> np.ndarray:
    theta = 10000.0 * ROPE_SCALE ** (ROPE_DIM_HW / (ROPE_DIM_HW - 2))
    r = torch.arange(0, ROPE_DIM_HW, 2)[: ROPE_DIM_HW // 2].float() / ROPE_DIM_HW
    return (1.0 / (theta**r)).numpy().astype(np.float32)


def rope_table(n: int) -> np.ndarray:
    return (
        torch.outer(torch.arange(n).float(), torch.from_numpy(rope_freqs()))
        .half()
        .numpy()
    )


def consts(g) -> dict[str, np.ndarray]:
    out = {
        t.name: numpy_helper.to_array(t)
        for t in g.initializer
        if t.data_location != TensorProto.EXTERNAL
    }
    for n in g.node:
        if n.op_type == "Constant":
            out[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    return out


def fit_shape(name, vals, syms):
    """Per element: ('c', value) or ('s', symbol), consistent over every bucket."""
    out = []
    for i in range(len(vals[0])):
        col = [int(v[i]) for v in vals]
        if len(set(col)) == 1 and not any(
            all(col[b] == syms[b][s] for b in range(len(col))) for s in SYMS
        ):
            out.append(("c", col[0]))
            continue
        hits = [s for s in SYMS if all(col[b] == syms[b][s] for b in range(len(col)))]
        if len(hits) != 1:
            raise SystemExit(f"{name}[{i}] = {col}: symbols {hits}")
        out.append(("s", hits[0]))
    return out


def convert(graphs, k):
    """Template bucket's seg k with every resolution constant rebuilt in-graph."""
    syms = [bucket_syms_cache[b] for b in FIT_BUCKETS]
    cs = [consts(g.graph) for g in graphs]
    diff = {}
    for name, v in cs[0].items():
        vals = [c[name] for c in cs]
        if any(x.shape != v.shape or not np.array_equal(x, v) for x in vals[1:]):
            diff[name] = vals

    m = onnx.ModelProto()
    m.CopyFrom(graphs[0])
    g = m.graph
    tsyms = syms[0]
    pre = []  # nodes that compute the symbols and the rebuilt constants

    def c64(name, arr):
        pre.append(
            helper.make_node(
                "Constant",
                [],
                [name],
                value=numpy_helper.from_array(np.asarray(arr, np.int64), name),
            )
        )
        return name

    if diff:
        pre += [helper.make_node("Shape", ["latent"], ["dyn__lshape"])]
        c64("dyn__i3", [3])
        c64("dyn__i4", [4])
        c64("dyn__i5", [5])
        c64("dyn__two", [2])
        pre += [
            helper.make_node(
                "Slice", ["dyn__lshape", "dyn__i3", "dyn__i4"], ["dyn__H"]
            ),
            helper.make_node(
                "Slice", ["dyn__lshape", "dyn__i4", "dyn__i5"], ["dyn__W"]
            ),
            helper.make_node("Div", ["dyn__H", "dyn__two"], ["dyn__ph"]),
            helper.make_node("Div", ["dyn__W", "dyn__two"], ["dyn__pw"]),
            helper.make_node("Mul", ["dyn__ph", "dyn__pw"], ["dyn__N"]),
        ]

    replaced = {}  # const name -> new tensor name
    for name, vals in diff.items():
        v0 = vals[0]
        if v0.dtype == np.int64 and v0.ndim == 1:
            parts = fit_shape(name, vals, syms)
            ins = []
            for j, (kind, x) in enumerate(parts):
                ins.append(c64(f"dyn__{name}_{j}", [x]) if kind == "c" else f"dyn__{x}")
            pre.append(helper.make_node("Concat", ins, [f"dyn__{name}"], axis=0))
        elif name == "repeat" or name == "repeat_3":
            # zeros: (1,1,1,H,W) pad mask / (1,ph,pw,22) T-axis angles
            assert all(not x.any() for x in vals), name
            last = v0.shape[-1]
            if name == "repeat":
                shp = [c64("dyn__ones3", [1, 1, 1]), "dyn__H", "dyn__W"]
            else:
                shp = [
                    c64("dyn__one", [1]),
                    "dyn__ph",
                    "dyn__pw",
                    c64(f"dyn__{name}_d", [last]),
                ]
            pre += [
                helper.make_node("Concat", shp, [f"dyn__{name}_shape"], axis=0),
                helper.make_node(
                    "ConstantOfShape",
                    [f"dyn__{name}_shape"],
                    [f"dyn__{name}"],
                    value=numpy_helper.from_array(np.zeros(1, np.float16)),
                ),
            ]
        elif name in ("repeat_1", "repeat_2"):
            axis = 1 if name == "repeat_1" else 2  # rows / columns of (1, ph, pw, 21)
            for b, x in enumerate(vals):
                n = x.shape[axis]
                ref = x[0][:, 0, :] if axis == 1 else x[0][0, :, :]
                if not np.array_equal(rope_table(n), ref):
                    raise SystemExit(
                        f"{name}: RoPE formula misses bucket {FIT_BUCKETS[b]}"
                    )
                assert np.array_equal(
                    np.broadcast_to(np.expand_dims(ref, 2 - axis + 0), x[0].shape), x[0]
                ), name
            p = f"dyn__{name}"
            s = "dyn__ph" if axis == 1 else "dyn__pw"
            freqs = numpy_helper.from_array(rope_freqs()[None, :], f"{p}_f")
            pre += [
                helper.make_node("Constant", [], [f"{p}_f"], value=freqs),
                helper.make_node("Squeeze", [s], [f"{p}_n"]),
                helper.make_node("Cast", [f"{p}_n"], [f"{p}_nf"], to=TensorProto.FLOAT),
                helper.make_node(
                    "Constant",
                    [],
                    [f"{p}_z"],
                    value=numpy_helper.from_array(np.array(0, np.float32)),
                ),
                helper.make_node(
                    "Constant",
                    [],
                    [f"{p}_o"],
                    value=numpy_helper.from_array(np.array(1, np.float32)),
                ),
                helper.make_node(
                    "Range", [f"{p}_z", f"{p}_nf", f"{p}_o"], [f"{p}_pos"]
                ),
                helper.make_node(
                    "Unsqueeze", [f"{p}_pos", c64(f"{p}_ax1", [1])], [f"{p}_col"]
                ),
                helper.make_node("Mul", [f"{p}_col", f"{p}_f"], [f"{p}_ang"]),
                helper.make_node(
                    "Cast", [f"{p}_ang"], [f"{p}_h"], to=TensorProto.FLOAT16
                ),
                # (n, 21) -> (1, n, 1, 21) for rows, (1, 1, n, 21) for columns
                helper.make_node(
                    "Unsqueeze",
                    [f"{p}_h", c64(f"{p}_ax", [0, 2] if axis == 1 else [0, 1])],
                    [f"{p}_u"],
                ),
                helper.make_node(
                    "Concat",
                    [
                        c64(f"{p}_one", [1]),
                        "dyn__ph",
                        "dyn__pw",
                        c64(f"{p}_d", [v0.shape[-1]]),
                    ],
                    [f"{p}_shape"],
                    axis=0,
                ),
                helper.make_node("Expand", [f"{p}_u", f"{p}_shape"], [p]),
            ]
        else:
            raise SystemExit(
                f"seg{k}: unhandled resolution constant {name} {v0.dtype} {v0.shape}"
            )
        replaced[name] = f"dyn__{name}"

    keep_nodes = [
        n for n in g.node if not (n.op_type == "Constant" and n.output[0] in replaced)
    ]
    for n in keep_nodes:
        for i, x in enumerate(n.input):
            if x in replaced:
                n.input[i] = replaced[x]
    keep_init = [t for t in g.initializer if t.name not in replaced]
    del g.initializer[:]
    g.initializer.extend(keep_init)
    del g.node[:]
    g.node.extend(pre + keep_nodes)

    # graph I/O: latent dims and token dims symbolic
    by_val = {tsyms["H"]: "H", tsyms["W"]: "W", tsyms["N"]: "N"}
    if not any(i.name == "latent" for i in g.input):
        g.input.append(
            helper.make_tensor_value_info(
                "latent", TensorProto.FLOAT, [1, 16, 1, "H", "W"]
            )
        )
    for v in list(g.input) + list(g.output):
        for d in v.type.tensor_type.shape.dim:
            if d.HasField("dim_value") and d.dim_value in by_val:
                d.dim_param = by_val[d.dim_value]
    del g.value_info[:]
    return m, sorted(replaced)


bucket_syms_cache: dict[str, dict[str, int]] = {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf_dir", required=True, help="dir holding the onnx/ graphs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--name", default="dit_dyn_fp16_r16")
    a = ap.parse_args()

    src = (
        os.path.join(a.hf_dir, "onnx")
        if os.path.isdir(os.path.join(a.hf_dir, "onnx"))
        else a.hf_dir
    )
    man = json.load(open(os.path.join(src, f"dit_{TEMPLATE}_fp16_r16.segments.json")))
    os.makedirs(a.out_dir, exist_ok=True)
    segs = []
    for k in range(NSEG):
        graphs = [
            onnx.load(
                os.path.join(src, f"dit_{b}_fp16_r16.seg{k}.onnx"),
                load_external_data=False,
            )
            for b in FIT_BUCKETS
        ]
        if k == 0:
            for b, gm in zip(FIT_BUCKETS, graphs):
                bucket_syms_cache[b] = bucket_syms(gm.graph)
        sig = [
            [
                (n.op_type, tuple(n.input), tuple(n.output))
                for n in gm.graph.node
                if n.op_type != "Constant"
            ]
            for gm in graphs
        ]
        if any(s != sig[0] for s in sig[1:]):
            raise SystemExit(f"seg{k}: buckets differ in structure")
        m, replaced = convert(graphs, k)
        old = f"dit_{TEMPLATE}_fp16_r16.seg{k}."
        for t in m.graph.initializer:
            for e in t.external_data:
                if e.key == "location":
                    assert e.value.startswith(old), e.value
                    e.value = f"{a.name}.seg{k}." + e.value[len(old) :]
        path = os.path.join(a.out_dir, f"{a.name}.seg{k}.onnx")
        onnx.save(m, path)
        s = man["segments"][k]
        inputs = list(s["inputs"]) + ([] if "latent" in s["inputs"] else ["latent"])
        segs.append(
            {
                "graph": f"{a.name}.seg{k}.onnx",
                "data": [
                    {
                        "name": d["name"].replace(old, f"{a.name}.seg{k}."),
                        "range": d["range"],
                    }
                    for d in s["data"]
                ],
                "inputs": inputs,
                "outputs": s["outputs"],
            }
        )
        print(
            f"seg{k}: {len(replaced)} constants made dynamic, {os.path.getsize(path) / 2**20:.2f} MiB"
        )
    out = {
        "model": a.name,
        "sidecar": man["sidecar"],
        "outputs": man["outputs"],
        "segments": segs,
    }
    with open(os.path.join(a.out_dir, f"{a.name}.segments.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
