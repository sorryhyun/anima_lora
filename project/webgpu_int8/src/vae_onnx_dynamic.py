"""One resolution-free VAE decoder graph from the published per-bucket graphs.

``vae_decoder_<bucket>_fp16`` bakes the latent grid (H, W) into three kinds of constant;
624x416 and 416x624 share one node sequence (512 keeps its index tables in the sidecar,
688x384 / 384x688 come from another export); the two are aligned by node position:

- int64 shape vectors (Expand / Reshape targets) holding s·H, s·W (s = 1, 2, 4, 8) and
  H·W (the mid-block attention). Fitted per element over the two buckets (H and W swap between them) and rebuilt
  from ``Shape(latent)``.
- the nearest-2x upsample index tables, rows ⌊i/2⌋ and columns ⌊j/2⌋ of a (2h, 2w)
  grid, fed to GatherND over the (h, w, 1, C) transposed feature map. Rebuilt from that
  map's own shape with Range / Div / Expand.

Weights stay in the shared ``vae_decoder_512_fp16.onnx.data`` (the external-data
``location`` is untouched).

    uv run --no-project --with onnx --with numpy python \
        project/webgpu_int8/src/vae_onnx_dynamic.py --hf_dir <dir with the graphs> --out <onnx>
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FIT_BUCKETS = ("624x416", "416x624")


def grid(g):
    d = [x.dim_value for x in g.input[0].type.tensor_type.shape.dim]
    return d[2], d[3]


def consts(g):
    out = {
        t.name: numpy_helper.to_array(t)
        for t in g.initializer
        if t.data_location != TensorProto.EXTERNAL
    }
    for n in g.node:
        if n.op_type == "Constant":
            out[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    return out


def symbols(H, W):
    s = {}
    for k in (1, 2, 4, 8):
        s[f"{k}H"], s[f"{k}W"] = k * H, k * W
    s["HW"] = H * W
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    models = [
        onnx.load(
            os.path.join(a.hf_dir, f"vae_decoder_{b}_fp16.onnx"),
            load_external_data=False,
        )
        for b in FIT_BUCKETS
    ]
    gs = [m.graph for m in models]
    nodes = [[n for n in g.node if n.op_type != "Constant"] for g in gs]
    if any(
        [n.op_type for n in ns] != [n.op_type for n in nodes[0]] for ns in nodes[1:]
    ):
        raise SystemExit("buckets differ in structure")
    cs = [consts(g) for g in gs]
    syms = [symbols(*grid(g)) for g in gs]
    names = list(syms[0])

    # template constant name -> its value in every bucket, by node position
    vals = {}
    for i, n in enumerate(nodes[0]):
        for j, x in enumerate(n.input):
            if x in cs[0]:
                vals.setdefault(
                    x, [cs[b][nodes[b][i].input[j]] for b in range(len(gs))]
                )

    def tied(v):
        # equal in both buckets but still a symbol: H·W is the same when H and W swap
        return (
            v[0].dtype == np.int64
            and v[0].ndim == 1
            and any(
                int(e) == syms[0]["HW"]
                and all(int(x[i]) == syms[b]["HW"] for b, x in enumerate(v))
                for i, e in enumerate(v[0])
            )
        )

    diff = {
        k: v
        for k, v in vals.items()
        if any(x.shape != v[0].shape or not np.array_equal(x, v[0]) for x in v[1:])
        or tied(v)
    }

    m = onnx.ModelProto()
    m.CopyFrom(models[0])
    g = m.graph
    lat = g.input[0].name
    pre, idx_nodes = [], {}
    counter = [0]

    def c64(arr):
        counter[0] += 1
        nm = f"dyn__c{counter[0]}"
        pre.append(
            helper.make_node(
                "Constant",
                [],
                [nm],
                value=numpy_helper.from_array(np.asarray(arr, np.int64), nm),
            )
        )
        return nm

    pre += [
        helper.make_node("Shape", [lat], ["dyn__ls"]),
        helper.make_node("Slice", ["dyn__ls", "dyn__i2", "dyn__i3"], ["dyn__H"]),
        helper.make_node("Slice", ["dyn__ls", "dyn__i3", "dyn__i4"], ["dyn__W"]),
        helper.make_node("Mul", ["dyn__H", "dyn__W"], ["dyn__HW"]),
    ]
    for k in (2, 3, 4):
        pre.insert(
            0,
            helper.make_node(
                "Constant",
                [],
                [f"dyn__i{k}"],
                value=numpy_helper.from_array(np.array([k], np.int64)),
            ),
        )
    for s in (1, 2, 4, 8):
        for ax in "HW":
            if s == 1:
                pre.append(
                    helper.make_node("Identity", [f"dyn__{ax}"], [f"dyn__1{ax}"])
                )
            else:
                pre.append(
                    helper.make_node("Mul", [f"dyn__{ax}", c64([s])], [f"dyn__{s}{ax}"])
                )

    consumers = {}
    for n in g.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    replaced = {}
    for name, v in diff.items():
        v0 = v[0]
        if v0.dtype == np.int64 and v0.ndim == 1:
            parts = []
            for e in range(len(v0)):
                col = [int(x[e]) for x in v]
                hits = [
                    s for s in names if all(col[b] == syms[b][s] for b in range(len(v)))
                ]
                if len(set(col)) == 1 and not hits:
                    parts.append(c64([col[0]]))
                elif len(hits) == 1 or (
                    hits
                    and all(
                        syms[b][h] == syms[b][hits[0]]
                        for h in hits
                        for b in range(len(v))
                    )
                ):
                    parts.append(f"dyn__{hits[0]}")
                else:
                    raise SystemExit(f"{name}[{e}] = {col}: symbols {hits}")
            out = f"dyn__{name}"
            pre.append(helper.make_node("Concat", parts, [out], axis=0))
            replaced[name] = out
        elif v0.dtype == np.int64 and v0.ndim == 2:
            # upsample index table: Unsqueeze(table) -> Concat -> GatherND(data, idx)
            un = consumers[name][0]
            cat = consumers[un.output[0]][0]
            gnd = consumers[cat.output[0]][0]
            assert (
                un.op_type == "Unsqueeze"
                and cat.op_type == "Concat"
                and gnd.op_type == "GatherND"
            ), name
            r, c = np.arange(v0.shape[0]), np.arange(v0.shape[1])
            if np.array_equal(v0, np.broadcast_to((r // 2)[:, None], v0.shape)):
                axis = 0
            elif np.array_equal(v0, np.broadcast_to((c // 2)[None, :], v0.shape)):
                axis = 1
            else:
                raise SystemExit(f"{name}: not a nearest-2x index table")
            data = gnd.input[0]
            p = f"dyn__{name}"
            if data not in idx_nodes:
                d = f"dyn__{data}"
                idx_nodes[data] = d
                pre_local = [
                    helper.make_node("Shape", [data], [f"{d}_s"]),
                    helper.make_node(
                        "Slice", [f"{d}_s", c64([0]), c64([1])], [f"{d}_h"]
                    ),
                    helper.make_node(
                        "Slice", [f"{d}_s", c64([1]), c64([2])], [f"{d}_w"]
                    ),
                    helper.make_node("Mul", [f"{d}_h", c64([2])], [f"{d}_h2"]),
                    helper.make_node("Mul", [f"{d}_w", c64([2])], [f"{d}_w2"]),
                    helper.make_node(
                        "Concat", [f"{d}_h2", f"{d}_w2"], [f"{d}_hw2"], axis=0
                    ),
                ]
            else:
                pre_local = []
            d = idx_nodes[data]
            n2 = f"{d}_h2" if axis == 0 else f"{d}_w2"
            pre_local += [
                helper.make_node("Squeeze", [n2], [f"{p}_n"]),
                helper.make_node("Range", [c64(0), f"{p}_n", c64(1)], [f"{p}_r"]),
                helper.make_node("Div", [f"{p}_r", c64(2)], [f"{p}_q"]),
                helper.make_node("Unsqueeze", [f"{p}_q", c64([1 - axis])], [f"{p}_u"]),
                helper.make_node("Expand", [f"{p}_u", f"{d}_hw2"], [p]),
            ]
            # data is computed mid-graph: insert these nodes right before the Unsqueeze
            idx_nodes.setdefault("__insert", []).append((un, pre_local))
            replaced[name] = p
        else:
            raise SystemExit(f"unhandled constant {name} {v0.dtype} {v0.shape}")

    inserts = {id(un): extra for un, extra in idx_nodes.get("__insert", [])}
    out_nodes = list(pre)
    for n in g.node:
        if n.op_type == "Constant" and n.output[0] in replaced:
            continue
        if id(n) in inserts:
            out_nodes += inserts[id(n)]
        for i, x in enumerate(n.input):
            if x in replaced:
                n.input[i] = replaced[x]
        out_nodes.append(n)
    keep = [t for t in g.initializer if t.name not in replaced]
    del g.initializer[:]
    g.initializer.extend(keep)
    del g.node[:]
    g.node.extend(out_nodes)
    H0, W0 = grid(gs[0])
    for v in list(g.input) + list(g.output):
        dims = v.type.tensor_type.shape.dim
        for d in dims:
            for s, val in symbols(H0, W0).items():
                if d.HasField("dim_value") and d.dim_value == val and s != "HW":
                    d.dim_param = s
                    break
    del g.value_info[:]
    onnx.save(m, a.out)
    print(
        f"{len(replaced)} constants made dynamic -> {a.out} ({os.path.getsize(a.out) / 2**20:.2f} MiB)"
    )


if __name__ == "__main__":
    main()
