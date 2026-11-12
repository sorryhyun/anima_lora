"""Rewrite the browser DiT (HF ``onnx/dit_*_fp16_r16``) onto new weights, optionally int8.

The source is the published segmented fp16 export: five resolution buckets, seven
segments each, one shared sidecar (or, with ``--buckets dyn``, dit_onnx_dynamic.py's
one resolution-free set over that sidecar). Its weight initializers are mostly anonymous
(``val_N``) and some carry the r16 residual rescale (×1/16), so each 2-D fp16
initializer is identified by value: cosine against every ``--base`` tensor of the same
shape (or its transpose), and the scale factor read off the projection and snapped to
1 or 1/16. Every identified weight is replaced by the ``--dit`` tensor in the same
layout times that factor. Other initializers (RoPE tables, norms, shapes) are kept.

With ``--gptq`` (dit_convrot_gptq.py's file) each converted linear's MatMul becomes
``X -> Reshape -> MatMul H256 -> Reshape -> MatMulNBits`` (bits 8, block from the file,
zero point 128, ``--acc``), the factor folded into the fp16 scales. Linears reading the
same X share one rotation.

Output layout (``--out_dir``): per bucket ``<prefix>_<bucket>.segN.onnx`` and
``<prefix>_<bucket>.segments.json``, one shared ``<prefix>.onnx.data``. Initializers
identical across the five buckets of a segment live in the sidecar, one contiguous
range per segment; bucket-specific ones (resolution constants) are inlined in the graph.

    uv run --no-project --with onnx --with numpy --with safetensors --with torch \
        python project/webgpu_int8/src/dit_onnx_int8.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from safetensors import safe_open

BUCKETS = ("512", "624x416", "416x624", "688x384", "384x688")
NSEG = 7
ROT = 256
ALIGN = 64
CONVERTED = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.output_proj",
    "mlp.layer1",
    "mlp.layer2",
)


def hadamard(n):
    h = np.ones((1, 1), np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float16)


class Source:
    """The published graphs, manifests and sidecar."""

    def __init__(self, hf_dir, src_prefix, sidecar):
        self.graphs, self.man = {}, {}
        for b in BUCKETS:
            base = os.path.join(hf_dir, f"{src_prefix.format(b=b)}")
            self.man[b] = json.load(open(base + ".segments.json"))
            self.graphs[b] = [
                onnx.load(base + f".seg{k}.onnx", load_external_data=False)
                for k in range(NSEG)
            ]
        self.data = np.memmap(sidecar, dtype=np.uint8, mode="r")

    def array(self, b, k, t) -> np.ndarray:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            return numpy_helper.to_array(t)
        ext = {e.key: e.value for e in t.external_data}
        rng = {d["name"]: d["range"] for d in self.man[b]["segments"][k]["data"]}
        lo = rng[ext["location"]][0] + int(ext.get("offset", 0))
        raw = self.data[lo : lo + int(ext["length"])].tobytes()
        dt = helper.tensor_dtype_to_np_dtype(t.data_type)
        return np.frombuffer(raw, dtype=dt).reshape(tuple(t.dims))


def identify(src: Source, base_path: str):
    """{(k, init name) in bucket 512: (ckpt key, transposed, factor)}."""
    with safe_open(base_path, "pt") as f:
        base = {
            k: f.get_tensor(k)
            for k in f.keys()
            if k.startswith("net.")
            and f.get_slice(k).get_shape().__len__() == 2
            and "llm_adapter" not in k
        }
    by_shape = defaultdict(list)
    for k, w in base.items():
        by_shape[tuple(w.shape)].append((k, False))
        if w.shape[0] != w.shape[1]:
            by_shape[tuple(w.shape[::-1])].append((k, True))
        else:
            by_shape[tuple(w.shape)].append((k, True))
    inits = defaultdict(list)
    for k, g in enumerate(src.graphs[BUCKETS[0]]):
        for t in g.graph.initializer:
            if t.data_type == onnx.TensorProto.FLOAT16 and len(t.dims) == 2:
                if tuple(t.dims) in by_shape:
                    inits[tuple(t.dims)].append((k, t))
    found = {}
    for shape, items in inits.items():
        cands = by_shape[shape]
        C = torch.stack(
            [
                (base[k].float().T if tr else base[k].float()).reshape(-1)
                for k, tr in cands
            ]
        )
        Cn = C / C.norm(dim=1, keepdim=True)
        X = torch.stack(
            [
                torch.from_numpy(
                    src.array(BUCKETS[0], k, t).astype(np.float32)
                ).reshape(-1)
                for k, t in items
            ]
        )
        cos = (X / X.norm(dim=1, keepdim=True)) @ Cn.T
        top2 = cos.topk(min(2, cos.shape[1]), dim=1)
        for i, (k, t) in enumerate(items):
            j = top2.indices[i, 0].item()
            c1 = top2.values[i, 0].item()
            c2 = top2.values[i, 1].item() if top2.values.shape[1] > 1 else 0.0
            key, tr = cands[j]
            if c1 < 0.8 or c1 - c2 < 0.05:
                print(f"  unmatched {t.name} {shape} cos {c1:.3f} / {c2:.3f} ({key})")
                continue
            f = (X[i] @ C[j] / (C[j] @ C[j])).item()
            snap = min((1.0, 1 / 16), key=lambda r: abs(np.log(abs(f) / r)))
            if abs(f / snap - 1) > 0.15:
                raise SystemExit(f"{t.name} -> {key}: factor {f:.4f} snaps to neither")
            found[(k, t.name)] = (key, tr, snap, c1)
    keys = [v[0] for v in found.values()]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise SystemExit(f"tensors matched twice: {sorted(dup)[:5]}")
    return found


def main():
    global BUCKETS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf_dir", required=True, help="dir holding the source graphs")
    ap.add_argument("--src_prefix", default="dit_{b}_fp16_r16")
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--base", required=True, help="checkpoint to identify weights by")
    ap.add_argument("--dit", required=True, help="checkpoint whose weights are written")
    ap.add_argument("--gptq", default="", help="int8 file; omit for an fp16 rewrite")
    ap.add_argument("--acc", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", default="dit_int8")
    ap.add_argument(
        "--path_prefix",
        default="int8/",
        help="prefix of graph / sidecar paths in the manifests",
    )
    ap.add_argument(
        "--buckets",
        default=",".join(BUCKETS),
        help="source buckets, the first identifies weights; 'dyn' for dit_onnx_dynamic.py's set",
    )
    a = ap.parse_args()
    BUCKETS = tuple(a.buckets.split(","))

    src = Source(a.hf_dir, a.src_prefix, a.sidecar)
    found = identify(src, a.base)
    per = defaultdict(int)
    for key, tr, snap, c in found.values():
        per[(key.split(".")[-2] if "blocks" in key else key, snap)] += 1
    print(
        f"identified {len(found)} weights; min cos "
        f"{min(v[3] for v in found.values()):.3f}"
    )
    for k, v in sorted(per.items()):
        print(f"  {v:3d} {k[0]} x{k[1]:g}")

    q = {}
    if a.gptq:
        with safe_open(a.gptq, "pt") as f:
            q = {k: f.get_tensor(k) for k in f.keys()}
    dit = safe_open(a.dit, "pt")
    H = numpy_helper.from_array(hadamard(ROT), "convrot_h256")

    # content of each first-bucket initializer by name, to map other buckets onto it
    h512 = {}
    for k, g in enumerate(src.graphs[BUCKETS[0]]):
        for t in g.graph.initializer:
            if (k, t.name) in found:
                h512[
                    (k, hashlib.sha1(src.array(BUCKETS[0], k, t).tobytes()).hexdigest())
                ] = t.name

    new_w_cache = {}

    def new_weight(key, tr, snap):
        if (key, tr, snap) not in new_w_cache:
            w = dit.get_tensor(key).float()
            w = (w.T if tr else w) * snap
            new_w_cache[(key, tr, snap)] = w.contiguous().numpy().astype(np.float16)
        return new_w_cache[(key, tr, snap)]

    n_conv = defaultdict(int)

    def build(b, k, g):
        m = onnx.ModelProto()
        m.CopyFrom(g)
        gg = m.graph
        # name -> (ckpt key, transposed, factor) for this graph's weights
        ident = {}
        arrays = {}
        for t in gg.initializer:
            arr = src.array(b, k, t)
            arrays[t.name] = arr
            if t.data_type == onnx.TensorProto.FLOAT16 and len(t.dims) == 2:
                h = (k, hashlib.sha1(arr.tobytes()).hexdigest())
                if h in h512:
                    ident[t.name] = found[(k, h512[h])][:3]
        nodes, extra, drop, rot_cache = [], [], set(), {}
        for n in gg.node:
            key = None
            if n.op_type == "MatMul" and n.input[1] in ident:
                ck, tr, snap = ident[n.input[1]]
                short = ck.removeprefix("net.").removesuffix(".weight")
                if q and short.split(".", 2)[-1] in CONVERTED and tr:
                    key = short
            if key is None:
                nodes.append(n)
                continue
            Kd, Nd = arrays[n.input[1]].shape
            qi = q[f"{key}.q"].numpy()  # (N, K) int8, rotated basis
            sc = q[f"{key}.scale"].float().numpy() * snap  # (N, K/blk)
            assert qi.shape == (Nd, Kd), (key, qi.shape, (Nd, Kd))
            blk = Kd // sc.shape[1]
            x = n.input[0]
            if x not in rot_cache:
                tag = f"{x}__rot"
                extra.append(
                    numpy_helper.from_array(
                        np.array([-1, Kd // ROT, ROT], np.int64), f"{tag}_s3"
                    )
                )
                nodes += [
                    helper.make_node(
                        "Shape", [x], [f"{tag}_shape"], name=f"{tag}_shape"
                    ),
                    helper.make_node(
                        "Reshape", [x, f"{tag}_s3"], [f"{tag}_r3"], name=f"{tag}_r3"
                    ),
                    helper.make_node(
                        "MatMul",
                        [f"{tag}_r3", "convrot_h256"],
                        [f"{tag}_mm"],
                        name=f"{tag}_mm",
                    ),
                    helper.make_node(
                        "Reshape",
                        [f"{tag}_mm", f"{tag}_shape"],
                        [tag],
                        name=f"{tag}_back",
                    ),
                ]
                rot_cache[x] = tag
            bq = (
                (qi.astype(np.int16) + 128).astype(np.uint8).reshape(Nd, Kd // blk, blk)
            )
            extra += [
                numpy_helper.from_array(bq, f"{key}.qweight"),
                numpy_helper.from_array(
                    sc.astype(np.float16).reshape(-1), f"{key}.scales"
                ),
            ]
            nodes.append(
                helper.make_node(
                    "MatMulNBits",
                    [rot_cache[x], f"{key}.qweight", f"{key}.scales"],
                    list(n.output),
                    name=f"{key}.matmulnbits",
                    domain="com.microsoft",
                    K=Kd,
                    N=Nd,
                    bits=8,
                    block_size=blk,
                    accuracy_level=a.acc,
                )
            )
            drop.add(n.input[1])
            n_conv[b] += 1
        used = {i for n in nodes for i in n.input}
        keep = []
        for t in gg.initializer:
            if t.name in drop and t.name not in used:
                continue
            arr = arrays[t.name]
            if t.name in ident:
                arr = new_weight(*ident[t.name])
            keep.append(numpy_helper.from_array(np.ascontiguousarray(arr), t.name))
        if rot_cache:
            keep.append(H)
        del gg.initializer[:]
        gg.initializer.extend(keep + extra)
        del gg.node[:]
        gg.node.extend(nodes)
        if q and not any(o.domain == "com.microsoft" for o in m.opset_import):
            m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
        return m

    # per segment: build the five bucket graphs, put the initializers whose bytes every
    # bucket shares in the sidecar (one range per segment), save, free
    os.makedirs(a.out_dir, exist_ok=True)
    side_name = f"{a.prefix}.onnx.data"
    side = open(os.path.join(a.out_dir, side_name), "wb")
    pos, ranges, inlined = 0, [], 0
    for k in range(NSEG):
        seg_graphs = {b: build(b, k, src.graphs[b][k]) for b in BUCKETS}
        new_w_cache.clear()
        hashes = [
            {
                t.name: hashlib.sha1(t.raw_data).hexdigest()
                for t in seg_graphs[b].graph.initializer
            }
            for b in BUCKETS
        ]
        shared_hash = set(hashes[0].values()).intersection(
            *[set(h.values()) for h in hashes[1:]]
        )
        lo, placed = pos, {}
        for b_i, b in enumerate(BUCKETS):
            for t in seg_graphs[b].graph.initializer:
                h = hashes[b_i][t.name]
                if h not in shared_hash or len(t.raw_data) < 1024:
                    inlined += len(t.raw_data) if b == BUCKETS[0] else 0
                    continue
                if h not in placed:
                    pad = (-pos) % ALIGN
                    side.write(b"\0" * pad)
                    pos += pad
                    placed[h] = (pos - lo, len(t.raw_data))
                    side.write(t.raw_data)
                    pos += len(t.raw_data)
                off, ln = placed[h]
                t.ClearField("raw_data")
                t.data_location = onnx.TensorProto.EXTERNAL
                for kk, vv in (
                    ("location", f"{a.prefix}.seg{k}.data"),
                    ("offset", str(off)),
                    ("length", str(ln)),
                ):
                    e = t.external_data.add()
                    e.key, e.value = kk, vv
        ranges.append((lo, pos))
        for b in BUCKETS:
            onnx.save(
                seg_graphs[b], os.path.join(a.out_dir, f"{a.prefix}_{b}.seg{k}.onnx")
            )
        del seg_graphs
        print(f"seg{k}: sidecar [{lo}, {pos}) {(pos - lo) / 2**20:.1f} MiB", flush=True)
    side.close()
    print(
        f"sidecar {pos / 2**30:.3f} GiB; inlined (first bucket) {inlined / 2**20:.2f} MiB"
    )
    if q:
        print("converted MatMuls per bucket", dict(n_conv))

    for b in BUCKETS:
        name = f"{a.prefix}_{b}"
        segs = []
        for k in range(NSEG):
            s = src.man[b]["segments"][k]
            segs.append(
                {
                    "graph": a.path_prefix + f"{name}.seg{k}.onnx",
                    "data": [
                        {"name": f"{a.prefix}.seg{k}.data", "range": list(ranges[k])}
                    ],
                    "inputs": s["inputs"],
                    "outputs": s["outputs"],
                }
            )
        man = {
            "model": a.path_prefix + name,
            "sidecar": a.path_prefix + side_name,
            "outputs": src.man[b]["outputs"],
            "segments": segs,
        }
        with open(os.path.join(a.out_dir, f"{name}.segments.json"), "w") as f:
            json.dump(man, f, indent=1)
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
