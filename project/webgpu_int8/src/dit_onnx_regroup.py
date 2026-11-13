"""Regroup a segmented browser DiT set (dit_onnx_int8.py's output) into fewer segments.

Consecutive source segments are concatenated into one graph; the split is the contiguous
one with the smallest largest sidecar range. The sidecar bytes do not move: a group's
range is the union of its segments' ranges, and each external initializer's offset is
shifted by its segment's start within the group. Names private to a segment (initializers,
node outputs that are not segment outputs) get a ``__s<k>`` suffix, since the export reuses
``val_N`` and dit_onnx_dynamic.py's ``dyn__*`` names in every segment.

    uv run --no-project --with onnx --with onnxruntime --with numpy python \
        project/webgpu_int8/src/dit_onnx_regroup.py \
        --src output/webgpu_int8/onnx_v4/dit_int8_v4_dyn --nseg 2 \
        --out output/webgpu_int8/onnx_v4_2seg/dit_int8_v4_2seg --check 32x48

``--src`` and ``--out`` are ``<dir>/<model name>``; the manifest is
``<model name>.segments.json``. The sidecar bytes are unchanged and hard-linked into the
output dir (renamed with ``--sidecar_name``). ``--check`` runs both sets with ORT's
default graph optimisation and prints each group boundary tensor's distance to the
source's. On the CPU EP an fp16 set does not match: CPU has no fp16 kernels for most ops,
wraps them in fp32 Casts and drops back-to-back Cast pairs at every optimisation level,
so a consumer that read a segment's fp16 graph input reads the producer's unrounded fp32
value once merged (v4_preview int8, 32x48: ``noise_pred`` rel 0.025, the int8 activation
rounding of ``accuracy_level`` 4 amplifying it). Promoted to fp32, the merged and source
sets agree bit for bit, and with the ``__s<k>`` suffixes stripped the nodes are the
source's one for one.
"""

from __future__ import annotations

import argparse
import json
import os

import onnx


def split_points(sizes, n):
    """Contiguous split of sizes into n groups minimising the largest group sum."""
    best = None

    def rec(start, left, cur):
        nonlocal best
        if left == 1:
            groups = cur + [list(range(start, len(sizes)))]
            worst = max(sum(sizes[i] for i in g) for g in groups)
            if best is None or worst < best[0]:
                best = (worst, groups)
            return
        for end in range(start + 1, len(sizes) - left + 2):
            rec(end, left - 1, cur + [list(range(start, end))])

    rec(0, n, [])
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="<dir>/<model name> of the source set")
    ap.add_argument("--out", required=True, help="<dir>/<model name> to write")
    ap.add_argument("--nseg", type=int, default=2)
    ap.add_argument(
        "--sidecar_name",
        default="",
        help="file name of the output sidecar (default: the source's)",
    )
    ap.add_argument(
        "--check",
        default="",
        help="latent grids HxW (comma list) to run source and output on ORT CPU and "
        "compare every group boundary tensor",
    )
    ap.add_argument("--ep", default="cpu", choices=("cpu", "cuda"))
    a = ap.parse_args()

    src_dir, src_name = os.path.split(a.src)
    out_dir, out_name = os.path.split(a.out)
    man = json.load(open(a.src + ".segments.json"))
    path_prefix = man["model"][: -len(src_name)]
    segs = man["segments"]
    for s in segs:
        assert len(s["data"]) == 1, s["graph"]
    rng = [s["data"][0]["range"] for s in segs]
    for (lo0, hi0), (lo1, _) in zip(rng, rng[1:]):
        assert hi0 <= lo1, "segment ranges overlap"
    groups = split_points([hi - lo for lo, hi in rng], a.nseg)

    src_side = os.path.join(src_dir, os.path.basename(man["sidecar"]))
    side_name = a.sidecar_name or os.path.basename(man["sidecar"])
    os.makedirs(out_dir, exist_ok=True)
    dst_side = os.path.join(out_dir, side_name)
    if not os.path.exists(dst_side):
        os.link(src_side, dst_side)
    data_stem = side_name.removesuffix(".onnx.data")

    # tensors a group must export: inputs of later groups and the model outputs
    needed_after = [set() for _ in groups]
    for gi in range(len(groups)):
        for gj in range(gi + 1, len(groups)):
            for k in groups[gj]:
                needed_after[gi].update(segs[k]["inputs"])
    needed_after[-1].update(man["outputs"])

    out_segs = []
    for gi, g in enumerate(groups):
        glo, ghi = rng[g[0]][0], rng[g[-1]][1]
        data_name = f"{data_stem}.seg{gi}.data"
        merged = None
        produced, inputs, outputs = set(), [], []
        seen_in = set()
        for k in g:
            m = onnx.load(
                os.path.join(src_dir, os.path.basename(segs[k]["graph"])),
                load_external_data=False,
            )
            gg = m.graph
            boundary = {i.name for i in gg.input} | {o.name for o in gg.output}
            private = {t.name for t in gg.initializer} | {
                o for n in gg.node for o in n.output
            }
            private -= boundary | {""}  # "" marks an absent optional input / output
            ren = {nm: f"{nm}__s{k}" for nm in private}
            for t in gg.initializer:
                t.name = ren.get(t.name, t.name)
                if t.data_location == onnx.TensorProto.EXTERNAL:
                    for e in t.external_data:
                        if e.key == "location":
                            e.value = data_name
                        elif e.key == "offset":
                            e.value = str(int(e.value) + rng[k][0] - glo)
            for n in gg.node:
                if n.name:
                    n.name = f"{n.name}__s{k}"
                for i, x in enumerate(n.input):
                    if x in ren:
                        n.input[i] = ren[x]
                for i, x in enumerate(n.output):
                    if x in ren:
                        n.output[i] = ren[x]
            for vi in gg.value_info:
                vi.name = ren.get(vi.name, vi.name)
            for i in gg.input:
                if i.name not in produced and i.name not in seen_in:
                    inputs.append(i)
                    seen_in.add(i.name)
            for o in gg.output:
                produced.add(o.name)
                outputs.append(o)
            if merged is None:
                merged = m
                del merged.graph.input[:]
                del merged.graph.output[:]
            else:
                merged.graph.node.extend(gg.node)
                merged.graph.initializer.extend(gg.initializer)
                merged.graph.value_info.extend(gg.value_info)
                have = {(o.domain, o.version) for o in merged.opset_import}
                for o in m.opset_import:
                    if (o.domain, o.version) not in have:
                        merged.opset_import.append(o)
        out_list = []
        names = set()
        for o in outputs:
            if o.name in needed_after[gi] and o.name not in names:
                out_list.append(o)
                names.add(o.name)
        merged.graph.input.extend(inputs)
        merged.graph.output.extend(out_list)
        merged.graph.name = f"{out_name}_seg{gi}"
        gpath = os.path.join(out_dir, f"{out_name}.seg{gi}.onnx")
        onnx.save(merged, gpath)
        out_segs.append(
            {
                "graph": f"{path_prefix}{out_name}.seg{gi}.onnx",
                "data": [{"name": data_name, "range": [glo, ghi]}],
                "inputs": [i.name for i in inputs],
                "outputs": [o.name for o in out_list],
            }
        )
        print(
            f"seg{gi} <- source {g}: sidecar [{glo}, {ghi}) "
            f"{(ghi - glo) / 2**30:.3f} GiB, graph {os.path.getsize(gpath) / 2**20:.2f} MiB, "
            f"{len(inputs)} in / {len(out_list)} out"
        )

    out_man = {
        "model": path_prefix + out_name,
        "sidecar": path_prefix + side_name,
        "outputs": man["outputs"],
        "segments": out_segs,
    }
    with open(a.out + ".segments.json", "w") as f:
        json.dump(out_man, f, indent=1)
    print("wrote", a.out + ".segments.json")
    if a.check:
        check(a, src_side, dst_side)


def check(a, src_side, dst_side):
    """Source and regrouped set on one EP, same input."""
    import numpy as np
    from dit_onnx_dyn_eval import Segmented, feeds_for, rel

    kw = dict(
        providers=("CUDAExecutionProvider",)
        if a.ep == "cuda"
        else ("CPUExecutionProvider",),
    )
    src = Segmented(a.src + ".segments.json", os.path.dirname(a.src), src_side, **kw)
    out = Segmented(a.out + ".segments.json", os.path.dirname(a.out), dst_side, **kw)
    for hw in a.check.split(","):
        H, W = map(int, hw.split("x"))
        f = feeds_for(H, W)
        es, eo = src.run(f), out.run(f)
        es2 = src.run(f)
        print(f"{H}x{W}: source rerun noise_pred rel {rel(es2['noise_pred'], es['noise_pred']):.2e}")
        for s in out.man["segments"]:
            for k in s["outputs"]:
                x, y = es[k].astype(np.float32), eo[k].astype(np.float32)
                print(
                    f"  {k:24s} rel {rel(y, x):.2e} max|d| {np.abs(y - x).max():.3g}"
                )


if __name__ == "__main__":
    main()
