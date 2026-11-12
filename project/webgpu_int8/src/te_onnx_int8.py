"""Rewrite text_encoder_fp16.onnx into the int8 ConvRot graph.

Each linear MatMul(X, Wᵀ) becomes  X -> [Reshape, MatMul H256, Reshape] -> MatMulNBits
(bits 8, block from the GPTQ file, default zero point 128, accuracy_level 4 -> DP4A on
WebGPU). The rotation is shared by linears reading the same X. The embedding Gather
becomes GatherBlockQuantized (uint8, bits 8, per-128 block, explicit zero point 128).

Linears are matched to (layer, name) by value against the original safetensors, not by
node order.

    uv run --no-project --with onnx --with numpy --with safetensors python \
        project/webgpu_int8/src/te_onnx_int8.py \
        <hf>/onnx/text_encoder_fp16.onnx output/webgpu_int8/te_int8.aa32k.safetensors \
        models/text_encoders/qwen_3_06b_base.safetensors \
        output/webgpu_int8/onnx/int8/text_encoder_int8.onnx
"""

import argparse
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper
from safetensors.numpy import load_file

ROT = 256
EBLK = 128


def hadamard(n):
    h = np.ones((1, 1), np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("gptq")
    ap.add_argument("te")
    ap.add_argument("out")
    ap.add_argument("--acc", type=int, default=4)
    a = ap.parse_args()

    m = onnx.load(a.src)  # with external data
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    q = load_file(a.gptq)
    te = load_file(a.te)
    names = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    want = {k[:-2] for k in q if k.endswith(".q")}
    lin_idx = [0]
    print("linears in gptq file:", len(want))

    new_nodes, new_inits = [], []
    rot_cache = {}
    used, drop, kept = set(), set(), []
    H = numpy_helper.from_array(hadamard(ROT), "convrot_h256")
    new_inits.append(H)

    def shape_const(name, vals):
        new_inits.append(numpy_helper.from_array(np.array(vals, np.int64), name))
        return name

    def rotated(x, K):
        if x in rot_cache:
            return rot_cache[x], []
        s4 = shape_const(f"{x}__rs4", [0, 0, K // ROT, ROT])
        s3 = shape_const(f"{x}__rs3", [0, 0, K])
        r4, mm, out = f"{x}__r4", f"{x}__rm", f"{x}__rot"
        nodes = [
            helper.make_node("Reshape", [x, s4], [r4], name=f"{x}__rot_reshape4"),
            helper.make_node("MatMul", [r4, "convrot_h256"], [mm], name=f"{x}__rot_mm"),
            helper.make_node("Reshape", [mm, s3], [out], name=f"{x}__rot_reshape3"),
        ]
        rot_cache[x] = out
        return out, nodes

    for n in g.node:
        key = None
        if n.op_type == "MatMul" and n.input[1] in inits:
            t = inits[n.input[1]]
            arr = numpy_helper.to_array(t)
            i = lin_idx[0]
            lin_idx[0] += 1
            name = f"layers.{i // 7}.{names[i % 7]}"
            w = te[f"model.{name}.weight"].astype(np.float32).T
            dmax = np.abs(arr.astype(np.float32) - w).max()
            if arr.shape != w.shape or dmax > 1e-6:
                raise SystemExit(
                    f"{n.name} is not {name}: {arr.shape} {w.shape} {dmax}"
                )
            if name in want:
                key = name
            else:
                kept.append(name)
        if key is not None:
            used.add(key)
            drop.add(t.name)
            K, N = arr.shape
            qi = q[f"{key}.q"]  # (N, K) int8, rotated basis
            sc = q[f"{key}.scale"]  # (N, K/b) fp16
            b = K // sc.shape[1]
            xr, rn = rotated(n.input[0], K)
            new_nodes += rn
            bq = (qi.astype(np.int16) + 128).astype(np.uint8).reshape(N, K // b, b)
            bn, sn = f"{key}.qweight", f"{key}.scales"
            new_inits.append(numpy_helper.from_array(bq, bn))
            new_inits.append(
                numpy_helper.from_array(sc.astype(np.float16).reshape(-1), sn)
            )
            new_nodes.append(
                helper.make_node(
                    "MatMulNBits",
                    [xr, bn, sn],
                    list(n.output),
                    name=f"{key}.matmulnbits",
                    domain="com.microsoft",
                    K=K,
                    N=N,
                    bits=8,
                    block_size=b,
                    accuracy_level=a.acc,
                )
            )
        elif (
            n.op_type == "Gather"
            and n.input[0] in inits
            and len(inits[n.input[0]].dims) == 2
        ):
            t = inits[n.input[0]]
            e = numpy_helper.to_array(t).astype(np.float32)
            V, D = e.shape
            blk = e.reshape(V, D // EBLK, EBLK)
            s = np.maximum(np.abs(blk).max(-1), 1e-12) / 127
            s16 = s.astype(np.float16)
            qe = np.clip(np.round(blk / s16.astype(np.float32)[..., None]), -127, 127)
            qe = (qe + 128).astype(np.uint8).reshape(V, D)
            drop.add(t.name)
            new_inits += [
                numpy_helper.from_array(qe, "embed.qdata"),
                numpy_helper.from_array(s16, "embed.scales"),
                numpy_helper.from_array(np.full(s16.shape, 128, np.uint8), "embed.zp"),
            ]
            new_nodes.append(
                helper.make_node(
                    "GatherBlockQuantized",
                    ["embed.qdata", n.input[1], "embed.scales", "embed.zp"],
                    list(n.output),
                    name="embed.gbq",
                    domain="com.microsoft",
                    gather_axis=0,
                    quantize_axis=1,
                    block_size=EBLK,
                    bits=8,
                )
            )
        else:
            new_nodes.append(n)
    assert used == want and lin_idx[0] == 196, (want - used, lin_idx)
    print("int8 linears", len(used), "kept fp16 MatMul with initializer:", kept)
    keep = [t for t in g.initializer if t.name not in drop]
    del g.initializer[:]
    g.initializer.extend(keep + new_inits)
    del g.node[:]
    g.node.extend(new_nodes)
    if not any(o.domain == "com.microsoft" for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    data = os.path.basename(a.out) + ".data"
    if os.path.exists(a.out + ".data"):
        os.remove(a.out + ".data")
    onnx.save(
        m,
        a.out,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data,
        size_threshold=1024,
    )
    print("wrote", a.out, os.path.getsize(a.out), os.path.getsize(a.out + ".data"))


if __name__ == "__main__":
    main()
