"""int8 ConvRot GEMMs at the PaddleOCR-VL-1.6 vision tower's shapes
(``docs/proposal/ocr_additionals.md`` § 5).

The tower: 27 layers, hidden 1152, MLP 4304 (gelu tanh), 16 heads. comfy_kitchen's
Hadamard takes groups 16 / 64 / 256 and fuses rotate + quantize only at group 256, so
K = 1152 runs staged at group 64 or padded to 1280 (5 x 256) for the fused path; fc1 gets
48 zero output rows (N 4352) so fc2's K = 4352 = 17 x 256 is fused with no copy.

Arms, one tower layer (q, k, v, out, fc1, gelu, fc2), times 27:

* ``bf16`` — F.linear throughout.
* ``fc2`` — bf16, but fc2 int8 group 256 fused with gelu folded into its quantizer.
* ``g64`` — K 1152 linears int8 group 64 staged; fc2 as in ``fc2``.
* ``pad`` — K 1152 linears padded to 1280 per call (F.pad), int8 group 256 fused; fc2 as above.

Forward and the input-gradient GEMM (``full`` mode: ``dY @ W`` on int8 ``Wᵀ``, K = N)
are timed separately; the tower is frozen under a LoRA, so its weight gradient never runs.
``--real_acts N`` captures the inputs of layers 0 / 13 / 26 on N val crops and reports each
arm's relative error against fp32 on the real weights.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import re
import statistics
import sys

import comfy_kitchen
import torch
import torch.nn.functional as F
from safetensors import safe_open

from bench._common import make_run_dir, write_result
from library.env import resolve_under_home
from library.int8.convrot import INT8_MAX, rotate, row_scale

MODEL = "models/paddleocr_vl_1.6"
OCR_DIR = "project/finished/ocr_reader/ocr"
H, MLP, MLP_PAD, LAYERS = 1152, 4304, 4352, 27
KEY = "visual.vision_model.encoder.layers.{i}.{m}"
MODS = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "out": "self_attn.out_proj",
    "fc1": "mlp.fc1",
    "fc2": "mlp.fc2",
}


def quant(w: torch.Tensor, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    w_rot = rotate(w.float(), group)
    s = row_scale(w_rot)
    q = (w_rot / s[:, None]).round().clamp(-INT8_MAX - 1, INT8_MAX)
    return q.to(torch.int8).contiguous(), s[:, None].contiguous()


def i8(x, wq, s, b, group, act=None):
    return comfy_kitchen.int8_linear(
        x,
        wq,
        s,
        bias=b,
        out_dtype=torch.bfloat16,
        convrot=True,
        convrot_groupsize=group,
        input_act=act,
    )


def load_layer(i: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    path = resolve_under_home(f"{MODEL}/model.safetensors")
    with safe_open(str(path), "pt", device="cuda") as f:
        return {
            n: (
                f.get_tensor(KEY.format(i=i, m=m) + ".weight"),
                f.get_tensor(KEY.format(i=i, m=m) + ".bias"),
            )
            for n, m in MODS.items()
        }


class Layer:
    """One layer's weights in every form the arms use."""

    def __init__(self, wb: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        self.w = {n: w.bfloat16() for n, (w, _) in wb.items()}
        self.b = {n: b.bfloat16() for n, (_, b) in wb.items()}
        w1, b1 = self.w["fc1"], self.b["fc1"]
        # fc1 with 48 zero output rows: gelu(0) = 0 feeds fc2's padded K
        self.w["fc1p"] = F.pad(w1, (0, 0, 0, MLP_PAD - MLP))
        self.b["fc1p"] = F.pad(b1, (0, MLP_PAD - MLP))
        self.w["fc2p"] = F.pad(self.w["fc2"], (0, MLP_PAD - MLP))
        self.b["fc2p"] = self.b["fc2"]
        self.q: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        for n in ("q", "k", "v", "out", "fc1p"):
            w = self.w[n]
            self.q[n, "g64"] = quant(w, 64)
            self.q[n, "pad"] = quant(F.pad(w, (0, 1280 - H)), 256)
        self.q["fc2p", "g256"] = quant(self.w["fc2p"], 256)
        self.q["fc2p", "g64"] = quant(self.w["fc2p"], 64)
        # transposed copies for the input gradient (K = N)
        for n in ("q", "k", "v", "out", "fc2p"):  # N = 1152
            wt = self.w[n].t()
            self.q[n + "_t", "g64"] = quant(wt, 64)
            self.q[n + "_t", "pad"] = quant(F.pad(wt, (0, 1280 - H)), 256)
        self.q["fc1p_t", "g256"] = quant(self.w["fc1p"].t(), 256)  # N = 4352


def fwd(L: Layer, x: torch.Tensor, arm: str) -> torch.Tensor:
    """q, k, v, out, fc1, gelu, fc2 on the same (M, 1152) input (attention core omitted)."""
    w, b, q = L.w, L.b, L.q
    xp = F.pad(x, (0, 1280 - H)) if arm == "pad" else None
    for n in ("q", "k", "v", "out"):
        if arm in ("bf16", "fc2"):
            y = F.linear(x, w[n], b[n])
        elif arm == "g64":
            y = i8(x, *q[n, "g64"], b[n], 64)
        else:
            y = i8(xp, *q[n, "pad"], b[n], 256)
    if arm == "bf16":
        h = F.linear(x, w["fc1"], b["fc1"])
        return F.linear(F.gelu(h, approximate="tanh"), w["fc2"], b["fc2"]) + y
    if arm == "fc2":
        h = F.linear(x, w["fc1p"], b["fc1p"])
    elif arm == "g64":
        h = i8(x, *q["fc1p", "g64"], b["fc1p"], 64)
    else:
        h = i8(xp, *q["fc1p", "pad"], b["fc1p"], 256)
    return i8(h, *q["fc2p", "g256"], b["fc2p"], 256, act="gelu_tanh") + y


def bwd(L: Layer, dy: torch.Tensor, dh: torch.Tensor, arm: str) -> torch.Tensor:
    """Input-gradient GEMMs of the same seven ops: dy (M, 1152) through q / k / v / out /
    fc2, dh (M, 4352) through fc1."""
    w, q = L.w, L.q
    dyp = F.pad(dy, (0, 1280 - H)) if arm == "pad" else None
    acc = None
    for n in ("q", "k", "v", "out", "fc2p"):
        if arm == "bf16" or (arm == "fc2" and n != "fc2p"):
            g = dy @ w[n]
        elif arm == "fc2" or arm == "g64":
            g = i8(dy, *q[n + "_t", "g64"], None, 64)
        else:
            g = i8(dyp, *q[n + "_t", "pad"], None, 256)
        acc = g[:, :8] if acc is None else acc + g[:, :8]
    if arm == "bf16":
        g = dh[:, :MLP] @ w["fc1"]
    else:
        g = i8(dh, *q["fc1p_t", "g256"], None, 256)
    return acc + g[:, :8]


def time_ms(fn, warm: int, iters: int) -> float:
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(True)
        a.record()
        fn()
        z.record()
        z.synchronize()
        ts.append(a.elapsed_time(z))
    return statistics.median(ts)


def capture_acts(
    n_crops: int, layers: list[int]
) -> dict[tuple[int, str], torch.Tensor]:
    """Inputs of each tower linear in ``layers`` on ``n_crops`` val crops (bf16, (T, K))."""
    sys.path.insert(0, str(resolve_under_home(OCR_DIR)))
    import crop_dataset as cd
    from finetune_vl16_lora import Collate
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base = str(resolve_under_home(MODEL))
    proc = AutoProcessor.from_pretrained(base)
    model = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    df = cd.load_sfxp("val", limit=n_crops)
    ds = cd.CropDataset(df, augment=False)
    chats = {
        t: proc.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": t}],
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        for t in (cd.OCR_TASK, cd.SFX_TASK)
    }
    collate = Collate(
        proc,
        [chats[t] for t in df.task],
        proc.image_processor.size["shortest_edge"],
        2048 * 196,
    )
    enc, _ = collate([ds[i] for i in range(len(ds))])
    enc.pop("labels"), enc.pop("n_boundary_fallback")
    pat = re.compile(r"visual\.vision_model\.encoder\.layers\.(\d+)\.(.+)$")
    rev = {m: n for n, m in MODS.items()}
    acts, hooks = {}, []
    for name, mod in model.named_modules():
        hit = pat.search(name)
        if hit and int(hit[1]) in layers and hit[2] in rev:
            key = (int(hit[1]), rev[hit[2]])
            hooks.append(
                mod.register_forward_pre_hook(
                    lambda _m, args, key=key: acts.__setitem__(
                        key, args[0].detach().reshape(-1, args[0].shape[-1]).clone()
                    )
                )
            )
    with torch.no_grad():
        model(**enc.to("cuda"), logits_to_keep=1)
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return acts


def rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return float((y.float() - ref).norm() / ref.norm())


def errors(acts, layers: list[int]) -> dict:
    out = {}
    for i in layers:
        L = Layer(load_layer(i))
        row = {}
        for n in ("q", "k", "v", "out", "fc1", "fc2"):
            x = acts[i, n].bfloat16()
            w, b = L.w[n], L.b[n]
            ref = F.linear(x.float(), w.float(), b.float())
            r = {"bf16": rel(F.linear(x, w, b), ref)}
            if n == "fc2":
                xp = F.pad(x, (0, MLP_PAD - MLP))
                r["g256"] = rel(i8(xp, *L.q["fc2p", "g256"], b, 256), ref)
                r["g64"] = rel(i8(xp, *L.q["fc2p", "g64"], b, 64), ref)
            else:
                nn_ = "fc1p" if n == "fc1" else n
                r["g64"] = rel(
                    i8(x, *L.q[nn_, "g64"], L.b[nn_], 64)[:, : ref.shape[1]], ref
                )
                r["pad"] = rel(
                    i8(F.pad(x, (0, 1280 - H)), *L.q[nn_, "pad"], L.b[nn_], 256)[
                        :, : ref.shape[1]
                    ],
                    ref,
                )
            r["tokens"] = x.shape[0]
            # activation outlier read: max |x| over the row RMS, median over rows
            rms = x.float().pow(2).mean(1).sqrt().clamp(min=1e-6)
            r["peak_over_rms_p50"] = float((x.float().abs().amax(1) / rms).median())
            row[n] = r
        out[f"layer{i}"] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--m", type=int, nargs="+", default=[650, 3072])
    ap.add_argument("--layer", type=int, default=13, help="weights for the timing")
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--real_acts", type=int, default=8, help="val crops (0 = skip)")
    ap.add_argument("--err_layers", type=int, nargs="+", default=[0, 13, 26])
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    print(f"comfy_kitchen {importlib.metadata.version('comfy-kitchen')}", flush=True)

    err = {}
    if a.real_acts:
        acts = capture_acts(a.real_acts, a.err_layers)
        err = errors(acts, a.err_layers)
        for li, row in err.items():
            for n, r in row.items():
                print(
                    f"[err {li} {n}] "
                    + " ".join(
                        f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                        for k, v in r.items()
                    ),
                    flush=True,
                )
        del acts

    L = Layer(load_layer(a.layer))
    arms = ("bf16", "fc2", "g64", "pad")
    timing = {}
    g = torch.Generator(device="cuda").manual_seed(0)
    for m in a.m:
        x = torch.randn(m, H, device="cuda", generator=g).bfloat16()
        dy = torch.randn(m, H, device="cuda", generator=g).bfloat16()
        dh = torch.randn(m, MLP_PAD, device="cuda", generator=g).bfloat16()
        dh[:, MLP:] = 0
        row = {}
        for arm in arms:
            f_ms = time_ms(lambda: fwd(L, x, arm), a.warm, a.iters)
            b_ms = time_ms(lambda: bwd(L, dy, dh, arm), a.warm, a.iters)
            row[arm] = {
                "fwd_ms_layer": f_ms,
                "dx_ms_layer": b_ms,
                "fwd_ms_tower": f_ms * LAYERS,
                "dx_ms_tower": b_ms * LAYERS,
            }
        for arm in arms:
            r = row[arm]
            r["fwd_vs_bf16"] = r["fwd_ms_layer"] / row["bf16"]["fwd_ms_layer"]
            r["dx_vs_bf16"] = r["dx_ms_layer"] / row["bf16"]["dx_ms_layer"]
            print(
                f"[M {m}] {arm:5s} fwd {r['fwd_ms_layer']:.3f} ms/layer "
                f"({r['fwd_vs_bf16']:.2f}x)  dx {r['dx_ms_layer']:.3f} ms/layer "
                f"({r['dx_vs_bf16']:.2f}x)  tower fwd+dx "
                f"{r['fwd_ms_tower'] + r['dx_ms_tower']:.1f} ms",
                flush=True,
            )
        timing[f"M{m}"] = row

    # per-op forward split at the largest M (where does the time go)
    m = max(a.m)
    x = torch.randn(m, H, device="cuda", generator=g).bfloat16()
    xp = F.pad(x, (0, 1280 - H))
    h = torch.randn(m, MLP_PAD, device="cuda", generator=g).bfloat16()
    w, b, q = L.w, L.b, L.q
    ops = {
        "q_bf16": lambda: F.linear(x, w["q"], b["q"]),
        "q_g64": lambda: i8(x, *q["q", "g64"], b["q"], 64),
        "q_pad_incl_copy": lambda: i8(F.pad(x, (0, 128)), *q["q", "pad"], b["q"], 256),
        "q_pad_no_copy": lambda: i8(xp, *q["q", "pad"], b["q"], 256),
        "fc1_bf16": lambda: F.linear(x, w["fc1"], b["fc1"]),
        "fc1_g64": lambda: i8(x, *q["fc1p", "g64"], b["fc1p"], 64),
        "fc2_bf16_gelu": lambda: F.linear(
            F.gelu(h[:, :MLP], approximate="tanh"), w["fc2"], b["fc2"]
        ),
        "fc2_g256_actfold": lambda: i8(
            h, *q["fc2p", "g256"], b["fc2p"], 256, act="gelu_tanh"
        ),
        "fc2_g256_gelu_eager": lambda: i8(
            F.gelu(h, approximate="tanh"), *q["fc2p", "g256"], b["fc2p"], 256
        ),
        "fc2_g64": lambda: i8(h, *q["fc2p", "g64"], b["fc2p"], 64),
    }
    per_op = {k: time_ms(f, a.warm, a.iters) for k, f in ops.items()}
    for k, v in per_op.items():
        print(f"[op M {m}] {k:22s} {v * 1000:.1f} us", flush=True)

    run_dir = make_run_dir("ocr_tower_int8", label=a.label or None)
    write_result(
        run_dir,
        script=__file__,
        args=a,
        metrics={
            "comfy_kitchen": importlib.metadata.version("comfy-kitchen"),
            "timing": timing,
            "per_op_ms": per_op,
            "error": err,
        },
    )
    print(f"result: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
