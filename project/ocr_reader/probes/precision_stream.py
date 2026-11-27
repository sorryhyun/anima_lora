#!/usr/bin/env python3
"""How far the bf16 forward of OvisOCR3 / PaddleOCR-VL 1.6 sits from fp32, and how much an
fp32 residual stream (MOSS / Anima's ``stream fp32``) takes back.

    make daemon-run ARGS="--queue --stall-timeout 600 \\
        project/ocr_reader/probes/precision_stream.py --model ovis3 --run prec_ovis3_<mmdd_hhmm>"

Both checkpoints are all-bf16, so every arm runs the same weight values; only where the
arithmetic rounds changes. Arms, on the base model (no adapter), the same val crops
(``load_mainsub("val")``) and the same batches:

* ``fp32``: every parameter and activation fp32, TF32 off. The reference.
* ``stream``: fp32 model; every ``nn.Linear`` inside a vision block or an LM decoder layer
  runs as a bf16 GEMM (input, weight and output rounded), and so do the attention core
  (SDPA on bf16 q / k / v) and the GDN chunk / recurrent rules (bf16 q / k / v / beta, the
  gate fp32 as in the bf16 model). Residual stream, norms, RoPE, GDN conv, embeddings,
  the merger and ``lm_head`` stay fp32.
* ``bf16``: the model as the trainers load it (``dtype=torch.bfloat16``, no autocast).
* ``bf16_head32``: ``bf16`` with ``lm_head`` run in fp32 on the bf16 hidden state.

Measured against ``fp32``:

* teacher-forced target logits (training's collate, ``--max_tokens`` cap): relative
  error, top-1 agreement, mean target CE;
* the CE gradient of every block linear's weight, summed over ``--n_grad`` crops:
  relative distance per linear, median over the tower and the LM's halves. ``floor`` is
  the fp32 gradient rounded to bf16 (what a bf16 gradient costs even when exact);
* greedy decoding of ``--n_decode`` crops with each trainer's own ``predict``: share of
  strings equal to the fp32 arm's, and main / sub exact against the labels.

Writes ``output/ocr/<run>/precision.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from project.ocr_reader.core import crop_dataset as cd
from project.ocr_reader.core.paths import OUTPUT, OVIS3_BASE, VL16_BASE
from project.ocr_reader.core.scoring import score
from project.ocr_reader.ocr import finetune_vl16_lora as vl

BLOCK = re.compile(r"\.(layers|blocks)\.\d+\.")
BF = torch.bfloat16


# --- arms -------------------------------------------------------------------------


class Stream:
    """The stream arm's wrappers; ``on`` toggles them, ``calls`` counts what they caught."""

    def __init__(self, model, model_kind: str):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        self.on = False
        self.calls = {"linear": 0, "attn": 0, "gdn": 0}
        self.linears = [
            (n, m)
            for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and BLOCK.search(n + ".")
        ]
        for _, m in self.linears:
            m.forward = self._linear(m)

        orig = ALL_ATTENTION_FUNCTIONS["sdpa"]

        def attn(module, query, key, value, attention_mask, *args, **kw):
            if not self.on:
                return orig(module, query, key, value, attention_mask, *args, **kw)
            self.calls["attn"] += 1
            dt = query.dtype
            if attention_mask is not None and attention_mask.is_floating_point():
                attention_mask = attention_mask.clamp(min=torch.finfo(BF).min).to(BF)
            o, w = orig(
                module,
                query.to(BF),
                key.to(BF),
                value.to(BF),
                attention_mask,
                *args,
                **kw,
            )
            return o.to(dt), w

        ALL_ATTENTION_FUNCTIONS.register("sdpa", attn)

        if model_kind == "ovis3":
            from transformers.models.qwen3_5 import modeling_qwen3_5 as q35

            for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
                setattr(q35, name, self._gdn(getattr(q35, name)))

    def _linear(self, m):
        def forward(x):
            if not self.on:
                return F.linear(x, m.weight, m.bias)
            self.calls["linear"] += 1
            b = None if m.bias is None else m.bias.to(BF)
            return F.linear(x.to(BF), m.weight.to(BF), b).to(x.dtype)

        return forward

    def _gdn(self, rule):
        def wrapped(query, key, value, g, beta, **kw):
            if not self.on:
                return rule(query, key, value, g=g, beta=beta, **kw)
            self.calls["gdn"] += 1
            dt = query.dtype
            o, state = rule(
                query.to(BF), key.to(BF), value.to(BF), g=g, beta=beta.to(BF), **kw
            )
            return o.to(dt), state

        return wrapped


@contextmanager
def head32(model):
    """lm_head in fp32 on whatever hidden state comes in; weights untouched."""
    head = model.get_output_embeddings()
    orig = head.forward
    head.forward = lambda x: F.linear(x.float(), head.weight.float())
    try:
        yield
    finally:
        head.forward = orig


# --- measurements -----------------------------------------------------------------


def teacher_forced(model, batches, params):
    """Target logits (fp32, CPU) and the summed CE gradient of ``params`` (fp32, CPU)."""
    for p in params:
        p.grad = None
    logits, tgts = [], []
    for enc in batches:
        enc = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in enc.items()}
        labels = enc.pop("labels")
        enc.pop("n_boundary_fallback", None)
        K = int((labels != -100).sum(1).max()) + 1
        z = model(**enc, logits_to_keep=K).logits[:, :-1]
        tgt = labels[:, -(K - 1) :]
        keep = tgt != -100
        z = z[keep].float()
        loss = F.cross_entropy(z, tgt[keep]) / len(batches)
        loss.backward()
        logits.append(z.detach().cpu())
        tgts.append(tgt[keep].cpu())
    grads = [p.grad.float().cpu() for p in params]
    for p in params:
        p.grad = None
    return torch.cat(logits), torch.cat(tgts), grads


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def grad_stats(names, grads, ref) -> dict:
    d = np.array([rel(g, r) for g, r in zip(grads, ref)])
    groups = {}
    lm = [i for i, n in enumerate(names) if "language_model" in n]
    tower = [i for i, n in enumerate(names) if "language_model" not in n]
    depth = lambda n: int(BLOCK.search(n + ".").group(0).split(".")[2])  # noqa: E731
    n_lm = max(depth(names[i]) for i in lm) + 1
    groups["tower"] = tower
    groups["lm_lo"] = [i for i in lm if depth(names[i]) < n_lm // 2]
    groups["lm_hi"] = [i for i in lm if depth(names[i]) >= n_lm // 2]
    return {k: float(np.median(d[v])) for k, v in groups.items() if v}


# --- main -------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", choices=("ovis3", "vl16"), required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n_grad", type=int, default=16)
    ap.add_argument("--grad_bs", type=int, default=4)
    ap.add_argument("--n_decode", type=int, default=128)
    ap.add_argument("--val_bs", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = OUTPUT / a.run
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    base = OVIS3_BASE if a.model == "ovis3" else VL16_BASE
    proc = AutoProcessor.from_pretrained(str(base))
    tok = proc.tokenizer

    def chat(task: str) -> str:
        kw = {"enable_thinking": False} if a.model == "ovis3" else {}
        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": task}]}]
        return proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, **kw)

    if a.model == "ovis3":
        from project.ocr_reader.ocr import finetune_ovis3_lora as ov

        plain = (OVIS3_BASE / "prompts" / "plain_ocr.txt").read_text(encoding="utf-8")
        chats = {cd.OCR_TASK: chat(plain.strip()), cd.OCR2_TASK: chat(cd.OCR2_TASK)}
        min_edge, max_px = ov.MIN_PX, a.max_tokens * ov.PATCH_PX
        eos = sorted({tok.eos_token_id, tok.convert_tokens_to_ids("<|endoftext|>")})

        def decode(model, prompts, ds):
            return ov.predict(model, proc, prompts, ds, a.val_bs, a.workers, eos)
    else:
        chats = {t: chat(t) for t in (cd.OCR_TASK, cd.OCR2_TASK)}
        min_edge = proc.image_processor.size["shortest_edge"]
        max_px = a.max_tokens * 196

        def decode(model, prompts, ds):
            return vl.predict(model, proc, prompts, min_edge, ds, a.val_bs, a.workers)

    va_df = cd.load_mainsub("val", neg_share=0.2, limit=a.n_decode, seed=0)
    va = cd.CropDataset(va_df, augment=False)
    prompts = [chats[t] for t in va_df.task]
    collate = vl.Collate(proc, prompts, min_edge, max_px)
    g_idx = list(range(min(a.n_grad, len(va))))
    batches = [
        collate([va[i] for i in g_idx[s : s + a.grad_bs]])[0]
        for s in range(0, len(g_idx), a.grad_bs)
    ]
    print(
        f"{a.model}: val {len(va)} crops ({va_df.task.value_counts().to_dict()}), "
        f"grad on {len(g_idx)} in {len(batches)} batches",
        flush=True,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        str(base), dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    stream = Stream(model, a.model)
    names = [n for n, _ in stream.linears]
    params = [m.weight for _, m in stream.linears]
    for p in params:
        p.requires_grad_(True)
    print(f"{len(params)} block linears, {sum(p.numel() for p in params) / 1e6:.0f}M", flush=True)

    arms = {}
    convert = {"fp32": model.float, "bf16": lambda: model.to(BF)}
    ref = None
    for arm in ("fp32", "stream", "bf16", "bf16_head32"):
        convert.get(arm, lambda: None)()
        stream.on = arm == "stream"
        stream.calls = dict.fromkeys(stream.calls, 0)
        t0 = time.time()
        with head32(model) if arm == "bf16_head32" else nullcontext():
            model.eval()  # predict() leaves the model in train(); no dropout either way
            z, tgt, grads = teacher_forced(model, batches, params)
            calls = dict(stream.calls)
            preds = decode(model, prompts, va)
        scored = score(va_df, preds)
        rec = {
            "ce": F.cross_entropy(z, tgt).item(),
            "val": vl.val_metrics(scored),
            "wall": round(time.time() - t0, 1),
        }
        if arm == "stream":
            rec["calls"] = calls
        if ref is None:
            ref = {"z": z, "grads": grads, "preds": preds}
            rec["floor"] = grad_stats(names, [g.to(BF).float() for g in grads], grads)
        else:
            rec["logit_rel"] = rel(z, ref["z"])
            rec["top1_agree"] = (z.argmax(-1) == ref["z"].argmax(-1)).float().mean().item()
            rec["grad_rel"] = grad_stats(names, grads, ref["grads"])
            rec["decode_same"] = float(np.mean([p == q for p, q in zip(preds, ref["preds"])]))
        arms[arm] = rec
        print(f"[{arm}] {json.dumps(rec)}", flush=True)
        del z, grads
        torch.cuda.empty_cache()

    res = {
        "model": a.model,
        "n_grad": len(g_idx),
        "n_decode": len(va),
        "target_tokens": int(ref["z"].shape[0]),
        "arms": arms,
    }
    (out / "precision.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {out / 'precision.json'}", flush=True)


if __name__ == "__main__":
    main()
