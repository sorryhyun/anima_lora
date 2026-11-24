#!/usr/bin/env python3
"""OvisOCR3 wall with and without the ``causal_conv1d`` kernel (GDN runs on
``fla`` either way). One arm per process — transformers binds the kernel when
``modeling_qwen3_5`` is imported, so ``--no-cc1d`` hides the package first.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="project/ocr_reader/ocr/bench_ovis_kernels.py --arm cc1d"
    … --arm torch --no-cc1d

Three loads on the sincos gate: (a) the 908 crops read with ``plain_ocr``
(bs 32), (b) 16 pages with ``word_spotting`` (bs 8, ``max_new_tokens`` 1024),
(c) a teacher-forced train step on crops + their hand label (bs 4,
forward + backward, no optimizer). Writes
``output/ocr/eval/bench_kernels_<arm>.json`` with the walls and the outputs, so
the two arms' reads can be diffed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--no-cc1d", action="store_true")
    ap.add_argument("--train_steps", type=int, default=20)
    ap.add_argument("--train_bs", type=int, default=4)
    a = ap.parse_args()
    if a.no_cc1d:
        sys.modules["causal_conv1d"] = None  # import → ImportError → torch path

    import cv2
    import torch

    import eval_manga109 as ev
    import eval_sfx
    import transformers.models.qwen3_5.modeling_qwen3_5 as q

    conv = q.causal_conv1d_fn.__closure__
    impl = [c.cell_contents for c in conv if callable(c.cell_contents)]
    print("causal_conv1d_fn →", [f.__module__ for f in impl], flush=True)

    df = eval_sfx.load_labels(["sfx", "speech", "chrome"], False, eval_sfx.LABELS)
    crops, _ = eval_sfx.crops_for(df, eval_sfx.FROZEN_PAD)
    reader = ev.READERS["ovis3"](None, "cuda")
    tok = reader.proc.tokenizer
    out = dict(arm=a.arm, cc1d=not a.no_cc1d)

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.time()
        r = fn()
        torch.cuda.synchronize()
        return r, time.time() - t0

    # (a) crops — one warm batch first
    reader.max_tokens = ev.MAX_NEW_TOKENS
    reader.read(crops[:32], [None] * 32, 32)
    preds, wall = timed(lambda: reader.read(crops, [None] * len(crops), 32))
    out["crops"] = dict(
        n=len(crops),
        wall=wall,
        per_s=len(crops) / wall,
        gen_tokens=sum(len(tok(p)["input_ids"]) for p in preds),
        preds=preds,
    )
    print(f"crops: {wall:.1f} s, {len(crops) / wall:.1f} crops/s", flush=True)

    # (b) pages
    stems = sorted(df.stem.unique())[:16]
    pages = [cv2.imread(str(eval_sfx.FROZEN / "pages" / f"{s}.png")) for s in stems]
    prompt = reader.prompt
    reader.prompt = (
        (ev.m109.REPO / reader.DEFAULT / "prompts" / "word_spotting.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
    reader.max_tokens = 1024
    reader.filter_imgtags = False
    raws, wall = timed(lambda: reader.read(pages, [None] * len(pages), 8))
    n_tok = sum(len(tok(r)["input_ids"]) for r in raws)
    out["pages"] = dict(n=len(pages), wall=wall, gen_tokens=n_tok, raws=raws)
    print(f"pages: {wall:.1f} s, {n_tok / wall:.0f} gen tok/s", flush=True)
    reader.prompt = prompt

    # (c) train step
    from PIL import Image

    model = reader.model
    model.train()
    msgs = lambda t: [  # noqa: E731
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt}],
        },
        {"role": "assistant", "content": [{"type": "text", "text": t}]},
    ]
    texts = [
        reader.proc.apply_chat_template(msgs(t), tokenize=False, enable_thinking=False)
        for t in df.text
    ]
    bs = a.train_bs
    walls = []
    for step in range(5 + a.train_steps):
        s = (step * bs) % (len(crops) - bs)
        inputs = reader.proc(
            text=texts[s : s + bs],
            images=[Image.fromarray(c[:, :, ::-1]) for c in crops[s : s + bs]],
            padding=True,
            return_tensors="pt",
            images_kwargs={
                "size": {"shortest_edge": 448 * 448, "longest_edge": 2880 * 2880}
            },
        ).to("cuda")
        labels = inputs["input_ids"].masked_fill(inputs["attention_mask"] == 0, -100)

        def step_fn():
            loss = model(**inputs, labels=labels).loss
            loss.backward()
            model.zero_grad(set_to_none=True)
            return float(loss)

        loss, w = timed(step_fn)
        if step >= 5:
            walls.append(w)
    out["train"] = dict(
        bs=bs,
        steps=len(walls),
        step_s=sum(walls) / len(walls),
        peak_gb=torch.cuda.max_memory_allocated() / 2**30,
        last_loss=loss,
    )
    print(
        f"train: {out['train']['step_s'] * 1000:.0f} ms/step (bs {bs}), "
        f"peak {out['train']['peak_gb']:.1f} GB",
        flush=True,
    )
    ev.OUT.mkdir(parents=True, exist_ok=True)
    (ev.OUT / f"bench_kernels_{a.arm}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
