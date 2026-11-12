"""Per-step decode time of ``rewrite_set``'s vision PE generate: eager vs ``fast_decode``.

Batch of ``--bs`` caststation images, ``--n`` forced tokens, a logits processor stamps each
step after a sync; reports prefill, median / p90 decode step, dynamo counters and the cache
class ``generate`` built. Two passes per arm, the second is reported (the first absorbs the
compile).

    make daemon-run ARGS="--stall-timeout 1200 project/pe_lora/src/bench_fast_vl.py --arm fast"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import LogitsProcessor, LogitsProcessorList

import fast_decode
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import vision_te
from library.moss.loader import load_tokenizer
from pe_mode import PEMode
from rewrite_set import brief_of, vision_causal


class Stamp(LogitsProcessor):
    def __init__(self):
        self.t = []

    def __call__(self, input_ids, scores):
        torch.cuda.synchronize()
        self.t.append(time.perf_counter())
        return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("eager", "fast"), required=True)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--src", default="post_image_dataset/resized/channel_(caststation)")
    args = ap.parse_args()

    gen_kw = fast_decode.enable() if args.arm == "fast" else {}
    tok = load_tokenizer(default_moss_dir())
    te = vision_te.load_te(tok, torch.device("cuda"), precision="int8")
    pm = PEMode(
        te,
        load_file(
            "/media/sorryhyun/new/qwen35_pe/lora/pe_mode_int8_r128d256.safetensors"
        ),
    )
    pm.set_mode("pe")
    vl = vision_causal(te, pm.causal.lm_head)
    proc = vision_te.processor(tok, 896 * 896)
    system = resolve_under_home(
        Path("project/pe_lora/system_caption_vl.txt")
    ).read_text()

    caps = sorted(
        p for p in resolve_under_home(args.src).glob("*.txt") if p.name.count(".") == 1
    )
    caps = caps[: args.bs]
    texts = [
        tok.apply_chat_template(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": brief_of(p.read_text().strip(), None)},
                    ],
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for p in caps
    ]
    imgs = [Image.open(p.with_suffix(".png")).convert("RGB") for p in caps]
    tok.padding_side = "left"
    enc = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to("cuda")

    import torch._dynamo.utils as du

    for rep in range(2):
        stamp = Stamp()
        t0 = time.perf_counter()
        out = vl.generate(
            **enc,
            max_new_tokens=args.n,
            min_new_tokens=args.n,
            do_sample=True,
            top_k=20,
            top_p=0.95,
            logits_processor=LogitsProcessorList([stamp]),
            pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True,
            **gen_kw,
        )
        torch.cuda.synchronize()
        steps = sorted(b - a for a, b in zip(stamp.t, stamp.t[1:]))
        print(
            f"[{args.arm} rep{rep}] bs={args.bs} prefill={stamp.t[0] - t0:.1f}s "
            f"decode median={1000 * steps[len(steps) // 2]:.1f}ms "
            f"p90={1000 * steps[int(len(steps) * 0.9)]:.1f}ms "
            f"total={time.perf_counter() - t0:.1f}s cache={type(out.past_key_values).__name__}",
            flush=True,
        )
    print(
        "dynamo:",
        dict(du.counters["stats"]),
        dict(du.counters["graph_break"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
