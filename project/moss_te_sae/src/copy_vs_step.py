"""When does the reference copy form? b39 h19 own-cell copy mass and luma L1 against
the reference, per saved checkpoint of our i2i runs (sanitize, swap_text, uncensor), on
the tasks' eval inputs with the shipped generate settings.

Checkpoints are the epoch saves beside each final file (coarse: half / final, and the
uncensor int8 run's e5 / e10 / e15 / e20); ``--extra task=label:path,…`` adds files from
a step-saving micro arm. ``base`` (multiplier 0) renders once per task as the floor.

Reads per render: ``copy`` of b39 h19 at step 0 and in the last third of the steps, the
copy mass summed over the eight largest heads at step 0, the number of (block, head)
pairs with copy mass over 0.1, luma L1 against the reference (swap_text: outside its
text boxes). Sheets: one row per input, columns ref then the checkpoints in order.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import library.moss.generate as gen
from copy_head import EVAL, STATS, Probe, log, luma, text_mask
from library.env import resolve_under_home
from library.moss.requests import GenerateRequest

CK = "output/ckpt_moss_i2i"
SAN = f"{CK}/moss_i2i_sanitize_idt_w0003_late_lr5e5_e8"
SWP = f"{CK}/moss_i2i_swap_text_v3_en400_ko1709_ksplit_rw2_mod16_ema5e5_int8_e2"
UNC = f"{CK}/moss_i2i_uncensor_combined_muon_lr5e5_rw10to1_mod16_int8_e20"
TASKS = {
    "sanitize": dict(
        ckpts=[
            (
                "s1108",
                f"{SAN}/moss_i2i_sanitize_idt_w0003_late_lr5e5_e8_e4.safetensors",
            ),
            ("s2216", f"{SAN}.safetensors"),
        ],
        args=["--config", "configs/moss/i2i/sanitize/generate.toml"],
        inputs=[
            (f"{EVAL}/sanitize/inputs/{s}.png", None)
            for s in ("usual_case1", "hard_to_sanitize1")
        ],
    ),
    "swap_text": dict(
        ckpts=[
            (
                "s2109",
                f"{SWP}/moss_i2i_swap_text_v3_en400_ko1709_ksplit_rw2_mod16_ema5e5_int8_e2_e1.safetensors",
            ),
            ("s4218", f"{SWP}.safetensors"),
        ],
        args=["--config", "configs/moss/i2i/swap_text/generate.toml"],
        inputs=[
            (
                f"{EVAL}/swap_text/inputs/{s}.src.webp",
                f"{EVAL}/swap_text/inputs/{s}.txt",
            )
            for s in ("1483610_0007_en", "589883_0004_ko")
        ],
    ),
    "uncensor": dict(
        ckpts=[
            (
                f"s{66 * e}",
                f"{UNC}/moss_i2i_uncensor_combined_muon_lr5e5_rw10to1_mod16_int8_e20_e{e}.safetensors",
            )
            for e in (5, 10, 15)
        ]
        + [("s1320", f"{UNC}.safetensors")],
        args=[
            "--prompt",
            "no bar censor, no mosaic censor, uncensored",
            "--prompt_format",
            "text",
            "--prompt_tail",
            "off",
            "--ref_scale",
            "1",
        ],  # fmt: skip
        inputs=[
            ("post_image_dataset/i2i/uncensor/selected/Pasted image (5).png", None)
        ],
    ),
}


def read_attn(probe):
    br = probe.records[0]
    cond = br[max(br)]
    acc = (cond["acc"] / cond["count"][:, :, None, None].clamp(min=1)).float()
    ci = STATS.index("copy")
    copy0 = acc[0, :, :, ci]
    return dict(
        h19_copy_s0=round(float(acc[0, 39, 19, ci]), 4),
        h19_copy_late=round(float(acc[3, 39, 19, ci]), 4),
        h19_ref_s0=round(float(acc[0, 39, 19, STATS.index("ref")]), 4),
        top8_copy_s0=round(float(copy0.flatten().topk(8).values.sum()), 4),
        n_copy_heads_s0=int((copy0 > 0.1).sum()),
        top_copy_s0=[
            (int(i) // acc.shape[2], int(i) % acc.shape[2], round(float(v), 3))
            for v, i in zip(*copy0.flatten().topk(5))
        ],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--extra", default="", help="task=label:path,label:path;task=…")
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--max_inputs", type=int, default=0, help="first N inputs per task")
    ap.add_argument(
        "--only", default="", help="replace a task's checkpoints with --extra"
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or resolve_under_home(
        f"output/tests/copy_vs_step/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    for part in filter(None, args.extra.split(";")):
        task, items = part.split("=", 1)
        if task in args.only.split(","):
            TASKS[task]["ckpts"] = []
        TASKS[task]["ckpts"] += [tuple(it.split(":", 1)) for it in items.split(",")]
        if args.max_inputs:
            TASKS[task]["inputs"] = TASKS[task]["inputs"][: args.max_inputs]
    summary = {}
    for task in args.tasks.split(","):
        spec = TASKS[task]
        arms = ([] if args.no_base else [("base", spec["ckpts"][-1][1])]) + spec[
            "ckpts"
        ]
        for label, lora in arms:
            for n, (ref, prompt_file) in enumerate(spec["inputs"]):
                if label == "base" and n > 0:
                    continue
                argv = spec["args"] + [
                    "--lora", lora,
                    "--lora_multiplier", "0" if label == "base" else "1",
                    "--ref_image", ref,
                    "--seed", "0",
                    "--out_dir", str(out / task / label / f"in{n}"),
                ]  # fmt: skip
                if prompt_file:
                    argv += [
                        "--prompt",
                        resolve_under_home(prompt_file).read_text().strip(),
                    ]
                req = GenerateRequest.from_argv(argv)
                t0 = time.time()
                with Probe(out / "_te_cache") as probe:
                    probe.capture = True
                    gen.run_generate(req)
                man = json.loads(
                    (out / task / label / f"in{n}" / "manifest.json").read_text()
                )
                rec = man["images"][0]
                im = Image.open(rec["image"]).convert("RGB")
                L = luma(im)
                ref_l = luma(Image.open(resolve_under_home(ref)).resize(im.size))
                e = dict(image=rec["image"], ref=ref, lora=lora, **read_attn(probe))
                e["luma_vs_ref"] = round(float(np.abs(L - ref_l).mean()), 2)
                if task == "swap_text":
                    keep = ~text_mask(req.prompt, im.height, im.width)
                    e["luma_vs_ref_outside_text"] = round(
                        float(np.abs(L - ref_l)[keep].mean()), 2
                    )
                summary.setdefault(task, {}).setdefault(label, []).append(e)
                (out / "summary.json").write_text(
                    json.dumps(summary, indent=1, ensure_ascii=False)
                )
                log(
                    f"{task} {label} in{n}: {time.time() - t0:.0f}s h19 s0 "
                    f"{e['h19_copy_s0']:.3f} late {e['h19_copy_late']:.3f} top8 "
                    f"{e['top8_copy_s0']:.2f} heads>0.1 {e['n_copy_heads_s0']} "
                    f"L/ref {e['luma_vs_ref']:.1f}"
                    + (
                        f" out-text {e['luma_vs_ref_outside_text']:.1f}"
                        if "luma_vs_ref_outside_text" in e
                        else ""
                    )
                )
    for task, arms in summary.items():
        rows = []
        for n in range(len(TASKS[task]["inputs"])):
            row = [
                Image.open(resolve_under_home(TASKS[task]["inputs"][n][0])).convert(
                    "RGB"
                )
            ]
            for label, entries in arms.items():
                if n < len(entries):
                    row.append(Image.open(entries[n]["image"]).convert("RGB"))
            rows.append(row)
        tw = 384
        th = max(int(r[0].height * tw / r[0].width) for r in rows)
        cols = max(len(r) for r in rows)
        sheet = Image.new("RGB", (tw * cols, th * len(rows)), "white")
        for y, row in enumerate(rows):
            for x, im in enumerate(row):
                im = im.copy()
                im.thumbnail((tw, th))
                sheet.paste(im, (x * tw, y * th))
        sheet.save(out / f"sheet_{task}.png")
        log(f"{task}: sheet columns ref, {', '.join(arms)}")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
