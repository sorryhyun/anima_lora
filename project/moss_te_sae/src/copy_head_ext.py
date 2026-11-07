"""The reference copy head on an outside edit LoRA: chinoll/Moss-Image-Edit.

That LoRA (rank 64, all 42 blocks, ConceptEdit-12M instruction edits, own trainer) uses
the same reference contract as ours (reference at t = 0, reference queries isolated)
but places the reference on RoPE frame 0, where ours puts it on frame = text length.
Its weights put 0.56 / 0.81 of block 39's v / o energy on head 19, as every one of our
i2i LoRAs does. This run asks whether it also builds the own-cell copy head there, and
whether the copy gate follows the edit region.

Converted file: ``output/ckpt_moss_i2i/ext/moss_image_edit_chinoll.safetensors``
(lora_A → down, lora_B → up, rank 64 / alpha 64, ref_timestep 0, ref_isolate 1).

Per input (two colour illustrations with a local edit, one manga page with a removal),
arms in order:

- ``base``: the LoRA at multiplier 0;
- ``lora``: the LoRA, reference on frame 0 (its training contract);
- ``lora_ourrope``: the LoRA with our reference frame (text length), first input only;
- ``cut_h19``: b39 h19's image queries cannot attend to reference keys;
- ``cut_top8``: the same cut on this LoRA's own top eight copy heads (from ``lora``);
- ``cut_ctrl``: the cut on b39 h5.

Sampler as the LoRA's own ``infer.py``: Euler 50, CFG 4, plain CFG (no uncond skip),
negative ``""``, plain instruction through the chat template with no tail.

Reads: copy mass per block / head by step bucket (``copy_head.Probe``), the b39 h19
copy map at step 0 and over all steps, luma L1 against the reference and against the
``lora`` render, and the copy map's agreement with the edit region (pixels where the
``lora`` render departs from the reference).
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import library.moss.generate as gen
import library.moss.model as mm
import copy_head
from copy_head import STATS, Probe, log, luma
from library.env import resolve_under_home
from library.moss.requests import GenerateRequest

LORA = "output/ckpt_moss_i2i/ext/moss_image_edit_chinoll.safetensors"
INPUTS = [
    ("image_dataset/sushispin/10043045.webp", "Change her hair color to bright pink."),
    ("image_dataset/sushispin/9859367.webp", "Change her black skirt to a red skirt."),
    (
        "output/moss_i2i_eval/sanitize/inputs/usual_case1.png",
        "Remove all the text and sound effects from the image.",
    ),
]
SAMPLER = [
    "--prompt_format", "text",
    "--prompt_tail", "off",
    "--sampler", "euler",
    "--steps", "50",
    "--cfg", "4",
    "--uncond_skip", "",
    "--ref_scale", "1",
]  # fmt: skip


class FrameZeroRoPE:
    """Reference rows on RoPE frame 0 instead of the text length."""

    def __init__(self, on=True):
        self.on = on
        self.orig = mm._RoPE.forward

    def __enter__(self):
        orig = self.orig
        if not self.on:
            return self

        def forward(rope, height, width, text_length, device, ref_hw=None):
            out = orig(rope, height, width, text_length, device, ref_hw)
            if ref_hw is None:
                return out
            rh, rw = ref_hw
            h, w = torch.meshgrid(
                (torch.arange(rh, device=device) + 0.5) * height / rh - 0.5,
                (torch.arange(rw, device=device) + 0.5) * width / rw - 0.5,
                indexing="ij",
            )
            ref = torch.stack(
                (torch.ones_like(h), torch.zeros_like(h), h, w), -1
            ).flatten(0, 1)
            ref[:, 2] -= (height - 1) / 2
            ref[:, 3] -= (width - 1) / 2
            phase = ref.unsqueeze(-1) * rope.bands
            ref_rope = torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)
            out = out.clone()
            out[0, -ref.shape[0] :] = ref_rope.to(out.dtype)
            return out

        mm._RoPE.forward = forward
        return self

    def __exit__(self, *exc):
        mm._RoPE.forward = self.orig


def find_image(stem: str) -> str:
    p = resolve_under_home(stem)
    if p.exists():
        return str(p)
    for ext in (".png", ".jpg", ".webp"):
        q = p.with_suffix(ext)
        if q.exists():
            return str(q)
    raise FileNotFoundError(stem)


def render(ref, prompt, arm_dir, mult, cut, capture, frame0, te_cache, lora=LORA):
    argv = [
        "--lora", lora,
        "--lora_multiplier", str(mult),
        "--ref_image", ref,
        "--prompt", prompt,
        "--seed", "0",
        "--out_dir", str(arm_dir),
        *SAMPLER,
    ]  # fmt: skip
    req = GenerateRequest.from_argv(argv)
    with FrameZeroRoPE(frame0), Probe(te_cache) as probe:
        # Python-state branches (block index, cut) inside the compiled block would
        # recompile per block; run the probed attention outside dynamo.
        mm._attention = torch.compiler.disable(probe._attention)
        probe.capture = capture
        probe.cut = cut
        gen.run_generate(req)
    man = json.loads((arm_dir / "manifest.json").read_text())
    rec = man["images"][0]
    entry = dict(image=rec["image"], width=rec["width"], height=rec["height"])
    if capture and probe.records:
        br = probe.records[0]
        cond = br[max(br)]
        acc = cond["acc"] / cond["count"][:, :, None, None].clamp(min=1)
        h19 = (cond["h19"] / max(cond["h19_count"], 1)).float()
        torch.save(
            dict(acc=acc.float(), stats=STATS, h19=h19,
                 h19_step0=cond["h19_step0"], n_img=cond["n_img"]),
            arm_dir / "attn.pt",
        )  # fmt: skip
        a = acc.float()
        entry["b39h19"] = {s: a[:, 39, 19, i].tolist() for i, s in enumerate(STATS)}
        allstep = a[1:, :, :, 3].mean(0)
        top = allstep.flatten().topk(8)
        entry["top_copy"] = [
            (int(i) // 24, int(i) % 24, round(float(v), 3))
            for v, i in zip(top.values, top.indices)
        ]
        top0 = a[0, :, :, 3].flatten().topk(8)
        entry["top_copy_step0"] = [
            (int(i) // 24, int(i) % 24, round(float(v), 3))
            for v, i in zip(top0.values, top0.indices)
        ]
    return entry


def gate_vs_edit(arm_dir, ref, lora_img, W, H):
    """Copy map (latent grid) against the edit region of the lora render."""
    d = torch.load(arm_dir / "attn.pt")
    gh, gw = H // 16, W // 16
    out = {}
    L = luma(Image.open(lora_img))
    R = luma(Image.open(ref).resize((W, H)))
    rgb_l = np.asarray(Image.open(lora_img).convert("RGB"), np.float32)
    rgb_r = np.asarray(Image.open(ref).convert("RGB").resize((W, H)), np.float32)
    diff = np.abs(rgb_l - rgb_r).mean(-1)
    cell = diff.reshape(gh, 16, gw, 16).mean((1, 3))
    edit = cell > max(np.percentile(cell, 90), 12.0)
    out["edit_cells"] = int(edit.sum())
    out["luma_l1"] = float(np.abs(L - R).mean())
    for key in ("h19_step0", "h19"):
        m = d[key]
        if m is None:
            continue
        m = m.numpy().reshape(gh, gw)
        out[key] = dict(
            inside=float(m[edit].mean()) if edit.any() else None,
            outside=float(m[~edit].mean()),
            corr=float(np.corrcoef(m.ravel(), cell.ravel())[0, 1]),
        )
        img = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8)).resize(
            (W, H), Image.NEAREST
        )
        img.save(arm_dir / f"{key}_map.png")
    Image.fromarray(edit.astype(np.uint8) * 255).resize((W, H), Image.NEAREST).save(
        arm_dir / "edit_region.png"
    )
    return out


def main():
    copy_head.HEAD_CHUNK = 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--inputs", default="0,1,2")
    ap.add_argument(
        "--size", default="", help="WxH area cap (the LoRA trained at max edge 1024)"
    )
    args = ap.parse_args()
    if args.size:
        w, h = args.size.split("x")
        SAMPLER.extend(["--width", w, "--height", h])
    out = args.out or resolve_under_home(
        f"output/tests/copy_head_ext/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    te_cache = out / "_te_cache"
    summary = {}
    for n in map(int, args.inputs.split(",")):
        ref, prompt = INPUTS[n]
        ref = find_image(ref)
        rows = {}
        arms = ["base", "lora"] + (["lora_ourrope"] if n == 0 else [])
        for arm in arms:
            t0 = time.time()
            rows[arm] = render(
                ref, prompt, out / f"in{n}" / arm,
                mult=0 if arm == "base" else 1, cut=None, capture=True,
                frame0=arm != "lora_ourrope", te_cache=te_cache,
            )  # fmt: skip
            log(f"in{n} {arm}: {time.time() - t0:.0f}s top {rows[arm].get('top_copy')}")
        own8 = {}
        for b, h, _ in rows["lora"]["top_copy"]:
            own8.setdefault(b, []).append(h)
        for arm, cut in (
            ("cut_h19", {39: [19]}),
            ("cut_top8", own8),
            ("cut_ctrl", {39: [5]}),
        ):
            t0 = time.time()
            rows[arm] = render(
                ref, prompt, out / f"in{n}" / arm, mult=1, cut=cut,
                capture=False, frame0=True, te_cache=te_cache,
            )  # fmt: skip
            rows[arm]["cut"] = {str(k): v for k, v in cut.items()}
            log(f"in{n} {arm}: {time.time() - t0:.0f}s")
        lora_img = rows["lora"]["image"]
        W, H = rows["lora"]["width"], rows["lora"]["height"]
        L_lora = luma(Image.open(lora_img))
        R = luma(Image.open(ref).resize((W, H)))
        for arm, e in rows.items():
            Li = luma(Image.open(e["image"]))
            e["luma_vs_ref"] = float(np.abs(Li - R).mean())
            e["luma_vs_lora"] = float(np.abs(Li - L_lora).mean())
        for arm in ("base", "lora") + (("lora_ourrope",) if n == 0 else ()):
            rows[arm]["gate"] = gate_vs_edit(out / f"in{n}" / arm, ref, lora_img, W, H)
        summary[f"in{n}"] = dict(ref=ref, prompt=prompt, arms=rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        # sheet: ref, arms, h19 map (lora), edit region
        tiles = [Image.open(ref).convert("RGB").resize((W, H))]
        tiles += [Image.open(rows[a]["image"]).convert("RGB") for a in rows]
        tiles += [
            Image.open(out / f"in{n}" / "lora" / "h19_map.png").convert("RGB"),
            Image.open(out / f"in{n}" / "lora" / "edit_region.png").convert("RGB"),
        ]
        tw = 320
        th = int(H * tw / W)
        sheet = Image.new("RGB", (tw * len(tiles), th), "white")
        for i, t in enumerate(tiles):
            sheet.paste(t.resize((tw, th)), (i * tw, 0))
        sheet.save(out / f"sheet_in{n}.png")
        log(f"in{n} sheet: ref, {', '.join(rows)}, h19 map, edit region")
        for arm, e in rows.items():
            g = e.get("gate", {})
            h = e.get("b39h19", {}).get("copy")
            log(
                f"  {arm:12} L/ref {e['luma_vs_ref']:.1f} L/lora {e['luma_vs_lora']:.1f}"
                + (f" h19 copy s0 {h[0]:.3f} late {h[3]:.3f}" if h else "")
                + (
                    f" gate in/out {g['h19']['inside']:.2f}/{g['h19']['outside']:.2f}"
                    f" corr {g['h19']['corr']:.2f}"
                    if g.get("h19") and g["h19"]["inside"] is not None
                    else ""
                )
            )
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
