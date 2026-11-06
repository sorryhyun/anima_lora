"""adapter_render — Japanese renders on the shipped pack, the fp32 stream with the bf16
``llm_adapter`` (generation's default) against ``adapter_fp32_`` (bf16 weights, fp32
compute), same captions and seeds.

Captions: the identity grid's 13 ``read`` words plus ``SENTENCES`` (kanji), each on the
scale line's plain-read prompts p00 / p01 as ``<prompt>. Text reads as "<key>".``; seeds
0 / 1, 512², 28 steps, cfg 4, compiled blocks. The pack is ``configs/base.toml``'s
(``jp_v1``) with its own routing. No ExtDelta: the pack's rows as shipped.

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/adapter_render.py"

→ ``results/<ts>-adapter_render/``: ``<arm>/<stem>.png``, ``sheet_<n>.png`` (per caption:
seed 0 and 1, stream | adapter_fp32), ``result.json`` (mean |Δpx| per pair).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["ANIMA_VOCAB_PACK"] = "models/vocab_packs/anima_cjk_vocab_pack_jp_v1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import HOME, SCALE_OUT, bootstrap  # noqa: E402

bootstrap()

import torch  # noqa: E402

SENTENCES = ["今日はいい天気", "ありがとう", "大丈夫ですか", "好きです", "待って"]
SEEDS = (0, 1)
PROMPTS = (0, 1)
SRC = SCALE_OUT / "retrain_kana" / "native_r4_plain" / "native_reads.json"
ARMS = ("stream", "adapter_fp32")


def main():
    import numpy as np
    from PIL import Image, ImageDraw

    from common.models import decode_image, load_generator, load_vae
    from library.anima.stream import adapter_fp32_
    from library.anima.vocab_pack import attached_pack_rows
    from library.inference.generation import generate
    from reseed.config import load

    out = HOME / "results" / (time.strftime("%Y%m%d-%H%M") + "-adapter_render")
    prompts = {
        m["pi"]: m["prompt"] for m in json.loads(SRC.read_text(encoding="utf-8"))
    }
    keys = list(load("hira_cold_hp").read) + SENTENCES
    items = [
        {
            "key": k,
            "pi": pi,
            "caption": f'{prompts[pi]}. Text reads as "{k}".',
            "stem": f"{n:02d}_p{pi:02d}",
        }
        for n, k in enumerate(keys)
        for pi in PROMPTS
    ]

    args, gen, device, shared = load_generator(512, 28, 4.0, out / "_tmp")
    anima = shared["model"]
    anima.eval()
    assert getattr(anima, "stream", "bf16") == "fp32", (
        "generation loaded the bf16 stream"
    )
    print(f"pack rows attached: {attached_pack_rows(anima)}", flush=True)
    vae = load_vae(device)
    seen = {}
    anima.llm_adapter.register_forward_hook(
        lambda m, i, o: seen.__setitem__("dtype", o.dtype)
    )

    import copy

    t0 = time.time()
    for arm in ARMS:
        if arm == "adapter_fp32":
            adapter_fp32_(anima)
        shared["conds_cache"].clear()
        seen.clear()
        for n, m in enumerate(items):
            for seed in SEEDS:
                fn = out / arm / f"{m['stem']}_s{seed}.png"
                fn.parent.mkdir(parents=True, exist_ok=True)
                a2 = copy.deepcopy(args)
                a2.prompt, a2.seed = m["caption"], seed
                with torch.no_grad():
                    lat = generate(a2, gen, shared)
                decode_image(vae, lat, device).save(fn)
            print(
                f"{arm}: {n + 1}/{len(items)} adapter out {seen.get('dtype')} "
                f"({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )

    diffs = {}
    font = None
    try:
        from PIL import ImageFont

        font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20
        )
    except OSError:
        pass
    per_sheet = 6
    for s0 in range(0, len(items), per_sheet):
        chunk = items[s0 : s0 + per_sheet]
        W, H, lab = 512, 512, 28
        sheet = Image.new("RGB", (W * 4, (H + lab) * len(chunk)), "white")
        d = ImageDraw.Draw(sheet)
        for r, m in enumerate(chunk):
            for ci, (seed, arm) in enumerate([(s, a) for s in SEEDS for a in ARMS]):
                im = Image.open(out / arm / f"{m['stem']}_s{seed}.png").convert("RGB")
                sheet.paste(im.resize((W, H)), (ci * W, r * (H + lab) + lab))
                d.text(
                    (ci * W + 6, r * (H + lab) + 2),
                    f"{m['key']}  p{m['pi']:02d}  s{seed}  {arm}",
                    fill="black",
                    font=font,
                )
            for seed in SEEDS:
                a = np.asarray(
                    Image.open(out / "stream" / f"{m['stem']}_s{seed}.png"), np.float32
                )
                b = np.asarray(
                    Image.open(out / "adapter_fp32" / f"{m['stem']}_s{seed}.png"),
                    np.float32,
                )
                diffs[f"{m['stem']}_s{seed}"] = {
                    "key": m["key"],
                    "mean_abs_dpx": float(np.abs(a - b).mean()),
                }
        sheet.save(out / f"sheet_{s0 // per_sheet}.png")
    vals = [v["mean_abs_dpx"] for v in diffs.values()]
    (out / "result.json").write_text(
        json.dumps(
            {
                "pack": os.environ["ANIMA_VOCAB_PACK"],
                "items": diffs,
                "mean_abs_dpx": float(np.mean(vals)),
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    print(
        f"mean |Δpx| stream vs adapter_fp32: {np.mean(vals):.2f} / 255 (median {np.median(vals):.2f}, max {np.max(vals):.2f})"
    )
    print(f"→ {out} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
