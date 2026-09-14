"""Order probe — does the frozen adapter + DiT read a *sequence* of T5 pieces?

No training, no delta. The EN control of `wake_probe.py` already renders
multi-piece words (HELLO = ▁H·ELL·O, SORRY = ▁S·OR·RY) 24/24, but those are
real words the adapter's self-attn could recognise as a whole. This probe
renders *nonsense* multi-piece words (GLORPAX = ▁·GL·OR·PA·X): exact
renders here can only come from reading 4–5 addresses in order.

Groups: `real` (multi-piece real words, sanity), `nonsense` (4–5 pieces,
not words), `two` (two nonsense words, a space between). Same template,
size, steps, cfg and seeds as the wake eval EN group.

    make daemon-run ARGS="--label order-probe --queue \\
        project/cjk_renderable_anima/probes/order_probe.py"

Outputs `output/wake_probe/order_probe/{img/,reads.json,report.md,sheet_*.png}`.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wake.common import OUT, TPL_EN, cer, norm  # noqa: E402
from wake.models import decode_image, gen_args, load_vae  # noqa: E402
from wake.readers import Readers, contact_sheet, load_bgr  # noqa: E402

REAL = ["HELLO", "STOP", "SORRY", "HELP", "PLOVEN", "OSTREB"]
NONSENSE = [
    "GLORPAX",
    "MIZUKANE",
    "SUMIMASEN",
    "TOBRINEK",
    "VASQUOLM",
    "PILDROME",
    "KANTOBRE",
    "ZEMURIAL",
    "DRAVOKIN",
    "NURPELTA",
    "OKTABRIS",
    "FELMUNDO",
]
TWO = ["ZORP KAV", "BLIM TOK", "WAY NO"]


def build_items(t5_tok):
    items = []
    for group, words in (("real", REAL), ("nonsense", NONSENSE), ("two", TWO)):
        for w in words:
            pieces = t5_tok.tokenize(w)
            items.append(
                {
                    "group": group,
                    "text": w,
                    "pieces": pieces,
                    "n_pieces": len(pieces),
                    "caption": TPL_EN.format(w),
                }
            )
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seeds", type=int, default=2)
    a = p.parse_args()

    import torch

    from library.anima.weights import load_t5_tokenizer
    from library.inference.generation import generate, get_generation_settings
    from library.inference.models import load_dit_model, load_shared_models

    out = OUT / "order_probe"
    (out / "img").mkdir(parents=True, exist_ok=True)
    items = build_items(load_t5_tokenizer(None))
    for it in items:
        print(f"{it['group']:8s} {it['text']:10s} {it['n_pieces']} {it['pieces']}")

    args = gen_args(a.size, a.steps, a.cfg, out / "img")
    gen = get_generation_settings(args)
    device = gen.device
    shared = load_shared_models(args)
    shared["conds_cache"] = {}
    anima = load_dit_model(args, device, torch.bfloat16)
    anima.eval()
    shared["model"] = anima
    vae = load_vae(device)
    manifest = []
    t0 = time.time()
    for ei, e in enumerate(items):
        for seed in range(a.seeds):
            fn = out / "img" / f"{e['group']}_{ei:03d}_s{seed}.png"
            if not fn.exists():
                a2 = copy.deepcopy(args)
                a2.prompt = e["caption"]
                a2.seed = seed
                with torch.no_grad():
                    lat = generate(a2, gen, shared)
                decode_image(vae, lat, device).save(fn)
            manifest.append({"file": str(fn), "seed": seed, **e})
    print(
        f"gen: {len(manifest)} images in {(time.time() - t0) / 60:.1f} min", flush=True
    )
    del anima, vae, shared
    torch.cuda.empty_cache()

    from PIL import Image

    rd = Readers(a.device)
    for m in manifest:
        reads = rd.read_image(load_bgr(Path(m["file"])), whole=True)
        m["reads"] = reads
        m["cer_sfx"] = min([cer(r["sfx"] or "", m["text"]) for r in reads] or [1.0])
        m["cer_vl"] = min([cer(r["vl"] or "", m["text"]) for r in reads] or [1.0])
        m["exact_sfx"] = any(norm(r["sfx"] or "") == norm(m["text"]) for r in reads)
        m["exact_vl"] = any(norm(r["vl"] or "") == norm(m["text"]) for r in reads)
        m["exact"] = m["exact_sfx"] or m["exact_vl"]
    (out / "reads.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    agg = defaultdict(list)
    for m in manifest:
        agg[m["group"]].append(m)
    lines = [
        "# order probe — base model, EN multi-piece words, no delta",
        "",
        f"size {a.size} steps {a.steps} cfg {a.cfg} seeds {a.seeds}; template `{TPL_EN}`",
        "",
        "| group | n | exact (sfx) | exact (vl) | exact (either) | CER sfx | CER vl |",
        "|---|---|---|---|---|---|---|",
    ]
    for g, ms in agg.items():
        n = len(ms)
        lines.append(
            f"| {g} | {n} | {sum(m['exact_sfx'] for m in ms)}/{n} | "
            f"{sum(m['exact_vl'] for m in ms)}/{n} | {sum(m['exact'] for m in ms)}/{n} | "
            f"{sum(m['cer_sfx'] for m in ms) / n:.2f} | {sum(m['cer_vl'] for m in ms) / n:.2f} |"
        )
    lines += [
        "",
        "| group | text | pieces | seed | sfx read | vl read | exact |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in manifest:
        best = max(
            m["reads"],
            key=lambda r: (
                (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1])
                if not r["whole"]
                else -1
            ),
            default=None,
        )
        s = (best or {}).get("sfx") or ""
        v = (best or {}).get("vl") or ""
        lines.append(
            f"| {m['group']} | {m['text']} | {'·'.join(m['pieces'])} | {m['seed']} | {s} | {v} | {'Y' if m['exact'] else ''} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))

    for g, ms in agg.items():
        rows = []
        for m in ms:
            best = max(
                m["reads"],
                key=lambda r: (
                    (r["box"][2] - r["box"][0]) * (r["box"][3] - r["box"][1])
                    if not r["whole"]
                    else -1
                ),
                default=None,
            )
            rows.append(
                (
                    Image.open(m["file"]).convert("RGB"),
                    [
                        f"{m['text']} s{m['seed']} {'OK' if m['exact'] else ''}",
                        f"sfx {(best or {}).get('sfx') or ''}",
                        f"vl {(best or {}).get('vl') or ''}",
                    ],
                )
            )
        contact_sheet(rows, out / f"sheet_{g}.png", cols=4)


if __name__ == "__main__":
    main()
