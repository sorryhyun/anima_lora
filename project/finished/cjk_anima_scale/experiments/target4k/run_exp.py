#!/usr/bin/env python
"""target4k — the seed's ``target`` ruler at 4 k tokens (2026-09-30)

``future.md`` § 3's first question, on the ``target`` ruler only: do
``seed_retrain_0930``'s rows read at 768×1344 (48 × 84 = 4 032 tokens: the
shape of the user's own ComfyUI target renders, ``target_prompts.txt``'s
header) as they do at the ruler's 512²? No training. The ``target`` stage exactly as
``eval.ruler_args`` runs it (the user's verbatim captions, 2 seeds, 28 steps,
cfg 4.0, routed — the floor arm of a routed run is ``<seed>/routed/``), with
``--eval_shape 768x1344`` and ``--eval_tag 4k``: the renders land in
``<seed>/routed/target_4k/``, beside the 512² cache ``target/`` they pair
with (same prompt × seed). The 512² cache is read, never re-rendered.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="--label target4k \\
      project/cjk_anima_scale/experiments/target4k/run_exp.py --label s0930"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the seed trained routed, reads routed
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

RUN = (
    "retrain_kanji_b4"  # a routed run on the new seed: its floor arm is <seed>/routed/
)
SHAPE = "768x1344"  # W × H, the user's ComfyUI renders: 48 × 84 = 4 032 tokens
TAG = "4k"


def per_string(recs: list) -> dict:
    from cjk_scale.eval import _metrics

    by = defaultdict(list)
    for m in recs:
        by[m["text"]].append(m)
    return {t: _metrics("target", ms) for t, ms in by.items()} | {
        "total": _metrics("target", recs)
    }


def paired(lo: list, hi: list) -> dict:
    """Official hit, 512² vs ``SHAPE``, on shared (prompt, seed)."""

    def off(m):
        return bool(m.get("hit_sfx")) and bool(m.get("hit_vl"))

    a = {(m["pi"], m["seed"]): off(m) for m in lo}
    b = {(m["pi"], m["seed"]): off(m) for m in hi}
    keys = sorted(a.keys() & b.keys())
    return {
        "n": len(keys),
        "only_4k": sum(b[k] and not a[k] for k in keys),
        "only_1k": sum(a[k] and not b[k] for k in keys),
        "both": sum(a[k] and b[k] for k in keys),
    }


def render_base(label: str) -> None:
    """``--base``: the target captions at ``SHAPE`` on Anima with **no vocab
    pack** (T5 encodes each Japanese span as one ``<unk>``; Qwen3 still reads
    it), same seeds / steps / cfg as the stage → ``<seed>/routed/target_4k_base/``
    beside the seed's and the raw pack's renders. Render + read, no delta."""
    import copy

    import torch

    from anima_lora.inference import GenerationRequest
    from cjk_scale.config import load_run
    from cjk_scale.eval import SEEDS, GEN_STEPS, GEN_CFG, floor_arm_dir, _load_reads
    from common.models import checkpoints, decode_image, load_vae
    from common.prompts import TARGET_PROMPTS
    from common.shapes import parse_shape
    from eval.native import target_items
    from library.inference.generation import generate, get_generation_settings
    from library.inference.models import load_dit_model, load_shared_models

    os.environ.pop("ANIMA_VOCAB_GLYPH_ROUTE", None)
    os.environ.pop("ANIMA_VOCAB_PACK", None)
    arm = floor_arm_dir(load_run(RUN))
    out = arm / f"target_{TAG}_base"
    (out / "img").mkdir(parents=True, exist_ok=True)
    W, H = parse_shape(SHAPE)
    ck = checkpoints()
    args = GenerationRequest(
        prompt="",
        image_size=(H, W),
        infer_steps=GEN_STEPS,
        guidance_scale=GEN_CFG,
        seed=0,
        dit=ck.dit,
        vae=ck.vae,
        text_encoder=ck.text_encoder,
        attn_mode="flash",
        save_path=str(out / "img"),
        no_vocab_pack=True,
    ).to_args()
    args.compile_blocks = True
    gen = get_generation_settings(args)
    shared = load_shared_models(args)
    shared["text_encoder"].to(gen.device)
    shared["conds_cache"] = {}
    shared["model"] = load_dit_model(args, gen.device, torch.bfloat16)
    vae = load_vae(gen.device)
    items = target_items(Path(TARGET_PROMPTS))
    manifest = []
    for it in items:
        for seed in range(SEEDS):
            fn = out / "img" / f"trained_p{it['pi']:02d}_{it['text']}_s{seed}.png"
            if not fn.exists():
                a2 = copy.deepcopy(args)
                a2.prompt, a2.seed = it["caption"], seed
                with torch.no_grad():
                    decode_image(vae, generate(a2, gen, shared), gen.device).save(fn)
            manifest.append({"file": str(fn), "cond": "base", "seed": seed, **it})
    print(f"rendered {len(manifest)} → {out}", flush=True)
    del shared, vae
    torch.cuda.empty_cache()
    from eval.native import Readers, CJK_RE, hit, read_scored

    rd = Readers(gen.device)
    for m in manifest:
        reads = read_scored(rd, m)
        m["reads"] = reads
        m["hit_sfx"], m["hit_vl"] = (
            hit(reads, m["text"], "sfx"),
            hit(reads, m["text"], "vl"),
        )
        m["any_cjk"] = any(CJK_RE.search(r["sfx"] or "") for r in reads)
    (out / "native_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    seed4k = _load_reads(arm / f"target_{TAG}" / "native_reads.json")
    metrics = {
        "base": per_string(manifest),
        "seed_4k": per_string(seed4k),
        "paired_official": paired(seed4k, manifest),
        "out": str(out),
    }
    print(json.dumps(metrics["base"], ensure_ascii=False), flush=True)
    run_dir = make_run_dir(
        "target4k", label=label, root=LINE / "experiments" / "target4k" / "results"
    )
    write_result(
        run_dir,
        script=__file__,
        args=argparse.Namespace(label=label, base=True),
        label=label,
        metrics=metrics,
        artifacts=[str(out)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument(
        "--raw",
        action="store_true",
        help="the raw pack (Δ scale 0) at the same shape → target_4k_raw/, paired "
        "against the seed's target_4k/ (is the 4 k letterbox the rows or the base?)",
    )
    ap.add_argument(
        "--base", action="store_true", help="no vocab pack at all (render_base)"
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.base:
        return render_base(args.label)
    tag = f"{TAG}_raw" if args.raw else TAG

    from cjk_scale.config import load_run
    from cjk_scale.eval import (
        FLOOR_ARM,
        READ_FILES,
        _load_reads,
        floor_arm_dir,
        probe_args,
        routed,
    )

    rc = load_run(RUN)
    assert routed(rc), f"{RUN}: expected a routed run"
    arm = floor_arm_dir(rc)
    # --raw pairs against the seed at the same shape; else the 512² cache
    lo_f = (
        arm / f"target_{TAG}" / "native_reads.json"
        if args.raw
        else arm / READ_FILES["target"]
    )
    hi_f = arm / f"target_{tag}" / "native_reads.json"
    lo = _load_reads(lo_f)
    assert lo, f"no target cache at {lo_f}"
    a = probe_args(
        rc,
        FLOOR_ARM,
        ["target"],
        ["--eval_shape", SHAPE, "--eval_tag", tag]
        + (["--delta_scale", "0"] if args.raw else []),
    )
    print(
        f"arm {arm} (trained.pt → {(arm / 'trained.pt').resolve()})\n"
        f"paired against {lo_f.parent} ({len(lo)} renders); {SHAPE} → {hi_f.parent}",
        flush=True,
    )
    if args.dry_run:
        return
    run_dir = make_run_dir(
        "target4k", label=args.label, root=LINE / "experiments" / "target4k" / "results"
    )
    from stages import run as run_stage

    run_stage("target", a)
    hi = _load_reads(hi_f)
    metrics = {
        "seed_rows": str((arm / "trained.pt").resolve()),
        "size_1k": 512,
        "shape_4k": SHAPE,
        # t1k = the paired side: the 512² cache, or (--raw) the seed at SHAPE;
        # t4k = this run's renders (the seed, or --raw the raw pack, at SHAPE)
        "raw": args.raw,
        "paired_with": str(lo_f.parent),
        "t1k": per_string(lo),
        "t4k": per_string(hi),
        "paired_official": paired(lo, hi),
    }
    for t in metrics["t4k"]:
        p1, h = metrics["t1k"].get(t, {}), metrics["t4k"][t]
        print(
            f"{t:<12} {'seed' if args.raw else '512²'} {p1.get('official')}/{p1.get('n')} "
            f"(loose {p1.get('loose')}, contained {p1.get('contained')})  "
            f"{'raw ' if args.raw else ''}{SHAPE} {h['official']}/{h['n']} "
            f"(loose {h['loose']}, contained {h['contained']})",
            flush=True,
        )
    print("paired official", metrics["paired_official"], flush=True)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(hi_f.parent)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
