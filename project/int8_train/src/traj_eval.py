"""Does the GPTQ gain hold on the base's own sampling trajectories?

``dit_gptq.py`` calibrates on forward-noised real images (the LoRA training input). At
inference the DiT sees its own trajectory states instead. This records the bf16
base's Euler trajectories (CFG, the inference schedule) for the A/B prompts, plus
forward-noised real images held out of the calibration set, and scores the int8 base
at those states against bf16: round to nearest, then the ``--int8_file`` copies.
The DiT is converted once in mode ``fwd`` (RTN) and its ``wq`` / ``s`` are then
overwritten with the file's, so both arms run the same kernel on the same states.

Per state, ``‖v_arm − v_bf16‖ / ‖v_bf16‖`` for the cond and uncond predictions and the
CFG-combined velocity, averaged per σ band.

    make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/traj_eval.py \\
        --int8_file models/diffusion_models/anima-base-v1.0_int8_gptq.safetensors"
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import defaultdict

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.io.cache import (  # noqa: E402
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from dit_gptq import (  # noqa: E402  (sibling module)
    NEGATIVE,
    _WXH,
    encode_negative,
    pick_images,
    schedule_sigmas,
)
from int8_linear import (  # noqa: E402
    base_fingerprint,
    convert_dit_,
    convert_targets,
    load_int8_file,
)

BANDS = (("high σ>0.7", 0.7, 1.01), ("mid", 0.3, 0.7), ("low σ<0.3", -1, 0.3))


def band(sigma: float) -> str:
    return next(name for name, lo, hi in BANDS if lo < sigma <= hi)


@torch.no_grad()
def pred(anima, x, sigma, c, device):
    pad = torch.zeros(1, 1, *x.shape[-2:], dtype=torch.bfloat16, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        v = anima(
            x.unsqueeze(2).to(device, torch.bfloat16),
            torch.tensor([sigma], device=device),
            c.to(device, torch.bfloat16),
            padding_mask=pad,
        )
    return v.squeeze(2).float()


def held_out_images(root, n, min_tokens, max_tokens, seed):
    """``n`` cached images outside ``dit_gptq.pick_images``'s calibration pick."""
    calib = {
        ci.npz_path
        for pairs in pick_images(root, 2, min_tokens, max_tokens, 0).values()
        for ci in pairs
    }
    pool = []
    for ci in discover_cached_pairs(str(root)):
        m = _WXH.search(ci.npz_path)
        if (
            ci.npz_path not in calib
            and m
            and min_tokens <= (int(m[1]) // 16) * (int(m[2]) // 16) <= max_tokens
        ):
            pool.append(ci)
    return random.Random(seed).sample(pool, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(ap, vae=False, text_encoder=True)
    add_common_args(ap, include_compile=False, include_checkpointing=False)
    ap.add_argument("--int8_file", required=True)
    ap.add_argument("--prompts", default="project/int8_train/src/ab_prompts.txt")
    ap.add_argument("--seeds", default="11,22")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--flow_shift", type=float, default=3.0)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--negative_prompt", default=NEGATIVE)
    ap.add_argument("--cache_root", default="post_image_dataset/lora")
    ap.add_argument("--n_real", type=int, default=28)
    ap.add_argument("--out", default="project/int8_train/assets/traj_eval.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    t0 = time.perf_counter()

    bundle = build_anima(args, adapter=None, train_mode=False)
    anima, device = bundle.anima, bundle.device
    # the config-default CJK pack, as inference.py hooks it; the tokenizer routes
    # through it either way
    from library.anima.vocab_pack import attach_vocab_pack, resolve_active_pack

    pack = resolve_active_pack(args)
    if pack is not None:
        attach_vocab_pack(anima, pack)
    copies, meta = load_int8_file(str(resolve_under_home(args.int8_file)))
    if meta.get("base_fingerprint") != base_fingerprint(anima):
        raise SystemExit(f"{args.int8_file} was made from {meta.get('source')}")
    neg = encode_negative(anima, args, device).cpu()

    from library.inference.text import prepare_text_inputs

    prompts = []
    for line in resolve_under_home(args.prompts).read_text().splitlines():
        p = re.sub(r"\s--d\s+\d+\s*$", "", line).strip()
        if p and p not in prompts:
            prompts.append(p)
    conds = [
        prepare_text_inputs(
            device=device,
            anima=anima,
            prompt=p,
            negative_prompt=args.negative_prompt,
            text_encoder_path=args.text_encoder,
        )[0]["embed"][0].cpu()
        for p in prompts
    ]

    # states: (kind, σ, x, cond); refs: the bf16 (v_c, v_u) at each
    states, refs = [], []
    sigmas = schedule_sigmas(args.steps, args.flow_shift) + [0.0]
    side = args.image_size // 8
    for pi, c in enumerate(conds):
        for seed in (int(s) for s in args.seeds.split(",")):
            gen = torch.Generator(device=device).manual_seed(seed)
            x = torch.randn(1, 16, side, side, generator=gen, device=device)
            for k in range(args.steps):
                s, s_next = sigmas[k], sigmas[k + 1]
                v_c = pred(anima, x, s, c, device)
                v_u = pred(anima, x, s, neg, device)
                states.append(("traj", s, x.cpu(), c))
                refs.append((v_c.cpu(), v_u.cpu()))
                x = x + (s_next - s) * (v_u + args.cfg * (v_c - v_u))
        print(
            f"trajectories: prompt {pi + 1}/{len(conds)} {time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    root = resolve_under_home(args.cache_root)
    real = held_out_images(root, args.n_real, 3000, 4608, args.seed)
    grid = schedule_sigmas(args.steps, args.flow_shift)
    for j, ci in enumerate(real):
        lat = load_cached_latents(ci.npz_path)[0].unsqueeze(0).to(device)
        cap, _ = load_cached_text_features(ci.te_path, variant=0)
        gen = torch.Generator(device=device).manual_seed(args.seed * 1000 + j)
        noise = torch.randn(lat.shape, generator=gen, device=device)
        s = grid[j % len(grid)]
        x = (1 - s) * lat + s * noise
        c = cap.unsqueeze(0)
        states.append(("real", s, x.cpu(), c))
        refs.append(
            (pred(anima, x, s, c, device).cpu(), pred(anima, x, s, neg, device).cpu())
        )
    print(
        f"{len(states)} states, bf16 done {time.perf_counter() - t0:.0f}s", flush=True
    )

    convert_dit_(anima, "fwd")
    targets = dict(convert_targets(anima))
    errs = defaultdict(
        lambda: defaultdict(list)
    )  # (kind, band, arm) -> metric -> values
    for arm in ("rtn", "gptq"):
        if arm == "gptq":
            for name, (wq, s) in copies.items():
                targets[name].wq.copy_(wq)
                targets[name].s.copy_(s)
        for (kind, s, x, c), (r_c, r_u) in zip(states, refs):
            v_c = pred(anima, x, s, c, device).cpu()
            v_u = pred(anima, x, s, neg, device).cpu()
            r = r_u + args.cfg * (r_c - r_u)
            v = v_u + args.cfg * (v_c - v_u)
            e = errs[kind, band(s), arm]
            e["cond"].append(float((v_c - r_c).norm() / r_c.norm()))
            e["uncond"].append(float((v_u - r_u).norm() / r_u.norm()))
            e["cfg"].append(float((v - r).norm() / r.norm()))
        print(f"{arm} done {time.perf_counter() - t0:.0f}s", flush=True)

    rows = {}
    for kind in ("traj", "real"):
        for name, *_ in BANDS:
            rtn, gq = errs[kind, name, "rtn"], errs[kind, name, "gptq"]
            if not rtn["cond"]:
                continue
            row = {"n": len(rtn["cond"])}
            for m in ("cond", "uncond", "cfg"):
                a, b = statistics.mean(rtn[m]), statistics.mean(gq[m])
                row[m] = {"rtn": a, "gptq": b, "gptq_over_rtn": b / a}
            rows[f"{kind} {name}"] = row
            print(
                f"{kind:4s} {name:10s} n={row['n']:3d} "
                + " ".join(
                    f"{m} rtn {row[m]['rtn']:.4f} gptq {row[m]['gptq']:.4f} "
                    f"({row[m]['gptq_over_rtn']:.2f}x)"
                    for m in ("cond", "uncond", "cfg")
                ),
                flush=True,
            )
    out = resolve_under_home(args.out)
    out.write_text(
        json.dumps(
            {
                "args": vars(args),
                "int8_id": meta.get("int8_id"),
                "prompts": prompts,
                "real": [ci.npz_path for ci in real],
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
