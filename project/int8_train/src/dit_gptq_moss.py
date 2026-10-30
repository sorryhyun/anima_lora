"""Offline GPTQ for MOSS-Image's 252 int8 ConvRot linears, for both t2i and i2i
(the MOSS counterpart of ``dit_gptq.py``), written as a full DiT file in ComfyUI's
``int8_tensorwise`` + ``convrot`` layout.

Calibration: up to ``--per_set`` cached items from each of ``--sets`` (token count in
range), each noised at one σ drawn from the inference schedule and run twice, as CFG
does: with its own caption, and with the empty prompt skipping block 22. A cache with
``*.moss_ref`` files adds the reference tokens, modulated at ``ref_timestep`` 0 and
isolated, as every i2i recipe trains and generates; in the skip-22 branch those tokens
still run block 22. The sets keep ``channel_(caststation)_layout`` out, the images
``grad_error_moss.py`` scores on.

Blocks are quantized in order, each quantized block (W8A8, the kernel) feeding the
next. GPTQ runs against the summed t2i + i2i input statistics. ``rtn`` and ``gptq`` are
scored per domain as well, so a cost to t2i from the i2i tokens would show. The file
and metadata are ``dit_gptq.py``'s; ``train_int8_moss.py`` and ``infer_int8_moss.py``
take it as ``--int8_file``. ``--max_blocks`` stops early and writes nothing.

    make daemon-run ARGS="--stall-timeout 1800 project/int8_train/src/dit_gptq_moss.py"
"""

from __future__ import annotations

import argparse
import json
import random
import time

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import torch  # noqa: E402

from library.env import resolve_under_home  # noqa: E402
from library.moss import cache as moss_cache  # noqa: E402
from library.moss import int8  # noqa: E402
from library.moss.loader import load_transformer, variant_dir  # noqa: E402
from library.moss.model import snapped_sigmas  # noqa: E402
from library.moss.requests import (  # noqa: E402
    DEFAULT_SHIFT,
    DEFAULT_STEPS,
    DEFAULT_UNCOND_SKIP,
)
from library.moss.train import EMPTY_PROMPT_PATH, NUM_TRAIN_TIMESTEPS  # noqa: E402

# sibling modules
from dit_gptq import Acc, gptq, out_err, sha256_file, w8a8_err, write_file  # noqa: E402
from int8_linear import base_fingerprint  # noqa: E402
from train_int8_moss import convert_moss_, moss_keys, moss_targets  # noqa: E402

SETS = (
    "ama_mitsuki_layout,group_wsw_layout,hews_layout,sincos_layout,_calib_mixed,"
    "i2i_sanitize_idt,i2i_swap_text,i2i_uncensor_combined"
)
SUFFIXES = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


def pick_items(root, sets, per_set, min_tokens, max_tokens, rng) -> dict:
    picked = {}
    for name in sets:
        items = [
            it
            for it in moss_cache.list_items(root / name)
            if min_tokens <= (it.width // 16) * (it.height // 16) <= max_tokens
        ]
        picked[name] = rng.sample(items, min(per_set, len(items)))
    return picked


def samples_of(tr, item, empty, sigma, gen, device) -> list[dict]:
    """The cond and the skip-22 uncond entering block 0, on the CPU."""
    latent, embeds = moss_cache.load_item(item)
    latent = latent[None].to(device)
    ref = moss_cache.load_ref(item)
    ref = None if ref is None else ref[None].to(device)
    noise = torch.randn(latent.shape, generator=gen, device=device)
    noisy = (1.0 - sigma) * latent + sigma * noise
    mod, _ = tr.time_conditioning(NUM_TRAIN_TIMESTEPS * sigma, 1)
    ref_hw = None if ref is None else tuple(ref.shape[-2:])
    mod, n_ref = tr.ref_modulation(mod, ref_hw)
    out = []
    for context, skip in ((embeds, False), (empty, True)):
        x, rope = tr.embed_sample(noisy, context[None].to(device), ref)
        out.append(
            {
                "x": x.cpu(),
                "rope": rope.cpu(),
                "mod": mod.cpu(),
                "ref_start": x.shape[1] - n_ref if n_ref else None,
                "skip": skip,
                "domain": "i2i" if n_ref else "t2i",
            }
        )
    return out


def run_block(tr, layer, index, s, device):
    """``s``'s block output, or None when the uncond skips it with no reference."""
    x, rope, mod = (s[k].to(device) for k in ("x", "rope", "mod"))
    start = s["ref_start"]
    if s["skip"] and index in DEFAULT_UNCOND_SKIP:
        if start is None:
            return None
        return tr._ref_only(layer, x, rope, mod, start)
    return layer(x, rope, mod, None, start, start is not None)


def scores(Q, Wr, Hr, Hq, G) -> dict:
    return {
        "w8a8": w8a8_err(Q, Wr, Hq, G, Hr),
        "w8a16": out_err(Q - Wr, Hr, Wr),
    }


def linear_row(acc_by_domain, Wr, rtn, Q) -> dict:
    """Errors of both roundings over all samples and per domain."""
    groups = {
        "": [sum(getattr(a, k) for a in acc_by_domain.values()) for k in ATTRS],
        **{f"{d}_": [getattr(a, k) for k in ATTRS] for d, a in acc_by_domain.items()},
    }
    row = {}
    for prefix, (Hr, Hq, G) in groups.items():
        for arm, q in (("rtn", rtn), ("gptq", Q)):
            for k, v in scores(q, Wr, Hr, Hq, G).items():
                row[f"{prefix}{k}_{arm}"] = v
    return row


ATTRS = ("Hr", "Hq", "G")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--model", default="base")
    ap.add_argument("--cache_root", default="post_image_dataset/moss")
    ap.add_argument("--sets", default=SETS)
    ap.add_argument("--per_set", type=int, default=30)
    ap.add_argument("--min_tokens", type=int, default=3000)
    ap.add_argument("--max_tokens", type=int, default=4800)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--shift", type=float, default=DEFAULT_SHIFT)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--no_act_order", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_blocks", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    t0 = time.perf_counter()
    device = torch.device("cuda")
    out = resolve_under_home(
        args.out or f"models/moss_image/moss_image_{args.model}_int8_gptq.safetensors"
    )

    tr = load_transformer(args.model_dir, args.model, device=device)
    tr.ref_timestep = 0.0
    targets = dict(moss_targets(tr))
    fingerprint = base_fingerprint(tr, list(targets.items()))
    empty = moss_cache.read_te(resolve_under_home(EMPTY_PROMPT_PATH))

    rng = random.Random(args.seed)
    picked = pick_items(
        resolve_under_home(args.cache_root),
        args.sets.split(","),
        args.per_set,
        args.min_tokens,
        args.max_tokens,
        rng,
    )
    sigmas = snapped_sigmas(args.steps, args.shift).tolist()
    samples, n = [], 0
    for items in picked.values():
        for it in items:
            gen = torch.Generator(device=device).manual_seed(args.seed * 100003 + n)
            samples += samples_of(tr, it, empty, rng.choice(sigmas), gen, device)
            n += 1
    tokens = {d: 0 for d in ("t2i", "i2i")}
    for s in samples:
        tokens[s["domain"]] += s["x"].shape[1]
    domains = [d for d, v in tokens.items() if v]
    calib = {
        "sets": {k: len(v) for k, v in picked.items()},
        "images": n,
        "samples": len(samples),
        "tokens": tokens,
        "per_set": args.per_set,
        "token_range": [args.min_tokens, args.max_tokens],
        "sigma_schedule": f"{args.steps} steps, shift {args.shift}",
        "seed": args.seed,
    }
    print(
        f"[calib] {json.dumps(calib)} {sum(s['x'].numel() for s in samples) * 2 / 2**30:.1f} "
        f"GiB of x, {time.perf_counter() - t0:.0f}s",
        flush=True,
    )

    copies, report = {}, {}
    current = {"domain": domains[0]}
    for i, layer in enumerate(tr.layers):
        lins = {suf: targets[f"layers.{i}.proj.{suf}"] for suf in SUFFIXES}
        accs = {
            suf: {d: Acc(m.in_features, device) for d in domains}
            for suf, m in lins.items()
        }

        def hook_of(by_domain):
            return lambda mod, inputs: by_domain[current["domain"]].hook(mod, inputs)

        hooks = [m.register_forward_pre_hook(hook_of(accs[s])) for s, m in lins.items()]
        for s in samples:
            current["domain"] = s["domain"]
            run_block(tr, layer, i, s, device)
        for h in hooks:
            h.remove()

        for suf, m in lins.items():
            W = m.weight.detach().float()
            Wr = int8.rotate(W)
            scale = int8.row_scale(Wr)
            Hr = sum(a.Hr for a in accs[suf].values())

            def grid(c, s=scale):
                return int8.qdq(c, s)

            Q = gptq(Wr, Hr, grid, args.block, args.damp, not args.no_act_order)
            name = f"layers.{i}.proj.{suf}"
            report[name] = linear_row(accs[suf], Wr, int8.qdq(Wr, scale), Q)
            wq = (Q / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
            copies[name] = (wq.cpu(), scale[:, None].cpu())
            convert_moss_(m, "infer", (wq, scale[:, None]))
        del accs
        for s in samples:
            y = run_block(tr, layer, i, s, device)
            if y is not None:
                s["x"] = y.cpu()
        torch.cuda.empty_cache()
        rows = [report[f"layers.{i}.proj.{s}"] for s in SUFFIXES]
        mean = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        print(
            f"[gptq] block {i:2d} w8a8 rtn {mean['w8a8_rtn']:.2e} gptq "
            f"{mean['w8a8_gptq']:.2e} | w8a16 rtn {mean['w8a16_rtn']:.2e} gptq "
            f"{mean['w8a16_gptq']:.2e} | peak "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB | "
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )
        if args.max_blocks and i + 1 >= args.max_blocks:
            print("[gptq] --max_blocks reached, nothing written", flush=True)
            return

    keys = next(iter(report.values())).keys()
    total = {k: sum(r[k] for r in report.values()) / len(report) for k in keys}
    print("[gptq] mean over linears " + json.dumps(total), flush=True)
    del tr, samples
    torch.cuda.empty_cache()
    src_dir = variant_dir(args.model_dir, args.model) / "transformer"
    srcs = sorted(src_dir.glob("*.safetensors"))
    meta = {
        "source": f"MOSS-Image {args.model}",
        "source_sha256": json.dumps({p.name: sha256_file(p) for p in srcs}),
        "base_fingerprint": fingerprint,
        "method": "gptq",
        "calibration": json.dumps(calib),
        "gptq": json.dumps(
            {
                "block": args.block,
                "damp": args.damp,
                "act_order": not args.no_act_order,
            }
        ),
    }
    int8_id = write_file(srcs, out, copies, meta, moss_keys)
    out.with_suffix(".report.json").write_text(
        json.dumps(
            {
                "meta": meta,
                "int8_id": int8_id,
                "stems": {k: [it.stem for it in v] for k, v in picked.items()},
                "mean": total,
                "linears": report,
            },
            indent=1,
        )
    )
    print(
        f"[gptq] wrote {out} ({out.stat().st_size / 2**30:.2f} GiB) int8_id "
        f"{int8_id[:16]} in {time.perf_counter() - t0:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
