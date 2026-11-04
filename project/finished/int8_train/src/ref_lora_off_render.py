#!/usr/bin/env python3
"""Render the swap_text eval pages with the LoRA delta off on the reference stream.

Probe for ``_archive/proposals/reference_adapters.md``: does the LoRA's change to the
reference rows (the "text mark") show in a render? The LoRA trained with adapters on
the reference rows; here ``RefCache`` builds the reference keys and values with every
adapter at multiplier 0, so the image and text rows (LoRA on) read Base reference
features. Control: the existing v3 renders (same seed, prompt, size and sampler).

``--mode blocks``: block adapters off on the reference, the reference's t = 0
modulation keeps the mod adapter.
``--mode all``: the modulation at t = 0 from the Base linear too (the frozen setting).
``--mode kv``: only ``proj_k`` / ``proj_v`` keep their adapters on the reference (the
K / V the image rows read); ``--mode nokv``: the opposite, k / v off, the rest on.

    make daemon-run ARGS="--stall-timeout 600 project/finished/int8_train/src/ref_lora_off_render.py --mode all"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

LORA = "output/ckpt_moss_i2i/moss_i2i_swap_text_v3_en400_ko1709_rw2_mod16_ema5e5_int8_e2.safetensors"
INPUTS = Path("output/moss_i2i_eval/swap_text/inputs")
CONTROL = Path(
    "output/moss_i2i_eval/swap_text/results/v3_en400_ko1709_e2_ema_bf16_fp32s"
)
PAGES = ("1483610_0007_en", "3124789_0042_en", "589883_0004_ko", "737486_0019_ko")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--mode", choices=("blocks", "all", "kv", "nokv"), required=True)
    ap.add_argument("--pages", nargs="*", default=list(PAGES))
    ap.add_argument(
        "--out", default=None, help="results dir; default v3_..._reflora_<mode>"
    )
    args = ap.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    import library.moss.lora as lora_mod
    from library.moss.generate import run_generate
    from library.moss.model import RefCache
    from library.moss.requests import GenerateRequest

    state: dict = {}
    orig_load = lora_mod.load_for_inference

    def load_for_inference(transformer, path, multiplier=1.0, before_apply=None):
        network = orig_load(transformer, path, multiplier, before_apply)
        state.update(transformer=transformer, network=network, multiplier=multiplier)
        return network

    lora_mod.load_for_inference = load_for_inference
    orig_lookup = RefCache.lookup

    def lookup(self, layers, tokens, rope, mod):
        hit = (
            self.tokens is not None
            and self.tokens.shape == tokens.shape
            and torch.equal(self.tokens, tokens)
        )
        if hit:
            return self.kv
        transformer, network = state["transformer"], state["network"]
        for name, adapter in network.adapters.items():
            is_kv = name.endswith(("proj_k", "proj_v"))
            keep = (args.mode == "kv" and is_kv) or (args.mode == "nokv" and not is_kv)
            adapter.multiplier = state["multiplier"] if keep else 0.0
        try:
            if args.mode == "all":
                linear = transformer.shared_block_modulation
                patched = linear.__dict__.pop("forward", None)
                try:
                    base_mod, _ = transformer.time_conditioning(
                        transformer.ref_timestep, tokens.shape[0]
                    )
                finally:
                    if patched is not None:
                        linear.__dict__["forward"] = patched
                delta = (
                    base_mod.to(mod.dtype) - mod
                ).float().norm() / mod.float().norm()
                print(
                    f"[probe] ref mod adapter removed: rel delta {delta:.4f}",
                    flush=True,
                )
                mod = base_mod.to(mod.dtype)
            kv = orig_lookup(self, layers, tokens, rope, mod)
        finally:
            network.set_multiplier(state["multiplier"])
        print(f"[probe] reference K/V built at multiplier 0 ({args.mode})", flush=True)
        return kv

    RefCache.lookup = lookup

    out_root = Path(args.out or f"{CONTROL}_reflora_{args.mode}")
    for page in args.pages:
        control = json.load(open(CONTROL / page / "manifest.json"))["images"][0]
        argv = [
            "--prompt",
            (INPUTS / f"{page}.txt").read_text().strip(),
            "--prompt_format",
            "text",
            "--lora",
            LORA,
            "--width",
            str(control["width"]),
            "--height",
            str(control["height"]),
            "--seed",
            str(control["seed"]),
            "--out_dir",
            str(out_root / page),
            "--ref_image",
            str(INPUTS / f"{page}.src.webp"),
            "--stream",
            "fp32",
            "--blocks_to_swap",
            "10",
        ]
        print(
            f"[probe] {page} {control['width']}x{control['height']} seed {control['seed']}",
            flush=True,
        )
        run_generate(GenerateRequest.from_argv(argv))
    print(f"[probe] done: {out_root}", flush=True)


if __name__ == "__main__":
    main()
