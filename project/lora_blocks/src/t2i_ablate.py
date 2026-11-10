"""Where a t2i LoRA's effect sits: block ranges and row kinds switched off at inference.

One trained LoRA, one DiT, the same noise per (prompt, seed); each arm keeps a part of
the delta:

- ``none`` / ``full``: multiplier 0 / 1 everywhere.
- ``b14_41`` / ``b26_41``: the adapters of blocks 0-13 / 0-25 off.
- ``img_rows``: the delta on the image rows only (the text rows run on Base).
- ``text_rows``: the delta on the text rows only.
- ``iso_none`` / ``iso_full``: ``none`` / ``full`` with the text queries reading only the
  text keys in every block (the image still reads the text): the text stream then
  depends on the prompt and σ, not on the image (``request.md``).

Every arm renders eager: the row arms set a per-forward mask on the adapters, which the
compiled blocks do not take, and compiled and eager renders are not compared with each
other. Prompts are ``bench.moss_ab.lora_ab.PROMPTS``; the TE settings (encoder, tail)
have to match the LoRA's cache.

    make daemon-run ARGS="--stall-timeout 600 project/lora_blocks/src/t2i_ablate.py \
        --lora output/ckpt_moss/caststation_fwdpv.safetensors"

Writes PNGs, ``sheet_p<i>[_<label>].png`` (rows = seeds, columns = arms) and
``index[_<label>].json`` to ``output/tests/t2i_ablate/<lora stem>/``.
"""

import argparse
import json
import re
import time

import torch

import library.moss.model as moss_model
from bench.moss_ab._render import (
    PROMPT_TE,
    decoded,
    grid,
    load_dit,
    lora_id,
    render,
    save_latent,
    settings_tag,
    spectrum_settings,
)
from bench.moss_ab.lora_ab import PROMPTS
from library.env import resolve_under_home
from library.moss.cache import caption_converter
from library.moss.generate import cached_embeds
from library.moss.loader import TE_PRECISIONS, empty_cache, load_vae, pad_embeds
from library.moss.model import snapped_sigmas

# arm -> (first block that keeps its adapters, rows that keep the delta, text isolated)
ARMS = {
    "none": (None, "all", False),
    "full": (0, "all", False),
    "b14_41": (14, "all", False),
    "b26_41": (26, "all", False),
    "img_rows": (0, "image", False),
    "text_rows": (0, "text", False),
    "iso_none": (None, "all", True),
    "iso_full": (0, "all", True),
}


def set_blocks(net, first: int | None) -> None:
    """Multiplier 1 on the adapters of blocks ``first``.. , 0 below (None: 0 on all)."""
    for name, adapter in net.adapters.items():
        block = int(re.search(r"layers_(\d+)_", name).group(1))
        adapter.multiplier = float(first is not None and block >= first)


def confine(tr, net, state: dict) -> None:
    """Wrap ``tr.run_layers`` so the delta keeps ``state["rows"]``: ``all``, the
    ``image`` rows or the ``text`` rows, on cond and uncond alike."""
    run = tr.run_layers

    def run_layers(
        x, rope, mod, skip_layers=frozenset(), key_mask=None, n_img=None, n_ref=0
    ):
        rows = state["rows"]
        if rows != "all":
            mask = torch.zeros(1, x.shape[1], 1, dtype=torch.bfloat16, device=x.device)
            if rows == "image":
                mask[:, :n_img] = 1
            else:
                mask[:, n_img : x.shape[1] - n_ref] = 1
            net.set_row_mask(mask)
        state["n_img"] = n_img
        try:
            return run(x, rope, mod, skip_layers, key_mask, n_img, n_ref)
        finally:
            state["n_img"] = None
            net.set_row_mask(None)

    tr.run_layers = run_layers


def isolate_text(state: dict) -> None:
    """Wrap ``model._attention`` (eager blocks look it up per call): under
    ``state["iso"]``, inside ``run_layers``, the text queries attend to the text keys
    alone. t2i only, no reference rows."""
    attention = moss_model._attention

    def _attention(q, k, v, rope, heads, key_mask=None, *rest, **kw):
        out = attention(q, k, v, rope, heads, key_mask, *rest, **kw)
        n = state["n_img"]
        if not state["iso"] or n is None:
            return out
        text = attention(
            q[:, n:],
            k[:, n:],
            v[:, n:],
            rope[:, n:],
            heads,
            None if key_mask is None else key_mask[:, n:],
        )
        return torch.cat((out[:, :n], text), 1)

    moss_model._attention = _attention


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--artist", default="@channel (caststation)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--stream", default="fp32", choices=("bf16", "fp32"))
    ap.add_argument("--te_precision", default="int8", choices=TE_PRECISIONS)
    ap.add_argument("--prompt_tail", default="sure")
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="", help="suffix of the sheets and the index")
    args = ap.parse_args()

    lora = resolve_under_home(args.lora)
    out = resolve_under_home(args.out or f"output/tests/t2i_ablate/{lora.stem}")
    (out / "latents").mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    w, h = map(int, args.size.split("x"))
    device = torch.device("cuda")

    prompts = [
        caption_converter("layout_json")(p.format(artist=args.artist)) for p in PROMPTS
    ]
    embeds = cached_embeds(
        "",
        ["", *prompts],
        resolve_under_home(PROMPT_TE),
        precision=args.te_precision,
        prompt_tail=args.prompt_tail,
    )
    uncond, uncond_mask = pad_embeds([embeds[""]], device)
    sigmas = snapped_sigmas(args.steps)
    tag = settings_tag(
        lora=lora_id(lora),
        prompts=prompts,
        steps=args.steps,
        cfg=args.cfg,
        size=args.size,
        variant=args.variant,
        stream=args.stream,
        te=args.te_precision,
        tail=args.prompt_tail,
        spectrum=spectrum_settings(True),
        compile=False,
    )
    jobs = [(a, pi, s) for pi in range(len(prompts)) for s in seeds for a in arms]

    def lat_path(job):
        arm, pi, seed = job
        return out / "latents" / f"{arm}_p{pi}_s{seed}_{tag}.safetensors"

    todo = [j for j in jobs if not lat_path(j).exists()]
    if todo:
        tr, net = load_dit(
            lora, device, args.variant, compile=False, stream=args.stream
        )
        state = {"rows": "all", "iso": False, "n_img": None}
        confine(tr, net, state)
        isolate_text(state)
        for job in todo:
            arm, pi, seed = job
            first, state["rows"], state["iso"] = ARMS[arm]
            set_blocks(net, first)
            t0 = time.time()
            z = render(
                tr,
                embeds[prompts[pi]],
                uncond,
                uncond_mask,
                sigmas,
                seed=seed,
                height=h,
                width=w,
                cfg=args.cfg,
            )
            save_latent(z, lat_path(job))
            print(f"{arm} p{pi} s{seed}: {time.time() - t0:.1f} s", flush=True)
        del net, tr
        empty_cache()

    vae = load_vae("", device=device)
    suffix = f"_{args.label}" if args.label else ""
    rows = []
    for pi in range(len(prompts)):
        sheet = []
        for seed in seeds:
            cells = []
            for arm in arms:
                png = out / f"{arm}_p{pi}_s{seed}.png"
                cells.append(
                    (arm, decoded(lat_path((arm, pi, seed)), vae, device, png))
                )
                rows.append({"arm": arm, "prompt": pi, "seed": seed, "path": str(png)})
            sheet.append((f"p{pi} s{seed}", cells))
        grid(sheet).save(out / f"sheet_p{pi}{suffix}.png")
    (out / f"index{suffix}.json").write_text(
        json.dumps(
            {"args": vars(args), "tag": tag, "prompts": prompts, "rows": rows}, indent=2
        )
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
