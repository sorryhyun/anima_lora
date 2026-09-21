"""Sweep attention backends x block compilation on one Qwen-Image-2.1 load.

Every arm denoises the same prompt at the same seed and prints the max absolute
latent deviation from the first arm, so a backend that changes the picture shows
up next to its speedup.

One process, because a reload is 33 GB off disk and an arm is 20 seconds: the
text encoder is streamed, used and dropped once, then the transformer stays on
the card while the arms reconfigure it in place. Compiled arms run last — the
attention backend is read inside the traced region, so switching it after a
compile forces a recompile.

    make daemon-run ARGS="project/qwen21_lora/src/bench_accel.py --steps 20"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from accel import compile_blocks, recompile_report, set_attention_backend  # noqa: E402
from loader import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    TEXT_ENCODER_BLOCKS,
    TRANSFORMER_BLOCKS,
    drop_text_encoder,
    empty_cache,
    encode_prompts,
    free_vram_gb,
    load_pipeline,
    load_text_encoder,
    place,
)

PROMPT = (
    'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, '
    "reflections on wet pavement"
)


def profile_arm(pipe, call, steps: int, seed: int) -> None:
    """Print the top CUDA ops of a short denoise, self time first.

    A block-swap stall shows up as ``Memcpy DtoH/HtoD`` and as wall time the
    kernel table does not account for; a kernel-bound loop shows GEMMs at the
    top.
    """
    from torch.profiler import ProfilerActivity, profile

    warm = dict(call, num_inference_steps=steps)
    pipe(
        output_type="latent",
        generator=torch.Generator("cuda").manual_seed(seed),
        **warm,
    )
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        pipe(
            output_type="latent",
            generator=torch.Generator("cuda").manual_seed(seed),
            **warm,
        )
    print(
        prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=20, max_name_column_width=55
        ),
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeat", type=int, default=3, help="passes per arm")
    ap.add_argument("--blocks_to_swap", type=int, default=None)
    ap.add_argument("--te_blocks_to_swap", type=int, default=None)
    ap.add_argument(
        "--backends",
        default="native,flash",
        help="comma-separated diffusers attention backends to sweep",
    )
    ap.add_argument("--compile_mode", default=None)
    ap.add_argument("--no_compile_arms", action="store_true")
    ap.add_argument(
        "--swap_arms",
        default=None,
        help="comma-separated block-swap configs to sweep instead of the "
        "backend/compile arms, each `<blocks_to_swap>[:ring]` — e.g. "
        "'0,4,4:ring,12,12:ring'. `ring` is ModelOffloader's own N-move "
        "schedule, the default is the 2S-move one in blockswap.py",
    )
    ap.add_argument(
        "--profile",
        type=int,
        default=0,
        help="profile this many steps of each swept backend and print the top CUDA ops",
    )
    ap.add_argument("--out", default="project/qwen21_lora/out/bench_accel.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.cuda.init()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    te = load_text_encoder(args.model_dir)
    pipe = load_pipeline(args.model_dir, text_encoder=te)

    te_attached = place(
        te,
        TEXT_ENCODER_BLOCKS,
        device,
        blocks_to_swap=args.te_blocks_to_swap,
        label="text_encoder",
    )
    embeds, mask, _pad = encode_prompts(pipe, [args.prompt], device="cuda")[0]
    if te_attached is not None:
        te_attached.detach()
    del te, te_attached
    drop_text_encoder(pipe)

    # Held for the whole sweep: detaching an attachment takes the swapper's
    # hooks off with it.
    attachment = [None]

    def attach_dit(blocks_to_swap, minimal=True):
        if attachment[0] is not None:
            attachment[0].detach()
            attachment[0] = None
            empty_cache()
        attachment[0] = place(
            pipe.transformer,
            TRANSFORMER_BLOCKS,
            device,
            blocks_to_swap=blocks_to_swap,
            label="transformer",
            minimal_schedule=minimal,
        )
        empty_cache()
        print(f"free VRAM with DiT placed: {free_vram_gb():.2f} GB\n", flush=True)

    attach_dit(args.blocks_to_swap)

    call = dict(
        prompt_embeds=embeds.to("cuda"),
        prompt_embeds_mask=None if mask is None else mask.to("cuda"),
        num_inference_steps=args.steps,
        output_resolution=args.resolution,
        true_cfg_scale=1.0,
    )

    def run_arm(name: str, backend: str) -> dict:
        set_attention_backend(pipe.transformer, backend, padded_prompt=mask is not None)
        times, peak, latents = [], 0.0, None
        for _ in range(args.repeat):
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            latents = pipe(
                output_type="latent",
                generator=torch.Generator("cuda").manual_seed(args.seed),
                **call,
            ).images
            times.append(time.time() - t0)
            peak = max(peak, torch.cuda.max_memory_allocated() / 1024**3)
        steady = min(times[1:]) if len(times) > 1 else times[0]
        print(
            f"{name}: first {times[0]:.1f}s, steady {steady:.1f}s "
            f"({steady / args.steps:.3f}s/step), peak {peak:.2f} GB",
            flush=True,
        )
        return {
            "arm": name,
            "backend": backend,
            "times_s": [round(t, 2) for t in times],
            "steady_s": round(steady, 2),
            "s_per_step": round(steady / args.steps, 4),
            "peak_gb": round(peak, 2),
            "latents": latents.float().cpu(),
        }

    if args.profile:
        for backend in backends:
            set_attention_backend(
                pipe.transformer, backend, padded_prompt=mask is not None
            )
            print(f"\n=== profile: {backend} ===", flush=True)
            profile_arm(pipe, call, args.profile, args.seed)

    if args.swap_arms:
        results = []
        for spec in args.swap_arms.split(","):
            count, _, kind = spec.strip().partition(":")
            minimal = kind != "ring"
            attach_dit(int(count), minimal=minimal)
            label = f"swap {count}" + ("" if minimal else " (ring)")
            results.append(run_arm(label, backends[0]))
    else:
        results = [run_arm(b, b) for b in backends]

        if not args.no_compile_arms:
            print(flush=True)
            compile_blocks(pipe.transformer.transformer_blocks, mode=args.compile_mode)
            results += [run_arm(f"{b}+compile", b) for b in backends]
            print(recompile_report(), flush=True)

    reference = results[0]["latents"]
    print(f"\n{'arm':<24} {'s/step':>8} {'speedup':>8} {'peak GB':>8} {'max|Δ|':>9}")
    rows = []
    for row in results:
        delta = (row.pop("latents") - reference).abs().max().item()
        row["max_abs_delta_vs_" + results[0]["arm"]] = round(delta, 5)
        row["speedup"] = round(results[0]["steady_s"] / row["steady_s"], 3)
        print(
            f"{row['arm']:<24} {row['s_per_step']:>8.3f} {row['speedup']:>8.3f} "
            f"{row['peak_gb']:>8.2f} {delta:>9.5f}",
            flush=True,
        )
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "resolution": args.resolution,
                "steps": args.steps,
                "repeat": args.repeat,
                "compile_mode": args.compile_mode,
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "arms": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
