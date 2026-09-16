#!/usr/bin/env python3
"""From a method config to a built network to an in-process training run.

Three parts; the config part runs by default, the GPU parts are opt-in:

  1. load_method_preset() — the config merge chain (no GPU, no weights)
     base.toml → presets.toml[<preset>] → methods/<method>.toml → (CLI on top).
     Prints the network keys, including the three-axis LoRA routing surface
     (use_moe_style / route_per_layer / router_source).

  2. create_network()  (--build-network) — turn the resolved config into a live
     LoRA network bound to the DiT. Needs the DiT checkpoint.

  3. AnimaTrainer().train()  (--train) — run the training loop in-process on a
     single GPU, as `make lora` does by default. Multi-GPU runs need
     `accelerate launch` (`ANIMA_ACCELERATE_LAUNCH=1 make lora`).

    python examples/02_config_and_train.py --method lora --preset default
    python examples/02_config_and_train.py --method lora --build-network
    python examples/02_config_and_train.py --train --max_train_epochs 8 --network_dim 32

Part 3 prereq: `make download-models` and `make preprocess` (training reads only
the cached latents/embeddings under post_image_dataset/lora/). Any extra argv is
forwarded verbatim to the trainer (same override semantics as the CLI), so method
settings still win over preset on overlap.
"""

from __future__ import annotations

import argparse

from anima_lora.config import load_method_preset

# The LoRA routing/shape keys printed from the merged config.
NETWORK_KEYS = (
    "network_module",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "use_moe_style",
    "route_per_layer",
    "router_source",
)


def show_config(method: str, preset: str) -> dict:
    """Part 1 — merge + print the network-relevant keys with provenance."""
    merged, provenance = load_method_preset(method, preset, return_provenance=True)

    print(f"\nmethod={method!r}  preset={preset!r}")
    print("-" * 72)
    for k in NETWORK_KEYS:
        src = provenance.get(k, "(unset → code default)")
        print(f"  {k:16} = {merged.get(k)!r:30}  ← {src}")
    print("-" * 72)
    print(f"  ({len(merged)} keys total in the merged config)\n")
    return merged


def build_network(merged: dict):
    """Part 2 — instantiate the network against the real DiT.

    Mirrors how train.py wires the adapter: the resolved routing keys are
    forwarded as **kwargs to the network module's create_network().
    """
    import torch

    from library.anima import weights as anima_weights

    from anima_lora.training import create_network

    # Only the raw DiT is needed (no adapter to attach), so use the
    # explicit-argument primitive, as examples/04_load_models.py does.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = anima_weights.load_anima_model(
        device=device,
        dit_path="models/diffusion_models/anima-base-v1.0.safetensors",
        attn_mode="torch",
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )

    # Forward the routing surface (skip None — let create_network use its own
    # defaults) plus any other str-valued knobs the module reads from kwargs.
    routing = {
        k: merged[k]
        for k in NETWORK_KEYS
        if merged.get(k) is not None and k != "network_module"
    }
    network = create_network(
        multiplier=1.0,
        network_dim=merged.get("network_dim"),
        network_alpha=merged.get("network_alpha"),
        vae=None,
        text_encoders=[],
        unet=unet,
        **{
            k: v
            for k, v in routing.items()
            if k not in ("network_dim", "network_alpha")
        },
    )
    n_params = sum(p.numel() for p in network.parameters())
    print(f"built {type(network).__name__}: {n_params:,} trainable params")
    return network


def run_training(method: str, preset: str, extra_argv: list[str]) -> None:
    """Part 3 — reproduce train.py's __main__ block and run the trainer.

    setup_parser() + populate_schema()  →  parse  →  read_config_from_file  →
    AnimaTrainer().train(args)
    """
    from library.config import schema as config_schema

    # `anima_lora.training` loads repo-root train.py by path (any CWD works).
    from anima_lora.config import read_config_from_file
    from anima_lora.training import (
        AnimaTrainer,
        build_network_extras,
        setup_parser,
        verify_command_line_training_args,
    )

    argv = ["--method", method, "--preset", preset, *extra_argv]

    parser = setup_parser()
    # populate_schema adds the config-driven flags, including the routing keys
    # create_network reads.
    config_schema.populate_schema(parser, extras=build_network_extras())

    args = parser.parse_args(argv)
    verify_command_line_training_args(args)
    # base→preset→method merge, then CLI overrides on top. `argv` is passed
    # explicitly so the override layer reads our list, not sys.argv.
    args = read_config_from_file(args, parser, argv=argv)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    AnimaTrainer().train(args)


def main() -> None:
    # parse_known_args: unknown args (e.g. --max_train_epochs 8) go to the trainer.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", default="lora")
    p.add_argument("--preset", default="default")
    p.add_argument(
        "--build-network",
        action="store_true",
        help="part 2: also instantiate the network (loads the DiT — slow)",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="part 3: run the training loop in-process (needs preprocessed cache)",
    )
    opts, extra_argv = p.parse_known_args()

    if opts.train:
        # Training assembles + merges the config itself (via train.py's parser),
        # so go straight to the trainer with method/preset + forwarded overrides.
        run_training(opts.method, opts.preset, extra_argv)
        return

    merged = show_config(opts.method, opts.preset)
    if opts.build_network:
        build_network(merged)


if __name__ == "__main__":
    main()
