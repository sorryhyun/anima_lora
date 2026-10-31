"""Adapter-network construction for the Anima trainer: the optional
gradient-SVD basis sketch (before the network exists), then build → apply →
load weights → compile."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Optional

from library.anima.trainer.cli import resolve_network_kwargs
from library.env import project_root
from library.runtime.device import clean_memory_on_device
from library.training import NetworkBundle

logger = logging.getLogger(__name__)


def sketch_grad_basis(
    args, accelerator, unet, train_dataset_group, weight_dtype, *, swapping: bool
) -> None:
    """``down_init="grad_svd"``: sketch this run's own gradient row space.

    A LoRA-GA / LoRA-One style seed needs the task gradient *before* the
    adapter exists, so this runs one frozen-DiT forward/backward per
    (image, σ) over the run's cached latents+TE, takes the top-r right
    singular vectors per target Linear, writes them beside the checkpoint,
    and hands the path to the factory as ``grad_basis_file`` — the same
    artifact ``down_init="basis_file"`` reads, so the two modes share one
    load path. Output is an ordinary LoRA either way (B=0, ΔW=0 at step 0).

    Refused under block swap: the swapper's residency plan assumes the
    training loop's forward cadence and desyncs on extra forwards. Build the basis with
    ``bench/grad_init/build_universal_basis.py`` and pass ``basis_file``
    instead on a swap preset.
    """
    net_kwargs = resolve_network_kwargs(args)
    if net_kwargs.get("down_init") != "grad_svd":
        return
    if net_kwargs.get("grad_basis_file"):
        logger.info(
            "down_init=grad_svd: grad_basis_file already set "
            f"({net_kwargs['grad_basis_file']}); skipping the sketch pass."
        )
        return
    if swapping:
        raise ValueError(
            "down_init='grad_svd' needs a resident DiT for its sketch pass, "
            "but blocks_to_swap>0. Use down_init='basis_file' with a basis "
            "built by bench/grad_init/build_universal_basis.py, or set "
            "blocks_to_swap=0."
        )

    from library.env import resolve_under_home
    from networks.grad_basis import (
        BASIS_SUFFIX,
        basis_from_sketches,
        dit_num_blocks,
        save_basis,
        sketch_dataset,
    )

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for info in train_dataset_group.image_data.values():
        if info.is_reg:
            continue
        npz, te = info.latents_npz, info.text_encoder_outputs_npz
        if not npz or not te or npz in seen:
            continue
        seen.add(npz)
        pairs.append((npz, te))
    pairs.sort()  # dict order follows the glob; keep the sketch reproducible
    if not pairs:
        raise RuntimeError(
            "down_init='grad_svd' needs cached latents + text-encoder "
            "outputs; none of the training images have both. Run "
            "`make preprocess` first."
        )

    rank = int(args.network_dim or 4)
    samples = int(net_kwargs.get("grad_basis_samples", 64) or 0)
    passes = int(net_kwargs.get("grad_basis_passes", 1) or 1)
    oversample = int(net_kwargs.get("grad_basis_oversample", 32) or 32)
    seed = int(net_kwargs.get("grad_basis_seed", args.seed or 42) or 42)

    out_path = resolve_under_home(
        os.path.join(args.output_dir, f"{args.output_name}{BASIS_SUFFIX}")
    )
    logger.info(
        f"down_init=grad_svd: sketching r={rank} (q={rank + oversample}) over "
        f"{min(samples, len(pairs)) if samples else len(pairs)} of "
        f"{len(pairs)} cached images x {passes} pass(es)"
    )

    was_training = unet.training
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    # Gradient checkpointing is gated on module.training (models.py), and a
    # 4k-token backward without it does not fit the 16 GB envelope — turn it
    # on for the sketch regardless of args.gradient_checkpointing, then put
    # the model back exactly as the trainer expects it.
    had_ckpt = bool(getattr(unet, "gradient_checkpointing", False))
    unet.train()
    if not had_ckpt:
        unet.enable_gradient_checkpointing()
    try:
        sketches, meta = sketch_dataset(
            unet,
            pairs,
            rank=rank,
            device=accelerator.device,
            oversample=oversample,
            passes=passes,
            seed=seed,
            max_samples=samples,
        )
    finally:
        if not had_ckpt:
            unet.disable_gradient_checkpointing()
        if not was_training:
            unet.eval()
        unet.to("cpu")
        unet.zero_grad(set_to_none=True)
        clean_memory_on_device(accelerator.device)

    basis = basis_from_sketches(sketches, rank)
    save_basis(
        out_path,
        basis,
        num_blocks=dit_num_blocks(unet),
        extra_metadata={
            "source": "grad_svd (per-run sketch)",
            "n_used": meta["n_used"],
            "passes": meta["passes"],
            "seed": meta["seed"],
            "mean_loss": round(meta["mean_loss"], 6),
        },
    )
    logger.info(
        f"down_init=grad_svd: basis written to {out_path} "
        f"({len(basis)} layers, {meta['n_used']} samples, {meta['seconds']}s)"
    )
    # Mutating the cached dict is how the factory sees it — net_kwargs IS
    # args._network_kwargs, which create_and_apply_network re-reads.
    net_kwargs["grad_basis_file"] = str(out_path)


def create_and_apply_network(
    args,
    accelerator,
    vae,
    text_encoders,
    unet,
    weight_dtype,
    *,
    step,
    token_budget,
) -> Optional[NetworkBundle]:
    """Import the network module, merge base weights, build the adapter,
    hand it to ``step`` (aux-loss wiring), apply it to the DiT, load weights,
    then compile. ``token_budget`` is :func:`data.derive_token_budget`'s
    triple."""
    sys.path.append(str(project_root()))
    accelerator.print("import network module:", args.network_module)
    network_module = importlib.import_module(args.network_module)

    if args.base_weights is not None:
        for i, weight_path in enumerate(args.base_weights):
            if (
                args.base_weights_multiplier is None
                or len(args.base_weights_multiplier) <= i
            ):
                multiplier = 1.0
            else:
                multiplier = args.base_weights_multiplier[i]

            accelerator.print(
                f"merging module: {weight_path} with multiplier {multiplier}"
            )

            module, weights_sd = network_module.create_network_from_weights(
                multiplier, weight_path, vae, text_encoders, unet, for_inference=True
            )
            module.merge_to(
                text_encoders,
                unet,
                weights_sd,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
            )

        accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

    # Copied so the dropout default below stays a factory-call detail,
    # not part of the cached ``args._network_kwargs`` view other
    # consumers read.
    net_kwargs = dict(resolve_network_kwargs(args))

    if args.dim_from_weights:
        network, _ = network_module.create_network_from_weights(
            1, args.network_weights, vae, text_encoders, unet, **net_kwargs
        )
    else:
        if "dropout" not in net_kwargs:
            net_kwargs["dropout"] = args.network_dropout

        network = network_module.create_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            vae,
            text_encoders,
            unet,
            neuron_dropout=args.network_dropout,
            **net_kwargs,
        )
    if network is None:
        return None

    if hasattr(network, "prepare_network"):
        network.prepare_network(args)
    if args.scale_weight_norms and not hasattr(
        network, "apply_max_norm_regularization"
    ):
        logger.warning(
            "warning: scale_weight_norms is specified but the network does not support it"
        )
        args.scale_weight_norms = False

    step.attach_network(args, network, unet)

    # Token-adding adapters (register tokens) do mid-stack token surgery
    # via block pre-hooks — unaudited against the block-swap offloader,
    # which desyncs on unexpected per-step forward patterns.
    if (
        int(getattr(network, "extra_seq_tokens", 0) or 0) > 0
        and args.blocks_to_swap is not None
        and args.blocks_to_swap > 0
    ):
        raise ValueError(
            "Register tokens (extra_seq_tokens>0) require blocks_to_swap=0 — "
            "the mid-stack token insertion pre-hooks have not been validated "
            "against the block-swap offloader and can silently desync it. "
            "Use block compile for memory instead."
        )

    int8 = getattr(args, "int8", "off")
    if int8 != "off":
        from library.anima.int8 import convert_for_training

        convert_for_training(unet, network, args)

    # DiT only: the text encoder is never trained.
    network.apply_to(text_encoders, unet, False, True)

    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        accelerator.print(f"load network weights from {args.network_weights}: {info}")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        network.enable_gradient_checkpointing()  # may have no effect

    # Native-shape flattening + per-block torch.compile. COMPILE LAST —
    # after apply_to + load_weights so dynamo traces the adapter's
    # monkey-patched forwards, not the bare DiT (see harness.py).
    if args.torch_compile:
        from library.runtime.harness import compile_blocks_for_training

        n_token_families, seq_range, seq_bands = token_budget
        if not getattr(args, "compile_seq_bands", False):
            seq_bands = None
        # Register tokens grow the seq by a constant K, so widen the
        # dynamic-seq bound's MAX by K or the compiled block's bound is
        # violated (the min/family-count stay: mid-stack insertion still
        # runs blocks before the insert at the bare seq). Per-band mode
        # widens each band's hi; widen_bands raises if K would make
        # adjacent bands touch (silent merging would un-tighten them).
        extra_seq = int(getattr(network, "extra_seq_tokens", 0) or 0)
        if extra_seq and seq_range is not None:
            seq_range = (seq_range[0], seq_range[1] + extra_seq)
            if seq_bands:
                from library.datasets.buckets import widen_bands

                seq_bands = widen_bands(seq_bands, extra_seq)
        compile_blocks_for_training(
            unet,
            network,
            backend=args.dynamo_backend,
            mode=getattr(args, "compile_inductor_mode", None),
            n_token_families=n_token_families,
            seq_range=seq_range,
            seq_bands=seq_bands,
            dynamic_seq=bool(getattr(args, "compile_dynamic_seq", False)),
            signature_suffix=""
            if int8 == "off"
            else f";int8={int8};int8_in=fp32;flops=fa2,int8",
            activation_memory_budget=float(
                getattr(args, "activation_memory_budget", 1.0) or 1.0
            ),
            partitioner_recompute_views=bool(
                getattr(args, "partitioner_recompute_views", False)
            ),
            partitioner_aggressive_recomputation=bool(
                getattr(args, "partitioner_aggressive_recomputation", False)
            ),
            grad_ckpt=bool(getattr(args, "gradient_checkpointing", False)),
            logger=logger,
        )

    return NetworkBundle(network=network, net_kwargs=net_kwargs)
