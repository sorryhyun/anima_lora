"""Shared DiT + adapter run harness.

Loads the DiT, attaches an optional adapter, and compiles, in the required
order: ``torch.compile`` traces the adapter's monkey-patched forward, so
``compile_blocks`` must run after ``network.apply_to`` + ``load_weights`` or it
silently compiles the unpatched forward.

Usage::

    from library.runtime.harness import build_anima

    bundle = build_anima(args, dit_path=..., adapter=..., train_mode=False)
    anima, network = bundle.anima, bundle.network

Knobs come off an ``argparse.Namespace`` (``library.runtime.argparse_groups``).
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from library.runtime.dynamo import (
    compile_signature,
    isolate_compile_cache,
    set_activation_memory_budget,
)

log = logging.getLogger("library.runtime.harness")


@dataclass
class AnimaBundle:
    """``build_anima``'s result; ``network`` is ``None`` without an adapter."""

    anima: object  # library.anima Anima (object: avoids the heavy import)
    network: Optional[object]  # networks.lora_anima.network.LoRANetwork or None
    device: torch.device
    dtype: torch.dtype


def build_anima(
    args: argparse.Namespace,
    *,
    dit_path: str | None = None,
    adapter: str | None = None,
    train_mode: bool = False,
    network_requires_grad: bool = True,
    multiplier: float = 1.0,
) -> AnimaBundle:
    """Load the DiT (+ optional adapter) in the required order.

    load DiT → freeze + ``reset_mod_guidance`` → adapter ``apply_to`` →
    ``load_weights`` → requires_grad per ``train_mode`` → gradient checkpointing
    → train()/eval() → ``compile_blocks`` last.

    Arguments:
        args: reads ``device``, ``dtype``, ``attn_mode``,
            ``gradient_checkpointing``, ``compile``, ``compile_mode``.
        dit_path: defaults to ``args.dit`` if the caller's argparse exposed one.
        adapter: optional adapter safetensors path.
        train_mode: required before any ``backward()``: the LoRA training
            forward, T-LoRA mask and gradient checkpointing gate on
            ``self.training``.
        network_requires_grad: under ``train_mode``, whether adapter params train.
        multiplier: adapter scale; ``set_multiplier(0.0)`` recovers the base.
    """
    # keep this module cheap to import on CPU-only runs that never load a DiT
    from library.anima import weights as anima_utils
    from library.runtime.device import str_to_dtype

    device = torch.device(getattr(args, "device", "cuda"))
    dtype = str_to_dtype(getattr(args, "dtype", "bf16"))
    from library.runtime.backend import (
        needs_rocm_attention_fallback,
        resolve_attention_mode,
    )

    requested_attn_mode = getattr(args, "attn_mode", "flash")
    attn_mode = resolve_attention_mode(requested_attn_mode, torch)
    if needs_rocm_attention_fallback(requested_attn_mode, torch):
        log.warning("ROCm detected: using PyTorch SDPA instead of CUDA Flash Attention")

    if dit_path is None:
        dit_path = getattr(args, "dit", None)
    if dit_path is None:
        raise SystemExit(
            "build_anima: no DiT path. Pass dit_path= explicitly or expose "
            "--dit in your argparse."
        )

    log.info(f"loading base DiT: {dit_path}")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=dit_path,
        attn_mode=attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device, dtype=dtype).requires_grad_(False)
    anima.reset_mod_guidance()

    network = None
    if adapter is not None:
        log.info(f"loading adapter:  {adapter}")
        from networks.lora_anima.factory import create_network_from_weights

        network, _sd = create_network_from_weights(
            multiplier,
            adapter,
            None,  # ae (unused for harness callers)
            None,  # text_encoders (unused for harness callers)
            anima,
            for_inference=not train_mode,
        )
        network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
        info = network.load_weights(adapter)
        log.info(f"adapter loaded — {info}")

        network.to(device=device, dtype=dtype)
        if train_mode and network_requires_grad:
            network.requires_grad_(True)
        else:
            network.requires_grad_(False)
        anima.requires_grad_(False)

        trainable = [p for p in network.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in trainable)
        if train_mode and network_requires_grad:
            if n_train == 0:
                raise SystemExit(
                    "build_anima: adapter loaded with train_mode=True but "
                    "no trainable parameters were detected. Check the "
                    "checkpoint."
                )
            log.info(
                f"adapter trainable params: {n_train:,} ({len(trainable)} tensors)"
            )

    # gated on anima.training (models.py); effect requires train_mode below
    if getattr(args, "gradient_checkpointing", False):
        log.info("enabling gradient checkpointing")
        anima.enable_gradient_checkpointing()

    if train_mode:
        anima.train()
        if network is not None:
            network.train()
    else:
        anima.eval()
        if network is not None:
            network.eval()

    # Compile last (module docstring).
    if getattr(args, "compile", False):
        mode = getattr(args, "compile_mode", None)
        log.info(
            f"compiling DiT blocks{' (mode=' + mode + ')' if mode else ''} "
            "— first batch pays ~30-60s compile cost"
        )
        anima.compile_blocks(mode=mode)

    return AnimaBundle(anima=anima, network=network, device=device, dtype=dtype)


@dataclass
class InferenceBundle:
    """Everything a probe needs to drive (and hook) a real ``generate()`` call.

    The DiT sits in ``shared_models["model"]``, so ``generate()`` reuses this
    instance and a forward hook on ``bundle.model`` stays live during sampling.
    ``vae`` is ``None`` with ``with_vae=False`` or no ``--vae`` path.
    """

    model: object  # the loaded DiT (also stashed in shared_models["model"])
    vae: Optional[object]  # AutoencoderKLQwenImage, or None
    text_encoder: object  # on CPU; generate()/prepare_text_inputs moves it
    gen_settings: object  # library.inference.GenerationSettings
    shared_models: dict  # {"model", "text_encoder", "conds_cache"} -> generate()
    args: argparse.Namespace  # the namespace the bundle was built from
    device: torch.device

    def generate(self, args: Optional[argparse.Namespace] = None):
        """``library.inference.generate`` on the loaded models; returns the latent."""
        from library.inference import generate as _generate

        return _generate(args or self.args, self.gen_settings, self.shared_models)


def build_inference_bundle(
    args: argparse.Namespace,
    device: torch.device | str | None = None,
    *,
    with_vae: bool = True,
) -> InferenceBundle:
    """Assemble the text-encoder + DiT (+ optional VAE) set for a generation.

    The sequence ``inference.main()`` open-codes, for bench/probe callers.

    Args:
        args: a fully-defaulted namespace (``GenerationRequest.to_args()`` /
            ``build_default_args``).
        device: written back to ``args.device`` so every loader agrees.
            ``None`` resolves cuda-else-cpu.
        with_vae: ``False`` skips the VAE for latent-space probes.
    """
    from library.inference import (
        get_generation_settings,
        load_dit_model,
        load_shared_models,
    )

    if device is not None:
        args.device = str(device) if not isinstance(device, str) else device

    gen_settings = get_generation_settings(args)
    resolved_device = gen_settings.device

    shared_models = load_shared_models(args)  # text encoder on CPU
    shared_models["conds_cache"] = {}

    anima = load_dit_model(args, resolved_device, torch.bfloat16)
    # generate() reuses this instance: the seam for hooking the DiT.
    shared_models["model"] = anima

    vae = None
    vae_path = getattr(args, "vae", None)
    if with_vae and vae_path:
        from library.models.qwen_vae import load_vae

        vae = load_vae(
            vae_path,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=getattr(args, "vae_chunk_size", None),
            disable_cache=getattr(args, "vae_disable_cache", False),
            dtype=torch.bfloat16,
            eval=True,
        )
    elif with_vae:
        log.warning(
            "build_inference_bundle(with_vae=True) but args.vae is unset; "
            "bundle.vae is None (latent decode will be unavailable)."
        )

    return InferenceBundle(
        model=anima,
        vae=vae,
        text_encoder=shared_models["text_encoder"],
        gen_settings=gen_settings,
        shared_models=shared_models,
        args=args,
        device=resolved_device,
    )


# Training-side helpers: distillation trainers build a fresh network with their
# own freeze/optimizer/swap ordering and can't call build_anima wholesale, so
# these factor out the shared parts. compile-after-apply still applies.


def place_dit_for_training(
    anima: object, device: torch.device, *, blocks_to_swap: int = 0
) -> None:
    """Move a (frozen-base) DiT onto ``device`` for a training run.

    With block swap, swapped blocks stay on CPU and ride the forward+backward
    swap hooks. Call before ``compile_dit_blocks`` / ``train()``.
    """
    if blocks_to_swap > 0:
        anima.enable_block_swap(blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.switch_block_swap_for_training()
    else:
        anima.to(device)


def compile_dit_blocks(
    anima: object,
    *,
    enabled: bool = True,
    cache_size_limit: int = 64,
    backend: str = "inductor",
    mode: Optional[str] = None,
    dynamic_seq: bool = False,
    n_token_families: Optional[int] = None,
    seq_range: Optional[tuple] = None,
) -> None:
    """``torch.compile`` each ``Block._forward`` for a distillation/training run.

    Native-shape flattening traces one graph per token count, so the dynamo
    cache is raised to ``cache_size_limit`` first, else shapes fall back to
    eager mid-warmup. ``dynamic_seq`` marks only the seq axis dynamic (one
    graph), bounded by ``seq_range``. Run after the adapter's ``apply_to``.
    """
    if not enabled:
        return
    from library.runtime.dynamo import pin_dynamo_limit

    # pin the canonical .default so it survives into the backward compile context
    pin_dynamo_limit("recompile_limit", cache_size_limit)
    anima.compile_blocks(
        backend,
        mode=mode,
        n_token_families=n_token_families,
        dynamic_seq=dynamic_seq,
        seq_range=seq_range,
    )


def _apply_activation_memory_budget(
    budget: float, *, grad_ckpt: bool, logger: logging.Logger = log
) -> None:
    """Cap the AOT min-cut partitioner's saved-for-backward set.

    ``budget < 1.0`` recomputes cheap intermediates instead of saving them
    (0.85 fixed a first-step OOM at 16 GB / 4200 tokens without grad-ckpt).
    Set before the block compile (partitioning happens at first-forward
    compile). Skipped under gradient checkpointing: repartitioning lets the
    recompute pass pick a different graph than forward, raising
    ``CheckpointError`` (torch #166926).
    """
    if budget < 1.0 and not grad_ckpt:
        set_activation_memory_budget(budget)
        logger.info(
            "torch.compile activation_memory_budget = %.3g "
            "(partitioner recomputes cheap intermediates in backward)",
            budget,
        )
    elif budget < 1.0:
        logger.info(
            "activation_memory_budget ignored: incompatible with "
            "gradient_checkpointing (and redundant under it)"
        )


def _apply_partitioner_tuning(
    *,
    recompute_views: bool,
    aggressive_recomputation: bool,
    grad_ckpt: bool,
    logger: logging.Logger = log,
) -> None:
    """Tune the AOT min-cut partitioner's default save/recompute heuristics.

    These change what the default partition will recompute at all:
    ``recompute_views`` stops saving view ops, ``aggressive_recomputation``
    drops the recompute ban list when the min-cut is cheap
    (docs/anima/findings/custom_autograd_removal_partitioner_oom.md). Skipped
    under gradient checkpointing, as for the budget.
    """
    if not (recompute_views or aggressive_recomputation):
        return
    if grad_ckpt:
        logger.info(
            "partitioner tuning (recompute_views=%s, aggressive_recomputation=%s) "
            "ignored: incompatible with gradient_checkpointing (and redundant "
            "under it)",
            recompute_views,
            aggressive_recomputation,
        )
        return
    import torch._functorch.config as _functorch_config

    if recompute_views:
        _functorch_config.recompute_views = True
    if aggressive_recomputation:
        _functorch_config.aggressive_recomputation = True
    logger.info(
        "partitioner tuning: recompute_views=%s aggressive_recomputation=%s",
        recompute_views,
        aggressive_recomputation,
    )


def _apply_cudagraph_skip_dynamic(
    mode: Optional[str], *, logger: logging.Logger = log
) -> None:
    """Opt-in: skip CUDAGraph capture for dynamic-shape graphs (env toggle).

    Under ``reduce-overhead`` inductor records a CUDAGraph per input size, and
    free-fit bucketing yields many seq lengths, each pinning its own static
    buffers. ``ANIMA_CUDAGRAPH_SKIP_DYNAMIC=1`` keeps CUDAGraphs for static
    graphs only.
    """
    if os.environ.get("ANIMA_CUDAGRAPH_SKIP_DYNAMIC", "0") not in ("1", "true", "True"):
        return
    if mode not in ("reduce-overhead", "max-autotune"):
        logger.info(
            "ANIMA_CUDAGRAPH_SKIP_DYNAMIC set but mode=%r has no cudagraphs — no-op",
            mode,
        )
        return
    import torch._inductor.config as _inductor_config

    _inductor_config.triton.cudagraph_skip_dynamic_graphs = True
    logger.info(
        "cudagraph_skip_dynamic_graphs = True "
        "(dynamic-seq blocks run as inductor launches; cudagraphs only for static graphs)"
    )


def compile_blocks_for_training(
    unet: object,
    network: object,
    *,
    backend: str,
    mode: Optional[str] = None,
    n_token_families: Optional[int] = None,
    seq_range: Optional[tuple] = None,
    seq_bands: Optional[list] = None,
    dynamic_seq: bool = False,
    signature_suffix: str = "",
    activation_memory_budget: float = 1.0,
    partitioner_recompute_views: bool = False,
    partitioner_aggressive_recomputation: bool = False,
    grad_ckpt: bool = False,
    logger: logging.Logger = log,
) -> None:
    """The LoRA-training (``train.py``) compile sequence, post ``apply_to``.

    Run after ``network.apply_to`` + ``load_weights``. ``signature_suffix``
    keeps a base the signature cannot see (int8) in its own compile cache.

    EasyControl's patched ``Block.forward`` routes its cond path through
    ``_two_stream_inner``, bypassing the compiled ``block._forward``, so
    ``compile_cond_stream`` must run too or the cond stream stays eager.
    """
    _apply_activation_memory_budget(
        activation_memory_budget, grad_ckpt=grad_ckpt, logger=logger
    )
    _apply_partitioner_tuning(
        recompute_views=partitioner_recompute_views,
        aggressive_recomputation=partitioner_aggressive_recomputation,
        grad_ckpt=grad_ckpt,
        logger=logger,
    )
    _apply_cudagraph_skip_dynamic(mode, logger=logger)
    isolate_compile_cache(
        compile_signature(
            n_token_families=n_token_families,
            seq_range=seq_range,
            dynamic_seq=dynamic_seq,
            backend=backend,
            mode=mode,
            seq_bands=seq_bands,
        )
        + signature_suffix
    )
    unet.compile_blocks(
        backend,
        mode=mode,
        n_token_families=n_token_families,
        dynamic_seq=dynamic_seq,
        seq_range=seq_range,
        seq_bands=seq_bands,
    )
    # The cond stream stays on the union range (no seq_bands).
    if hasattr(network, "compile_cond_stream"):
        network.compile_cond_stream(
            backend,
            mode=mode,
            n_token_families=n_token_families,
            dynamic_seq=dynamic_seq,
            seq_range=seq_range,
        )


@dataclass
class PoolCompileResult:
    """What :func:`compile_dit_blocks_for_pool` derived, for caller-side logging.

    ``n_token_families`` / ``seq_range`` are ``None`` on the static (per-shape)
    path; ``n_shapes`` is always the distinct-token-count tally (≥1).
    """

    n_shapes: int
    n_token_families: Optional[int]
    seq_range: Optional[tuple]


def compile_dit_blocks_for_pool(
    anima: object,
    token_counts,
    *,
    enabled: bool = True,
    dynamic_seq: bool = True,
    backend: str = "inductor",
    mode: Optional[str] = None,
    activation_memory_budget: float = 1.0,
    grad_ckpt: bool = False,
    cache_size_limit: Optional[int] = None,
    logger: logging.Logger = log,
) -> PoolCompileResult:
    """Partitioner budget → per-signature cache isolation → block compile, for a
    self-describing distillation pool.

    Shared by ``distill_turbo`` / ``distill_mod``. ``token_counts`` are the
    distinct ``(W//patch)*(H//patch)`` counts of the cached pool. The dynamo
    cache defaults to ``2*n_shapes + 8``. Returns the result even when
    ``enabled`` is False, for logging. Run after the network's ``apply_to``.
    """
    counts = {int(c) for c in token_counts}
    n_shapes = max(1, len(counts))
    if dynamic_seq and counts:
        # one symbolic-seq graph for every shape, bounded by the pool's token range
        n_token_families: Optional[int] = n_shapes
        seq_range: Optional[tuple] = (min(counts), max(counts))
    else:
        # static path: one graph per distinct token count (no padding)
        n_token_families = None
        seq_range = None
    result = PoolCompileResult(n_shapes, n_token_families, seq_range)

    if not enabled:
        return result

    _apply_activation_memory_budget(
        activation_memory_budget, grad_ckpt=grad_ckpt, logger=logger
    )

    isolate_compile_cache(
        compile_signature(
            n_token_families=n_token_families,
            seq_range=seq_range,
            dynamic_seq=dynamic_seq,
            backend=backend,
            mode=mode,
        )
    )
    compile_dit_blocks(
        anima,
        enabled=True,
        cache_size_limit=(
            2 * n_shapes + 8 if cache_size_limit is None else cache_size_limit
        ),
        backend=backend,
        mode=mode,
        dynamic_seq=dynamic_seq,
        n_token_families=n_token_families,
        seq_range=seq_range,
    )
    return result


def enable_training_grad_ckpt(anima: object, *, enabled: bool) -> None:
    """Toggle unsloth CPU-offload gradient checkpointing for a training run.

    The model must stay in ``train()`` mode: ``Block.forward`` gates
    checkpointing on ``self.training``.
    """
    if enabled:
        anima.enable_gradient_checkpointing(unsloth_offload=True)
        log.info("gradient checkpointing: on (unsloth CPU offload)")
    else:
        log.info("gradient checkpointing: off")
