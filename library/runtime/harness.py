"""Shared DiT + adapter run harness.

Loads the DiT, attaches an optional adapter, and applies ``torch.compile`` in
the one order that's required: ``torch.compile`` traces the adapter's
monkey-patched forward, so ``compile_blocks`` MUST run after
``network.apply_to`` + ``load_weights`` — otherwise it silently compiles the
wrong (unpatched) forward. Shared by ``bench`` / ``scripts`` / ``preprocess``.

Usage::

    from library.runtime.harness import build_anima

    bundle = build_anima(args, dit_path=..., adapter=..., train_mode=False)
    anima, network = bundle.anima, bundle.network

Reads its knobs off an argparse ``Namespace`` (``device`` / ``dtype`` /
``attn_mode`` / ``gradient_checkpointing`` / ``compile`` / ``compile_mode``);
see ``library.runtime.argparse_groups.add_device_args``. Callers without a
parser can pass a plain ``argparse.Namespace(**kwargs)``.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

log = logging.getLogger("library.runtime.harness")


@dataclass
class AnimaBundle:
    """Container for a built Anima model + optional adapter.

    Returned by ``build_anima``. ``network`` is ``None`` when no adapter
    was requested.
    """

    anima: object  # library.anima Anima — typed as object to avoid heavy import
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

    Sequence: load DiT → freeze + ``reset_mod_guidance`` → (if ``adapter``)
    ``create_network_from_weights`` → ``apply_to`` → ``load_weights`` →
    freeze/unfreeze per ``train_mode`` → gradient checkpointing → train()/
    eval() → **``compile_blocks`` last** (compile-after-apply, see module
    docstring).

    Arguments:
        args: reads ``device``, ``dtype``, ``attn_mode``,
            ``gradient_checkpointing``, ``compile``, ``compile_mode``.
        dit_path: defaults to ``args.dit`` if the caller's argparse exposed one.
        adapter: optional adapter safetensors path, applied with
            ``multiplier`` as the apply-time scale.
        train_mode: puts anima + network in train mode; required before any
            ``backward()`` call (LoRA training-path forward, T-LoRA mask, and
            gradient checkpointing are all gated on ``self.training``).
        network_requires_grad: under ``train_mode=True``, freeze/unfreeze
            adapter params.
        multiplier: adapter forward-time scale; ``set_multiplier(0.0)``
            later recovers the base-model output.

    Returns:
        ``AnimaBundle(anima, network, device, dtype)``.
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
        anima.requires_grad_(False)  # always — DiT stays frozen in the harness

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

    # COMPILE LAST. Adapter monkey-patches must be installed first or
    # torch.compile traces the wrong forward.
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

    Where :class:`AnimaBundle` stops at *DiT + adapter*, this carries the full
    inference set — text encoder, VAE, resolved ``GenerationSettings`` — plus
    ``shared_models`` primed with ``model`` / ``text_encoder`` / ``conds_cache``.
    Because the DiT is pre-loaded into ``shared_models["model"]``, ``generate()``
    reuses *this* instance — so a forward-hook installed on ``bundle.model``
    before generating stays live during sampling. ``vae`` is ``None`` when built
    with ``with_vae=False`` or no ``--vae`` path.
    """

    model: object  # the loaded DiT (also stashed in shared_models["model"])
    vae: Optional[object]  # AutoencoderKLQwenImage, or None
    text_encoder: object  # on CPU; generate()/prepare_text_inputs moves it
    gen_settings: object  # library.inference.GenerationSettings
    shared_models: dict  # {"model", "text_encoder", "conds_cache"} -> generate()
    args: argparse.Namespace  # the namespace the bundle was built from
    device: torch.device

    def generate(self, args: Optional[argparse.Namespace] = None):
        """Run ``library.inference.generate`` reusing this bundle's loaded models.

        Defaults to the namespace the bundle was built from; pass a per-call
        ``args`` to override. Returns the latent tensor ``generate()`` produces.
        """
        from library.inference import generate as _generate

        return _generate(args or self.args, self.gen_settings, self.shared_models)


def build_inference_bundle(
    args: argparse.Namespace,
    device: torch.device | str | None = None,
    *,
    with_vae: bool = True,
) -> InferenceBundle:
    """Assemble the text-encoder + DiT (+ optional VAE) set for a generation.

    The inference-side counterpart to :func:`build_anima` — bundles the
    sequence ``inference.main()`` open-codes (load text encoder, load DiT,
    stash it in ``shared_models["model"]`` so ``generate()`` reuses the
    instance, load the VAE) for bench/probe callers.

    Args:
        args: a fully-defaulted namespace (``inference.parse_args`` /
            ``GenerationRequest.to_args()`` / ``build_default_args``). Reads
            ``vae`` / ``text_encoder`` / ``dit`` / adapter + sampler knobs.
        device: optional explicit device, written back to ``args.device`` so
            every downstream loader agrees. ``None`` resolves cuda-else-cpu.
        with_vae: load the VAE (needed only to decode latents → pixels). A
            latent-space probe can pass ``False`` to skip the load.

    Returns:
        :class:`InferenceBundle` — pass ``.shared_models`` to ``generate()`` or
        call ``.generate()``.
    """
    # keep this module import-cheap on CPU-only runs
    from library.inference import (
        get_generation_settings,
        load_dit_model,
        load_shared_models,
    )

    if device is not None:
        # Pin it on the namespace so get_generation_settings + load_dit_model all
        # resolve to the same device (mirrors inference.main()).
        args.device = str(device) if not isinstance(device, str) else device

    gen_settings = get_generation_settings(args)
    resolved_device = gen_settings.device

    shared_models = load_shared_models(args)  # text encoder on CPU
    shared_models["conds_cache"] = {}

    anima = load_dit_model(args, resolved_device, torch.bfloat16)
    # Stash so generate() reuses *this* instance — the only seam for hooking the
    # DiT before the sampler loop runs.
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

    With block swap on, swapped blocks stay on CPU and ride the
    forward+backward swap hooks while everything else moves to ``device``;
    without it the whole model moves. Call before ``compile_dit_blocks`` /
    ``train()``.
    """
    if blocks_to_swap > 0:
        anima.enable_block_swap(blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.switch_block_swap_for_training()  # forward+backward block movement
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

    Native-shape flattening traces one block graph per distinct token count;
    pre-raises the dynamo cache to ``cache_size_limit`` so each shape traces
    instead of falling back to eager mid-warmup. No-op when ``enabled`` is
    False.

    ``dynamic_seq`` collapses the per-token-count graphs to one graph by
    marking only the seq-length axis dynamic; ``seq_range`` bounds that
    symbolic axis and ``n_token_families`` sizes the dynamo cache budget.

    COMPILE LAST — after the adapter's ``apply_to`` (see module docstring).
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


def compile_signature(
    *,
    n_token_families: Optional[int],
    seq_range: Optional[tuple],
    dynamic_seq: bool,
    backend: str = "inductor",
    mode: Optional[str] = None,
    seq_bands: Optional[list] = None,
) -> str:
    """Canonical signature string for ``maybe_clear_stale_compile_cache``.

    Every compile entry point must build the marker through this one
    formatter so equivalent configs serialize identically. ``mode`` is
    normalized so ``None`` and ``""`` ("inductor default") don't read as a
    signature change. ``seq_bands`` (per-band dynamic-seq) is appended only
    when set, so pre-existing union-range signatures — and their persistent
    cache dirs — stay byte-identical.
    """
    sig = (
        f"families={n_token_families};seq_range={seq_range};"
        f"dynamic_seq={dynamic_seq};backend={backend};mode={mode or None}"
    )
    if seq_bands:
        sig += f";seq_bands={sorted(tuple(b) for b in seq_bands)}"
    return sig


# captured on the first isolate_compile_cache call so later calls in the same
# process re-derive from the same root instead of nesting per-signature dirs
_compile_cache_base: Optional[str] = None


def isolate_compile_cache(signature: str) -> str:
    """Route this run's torch.compile caches to a per-signature directory.

    GOTCHA: the persistent compile caches key on the FX graph but NOT on the
    ``mark_dynamic`` value range, so processes compiled with different
    seq-range bounds poison each other through the shared default cache dir —
    e.g. an inference run's narrower stored guard gets replayed into a wider
    training-run mark and raises ``ConstraintViolationError`` instead of
    cleanly missing (hint-dependent, so it strikes "sometimes"). Wiping the
    shared dir doesn't fix it since inference keeps re-depositing; instead
    this points ``TORCHINDUCTOR_CACHE_DIR`` at a per-signature subdir so
    every entry inside was compiled under the same seq bounds.

    Must run BEFORE the first ``torch.compile`` trace in the process (torch
    reads the env var lazily per cache access). Build ``signature`` via
    ``compile_signature``. Returns the directory used.
    """
    global _compile_cache_base
    import hashlib
    import os

    if _compile_cache_base is None:
        base = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        if not base:
            try:
                from torch._inductor.runtime.cache_dir_utils import default_cache_dir

                base = default_cache_dir()
            except Exception:  # noqa: BLE001 — torch internals move across versions
                import getpass
                import tempfile

                base = os.path.join(
                    tempfile.gettempdir(), f"torchinductor_{getpass.getuser()}"
                )
        _compile_cache_base = base

    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:16]
    target = os.path.join(_compile_cache_base, f"anima-sig-{digest}")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = target
    log.info(f"torch.compile cache isolated per compile signature: {target}")
    log.info(f"compile signature: {signature}")
    return target


def _apply_activation_memory_budget(
    budget: float, *, grad_ckpt: bool, logger: logging.Logger = log
) -> None:
    """Cap the AOT min-cut partitioner's saved-for-backward set.

    ``budget < 1.0`` makes the partitioner recompute cheap intermediates
    instead of saving them (0.85 was measured to fix a first-step OOM at
    16GB/4200 tokens without grad-ckpt, at negligible step-time cost).

    Must be set BEFORE the block compile — partitioning happens at
    first-forward compile, and this is a plain module attr (no ContextVar
    revert, unlike dynamo's recompile_limit). GOTCHA: skipped under gradient
    checkpointing — repartitioning the joint graph lets checkpoint's recompute
    pass select a different graph than forward, raising ``CheckpointError``
    (torch #166926).
    """
    if budget < 1.0 and not grad_ckpt:
        import torch._functorch.config as _functorch_config

        _functorch_config.activation_memory_budget = budget
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

    Unlike ``activation_memory_budget`` (a cap applied on top of the default
    partition), these flags change what the default partition is willing to
    recompute at all: ``recompute_views`` stops saving view ops,
    ``aggressive_recomputation`` drops the ban list on recomputing whole op
    classes when the min-cut is cheap. See
    docs/findings/custom_autograd_removal_partitioner_oom.md.

    Skipped under gradient checkpointing for the same reason as the budget
    (see :func:`_apply_activation_memory_budget`).
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

    Under ``mode='reduce-overhead'`` inductor records a fresh CUDAGraph per
    distinct input size; free-fit's native-shape bucketing populates many
    distinct seq lengths per tier, so each pins its own static I/O buffers
    and fights the partitioner's OOM budget. ``ANIMA_CUDAGRAPH_SKIP_DYNAMIC=1``
    keeps CUDAGraphs only for fully-static graphs. No-op unless cudagraphs are
    actually active (reduce-overhead / max-autotune).
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
    activation_memory_budget: float = 1.0,
    partitioner_recompute_views: bool = False,
    partitioner_aggressive_recomputation: bool = False,
    grad_ckpt: bool = False,
    logger: logging.Logger = log,
) -> None:
    """The LoRA-training (``train.py``) compile sequence, post ``apply_to``.

    Native-shape flattening + per-block torch.compile. COMPILE LAST — run only
    after ``network.apply_to`` + ``load_weights``.

    Sequence: partitioner budget/tuning (skipped under grad-ckpt) →
    per-signature compile cache isolation → ``unet.compile_blocks(...)`` with
    the caller-derived token budget (``train.py::_derive_token_budget``) →
    ``network.compile_cond_stream(...)`` when the adapter exposes it. GOTCHA:
    EasyControl's patched ``Block.forward`` routes its cond path through
    ``_two_stream_inner``, bypassing the just-compiled ``block._forward`` —
    so the cond stream (incl. cond LoRA projections) needs this separate
    compile call or it silently stays eager.
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
    )
    unet.compile_blocks(
        backend,
        mode=mode,
        n_token_families=n_token_families,
        dynamic_seq=dynamic_seq,
        seq_range=seq_range,
        seq_bands=seq_bands,
    )
    # NB: compile_cond_stream (EasyControl) stays on the union range —
    # the cond stream runs at the same seq as the target stream, but its
    # compile surface is separate.
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

    The compile-setup sequence ``distill_turbo`` / ``distill_mod`` share. The
    caller derives ``token_counts`` (the distinct ``(W//patch)*(H//patch)``
    counts its cached pool populates); this owns everything after: derive
    ``n_shapes`` and, under ``dynamic_seq``, the symbolic-seq
    ``n_token_families`` / ``seq_range`` → partitioner budget (skipped under
    ``grad_ckpt``, see :func:`_apply_activation_memory_budget`) →
    :func:`isolate_compile_cache` → ``compile_dit_blocks`` with the dynamo
    cache sized to ``2*n_shapes + 8`` (override via ``cache_size_limit``).

    Returns the derived :class:`PoolCompileResult` even when ``enabled`` is
    False, so callers can still log it. Run AFTER the network ``apply_to``
    (compile-after-apply invariant).
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
    # compile-after-apply invariant
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

    Recomputes block activations in backward, offloading saved tensors to CPU
    between forward/backward. The model must stay in ``train()`` mode —
    ``Block.forward`` gates checkpointing on ``self.training``. Logs and no-ops
    when ``enabled`` is False.
    """
    if enabled:
        anima.enable_gradient_checkpointing(unsloth_offload=True)
        log.info("gradient checkpointing: on (unsloth CPU offload)")
    else:
        log.info("gradient checkpointing: off")
