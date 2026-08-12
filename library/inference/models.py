"""Model loading for Anima inference: DiT, text encoder, shared model management."""

import argparse
import logging
from typing import Optional, Dict

import torch
from safetensors.torch import load_file

from library.anima import models as anima_models, weights as anima_utils
from library.runtime.device import clean_memory_on_device

logger = logging.getLogger(__name__)


def _is_hydra_moe(path: str) -> bool:
    """Cheap check: peek at the safetensors header for a per-expert ups key.

    HydraLoRA moe files carry per-expert ``.lora_ups.N.weight`` keys;
    chimera files carry the dual-pool variants ``.lora_ups_c.N.weight`` /
    ``.lora_ups_f.N.weight``. Either signal means router-live: skip static
    merge. Regular LoRA files have none of these. Uses ``safe_open`` so
    only the header is read.

    Callers that need to disambiguate chimera from plain Hydra should also
    consult ``_is_chimera_moe``.
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            return any(
                ".lora_ups." in k or ".lora_ups_c." in k or ".lora_ups_f." in k
                for k in f.keys()
            )
    except Exception:
        return False


def _has_register_tokens(path: str) -> bool:
    """Cheap header peek: does this LoRA carry a ``register_tokens`` key?

    LoRA-family checkpoints trained with ``num_registers > 0`` hold K DSR
    register tokens that ride the self-attn sequence — they can't fold into a
    static weight merge, so the network must stay live (same treatment as
    P-GRAFT: ``create_network_from_weights`` + ``apply_to`` dynamic hooks; the
    register injection installs in ``LoRANetwork.apply_to``).
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            return "register_tokens" in f.keys()
    except Exception:
        return False


def _has_te_keys(path: str) -> bool:
    """Cheap header peek: does this safetensors LoRA carry any ``lora_te_*`` keys?

    Lets ``load_text_encoder`` skip a redundant ``load_file`` + empty-dict merge
    when the LoRA is DiT-only (the common case — turbo, plain LoRA, postfix, …).
    Returns False on any read error so the caller falls back to the no-LoRA TE
    load path (a truly broken file would have already tripped the DiT loader).
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            return any(k.startswith("lora_te_") for k in f.keys())
    except Exception:
        return False


def _is_chimera_moe(path: str) -> bool:
    """Peek at safetensors metadata for ``ss_use_chimera_hydra="true"``.

    Chimera files share the Hydra-MoE on-disk shape (so ``_is_hydra_moe``
    also returns True) but carry the dual-pool runtime contract — they
    additionally hold a top-level ``freq_router.*`` block and need the
    per-Linear router narrowed to K_c outputs. Inference / load paths
    must read this flag to wire the network correctly.
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            md = f.metadata() or {}
            return str(md.get("ss_use_chimera_hydra", "")).strip().lower() == "true"
    except Exception:
        return False


def _is_step_expert_turbo(path: str) -> bool:
    """Peek at safetensors metadata for ``ss_turbo_per_step_expert``.

    Per-step-expert turbo files carry ``.lora_ups.{k}.weight`` keys (so
    ``_is_hydra_moe`` ALSO returns True), but they are not Hydra — head
    selection is by denoise-step counter, not a router. This metadata stamp is
    the authoritative discriminator and must be checked before ``hydra_mode``.
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            md = f.metadata() or {}
            return str(md.get("ss_turbo_per_step_expert", "")).strip() in (
                "1",
                "true",
                "True",
            )
    except Exception:
        return False


def attach_adapters(
    model: anima_models.Anima,
    args: argparse.Namespace,
    device: torch.device,
    *,
    pgraft_mode: bool,
    hydra_mode: bool,
    step_expert_mode: bool = False,
    register_mode: bool = False,
) -> None:
    """Attach LoRA-family adapters that ride as dynamic forward hooks.

    Covers the two routes that can't go through ``load_anima_model``'s static
    merge: **P-GRAFT** (toggleable mid-denoising) and **HydraLoRA moe / chimera**
    (router-live, runs per-sample). Both rehydrate a network, ``apply_to`` the
    already-loaded ``model`` in place, and stash it on ``model`` for the sampler
    toggle sites to find. No-op when neither mode is set. Mutates ``model``;
    returns nothing. The static-merge path and ``torch.compile`` stay in
    :func:`load_dit_model` — this does only the dynamic-hook attach.

    ``pgraft_mode`` / ``hydra_mode`` are passed in (not recomputed) because the
    caller already derives them to decide whether to skip the static merge.
    """
    # P-GRAFT / register tokens: attach LoRA as dynamic hooks. P-GRAFT wants
    # the toggle; register-token checkpoints CAN'T merge (K sequence-riding
    # tokens), so both skip the static merge and keep the network live.
    if (pgraft_mode or register_mode) and not hydra_mode:
        from safetensors import safe_open
        from networks import lora_anima

        logger.info(
            "%s: Loading LoRA as dynamic hooks (not static merge)",
            "P-GRAFT" if pgraft_mode else "register tokens",
        )
        for lora_weight_path in args.lora_weight:
            # Metadata carries ss_register_insert_block (and the three-axis
            # stamps) — load_file() drops __metadata__, so read it separately.
            with safe_open(lora_weight_path, framework="pt") as f:
                lora_metadata = dict(f.metadata() or {})
            lora_sd = load_file(lora_weight_path)
            lora_sd = {
                k: v
                for k, v in lora_sd.items()
                if k.startswith("lora_unet_") or k == "register_tokens"
            }

            multiplier = (
                args.lora_multiplier
                if isinstance(args.lora_multiplier, (int, float))
                else args.lora_multiplier[0]
            )
            network, weights_sd = lora_anima.create_network_from_weights(
                multiplier=multiplier,
                file=None,
                ae=None,
                text_encoders=[],
                unet=model,
                weights_sd=lora_sd,
                metadata=lora_metadata,
                for_inference=True,
            )
            network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
            info = network.load_state_dict(weights_sd, strict=False)
            if info.unexpected_keys:
                logger.debug(
                    f"P-GRAFT: unexpected keys in LoRA state dict: {info.unexpected_keys[:5]}..."
                )
            network.to(device, dtype=torch.bfloat16)
            network.eval()
            model._pgraft_network = network
            logger.info(
                f"P-GRAFT: LoRA attached with cutoff_step={getattr(args, 'lora_cutoff_step', None)}"
            )

    # HydraLoRA moe: rehydrate the trained router-live network and attach it
    # as dynamic forward hooks, identical shape to the P-GRAFT path above.
    # The router runs per-sample on each adapted module, so the net stays in
    # eval mode with requires_grad_(False).
    if hydra_mode:
        from networks import lora_anima

        logger.info("HydraLoRA: loading moe file as router-live dynamic hooks")
        from safetensors import safe_open

        for lora_weight_path in args.lora_weight:
            # Read the three-axis routing stamps (and chimera stamps) from
            # on-disk __metadata__ — load_file() drops it. Chimera files
            # (dual-pool) carry top-level ``freq_router.*`` keys outside the
            # ``lora_unet_*`` namespace, so they must NOT be filtered; plain
            # Hydra moe keeps the lora_unet_* filter. Passing ``metadata=``
            # alongside ``weights_sd=`` lets both layouts go through one code
            # path — no more file=path vs weights_sd= fork.
            with safe_open(lora_weight_path, framework="pt") as f:
                lora_metadata = dict(f.metadata() or {})
            is_chimera = _is_chimera_moe(lora_weight_path)
            lora_sd = load_file(lora_weight_path)
            if is_chimera:
                logger.info("HydraLoRA: chimera file — dual-pool routing wired")
            else:
                lora_sd = {
                    k: v
                    for k, v in lora_sd.items()
                    if k.startswith("lora_unet_") or k == "register_tokens"
                }

            multiplier = (
                args.lora_multiplier
                if isinstance(args.lora_multiplier, (int, float))
                else args.lora_multiplier[0]
            )
            network, weights_sd = lora_anima.create_network_from_weights(
                multiplier=multiplier,
                file=None,
                ae=None,
                text_encoders=[],
                unet=model,
                weights_sd=lora_sd,
                metadata=lora_metadata,
                for_inference=True,
            )
            network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
            info = network.load_state_dict(weights_sd, strict=False)
            if info.unexpected_keys:
                logger.warning(
                    f"HydraLoRA: unexpected keys in state dict: {info.unexpected_keys[:5]}..."
                )
            if info.missing_keys:
                logger.warning(
                    f"HydraLoRA: missing keys in state dict: {info.missing_keys[:5]}..."
                )
            network.to(device, dtype=torch.bfloat16)
            network.eval().requires_grad_(False)
            hydra_networks = list(getattr(model, "_hydra_networks", []))
            hydra_networks.append(network)
            model._hydra_networks = hydra_networks
            model._hydra_network = network
            # Reuse the P-GRAFT cutoff slot so existing toggle sites
            # (inference_pipeline loops + spectrum_denoise) honor
            # --lora_cutoff_step without further plumbing.
            model._pgraft_network = network
            logger.info(
                f"HydraLoRA: router-live attached "
                f"({len(network.unet_loras)} modules, "
                f"cutoff_step={getattr(args, 'lora_cutoff_step', None)})"
            )

    # Per-step-expert turbo: kept-live (K up-heads can't merge), head selected
    # by the denoise step counter in the sampler loop. Like Hydra it can't ride
    # the static merge; unlike Hydra there is no router — generation.py calls
    # network.set_step_index(i) once per step.
    if step_expert_mode:
        from safetensors import safe_open
        from networks.methods.turbo_dmd import load_step_expert_student

        logger.info("step-expert turbo: loading as router-free kept-live hooks")
        for lora_weight_path in args.lora_weight:
            with safe_open(lora_weight_path, framework="pt") as f:
                se_metadata = dict(f.metadata() or {})
            lora_sd = load_file(lora_weight_path)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
            multiplier = (
                args.lora_multiplier
                if isinstance(args.lora_multiplier, (int, float))
                else args.lora_multiplier[0]
            )
            network = load_step_expert_student(
                model, lora_sd, se_metadata, multiplier=multiplier
            )
            network.to(device, dtype=torch.bfloat16)
            network.eval().requires_grad_(False)
            # Dedicated slot so generation.py finds it for set_step_index, kept
            # separate from the Hydra/P-GRAFT slots (no router, no cutoff reuse).
            step_nets = list(getattr(model, "_step_expert_networks", []))
            step_nets.append(network)
            model._step_expert_networks = step_nets


def _attach_res_cond(
    model: anima_models.Anima,
    lora_weight_paths: list[str],
    device: torch.device,
    s: float = 0.0,
) -> None:
    """E25b res-cond: install the trained resolution-conditioning projection.

    A rescond checkpoint carries ``sigma_lowres_res_cond_proj`` + the
    ``ss_sigma_lowres_res_cond`` metadata stamp. Native training steps ran the
    module at s = 0, so the trained W·φ(0) offset is part of the function the
    run converged to — rendering without it is off-distribution. Inference is
    always on the native grid, so the attach is a persistent ``(proj, s=0)``
    on the DiT (the forward reads it per call in the eager region; the
    trainer's try/finally scoping is unnecessary because s never changes).
    ``s`` defaults to 0 (the trained native point); a nonzero value is the
    E25c inference knob (``--res_cond_s``) and requires a carrier checkpoint
    — a knob that silently does nothing is the failure mode this helper
    exists to close. Stamp without tensor ⇒ hard fail (a stripped/re-saved file — the
    silent-drop failure mode this helper closes). The projection is kept fp32
    on device (~2 MB); the forward casts the delta to the t-embedding dtype.
    Unscaled by ``lora_multiplier`` — the projection is a conditioning input,
    not a ΔW.
    """
    from safetensors import safe_open

    carriers: list[tuple[str, torch.Tensor]] = []
    for path in lora_weight_paths:
        with safe_open(path, framework="pt") as f:
            has_key = "sigma_lowres_res_cond_proj" in f.keys()
            stamped = dict(f.metadata() or {}).get("ss_sigma_lowres_res_cond") == "true"
            if stamped and not has_key:
                raise RuntimeError(
                    f"{path}: ss_sigma_lowres_res_cond stamp without the "
                    "sigma_lowres_res_cond_proj tensor — the projection was "
                    "stripped; refusing to render off-distribution."
                )
            if has_key:
                carriers.append((path, f.get_tensor("sigma_lowres_res_cond_proj")))
    if not carriers:
        if s != 0.0:
            raise RuntimeError(
                f"--res_cond_s {s} needs a --lora_weight carrying "
                "sigma_lowres_res_cond_proj; none of the loaded files does."
            )
        return
    if len(carriers) > 1:
        raise RuntimeError(
            "Multiple --lora_weight files carry sigma_lowres_res_cond_proj — "
            "composing res-cond adapters is untested; load them separately."
        )
    path, proj = carriers[0]
    model._sigma_lowres_res_cond = (
        proj.to(device=device, dtype=torch.float32),
        float(s),
    )
    logger.info(f"E25b res-cond: projection installed from {path} (inference s={s})")


def load_dit_model(
    args: argparse.Namespace,
    device: torch.device,
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> anima_models.Anima:
    """Load DiT model with optional LoRA merge, P-GRAFT hooks, and torch.compile.

    Namespace-driven adapter over the explicit-argument primitive
    ``library.anima.weights.load_anima_model``: it pulls ``dit``/``attn_mode``/
    ``lora_weight``/etc. off ``args``, then hands the dynamic-hook adapter attach
    to :func:`attach_adapters` and applies ``torch.compile``. Reach for
    ``load_anima_model`` directly when you want just the weights and no Namespace.
    """

    loading_device = device

    # HydraLoRA moe (incl. FeRA-style stacked-experts global FEI): router-live
    # inference can't go through static merge. Detect early so we can skip the
    # baked-down path and take the dynamic hook route regardless of whether
    # --pgraft is set. ``_is_hydra_moe`` matches the ``lora_ups.{i}.weight``
    # key pattern shared by both shared-A Hydra and the plan2 stacked-experts
    # save format.
    # Per-step-expert turbo is detected FIRST: its files also match
    # ``_is_hydra_moe`` (shared ``.lora_ups.{k}.weight`` key shape) but are
    # router-free and head-selected by step counter, so the metadata stamp wins.
    step_expert_mode = False
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        se_flags = [_is_step_expert_turbo(p) for p in args.lora_weight]
        if any(se_flags):
            if not all(se_flags) or len(args.lora_weight) > 1:
                raise ValueError(
                    "Per-step-expert turbo must be loaded alone (one "
                    "--lora_weight). Its K up-heads are kept-live and head "
                    "selection is by denoise step — composing it with other "
                    "LoRAs / merge is unsupported."
                )
            step_expert_mode = True

    hydra_mode = False
    if (
        not step_expert_mode
        and args.lora_weight is not None
        and len(args.lora_weight) > 0
    ):
        hydra_flags = [_is_hydra_moe(p) for p in args.lora_weight]
        if any(hydra_flags):
            if not all(hydra_flags):
                raise ValueError(
                    "Mixing HydraLoRA moe files with regular LoRA files in a "
                    "single --lora_weight list is not supported. The static "
                    "merge + dynamic hook interaction is untested. Pass them "
                    "in separate invocations."
                )
            hydra_mode = True

    # P-GRAFT: load without LoRA merge, attach dynamic hooks instead
    pgraft_mode = (
        getattr(args, "pgraft", False)
        and args.lora_weight is not None
        and len(args.lora_weight) > 0
    )

    # Register tokens (LoRA trained with num_registers > 0): the K
    # sequence-riding tokens can't fold into a static merge — force the
    # dynamic-hook route (LoRANetwork.apply_to installs the injection).
    register_mode = False
    if (
        not step_expert_mode
        and args.lora_weight is not None
        and len(args.lora_weight) > 0
    ):
        register_flags = [_has_register_tokens(p) for p in args.lora_weight]
        if any(register_flags):
            if len(args.lora_weight) > 1:
                raise ValueError(
                    "A register-token LoRA must be loaded alone (one "
                    "--lora_weight). Its registers are kept-live sequence "
                    "tokens — composing with statically-merged LoRAs in one "
                    "invocation is untested."
                )
            register_mode = True

    # load LoRA weights (skip static merge for P-GRAFT, HydraLoRA moe,
    # register tokens, and per-step-expert turbo — all ride dynamic hooks)
    if (
        not pgraft_mode
        and not hydra_mode
        and not step_expert_mode
        and not register_mode
        and args.lora_weight is not None
        and len(args.lora_weight) > 0
    ):
        lora_weights_list = []
        for lora_weight in args.lora_weight:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            lora_sd = {
                k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")
            }  # only keep unet lora weights
            lora_weights_list.append(lora_sd)
    else:
        lora_weights_list = None

    model = anima_utils.load_anima_model(
        device,
        args.dit,
        args.attn_mode,
        loading_device,
        dit_weight_dtype,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.lora_multiplier,
    )

    # Modulation guidance: load trained pooled_text_proj weights before .to()
    # (pooled_text_proj params are meta tensors when not in the pretrained checkpoint)
    pooled_text_proj_path = getattr(args, "pooled_text_proj", None)
    if pooled_text_proj_path is not None:
        anima_utils.load_pooled_text_proj(model, pooled_text_proj_path, "cpu")

    target_dtype = dit_weight_dtype
    if target_dtype is not None:
        logger.info(f"Convert model to {target_dtype}")
    logger.info(f"Move model to device: {device}")
    model.to(device, target_dtype)

    model.to(device, dtype=torch.bfloat16)  # ensure model is in bfloat16 for inference

    model.eval().requires_grad_(False)

    # Dynamic-hook adapters (P-GRAFT toggle / HydraLoRA router-live) that can't
    # ride the static merge above.
    attach_adapters(
        model,
        args,
        device,
        pgraft_mode=pgraft_mode,
        hydra_mode=hydra_mode,
        step_expert_mode=step_expert_mode,
        register_mode=register_mode,
    )

    # E25b res-cond (Stage 2.0): every adapter route above filters the state
    # dict to lora_unet_*/register_tokens, so the (non-bakeable) res-cond
    # projection never reaches the network — install it on the DiT here.
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        _attach_res_cond(
            model,
            list(args.lora_weight),
            device,
            s=getattr(args, "res_cond_s", 0.0),
        )
    elif getattr(args, "res_cond_s", 0.0) != 0.0:
        raise RuntimeError("--res_cond_s needs a rescond --lora_weight; none given.")

    if getattr(args, "compile", False):
        logger.info("Compiling DiT model with torch.compile...")
        model = torch.compile(model)
    elif getattr(args, "compile_blocks", False):
        model.compile_blocks(mode=getattr(args, "compile_inductor_mode", None))

    clean_memory_on_device(device)

    return model


def load_text_encoder(
    args: argparse.Namespace | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cpu"),
    *,
    text_encoder: str | None = None,
    lora_weight: list[str] | None = None,
    lora_multiplier: float | list[float] | None = None,
) -> torch.nn.Module:
    """Load the Qwen3 text encoder (optionally folding in TE-side LoRA).

    Only three fields matter here: the encoder path and the LoRA weight/multiplier
    list. Pass them as keywords for a self-documenting call::

        load_text_encoder(text_encoder=TEXT_ENCODER, device=device)

    The legacy ``args`` namespace is still accepted as a fallback (the CLI and
    ``prepare_text_inputs`` pass one); explicit keywords win over it when both are
    given. Don't reach for ``inference.parse_args`` just to feed this — that drags
    in unrelated required flags (``--save_path``) and reads nothing else here.
    """
    # Explicit keyword wins; otherwise fall back to the namespace; otherwise default.
    te_path = (
        text_encoder
        if text_encoder is not None
        else getattr(args, "text_encoder", None)
    )
    if te_path is None:
        raise ValueError(
            "load_text_encoder needs a text_encoder path "
            "(pass text_encoder=... or an args namespace with .text_encoder)"
        )
    lw = lora_weight if lora_weight is not None else getattr(args, "lora_weight", None)
    lm = (
        lora_multiplier
        if lora_multiplier is not None
        else getattr(args, "lora_multiplier", 1.0)
    )

    lora_weights_list = None
    if lw is not None and len(lw) > 0 and any(_has_te_keys(p) for p in lw):
        lora_weights_list = []
        for lora_weight_path in lw:
            logger.info(f"Loading LoRA weight from: {lora_weight_path}")
            lora_sd = load_file(lora_weight_path)  # load on CPU, dtype is as is
            lora_sd = {
                "model_" + k[len("lora_te_") :]: v
                for k, v in lora_sd.items()
                if k.startswith("lora_te_")
            }  # only keep Text Encoder lora weights, remove prefix "lora_te_" and add "model_" prefix
            lora_weights_list.append(lora_sd)

    lora_multipliers = lm
    if lora_multipliers is not None and not isinstance(lora_multipliers, list):
        lora_multipliers = [lora_multipliers]
    text_encoder, _ = anima_utils.load_qwen3_text_encoder(
        te_path,
        dtype=dtype,
        device=device,
        lora_weights=lora_weights_list,
        lora_multipliers=lora_multipliers,
    )
    text_encoder.eval()
    return text_encoder


def load_shared_models(args: argparse.Namespace) -> Dict:
    """Load shared models for batch processing or interactive mode.
    Models are loaded to CPU to save memory. VAE is NOT loaded here.
    DiT model is also NOT loaded here, handled by process_batch_prompts or generate.
    """
    shared_models = {}
    text_encoder_dtype = torch.bfloat16
    text_encoder = load_text_encoder(
        args, dtype=text_encoder_dtype, device=torch.device("cpu")
    )
    shared_models["text_encoder"] = text_encoder
    return shared_models
