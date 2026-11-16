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

    Per-expert ``.lora_ups.N.weight`` keys mean router-live: skip the static
    merge. The removed ChimeraHydra layout (``.lora_ups_c.`` / ``.lora_ups_f.``)
    is matched too so it reaches ``create_network_from_weights``' refusal
    instead of a silent partial merge.
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


def _has_te_keys(path: str) -> bool:
    """Cheap header peek: does this safetensors LoRA carry any ``lora_te_*`` keys?

    Lets ``load_text_encoder`` skip the load for DiT-only LoRAs. False on a
    read error (a broken file has already tripped the DiT loader).
    """
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt") as f:
            return any(k.startswith("lora_te_") for k in f.keys())
    except Exception:
        return False


def _is_step_expert_turbo(path: str) -> bool:
    """Peek at safetensors metadata for ``ss_turbo_per_step_expert``.

    These files also match ``_is_hydra_moe`` (same ``.lora_ups.{k}.weight``
    keys) but select heads by denoise step, not a router, so this stamp must be
    checked before ``hydra_mode``.
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
) -> None:
    """Attach LoRA-family adapters that ride as dynamic forward hooks.

    For the routes that can't take ``load_anima_model``'s static merge:
    P-GRAFT (toggleable mid-denoising), HydraLoRA moe (router-live) and
    per-step-expert turbo. Each network is ``apply_to``'d onto ``model`` in
    place and stashed on it for the sampler sites to find.
    """
    if pgraft_mode and not hydra_mode:
        from safetensors import safe_open
        from networks import lora_anima

        logger.info("P-GRAFT: Loading LoRA as dynamic hooks (not static merge)")
        for lora_weight_path in args.lora_weight:
            # load_file() drops __metadata__ (the routing stamps)
            with safe_open(lora_weight_path, framework="pt") as f:
                lora_metadata = dict(f.metadata() or {})
            lora_sd = load_file(lora_weight_path)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}

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
            if getattr(args, "lora_timestep_mask", False):
                import dataclasses

                network.cfg = dataclasses.replace(
                    network.cfg,
                    use_timestep_mask=True,
                    min_rank=args.lora_mask_min_rank,
                    alpha_rank_scale=args.lora_mask_alpha,
                )
                for lora in network.unet_loras:
                    lora._eval_mask = True
            model._pgraft_network = network
            model._pgraft_networks = [*getattr(model, "_pgraft_networks", []), network]
            logger.info(
                f"P-GRAFT: LoRA attached with cutoff_step={getattr(args, 'lora_cutoff_step', None)}"
            )

    if hydra_mode:
        from networks import lora_anima

        logger.info("HydraLoRA: loading moe file as router-live dynamic hooks")
        from safetensors import safe_open

        for lora_weight_path in args.lora_weight:
            # load_file() drops __metadata__ (the routing stamps)
            with safe_open(lora_weight_path, framework="pt") as f:
                lora_metadata = dict(f.metadata() or {})
            lora_sd = load_file(lora_weight_path)
            lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}

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
            # The P-GRAFT slot makes the toggle sites honor --lora_cutoff_step.
            model._pgraft_network = network
            logger.info(
                f"HydraLoRA: router-live attached "
                f"({len(network.unet_loras)} modules, "
                f"cutoff_step={getattr(args, 'lora_cutoff_step', None)})"
            )

    # Per-step-expert turbo: K up-heads can't merge; generation.py selects the
    # head once per step.
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
            # own slot: no router, no cutoff reuse
            step_nets = list(getattr(model, "_step_expert_networks", []))
            step_nets.append(network)
            model._step_expert_networks = step_nets


_ADAPTER_SLOTS = (
    "_pgraft_network",
    "_pgraft_networks",
    "_hydra_network",
    "_hydra_networks",
    "_step_expert_networks",
)


def detach_adapters(model: anima_models.Anima) -> None:
    """Undo :func:`attach_adapters`: restore every forward a kept-live network
    patched and clear the slots the sampler reads."""
    model = getattr(model, "_orig_mod", model)
    networks = []
    for slot in _ADAPTER_SLOTS:
        held = model.__dict__.get(slot, model._modules.get(slot))
        if held is None:
            continue
        networks.extend(held if isinstance(held, (list, tuple)) else [held])
        delattr(model, slot)
    owned = {id(m): m for net in networks for m in net.modules()}
    if not owned:
        return
    for mod in model.modules():
        # Stacked LoRAs chain: each one's org_forward is the previous one's forward.
        while True:
            fwd = mod.__dict__.get("forward")
            lora = owned.get(id(getattr(fwd, "__self__", None)))
            if lora is None:
                break
            org = lora.org_forward
            if (
                getattr(org, "__self__", None) is mod
                and getattr(org, "__func__", None) is type(mod).forward
            ):
                del mod.forward
            else:
                mod.forward = org


def swap_adapters(
    model: anima_models.Anima,
    args: argparse.Namespace,
    device: torch.device,
    lora_weight: list[str],
) -> None:
    """Replace the kept-live adapters on a loaded DiT with ``lora_weight``."""
    detach_adapters(model)
    args.lora_weight = list(lora_weight)
    step_expert = any(_is_step_expert_turbo(p) for p in lora_weight)
    hydra = not step_expert and any(_is_hydra_moe(p) for p in lora_weight)
    attach_adapters(
        getattr(model, "_orig_mod", model),
        args,
        device,
        pgraft_mode=not (hydra or step_expert),
        hydra_mode=hydra,
        step_expert_mode=step_expert,
    )


def load_dit_model(
    args: argparse.Namespace,
    device: torch.device,
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> anima_models.Anima:
    """Load DiT model with optional LoRA merge, P-GRAFT hooks, and torch.compile.

    Namespace wrapper over ``library.anima.weights.load_anima_model`` (use that
    directly for just the weights).
    """

    loading_device = device

    # CJK vocab pack: --no_vocab_pack > --vocab_pack > base.toml `vocab_pack`.
    # The prompt side follows the same selection via prepare_text_inputs.
    from library.anima.vocab_pack import resolve_active_pack

    vocab_pack = resolve_active_pack(args)

    # Hydra and step-expert files skip the static merge. Step-expert is checked
    # first: its files also match _is_hydra_moe.
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

    # These modes keep the LoRA as live hooks instead of a static merge.
    int8 = getattr(args, "int8", False)
    pgraft_mode = (
        (
            getattr(args, "pgraft", False)
            or getattr(args, "lora_sigma_range", None) is not None
            or getattr(args, "lora_timestep_mask", False)
            or int8
            or bool(getattr(args, "lora_sweep", None))
        )
        and args.lora_weight is not None
        and len(args.lora_weight) > 0
    )

    if (
        not pgraft_mode
        and not hydra_mode
        and not step_expert_mode
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

    from library.runtime.backend import (
        needs_rocm_attention_fallback,
        resolve_attention_mode,
    )

    attn_mode = resolve_attention_mode(args.attn_mode, torch)
    if needs_rocm_attention_fallback(args.attn_mode, torch):
        logger.warning(
            "ROCm detected: using PyTorch SDPA instead of CUDA Flash Attention"
        )

    infer_file = ""
    if int8:
        from library.anima.int8 import infer_load_file

        infer_file = infer_load_file(getattr(args, "int8_file", None), args.dit)
    model = anima_utils.load_anima_model(
        device,
        args.dit,
        attn_mode,
        loading_device,
        dit_weight_dtype,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.lora_multiplier,
        vocab_pack=vocab_pack,
        int8_infer_file=infer_file or None,
    )

    # Before .to(): pooled_text_proj params are meta tensors when absent from
    # the checkpoint.
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

    if int8:
        from library.anima.int8 import convert_for_inference

        convert_for_inference(model, args)

    if getattr(args, "stream", "bf16") == "fp32":
        if getattr(args, "easycontrol_weight", None):
            logger.warning(
                "EasyControl runs on the bf16 stream; ignoring --stream fp32"
            )
        else:
            from library.anima.stream import stream_fp32_

            stream_fp32_(model)

    attach_adapters(
        model,
        args,
        device,
        pgraft_mode=pgraft_mode,
        hydra_mode=hydra_mode,
        step_expert_mode=step_expert_mode,
    )

    if getattr(args, "compile", False):
        logger.info("Compiling DiT model with torch.compile...")
        model = torch.compile(model)
    elif getattr(args, "compile_dynamic_seq", False):
        # compile_seq_range / compile_token_bands come from the batch path;
        # unset falls back to compile_blocks' 1024-tier band. A cached graph
        # carries its builder's seq-range guard, and a wider band reusing it hits
        # ConstraintViolationError → eager fallback, so compile fresh.
        import torch._functorch.config as functorch_config
        from library.runtime.dynamo import pin_inductor_flag

        pin_inductor_flag("fx_graph_cache", False)
        functorch_config.enable_autograd_cache = False
        functorch_config._config["enable_autograd_cache"].default = False

        bands = getattr(args, "compile_token_bands", None)
        model.compile_blocks(
            mode=getattr(args, "compile_inductor_mode", None),
            # recompile budget 2n+8: each band also splits on the cond/uncond
            # requires_grad guard and the xattn_boost renorm toggle
            n_token_families=2 * len(bands) if bands else None,
            dynamic_seq=True,
            seq_range=getattr(args, "compile_seq_range", None),
            seq_bands=bands,
        )
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

    Reads the encoder path and LoRA weight/multiplier as keywords or from
    ``args`` (keywords win)::

        load_text_encoder(text_encoder=TEXT_ENCODER, device=device)
    """
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
            }  # lora_te_* → model_*
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
