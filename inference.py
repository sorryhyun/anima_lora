"""Anima inference CLI -- argument parsing, batch/interactive modes, and main entry point."""

import argparse
import copy
import gc
import os
import random
from typing import List, Dict, Any

from library.runtime.proc import install_no_window_default

install_no_window_default()

import torch
from safetensors.torch import load_file
from safetensors import safe_open
from diffusers.utils.torch_utils import randn_tensor

from library.anima import (
    models as anima_models,
    strategy as strategy_anima,
    text_strategies,
)
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.runtime.device import clean_memory_on_device
from library.inference import (
    build_default_args,
    get_generation_settings,
    resolve_seed,
    check_inputs,
    load_dit_model,
    load_text_encoder,
    load_shared_models,
    prepare_text_inputs,
    generate,
    generate_body,
    save_latent,
    save_output,
    write_gen_manifest,
)
from library.inference.text import MAX_CROSSATTN_TOKENS

# Side-effect import: registers spectrum_denoise with library.inference.generation
# so --spectrum dispatches without library.inference holding a hard edge into networks/.
import networks.spectrum  # noqa: F401, E402

# Same pattern for SPD (Spectral Progressive Diffusion) — registers spd_denoise.
import networks.spd  # noqa: F401, E402

# Same pattern for the deferred-foveated merge — registers foveated_denoise.
import networks.foveated  # noqa: F401, E402

from library.log import setup_logging  # noqa: E402

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


# region Argument parsing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    ``argv=None`` reads ``sys.argv`` (the CLI path). Pass an explicit list to
    build an args namespace programmatically -- see ``examples/01_generate.py``.

    Thin delegate to :func:`library.inference.args.build_default_args`: the
    parser definition lives in ``library`` so programmatic callers
    (``GenerationRequest``, ``bench/`` probes) don't have to import this
    entry-point script. ``from inference import parse_args`` still works.
    """
    return build_default_args(argv)


# endregion


# region Prompt parsing


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line into a dictionary of argument overrides."""
    parts = line.split(" --")
    prompt = parts[0].strip()

    overrides = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        if option == "w":
            overrides["image_size_width"] = int(value)
        elif option == "h":
            overrides["image_size_height"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "n":
            overrides["negative_prompt"] = value

    return overrides


def apply_overrides(
    args: argparse.Namespace, overrides: Dict[str, Any]
) -> argparse.Namespace:
    """Apply overrides to args, returning a new copy."""
    args_copy = copy.deepcopy(args)

    for key, value in overrides.items():
        if key == "image_size_width":
            args_copy.image_size[1] = value
        elif key == "image_size_height":
            args_copy.image_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def preprocess_prompts_for_batch(
    prompt_lines: List[str], base_args: argparse.Namespace
) -> List[Dict]:
    """Process multiple prompt lines for batch mode."""
    prompts_data = []

    for line in prompt_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        prompt_data = parse_prompt_line(line)
        logger.info(f"Parsed prompt data: {prompt_data}")
        prompts_data.append(prompt_data)

    return prompts_data


# endregion


# region Batch / interactive modes


def process_batch_prompts(prompts_data: List[Dict], args: argparse.Namespace) -> None:
    """Process multiple prompts with model reuse and batched precomputation."""
    if not prompts_data:
        logger.warning("No valid prompts found")
        return

    gen_settings = get_generation_settings(args)
    dit_weight_dtype = torch.bfloat16
    device = gen_settings.device

    # 1. Prepare VAE
    logger.info("Loading VAE for batch generation...")
    vae_for_batch = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
        vae_2d=args.vae_2d,
    )
    vae_for_batch.to(device, dtype=torch.bfloat16)
    vae_for_batch.eval()

    all_prompt_args_list = [apply_overrides(args, pd) for pd in prompts_data]
    for prompt_args in all_prompt_args_list:
        check_inputs(prompt_args)

    # 2. Load DiT Model once
    logger.info("Loading DiT model for batch generation...")
    first_prompt_args = all_prompt_args_list[0]
    anima = load_dit_model(first_prompt_args, device, dit_weight_dtype)

    # 3. Precompute Text Data (Text Encoder)
    logger.info("Loading Text Encoder for batch text preprocessing...")

    text_encoder_dtype = torch.bfloat16
    text_encoder_batch = load_text_encoder(
        args, dtype=text_encoder_dtype, device=torch.device("cpu")
    )

    text_encoder_device = torch.device("cpu") if args.text_encoder_cpu else device
    text_encoder_batch.to(text_encoder_device)

    all_precomputed_text_data = []
    conds_cache_batch = {}

    logger.info("Preprocessing text and LLM/TextEncoder encoding for all prompts...")
    temp_shared_models_txt = {
        "text_encoder": text_encoder_batch,
        "conds_cache": conds_cache_batch,
    }

    for i, prompt_args_item in enumerate(all_prompt_args_list):
        logger.info(
            f"Text preprocessing for prompt {i + 1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}"
        )

        context, context_null = prepare_text_inputs(
            prompt_args_item, device, anima, temp_shared_models_txt
        )
        text_data = {"context": context, "context_null": context_null}
        all_precomputed_text_data.append(text_data)

    del text_encoder_batch, temp_shared_models_txt, conds_cache_batch
    gc.collect()
    clean_memory_on_device(device)

    # 4. Setup modulation guidance once (shared across all prompts)
    if (
        getattr(args, "pooled_text_proj", None) is not None
        and getattr(args, "mod_w", 0.0) != 0.0
    ):
        from library.inference.corrections.mod_guidance import setup_mod_guidance

        setup_mod_guidance(args, anima, device)
    else:
        anima.reset_mod_guidance()

    # 5. Group prompts by text content for batched denoising
    infer_batch_size = getattr(args, "infer_batch_size", 1)

    groups = []
    for i, prompt_args_item in enumerate(all_prompt_args_list):
        key = (prompt_args_item.prompt, tuple(prompt_args_item.image_size))
        if groups and groups[-1][0] == key and len(groups[-1][1]) < infer_batch_size:
            groups[-1][1].append(i)
        else:
            groups.append((key, [i]))

    logger.info(
        f"Generating latents: {len(all_prompt_args_list)} prompts in {len(groups)} group(s) (batch_size={infer_batch_size})"
    )

    with torch.no_grad():
        for (_prompt_text, _img_size), indices in groups:
            batch_size = len(indices)
            first_args = all_prompt_args_list[indices[0]]
            first_text_data = all_precomputed_text_data[indices[0]]
            height, width = check_inputs(first_args)

            logger.info(
                f"Batched generation ({batch_size}x) for: {first_args.prompt[:80]}... "
                f"size={height}x{width}, steps={first_args.infer_steps}"
            )

            try:
                # Collect seeds and create batched latent noise
                seeds = []
                num_channels_latents = anima_models.Anima.LATENT_CHANNELS
                single_shape = (1, num_channels_latents, 1, height // 8, width // 8)
                latent_list = []

                for idx in indices:
                    pa = all_prompt_args_list[idx]
                    s = pa.seed if pa.seed is not None else random.randint(0, 2**32 - 1)
                    seeds.append(s)
                    g = torch.Generator(device=device)
                    g.manual_seed(s)
                    latent_list.append(
                        randn_tensor(
                            single_shape,
                            generator=g,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                    )

                batched_latents = torch.cat(latent_list, dim=0)  # (B, C, T, H, W)

                # Store first seed on args for ER-SDE sampler
                first_args.seed = seeds[0]

                # Run unified denoising loop
                latents = generate_body(
                    first_args,
                    anima,
                    first_text_data["context"],
                    first_text_data["context_null"],
                    device,
                    seeds,
                    latents=batched_latents,
                )

                # Split batch and save
                for j, idx in enumerate(indices):
                    single_latent = latents[j : j + 1]
                    all_prompt_args_list[idx].seed = seeds[j]
                    if all_prompt_args_list[idx].output_type in [
                        "latent",
                        "latent_images",
                    ]:
                        save_latent(
                            single_latent, all_prompt_args_list[idx], height, width
                        )

                    if args.output_type != "latent":
                        current_args = all_prompt_args_list[idx]
                        if current_args.output_type == "latent_images":
                            current_args.output_type = "images"
                        save_output(current_args, vae_for_batch, single_latent, device)

                logger.info(f"Saved {batch_size} image(s) to disk")

            except Exception as e:
                logger.error(
                    f"Error generating batch for prompt: {first_args.prompt}. Error: {e}",
                    exc_info=True,
                )
                continue

    # Free models
    logger.info("Releasing models from memory...")
    del anima, vae_for_batch
    clean_memory_on_device(device)


def process_interactive(args: argparse.Namespace) -> None:
    """Process prompts in interactive mode."""
    gen_settings = get_generation_settings(args)
    device = gen_settings.device
    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}

    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
        vae_2d=args.vae_2d,
    )
    vae.to(torch.bfloat16)
    vae.eval()

    print("Interactive mode. Enter prompts (Ctrl+D or Ctrl+Z (Windows) to exit):")

    try:
        import prompt_toolkit
    except ImportError:
        logger.warning("prompt_toolkit not found. Using basic input instead.")
        prompt_toolkit = None

    if prompt_toolkit:
        session = prompt_toolkit.PromptSession()

        def input_line(prompt: str) -> str:
            return session.prompt(prompt)

    else:

        def input_line(prompt: str) -> str:
            return input(prompt)

    try:
        while True:
            try:
                line = input_line("> ")
                if not line.strip():
                    continue
                if len(line.strip()) == 1 and line.strip() in [
                    "\x04",
                    "\x1a",
                ]:  # Ctrl+D or Ctrl+Z with prompt_toolkit
                    raise EOFError

                prompt_data = parse_prompt_line(line)
                prompt_args = apply_overrides(args, prompt_data)

                # Pin the resolved seed for save_output (generate() no longer
                # writes it back to the namespace).
                prompt_args.seed = resolve_seed(prompt_args)
                latent = generate(prompt_args, gen_settings, shared_models)

                save_output(prompt_args, vae, latent, device)

            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue

    except EOFError:
        print("\nExiting interactive mode")


# endregion


# region Main


def _list_pngs(path) -> set:
    """The set of ``*.png`` under ``path`` (empty if it doesn't exist yet). Used
    to diff before/after a run so the daemon manifest captures exactly the images
    this run produced, without threading a collector through every save path."""
    try:
        return {os.path.join(path, n) for n in os.listdir(path) if n.endswith(".png")}
    except OSError:
        return set()


def main():
    args = parse_args()

    # Snapshot the output dir so we can attribute freshly-written images to this
    # run for the daemon result manifest (Phase 1a lift). No-op cost when inline.
    _pngs_before = _list_pngs(getattr(args, "save_path", None))

    # Check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device = (
        args.device
        if args.device is not None
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device

    if latents_mode:
        # Original latent decode mode
        original_base_names = []
        latents_list = []
        seeds = []

        for latent_path in args.latent_path:
            original_base_names.append(
                os.path.splitext(os.path.basename(latent_path))[0]
            )
            seed = 0

            if os.path.splitext(latent_path)[1] != ".safetensors":
                latents = torch.load(latent_path, map_location="cpu")
            else:
                latents = load_file(latent_path)["latent"]
                with safe_open(latent_path, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is None:
                    metadata = {}
                logger.info(f"Loaded metadata: {metadata}")

                if "seeds" in metadata:
                    seed = int(metadata["seeds"])
                if "height" in metadata and "width" in metadata:
                    height = int(metadata["height"])
                    width = int(metadata["width"])
                    args.image_size = [height, width]

            seeds.append(seed)
            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")

            if latents.ndim == 5:  # [BCTHW]
                latents = latents.squeeze(0)  # [CTHW]

            latents_list.append(latents)

        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device=device,
            disable_mmap=True,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
            vae_2d=args.vae_2d,
        )
        vae.to(torch.bfloat16)
        vae.eval()

        for i, latent in enumerate(latents_list):
            args.seed = seeds[i]
            save_output(args, vae, latent, device, original_base_names[i])

    else:
        from library.anima.vocab_pack import make_tokenize_strategy, resolve_active_pack

        # --no_vocab_pack > --vocab_pack > base.toml `vocab_pack` ("" = off).
        # load_dit_model hooks the same (memoised) pack onto llm_adapter.embed.
        tokenize_strategy = make_tokenize_strategy(
            resolve_active_pack(args),
            qwen3_path=args.text_encoder,
            t5_tokenizer_path=None,
            qwen3_max_length=MAX_CROSSATTN_TOKENS,
            t5_max_length=MAX_CROSSATTN_TOKENS,
        )
        text_strategies.TokenizeStrategy.set_strategy(tokenize_strategy)

        encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
        text_strategies.TextEncodingStrategy.set_strategy(encoding_strategy)

        if args.from_file:
            # Batch mode from file
            with open(args.from_file, "r", encoding="utf-8") as f:
                prompt_lines = f.readlines()

            prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
            process_batch_prompts(prompts_data, args)

        elif args.interactive:
            process_interactive(args)

        else:
            # Single prompt mode
            gen_settings = get_generation_settings(args)
            # generate() no longer writes the resolved seed back to args, so
            # pin it here for save_output()'s filename + metadata.
            args.seed = resolve_seed(args)
            latent = generate(args, gen_settings)

            clean_memory_on_device(device)

            vae = qwen_image_autoencoder_kl.load_vae(
                args.vae,
                device="cpu",
                disable_mmap=True,
                spatial_chunk_size=args.vae_chunk_size,
                disable_cache=args.vae_disable_cache,
                vae_2d=args.vae_2d,
            )
            vae.to(torch.bfloat16)
            vae.eval()
            save_output(args, vae, latent, device)

    # Record a generation manifest + result pointer when running as a daemon job
    # (proposal Phase 1a). Inline runs: no-op. Sorted so the manifest is stable.
    new_images = sorted(_list_pngs(getattr(args, "save_path", None)) - _pngs_before)
    write_gen_manifest(args, new_images)

    logger.info("Done!")


# endregion


if __name__ == "__main__":
    main()
