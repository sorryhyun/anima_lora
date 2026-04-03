"""Anima inference CLI — argument parsing, batch/interactive modes, and main entry point."""

import argparse
import copy
import gc
import os
import random
from importlib.util import find_spec
from typing import List, Dict, Any

import torch
from safetensors.torch import load_file
from safetensors import safe_open
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor

from library import (
    anima_models,
    inference_utils,
    qwen_image_autoencoder_kl,
    strategy_anima,
    strategy_base,
)
from library.device_utils import clean_memory_on_device
from library.inference_pipeline import (
    get_generation_settings,
    check_inputs,
    load_dit_model,
    load_text_encoder,
    load_shared_models,
    prepare_text_inputs,
    generate,
    save_latent,
    save_output,
)

lycoris_available = find_spec("lycoris") is not None
if lycoris_available:
    pass

from library.utils import setup_logging  # noqa: E402

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)


# region Argument parsing


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="HunyuanImage inference script")

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument(
        "--vae_chunk_size",
        type=int,
        default=None,
        help="Spatial chunk size for VAE encoding/decoding to reduce memory usage. Must be even number. If not specified, chunking is disabled (official behavior)."
        + " / メモリ使用量を減らすためのVAEエンコード/デコードの空間チャンクサイズ。偶数である必要があります。未指定の場合、チャンク処理は無効になります（公式の動作）。",
    )
    parser.add_argument(
        "--vae_disable_cache",
        action="store_true",
        help="Disable internal VAE caching mechanism to reduce memory usage. Encoding / decoding will also be faster, but this differs from official behavior."
        + " / VAEのメモリ使用量を減らすために内部のキャッシュ機構を無効にします。エンコード/デコードも速くなりますが、公式の動作とは異なります。",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Text Encoder 1 (Qwen2.5-VL) directory or path",
    )

    # LoRA
    parser.add_argument(
        "--lora_weight",
        type=str,
        nargs="*",
        required=False,
        default=None,
        help="LoRA weight path",
    )
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier"
    )
    parser.add_argument(
        "--include_patterns",
        type=str,
        nargs="*",
        default=None,
        help="LoRA module include patterns",
    )
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        nargs="*",
        default=None,
        help="LoRA module exclude patterns",
    )

    # inference
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for classifier free guidance. Default is 3.5.",
    )
    parser.add_argument(
        "--prompt", type=str, default=None, help="prompt for generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="negative prompt for generation, default is empty string",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        help="image size, height and width",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=50,
        help="number of inference steps, default is 50",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="path to save generated video"
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=5.0,
        help="Shift factor for flow matching schedulers. Default is 5.0.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
        choices=["euler", "er_sde"],
        help="Sampler to use: 'euler' (deterministic ODE) or 'er_sde' (Extended Reverse-Time SDE). Default is euler.",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument(
        "--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8"
    )

    parser.add_argument(
        "--text_encoder_cpu",
        action="store_true",
        help="Inference on CPU for Text Encoders",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device to use for inference. If None, use CUDA if available, otherwise use CPU",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=[
            "flash",
            "flash4",
            "torch",
            "sageattn",
            "flex",
            "xformers",
            "sdpa",
        ],  #  "sdpa" for backward compatibility
        help="attention mode",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="images",
        choices=["images", "latent", "latent_images"],
        help="output type",
    )
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata"
    )
    parser.add_argument(
        "--latent_path",
        type=str,
        nargs="*",
        default=None,
        help="path to latent for decode. no inference",
    )
    parser.add_argument(
        "--lycoris",
        action="store_true",
        help=f"use lycoris for inference{'' if lycoris_available else ' (not available)'}",
    )

    # P-GRAFT
    parser.add_argument(
        "--pgraft",
        action="store_true",
        help="Enable P-GRAFT: load LoRA as dynamic hooks instead of static merge, allowing mid-denoising cutoff",
    )
    parser.add_argument(
        "--lora_cutoff_step",
        type=int,
        default=None,
        help="Step at which to disable LoRA during inference (for P-GRAFT). "
        "LoRA is active for steps 0..cutoff_step-1, disabled for cutoff_step..end.",
    )

    # Tiled diffusion
    parser.add_argument(
        "--tiled_diffusion",
        action="store_true",
        help="Enable MultiDiffusion-style tiled generation for VRAM reduction at high resolutions",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=128,
        help="Tile size in latent space (default 128 = 1024px). Must be even.",
    )
    parser.add_argument(
        "--tile_overlap",
        type=int,
        default=16,
        help="Overlap between tiles in latent space (default 16 = 128px). Must be even and < tile_size.",
    )

    # Spectrum acceleration
    parser.add_argument(
        "--spectrum",
        action="store_true",
        help="Enable Spectrum inference acceleration (Chebyshev polynomial feature forecasting). "
        "Skips transformer blocks on predicted steps, running only final_layer + unpatchify.",
    )
    parser.add_argument(
        "--spectrum_window_size",
        type=float,
        default=2.0,
        help="Spectrum initial window size N (default 2.0)",
    )
    parser.add_argument(
        "--spectrum_flex_window",
        type=float,
        default=0.25,
        help="Spectrum flex parameter alpha — N grows by this after each actual forward (default 0.25)",
    )
    parser.add_argument(
        "--spectrum_warmup",
        type=int,
        default=6,
        help="Spectrum warmup steps (always run full forward) (default 6)",
    )
    parser.add_argument(
        "--spectrum_w",
        type=float,
        default=0.3,
        help="Spectrum Chebyshev/Taylor blend weight (1.0=pure Chebyshev, default 0.3)",
    )
    parser.add_argument(
        "--spectrum_m",
        type=int,
        default=3,
        help="Spectrum number of Chebyshev basis functions (default 3)",
    )
    parser.add_argument(
        "--spectrum_lam",
        type=float,
        default=0.1,
        help="Spectrum ridge regression regularization (default 0.1)",
    )
    parser.add_argument(
        "--spectrum_stop_caching_step",
        type=int,
        default=-1,
        help="Force actual forwards from this step onward (-1 = auto: total_steps - 3)",
    )
    parser.add_argument(
        "--spectrum_calibration",
        type=float,
        default=0.0,
        help="Spectrum residual calibration strength (0.0=disabled, default 0.0). "
        "Adds residual bias correction from last actual forward to cached predictions.",
    )

    # arguments for batch and interactive modes
    parser.add_argument(
        "--from_file", type=str, default=None, help="Read prompts from a file"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: read prompts from console",
    )
    parser.add_argument(
        "--infer_batch_size",
        type=int,
        default=1,
        help="Batch size for denoising. Prompts sharing the same text are batched together. Higher values use more VRAM.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the DiT model with torch.compile for faster inference. First run incurs compilation overhead.",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError(
            "Cannot use both --from_file and --interactive at the same time"
        )

    if args.latent_path is None or len(args.latent_path) == 0:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError(
                "Either --prompt, --from_file or --interactive must be specified"
            )

    if args.lycoris and not lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    if args.tiled_diffusion:
        if args.tile_size % 2 != 0:
            raise ValueError(
                f"--tile_size must be even (patch_spatial=2 requires it), got {args.tile_size}"
            )
        if args.tile_overlap % 2 != 0:
            raise ValueError(
                f"--tile_overlap must be even (patch_spatial=2 requires it), got {args.tile_overlap}"
            )
        if args.tile_overlap >= args.tile_size:
            raise ValueError(
                f"--tile_overlap ({args.tile_overlap}) must be less than --tile_size ({args.tile_size})"
            )

    return args


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
    dit_weight_dtype = gen_settings.dit_weight_dtype
    device = gen_settings.device

    # 1. Prepare VAE
    logger.info("Loading VAE for batch generation...")
    vae_for_batch = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
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

    shared_models_for_generate = {"model": anima}

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

    # Group prompts by text content for batched denoising
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

    anima: anima_models.Anima = shared_models_for_generate["model"]

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

            group_latents = []
            try:
                # Collect seeds and create batched latent noise
                seeds = []
                seed_generators = []
                for idx in indices:
                    pa = all_prompt_args_list[idx]
                    s = pa.seed if pa.seed is not None else random.randint(0, 2**32 - 1)
                    seeds.append(s)
                    g = torch.Generator(device=device)
                    g.manual_seed(s)
                    seed_generators.append(g)

                num_channels_latents = anima_models.Anima.LATENT_CHANNELS
                single_shape = (1, num_channels_latents, 1, height // 8, width // 8)
                latent_list = [
                    randn_tensor(
                        single_shape, generator=g, device=device, dtype=torch.bfloat16
                    )
                    for g in seed_generators
                ]
                latents = torch.cat(latent_list, dim=0)  # (B, C, T, H, W)

                bs = batch_size
                h_latent = latents.shape[-2]
                w_latent = latents.shape[-1]
                padding_mask = torch.zeros(
                    bs, 1, h_latent, w_latent, dtype=torch.bfloat16, device=device
                )

                # Expand shared text embeddings to batch
                embed = first_text_data["context"]["embed"][0].to(
                    device, dtype=torch.bfloat16
                )
                context_null = (
                    first_text_data.get("context_null") or first_text_data["context"]
                )
                negative_embed = context_null["embed"][0].to(
                    device, dtype=torch.bfloat16
                )
                embed = embed.expand(bs, -1, -1)
                negative_embed = negative_embed.expand(bs, -1, -1)

                timesteps, sigmas = inference_utils.get_timesteps_sigmas(
                    first_args.infer_steps, first_args.flow_shift, device
                )
                timesteps = (timesteps / 1000).to(device, dtype=torch.bfloat16)

                # Create sampler
                er_sde = None
                if first_args.sampler == "er_sde":
                    er_sde = inference_utils.ERSDESampler(
                        sigmas, seed=first_args.seed, device=device
                    )

                do_cfg = first_args.guidance_scale != 1.0
                autocast_enabled = first_args.fp8

                # P-GRAFT: get network reference for mid-denoising cutoff
                pgraft_network = getattr(anima, "_pgraft_network", None)
                lora_cutoff_step = getattr(first_args, "lora_cutoff_step", None)

                with tqdm(
                    total=len(timesteps), desc=f"Denoising ({bs}x batch)"
                ) as pbar:
                    for step_i, t in enumerate(timesteps):
                        if (
                            pgraft_network is not None
                            and lora_cutoff_step is not None
                            and step_i == lora_cutoff_step
                        ):
                            pgraft_network.set_enabled(False)
                            logger.info(
                                f"P-GRAFT: Disabled LoRA at step {step_i}/{len(timesteps)}"
                            )

                        t_expand = t.expand(bs)

                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.bfloat16,
                            enabled=autocast_enabled,
                        ):
                            noise_pred = anima(
                                latents, t_expand, embed, padding_mask=padding_mask
                            )

                        if do_cfg:
                            with torch.autocast(
                                device_type=device.type,
                                dtype=torch.bfloat16,
                                enabled=autocast_enabled,
                            ):
                                uncond_noise_pred = anima(
                                    latents,
                                    t_expand,
                                    negative_embed,
                                    padding_mask=padding_mask,
                                )
                            noise_pred = (
                                uncond_noise_pred
                                + first_args.guidance_scale
                                * (noise_pred - uncond_noise_pred)
                            )

                        if er_sde is not None:
                            denoised = (
                                latents.float() - sigmas[step_i] * noise_pred.float()
                            )
                            latents = er_sde.step(latents, denoised, step_i).to(
                                latents.dtype
                            )
                        else:
                            latents = inference_utils.step(
                                latents, noise_pred, sigmas, step_i
                            ).to(latents.dtype)
                        pbar.update()

                # P-GRAFT: restore LoRA for next group
                if pgraft_network is not None and lora_cutoff_step is not None:
                    pgraft_network.set_enabled(True)

                # Split batch and decode+save immediately
                for j, idx in enumerate(indices):
                    single_latent = latents[j : j + 1]  # keep batch dim
                    all_prompt_args_list[idx].seed = seeds[j]
                    if all_prompt_args_list[idx].output_type in [
                        "latent",
                        "latent_images",
                    ]:
                        save_latent(
                            single_latent, all_prompt_args_list[idx], height, width
                        )
                    group_latents.append((idx, single_latent))

            except Exception as e:
                logger.error(
                    f"Error generating batch for prompt: {first_args.prompt}. Error: {e}",
                    exc_info=True,
                )
                continue

            # Decode and save this group's latents immediately
            if args.output_type != "latent" and group_latents:
                for idx, latent in group_latents:
                    current_args = all_prompt_args_list[idx]
                    if current_args.output_type == "latent_images":
                        current_args.output_type = "images"
                    save_output(current_args, vae_for_batch, latent, device)
                logger.info(f"Saved {len(group_latents)} image(s) to disk")

    # Free DiT model
    logger.info("Releasing DiT model from memory...")

    del shared_models_for_generate["model"]
    del anima
    del vae_for_batch
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

                latent = generate(prompt_args, gen_settings, shared_models)

                save_output(prompt_args, vae, latent, device)

            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue

    except EOFError:
        print("\nExiting interactive mode")


# endregion


# region Main


def main():
    args = parse_args()

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
        )
        vae.to(torch.bfloat16)
        vae.eval()

        for i, latent in enumerate(latents_list):
            args.seed = seeds[i]
            save_output(args, vae, latent, device, original_base_names[i])

    else:
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.text_encoder,
            t5_tokenizer_path=None,
            qwen3_max_length=512,
            t5_max_length=512,
        )
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

        encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
        strategy_base.TextEncodingStrategy.set_strategy(encoding_strategy)

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
            latent = generate(args, gen_settings)

            clean_memory_on_device(device)

            vae = qwen_image_autoencoder_kl.load_vae(
                args.vae,
                device="cpu",
                disable_mmap=True,
                spatial_chunk_size=args.vae_chunk_size,
                disable_cache=args.vae_disable_cache,
            )
            vae.to(torch.bfloat16)
            vae.eval()
            save_output(args, vae, latent, device)

    logger.info("Done!")


# endregion


if __name__ == "__main__":
    main()
