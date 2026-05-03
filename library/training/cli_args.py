"""Argparse argument adders for the Anima training CLI.

Each ``add_*_arguments`` call plugs a related group of flags into the training
parser. The groups split the flag surface into coherent chunks so individual
entry points (training, preprocessing, distillation, ...) can opt into only
what they need.

No real logic lives here — these are pure argparse declarations.
"""

from __future__ import annotations

import argparse
import logging

from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def add_sd_models_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--v2",
        action="store_true",
        help="load Stable Diffusion v2.0 model",
    )
    parser.add_argument(
        "--v_parameterization",
        action="store_true",
        help="enable v-parameterization training",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training)",
    )


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    def int_or_float(value):
        if value.endswith("%"):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Value '{value}' is not a valid percentage"
                )
        try:
            float_value = float(value)
            if float_value >= 1:
                return int(value)
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not an int or float")

    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use"
        "Lion8bit, PagedLion8bit, Lion, SGDNesterov, SGDNesterov8bit, "
        "DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, "
        "AdaFactor.",
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="use 8bit AdamW optimizer (requires bitsandbytes)",
    )
    parser.add_argument(
        "--use_lion_optimizer",
        action="store_true",
        help="use Lion optimizer (requires lion-pytorch)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2.0e-6, help="learning rate"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm, 0 for no clipping",
    )

    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...")',
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="",
        help="custom scheduler module",
    )
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100")',
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the warmup in the lr scheduler (default is 0) or float with ratio of train steps",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the decay in the lr scheduler (default is 0) or float (<1) with ratio of train steps",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler",
    )
    parser.add_argument(
        "--fused_backward_pass",
        action="store_true",
        help="Combines backward pass and optimizer step to reduce VRAM usage.",
    )
    parser.add_argument(
        "--lr_scheduler_timescale",
        type=int,
        default=None,
        help="Inverse sqrt timescale for inverse sqrt scheduler,defaults to `num_warmup_steps`",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr_ratio",
        type=float,
        default=None,
        help="The minimum learning rate as a ratio of the initial learning rate for cosine with min lr scheduler and warmup decay scheduler",
    )


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to output trained model",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="base name of trained model file",
    )
    parser.add_argument(
        "--huggingface_repo_id",
        type=str,
        default=None,
        help="huggingface repo name to upload",
    )
    parser.add_argument(
        "--huggingface_repo_type",
        type=str,
        default=None,
        help="huggingface repo type to upload",
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files",
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="huggingface token",
    )
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private)",
    )
    parser.add_argument(
        "--save_state_to_huggingface",
        action="store_true",
        help="save state to huggingface",
    )
    parser.add_argument(
        "--resume_from_huggingface", action="store_true", help="resume from huggingface"
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving",
    )
    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="save checkpoint every N epochs",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="save checkpoint every N steps",
    )
    parser.add_argument(
        "--save_n_epoch_ratio",
        type=int,
        default=None,
        help="save checkpoint N epoch ratio",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints)",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed)",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps)",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.) when saving model",
    )
    parser.add_argument(
        "--save_state_on_train_end",
        action="store_true",
        help="save training state (including optimizer states etc.) on train end",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="saved state to resume training",
    )
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=None,
        help="save resumable checkpoint every N epochs (overwrites previous, auto-resumes on next run)",
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="batch size for training",
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=None,
        choices=[None, 150, 225],
        help="max token length of text encoder (default for 75, 150 or 225)",
    )
    parser.add_argument(
        "--mem_eff_attn",
        action="store_true",
        help="use memory efficient attention for CrossAttention",
    )
    parser.add_argument(
        "--profile_steps",
        type=str,
        default=None,
        help=(
            "toggle the CUDA profiler for the given step range, e.g. '3-5'. "
            "Pair with: nsys profile --capture-range=cudaProfilerApi "
            "--capture-range-end=stop ... so nsys only records that window. "
            "NVTX ranges (step / forward / backward / optimizer) label the timeline."
        ),
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="use torch.compile (requires PyTorch 2.0)",
    )
    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="inductor",
        choices=[
            "eager",
            "aot_eager",
            "inductor",
            "aot_ts_nvfuser",
            "nvprims_nvfuser",
            "cudagraphs",
            "ofi",
            "fx2trt",
            "onnxrt",
            "tensort",
            "ipex",
            "tvm",
        ],
        help="dynamo backend type (default is inductor)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="blocks",
        choices=["blocks", "full"],
        help="torch.compile mode: 'blocks' compiles each DiT block individually (default), "
        "'full' compiles the entire model as one graph for cross-block memory optimization "
        "(incompatible with gradient checkpointing and block swap)",
    )
    parser.add_argument(
        "--compile_inductor_mode",
        type=str,
        default=None,
        choices=[None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="Inductor preset forwarded as torch.compile(..., mode=...). "
        "'reduce-overhead' enables CUDAGraphs — requires stable tensor addresses "
        "across steps and is incompatible with block swap. Works with both "
        "--compile_mode=blocks and --compile_mode=full (per-block graph vs one graph).",
    )
    parser.add_argument(
        "--xformers", action="store_true", help="use xformers for CrossAttention"
    )
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="use sdpa for CrossAttention (requires PyTorch 2.0)",
    )
    parser.add_argument(
        "--vae", type=str, default=None, help="path to checkpoint of vae to replace"
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1600,
        help="training steps",
    )
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps)",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=4,
        help="max num workers for DataLoader",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers",
    )
    parser.add_argument(
        "--dataloader_pin_memory", action="store_true", help="pin DataLoader memory"
    )
    parser.add_argument(
        "--dataloader_prefetch_factor",
        type=int,
        default=12,
        help="prefetch_factor for DataLoader workers (only valid when num_workers>0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="enable gradient checkpointing",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="use mixed precision",
    )
    parser.add_argument(
        "--full_fp16", action="store_true", help="fp16 training including gradients"
    )
    parser.add_argument(
        "--full_bf16", action="store_true", help="bf16 training including gradients"
    )
    # FP8 is not supported yet — flag kept for CLI compatibility but force-disabled in assert_extra_args.
    parser.add_argument(
        "--fp8_base",
        action="store_true",
        help="(not supported yet) use fp8 for base model. This flag is force-disabled.",
    )

    parser.add_argument(
        "--clip_skip",
        type=int,
        default=None,
        help="use output of nth layer from back of text encoder (n>=1)",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use",
    )
    parser.add_argument(
        "--log_prefix", type=str, default=None, help="add prefix for each log directory"
    )
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the specific wandb session",
    )
    parser.add_argument(
        "--log_tracker_config",
        type=str,
        default=None,
        help="path to tracker config file to use for logging",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional).",
    )
    parser.add_argument(
        "--log_config", action="store_true", help="log training configuration"
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=1,
        help="only emit step-level metrics every N global steps (default 1 = every step). "
        "Validation and epoch logs are unaffected. Useful for long W&B runs.",
    )

    parser.add_argument(
        "--noise_offset",
        type=float,
        default=None,
        help="enable noise offset with this value (if enabled, around 0.1 is recommended)",
    )
    parser.add_argument(
        "--noise_offset_random_strength",
        action="store_true",
        help="use random strength between 0~noise_offset for noise offset.",
    )
    parser.add_argument(
        "--multires_noise_iterations",
        type=int,
        default=None,
        help="enable multires noise with this number of iterations (if enabled, around 6-10 is recommended)",
    )
    parser.add_argument(
        "--ip_noise_gamma",
        type=float,
        default=None,
        help="enable input perturbation noise. recommended value: around 0.1",
    )
    parser.add_argument(
        "--ip_noise_gamma_random_strength",
        action="store_true",
        help="Use random strength between 0~ip_noise_gamma for input perturbation noise.",
    )
    parser.add_argument(
        "--multires_noise_discount",
        type=float,
        default=0.3,
        help="set discount value for multires noise",
    )
    parser.add_argument(
        "--adaptive_noise_scale",
        type=float,
        default=None,
        help="add `latent mean absolute value * this value` to noise_offset (disabled if None, default)",
    )
    parser.add_argument(
        "--zero_terminal_snr",
        action="store_true",
        help="fix noise scheduler betas to enforce zero terminal SNR",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=None,
        help="set minimum time step for U-Net training (0~999, default is 0)",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for U-Net training (1~1000, default is 1000)",
    )
    parser.add_argument(
        "--t_min",
        type=float,
        default=None,
        help="Restrict training sigma range: minimum sigma (0.0~1.0).",
    )
    parser.add_argument(
        "--t_max",
        type=float,
        default=None,
        help="Restrict training sigma range: maximum sigma (0.0~1.0). Default 1.0.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l1", "l2", "huber", "smooth_l1", "pseudo_huber"],
        help="The type of loss function to use (L1, L2, Huber, smooth L1, or pseudo-Huber), default is L2",
    )
    parser.add_argument(
        "--huber_schedule",
        type=str,
        default="snr",
        choices=["constant", "exponential", "snr"],
        help="The scheduling method for Huber loss. default is snr",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.1,
        help="The Huber loss decay parameter. default is 0.1",
    )
    parser.add_argument(
        "--huber_scale",
        type=float,
        default=1.0,
        help="The Huber loss scale parameter. default is 1.0",
    )
    parser.add_argument(
        "--pseudo_huber_c",
        type=float,
        default=0.03,
        help="Pseudo-Huber loss parameter c. Small c -> L1-like, large c -> MSE-like. default is 0.03",
    )
    parser.add_argument(
        "--multiscale_loss_weight",
        type=float,
        default=0,
        help="Weight for 2x-downsampled multiscale loss term. 0 = disabled. default is 0",
    )
    parser.add_argument(
        "--lowram", action="store_true", help="enable low RAM optimization."
    )
    parser.add_argument(
        "--highvram", action="store_true", help="disable low VRAM optimization."
    )

    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=None,
        help="generate sample images every N steps",
    )
    parser.add_argument(
        "--sample_at_first",
        action="store_true",
        help="generate sample images before training",
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps)",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="file for prompts to generate sample images",
    )
    parser.add_argument(
        "--sample_sampler",
        type=str,
        default="ddim",
        choices=[
            "ddim",
            "pndm",
            "lms",
            "euler",
            "euler_a",
            "heun",
            "dpm_2",
            "dpm_2_a",
            "dpmsolver",
            "dpmsolver++",
            "dpmsingle",
            "k_lms",
            "k_euler",
            "k_euler_a",
            "k_dpm_2",
            "k_dpm_2_a",
        ],
        help="sampler (scheduler) type for sample images",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="method name under configs/methods/ (e.g. 'tlora', 'hydralora', 'postfix'). Merged after preset so method settings win on overlap.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="default",
        help="hardware preset section name in configs/presets.toml (e.g. 'default', 'fast_16gb', 'low_vram').",
    )
    parser.add_argument(
        "--methods_subdir",
        type=str,
        default="methods",
        help="subfolder under configs/ that holds the method file (default 'methods'). Use 'gui-methods' for the GUI-friendly per-variant configs (lora, ortholora, tlora, reft, hydralora, …).",
    )
    parser.add_argument(
        "--output_config",
        action="store_true",
        help="output command line args to given .toml file",
    )
    if support_dreambooth:
        parser.add_argument(
            "--prior_loss_weight",
            type=float,
            default=1.0,
            help="loss weight for regularization images",
        )


def add_masked_loss_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--conditioning_data_dir",
        type=str,
        default=None,
        help="conditioning data directory",
    )
    parser.add_argument(
        "--masked_loss", action="store_true", help="apply mask for calculating loss."
    )


def add_dit_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cache_text_encoder_outputs",
        action="store_true",
        help="cache text encoder outputs",
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        type=int,
        default=None,
        help="text encoder batch size (default: None, use dataset's batch size)",
    )
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=[
            "sigma_sqrt",
            "logit_normal",
            "mode",
            "cosmap",
            "apex_omega",
            "none",
            "uniform",
        ],
        help="weighting scheme for timestep distribution. Default is uniform",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean for logit_normal weighting scheme",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std for logit_normal weighting scheme",
    )
    parser.add_argument(
        "--mode_scale", type=float, default=1.29, help="Scale of mode weighting scheme"
    )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=None,
        help="[EXPERIMENTAL] Sets the number of blocks to swap during the forward and backward passes.",
    )


def add_network_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--network_weights",
        type=str,
        default=None,
        help="pretrained weights for network",
    )
    parser.add_argument(
        "--network_module",
        type=str,
        default=None,
        help="network module to train",
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network)",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version)",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons)",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value)",
    )
    parser.add_argument(
        "--network_train_unet_only",
        action="store_true",
        help="only training U-Net part",
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point)",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="path to a pretrained LoRA checkpoint to merge into DiT weights before training. "
        "Intended for postfix/prefix training on top of a fixed LoRA. "
        "The LoRA is baked into the base weights at load time — no runtime hooks.",
    )
    parser.add_argument(
        "--lora_multiplier",
        type=float,
        default=1.0,
        help="multiplier applied to the frozen LoRA merged via --lora_path",
    )


def get_sanitized_config_or_none(args: argparse.Namespace):
    if not args.log_config:
        return None

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]
    filtered_args = {}
    for k, v in vars(args).items():
        if k not in sensitive_args + sensitive_path_args:
            if (
                v is None
                or isinstance(v, bool)
                or isinstance(v, str)
                or isinstance(v, float)
                or isinstance(v, int)
            ):
                filtered_args[k] = v
            elif isinstance(v, list):
                filtered_args[k] = f"{v}"
            elif isinstance(v, object):
                filtered_args[k] = f"{v}"

    return filtered_args


def verify_command_line_training_args(args: argparse.Namespace):
    wandb_enabled = args.log_with is not None and args.log_with != "tensorboard"
    if not wandb_enabled:
        return

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
    ]

    for arg in sensitive_args:
        if getattr(args, arg, None) is not None:
            logger.warning(
                f"wandb is enabled, but option `{arg}` is included in the command line. It is recommended to move it to the `.toml` file."
            )

    for arg in sensitive_path_args:
        if getattr(args, arg, None) is not None and os.path.isabs(getattr(args, arg)):
            logger.info(
                f"wandb is enabled, but option `{arg}` is included in the command line and it is an absolute path. It is recommended to move it to the `.toml` file or use relative path."
            )

    if getattr(args, "config_file", None) is not None:
        logger.info(
            "wandb is enabled, but option `config_file` is included in the command line. Please be careful about the information included in the path."
        )

    if (
        args.huggingface_repo_id is not None
        and args.huggingface_repo_visibility != "public"
    ):
        logger.info(
            "wandb is enabled, but option huggingface_repo_id is included in the command line and huggingface_repo_visibility is not 'public'. It is recommended to move it to the `.toml` file."
        )


def enable_high_vram(args: argparse.Namespace):
    if args.highvram:
        logger.info("highvram is enabled")
        _datasets_base.enable_high_vram()


def verify_training_args(args: argparse.Namespace):
    enable_high_vram(args)

    if args.v2 and args.clip_skip is not None:
        logger.warning("v2 with clip_skip will be unexpected")

    if args.cache_latents_to_disk and not args.cache_latents:
        args.cache_latents = True
        logger.warning(
            "cache_latents_to_disk is enabled, so cache_latents is also enabled"
        )

    if args.adaptive_noise_scale is not None and args.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset")

    if args.scale_v_pred_loss_like_noise_pred and not args.v_parameterization:
        raise ValueError(
            "scale_v_pred_loss_like_noise_pred can be enabled only with v_parameterization"
        )

    if args.v_pred_like_loss and args.v_parameterization:
        raise ValueError("v_pred_like_loss cannot be enabled with v_parameterization")

    if args.zero_terminal_snr and not args.v_parameterization:
        logger.warning(
            "zero_terminal_snr is enabled, but v_parameterization is not enabled. training will be unexpected"
        )

    if args.sample_every_n_epochs is not None and args.sample_every_n_epochs <= 0:
        logger.warning(
            "sample_every_n_epochs is less than or equal to 0, so it will be disabled"
        )
        args.sample_every_n_epochs = None

    if args.sample_every_n_steps is not None and args.sample_every_n_steps <= 0:
        logger.warning(
            "sample_every_n_steps is less than or equal to 0, so it will be disabled"
        )
        args.sample_every_n_steps = None


def add_dataset_arguments(
    parser: argparse.ArgumentParser,
    support_dreambooth: bool,
    support_caption: bool,
    support_caption_dropout: bool,
):
    parser.add_argument(
        "--train_data_dir", type=str, default=None, help="directory for train images"
    )
    parser.add_argument(
        "--cache_info",
        action="store_true",
        help="cache meta information for faster dataset loading",
    )
    parser.add_argument(
        "--caption_separator", type=str, default=",", help="separator for caption"
    )
    parser.add_argument(
        "--caption_extension",
        type=str,
        default=".caption",
        help="extension of caption files",
    )
    parser.add_argument(
        "--caption_extention",
        type=str,
        default=None,
        help="extension of caption files (backward compatibility)",
    )
    parser.add_argument(
        "--keep_tokens",
        type=int,
        default=0,
        help="keep heading N tokens when shuffling caption tokens",
    )
    parser.add_argument(
        "--keep_tokens_separator",
        type=str,
        default="",
        help="A custom separator to divide the caption into fixed and flexible parts.",
    )
    parser.add_argument(
        "--secondary_separator",
        type=str,
        default=None,
        help="a secondary separator for caption.",
    )
    parser.add_argument(
        "--enable_wildcard", action="store_true", help="enable wildcard for caption"
    )
    parser.add_argument(
        "--caption_prefix", type=str, default=None, help="prefix for caption text"
    )
    parser.add_argument(
        "--caption_suffix", type=str, default=None, help="suffix for caption text"
    )
    parser.add_argument(
        "--color_aug", action="store_true", help="enable weak color augmentation"
    )
    parser.add_argument(
        "--flip_aug", action="store_true", help="enable horizontal flip augmentation"
    )
    parser.add_argument(
        "--face_crop_aug_range",
        type=str,
        default=None,
        help="enable face-centered crop augmentation and its range (e.g. 2.0,4.0)",
    )
    parser.add_argument("--random_crop", action="store_true", help="enable random crop")
    parser.add_argument(
        "--debug_dataset",
        action="store_true",
        help="show images for debugging (do not train)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default=None,
        help="resolution in training ('size' or 'width,height')",
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        help="cache latents to main memory to reduce VRAM usage",
    )
    parser.add_argument(
        "--vae_batch_size", type=int, default=1, help="batch size for caching latents"
    )
    parser.add_argument(
        "--cache_latents_to_disk",
        action="store_true",
        help="cache latents to disk to reduce VRAM usage",
    )
    parser.add_argument(
        "--skip_cache_check",
        action="store_true",
        help="skip the content validation of cache",
    )
    parser.add_argument(
        "--enable_bucket",
        action="store_true",
        help="enable buckets for multi aspect ratio training",
    )
    parser.add_argument(
        "--min_bucket_reso",
        type=int,
        default=256,
        help="minimum resolution for buckets",
    )
    parser.add_argument(
        "--max_bucket_reso",
        type=int,
        default=1024,
        help="maximum resolution for buckets",
    )
    parser.add_argument(
        "--bucket_reso_steps",
        type=int,
        default=64,
        help="steps of resolution for buckets",
    )
    parser.add_argument(
        "--bucket_no_upscale",
        action="store_true",
        help="make bucket for each image without upscaling",
    )
    parser.add_argument(
        "--resize_interpolation",
        type=str,
        default=None,
        choices=[
            "lanczos",
            "nearest",
            "bilinear",
            "linear",
            "bicubic",
            "cubic",
            "area",
        ],
        help="Resize interpolation",
    )
    parser.add_argument(
        "--token_warmup_min", type=int, default=1, help="start learning at N tags"
    )
    parser.add_argument(
        "--token_warmup_step",
        type=float,
        default=0,
        help="tag length reaches maximum on N steps",
    )
    parser.add_argument(
        "--alpha_mask",
        action="store_true",
        help="use alpha channel as mask for training",
    )
    parser.add_argument(
        "--dataset_class",
        type=str,
        default=None,
        help="dataset class for arbitrary dataset (package.module.Class)",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help=(
            "Global override applied to every subset's sample_ratio (0<r≤1). "
            "Unset = use each subset's own value. Exposed here so presets like "
            "`[half]` can propagate a single value across the dataset blueprint."
        ),
    )

    if support_caption_dropout:
        parser.add_argument(
            "--caption_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout caption(0.0~1.0)",
        )
        parser.add_argument(
            "--caption_dropout_every_n_epochs",
            type=int,
            default=0,
            help="Dropout all captions every N epochs",
        )
        parser.add_argument(
            "--caption_tag_dropout_rate",
            type=float,
            default=0.0,
            help="Rate out dropout comma separated tokens(0.0~1.0)",
        )

    if support_dreambooth:
        parser.add_argument(
            "--reg_data_dir",
            type=str,
            default=None,
            help="directory for regularization images",
        )

    if support_caption:
        parser.add_argument(
            "--in_json", type=str, default=None, help="json metadata for dataset"
        )
        parser.add_argument(
            "--dataset_repeats",
            type=int,
            default=1,
            help="repeat dataset when training with captions",
        )


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--save_model_as",
        type=str,
        default=None,
        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
        help="format to save the model (default is same to original)",
    )
    parser.add_argument(
        "--use_safetensors", action="store_true", help="use safetensors format to save"
    )
