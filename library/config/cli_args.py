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
import os

from library.datasets import base as _datasets_base
from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _optional_bool(value):
    """Parse a tri-state flag from the CLI: 'auto'/'' → None, else a bool.

    TOML-sourced values arrive already typed (bool/None) and bypass this, so it
    only needs to cope with the string forms argparse hands over on the CLI."""
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in ("", "auto", "none"):
        return None
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false/auto, got {value!r}")


def add_sd_models_arguments(parser: argparse.ArgumentParser):
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
        "DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD.",
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
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts",
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
        "--progress_jsonl",
        type=str,
        default=None,
        help="path to the structured progress JSONL event stream. Unset → derive "
        "<output_dir>/<output_name>.progress.jsonl (default on). Pass an empty string "
        "/ 'none' / 'off' to disable.",
    )
    parser.add_argument(
        "--huggingface_token",
        type=str,
        default=None,
        help="huggingface token",
    )
    parser.add_argument(
        "--resume_from_huggingface", action="store_true", help="resume from huggingface"
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default="bf16",
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving (None → save in the training weight dtype)",
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
        "--target_res",
        type=int,
        nargs="+",
        default=None,
        metavar="EDGE",
        help=(
            "Multi-scale constant-token tiers the dataset was preprocessed with "
            "(512 768 896 1024 1280 1536). Drives BOTH the training bucket table "
            "(union of these tiers — list every tier on disk or its caches get "
            "AR-snapped into a 1024 bucket and never loaded) and the torch.compile "
            "dynamo cache budget so multi-tier training does not recompile-storm. "
            "Default (unset) = single 1024 tier."
        ),
    )
    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="inductor",
        choices=["eager", "inductor"],
        help="torch.compile backend. 'inductor' (default) is the production path; "
        "'eager' disables compilation for debugging. (The wider Accelerate backend "
        "list was never used on Anima.)",
    )
    parser.add_argument(
        "--compile_inductor_mode",
        type=str,
        default=None,
        choices=[
            None,
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        help="Inductor preset forwarded as torch.compile(..., mode=...). "
        "'reduce-overhead' enables CUDAGraphs — requires stable tensor addresses "
        "across steps and is incompatible with block swap.",
    )
    parser.add_argument(
        "--activation_memory_budget",
        type=float,
        default=1.0,
        help="torch.compile AOT partitioner budget (torch._functorch.config."
        "activation_memory_budget): fraction of the default saved-for-backward "
        "activation memory the min-cut partitioner may keep; below 1.0 it "
        "recomputes cheap intermediates in backward instead. 0.85 restores the "
        "pre-2026-06-10 (custom-autograd-era) no-grad-ckpt training footprint "
        "at identical step time. 1.0 = torch default (no cap). Only applies "
        "when torch_compile is on; ignored (with a log line) under "
        "gradient_checkpointing — repartitioning breaks checkpoint's "
        "recompute-graph match (torch #166926) and ckpt already minimizes "
        "saved activations.",
    )
    parser.add_argument(
        "--partitioner_recompute_views",
        action="store_true",
        help="torch._functorch.config.recompute_views: let the AOT min-cut "
        "partitioner recompute view ops in backward instead of saving them "
        "(views are free to recompute but saving one pins its base tensor "
        "alive). Lowers the saved-for-backward set without engaging the "
        "activation_memory_budget knapsack. Ignored (with a log line) under "
        "gradient_checkpointing — same repartitioning hazard as the budget.",
    )
    parser.add_argument(
        "--partitioner_aggressive_recomputation",
        action="store_true",
        help="torch._functorch.config.aggressive_recomputation: drop the "
        "min-cut partitioner's ban-recompute heuristics so more op classes "
        "may be recomputed in backward when the cut is cheap. Can trade "
        "backward time for VRAM — see bench/freefit_vram before adopting. "
        "Ignored (with a log line) under gradient_checkpointing.",
    )
    parser.add_argument(
        "--compile_dynamic_seq",
        action="store_true",
        help="Marks the sequence-length axis dynamic (torch._dynamo.mark_dynamic), "
        "keeping all other dims static, instead of one static graph per "
        "token-count family. Under native_flatten the only varying in-block dim "
        "is seq_len (x dim 2 + the RoPE cos/sin), so a single graph symbolic in "
        "seq_len alone covers every token bucket (4032/4200/3024/3000/...) — "
        "tighter than blanket dynamic=True. Collapses the N-graph compile cascade "
        "— and its CUDA-context VRAM peak — to one graph. Off by default; bench "
        "graph-count / mem_get_info peak / step-time / bit-exactness before "
        "trusting it on a new config.",
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
        default=1,
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
        default=1,
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
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="use mixed precision (Anima trains in bf16; fp16/no are untested)",
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
        "--log_every_n_steps",
        type=int,
        default=1,
        help="only emit step-level metrics every N global steps (default 1 = every step). "
        "Validation and epoch logs are unaffected. Useful for long W&B runs.",
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
        "--sigma_lowres",
        action="store_true",
        help="σ-conditional low-res training (sigma_lowres Phase 1b): when a "
        "step's σ draw exceeds --sigma_lowres_threshold, train on the 896-tier "
        "sibling latent (the demoted_* npz key emitted by `make "
        "preprocess-demote`) instead of the native 1024-tier latent. Only the "
        "measured-safe 1024→896 route; images native to other tiers train "
        "as-is. Opt-in; plain train.py methods only (bespoke adapter loops "
        "need their own operating-point probe).",
    )
    parser.add_argument(
        "--sigma_lowres_threshold",
        type=float,
        default=0.5,
        help="σ gate for --sigma_lowres: demote only when every σ in the batch "
        "exceeds this. 0.5 is the measured-safe boundary for 1024→896 — lower "
        "values are OUTSIDE the probe's safe region.",
    )
    parser.add_argument(
        "--sigma_lowres_threshold_max",
        type=float,
        default=None,
        metavar="SIGMA",
        help="optional upper σ bound for --sigma_lowres: demote only when "
        "every σ in the batch is BELOW this too, turning the half-line gate "
        "into a window (threshold, threshold_max). The measured demote-safe "
        "region is a per-route window — e.g. 768's least-liability region is "
        "~(0.65, 0.95), with the σ=1 endpoint elevated again. Default: no "
        "upper bound (the shipped 1024:896 half-line gate).",
    )
    parser.add_argument(
        "--sigma_lowres_route",
        type=str,
        default="1024:896",
        metavar="NATIVE:DEMOTE",
        help="demote route for --sigma_lowres as native:demote edge. The "
        "default 1024:896 is the only measured-safe route; other values "
        "(e.g. 1024:768 — the E4 negative-control arm) are for probe/control "
        "runs only. The sibling latents must have been emitted with the SAME "
        'route (`make preprocess-demote ARGS="--sigma_demote N:D"`) — a '
        "missing key degrades that batch to native with a warn-once.",
    )
    parser.add_argument(
        "--sigma_lowres_yarnsig",
        type=str,
        nargs="?",
        const="1,4,0.35,2",
        default=None,
        metavar="ALPHA,BETA,CENTER,GAMMA",
        help="on demoted steps, build RoPE with the σ-gated YaRN banded "
        "alignment (yarnsig). Spatial frequency bands with fewer than "
        "ALPHA·μ(σ) rotations across the demoted extent get the full PI "
        "stretch to native coordinates, bands above BETA·μ(σ) keep native "
        "spacing, linear ramp between; μ(σ) = sigmoid(GAMMA·[logit(σ) − "
        "logit(CENTER)]) is SigMa's boundary gate. No second σ-threshold — "
        "the gate lives inside the rope schedule; native steps are untouched. "
        "ON BY DEFAULT whenever --sigma_lowres is set, at the probe's "
        "operating point (1,4,0.35,2) — it is part of the shipped combo "
        "recipe. Pass ALPHA,BETA,CENTER,GAMMA to move the operating point, or "
        "`off` to disable it. Applies to PRIMARY-rule demotes only.",
    )
    parser.add_argument(
        "--sigma_lowres_route2",
        type=str,
        default=None,
        metavar="NATIVE:DEMOTE",
        help="secondary demote rule (E16 stacked router), PRIORITY over the "
        "primary: when set, steps passing --sigma_lowres_threshold2/_max "
        "(and --sigma_lowres_span2) demote on THIS route instead — e.g. "
        "1024:768 inside its measured window (0.65, 0.95) stacked on the "
        "shipped σ>0.5 1024:896 rule. yarnsig applies to primary-rule "
        "demotes only. Sibling latents for BOTH routes must be emitted.",
    )
    parser.add_argument(
        "--sigma_lowres_threshold2",
        type=float,
        default=None,
        help="lower σ bound for the secondary rule (--sigma_lowres_route2). "
        "REQUIRED with route2 — the secondary rule has priority, so an "
        "unbounded gate here would shadow the primary rule at every σ (and "
        "silence yarnsig, which is primary-only).",
    )
    parser.add_argument(
        "--sigma_lowres_threshold2_max",
        type=float,
        default=None,
        metavar="SIGMA",
        help="upper σ bound for the secondary rule — window semantics as "
        "--sigma_lowres_threshold_max. REQUIRED with route2: the deep route's "
        "safe region is a WINDOW, so a half-line gate here demotes below the "
        "window where the measured gap is +0.25…+0.38 (badly off-map).",
    )
    parser.add_argument(
        "--sigma_lowres_span2",
        type=str,
        default=None,
        metavar="MODE[:FRAC]",
        help="step-span gate for the secondary rule — semantics as "
        "--sigma_lowres_span.",
    )
    parser.add_argument(
        "--sigma_lowres_span",
        type=str,
        default=None,
        metavar="MODE[:FRAC]",
        help="step-span gate on top of the σ gate (E16 placement probe): "
        "demote only inside a span of training progress. MODE early = first "
        "FRAC of train steps, late = last FRAC, spread = seed-keyed per-step "
        "coin with p=FRAC (deterministic in --seed + step index, touches no "
        "RNG stream — CRN pairing holds). FRAC defaults to 0.5. Steps outside "
        "the span train native; the σ draw itself is untouched.",
    )
    parser.add_argument(
        "--sigma_lowres_res_cond",
        action="store_true",
        help="E25b explicit resolution conditioning: embed the step's known "
        "grid scale s = log2(step_edge/native_edge) (0 on native steps, the "
        "route's config value on demoted steps) via the model's sinusoidal "
        "t-embedding function (dim 256) through a zero-init projection added "
        "to the t-embedding output before the adaln trunk. Trains only the "
        "projection (~0.5 MB) alongside the LoRA; zero-init ⇒ bit-exact to a "
        "control arm at step 0. Requires --sigma_lowres. The checkpoint "
        "carries the projection (not a ΔW — make merge refuses it).",
    )
    parser.add_argument(
        "--sigma_lowres_res_cond_centered",
        action="store_true",
        help="E25e re-centered variant of --sigma_lowres_res_cond (requires "
        "it): the delta becomes W·(φ(s) − φ(0)), so native (s = 0) forwards "
        "are bit-identical to a control arm for ANY projection value and "
        "contribute zero gradient to it. Removes, at the parameterization, "
        "the common-mode-offset channel the E25b post-close decomposition "
        "measured (~90 % of the trained lever was a resolution-independent "
        "W·φ(0) bias that shifted the native operating point). Checkpoint "
        "stamp: ss_sigma_lowres_res_cond = 'centered'.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Bit-exact reproducibility: deterministic flash-attn backward "
        "(the dK/dV atomic-add order is the one un-seedable noise source) + "
        "torch.use_deterministic_algorithms(warn_only) + cudnn determinism. "
        "Two runs of the identical command then produce identical "
        "checkpoints, so paired A/B endpoint deltas (--paired_step_rng) are "
        "pure treatment instead of riding chaos-amplified hardware wobble. "
        "Slightly slower backward; same GPU/driver/library versions assumed.",
    )
    parser.add_argument(
        "--paired_step_rng",
        action="store_true",
        help="Common-random-numbers mode for A/B arms: draw each train step's "
        "σ and noise from dedicated per-step-seeded generators (derived from "
        "--seed + step counter) instead of the global torch stream. Arms "
        "sharing a seed then see identical (data, σ, noise) sequences, so "
        "checkpoint differences isolate the intervention (e.g. --sigma_lowres "
        "thresholds) instead of noise-lottery divergence. Statistically "
        "equivalent to the default draw; use for paired comparisons.",
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
        "--sample_decode_inline",
        type=_optional_bool,
        default=None,
        help=(
            "Decode each round of sample latents to PNG right after that "
            "sampling event (per-epoch visibility) instead of batching the "
            "decode to the end of training. The inline decode parks the DiT on "
            "CPU while the VAE runs (never co-resident), so it's OOM-safe. "
            "true/false, or 'auto' (default): auto decodes inline unless the run "
            "is block-swapping (blocks_to_swap>0), where the deferred end-of-"
            "training decode avoids paying a full DiT CPU<->GPU round-trip per "
            "sample event on a tight card."
        ),
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
        help="method name under configs/methods/ (e.g. 'tlora', 'hydralora', 'chimera'). Merged after preset so method settings win on overlap.",
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
        help="subfolder under configs/ that holds the method file (default 'methods'). Use 'gui-methods' for the GUI-friendly per-variant configs (lora, ortholora, tlora, hydralora, …).",
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
        "--masked_loss", action="store_true", help="apply mask for calculating loss."
    )


def add_dit_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--use_text_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache text-encoder outputs to disk and read them during training "
        "(load TE → cache → free → load DiT). Set false for live encoding "
        "(e.g. IP-Adapter live mode). Subsumes the legacy "
        "cache_text_encoder_outputs{,_to_disk} pair.",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        type=int,
        default=None,
        help="text encoder batch size (default: None, use dataset's batch size)",
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
        "Intended for adapter training on top of a fixed LoRA. "
        "The LoRA is baked into the base weights at load time — no runtime hooks.",
    )
    parser.add_argument(
        "--lora_multiplier",
        type=float,
        default=1.0,
        help="multiplier applied to the frozen LoRA merged via --lora_path",
    )


def add_validation_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Validation seed for shuffling validation dataset, training `--seed` used otherwise",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset",
    )
    parser.add_argument(
        "--validation_split_num",
        type=int,
        default=0,
        help=(
            "Count-based validation split (number of held-out images). When "
            "set (>0), wins over the fractional `--validation_split`. Also "
            "determines how many samples CMMD evaluation generates per pass."
        ),
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation on validation dataset every N steps. By default, validation will only occur every epoch if a validation dataset is available",
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=None,
        help="Run validation dataset every N epochs. By default, validation will run every epoch if a validation dataset is available",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Max number of validation dataset items processed. By default, validation will run the entire validation dataset",
    )
    parser.add_argument(
        "--validation_sigmas",
        type=float,
        nargs="+",
        default=None,
        help="Sigma values for validation loss (0.0~1.0). Low values = fine detail. Default: 0.1 0.4 0.7. (Legacy FM-val path — unused under the CMMD val replacement.)",
    )
    parser.add_argument(
        "--validation_sample_steps",
        type=int,
        default=20,
        help="Denoising steps used by CMMD validation when sampling each held-out item. Default 20.",
    )
    parser.add_argument(
        "--validation_cfg_scale",
        type=float,
        default=1.0,
        help="CFG scale used by CMMD validation. Default 1.0 (no CFG, fastest). Bump to 4.0 to match production sampling but generation cost ~2×.",
    )
    parser.add_argument(
        "--use_cmmd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CMMD (PE-Core MMD²) as the validation signal. Set "
        "`use_cmmd = false` in the method TOML (or pass `--no-use_cmmd`) to "
        "skip CMMD and run only the legacy per-σ FM-MSE val pass — useful "
        "on tight VRAM where the PE encoder + sampling path doesn't fit.",
    )
    parser.add_argument(
        "--validation_fm_with_cmmd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also run the per-σ FM-MSE pass alongside a successful CMMD "
        "validation, logged under separate `loss/validation/fm_*` keys (CMMD "
        "stays the primary signal / best-ckpt selector). Set "
        "`validation_fm_with_cmmd = false` (or pass `--no-validation_fm_with_cmmd`) "
        "to skip the extra DiT forwards when you only want CMMD. No effect when "
        "`use_cmmd` is off — FM-MSE is already the primary signal then.",
    )
    parser.add_argument(
        "--validation_baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run each method adapter's validation baselines (e.g. IP-Adapter "
        "no_ip / shuffled_ref) as FM-MSE delta diagnostics during validation. "
        "Set `validation_baselines = false` in the method TOML (or pass "
        "`--no-validation_baselines`) to skip them — each baseline adds a full "
        "extra val forward per (batch, σ), so this roughly halves IP-Adapter "
        "validation time when you don't need the deltas.",
    )


def add_train_misc_arguments(parser: argparse.ArgumentParser):
    """Train-loop one-offs that don't belong to a larger flag group.

    Checkpoint-format / LR / resume-position knobs plus the config-provenance
    flags (``--print-config`` / ``--config-snapshot`` / ``--config-strict``).
    """
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save metadata in output model",
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors)",
    )
    parser.add_argument(
        "--unet_lr",
        type=float,
        default=None,
        help="learning rate for U-Net",
    )
    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=None,
        nargs="*",
        help="learning rate for Text Encoder, can be multiple",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + "",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + "",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers. "
        "Cannot be used with --blocks_to_swap.",
    )
    parser.add_argument(
        "--print-config",
        dest="print_config",
        action="store_true",
        help="Dump the fully merged config (base → preset → method → CLI) as TOML "
        "with provenance comments, then exit 0. Does not start training.",
    )
    parser.add_argument(
        "--config-snapshot",
        dest="config_snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write output/<output_name>.snapshot.toml next to the checkpoint on every real "
        "run (provenance + git SHA). Pass --no-config-snapshot to disable.",
    )
    parser.add_argument(
        "--config-strict",
        dest="config_strict",
        action="store_true",
        help="Treat config-schema warnings (unknown keys, off-list choices) as errors.",
    )


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


def enable_high_vram(args: argparse.Namespace):
    if args.highvram:
        logger.info("highvram is enabled")
        _datasets_base.enable_high_vram()


def verify_training_args(args: argparse.Namespace):
    enable_high_vram(args)

    # Expand the two semantic cache knobs into the legacy internal flags that
    # the dataset / strategy / metadata code still reads. `use_vae_cache` and
    # `use_text_cache` are the only config-facing surface; disk caching is the
    # only supported mode (RAM-only was never used), so the `_to_disk` siblings
    # always track their base flag. The old keys survive as schema aliases
    # (see library/config/schema.py) so pre-existing configs still resolve.
    args.cache_latents = args.cache_latents_to_disk = bool(args.use_vae_cache)
    args.cache_text_encoder_outputs = args.cache_text_encoder_outputs_to_disk = bool(
        args.use_text_cache
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
        "--debug_dataset",
        action="store_true",
        help="show images for debugging (do not train)",
    )
    parser.add_argument(
        "--use_vae_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache VAE-encoded latents to disk and read them during training. "
        "Set false for live VAE encoding (e.g. IP-Adapter live mode, where "
        "batch['images'] carries the raw reference). Subsumes the legacy "
        "cache_latents{,_to_disk} pair.",
    )
    parser.add_argument(
        "--vae_batch_size", type=int, default=1, help="batch size for caching latents"
    )
    parser.add_argument(
        "--skip_cache_check",
        action="store_true",
        help="skip the content validation of cache",
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
            "Unset or 1.0 = use each subset's own value. Exposed here so "
            "presets like `[half]` and the GUI data-scope field can propagate "
            "a single value across the dataset blueprint."
        ),
    )
    parser.add_argument(
        "--path_pattern",
        type=str,
        default=None,
        help=(
            "fnmatch glob applied to each image's path relative to its "
            "subset's image_dir. `|` separates alternatives — e.g. "
            "`char_a/*` keeps only the char_a/ subfolder, "
            "`char_a/*|char_b/*` keeps either. Unset / `*` = use everything. "
            "Validation and image-count thresholds honour the filtered pool."
        ),
    )
    parser.add_argument(
        "--artists_shard",
        type=str,
        default=None,
        help=(
            "Restrict training to one round-robin shard of the artist "
            "subdirectories under each subset's image_dir. Format `k_N` (e.g. "
            "`1_4` = shard 1 of 4): artists are sorted and artist i is assigned "
            "to shard i mod N, so each shard is a balanced cross-alphabet mix. "
            "Expands to a `<artist>/*|...` path_pattern at load — mutually "
            "exclusive with an explicit --path_pattern. Used by the "
            "lora_merge_interference partition-coherence sweep."
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
