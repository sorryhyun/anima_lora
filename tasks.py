#!/usr/bin/env python3
"""Cross-platform task runner — replaces Makefile for Windows compatibility.

Usage:
    python tasks.py <command> [extra args...]

Examples:
    python tasks.py lora
    python tasks.py lora --network_dim 32 --max_train_epochs 64
    python tasks.py test
    python tasks.py test-spectrum
    python tasks.py download-models
    python tasks.py exp-apex                 # experimental method
    python tasks.py exp-test-ip ref.png      # experimental inference

Command implementations live under ``scripts/tasks/`` (shipped methods) and
``scripts/experimental_tasks/`` (unstable methods exposed under ``exp-*``).
This file is just a name → callable dispatch table.
"""

import sys

from scripts.experimental_tasks import inference as exp_inference
from scripts.experimental_tasks import training as exp_training
from scripts.tasks import (
    dcw,
    downloads,
    gui,
    inference,
    masking,
    preprocess,
    training,
    utilities,
)

COMMANDS = {
    # ── Training ──────────────────────────────────────────────────────
    "lora": (
        training.cmd_lora,
        "LoRA family (lora|tlora|tlora_rf|hydralora via configs/methods/lora.toml)",
    ),
    "lora-gui": (
        training.cmd_lora_gui,
        "Train from a self-contained configs/gui-methods/<variant>.toml "
        "(variant from GUI_PRESETS env or 1st positional; e.g. tlora, hydralora, reft, postfix_exp).",
    ),
    # ── Inference ─────────────────────────────────────────────────────
    "test": (inference.cmd_test, "Inference with latest LoRA"),
    "test-mod": (
        inference.cmd_test_mod,
        "Inference with latest pooled_text_proj (modulation guidance)",
    ),
    "test-hydra": (
        inference.cmd_test_hydra,
        "Inference with latest HydraLoRA moe (router-live)",
    ),
    "test-merge": (
        inference.cmd_test_merge,
        "Inference with latest *_merged.safetensors (MODEL_DIR=..., default 'output_temp')",
    ),
    "test-spectrum": (inference.cmd_test_spectrum, "Spectrum-accelerated inference"),
    "test-dcw": (
        inference.cmd_test_dcw,
        "Inference with latest LoRA + DCW post-step bias correction",
    ),
    "test-dcw-v4": (
        inference.cmd_test_dcw_v4,
        "Inference with latest LoRA + DCW v4 learnable calibrator (auto-resolves fusion_head.safetensors)",
    ),
    "dcw": (
        dcw.cmd_dcw,
        "Calibrate DCW v4: sample 5 aspect buckets (default 130×1 seeds, "
        "shuffle_seed=0) + train fusion head",
    ),
    "dcw-train": (
        dcw.cmd_dcw_train,
        "Train-only on existing bench/dcw/results/ pool (~30s, no sampling)",
    ),
    "test-spectrum-dcw": (
        inference.cmd_test_spectrum_dcw,
        "Spectrum-accelerated inference + DCW post-step bias correction",
    ),
    "test-dcw-v4-spectrum": (
        inference.cmd_test_dcw_v4_spectrum,
        "Spectrum-accelerated inference + DCW v4 learnable calibrator (auto-resolves fusion_head.safetensors)",
    ),
    # ── Preprocess ────────────────────────────────────────────────────
    "preprocess": (
        preprocess.cmd_preprocess,
        "Full preprocessing (resize + VAE + text embeddings)",
    ),
    "preprocess-resize": (
        preprocess.cmd_preprocess_resize,
        "Resize images to bucket resolutions",
    ),
    "preprocess-vae": (preprocess.cmd_preprocess_vae, "Cache VAE latents"),
    "preprocess-te": (preprocess.cmd_preprocess_te, "Cache text encoder embeddings"),
    "preprocess-pe": (
        preprocess.cmd_preprocess_pe,
        "Cache PE-Core (or other registered) vision-encoder features into the "
        "LoRA cache dir. Consumed by REPA (--use_repa) and IP-Adapter live-disk "
        "mode. PE_ENCODER=pe|pe-g.",
    ),
    # ── Downloads ─────────────────────────────────────────────────────
    "download-models": (downloads.cmd_download_models, "Download all models"),
    "download-anima": (downloads.cmd_download_anima, "Download Anima model"),
    "download-sam3": (downloads.cmd_download_sam3, "Download SAM3 model"),
    "download-mit": (downloads.cmd_download_mit, "Download MIT model"),
    "download-tipsv2": (
        downloads.cmd_download_tipsv2,
        "Download TIPSv2-L/14 (img2emb encoder)",
    ),
    "download-pe": (
        downloads.cmd_download_pe,
        "Download PE-Core-L14-336 (img2emb encoder)",
    ),
    "download-pe-g": (
        downloads.cmd_download_pe_g,
        "Download PE-Core-G14-448 (larger img2emb encoder)",
    ),
    # ── Masking ───────────────────────────────────────────────────────
    "mask": (masking.cmd_mask, "Generate SAM3 + MIT masks, then merge"),
    "mask-sam": (masking.cmd_mask_sam, "Generate SAM3 masks only"),
    "mask-mit": (masking.cmd_mask_mit, "Generate MIT masks only"),
    "mask-clean": (masking.cmd_mask_clean, "Remove all generated masks"),
    # ── GUI ───────────────────────────────────────────────────────────
    "gui": (gui.cmd_gui, "Launch PySide6 GUI"),
    "gui-shortcut": (
        gui.cmd_gui_shortcut,
        "Create a Windows desktop shortcut that launches the GUI (no console window)",
    ),
    # ── Utilities ─────────────────────────────────────────────────────
    "merge": (
        utilities.cmd_merge,
        "Bake latest LoRA (ADAPTER_DIR=..., default 'output/ckpt') into base DiT",
    ),
    "comfy-batch": (utilities.cmd_comfy_batch, "Run ComfyUI batch workflow"),
    "distill-mod": (
        utilities.cmd_distill_mod,
        "Distill pooled_text_proj MLP for modulation guidance",
    ),
    "test-unit": (utilities.cmd_test_unit, "Run smoke/unit tests (pytest tests/)"),
    "export-logs": (
        utilities.cmd_export_logs,
        "Dump TB scalar logs to JSON (RUN=<dir>, ALL=1 for every subrun, JSONL=1 for line-delimited)",
    ),
    "print-config": (
        utilities.cmd_print_config,
        "Dump merged config (METHOD=<name> PRESET=<name>) with provenance",
    ),
    "update": (
        utilities.cmd_update,
        "Update from GitHub release (preserves datasets/output/models, prompts on "
        "config conflicts, runs uv sync). Pass --dry-run / --version v1.0 / --no-sync.",
    ),
    # ── Experimental ──────────────────────────────────────────────────
    # Unstable methods kept under exp-* so they don't pollute the main command
    # surface. May produce broken output, change without notice, or be removed.
    "exp-apex": (
        exp_training.cmd_apex,
        "[experimental] APEX distillation (condition-shift self-adversarial)",
    ),
    "exp-postfix": (
        exp_training.cmd_postfix,
        "[experimental] Postfix/prefix tuning (mode selected in configs/methods/postfix.toml)",
    ),
    "exp-ip-adapter": (
        exp_training.cmd_ip_adapter,
        "[experimental] IP-Adapter training (decoupled image cross-attention)",
    ),
    "exp-ip-adapter-preprocess": (
        exp_training.cmd_ip_adapter_preprocess,
        "[experimental] Alias for `preprocess` + `preprocess-pe` (IP-Adapter "
        "reuses the LoRA pipeline's caches under post_image_dataset/lora/).",
    ),
    "exp-easycontrol": (
        exp_training.cmd_easycontrol,
        "[experimental] EasyControl training (extended self-attn KV with VAE-encoded reference)",
    ),
    "exp-easycontrol-preprocess": (
        exp_training.cmd_easycontrol_preprocess,
        "[experimental] Full EasyControl preprocess: latents + text emb. "
        "Source: easycontrol-dataset/  Cache: post_image_dataset/easycontrol/.",
    ),
    "exp-test-apex": (
        exp_inference.cmd_test_apex,
        "[experimental] Inference with latest APEX LoRA",
    ),
    "exp-test-prefix": (
        exp_inference.cmd_test_prefix,
        "[experimental] Inference with latest prefix weight",
    ),
    "exp-test-ref": (
        exp_inference.cmd_test_ref,
        "[experimental] Inference with latest reference-inversion prefix (output/ckpt/anima_ref*.safetensors)",
    ),
    "exp-test-postfix": (
        exp_inference.cmd_test_postfix,
        "[experimental] Inference with latest postfix weight",
    ),
    "exp-test-postfix-exp": (
        exp_inference.cmd_test_postfix_exp,
        "[experimental] Inference with latest postfix-exp weight",
    ),
    "exp-test-postfix-func": (
        exp_inference.cmd_test_postfix_func,
        "[experimental] Inference with latest postfix-func weight",
    ),
    "exp-test-ip": (
        exp_inference.cmd_test_ip,
        "[experimental] Inference with latest IP-Adapter weight. Usage: exp-test-ip <ref_image> [--prompt ... --ip_scale ...]",
    ),
    "exp-test-easycontrol": (
        exp_inference.cmd_test_easycontrol,
        "[experimental] Inference with latest EasyControl weight. Usage: exp-test-easycontrol <ref_image> [--prompt ... --easycontrol_scale ...]",
    ),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python tasks.py <command> [extra args...]\n")
        print("Commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:30s} {desc}")
        print("\nExtra arguments are forwarded to the underlying command.")
        print("Example: python tasks.py lora --network_dim 32 --max_train_epochs 64")
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'python tasks.py --help' for available commands", file=sys.stderr)
        sys.exit(1)

    extra = sys.argv[2:]
    fn, desc = COMMANDS[command]
    if extra and extra[0] in ("-h", "--help"):
        print(f"python tasks.py {command} — {desc}\n")
        if fn.__doc__:
            print(fn.__doc__.strip())
        else:
            print("(no detailed help available)")
        print(
            "\nUnrecognised flags are forwarded verbatim to the underlying script. "
            "Run the underlying script with --help for its full flag set."
        )
        sys.exit(0)
    fn(extra)


if __name__ == "__main__":
    main()
