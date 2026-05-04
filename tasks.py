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

Command implementations live under ``scripts/tasks/`` (one module per
category). This file is just a name → callable dispatch table.
"""

import sys

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
    "apex": (training.cmd_apex, "APEX distillation (condition-shift self-adversarial)"),
    "postfix": (
        training.cmd_postfix,
        "Postfix/prefix tuning (mode selected in configs/methods/postfix.toml)",
    ),
    "ip-adapter": (
        training.cmd_ip_adapter,
        "IP-Adapter training (decoupled image cross-attention)",
    ),
    "ip-adapter-preprocess": (
        training.cmd_ip_adapter_preprocess,
        "Alias for `preprocess` + `preprocess-pe` (IP-Adapter reuses the LoRA "
        "pipeline's caches under post_image_dataset/lora/).",
    ),
    "easycontrol": (
        training.cmd_easycontrol,
        "EasyControl training (extended self-attn KV with VAE-encoded reference)",
    ),
    "easycontrol-preprocess": (
        training.cmd_easycontrol_preprocess,
        "Full EasyControl preprocess: latents + text emb. "
        "Source: easycontrol-dataset/  Cache: post_image_dataset/easycontrol/.",
    ),
    # ── Inference ─────────────────────────────────────────────────────
    "test": (inference.cmd_test, "Inference with latest LoRA"),
    "test-mod": (
        inference.cmd_test_mod,
        "Inference with latest pooled_text_proj (modulation guidance)",
    ),
    "test-apex": (inference.cmd_test_apex, "Inference with latest APEX LoRA"),
    "test-hydra": (
        inference.cmd_test_hydra,
        "Inference with latest HydraLoRA moe (router-live)",
    ),
    "test-prefix": (inference.cmd_test_prefix, "Inference with latest prefix weight"),
    "test-ref": (
        inference.cmd_test_ref,
        "Inference with latest reference-inversion prefix (output/ckpt/anima_ref*.safetensors)",
    ),
    "test-postfix": (inference.cmd_test_postfix, "Inference with latest postfix weight"),
    "test-postfix-exp": (
        inference.cmd_test_postfix_exp,
        "Inference with latest postfix-exp weight",
    ),
    "test-postfix-func": (
        inference.cmd_test_postfix_func,
        "Inference with latest postfix-func weight",
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
        "Calibrate DCW v4: sample 3 aspect buckets (default 80×3 seeds) + train fusion head",
    ),
    "dcw-train": (
        dcw.cmd_dcw_train,
        "Train-only on existing bench/dcw/results/ pool (~30s, no sampling)",
    ),
    "test-spectrum-dcw": (
        inference.cmd_test_spectrum_dcw,
        "Spectrum-accelerated inference + DCW post-step bias correction",
    ),
    "test-ip": (
        inference.cmd_test_ip,
        "Inference with latest IP-Adapter weight. Usage: test-ip <ref_image> [--prompt ... --ip_scale ...]",
    ),
    "test-easycontrol": (
        inference.cmd_test_easycontrol,
        "Inference with latest EasyControl weight. Usage: test-easycontrol <ref_image> [--prompt ... --easycontrol_scale ...]",
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
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python tasks.py <command> [extra args...]\n")
        print("Commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:20s} {desc}")
        print("\nExtra arguments are forwarded to the underlying command.")
        print("Example: python tasks.py lora --network_dim 32 --max_train_epochs 64")
        sys.exit(0)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Run 'python tasks.py --help' for available commands", file=sys.stderr)
        sys.exit(1)

    extra = sys.argv[2:]
    fn, _ = COMMANDS[command]
    fn(extra)


if __name__ == "__main__":
    main()
