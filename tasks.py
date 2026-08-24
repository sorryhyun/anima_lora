#!/usr/bin/env python3
"""Cross-platform task runner -- replaces Makefile for Windows compatibility.

Usage:
    python tasks.py <command> [extra args...]

Examples:
    python tasks.py lora
    python tasks.py lora --network_dim 32 --max_train_epochs 64
    python tasks.py test
    python tasks.py test                     # add SPECTRUM=1 to enable Spectrum
    python tasks.py test                     # add MOD=1 to enable modulation guidance
    python tasks.py test                     # add NOLORA=1 to run against the bare DiT
    python tasks.py download-models
    python tasks.py turbo                    # DP-DMD 4-step distillation
    python tasks.py exp-chimera              # experimental method
    python tasks.py test                     # add SPD=1 for progressive-resolution inference

Command implementations live under ``scripts/tasks/`` (shipped methods) and
``scripts/experimental_tasks/`` (unstable methods exposed under ``exp-*``).
This file is just a name → callable dispatch table.
"""

import importlib
import sys


class _LazyCmd:
    """A command callable that imports its module only when first invoked.

    tasks.py is a pure dispatch table: a single ``python tasks.py <cmd>`` needs
    exactly one command module, but importing all of them up front to build
    ``COMMANDS`` cost ~100ms (the daemon client's urllib/http chain dominates) —
    wasted for the common case (``make gui`` immediately spawns a child for the
    real work). Wrapping each entry defers the import to dispatch time, so every
    target stops paying for modules it won't run.
    """

    def __init__(self, modpath: str, name: str):
        self._modpath = modpath
        self._name = name

    def resolve(self):
        return getattr(importlib.import_module(self._modpath), self._name)

    def __call__(self, extra):
        return self.resolve()(extra)


class _LazyModule:
    """``<alias>.cmd_x`` in the COMMANDS table → a _LazyCmd, with no import yet."""

    def __init__(self, modpath: str):
        self._modpath = modpath

    def __getattr__(self, name: str) -> _LazyCmd:
        return _LazyCmd(self._modpath, name)


curate = _LazyModule("scripts.tasks.curate")
daemon = _LazyModule("scripts.tasks.daemon")
downloads = _LazyModule("scripts.tasks.downloads")
gui = _LazyModule("scripts.tasks.gui")
inference = _LazyModule("scripts.tasks.inference")
masking = _LazyModule("scripts.tasks.masking")
preprocess = _LazyModule("scripts.tasks.preprocess")
tagger = _LazyModule("scripts.tasks.tagger")
training = _LazyModule("scripts.tasks.training")
utilities = _LazyModule("scripts.tasks.utilities")
exp_inference = _LazyModule("scripts.experimental_tasks.inference")
exp_training = _LazyModule("scripts.experimental_tasks.training")

COMMANDS = {
    # ── Training ──────────────────────────────────────────────────────
    "lora": (
        training.cmd_lora,
        "LoRA family (lora|tlora|hydralora via configs/methods/lora.toml)",
    ),
    "register": (
        training.cmd_register,
        "Register-token adapter on a frozen DiT (DSR registers + self-attn QKV "
        "surface; configs/methods/register.toml). Kept-live at inference via the "
        "comfyui-anima-register node.",
    ),
    "lora-gui": (
        training.cmd_lora_gui,
        "Train from a self-contained configs/gui-methods/<variant>.toml "
        "(variant from GUI_PRESETS env or 1st positional; e.g. tlora, hydralora).",
    ),
    "soup": (
        training.cmd_soup,
        "Uncond-init soup (TARGET=<artist> required): uncond inter-train on the "
        "target's artist shard (reused if it exists) → 3 seeded fine-tunes → ΔW "
        "soup SVD-truncated to network_dim (bench/memorization/report.md).",
    ),
    "turbo": (
        training.cmd_turbo,
        "Turbo (DP-DMD) distillation — bakes CFG=4 / 28-step Anima into a 4-step "
        "LoRA student (configs/methods/turbo.toml). Single-GPU bespoke loop "
        "(bypasses train.py/accelerate). Output is a normal LoRA "
        "(https://huggingface.co/sorryhyun/anima-turbo-4step).",
    ),
    "easycontrol": (
        training.cmd_easycontrol,
        "EasyControl training (extended self-attn KV with VAE-encoded reference). "
        "EASYADAPTER=<task> (e.g. colorize) selects a control-task project.",
    ),
    "easycontrol-staging": (
        training.cmd_easycontrol_staging,
        "Generate an EasyControl adapter's staging dataset (no VAE/TE caching). "
        "EASYADAPTER=near_twin → mine the in-artist near-twin pair tree.",
    ),
    "easycontrol-preprocess": (
        training.cmd_easycontrol_preprocess,
        "Full EasyControl preprocess: latents + text emb. "
        "Source: easycontrol-dataset/  Cache: post_image_dataset/easycontrol/.",
    ),
    # ── Training daemon ───────────────────────────────────────────────
    "daemon": (
        daemon.cmd_daemon,
        "Start the local training-job daemon (idempotent; detached, waits for /health).",
    ),
    "daemon-status": (
        daemon.cmd_daemon_status,
        "Daemon status as JSON (health + resolved base_url + compact job "
        "summaries; --full for raw records, --job <id>/JOB=<id> for one record "
        "+ its result envelope). Passive — never starts a daemon; exit 1 when down.",
    ),
    "daemon-run": (
        daemon.cmd_daemon_run,
        'Run an arbitrary command on the GPU queue: ARGS="<script.py> [flags]". '
        "Attach-by-default (ctrl-C detaches, exits with the job's code); --queue "
        "detaches, --inline bypasses the daemon; --stall-timeout S (0=off) for a "
        "legitimately quiet loop.",
    ),
    "daemon-wait": (
        daemon.cmd_daemon_wait,
        "Block until JOB=<id> (or the active job) is terminal, print its record "
        "+ bench result envelope as JSON, and exit with the job's own exit code. "
        'ARGS="--timeout S" bounds the wait (exit 124).',
    ),
    "daemon-attach": (
        daemon.cmd_daemon_attach,
        "Follow the daemon (read-only). JOB=<id> tails that job's stdout; "
        "ctrl-C detaches only — training keeps running.",
    ),
    "daemon-pause": (
        daemon.cmd_daemon_pause,
        "Freeze the running job (or JOB=<id>) in place — SIGSTOP the process "
        "tree; VRAM stays allocated, SM util drops to zero, resume is instant. "
        "The queue does not advance past it.",
    ),
    "daemon-resume": (
        daemon.cmd_daemon_resume,
        "Thaw a paused job (or JOB=<id>) — SIGCONT the process tree back to running.",
    ),
    "daemon-kill": (
        daemon.cmd_daemon_kill,
        "Abort the running job (or JOB=<id>) and free the GPU; daemon stays up "
        "and starts the next queued job.",
    ),
    "daemon-terminate": (
        daemon.cmd_daemon_terminate,
        "Stop the daemon entirely (active job killed, GPU freed, queue discarded).",
    ),
    "daemon-prune": (
        daemon.cmd_daemon_prune,
        "Delete old terminal job dirs (the daemon also sweeps at boot). Dry-run "
        'unless ARGS="--apply"; --days N / --keep N override the retention knobs.',
    ),
    # ── Inference ─────────────────────────────────────────────────────
    "test": (
        inference.cmd_test,
        "Inference with latest LoRA. SPECTRUM=1 enables Spectrum acceleration; "
        "MOD=1 adds the latest distilled pooled_text_proj (modulation guidance); "
        "NOLORA=1 runs against the bare DiT (skips --lora_weight).",
    ),
    "gen": (
        inference.cmd_gen,
        "Batch generation via the daemon (attach-by-default): queues behind a "
        "live train run instead of OOM-colliding, survives the terminal, and "
        "lands a result manifest in the job record. Same argv/env levers as "
        "'test' (NOLORA/SPECTRUM/MOD/DAVE/FSG). --queue detaches; --inline "
        "bypasses the daemon. Target adapters/prompts/seeds via ARGS.",
    ),
    "test-hydra": (
        inference.cmd_test_hydra,
        "Inference with latest HydraLoRA moe (router-live)",
    ),
    "test-merge": (
        inference.cmd_test_merge,
        "Inference with latest *_merged.safetensors (MODEL_DIR=..., default 'output_temp')",
    ),
    "test-smc-cfg": (
        inference.cmd_test_smc_cfg,
        "Inference with latest LoRA + SMC-CFG (sliding-mode control CFG, arXiv:2603.03281). "
        "Honors SPECTRUM=1 / MOD=1 / NOLORA=1.",
    ),
    "test-easycontrol": (
        inference.cmd_test_easycontrol,
        "Inference with latest EasyControl weight. Usage: test-easycontrol <ref_image> [--prompt ... --easycontrol_scale ...]",
    ),
    "test-turbo": (
        inference.cmd_test_turbo,
        "Inference with latest Turbo student LoRA at 4 steps, cfg=1.0 "
        "(CFG is baked into the student).",
    ),
    # ── Preprocess ────────────────────────────────────────────────────
    "preprocess": (
        preprocess.cmd_preprocess,
        "Full preprocessing (resize + VAE + text embeddings + caption index "
        "when the tagger vocab is present)",
    ),
    "preprocess-config": (
        preprocess.cmd_preprocess_config,
        "Preprocess the dirs named in a --dataset_config TOML (resize --src "
        "→ image_dir, then VAE + TE caches → cache_dir). Used by the trainer node.",
    ),
    "preprocess-resize": (
        preprocess.cmd_preprocess_resize,
        "Resize images to bucket resolutions",
    ),
    "preprocess-reconcile": (
        preprocess.cmd_preprocess_reconcile,
        "Remove resized/latent/PE/mask caches stale for the configured "
        'target_res (dry-run; ARGS="--delete" to act). Run after changing tiers.',
    ),
    "preprocess-vae": (preprocess.cmd_preprocess_vae, "Cache VAE latents"),
    "preprocess-demote": (
        preprocess.cmd_preprocess_demote,
        "Emit σ-demote sibling latents (sigma_lowres: demoted_{H}x{W} keys "
        "inside the native npz) for --sigma_lowres training — one pass per "
        "route in preprocess.toml's sigma_demote (default 1024:896)",
    ),
    "preprocess-te": (preprocess.cmd_preprocess_te, "Cache text encoder embeddings"),
    "preprocess-captions": (
        preprocess.cmd_preprocess_captions,
        "Write corrected caption sidecars next to resized preprocessing images",
    ),
    "preprocess-pe": (
        preprocess.cmd_preprocess_pe,
        "Cache PE-Core vision-encoder features into the LoRA cache dir. "
        "Consumed by CMMD validation.",
    ),
    "preprocess-pe-spatial": (
        preprocess.cmd_preprocess_pe_spatial,
        "Cache PE-Spatial (dense, B16-512) patch tokens into the LoRA cache "
        "dir as {stem}_anima_pe_spatial.safetensors. Consumed by REPA v2.",
    ),
    "caption-index": (
        preprocess.cmd_caption_index,
        "Build the typed-tag caption index (character/copyright/artist groups) "
        "at post_image_dataset/captions/caption_index.json. Pure data, no GPU.",
    ),
    "caption-autotag": (
        preprocess.cmd_caption_autotag,
        "Auto-tag the dataset with the Anima Tagger and write .txt caption "
        'sidecars. ARGS="--mode missing|merge|overwrite" (default missing = '
        'only uncaptioned images). Dry-run by default; ARGS="--apply" writes, '
        "then `make preprocess-te` is REQUIRED.",
    ),
    "caption-position": (
        preprocess.cmd_caption_position,
        "Append position-aware clauses ('On the left, <tags>. ...') to "
        "multi-subject captions via SAM3 + Anima Tagger. Writes the resized "
        "captions (post_image_dataset/), never the image_dataset/ master. "
        'Dry-run by default; ARGS="--apply" writes, then `make preprocess-te` '
        "is REQUIRED.",
    ),
    # ── Curation ──────────────────────────────────────────────────────
    "curate-group": (
        curate.cmd_curate_group,
        "Group dataset images by PE-Spatial visual similarity (per-artist "
        "connected-components) → post_image_dataset/groups/groups.json. The GUI "
        'Dataset tab reads it to filter by group. ARGS="--threshold 0.95".',
    ),
    # ── Anima Tagger ──────────────────────────────────────────────────
    "preprocess-tagger": (
        tagger.cmd_preprocess_tagger,
        "Build the Anima Tagger vocab/manifest + cache PE-Core & PE-Spatial "
        "features (build_vocab + build_features). Needs CAPTION_CORPUS_DIR "
        "in .env.",
    ),
    "tagger": (
        tagger.cmd_tagger,
        "Train the dual-encoder hard-routed tagger head on cached PE-Core + "
        "PE-Spatial features. Requires `make preprocess-tagger` first.",
    ),
    "test-tagger": (
        tagger.cmd_test_tagger,
        "Predict tags for a single image (--image <path>) or sample a random "
        "val-split stem. Pass --show_scores for rating + top-K kept tags.",
    ),
    "autotag": (
        tagger.cmd_autotag,
        "Autotag one image (--image <path>) with the Anima Tagger "
        "(auto-downloaded on first use); prints the predicted caption. CLI "
        "one-shot — the GUI Dataset tab uses a resident worker instead.",
    ),
    # ── Downloads ─────────────────────────────────────────────────────
    "download-models": (downloads.cmd_download_models, "Download all models"),
    "download-anima": (downloads.cmd_download_anima, "Download Anima model"),
    "download-anima-variant": (
        downloads.cmd_download_anima_variant,
        "Download an alternate Anima base DiT (aesthetic / turbo / preview / "
        "2.9B); ARGS=<name>, no args lists them",
    ),
    "download-sam3": (downloads.cmd_download_sam3, "Download SAM3 model"),
    "download-mit": (downloads.cmd_download_mit, "Download MIT model"),
    "download-pe": (
        downloads.cmd_download_pe,
        "Download PE-Core-L14-336 (img2emb encoder)",
    ),
    "download-pe-spatial": (
        downloads.cmd_download_pe_spatial,
        "Download PE-Spatial-B16-512 (Anima Tagger aux encoder)",
    ),
    "download-tagger": (
        downloads.cmd_download_tagger,
        "Download Anima Tagger v2 vocab.json (caption-index dependency; not the full model)",
    ),
    "download-danbooru-tags": (
        downloads.cmd_download_danbooru_tags,
        "Download danbooru tag tables (KR base + EN sibling) for caption correction",
    ),
    # ── Masking ───────────────────────────────────────────────────────
    "mask": (
        masking.cmd_mask,
        "Run SAM + MIT (via tempdir) and write merged masks under post_image_dataset/masks/",
    ),
    "mask-clean": (
        masking.cmd_mask_clean,
        "Remove post_image_dataset/masks/",
    ),
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
    "test-unit": (utilities.cmd_test_unit, "Run smoke/unit tests (pytest tests/)"),
    "export-logs": (
        utilities.cmd_export_logs,
        "Dump TB scalar logs to JSON (RUN=<dir>, ALL=1 for every subrun, JSONL=1 for "
        "line-delimited, SUMMARY=1 for max-step + last value per tag)",
    ),
    "run-status": (
        utilities.cmd_run_status,
        "Step/ETA/last-ckpt one-liner for a run from its progress.jsonl "
        "(RUN=<output_name|path>, default newest; ARGS='--list|--json')",
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
    "vendor-sync": (
        utilities.cmd_vendor_sync,
        "Refresh custom_nodes/*/_vendor/ from live library/* (run before publishing nodes)",
    ),
    # ── Experimental ──────────────────────────────────────────────────
    # Unstable methods kept under exp-* so they don't pollute the main command
    # surface. May produce broken output, change without notice, or be removed.
    "exp-soft-tokens": (
        exp_training.cmd_soft_tokens,
        "[experimental] SoftREPA-style per-layer × per-t soft tokens (training-only v1)",
    ),
    "exp-chimera": (
        exp_training.cmd_chimera,
        "[experimental] ChimeraHydra dual-pool additive routing "
        "(content + freq pools on OrthoHydra; configs/methods/chimera.toml)",
    ),
    "exp-byg": (
        exp_training.cmd_byg,
        "[experimental] BYG unpaired instruction-editing training (plain LoRA, "
        "bootstrap + DDS prior + cycle + identity; configs/methods/byg.toml). "
        "Run exp-byg-data first.",
    ),
    "exp-byg-data": (
        exp_training.cmd_byg_data,
        "[experimental] Build BYG edit-tuple sidecars (tag-swap) into "
        "post_image_dataset/byg/. Usage: exp-byg-data [--limit N --overwrite].",
    ),
    "exp-cjk-cache": (
        exp_training.cmd_cjk_cache,
        "[experimental] Stage the CJK distillation cache (Qwen hidden states + "
        "frozen-teacher adapter outputs) from post_image_dataset/cjk_distill/"
        "pairs.jsonl. Reused by every exp-distill-cjk arm.",
    ),
    "exp-distill-cjk": (
        exp_training.cmd_distill_cjk,
        "[experimental] Distill the extended CJK T5-vocab rows against the "
        "en-translation teacher. Gates in order: ARGS='--mode oracle' → "
        "'--mode capacity' → '--mode train'. Emits a vocab pack, not a LoRA.",
    ),
    "exp-cjk-gates": (
        exp_training.cmd_cjk_gates,
        "[experimental] CJK Phase-2b closing gates: G3 (teacher ceiling per "
        "register — is the 2c cos>=0.6 gate even the right number?) and G4 "
        "(corpus health + trust ablation). ARGS='--gates g3,g4a,g4b'.",
    ),
    "exp-test-soft": (
        exp_inference.cmd_test_soft,
        "[experimental] Inference with latest soft_tokens weight "
        "(SoftREPA-style per-layer × per-t bank, spliced into cross-attn via "
        "monkey-patched Block.forward). Composes freely with --spectrum.",
    ),
    "exp-test-byg": (
        exp_inference.cmd_test_byg,
        "[experimental] Inference with latest BYG editing LoRA. Usage: exp-test-byg <ref_image> --prompt 'change background to a forest'",
    ),
    "exp-test-directedit": (
        exp_inference.cmd_test_directedit,
        "[experimental] DirectEdit on a random source image. PROMPT='...' supplies the edit "
        "instruction (appended to the Anima Tagger source caption). REF_IMAGE=path overrides the "
        "random pick. Usage: exp-test-directedit [ref_image] [extra...]",
    ),
    "exp-test-directedit-dry": (
        exp_inference.cmd_test_directedit_dry,
        "[experimental] DirectEdit functional check: random source image + random crossattn "
        "embed (no TE, no captioner); ψ_tar == ψ_src so output should reconstruct the source. "
        "REF_IMAGE=path overrides the random pick. Usage: exp-test-directedit-dry [ref_image] [extra...]",
    ),
}


def _force_utf8_stdio():
    """Make stdout/stderr UTF-8 so non-UTF-8 consoles don't crash on glyphs.

    Several commands print Unicode status glyphs (``✓``/``✗``). On a Windows
    console whose code page isn't UTF-8 (e.g. cp949 on a Korean install)
    ``print`` raises ``UnicodeEncodeError`` and aborts the whole task. Re-encode
    stdio as UTF-8 with ``errors="replace"`` so output is never fatal — UTF-8
    when the terminal can show it, a replacement char at worst when it can't.
    Best-effort: some wrapped streams (pytest capture, certain pipes) lack
    ``reconfigure``; skip them silently.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main():
    _force_utf8_stdio()
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
        print(f"python tasks.py {command} -- {desc}\n")
        doc = fn.resolve().__doc__ if isinstance(fn, _LazyCmd) else fn.__doc__
        if doc:
            print(doc.strip())
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
