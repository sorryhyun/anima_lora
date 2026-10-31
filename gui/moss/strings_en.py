"""English strings for the MOSS-Image GUI; assembled in ``gui.moss.strings``."""

from __future__ import annotations

UI: dict[str, str] = {
    "window_title": "MOSS-Image LoRA",
    "model_missing": "Model not found: {path}\nRun `make download-model ARGS=moss-base` "
    "(ARGS=moss-sft for the SFT DiT), set Advanced → model_dir, or set "
    "ANIMA_MOSS_MODEL_DIR in .env.",
    "scan_stale": "{stale} captions changed since their text cache — the next "
    "preprocessing run re-encodes them.",
    "scan_dupes": "{n} images share a file name with one in another subfolder — "
    "their cache files are prefixed with the folder name.",
    "tab_i2i_preprocess": "i2i Preprocess",
    "tab_i2i_train": "i2i Train",
    "i2i_task": "Task",
    "i2i_no_tasks": "No i2i task configs under configs/moss/i2i/.",
    "i2i_guide": "An i2i LoRA learns to turn a reference image into a target: the "
    "reference rides every forward as clean latent tokens. Stage builds the "
    "pairs, Preprocess caches each target with its reference (and caption), Train "
    "fits the LoRA, Test renders new references with it. Each form starts from "
    "configs/moss/i2i/<task>/; only your changes are saved.",
    "i2i_task_colorize": "colorize: B&W manga page → colour. Staging mangafies the "
    "colour corpus; the page is a full-size reference and the caption carries its "
    'OCR\'d lines (no vision tower). Test with the prompt "comic".',
    "i2i_task_sanitize": "sanitize: removes text and speech bubbles. Staging mines "
    "in-artist near-twin pairs (texted / clean); the texted page is a full-size "
    "reference. Test with the page's caption minus its text tags.",
    "i2i_stage_group": "1 · Stage",
    "i2i_stage_hint": "Builds the task's pair tree under post_image_dataset/i2i/ "
    "from the dataset (make i2i-stage TASK={task}; knobs in the [staging] table "
    "of i2i_tasks/configs/{task}.toml). EasyControl reads the same tree.",
    "run_stage": "Stage",
    "i2i_cache_group": "2 · Preprocess",
    "i2i_train_group": "3 · Train",
    "i2i_chain": "Stage, preprocess, then train",
    "i2i_test_group": "4 · Test",
    "i2i_test_hint": "Renders each reference with the LoRA, shown beside it. "
    "ref_image is one image or a folder of images; ref_scale and ref_vision must "
    "match the cache the LoRA trained on.",
    "i2i_staged": "{n} staged pairs.",
    "i2i_not_staged": "Not staged yet: {path} is missing. Run Stage first.",
    "i2i_stage_next": "Staging done — preprocessing.",
    "i2i_need_ref": "Set ref_image to an image or a folder of images.",
    "i2i_need_prompt": "With ref_vision off, the test needs a prompt: the page's "
    "caption.",
    "test_ref": "reference",
    "model_help_test": "The DiT to render with (base / sft) and how it runs. bf16 "
    "(default): 10.7 GiB at 1024². int8: its block weights run as int8 on an fp32 stream, "
    "7.8 GiB at 1024²; the Base file downloads on first use, sft rounds to nearest. It "
    "refuses loha and block swap. Changing between them sets int8, stream and "
    "blocks_to_swap, which stay editable under Advanced.",
    "model_help_i2i": "The DiT to train (base / sft) and how it runs. bf16 (default "
    "here): this task's measured settings. int8: swap 20 at budget 0.1, which fit "
    "sanitize's largest sample at a 12.3 GB peak over six steps; no i2i LoRA has been "
    "trained on it yet. It refuses loha. Changing between them sets int8, stream, "
    "ema_card, blocks_to_swap, swap_h2d_only and activation_memory_budget, which stay "
    "editable under Advanced.",
    "model_help": "The DiT to train (base / sft) and how it runs. int8 (default): its "
    "block weights run as int8, about 1.1 s/step at a 13 GB peak, and 4.4 GiB of system "
    "memory stays pinned; the Base file downloads on first use, sft rounds to nearest. "
    "It refuses loha. bf16: about 1.6 s/step at the same peak with 36 blocks swapped, "
    "and it runs loha. Changing between them sets int8, stream, ema_card, "
    "blocks_to_swap, swap_h2d_only and activation_memory_budget, which stay editable "
    "under Advanced.",
}
