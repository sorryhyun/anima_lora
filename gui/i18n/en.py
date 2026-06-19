"""English strings for the Anima LoRA GUI."""

from __future__ import annotations

STRINGS: dict[str, str] = {
    # Window / tabs
    "window_title": "Anima LoRA",
    "tab_config": "Training Config",
    "tab_easycontrol": "EasyControl",
    "tab_spd": "SPD",
    "tab_turbo": "Turbo",
    "tab_experimental": "Experimental",
    "tab_images": "Dataset Viewer",
    "tab_merge": "Merge",
    "tab_queue": "Queue",
    "tab_preprocess": "Preprocessing",
    "tab_tensorboard": "TensorBoard",
    # PreprocessingTab
    "preprocess_intro": (
        "Configure caption shuffling and text-bubble masking, then run each "
        "step on demand. The Training Config tab's Train button auto-runs "
        "preprocess with default settings when no cache exists — this tab "
        "is for tuning and for re-running individual steps."
    ),
    "preprocess_image_prep": "Image preprocessing (resize / filter)",
    "preprocess_source_image_dir": "Source image dir:",
    "preprocess_source_image_dir_tip": (
        "Base raw-image root for the selected GUI method (defaults to "
        "configs/preprocess.toml; edits save onto the variant). path_scope is "
        "appended on top at run time, so this is the unscoped root, not the final "
        "scoped path. Use the preprocess path filter below to run only part of "
        "this tree without changing where files are stored."
    ),
    "preprocess_path_pattern": "Preprocess path filter:",
    "preprocess_path_pattern_tip": (
        "path_scope is applied first to choose the effective source image root. "
        "For example, path_scope=data_group1 makes the preprocess root "
        "image_dataset/data_group1. This filter is then matched relative to "
        "that root. '*' (or blank) processes everything, '1/*' processes only "
        "data_group1/1, and '1/*|2/*' processes both subfolders."
    ),
    "preprocess_drop_lowres": "Drop low-resolution images",
    "preprocess_drop_lowres_tip": (
        "Skip source images smaller than the pixel threshold below so they "
        "never enter the resize / VAE / text caches. Uncheck to keep every "
        "image regardless of size."
    ),
    "preprocess_min_pixels": "Min pixels (filter threshold):",
    "preprocess_min_pixels_tip": (
        "Pixel-count threshold for the low-res filter. 500000 = 0.5MP. "
        "Ignored when 'Drop low-resolution images' is unchecked."
    ),
    "preprocess_target_res": "Resolution tiers (target_res):",
    "preprocess_freefit_max_ratio": "Max aspect ratio:",
    "preprocess_freefit_max_ratio_tip": (
        "Clamp for free-fit: aspect ratios beyond 1:R / R:1 are cover-cropped to "
        "the limit (honoring the crop position above). Default 4.0 matches the "
        "old bucket table's most-elongated reach and blocks degenerate 1:5 / 1:6 "
        "inputs."
    ),
    "preprocess_text_caching": "Caching (VAE + text)",
    "preprocess_caption_shuffle_variants": "Shuffle variants per caption (N):",
    "preprocess_caption_shuffle_variants_tip": (
        "Generate N caption variants per image. v0 is the pristine original; "
        "v1..v(N-1) are smart-shuffled and (if tag-dropout > 0) have non-prefix "
        "tags independently dropped. The dataloader picks v0 with 20% probability "
        "and v1..v(N-1) uniformly otherwise when use_shuffled_caption_variants=true. "
        "Set to 0 to cache a single pristine caption only."
    ),
    "preprocess_caption_tag_dropout_rate": "Tag dropout rate (0.0–1.0):",
    "preprocess_caption_tag_dropout_rate_tip": (
        "Per-tag dropout probability applied to v1..v(N-1). Tags up to and "
        "including the first @artist marker are never dropped. Ignored when "
        "shuffle variants ≤ 0."
    ),
    "preprocess_caption_correct_order": "Correct caption order",
    "preprocess_caption_correct_order_tip": (
        "Write corrected .txt captions next to resized images and use them for "
        "text-encoder caching. Original source captions are not modified."
    ),
    "preprocess_caption_insert_no_artist": "Insert @no-artist",
    "preprocess_caption_insert_no_artist_tip": (
        "When caption order correction is enabled, insert @no-artist at the "
        "artist slot only when there is no artist marker. A trigger word in the "
        "artist slot takes priority and suppresses @no-artist."
    ),
    "preprocess_caption_trigger_word": "Trigger word:",
    "preprocess_caption_trigger_word_tip": (
        "Optional trigger tag to move into the corrected caption. With the "
        "front option off it is placed in the artist slot before existing "
        "artist tags. Include @ if you want it to behave as an artist marker "
        "for shuffle/dropout protection. Underscores in trigger words are "
        "preserved."
    ),
    "preprocess_caption_trigger_at_front": "Keep trigger first",
    "preprocess_caption_trigger_at_front_tip": (
        "Place the trigger word at the very front of the caption. In this mode "
        "@no-artist insertion follows the normal no-artist option and existing "
        "artist tags."
    ),
    "preprocess_run_te": "Run caching (VAE + text)",
    "preprocess_run_pe": "Run PE caching",
    "preprocess_add_to_queue": "Add to queue",
    "preprocess_queued": "Queued {label} (job {job_id}) — watch it in the Queue tab.",
    "preprocess_masking_sam": "SAM3 masking (text bubbles)",
    "preprocess_masking_mit": "MIT masking (manga text)",
    "preprocess_sam_prompts": "SAM prompts (one per line):",
    "preprocess_sam_prompts_tip": (
        "Text prompts SAM3 looks for. One per line. Defaults to 'speech bubble' "
        "and 'text bubble'."
    ),
    "preprocess_sam_focus_prompts": "SAM focus prompts (one per line):",
    "preprocess_sam_focus_prompts_tip": (
        "Reversed polarity: subjects to KEEP. When set, the mask trains ONLY on "
        "these subjects and ignores everything else (e.g. 'girl' masks all "
        "background). Composes with the prompts above — final trainable region "
        "is the focus subject minus those ignore regions. Leave empty for the "
        "default ignore-only behaviour."
    ),
    "preprocess_sam_rule": "Mask rule",
    "preprocess_sam_add_rule": "+ Add rule",
    "preprocess_sam_add_rule_tip": (
        "Add another mask rule. Each rule targets a subset of images by path "
        "pattern; rules whose pattern matches an image compose together."
    ),
    "preprocess_sam_remove_rule": "Remove rule",
    "preprocess_sam_rule_path_pattern": "Path pattern (this rule):",
    "preprocess_sam_rule_path_pattern_tip": (
        "Which images this rule applies to — an fnmatch glob ('|'-OR-combined) "
        "on each image's path relative to the dataset root, e.g. 'character_a/*'. "
        "Empty or '*' matches every image (a catch-all default rule)."
    ),
    "preprocess_sam_threshold": "SAM threshold (0.0–1.0):",
    "preprocess_sam_threshold_tip": (
        "Minimum confidence for a SAM3 detection to be kept. Lower = more masks "
        "(may include false positives), higher = stricter. Default 0.5."
    ),
    "preprocess_dilate": "Dilate (px):",
    "preprocess_dilate_tip": (
        "Pixels of dilation applied to the binary mask. Larger values blur "
        "mask edges outward. Default 5. Set to 0 to disable."
    ),
    "preprocess_mit_threshold": "MIT text threshold (0.0–1.0):",
    "preprocess_mit_threshold_tip": (
        "Confidence threshold for the MIT/ComicTextDetector text segmenter. "
        "Default 0.8."
    ),
    "preprocess_mask_path_pattern": "Mask path filter:",
    "preprocess_mask_path_pattern_tip": (
        "fnmatch glob restricting which resized images get masked, matched on "
        "each path relative to post_image_dataset/resized. Scopes BOTH SAM and "
        "MIT. Same syntax as the training path_pattern: '*' (or blank) masks "
        "everything; 'char_a/*' one subfolder; 'char_a/*|char_b/*' to OR-combine."
    ),
    "preprocess_run_mask": "Run masking",
    "preprocess_run_sam_mask": "Run SAM masking",
    "preprocess_run_sam_mask_tip": (
        "Run SAM3 bubble segmentation as part of mask generation. "
        "Uncheck to skip SAM and use only MIT (or whichever other "
        "backends are enabled)."
    ),
    "preprocess_run_mit_mask": "Run MIT masking",
    "preprocess_run_mit_mask_tip": (
        "Run MIT/ComicTextDetector text segmentation as part of mask "
        "generation. Uncheck to skip MIT and use only SAM."
    ),
    "preprocess_mask_nothing_enabled": (
        "At least one of SAM or MIT masking must be enabled."
    ),
    "preprocess_status_resized": "Resized images: {n}",
    "preprocess_status_caches": "Caches — latents: {lat}, text: {te}, PE: {pe}",
    "preprocess_status_masks": "Masks: {masks}",
    "preprocess_status_no_resized": "No resized images yet.",
    "preprocess_open_dataset_dir": "Open cache folder",
    "preprocess_open_dataset_dir_tooltip": "Open the post_image_dataset/ folder (resized images + caches) in your file manager.",
    "preprocess_clear_scope_cache": "Delete current-scope cache",
    "preprocess_clear_scope_cache_tooltip": "Delete resized images and LoRA cache folders for the current path_scope.",
    "preprocess_clear_scope_cache_all_scope": "no scope / all",
    "preprocess_clear_scope_cache_empty": "No resized image or LoRA cache files to delete.",
    "preprocess_clear_scope_cache_outside_root": "The GUI will not delete paths outside the project folder:\n{path}",
    "preprocess_clear_scope_cache_confirm": (
        "Delete preprocess files for the current scope?\n\n"
        "scope: {scope}\n"
        "resize: {resized}\n  {resized_count} files\n"
        "lora: {lora}\n  {lora_count} files\n\n"
        "Run preprocessing again afterward to regenerate them."
    ),
    "preprocess_clear_scope_cache_done": "Deleted {count} preprocess files.",
    "preprocess_invalid_path_scope": "Invalid path_scope value: {value}",
    "preprocess_log_placeholder": "Preprocessing output will appear here...",
    "preprocess_save_settings": "Save",
    "preprocess_save_settings_tip": "Persist these settings to the selected GUI method profile. Mask runs receive the current profile's mask settings with the job.",
    "preprocess_settings_saved": "Preprocessing settings saved.",
    "preprocess_invalid_float": "Invalid number for {field}: {value}",
    "preprocess_already_running": "A preprocessing step is already running.",
    # ConfigTab
    "preset": "Preset:",
    "save": "Save",
    "save_dirty_tooltip": "Form has unsaved edits. Click Save to write them to the variant file (Train/Preprocess auto-saves first if you skip this).",
    "train": "Train",
    "train_tooltip": "Train the current variant now. Open the dropdown to queue it on the daemon instead (don't start now — add to the queue).",
    "train_busy_use_queue": "A job is already attached to this tab. Use the Train dropdown to queue another behind it, or Stop the current one first.",
    "queue": "Queue",
    "queue_tooltip": "Add the current variant to the daemon queue. Open the menu to choose train-after-preprocess or preprocess only.",
    "queue_train_preprocess": "Queue: Train + Preprocess",
    "queue_train_only": "Queue: Train only",
    "queue_preprocess_only": "Queue: Preprocess only",
    "test": "Test",
    "stop": "Stop",
    "log_placeholder": "Training output will appear here...",
    "copy_log": "Copy",
    "copy_log_tooltip": "Copy the full training log to the clipboard",
    "copy_log_done": "Copied",
    "gpu_probing": "GPU: probing…",
    "gpu_stat": "GPU{i}: {util}%  ·  {used}/{total} GiB  ·  {temp}°C",
    "from_base": "From base.toml",
    "saved": "Saved",
    "saved_file": "Saved {name}",
    "invalid_toml": "Invalid TOML",
    "config_bad_keys_header": "Unknown dataset keys — training will fail until these are removed:",
    "config_remove_keys_btn": "Remove",
    "config_remove_keys_confirm": "Delete these {n} stale key(s) from their config files?\n\n{keys}",
    "config_remove_keys_none": "No keys were removed (the flagged lines may have changed on disk).",
    "error": "Error",
    "accelerate_not_found": "accelerate not found on PATH",
    "preprocess": "Preprocess",
    "preprocess_current_tooltip": "Run preprocessing for the current variant using its GUI path scope.",
    "preprocess_required": "Preprocess will run before training starts.",
    "preprocess_existing_caches_title": "Existing caches will be reused",
    "preprocess_existing_caches_body": (
        "Cache files already exist in:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "Preprocess will REUSE these — they are NOT deleted or "
        "regenerated. Only missing entries will be processed.\n\n"
        "If you want to force a full rebuild (e.g. after editing "
        "captions or changing tokenizer settings), cancel and delete "
        "the cache directory first."
    ),
    "preprocess_cache_count_latents": "{n} VAE latents (.npz)",
    "preprocess_cache_count_te": "{n} text embeddings (_te.safetensors)",
    "preprocess_cache_count_pe": "{n} PE features (_pe.safetensors)",
    "train_using_cache_title": "Use cached dataset?",
    "train_using_cache_body": (
        "A preprocessed dataset cache already exists at:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "Training will reuse this cache as-is. If you've added new images "
        "or edited captions and want them included, cancel and run "
        "Preprocess first.\n\n"
        "Proceed with the existing cache?"
    ),
    "train_autopreprocess_log": (
        "No preprocessed cache found — running preprocess before training starts.\n"
    ),
    "train_preprocessing": "Preprocessing…",
    "no_lora_for_test": "No LoRA in output/ckpt/ to test. Run training first.",
    "test_output_title": "Latest test output",
    "test_output_empty": "output/tests/ is empty.",
    "sample_output_title": "Latest training samples",
    "sample_output_empty": "No samples yet — they appear under the output dir's sample/ folder as training generates them.",
    "sample_prompt_col_prompt": "Prompt",
    "sample_prompt_col_width": "W",
    "sample_prompt_col_height": "H",
    "sample_prompt_col_steps": "Steps",
    "sample_prompt_col_seed": "Seed",
    "sample_prompt_col_cfg": "CFG",
    "sample_prompt_col_guidance": "Guidance",
    "sample_prompt_col_shift": "Shift",
    "sample_prompt_col_negative": "Negative",
    "sample_prompt_col_extra": "Extra",
    "sample_prompt_add": "Add prompt",
    "sample_prompt_select_all": "Select all",
    "sample_prompt_remove": "Remove selected",
    "sample_prompt_remove_confirm_title": "Remove sample prompts",
    "sample_prompt_remove_confirm_body": "Remove {n} selected sample prompt(s)?",
    "sample_prompt_expand": "Expand editor",
    "sample_prompt_collapse": "Collapse editor",
    "sample_prompt_edit_button": "Edit sample prompts…",
    "sample_prompt_dialog_title": "Sample prompts",
    "sample_prompt_summary_none": "No sample prompts",
    "sample_prompt_summary_count": "{n} prompt(s) · {first}",
    "sample_prompt_select": "Select",
    "sample_prompt_prompt_placeholder": "Prompt text. Line breaks are shown here and saved as spaces.",
    "sample_prompt_hint": "Blank/default values are not written to the prompt line.",
    "sample_prompt_default_width": "default 512",
    "sample_prompt_default_height": "default 512",
    "sample_prompt_default_steps": "default 30",
    "sample_prompt_default_seed": "auto seed",
    "sample_prompt_default_cfg": "default 7.5",
    "sample_prompt_default_guidance": "default 1.0",
    "sample_prompt_default_shift": "default 3.0",
    "sample_prompt_default_negative": "default: none",
    "sample_prompt_tip_width": "Image width (`--w`). Blank uses train.py default 512.",
    "sample_prompt_tip_height": "Image height (`--h`). Blank uses train.py default 512.",
    "sample_prompt_tip_steps": "Sampling steps (`--s`). Blank uses train.py default 30.",
    "sample_prompt_tip_seed": "Seed (`--d`). Auto seed keeps each prompt comparable across epochs.",
    "sample_prompt_tip_cfg": "CFG scale (`--l`). Blank uses train.py default 7.5.",
    "sample_prompt_tip_guidance": "Guidance scale (`--g`). Blank uses train.py default 1.0.",
    "sample_prompt_tip_shift": "Flow shift (`--fs`) for the sampling sigma schedule. Blank uses train.py default 3.0.",
    "sample_prompt_tip_negative": "Negative prompt (`--n`) for this sample only.",
    "sample_prompt_tip_extra": "Additional raw sample arguments, preserved as typed.",
    "finished": "--- Finished (exit code {code}) ---",
    "starting": "Starting… (loading torch / accelerate)",
    # Daemon-backed training (Phase 2 — training survives GUI close)
    "daemon_submitting": "Submitting job to the training daemon…",
    "daemon_submit_failed": "Could not reach the training daemon: {err}",
    "daemon_queued": "Queued job {job_id} on the training daemon.\n",
    "queue_submitting": "Queueing {variant} on the training daemon…",
    "queue_submitting_train_preprocess": "Queueing preprocess + training for {variant} on the training daemon…",
    "queue_submitting_preprocess": "Queueing preprocess for {variant} on the training daemon…",
    "queue_added_train": "Queued {variant} as training job {job_id}.\n",
    "queue_added_preprocess": "Queued {variant} as preprocess job {job_id}; training will chain after it.\n",
    "queue_added_preprocess_only": "Queued {variant} as preprocess job {job_id}.\n",
    "queue_refresh": "Refresh",
    "queue_start": "Start Queue",
    "queue_pause": "Pause Queue",
    "queue_start_tooltip": "Start running the queued jobs (added via the queue dropdowns). Runs them one at a time.",
    "queue_pause_tooltip": "Hold the queue — the running job continues, but no further queued job starts until you press Start Queue.",
    "queue_stop_selected": "Stop selected",
    "queue_copy_output": "Copy output",
    "queue_status": "{live} live / {total} total jobs",
    "queue_status_paused": "{live} live / {total} total jobs — queue paused",
    "queue_daemon_unavailable": "Daemon unavailable",
    "queue_detail_placeholder": "Select a queue item to inspect its details.",
    "queue_log_placeholder": "Selected job output will appear here...",
    "queue_log_missing": "(No output log yet.)",
    "queue_log_read_failed": "(Could not read output log: {err})",
    "queue_log_truncated": "--- Showing the last {mb} MB of output ---\n",
    "queue_detail_id": "id: {job_id}",
    "queue_detail_state": "state: {state}",
    "queue_detail_kind": "kind: {kind}",
    "queue_detail_method": "method: {method}",
    "queue_detail_submitted": "submitted: {time}",
    "queue_detail_started": "started: {time}",
    "queue_detail_ended": "ended: {time}",
    "queue_detail_from_chain": "from_chain: true",
    "queue_detail_chain": "chains to train: {method}",
    "queue_detail_chained_id": "chained job: {job_id}",
    "queue_detail_pid": "pid: {pid}",
    "queue_detail_error": "error: {error}",
    "queue_detail_status_detail": "detail: {detail}",
    "queue_detail_config": "config: {path}",
    "queue_detail_stdout": "stdout: {path}",
    "daemon_reattached": "Re-attached to running job {job_id} (started in a previous session).\n",
    "daemon_job_finished": "--- Job {job_id} {state} ---",
    "daemon_job_failed": "--- Job {job_id} {state}: {error} ---",
    "daemon_error_cause": "↳ likely cause: {summary}",
    "train_queued": "Train (queued)",
    "train_running_daemon": "Train (running…)",
    "update_success_title": "Update applied",
    "update_success_message": (
        "anima_lora was updated to {v}.\n\n"
        "Close and relaunch the GUI to load the new code."
    ),
    "update_success_badge": "Updated → {v} (relaunch to apply)",
    "update_dryrun_done_title": "Dry run finished",
    "update_dryrun_done_message": (
        "Dry run completed — no files were written. "
        "Review the log to see what a real update would change."
    ),
    "update_failed_title": "Update failed",
    "update_failed_message": (
        "Update exited with code {code}. "
        "See the log for details; the working tree may be partially modified."
    ),
    "resume_checkpoint_title": "Resume training?",
    "resume_checkpoint_question": (
        "A resumable checkpoint was found at step {step}.\n\n"
        "• Yes — resume training from step {step}\n"
        "• No — discard the checkpoint and start fresh\n"
        "• Cancel — don't launch training"
    ),
    "resume_checkpoint_delete_failed": "Could not remove old checkpoint state:\n{error}",
    "locked_by_preset": "Locked by preset (performance settings are fixed for this VRAM profile)",
    "lora_variants": "LoRA Variants",
    "variant": "Variant:",
    "apply_variant": "Apply",
    "apply_variant_tooltip": "Fill the form below with this variant's preset values. Nothing is saved until you click Save.",
    "show_guide": "Guide",
    "show_guide_tooltip": "Show the variant guide and Apply-semantics note in the right panel.",
    "click_field_for_help": "Click a field label to see its explanation here.",
    "no_help_available": "No help available for this field.",
    "extra_args_toggle": "+ Extra args",
    "extra_args_placeholder": "TOML lines for fields not in the form, e.g.\nmy_new_flag = true\nsome_value = 5e-5",
    "extra_args_tooltip": "Add config keys not shown in the form. Parsed as TOML on Save and merged into the current variant file. The form reloads so new keys appear as widgets afterwards. Overrides a form widget if the same key appears in both.",
    "new_variant": "+ New",
    "new_variant_tooltip": "Create a new custom variant under configs/gui-methods/custom/<name>.toml.",
    "new_variant_prompt": "Name for the new variant (saved to configs/gui-methods/custom/<name>.toml).\nLetters, digits, _ and - only.",
    "new_variant_invalid": "Invalid name. Use letters, digits, _, - only.",
    "new_variant_exists": "Variant '{name}' already exists.",
    "basic_section": "Basic",
    "advanced_section": "Advanced (click to expand)",
    # SPD / Turbo distillation config tabs (gui/tabs/distill_tab.py)
    "distill_general_section": "general",
    "distill_job_running": "A job is already running on this tab.",
    "distill_config_missing": "Could not read the config file: {err}",
    "n_images": "{n} images",
    # ImageViewerTab
    "directory": "Directory:",
    "dataset_reload": "Reload",
    "dataset_reload_tooltip": "Re-scan the current directory and refresh the image list and selection.",
    "dataset_open_dir": "Open",
    "dataset_open_dir_tooltip": "Open the current directory in the system file manager.",
    "dataset_add_dir": "Add directory…",
    "dataset_add_dir_tooltip": "Pick another directory and add it to the dropdown for this session.",
    "dataset_add_dir_picker": "Pick a directory to add",
    "dataset_add_dir_already": "Directory '{name}' is already in the list.",
    "dataset_search_placeholder": "Search filename…",
    "dataset_sort_asc_tooltip": "Sort A→Z (click to reverse)",
    "dataset_sort_desc_tooltip": "Sort Z→A (click to reverse)",
    "dataset_group_first_tooltip": "Group-first ordering: float every group to the top, flattened across folders (ungrouped images follow below in the folder tree).",
    "dataset_view_group": "Groups",
    "dataset_view_tree": "Tree",
    "dataset_group_sort_tooltip": "Sort order inside each group. Arrow-key navigation follows the visible tree order.",
    "dataset_group_sort_name": "Name",
    "dataset_group_sort_name_desc": "Name desc",
    "dataset_group_sort_size": "File size",
    "dataset_group_sort_size_desc": "File size desc",
    "dataset_group_sort_resolution": "Resolution",
    "dataset_group_sort_resolution_desc": "Resolution desc",
    "dataset_mask_overlay": "Show mask overlay",
    "dataset_resize_preview": "Show resize preview",
    "dataset_resize_preview_tooltip": "Show the center-crop area and final bucket selected by preprocessing target_res. Source files are not modified.",
    "dataset_resize_preview_label": "{width}x{height} @ {edge}",
    "dataset_preprocess_use_short": "Use (A)",
    "dataset_preprocess_use_tooltip": "Mark the current image as included for preprocessing. Source files are not modified.",
    "dataset_preprocess_skip_short": "Skip (S)",
    "dataset_preprocess_skip_tooltip": "Mark the current image to be skipped by preprocess resize. Source files are not modified.",
    "dataset_preprocess_clear_short": "Clear (F)",
    "dataset_preprocess_clear_tooltip": "Clear the current image's use/skip/move mark. Use the right-side menu to clear all marks.",
    "dataset_preprocess_clear_all": "Clear all marks",
    "dataset_preprocess_save": "Save preprocess decisions",
    "dataset_preprocess_save_tooltip": "Save per-image use/skip/move decisions as JSON for preprocessing. Move marks are excluded from preprocessing even before files are moved.",
    "dataset_preprocess_saved": "Preprocess decisions saved:\n{path}",
    "dataset_preprocess_decision_use": "Preprocess decision: use",
    "dataset_preprocess_decision_skip": "Preprocess decision: skip",
    "dataset_preprocess_decision_move": "Current state: marked to move",
    "dataset_image_meta_empty": "No image",
    "dataset_image_meta": "{width}x{height} · {size} · {fmt}",
    "dataset_image_meta_resize": "Resize {width}x{height} @ {edge}",
    "dataset_delete": "Move (D)",
    "dataset_delete_tooltip": "Move the images marked with the Delete or D key to post_image_dataset/moved/, along with sidecars.",
    "dataset_delete_confirm_title": "Move images",
    "dataset_delete_confirm_body": "Move {n} image(s) and sidecars to post_image_dataset/moved/?",
    "dataset_delete_failed": "Some images could not be moved:\n{err}",
    "dataset_group_label": "Group {n} — {size} imgs",
    "dataset_group_rebuild": "Group",
    "dataset_group_rebuild_tooltip": "Group images by PE-Spatial visual similarity (per-artist). Runs in the job queue.",
    "dataset_group_queued": "Grouping queued (job {job_id}). Reload this directory when it finishes to see the groups.",
    "n_images_filtered": "{shown} / {total} images",
    "caption": "Caption:",
    "no_caption": "(no caption)",
    "caption_save": "Save",
    "caption_revert": "Revert",
    "caption_autotag": "Autotag",
    "caption_autotag_tooltip": (
        "Run the Anima Tagger on this image and append its predicted tags to "
        "the caption. The model is downloaded automatically on first use; "
        "review the result, then Save to write the .txt."
    ),
    "caption_autotag_running": "Autotagging…",
    "caption_autotag_loading": "Loading tagger…",
    "caption_autotag_ready": "Tagger loaded · ready",
    "caption_autotag_busy": (
        "The GPU is busy with another job (training / preprocessing / "
        "grouping). Try autotagging again once it finishes."
    ),
    "caption_autotag_error": "Autotag failed: {err}",
    "caption_autotag_empty": "The tagger returned no tags for this image.",
    "caption_correct": "Correct order",
    "caption_correct_tooltip": (
        "Use danbooru_tags_classified.csv to reorder this caption into the "
        "recommended ANIMA order, optionally inserting @no-artist."
    ),
    "caption_correct_visible": "Correct current list",
    "caption_correct_visible_confirm": "Correct {n} caption(s) in the current list?",
    "caption_correct_visible_done": "Corrected {n} caption(s).",
    "caption_correct_visible_failed": "Corrected {n} caption(s).\n\nFailed:\n{err}",
    "caption_correct_no_change": "No caption changes to apply.",
    "caption_correct_db_missing": (
        "danbooru_tags_classified.csv was not found.\n\n"
        "Download the Danbooru tag DB from the Models dialog, or place it here:\n{paths}"
    ),
    "caption_correct_db_failed": "Failed to load tag DB: {err}",
    "caption_versions": "Versions…",
    "caption_dirty_marker": " *",
    "caption_diff_stats": "(+{add} / −{rem})",
    "caption_diff_clean": "(no changes)",
    "caption_save_failed": "Failed to save caption: {err}",
    "caption_unsaved_title": "Unsaved caption",
    "caption_unsaved_body": "You have unsaved caption edits. Save before switching?",
    "caption_versions_title": "Caption history — {name}",
    "caption_versions_empty": "(no prior versions)",
    "caption_versions_restore": "Restore selected",
    "caption_versions_close": "Close",
    "caption_no_history": "No history yet for this caption.",
    "caption_guideline_html": (
        "<b>Order:</b> rating → count → character (series) → series → "
        "<span style='color:#c9a227;'>@artist</span> → content tags. "
        "Per-region sub-sections: end the previous tag with <code>.</code> and "
        "start the next with <span style='color:#5e8eb0;'>On the&nbsp;…,</span> "
        "or <span style='color:#5e8eb0;'>In the&nbsp;…,</span>. "
        "Tags up to and including the first <code>@artist</code> are kept fixed; "
        "everything after is shuffled within each section. "
        "<b>No artist?</b> Drop in "
        "<span style='color:#c9a227;'>@no-artist</span> as a placeholder — "
        "it anchors the shuffle boundary the same way and is stripped before "
        "tokenization, so it never reaches the model."
    ),
    # Language
    "language": "Language:",
    # Settings dialog
    "settings_btn": "⚙ Settings",
    "settings_btn_tooltip": "Application settings — language, preferences, MCP server registration",
    "settings_title": "Settings",
    "settings_prefs_header": "Preferences",
    "settings_autotag_confidence": "Autotag confidence:",
    "settings_autotag_confidence_tooltip": (
        "Extra probability floor (0–1) applied on top of the tagger's per-tag "
        "thresholds. Higher = fewer, more confident tags. Default 0.50."
    ),
    "settings_caption_insert_no_artist": "Insert @no-artist during caption correction",
    "settings_caption_insert_no_artist_tooltip": (
        "If the corrected caption has no artist tag, insert @no-artist at the "
        "artist position. It only anchors caption shuffle and is stripped before tokenization."
    ),
    "settings_caption_validate_artist_tags": "Validate artist tags with DB",
    "settings_caption_validate_artist_tags_tooltip": (
        "When enabled, only @tags classified as artists in danbooru_tags_classified.csv "
        "move to the artist position. When disabled, any @tag is treated as an artist tag."
    ),
    "settings_group_match_frac": "Grouping tightness:",
    "settings_group_match_frac_tooltip": (
        "Inlier fraction (0–1) needed to group two Dataset images when you press "
        "Group. Higher = tighter, cleaner groups. Default 0.25."
    ),
    "settings_group_cell_match": "Grouping cell match:",
    "settings_group_cell_match_tooltip": (
        "Per-cell cosine floor (0–1) for an inlier match during Dataset grouping. "
        "Higher = stricter cell agreement. Default 0.93."
    ),
    "settings_theme": "Theme:",
    "settings_theme_tooltip": (
        "Overall color theme for the interface. Applies immediately; the window "
        "rebuilds when you close Settings to fully repaint."
    ),
    "settings_theme_dark": "Dark",
    "settings_theme_light": "Light",
    "settings_theme_sepia": "Sepia",
    "settings_font_size": "Font size:",
    "settings_font_size_tooltip": (
        "Point size of the interface font. Applies immediately; the window "
        "rebuilds when you close Settings so every panel relayouts. Default 10."
    ),
    "settings_debug_mode": "Debug mode",
    "settings_debug_mode_tooltip": (
        "Log the training daemon at DEBUG level so a bug report can include why a "
        "job hung. Takes effect the next time the daemon starts (close the app, "
        "or stop the daemon, then reopen)."
    ),
    "settings_debug_report_desc": (
        "Stuck on an endless spinner? Enable Debug mode, reproduce it, then click "
        "Copy debug report and paste the result into your bug report — it bundles "
        "the daemon log and recent job states."
    ),
    "settings_debug_copy_report": "Copy debug report",
    "settings_mcp_header": "MCP server (agent access)",
    "settings_mcp_desc": "Expose the local training daemon to MCP clients (Claude Code, "
    "Claude Desktop, …). Run this in a terminal to register it with Claude Code:",
    "settings_mcp_desc_json": "For other MCP clients (Claude Desktop, OpenClaw, …), "
    "the equivalent JSON config:",
    "settings_mcp_copy": "Copy",
    "settings_mcp_copied": "Copied ✓",
    "settings_close": "Close",
    "settings_lang_apply_title": "Language",
    "settings_lang_apply_question": "Reload the interface now to apply the language?\n\n"
    "Unsaved edits in open tabs are lost. Queued/running training jobs live in "
    "the daemon and are not affected.\n\nChoosing No applies it on next launch.",
    # Guidebook
    "guidebook": "📖 Guidebook",
    "guidebook_tooltip": "Open the end-to-end guide (docs/guidelines/guidebook.md)",
    "guidebook_missing": "Guide not found at {path}",
    "guidebook_open_external": "Open in system viewer",
    "guidebook_close": "Close",
    # EasyControl adapter guide (build-your-own control task)
    "adapter_guide": "📘 Adapter Guide",
    "adapter_guide_tooltip": "How to build your own EasyControl adapter (easycontrol_adapters/ADAPTER_GUIDE.md)",
    "easycontrol_descriptor_note": "This control task is a self-contained descriptor with a multi-table shape, edited as raw TOML on the left:<br><br>• <b>name</b> — output slug; reroutes every derived cache/output path.<br>• <code>[staging]</code> — data generation that materializes the condition tree.<br>• <code>[preprocess]</code> — VAE/TE caching knobs over the staged tree.<br>• <code>[training]</code> — overrides folded onto the base EasyControl method.<br>• <code>[general]</code> / <code>[[datasets]]</code> — the dataset blueprint train.py reads.<br>• <code>[variant]</code> — this dropdown entry's GUI metadata.<br><br>The <b>Preprocess</b> button synthesizes the condition tree + caches it; <b>Train</b> trains the base EasyControl method with this descriptor's <code>[training]</code> overrides folded in. Both survive the GUI closing.",
    "easycontrol_descriptor_form_header": "Editing descriptor <b>{path}</b>. The knob tables below edit as a form; Save writes changed values back, preserving comments and the <code>[[datasets]]</code> blueprint. The blueprint and <code>[variant]</code> metadata aren't shown here — edit the file directly for those. Click a field name for help.",
    "ec_desc_group_top": "descriptor",
    # Top-bar buttons (models / update / report issue)
    "models_btn": "Models",
    "models_btn_tooltip": "Download or re-download model checkpoints (Anima base, SAM3, MIT, PE vision encoder)",
    "update_btn": "Update",
    "update_btn_tooltip": "Pull the latest anima_lora release from GitHub and run uv sync",
    "update_btn_available": "Update ●",
    "update_btn_available_tooltip": "New release {v} available — click to view release notes",
    "report_issue": "Report Issue",
    "report_issue_tooltip": "Open the GitHub issue tracker in your browser",
    "visit_github": "Visit GitHub page",
    "open_in_system_viewer": "Open in system viewer",
    # Models dialog
    "models_title": "Download Models",
    "models_intro": "Pick a model group below or use 'Download all' for the standard set "
    "(Anima + SAM3 + MIT + PE + tag DB). Files are saved under models/.",
    "models_download_all": "Download all (Anima + SAM3 + MIT + PE + tag DB)",
    "models_download": "Download",
    "models_redownload": "Re-download",
    "models_installed": "✓ Installed",
    "models_missing": "✗ Missing",
    "models_done_title": "Download complete",
    "models_done_message": "Models downloaded successfully. Files are saved under models/.",
    "models_failed_title": "Download failed",
    "models_failed_message": "Download exited with code {code}. See the log for details.",
    "model_anima": "Anima — DiT + text encoder + VAE",
    "model_sam3": "SAM3 — text-bubble masking",
    "model_mit": "MIT — manga text masking",
    "model_pe": "PE-Core-L14-336 — vision encoder (CMMD validation / DCW)",
    "model_danbooru_tags": "Danbooru tag DB — caption order correction",
    # HuggingFace authentication (Models dialog)
    "models_hf_token_placeholder": "Paste your HuggingFace token (hf_…)",
    "models_hf_authenticate": "Authenticate",
    "models_hf_token_hint": "Needed for gated / rate-limited downloads (e.g. SAM3). "
    'Create a token at <a href="https://huggingface.co/settings/tokens">'
    "huggingface.co/settings/tokens</a> · request SAM3 access at "
    '<a href="https://huggingface.co/facebook/sam3">huggingface.co/facebook/sam3</a>.',
    "models_hf_token_present": "✓ A HuggingFace token is already saved.",
    "models_hf_not_authenticated": "Not authenticated — paste a token to enable gated downloads.",
    "models_hf_token_empty": "Paste a token first.",
    "models_hf_authenticating": "Authenticating…",
    "models_hf_logged_in": "✓ Logged in as {name}.",
    "models_hf_login_failed": "Authentication failed: {err}",
    # Update dialog
    "update_title": "Update anima_lora",
    "update_warning": "Update will pull the latest release from GitHub and overwrite the working "
    "tree (datasets, output/, models/ are preserved). For configs/methods/ "
    "and configs/gui-methods/, choose whether to keep your edits or overwrite "
    "them with upstream (your version is backed up first). Run 'Dry run' to "
    "preview the changes.",
    "update_dry_run": "Dry run",
    "update_run": "Run update",
    "update_run_keep": "Update — keep my configs",
    "update_run_overwrite": "Update — overwrite configs (back up mine)",
    "update_confirm": "This will rewrite anima_lora source files. Continue?",
    "update_check_now": "Check now",
    "update_view_release": "View on GitHub",
    "update_current_version": "Current: {v}",
    "update_latest_version": "Latest: {v}",
    "update_no_baseline": "unknown (no manifest)",
    "update_status_checking": "Checking…",
    "update_status_uptodate": "✓ Up to date",
    "update_status_available": "● Update available",
    "update_status_unknown": "? Cannot compare (no local manifest)",
    "update_status_failed": "✗ Check failed",
    "update_release_notes": "Release notes:",
    "update_no_release_notes": "(release has no description)",
    "update_check_error": "Could not reach GitHub: {err}",
    # MergeTab
    "n_files": "{n} files",
    "merge_no_adapter": "No adapters found",
    "merge_no_adapter_msg": "No adapter selected or the file doesn't exist.",
    "merge_no_selection": "Select a checkpoint from the list to scan it.",
    "merge_verdict_ready": "✓ Ready to bake",
    "merge_verdict_hydra": "✗ HydraLoRA moe — layer-local router can't be baked",
    "merge_verdict_postfix_only": "✗ Postfix/prefix only — not a weight delta",
    "merge_verdict_unknown": "? No recognized adapter keys",
    "merge_options": "Merge Options",
    "merge_base_dit": "Base DiT:",
    "merge_multiplier": "Multiplier:",
    "merge_multiplier_tip": "LoRA strength to bake in (1.0 = full strength).",
    "merge_dtype": "Save dtype:",
    "merge_out": "Output:",
    "merge_out_placeholder": "(auto: <adapter>_merged.safetensors)",
    "merge_allow_partial": "Allow partial merge (drop Hydra / postfix keys)",
    "merge_allow_partial_tip": "Proceed even if the adapter contains non-bakeable components. Dropped components will be absent from the merged DiT.",
    "merge_button": "Merge into DiT",
    "merge_log_placeholder": "Merge output will appear here...",
    "merge_pick_dir": "Select adapter directory",
    "merge_pick_file": "Select adapter .safetensors",
    "merge_pick_dit": "Select base DiT .safetensors",
    "merge_pick_out": "Save merged DiT as...",
    "browse": "Browse…",
    # Multi-scale target_res tiers
    "target_res_bucket_tooltip": "Limit preprocessing to this {edge}px bucket resolution. Leave all unchecked to use every bucket in the selected tiers.",
    "target_res_danger_tooltip": "Heavy tier: {edge}px runs ~{tokens} tokens per image and adds an extra compiled block graph (slower compile, higher VRAM). Only enable if you actually need this resolution.",
    "resize_crop_anchor": "Resize crop position:",
    "resize_crop_anchor_tip": "Choose which side/corner is preserved when preprocessing crops the cover-resized image to the target bucket.",
    "resize_crop_anchor_top_left": "Top-left",
    "resize_crop_anchor_top": "Top",
    "resize_crop_anchor_top_right": "Top-right",
    "resize_crop_anchor_left": "Left",
    "resize_crop_anchor_center": "Center",
    "resize_crop_anchor_right": "Right",
    "resize_crop_anchor_bottom_left": "Bottom-left",
    "resize_crop_anchor_bottom": "Bottom",
    "resize_crop_anchor_bottom_right": "Bottom-right",
    "resize_crop_margins": "Resize margins:",
    "resize_crop_margin_top": "Top",
    "resize_crop_margin_right": "Right",
    "resize_crop_margin_bottom": "Bottom",
    "resize_crop_margin_left": "Left",
    # TensorBoard panel
    "tb_panel_title": "TensorBoard Runs",
    "tb_open": "Open TensorBoard",
    "tb_stop": "Stop Server",
    "tb_remove": "Remove",
    "tb_view": "View",
    "tb_view_tip": "Open TensorBoard scoped to this run only.",
    "tb_no_runs": "No runs yet — start training to populate this list.",
    "tb_status_running": "Running on port {port}",
    "tb_status_stopped": "",
    "tb_not_installed": "tensorboard is not installed. Run: pip install tensorboard",
    "tb_current_run_label": " (current)",
    "tb_open_current": "View Current Run",
    "tb_open_current_tip": "Open TensorBoard scoped to the in-progress training run only.",
    "tb_open_current_idle_tip": "Available while a training run is active.",
    "tb_appear_hint": "If the run does not appear in the list, try pressing TensorBoard's reload (update) button.",
}
