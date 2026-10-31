"""English strings for the Anima LoRA GUI."""

from __future__ import annotations

STRINGS: dict[str, str] = {
    # Window / tabs
    "window_title": "Anima LoRA",
    "tab_config": "Training Config",
    "tab_easycontrol": "EasyControl",
    "tab_turbo": "Turbo",
    "tab_experimental": "Experimental",
    "tab_merge": "Merge",
    "tab_queue": "Queue",
    "tab_anime_tools": "anime_tools",
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
    "preprocess_caption_editing": "Caption rewriting",
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
    "preprocess_invalid_stage": "{stage}: {err}",
    "preprocess_status_resized": "Resized images: {n}",
    "preprocess_status_caches": "Caches — latents: {lat}, text: {te}, PE: {pe}",
    "preprocess_status_masks": "Masks: {masks}",
    "preprocess_status_no_resized": "No resized images yet.",
    "preprocess_open_dataset_dir": "Open cache folder",
    "preprocess_open_dataset_dir_tooltip": "Open the post_image_dataset/ folder (resized images + caches) in your file manager.",
    "preprocess_status_export_stale": "anime_tools has edits newer than its last Export",
    "preprocess_resize_preview": "Resize preview",
    "preprocess_resize_preview_tooltip": "Per source image: the bucket and tier the resize picks under the settings below, and the crop it takes. Source files are not modified.",
    "preprocess_resize_preview_title": "Resize preview",
    "preprocess_resize_preview_refresh": "Refresh",
    "preprocess_resize_preview_col_image": "Image",
    "preprocess_resize_preview_col_source": "Source",
    "preprocess_resize_preview_col_bucket": "Bucket",
    "preprocess_resize_preview_col_tier": "Tier",
    "preprocess_resize_preview_col_kept": "Kept",
    "preprocess_resize_preview_summary": "{n} images in {buckets} buckets",
    "preprocess_resize_preview_skipped": "{n} excluded",
    "preprocess_resize_preview_unreadable": "{n} unreadable",
    "preprocess_resize_preview_empty": "No images under {path}",
    "anime_tools_failed": "Could not start anime_tools: {err}",
    "anime_tools_root_warning": "The panel's saved roots point into post_image_dataset/ ({roots}), so its stages would write over the trainer's tree. Clear them in the panel's ⚙ Settings (or in {path}).",
    "anime_tools_starting": "Starting anime_tools… (log: {log})",
    "anime_tools_timeout": "anime_tools did not come up. See the log: {log}",
    "anime_tools_no_webengine": "Qt WebEngine is not installed, so the panel can't be shown here. Open {url} in your browser.",
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
    "daemon_next_queued": "Following the next queued job {job_id} on the training daemon.\n",
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
    "hardware_preset": "Hardware:",
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
    # Turbo distillation config tab (gui/tabs/distill_tab.py)
    "distill_general_section": "general",
    "distill_job_running": "A job is already running on this tab.",
    "distill_config_missing": "Could not read the config file: {err}",
    # Soup pipeline tab (gui/tabs/soup_tab.py) — run-only fields; the pool /
    # dose / seeds / rank knobs are edited in the [soup] section form below.
    "soup_run_section": "Run",
    "soup_path_pattern": "Path pattern",
    "soup_path_pattern_tip": "fnmatch glob selecting the fine-tune images, matched against each image's path relative to its subset image_dir. '|' separates alternatives, e.g. 'sincos/*' or 'art_a/*|art_b/*'. Required.",
    "soup_name": "Output name",
    "soup_name_tip": "Output slug → output/ckpt/anima_soup_<name>.safetensors. Blank = derived from the path pattern (a plain '<dir>/*' becomes '<dir>').",
    "soup_ft_args": "Fine-tune args",
    "soup_ft_args_tip": "Extra CLI args forwarded to each fine-tune run, e.g. --network_dim 32 --max_train_epochs 8.",
    "soup_path_pattern_required": "Soup needs a path pattern (e.g. 'sincos/*' or 'art_a/*|art_b/*').",
    # Shared labels
    "directory": "Directory:",
    # Language
    "language": "Language:",
    # Settings dialog
    "settings_btn": "⚙ Settings",
    "settings_btn_tooltip": "Application settings — language, preferences, MCP server registration",
    "mode_label": "Model",
    "mode_tooltip": "Which model the window trains and tests. Anima and MOSS-Image keep separate caches, settings and LoRAs; switching only changes the view, running jobs keep going.",
    "mode_anima": "Anima",
    "mode_moss": "MOSS-Image",
    "settings_title": "Settings",
    "settings_prefs_header": "Preferences",
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
    "adapter_guide_tooltip": "How to build your own EasyControl adapter (i2i_tasks/ADAPTER_GUIDE.md)",
    "easycontrol_descriptor_note": "This control task is an i2i task file (<code>i2i_tasks/configs/</code>) plus an EasyControl descriptor (<code>configs/easycontrol/</code>), edited as a form on the left:<br><br>• <b>name</b> — output slug; reroutes every derived cache/output path (task file).<br>• <b>stage</b> — which staging tool builds the tree (task file).<br>• <code>[staging]</code> — data generation that materializes the condition tree (task file).<br>• <code>[preprocess]</code> — VAE/TE caching knobs over the staged tree.<br>• <code>[training]</code> — overrides folded onto the base EasyControl method.<br>• <code>[general]</code> / <code>[[datasets]]</code> — the dataset blueprint train.py reads.<br>• <code>[variant]</code> — this dropdown entry's GUI metadata.<br><br>The <b>Preprocess</b> button synthesizes the condition tree + caches it; <b>Train</b> trains the base EasyControl method with this descriptor's <code>[training]</code> overrides folded in. Both survive the GUI closing.",
    "easycontrol_descriptor_form_header": "Editing descriptor <b>{path}</b>. The knob tables below edit as a form; Save writes changed values back, preserving comments and the <code>[[datasets]]</code> blueprint. The blueprint and <code>[variant]</code> metadata aren't shown here — edit the file directly for those. Click a field name for help.",
    "ec_desc_group_top": "descriptor",
    # Top-bar buttons (models / update / report issue)
    "models_btn": "Models",
    "models_btn_tooltip": "Download or re-download model checkpoints — the Anima weights (incl. the CJK vocab pack) and the anime_tools curation packs, in one place",
    "update_btn": "Update",
    "update_btn_tooltip": "Pull the latest anima_lora release from GitHub and run uv sync",
    "update_btn_available": "Update ●",
    "update_btn_available_tooltip": "New release {v} available — click to view release notes",
    "report_issue": "Report Issue",
    "report_issue_tooltip": "Open the GitHub issue tracker in your browser",
    "visit_github": "Visit GitHub page",
    # Models dialog
    "models_title": "Download Models",
    "models_intro": "The weights a training or inference run needs, grouped by pack. 'Download the first-run set' fetches the three Anima weights, PE-Spatial (REPA), the CJK vocab pack (on by default since v2), the tagger checkpoint and the tag DB. SAM3 (masking) and OCR are opt-in packs on the Curation tab. Files are saved under models/.",
    "models_download_all": "Download the first-run set",
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
    "model_pe": "PE-Core-L14-336 — vision encoder (CMMD validation)",
    "model_anima_dit": "Anima base DiT — the model being trained",
    "model_anima_te": "Qwen3-0.6B text encoder — prompt embeddings",
    "model_anima_vae": "Qwen-Image VAE — latent encode / decode",
    "model_vocab_pack": "CJK vocab pack — JA / KO / ZH caption rows (on by default)",
    "model_pe_spatial": "PE-Spatial-B16-512 — REPA + near-twin grouping",
    "models_used_by": "Used by: {what}",
    "models_tab_anima": "Anima",
    "models_tab_curation": "Curation (anime_tools)",
    "curation_models_intro": "Weights for the curation stages, grouped by pack — tagger, tag DB, masking (SAM3, opt-in), OCR (opt-in), grouping. Read straight from the anime_tools catalog. Each stage also fetches what it needs on first use; this only moves the wait.",
    "models_download_missing": "Download all missing ({n})",
    "models_all_installed": "✓ Everything installed",
    "models_download_pack": "Download pack",
    "models_redownload_pack": "Re-download pack",
    "models_pack_anima": "Anima base",
    "models_pack_anima_desc": "The DiT, the Qwen3-0.6B text encoder and the Qwen-Image VAE — every training and inference run needs all three.",
    "models_pack_pe": "PE-Core",
    "models_pack_pe_desc": "PE-Core-L14-336: the CMMD validation metric only (use_cmmd, off by default). The REPA feature cache uses PE-Spatial — the Grouping pack on the Curation tab.",
    "models_pack_cjk": "CJK vocab pack",
    "models_pack_cjk_desc": "Extra text-encoder rows for Japanese / Korean / Chinese caption and prompt spans. On by default since v2; English text is bit-exact either way.",
    "models_pack_moss_base": "MOSS-Image Base",
    "models_pack_moss_base_desc": "MOSS-Image-Base (not Anima): the 5B DiT, the Qwen3.5-9B text encoder, the FLUX.2 VAE and tokenizer — ~27 GB. The MOSS line's LoRA substrate.",
    "models_pack_moss_sft": "MOSS-Image SFT",
    "models_pack_moss_sft_desc": "MOSS-Image-SFT's DiT alone (~10 GB); it reuses the Base pack's text encoder, VAE and tokenizer.",
    "models_pack_moss_dit_int8": "MOSS-Image INT8 DiT",
    "models_pack_moss_dit_int8_desc": "MOSS-Image Base's DiT with its 252 block linears in int8 GPTQ (~5 GB), the default int8 training base: fetched on first use.",
    "models_pack_moss_te_int8": "MOSS-Image INT8 text encoder",
    "models_pack_moss_te_int8_desc": "The Qwen3.5-9B text encoder in INT8 W8A8 (~9.2 GB instead of 16.8 GB), the default text_encoder=int8: all 32 layers fit on a 16 GB card. The tokenizer still comes from the Base pack.",
    "models_pack_moss_te_fp8": "MOSS-Image fp8 text encoder",
    "models_pack_moss_te_fp8_desc": "The Qwen3.5-9B text encoder in fp8 (~9.2 GB instead of 16.8 GB) for text_encoder=fp8: all 32 layers fit on a 16 GB card. The tokenizer still comes from the Base pack.",
    "models_pack_tagger": "Tagger",
    "models_pack_tagger_desc": "The Anima Tagger: its checkpoint and the pixai-tagger-v1.0 backbone (Apache-2.0, ~2 GB).",
    "models_pack_tags": "Danbooru tag DB",
    "models_pack_tags_desc": "The ~114k-row tag table caption correction types against, and its English descriptions.",
    "models_pack_masking": "Masking",
    "models_pack_masking_desc": "SAM3 subject masks (gated; masking is opt-in since v2), and the subject soft prompt the position stages detect with.",
    "models_pack_ocr": "OCR",
    "models_pack_ocr_desc": "The AnimeText text-block detector and the manga VL reader (PaddleOCR-VL-1.6 base + the SFX fine-tune). Opt-in.",
    "models_pack_grouping": "Grouping",
    "models_pack_grouping_desc": "PE-Spatial-B16-512: the REPA feature cache (preprocess-pe) and the near-twin grouping tower.",
    "model_danbooru_tags": "Danbooru tag DB — caption order correction",
    "model_tagger": "Anima Tagger — pixai-tagger-v1.0 backbone",
    # HuggingFace authentication (Models dialog)
    "models_hf_token_placeholder": "Paste your HuggingFace token (hf_…)",
    "models_hf_authenticate": "Authenticate",
    "models_hf_token_hint": "Needed for gated / rate-limited downloads (e.g. SAM3). "
    'Create a token at <a href="https://huggingface.co/settings/tokens">'
    "huggingface.co/settings/tokens</a> · request SAM3 access at "
    '<a href="https://huggingface.co/facebook/sam3">huggingface.co/facebook/sam3</a>. '
    "The Anima Tagger backbone is public and needs no token.",
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
    # LoRA ⊕ LoRA → LoRA merge (merge_loras.py)
    "merge_mode": "Mode:",
    "merge_mode_dit": "Into DiT",
    "merge_mode_loras": "Merge LoRAs",
    "merge_lora_options": "LoRA Merge Options",
    "merge_lora_select_hint": "Pick 2+ adapters (Ctrl/Shift-click) to fuse into one LoRA, usable via the normal --lora_weight path.",
    "merge_lora_selected": "Selected:",
    "merge_lora_need_two": "Select at least 2 LoRAs.",
    "merge_weights": "Weights:",
    "merge_weights_placeholder": "(optional: e.g. 1.0,0.7,0.7 — list order)",
    "merge_weights_tip": "Comma-separated per-LoRA strengths, one per selected adapter in list order. Blank = all 1.0.",
    "merge_weights_mismatch": "{n} weights given but {m} LoRAs selected.",
    "merge_normalize": "Normalize:",
    "merge_normalize_tip": "Rescale the merged delta to single-LoRA magnitude. 'global' (default) matches the global RMS to the average single-LoRA norm; 'per_module' matches each module but reweights layers; 'off' = raw exact-concat sum (can overdrive the DiT).",
    "merge_lora_button": "Merge LoRAs",
    "merge_analyze_button": "Analyze interference",
    "merge_analyze_tip": "Dry-run: report how the selected LoRAs interfere in weight space (constructive reinforce vs destructive cancel, per pair and per layer) without writing a merged file.",
    "merge_analysis_safe": "✓ Safe to merge — LoRAs are near-orthogonal · strongest pair |cos| {cos} · energy ratio {ratio} · {shared}/{modules} shared modules",
    "merge_analysis_reinforce": "⚠ {strength} reinforcing — {a} ↔ {b} cos {cos} · energy ratio {ratio} (normalize=global rescales the summed magnitude) · {shared}/{modules} shared modules",
    "merge_analysis_cancel": "⚠ {strength} cancellation — {a} ↔ {b} cos {cos} · these LoRAs partially erase each other · energy ratio {ratio} · {shared}/{modules} shared modules",
    "merge_analysis_overlap": "⚠ Subspace collision — {a} ↔ {b} write into the same weight subspaces (overlap {overlap}, {x}× random) despite near-orthogonal directions · styles may fight at inference · energy ratio {ratio} · {shared}/{modules} shared modules",
    "merge_analysis_moderate": "Moderate",
    "merge_analysis_strong": "Strong",
    "merge_lora_out_placeholder": "(auto: <first-lora>_merged<N>.safetensors)",
    "merge_mode_extract": "Extract LoRA",
    "merge_extract_options": "Extract LoRA Options",
    "merge_extract_hint": "Extract a plain LoRA from the weight delta (ΔW = tuned − base) between two full DiT checkpoints, SVD-truncated to the chosen rank. The result loads through every plain-LoRA consumer (--lora_weight, Merge into DiT, ComfyUI, turbo warm-start). Both checkpoints must be the net.-prefixed Anima layout.",
    "merge_extract_base": "Base DiT:",
    "merge_extract_tuned": "Tuned checkpoint:",
    "merge_extract_tuned_placeholder": "(required: click a checkpoint in the list, or browse)",
    "merge_extract_rank": "Rank:",
    "merge_extract_rank_tip": "Rank the ΔW is SVD-truncated to (network_dim of the output LoRA). 96 captures most of a typical full-model delta.",
    "merge_extract_include_adaln": "Include adaln up-projections",
    "merge_extract_include_adaln_tip": "Also extract the adaln_up_{branch} deltas (the largest movers in an official turbo delta). Off by default: the plain student doesn't target adaln and stock loaders skip these keys.",
    "merge_extract_adaln_layout": "adaln layout:",
    "merge_extract_adaln_layout_tip": "Key naming for the adaln LoRAs: 'runtime' = in-repo module names (for the turbo warm-start path), 'comfy' = ComfyUI state-dict paths (loads natively in core). Only used when adaln is included.",
    "merge_extract_out_placeholder": "(required: output LoRA .safetensors)",
    "merge_extract_pick_ckpt": "Select full DiT .safetensors",
    "merge_extract_no_tuned": "Select a tuned checkpoint to diff against the base.",
    "merge_extract_no_out": "Choose an output path for the extracted LoRA.",
    "merge_extract_button": "Extract LoRA",
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
