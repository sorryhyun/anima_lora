# tests/

| Command | Runs |
|---|---|
| `make test-core` | `pytest -m core`: the line-level promises below |
| `make test-unit` | everything: `-m "not slow"` serially, then `-m slow -n auto` (with `ARGS=…`, one serial run) |

`-m core` is one list to read: if a core test fails, a promise of a line or of the shared
infrastructure is broken. A test elsewhere in the suite guards something narrower.

## What belongs here

Add a test for what would otherwise fail silently: wrong images, wrong math, dropped LoRA
keys, a stale cache used without a warning, a corrupted caption, a job that hangs. A
feature-landing check ("the flag exists", "the happy path returns the right shape") does
not belong; the first real run catches that. Mark `@pytest.mark.core` only when a failure
breaks a line-level promise, name what it protects, and add it to the map. A parametrized
test counts each case: keep the matrix out of core.

## Core tests that skip

pytest hides the GPU (`tests/conftest.py`), so a green run says nothing about these. Read
the `skipped` count.

| Needs | Tests | Run |
|---|---|---|
| a card | `test_anima_int8.py::test_swapped_step_is_the_resident_step`; `test_moss_int8.py::{test_forward_and_dx_match_bf16, test_lora_grads_match_eager_reference, test_swapped_dx_is_resident_dx}`; `test_blockswap_h2d.py::test_h2d_only_matches_resident_and_classic` (the file does not collect without a card) | `ANIMA_TEST_GPU=1 make daemon-run ARGS="--stall-timeout 300 -m pytest tests/<file>.py -q -m core"` |
| Qwen3 TE + vocab pack | `test_vocab_pack.py::{test_en_prompts_tokenize_bit_identically_through_the_pack, test_cjk_prompt_lands_on_pack_rows}` | `make download-models` |
| Qwen-Image VAE | `test_qwen_vae_2d.py::test_full_vae_2d_matches_3d_latents` | the Qwen weights on disk |

## MOSS-Image

| Invariant | Test |
|---|---|
| `forward_batched` masks padded keys and stays op-for-op with the unbatched reference forward, skip-22 uncond and i2i reference included | `test_moss_model.py::{test_batched_matches_reference, test_batched_per_sample_skip, test_ref_isolate_batched_matches_reference}` |
| Sampler matches the reference scheduler; shift snapped to the training σ grid; both samplers land on x0 | `test_moss_model.py::{test_denoise_matches_scheduler_loop, test_snapped_sigmas_match_scheduler, test_denoise_samplers_land_on_exact_x0}` |
| An isolated reference runs once per loop and changes nothing | `test_moss_model.py::test_ref_cache_matches_uncached` |
| Latents are 4D packed; sides are multiples of 16, never upscaled | `test_moss_cache.py::{test_pack_matches_flux2_patchify, test_pack_is_inverse_of_unpack, test_target_size_invariants}` |
| TE cache invalidates on caption and encoder; variants survive a re-encode; layout-JSON captions | `test_moss_cache.py::{test_te_current_tracks_the_plain_caption, test_cache_text_keeps_variants_of_an_unchanged_caption, test_find_pairs_layout_json}`, `test_moss_fp8.py::test_encoder_stamp_reencodes_on_a_switch` |
| Adapter off is the base; six targets per block; save / load; ComfyUI keys; one train step per algo; stamped `ref_*` settings apply | `test_moss_lora.py::{test_zero_init_and_multiplier_zero_are_base, test_targets_six_per_block, test_save_load_round_trip, test_saved_keys_match_comfy_lora_map, test_one_step_decreases_loss, test_load_for_inference_sets_ref_settings}` |
| Training defaults to the int8 base; `lora_bf16.toml` is the swap plan; int8 refuses LoHa | `test_moss_int8.py::{test_default_plan_is_stream_at_budget_03_swap_0, test_bf16_config_is_the_swap_plan, test_int8_refuses_loha_and_runs_swap}` |
| The int8 base matches bf16; swap matches resident; LoRA gradients match the eager reference (GPU) | `test_moss_int8.py::{test_forward_and_dx_match_bf16, test_swapped_dx_is_resident_dx, test_lora_grads_match_eager_reference}` |
| i2i recipes agree on cache folder and reference scale, on the bf16 plan | `test_moss_i2i.py::test_task_recipes_load`, `test_moss_gui.py::test_i2i_forms_return_to_the_tasks_bf16_plan` |
| GUI stays torch-free; the model combo switches the whole plan | `test_moss_gui.py::{test_gui_is_torch_free, test_model_items_change_the_whole_plan}` |
| Field default → TOML → CLI; shipped TOMLs equal the defaults | `test_lines_requests_config.py::{test_cli_wins_over_config, test_shipped_moss_config_is_the_defaults, test_shipped_moss_cache_config_is_the_defaults}` |
| Nothing Anima-side imports `library.moss` | `test_moss_boundary.py::test_anima_side_never_imports_moss` |

## Qwen-Image-2.1

| Invariant | Test |
|---|---|
| Block swap: each block resident when it runs, within budget, end state equals start state | `test_qwen21_swap_schedule.py::test_schedule_round_trips` |
| fp32 masters; rank GEMMs in the model dtype | `test_qwen21_lora_dtype.py::{test_default_master_weights_are_fp32, test_fp32_adapter_computes_in_the_model_dtype}` |
| LoRA files load across the shared adapters | `test_lines_adapters.py::{test_pre_shared_qwen21_file_loads, test_algos_round_trip}` |
| 2D VAE fold matches the 3D VAE | `test_qwen_vae_2d.py::test_full_vae_2d_matches_3d_latents` |
| Nothing Anima-side imports `library.qwen21` | `test_qwen21_boundary.py::test_anima_side_never_imports_qwen21` |

## Anima

| Invariant | Test |
|---|---|
| Depth from the checkpoint under either prefix; both prefixes load real adapter weights; q/k/v and AdaLN fuse whatever the key order | `test_dit_arch_probe.py::test_depth_is_counted_under_either_prefix`, `test_dit_prefix_variants.py::test_load_llm_adapter_finds_weights_under_either_prefix`, `test_weight_convert_hooks.py` |
| Free-fit: grids in the tier's band, native shapes, frozen 1024 band; compiled flatten bit-exact to the eager 5D path | `test_freefit_bucket.py::{test_grid_lands_in_band, test_bucketmanager_freefit_keeps_native_shapes}`, `test_multiscale_buckets.py::test_1024_band_frozen`, `test_native_flatten.py::test_flatten_is_bit_exact_4032_family` |
| fp32 stream is the default, keeps weights and bf16 block GEMMs; bf16 stays bit-identical; LoRA gradients reach past it | `test_anima_stream.py` (core marks) |
| int8 forward copies, checkpoints and LoRA stamps; swapped step equals resident (GPU) | `test_anima_int8.py` (core marks) |
| Both TE cache formats load whatever the flag | `test_te_cache_pruned_load.py` |
| CJK pack: ext rows through hooks, state dict untouched, EN bit-identical, max-padded, inert when off; digest and cache stamp; merge carries the pack | `test_vocab_pack.py` (core marks), `test_ext_vocab_iso.py::test_pack_digest_is_stable_and_regeneration_invariant` |
| TE reads only the derived caption; chain order autotag → position → TE; scoped runs stay scoped; the mirror survives correction off | `test_preprocess_tasks.py` (core marks) |
| Artist prefix survives shuffle and dropout; the sentinel is never encoded; clauses survive the mirror; the revised caption is the authority; sidecars reach the TE cache | `test_caption_shuffle.py`, `test_caption_variant_sidecars.py`, `test_ext_vocab_iso.py::test_caption_grammar_keeps_a_quoted_line_whole` (core marks) |
| Stale caches re-encode; master read-only; resize idempotent, no upscale; reconcile removes only wrong-bucket caches; nested layout | `test_preprocess_dataset.py` (core marks), `test_nested_paths.py::test_resolve_cache_path_nested` |
| `masked_loss` is the one switch | `test_masked_loss_gate.py::test_masked_loss_off_gates_before_construction` |
| Merge chain: every shipped GUI variant merges, none pins hardware, a base default reaches the network, every `args.x` read is a declared flag | `test_config.py::{test_hardware_preset_composes_with_gui_variant, test_train_adaln_defaults_on_and_reaches_the_network}`, `test_inference_arg_closure.py`, `test_gui_variants.py::test_builtin_loads_clean` |

### Adapters (`networks/`)

| Invariant | Test |
|---|---|
| Every LoRA-family forward and input gradient stays bit-exact; rank GEMMs in the model dtype | `test_lora_module_equivalence.py::test_module_forward_matches_golden`, `test_lora_dtype_policy.py::test_lora_fp32_activation_no_upcast` |
| Save writes split q/k/v and the ComfyUI adaln layout; mixed Hydra defuses; merge applies comfy-layout rows; a baked save reproduces the channel-scaled forward | `test_network_registry.py` (core marks), `test_lora_merge_adaln_layout.py::test_comfy_layout_adaln_keys_merge`, `test_per_channel_scaling_roundtrip.py::test_baked_save_reproduces_channel_scaled_forward` |
| `NETWORK_KWARGS` derives from the factory; string bools parse | `test_network_registry.py::test_factory_only_keys_are_derived`, `test_network_cfg.py::test_string_bool_parsing_matches_old_factory_path` |
| T-LoRA mask per rank group: same schedule fraction, restored on clear | `test_timestep_mask_mixed_rank.py::{test_rank_groups_follow_the_same_schedule_fraction, test_clear_restores_full_rank_on_every_group}` |
| Hydra routing reaches every module after a device move and at each denoising step; training kernels are the inference kernels | `test_global_router.py`, `test_hydra_sigma_band.py`, `test_inference_hydra_sigma.py`, `test_router_compute.py` (core marks) |
| Method adapters run on both text branches; method checkpoints round-trip; EasyControl step 0 bit-exact; dual bank loads as ψ⁺ | `test_adapter_dispatch.py`, `test_method_network_lifecycle.py`, `test_easycontrol_body.py`, `test_soft_tokens_contrastive.py` (core marks) |
| Inference never imports training | `test_adapter_protocol.py::test_inference_surface_does_not_import_training` |
| REPA (shipped path): layout-agnostic, portrait and landscape PE grids, weight gate, DoG target and gradient; a redirected TE cache still finds the PE sidecar | `test_repa.py` (core marks), `test_repa_pe_sidecar.py::test_redirected_te_falls_back_to_latent_cache_dir` |
| The reentrant checkpoint keeps gradients on LoRA params | `test_unsloth_checkpoint.py` |

### Turbo, soup, σ-lowres

| Invariant | Test |
|---|---|
| Resume continues exactly; extended runs land on the new curve; topology changes refused; a crash mid-save keeps the bundle | `test_turbo_resume.py` (core marks) |
| Warm start reproduces the extracted delta | `test_turbo_warmstart.py` (core marks) |
| Shipped objective terms and schedule: L_CDM, dynamic grid, token head layout, adaln, the int8 hook before LoRA patching | `test_turbo_cdm.py`, `test_turbo_dynamic_schedule.py`, `test_turbo_gan_disc.py`, `test_turbo_adaln.py`, `test_turbo_mfd.py` (core marks) |
| A soup is the exact ΔW average and never drops a module | `test_soup.py::TestSoupStateDicts` (core marks) |
| Demoted latents stay outside `latents_*`; emit keeps native keys, is idempotent, loads back | `test_sigma_lowres.py` (core marks) |
| Front door cannot drift from the parser; inference writes what the daemon lifts | `test_generation_request.py::test_core_defaults_match_parser`, `test_gen_manifest.py::test_daemon_writes_manifest_and_pointer` |

## Shared

| Area | Invariant | Test |
|---|---|---|
| Daemon | One job at a time; a bad job cannot wedge the queue; exit codes truthful; stop kills the tree; restart recovers; stale code detected; only our own pid reaped; release pause; stdlib-only; `daemon-run` end to end | `test_daemon.py` (core marks) |
| Daemon | A symlink cycle or a silent job cannot spin forever; a dead run does not read as running; `progress.jsonl` schema; a configured aux loss that never fires is an error | `test_hang_guards.py`, `test_progress_sink.py`, `test_liveness.py` (core marks) |
| GUI | The Preprocess tab's contract, byte for byte; cache-only; no preprocess key reaches training; saved forms build the request; `path_scope` applied once; no hardware baked into a variant; no torch at launch; mode switch | `test_gui_preprocess_characterization.py`, `test_gui_preprocess_tab.py`, `test_gui_snapshot_preprocess_keys.py`, `test_gui_stage_form.py`, `test_gui_submit.py`, `test_gui_variant_form.py`, `test_gui_launch_speed.py`, `test_gui_model_mode.py` (core marks) |
| Curation | Trainer → `anime_tools`, never the reverse; every import and `-m` string the trainer spells exists at the pin | `test_curation_boundary.py`, `test_anime_tools_cli_contract.py` (core marks) |
| Catalog | Rows land where the loaders and `base.toml` look; the first-run set is the mandatory set | `test_downloads.py` (core marks) |
| Install | Tarballs extract, `uv sync` works, no silent demotion to ROCm or CPU torch | `test_repo_hygiene.py`, `test_rocm_backend.py` (core marks) |
