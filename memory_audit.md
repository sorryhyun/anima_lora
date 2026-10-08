Memory audit (read-only; 56 files + MEMORY.md). Repo = /home/sorryhyun/anima/anima_lora, memory = ~/.claude/projects/-home-sorryhyun-anima-anima-lora/memory/.

| file | verdict | repo coverage / why keep | reason |
|---|---|---|---|
| feedback_cjk_vocab_idx_row_terminology | DELETE | project/cjk_anima_scale/CLAUDE.md § Invariants (vocab/idx/row, "table"/"ctx" retired, trained.pt merged at save, on-disk keys stay) | duplicate |
| feedback_commit_on_main | KEEP | not in repo; overrides harness "branch first" default | user preference |
| feedback_default_block_compile | DELETE | docs/guidelines/base-config.md l.113 "On OOM, enable this first, before gradient checkpointing"; qwen exception in library/qwen21/CLAUDE.md l.34 | duplicate |
| feedback_emoticon_tags_stay_latin | DELETE | scripts/distill_cjk/corpus/tag_glossary.py l.282-294 (`han_allowed`/`is_japanese`) + comment l.638-642 (cites the memory — drop the cite) | encoded in code |
| feedback_experiments_use_existing_floor | TRIM | cjk CLAUDE.md l.97-99 has floor mechanics, not the preference | → "CJK experiments read only strings whose floor is already cached; ask before adding a floor key." |
| feedback_heavy_deps_ok | TRIM | not in repo | → "Heavy native deps are fine (`uv add`); ship the upstream method, flag runtime cost only." |
| feedback_micro_arms_and_per_glyph_reads | TRIM | not in repo | → "Mechanism questions: ~25-min micro arm before a full run; native verdicts per glyph with sheets viewed, never totals." |
| feedback_no_cpu_contention_during_jobs | TRIM | not in repo | → "No CPU-heavy side work while a build or GPU job runs." |
| feedback_no_ruff_format_mid_edit | TRIM | CLAUDE.md l.77 gives the command, not the timing | → fold into the ruff line: "format once, before commit, touched files only" |
| feedback_no_teaching_stock_behaviour | KEEP | not in repo; **not indexed in MEMORY.md** | user preference; trim to one line |
| feedback_no_unasked_memory_writes | KEEP | not in repo | already one line |
| feedback_push_back_on_unclear_requests | TRIM | not in repo | → "Request off-plan or meaningless → say so bluntly and ask; never substitute the nearest probe." |
| feedback_ruff_scope_collateral | TRIM | CLAUDE.md l.77 has the rule and cites this file | → one line "F401 autofix strips re-exports — never blanket-run"; or put that clause in CLAUDE.md and delete |
| feedback_test_run_budget | DELETE | CLAUDE.md l.78-81 | duplicate |
| feedback_text_embedding_terminology | KEEP | not in repo | user vocabulary |
| feedback_use_uv | KEEP | CLAUDE.md shows `uv sync` but not "never pip" | one line already |
| feedback_vendor_sync | DELETE | CLAUDE.md l.289-291 + .claude/skills/custom-nodes § Vendor trees (drop the `[[feedback_vendor_sync]]` cite in CLAUDE.md) | duplicate |
| feedback_verification_general_agent | TRIM | not in repo | → "'검증은 agent spawn' = general-purpose agents, not `verifier`." |
| user_no_papers_blog_only | TRIM | not in repo; dangling `[[project_wake_line_rollup]]` | → "Blog over paper; write up only a transferable trick; flag when a lever plausibly transfers." |
| user_taste_profile | TRIM | mechanics in scripts/toolkits/build_randoms.py docstring l.125-133 | → persona one-liner (순정파 axes; dark skin / loli avoided), drop the numbers |
| reference_spectrum_node_publish | TRIM | custom-nodes skill has vendor-sync-before-publish, not the token | → "bump pyproject version, push, `comfy node publish --token $COMFY_REG` (.env; CLI at .venv/bin/comfy); output echoes the PAT." |
| project_anime_tools_pinned_copy_in_venv | DELETE | .claude/skills/anime-tools/SKILL.md § The pin l.65-90 (pinned copy, stale venv, upstream index leak, PYTHONPATH, DaemonClient extra_env) | duplicate |
| project_bench_run_dir_collision | TRIM | not in repo (bench/_common.py l.180 `exist_ok=True`, undocumented) | → "Same-minute bench runs share a dir and overwrite result.json — always pass --label." (better: one line in bench/README.md, then delete) |
| project_blockswap_extra_forwards_gradcache | TRIM | not in repo (offloading.py has no comment); cited by docs/experimental/soft_tokens.md, docs/findings/freetext_text_rendering.md, docs/methods/turbo.md | → "ModelOffloader = one fwd + one bwd per step; a second DiT forward with blocks_to_swap>0 crashes `aten.mm cuda vs cpu`; VR-loss/inversion/IP-Adapter sites unaudited." (belongs in offloading.py docstring) |
| project_cache_invalidation_rules | TRIM | code: library/preprocess/text.py::_cache_is_current (mtime); no doc states it; **CLAUDE.md l.223 "TE caches skip on existence only" is wrong** | → "TE caches are mtime-aware; latents/PE are existence-keyed — crop or vocab-pack changes need --overwrite." Fix CLAUDE.md, then delete |
| project_caption_index_shared_artifact | DELETE | tasks.py l.245-248 (`caption-index` help: path, pure data, auto-run) | duplicate |
| project_cjk_ocr_o4_sfx_reader_state | DELETE | project/finished/cjk_aware_anima_dit/findings.md l.88-96 (arms incl. hayai), l.205 (Hub = v3 b2_norm4), l.209-218 (closed levers) | research record |
| project_closed_lines_rollup | DELETE | every entry already names its repo doc; ~70 pointers verified present. Stale pointers: docs/structure/ortholora.md, docs/experimental/chimera-hydra.md, bench/torch_bump/README.md (gone), project/cjk_aware_anima/findings.md (now project/finished/…), "hydra-lora.md l.69 still says 0.001" (doc now says ~5e-5 at l.57). Three "verdict here only" lines (turbo_R_plateau, pid_superseded_history, dcw_line_shelved) → paste into docs/methods/turbo.md / _archive first | research index |
| project_comfy_node_gotchas | KEEP | not in Spectrum/BlockCompile READMEs or networks/CLAUDE.md (checked memory-leak cycle, vbar device, int8, `0530` CNS tag, `_DCW_INPUTS`) | live gotchas on external node repos; cut the DCW verdict + cross-refs |
| project_daemon_gotchas | TRIM | anima_daemon/README.md l.33-49,149-154,218-237,425 + daemon skill cover stall watchdog, flag order, heartbeat, RELEASE=1, SIGKILL | → "Bespoke loops (scripts/distill_turbo etc.) bypass train.py — queue/compile-cache/--deterministic infra does not reach them." Drop the pause anecdote |
| project_dataset_id_path_gotchas | TRIM | not in repo | → "image_dataset/post_image_dataset are symlinks into /media/sorryhyun/new/dataset (use find -L); stems mix gelbooru ids and dan_-prefixed danbooru ids — filter dan_ before any danbooru join." |
| project_gh92_windows_backend_default_group | TRIM | tests/test_repo_hygiene.py covers symlinks + out-of-tree lock | → "Tag push publishes via CI → `gh release edit` for notes; stub extras stay until ~v1.18; previous release's update.py runs during the transition." |
| project_harness_kill_systemd_run_escape | TRIM | CLAUDE.md + daemon skill cover GPU SIGKILL; systemd-run / pkill not in repo | → "Non-GPU long jobs: `systemd-run --user --unit X --collect <cmd>`; never `pkill -f` from the agent shell (self-match); wait on daemon jobs with Monitor on job.json." |
| project_line_block_pack_uncommitted_nodes | TRIM | Adapter repo clean at 3.14.0 (verified); Spectrum `_vendor` drift still uncommitted (git status: 8 _vendor files + node.zip); line-mode history is finished | → "~/ComfyUI-Spectrum-KSampler has uncommitted `_vendor` drift — check git status / ask before vendor-sync or publish; always bump version." |
| project_lora_family_pruned_2026_09_21 | DELETE | networks/CLAUDE.md (registry lora/hydra/step_expert), library/inference/router_compute.py docstring l.2-10 (node vendors it), factory.py `_detect_removed_variant` | duplicate |
| project_manga109s_coo_sfx_reader | TRIM | findings.md covers eval/baselines + licensing (l.327-330) | → "Manga109-s at ~/manga109s (never copy into repo — no redistribution); daemon forwards only ANIMA_* env → ANIMA_MANGA109S_ROOT; AnimeText pool at /media/sorryhyun/new/dataset is NC." |
| project_media_volume_renamed_new | TRIM | not in repo | → "Dataset volume mounts at /media/sorryhyun/new; the old `새 볼륨` mount point exists empty, so stale absolute paths read as missing." |
| project_models_on_nvme_symlinks | TRIM | not in repo (symlinks verified) | → "models/{diffusion_models,text_encoders,vae} → /media/sorryhyun/data/anima_models (nofail NVMe; dangling = ENOENT); vocab_packs stays a real dir; comfy reads packs from /media/sorryhyun/data/comfy_models/vocab_packs via extra_model_paths.yaml." |
| project_ocr_eval_basis_rebases | DELETE | findings.md l.26 "sincos label basis changed — no conversion exists", l.47-48 exact_key ellipsis fold | research record |
| project_ocr_finetune_wall_field_trap | DELETE (stale) | script is in project/finished/cjk_aware_anima_dit/ocr/finetune_vl16_lora.py (l.430); line finished | if wanted, one comment at l.430 |
| project_ocr_stage_by_module_paths | DELETE (stale) | "no make target for OCR" is false — `make caption-full` runs it (tasks.py l.268-273; scripts/tasks/preprocess.py DEFAULT_OCR_DIR, `_GPU_STAGES`); O4e bug in findings.md; anime-tools skill: wrappers build the request | obsolete |
| project_qwen21_eval_ruler_unusable | DELETE | project/qwen21_lora/report.md l.4-5, 122-141 | duplicate |
| project_qwen21_lora_backward_gate | DELETE | report.md l.25-27 (9.59 GB, 1.18 s/step, PCIe-bound), l.105, 154; library/qwen21/CLAUDE.md l.16, 34, 37 | duplicate |
| project_qwen21_not_anima_boundary | DELETE | root CLAUDE.md § Architecture ("library/qwen21/ … not Anima … read library/qwen21/CLAUDE.md, load the qwen21 skill") | duplicate |
| project_resized_captions_trigger_pollution_2026_09_03 | DELETE (stale) | incident fixed in a13ba5b8; the proposed conftest guard was never added (tests/conftest.py autouse fixture is only chdir) — belongs in a test, not memory | history |
| project_sea_delta_generalizes_guidance | TRIM | docs/inference/mod-guidance.md l.105 shows `--pooled_text_proj` but not that omitting it leaves mod silently inert; docs/methods/adaln.md cites this memory | → one line, or add the sentence to mod-guidance.md and delete |
| project_shelved_explorations | DELETE | every entry points at docs/findings/*, _archive/bench/*, _archive/shelved_benches.md (verified); `bench/{rt_lynx,oscar,qkv_packed,sigma_reshape,cross_attn_drive,res_curriculum,chimera}` pointers now live under _archive/bench/ | research index |
| project_sigma_lowres_archived | TRIM | docs/optimizations/sigma_lowres.md l.14 points at `_archive/sigma_lowres/` — **that dir does not exist** (only project/sigma_lowres/{bench,paper_bench} remnants); `preserve` remote exists | → "sigma_lowres research is only on private remote `preserve` (sorryhyun/demoted-training, sigma-lowres-yarnsig) — never push to origin." Verdicts → paper on preserve |
| project_sr_sidecar_finished | DELETE (after move) | project/finished/sr/STATUS.md l.55 says "full numbers live in project memory" — repo delegates to this file | paste Training/Tiling/Text bullets into STATUS.md, then delete |
| project_tagger_dbv4_backend | TRIM | ../anime_tools/docs/anima_tagger.md l.19 (TAGGER_HF_SUBFOLDER), l.59-70 (aliases, artist-OC drop), l.140 (tag_rules.yaml) | → "~/gelcrawl/tag_rules.yaml is user-owned and untracked — back up before editing." Rest is research |
| project_tagger_resident_mmap_ram_budget | TRIM | not in repo | → "Box RAM 64 GB (~58 available) + 8 GB swap; user kills runs past ~90 %." |
| project_text_cache_dir_te_redirect | TRIM | `latent_cache_dir` absent from docs/configs (rg none) | → "Subset `latent_cache_dir` redirects target latents only; a missing one is silently encoded from the ORIGINAL image." (belongs in configs/CLAUDE.md) |
| project_turbo_rollup | DELETE (after move) | docs/methods/turbo.md l.19/165 (sectioned schema), 54 (set_view), 116-150 (warm start), 161 (fm_mse not a ranker), 184-191 (div_weight, LR, flow_shift); docs/optimizations/channel_scaling.md l.98 (α on turbo). Not in docs: wrong-section TOML silently defaults; NFE=2 GAN-runaway signature; pooled-head zero metrics → add to turbo.md first | research record |
| project_user_community_audience | TRIM | not in README/CONTRIBUTING | → "Real users: Arca Live AI-art channel (KR) + CN/JP + Civitai; weigh UX/config regressions across 3 languages." |
| project_uv_run_resolve_broken | TRIM | pyproject l.146-156: cu132 index `explicit = true`, rocm deliberately not — conflict may persist (not run: GPU job live) | → "Use `.venv/bin/python` / `uv run --no-sync`; bare `uv run` re-resolves and can fail on the win32 torch index split." |
| project_xid8_gpu_hang_recurring | TRIM | docs/findings/README.md l.66 points at this memory (repo delegates) | → "Long runs die to Xid 8 (SIGABRT, no precursor; infra, no knob): `journalctl -k -b <n> | grep Xid` per boot before debugging; bespoke loops have no resume." Or move into docs/findings and delete |

MEMORY.md changes
- Drop section "Anti-re-proposal indexes" entirely (all 3 files deleted) and the "(check before reopening a line)" framing; the phrase "re-propose"/"anti-re-proposal" occurs in 8 memory files, all DELETE/TRIM above.
- Drop section "Closed / finished research lines" (turbo, sr deleted; sigma_lowres one-liner moves to Infra).
- Drop section "Qwen-Image-2.1 LoRA line" (all 3 deleted; root CLAUDE.md carries the boundary).
- Drop section "OCR line" (eval basis, SFX state, stage paths deleted; manga109s one-liner → Data).
- Add the missing index line for feedback_no_teaching_stock_behaviour.
- Cut prose on index lines 2, 4, 21-23, 35, 37, 43, 55, 70 (they carry findings/numbers); title + ≤6 words each.
- Four sections remain: Who the user is / Working preferences / Infra & tooling / Data & caches.

Contradictions / repo→memory pointers to reverse
- CLAUDE.md l.77 spells `ruff check . --fix && ruff format .` then "(touched files only)" — command contradicts the parenthetical; also cites `[[feedback_ruff_scope_collateral]]`.
- CLAUDE.md l.223 "TE caches skip on existence only" vs library/preprocess/text.py mtime check (memory project_cache_invalidation_rules is right, CLAUDE.md is wrong).
- docs/optimizations/sigma_lowres.md l.14 → nonexistent `_archive/sigma_lowres/`.
- Repo files that currently point INTO memory (the direction the user wants reversed): project/finished/sr/STATUS.md l.55, docs/findings/README.md l.66 (xid8), CLAUDE.md (2 feedback cites), tag_glossary.py l.639, docs/methods/adaln.md (sea_delta), docs/experimental/soft_tokens.md + docs/methods/turbo.md + docs/findings/freetext_text_rendering.md (blockswap). Plus ~50 dead `[[project_*]]` IDs in docs/methods/{turbo,adaln}.md, docs/findings/*, bench/turbo/dual_pool/README.md, docs/experimental/vr_loss.md — memories that no longer exist.
- No feedback memory contradicts another; feedback_default_block_compile vs the qwen activation-checkpointing exception is resolved by library/qwen21/CLAUDE.md l.34.

Totals: 20 delete (4 of them stale/obsolete; 2 need a paste into STATUS.md / turbo.md first), 30 trim, 6 keep.
