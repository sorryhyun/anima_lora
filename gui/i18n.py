"""Internationalization for the Anima LoRA GUI."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Window / tabs
        "window_title": "Anima LoRA",
        "tab_config": "Training Config",
        "tab_ip_adapter": "IP-Adapter",
        "tab_easycontrol": "EasyControl",
        "tab_postfix": "Postfix",
        "tab_images": "Dataset",
        "tab_merge": "Merge",
        # ConfigTab
        "preset": "Preset:",
        "save": "Save",
        "train": "Train",
        "test": "Test",
        "stop": "Stop",
        "log_placeholder": "Training output will appear here...",
        "from_base": "From base.toml",
        "saved": "Saved",
        "saved_file": "Saved {name}",
        "invalid_toml": "Invalid TOML",
        "error": "Error",
        "accelerate_not_found": "accelerate not found on PATH",
        "preprocess": "Preprocess",
        "preprocess_required": "Please run Preprocess before training.",
        "no_lora_for_test": "No LoRA in output/ckpt/ to test. Run training first.",
        "test_output_title": "Latest test output",
        "test_output_empty": "output/tests/ is empty.",
        "finished": "--- Finished (exit code {code}) ---",
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
        # AdapterTab (IP-Adapter / EasyControl)
        "adapter_source_dir": "Source dataset:",
        "adapter_cache_dir": "Cache directory:",
        "adapter_n_pairs": "{n} image / {c} caption pairs",
        "adapter_n_caches": "{n} cached",
        "adapter_preprocess": "Preprocess (resize + VAE + text)",
        "adapter_preprocess_pe": "Preprocess (resize + VAE + text + PE)",
        "adapter_train": "Train",
        "adapter_stop": "Stop",
        "adapter_log_placeholder": "Run output will appear here...",
        "adapter_no_dataset": "Source dataset directory does not exist. Create it and drop in image+caption pairs.",
        "adapter_open_dir": "Open directory",
        "n_images": "{n} images",
        # ImageViewerTab
        "directory": "Directory:",
        "caption": "Caption:",
        "no_caption": "(no caption)",
        "caption_save": "Save",
        "caption_revert": "Revert",
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
            "everything after is shuffled within each section."
        ),
        # Language
        "language": "Language:",
        # Guidebook
        "guidebook": "📖 Guide",
        "guidebook_tooltip": "Open the Korean end-to-end guide (docs/guidelines/가이드북.md)",
        "guidebook_missing": "Guide not found at {path}",
        "guidebook_open_external": "Open in system viewer",
        "guidebook_close": "Close",
        # Top-bar buttons (models / update / report issue)
        "models_btn": "Models",
        "models_btn_tooltip": "Download or re-download model checkpoints (Anima base, SAM3, MIT, IP-Adapter encoders)",
        "update_btn": "Update",
        "update_btn_tooltip": "Pull the latest anima_lora release from GitHub and run uv sync",
        "report_issue": "Report Issue",
        "report_issue_tooltip": "Open the GitHub issue tracker in your browser",
        "experimental_features": "🧪 Experimental",
        "experimental_features_tooltip": "Open Postfix, IP-Adapter, and EasyControl tabs (experimental methods)",
        "experimental_features_title": "Experimental Features",
        # Models dialog
        "models_title": "Download Models",
        "models_intro": "Pick a model group below or use 'Download all' for the standard set "
        "(Anima + SAM3 + MIT + TIPSv2). Files are saved under models/.",
        "models_download_all": "Download all (Anima + SAM3 + MIT + TIPSv2)",
        "models_download": "Download",
        "models_redownload": "Re-download",
        "models_installed": "✓ Installed",
        "models_missing": "✗ Missing",
        "model_anima": "Anima — DiT + text encoder + VAE",
        "model_sam3": "SAM3 — text-bubble masking",
        "model_mit": "MIT — manga text masking",
        "model_tipsv2": "TIPSv2-L/14 — img2emb encoder",
        "model_pe": "PE-Core-L14-336 — IP-Adapter vision encoder",
        "model_pe_g": "PE-Core-G14-448 — larger IP-Adapter vision encoder",
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
        "merge_verdict_partial": "△ Partial — LoRA bakeable, ReFT will be dropped",
        "merge_verdict_hydra": "✗ HydraLoRA moe — layer-local router can't be baked",
        "merge_verdict_postfix_only": "✗ Postfix/prefix only — not a weight delta",
        "merge_verdict_reft_only": "✗ ReFT only — block-level hook, no LoRA to bake",
        "merge_verdict_unknown": "? No recognized adapter keys",
        "merge_options": "Merge Options",
        "merge_base_dit": "Base DiT:",
        "merge_multiplier": "Multiplier:",
        "merge_multiplier_tip": "LoRA strength to bake in (1.0 = full strength).",
        "merge_dtype": "Save dtype:",
        "merge_out": "Output:",
        "merge_out_placeholder": "(auto: <adapter>_merged.safetensors)",
        "merge_allow_partial": "Allow partial merge (drop ReFT / Hydra / postfix keys)",
        "merge_allow_partial_tip": "Proceed even if the adapter contains non-bakeable components. Dropped components will be absent from the merged DiT.",
        "merge_button": "Merge into DiT",
        "merge_log_placeholder": "Merge output will appear here...",
        "merge_pick_dir": "Select adapter directory",
        "merge_pick_file": "Select adapter .safetensors",
        "merge_pick_dit": "Select base DiT .safetensors",
        "merge_pick_out": "Save merged DiT as...",
        "browse": "Browse…",
    },
    "ko": {
        # Window / tabs
        "window_title": "Anima LoRA",
        "tab_config": "학습 설정",
        "tab_ip_adapter": "IP-Adapter",
        "tab_easycontrol": "EasyControl",
        "tab_postfix": "Postfix",
        "tab_images": "데이터셋",
        "tab_merge": "병합",
        # ConfigTab
        "preset": "프리셋:",
        "save": "저장",
        "train": "학습",
        "test": "테스트",
        "stop": "정지",
        "log_placeholder": "학습 출력이 여기에 표시됩니다...",
        "from_base": "base.toml에서 상속",
        "saved": "저장 완료",
        "saved_file": "{name} 저장됨",
        "invalid_toml": "잘못된 TOML",
        "error": "오류",
        "accelerate_not_found": "PATH에서 accelerate를 찾을 수 없습니다",
        "preprocess": "전처리",
        "preprocess_required": "학습 전에 전처리를 먼저 실행해주세요.",
        "no_lora_for_test": "테스트할 LoRA가 output/ckpt/에 없습니다. 먼저 학습을 실행하세요.",
        "test_output_title": "최신 테스트 출력",
        "test_output_empty": "output/tests/가 비어 있습니다.",
        "finished": "--- 완료 (종료 코드 {code}) ---",
        "locked_by_preset": "프리셋에 의해 잠김 (이 VRAM 프로필의 성능 설정은 고정되어 있습니다)",
        "lora_variants": "LoRA 변형",
        "variant": "변형:",
        "apply_variant": "적용",
        "apply_variant_tooltip": "아래 폼을 이 variant의 프리셋 값으로 채웁니다. Save를 누르기 전까지는 디스크에 저장되지 않습니다.",
        "show_guide": "가이드",
        "show_guide_tooltip": "오른쪽 패널에 variant 가이드와 Apply 동작 설명을 표시합니다.",
        "click_field_for_help": "필드 라벨을 클릭하면 설명이 여기에 표시됩니다.",
        "no_help_available": "이 필드에 대한 설명이 없습니다.",
        "extra_args_toggle": "+ 추가 인자",
        "extra_args_placeholder": "폼에 없는 필드를 TOML 형식으로 입력. 예:\nmy_new_flag = true\nsome_value = 5e-5",
        "extra_args_tooltip": "폼에 없는 설정 키를 추가합니다. Save 시 TOML로 파싱되어 현재 variant 파일에 병합되며, 폼이 새로고침되어 위젯으로 표시됩니다. 동일 키가 폼에도 있는 경우 여기 입력값이 우선합니다.",
        "new_variant": "+ 새 Variant",
        "new_variant_tooltip": "configs/gui-methods/custom/<name>.toml에 새 커스텀 variant를 생성합니다.",
        "new_variant_prompt": "새 variant 이름 (configs/gui-methods/custom/<name>.toml에 저장됨).\n영문/숫자/_/- 만 사용 가능합니다.",
        "new_variant_invalid": "잘못된 이름. 영문, 숫자, _, - 만 사용 가능합니다.",
        "new_variant_exists": "Variant '{name}'이(가) 이미 존재합니다.",
        "basic_section": "기본 설정",
        "advanced_section": "고급 설정 (클릭하여 펼치기)",
        # AdapterTab (IP-Adapter / EasyControl)
        "adapter_source_dir": "소스 데이터셋:",
        "adapter_cache_dir": "캐시 디렉토리:",
        "adapter_n_pairs": "이미지 {n}개 / 캡션 {c}개 쌍",
        "adapter_n_caches": "캐시 {n}개",
        "adapter_preprocess": "전처리 (리사이즈 + VAE + 텍스트)",
        "adapter_preprocess_pe": "전처리 (리사이즈 + VAE + 텍스트 + PE)",
        "adapter_train": "학습",
        "adapter_stop": "정지",
        "adapter_log_placeholder": "실행 출력이 여기에 표시됩니다...",
        "adapter_no_dataset": "소스 데이터셋 디렉토리가 없습니다. 디렉토리를 만들고 이미지+캡션 쌍을 넣어주세요.",
        "adapter_open_dir": "디렉토리 열기",
        "n_images": "이미지 {n}개",
        # ImageViewerTab
        "directory": "디렉토리:",
        "caption": "캡션:",
        "no_caption": "(캡션 없음)",
        "caption_save": "저장",
        "caption_revert": "되돌리기",
        "caption_versions": "이력…",
        "caption_dirty_marker": " *",
        "caption_diff_stats": "(+{add} / −{rem})",
        "caption_diff_clean": "(변경 없음)",
        "caption_save_failed": "캡션 저장 실패: {err}",
        "caption_unsaved_title": "저장되지 않은 캡션",
        "caption_unsaved_body": "캡션 편집 사항이 저장되지 않았습니다. 전환하기 전에 저장할까요?",
        "caption_versions_title": "캡션 이력 — {name}",
        "caption_versions_empty": "(이전 버전 없음)",
        "caption_versions_restore": "선택 버전으로 되돌리기",
        "caption_versions_close": "닫기",
        "caption_no_history": "이 캡션에는 아직 이력이 없습니다.",
        "caption_guideline_html": (
            "<b>순서:</b> 등급 → 인원수 → 캐릭터 (작품) → 작품 → "
            "<span style='color:#c9a227;'>@작가</span> → 내용 태그. "
            "영역별 하위 섹션: 직전 태그를 <code>.</code> 으로 끝낸 뒤 "
            "<span style='color:#5e8eb0;'>On the&nbsp;…,</span> 또는 "
            "<span style='color:#5e8eb0;'>In the&nbsp;…,</span> 로 시작. "
            "첫 <code>@작가</code> 태그까지는 순서가 고정되고, 그 이후는 "
            "섹션 내에서 셔플됩니다."
        ),
        # Language
        "language": "언어:",
        # Guidebook
        "guidebook": "📖 가이드북",
        "guidebook_tooltip": "한국어 종합 가이드 열기 (docs/guidelines/가이드북.md)",
        "guidebook_missing": "가이드를 찾을 수 없습니다: {path}",
        "guidebook_open_external": "시스템 뷰어로 열기",
        "guidebook_close": "닫기",
        # Top-bar buttons (models / update / report issue)
        "models_btn": "모델",
        "models_btn_tooltip": "모델 체크포인트 다운로드 / 재다운로드 (Anima 베이스, SAM3, MIT, IP-Adapter 인코더)",
        "update_btn": "업데이트",
        "update_btn_tooltip": "GitHub에서 최신 anima_lora 릴리스를 가져오고 uv sync를 실행합니다",
        "report_issue": "이슈 신고",
        "report_issue_tooltip": "브라우저에서 GitHub 이슈 트래커 열기",
        "experimental_features": "🧪 실험 기능",
        "experimental_features_tooltip": "Postfix, IP-Adapter, EasyControl 탭 열기 (실험적 학습 방식)",
        "experimental_features_title": "실험 기능",
        # Models dialog
        "models_title": "모델 다운로드",
        "models_intro": "아래에서 모델 그룹을 선택하거나 '전체 다운로드'로 표준 세트 "
        "(Anima + SAM3 + MIT + TIPSv2)를 받으세요. 파일은 models/ 아래에 저장됩니다.",
        "models_download_all": "전체 다운로드 (Anima + SAM3 + MIT + TIPSv2)",
        "models_download": "다운로드",
        "models_redownload": "재다운로드",
        "models_installed": "✓ 설치됨",
        "models_missing": "✗ 없음",
        "model_anima": "Anima — DiT + 텍스트 인코더 + VAE",
        "model_sam3": "SAM3 — 말풍선 마스킹",
        "model_mit": "MIT — 만화 텍스트 마스킹",
        "model_tipsv2": "TIPSv2-L/14 — img2emb 인코더",
        "model_pe": "PE-Core-L14-336 — IP-Adapter 비전 인코더",
        "model_pe_g": "PE-Core-G14-448 — 대형 IP-Adapter 비전 인코더",
        # Update dialog
        "update_title": "anima_lora 업데이트",
        "update_warning": "업데이트는 GitHub에서 최신 릴리스를 받아 작업 트리를 덮어씁니다 "
        "(datasets, output/, models/는 보존됩니다). configs/methods/와 "
        "configs/gui-methods/에 직접 수정한 내용은, 그대로 유지할지 또는 "
        "최신 버전으로 덮어쓸지(기존 파일은 자동 백업됨) 선택하세요. "
        "먼저 'Dry run'으로 변경사항을 미리 확인할 수 있습니다.",
        "update_dry_run": "Dry run",
        "update_run": "업데이트 실행",
        "update_run_keep": "업데이트 — 내 설정 유지",
        "update_run_overwrite": "업데이트 — 설정 덮어쓰기 (기존 백업)",
        "update_confirm": "anima_lora 소스 파일이 다시 작성됩니다. 계속하시겠습니까?",
        "update_check_now": "업데이트 확인",
        "update_view_release": "GitHub에서 보기",
        "update_current_version": "현재: {v}",
        "update_latest_version": "최신: {v}",
        "update_no_baseline": "알 수 없음 (manifest 없음)",
        "update_status_checking": "확인 중…",
        "update_status_uptodate": "✓ 최신 버전입니다",
        "update_status_available": "● 업데이트 있음",
        "update_status_unknown": "? 비교 불가 (로컬 manifest 없음)",
        "update_status_failed": "✗ 확인 실패",
        "update_release_notes": "릴리스 노트:",
        "update_no_release_notes": "(릴리스 설명이 없습니다)",
        "update_check_error": "GitHub에 접속할 수 없습니다: {err}",
        # MergeTab
        "n_files": "파일 {n}개",
        "merge_no_adapter": "어댑터를 찾을 수 없습니다",
        "merge_no_adapter_msg": "어댑터가 선택되지 않았거나 파일이 존재하지 않습니다.",
        "merge_no_selection": "목록에서 체크포인트를 선택하여 스캔하세요.",
        "merge_verdict_ready": "✓ 병합 준비됨",
        "merge_verdict_partial": "△ 부분 병합 — LoRA는 병합되고 ReFT는 제외됩니다",
        "merge_verdict_hydra": "✗ HydraLoRA moe — 레이어 로컬 라우터는 병합할 수 없습니다",
        "merge_verdict_postfix_only": "✗ postfix/prefix 전용 — 가중치 델타가 아닙니다",
        "merge_verdict_reft_only": "✗ ReFT 전용 — 블록 후크만 있고 병합할 LoRA가 없습니다",
        "merge_verdict_unknown": "? 인식되는 어댑터 키가 없습니다",
        "merge_options": "병합 옵션",
        "merge_base_dit": "베이스 DiT:",
        "merge_multiplier": "강도 배수:",
        "merge_multiplier_tip": "병합 시 적용할 LoRA 강도 (1.0 = 전체 강도)",
        "merge_dtype": "저장 dtype:",
        "merge_out": "출력:",
        "merge_out_placeholder": "(자동: <adapter>_merged.safetensors)",
        "merge_allow_partial": "부분 병합 허용 (ReFT / Hydra / postfix 키 제외)",
        "merge_allow_partial_tip": "병합 불가능한 컴포넌트가 있어도 진행합니다. 제외된 컴포넌트는 병합된 DiT에 반영되지 않습니다.",
        "merge_button": "DiT에 병합",
        "merge_log_placeholder": "병합 출력이 여기에 표시됩니다...",
        "merge_pick_dir": "어댑터 디렉토리 선택",
        "merge_pick_file": "어댑터 .safetensors 선택",
        "merge_pick_dit": "베이스 DiT .safetensors 선택",
        "merge_pick_out": "병합된 DiT 저장 위치...",
        "browse": "찾아보기…",
    },
}

_current_lang = "en"
_SETTINGS_FILE = None


def _settings_path():
    global _SETTINGS_FILE
    if _SETTINGS_FILE is None:
        from pathlib import Path

        _SETTINGS_FILE = Path(__file__).resolve().parent / "gui_settings.json"
    return _SETTINGS_FILE


def load_language() -> str:
    """Load saved language preference."""
    global _current_lang
    import json

    p = _settings_path()
    if p.exists():
        try:
            _current_lang = json.loads(p.read_text(encoding="utf-8")).get(
                "language", "en"
            )
        except (json.JSONDecodeError, OSError):
            _current_lang = "en"
    return _current_lang


def save_language(lang: str):
    """Persist language preference."""
    global _current_lang
    import json

    _current_lang = lang
    p = _settings_path()
    settings = {}
    if p.exists():
        try:
            settings = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    settings["language"] = lang
    p.write_text(json.dumps(settings), encoding="utf-8")


def set_language(lang: str):
    global _current_lang
    _current_lang = lang


def t(key: str, **kwargs) -> str:
    """Translate a key using the current language."""
    s = TRANSLATIONS.get(_current_lang, TRANSLATIONS["en"]).get(key)
    if s is None:
        s = TRANSLATIONS["en"].get(key, key)
    if kwargs:
        s = s.format(**kwargs)
    return s


def current_language() -> str:
    return _current_lang


def available_languages() -> list[str]:
    return list(TRANSLATIONS.keys())
