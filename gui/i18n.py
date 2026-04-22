"""Internationalization for the Anima LoRA GUI."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Window / tabs
        "window_title": "Anima LoRA",
        "tab_config": "Config",
        "tab_graft": "GRAFT",
        "tab_images": "Images",
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
        # GraftTab
        "iterations": "Iterations",
        "refresh": "Refresh",
        "select_all": "Select All",
        "invert": "Invert",
        "deselect": "Deselect",
        "n_images": "{n} images",
        "n_images_selected": "{n} images, {s} selected",
        "delete_selected": "Delete Selected",
        "delete": "Delete",
        "delete_confirm": "Delete {n} image(s)?",
        "graft_config": "GRAFT Config",
        "save_graft_config": "Save GRAFT Config",
        "graft_saved": "GRAFT config saved.",
        # ImageViewerTab
        "directory": "Directory:",
        "caption": "Caption:",
        "no_caption": "(no caption)",
        # Language
        "language": "Language:",
        # Guidebook
        "guidebook": "📖 Guide",
        "guidebook_tooltip": "Open the Korean end-to-end guide (docs/guidelines/가이드북.md)",
        "guidebook_missing": "Guide not found at {path}",
        "guidebook_open_external": "Open in system viewer",
        "guidebook_close": "Close",
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
        "tab_config": "설정",
        "tab_graft": "GRAFT",
        "tab_images": "이미지",
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
        # GraftTab
        "iterations": "반복",
        "refresh": "새로고침",
        "select_all": "모두 선택",
        "invert": "선택 반전",
        "deselect": "선택 해제",
        "n_images": "이미지 {n}개",
        "n_images_selected": "이미지 {n}개, {s}개 선택됨",
        "delete_selected": "선택 삭제",
        "delete": "삭제",
        "delete_confirm": "이미지 {n}개를 삭제하시겠습니까?",
        "graft_config": "GRAFT 설정",
        "save_graft_config": "GRAFT 설정 저장",
        "graft_saved": "GRAFT 설정이 저장되었습니다.",
        # ImageViewerTab
        "directory": "디렉토리:",
        "caption": "캡션:",
        "no_caption": "(캡션 없음)",
        # Language
        "language": "언어:",
        # Guidebook
        "guidebook": "📖 가이드북",
        "guidebook_tooltip": "한국어 종합 가이드 열기 (docs/guidelines/가이드북.md)",
        "guidebook_missing": "가이드를 찾을 수 없습니다: {path}",
        "guidebook_open_external": "시스템 뷰어로 열기",
        "guidebook_close": "닫기",
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
