"""Internationalization for the Anima LoRA GUI."""

from __future__ import annotations

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # Window / tabs
        "window_title": "Anima LoRA",
        "tab_config": "Config",
        "tab_graft": "GRAFT",
        "tab_images": "Images",
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
        "no_lora_for_test": "No LoRA in output/ to test. Run training first.",
        "test_output_title": "Latest test output",
        "test_output_empty": "test_output/ is empty.",
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
    },
    "ko": {
        # Window / tabs
        "window_title": "Anima LoRA",
        "tab_config": "설정",
        "tab_graft": "GRAFT",
        "tab_images": "이미지",
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
        "no_lora_for_test": "테스트할 LoRA가 output/에 없습니다. 먼저 학습을 실행하세요.",
        "test_output_title": "최신 테스트 출력",
        "test_output_empty": "test_output/가 비어 있습니다.",
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
