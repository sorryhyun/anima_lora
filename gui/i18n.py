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
        "stop": "Stop",
        "log_placeholder": "Training output will appear here...",
        "from_base": "From base.toml",
        "dataset_config": "Dataset Config",
        "save_dataset_config": "Save Dataset Config",
        "saved": "Saved",
        "saved_file": "Saved {name}",
        "dataset_saved": "Dataset config saved.",
        "invalid_toml": "Invalid TOML",
        "error": "Error",
        "accelerate_not_found": "accelerate not found on PATH",
        "preprocess": "Preprocess",
        "preprocess_required": "Please run Preprocess before training.",
        "finished": "--- Finished (exit code {code}) ---",
        "locked_by_preset": "Locked by preset (performance settings are fixed for this VRAM profile)",
        "lora_variants": "LoRA Variants",

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
        "stop": "정지",
        "log_placeholder": "학습 출력이 여기에 표시됩니다...",
        "from_base": "base.toml에서 상속",
        "dataset_config": "데이터셋 설정",
        "save_dataset_config": "데이터셋 설정 저장",
        "saved": "저장 완료",
        "saved_file": "{name} 저장됨",
        "dataset_saved": "데이터셋 설정이 저장되었습니다.",
        "invalid_toml": "잘못된 TOML",
        "error": "오류",
        "accelerate_not_found": "PATH에서 accelerate를 찾을 수 없습니다",
        "preprocess": "전처리",
        "preprocess_required": "학습 전에 전처리를 먼저 실행해주세요.",
        "finished": "--- 완료 (종료 코드 {code}) ---",
        "locked_by_preset": "프리셋에 의해 잠김 (이 VRAM 프로필의 성능 설정은 고정되어 있습니다)",
        "lora_variants": "LoRA 변형",

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
            _current_lang = json.loads(p.read_text(encoding="utf-8")).get("language", "en")
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
