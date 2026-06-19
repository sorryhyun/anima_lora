"""Korean strings for the Anima LoRA GUI."""

from __future__ import annotations

STRINGS: dict[str, str] = {
    # Window / tabs
    "window_title": "Anima LoRA",
    "tab_config": "학습 설정",
    "tab_easycontrol": "EasyControl",
    "tab_spd": "SPD",
    "tab_turbo": "Turbo",
    "tab_experimental": "실험기능",
    "tab_images": "데이터셋 뷰어",
    "tab_merge": "병합",
    "tab_queue": "큐 현황",
    "tab_preprocess": "전처리",
    "tab_tensorboard": "텐서보드",
    # PreprocessingTab
    "preprocess_intro": (
        "캡션 셔플과 말풍선 마스킹을 설정하고, 각 단계를 원할 때 실행합니다. "
        "학습 설정 탭의 학습 버튼은 캐시가 없을 때 기본값으로 전처리를 "
        "자동 실행합니다 — 이 탭은 세부 조정 및 단계별 재실행용입니다."
    ),
    "preprocess_image_prep": "이미지 전처리 (리사이즈 / 필터)",
    "preprocess_source_image_dir": "소스 이미지 폴더:",
    "preprocess_source_image_dir_tip": (
        "선택한 GUI method의 베이스 원본 이미지 루트입니다 (기본값은 configs/preprocess.toml; "
        "편집 내용은 해당 variant에 저장됩니다). 실행 시 path_scope가 이 위에 "
        "덧붙여지므로, 여기서 표시되는 것은 최종 스코프 경로가 아닌 스코프 없는 루트입니다. "
        "파일 저장 위치를 바꾸지 않고 트리의 일부만 전처리하려면 아래 전처리 경로 필터를 사용하세요."
    ),
    "preprocess_path_pattern": "전처리 경로 필터:",
    "preprocess_path_pattern_tip": (
        "path_scope가 먼저 소스 이미지 폴더를 정합니다. 예를 들어 path_scope=data_group1이면 "
        "전처리 루트는 image_dataset/data_group1이 됩니다. 이 필터는 그 루트 기준 상대 경로에 "
        "적용됩니다. '*'(또는 빈 값)은 전체, '1/*'는 data_group1/1 하위만, "
        "'1/*|2/*'는 두 하위 폴더를 처리합니다."
    ),
    "preprocess_drop_lowres": "저해상도 이미지 제외",
    "preprocess_drop_lowres_tip": (
        "아래 픽셀 임계값보다 작은 소스 이미지를 건너뛰어 리사이즈 / VAE / "
        "텍스트 캐시에 포함되지 않도록 합니다. "
        "체크 해제 시 크기에 상관없이 모든 이미지를 유지합니다."
    ),
    "preprocess_min_pixels": "최소 픽셀 수 (필터 임계값):",
    "preprocess_min_pixels_tip": (
        "저해상도 필터의 픽셀 수 임계값. 500000 = 0.5MP. "
        "'저해상도 이미지 제외'가 해제되면 무시됩니다."
    ),
    "preprocess_target_res": "해상도 티어 (target_res):",
    "preprocess_freefit_max_ratio": "최대 종횡비:",
    "preprocess_freefit_max_ratio_tip": (
        "프리핏 클램프: 1:R / R:1을 초과하는 종횡비의 이미지는 "
        "위에서 선택한 크롭 위치에 따라 커버 크롭됩니다. "
        "기본값 4.0은 기존 버킷 테이블의 가장 긴 종횡비와 일치하며 "
        "1:5 / 1:6 같은 극단적인 입력을 차단합니다."
    ),
    "preprocess_text_caching": "캐싱 (VAE + 텍스트)",
    "preprocess_caption_shuffle_variants": "캡션당 셔플 변형 수 (N):",
    "preprocess_caption_shuffle_variants_tip": (
        "이미지당 N개의 캡션 변형을 생성합니다. v0은 원본 그대로이고, "
        "v1..v(N-1)은 스마트 셔플되며 (태그 드롭아웃 > 0이면) @artist 이후 "
        "태그가 독립적으로 드롭됩니다. use_shuffled_caption_variants=true일 때 "
        "데이터로더는 v0을 20% 확률로, 나머지는 v1..v(N-1) 균등 분포로 선택합니다. "
        "0으로 설정하면 원본 캡션 하나만 캐싱합니다."
    ),
    "preprocess_caption_tag_dropout_rate": "태그 드롭아웃 비율 (0.0–1.0):",
    "preprocess_caption_tag_dropout_rate_tip": (
        "v1..v(N-1)에만 적용되는 태그별 드롭아웃 확률입니다. "
        "첫 번째 @artist 마커까지의 태그는 절대 드롭되지 않습니다. "
        "셔플 변형 수가 0 이하이면 무시됩니다."
    ),
    "preprocess_caption_correct_order": "캡션 순서 교정",
    "preprocess_caption_correct_order_tip": (
        "교정된 .txt 캡션을 리사이즈 이미지 옆에 저장하고 텍스트 인코더 "
        "캐싱에 사용합니다. 원본 소스 캡션은 수정하지 않습니다."
    ),
    "preprocess_caption_insert_no_artist": "@no-artist 삽입",
    "preprocess_caption_insert_no_artist_tip": (
        "캡션 순서 교정이 켜져 있을 때 작가 마커가 없으면 artist 위치에 "
        "@no-artist를 넣습니다. 트리거 워드가 artist 위치에 있으면 "
        "트리거를 우선하며 @no-artist는 넣지 않습니다."
    ),
    "preprocess_caption_trigger_word": "트리거 워드:",
    "preprocess_caption_trigger_word_tip": (
        "교정 캡션에 배치할 선택 트리거 태그입니다. 맨 앞 고정 옵션이 "
        "꺼져 있으면 기존 작가 태그보다 앞의 artist 위치에 둡니다. "
        "셔플/드롭아웃 보호용 작가 마커처럼 쓰려면 @를 포함하세요. "
        "트리거 워드의 underscore는 보존됩니다."
    ),
    "preprocess_caption_trigger_at_front": "트리거를 맨 앞에 고정",
    "preprocess_caption_trigger_at_front_tip": (
        "트리거 워드를 캡션의 가장 앞에 둡니다. 이 모드에서는 "
        "@no-artist 삽입이 기존 작가 태그 유무와 no-artist 옵션에 따라 "
        "별도로 동작합니다."
    ),
    "preprocess_run_te": "캐싱 실행 (VAE + 텍스트)",
    "preprocess_run_pe": "PE 캐싱 실행",
    "preprocess_add_to_queue": "큐에 추가",
    "preprocess_queued": "{label} 큐에 추가됨 (작업 {job_id}) — 큐 탭에서 확인하세요.",
    "preprocess_masking_sam": "SAM3 마스킹 (말풍선)",
    "preprocess_masking_mit": "MIT 마스킹 (만화 텍스트)",
    "preprocess_sam_prompts": "SAM 프롬프트 (한 줄에 하나):",
    "preprocess_sam_prompts_tip": (
        "SAM3이 찾을 텍스트 프롬프트. 한 줄에 하나씩. "
        "기본값: 'speech bubble', 'text bubble'."
    ),
    "preprocess_sam_focus_prompts": "SAM 포커스 프롬프트 (한 줄에 하나):",
    "preprocess_sam_focus_prompts_tip": (
        "반전 극성: 유지할 피사체를 지정합니다. 설정하면 마스크는 해당 피사체에만 "
        "학습이 적용되고 나머지는 모두 무시됩니다 (예: 'girl'을 지정하면 배경 전체가 "
        "무시 영역이 됩니다). 위의 프롬프트와 합성되어, 최종 학습 영역은 포커스 "
        "피사체에서 무시 영역을 뺀 부분이 됩니다. 비워두면 기본 무시 전용 동작이 "
        "적용됩니다."
    ),
    "preprocess_sam_rule": "마스크 규칙",
    "preprocess_sam_add_rule": "+ 규칙 추가",
    "preprocess_sam_add_rule_tip": (
        "마스크 규칙을 하나 더 추가합니다. 각 규칙은 경로 패턴으로 이미지 "
        "서브셋을 대상으로 하며, 패턴이 일치하는 규칙들은 서로 합성됩니다."
    ),
    "preprocess_sam_remove_rule": "규칙 삭제",
    "preprocess_sam_rule_path_pattern": "경로 패턴 (이 규칙):",
    "preprocess_sam_rule_path_pattern_tip": (
        "이 규칙이 적용될 이미지를 지정합니다 — 데이터셋 루트 기준 각 이미지 "
        "경로에 대한 fnmatch 글로브 ('|'로 OR 조합). 예: 'character_a/*'. "
        "빈 값 또는 '*'는 모든 이미지에 매칭되는 기본 규칙입니다."
    ),
    "preprocess_sam_threshold": "SAM 임계값 (0.0–1.0):",
    "preprocess_sam_threshold_tip": (
        "SAM3 탐지를 유지할 최소 신뢰도. 낮을수록 더 많은 마스크 "
        "(오탐 포함 가능), 높을수록 엄격. 기본값 0.5."
    ),
    "preprocess_dilate": "팽창 (px):",
    "preprocess_dilate_tip": (
        "이진 마스크에 적용할 팽창 픽셀 수. 값이 클수록 마스크 가장자리가 "
        "바깥으로 번집니다. 기본값 5. 0으로 비활성화."
    ),
    "preprocess_mit_threshold": "MIT 텍스트 임계값 (0.0–1.0):",
    "preprocess_mit_threshold_tip": (
        "MIT/ComicTextDetector 텍스트 세그멘터의 신뢰도 임계값. 기본값 0.8."
    ),
    "preprocess_mask_path_pattern": "마스크 경로 필터:",
    "preprocess_mask_path_pattern_tip": (
        "마스킹할 리사이즈 이미지를 제한하는 fnmatch glob 패턴. "
        "post_image_dataset/resized 기준 각 경로에 대해 매칭됩니다. "
        "SAM과 MIT 모두에 적용됩니다. 학습용 path_pattern과 동일한 문법: "
        "'*'(또는 빈 값)이면 전체 마스킹; 'char_a/*'이면 한 하위 폴더; "
        "'char_a/*|char_b/*'으로 OR 조합."
    ),
    "preprocess_run_mask": "마스킹 실행",
    "preprocess_run_sam_mask": "SAM 마스킹 실행",
    "preprocess_run_sam_mask_tip": (
        "마스크 생성 단계에서 SAM3 말풍선 분할을 실행합니다. "
        "체크 해제하면 SAM을 건너뛰고 MIT(또는 활성화된 다른 백엔드)만 사용합니다."
    ),
    "preprocess_run_mit_mask": "MIT 마스킹 실행",
    "preprocess_run_mit_mask_tip": (
        "마스크 생성 단계에서 MIT/ComicTextDetector 텍스트 분할을 "
        "실행합니다. 체크 해제하면 MIT를 건너뛰고 SAM만 사용합니다."
    ),
    "preprocess_mask_nothing_enabled": (
        "SAM 또는 MIT 마스킹 중 최소 하나는 활성화되어야 합니다."
    ),
    "preprocess_status_resized": "리사이즈된 이미지: {n}장",
    "preprocess_status_caches": "캐시 — latents: {lat}, text: {te}, PE: {pe}",
    "preprocess_status_masks": "마스크: {masks}장",
    "preprocess_status_no_resized": "리사이즈된 이미지가 없습니다.",
    "preprocess_open_dataset_dir": "캐시 폴더 열기",
    "preprocess_open_dataset_dir_tooltip": "post_image_dataset/ 폴더(리사이즈된 이미지 + 캐시)를 파일 탐색기에서 엽니다.",
    "preprocess_clear_scope_cache": "현재 scope 캐시 삭제",
    "preprocess_clear_scope_cache_tooltip": "현재 path_scope가 적용된 리사이즈 이미지와 LoRA 캐시 폴더를 삭제합니다.",
    "preprocess_clear_scope_cache_all_scope": "전체 scope 없음",
    "preprocess_clear_scope_cache_empty": "삭제할 리사이즈 이미지 또는 LoRA 캐시 파일이 없습니다.",
    "preprocess_clear_scope_cache_outside_root": "프로젝트 폴더 밖의 경로는 GUI에서 삭제하지 않습니다:\n{path}",
    "preprocess_clear_scope_cache_confirm": (
        "현재 scope의 전처리 파일을 삭제합니다.\n\n"
        "scope: {scope}\n"
        "resize: {resized}\n  파일 {resized_count}개\n"
        "lora: {lora}\n  파일 {lora_count}개\n\n"
        "삭제 후 전처리를 다시 실행해 재생성해야 합니다."
    ),
    "preprocess_clear_scope_cache_done": "전처리 파일 {count}개를 삭제했습니다.",
    "preprocess_invalid_path_scope": "path_scope 값이 올바르지 않습니다: {value}",
    "preprocess_log_placeholder": "전처리 출력이 여기에 표시됩니다...",
    "preprocess_save_settings": "저장",
    "preprocess_save_settings_tip": "이 설정들을 선택한 GUI method 프로필에 저장합니다. 마스킹 실행 시에는 현재 프로필의 마스크 설정이 작업에 함께 전달됩니다.",
    "preprocess_settings_saved": "전처리 설정이 저장되었습니다.",
    "preprocess_invalid_float": "{field}에 잘못된 숫자: {value}",
    "preprocess_already_running": "이미 전처리 단계가 실행 중입니다.",
    # ConfigTab
    "preset": "프리셋:",
    "save": "저장",
    "save_dirty_tooltip": "저장되지 않은 편집이 있습니다. Save를 누르면 variant 파일에 기록됩니다 (학습/전처리 시작 시 자동 저장됨).",
    "train": "학습",
    "train_tooltip": "현재 variant를 지금 학습합니다. 펼침 메뉴를 열면 지금 시작하지 않고 daemon 큐에 추가합니다.",
    "train_busy_use_queue": "이미 이 탭에 작업이 연결되어 있습니다. Train 펼침 메뉴로 다른 작업을 큐에 추가하거나 먼저 현재 작업을 중지하세요.",
    "queue": "큐 추가",
    "queue_tooltip": "현재 variant를 daemon 큐에 추가합니다. 펼침 메뉴에서 학습+전처리 또는 전처리만 선택할 수 있습니다.",
    "queue_train_preprocess": "큐 추가: 학습 + 전처리",
    "queue_train_only": "큐 추가: 학습만",
    "queue_preprocess_only": "큐 추가: 전처리만",
    "test": "테스트",
    "stop": "정지",
    "log_placeholder": "학습 출력이 여기에 표시됩니다...",
    "copy_log": "복사",
    "copy_log_tooltip": "전체 학습 로그를 클립보드에 복사",
    "copy_log_done": "복사됨",
    "gpu_probing": "GPU: 확인 중…",
    "gpu_stat": "GPU{i}: {util}%  ·  {used}/{total} GiB  ·  {temp}°C",
    "from_base": "base.toml에서 상속",
    "saved": "저장 완료",
    "saved_file": "{name} 저장됨",
    "invalid_toml": "잘못된 TOML",
    "config_bad_keys_header": "알 수 없는 데이터셋 키 — 이 키들을 제거하기 전까지 학습이 실패합니다:",
    "config_remove_keys_btn": "제거",
    "config_remove_keys_confirm": "이 오래된 키 {n}개를 설정 파일에서 삭제할까요?\n\n{keys}",
    "config_remove_keys_none": "제거된 키가 없습니다 (디스크의 해당 줄이 변경되었을 수 있습니다).",
    "error": "오류",
    "accelerate_not_found": "PATH에서 accelerate를 찾을 수 없습니다",
    "preprocess": "전처리",
    "preprocess_current_tooltip": "현재 variant의 GUI 경로 스코프를 적용해 전처리를 실행합니다.",
    "preprocess_required": "학습을 시작하면 전처리가 먼저 실행됩니다.",
    "preprocess_existing_caches_title": "기존 캐시를 그대로 재사용합니다",
    "preprocess_existing_caches_body": (
        "다음 경로에 이미 캐시 파일이 있습니다:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "전처리는 기존 캐시를 그대로 재사용합니다 — 삭제하거나 다시 "
        "만들지 않습니다. 누락된 항목만 새로 처리됩니다.\n\n"
        "캡션을 수정했거나 토크나이저 설정 등을 바꿔서 캐시를 처음부터 "
        "다시 만들고 싶다면, 취소를 누르고 캐시 폴더를 직접 삭제한 뒤 "
        "다시 실행하세요."
    ),
    "preprocess_cache_count_latents": "VAE 잠재변수 {n}개 (.npz)",
    "preprocess_cache_count_te": "텍스트 임베딩 {n}개 (_te.safetensors)",
    "preprocess_cache_count_pe": "PE 피처 {n}개 (_pe.safetensors)",
    "train_using_cache_title": "기존 캐시 데이터셋으로 학습할까요?",
    "train_using_cache_body": (
        "다음 경로에 이미 전처리된 데이터셋 캐시가 있습니다:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "학습은 이 캐시를 그대로 재사용합니다. 새 이미지를 추가했거나 "
        "캡션을 수정해서 다시 반영하고 싶다면, 취소를 누르고 먼저 "
        "전처리를 실행하세요.\n\n"
        "기존 캐시로 학습을 진행할까요?"
    ),
    "train_autopreprocess_log": (
        "전처리 캐시가 없어 학습 시작 전에 전처리를 먼저 실행합니다.\n"
    ),
    "train_preprocessing": "전처리 중…",
    "no_lora_for_test": "테스트할 LoRA가 output/ckpt/에 없습니다. 먼저 학습을 실행하세요.",
    "test_output_title": "최신 테스트 출력",
    "test_output_empty": "output/tests/가 비어 있습니다.",
    "sample_output_title": "최신 학습 샘플",
    "sample_output_empty": "아직 샘플이 없습니다 — 학습이 생성하면 출력 디렉터리의 sample/ 폴더에 나타납니다.",
    "sample_prompt_col_prompt": "프롬프트",
    "sample_prompt_col_width": "W",
    "sample_prompt_col_height": "H",
    "sample_prompt_col_steps": "스텝",
    "sample_prompt_col_seed": "시드",
    "sample_prompt_col_cfg": "CFG",
    "sample_prompt_col_guidance": "Guidance",
    "sample_prompt_col_shift": "Shift",
    "sample_prompt_col_negative": "네거티브",
    "sample_prompt_col_extra": "추가 옵션",
    "sample_prompt_add": "프롬프트 추가",
    "sample_prompt_select_all": "전체 선택",
    "sample_prompt_remove": "선택 삭제",
    "sample_prompt_remove_confirm_title": "샘플 프롬프트 삭제",
    "sample_prompt_remove_confirm_body": "선택한 샘플 프롬프트 {n}개를 삭제할까요?",
    "sample_prompt_expand": "입력창 키우기",
    "sample_prompt_collapse": "입력창 줄이기",
    "sample_prompt_edit_button": "샘플 프롬프트 편집…",
    "sample_prompt_dialog_title": "샘플 프롬프트",
    "sample_prompt_summary_none": "샘플 프롬프트 없음",
    "sample_prompt_summary_count": "{n}개 프롬프트 · {first}",
    "sample_prompt_select": "선택",
    "sample_prompt_prompt_placeholder": "프롬프트 본문. 줄바꿈은 여기서 보이고 저장 시 공백으로 정리됩니다.",
    "sample_prompt_hint": "기본값으로 표시된 항목은 프롬프트 줄에 저장하지 않습니다.",
    "sample_prompt_default_width": "기본 512",
    "sample_prompt_default_height": "기본 512",
    "sample_prompt_default_steps": "기본 30",
    "sample_prompt_default_seed": "자동 시드",
    "sample_prompt_default_cfg": "기본 7.5",
    "sample_prompt_default_guidance": "기본 1.0",
    "sample_prompt_default_shift": "기본 3.0",
    "sample_prompt_default_negative": "기본: 없음",
    "sample_prompt_tip_width": "이미지 폭(`--w`)입니다. 비우면 train.py 기본값 512를 사용합니다.",
    "sample_prompt_tip_height": "이미지 높이(`--h`)입니다. 비우면 train.py 기본값 512를 사용합니다.",
    "sample_prompt_tip_steps": "샘플링 스텝(`--s`)입니다. 비우면 train.py 기본값 30을 사용합니다.",
    "sample_prompt_tip_seed": "시드(`--d`)입니다. 자동 시드는 에폭 간 비교가 가능하도록 프롬프트별 기준을 유지합니다.",
    "sample_prompt_tip_cfg": "CFG scale(`--l`)입니다. 비우면 train.py 기본값 7.5를 사용합니다.",
    "sample_prompt_tip_guidance": "Guidance scale(`--g`)입니다. 비우면 train.py 기본값 1.0을 사용합니다.",
    "sample_prompt_tip_shift": "샘플링 시그마 스케줄의 flow shift(`--fs`)입니다. 비우면 train.py 기본값 3.0을 사용합니다.",
    "sample_prompt_tip_negative": "이 샘플에만 적용할 네거티브 프롬프트(`--n`)입니다.",
    "sample_prompt_tip_extra": "추가 sample 인자입니다. 입력한 원문을 그대로 보존합니다.",
    "finished": "--- 완료 (종료 코드 {code}) ---",
    "starting": "시작 중… (torch / accelerate 로딩)",
    "queue_submitting": "{variant}을(를) 학습 daemon 큐에 추가 중…",
    "queue_submitting_train_preprocess": "{variant}의 전처리와 학습을 daemon 큐에 추가 중…",
    "queue_submitting_preprocess": "{variant}의 전처리를 daemon 큐에 추가 중…",
    "queue_added_train": "{variant}을(를) 학습 job {job_id}(으)로 큐에 추가했습니다.\n",
    "queue_added_preprocess": "{variant}을(를) 전처리 job {job_id}(으)로 큐에 추가했습니다. 완료 후 학습이 이어집니다.\n",
    "queue_added_preprocess_only": "{variant}을(를) 전처리 job {job_id}(으)로 큐에 추가했습니다.\n",
    "queue_refresh": "새로고침",
    "queue_start": "큐 시작",
    "queue_pause": "큐 일시정지",
    "queue_start_tooltip": "대기 중인 작업(큐 드롭다운으로 추가한 작업)을 실행합니다. 한 번에 하나씩 처리합니다.",
    "queue_pause_tooltip": "큐를 멈춥니다 — 실행 중인 작업은 계속되지만, 큐 시작을 누를 때까지 다음 대기 작업은 시작되지 않습니다.",
    "queue_stop_selected": "선택 항목 정지",
    "queue_copy_output": "출력 복사",
    "queue_status": "진행/대기 {live}개 / 전체 {total}개",
    "queue_status_paused": "진행/대기 {live}개 / 전체 {total}개 — 큐 일시정지됨",
    "queue_daemon_unavailable": "daemon 연결 불가",
    "queue_detail_placeholder": "큐 항목을 선택하면 상세 정보가 표시됩니다.",
    "queue_log_placeholder": "선택한 job 출력이 여기에 표시됩니다...",
    "queue_log_missing": "(아직 출력 로그가 없습니다.)",
    "queue_log_read_failed": "(출력 로그를 읽을 수 없습니다: {err})",
    "queue_log_truncated": "--- 마지막 {mb} MB 출력만 표시 중 ---\n",
    "queue_detail_id": "id: {job_id}",
    "queue_detail_state": "상태: {state}",
    "queue_detail_kind": "종류: {kind}",
    "queue_detail_method": "대상: {method}",
    "queue_detail_submitted": "추가됨: {time}",
    "queue_detail_started": "시작됨: {time}",
    "queue_detail_ended": "종료됨: {time}",
    "queue_detail_from_chain": "전처리 chain에서 생성됨",
    "queue_detail_chain": "완료 후 학습: {method}",
    "queue_detail_chained_id": "연결된 job: {job_id}",
    "queue_detail_pid": "pid: {pid}",
    "queue_detail_error": "오류: {error}",
    "queue_detail_status_detail": "상세: {detail}",
    "queue_detail_config": "설정 스냅샷: {path}",
    "queue_detail_stdout": "출력 로그: {path}",
    "daemon_job_failed": "--- Job {job_id} {state}: {error} ---",
    "daemon_error_cause": "↳ 추정 원인: {summary}",
    "update_success_title": "업데이트 완료",
    "update_success_message": (
        "anima_lora이(가) {v}(으)로 업데이트되었습니다.\n\n"
        "변경 사항을 적용하려면 GUI를 종료하고 다시 실행해 주세요."
    ),
    "update_success_badge": "{v}(으)로 업데이트됨 (재실행 필요)",
    "update_dryrun_done_title": "드라이런 완료",
    "update_dryrun_done_message": (
        "드라이런이 완료되었습니다. 실제 변경된 파일은 없습니다. "
        "어떤 변경이 일어날지 로그를 확인하세요."
    ),
    "update_failed_title": "업데이트 실패",
    "update_failed_message": (
        "업데이트가 코드 {code}(으)로 종료되었습니다. "
        "자세한 내용은 로그를 확인하세요. 작업 트리가 일부만 변경되었을 수 있습니다."
    ),
    "resume_checkpoint_title": "학습을 재개할까요?",
    "resume_checkpoint_question": (
        "재개 가능한 체크포인트가 감지되었습니다 (스텝 {step}).\n\n"
        "• 예 — 스텝 {step}부터 학습을 재개합니다\n"
        "• 아니오 — 기존 체크포인트를 삭제하고 처음부터 새로 학습합니다\n"
        "• 취소 — 학습을 시작하지 않습니다"
    ),
    "resume_checkpoint_delete_failed": "기존 체크포인트 상태를 삭제하지 못했습니다:\n{error}",
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
    # SPD / Turbo 증류 설정 탭 (gui/tabs/distill_tab.py)
    "distill_general_section": "일반",
    "distill_job_running": "이 탭에서 이미 작업이 실행 중입니다.",
    "distill_config_missing": "설정 파일을 읽을 수 없습니다: {err}",
    "n_images": "이미지 {n}개",
    # ImageViewerTab
    "directory": "디렉토리:",
    "dataset_reload": "새로고침",
    "dataset_reload_tooltip": "현재 디렉토리를 다시 스캔해서 이미지 목록과 선택을 갱신합니다.",
    "dataset_open_dir": "열기",
    "dataset_open_dir_tooltip": "현재 디렉토리를 시스템 파일 관리자에서 엽니다.",
    "dataset_add_dir": "디렉토리 추가…",
    "dataset_add_dir_tooltip": "다른 디렉토리를 골라 이번 세션 동안 드롭다운에 추가합니다.",
    "dataset_add_dir_picker": "추가할 디렉토리 선택",
    "dataset_add_dir_already": "'{name}' 디렉토리는 이미 목록에 있습니다.",
    "dataset_search_placeholder": "파일 이름 검색…",
    "dataset_sort_asc_tooltip": "오름차순 정렬 (A→Z, 클릭하여 반전)",
    "dataset_sort_desc_tooltip": "내림차순 정렬 (Z→A, 클릭하여 반전)",
    "dataset_group_first_tooltip": "그룹 우선 정렬: 묶인 이미지를 폴더 구분 없이 맨 위로 모아서 보여줍니다 (그룹 외 이미지는 아래에 폴더 트리로).",
    "dataset_view_group": "그룹",
    "dataset_view_tree": "트리",
    "dataset_group_sort_tooltip": "그룹 내부 이미지 정렬 기준입니다. 방향키 이동은 왼쪽 트리에 보이는 순서를 따릅니다.",
    "dataset_group_sort_name": "이름순",
    "dataset_group_sort_name_desc": "이름 역순",
    "dataset_group_sort_size": "용량순",
    "dataset_group_sort_size_desc": "용량 역순",
    "dataset_group_sort_resolution": "해상도순",
    "dataset_group_sort_resolution_desc": "해상도 역순",
    "dataset_mask_overlay": "마스크 오버레이 표시",
    "dataset_resize_preview": "리사이즈 미리보기 표시",
    "dataset_resize_preview_tooltip": "전처리 target_res가 선택할 중앙 크롭 영역과 최종 bucket을 표시합니다. 원본 파일은 변경하지 않습니다.",
    "dataset_resize_preview_label": "{width}x{height} @ {edge}",
    "dataset_preprocess_use_short": "사용 (A)",
    "dataset_preprocess_use_tooltip": "현재 이미지를 전처리 대상에 포함하도록 표시합니다. 원본 파일은 변경하지 않습니다.",
    "dataset_preprocess_skip_short": "생략 (S)",
    "dataset_preprocess_skip_tooltip": "현재 이미지를 전처리 resize 단계에서 생략하도록 표시합니다. 원본 파일은 변경하지 않습니다.",
    "dataset_preprocess_clear_short": "해제 (F)",
    "dataset_preprocess_clear_tooltip": "현재 이미지의 사용/생략/이동 표시를 해제합니다. 우측 메뉴에서 전체 표시를 해제할 수 있습니다.",
    "dataset_preprocess_clear_all": "전체 표시 해제",
    "dataset_preprocess_save": "전처리 결정 저장",
    "dataset_preprocess_save_tooltip": "전처리에서 사용할 이미지별 사용/생략/이동 결정을 JSON으로 저장합니다. 이동 표시는 실제 이동 전에도 전처리에서 제외됩니다.",
    "dataset_preprocess_saved": "전처리 결정이 저장되었습니다:\n{path}",
    "dataset_preprocess_decision_use": "전처리 결정: 사용",
    "dataset_preprocess_decision_skip": "전처리 결정: 생략",
    "dataset_preprocess_decision_move": "현재 상태: 이동 예정",
    "dataset_image_meta_empty": "이미지 없음",
    "dataset_image_meta": "{width}x{height} · {size} · {fmt}",
    "dataset_image_meta_resize": "리사이즈 {width}x{height} @ {edge}",
    "dataset_delete": "이동 (D)",
    "dataset_delete_tooltip": "Delete 또는 D 키로 표시한 이미지를 사이드카와 함께 post_image_dataset/moved/로 이동합니다.",
    "dataset_delete_confirm_title": "이미지 이동",
    "dataset_delete_confirm_body": "이미지 {n}개와 사이드카를 post_image_dataset/moved/로 이동할까요?",
    "dataset_delete_failed": "일부 이미지를 이동하지 못했습니다:\n{err}",
    "dataset_group_label": "그룹 {n} — {size}장",
    "dataset_group_rebuild": "그룹화",
    "dataset_group_rebuild_tooltip": "PE-Spatial 시각적 유사도로 이미지 그룹화 (작가별). 작업 큐에서 실행됩니다.",
    "dataset_group_queued": "그룹화 작업이 큐에 추가됨 (작업 {job_id}). 완료되면 이 디렉터리를 새로고침하면 그룹이 보입니다.",
    "n_images_filtered": "{shown} / {total} 이미지",
    "caption": "캡션:",
    "no_caption": "(캡션 없음)",
    "caption_save": "저장",
    "caption_revert": "되돌리기",
    "caption_autotag": "자동 태깅",
    "caption_autotag_tooltip": (
        "Anima Tagger를 이 이미지에 실행해 예측된 태그를 캡션에 추가합니다. "
        "모델은 최초 사용 시 자동으로 내려받습니다. 결과를 확인한 뒤 저장하면 "
        ".txt 파일에 기록됩니다."
    ),
    "caption_autotag_running": "자동 태깅 중…",
    "caption_autotag_loading": "태거 로딩 중…",
    "caption_autotag_ready": "태거 로드됨 · 대기 중",
    "caption_autotag_busy": (
        "다른 작업(학습 / 전처리 / 그룹화)이 GPU를 사용 중입니다. "
        "완료된 뒤 다시 자동 태깅하세요."
    ),
    "caption_autotag_error": "자동 태깅 실패: {err}",
    "caption_autotag_empty": "태거가 이 이미지에서 태그를 찾지 못했습니다.",
    "caption_correct": "순서 교정",
    "caption_correct_tooltip": (
        "danbooru_tags_classified.csv를 사용해 캡션을 ANIMA 권장 순서로 "
        "재배치하고, 설정에 따라 @no-artist를 삽입합니다."
    ),
    "caption_correct_visible": "현재 목록 전체 교정",
    "caption_correct_visible_confirm": "현재 목록의 캡션 {n}개를 교정할까요?",
    "caption_correct_visible_done": "캡션 {n}개를 교정했습니다.",
    "caption_correct_visible_failed": "캡션 {n}개를 교정했습니다.\n\n실패:\n{err}",
    "caption_correct_no_change": "교정할 변경사항이 없습니다.",
    "caption_correct_db_missing": (
        "danbooru_tags_classified.csv를 찾을 수 없습니다.\n\n"
        "모델 창에서 Danbooru 태그 DB를 다운로드하거나 다음 위치에 배치하세요:\n{paths}"
    ),
    "caption_correct_db_failed": "태그 DB 로드 실패: {err}",
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
        "섹션 내에서 셔플됩니다. "
        "<b>작가 정보가 없을 때:</b> "
        "<span style='color:#c9a227;'>@no-artist</span> 를 자리표시자로 "
        "넣어주세요 — 셔플 경계 역할만 하고 토큰화 직전에 제거되어 "
        "모델까지 전달되지 않습니다."
    ),
    # Language
    "language": "언어:",
    # Settings dialog
    "settings_btn": "⚙ 설정",
    "settings_btn_tooltip": "앱 설정 — 언어, 환경설정, MCP 서버 등록",
    "settings_title": "설정",
    "settings_prefs_header": "환경설정",
    "settings_autotag_confidence": "자동 태그 신뢰도:",
    "settings_autotag_confidence_tooltip": (
        "태거의 태그별 임계값 위에 추가로 적용되는 확률 하한(0–1)입니다. "
        "높을수록 더 확실한 태그만 적게 남습니다. 기본값 0.50."
    ),
    "settings_caption_insert_no_artist": "캡션 교정 시 @no-artist 삽입",
    "settings_caption_insert_no_artist_tooltip": (
        "교정 결과에 작가 태그가 없으면 작가 위치에 @no-artist를 넣습니다. "
        "캡션 셔플 경계로만 쓰이며 토큰화 직전에 제거됩니다."
    ),
    "settings_caption_validate_artist_tags": "작가 태그를 DB로 검증",
    "settings_caption_validate_artist_tags_tooltip": (
        "켜면 danbooru_tags_classified.csv에서 작가로 분류된 @태그만 작가 위치로 "
        "옮깁니다. 끄면 @로 시작하는 태그를 작가 태그로 취급합니다."
    ),
    "settings_group_match_frac": "그룹화 엄격도:",
    "settings_group_match_frac_tooltip": (
        "데이터셋 탭에서 그룹화를 누를 때 두 이미지를 묶는 데 필요한 일치 "
        "비율(0–1)입니다. 높을수록 더 엄격하고 깔끔한 그룹이 됩니다. 기본값 0.25."
    ),
    "settings_group_cell_match": "그룹화 셀 일치도:",
    "settings_group_cell_match_tooltip": (
        "데이터셋 그룹화 시 셀 단위 일치로 인정하는 코사인 하한(0–1)입니다. "
        "높을수록 셀 일치 기준이 엄격해집니다. 기본값 0.93."
    ),
    "settings_theme": "테마:",
    "settings_theme_tooltip": (
        "인터페이스 전체 색상 테마입니다. 즉시 적용되며, 설정 창을 닫으면 "
        "창이 다시 그려져 완전히 반영됩니다."
    ),
    "settings_font_size": "글꼴 크기:",
    "settings_font_size_tooltip": (
        "인터페이스 글꼴의 포인트 크기입니다. 즉시 적용되며, 설정 창을 닫으면 "
        "창이 다시 그려져 모든 패널이 다시 배치됩니다. 기본값 10."
    ),
    "settings_theme_dark": "다크",
    "settings_theme_light": "라이트",
    "settings_theme_sepia": "세피아",
    "settings_debug_mode": "디버그 모드",
    "settings_debug_mode_tooltip": (
        "학습 데몬을 DEBUG 레벨로 기록해, 작업이 멈춘 원인을 버그 리포트에 담을 수 "
        "있게 합니다. 데몬이 다음에 시작될 때 적용됩니다(앱을 닫거나 데몬을 종료한 뒤 "
        "다시 여세요)."
    ),
    "settings_debug_report_desc": (
        "무한 로딩에 걸렸나요? 디버그 모드를 켜고 증상을 재현한 다음 '디버그 리포트 "
        "복사'를 눌러 결과를 버그 리포트에 붙여넣어 주세요. 데몬 로그와 최근 작업 "
        "상태가 함께 담깁니다."
    ),
    "settings_debug_copy_report": "디버그 리포트 복사",
    "settings_mcp_header": "MCP 서버 (에이전트 연동)",
    "settings_mcp_desc": "로컬 학습 데몬을 MCP 클라이언트(Claude Code, Claude Desktop 등)에 "
    "노출합니다. 아래 명령을 터미널에서 실행하면 Claude Code에 등록됩니다:",
    "settings_mcp_desc_json": "다른 MCP 클라이언트(Claude Desktop, OpenClaw 등)에는 "
    "동일한 내용의 JSON 설정을 사용하세요:",
    "settings_mcp_copy": "복사",
    "settings_mcp_copied": "복사됨 ✓",
    "settings_close": "닫기",
    "settings_lang_apply_title": "언어",
    "settings_lang_apply_question": "지금 인터페이스를 다시 불러와 언어를 적용할까요?\n\n"
    "탭에서 저장하지 않은 편집 내용은 사라집니다. 대기/실행 중인 학습 작업은 "
    "데몬에서 돌아가므로 영향이 없습니다.\n\n'아니요'를 선택하면 다음 실행 시 적용됩니다.",
    # Guidebook
    "guidebook": "📖 가이드북",
    "guidebook_tooltip": "한국어 종합 가이드 열기 (docs/guidelines/가이드북.md)",
    "guidebook_missing": "가이드를 찾을 수 없습니다: {path}",
    "guidebook_open_external": "시스템 뷰어로 열기",
    "guidebook_close": "닫기",
    # EasyControl 어댑터 가이드 (직접 컨트롤 태스크 만들기)
    "adapter_guide": "📘 어댑터 가이드",
    "adapter_guide_tooltip": "나만의 EasyControl 어댑터 만들기 (easycontrol_adapters/ADAPTER_GUIDE.md)",
    "easycontrol_descriptor_note": "이 컨트롤 태스크는 다중 테이블 구조를 가진 독립형 디스크립터로, 왼쪽에서 원시 TOML로 편집합니다:<br><br>• <b>name</b> — 출력 슬러그; 파생되는 모든 캐시/출력 경로를 재지정합니다.<br>• <code>[staging]</code> — 조건 트리를 구체화하는 데이터 생성 단계.<br>• <code>[preprocess]</code> — 스테이징된 트리에 대한 VAE/TE 캐싱 설정.<br>• <code>[training]</code> — 기본 EasyControl 메소드에 병합될 오버라이드.<br>• <code>[general]</code> / <code>[[datasets]]</code> — train.py가 읽는 데이터셋 청사진.<br>• <code>[variant]</code> — 이 드롭다운 항목의 GUI 메타데이터.<br><br><b>전처리</b> 버튼은 조건 트리를 합성하고 캐싱합니다; <b>학습</b>은 이 디스크립터의 <code>[training]</code> 오버라이드를 병합하여 기본 EasyControl 메소드를 학습합니다. 두 작업 모두 GUI를 닫아도 계속 실행됩니다.",
    "easycontrol_descriptor_form_header": "디스크립터 <b>{path}</b> 편집 중. 아래 설정 테이블은 폼으로 편집합니다; 저장 시 변경된 값을 기록하되 주석과 <code>[[datasets]]</code> 청사진은 유지됩니다. 청사진과 <code>[variant]</code> 메타데이터는 여기에 표시되지 않으므로 해당 항목은 파일을 직접 편집하세요. 필드 이름을 클릭하면 도움말이 표시됩니다.",
    "ec_desc_group_top": "디스크립터",
    # Top-bar buttons (models / update / report issue)
    "models_btn": "모델",
    "models_btn_tooltip": "모델 체크포인트 다운로드 / 재다운로드 (Anima 베이스, SAM3, MIT, PE 비전 인코더)",
    "update_btn": "업데이트",
    "update_btn_tooltip": "GitHub에서 최신 anima_lora 릴리스를 가져오고 uv sync를 실행합니다",
    "update_btn_available": "업데이트 ●",
    "update_btn_available_tooltip": "새 릴리스 {v} 가 있습니다 — 클릭하여 릴리스 노트 보기",
    "report_issue": "이슈 신고",
    "report_issue_tooltip": "브라우저에서 GitHub 이슈 트래커 열기",
    "visit_github": "GitHub 페이지 방문",
    "open_in_system_viewer": "시스템 뷰어로 열기",
    # Models dialog
    "models_title": "모델 다운로드",
    "models_intro": "아래에서 모델 그룹을 선택하거나 '전체 다운로드'로 표준 세트 "
    "(Anima + SAM3 + MIT + PE + 태그 DB)를 받으세요. 파일은 models/ 아래에 저장됩니다.",
    "models_download_all": "전체 다운로드 (Anima + SAM3 + MIT + PE + 태그 DB)",
    "models_download": "다운로드",
    "models_redownload": "재다운로드",
    "models_installed": "✓ 설치됨",
    "models_missing": "✗ 없음",
    "model_anima": "Anima — DiT + 텍스트 인코더 + VAE",
    "model_sam3": "SAM3 — 말풍선 마스킹",
    "model_mit": "MIT — 만화 텍스트 마스킹",
    "model_pe": "PE-Core-L14-336 — 비전 인코더 (CMMD 검증 / DCW)",
    "model_danbooru_tags": "Danbooru 태그 DB — 캡션 순서 교정",
    # HuggingFace 인증 (모델 다이얼로그)
    "models_hf_token_placeholder": "HuggingFace 토큰을 붙여넣으세요 (hf_…)",
    "models_hf_authenticate": "인증",
    "models_hf_token_hint": "게이트/속도 제한 다운로드(예: SAM3)에 필요합니다. "
    '<a href="https://huggingface.co/settings/tokens">'
    "huggingface.co/settings/tokens</a> 에서 토큰을 생성하고 "
    '<a href="https://huggingface.co/facebook/sam3">huggingface.co/facebook/sam3</a> 에서 SAM3 접근을 요청하세요.',
    "models_hf_token_present": "✓ HuggingFace 토큰이 이미 저장되어 있습니다.",
    "models_hf_not_authenticated": "인증되지 않음 — 토큰을 붙여넣어 게이트 다운로드를 활성화하세요.",
    "models_hf_token_empty": "먼저 토큰을 붙여넣으세요.",
    "models_hf_authenticating": "인증 중…",
    "models_hf_logged_in": "✓ {name} (으)로 로그인되었습니다.",
    "models_hf_login_failed": "인증 실패: {err}",
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
    "merge_verdict_hydra": "✗ HydraLoRA moe — 레이어 로컬 라우터는 병합할 수 없습니다",
    "merge_verdict_postfix_only": "✗ postfix/prefix 전용 — 가중치 델타가 아닙니다",
    "merge_verdict_unknown": "? 인식되는 어댑터 키가 없습니다",
    "merge_options": "병합 옵션",
    "merge_base_dit": "베이스 DiT:",
    "merge_multiplier": "강도 배수:",
    "merge_multiplier_tip": "병합 시 적용할 LoRA 강도 (1.0 = 전체 강도)",
    "merge_dtype": "저장 dtype:",
    "merge_out": "출력:",
    "merge_out_placeholder": "(자동: <adapter>_merged.safetensors)",
    "merge_allow_partial": "부분 병합 허용 (Hydra / postfix 키 제외)",
    "merge_allow_partial_tip": "병합 불가능한 컴포넌트가 있어도 진행합니다. 제외된 컴포넌트는 병합된 DiT에 반영되지 않습니다.",
    "merge_button": "DiT에 병합",
    "merge_log_placeholder": "병합 출력이 여기에 표시됩니다...",
    "merge_pick_dir": "어댑터 디렉토리 선택",
    "merge_pick_file": "어댑터 .safetensors 선택",
    "merge_pick_dit": "베이스 DiT .safetensors 선택",
    "merge_pick_out": "병합된 DiT 저장 위치...",
    "browse": "찾아보기…",
    # Multi-scale target_res tiers
    "target_res_bucket_tooltip": "{edge}px tier에서 이 bucket 해상도만 전처리에 사용합니다. 모두 해제하면 선택한 tier의 모든 bucket을 사용합니다.",
    "target_res_danger_tooltip": "고비용 티어: {edge}px는 이미지당 약 {tokens} 토큰을 사용하고 컴파일된 블록 그래프를 하나 더 추가합니다(컴파일 느려짐, VRAM 증가). 이 해상도가 정말 필요할 때만 켜세요.",
    "resize_crop_anchor": "리사이즈 크롭 위치:",
    "resize_crop_anchor_tip": "cover resize 후 target bucket으로 crop할 때 어느 방향을 보존할지 선택합니다.",
    "resize_crop_anchor_top_left": "좌상단",
    "resize_crop_anchor_top": "상단",
    "resize_crop_anchor_top_right": "우상단",
    "resize_crop_anchor_left": "좌측",
    "resize_crop_anchor_center": "중앙",
    "resize_crop_anchor_right": "우측",
    "resize_crop_anchor_bottom_left": "좌하단",
    "resize_crop_anchor_bottom": "하단",
    "resize_crop_anchor_bottom_right": "우하단",
    "resize_crop_margins": "리사이즈 여백:",
    "resize_crop_margin_top": "상",
    "resize_crop_margin_right": "우",
    "resize_crop_margin_bottom": "하",
    "resize_crop_margin_left": "좌",
    # TensorBoard panel
    "tb_panel_title": "TensorBoard 실행 목록",
    "tb_open": "TensorBoard 열기",
    "tb_stop": "서버 중지",
    "tb_remove": "삭제",
    "tb_view": "조회",
    "tb_view_tip": "이 실행만 TensorBoard로 엽니다.",
    "tb_no_runs": "아직 실행 기록이 없습니다. 학습을 시작하면 목록이 채워집니다.",
    "tb_status_running": "포트 {port}에서 실행 중",
    "tb_status_stopped": "",
    "tb_not_installed": "tensorboard가 설치되지 않았습니다. 실행: pip install tensorboard",
    "tb_current_run_label": " (현재)",
    "tb_open_current": "현재 학습 조회",
    "tb_open_current_tip": "진행 중인 학습 실행만 TensorBoard로 엽니다.",
    "tb_open_current_idle_tip": "학습이 진행 중일 때 사용할 수 있습니다.",
    "tb_appear_hint": "실행이 목록에 보이지 않으면 TensorBoard의 새로고침(업데이트) 버튼을 눌러보세요.",
}
