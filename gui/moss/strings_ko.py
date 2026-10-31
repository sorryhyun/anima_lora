"""Korean strings for the MOSS-Image GUI; assembled in ``gui.moss.strings``."""

from __future__ import annotations

UI: dict[str, str] = {
    "window_title": "MOSS-Image LoRA",
    "model_missing": "모델을 찾을 수 없습니다: {path}\n`make download-model ARGS=moss-base`"
    "를 실행하거나 (SFT DiT는 ARGS=moss-sft), 고급 → model_dir을 지정하거나, "
    ".env에 ANIMA_MOSS_MODEL_DIR을 설정하세요.",
    "scan_stale": "캡션 {stale}개가 텍스트 캐시 이후 수정되었습니다 — 다음 전처리 때 "
    "다시 인코딩됩니다.",
    "scan_dupes": "이미지 {n}장이 다른 하위 폴더의 이미지와 파일 이름이 같습니다 — "
    "해당 캐시 파일 이름에는 폴더 이름이 접두사로 붙습니다.",
    "tab_i2i_preprocess": "i2i 전처리",
    "tab_i2i_train": "i2i 학습",
    "i2i_task": "태스크",
    "i2i_no_tasks": "configs/moss/i2i/ 아래에 i2i 태스크 설정이 없습니다.",
    "i2i_guide": "i2i LoRA는 참조 이미지를 타깃 이미지로 바꾸는 법을 배웁니다: 참조 "
    "이미지는 깨끗한 잠재변수 토큰으로 매 forward에 함께 들어갑니다. 스테이징은 쌍을 "
    "만들고, 전처리는 각 타깃을 참조 이미지(와 캡션)와 함께 캐시하고, 학습은 LoRA를 "
    "학습하며, 테스트는 그 LoRA로 새 참조 이미지를 변환합니다. 각 폼은 "
    "configs/moss/i2i/<task>/에서 시작하며, 바꾼 값만 저장됩니다.",
    "i2i_task_colorize": "colorize: 흑백 만화 페이지 → 컬러. 스테이징은 컬러 코퍼스를 "
    "만화풍으로 변환합니다. 페이지는 원본 크기의 참조 이미지가 되고, 캡션에는 "
    "OCR로 읽은 대사가 들어갑니다(비전 타워는 사용하지 않음). "
    '테스트 프롬프트는 "comic"을 쓰세요.',
    "i2i_task_sanitize": "sanitize: 글자와 말풍선을 지웁니다. 스테이징은 같은 작가의 "
    "거의 같은 이미지 쌍(글자 있음 / 없음)을 찾습니다. 글자가 있는 페이지가 원본 크기의 "
    "참조 이미지가 됩니다. 테스트 프롬프트는 페이지의 캡션에서 글자 관련 태그를 뺀 "
    "것입니다.",
    "i2i_stage_group": "1 · 스테이징",
    "i2i_stage_hint": "데이터셋에서 post_image_dataset/i2i/ 아래에 태스크의 쌍 "
    "트리를 만듭니다 (make i2i-stage TASK={task}; 설정은 "
    "i2i_tasks/configs/{task}.toml의 [staging] 테이블). EasyControl도 같은 트리를 "
    "읽습니다.",
    "run_stage": "스테이징",
    "i2i_cache_group": "2 · 전처리",
    "i2i_train_group": "3 · 학습",
    "i2i_chain": "스테이징, 전처리 후 학습",
    "i2i_test_group": "4 · 테스트",
    "i2i_test_hint": "각 참조 이미지를 LoRA로 변환해 나란히 보여 줍니다. ref_image는 "
    "이미지 한 장 또는 이미지 폴더이며, ref_scale과 ref_vision은 LoRA를 학습한 캐시와 "
    "같아야 합니다.",
    "i2i_staged": "스테이징된 쌍 {n}개.",
    "i2i_not_staged": "아직 스테이징되지 않았습니다: {path}가 없습니다. 먼저 "
    "스테이징을 실행하세요.",
    "i2i_stage_next": "스테이징 완료 — 전처리를 시작합니다.",
    "i2i_need_ref": "ref_image를 이미지 한 장 또는 이미지 폴더로 지정하세요.",
    "i2i_need_prompt": "ref_vision이 꺼져 있으면 테스트에 프롬프트가 필요합니다: "
    "페이지의 캡션.",
    "test_ref": "참조 이미지",
    "model_help_test": "생성에 쓸 DiT(base / sft)와 돌리는 방식. bf16(기본값): 1024²에서 "
    "10.7 GiB. int8: 블록 가중치를 int8로, 나머지는 fp32 stream으로 돌립니다. 1024²에서 "
    "7.8 GiB이고, Base 파일은 처음 쓸 때 받으며 sft는 반올림 양자화를 씁니다. loha와 블록 "
    "스왑은 쓸 수 없습니다. 둘 사이를 바꾸면 int8, stream, blocks_to_swap이 함께 바뀌며, "
    "이 값들은 고급에서 계속 조정할 수 있습니다.",
    "model_help_i2i": "학습할 DiT(base / sft)와 돌리는 방식. bf16(여기서 기본값): 이 "
    "태스크에서 측정된 설정. int8: swap 20, budget 0.1. sanitize의 가장 큰 샘플이 6스텝 "
    "동안 피크 12.3 GB로 들어갔지만, 아직 이 설정으로 i2i LoRA를 끝까지 학습한 적은 "
    "없습니다. loha는 쓸 수 없습니다. 둘 사이를 바꾸면 int8, stream, ema_card, "
    "blocks_to_swap, swap_h2d_only, activation_memory_budget이 함께 바뀌며, 이 값들은 "
    "고급에서 계속 조정할 수 있습니다.",
    "model_help": "학습할 DiT(base / sft)와 돌리는 방식. int8(기본값): 블록 가중치를 "
    "int8로 돌립니다. 약 1.1 s/step, 피크 13 GB에 시스템 메모리 4.4 GiB가 고정되고, Base "
    "파일은 처음 쓸 때 받으며 sft는 반올림 양자화를 씁니다. loha는 쓸 수 없습니다. bf16: "
    "블록 36개 스왑으로 같은 피크에서 약 1.6 s/step이며 loha를 쓸 수 있습니다. 둘 사이를 "
    "바꾸면 int8, stream, ema_card, blocks_to_swap, swap_h2d_only, "
    "activation_memory_budget이 함께 바뀌며, 이 값들은 고급에서 계속 조정할 수 있습니다.",
}

FIELDS: dict[str, tuple[str, str]] = {
    "src": (
        "이미지 폴더",
        "이미지와 같은 이름의 .txt 캡션이 있는 폴더, 하위 폴더 포함"
        "(기본값: Anima의 resized 트리, 그 안의 캡션은 수정본입니다. "
        "그 트리에 캡션만 있고 이미지가 없으면 image_dataset/의 원본을 씁니다)",
    ),
    "out": (
        "캐시 폴더",
        "이미지마다 캐시 파일 두 개(텍스트 임베딩 + 잠재변수)를 씁니다",
    ),
    "resolution": (
        "해상도",
        "픽셀 면적 상한을 정사각형 변으로 — 1024는 원본 비율 그대로 최대 약 1024² 픽셀, "
        "양 변은 16의 배수. 확대는 하지 않습니다",
    ),
    "caption_format": (
        "캡션 형식",
        "텍스트 인코더가 인코딩하는 내용: layout_json = Anima 캡션을 MOSS의 layout JSON "
        "프롬프트로 변환(danbooru 태그 CSV 필요), text = .txt 그대로, fixed = 모든 이미지에 "
        "fixed_caption을 인코딩. 생성할 때도 같은 형식을 쓰세요",
    ),
    "overwrite": ("덮어쓰기", "캐시가 최신이어도 다시 인코딩"),
    "skip_text": ("텍스트 건너뛰기", "텍스트 인코더 단계를 건너뜀"),
    "skip_latents": ("잠재변수 건너뛰기", "VAE 인코딩을 건너뜀"),
    "model_dir": (
        "모델 폴더",
        "MOSS-Image-Base/와 MOSS-Image-SFT/가 있는 폴더"
        "(기본값: $ANIMA_MOSS_MODEL_DIR 또는 models/moss_image)",
    ),
    "cache": ("캐시 폴더", "전처리가 쓴 캐시 폴더"),
    "output": ("출력", "LoRA 출력 경로(같은 폴더에 .json 실행 리포트가 생깁니다)"),
    "model": ("모델", "학습할 DiT(base / sft)"),
    "epochs": ("에폭", "캐시를 반복하는 횟수"),
    "max_steps": ("최대 스텝", "이 스텝 수에서 멈춤(0 = 에폭으로 결정)"),
    "algo": (
        "어댑터",
        "lora, 또는 LyCORIS의 lokr(크로네커 곱) / loha(랭크 r 행렬 두 쌍의 아다마르 곱)",
    ),
    "rank": ("랭크", "LoRA 랭크(lokr: 큰 크로네커 인수의 랭크)"),
    "alpha": (
        "alpha",
        "LoRA alpha(lora 기본값 128, lokr / loha 기본값은 랭크와 같음); 스케일 = alpha / 랭크",
    ),
    "factor": (
        "lokr 인수",
        "lokr 전용: 작은 크로네커 인수의 변 길이 상한(-1 = 정사각형에 가까운 분할)",
    ),
    "down_init": (
        "down 초기화",
        "lora 전용: kaiming = 기본 무작위 초기화; weight_svd = 베이스 가중치의 상위 r개 "
        "오른쪽 특이벡터 / √3으로 초기화(Anima의 SVD-Down); weight_svd_nodc = 같은 SVD "
        "기저에서 보정된 입력 DC 방향(channel_mean)을 뺀 것, 초기 down · x̄ = 0",
    ),
    "lr": ("학습률", "학습률 (Muon 기본, lokr / loha는 AdamW)"),
    "warmup_ratio": ("워밍업 비율", "선형 워밍업이 전체 스텝에서 차지하는 비율"),
    "grad_accum": (
        "그래디언트 누적",
        "옵티마이저 업데이트 한 번에 누적하는 샘플 수. 스텝, 에폭, 손실 로그는 여전히 "
        "샘플 단위, 워밍업은 업데이트 단위로 셉니다",
    ),
    "save_every_epochs": (
        "N 에폭마다 저장",
        "N 에폭마다 추가로 저장(0 = 최종 결과만)",
    ),
    "caption_dropout": (
        "캡션 드롭아웃",
        "빈 프롬프트 임베딩(CFG의 무조건 분기)으로 한 스텝을 학습할 확률",
    ),
    "plain_caption_rate": (
        "Anima 캡션 비율",
        "layout JSON 대신 원래 Anima 형식 캡션으로 한 스텝을 학습할 확률. "
        "caption_format=layout_json 캐시에만 원래 캡션이 들어 있습니다",
    ),
    "lora_dtype": (
        "LoRA 정밀도",
        "저장되는 LoRA 파일의 정밀도. 학습 중에는 fp32 마스터 가중치를 유지하고 "
        "랭크 행렬곱은 bf16으로 수행합니다",
    ),
    "grad_checkpointing": ("그래디언트 체크포인팅", "블록 단위 활성값 체크포인팅"),
    "seed": ("시드", "난수 시드"),
    "n": (
        "일러스트 쌍 수",
        "학습에 쓸 일러스트 쌍 수(스테이징된 만화풍 페이지 + resized의 컬러 타깃)",
    ),
    "manga": ("만화 페이지 수", "추가할 manga_color 만화 페이지 수(0 = 일러스트만)"),
    "holdout": ("홀드아웃 일러스트 수", "평가용으로 남겨 두는 일러스트 쌍 수"),
    "manga_holdout": (
        "홀드아웃 만화 작품 수",
        "평가용으로 남겨 두는 만화 작품 수, 작품당 한 페이지. 그 작품의 어떤 판본도 "
        "학습에 쓰이지 않습니다",
    ),
    "ref_scale": (
        "참조 이미지 축소",
        "참조 잠재변수의 변 = 타깃 변 ÷ 이 값, 16의 배수(생성할 때도 같은 값 사용)",
    ),
    "variants": (
        "캡션 변형 수",
        "희소 캡션 수 + 1: embeds_v1 … embeds_v{variants-1}은 OCR 레이아웃이 붙은 컬러 "
        "캡션이며, 홀수 k는 태그의 tag_dropout 비율을, 짝수 k는 전부를 뺍니다(만화 "
        "페이지는 만화 태그 하나를 남김). 최소 3, 홀수면 두 종류가 반반",
    ),
    "tag_dropout": (
        "태그 드롭아웃 비율",
        "홀수 k 희소 캡션이 컬러 캡션에서 빼는 태그 비율",
    ),
    "vision": (
        "비전 타워로 페이지 인코딩",
        "페이지가 비전 타워를 거쳐 모든 캡션 앞에 텍스트 인코더로 들어갑니다. "
        "끔 = 캡션만 인코딩, embeds_plain을 쓰지 않음",
    ),
}

FIELDS_GENERATE: dict[str, tuple[str, str]] = {
    "prompt": ("프롬프트", "프롬프트(prompts_file 또는 example을 지정하면 무시)"),
    "negative_prompt": (
        "네거티브 프롬프트",
        "네거티브 프롬프트(파이프라인 기본값은 빈 값)",
    ),
    "prompt_format": (
        "프롬프트 형식",
        "layout_json = Anima 형식 프롬프트를 MOSS의 layout JSON으로 변환(캐시의 캡션 "
        "형식과 동일); text = 입력 그대로, 이미 JSON인 프롬프트나 text 형식 캐시로 학습한 "
        "LoRA용",
    ),
    "lora": (
        "LoRA",
        "테스트할 LoRA(비우면 학습 출력). Base에서 학습한 LoRA는 SFT에도 쓸 수 있습니다",
    ),
    "lora_multiplier": ("LoRA 강도", "LoRA 강도. 0.0이면 베이스 모델 그대로"),
    "width": ("너비", "16의 배수"),
    "height": ("높이", "16의 배수"),
    "steps": ("스텝", "디노이징 스텝 수"),
    "cfg": ("CFG", "CFG 스케일. 1.0 = 무조건 forward 없음"),
    "sampler": (
        "샘플러",
        "er_sde = 같은 σ 그리드에서 ER-SDE 확률적 샘플링(선이 더 선명, 추가 forward 없음); "
        "euler = 파이프라인 기본 ODE",
    ),
    "uncond_skip": (
        "무조건 분기 건너뛸 블록",
        "쉼표로 구분한 DiT 블록 번호('' = 일반 CFG). 파이프라인 기본값은 22번 블록을 "
        "건너뜁니다",
    ),
    "seed": ("시드", "첫 프롬프트의 시드(프롬프트마다 +1)"),
    "out_dir": ("출력 폴더", "이미지와 manifest.json이 저장되는 곳"),
    "spectrum": (
        "Spectrum",
        "대부분의 스텝에서 42개 블록을 모두 돌리는 대신 마지막 블록의 출력을 예측",
    ),
    "dave": (
        "DAVE",
        "같은 프롬프트 다양성: 처음 몇 스텝 동안 블록 묶음에서 이미지 토큰의 채널별 평균을 "
        "약하게 합니다. '' = 끔, 'auto' = 내장 MOSS 마스크, 'lo-hi' = lo..hi번 블록, "
        "또는 dave_alpha.npz 경로",
    ),
    "dave_strength": (
        "DAVE 강도",
        "묶음 안의 블록에서 제거하는 평균의 비율(0 = 끔, 1 = 전부)",
    ),
    "cns": (
        "CNS",
        "er_sde 전용: 샘플러가 넣는 노이즈를 CNS로 재착색합니다. 'auto' = 변형·cfg에 맞춰 "
        "보정된 내장 γ, 또는 npz 경로, '' = 끔. ComfyUI의 er_sde_cns와 같습니다",
    ),
    "ref_image": (
        "참조 이미지",
        "i2i 참조: 이미지 한 장 또는 이미지 폴더(이미지 × 프롬프트마다 작업 하나). "
        "출력은 참조 이미지 비율을 유지하고 면적 상한은 너비 × 높이. LoRA에 기록된 "
        "ref_timestep / ref_isolate가 적용됩니다",
    ),
    "ref_scale": (
        "참조 이미지 축소",
        "ref_image 지정 시: 참조 잠재변수의 변 = 출력 변 ÷ 이 값, 16의 배수. LoRA를 "
        "학습한 캐시와 같아야 합니다(sanitize 1, colorize 2)",
    ),
    "ref_vision": (
        "참조 이미지를 비전 타워로",
        "ref_image 지정 시: 참조 이미지가 비전 타워를 거쳐 프롬프트 앞에 텍스트 인코더로도 "
        "들어갑니다(colorize 캐시가 페이지를 인코딩하는 방식과 동일). 이때 프롬프트를 "
        "비우면 페이지만 입력합니다",
    ),
}
