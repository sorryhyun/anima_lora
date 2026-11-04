# 나만의 EasyControl 어댑터 만들기

이 가이드는 Anima에 **새로운 EasyControl 컨트롤 태스크**를 개인 용도로 추가하는 방법을 안내합니다. git에 기여하는 것이 아니라 `i2i_tasks/<your_task>/` 아래에 두는 로컬 어댑터입니다. 대표 예제인 **colorize** (`i2i_tasks/colorization/`)를 단계별로 따라가면서, 자신의 태스크를 위해 정확히 무엇을 바꿔야 하는지 짚어 줍니다.

기억할 것은 단 하나:

> **모델 코드를 작성하는 것이 아닙니다.** 네트워크, 포워드 패스, `b_cond` 게이트, 인퍼런스 캐시 — 이 모든 것은 이미 들어 있고 모든 컨트롤 태스크가 공유합니다. 태스크마다 다른 것은 **레퍼런스 이미지를 어떻게 만드느냐**뿐입니다. 이 가이드의 나머지 내용은 모두 그것을 둘러싼 배관입니다.

---

## 0. 아이디어 — EasyControl 어댑터란 무엇인가

EasyControl은 **레퍼런스 이미지**를 사용해 생성을 안내합니다. 레퍼런스는 VAE를 거쳐 *cond 토큰*으로 변환되고, 생성 중인 이미지와 나란히 흘러갑니다. 각 스텝마다 모델은 두 가지를 모두 봅니다. (전체 아키텍처: `docs/anima/experimental/easycontrol.md`.)

기본 EasyControl은 **동일한 이미지**를 레퍼런스이자 타깃으로 사용합니다 — 그래서 그냥 복사하는 것을 학습합니다. 컨트롤 태스크는 그것을 바꿉니다: 각 타깃을 **다른** 레퍼런스와 짝지어, 모델이 복사 대신 `reference → target`을 학습하도록 합니다.

| | 기본 EasyControl | 컨트롤 태스크 (예: colorize) |
|---|---|---|
| 타깃 (원하는 것) | 이미지 X | 컬러 이미지 X |
| 레퍼런스 (힌트) | 이미지 X (동일) | X의 *변환* 버전 (X의 흑백 망가) |
| 학습 내용 | 복사 | manga → color |
| 텍스트 | 전체 캡션 | (선택) 짧은 캡션 |

colorize의 방식 — 재사용할 수 있는 아이디어입니다: **실제 흑백 망가에는 컬러 버전이 없으므로** `(흑백, 컬러)` 쌍을 수집해 만들 수 없습니다. 그래서 방향을 뒤집습니다 — 이미 보유한 컬러 이미지를 타깃으로 삼고(그것이 원하는 결과이므로), 각 이미지로부터 흑백 레퍼런스를 **직접 만듭니다**(선화 + 스크린톤, 알고리즘으로). 핵심은 만든 흑백이 인퍼런스 시 실제로 넣을 흑백처럼 보여야 한다는 것입니다. `depth → image`나 `pose → image`를 만든다면, "레퍼런스 만들기" 단계는 학습 이미지에 대해 실행하는 depth estimator나 pose detector가 될 것입니다.

즉, 할 일은: **함수 `target_image → reference_image`를 작성하고, 그 출력을 캐시하고, 데이터셋이 그것을 가리키게 하고, 디스크립터에 데이터셋을 기술하고, 태스크 이름을 등록하는 것**입니다.

---

## 1. 건드리는 네 가지

`<task>`라는 이름의 어댑터를 추가하려면 정확히 다음 네 가지를 만들거나 편집합니다:

| # | 항목 | colorize 버전 | 역할 |
|---|---------|-------------------|--------------|
| 1 | `i2i_tasks/<task>/` 프로젝트 | `colorization/` (`mangafy*.py`, `prep.py`) | 레퍼런스 이미지를 만들고 캐시 (선택적으로 짧은 텍스트 캐시도) |
| 2 | `i2i_tasks/configs/<task>.toml` + `configs/easycontrol/<task>.toml` | `i2i_tasks/configs/colorize.toml` + `configs/easycontrol/colorize.toml` | 태스크 파일(`name` 슬러그, 선택적 `stage`, `[staging]`)과 EasyControl 디스크립터(`[preprocess]` / `[training]` 테이블, `[[datasets]]` 블루프린트, 선택적 `[variant]`) |
| 3 | `scripts/tasks/training.py`의 `_I2I_STAGES` | `"colorize": {"stage": _colorize_stage, "preprocess": _colorize_preprocess}` | `make i2i-stage`에서 `TASK=<task>`가, `easycontrol-preprocess` / `easycontrol`에서 `EASYADAPTER=<task>`가 작동하게 함 |
| 4 | `cmd_test_easycontrol`의 `_ADAPTERS` (`scripts/tasks/inference.py`) | `"colorize"` 행 | `make test-easycontrol`용 체크포인트 접두사, 출력 폴더, 레퍼런스 폴백 폴더, 빈 프롬프트 기본값 |

순서대로 살펴봅니다. 어느 것도 `networks/`를 건드리지 않습니다.

---

## 2. 항목 1 — 어댑터 프로젝트 (`i2i_tasks/<task>/`)

실제 작업이 여기에 있습니다. 두 가지 일을 합니다: **레퍼런스 이미지 만들기**와 **캐시하기**. (선택적으로 세 번째로, 태스크별 텍스트 캐시를 구축합니다.)

### 2a. 레퍼런스를 만드는 함수

일반 함수입니다: 컬러 이미지를 RGB `uint8 (H,W,3)`과 seed로 받아, **같은 크기**의 레퍼런스 이미지를 RGB `uint8 (H,W,3)`으로 반환합니다. (같은 크기가 중요합니다 — §3 참조.) 동일한 seed에 대해 동일한 출력을 내야 하므로, 재실행과 병렬 워커가 모두 일치합니다.

colorize에서 이는 `mangafy.py::mangafy_array`(및 GPU 버전 `mangafy_gpu.py::mangafy_array_gpu`)입니다:

```python
# i2i_tasks/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

복사할 가치가 있는 네 가지:

- **이름으로부터 결정론적으로 seed를 만드세요.** colorize는 `zlib.crc32(stem)`을 사용합니다 — 프로세스마다 달라서 병렬 워커가 불일치하게 만드는 Python의 `hash()`가 *아닙니다*. 다양성을 추가하는 것은 괜찮습니다(colorize는 스크린톤 각도를 페이지마다 조금씩 다르게 합니다) — seed에서 파생하기만 하면 재현 가능합니다.
- **무거운 것은 지연 import하세요.** colorize에는 세 개의 엔진(`cv2` / `gpu` / `sd`)이 있고, 페이지가 실제로 그쪽으로 라우팅될 때만 3.5 GB짜리 SD 모델을 로드합니다. 빌더가 모델(depth net, line extractor)을 필요로 한다면, 사용할 때만 import하세요.
- **다운로드 없는 폴백은 큰 장점입니다.** colorize의 `cv2`/`gpu` 엔진은 다운로드가 전혀 필요 없어서, 새 체크아웃에서 추가 다운로드 단계 없이 prep + 학습이 가능합니다. 가능하다면 이런 폴백을 만들어 두세요.
- **파일을 원자적으로 쓰세요.** colorize의 `_save_png_atomic`은 임시 파일에 쓴 뒤 이름을 바꿉니다. 이것 없이는 중단된 실행이 반쯤 쓰인 PNG를 남길 수 있고, "이미 있으면 건너뛰기" 검사가 그 파일을 영원히 신뢰하게 됩니다. 실제로 발생하는 버그입니다 — 이 패턴을 복사하세요.

레퍼런스가 **이미 디스크에 있다면**(실제 depth map, 실제 스케치 등), 빌드 단계를 완전히 건너뛰고 그것을 직접 캐시하면 됩니다. 타깃으로부터 레퍼런스를 파생해야 할 때만 빌드가 필요합니다.

### 2b. (선택) 짧은 텍스트 캐시

colorize는 레퍼런스만 바꾸는 게 아니라 **캡션을 색상 단어로만 축소**하기도 합니다(`anime_tools.captions.color_tags.filter_to_colors`). 이유를 이해할 가치가 있습니다: 레퍼런스(선화 + 스크린톤)가 이미 *형태와 레이아웃에 관한 모든 것*을 인코딩하므로, 텍스트가 여전히 말해야 할 것은 흑백이 표현할 수 없는 단 하나 — **색**뿐입니다. 캡션을 색상 단어로 축소하면 남은 단어가 모두 모델이 레퍼런스에서 얻을 수 없는 정보가 되어, 긴 캡션 속에 묻힌 색상 단어로 약하게 스티어링하는 대신 강한 `prompt → color` 연결이 생깁니다.

자신의 태스크에 같은 질문을 해보세요: **레퍼런스가 이미 결정하는 것은 무엇이고, 텍스트가 결정해야 할 것은 무엇인가?** `pose → image` 레퍼런스는 포즈는 고정하지만 의상이나 배경은 고정하지 않습니다 — 아마 *전체* 캡션을 유지할 것입니다. `depth → image` 레퍼런스는 레이아웃은 고정하지만 정체성이나 색상은 고정하지 않습니다. colorize의 "캡션을 여전히 모호한 것으로 축소한다"는 방식은 고려할 *패턴*이지 규칙이 아닙니다 — 많은 어댑터가 캡션을 그대로 두고 텍스트 캐시를 완전히 건너뜁니다(데이터셋에서 `text_cache_dir`를 생략하기만 하면 됩니다 — §4 참조).

캡션을 축소한다면, colorize의 두 개의 독립적인 노브에 유의하세요(혼동하지 마세요):

- **`caption_dropout_rate`** — 자동 채색의 *하한(floor)*. 약 5%의 학습 스텝에서 캡션을 통째로 드롭하여, 프롬프트를 주지 않았을 때의 동작을 학습시킵니다. **낮게** 유지하세요(`0.05`); 높은 값은 무조건부 경로를 과하게 학습시켜 프롬프트를 약하게 만듭니다.
- **`use_shuffled_caption_variants`** — 전체 대 부분의 *균형*. 텍스트 캐시는 여러 버전을 가집니다(v0 = 전체 색상 집합, v1+ = 각 단어가 약 절반의 확률로 드롭된 셔플 버전). 로더는 v0를 20%, v1+를 80%로 추출하므로, "pink hair" 같은 부분 프롬프트만으로도 잘 동작합니다.

### 2c. `prep.py` — 캐시 빌더

세 단계, 각각 **멱등(idempotent)**(이미 완료된 작업은 건너뛰므로 재실행해도 안전합니다):

1. **빌드** — `--src`(`post_image_dataset/resized`) 아래 모든 컬러 이미지를 순회하며 빌더 함수를 실행하고, 소스 레이아웃을 미러링하는 `--staging` 폴더에 레퍼런스 PNG를 씁니다.
2. **인코드** — 스테이징된 레퍼런스를 `library.preprocess.cache_latents`로 `--cond_cache_dir`에 VAE 인코딩합니다. 각 이미지의 **원본 크기**에서 인코딩하므로 레퍼런스 레이턴트가 타깃 레이턴트와 같은 형태가 됩니다. 일반 캐시와 동일한 `{stem}_{WxH}_anima.npz` 포맷입니다.
3. **(선택) 텍스트** — 캡션을 필터를 통해 `--text_cache_dir`로 재인코딩합니다. `library.preprocess.cache_text_embeddings`에 `caption_transform=`(및 `caption_shuffle_variants` / `caption_tag_dropout_rate`)를 주어 수행합니다.

기존 라이브러리 헬퍼를 사용하세요 — `library.preprocess.{cache_latents, cache_text_embeddings, tqdm_progress}` 및 `library.preprocess._dataset.walk_images`. 직접 인코드 루프를 짜지 마세요; `prep.py`는 `scripts/preprocess/*.py`처럼 이들 위에 얹힌 얇은 셸입니다.

colorize가 처리하는, 그 구조를 복사하면 공짜로 물려받는 두 가지 정확성 함정:

- **Stem이 일치해야 합니다.** 텍스트 단계는 캡션 마스터(`image_dataset/`, `resized/`와 같은 레이아웃)에서 `.txt` 캡션을 읽으므로, 결과 텍스트 캐시 파일명이 로더가 찾는 것(`image_dir=post_image_dataset/resized` 기준)과 일치합니다. 캐시 파일명이 타깃 stem과 맞지 않으면 로더가 아무 말 없이 짝짓기를 못 합니다.
- **uncond 사이드카.** colorize의 텍스트 단계는 공유 `T5("")` 빈 프롬프트 사이드카가 없으면 다시 만듭니다. 텍스트 캐시를 만들고 캡션 드롭아웃을 사용한다면 똑같이 하세요(`library.preprocess.uncond.stage_uncond_sidecar_with_models`).

---

## 3. 지켜야 할 규칙 — 레퍼런스와 타깃의 토큰 카운트가 일치해야 한다

DiT는 Anima의 **원본 형태 버킷팅**으로 동작합니다(두 개의 토큰 카운트 패밀리, 4032와 4200; CLAUDE.md 및 `docs/anima/experimental/easycontrol.md` "Cond token count" 참조). **패딩 노브가 없습니다** — 레퍼런스는 레이턴트의 실제 토큰 카운트로 실행됩니다.

그래서 §2a에서 레퍼런스가 입력과 **같은 크기**여야 한다고 하고, §2c에서 **원본 크기**로 인코딩한다고 하는 것입니다: 두 가지를 모두 지키면 레퍼런스 레이턴트가 자동으로 타깃과 같은 버킷에 들어가고, 모든 것이 그냥 작동합니다. 더 작은 레퍼런스를 원한다면(완전히 괜찮습니다 — 더 작으면 메모리와 속도 모두 이득), **이미지** 수준에서, 인코딩 전에 줄이세요. 그러면 레이턴트가 여전히 실제 버킷에 들어갑니다. 네트워크 안에서 토큰 카운트를 제한하려 하지 마세요.

---

## 4. 항목 2 — 태스크 파일과 디스크립터

두 파일이 태스크별 설정을 모두 담습니다 (`i2i_tasks/configs/colorize.toml`과
`configs/easycontrol/colorize.toml` 참조). 모델과 무관한 태스크 파일
`i2i_tasks/configs/<task>.toml`에는 `name`, `stage`, `[staging]`이 들어가고,
나머지는 EasyControl 디스크립터 `configs/easycontrol/<task>.toml`에 들어갑니다:

- 태스크 파일 최상위 **`name`** — 슬러그. 트리는
  `post_image_dataset/i2i/<name>/` 아래에 만들어지며, `output_name`의 기본값은
  `anima_easycontrol_<name>`입니다.
- 태스크 파일 **`stage`** — `_I2I_STAGES`의 키 (기본값: 태스크 id).
- **`[staging]`** (태스크 파일) / **`[preprocess]`** (디스크립터) — stage/preprocess
  함수가 `prep.py` 플래그로 바꿔 주는 평평한 테이블 (`_toml_table_to_argv`: `--key value`,
  리스트는 펼침, `true` → 값 없는 `--flag`, `false` → 생략).
- **`[training]`** — 기본 `easycontrol` 메서드 위에 `train.py` CLI 오버라이드로
  병합됩니다.
- **`[general]` / `[[datasets]]`** — 데이터셋 블루프린트. `train.py`의 데이터셋 검증기가 다른
  최상위 키를 거부하므로, `make easycontrol`은 이를 (`{name}`을 치환해) 슬러그 디렉터리 아래의
  생성된 `dataset_config.toml`로 기록합니다.
- **`[variant]`** (선택) — `family = "easycontrol"`, `label`, `description`,
  `order`; 이 블록이 있는 디스크립터는 GUI의 EasyControl 탭에 나타납니다.

colorize 블루프린트의 서브셋:

```toml
  [[datasets.subsets]]
  image_dir = 'post_image_dataset/resized'                         # the color targets
  cache_dir = 'post_image_dataset/lora'                            # shared TE/PE cache, reused
  cond_cache_dir = 'post_image_dataset/i2i/{name}/cond'    # reference latents (prep.py)
  text_cache_dir = 'post_image_dataset/i2i/{name}/text'    # color-only text cache (prep.py)
  latent_cache_dir = 'post_image_dataset/i2i/{name}/target' # WB target latents (prep.py)
  recursive = true
  flip_aug = false
  num_repeats = 1
```

- **`cond_cache_dir`** — 이것이 컨트롤 태스크로 만들어 줍니다. 로더는 각 타깃을
  stem으로 레퍼런스 레이턴트와 매칭하고, 레퍼런스가 있는 타깃만 남깁니다.
- **`text_cache_dir`** — 텍스트 캐시만 리다이렉트합니다. 생략하면 공유 텍스트 캐시를
  사용하고 텍스트 단계를 건너뜁니다.
- **`latent_cache_dir`** — 타깃 레이턴트만 리다이렉트합니다. 생략하면
  `cache_dir`을 사용합니다. 여기에 없는 레이턴트는 보정되지 않은 원본 이미지에서 (경고와 함께)
  다시 인코딩되므로, 학습 전에 채워 두세요.
- **`flip_aug = false`** — 필수입니다. cond 캐시에는 뒤집힌 레이턴트가 없고,
  플립이 켜져 있으면 로더가 에러를 냅니다.

고민할 `[training]` 노브들:

- **`network_args`** — `b_cond_init`은 step 0에서 레퍼런스가 얼마나 기여하는지 정합니다
  (`-10` ≈ step 0에서 기본 DiT와 동일; `docs/anima/experimental/easycontrol.md`의
  "Step-0 baseline equivalence" 참조; colorize는 레퍼런스가 더 빨리 작동하도록 완화합니다;
  어느 쪽이든 학습 가능), `cond_scale`, `apply_ffn_lora` (`0`이면 FFN LoRA를 제거,
  학습 파라미터의 약 절반), `cond_res_scale`.
- **`easycontrol_cond_noise_max`** — 학습 중 레퍼런스에 더하는 노이즈
  (σ ~ `U(0, max)`, `cond + σ·ε`). `0`은 레퍼런스를 정확한 청사진으로 취급하고,
  높은 값은 대략적인 힌트로 만들어 디테일을 텍스트로 넘깁니다.
- **`easycontrol_drop_p`** — image-CFG를 위해 레퍼런스 전체를 드롭하는 빈도.
  항상 레퍼런스가 있는 태스크는 `0`으로 설정합니다.
- **`output_name`** — 인퍼런스 selector가 이 접두사로 최신 체크포인트를 찾습니다 (§5).
- **`blocks_to_swap = 0`** — 배포된 디스크립터들이 사용하는 값입니다.

---

## 5. 항목 3, 4 — 태스크 등록

**`scripts/tasks/training.py`:** 시그니처가 `(adapter, cfg, base, extra)`인 함수 두 개를 작성하고
`_I2I_STAGES`(stage 종류를 키로 사용)에 추가합니다:

```python
_I2I_STAGES = {
    ...
    "colorize": {"stage": _colorize_stage, "preprocess": _colorize_preprocess},
    "<task>": {"stage": _task_stage, "preprocess": _task_preprocess},
}
```

`_colorize_stage` / `_colorize_preprocess`가 모범 예시입니다: 각각 stage 건너뛰기 플래그,
슬러그에서 파생된 경로(`_colorize_prep_paths(base)`), 테이블 노브를 붙여
`i2i_tasks/colorization/prep.py`를 실행하며, `extra`를 맨 뒤에 붙여 사용자 `ARGS`가 우선하도록 합니다.
`_easyadapter()`는 `i2i_tasks/configs/<task>.toml`이 없는 태스크를 거부하고, 태스크 파일의 `stage`가 이 dict에 없으면 stage 시점에 실패합니다.
`make easycontrol EASYADAPTER=<task>`는 더 이상 편집할 필요가 없습니다.

**`scripts/tasks/inference.py`** (`cmd_test_easycontrol`): `_ADAPTERS`에 행을 추가합니다:

```python
"<task>": {
    "weight": "anima_easycontrol_<task>",   # prefix of output_name
    "out": "<task>",                         # output/tests/<out>/
    "ref_dir": ROOT / "post_image_dataset" / "resized",  # random ref if none given
    "empty_prompt": False,                   # True → default to --prompt ""
},
```

---

## 6. 실행하기

```bash
# First time only: shared target latents + text cache in post_image_dataset/lora.
make preprocess

# 1. Build the reference tree. QA a few first, then eyeball the PNGs under
#    post_image_dataset/i2i/<name>/staging/.
make i2i-stage TASK=<task> ARGS="--limit 8"
make i2i-stage TASK=<task>

# 2. Encode the caches. Idempotent.
make easycontrol-preprocess EASYADAPTER=<task>

# 3. Train (DiT frozen, adapter only).
make easycontrol EASYADAPTER=<task>

# 4. Inference — give it a real, in-distribution reference image.
REF_IMAGE=path/to/condition.png make test-easycontrol EASYADAPTER=<task>
#    Steer with text:  ... ARGS='--prompt "..."'
```

### 인퍼런스 노트

- 인퍼런스 시 레퍼런스는 빌드 단계 없이 그대로 VAE 인코딩됩니다. 빌더가 흉내 내려 했던 것을 입력하세요
  (colorize: 실제 스크린톤이 입혀진 페이지이지, 그레이스케일 사진이 아닙니다).
- `make test-easycontrol`은 항상 `--easycontrol_image_match_size`를 전달하며, 이는
  레퍼런스의 종횡비에 맞는 토큰 버킷을 선택합니다.
- `--easycontrol_scale` (`EC_SCALE=`)은 출력이 레퍼런스를 얼마나 따를지 정합니다;
  `1.0`이 학습 기본값입니다.
- colorize의 튜닝된 `--guidance_scale` / scale / 스텝 범위는
  `colorization/README.md`("Inference settings")에 있습니다.

---

## 7. 체크리스트

- [ ] `i2i_tasks/<task>/`에 결정론적이고 원자적으로 쓰는 빌더
      (`(img, seed) → reference`, 같은 크기)와 멱등 `prep.py`.
- [ ] 레퍼런스를 **원본 크기**로 인코딩 (§3).
- [ ] `i2i_tasks/configs/<task>.toml`: `name`, `[staging]`; 그리고
      `configs/easycontrol/<task>.toml`: `[preprocess]` / `[training]`, `cond_cache_dir`(+ 선택적
      `text_cache_dir` / `latent_cache_dir`)와 `flip_aug = false`를 가진 블루프린트.
- [ ] `_I2I_STAGES` 항목 (training.py)과 `_ADAPTERS` 행 (inference.py).
- [ ] 전체 실행 전에 `--limit 8` 스테이징 배치를 눈으로 확인.

---

## 8. 더 읽을 곳

- **`i2i_tasks/colorization/README.md`** — colorize 레퍼런스 구현:
  캡션 정책, 스크린톤 밴드, 태그 슬라이스, 인퍼런스 설정.
- **`docs/anima/experimental/easycontrol.md`** — 네트워크 자체: 2-스트림 포워드,
  `b_cond` step-0 벤치, `cond_res_scale`, 인퍼런스 캐시, 메모리, 한계.
- **`networks/methods/easycontrol.py`** — `EasyControlNetwork`와 패치된
  `Block.forward`.
- **`networks/CLAUDE.md`** — 모듈별 맵과 디스패치 규칙.
