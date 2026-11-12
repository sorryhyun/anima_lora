# 1. Anima LoRA 학습 레시피

> **요약.** 터보 파이프라인의 첫 단계는 작가 하나에 plain LoRA 하나를 만드는 일이다. 이 LoRA들이 다음 단계의
> soup 재료가 된다. 데이터는 `anime_tools`에서 정리하고 `make preprocess`로 캐시한다. TE는 512 토큰까지
> max-padding되고, 리사이즈는 free-fit 버킷으로 한다. 학습은 `make lora` 한 줄이다. 기본 설정
> (`configs/methods/lora.toml`)은 블라인드 A/B 14회(s29–s42)로 굳힌 레시피로, rank 32 / alpha 128 / lr 5e-5
> constant, GA2, σ 구간별 median clip, adapter EMA 0.1, shift 1이며 REPA와 T-LoRA mask는 끈다. 출처는
> `docs/anima/findings/lora_recipe.md`다. rank, alpha, shift, REPA, down init은 모두 결과 차이가 없었다(null).
> 런을 판정할 때 loss 수치는 쓰지 않는다. 블라인드로 렌더를 비교해서 판정한다.

이 문서가 다루는 범위는 Anima 라인(`train.py`, `networks/`, `configs/{base,presets}.toml`, `configs/methods/`)뿐이다.
MOSS-Image와 Qwen-Image-2.1은 불변식이 다른 별개 라인이므로 여기서 다루지 않는다(`CLAUDE.md`).

---

## 0. 파이프라인에서 이 단계의 위치

| 단계 | 산출물 | 문서 |
|---|---|---|
| **1. LoRA 학습** | 작가별 plain LoRA (`output/ckpt/<output_name>.safetensors`) | 이 문서 |
| 2. soup / merge | 여러 LoRA를 하나로 (`merge_loras.py`, concat_exact, merged14) | → `02_soup_merge.md` |
| 3. bake / int8 / GPTQ | soup를 DiT에 굽고, 그 DiT 전용 GPTQ 파일 생성 | → `03_bake_int8_gptq.md` |
| 4. 터보 증류 | DP-DMD 4-step student LoRA | → `04_turbo_distillation.md` |
| 5. 실전 팁 | fp32 stream, daemon, compile, VRAM | → `05_practical_tips.md` |

이 단계에서 어댑터 종류를 고를 때 따져야 할 제약은 하나다. soup 단계는 **plain LoRA 키 구조만 받는다**.
Hydra는 거부되고(`scripts/soup/build.py`, `.claude/skills/soup/SKILL.md`), T-LoRA는 mask가 학습에만 쓰이기
때문에 저장된 체크포인트가 plain 구조라서 받아진다(§5.1).

---

## 1. 데이터와 캡션

### 1.1 큐레이션은 `anime_tools`에서

캡션 문법, 태그 분류, Anima Tagger, 캡션 스테이지, 마스킹, 그룹핑은 이 저장소가 아니라 형제 저장소
`anime_tools`(`../anime_tools`)에 있다. 의존 방향은 trainer → `anime_tools` 한쪽뿐이다. `pyproject.toml`에서
릴리스 태그로 고정한 git 의존성이라서, `../anime_tools`를 고쳐도 태그를 새로 찍고 pin을 올리기 전에는
`make` 타깃과 daemon job에 반영되지 않는다(`CLAUDE.md` § Curation lives in `anime_tools`). GUI의 anime_tools
탭은 이 패키지의 웹 패널을 띄운다. 패널은 자체 `workspace/` 안에서 작업한다. Export를 누르면 캡션,
마스크, 수정된 master만 내보내고 이미지는 복사하지 않는다. 순서는 **Export → preprocess**다. 제외하기로 한
이미지는 리사이즈의 skip 목록에 합쳐진다(`library/datasets/curation_actions.py`,
`docs/guidelines/가이드북.md` §6.3).

### 1.2 캡션 문법

원본은 `image_dataset/`에 두고, 이미지마다 같은 이름의 `.txt` 사이드카를 붙인다. 태그 순서는
`[meta] [character] [series] [artist] [general]`이고 작가 태그는 `@` 접두사를 붙인다
(`docs/guidelines/가이드북.md` §4.1).

```
absurdres, safe, 1girl, chitanda eru, hyouka, @channel (caststation), full body, serafuku, She is saying hi.
```

캡션에는 태그 묶음 뒤로 절이 더 붙을 수 있다.

- **위치 절**: `<tag bag>. On the left, akita neru, yellow eyes. On the right, kasane teto.` 절과 절은 마침표로
  나누고, 한 절 안의 태그는 콤마로 나눈다.
- **OCR 텍스트 절**: `… Japanese text reads as "…"`. 그릴 텍스트는 영어 문장 안의 큰따옴표에 넣는다.

**캡션을 손으로 쪼개면 안 된다.** `caption.split(",")`를 쓰면 절이 조용히 깨진다. 파싱과 조립은
`anime_tools.captions.position_clauses`(`parse_caption` / `compose_caption`) 하나로만 하고, 학습 시점의 셔플과
`@no-artist` 처리는 `anime_tools.captions.shuffle`이 맡는다(`.claude/skills/captions/SKILL.md`).

캡션을 만드는 타깃은 아래와 같다(`tasks.py`).

| 타깃 | 기본 동작 | 쓰기 위치 |
|---|---|---|
| `make caption-autotag` | dry-run. `ARGS="--apply"`를 줘야 쓴다. `--mode missing\|merge\|overwrite`(기본 `missing`) | revised caption (`post_image_dataset/resized/`) |
| `make caption-position` | dry-run. SAM3 + tagger로 위치 절을 붙이는 GPU 작업이라 daemon을 거친다 | revised caption |
| `make caption-full` | **기본이 쓰기.** position → OCR → 텍스트 절 결합. `ARGS="--dry_run"`이면 계획만 출력 | revised caption |

셋 다 `image_dataset/`의 master는 건드리지 않는다. 실행한 뒤에는 TE를 다시 캐시해야 한다(§1.4).

### 1.3 `make preprocess`의 단계

`make preprocess`의 단계 순서는 `scripts/tasks/preprocess.py::cmd_preprocess`에 고정되어 있고,
`tests/test_preprocess_tasks.py`가 이 순서를 검사한다.

| 순서 | 단계 | 단일 타깃 | 기본 실행 여부 | 산출물 |
|---|---|---|---|---|
| 1 | resize (free-fit, §2) | `make preprocess-resize` | 항상 | `post_image_dataset/resized/` |
| 2 | autotag | `make caption-autotag` | **꺼짐.** `caption_autotag` 키나 `CAPTION_AUTOTAG` env로 켠다 | revised caption |
| 3 | VAE latent | `make preprocess-vae` | 항상 | `post_image_dataset/lora/{stem}_{WxH}_anima.npz` |
| 4 | position clauses | `make caption-position` | **꺼짐.** `configs/preprocess.toml`의 `caption_position_clauses = false` | revised caption |
| 5 | TE | `make preprocess-te` | 항상 | `post_image_dataset/lora/{stem}_anima_te.safetensors` |
| 6 | caption index | `make caption-index` | tagger vocab이 있을 때 | `post_image_dataset/captions/caption_index.json` |
| (7) | PE | `make preprocess-pe` / `preprocess-pe-spatial` | `use_repa = true`일 때만 이어 실행 | `{stem}_anima_pe*.safetensors` |

autotag와 position 단계의 on/off 판정은 `_caption_correction_config`(`scripts/tasks/preprocess.py`)가 한다.
`CLAUDE.md`에는 "resize → autotag → VAE → position clauses → TE → caption index"로 적혀 있는데, 이것은 체인
안에서의 자리를 말한 것이고 2번과 4번은 기본으로 꺼져 있다. 마스크(`make mask`)는 따로 실행하는 단계다.
`masked_loss = false`가 기본이라서, 디스크에 마스크 트리가 있어도 그것만으로 마스크가 켜지지는 않는다
(`configs/CLAUDE.md`).

### 1.4 TE 캐시 규칙

이 부분을 틀리면 학습이 조용히 망가진다. Anima 라인의 규칙이고 MOSS와는 정반대다.

1. **max-padding.** Anima는 TE 출력을 `max_length`까지 패딩한 상태를 전제로 한다. 0으로 패딩된 위치는
   cross-attention에서 attention sink 역할을 한다. 텍스트 길이에 맞춰 잘라 내면 **검은 이미지**가 나온다.
   학습과 추론 모두 `max_length`까지 패딩하고, `crossattn_seqlens`로 패딩을 마스킹하지 않는다(`CLAUDE.md`
   § Invariants). 코드상 패딩 길이는 Qwen3와 T5 쪽 모두 512다(`library/anima/strategy.py`,
   `padding="max_length"`, `qwen3_max_length` / `t5_max_length` = 512).
2. **LLM adapter 출력까지 캐시한다.** `configs/methods/lora.toml`에 `cache_llm_adapter_outputs = true`가 있다.
   그래서 캐시 파일에는 `crossattn_emb`(LLM adapter 출력)와 `t5_attn_mask`만 남는다. 패딩 길이는 512 그대로라
   값은 비트 단위로 같다(`library/preprocess/text.py`의 주석).
3. **TE가 읽는 것은 revised caption뿐이다.** `post_image_dataset/resized/**/{stem}.txt`만 읽고 `image_dataset/`의
   master로 대신하지 않는다. 사이드카가 없으면 **빈 캡션이 인코딩된다.** 그래서 resize와 `preprocess-te`만
   돌리면 빈 프롬프트가 캐시되고, master를 손으로 고친 내용은 캡션 스테이지를 거쳐야 학습에 들어간다
   (`CLAUDE.md` § Preprocessing and caches).
4. **무효화 기준은 mtime이다.** 캐시 파일이 캡션 `.txt`나 `.variants.txt`보다 오래되었으면 다시 인코딩한다
   (`library/preprocess/text.py::_cache_is_current`). 그래서 캡션 스테이지를 `--apply`로 돌린 뒤에는 그냥
   `make preprocess-te`만 실행하면 바뀐 stem만 다시 인코딩된다. mtime으로 알 수 없는 변경, 즉 vocab pack이나
   variant 개수를 바꾼 경우에만 `make preprocess-te ARGS=--overwrite`가 필요하다. CJK vocab pack(`vocab_pack`,
   `configs/base.toml`, 기본 켜짐)을 바꿀 때도 같다.
5. **캡션 변형.** `configs/preprocess.toml`의 기본값은 `caption_shuffle_variants = 4`,
   `caption_tag_dropout_rate = 0.1`이고, 학습 쪽 `use_shuffled_caption_variants = true`가 이 변형을 쓴다.
   v0은 셔플하지 않은 원본이고, 학습 때는 v0을 20%, v1..v3을 80% 확률로 뽑는다(`library/anima/strategy.py`,
   `v0_intact`). 셔플은 `@artist` 태그와 절 표지(`On the ...`)를 고정한 채 절 안의 태그만 돌린다
   (`docs/anima/guidelines/training.md`). 이와 별개로 `caption_dropout_rate = 0.1`은 캡션 전체를 비우는
   dropout이다(`configs/methods/lora.toml`).

> 주의: `docs/guidelines/가이드북.md` §5에는 "캡션을 수정했다면 단순 재실행으로는 반영되지 않으니 캐시 폴더를
> 지우라"고 되어 있다. 코드(mtime 비교)와 같은 문서 §6.3, `captions` skill은 모두 재실행하면 반영된다고 한다.
> 이 리포트는 코드를 따른다. 부록 A 참고.

---

## 2. Free-fit 버킷

Anima가 쓰는 리사이즈 방식은 free-fit 하나다. 이미지마다 **원래 종횡비를 유지**한 채, 패치 그리드의 토큰
수가 자기 tier의 band 안 어디든 들어가도록 크기를 정한다. 그래서 크롭되는 부분은 16px 미만 수준으로 거의
없다(`.claude/skills/bucketing/SKILL.md`). 실제 구현은 `anime_tools.buckets`가 갖고 있다.

| tier edge | 토큰 band (natural) |
|---|---|
| 512 | 1008–1024 |
| 768 | 2160 |
| 896 | 3000–3024 |
| 1024 | 4032–4200 (고정, `FREEFIT_FROZEN_EDGES`) |
| 1280 | 6300 |
| 1536 | 8640 |

1024가 아닌 tier는 ±2.5%만큼 band를 넓혀 종횡비를 맞출 여유를 둔다(`anime_tools/buckets.py`,
`FREEFIT_BAND_TOLERANCE`). 종횡비는 `freefit_max_ratio = 4.0`(`configs/preprocess.toml`)에서 잘린다.

결정 사항:

- **tier는 preprocess 때 정한다.** `configs/preprocess.toml`의 기본값은 `target_res = [1024, 896]`이다.
  이미지마다 리사이즈 폭이 가장 작은 tier로 들어간다. 판정은 `|log(nominal_tokens / native_tokens)|`를
  최소화하는 방식이라, 0.95MP 이미지는 768로 줄지 않고 1024에 남는다.
- **학습에는 `--target_res`가 필요 없다.** 디스크에 있는 캐시의 실제 `(W,H)` 목록이 곧 버킷 집합이다.
  compile 그래프 예산도 실제로 채워진 버킷에서 계산한다(`library/anima/trainer/data.py::derive_token_budget`).
- **free-fit은 `compile_dynamic_seq`를 요구한다.** band 하나에 서로 다른 `(W,H)`가 많이 생기므로 seq 축만
  dynamic으로 두어 tier 하나당 그래프 하나로 묶는다. `configs/base.toml`이 `torch_compile = true`,
  `compile_dynamic_seq = true`를 기본으로 켜 두므로 따로 할 일은 없다.
- **tier를 바꿨다면** `make preprocess-reconcile`(dry-run)로 버킷이 바뀐 이미지의 낡은 latent, resized PNG,
  PE, mask를 확인하고 `ARGS="--delete"`로 지운다. TE 캐시는 텍스트만 담으므로 건드리지 않는다.
- snap 방식 시절의 캐시도 그대로 학습된다. 크롭을 줄이는 이득을 보고 싶을 때만 다시 전처리하면 된다.

> 주의: `docs/guidelines/가이드북.md` §5의 리사이즈 설명("이미지별 고정 토큰 버킷 배정", "0.25MP 미만 제외")은
> free-fit 이전 기준이다. 0.25MP 하한은 `anime_tools.stages.resize`에서 확인하지 못했다.
> `> TODO(확인 필요): 현재 리사이즈에 최소 해상도 하한이 있는지, 있다면 값이 얼마인지.`

---

## 3. config 체인과 `make lora`

### 3.1 병합 순서

```
configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI
```

겹치는 키는 **method가 preset을 이긴다**(`configs/CLAUDE.md`). 각 파일의 성격은 다음과 같다.

| 파일 | 역할 | 주의 |
|---|---|---|
| `configs/base.toml` | 모델 경로, optimizer, σ 샘플링, compile, stream, adaln, channel scaling, 데이터셋 blueprint | **`make update`가 덮어쓴다.** 사용자 값은 여기 두지 않는다 |
| `configs/preprocess.toml` | `target_res`, `mask_dir`, `caption_*`, `freefit_max_ratio` | 사용자 소유. 업데이트해도 보존된다 |
| `configs/presets.toml` | 하드웨어(`default`, `low_vram`, `graft`)와 데이터 범위(`half`, `quarter`, `tenth`, `debug`) | `default`는 `blocks_to_swap = 0`, `low_vram`은 grad ckpt + unsloth offload |
| `configs/methods/lora.toml` | 확정 레시피(§4) | `make lora`가 읽는다 |
| `configs/gui-methods/<variant>.toml` | GUI 변형별 독립 파일 | `make lora-gui GUI_PRESETS=<variant>` |

병합 결과는 `make print-config METHOD=lora PRESET=default`로 확인한다. 학습한 런이 실제로 쓴 설정은
`output/ckpt/<output_name>.snapshot.toml`에 남는다.

### 3.2 명령

```bash
make preprocess                                # 캐시 (§1.3)
make lora                                      # methods/lora.toml + presets[default]
make lora PRESET=low_vram                      # 8GB급
make lora ARTIST=<name>                        # --artist_filter: @<name> 캡션만, 출력은 output/ckpt-artist/
make lora ARGS="--seed 43 --output_name x"     # CLI 덮어쓰기
make print-config METHOD=lora PRESET=default
```

train 타깃은 기본적으로 daemon을 거쳐 실행된다(attach). `--queue`를 주면 붙지 않고 큐에만 넣고, `--inline`을
주면 daemon을 우회한다(`CLAUDE.md` § Commands, `scripts/tasks/_common.py::train`). Claude Code의 백그라운드
Bash에서 GPU 프로세스를 직접 띄우면 sandbox가 약 1분 뒤 죽인다. daemon 사용법은 → `05_practical_tips.md`.

### 3.3 `base.toml`이 이 레시피에 주는 기본값

method 파일이 덮어쓰지 않는 값들이라 레시피의 일부로 봐야 한다(`configs/base.toml`에서 확인).

| 키 | 값 | 의미 |
|---|---|---|
| `optimizer_type` | `AdamW` | |
| `lr_warmup_steps` | `0.05` | warmup은 전체의 5% |
| `timestep_sampling` / `sigmoid_bias` / `discrete_flow_shift` | `sigmoid` / `0.0` / `1.0` | σ ~ logit-normal(0, 1), shift 1 |
| `train_adaln` / `adaln_rank` | `true` / `16` | §5.3 |
| `channel_scaling_alpha` | `0.5` | §5.4 |
| `stream` | `fp32` | → `05_practical_tips.md` |
| `int8` | `off` | → `03_bake_int8_gptq.md` |
| `torch_compile` / `compile_dynamic_seq` | `true` / `true` | §2 |
| `masked_loss` | `false` | |
| `use_cmmd` | `false` | §6.2 |
| `batch_size` (blueprint) | `1` | |

`lr_scheduler`의 base 기본값은 `cosine`이고, `configs/methods/lora.toml`이 이를 `constant_with_warmup`으로
덮어쓴다.

---

## 4. 확정 레시피와 그 근거

### 4.1 레시피

| 항목 | 값 | 어디서 설정되나 | 근거 (`docs/anima/findings/lora_recipe.md`) |
|---|---|---|---|
| rank / alpha | 32 / 128 | `methods/lora.toml` | rank 16은 null(s34 28–20, s41 17–15), alpha 512 / lr 1.25e-5도 null(s35 25–22, s36 18–13, s42 23–22) |
| lr / 스케줄 | 5e-5, `constant_with_warmup` (warmup 5%) | `methods/lora.toml` + `base.toml` | 옛 기본값(2e-5 cosine bs1 clip 1.0)을 41–16으로 이김(s29). 단 ‖dW‖를 맞춘 재실험(s33)에서는 A가 52–42로 앞섬. 1e-4는 손이 엉킴 |
| batch | bs 1, GA 2 | blueprint + `gradient_accumulation_steps = 2` | per-micro clip 아래에서 스파이크 하나가 갖는 몫은 update의 `cap / accum` 이하 |
| clip | `clip_mode = "median"`, `clip_k = 2.0`, `clip_per_micro = true`, `clip_sigma_bands = "0.2,0.4,0.6,0.8"` | `methods/lora.toml` (`clip_window` 기본 50, `library/config/cli_args.py`) | 구간별 clip이 단일 window를 39–21로 이김(s30, p 0.047) |
| adapter EMA | 0.1 | `--ema_window` 기본값 (`library/anima/training.py`) | 일정한 lr에서 raw 가중치에 남는 말풍선 같은 잔여물을 없앤다 |
| σ 샘플링 | logit-normal(0, 1), shift 1 | `base.toml` | shift 2와 무승부(s32 22–23). REPA를 끄면 shift 1이 28–20(s39) |
| REPA | off | `methods/lora.toml` | alpha 512에서는 off가 32–16(s37+s38, pair p 0.015). alpha 128에서는 23–22 무승부(s40) |
| T-LoRA mask | off | `methods/lora.toml` (주석 처리) | 논문대로 학습과 추론 모두 mask를 건 경우 plain과 11–13(s31) |
| down init | `weight_svd` | `methods/lora.toml` | 렌더 차이 없음(§5.2). `weight_svd_nodc` + floor 0.01도 25–23 null(s43) |
| 기타 | AdamW, 8 epoch, caption dropout 0.1, fp32 stream | `methods/lora.toml`, `base.toml` | 이 A/B 라인 이전에 정해진 값 |

**무승부의 뜻.** 이 레시피에서 대부분의 손잡이보다 학습 seed가 결과를 더 크게 흔든다. (작가, seed) 묶음
하나만 봐도 어느 A/B에서든 6–2로 한쪽에 쏠린다. 무승부일 때는 더 단순하거나 싼 쪽을 고르고, 차이가 없으면
원래 출시된 쪽을 유지한다(`lora_recipe.md`).

### 4.2 어떻게 판정했나 (프로토콜)

- arm마다 학습 seed를 여러 개(42–44, 때로 45) 쓰고, 렌더 seed는 2개나 4개, 프롬프트는 4개(교실/교복, 포옹,
  메이드/트레이/카페, 비키니/해변)다. 렌더는 `make test` 설정(28 steps euler, cfg 4, shift 3, 1024²)으로 한다.
  이 값은 `scripts/tasks/_common.py`의 inference base argv와 같다. 채점자 한 명이 블라인드 쌍으로 채점했다.
- **‖dW‖ 맞추기(`match_dw`, s30부터).** treatment의 `lora_up`을 다시 스케일해서 모듈별
  `‖(alpha/rank)·up@down‖`의 중앙값을 같은 (작가, seed)의 control에 맞춘다. 이렇게 하면 보폭이 아니라
  방향만 비교하게 된다. s29는 이 보정 없이 돌았고, B가 2.3–2.5배 멀리 움직인 효과와 섞였다.
- **판정 통계.** (작가, seed) 묶음 단위로 sign test를 하고, 쌍 단위 이항검정은 옆에 참고로 적는다. 학습 seed가
  2개면 묶음 단위 p의 하한이 0.25, 3개면 0.125다.
- 데이터셋: caststation(30장, GA2에서 약 120 update), sweetonedollar(31장, 약 124), ama mitsuki(106장, 424),
  sincos(354장, 약 1416).
- 실행 스크립트는 `_archive/bench/anima_abclip/run_ab.py --exp <name>`이고, 실험별 집계는
  `_archive/bench/anima_abclip/report_<exp>.md`에 있다. 이 경로는 gitignore된 `_archive/` 아래다. 다음 실험
  id는 s44다(memory `project_lora_alpha_lr.md`).

### 4.3 clip: 그래디언트 꼬리는 σ 양 끝에 몰린다

ama mitsuki, bs1 / 5e-5, 848 update에서 측정한 결과다. epoch별 평균 pre-clip norm은 2.20에서 약 0.95로
떨어졌다(전체 중앙값 1.01). 스파이크는 12.8까지 갔고 loss와의 상관은 약했다(r = 0.27). 고정 cap 1.0은
중앙값 바로 위에 있어서, epoch 1에서는 update의 94%를, epoch 5–8에서는 22–29%를 잘랐다. 스파이크를 잡는다기보다
update 절반을 정규화한 셈이다.

꼬리를 만드는 것은 특정 이미지가 아니라 σ다(`--grad_sigma_log`, caststation / ama mitsuki / sweetonedollar,
seed 42).

| σ | 0–0.2 | 0.2–0.4 | 0.4–0.6 | 0.6–0.8 | 0.8–1.0 |
|---|---:|---:|---:|---:|---:|
| 샘플 비율 (합산) | 6.7 % | 23.5 % | 34.3 % | 28.0 % | 7.4 % |
| pre-clip norm 중앙값 | 1.6–2.7 | 0.9–1.4 | 0.9–1.3 | 1.4–2.2 | 2.5–4.9 |
| 전역 median clip에서 잘린 비율 | 33 % | 2.6 % | 1.1 % | 9.8 % | 62 % |

σ 양 끝(σ < 0.2 또는 ≥ 0.8)은 샘플의 13–16%인데 잘린 micro-batch의 50–80%를 차지한다. 이미지 하나가 지배하는
경우는 없었다(8 epoch 동안 이미지당 많아야 2–4회 clip). σ 전체에서 median 하나를 쓰면 σ 끝의 정상 샘플을
스파이크로 오판한다. 그래서 σ 구간마다 window를 따로 둔다(`lora_recipe.md` § Gradients). 이 분포 모양은
T-LoRA mask 유무, rank 16, REPA off 조건에서도 그대로였다.

레시피에 반영한 결정:

- **median 기준 clip을 micro-batch마다 건다.** 스파이크를 `k ×` 평소 norm으로 줄이되 방향은 유지하므로 그
  이미지도 학습된다. 평균을 낸 뒤에 자르면 이미 늦다. norm 10짜리 micro-batch가 norm 1짜리 옆에서 방향을
  정해 버리기 때문이다. 자르지 않은 스파이크는 Adam의 1차 모멘트를 약 10 update 동안 끌고, 2차 모멘트를
  수백 update 동안 부풀린다.
- **GA2로 충분하다.** GA4로 가면 update 수가 반으로 준다. 30장 세트면 60 update인데, 여기서 EMA 0.1의 span은
  6이다.
- 구현은 `library/training/grad_clip.py`에 있다. window는 체크포인트에 저장되지 않으므로 재개한 런은 warmup을
  다시 거친다. 처음 10개 norm이 쌓일 때까지는 `max_grad_norm`(기본 1.0)이 cap이다. clip 비율을 셀 때 이
  구간은 빼야 한다.

### 4.4 lr, 스케줄, EMA

- **EMA에는 일정한 lr이 필요하다.** cosine 감쇠에서는 마지막 약 10%의 update가 거의 움직이지 않아서 EMA 0.1이
  raw 가중치와 같아진다(ama mitsuki: ‖dW‖ 0.997배, cos 1.000). constant 5e-5에서는 0.91배 / cos 0.974이고,
  raw에 남는 말풍선이나 엉뚱한 "?" 같은 잔여물을 없앤다(`_archive/bench/anima_ema/render_ab.py`).
- `--ema_window`의 감쇠는 `1 - 1/(ema_window × updates)`다. 848 update에서 0.988이다. EMA는 lr warmup이 끝난
  시점부터 fp32 shadow로 시작하고, step / epoch / final 저장 때 live 가중치 대신 **EMA를 저장한다.** raw도
  남기려면 `--ema_save_raw`를 준다(`library/anima/training.py`).
- **lr은 변위를 정하고, 변위에는 상한이 있다.** GA2 / 1e-4(bs1 EMA ‖dW‖의 1.41배)에서는 포옹과 메이드
  프롬프트에서 팔과 손이 엉킨다. GA2 / 5e-5(0.68배)는 깨끗하다. 이것은 블라인드 A/B가 아니라 렌더를 읽은
  결과다.
- s29의 승리(41–16)는 T-LoRA mask를 켠 상태였고, ‖dW‖도 맞추지 않았다. mask를 끄고 ‖dW‖를 맞춘 재실험(s33)에서는
  옛 레시피 A가 52–42로 앞섰다(cluster p 0.17). 그러니 B의 이점은 update 방향이 아니다. "B를 자기 ‖dW‖ 그대로
  두고 A와 비교"하는 실험은 아직 열려 있다(§4.8).

### 4.5 rank와 alpha: 손대지 않는다

arXiv 2606.12883은 LoRA 한 step을 task 방향 신호 `s·(BΔA + ΔB·A)`(Θ(αη))와 bilinear drift `s·ΔB·ΔA`(Θ(αη²))로
나눈다. α·η가 같으면 1차 update는 같고 drift는 1/η에 비례해 줄어든다. 그래서 "alpha를 높이고 lr을 낮추면 더
깨끗하다"는 예측이 나온다. 실험은 α·η를 출시값에 고정하고 둘의 배분만 바꿨다.

- **α·η가 같으면 ‖dW‖도 같다.** a512는 a128의 0.96–0.99배였다(30장). sincos(약 1416 update)에서는 0.97배였다.
- **alpha/rank를 고정하면 Adam에서 ‖dW‖는 √r에 비례한다.** r16 / alpha 64는 r32의 0.72배만 움직였다(s34).
  alpha를 √r에 비례시키면(r16 / a362 vs r32 / a512) 학습 중 보폭이 맞았다(0.95배 / 1.09배, s41).
- **drift는 실제로 있지만 렌더에는 나타나지 않는다.** 모듈 중앙값 기준 `‖B·D‖ / ‖B·A0‖`는 a128에서 0.15–0.17,
  a512에서 0.06이었다. 블라인드 다섯 세트(s35, s36, s42; rank는 s34, s41)가 두 arm을 구별하지 못했다.
- rank는 세 작가에서 REPA 유무와 관계없이 null이었다. rank 16으로 얻는 것은 파일 크기가 절반이 된다는 점뿐이다.

결론: **rank 32 / alpha 128 / lr 5e-5를 유지한다.** rank를 바꿔야 한다면 alpha를 √r 비율로 맞춘다.
adaln 브랜치는 이미 이 규칙을 따른다(§5.3).

### 4.6 σ shift: 1을 유지한다

`inference.py --lora_sigma_range LO HI`는 `[LO, HI)` 밖에서 LoRA를 끈다. sweetonedollar, 네 프롬프트, 평균 절대
픽셀 차(0–255)로 측정했다.

| 조건 | vs LoRA 전 구간 on | vs base |
|---|---:|---:|
| static merge ×1.0 (hook 경로의 잡음 바닥) | 1.6–2.0 | — |
| σ < 0.4에서 off (28 step 중 5) | 2.4–2.6 | — |
| σ ≥ 0.8에서 off (28 step 중 12) | 48–58 | **8.5–8.9** |
| LoRA 전 구간 on | — | 50–59 |

σ ≥ 0.8에서 LoRA를 끄면 렌더가 base와 거의 같아진다. 반대로 σ < 0.4에서 끄면 잡음 바닥 이상의 변화가 없다.
그러니 LoRA가 일하는 곳은 높은 σ다. shift 1에서 이 구간은 학습 샘플의 7.4%(런당 약 18 micro-batch)만 받는다.
그래서 shift를 올려 보는 것이 자연스러운 시도였지만, shift 2는 무승부였다(s32 22–23). REPA를 끄자
caststation은 동률, sweetonedollar는 shift 1 쪽으로 남았다(s39 28–20). shift 1을 유지한다.

### 4.7 REPA와 T-LoRA mask: 끈다

- **REPA**(relational, weight 0.05, layer 8, PE-Spatial). 손해 여부가 step 크기에 따라 달랐다. a512 / lr 1.25e-5에서는
  16–32로 졌고(s37+s38), a128 / lr 5e-5에서는 23–22로 비겼다(s40). 그래서 출시 step에서 REPA를 켜고 학습한
  LoRA가 손해를 본 것은 측정되지 않는다. **loss로는 REPA를 걸러낼 수 없다.** REPA 항을 빼면 두 arm의 fm loss가
  같았고(caststation s42: 0.0731 vs 0.0723), grad norm과 ‖dW‖도 거의 그대로였다. REPA는 fm 적합도는 그대로 두고
  렌더만 바꾼다. EasyControl 설정에서는 계속 켜 둔다(cond ≠ target인 다른 regime이라 여기서 검증하지 않았다).
- **T-LoRA mask.** 2026-11-05까지 출시본은 학습에만 mask를 걸고 추론은 full rank로 했다. 학습한 모델과 렌더하는
  모델이 서로 달랐던 셈이다. 논문대로 추론에도 mask를 걸면(`inference.py --lora_timestep_mask`, ComfyUI에서는
  custom node 필요) plain과 11–13으로 비겼다(s31, sweetonedollar 한 명). 추론 쪽 비용만 늘고 얻은 것이 없어서
  기본값에서 뺐다.

### 4.8 열린 문제

`lora_recipe.md` § Open에 남아 있는 항목을 그대로 옮긴다.

- B를 자기 ‖dW‖ 그대로 두고 A와 비교하기(둘 다 mask 없이). B가 이기면 지렛대는 clip이 아니라 보폭(lr / 스케줄)이다.
- 긴 런(sincos, 약 1416 update)에서의 REPA. 지금까지의 검증은 모두 30–106장이었다.
- T-LoRA, Hydra, soup은 REPA off를 검증 없이 물려받았다.
- σ 구간별 clip은 두 작가, p 0.047로 출시되었다.
- sweetonedollar는 ×1.0에서 기형적으로 큰 손을 그리고, ×0.8 / ×0.6에서는 손이 줄어든다. 손과 스타일을 무엇이
  가르는지(σ-gated LoRA, 렌더 종횡비, 데이터)는 검증하지 않았다.

---

## 5. 어댑터 선택

### 5.1 plain LoRA, T-LoRA, Hydra

| 변형 | 설정 | soup(`02`) | `make merge` / bake(`03`) | ComfyUI 기본 LoraLoader |
|---|---|---|---|---|
| **plain LoRA** | `configs/methods/lora.toml` | 가능 | 가능 | 가능 (`lora_unet_` 키) |
| T-LoRA | `use_timestep_mask = true` (`gui-methods/tlora.toml`) | 가능 (mask는 학습 전용이라 키 구조가 plain) | 가능 | 가능. 단 논문식 추론 mask는 custom node 필요 |
| HydraLoRA | `use_moe_style = "shared_A"` + `router_source = "sigma"` (`gui-methods/hydralora.toml`) | **거부** | router-live는 불가. moe는 `--allow-partial` 없이 거부 | baked-down 파일만 가능. router 키는 버려짐 |

출처: `scripts/soup/build.py`, `CLAUDE.md` § Commands(`make merge`), `docs/anima/structure/lora.md` §4.

이 파이프라인의 재료로는 **plain LoRA**를 쓴다. T-LoRA는 §4.7에서 본 대로 이득이 측정되지 않았고, Hydra는 soup
단계를 통과하지 못한다. routing 축의 자세한 내용은 `lora-routing` skill과 `networks/CLAUDE.md`에 있다.

### 5.2 `down_init = "weight_svd"`

`lora_down`을 Kaiming 대신 `W₀`의 상위 r개 right singular vector로 시작한다. 정확히는 `A₀ = V_rᵀ/√3`,
`B₀ = 0`이다. `1/√3`은 Kaiming의 기대 row-norm에 맞춘 값이라 바뀌는 것은 보폭이 아니라 방향뿐이다. 초기 ΔW는
0이다. 저장되는 키, 병합, 추론 경로는 plain LoRA와 같다(`docs/anima/methods/svd-down-lora.md`). 코드 기본값은
`"kaiming"`이고(`networks/lora_anima/config.py`), `configs/methods/lora.toml`이 `weight_svd`로 바꾼다.

측정된 사실:

- 첫 step의 gradient 에너지 중 `weight_svd`가 담는 비율은 0.21, 작가 자신의 gradient basis는 0.74다. 그런데
  256 step(`aak`, r32 / α128, 8 epoch)을 학습하고 나면 **어떤 init도 렌더를 바꾸지 못한다.** weight_svd vs
  kaiming 12–9, vs grad_svd 11–9, vs basis_file 10–10이고 모두 seed 쌍 잡음 바닥(15–9) 안이다
  (`bench/grad_init/README.md` § E1, 블라인드 s22–s25).
- `weight_svd_nodc` + `channel_scaling_floor 0.01` vs `weight_svd`는 25–23 null이었다(s43). 두 손잡이 모두 살아
  있지만 기본값은 바꾸지 않았다.

렌더에서 차이가 없는데도 `weight_svd`를 쓰는 실질적 이유는 soup 쪽에 있다. `svd_slice = k`는 `lora_down`을
`[k·r, (k+1)·r)` 번째 singular vector에서 시작시킨다. slice끼리는 서로 직교하므로, 작가마다 다른 k를 주면 병합할
때 입력 부분공간이 겹치지 않는다. r=32에서는 256행짜리 adaln `.1` Linear가 상한이 되어 k ≤ 7, 즉 작가 8명까지다.
`svd_slice_count`는 slice를 교차 배치하는 방식이다(`configs/methods/lora.toml` 주석). adaln 모듈은 slice 설정과
관계없이 항상 top-r로 시작한다. slice 배치와 병합 방식은 → `02_soup_merge.md`.

### 5.3 adaln LoRA (`train_adaln`)

Anima 블록마다 있는 세 modulation MLP(`adaln_modulation_{self_attn,cross_attn,mlp}`, 2048→256→6144)는 timestep
embedding만 입력으로 받는다. 그래서 여기에 붙는 LoRA는 σ에 따른 전역 톤, 대비, gate만 배울 수 있고 텍스트나
공간 정보는 배울 수 없다(`docs/anima/methods/adaln.md`).

- `configs/base.toml`에 `train_adaln = true`, `adaln_rank = 16`이 있어서 모든 LoRA 계열 method에 적용된다.
  기본 exclude에 `adaln_up_`이 들어 있지만 `train_adaln`이 include pattern으로 다시 넣는다
  (`networks/lora_anima/config.py::_DEFAULT_EXCLUDE`).
- `adaln_alpha`를 비워 두면 √r 규칙 `network_alpha · sqrt(adaln_rank / network_dim)`으로 계산된다. 출시 레시피에서는
  128 · √(16/32) ≈ 90.5다(공식은 `adaln.md`, 수치는 이 문서에서 계산한 값).
- rank 16의 근거: 학습된 adaln ΔW는 유효 rank가 낮다. 공식 turboV10의 full-rank delta도 r16에서 에너지의 93%가
  담긴다. soup s1001 스타일 LoRA는 r16에서 89%다(`adaln.md` § Sizing).
- 저장할 때 키를 ComfyUI 배치(`blocks.{b}.adaln_modulation_{br}.2`)로 바꿔 쓰고 `ss_adaln_layout = "comfy"`를
  찍는다. 저장소 안에서 로드할 때는 runtime 이름으로 되돌린다. 파일 하나로 두 생태계에서 다 쓸 수 있다.
- soup에서는 shard별 adaln delta를 합치지 않고 평균 낸다(`merge_loras.py --adaln average`가 기본). → `02_soup_merge.md`.

### 5.4 Channel scaling (`channel_scaling_alpha = 0.5`)

SmoothQuant 방식의 재매개화다. 보정된 채널별 스케일 `s[c] = (mean|x[c]|)^α`를 `lora_down` 열에 흡수시키고
forward에서는 `x / s`를 쓴다. 초기 출력은 비트 단위로 같고, 달라지는 것은 Adam의 채널별 보폭뿐이다. α = 0.5에서
채널 지배도(max/mean leverage)는 약 13–80배에서 3.8–9.3배로 내려간다(`docs/anima/optimizations/channel_scaling.md`).

- 보정 파일은 저장소에 포함되어 있다(`networks/calibration/channel_stats.safetensors`). base DiT가 바뀌지 않는 한
  다시 만들 필요가 없다. 이 파일은 depth에 묶여 있다(`CLAUDE.md` § Invariants).
- 저장할 때 `bake_inv_scale`이 `inv_scale`을 `lora_down`에 접어 넣으므로, 디스크의 체크포인트는 그냥 plain
  LoRA다. 추론에는 영향이 없다.
- **일반 FM LoRA에서 α 0.5 vs 0을 비교한 블라인드 A/B는 없다**(`channel_scaling.md` § When this helps more).
  관련된 측정은 하나다. 거의 죽은 채널은 `lora_down` 열이 random walk를 하는데, 676-update soup shard에서 이 열들이
  저장된 ‖ΔW‖²의 33–43%를 차지했다. 이것이 ‖ΔW‖를 읽는 도구(`merge_loras.py --normalize global`, `match_dw` 등)를
  왜곡한다. `channel_scaling_floor = 0.01`이면 이 비중이 0이 되지만, 렌더 A/B(s43)에서는 null이었고 기본값은
  0(off)이다(`networks/lora_anima/factory.py`).
- 터보 증류에서는 α 0.5가 확실히 손해였고 기본값이 0.0이다. → `04_turbo_distillation.md`.

---

## 6. 런 판정

### 6.1 무엇이 기록되나

TensorBoard 스칼라는 `output/logs/<run>/`에, `train.py` 런의 구조화 이벤트는 `output/logs/<output_name>.progress.jsonl`에
남는다(`make run-status`로 추적). JSON으로 뽑으려면 다음과 같이 한다.

```bash
python scripts/toolkits/export_logs_json.py output/logs/<run_dir> --summary   # 태그별 마지막 값
python scripts/toolkits/export_logs_json.py output/logs/<run_dir> --stdout    # {"tags": {tag: [[step, wall, value], ...]}}
```

값은 `row[2]`다(`scripts/toolkits/export_logs_json.py` docstring). 이 레시피에서 볼 만한 태그는 다음과 같다.

| 태그 | 읽는 법 |
|---|---|
| `loss/current`, `loss/average` | fm loss다(REPA가 켜져 있으면 `0.05 · repa/align_loss`가 더해진다). 수렴과 발산을 확인하는 용도이고, **arm 간 품질 순위를 매기는 데 쓰지 않는다** |
| `grad/norm`, `grad/norm_max`, `grad/clip_frac`, `grad/clip_cap`, `grad/micro_norm_sq` | median clip 진단. 건강한 런은 epoch 평균 norm이 떨어진다(ama mitsuki에서 2.20 → 약 0.95). 처음 10 micro-batch는 warmup cap 1.0에 걸려 있다 |
| `lr/<group>` | constant_with_warmup이면 5% 지점부터 평평해야 한다 |
| `liveness/<name>` | REPA 같은 보조 loss가 켜져 있는데 입력을 한 번도 소비하지 못했으면 1.0보다 작고, 로그에 `LIVENESS:` ERROR가 찍힌다(`library/training/loop.py::_audit_liveness`) |

`--grad_sigma_log`를 주면 `<output_name>.grad_sigma.jsonl`에 clip 단위별 norm, cap, σ, loss, image key가 기록된다.
§4.3의 σ 분포를 다시 그릴 때 쓴다.

### 6.2 loss가 낮다고 좋은 런이 아니다

- REPA 실험이 그 예다. REPA를 켠 arm과 끈 arm은 fm loss가 같았는데(0.0731 vs 0.0723) 렌더는 블라인드에서
  16–32로 갈렸다(§4.7). loss는 렌더 차이를 보지 못한다.
- `docs/anima/guidelines/base-config.md`: per-σ FM-MSE 검증 loss는 Anima에서 샘플 품질과 상관관계를 보이지 않았다.
  검증을 켠다면 `use_cmmd = true`(PE-Core 기반 paired CMMD², `make preprocess-pe` 필요)를 쓴다. 검증은 기본으로
  꺼져 있다(`validation_split_num = 0`, `base.toml`의 `use_cmmd = false`). 참고로 `--use_cmmd`의 argparse 기본값은
  `True`이고 `base.toml`이 이를 끈다(`library/config/cli_args.py`).
- 다만 CMMD도 만능은 아니다. down_init E1 실험에서는 모든 arm의 CMMD가 real-vs-real 바닥(0.61) 아래였고, 그래서
  순위를 매길 수 없었다(`bench/grad_init/README.md`).
- 마지막 판정은 렌더로 한다. 레시피 A/B와 같은 방식이 기준이다. 학습 seed를 2–3개, 렌더 seed를 2개 이상 쓰고,
  `make test` 설정(28 step, cfg 4, shift 3, 1024²)으로 렌더해서 블라인드 쌍으로 비교하며, ‖dW‖를 맞춘다(§4.2).
  한 번에 여러 장을 렌더할 때는 `make gen`(daemon 배치)을 쓴다.

### 6.3 판정할 때 빠지기 쉬운 함정

`lora_recipe.md` § Traps:

- **arm 간 끝점 ΔW cosine에는 신호가 없다.** 같은 설정을 두 번 돌려도 dW cosine 중앙값이 0.146(bs1 5e-5)이다.
  반면 변위(‖dW‖) 자체는 ±3% 안에서 재현된다(`docs/anima/findings/paired_dw_chaos_floor_deterministic.md`).
- **런 하나의 렌더 하나는 그 설정의 성질이 아니다.** bs1 쌍둥이 런은 말풍선은 재현했지만 첫 런의 과포화 마젠타는
  재현하지 않았다. 학습 seed를 바꿔도 반복되는 결함만 그 설정의 것으로 본다.
- **seed가 같아도 쌍 비교가 아니다.** 두 arm이 같은 σ를 뽑더라도 clip이 한 번 다르게 걸리는 순간 갈라진다.
- **cross-seed MAD로 사전 선별하지 않는다.** s32–s35에서는 6건 중 5건이 맞았지만 s39–s42에서는 맞지 않았다.

### 6.4 학습 규모 감 잡기

update 수는 이미지 수 × epoch / GA다. 30장 × 8 epoch / 2 = 120으로, `lora_recipe.md`의 caststation 수치(약 120)와
같다. 레시피 검증은 30–354장 범위에서 했다. 이보다 훨씬 큰 세트에서 REPA나 rank가 다르게 작동하는지는 열린
문제다(§4.8).

> TODO(확인 필요): 5070 Ti에서 30장 / 8 epoch 기준 런 하나의 소요 시간과 피크 VRAM. 이 저장소 문서에서 LoRA 학습
> 1회의 wall-clock 수치를 찾지 못했다.

---

## 7. 이 단계의 산출물과 다음 단계

- 출력: `output/ckpt/<output_name>.safetensors`(EMA 가중치), `.snapshot.toml`. `make lora ARTIST=<name>`이면
  `output/ckpt-artist/<output_name>_<artist>.safetensors`다. 메타데이터에 `ss_clip_mode`, `ss_clip_k`, `ss_svd_slice`,
  `ss_adaln_layout`, `ss_num_blocks`가 찍힌다.
- LoRA는 depth에 묶여 있다. 28-block base에서 학습한 LoRA를 40-block Anima-2.9B에 올리면 꼬리가 버려진다
  (`CLAUDE.md` § Invariants).
- 다음 단계: 작가별 LoRA 여러 개를 soup나 merge로 묶는다. → `02_soup_merge.md`. `make soup`는 별도 설정
  (`configs/soup/soup.toml`: lr 2e-5, 4 epoch, uncond-init)을 쓰므로 이 문서의 레시피와 다르다. 차이는 02에서 다룬다.

---

## 부록 A. 작성 중 발견한 문서와 코드의 불일치

| 위치 | 문서 내용 | 실제 (코드 / config) |
|---|---|---|
| `docs/anima/structure/lora.md` §3.4 | `lora.toml`이 "the REPA auxiliary alignment loss"를 쌓는다 | `use_repa = false` (2026-11-06부터) |
| `docs/anima/guidelines/training.md` § LoRA family | `lora.toml`은 "T-LoRA and REPA on" | 둘 다 off. 같은 문서의 몇 줄 아래 T-LoRA 절은 off라고 적혀 있다 |
| `docs/guidelines/가이드북.md` §7, §8 | `tlora`로 시작하라고 권장 ("🌟 추천") | 확정 레시피는 plain LoRA다. T-LoRA mask는 null(s31) |
| `configs/gui-methods/tlora.toml`, `hydralora.toml` | GUI 변형 | 확정 레시피를 따르지 않는다. alpha 32, 4 epoch, base의 cosine 스케줄, GA 1, 고정 clip(`clip_mode` 기본 `fixed`)이다. EMA 0.1은 argparse 기본값이라 적용되지만, cosine에서는 raw와 거의 같아진다(§4.4) |
| `configs/gui-methods/lora.toml` | GUI `lora` 변형 | `down_init`이 없어서 코드 기본값 `kaiming`이 된다. `methods/lora.toml`은 `weight_svd`다. 렌더 차이는 null(E1)이지만 `svd_slice`를 쓸 수 없다 |
| `docs/guidelines/가이드북.md` §5 | 캡션을 수정하면 캐시 폴더를 직접 지워야 한다 | TE 캐시는 mtime을 보고 다시 인코딩한다(`library/preprocess/text.py::_cache_is_current`) |
| `tasks.py` `caption-autotag` / `cmd_caption_autotag` docstring | "caption edits do NOT invalidate the TE caches" | 위와 같다. `make preprocess-te`를 다시 실행하라는 지시 자체는 맞다 |
| `docs/guidelines/가이드북.md` §5 | 리사이즈가 "고정 토큰 버킷 배정, 0.25MP 미만 제외" | free-fit band다. 0.25MP 하한은 확인하지 못했다 |
| `.claude/agents/log-analyst.md` | `liveness/<name> < 1.0` = "some adapter params never moved" | 보조 loss가 입력을 소비한 비율이다(`library/training/losses.py::LivenessLedger`). 이 에이전트가 참조하는 memory 파일(`project_cmmd_val_signal`, `project_fm_val_loss_uninformative` 등)도 현재 memory 디렉터리에 없다 |
| memory `project_lora_alpha_lr.md` | runner 경로 `bench/anima_abclip/run_ab.py` | `_archive/bench/anima_abclip/run_ab.py` (`bench/anima_abclip/`는 없다) |
| `docs/anima/guidelines/base-config.md` `int8` 행 | "Refuses `blocks_to_swap`" | `CLAUDE.md`는 학습에서 `blocks_to_swap`이 int8 위에서 돈다고 한다. 이 리포트의 범위 밖이다. → `03_bake_int8_gptq.md`에서 확인 |
