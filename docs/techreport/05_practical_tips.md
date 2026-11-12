# 5. 실전 팁: 16 GB 카드에서 파이프라인 굴리기

> 초안 (2026-11-12). 수치는 모두 본문에 적은 경로에서 가져왔다. 측정 날짜가 오래된 것은 날짜를 함께
> 적었다. 확인하지 못한 부분은 `TODO(확인 필요)`로 남겼다.

**요약.** 이 섹션은 1–4장(LoRA → soup → bake/int8 → turbo)에 공통으로 걸리는 엔지니어링 팁을 모은다.
조용히 결과를 망가뜨리는 모델 불변식(TE padding, 5D latent, compile 순서), 기본값인 fp32 residual
stream, torch.compile과 free-fit 버킷의 결합, 16 GB에서 VRAM을 맞추는 순서(activation budget →
block swap → grad checkpointing), GPU 작업을 daemon으로 돌리는 방법, 추론 쪽 설정, 그리고 run noise를
넘는 판정을 내리기 위한 실험 위생을 다룬다. 각 팁은 무엇을 할지, 왜인지, 그걸 알려준 수치나 실패,
출처 경로 순으로 적는다.

기준 환경은 RTX 5070 Ti 16 GB, RAM 64 GB(사용 가능 55–58 GB) + swap 8 GB다
(`~/.claude/.../memory/project_box_environment.md`). 아래 수치는 별도 표기가 없으면 이 카드에서 잰 것이다.

---

## 5.1 깨도 에러가 안 나는 모델 불변식

Anima 라인의 불변식은 대부분 위반해도 예외가 나지 않는다. 결과만 틀어진다. MOSS / Qwen 라인의
불변식은 반대인 경우가 많으므로(예: MOSS는 TE를 절대 padding하지 않는다) 파일이 어느 라인 소속인지 먼저
확인한다(`CLAUDE.md`).

### TE 출력은 512 토큰까지 zero-padding한 채로 쓴다

- **할 일.** 학습과 추론 모두 TE 출력을 `max_length`(512)까지 padding한다. padding 위치를
  `crossattn_seqlens`나 attention mask로 가리지 않는다.
- **이유.** 사전학습된 Anima는 zero-padding 위치를 cross-attention의 attention sink("볼 곳이 없을 때 보는
  곳")로 쓰도록 학습됐다. caption 길이로 잘라내거나 mask로 지우면 softmax 분모가 무너지고
  cross-attention이 포화되어 **검은 이미지**가 나온다(`docs/anima/structure/anima.md` §2.4).
- **부수 효과.** KV 길이가 항상 512로 고정되므로 cross-attention 쪽은 compile shape가 공짜로 안정된다
  (`docs/anima/optimizations/for_compile.md` 요약 표). 실제 caption은 30–80 토큰만 채운다
  (`docs/anima/optimizations/fa4.md`). FA4 시절 KV를 잘라내고 LSE로 sink 기여를 복원하던 경로는
  2026-05-20에 제거됐다.
- tokenizer나 padding을 바꿨다면 TE 캐시를 다시 만든다.

### Latent는 5D `(B, C, T=1, H, W)`, 바깥은 4D

- **할 일.** DiT 경계에서만 `unsqueeze(2)` / `squeeze(2)`를 쓴다. `squeeze()`나 `squeeze(0)`는 쓰지 않는다.
- **이유.** VAE encode, 캐시된 latent, 학습 루프, spectral helper, PE tower는 모두 4D다. 인자 없는
  `squeeze()`는 batch 1일 때 batch 축까지 지운다. sampler 경계 plug-in은 5D를 받고 reference latent는
  4D인 경우가 많으므로, 섞기 전에 `ndim`부터 맞춘다(`CLAUDE.md`, `docs/anima/inference/README.md`).

### 로딩 순서: TE → 캐시 → 해제 → VAE → 캐시 → 해제 → DiT → adapter → 학습

16 GB에 TE, VAE, DiT를 동시에 올리지 않기 위한 순서다(`CLAUDE.md` "Lazy loading"). 캐시가 이미 있으면
학습 시에는 TE와 VAE를 아예 올리지 않는다(`configs/base.toml`: `use_vae_cache = true`,
`use_text_cache = true`).

### compile은 adapter를 붙인 뒤에

- **할 일.** `network.apply_to` + `load_weights` 다음에 `compile_blocks()`를 부른다. 직접 조립하지 말고
  `library/runtime/harness.py::build_anima`를 쓴다.
- **이유.** `torch.compile`은 adapter가 monkey-patch한 forward를 trace한다. 순서가 바뀌면 patch 전
  forward를 조용히 compile하고, LoRA가 없는 것처럼 돈다(`library/runtime/harness.py` 모듈 docstring).
- fp32 stream 변환(5.2절)과 int8 변환은 그보다도 앞, adapter가 붙기 전에 일어난다
  (`library/anima/stream.py` docstring).

### 블록 수는 checkpoint에서 읽는다

- `library/anima/weights.py::probe_dit_arch()`가 `blocks.N.` 키를 세어 깊이를 정한다. 40-block
  Anima-2.9B도 플래그 없이 파일만 가리키면 된다(`docs/anima/methods/anima-2.9b.md`).
- **LoRA는 깊이에 묶인다.** 40-block LoRA를 28-block base에 올리면 뒤쪽 블록이
  `not all LoRA keys are used` 경고 하나만 남기고 버려진다. `ss_num_blocks`가 기록된다.
- `networks/calibration/`의 `channel_stats.safetensors`, `dave_alpha.npz`도 깊이에 묶여 있다. CNS γ는
  아니다(`CLAUDE.md`).
- int8 GPTQ 파일도 fingerprint로 원본 checkpoint를 확인하므로, 2.9B나 bake한 DiT는 자기 GPTQ 파일이
  필요하다(`docs/anima/optimizations/int8.md` § Forward copies, 3장 참조).
- 2.9B 학습은 16 GB에서 default preset이 바로 OOM이고(bf16 가중치만 5.9 GB), `--blocks_to_swap 16`은
  한 스텝 뒤 OOM, `--preset low_vram --blocks_to_swap 0`이 동작한다(4.6 → 2.2 s/it). 추론은 832×1216,
  32 step, CFG 4에서 이미지당 약 31.5 s로 28-block base(약 24.5 s)보다 28% 느리다
  (`docs/anima/methods/anima-2.9b.md`).

### Caption은 손으로 쪼개지 않는다

- caption은 `<tag bag>. On the left, akita neru, yellow eyes. On the right, kasane teto.` 형태로, 마침표가
  clause를, 쉼표가 clause 안의 tag를 나눈다. `caption.split(",")`는 clause를 조용히 깨뜨린다.
  `anime_tools.captions.position_clauses`의 `parse_caption` / `compose_caption`을 쓴다
  (`.claude/skills/captions/SKILL.md`).
- TE 캐시는 resize된 이미지 옆의 수정본 caption(`post_image_dataset/resized/**/{stem}.txt`)**만** 읽고,
  `image_dataset/` 원본으로 fallback하지 않는다. sidecar가 없으면 빈 caption을 인코딩한다. 원본을 손으로
  고쳤다면 caption stage를 거쳐야 학습에 반영된다(`CLAUDE.md` "Preprocessing and caches").
- TE 캐시는 caption mtime으로만 무효화된다. CJK vocab pack을 바꾸거나 끄면 mtime 검사가 못 잡으므로
  `make preprocess-te ARGS=--overwrite`를 돌린다(`configs/base.toml`의 `vocab_pack` 주석).
- `make caption-autotag` / `caption-position` 뒤에는 `make preprocess-te`가 필요하다(`tasks.py` help).

---

## 5.2 fp32 residual stream (기본값)

### 무엇이 fp32로 도는가

`--stream fp32`는 block GEMM 바깥을 fp32로 돌린다. block linear와 FA2는 bf16(또는 int8) 그대로이고,
파라미터 dtype은 바뀌지 않는다(`docs/anima/optimizations/fp32_stream.md`, `library/anima/stream.py`).

| fp32 | bf16 / int8 그대로 |
|---|---|
| residual stream `x`, modulation, q / k norm, RoPE, embedder, head | block linear(qkv, output_proj, cross q, mlp), FA2, LoRA 자체 가중치 |

구현은 embedder / head / modulation linear를 `Fp32Linear`(호출 시에만 weight를 cast, autocast 밖)로,
q / k norm을 `Fp32RMSNorm`으로 바꾸고 나머지 DiT forward를 bf16 autocast로 감싼다. 그래서 block linear
입력은 bf16으로 반올림되고 int8 linear는 fp32 kernel 입력을 받는다. 파라미터가 늘지 않으므로 가중치
VRAM은 그대로다(`library/anima/stream.py` docstring).

### 기본값과 스위치

- 학습: `configs/base.toml`의 `stream = "fp32"`. `make lora ARGS="--stream bf16"`로 끈다.
- 추론: `inference.py --stream` 기본값 `fp32`(`library/inference/args.py`). `make test STREAM=bf16`.
- Turbo: `configs/methods/turbo.toml`이 `stream = "fp32"`를 teacher까지 포함해 쓴다.
- **EasyControl은 예외**로 bf16 stream으로 돌고 경고를 남긴다
  (`library/anima/stream.py::settle_training_stream`). 단, `.snapshot.toml`에는 설정값인 `fp32`가 찍힌다
  (`fp32_stream.md`). snapshot만 보고 stream을 판단하지 않는다.
- LoRA에는 `ss_stream`이 기록만 되고, 로드 쪽에서 강제하지 않는다. compile 캐시 키에는 `;stream=fp32`
  접미사가 붙는다(`fp32_stream.md`).

### 왜 쓰는가: 수치

예측 오차 `‖v − v_fp32‖ / ‖v_fp32‖`(%, LoRA 없음, 6장 평균; `fp32_stream.md` Accuracy 표):

| σ | bf16 stream | fp32 stream | int8 GPTQ + bf16 kernel 입력(ComfyUI stock) | int8 GPTQ + fp32 stream |
|---|---|---|---|---|
| 0.1 | 1.76 | 0.52 | 2.29 | 1.32 |
| 0.5 | 0.85 | 0.22 | 0.99 | 0.48 |
| 0.9 | 1.15 | 0.23 | 1.07 | 0.50 |

- fp32 stream은 bf16 GEMM 주변 예측 오차를 1.5–2.1배, int8 base 주변 오차를 21–36% 줄인다. LoRA
  gradient는 bf16 GEMM 주변에서 블록 14–27이 개선되고, int8 주변에서는 섞여 있다
  (`project/finished/int8_train/reports/anima_fp32_stream.md`).
- 왜 fp16 stream이 아니라 fp32인가: stream은 블록 12–20에서 261k–285k까지 올라가 fp16 최대값(65504)의
  4.3배다. all-fp16 forward는 모든 이미지에서 블록 12부터 NaN이 된다. block GEMM 입출력은 ≤ 747로 fp16에
  충분히 들어간다(`docs/anima/findings/fp16_headroom.md`). 즉 문제는 GEMM이 아니라 stream이다.

### 비용

- 추론(8-step, 768², smoke 1회): bf16 261 ms/step, 4.11 GiB → fp32 280 ms/step, 4.17 GiB
  (`fp32_stream.md`, 결과가 아닌 smoke라고 문서에 명시).
- Turbo 학습(int8 `stream`, budget 0.99): bf16 stream 6.28 s/step, 11.87 GB → fp32 6.7–7.0 s/step,
  12.39 GB (`docs/anima/optimizations/int8.md` § Limits).
- fp32 stream의 saved set이 더 크므로, bf16 시절의 activation budget 표는 그대로 옮기지 않는다
  (`fp32_stream.md`).

### 주의: 같은 seed가 다른 그림을 낸다

noise 자체는 같지만(fp32 draw가 bf16 draw로 반올림됨) 최종 latent가 768²에서 6–12 step 5회 실행 기준
12–24% 움직인다(`fp32_stream.md`). bf16 시절 렌더와 seed 단위로 비교하지 않는다.

### "눈에 보이는 이득"에 대해

사용자가 렌더를 보고 "일관되게 낫다(글자 렌더링, 손가락 위치)"고 판정한 것은 **MOSS** 라인의 fp32
stream이다(`memory/feedback_settled_verdicts.md`). Anima 쪽 근거는 위의 수치(예측 오차, gradient)다.
2026-10-31 리포트는 "Anima keeps its bf16 stream"으로 끝났지만, 그 뒤 기본값이 fp32로 바뀌었다
(`configs/base.toml`).

> TODO(확인 필요): Anima에서 fp32 vs bf16 stream을 렌더로 A/B한 기록(작은 구조, 글자, 손가락 기준)을
> 찾지 못했다. 있으면 링크하고, 없으면 "Anima는 수치 근거로 기본값을 옮겼다"로 문장을 정리한다.

---

## 5.3 torch.compile

### 블록 단위 compile, `forward`가 아니라 `_forward`

- `compile_blocks()`가 유일한 compile 경로다. 각 블록의 `_forward`를 `torch.compile(..., dynamic=False)`로
  감싼다. DiT 전체 compile은 하지 않는다(`for_compile.md` §2.5, §4.1).
- `forward`가 아니라 `_forward`인 이유: gradient checkpointing wrapper(`unsloth_checkpoint`)가
  `@torch._disable_dynamo`라서, `forward`를 compile하면 곧바로 graph break가 나고 쓸모 있는 건 하나도
  compile되지 않은 채 shape guard만 검사하며 recompile 폭주가 난다.
- Accelerate의 `dynamo_backend`는 항상 `"NO"`다. 위에 또 compile을 얹으면 이중 compile이 된다
  (`for_compile.md` §4.2).

### free-fit 버킷과 `compile_dynamic_seq`는 한 묶음

- 이미지는 원래 비율을 유지하고, 토큰 수가 tier별 band 안 어디든 떨어진다(`EDGE_TOKEN_BANDS`).
  (`.claude/skills/bucketing/SKILL.md`)

  | edge | 토큰 family |
  |---|---|
  | 512 | 1008, 1024 |
  | 768 | 2160 |
  | 896 | 3000, 3024 |
  | 1024 | 4032, 4200 |
  | 1280 | 6300 |
  | 1536 | 8640 |

- band 안의 토큰 수마다 static graph를 만들면 폭발하므로, free-fit은 `compile_dynamic_seq`를 요구한다.
  `train.py`는 `torch_compile`이 켜지면 자동으로 켜고, turbo 같은 전용 루프는
  `ensure_dynamic_seq_for_freefit`으로 강제한다. seq 축만 dynamic으로 표시하고 tier의 `seq_range`로
  묶어서 **tier당 graph 하나**가 된다(`bucketing` skill).
- `compile_blocks`가 켜는 native flatten은 patch grid를 `(B, 1, seq_len, 1, D)`로 펴서 graph가 H, W가
  아니라 토큰 수로만 key된다. eager 5D 경로와 bit-exact다(`for_compile.md` §2.5).
- 어떤 버킷이 존재하는지는 **디스크의 캐시가 정한다**. `--target_res`는 preprocess 전용이고 학습에는
  필요 없다. tier를 바꿨다면 `make preprocess-reconcile`(dry-run, `ARGS="--delete"`로 실행)로 고아 캐시를
  지운다(`bucketing` skill).
- 마지막 불완전 batch는 버린다(batch 축을 고정해 epoch 경계 recompile을 막음). `sample_ratio < 1.0`이면
  예외다(`for_compile.md` §3.2).

### recompile 한도는 "올린 줄 알았는데 안 올라간" 상태가 기본이다

- 한도는 `recompile_limit = max(현재값, 2n + 8)`이다. n은 실제로 채워진 버킷에서
  `library/anima/trainer/data.py::derive_token_budget`이 계산한 토큰 family 수(sample prompt 해상도 포함).
  `2*`는 forward와 backward가 같은 `_forward` bytecode를 공유해서, `+8`은 requires_grad / stride
  specialization 몫이다(`for_compile.md` §2.5).
- **함정:** dynamo config는 `ContextVar` 기반이라, 그냥 `config.recompile_limit = N`을 하면 그 스레드에서만
  적용된다. grad가 걸린 `_forward`는 다른 context(AOTAutograd / backward compile)에서 compile되므로 기본값
  8로 돌아가고, 첫 grad forward에서 eager로 떨어진다. `library/runtime/dynamo.py::pin_dynamo_limit`은
  entry의 `.default`까지 고정해서 이를 막는다. inductor flag도 같은 이유로 `pin_inductor_flag`를 쓴다
  (`for_compile.md` §2.6).
- turbo 루프는 한 스텝에 여러 해상도 / 여러 view의 forward를 돌리므로 별도 knob
  `dynamo_recompile_limit`(블록 `_forward`당, 기본 64)을 쓴다(`scripts/distill_turbo/config.py`).

### compile 캐시 오염

- 영구 compile 캐시는 FX graph로 key되지만 `mark_dynamic`의 값 범위는 key에 없다. 범위가 다른 두
  프로세스(예: 좁은 범위로 compile한 추론과 넓은 범위의 학습)가 같은 캐시 디렉터리를 쓰면 저장된 guard가
  재생되며 `ConstraintViolationError`가 "가끔" 난다. `library/runtime/dynamo.py::isolate_compile_cache`가
  signature별로 `TORCHINDUCTOR_CACHE_DIR`을 나눈다.
- torch 2.12의 inductor `mix_order_reduction`은 4096 경계에서 guard를 기록해 4096을 걸치는 dynamic 범위와
  충돌한다. adaLN LoRA처럼 broadcast로 소비되는 Linear에 LoRA가 붙어 backward에 seq 축 reduction이 생기면
  터진다. `compile_blocks`가 해당 band일 때 끈다(`for_compile.md` §2.6).

### compile 설정은 기본값이 맞다

2026-09-09, 기본 `make lora`(70장, 210 step) 실측(`docs/anima/findings/torch214_no_win_on_sm120.md`):

| 설정 | s/it (warm) | 첫 스텝 cold / warm 캐시 | peak |
|---|---|---|---|
| torch 2.12 default | 0.588 | 17.1 s / 9.6 s | 13.9 GB |
| torch 2.14 default | 0.584 (−0.7%) | 19.4 s / 10.0 s | 13.9 GB |
| 2.12 `max-autotune` | 평균 1.032, 중앙 0.556, p90 2.77 s | 97 s | 15.1 GB |
| 2.12 + `combo_kernels=True` | 0.597 (+1.5%) | 16.9 s | 13.9 GB |

- `max-autotune`은 run 중간에 다시 autotune해서 평균이 1.7배 나빠지고 VRAM이 1.2 GB 늘어난다.
- `combo_kernels`는 2.12 / 2.14 모두 기본 off이고, 켜면 1.5–2.4% 느리다.
- torch 2.14의 NVGEMM은 sm_120(RTX 50)에서 `cudaErrorNoKernelImageForDevice`로 전부 실패한다.
  `cutlass.operators` 0.2.0이 sm100 계열 커널만 가지고 있다. 릴리스 노트의 "Blackwell"은 B200이지
  RTX 50이 아니다. 이런 "auto" backend에 GPU 시간을 쓰기 전에 `TORCH_LOGS=autotuning` GEMM smoke 10분으로
  확인한다.
- 2026-10-09에 기본 torch는 위생 목적으로 2.14로 올라갔다. 속도 이득은 아니다(같은 문서 Update).

### 추론에서도 compile한다

`CLAUDE.md`는 DiT를 이미지 몇 장 이상 돌리는 모든 작업(배치 렌더, `make gen`, A/B sweep)에서 compile을
켜라고 한다. 그런데 `make test` / `make gen`의 기본 argv(`scripts/tasks/_common.py::INFERENCE_BASE`)에는
compile 플래그가 없다. 배치에서는 직접 붙인다.

```bash
make gen ARGS="--compile_dynamic_seq --lora_weight output/ckpt/foo.safetensors --from_file prompts.txt"
```

`--compile_dynamic_seq`는 `--compile_blocks`를 포함하고, 범위는 run의 모든 크기(각 `--from_file` 줄)의
최소–최대 토큰 수, 없으면 1024 tier band다(`library/inference/args.py`). 한 장짜리 대화형 렌더만 예외다.

---

## 5.4 VRAM과 속도

### 레버를 쓰는 순서

16 GB에서 OOM이 나면 아래 순서로 내린다. 앞의 것일수록 속도 손해가 적다.

1. **`activation_memory_budget`** (기본 0.99). 안 맞으면 더 내린다.
2. **int8 base** (`--int8 stream`). 3장 참조.
3. **block swap** (`--blocks_to_swap N`).
4. **gradient checkpointing** (`--preset low_vram`).

### 1. `activation_memory_budget`

- AOT autograd의 min-cut partitioner가 backward용으로 무엇을 저장할지 정한다. 1.0 미만이면 knapsack으로 싼
  중간값을 backward에서 재계산한다. `train.py`가 `compile_blocks` 직전에
  `torch._functorch.config.activation_memory_budget`을 설정한다
  (`docs/anima/findings/custom_autograd_removal_partitioner_oom.md`).
- 기본값은 0.99다(`configs/base.toml`, turbo도 동일). free-fit 워크로드에서 1.0은 OOM, 0.99는 약 13.1 GB로
  학습된다(같은 문서, 2026-07-04 갱신).
- int8 `stream` 기준 budget sweep(`docs/anima/optimizations/int8.md`):

  | budget | ms/step | peak |
  |---|---|---|
  | 0.99 | 427 | 10.87 GB |
  | 0.4 | 434–439 | 10.00 GB |
  | 0.3 | 472 | 6.87 GB |
  | 0.1 | 511 | 5.32 GB |

  "16 GB 아래에서는 budget을 먼저 내려라. block swap보다 peak를 더 많이 깎는다"(같은 문서). 다만 0.99
  미만으로 끝까지 학습한 LoRA는 아직 없다.
- gradient checkpointing이 켜져 있으면 budget은 자동으로 건너뛴다. checkpoint의 재계산 pass가 다른 compiled
  graph를 골라 `CheckpointError`가 날 수 있고, 어차피 중복이다.

**배운 경위.** 2026-06-10, 수치상 아무 영향이 없다고 검증된 LoRA down-projection `autograd.Function`을
지웠더니 다음 no-grad-ckpt 학습이 step 0에서 OOM이 났다. custom Function은 `save_for_backward`로 저장 집합을
고정하고 있었고, 지우자 partitioner가 블록마다 약 0.8 GB를 더 저장했다. config 수준의 추측 다섯 개가 모두
빗나갔고, 인접 commit 두 개에서 26-step 실제 job을 돌린 bisect가 몇 분 만에 원인을 찾았다. 교훈:
**수치적으로 무해하다고 메모리도 무해한 것은 아니다.** compile 경로에서 `autograd.Function` / checkpoint
wrapper를 지울 때는 출력뿐 아니라 저장 텐서 크기(`torch.cuda.max_memory_allocated`)를 비교한다
(`custom_autograd_removal_partitioner_oom.md`).

### 2. int8 base (3장으로)

block GEMM을 int8로 돌리면 bf16 593 ms → int8 `full` 433 ms(1.37배), 최종 `avr_loss`는 0.0737로 같다
(`ama mitsuki`, 848 step, `int8.md` § Measurements). `stream`은 `Wᵀ` 사본을 host에 두어 `full`보다
peak가 1.4 GB 낮다(12.38 → 10.96 GB, `int8.md`). 모드 선택, GPTQ 파일, bake한 DiT의 전용 GPTQ는 3장이
다룬다.

### 3. Block swap

- `--blocks_to_swap`은 모든 int8 모드에서 동작하고, swap한 stack의 출력과 `dX`가 resident stack과 bit 단위로
  같다(`test_swapped_step_is_the_resident_step`, GPU 테스트라 `ANIMA_TEST_GPU=1` 없이는 skip;
  `int8.md` § Block swap).
- LoRA, int8 `stream`, fp32 stream, budget 0.99, 106 step(`int8.md`):

  | swap | median | peak |
  |---|---|---|
  | 0 | 451 ms | 11.47 GB |
  | 14 | 475 ms | 10.44 GB |
  | 24 | 524 ms | 9.70 GB |

- **turbo에서는 비싸다.** 매 forward가 swap을 되감기(`rewind_block_swap`) 때문에 한 스텝에 swap된 블록이
  여러 번 이동한다. 30 step, CDM on, GAN off: swap 0에서 5.8 s/step, 12.36 GB, swap 14에서 8.4 s/step,
  11.32 GB(`int8.md`). 약 1.5배 시간으로 1 GB를 산다. 그래서 turbo 기본 config는 swap을 끈다
  (`docs/anima/methods/turbo.md` § Limitations).
- swap을 켜면 같은 seed라도 다른 샘플을 뽑는다. swap / resident run의 loss를 step 단위로 비교하지 않는다
  (`loss/epoch_average` 0.0749 vs 0.0733; `int8.md`).
- `inference.py`에는 어떤 base에서도 block swap이 없다. 생성은 전부 resident다(`int8.md` § Limits).
- **preset이 base를 덮는다.** 2026-06-10 당시 `base.toml`의 `blocks_to_swap=24`는 `presets.toml[default]`의
  `blocks_to_swap=0`에 덮여 아무 효과가 없었다(`custom_autograd_removal_partitioner_oom.md`). 체인은
  `base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI`이고 method가 preset을 이긴다.
  의심되면 `make print-config METHOD=… PRESET=…`로 병합 결과를 본다(`CLAUDE.md`).

### 4. Gradient checkpointing

- `--preset low_vram` = `gradient_checkpointing` + `unsloth_offload_checkpointing`
  (`configs/presets.toml`). 2026-06-10 워크로드에서 grad ckpt는 step time을 약 18% 늘렸다
  (`custom_autograd_removal_partitioner_oom.md`).
- `blocks_to_swap`과 `unsloth_offload_checkpointing`은 같이 못 쓴다(hard assert,
  `library/anima/trainer/data.py`). 두 offload 레버는 쌓이지 않는다.
- turbo는 CDM과 GAN 경로에 자체 `grad_ckpt`가 있다(`configs/methods/turbo.toml`의 `[cdm]`, `[gan]`).
  checkpoint 재계산은 `.backward()` 시점에 일어나는데 LoRA view(`set_view`)는 전역 상태이므로, 각
  checkpointed forward는 자기 view가 살아 있을 때 backward해야 한다. 어긋나도 metric은 아무것도 잡지
  못한다(`turbo.md` § GAN).

### 기본 학습과 turbo의 실측 peak

| 작업 | 설정 | 속도 | peak | 출처 |
|---|---|---|---|---|
| `make lora` (default preset) | bf16 base, torch 2.12, 2026-09 | 0.588 s/it | 13.9 GB | `torch214_no_win_on_sm120.md` |
| LoRA | int8 `stream`, fp32 stream, swap 0 | 451 ms | 11.47 GB | `int8.md` |
| `make turbo` | int8 `stream`, fp32 stream, warm start | 6.7–7.0 s/step | 12.39 GB | `int8.md` § Limits |
| turbo 2-step smoke | int8 GPTQ, fp32 stream | – | 12.35 GB | `project/anima_turbo/reports/mfd.md` |
| turbo 4-step arm 0 | merged14 teacher, 250 it | 5.7–5.8 s/it, 약 27분 | – | `project/anima_turbo/reports/arm0.md` |

> TODO(확인 필요): 요청서에 있던 "turbo peak ~13.5 GB"의 출처를 찾지 못했다. 문서에 있는 turbo peak는
> 12.35–12.39 GB(swap 0)이고, GAN on / swap 14에서 11.35 GB다. GAN on, swap 0의 peak 실측이 있으면
> 여기에 추가한다.

### 옵티마이저: fused AdamW

- 기본은 `optimizer_type = "AdamW"`이고, CUDA에서는 `get_optimizer`가 `fused=True`를 자동으로 넣는다
  (`library/training/optimizers.py`).
- bitsandbytes `AdamW8bit`는 파라미터 블록마다 커널을 하나씩 띄워 optimizer 구간에서 GPU 사용률이 88%로
  떨어졌다. 한 스텝 1.17 s 중 308 ms가 optimizer였고, fused AdamW로 바꾸자 0.86 s가 됐다(RTX 5060 Ti,
  당시 static-pad 설정). 대가는 optimizer state +180 MB 정도(rank 32 LoRA, 약 30M 파라미터)
  (`docs/anima/optimizations/adamw_fused.md`).
- `optimizer_args`는 `ast.literal_eval`로 파싱하므로 `"fused=True"`이지 `"fused=true"`가 아니다.

### RTX 50용 FlashAttention 빌드

```bash
uv sync --no-default-groups --group fa2-sm120
```

- sm_120 전용 패치된 FA2 2.8.3. Anima 기준 fp16-accumulate backward가 step의 5.1%, fp16 input(`f16in`)이
  3.1%를 줄인다. peak VRAM은 같다(`docs/moss/optimization/fa2_sm120.md`).
- head dim 64 / 128만 지원한다. Anima `llm_adapter`의 64는 stock backward로 간다.
- **wheel이 빠지면 조용하다.** `attention_dispatch.py`가 ImportError를 SDPA로 돌려서, torch를 wheel의 ABI
  범위 밖으로 올리면 에러 없이 느려진다. 플래그 없는 `uv sync`는 stock pin으로 되돌리므로, 업데이트 때는
  `ANIMA_TORCH=2.14-sm120 make update`를 쓴다(같은 문서).
- venv 전체에 적용되므로 같은 venv의 ComfyUI도 이 빌드를 쓴다.

### fp16은 쓰지 않는다 (아직)

같은 카드에서 fp32 누적 fp16 GEMM은 bf16과 속도가 같고(약 102 TFLOPS), fp16 누적만 1.6배다. 그래도 int8
`_int_mm`(약 330 TFLOPS)보다 느리고, backward는 loss scale 2^20 근처가 필요하다. 출하된 fp16 계획은 없다
(`docs/anima/findings/fp16_headroom.md`).

### RAM과 디스크

- RAM 64 GB에서 사용자는 사용률 약 90%를 넘는 run을 죽인다(`memory/project_box_environment.md`). int8
  `stream`은 `Wᵀ` 사본을, block swap은 블록을 host에 두므로 RAM을 더 쓴다.
- run 데이터와 캐시는 루트가 아닌 별도 디스크에 둔다. CJK run 하나의 캐시가 12–55 GB(TE 23 GB, latent
  26 GB)였고, 루트 디스크가 가득 차 latent encode가 SIGBUS로 죽은 적이 있다(같은 메모).

---

## 5.5 긴 작업은 daemon으로

### 왜 daemon인가

- 학습 중인 job 뒤에 줄을 서서 OOM 충돌을 피하고, 터미널을 닫아도 살아남는다(`.claude/skills/daemon/SKILL.md`).
- Claude Code 같은 에이전트가 GPU 작업을 돌릴 때는 필수다. 에이전트의 background Bash에서 띄운 GPU
  프로세스는 sandbox에 의해 약 1분 뒤 흔적 없이 SIGKILL된다(2026-07-25 관측, 같은 skill).

### 기본 루프

```bash
make daemon-run ARGS="--queue --stall-timeout 900 bench/foo/run_bench.py --label x"  # 제출, job id 출력
make daemon-wait JOB=<id>        # 끝날 때까지 대기, job의 exit code로 종료
make run-status                  # step N/total, it/s, ETA, 최근 loss, 최근 ckpt
make daemon-log JOB=<id> ARGS="-n 200"
make lora ARGS="--queue"         # 학습 target도 같은 --queue / --inline
make turbo ARGS="--config project/anima_turbo/configs/<id>.toml --queue"
```

| 상황 | 명령 | 비고 |
|---|---|---|
| 지금 뭐가 도는가 | `make daemon-status` | 미완료 job만 보인다. 끝난 job은 `make daemon-jobs ARGS="--all"` |
| 잠깐 GPU 연산을 멈춤 | `make daemon-pause` | SIGSTOP. VRAM은 그대로, 큐는 진행하지 않음 |
| GPU를 비워야 함 | `make daemon-pause RELEASE=1` | `train.py` job만. 다음 optimizer step에서 저장 후 종료, 큐 진행. `make daemon-resume`이 `--resume`으로 재시작(재로드 + 재compile) |
| 중단 | `make daemon-kill JOB=<id>` | daemon은 살아 있고 다음 job 시작 |
| 모든 scalar | `make export-logs RUN=output/logs/<run> SUMMARY=1` | tag별 최대 step + 마지막 값 |

(`tasks.py`, `anima_daemon/README.md`)

### 함정

- **stall watchdog.** command job은 `stdout.log`와 `progress.jsonl`이 120초 동안 둘 다 멈추면 죽는다.
  compile이나 캐시 생성처럼 조용한 구간이 긴 job은 `--stall-timeout S`(0 = 끔)를 준다. daemon-run 자신의
  플래그(`--label`, `--stall-timeout`)는 **스크립트 경로 앞에** 둔다. 뒤에 두면 자식에게 넘어간다
  (`anima_daemon/README.md`, daemon skill).
- **환경변수는 daemon이 뜬 shell에서 온다.** submit shell이 설정하지 않은 whitelist 변수는 며칠 전 daemon을
  띄운 shell의 값을 쓴다. 예: daemon이 `ANIMA_VOCAB_PACK=<preview pack>`으로 떴다면, 그 변수 없이 제출한
  job은 preview pack으로 학습한다. 의존하는 환경 레버는 submit shell에 모두 설정하고 job의 `captured_env`나
  로그(vocab pack은 sha를 남김)로 확인한다. 아무것도 안 돌 때(`make daemon-jobs ARGS="--state
  queued,running,paused"`가 `0 of M`) `make daemon-terminate`가 값싼 리셋이다(daemon skill).
- `make run-status`는 `progress.jsonl`을 쓰는 루프(train.py, `make turbo`)만 본다. 전용 루프가 이를 쓰지
  않으면 `make daemon-log`에서 step을 읽는다.

### run 이름과 config

- run 이름은 제출 시각 `mmdd_hhmm`(예: `1111_1012`)이다. CLI가 아니라 run의 config 파일에 `output_name`으로
  쓰고, 무엇을 하는 run인지는 config의 머리 주석과 필드에 적는다. 설정을 이름에 이어 붙이지 않는다. 2-step
  turbo 이름이 `..._mfd_k6_ganlast_nocdm_500_tafake_const_ema10`까지 길어진 뒤 정한 규칙이다
  (`memory/feedback_experiments.md`). 예: `project/anima_turbo/configs/1112_0833.toml`(arm 0 control),
  `1112_0834.toml`(seed만 다른 null twin).
- 모든 학습은 해석된 설정을 `.snapshot.toml`로 남긴다. turbo snapshot의 `# source config:` 줄을 꼭 본다.
  GUI / daemon run은 `configs/gui-methods/custom/turbo.toml`을 읽기 때문이다. 그리고 turbo TOML에서 key를
  잘못된 section에 두면 경고 없이 기본값이 쓰인다. arm마다 `ss_turbo_*` metadata나 snapshot으로 확인한다
  (`docs/anima/methods/turbo.md` § Loop gotchas, 4장).

---

## 5.6 추론 쪽

### `make test`와 `inference.py`의 기본값은 다르다

| | `make test` (`INFERENCE_BASE`) | `inference.py` 단독 |
|---|---|---|
| steps | 28 | 50 |
| CFG (`--guidance_scale`) | 4.0 | 3.5 |
| `--flow_shift` | 3.0 | 3.0 |
| sampler | euler | euler |
| 해상도 | 1024 × 1024 | 1024 × 1024 |
| stream | fp32 | fp32 |

(`scripts/tasks/_common.py::INFERENCE_BASE`, `library/inference/args.py`.) 스크립트에서 `inference.py`를
직접 부를 때는 step과 CFG를 명시한다.

- `make test`는 대화형 1장, `make gen`은 같은 argv를 daemon job으로 돌려 학습 뒤에 줄을 서고 결과 manifest를
  남긴다. 배치는 `make gen`(+ 5.3절의 compile 플래그).
- 환경 레버는 모든 `test-*`에 조합된다: `NOLORA=1`, `SPECTRUM=1`, `MOD=1`, `INT8=1`, `STREAM=bf16`,
  `DAVE=1`, `FSG=1`, `XATTN_BOOST=2`(`scripts/tasks/inference.py::_base_test_args`). `SPECTRUM=1`과 `SPD=1`은
  같이 못 쓴다.
- `INT8=1`이면 LoRA가 정적 merge가 아니라 P-GRAFT hook으로 붙는다(merge하면 버려질 bf16 가중치에 접히므로).
  bf16 base를 int8 base로 바꾸면 생성 픽셀이 약 4/255 움직인다(`int8.md`).

### 학습 없는 추론 스택 (요약)

자세한 것은 `docs/anima/inference/README.md`.

| 기능 | 켜는 법 | 한 줄 |
|---|---|---|
| Spectrum | `make test SPECTRUM=1` | Chebyshev feature 예측으로 캐시된 step에서 28 블록을 모두 건너뛴다 |
| CNS | `--sampler er_sde --cns auto` | SDE noise를 아직 해상되지 않은 주파수 대역으로 재분배. **er_sde 전용**, euler에서는 no-op |
| mod-guidance | `make test MOD=1` | 학습된 `pooled_text_proj`로 AdaLN을 조절. 전역 톤 레버이지 내용 레버가 아님 |
| DAVE | `make test DAVE=1` | 같은 프롬프트의 seed 간 다양성 회복. 블록 hook이라 `compile_blocks`와 공존 |

### Turbo LoRA 추론

- `--infer_steps 4 --guidance_scale 1.0`(`make test-turbo`가 euler, 4 step, CFG 1.0으로 고정;
  `scripts/tasks/inference.py::cmd_test_turbo`). CFG는 증류 때 student에 구워졌으므로 1.0을 유지한다.
  두 번 guide하지 않는다.
- `flow_shift` 3.0은 turbo의 `[sampling] flow_shift = 3.0`과 맞는다. 하나를 바꾸면 다른 것도 바꾼다
  (`configs/methods/turbo.toml`, `turbo.md`).
- **Spectrum과 쌓지 않는다.** Spectrum의 Chebyshev 캐시는 ≥16 step을 가정한다. CNS도 er_sde 전용이라 turbo의
  euler 4-step에는 해당이 없다. mod-guidance는 turbo student에서 검증되지 않았다(`turbo.md` § Limitations).
- checkpoint가 학습한 step 수보다 더 많은 step에서 더 좋아 보이면, 아직 진짜 N-step 맵에 도달하지 못했다는
  신호다(`turbo.md` § Inference).
- 공개된 4-step student는 `huggingface.co/sorryhyun/anima-turbo-4step`. 증류 자체는 4장.

### ComfyUI

노드 대부분은 독립 repo에 있고 `../comfy/custom_nodes/`로 symlink된다. symlink가 아니라 원본 repo를
고친다. `_vendor/` 부분집합은 손으로 `cp`하지 말고 `make vendor-sync`로 다시 만들고, 노드 publish 전마다
돌린다(`.claude/skills/custom-nodes/SKILL.md`).

- `Anima Adapter Loader`의 `stream` 입력과 `Anima fp32 Stream` 노드가 fp32 stream을, `Anima INT8 Loader`가
  int8 파일을 담당한다(`fp32_stream.md`). ComfyUI 쪽 fp32 stream의 오차는 `library`와 같은 수준이다(5.2절 표).
- ComfyUI의 int8 kernel(`comfy_kitchen` 0.2.40/0.2.41)은 bf16 입력에서 activation 나눗셈을 bf16으로 반올림한다
  (comfy-kitchen#289, 미해결). Adapter 노드가 자체 우회를 갖고 있다(`memory/project_comfy_kitchen_fp32_division.md`).
- **Block Compile 노드는 DiT를 새로 만든다.** 그보다 앞에 설치한 forward hook은 죽은 모듈에 남아 조용히
  no-op이 된다. patch 노드는 Block Compile **뒤에** 둔다(`memory/project_comfy_node_gotchas.md`).
- W8A8 int8 모델에서는 `next(parameters()).dtype`이 int8을 돌려준다. 첫 번째 floating 파라미터를 고른다
  (같은 메모).

---

## 5.7 실험 위생

### 판정 기준은 결과를 보기 전에 정한다

- gate는 근거가 통과시킨다. 모든 지표와 렌더가 한 arm을 가리키면 차이가 작아도 통과다. 기준은 결과를 읽기
  전에 정하고, 구조적으로 작을 수밖에 없는 양(낮은 σ에서 one-step x̂₀ 차이처럼 x_t가 이미 이미지를 담고
  있는 경우)에는 threshold를 걸지 않는다(`CLAUDE.md` "Papers and gates").
- 예: turbo A/B 프로토콜은 "arm의 EMA 렌더가 두 prompt 모두에서 control–twin 차이보다 크게 이기고, S–T4
  nW2²가 |control − twin|보다 크게 낮으면 통과"를 run 전에 적어 두었다(`project/anima_turbo/plan.md`
  § A/B protocol). gradient norm 쪽 결정 규칙(pooled top/bottom ≤ 1.5 → adaptive clip alone)도 미리 고정됐고,
  arm 0 결과(1.43)는 그 규칙대로 읽혔다(`project/anima_turbo/reports/arm0.md`).

### null twin으로 noise floor를 먼저 잰다

- raw checkpoint 하나와 다른 하나를 비교하면 대부분 run noise를 읽는다(`plan.md`). control을 seed만 바꿔 한 번
  더 돌린 twin이 이후 모든 arm의 noise floor가 된다.
- arm 0 실측: seed만 다른 두 run의 ΔW가 공유하는 비율은 cos 0.09–0.10이다. 목적함수나 LR이 다른 쌍도
  0.09–0.11이었다. 즉 **ΔW overlap으로는 arm을 구분할 수 없다.** 판정은 렌더와 `moment_gap`으로 한다
  (`arm0.md`).
- 같은 run 안에서도 |ΔW|가 √t로 자라고 125-step 이동끼리 직교한다(cos 0.018 / 0.028). 상수 LR에서
  random walk다. EMA(window 0.2)가 배포 대상인 이유다(`arm0.md`, `plan.md`).
- **raw checkpoint는 발산 burst 안에 있을 수 있다.** control의 raw 250은 judge 12장 전부 평평한 noise
  texture였다. 마지막 4 step에서 `gn_student`가 231 / 244 / 106 / 72로 튀었고, 이 recipe에서 burst는 약
  250 iteration에 한 번 나온다. EMA 250은 정상 렌더였다(`arm0.md`).
- twin은 한 번만 돌린다. 통과한 arm은 새 control이 되고, step time이나 loss 항이 바뀌지 않는 한 새 twin을
  만들지 않는다(`plan.md`).

### 렌더가 gate다. scalar는 아니다

- 첫 checkpoint에서 grid를 렌더한다. APEX turbo arm은 loss, `x_pred` std, rollout std가 모두 평평하고
  건강했는데 4-step grid는 warm start보다 흐려지고 있었다. 흐린 샘플도 latent std는 정상이다
  (`project/anima_turbo/findings/turbo_apex_no_quality.md`).
- turbo의 `fm_mse`는 품질과 역상관이다. checkpoint 순위는 렌더한 4-step 샘플로 매긴다. pose 다양성은
  grid에서만 보이고, PE-pooled 지표는 pose를 못 본다(`turbo.md`).
- 판정용 prompt는 쉬우면 안 된다. ANIMA 표지판, nozomi prompt는 arm을 가르기에 너무 쉬워서 turbo는
  `bench/turbo/judge_hard.txt`(이름 있는 다섯 명 + position clause, 일본어 메뉴 + 영어 간판 카페, 아래에서
  본 점프 구도)를 768 × 1344에서 teacher 옆에 놓고 본다(`plan.md`).

### 작은 것부터, 한 번만

- 메커니즘 질문은 full run 전에 약 25분짜리 micro arm으로 답한다. 이전 세션에서 끝난 측정은 다시 재지 않고
  그 리포트 위에 쌓는다. job을 큐에 넣기 전에 "이 결과가 어떤 결정을 바꾸는가"를 말할 수 있어야 한다.
  없으면 코드를 읽는다(`memory/feedback_experiments.md`).
- 기존 checkpoint나 렌더가 control이다. code drift나 recipe 변경으로 불공정하면 결과 옆에 적고, 새 control은
  treatment 뒤에 follow-up으로 제안한다(같은 메모).
- 원인을 모를 때는 config 추측보다 실제 job의 bisect가 빠르다. 5.4절의 partitioner OOM은 추측 다섯 개가
  빗나가고, 인접 commit 두 개의 26-step run이 몇 분 만에 끝냈다.
- GPU job이 도는 동안 CPU를 많이 쓰는 부수 작업을 하지 않는다(같은 메모).

### 논문은 PDF로 읽는다

웹 요약은 표, 식, 부록을 버린다. arXiv는 `https://arxiv.org/pdf/<id>`를 받아 `pdftotext -layout`으로
읽는다(`CLAUDE.md`). 예: "MFD는 1-step용"이라는 인상과 달리, PDF의 T2I 결과는 SANA 4-step, 3000 step이었다
(`memory/project_soup_turbo.md`).

### bench는 envelope로 남긴다

수치 / 효율 변경(Tier 1.5)과 새 method(Tier 2)는 bench 스크립트와 invariant test가 필요하다. bench는
`bench/_common.py`를 공유하고 `bench/<name>/results/<YYYYMMDD-HHMM>[-label]/`에 `result.json`을 쓴다
(`CLAUDE.md`, `CONTRIBUTING.md`). pytest는 기본으로 GPU를 숨기므로(`tests/conftest.py`) 초록불이어도
GPU 테스트는 skip됐을 수 있다. `skipped` 수를 읽고, 필요하면
`ANIMA_TEST_GPU=1 make daemon-run ARGS="--stall-timeout 300 -m pytest tests/<file>.py -q -k <name>"`로 돌린다.

---

## 5.8 체크리스트

학습 전:

- [ ] TE 캐시가 현재 caption / vocab pack과 맞는가(pack 변경 시 `make preprocess-te ARGS=--overwrite`)
- [ ] `make print-config METHOD=… PRESET=…`로 병합된 설정 확인(preset이 base를 덮는다)
- [ ] `torch_compile = true`, `compile_dynamic_seq = true`, `activation_memory_budget = 0.99`, `stream = "fp32"`
- [ ] config에 `output_name = "mmdd_hhmm"`와 머리 주석
- [ ] `--queue`로 제출, `make daemon-wait`로 대기, 조용한 구간이 길면 `--stall-timeout`

학습 중 / 후:

- [ ] `make run-status`로 진행 확인, 첫 checkpoint에서 grid 렌더
- [ ] raw가 아니라 EMA로 판정(turbo), twin 대비 차이로 판정
- [ ] `.snapshot.toml`의 실제 값(특히 turbo의 `# source config:`)과 safetensors metadata 확인

추론:

- [ ] 배치는 `make gen` + `--compile_dynamic_seq`
- [ ] turbo는 euler, 4 step, CFG 1.0, Spectrum 없음
- [ ] LoRA 깊이(`ss_num_blocks`)가 DiT 깊이와 맞는가
