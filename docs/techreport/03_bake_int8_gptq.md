# 3. LoRA를 DiT에 bake하고, int8 base와 teacher 전용 GPTQ 만들기

**요약**

- soup LoRA(`output/ckpt/anima_merged14.safetensors`)를 multiplier 1.0으로 base DiT에 구워 teacher DiT
  `models/diffusion_models/anima_merged14.safetensors`(bf16, 체크포인트 레이아웃)를 만듭니다. 13초 걸리는 작업입니다.
- 터보는 이 teacher를 int8 base(`int8 = "stream"`) 위에서 돌립니다. 블록당 큰 linear 6개(28블록에서 168개)를 ConvRot
  int8 GEMM으로 바꾸고, 나머지는 float으로 둡니다.
- int8 forward copy는 GPTQ 파일에서 읽습니다. base용 GPTQ 파일은 `base_fingerprint`가 달라서 merged DiT에서는
  거부됩니다. 그래서 teacher에는 **teacher 전용** GPTQ 파일(`anima_merged14_int8_gptq.safetensors`, 2.37 GiB, 약 17분)을
  따로 만듭니다.
- 순서는 bake → GPTQ → 터보 config의 `dit_path`(GPTQ 파일)와 `dit_nonquant_path`(bf16 teacher)입니다.

---

## 3.1 파이프라인에서 이 단계의 위치

앞 단계([02_soup_merge.md](02_soup_merge.md))의 결과물은 **LoRA 파일**입니다. 8개 shard를 `concat_exact`로 합친 rank 256
LoRA이고, adaLN은 평균으로 합쳤습니다(`project/anima_turbo/README.md`). 터보 증류([04_turbo_distillation.md](04_turbo_distillation.md))는
이 soup이 반영된 모델을 teacher로 씁니다. 그 teacher를 int8 base 위에서 돌리려면 파일 세 개가 필요합니다.

| 파일 | 만드는 도구 | 내용 | 용도 |
|---|---|---|---|
| `output/ckpt/anima_merged14.safetensors` | 02의 soup / `merge_loras.py` | soup LoRA | bake 입력, 배포 시 터보 LoRA와 exact concat |
| `models/diffusion_models/anima_merged14.safetensors` | `scripts/toolkits/merge_to_dit.py` | bf16 DiT, soup이 1.0으로 구워짐 | 터보의 `dit_nonquant_path` (float 가중치) |
| `models/diffusion_models/anima_merged14_int8_gptq.safetensors` (+ `.report.json`) | `project/finished/int8_train/src/dit_gptq.py` | 블록 linear는 int8, 나머지는 bf16인 DiT 전체 | 터보의 `dit_path` (int8 forward copy) |

LoRA 파일과 bake된 DiT의 파일명이 둘 다 `anima_merged14.safetensors`입니다. 디렉터리(`output/ckpt/` 와
`models/diffusion_models/`)로만 구분되니 경로를 끝까지 적어야 합니다.

---

## 3.2 Bake: LoRA를 DiT 가중치에 굽기

### 명령

`tasks.py`의 `merge` 타깃은 `scripts/tasks/utilities.py::cmd_merge`로 가고, 이 함수가
`scripts/toolkits/merge_to_dit.py --adapter_dir $ADAPTER_DIR --multiplier $MULTIPLIER <ARGS>`를 실행합니다.

```bash
# ADAPTER_DIR(기본 output/ckpt)에서 가장 최근 LoRA를 base에 굽기
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=1.0

# 파일을 지정하고 출력 위치를 정하기 (ARGS가 뒤에 붙으므로 --adapter가 --adapter_dir보다 우선)
make merge ARGS="--adapter output/ckpt/anima_merged14.safetensors \
  --out models/diffusion_models/anima_merged14.safetensors"
```

merged14 teacher는 daemon job `20261110-102956-89d8d6`으로 만들었습니다. job.json의 argv는 아래와 같고,
10:29:56에 시작해 13초 뒤 끝났습니다(`output/daemon/jobs/20261110-102956-89d8d6/`).

```bash
make daemon-run ARGS="scripts/toolkits/merge_to_dit.py \
  --adapter output/ckpt/anima_merged14.safetensors --multiplier 1.0 --dtype bf16 \
  --out models/diffusion_models/anima_merged14.safetensors"
```

`merge_to_dit.py` 플래그 (`scripts/toolkits/merge_to_dit.py`):

| 플래그 | 기본값 | 비고 |
|---|---|---|
| `--adapter_dir` | `output/ckpt` | `--adapter`가 없을 때 여기서 가장 최근 파일을 고릅니다 |
| `--adapter` | 없음 | 명시 경로. `--adapter_dir`보다 우선합니다 |
| `--dit` | `models/diffusion_models/anima-base-v1.0.safetensors` | base DiT. **bf16 원본을 넣어야 합니다** (아래 주의) |
| `--multiplier` | 1.0 | LoRA 강도 |
| `--out` | `<adapter-stem>_merged.safetensors` (어댑터 옆) | teacher는 `models/diffusion_models/`에 두는 것이 좋습니다 (3.7) |
| `--dtype` | `bf16` | `fp32` / `fp16` / `bf16` |
| `--device` | GPU가 있으면 `cuda` | merge 연산 장치 |
| `--allow-partial` | 꺼짐 | bake할 수 없는 키를 버리고 나머지만 굽습니다 |
| `--network_module` | `networks.lora_anima` | `create_network_from_weights`를 제공하는 모듈 |

### 내부 동작

`library/anima/merge.py::merge_adapter_into_dit` 기준입니다.

1. 어댑터 state dict에서 bake할 수 없는 키를 찾습니다(`scan_non_bakeable_keys`).
2. base DiT를 `load_anima_model(..., dit_weight_dtype=dtype)`으로 올립니다. forward를 돌리지 않으므로 `attn_mode="torch"`입니다.
3. `create_network_from_weights(multiplier, adapter, ..., for_inference=True)`로 네트워크를 만들고
   `network.merge_to(...)`를 호출합니다.
4. 모듈마다 `networks/lora_modules/lora.py::LoRAModule.merge_to`가 **fp32로** `W += multiplier · (up @ down) · scale`을 계산합니다.
   channel scaling의 `inv_scale`이 있으면 `down`에 먼저 곱해서 흡수를 되돌립니다. 결과는 `--dtype`으로 저장됩니다.
5. `library/anima/weights.py::save_anima_model`이 **체크포인트 레이아웃**으로 저장합니다. fused `qkv_proj` 등을
   `unfuse_dit_state_dict`로 다시 나누고, 키에 `net.` 접두사를 붙입니다. ComfyUI와 `load_anima_model`, `dit_gptq.py`가
   읽는 형식입니다. 메타데이터는 `ss_merged_from`, `ss_merge_multiplier`, `ss_base_dit`, `format=pt`입니다.

이 레이아웃은 커밋 `b06783cb`부터 적용됩니다. 그 전의 `save_anima_model`은 `dit.state_dict()`를 fused 이름 그대로 써서,
`make merge` 결과를 ComfyUI와 `dit_gptq.py`가 읽지 못했습니다(커밋 메시지, merged13 작업 세션).

### 결과물 확인 (merged14)

safetensors 헤더를 CPU로 읽은 결과입니다.

| | `anima-base-v1.0.safetensors` | `anima_merged14.safetensors` |
|---|---|---|
| 텐서 수 | 685 | 685 |
| dtype | BF16 685 | BF16 685 |
| 크기 | 4,182,218,328 B | 4,182,218,480 B (메타데이터만큼 차이) |
| 메타데이터 | 없음 | `ss_base_dit=anima-base-v1.0.safetensors`, `ss_merged_from=anima_merged14.safetensors`, `ss_merge_multiplier=1.0` |

키 이름과 shape가 base와 같으므로, bake된 DiT는 base를 받는 곳(ComfyUI, `inference.py --dit`, `dit_gptq.py --dit`,
터보의 `dit_nonquant_path`)이라면 어디서나 base 대신 쓸 수 있습니다.

### bake를 거부하는 경우

`library/anima/merge.py::_NON_BAKEABLE_MARKERS`에 등록된 키가 하나라도 있으면 `NonBakeableError`를 내고 종료 코드 2로 끝납니다.

| 키 조각 | 종류 | 이유 |
|---|---|---|
| `.lora_up_weight` | HydraLoRA stacked | 레이어 router가 입력마다 head를 고릅니다. 고정 가중치로 옮길 수 없습니다 |
| `.lora_ups.` | HydraLoRA split / step-expert 터보(step별 head) | 위와 같습니다 |
| `postfix_` | postfix | cross-attn KV splice라서 가중치 delta가 아닙니다 |
| `prefix_` | prefix | 위와 같습니다 |
| `register_tokens` | register tokens | self-attn 시퀀스에 붙는 토큰이라서 가중치 delta가 아닙니다 |

`--allow-partial`을 주면 경고만 남기고 나머지 LoRA 부분을 굽습니다. 이렇게 만든 DiT는 해당 컴포넌트를 재현하지
못합니다. `ADAPTER_DIR`에서 자동으로 고를 때(`pick_latest_adapter`)는 `*_moe.safetensors`, `*.bak.*`, 파일명에
`postfix` / `prefix`가 들어간 파일을 처음부터 후보에서 뺍니다. 지원 대상은 plain LoRA와 T-LoRA입니다. T-LoRA의
timestep 마스크는 학습에서만 쓰이므로 bake해도 결과가 같습니다(모듈 docstring).

### 주의

- **`make merge`는 daemon을 거치지 않습니다.** `cmd_merge`는 `run()`으로 바로 subprocess를 띄우고, `--device` 기본값이
  `cuda`입니다. 에이전트가 실행할 때는 위 예처럼 `make daemon-run`으로 넣어야 합니다(CLAUDE.md, "Running work").
- **GPTQ 파일을 `--dit`로 주면 안 됩니다.** `merge_adapter_into_dit`는 `load_anima_model`에 `lora_weights_list`도
  `dit_nonquant_path`도 넘기지 않습니다. 그래서 int8 체크포인트를 넣으면 거부하지 않고, "no dit_nonquant_path"
  경고만 남긴 채 int8에서 복원한 가중치(`dequantize_checkpoint_`) 위에 굽습니다. 복원한 가중치는 원래 가중치와
  1.1% 다릅니다(`docs/anima/optimizations/int8.md`). 3.8의 문서/코드 불일치 1번을 보세요.
- 기본 `--out`은 어댑터 옆(`output/ckpt/`)입니다. 다음 단계의 "DiT 옆 GPTQ 파일" 규칙과 맞추려면 `--out`으로
  `models/diffusion_models/`를 지정합니다(3.7).

---

## 3.3 int8 base

### 무엇이 int8로 바뀌는가

`library/anima/int8.py::CONVERT_SUFFIXES`, `convert_targets`:

| int8 (블록당 6개, 28블록에서 168개) | float으로 남는 것 |
|---|---|
| `self_attn.qkv_proj` (fused), `self_attn.output_proj` | cross-attn `k` / `v` (M이 context 길이) |
| `cross_attn.q_proj`, `cross_attn.output_proj` | adaLN (M이 batch) |
| `mlp.layer1`, `mlp.layer2` | attention, norm, residual stream, embedder, head, `llm_adapter` |
| | LoRA 자체의 GEMM |

cross `k/v`와 adaLN은 phase 0에서 int8이 오히려 느리게 측정되어 bf16로 둡니다(`library/anima/int8.py` 주석).
대상은 `blocks.N.` 아래의 `nn.Linear`뿐이고, `llm_adapter`의 linear는 바꾸지 않습니다.

### ConvRot 형식

`library/int8/convrot.py`, `library/int8/linear.py`:

- 가중치를 256열 그룹마다 정규화된 Hadamard 행렬로 회전한 뒤 int8로 저장합니다. 스케일은 출력 행마다 fp32 하나이고,
  값은 `amax / 127`입니다. 이 Hadamard는 대칭이고 자기 자신이 역행렬이라서, 같은 회전으로 입력을 가중치 기저로
  옮길 수 있습니다.
- 커널은 `comfy_kitchen.int8_linear(..., convrot=True, convrot_groupsize=256)`입니다. activation은 토큰마다 int8로
  양자화하고(W8A8), 출력은 bf16입니다.
- base 가중치의 gradient는 계산하지 않습니다. LoRA 학습에 필요한 것은 입력 gradient `dX = dY·W`뿐입니다.

### 모드

학습은 `--int8 off|fwd|full|stream`(`library/config/cli_args.py`, `configs/base.toml` 기본 `"off"`)이고,
생성은 `inference.py --int8` 불리언(`library/inference/args.py`)입니다. `make test-*`에서는 `INT8=1`이
`--int8`을 붙입니다(`scripts/tasks/inference.py`). 내부 모드는 `library/int8/linear.py::MODES = ("fwd", "full", "infer")`이고,
`stream`은 `full`로 변환한 뒤 `stream_blocks_`로 바꿉니다.

| 모드 | forward | 입력 gradient `dX` | bf16 base 가중치 | 쓰는 곳 |
|---|---|---|---|---|
| `off` | bf16 | bf16 | 유지 | 기본값 |
| `fwd` | int8 | bf16, 유지된 `W`로 계산 | 유지 | 학습 |
| `full` | int8 | int8, `Wᵀ`의 두 번째 int8 copy(`wq_t`) | 버림 | 학습 |
| `stream` | int8 | `full`과 같지만 `wq_t`는 pinned host 메모리에 두고, backward가 지나는 블록 것만 카드로 보냄 | 버림 | 학습, 터보 |
| `infer` | int8 | 없음 | 버림 | `inference.py --int8` |

몇 가지는 코드에서만 보입니다.

- **GPTQ가 다루는 것은 forward copy(`wq`)뿐입니다.** backward copy `wq_t`는 항상 round to nearest로 만듭니다
  (`convert_`, 모듈 docstring). `wq_t`를 만들려면 float 가중치가 필요합니다. int8 체크포인트를 `dit_path`로 쓸 때
  `dit_nonquant_path`가 필요한 이유가 이것입니다(3.5).
- `stream`은 `WqtStream`(`library/int8/stream.py`)으로 동작합니다. 모든 블록의 `wq_t`를 page-locked host 버퍼에 두고,
  카드에는 블록 크기 slot 두 개만 둡니다. 하나는 backward가 지금 지나는 블록용이고, 하나는 worker thread가 미리 올리는
  아래 블록용입니다.
- 학습에서 변환하는 시점은 `create_network` 다음, `apply_to` 전입니다. 그래서 `weight_svd` / `grad_svd` 초기화는 아직
  bf16 가중치를 읽고, LoRA 모듈은 int8 forward를 `org_forward`로 잡습니다. 그 뒤에 `compile_blocks`가 옵니다.
  터보 루프는 `TurboDMDNetwork`의 `before_apply`에서 같은 시점에 변환합니다(`library/anima/int8.py` docstring).
- 생성(`infer`)에서는 LoRA가 `--pgraft` 값과 상관없이 P-GRAFT hook으로 붙습니다. 정적으로 merge하면 변환 과정에서 버려지는
  bf16 가중치에 LoRA가 접히기 때문입니다. base는 `int8(x, W_q) + B·A·x`를 계산하는데, 이는 trainer가 학습한 함수와 같습니다.
- 학습한 LoRA에는 `ss_int8_base`(모드)와 `ss_int8_base_id`(forward copy 파일의 `int8_id`, RTN이면 `rtn`)가
  찍힙니다(`library/anima/trainer/trainer.py`, `scripts/distill_turbo/distill.py`). `convert_for_inference`는 이 값을
  지금 쓰는 base와 비교해 `matches` / `differs from`을 로그에 남깁니다. 경고만 하고 거부하지는 않습니다.

### 거부하는 조합

| 조건 | 결과 | 위치 |
|---|---|---|
| network가 `LoRANetwork`가 아님 (EasyControl 등) | `SystemExit` | `convert_for_training` |
| `--base_weights`와 forward copy 파일을 함께 사용 | `SystemExit` (`--int8_file rtn`을 쓰라는 안내) | `check_plan` |
| 생성에서 `--lora_cutoff_step` / `--lora_sigma_range` / `--lora_timestep_mask` | `SystemExit` | `convert_for_inference` |
| int8 체크포인트에 정적 LoRA merge(`lora_weights_list`) | `ValueError` | `load_anima_model` |
| bias가 있는 linear | `ValueError` | `convert_` |

`inference.py`에는 base 종류와 상관없이 block swap이 없습니다(`docs/anima/optimizations/int8.md`, Limits).

### 측정값 (문서와 bench 결과에서 옮김)

학습 속도는 `ama mitsuki`, 848 steps, method `lora`, 5070 Ti 기준입니다(`docs/anima/optimizations/int8.md`, Measurements).

| | bf16 | int8 `full`, bf16 kernel input | int8 `full`, fp32 kernel input |
|---|---|---|---|
| step | 593 ms | 433 ms (1.37x) | 424 ms (첫 compile 포함) |
| 최종 `avr_loss` | 0.0737 | 0.0737 | 0.0739 |

두 LoRA를 bf16 base에서 생성한 샘플의 품질은 같았습니다. base를 bf16에서 int8로 바꾸면 생성 픽셀이 약 4/255
움직입니다(같은 문서).

`stream`과 `full`의 peak는 `project/finished/int8_train/reports/anima_wqt_stream.md`에서 가져왔습니다. budget 0.99, 106 샘플 전체 기준입니다.

| arm | median | peak |
|---|---|---|
| `full` | 482 ms | 12.38 GB |
| `stream` | 499 ms | 10.96 GB |
| `stream`, budget 1.0 | step 0에서 OOM | – |

`stream`은 17 ms/step을 더 쓰고 peak를 1.42 GB 줄입니다. loss는 `full`과 같습니다(같은 리포트의 Verdict).

block swap(`ama mitsuki`, 106 steps, budget 0.99, GPTQ forward copy, fp32 stream)에서 `stream`은 swap 0 / 14 / 24일 때
451 / 475 / 524 ms, 11.47 / 10.44 / 9.70 GB였습니다. swap 14에서 `full`은 526 ms / 11.10 GB, `fwd`는 667 ms / 11.92 GB였습니다
(`int8.md`, Block swap).

16 GB보다 작은 카드에서는 swap보다 activation budget을 먼저 내리라고 문서가 권합니다. 같은 문서의 budget 표에서
`stream`은 0.99일 때 427 ms / 10.87 GB, 0.3일 때 472 ms / 6.87 GB, 0.1일 때 511 ms / 5.32 GB입니다.

터보(`make turbo`)는 `int8 = "stream"`으로 배포되고, teacher도 int8로 돕니다(`configs/methods/turbo.toml`).
warm start, budget 0.99 기준 수치는 다음과 같습니다(`int8.md`, Limits).

| 터보 base | s/step | peak |
|---|---|---|
| bf16 base | 9.65 | – |
| int8 `stream`, bf16 stream | 6.28 | 11.87 GB |
| int8 `stream`, fp32 stream | 6.7–7.0 | 12.39 GB |

block swap을 켠 터보(30 steps, CDM on, GAN off, fp32 stream)는 swap 0에서 5.8 s/step / 12.36 GB, swap 14에서
8.4 s/step / 11.32 GB였습니다. 터보는 forward마다 swap을 되감기 때문에 swap 비용이 학습보다 큽니다(`int8.md`, Block swap).

생성 쪽은 smoke 한 번만 있습니다. 768², 8 steps, 한 셀에 한 번씩 그렸고, 전체 run은 중단됐습니다. 문서도
"not a result"라고 적었습니다(`docs/anima/optimizations/fp32_stream.md`). bf16 261 ms/step / 4.11 GiB,
int8 164 ms/step / 2.64 GiB, int8 + fp32 stream 179 ms/step / 2.68 GiB입니다.

> TODO(확인 필요): `inference.py --int8`의 정식 생성 속도와 VRAM 측정 (1024², 28 steps, compile on). 현재는 위 smoke뿐입니다.

메모리 노트의 "int8 swap ≤ 36" 한도(`project_box_environment.md`)는 MOSS(42블록, 블록 하나가 0.106 GB int8)에서
측정한 값입니다. 28블록 Anima에는 그대로 옮길 수 없습니다. Anima에서 측정된 swap 값은 위 표의 0 / 14 / 24뿐입니다.

### kernel input과 fp32 stream (짧게)

stock `comfy_kitchen`(0.2.40 / 0.2.41)은 bf16 입력을 받으면 activation 나눗셈 근처에서 값을 bf16로 반올림합니다
(comfy-kitchen#289). 그래서 `Int8TrainLinear.forward`는 `kernel_divides_fp32()` 검사가 False이면 입력을 fp32로 바꿔
넘깁니다. 이 fp32 kernel input은 bf16 input보다 예측 오차를 6–8% 줄입니다(`int8.md`, Accuracy). fp32 residual stream은
int8 base의 오차를 21–36% 더 줄입니다(같은 문서). stream 자체는 [05_practical_tips.md](05_practical_tips.md)에서 다룹니다.

---

## 3.4 GPTQ 파일 만들기

### 빌더

`make` 타깃은 없습니다. `tasks.py`, `scripts/`, `library/`에서 `gptq`를 찾으면 읽는 쪽 코드만 나옵니다. 빌더는 끝난
연구 라인에 있는 `project/finished/int8_train/src/dit_gptq.py`이고, daemon으로 실행합니다. 같은 폴더의
`int8_linear.py`는 `library/anima/int8.py`와 `library/int8/linear.py`를 다시 내보내는 import 경로일 뿐이라서,
fingerprint와 변환 코드는 trainer와 같습니다.

```bash
# base (2026-10-30, job 20261030-114754-5e5507, 17.6분; reports/gptq.md)
make daemon-run ARGS="--stall-timeout 1800 project/finished/int8_train/src/dit_gptq.py"

# merged14 teacher (2026-11-10, job 20261110-103019-98994f, 1045초)
make daemon-run ARGS="--queue --stall-timeout 1800 project/finished/int8_train/src/dit_gptq.py \
  --dit models/diffusion_models/anima_merged14.safetensors"
```

출력 경로 기본값은 `models/diffusion_models/{src.stem}_int8_{method}.safetensors`이고, 옆에 `.report.json`을 씁니다.
`--method gptq`(기본값)이면 이름이 `<dit>_int8_gptq.safetensors`가 되어, 3.5의 "DiT 옆 파일" 규칙과 맞습니다.
merged14 job 로그의 마지막 줄은
`wrote .../anima_merged14_int8_gptq.safetensors (2.37 GiB) int8_id b469fa22d262d231 in 1045s`입니다.

> TODO(확인 필요): GPTQ job의 peak VRAM. 로그에 남지 않습니다.

### calibration

`dit_gptq.py::pick_images`, `main` 기준입니다.

- `--cache_root`(기본 `post_image_dataset/lora`)의 작가 폴더마다 캐시된 이미지를 최대 `--per_artist 2`장 고릅니다.
  토큰 수가 `--min_tokens 3000` 이상 `--max_tokens 4608` 이하인 것만 고르고, `--seed 0`으로 섞습니다.
  캐시된 VAE latent와 TE 출력을 그대로 씁니다.
- 이미지마다 σ를 하나씩 뽑습니다. σ는 추론 schedule(`--steps 28`, `--flow_shift 3.0`)에서 고르고, 그 σ로 노이즈를 섞습니다.
- 각 샘플을 두 번 돌립니다. 한 번은 자기 캡션으로, 한 번은 `--negative_prompt`로 돌립니다(CFG uncond 쪽). 기본 negative는
  `worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia`입니다.
- 블록 0에 들어가는 입력을 잡아 두고, 블록을 **순서대로** 양자화합니다. 양자화가 끝난 블록(W8A8 kernel)의 출력이 다음 블록의
  입력이 되므로, 앞쪽 블록의 양자화 오차가 뒤쪽 calibration에 반영됩니다.

| | base | merged14 |
|---|---|---|
| 작가 / 이미지 / 샘플 | 91 / 182 / 364 | 92 / 184 / 368 |
| linear당 토큰 | 1,487,472 | 1,510,348 |
| σ schedule | 28 steps, shift 3.0 | 28 steps, shift 3.0 |

두 job은 같은 기본 설정으로 돌았습니다(merged14 job argv에는 `--dit`만 있습니다). 고른 작가 수만 91에서 92로 다른데,
그 사이 `post_image_dataset/lora`의 폴더가 늘었는지는 확인하지 않았습니다.

### 양자화 방법

`dit_gptq.py` docstring과 `roundings`, `gptq` 기준입니다. linear 하나마다 회전한 입력 `X_r`로 `H_r = X_rᵀX_r`를 모읍니다.
같은 grid(256열 회전, 행 스케일 `amax/127`)에서 반올림을 세 가지로 해 봅니다.

- `rtn`: round to nearest입니다. 파일 없이 `convert_`가 하는 것과 같습니다.
- `gptq`: `H_r`에 대한 GPTQ입니다. 열 단위로 반올림하고 오차를 남은 열로 넘깁니다. `--block 128`, `--damp 0.01`,
  act-order는 켭니다(`--no_act_order`로 끕니다).
- `act_aware`: activation 양자화까지 고려한 최소제곱 가중치 `W*`에 GPTQ를 적용합니다(`--ls_damp 1e-4` ridge).

세 방식 모두 W8A8 출력 오차 `‖X̃Qᵀ − X_r W_rᵀ‖² / ‖X_r W_rᵀ‖²`로 채점합니다. `rtn`과 `gptq`는 weight-only 오차(W8A16)로도
채점합니다. 파일에 쓰고 다음 블록의 입력을 만드는 데 쓰는 것은 `--method`(기본 `gptq`)입니다.

### 파일 구조

`write_file` 기준이고, merged14 파일의 헤더를 CPU로 읽어 확인했습니다.

- **DiT 전체**입니다. 원본의 텐서를 모두 담고, 변환 대상 linear만 int8로 바꿉니다. fused `qkv_proj`는 q / k / v로 다시 나눕니다.
  변환한 키마다 `weight`(I8), `weight_scale`(F32, `(out, 1)`), `comfy_quant`(U8, JSON 설정)가 붙습니다.
  ComfyUI의 `int8_tensorwise` + `convrot` 레이아웃입니다(`quantization` 메타데이터:
  `{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}`).
- merged14 GPTQ: 텐서 1133개(BF16 461, I8 224, F32 224, U8 224), 2,540,645,280 B(2.37 GiB). I8 224개는 28블록 × 8키
  (q, k, v, self out, cross q, cross out, layer1, layer2)입니다. base GPTQ 파일도 크기가 같습니다.
- 메타데이터: `source`, `source_sha256`, `base_fingerprint`, `method`, `calibration`, `gptq`, `int8_id`, `int8_modules`
  (변환한 168개 모듈 이름 JSON), `quantization`.

`.report.json`은 `{"meta", "int8_id", "mean", "linears"}` 구조입니다. `linears`에는 168개 linear마다 `w8a8_rtn`,
`w8a8_gptq`, `w8a8_act_aware`, `w8a16_rtn`, `w8a16_gptq`, `ls_damp`가 있고, `mean`은 그 평균입니다.

### 두 파일의 report 비교

`models/diffusion_models/anima-base-v1.0_int8_gptq.report.json`과
`models/diffusion_models/anima_merged14_int8_gptq.report.json`을 그대로 옮겼습니다. 168개 linear의 평균이고, 단위는
상대 출력 오차 에너지입니다.

| | base | merged14 |
|---|---|---|
| `int8_id` | `9dad0c30a67e4d48…` | `b469fa22d262d231…` |
| `base_fingerprint` | `f46caab29d8381b4…` | `28a3dc7eb701e073…` |
| `source_sha256` | `bd43b7cffe1ed115…` | `282bca0454602c14…` |
| W8A8 RTN → GPTQ | 6.40e-5 → 4.12e-5 (−35.5%) | 6.53e-5 → 4.24e-5 (−35.1%) |
| W8A8 act-aware | 4.10e-5 | 4.22e-5 |
| W8A16 RTN → GPTQ | 2.79e-5 → 0.52e-5 (−81.4%) | 2.84e-5 → 0.55e-5 (−80.6%) |
| 평균 `ls_damp` | 1.21e-4 | 1.11e-4 |
| W8A8 GPTQ 최악 linear | `blocks.17.cross_attn.output_proj` 1.41e-4 | `blocks.15.cross_attn.output_proj` 1.48e-4 |

linear 종류별 W8A8 GPTQ 평균(report에서 계산):

| linear | base | merged14 |
|---|---|---|
| `self_attn.qkv_proj` | 3.68e-5 | 3.72e-5 |
| `self_attn.output_proj` | 6.74e-5 | 7.04e-5 |
| `cross_attn.q_proj` | 4.64e-5 | 4.41e-5 |
| `cross_attn.output_proj` | 5.40e-5 | 5.57e-5 |
| `mlp.layer1` | 2.17e-5 | 2.20e-5 |
| `mlp.layer2` | 2.12e-5 | 2.53e-5 |

soup을 구웠다고 int8이 더 어려워지지는 않았습니다. 전체 평균 차이는 3% 안이고, 가장 크게 움직인 것은 `mlp.layer2`(+19%)입니다.
act-aware가 plain GPTQ보다 나은 점이 없다는 결론도 base에서와 같습니다. merged13 GPTQ(`int8_id 77f7e53e…`)도
W8A8 RTN → GPTQ가 6.51e-5 → 4.22e-5로 거의 같았습니다(작업 세션 `09b29148` 기록이며, merged13 report 파일은 지금
`models/diffusion_models/`에 없습니다).

### GPTQ가 실제로 줄이는 것 (base 측정)

`project/finished/int8_train/reports/gptq.md`(job `20261030-120640-c9289f`, `traj_eval.py`)의 결과입니다.

- GPTQ는 weight 항의 81%를 없앱니다. W8A8 오차는 36% 줄고, 남은 오차는 대부분 activation 항입니다.
- end to end로 보면 cond / uncond 예측 오차가 약 10% 줄고(0.87–0.92x), CFG velocity 오차는 2–7% 줄어듭니다(0.93–0.99x).
  base 자체의 trajectory에서 재도 held-out 실제 이미지에서 재도 같아서, calibration 소스는 결과에 영향이 없습니다.
- CFG velocity에서 이득이 작은 이유는 이렇습니다. weight 오차는 cond와 uncond가 공유하므로 `v_c − v_u`에서 대부분 상쇄되고,
  CFG가 증폭하는 것은 공유되지 않는 activation 반올림입니다.
- LoRA 학습은 CFG 없이 cond 예측을 쓰므로, 학습 쪽 이득은 약 10% 쪽 수치에 해당합니다.

> TODO(확인 필요): merged14 GPTQ의 end-to-end 오차(`traj_eval.py`를 merged14에서 돌린 것)는 측정하지 않았습니다. 위 수치는
> base 기준이고, merged14는 per-linear report만 있습니다.

### round to nearest fallback

forward copy 파일이 없거나 `--int8_file rtn`을 주면 `convert_`가 그 자리에서 RTN으로 양자화합니다(`quantize`). 이때 LoRA
stamp는 `ss_int8_base_id = "rtn"`입니다. GPTQ 대비 손해는 위에서 본 예측 오차 약 10%입니다. RTN은 fingerprint 검사를
하지 않으므로 어떤 DiT에서도 동작합니다. 40블록 Anima-2.9B나 GPTQ를 아직 만들지 않은 merged DiT에서 쓸 수 있는 유일한
선택지입니다.

### catalog row

`library/downloads.py`:

- `Asset(id="anima_dit_int8", pack="anima-dit-int8", repo="sorryhyun/anima-base-v1.0-int8", files=("anima-base-v1.0_int8_gptq.safetensors",), dest=models/diffusion_models)`.
  설명은 "~2.4 GB, 블록 linear는 INT8 ConvRot + offline GPTQ, 나머지는 bf16. fingerprint가 다른 DiT를 거부하므로 40블록
  Anima-2.9B는 자기 파일이 필요하거나 round to nearest를 써야 함"입니다.
- 받는 방법은 `make download-model ARGS=anima-dit-int8`입니다. 또는 `dit_path`가 catalog의 base DiT일 때 `--int8`로
  처음 실행하면 자동으로 받습니다(`forward_file`, `is_int8_checkpoint`). `make download-models`의 첫 실행 세트에는 들어 있지 않습니다.
- **catalog에는 base 파일만 있습니다.** merged DiT 옆의 GPTQ 파일은 직접 만들어야 합니다. 자동 다운로드는 경로가 catalog
  파일과 정확히 같을 때만 일어납니다.

> TODO(확인 필요): merged14 GPTQ 파일을 Hub에 올릴지. 지금은 로컬(`models/diffusion_models/`, NVMe로 가는 symlink)에만 있습니다.

---

## 3.5 `int8_id`, `base_fingerprint`, 거부 로직

### 정의

- **`int8_id`** (`dit_gptq.py::write_file`): 변환한 키 이름, int8 텐서, 스케일 바이트를 이름 순으로 이어 붙인 sha256입니다.
  GPTQ 결과물 자체의 ID이고, 이 base에서 학습한 LoRA에 `ss_int8_base_id`로 찍힙니다.
- **`base_fingerprint`** (`library/int8/linear.py::base_fingerprint`): 변환 **전** bf16 가중치에서, 변환 대상의 **첫 linear와
  마지막 linear** 두 개의 이름과 bf16 바이트를 sha256한 값입니다. Anima에서는 `blocks.0.self_attn.qkv_proj`와
  `blocks.27.mlp.layer2`입니다. GPTQ 파일은 자신을 만든 DiT의 fingerprint를 메타데이터에 기록합니다.
- **`source_sha256`**: 원본 파일 전체의 sha256입니다. 기록만 하고 로드할 때 검사하지는 않습니다.

### forward copy를 고르는 순서

`library/anima/int8.py::forward_file`:

1. `--int8_file <path>`. `rtn`이면 round to nearest입니다.
2. `dit_path` 자체가 int8 체크포인트(메타데이터에 `int8_modules`가 있음)이면 그 파일입니다.
3. `dit_path` 옆의 `<dit stem>_int8_gptq.safetensors`입니다.
4. 3의 경로가 catalog의 base 파일이면 받아 옵니다.
5. 모두 아니면 round to nearest입니다.

### 검사

`convert_dit_`는 파일의 `base_fingerprint`를 **지금 로드된 DiT**의 fingerprint와 비교해서, 다르면
`ValueError("<file> was made from <source>, not the loaded DiT")`를 냅니다. 예외가 하나 있습니다. DiT가 그 int8 체크포인트
자체에서 로드된 경우(`dit.int8_checkpoint == int8_file`)에는 검사를 건너뜁니다. 이때 로드된 가중치는 int8에서 복원한
값이라 fingerprint가 맞을 수 없기 때문입니다. 파일에 대상 모듈이 하나라도 없으면 `"has no forward copy for <name>"`로 거부합니다.

`dit_nonquant_path`를 주면 `load_anima_model`은 bf16 파일에서 float 가중치를 읽고 `int8_checkpoint`를 False로 둡니다.
그래서 fingerprint 검사는 **bf16 DiT와 GPTQ 파일 사이에서** 일어납니다. 두 파일이 짝이 맞는지를 여기서 확인합니다.

| `dit_path` | `dit_nonquant_path` | `--int8_file` | 결과 |
|---|---|---|---|
| `anima_merged14_int8_gptq` | `anima_merged14` | – | **통과** (터보 config, 3.7) |
| `anima_merged14` | – | – | 옆의 `anima_merged14_int8_gptq`를 찾아서 **통과** |
| `anima_merged14_int8_gptq` | – | – | 통과(검사 생략). 단 `wq_t`와 `weight_svd` 초기화가 int8에서 복원한 가중치를 읽음 (아래) |
| `anima-base-v1.0_int8_gptq` | `anima_merged14` | – | **거부**: 파일은 base에서 만들었고 로드된 DiT는 merged14 |
| `anima_merged14` | – | `anima-base-v1.0_int8_gptq` | **거부** |
| `anima_merged14_int8_gptq` | `anima_merged13` | – | **거부** (merged13 파일과 merged14 DiT의 조합도 마찬가지) |
| 아무거나 | – | `rtn` | 통과 (RTN, 검사 없음) |

`anima-base-v1.0_int8_gptq`와 `anima_merged14` 조합의 거부는 위 코드 경로에서 나옵니다. merged13 작업 세션
(`200d31dc`)에서 soup LoRA가 fingerprint가 해시하는 두 linear를 모두 건드리는 것을 확인했습니다. merged14의
fingerprint(`28a3dc7e…`)도 base(`f46caab2…`)와 다릅니다.

### `dit_nonquant_path` 없이 GPTQ 파일만 줄 때

`load_anima_model`은 `"no dit_nonquant_path; wq_t and LoRA init read weights rebuilt from int8"` 경고를 남기고,
`dequantize_checkpoint_`로 float 가중치를 다시 만듭니다. 문서에 따르면 다시 만든 가중치는 원래 가중치와 1.1% 다르고,
그 결과 `wq_t`가 1.6% 움직입니다(`docs/anima/optimizations/int8.md`, Forward copies). 생성(`infer`)은 `wq_t`가 필요 없으므로
GPTQ 파일 하나로 충분합니다. 학습과 터보에서는 `dit_nonquant_path`를 함께 줍니다.

### fingerprint의 한계

fingerprint는 첫 linear와 마지막 linear만 봅니다. 그래서 그 두 linear를 건드리지 않는 LoRA(예: 일부 블록 범위에만 붙은
LoRA)를 구운 DiT에는 base의 GPTQ 파일이 **조용히 통과**합니다. 이때 forward copy는 bake 전 가중치로 계산되므로, 그 블록들의
LoRA delta는 int8 forward에서 사라집니다. soup처럼 모든 블록을 건드리는 LoRA라면 문제가 없습니다(merged13과 merged14 모두
fingerprint가 바뀌었습니다). 직접 bake한 DiT라면 fingerprint가 달라졌는지 직접 확인하거나, 처음부터 그 DiT 전용 GPTQ를 만드는
것이 안전합니다.

---

## 3.6 teacher에 전용 GPTQ가 필요한 이유

1. **거부됩니다.** base GPTQ 파일의 `base_fingerprint`는 `anima-base-v1.0`의 것이고, soup을 구운 DiT에서는 첫 linear와 마지막
   linear의 바이트가 다릅니다. `convert_dit_`가 `ValueError`를 냅니다(3.5).
2. **거부되지 않더라도 틀린 값입니다.** forward copy는 그 자체가 가중치입니다. base 파일을 쓰면 teacher의 int8 forward는
   soup 없는 base를 계산합니다. 그러면 teacher가 soup이 아니게 되고, soup을 teacher로 쓰려던 목적이 사라집니다.
3. **GPTQ는 가중치와 activation 통계 모두에 묶여 있습니다.** 반올림은 그 가중치의 `H_r`(그 모델의 블록 입력)에 맞춰 최적화되고,
   블록 i의 calibration 입력은 양자화된 블록 0..i−1을 지나 만들어집니다. 그래서 다른 DiT의 파일을 다시 쓸 방법이 없습니다.
4. **RTN으로 버티는 것과 비교하면** `--int8_file rtn`으로도 merged DiT를 int8로 돌릴 수 있습니다. 대신 per-linear W8A8 오차가
   약 1.5배(merged14: 6.53e-5 대 4.24e-5)이고, base 측정 기준으로 예측 오차가 약 10% 큽니다. 만드는 데 17분이면 되므로
   전용 GPTQ를 만드는 쪽을 택했습니다.

soup이 바뀔 때마다(merged13 → merged14) bake와 GPTQ를 다시 해야 합니다. merged13 쌍은 merged14 쌍으로 대체되었습니다
(`project_soup_turbo.md`).

> TODO(확인 필요): int8 teacher(전용 GPTQ)와 bf16 teacher로 학습한 터보 student를 비교한 A/B는 찾지 못했습니다. int8 base가
> 터보 품질에 미치는 영향은 위의 per-linear / per-prediction 오차로만 간접적으로 알 수 있습니다.

> TODO(확인 필요): GPTQ calibration은 28-step, shift 3 추론 schedule에서 σ를 뽑습니다. 4-step student가 지나는 σ에서 따로
> 오차를 잰 적이 있는지 확인하지 못했습니다. base 측정(`gptq.md`)은 σ 구간 세 개 모두에서 0.87–0.92x로 고르게 나왔습니다.

---

## 3.7 실전 순서: bake → GPTQ → 터보

```bash
# 1) soup LoRA를 base에 bake (GPU, 약 13초)
make daemon-run ARGS="scripts/toolkits/merge_to_dit.py \
  --adapter output/ckpt/anima_merged14.safetensors --multiplier 1.0 --dtype bf16 \
  --out models/diffusion_models/anima_merged14.safetensors"

# 2) 그 DiT의 GPTQ forward copy (GPU, 약 17분; 출력은 models/diffusion_models/<stem>_int8_gptq.*)
make daemon-run ARGS="--queue --stall-timeout 1800 project/finished/int8_train/src/dit_gptq.py \
  --dit models/diffusion_models/anima_merged14.safetensors"
make daemon-wait JOB=<id>    # background Bash에서

# 3) 터보 config가 두 파일을 가리키게 하고 실행
make turbo ARGS="--config project/anima_turbo/configs/<id>.toml --queue"
```

터보 config(`project/anima_turbo/configs/1112_0833.toml`)에서 이 단계와 관련된 키는 다음과 같습니다.

```toml
dit_path = "models/diffusion_models/anima_merged14_int8_gptq.safetensors"
dit_nonquant_path = "models/diffusion_models/anima_merged14.safetensors"
activation_memory_budget = 0.99
int8 = "stream"
stream = "fp32"
```

배포용 `configs/methods/turbo.toml`은 같은 자리에 base 쌍(`anima-base-v1.0_int8_gptq` / `anima-base-v1.0`)을 둡니다.
merged14 쌍은 arm config에서만 바꾸고, 공유 config는 건드리지 않았습니다(작업 세션 `09b29148`). `scripts/distill_turbo/config.py`의
`int8` 기본값은 `"off"`이므로, `int8 = "stream"`은 TOML에 꼭 적어야 합니다.

### 점검 목록

- bake 출력이 `models/diffusion_models/`에 있는지 확인합니다. `dit_gptq.py`는 원본 위치와 상관없이
  `models/diffusion_models/`에 씁니다. bake 출력이 `output/ckpt/`에 있으면 GPTQ 파일이 "DiT 옆"에 생기지 않아서,
  3.5의 3번 규칙이 그 파일을 찾지 못합니다.
- `--method act_aware`로 만든 파일은 이름이 `_int8_act_aware`라서 자동으로 찾지 못합니다. `dit_path`나 `--int8_file`로
  직접 지정해야 합니다. act-aware는 이득도 없습니다(3.4).
- 터보 로그의 `int8 stream: converted 168 linears, forward copies <path>` 줄에서 파일 경로를 확인합니다
  (`library/anima/int8.py::_convert`). `round to nearest`라고 나오면 GPTQ 파일을 찾지 못한 것입니다.
- 결과 LoRA의 `ss_int8_base_id`가 `b469fa22…`인지 확인합니다. 이 값은 학습할 때의 teacher/base 짝을 기록합니다.
- 생성에서 teacher를 볼 때는 `inference.py --dit models/diffusion_models/anima_merged14.safetensors --int8`로 돌립니다.
  옆의 GPTQ 파일을 자동으로 씁니다. bf16로 보려면 `--int8`을 빼면 됩니다.
- 배포는 bake된 DiT로 하지 않습니다. soup LoRA와 터보 LoRA를 `merge_loras.py --normalize off --adaln sum`으로 exact concat해서
  base에 올라가는 LoRA 하나로 만듭니다. [02_soup_merge.md](02_soup_merge.md)와 [04_turbo_distillation.md](04_turbo_distillation.md)를 보세요.

### 소요 시간 (merged14, 5070 Ti)

| 단계 | job | 걸린 시간 |
|---|---|---|
| bake | `20261110-102956-89d8d6` | 13초 |
| GPTQ | `20261110-103019-98994f` | 1045초 (약 17.4분; calibration 수집 18초, 블록당 약 36초) |

---

## 3.8 문서와 코드가 어긋나는 곳

1. `docs/anima/optimizations/int8.md`에는 "A static LoRA merge on a rebuilt base is refused"라고 적혀 있습니다. 이것은
   `load_anima_model`의 `lora_weights_list` 경로(생성 시 정적 merge)에만 해당합니다. `merge_to_dit.py --dit <GPTQ file>`은
   `network.merge_to`를 쓰기 때문에 거부되지 않고, 경고만 남긴 채 int8에서 복원한 가중치 위에 굽습니다.
2. 문서(`int8.md`, catalog `notes`)는 "fingerprint가 다른 checkpoint의 파일을 거부한다"고 적었습니다. 실제 fingerprint는
   첫 linear와 마지막 linear 두 개만 해시합니다. 그래서 두 linear를 건드리지 않은 merged DiT에는 base 파일이 통과합니다(3.5).
3. CLAUDE.md는 `make merge`가 "Hydra-moe / postfix"를 거부한다고 적었습니다. 코드는 prefix, register tokens,
   step-expert 터보 head(`.lora_ups.`)도 거부합니다(`_NON_BAKEABLE_MARKERS`).
4. `int8.md`는 GPTQ 파일이 `dit_gptq.py`로 만들어진다고만 적었습니다. 이 스크립트는 `project/finished/` 아래에 있고
   `make` 타깃이 없습니다. merged DiT용 파일을 만들 공식 진입점이 없는 셈입니다.
5. 메모리 노트의 int8 swap 한도(≤ 36)는 MOSS의 값이고 Anima에는 해당하지 않습니다(3.3).

---

## 3.9 다른 섹션

- LoRA 학습 레시피: [01_lora_recipe.md](01_lora_recipe.md)
- soup과 `merge_loras.py`, merged14 LoRA를 만드는 과정: [02_soup_merge.md](02_soup_merge.md)
- DP-DMD 터보 루프, 레시피, 판정: [04_turbo_distillation.md](04_turbo_distillation.md)
- fp32 stream, daemon, compile, VRAM 운용: [05_practical_tips.md](05_practical_tips.md)
