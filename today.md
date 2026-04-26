# 오늘 한 일 (2026-04-26)

안녕하세요, Anima 한국 커뮤니티 여러분. 오늘 하루 동안 작업한 내용을 정리해서 공유드립니다.
오늘은 **레퍼런스 이미지 기반 생성** 두 종류 — IP-Adapter와 EasyControl — 을 Anima에
새로 붙인 큰 날이었습니다. 코드 줄 수로만 따지면 **+6500 / −700 줄** 정도 나왔네요.

## 왜 이걸 했나

지금까지 Anima에는 **레퍼런스(referencing) 기능 자체가 없었습니다.** 캐릭터/스타일을
잡으려면 LoRA를 새로 학습하는 것 외에 선택지가 없었어요. "이 그림 한 장처럼 그려줘"
같은 가장 기본적인 요구를 즉석에서 받아낼 방법이 없었다는 뜻입니다. 다른 모델
생태계에는 이미 IP-Adapter, EasyControl, Redux, ControlNet, Reference-only 같은
레퍼런스 모듈이 보편화돼 있는데 Anima에는 이 자리가 통째로 비어 있었습니다.

이걸 메우려고 며칠 전부터 **img2emb** (이미지를 prompt 임베딩 자리에 직접 끼워넣는
방식) 를 실험해 왔는데, 오늘 결과를 정리해 보니 레퍼런스 용도로는 별로였습니다.
그래서 오늘부터 방향을 틀어서 **이미 검증된 레퍼런스 아키텍처 두 개 (IP-Adapter,
EasyControl) 를 Anima에 직접 이식**하는 작업을 시작했습니다.

커밋 4개 요약:

| 시간  | 커밋        | 줄 수 (+/−)         | 내용 |
|------|-------------|--------------------|------|
| 15:18 | `0eddcbc`  | 0 / 0 (파일 이동)   | bench 폴더 정리 + img2emb 실험 결론 |
| 18:03 | `ab9b38d`  | +2,427 / −132      | **IP-Adapter** 구현 |
| 20:36 | `32b1fa8`  | +2,554 / −33       | **EasyControl** 1차 구현 |
| 22:09 | `3a551f5`  | +1,559 / −536      | EasyControl 2차 리팩터 (구조 갈아엎음) |

---

## 1. img2emb 실험 마무리 + bench 폴더 정리 (`0eddcbc`)

### img2emb 가 왜 별로였나

img2emb 는 **"이미지를 텍스트 임베딩처럼 쓰자"** 는 접근이었습니다. 비전 인코더
(TIPSv2 또는 PE-Core) 로 이미지를 토큰화한 뒤 그걸 작은 모델(Perceiver resampler)
로 압축해서, DiT가 받는 prompt 임베딩 자리에 같이 넣는 구조였어요.

며칠 동안 돌려본 결과 두 가지 한계가 또렷했습니다:

- **Identity가 안 잡혔습니다.** prompt 임베딩 한 자리에 이미지 한 장의 의미를 다
  욱여넣는 건 압축률이 너무 빡세서, 학습이 진행돼도 "비슷한 분위기" 까지만
  올라가고 같은 캐릭터로 안정적으로 수렴하지 않았습니다.
- **Prompt가 안 먹혔습니다.** 이미지와 텍스트가 같은 자리에 있다 보니, ref 가 강하면
  prompt 가 묻혀버렸습니다. "ref 의 스타일 + 다른 구도" 같은 조작이 거의 안 됐어요.
- 별별 트릭(anchor, teacher forcing 등) 을 다 붙여봤지만 위 두 문제가 **구조적**
  이라 안 풀렸습니다.

결론: **이미지 신호는 prompt 자리가 아니라 자기 자리(별도 attention 분기) 에 들어가야
한다.** 그래서 오늘부터 IP-Adapter (별도 cross-attention) 와 EasyControl
(self-attention 에 cond stream 합류) 로 방향을 틀게 됐습니다. img2emb 코드 자체는
PE-Core 인코더 래퍼 / Perceiver resampler / bucket spec 같은 재사용 가능한 부품이
많아서 죽이지 않았고, 실제로 IP-Adapter 가 이걸 그대로 가져다 씁니다.

### bench 폴더 정리

벤치마크 스크립트가 그동안 `bench/` 아래에 평면적으로 쌓여서 어디부터 봐야 할지
헷갈리는 상태였는데, 이걸 두 폴더로 갈랐습니다:

- `bench/active/` — 지금도 의미 있는 실험 (dcw, hydralora, img2emb, inversionv2, spectrum)
- `bench/archive/` — 끝난 실험 (fa4, postfix, inversion v1 등)

코드 변경은 0줄, 순수 파일 이동만 한 커밋입니다. 이후 EasyControl 벤치들이 자연스럽게
`bench/active/easycontrol/` 자리에 들어갔습니다.

---

## 2. IP-Adapter (`ab9b38d`, +2,427 / −132 줄)

원조 IP-Adapter (Ye et al. 2023) 의 **decoupled image cross-attention** 구조를
그대로 Anima에 옮겼습니다.

### 한 줄 요약

> "DiT 가 텍스트를 보는 cross-attention 옆에, 이미지를 보는 cross-attention 을 하나
> 더 만들어서 같이 쓴다."

```
ref image
   │
   ▼
PE-Core (frozen 비전 인코더) ──►  patch tokens
   │
   ▼
Perceiver resampler  ──► IP tokens [B, K=16, 1024]   (학습 대상)
   │
   ▼  DiT 의 cross-attn 28개 블록 각각에서:
   ┌─────────────────────────────────────────────────────────┐
   │  text_result = cross-attn(q, text K, text V)            │
   │  ip_out      = cross-attn(q, IP K, IP V)   ← 옆에 따로  │
   │  out         = output_proj(text_result + scale·ip_out)  │
   └─────────────────────────────────────────────────────────┘
```

### 알아두면 좋은 점 몇 개

- **텍스트 K/V 에 이어붙이지 않고 별도 softmax 로 분리.** 이어붙이면 softmax
  분모가 공유돼서, IP 분기가 신호를 실으려면 텍스트 attention 을 빼앗아야 합니다.
  분리하면 그런 trade-off 없이 양쪽이 공존할 수 있어요. 원조 IP-Adapter 가 이렇게
  설계된 이유이기도 합니다.
- **Per-block gate 안 씀.** 처음 학습 시점에 ip_out ≈ 0 이 되도록 하는 건 gate (`α`)
  대신 **V 의 초기값을 거의 0으로 두는 방식** 으로 해결했습니다. gate 도 같이
  넣어봤는데 학습이 잘 안 됐어요 (`α≈0` + `K/V` 가 70배로 커지는 식의 보상이
  일어남). 원조 논문도 gate 없이 갑니다.
- **DiT 는 통째로 freeze.** 학습되는 건 resampler + 블록별 K/V 프로젝션뿐 (~150M
  파라미터). 일반 LoRA 보다는 큽니다.
- **PE-Core-L14-336 비전 인코더.** 동적 해상도 지원 (다양한 가로/세로 비 가능),
  학습 전에 `make ip-adapter-cache` 로 한 번만 돌려서 디스크에 캐싱해 두면
  학습 때 비전 인코더를 메모리에 안 올려도 됩니다 (VRAM ~600 MB 절약).
- **Caption dropout 함정.** 원조 레시피는 caption_dropout 0.05 인데, 이건 수백만
  장 학습 기준입니다. Anima 처럼 작은 데이터셋에서 0.5 같은 큰 값을 쓰면 2 epoch
  안에 mode collapse 합니다. 권장 범위는 **0.10–0.20**.

### 학습/추론

```bash
make ip-adapter-cache         # PE 피처 캐시 (한 번만)
make ip-adapter               # 학습
make test-ip REF_IMAGE=foo.png PROMPT="..." IP_SCALE=0.8
```

자세한 사용법은 `docs/methods/ip-adapter.md` 에 길게 적어뒀습니다.

---

## 3. EasyControl — 1차 (`32b1fa8`) → 2차 (`3a551f5`)

이건 오늘 두 번 작업한 부분입니다. 1차 (+2,554 줄) 를 끝내고 학습을 돌려봤더니
메모리/구조 문제가 있어서 저녁에 **−536 / +1,559 줄로 통째로 다시 짰습니다.**

### 한 줄 요약

> "이미지를 DiT 의 self-attention 에 같이 흘려보내서, 모든 블록에서 target 토큰이
> ref 토큰을 함께 보게 만든다."

IP-Adapter 가 텍스트와 같은 자리(cross-attention)에 붙는다면, EasyControl 은
**self-attention 자리** 에 붙습니다. 이미지의 공간 정보가 그대로 흘러서
구도/포즈/디테일 같은 구조적인 conditioning 에 강합니다.

### 1차 시도 (Phase 1.5, `32b1fa8`)

처음에는 이렇게 짰어요:

- target forward 시작 전에 **cond stream 을 따로 한 번 다 돌려서** 블록별로
  `(K_c, V_c)` 를 미리 캐싱해 두고
- 캐싱된 K_c/V_c 를 target self-attention 에 합쳐서 사용
- backward 는 따로 호출해서 cond chain 을 거꾸로 한 번 더 돌려야 함

문제는 두 가지:
1. **메모리.** 블록별 K/V 캐시가 ~1.4 GiB 더 잡아먹어서 16 GiB GPU 에서 OOM 났습니다.
2. **autograd 가 깨졌어요.** unsloth 같은 메모리 절약 기법이 backward 를 부분적으로
   다시 돌리는데, cond 캐시 텐서를 잘못 건드려서 코드가 불안정했습니다.

### 2차 리팩터 (Two-stream, `3a551f5`)

저녁에 공식 EasyControl 레퍼런스 코드를 다시 보고 구조를 갈아엎었습니다. 핵심은
**"블록 forward 한 번 안에서 target 과 cond 를 동시에 처리"**:

```
                 한 블록 안에서

target stream                          cond stream (t=0 으로 처리)
─────────────                          ─────────────
self-attention                         self-attention
  ↑                                     ↑
  └── target Q 가 [target K; cond K]   └── cond Q 는 자기 K/V 만 봄
      를 같이 봄 (LSE 분해로 효율 챙김)

cross-attention (text)                 (cross-attn 생략)

mlp                                    mlp + cond LoRA
   ↓                                       ↓
다음 블록                              다음 블록
```

핵심 트릭 몇 개를 풀어보면:

- **`b_cond` 로 학습 초기 안정성 확보.** ref 토큰이 학습 0 step 부터 영향을 주면
  베이스 모델 동작이 깨지니까, cond logit 에 큰 음수 bias (`b_cond = -10`) 를
  걸어서 처음엔 ref 가 거의 무시되게 했습니다. 학습이 진행되면서 모델이 알아서
  `b_cond` 를 조정합니다. (단순히 V 를 0으로 초기화하는 방식은 attention softmax
  특성상 target 출력을 절반으로 깎아버리는 함정이 있어요.)
- **LSE 분해 attention.** target Q 가 `[target K; cond K]` 를 같이 보면 attention
  matrix 크기가 `(S_t + S_c)²` 로 커지는데, 이걸 직접 만들지 않고 **두 key 묶음에
  대해 flash-attention 을 따로 돌린 뒤 LSE 산술로 합치는** 방식으로 처리했습니다.
  Backward 도 직접 작성. (FA2 의 backward 가 일부 grad 를 떨어뜨려서 그렇습니다.)
- **Cond LoRA 는 cond stream 에만.** target 과 cond 가 별도 텐서니까 cond_x 에만
  LoRA delta 를 더하면 됩니다. mask trick 같은 거 필요 없음.

### 결과

| 구성                                       | Peak GPU memory |
|--------------------------------------------|----------------:|
| Baseline DiT only (no cond)                | ~5.0 GiB        |
| Two-stream, `cond_token_count=1024`        | ~5.4 GiB        |
| Two-stream, `cond_token_count=4096`        | ~6.3 GiB        |

실제 학습 step 은 16 GiB GPU 에서 **~7.8 GiB** 정도. Phase 1.5 가 같은 조건에서
OOM 났던 걸 생각하면 큰 개선입니다.

### 사용법

```bash
make easycontrol                          # 학습
make test-easycontrol REF_IMAGE=foo.png PROMPT="..."
```

VAE latent 캐시를 cond 입력으로 그대로 재사용하기 때문에 별도 사이드카 캐시는
필요없습니다. 자세한 설명은 `docs/methods/easycontrol.md` 에 있습니다.

---

## 정리 — IP-Adapter vs EasyControl, 뭐가 다른가?

오늘 두 개를 같이 넣어서 헷갈리실 수 있는데, 잘하는 게 다릅니다:

| 항목              | IP-Adapter                                  | EasyControl                                  |
|-------------------|--------------------------------------------|----------------------------------------------|
| 어디에 붙나       | **cross-attention** (텍스트 옆에)           | **self-attention** (이미지 토큰 옆에)         |
| Ref 인코더        | PE-Core (외부 비전 인코더)                  | DiT 자기 자신 (VAE latent 그대로)             |
| Ref 토큰 수       | 16 개 (압축)                                | ~4096 개 (공간 정보 유지)                     |
| 학습 파라미터     | ~150M                                       | ~14.7M                                       |
| 잘하는 일         | **identity / style** ("이 캐릭터", "이 그림체") | **구도 / 구조** ("이 포즈", "이 레이아웃")    |

한 줄로 말하면:

- **IP-Adapter** = "이 캐릭터로 다른 장면 그려줘"
- **EasyControl** = "이 구도/포즈로 다른 캐릭터 그려줘"

두 개를 같이 쓰는 것도 구조적으로는 가능합니다 (서로 다른 attention 자리에 붙으니까).
아직 검증은 안 해봤어요.

---

읽어주셔서 감사합니다. 질문/피드백 환영합니다 🌱
