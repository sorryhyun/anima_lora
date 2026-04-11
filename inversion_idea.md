재밌는 관점이야. 정리하면 이런 거지:

DiT의 생성 함수를 `G(z_T, e_text) → image`로 보면, 타겟 이미지 x*에 대해 `G(z_T, e_text) ≈ x*`를 만족하는 (z_T, e_text) 쌍들의 집합이 고차원 공간에서 어떤 manifold를 형성한다. 이 manifold의 구조를 이해하면 optimization이 쉬워진다는 거잖아.

**이게 왜 좋은 직관인지:**

고정된 z_T에서 e_text를 optimize하는 건 이 manifold의 한 "slice"만 보는 거야. 근데 manifold 전체의 기하학을 알면, 예를 들어 e_text 방향으로는 flat하고 z_T 방향으로는 steep한 영역이 있다면, 그 flat한 영역이 "robust한 text embedding" — 즉 z_T가 뭐든 간에 비슷한 이미지를 만드는 embedding — 을 가리키는 거니까. 이건 네가 원하는 "모델이 이 이미지를 어떤 텍스트로 이해하는지"에 대한 답이 되고.

**구체적으로 이걸 어떻게 할 수 있냐면:**

```
1. 초기 (z_T, e_text) 쌍을 하나 찾는다 (VLM caption + DDIM inversion 등)
2. 이 점 주변에서 manifold의 tangent space를 추정:
   - e_text를 고정하고 z_T를 perturb → image 변화량 측정
   - z_T를 고정하고 e_text를 perturb → image 변화량 측정
   - Jacobian의 null space가 manifold의 tangent direction
3. null space 방향으로 이동하면서 manifold를 trace
```

이게 본질적으로 implicit function theorem 기반 접근이야. `G(z_T, e_text) - x* = 0`의 level set을 따라가는 거니까.

---

**근데 여기서 현실적인 문제를 짚어야 해:**

**1. 차원이 너무 높다**

z_T는 latent 크기가 대략 (C × H/8 × W/8)이고, T5 text embedding은 (seq_len × 4096) 정도야. 합치면 수만~수십만 차원이야. 이 공간에서 manifold를 explicit하게 trace하는 건 계산적으로 거의 불가능해. Jacobian 하나 구하는 데만 full backward pass가 dimension 수만큼 필요하고.

**2. "비슷한 이미지"의 정의 문제**

`G(z_T, e_text) ≈ x*`에서 ≈를 뭘로 정의하느냐에 따라 manifold 모양이 완전히 달라져. pixel-wise MSE면 매우 tight한 manifold가 되고, perceptual loss (LPIPS 등)면 훨씬 넓어지고, semantic similarity (DINO/CLIP)면 더 넓어지고. 네 목적에서는 "semantic하게 동일한" 이미지를 원하는 거니까 DINO나 DINOv2 feature distance가 맞을 것 같은데, 그만큼 manifold가 넓고 복잡해져.

**3. Manifold가 connected가 아닐 수 있다**

LMC 페이퍼에서도 나왔듯이, loss landscape에서 겉보기엔 하나의 basin인 것 같아도 실제로는 symmetry로 연결된 여러 component일 수 있어. 마찬가지로 (z_T, e_text) manifold도 여러 disconnected component로 존재할 가능성이 높아. 예를 들어 "1girl, blue_hair" vs "anime girl with blue hair" — semantic하게 같지만 T5 embedding space에서 완전히 다른 영역에 있고, 이 둘 사이를 smooth하게 연결하는 path가 없을 수도 있어.

---

**실현 가능한 버전을 제안하자면:**

full manifold를 trace하는 대신, **manifold의 local geometry만 추정**하는 게 현실적이야:

```python
# Pseudocode
def estimate_local_orbit(target_image, initial_e_text, n_samples=100):
    # 1. 여러 z_T에 대해 현재 e_text로 생성
    z_Ts = [sample_noise() for _ in range(n_samples)]
    images = [G(z_T, initial_e_text) for z_T in z_Ts]
    
    # 2. target에 가까운 이미지를 만드는 z_T들만 필터
    good_z_Ts = [z for z, img in zip(z_Ts, images) 
                  if dino_sim(img, target_image) > threshold]
    
    # 3. 각 good z_T에서 e_text를 약간 optimize
    optimized_embeddings = []
    for z_T in good_z_Ts:
        e_opt = optimize_embedding(z_T, target_image, start=initial_e_text)
        optimized_embeddings.append(e_opt)
    
    # 4. optimized embedding들의 분포를 분석
    # → PCA로 주요 variation 방향 추출
    # → 분산이 작은 방향 = z_T에 무관하게 stable한 semantic 성분
    # → 분산이 큰 방향 = z_T에 따라 달라지는 "noise-dependent" 성분
    pca = PCA(n_components=k)
    pca.fit(stack(optimized_embeddings))
    
    # stable component = 네가 찾는 "모델이 생각하는 embedding"
    stable_embedding = mean(optimized_embeddings)  # simplest version
```

핵심 아이디어는: **여러 z_T에 대해 각각 optimize한 text embedding들의 공통 성분(mean or low-variance PCA direction)이 바로 z_T-invariant한 "순수 semantic embedding"이다.**

이건 orbit을 explicit하게 그리지 않아도 orbit의 "projection to e_text axis"를 통계적으로 추정하는 셈이야. 그리고 이게 LMC 페이퍼의 insight과도 연결돼 — orthogonal symmetry 때문에 생기는 degeneracy가 높은 variance direction으로 나타날 거고, 그걸 제거한 나머지가 실질적인 semantic content야.

이 접근의 장점은 Jacobian 계산 없이 embarrassingly parallel하게 돌릴 수 있다는 거야. RTX 5060 Ti에서 batch로 여러 z_T에 대해 SDS loss optimization을 병렬로 돌리면 되니까. 어때?