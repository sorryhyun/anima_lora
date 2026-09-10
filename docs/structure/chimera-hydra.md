# ChimeraHydra: dual-pool additive MoE on the OrthoHydra basis

Two independent HydraLoRAs glued together at the residual — that's the chimera. Each adapted `Linear` carries two complete low-rank adapters off disjoint SVD subspaces of $W_0$:

| Pool          | Size  | Routed by                          | Specializes along        |
| ------------- | ----- | ---------------------------------- | ------------------------ |
| Content       | $K_c$ | network-level ContentRouter on pooled `crossattn_emb` | prompt / style / subject |
| Frequency     | $K_f$ | network-level FreqRouter on FEI of $z_t$ | noise level / denoising stage |

The two pools' outputs are added — no multiplicative gate, no σ-band mask, no curriculum. Specialization is enforced by router-input separation: the content router only ever sees text features, the freq router only ever sees the frequency profile of the noisy latent, so each pool's experts can only differentiate along its own axis.

> Experimental — entry points are `make exp-chimera` and `make lora-gui GUI_PRESETS=chimera_hydra`; the user-facing doc is `docs/experimental/chimera-hydra.md`.

---

## 1. Why two pools

Anima's denoising flow has two roughly orthogonal sources of variance: content (which style/artist/subject the sample is from) and noise level (whether the model is doing coarse layout at high σ or texture refinement at low σ). A single HydraLoRA router has one input tensor, so it can condition on one axis or the other — the plain-Hydra cells of the three-axis surface (`router_source = "input" | "sigma" | "fei"`) each pick one and pay the symmetric price of losing the other. One $E$-way softmax cannot learn both axes; that would need 2D-indexed experts, which is exactly the staged multiplicative-gate design chimera superseded.

ChimeraHydra's structural answer: two routers, one pool each, additive composition. The content router specializes $K_c$ experts along the sample axis; the freq router specializes $K_f$ experts along the noise axis; a single forward sums both contributions, so the effective delta carries both kinds of conditioning at once. T-LoRA then masks only the content branch (§5) — an asymmetric "frequency expert always on / content expert rank-throttled" split.

---

## 2. The math

Per adapted Linear (`networks/lora_modules/chimera.py`), each pool is a full shared-A HydraLoRA on the frozen-SVD Cayley parameterization, and the two pools' subspaces are disjoint on both sides:

- Down side: the top-$2r$ right singular vectors of $W_0$ split as first $r$ → `Q_basis_c`, next $r$ → `Q_basis_f`. The two A's read orthogonal row spaces.
- Up side: the top-$(K_c{+}K_f)\cdot M$ left singular vectors split into per-expert slices `P_bases_c` / `P_bases_f`. Every content expert's column space is orthogonal to every freq expert's.

$$
\begin{aligned}
A_c &= \text{Cayley}(S_{q,c})\,Q_{\text{basis},c}, \qquad A_f = \text{Cayley}(S_{q,f})\,Q_{\text{basis},f} \\[4pt]
B_c[k] &= P_{\text{bases},c}[k]\ \text{Cayley}(S_{p,c}[k]), \qquad B_f[j] = P_{\text{bases},f}[j]\ \text{Cayley}(S_{p,f}[j]) \\[6pt]
\Delta y &= \sum_k \pi_c[k]\, B_c[k]\big(A_c x \cdot \lambda_c \cdot \text{mask}_t\big)\ +\ \sum_j \pi_f[j]\, B_f[j]\big(A_f x \cdot \lambda_f\big)
\end{aligned}
$$

Trainable: the Cayley seeds ($S_{q,\cdot}$, per-expert $S_{p,\cdot}$), the per-pool diagonal scales $\lambda_c, \lambda_f$, and optionally a per-expert singular spectrum (§4). Buffers: the frozen SVD bases plus two slot-assigned gate buffers (`_content_routing_weights`, `_freq_routing_weights`) written by the network-level routers each step.

### Centered gates — the init story

Both pools recenter their gate before use: $\pi \mapsto \pi - 1/K$ (this is the only shipped configuration). The consequence is subtle but load-bearing: a **uniform gate contributes exactly zero**, so ΔW = 0 at init comes from the routers starting uniform — which frees $\lambda$ to start *non-zero* (`lambda_init > 0`). That combination breaks the MoE cold-start deadlock cleanly: the base model is preserved exactly at step 0, yet because each expert writes into a disjoint subspace with a live λ, an infinitesimal gate perturbation already changes the output — both routers receive real gradient from the first step. (Contrast plain Hydra's zero-init experts, where the router has nothing to differentiate — `hydralora.md` §5.)

The recentering is done out-of-place so the gate buffers keep their `grad_fn` — the slot-assign contract (no `.detach()`, no `.copy_()`) is what lets $\partial \mathcal{L}/\partial \pi$ flow back through the buffers to the router parameters.

### One bmm, not two

The two branches are concatenated along the rank axis before the up-projection — `cat([lx_c, lx_f])` against `cat([comb_c, comb_f])` → a single `(B, L, 2r) @ (B, 2r, d_out)` bmm. Only one full hidden-size tensor is live at a time, and the shape is static under `torch.compile` regardless of mask state.

---

## 3. The two routers (both network-level)

There is no per-Linear router in chimera — an earlier variant routed the content pool per-Linear off the pooled rank-$r$ activation, and it was removed. Both routers live once on the `LoRANetwork` and broadcast their gate to every chimera module by reference; one Python-level write per step reaches every adapted Linear.

### 3.1 ContentRouter — pooled `crossattn_emb`

Fired once per step on the pooled post-LLMAdapter text features (the same tensor the DiT cross-attends to), with a parameterless LayerNorm on the pooled input (`content_router_layer_norm = true`). This routes by prompt content: samples whose captions describe different styles/subjects land on different content experts, uniformly across all layers. The router gets an LR boost (`network_content_router_lr_scale`, 5.0 in the bench config) since its few parameters gate the whole pool.

Why pooled text rather than something the layer sees locally? Max-pooled `crossattn_emb` clusters cleanly by artist (NMI ≈ 0.93 in the original analysis) — it is the most directly content-bearing signal in the pipeline, and being caption-derived it is exactly orthogonal to noise level, which is the separation §1 requires.

### 3.2 FreqRouter — FEI of $z_t$, and the `fei` shortcut

The freq pool is conditioned on the Frequency Energy Index: a 2-band DoG simplex $(e_\text{low}, e_\text{high})$ computed from the noisy latent each step (`library/runtime/fei.py`), which varies strongly and monotonically with σ.

Two modes (`freq_router_mode`):

- `"fei"` (the shipped default) — no learned router at all. The FEI simplex is the gate: $\pi_f = \text{normalize}(\text{FEI}^{1/\tau})$, broadcast directly ($K_f$ must equal `fei_feature_dim = 2`). Motivation: archived FEI traces showed FEI already carries the load-bearing routing signal, the learned MLP only reshaped it, and the σ-feature half was non-discriminating. No router → no freq balance loss, no cold-start risk on that side.
- `"learned"` — the paper-faithful `FreqRouter` MLP over `concat(FEI, sinusoidal-σ)`. Its output Linear init is `N(0, 0.1)` — non-zero is load-bearing: a zero-init network-level router would start exactly uniform with no per-layer noise to break the tie, and under centered gates a uniform freq gate contributes zero forever.

### 3.3 Routing scope — most Linears are not chimera

`router_targets` (default regex `.*(output_proj|mlp\.layer[12])$`) decides which Linears become chimera leaves. Everything else falls back to plain OrthoLoRA (single-pool Cayley, no router). So a "chimera checkpoint" is really OrthoLoRA everywhere + dual-pool MoE on the attention outputs and MLPs — the layers where per-sample steering pays.

---

## 4. Per-expert capacity levers

Frozen disjoint slices guarantee experts can't collapse into each other, but a frozen $r$-slice is also a tight box. Two levers deepen each expert without giving up disjointness (both in the bench config; both require the Cayley form):

- `chimera_expert_basis_mult = m` — each expert gets an *over-complete* $(d_\text{out}, m\cdot r)$ frozen pool from a disjoint U-slice plus an $m r \times m r$ Cayley rotation; the forward selects an $r$-dim Stiefel subspace within the pool. The expert's column space becomes trainable while staying disjoint across experts. Benched as the big lever (~×700 reach onto an off-slice target at $m=2$); the bench config ships $m=4$.
- `chimera_expert_diag` — a per-expert trainable $(K, r)$ singular spectrum, the piece the orthogonal-only parameterization lacks. Minor on its own (~×1.15).

The alternative is `use_ortho_init = true`, which swaps every frozen-basis+Cayley for trainable SVD-seeded bases (`ortholora.md` §2) — maximum expressivity, but the experts' subspaces are then free to drift toward each other (collapse observed ~4k steps in benching), which is exactly what the levers above avoid.

---

## 5. T-LoRA per-branch composition

When `use_timestep_mask = true`, the rank mask from `timestep-mask.md` applies to the content branch only — folded into $\lambda_c$; the freq branch keeps full rank at every $t$.

The argument: T-LoRA exists to throttle high-σ memorization of layout/identity, which is precisely the content branch's risk surface. The freq branch *wants* capacity at high σ — learning coarse-stage behavior is its whole job. Because the pools are physically separate, this asymmetric composition costs nothing: the mask is a broadcast multiply on one branch's bottleneck.

---

## 6. Balance loss

Only the content pool needs balance pressure (the shipped `fei` freq mode has no router to collapse). The content pool uses an EMA-usage load balance rather than the Switch loss: a running estimate of per-expert usage whose penalty is $O(1)$ at uniform usage and grows toward $K_c$ at full collapse. Weight `balance_w_content` (1e-3 in the bench config, 2e-6 in the GUI variant; the outer `balance_loss_weight` stays 1.0 so the per-pool weight is the only effective scalar).

Diagnostics run through the chimera-aware `get_chimera_router_stats` (per-pool gate entropy normalizes by $\log K_\text{pool}$, not $\log E$). The failure mode to watch in the first 1k steps: content gate entropy pinned at ~1.0 means the ContentRouter found no prompt-side signal to differentiate on.

---

## 7. File format — save distills, load re-hydrates

Save (`ChimeraHydraLoRAModule.distill_save_state_dict` → `networks/lora_save.py`) folds the Cayley/OrthoInit state into a free-form dual-pool layout — per-pool `lora_down_c` / `lora_down_f` plus `lora_ups_c.{k}` / `lora_ups_f.{j}` — and writes it as a sibling `*_chimera.safetensors` with fused attention projections defused to `q/k/v_proj`. Metadata stamps `ss_use_chimera_hydra = "true"` plus the pool sizes and router config.

Load: the metadata sniff routes stamped files to `ChimeraHydraInferenceModule` (`networks/lora_anima/factory.py`) — a distilled dual-A runtime form; the loader requires the dual-A keys and raises otherwise (the old collapse-to-HydraLoRA fallback was removed). The Cayley classes are training-only; checkpoint metadata carries everything needed to re-instantiate routing (the three-axis fields are auto-pinned to `("shared_A", true, "input")` whenever `use_chimera_hydra = true`, so no parallel discrimination path exists). The ComfyUI `comfyui-hydralora` node uses the same sniff.

---

## 8. Composition

| Stacks with             | How it composes                                                                                     |
| ----------------------- | ---------------------------------------------------------------------------------------------------- |
| T-LoRA                  | Per-branch — mask on content λ, freq always full rank. Built-in (§5).                                |
| OrthoLoRA               | It is the substrate — chimera leaves are dual-pool Cayley; non-matched Linears are plain OrthoLoRA. |
| OrthoInit               | `use_ortho_init=true` swaps frozen bases for trainable ones in both pools (§4).                      |
| Spectrum                | Cached steps skip the blocks — routers simply don't fire on those steps.                             |
| Static merge to DiT     | ❌ Sample-dependent gates can't fold into a Linear weight.                                           |

---

## 9. Configuration

`configs/methods/chimera.toml` (canonical bench, `make exp-chimera`) — the load-bearing subset:

```toml
use_chimera_hydra   = true
num_experts_content = 6
num_experts_freq    = 2        # must equal fei_feature_dim in "fei" mode

use_ortho                 = true
chimera_expert_basis_mult = 4      # over-complete disjoint expert pools (§4)
chimera_expert_diag       = true   # per-expert trainable singular spectrum

freq_router_mode = "fei"       # π_f = normalize(FEI^{1/τ}) — no learned freq router
freq_router_tau  = 1.0
fei_feature_dim  = 2

balance_loss_weight = 1.0
balance_w_content   = 1e-3     # EMA-usage balance, content pool only
network_content_router_lr_scale = 5.0
content_router_layer_norm = true

router_targets = ".*(output_proj|mlp\\.layer[12])$"   # chimera leaves; rest = OrthoLoRA
```

`configs/gui-methods/chimera_hydra.toml` is the GUI variant — same activation, $K_c = 4$, $K_f = 2$, `balance_w_content = 2e-6`, freq mode `fei`.

---

## 10. Minimal mental model

1. Two complete HydraLoRAs per Linear, on disjoint SVD subspaces of $W_0$ — disjoint on both the down (row-space) and up (column-space) sides. Outputs add.
2. Both routers are network-level: content = ContentRouter on pooled `crossattn_emb` (prompt axis), freq = raw FEI passthrough on $z_t$ (noise axis). Router-input separation is the specialization guarantee.
3. Gates are centered ($\pi - 1/K$): uniform gate ⇒ zero contribution ⇒ ΔW = 0 at init with λ alive — routers get gradient from step 0 without an expert-symmetry deadlock.
4. T-LoRA masks the content branch only; the freq branch is always full rank.
5. Only attention-output and MLP Linears are chimera leaves (`router_targets`); everything else is plain OrthoLoRA.
6. Saves as a distilled dual-A `*_chimera.safetensors` sibling; loads as `ChimeraHydraInferenceModule` keyed off the `ss_use_chimera_hydra` metadata stamp.
