# blind A/B — s32_SIGSHIFT

shift1 = the shipped lora.toml (B + σ-band clip, no mask; σ ~ logit-normal(0, 1)); shift2 = the same with timestep_sampling shift, discrete_flow_shift 2 (σ >= 0.8 draws 7.4 % -> 24 %); shift2 rescaled to shift1's median ||dW||

| arm | wins |
|---|---:|
| shift1 | 22 |
| shift2 | 23 |
| tie | 3 |

Binomial sign test, shift2 wins 23/45: one-sided p = 0.5000

Cluster sign-flip over 6 (artist, seed) pairs, shift2 − shift1 = 1: one-sided p = 0.5000

| artist | seed | shift1 | shift2 |
|---|---|---:|---:|
| cast | s42 | 3 | 4 |
| cast | s43 | 2 | 5 |
| cast | s44 | 3 | 5 |
| sweet | s42 | 6 | 2 |
| sweet | s43 | 5 | 3 |
| sweet | s44 | 3 | 4 |

| artist | prompt | shift1 | shift2 |
|---|---|---:|---:|
| cast | p0 | 3 | 3 |
| cast | p1 | 1 | 4 |
| cast | p2 | 2 | 3 |
| cast | p3 | 2 | 4 |
| sweet | p0 | 4 | 2 |
| sweet | p1 | 5 | 0 |
| sweet | p2 | 3 | 3 |
| sweet | p3 | 2 | 4 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s32_SIGSHIFT_cast | p01 | s44 | p3 | shift2 | shift1 | B |
| s32_SIGSHIFT_cast | p02 | s44 | p1 | shift2 | shift1 | B |
| s32_SIGSHIFT_cast | p03 | s43 | p1 | shift2 | shift1 | TIE |
| s32_SIGSHIFT_cast | p04 | s44 | p2 | shift1 | shift2 | B |
| s32_SIGSHIFT_cast | p05 | s44 | p3 | shift1 | shift2 | B |
| s32_SIGSHIFT_cast | p06 | s44 | p0 | shift1 | shift2 | B |
| s32_SIGSHIFT_cast | p07 | s44 | p2 | shift1 | shift2 | A |
| s32_SIGSHIFT_cast | p08 | s43 | p1 | shift2 | shift1 | A |
| s32_SIGSHIFT_cast | p09 | s43 | p2 | shift1 | shift2 | B |
| s32_SIGSHIFT_cast | p10 | s44 | p1 | shift2 | shift1 | A |
| s32_SIGSHIFT_cast | p11 | s43 | p2 | shift1 | shift2 | A |
| s32_SIGSHIFT_cast | p12 | s44 | p0 | shift1 | shift2 | B |
| s32_SIGSHIFT_cast | p13 | s43 | p3 | shift2 | shift1 | A |
| s32_SIGSHIFT_cast | p14 | s43 | p0 | shift2 | shift1 | B |
| s32_SIGSHIFT_cast | p15 | s42 | p2 | shift1 | shift2 | B |
| s32_SIGSHIFT_cast | p16 | s42 | p1 | shift2 | shift1 | A |
| s32_SIGSHIFT_cast | p17 | s42 | p0 | shift2 | shift1 | B |
| s32_SIGSHIFT_cast | p18 | s42 | p0 | shift2 | shift1 | B |
| s32_SIGSHIFT_cast | p19 | s43 | p3 | shift1 | shift2 | B |
| s32_SIGSHIFT_cast | p20 | s42 | p3 | shift1 | shift2 | B |
| s32_SIGSHIFT_cast | p21 | s43 | p0 | shift1 | shift2 | B |
| s32_SIGSHIFT_cast | p22 | s42 | p2 | shift2 | shift1 | TIE |
| s32_SIGSHIFT_cast | p23 | s42 | p3 | shift2 | shift1 | B |
| s32_SIGSHIFT_cast | p24 | s42 | p1 | shift2 | shift1 | A |
| s32_SIGSHIFT_sweet | p01 | s44 | p3 | shift2 | shift1 | B |
| s32_SIGSHIFT_sweet | p02 | s42 | p2 | shift1 | shift2 | A |
| s32_SIGSHIFT_sweet | p03 | s44 | p1 | shift1 | shift2 | TIE |
| s32_SIGSHIFT_sweet | p04 | s42 | p1 | shift1 | shift2 | A |
| s32_SIGSHIFT_sweet | p05 | s44 | p0 | shift1 | shift2 | B |
| s32_SIGSHIFT_sweet | p06 | s43 | p2 | shift2 | shift1 | B |
| s32_SIGSHIFT_sweet | p07 | s44 | p2 | shift1 | shift2 | B |
| s32_SIGSHIFT_sweet | p08 | s43 | p2 | shift2 | shift1 | A |
| s32_SIGSHIFT_sweet | p09 | s44 | p2 | shift2 | shift1 | A |
| s32_SIGSHIFT_sweet | p10 | s43 | p0 | shift1 | shift2 | A |
| s32_SIGSHIFT_sweet | p11 | s42 | p2 | shift2 | shift1 | B |
| s32_SIGSHIFT_sweet | p12 | s43 | p0 | shift1 | shift2 | B |
| s32_SIGSHIFT_sweet | p13 | s42 | p1 | shift2 | shift1 | B |
| s32_SIGSHIFT_sweet | p14 | s43 | p3 | shift2 | shift1 | A |
| s32_SIGSHIFT_sweet | p15 | s42 | p3 | shift1 | shift2 | B |
| s32_SIGSHIFT_sweet | p16 | s43 | p1 | shift1 | shift2 | A |
| s32_SIGSHIFT_sweet | p17 | s43 | p1 | shift1 | shift2 | A |
| s32_SIGSHIFT_sweet | p18 | s44 | p3 | shift2 | shift1 | A |
| s32_SIGSHIFT_sweet | p19 | s44 | p0 | shift1 | shift2 | A |
| s32_SIGSHIFT_sweet | p20 | s42 | p0 | shift2 | shift1 | B |
| s32_SIGSHIFT_sweet | p21 | s43 | p3 | shift1 | shift2 | A |
| s32_SIGSHIFT_sweet | p22 | s42 | p0 | shift1 | shift2 | A |
| s32_SIGSHIFT_sweet | p23 | s44 | p1 | shift1 | shift2 | A |
| s32_SIGSHIFT_sweet | p24 | s42 | p3 | shift1 | shift2 | B |
