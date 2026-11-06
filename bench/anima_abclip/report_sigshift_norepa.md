# blind A/B — s39_SIGSHIFT_NOREPA

s32 without REPA: shift1 = the shipped lora.toml (alpha 128, lr 5e-5, B + σ-band clip, no mask, REPA off; σ ~ logit-normal(0, 1)); shift2 = the same with timestep_sampling shift, discrete_flow_shift 2; shift2 rescaled to shift1's median ||dW||

| arm | wins |
|---|---:|
| shift1 | 28 |
| shift2 | 20 |

Binomial sign test, shift2 wins 20/48: one-sided p = 0.9033

Cluster sign-flip over 6 (artist, seed) pairs, shift2 − shift1 = -8: one-sided p = 0.9688

| artist | seed | shift1 | shift2 |
|---|---|---:|---:|
| cast | s42 | 3 | 5 |
| cast | s43 | 5 | 3 |
| cast | s44 | 4 | 4 |
| sweet | s42 | 6 | 2 |
| sweet | s43 | 5 | 3 |
| sweet | s44 | 5 | 3 |

| artist | prompt | shift1 | shift2 |
|---|---|---:|---:|
| cast | p0 | 3 | 3 |
| cast | p1 | 2 | 4 |
| cast | p2 | 3 | 3 |
| cast | p3 | 4 | 2 |
| sweet | p0 | 3 | 3 |
| sweet | p1 | 4 | 2 |
| sweet | p2 | 6 | 0 |
| sweet | p3 | 3 | 3 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s39_SIGSHIFT_NOREPA_cast | p01 | s43 | p0 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_cast | p02 | s44 | p1 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_cast | p03 | s43 | p3 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_cast | p04 | s44 | p0 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_cast | p05 | s42 | p1 | shift1 | shift2 | B |
| s39_SIGSHIFT_NOREPA_cast | p06 | s44 | p1 | shift1 | shift2 | B |
| s39_SIGSHIFT_NOREPA_cast | p07 | s43 | p3 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_cast | p08 | s43 | p2 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_cast | p09 | s42 | p0 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_cast | p10 | s42 | p2 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_cast | p11 | s44 | p0 | shift1 | shift2 | B |
| s39_SIGSHIFT_NOREPA_cast | p12 | s42 | p1 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_cast | p13 | s43 | p1 | shift1 | shift2 | B |
| s39_SIGSHIFT_NOREPA_cast | p14 | s42 | p2 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_cast | p15 | s43 | p1 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_cast | p16 | s44 | p3 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_cast | p17 | s44 | p2 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_cast | p18 | s44 | p2 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_cast | p19 | s42 | p3 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_cast | p20 | s44 | p3 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_cast | p21 | s42 | p0 | shift1 | shift2 | B |
| s39_SIGSHIFT_NOREPA_cast | p22 | s43 | p2 | shift1 | shift2 | B |
| s39_SIGSHIFT_NOREPA_cast | p23 | s43 | p0 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_cast | p24 | s42 | p3 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_sweet | p01 | s43 | p0 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_sweet | p02 | s43 | p3 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_sweet | p03 | s44 | p3 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_sweet | p04 | s44 | p2 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_sweet | p05 | s44 | p3 | shift1 | shift2 | B |
| s39_SIGSHIFT_NOREPA_sweet | p06 | s44 | p2 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_sweet | p07 | s42 | p3 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_sweet | p08 | s43 | p2 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_sweet | p09 | s44 | p0 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_sweet | p10 | s43 | p2 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_sweet | p11 | s43 | p3 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_sweet | p12 | s42 | p1 | shift1 | shift2 | B |
| s39_SIGSHIFT_NOREPA_sweet | p13 | s42 | p0 | shift1 | shift2 | B |
| s39_SIGSHIFT_NOREPA_sweet | p14 | s42 | p0 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_sweet | p15 | s42 | p1 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_sweet | p16 | s43 | p0 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_sweet | p17 | s43 | p1 | shift2 | shift1 | A |
| s39_SIGSHIFT_NOREPA_sweet | p18 | s43 | p1 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_sweet | p19 | s44 | p1 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_sweet | p20 | s42 | p2 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_sweet | p21 | s42 | p2 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_sweet | p22 | s44 | p1 | shift2 | shift1 | B |
| s39_SIGSHIFT_NOREPA_sweet | p23 | s42 | p3 | shift1 | shift2 | A |
| s39_SIGSHIFT_NOREPA_sweet | p24 | s44 | p0 | shift2 | shift1 | B |
