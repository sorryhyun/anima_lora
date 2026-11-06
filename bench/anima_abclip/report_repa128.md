# blind A/B — s40_REPA128

s37 / s38 at the shipped step: repa = the s31 / s32 nomask LoRAs (alpha 128, lr 5e-5, relational REPA 0.05 at layer 8); norepa = the s39 shift1 LoRAs (the same with use_repa off); norepa rescaled to repa's median ||dW||

| arm | wins |
|---|---:|
| repa | 23 |
| norepa | 22 |
| blank | 3 |

Binomial sign test, norepa wins 22/45: one-sided p = 0.6170

Cluster sign-flip over 6 (artist, seed) pairs, norepa − repa = -1: one-sided p = 0.6875

| artist | seed | repa | norepa |
|---|---|---:|---:|
| cast | s42 | 5 | 2 |
| cast | s43 | 4 | 4 |
| cast | s44 | 3 | 4 |
| sweet | s42 | 4 | 4 |
| sweet | s43 | 4 | 3 |
| sweet | s44 | 3 | 5 |

| artist | prompt | repa | norepa |
|---|---|---:|---:|
| cast | p0 | 2 | 4 |
| cast | p1 | 2 | 2 |
| cast | p2 | 4 | 2 |
| cast | p3 | 4 | 2 |
| sweet | p0 | 2 | 3 |
| sweet | p1 | 4 | 2 |
| sweet | p2 | 1 | 5 |
| sweet | p3 | 4 | 2 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s40_REPA128_cast | p01 | s44 | p3 | repa | norepa | A |
| s40_REPA128_cast | p02 | s44 | p2 | norepa | repa | A |
| s40_REPA128_cast | p03 | s44 | p0 | repa | norepa | B |
| s40_REPA128_cast | p04 | s43 | p1 | repa | norepa | A |
| s40_REPA128_cast | p05 | s43 | p2 | norepa | repa | A |
| s40_REPA128_cast | p06 | s42 | p2 | repa | norepa | A |
| s40_REPA128_cast | p07 | s44 | p3 | repa | norepa | B |
| s40_REPA128_cast | p08 | s42 | p1 | norepa | repa |  |
| s40_REPA128_cast | p09 | s43 | p2 | repa | norepa | A |
| s40_REPA128_cast | p10 | s42 | p1 | norepa | repa | A |
| s40_REPA128_cast | p11 | s43 | p0 | repa | norepa | A |
| s40_REPA128_cast | p12 | s42 | p0 | repa | norepa | A |
| s40_REPA128_cast | p13 | s44 | p1 | repa | norepa | A |
| s40_REPA128_cast | p14 | s43 | p3 | norepa | repa | B |
| s40_REPA128_cast | p15 | s44 | p2 | repa | norepa | A |
| s40_REPA128_cast | p16 | s43 | p1 | norepa | repa | A |
| s40_REPA128_cast | p17 | s42 | p2 | repa | norepa | A |
| s40_REPA128_cast | p18 | s42 | p3 | norepa | repa | B |
| s40_REPA128_cast | p19 | s43 | p0 | norepa | repa | A |
| s40_REPA128_cast | p20 | s42 | p3 | repa | norepa | A |
| s40_REPA128_cast | p21 | s42 | p0 | repa | norepa | B |
| s40_REPA128_cast | p22 | s44 | p0 | repa | norepa | B |
| s40_REPA128_cast | p23 | s44 | p1 | repa | norepa |  |
| s40_REPA128_cast | p24 | s43 | p3 | repa | norepa | B |
| s40_REPA128_sweet | p01 | s44 | p3 | repa | norepa | B |
| s40_REPA128_sweet | p02 | s44 | p1 | norepa | repa | B |
| s40_REPA128_sweet | p03 | s43 | p0 | norepa | repa |  |
| s40_REPA128_sweet | p04 | s44 | p0 | repa | norepa | A |
| s40_REPA128_sweet | p05 | s43 | p0 | norepa | repa | A |
| s40_REPA128_sweet | p06 | s43 | p1 | repa | norepa | B |
| s40_REPA128_sweet | p07 | s42 | p0 | norepa | repa | A |
| s40_REPA128_sweet | p08 | s42 | p1 | norepa | repa | B |
| s40_REPA128_sweet | p09 | s43 | p2 | norepa | repa | B |
| s40_REPA128_sweet | p10 | s44 | p3 | norepa | repa | B |
| s40_REPA128_sweet | p11 | s42 | p1 | norepa | repa | B |
| s40_REPA128_sweet | p12 | s42 | p0 | norepa | repa | B |
| s40_REPA128_sweet | p13 | s43 | p1 | repa | norepa | A |
| s40_REPA128_sweet | p14 | s44 | p1 | norepa | repa | A |
| s40_REPA128_sweet | p15 | s42 | p2 | repa | norepa | B |
| s40_REPA128_sweet | p16 | s44 | p2 | repa | norepa | B |
| s40_REPA128_sweet | p17 | s42 | p2 | repa | norepa | B |
| s40_REPA128_sweet | p18 | s42 | p3 | repa | norepa | B |
| s40_REPA128_sweet | p19 | s43 | p3 | repa | norepa | A |
| s40_REPA128_sweet | p20 | s44 | p2 | repa | norepa | B |
| s40_REPA128_sweet | p21 | s42 | p3 | repa | norepa | A |
| s40_REPA128_sweet | p22 | s44 | p0 | norepa | repa | A |
| s40_REPA128_sweet | p23 | s43 | p3 | repa | norepa | A |
| s40_REPA128_sweet | p24 | s43 | p2 | norepa | repa | A |
