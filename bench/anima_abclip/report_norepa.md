# blind A/B — s37_NOREPA

repa = the s35 a512 LoRAs (alpha 512, lr 1.25e-5, relational REPA weight 0.05 at layer 8); norepa = the same with use_repa off; norepa rescaled to repa's median ||dW||

| arm | wins |
|---|---:|
| repa | 8 |
| norepa | 16 |

Binomial sign test, norepa wins 16/24: one-sided p = 0.0758

Cluster sign-flip over 3 (artist, seed) pairs, norepa − repa = 8: one-sided p = 0.3750

| artist | seed | repa | norepa |
|---|---|---:|---:|
| cast | s42 | 3 | 5 |
| cast | s43 | 0 | 8 |
| cast | s44 | 5 | 3 |

| artist | prompt | repa | norepa |
|---|---|---:|---:|
| cast | p0 | 1 | 5 |
| cast | p1 | 3 | 3 |
| cast | p2 | 1 | 5 |
| cast | p3 | 3 | 3 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s37_NOREPA_cast | p01 | s43 | p0 | norepa | repa | A |
| s37_NOREPA_cast | p02 | s44 | p1 | norepa | repa | B |
| s37_NOREPA_cast | p03 | s44 | p0 | repa | norepa | B |
| s37_NOREPA_cast | p04 | s43 | p3 | repa | norepa | B |
| s37_NOREPA_cast | p05 | s42 | p2 | repa | norepa | B |
| s37_NOREPA_cast | p06 | s42 | p0 | repa | norepa | B |
| s37_NOREPA_cast | p07 | s43 | p2 | repa | norepa | B |
| s37_NOREPA_cast | p08 | s44 | p3 | norepa | repa | B |
| s37_NOREPA_cast | p09 | s43 | p2 | repa | norepa | B |
| s37_NOREPA_cast | p10 | s44 | p2 | repa | norepa | A |
| s37_NOREPA_cast | p11 | s42 | p3 | repa | norepa | A |
| s37_NOREPA_cast | p12 | s42 | p2 | repa | norepa | B |
| s37_NOREPA_cast | p13 | s43 | p0 | norepa | repa | A |
| s37_NOREPA_cast | p14 | s42 | p3 | norepa | repa | B |
| s37_NOREPA_cast | p15 | s43 | p1 | repa | norepa | B |
| s37_NOREPA_cast | p16 | s42 | p1 | repa | norepa | A |
| s37_NOREPA_cast | p17 | s43 | p3 | norepa | repa | A |
| s37_NOREPA_cast | p18 | s43 | p1 | repa | norepa | B |
| s37_NOREPA_cast | p19 | s42 | p0 | repa | norepa | B |
| s37_NOREPA_cast | p20 | s44 | p0 | norepa | repa | B |
| s37_NOREPA_cast | p21 | s44 | p3 | norepa | repa | A |
| s37_NOREPA_cast | p22 | s42 | p1 | repa | norepa | B |
| s37_NOREPA_cast | p23 | s44 | p1 | repa | norepa | A |
| s37_NOREPA_cast | p24 | s44 | p2 | repa | norepa | B |
