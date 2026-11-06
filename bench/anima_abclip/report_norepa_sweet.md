# blind A/B — s38_NOREPA

s37 on sweetonedollar: repa = the s35 a512 LoRAs, norepa = the same with use_repa off; norepa rescaled to repa's median ||dW||

| arm | wins |
|---|---:|
| repa | 8 |
| norepa | 16 |

Binomial sign test, norepa wins 16/24: one-sided p = 0.0758

Cluster sign-flip over 3 (artist, seed) pairs, norepa − repa = 8: one-sided p = 0.1250

| artist | seed | repa | norepa |
|---|---|---:|---:|
| sweet | s42 | 2 | 6 |
| sweet | s43 | 3 | 5 |
| sweet | s44 | 3 | 5 |

| artist | prompt | repa | norepa |
|---|---|---:|---:|
| sweet | p0 | 4 | 2 |
| sweet | p1 | 2 | 4 |
| sweet | p2 | 0 | 6 |
| sweet | p3 | 2 | 4 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s38_NOREPA_sweet | p01 | s42 | p0 | repa | norepa | A |
| s38_NOREPA_sweet | p02 | s43 | p3 | norepa | repa | A |
| s38_NOREPA_sweet | p03 | s42 | p3 | norepa | repa | A |
| s38_NOREPA_sweet | p04 | s44 | p3 | repa | norepa | B |
| s38_NOREPA_sweet | p05 | s43 | p2 | norepa | repa | A |
| s38_NOREPA_sweet | p06 | s42 | p3 | repa | norepa | B |
| s38_NOREPA_sweet | p07 | s43 | p0 | repa | norepa | B |
| s38_NOREPA_sweet | p08 | s44 | p0 | repa | norepa | B |
| s38_NOREPA_sweet | p09 | s42 | p2 | norepa | repa | A |
| s38_NOREPA_sweet | p10 | s44 | p2 | repa | norepa | B |
| s38_NOREPA_sweet | p11 | s42 | p2 | norepa | repa | A |
| s38_NOREPA_sweet | p12 | s44 | p0 | norepa | repa | B |
| s38_NOREPA_sweet | p13 | s44 | p1 | norepa | repa | A |
| s38_NOREPA_sweet | p14 | s43 | p1 | norepa | repa | A |
| s38_NOREPA_sweet | p15 | s43 | p0 | repa | norepa | A |
| s38_NOREPA_sweet | p16 | s42 | p0 | norepa | repa | B |
| s38_NOREPA_sweet | p17 | s44 | p1 | norepa | repa | B |
| s38_NOREPA_sweet | p18 | s43 | p3 | repa | norepa | A |
| s38_NOREPA_sweet | p19 | s43 | p1 | norepa | repa | B |
| s38_NOREPA_sweet | p20 | s42 | p1 | repa | norepa | B |
| s38_NOREPA_sweet | p21 | s42 | p1 | repa | norepa | B |
| s38_NOREPA_sweet | p22 | s44 | p3 | repa | norepa | A |
| s38_NOREPA_sweet | p23 | s43 | p2 | norepa | repa | A |
| s38_NOREPA_sweet | p24 | s44 | p2 | repa | norepa | B |
