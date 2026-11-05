# blind A/B — s30_SIGBAND

global = recipe B (median clip k2 per micro, one window), band = B with one window per σ band (0.2 / 0.4 / 0.6 / 0.8); band rescaled to global's median ||dW||

| arm | wins |
|---|---:|
| global | 21 |
| band | 39 |
| tie | 4 |

Binomial sign test, band wins 39/60: one-sided p = 0.0137

Cluster sign-flip over 8 (artist, seed) pairs, band − global = 18: one-sided p = 0.0469

| artist | seed | global | band |
|---|---|---:|---:|
| cast | s42 | 2 | 5 |
| cast | s43 | 3 | 4 |
| cast | s44 | 0 | 8 |
| cast | s45 | 5 | 3 |
| sweet | s42 | 4 | 4 |
| sweet | s43 | 1 | 6 |
| sweet | s44 | 3 | 4 |
| sweet | s45 | 3 | 5 |

| artist | prompt | global | band |
|---|---|---:|---:|
| cast | p0 | 4 | 4 |
| cast | p1 | 3 | 4 |
| cast | p2 | 2 | 6 |
| cast | p3 | 1 | 6 |
| sweet | p0 | 3 | 5 |
| sweet | p1 | 4 | 2 |
| sweet | p2 | 3 | 5 |
| sweet | p3 | 1 | 7 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s30_SIGBAND_cast | p01 | s43 | p1 | band | global | A |
| s30_SIGBAND_cast | p02 | s43 | p1 | band | global | B |
| s30_SIGBAND_cast | p03 | s45 | p2 | global | band | A |
| s30_SIGBAND_cast | p04 | s44 | p2 | band | global | A |
| s30_SIGBAND_cast | p05 | s43 | p3 | global | band | TIE |
| s30_SIGBAND_cast | p06 | s44 | p0 | global | band | B |
| s30_SIGBAND_cast | p07 | s42 | p1 | global | band | B |
| s30_SIGBAND_cast | p08 | s43 | p2 | band | global | A |
| s30_SIGBAND_cast | p09 | s42 | p2 | global | band | A |
| s30_SIGBAND_cast | p10 | s45 | p0 | band | global | B |
| s30_SIGBAND_cast | p11 | s45 | p0 | band | global | B |
| s30_SIGBAND_cast | p12 | s43 | p0 | band | global | B |
| s30_SIGBAND_cast | p13 | s45 | p3 | global | band | B |
| s30_SIGBAND_cast | p14 | s44 | p3 | band | global | A |
| s30_SIGBAND_cast | p15 | s44 | p0 | band | global | A |
| s30_SIGBAND_cast | p16 | s42 | p3 | band | global | A |
| s30_SIGBAND_cast | p17 | s45 | p2 | global | band | B |
| s30_SIGBAND_cast | p18 | s43 | p3 | global | band | B |
| s30_SIGBAND_cast | p19 | s42 | p1 | band | global | TIE |
| s30_SIGBAND_cast | p20 | s42 | p2 | global | band | B |
| s30_SIGBAND_cast | p21 | s45 | p1 | band | global | B |
| s30_SIGBAND_cast | p22 | s44 | p3 | band | global | A |
| s30_SIGBAND_cast | p23 | s43 | p2 | band | global | A |
| s30_SIGBAND_cast | p24 | s42 | p0 | band | global | A |
| s30_SIGBAND_cast | p25 | s43 | p0 | band | global | B |
| s30_SIGBAND_cast | p26 | s42 | p3 | global | band | A |
| s30_SIGBAND_cast | p27 | s45 | p1 | band | global | B |
| s30_SIGBAND_cast | p28 | s44 | p2 | global | band | B |
| s30_SIGBAND_cast | p29 | s45 | p3 | band | global | A |
| s30_SIGBAND_cast | p30 | s42 | p0 | band | global | A |
| s30_SIGBAND_cast | p31 | s44 | p1 | global | band | B |
| s30_SIGBAND_cast | p32 | s44 | p1 | global | band | B |
| s30_SIGBAND_sweet | p01 | s43 | p0 | band | global | A |
| s30_SIGBAND_sweet | p02 | s42 | p2 | global | band | A |
| s30_SIGBAND_sweet | p03 | s45 | p2 | band | global | A |
| s30_SIGBAND_sweet | p04 | s44 | p2 | global | band | B |
| s30_SIGBAND_sweet | p05 | s45 | p1 | global | band | B |
| s30_SIGBAND_sweet | p06 | s42 | p3 | band | global | A |
| s30_SIGBAND_sweet | p07 | s45 | p3 | global | band | B |
| s30_SIGBAND_sweet | p08 | s45 | p2 | global | band | B |
| s30_SIGBAND_sweet | p09 | s44 | p3 | global | band | B |
| s30_SIGBAND_sweet | p10 | s42 | p1 | band | global | B |
| s30_SIGBAND_sweet | p11 | s44 | p3 | global | band | B |
| s30_SIGBAND_sweet | p12 | s42 | p0 | band | global | A |
| s30_SIGBAND_sweet | p13 | s44 | p0 | global | band | A |
| s30_SIGBAND_sweet | p14 | s44 | p2 | band | global | B |
| s30_SIGBAND_sweet | p15 | s44 | p1 | band | global | B |
| s30_SIGBAND_sweet | p16 | s43 | p3 | global | band | B |
| s30_SIGBAND_sweet | p17 | s43 | p2 | global | band | B |
| s30_SIGBAND_sweet | p18 | s42 | p3 | band | global | A |
| s30_SIGBAND_sweet | p19 | s42 | p0 | global | band | A |
| s30_SIGBAND_sweet | p20 | s43 | p1 | global | band | A |
| s30_SIGBAND_sweet | p21 | s42 | p1 | global | band | B |
| s30_SIGBAND_sweet | p22 | s45 | p0 | band | global | A |
| s30_SIGBAND_sweet | p23 | s43 | p2 | band | global | A |
| s30_SIGBAND_sweet | p24 | s45 | p0 | band | global | B |
| s30_SIGBAND_sweet | p25 | s42 | p2 | global | band | A |
| s30_SIGBAND_sweet | p26 | s44 | p0 | band | global | A |
| s30_SIGBAND_sweet | p27 | s43 | p3 | band | global | A |
| s30_SIGBAND_sweet | p28 | s43 | p0 | global | band | B |
| s30_SIGBAND_sweet | p29 | s43 | p1 | band | global | TIE |
| s30_SIGBAND_sweet | p30 | s45 | p1 | band | global | B |
| s30_SIGBAND_sweet | p31 | s44 | p1 | global | band | TIE |
| s30_SIGBAND_sweet | p32 | s45 | p3 | band | global | B |
