# blind A/B — s33_ABCLIP2

s29 without the T-LoRA mask: A = old default (2e-5 cosine, bs1, fixed clip 1.0), B = 5e-5 constant GA2, median clip k2 per micro, one window (no σ bands); both use_timestep_mask off, EMA 0.1; B rescaled to A's median ||dW||

| arm | wins |
|---|---:|
| A | 52 |
| B | 42 |
| tie | 2 |

Binomial sign test, B wins 42/94: one-sided p = 0.8718

Cluster sign-flip over 12 (artist, seed) pairs, B − A = -10: one-sided p = 0.8828

| artist | seed | A | B |
|---|---|---:|---:|
| ama | s42 | 5 | 3 |
| ama | s43 | 3 | 4 |
| ama | s44 | 6 | 2 |
| ama | s45 | 3 | 4 |
| cast | s42 | 6 | 2 |
| cast | s43 | 5 | 3 |
| cast | s44 | 3 | 5 |
| cast | s45 | 2 | 6 |
| sweet | s42 | 5 | 3 |
| sweet | s43 | 4 | 4 |
| sweet | s44 | 4 | 4 |
| sweet | s45 | 6 | 2 |

| artist | prompt | A | B |
|---|---|---:|---:|
| ama | p0 | 5 | 3 |
| ama | p1 | 4 | 4 |
| ama | p2 | 4 | 4 |
| ama | p3 | 4 | 2 |
| cast | p0 | 2 | 6 |
| cast | p1 | 3 | 5 |
| cast | p2 | 6 | 2 |
| cast | p3 | 5 | 3 |
| sweet | p0 | 6 | 2 |
| sweet | p1 | 5 | 3 |
| sweet | p2 | 3 | 5 |
| sweet | p3 | 5 | 3 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s33_ABCLIP2_ama | p01 | s45 | p1 | A | B | B |
| s33_ABCLIP2_ama | p02 | s42 | p1 | A | B | A |
| s33_ABCLIP2_ama | p03 | s43 | p1 | B | A | A |
| s33_ABCLIP2_ama | p04 | s44 | p2 | A | B | B |
| s33_ABCLIP2_ama | p05 | s42 | p0 | A | B | A |
| s33_ABCLIP2_ama | p06 | s42 | p1 | A | B | B |
| s33_ABCLIP2_ama | p07 | s43 | p3 | B | A | B |
| s33_ABCLIP2_ama | p08 | s43 | p0 | B | A | A |
| s33_ABCLIP2_ama | p09 | s42 | p3 | B | A | A |
| s33_ABCLIP2_ama | p10 | s43 | p2 | A | B | B |
| s33_ABCLIP2_ama | p11 | s42 | p0 | A | B | A |
| s33_ABCLIP2_ama | p12 | s43 | p2 | A | B | A |
| s33_ABCLIP2_ama | p13 | s44 | p3 | A | B | A |
| s33_ABCLIP2_ama | p14 | s42 | p2 | A | B | A |
| s33_ABCLIP2_ama | p15 | s45 | p2 | A | B | A |
| s33_ABCLIP2_ama | p16 | s45 | p0 | B | A | A |
| s33_ABCLIP2_ama | p17 | s44 | p1 | B | A | B |
| s33_ABCLIP2_ama | p18 | s45 | p3 | A | B | B |
| s33_ABCLIP2_ama | p19 | s44 | p3 | B | A | B |
| s33_ABCLIP2_ama | p20 | s44 | p1 | B | A | B |
| s33_ABCLIP2_ama | p21 | s45 | p0 | B | A | B |
| s33_ABCLIP2_ama | p22 | s42 | p2 | A | B | B |
| s33_ABCLIP2_ama | p23 | s44 | p0 | B | A | A |
| s33_ABCLIP2_ama | p24 | s44 | p0 | B | A | B |
| s33_ABCLIP2_ama | p25 | s45 | p3 | A | B | TIE |
| s33_ABCLIP2_ama | p26 | s43 | p3 | A | B | TIE |
| s33_ABCLIP2_ama | p27 | s43 | p0 | B | A | B |
| s33_ABCLIP2_ama | p28 | s45 | p2 | B | A | A |
| s33_ABCLIP2_ama | p29 | s45 | p1 | A | B | A |
| s33_ABCLIP2_ama | p30 | s42 | p3 | B | A | B |
| s33_ABCLIP2_ama | p31 | s44 | p2 | B | A | B |
| s33_ABCLIP2_ama | p32 | s43 | p1 | B | A | A |
| s33_ABCLIP2_sweet | p01 | s42 | p1 | B | A | A |
| s33_ABCLIP2_sweet | p02 | s42 | p2 | A | B | A |
| s33_ABCLIP2_sweet | p03 | s45 | p2 | A | B | B |
| s33_ABCLIP2_sweet | p04 | s42 | p0 | A | B | A |
| s33_ABCLIP2_sweet | p05 | s42 | p3 | A | B | A |
| s33_ABCLIP2_sweet | p06 | s43 | p0 | B | A | B |
| s33_ABCLIP2_sweet | p07 | s44 | p0 | A | B | A |
| s33_ABCLIP2_sweet | p08 | s45 | p0 | A | B | A |
| s33_ABCLIP2_sweet | p09 | s44 | p0 | A | B | A |
| s33_ABCLIP2_sweet | p10 | s43 | p3 | A | B | B |
| s33_ABCLIP2_sweet | p11 | s43 | p2 | A | B | B |
| s33_ABCLIP2_sweet | p12 | s42 | p3 | A | B | A |
| s33_ABCLIP2_sweet | p13 | s45 | p1 | B | A | B |
| s33_ABCLIP2_sweet | p14 | s43 | p1 | B | A | B |
| s33_ABCLIP2_sweet | p15 | s44 | p3 | A | B | A |
| s33_ABCLIP2_sweet | p16 | s44 | p1 | A | B | B |
| s33_ABCLIP2_sweet | p17 | s43 | p0 | B | A | A |
| s33_ABCLIP2_sweet | p18 | s44 | p2 | B | A | A |
| s33_ABCLIP2_sweet | p19 | s44 | p2 | B | A | A |
| s33_ABCLIP2_sweet | p20 | s42 | p1 | B | A | B |
| s33_ABCLIP2_sweet | p21 | s42 | p2 | B | A | A |
| s33_ABCLIP2_sweet | p22 | s44 | p3 | B | A | A |
| s33_ABCLIP2_sweet | p23 | s43 | p1 | A | B | A |
| s33_ABCLIP2_sweet | p24 | s45 | p1 | A | B | B |
| s33_ABCLIP2_sweet | p25 | s45 | p3 | A | B | A |
| s33_ABCLIP2_sweet | p26 | s42 | p0 | B | A | A |
| s33_ABCLIP2_sweet | p27 | s43 | p2 | A | B | A |
| s33_ABCLIP2_sweet | p28 | s43 | p3 | B | A | A |
| s33_ABCLIP2_sweet | p29 | s45 | p0 | B | A | B |
| s33_ABCLIP2_sweet | p30 | s44 | p1 | A | B | A |
| s33_ABCLIP2_sweet | p31 | s45 | p2 | A | B | A |
| s33_ABCLIP2_sweet | p32 | s45 | p3 | A | B | A |
| s33_ABCLIP2_cast | p01 | s44 | p3 | A | B | B |
| s33_ABCLIP2_cast | p02 | s45 | p3 | A | B | A |
| s33_ABCLIP2_cast | p03 | s44 | p2 | A | B | A |
| s33_ABCLIP2_cast | p04 | s44 | p0 | A | B | B |
| s33_ABCLIP2_cast | p05 | s43 | p2 | B | A | A |
| s33_ABCLIP2_cast | p06 | s44 | p0 | B | A | A |
| s33_ABCLIP2_cast | p07 | s43 | p1 | B | A | B |
| s33_ABCLIP2_cast | p08 | s42 | p1 | A | B | B |
| s33_ABCLIP2_cast | p09 | s44 | p2 | B | A | B |
| s33_ABCLIP2_cast | p10 | s45 | p2 | B | A | A |
| s33_ABCLIP2_cast | p11 | s42 | p2 | B | A | B |
| s33_ABCLIP2_cast | p12 | s45 | p0 | A | B | B |
| s33_ABCLIP2_cast | p13 | s42 | p3 | A | B | B |
| s33_ABCLIP2_cast | p14 | s42 | p1 | A | B | A |
| s33_ABCLIP2_cast | p15 | s42 | p2 | A | B | A |
| s33_ABCLIP2_cast | p16 | s42 | p0 | B | A | B |
| s33_ABCLIP2_cast | p17 | s45 | p1 | A | B | B |
| s33_ABCLIP2_cast | p18 | s45 | p2 | A | B | A |
| s33_ABCLIP2_cast | p19 | s45 | p1 | A | B | B |
| s33_ABCLIP2_cast | p20 | s44 | p1 | B | A | A |
| s33_ABCLIP2_cast | p21 | s43 | p2 | A | B | A |
| s33_ABCLIP2_cast | p22 | s45 | p0 | A | B | B |
| s33_ABCLIP2_cast | p23 | s43 | p3 | B | A | B |
| s33_ABCLIP2_cast | p24 | s43 | p3 | A | B | A |
| s33_ABCLIP2_cast | p25 | s42 | p0 | B | A | B |
| s33_ABCLIP2_cast | p26 | s44 | p1 | B | A | A |
| s33_ABCLIP2_cast | p27 | s43 | p0 | B | A | A |
| s33_ABCLIP2_cast | p28 | s42 | p3 | B | A | B |
| s33_ABCLIP2_cast | p29 | s43 | p0 | B | A | A |
| s33_ABCLIP2_cast | p30 | s44 | p3 | A | B | A |
| s33_ABCLIP2_cast | p31 | s45 | p3 | A | B | B |
| s33_ABCLIP2_cast | p32 | s43 | p1 | B | A | B |
