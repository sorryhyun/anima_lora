# blind A/B — s29_ABCLIP

A = default (2e-5 cosine bs1 clip 1.0), B = 5e-5 constant GA2 median k2 per micro

| arm | wins |
|---|---:|
| A | 16 |
| B | 41 |
| tie | 7 |

Binomial sign test, B wins 41/57: one-sided p = 0.0006

Cluster sign-flip over 8 (artist, seed) pairs, B − A = 25: one-sided p = 0.0078

| artist | seed | A | B |
|---|---|---:|---:|
| ama | s42 | 2 | 5 |
| ama | s43 | 3 | 3 |
| ama | s44 | 2 | 6 |
| ama | s45 | 1 | 5 |
| cast | s42 | 3 | 5 |
| cast | s43 | 2 | 6 |
| cast | s44 | 1 | 6 |
| cast | s45 | 2 | 5 |

| artist | prompt | A | B |
|---|---|---:|---:|
| ama | p0 | 3 | 4 |
| ama | p1 | 0 | 5 |
| ama | p2 | 4 | 3 |
| ama | p3 | 1 | 7 |
| cast | p0 | 1 | 7 |
| cast | p1 | 4 | 3 |
| cast | p2 | 0 | 8 |
| cast | p3 | 3 | 4 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s29_ABCLIP_cast | p01 | s44 | p3 | B | A | TIE |
| s29_ABCLIP_cast | p02 | s44 | p2 | B | A | A |
| s29_ABCLIP_cast | p03 | s45 | p0 | B | A | B |
| s29_ABCLIP_cast | p04 | s45 | p3 | A | B | B |
| s29_ABCLIP_cast | p05 | s43 | p1 | B | A | A |
| s29_ABCLIP_cast | p06 | s43 | p3 | B | A | B |
| s29_ABCLIP_cast | p07 | s44 | p0 | B | A | A |
| s29_ABCLIP_cast | p08 | s44 | p2 | B | A | A |
| s29_ABCLIP_cast | p09 | s44 | p1 | B | A | A |
| s29_ABCLIP_cast | p10 | s42 | p3 | B | A | A |
| s29_ABCLIP_cast | p11 | s43 | p0 | B | A | A |
| s29_ABCLIP_cast | p12 | s45 | p1 | B | A | A |
| s29_ABCLIP_cast | p13 | s42 | p2 | A | B | B |
| s29_ABCLIP_cast | p14 | s45 | p0 | A | B | B |
| s29_ABCLIP_cast | p15 | s45 | p2 | A | B | B |
| s29_ABCLIP_cast | p16 | s44 | p3 | B | A | A |
| s29_ABCLIP_cast | p17 | s42 | p3 | A | B | A |
| s29_ABCLIP_cast | p18 | s43 | p3 | A | B | B |
| s29_ABCLIP_cast | p19 | s42 | p1 | A | B | A |
| s29_ABCLIP_cast | p20 | s42 | p0 | A | B | B |
| s29_ABCLIP_cast | p21 | s42 | p0 | A | B | B |
| s29_ABCLIP_cast | p22 | s43 | p2 | A | B | B |
| s29_ABCLIP_cast | p23 | s43 | p2 | A | B | B |
| s29_ABCLIP_cast | p24 | s45 | p3 | A | B | A |
| s29_ABCLIP_cast | p25 | s44 | p0 | B | A | A |
| s29_ABCLIP_cast | p26 | s42 | p1 | A | B | A |
| s29_ABCLIP_cast | p27 | s43 | p0 | A | B | B |
| s29_ABCLIP_cast | p28 | s45 | p2 | B | A | A |
| s29_ABCLIP_cast | p29 | s45 | p1 | A | B | TIE |
| s29_ABCLIP_cast | p30 | s44 | p1 | A | B | A |
| s29_ABCLIP_cast | p31 | s43 | p1 | A | B | A |
| s29_ABCLIP_cast | p32 | s42 | p2 | A | B | B |
| s29_ABCLIP_ama | p01 | s44 | p3 | B | A | A |
| s29_ABCLIP_ama | p02 | s45 | p3 | B | A | A |
| s29_ABCLIP_ama | p03 | s43 | p3 | A | B | B |
| s29_ABCLIP_ama | p04 | s45 | p2 | A | B | TIE |
| s29_ABCLIP_ama | p05 | s44 | p2 | B | A | B |
| s29_ABCLIP_ama | p06 | s44 | p3 | A | B | B |
| s29_ABCLIP_ama | p07 | s44 | p0 | A | B | B |
| s29_ABCLIP_ama | p08 | s43 | p2 | A | B | B |
| s29_ABCLIP_ama | p09 | s43 | p1 | A | B | TIE |
| s29_ABCLIP_ama | p10 | s42 | p1 | B | A | A |
| s29_ABCLIP_ama | p11 | s43 | p0 | B | A | B |
| s29_ABCLIP_ama | p12 | s42 | p2 | A | B | A |
| s29_ABCLIP_ama | p13 | s42 | p3 | B | A | A |
| s29_ABCLIP_ama | p14 | s42 | p2 | B | A | B |
| s29_ABCLIP_ama | p15 | s43 | p3 | A | B | A |
| s29_ABCLIP_ama | p16 | s42 | p0 | B | A | A |
| s29_ABCLIP_ama | p17 | s45 | p3 | B | A | A |
| s29_ABCLIP_ama | p18 | s45 | p0 | A | B | A |
| s29_ABCLIP_ama | p19 | s45 | p0 | A | B | B |
| s29_ABCLIP_ama | p20 | s43 | p0 | A | B | A |
| s29_ABCLIP_ama | p21 | s44 | p1 | B | A | A |
| s29_ABCLIP_ama | p22 | s42 | p1 | A | B | B |
| s29_ABCLIP_ama | p23 | s44 | p1 | B | A | A |
| s29_ABCLIP_ama | p24 | s42 | p0 | A | B | TIE |
| s29_ABCLIP_ama | p25 | s45 | p1 | A | B | TIE |
| s29_ABCLIP_ama | p26 | s43 | p1 | B | A | TIE |
| s29_ABCLIP_ama | p27 | s42 | p3 | B | A | A |
| s29_ABCLIP_ama | p28 | s44 | p0 | B | A | A |
| s29_ABCLIP_ama | p29 | s45 | p2 | A | B | B |
| s29_ABCLIP_ama | p30 | s45 | p1 | A | B | B |
| s29_ABCLIP_ama | p31 | s44 | p2 | B | A | B |
| s29_ABCLIP_ama | p32 | s43 | p2 | A | B | B |
