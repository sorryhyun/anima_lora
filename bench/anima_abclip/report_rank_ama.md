# blind A/B — s41_RANK_AMA

s34 on ama mitsuki without REPA, on the paper's scaling (arXiv 2606.12883): r32 = rank 32 / alpha 512 / lr 1.25e-5 (the s35 a512 arm); r16 = rank 16 / alpha 362 (alpha ∝ √r) at the same lr; REPA off in both; r16 rescaled to r32's median ||dW||

| arm | wins |
|---|---:|
| r32 | 17 |
| r16 | 15 |

Binomial sign test, r16 wins 15/32: one-sided p = 0.7017

Cluster sign-flip over 2 (artist, seed) pairs, r16 − r32 = -2: one-sided p = 1.0000

| artist | seed | r32 | r16 |
|---|---|---:|---:|
| ama | s42 | 8 | 8 |
| ama | s43 | 9 | 7 |

| artist | prompt | r32 | r16 |
|---|---|---:|---:|
| ama | p0 | 3 | 5 |
| ama | p1 | 3 | 5 |
| ama | p2 | 4 | 4 |
| ama | p3 | 7 | 1 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s41_RANK_AMA_ama | p01 | s42 | p3 | r32 | r16 | A |
| s41_RANK_AMA_ama | p02 | s43 | p1 | r16 | r32 | B |
| s41_RANK_AMA_ama | p03 | s43 | p1 | r32 | r16 | B |
| s41_RANK_AMA_ama | p04 | s42 | p0 | r32 | r16 | B |
| s41_RANK_AMA_ama | p05 | s43 | p2 | r32 | r16 | B |
| s41_RANK_AMA_ama | p06 | s43 | p3 | r16 | r32 | B |
| s41_RANK_AMA_ama | p07 | s42 | p1 | r32 | r16 | B |
| s41_RANK_AMA_ama | p08 | s42 | p0 | r32 | r16 | B |
| s41_RANK_AMA_ama | p09 | s43 | p1 | r32 | r16 | B |
| s41_RANK_AMA_ama | p10 | s43 | p3 | r16 | r32 | B |
| s41_RANK_AMA_ama | p11 | s42 | p3 | r32 | r16 | A |
| s41_RANK_AMA_ama | p12 | s43 | p0 | r16 | r32 | A |
| s41_RANK_AMA_ama | p13 | s43 | p3 | r16 | r32 | B |
| s41_RANK_AMA_ama | p14 | s42 | p2 | r32 | r16 | A |
| s41_RANK_AMA_ama | p15 | s42 | p0 | r32 | r16 | B |
| s41_RANK_AMA_ama | p16 | s42 | p1 | r32 | r16 | B |
| s41_RANK_AMA_ama | p17 | s42 | p3 | r16 | r32 | B |
| s41_RANK_AMA_ama | p18 | s43 | p0 | r16 | r32 | B |
| s41_RANK_AMA_ama | p19 | s43 | p3 | r32 | r16 | A |
| s41_RANK_AMA_ama | p20 | s42 | p0 | r16 | r32 | B |
| s41_RANK_AMA_ama | p21 | s43 | p2 | r32 | r16 | B |
| s41_RANK_AMA_ama | p22 | s43 | p2 | r32 | r16 | A |
| s41_RANK_AMA_ama | p23 | s42 | p2 | r32 | r16 | B |
| s41_RANK_AMA_ama | p24 | s43 | p0 | r32 | r16 | B |
| s41_RANK_AMA_ama | p25 | s42 | p2 | r32 | r16 | A |
| s41_RANK_AMA_ama | p26 | s43 | p1 | r32 | r16 | A |
| s41_RANK_AMA_ama | p27 | s43 | p0 | r16 | r32 | B |
| s41_RANK_AMA_ama | p28 | s42 | p3 | r16 | r32 | A |
| s41_RANK_AMA_ama | p29 | s42 | p1 | r32 | r16 | A |
| s41_RANK_AMA_ama | p30 | s42 | p1 | r32 | r16 | B |
| s41_RANK_AMA_ama | p31 | s42 | p2 | r16 | r32 | B |
| s41_RANK_AMA_ama | p32 | s43 | p2 | r16 | r32 | A |
