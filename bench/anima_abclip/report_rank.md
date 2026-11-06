# blind A/B — s34_RANK

r32 = the shipped lora.toml (rank 32 / alpha 128, B + σ-band clip, no mask; the s31 / s32 LoRAs); r16 = the same at rank 16 / alpha 64 (alpha / rank 4 in both); r16 rescaled to r32's median ||dW||

| arm | wins |
|---|---:|
| r32 | 20 |
| r16 | 28 |

Binomial sign test, r16 wins 28/48: one-sided p = 0.1562

Cluster sign-flip over 6 (artist, seed) pairs, r16 − r32 = 8: one-sided p = 0.2500

| artist | seed | r32 | r16 |
|---|---|---:|---:|
| cast | s42 | 4 | 4 |
| cast | s43 | 5 | 3 |
| cast | s44 | 1 | 7 |
| sweet | s42 | 1 | 7 |
| sweet | s43 | 5 | 3 |
| sweet | s44 | 4 | 4 |

| artist | prompt | r32 | r16 |
|---|---|---:|---:|
| cast | p0 | 2 | 4 |
| cast | p1 | 3 | 3 |
| cast | p2 | 3 | 3 |
| cast | p3 | 2 | 4 |
| sweet | p0 | 2 | 4 |
| sweet | p1 | 2 | 4 |
| sweet | p2 | 4 | 2 |
| sweet | p3 | 2 | 4 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s34_RANK_cast | p01 | s44 | p1 | r32 | r16 | B |
| s34_RANK_cast | p02 | s44 | p0 | r16 | r32 | A |
| s34_RANK_cast | p03 | s42 | p1 | r16 | r32 | B |
| s34_RANK_cast | p04 | s44 | p2 | r16 | r32 | A |
| s34_RANK_cast | p05 | s43 | p2 | r16 | r32 | A |
| s34_RANK_cast | p06 | s43 | p3 | r32 | r16 | A |
| s34_RANK_cast | p07 | s43 | p1 | r16 | r32 | B |
| s34_RANK_cast | p08 | s44 | p3 | r32 | r16 | B |
| s34_RANK_cast | p09 | s43 | p3 | r16 | r32 | A |
| s34_RANK_cast | p10 | s43 | p0 | r32 | r16 | B |
| s34_RANK_cast | p11 | s43 | p0 | r32 | r16 | A |
| s34_RANK_cast | p12 | s42 | p0 | r16 | r32 | A |
| s34_RANK_cast | p13 | s42 | p2 | r32 | r16 | A |
| s34_RANK_cast | p14 | s42 | p2 | r32 | r16 | B |
| s34_RANK_cast | p15 | s42 | p3 | r16 | r32 | A |
| s34_RANK_cast | p16 | s42 | p1 | r16 | r32 | A |
| s34_RANK_cast | p17 | s44 | p3 | r16 | r32 | A |
| s34_RANK_cast | p18 | s43 | p2 | r16 | r32 | B |
| s34_RANK_cast | p19 | s44 | p2 | r32 | r16 | A |
| s34_RANK_cast | p20 | s42 | p3 | r16 | r32 | B |
| s34_RANK_cast | p21 | s42 | p0 | r32 | r16 | A |
| s34_RANK_cast | p22 | s44 | p1 | r16 | r32 | A |
| s34_RANK_cast | p23 | s43 | p1 | r32 | r16 | A |
| s34_RANK_cast | p24 | s44 | p0 | r32 | r16 | B |
| s34_RANK_sweet | p01 | s43 | p1 | r32 | r16 | A |
| s34_RANK_sweet | p02 | s42 | p0 | r16 | r32 | A |
| s34_RANK_sweet | p03 | s44 | p2 | r16 | r32 | B |
| s34_RANK_sweet | p04 | s42 | p2 | r16 | r32 | A |
| s34_RANK_sweet | p05 | s44 | p3 | r16 | r32 | A |
| s34_RANK_sweet | p06 | s42 | p1 | r16 | r32 | A |
| s34_RANK_sweet | p07 | s43 | p0 | r16 | r32 | B |
| s34_RANK_sweet | p08 | s44 | p1 | r32 | r16 | B |
| s34_RANK_sweet | p09 | s44 | p2 | r16 | r32 | B |
| s34_RANK_sweet | p10 | s43 | p3 | r32 | r16 | A |
| s34_RANK_sweet | p11 | s43 | p2 | r16 | r32 | B |
| s34_RANK_sweet | p12 | s42 | p3 | r16 | r32 | A |
| s34_RANK_sweet | p13 | s42 | p2 | r16 | r32 | B |
| s34_RANK_sweet | p14 | s42 | p0 | r16 | r32 | A |
| s34_RANK_sweet | p15 | s44 | p1 | r32 | r16 | B |
| s34_RANK_sweet | p16 | s44 | p0 | r32 | r16 | A |
| s34_RANK_sweet | p17 | s42 | p1 | r16 | r32 | A |
| s34_RANK_sweet | p18 | s43 | p3 | r32 | r16 | B |
| s34_RANK_sweet | p19 | s43 | p1 | r16 | r32 | B |
| s34_RANK_sweet | p20 | s44 | p0 | r16 | r32 | A |
| s34_RANK_sweet | p21 | s43 | p0 | r16 | r32 | A |
| s34_RANK_sweet | p22 | s44 | p3 | r32 | r16 | A |
| s34_RANK_sweet | p23 | s42 | p3 | r32 | r16 | B |
| s34_RANK_sweet | p24 | s43 | p2 | r32 | r16 | B |
