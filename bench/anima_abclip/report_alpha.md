# blind A/B — s35_ALPHA

a128 = the shipped lora.toml (rank 32 / alpha 128, lr 5e-5, B + σ-band clip, no mask; the s31 / s32 LoRAs); a512 = alpha 512, lr 1.25e-5 (same alpha x lr, so the same first-order step and 1/4 of the bilinear drift, arXiv 2606.12883); a512 rescaled to a128's median ||dW||

| arm | wins |
|---|---:|
| a128 | 25 |
| a512 | 22 |
| tie | 1 |

Binomial sign test, a512 wins 22/47: one-sided p = 0.7200

Cluster sign-flip over 6 (artist, seed) pairs, a512 − a128 = -3: one-sided p = 0.6250

| artist | seed | a128 | a512 |
|---|---|---:|---:|
| cast | s42 | 5 | 3 |
| cast | s43 | 8 | 0 |
| cast | s44 | 3 | 5 |
| sweet | s42 | 2 | 6 |
| sweet | s43 | 3 | 5 |
| sweet | s44 | 4 | 3 |

| artist | prompt | a128 | a512 |
|---|---|---:|---:|
| cast | p0 | 5 | 1 |
| cast | p1 | 4 | 2 |
| cast | p2 | 5 | 1 |
| cast | p3 | 2 | 4 |
| sweet | p0 | 3 | 3 |
| sweet | p1 | 3 | 3 |
| sweet | p2 | 2 | 4 |
| sweet | p3 | 1 | 4 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s35_ALPHA_cast | p01 | s42 | p1 | a512 | a128 | B |
| s35_ALPHA_cast | p02 | s44 | p3 | a512 | a128 | A |
| s35_ALPHA_cast | p03 | s42 | p1 | a128 | a512 | A |
| s35_ALPHA_cast | p04 | s42 | p3 | a128 | a512 | B |
| s35_ALPHA_cast | p05 | s42 | p3 | a128 | a512 | B |
| s35_ALPHA_cast | p06 | s43 | p3 | a512 | a128 | B |
| s35_ALPHA_cast | p07 | s42 | p0 | a128 | a512 | A |
| s35_ALPHA_cast | p08 | s44 | p1 | a512 | a128 | A |
| s35_ALPHA_cast | p09 | s43 | p3 | a512 | a128 | B |
| s35_ALPHA_cast | p10 | s42 | p2 | a128 | a512 | A |
| s35_ALPHA_cast | p11 | s44 | p2 | a512 | a128 | A |
| s35_ALPHA_cast | p12 | s44 | p1 | a512 | a128 | A |
| s35_ALPHA_cast | p13 | s43 | p2 | a512 | a128 | B |
| s35_ALPHA_cast | p14 | s44 | p0 | a512 | a128 | B |
| s35_ALPHA_cast | p15 | s43 | p1 | a128 | a512 | A |
| s35_ALPHA_cast | p16 | s42 | p0 | a512 | a128 | A |
| s35_ALPHA_cast | p17 | s43 | p0 | a128 | a512 | A |
| s35_ALPHA_cast | p18 | s43 | p2 | a512 | a128 | B |
| s35_ALPHA_cast | p19 | s43 | p0 | a128 | a512 | A |
| s35_ALPHA_cast | p20 | s44 | p2 | a512 | a128 | B |
| s35_ALPHA_cast | p21 | s44 | p3 | a512 | a128 | A |
| s35_ALPHA_cast | p22 | s42 | p2 | a128 | a512 | A |
| s35_ALPHA_cast | p23 | s43 | p1 | a128 | a512 | A |
| s35_ALPHA_cast | p24 | s44 | p0 | a128 | a512 | A |
| s35_ALPHA_sweet | p01 | s42 | p0 | a512 | a128 | B |
| s35_ALPHA_sweet | p02 | s43 | p0 | a512 | a128 | A |
| s35_ALPHA_sweet | p03 | s43 | p1 | a128 | a512 | A |
| s35_ALPHA_sweet | p04 | s42 | p1 | a128 | a512 | B |
| s35_ALPHA_sweet | p05 | s43 | p1 | a512 | a128 | B |
| s35_ALPHA_sweet | p06 | s42 | p1 | a128 | a512 | B |
| s35_ALPHA_sweet | p07 | s44 | p1 | a512 | a128 | B |
| s35_ALPHA_sweet | p08 | s44 | p1 | a128 | a512 | B |
| s35_ALPHA_sweet | p09 | s42 | p3 | a128 | a512 | B |
| s35_ALPHA_sweet | p10 | s42 | p3 | a128 | a512 | B |
| s35_ALPHA_sweet | p11 | s44 | p0 | a512 | a128 | A |
| s35_ALPHA_sweet | p12 | s44 | p2 | a128 | a512 | A |
| s35_ALPHA_sweet | p13 | s43 | p0 | a128 | a512 | B |
| s35_ALPHA_sweet | p14 | s43 | p2 | a512 | a128 | A |
| s35_ALPHA_sweet | p15 | s43 | p2 | a128 | a512 | B |
| s35_ALPHA_sweet | p16 | s44 | p3 | a128 | a512 | B |
| s35_ALPHA_sweet | p17 | s44 | p0 | a512 | a128 | B |
| s35_ALPHA_sweet | p18 | s43 | p3 | a128 | a512 | A |
| s35_ALPHA_sweet | p19 | s44 | p2 | a128 | a512 | A |
| s35_ALPHA_sweet | p20 | s43 | p3 | a128 | a512 | B |
| s35_ALPHA_sweet | p21 | s42 | p2 | a128 | a512 | B |
| s35_ALPHA_sweet | p22 | s42 | p0 | a128 | a512 | A |
| s35_ALPHA_sweet | p23 | s42 | p2 | a512 | a128 | A |
| s35_ALPHA_sweet | p24 | s44 | p3 | a128 | a512 | TIE |
