# blind A/B — s42_ALPHA_NOREPA

s35 without REPA: a128 = the shipped lora.toml (alpha 128, lr 5e-5, REPA off; the s39 shift1 LoRAs); a512 = alpha 512, lr 1.25e-5, REPA off (the s37 / s38 norepa LoRAs); a512 rescaled to a128's median ||dW||

| arm | wins |
|---|---:|
| a128 | 23 |
| a512 | 22 |
| blank | 3 |

Binomial sign test, a512 wins 22/45: one-sided p = 0.6170

Cluster sign-flip over 6 (artist, seed) pairs, a512 − a128 = -1: one-sided p = 0.6406

| artist | seed | a128 | a512 |
|---|---|---:|---:|
| cast | s42 | 2 | 6 |
| cast | s43 | 4 | 3 |
| cast | s44 | 5 | 2 |
| sweet | s42 | 2 | 6 |
| sweet | s43 | 6 | 2 |
| sweet | s44 | 4 | 3 |

| artist | prompt | a128 | a512 |
|---|---|---:|---:|
| cast | p0 | 4 | 1 |
| cast | p1 | 4 | 2 |
| cast | p2 | 2 | 4 |
| cast | p3 | 1 | 4 |
| sweet | p0 | 2 | 3 |
| sweet | p1 | 4 | 2 |
| sweet | p2 | 3 | 3 |
| sweet | p3 | 3 | 3 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s42_ALPHA_NOREPA_cast | p01 | s43 | p3 | a512 | a128 | A |
| s42_ALPHA_NOREPA_cast | p02 | s42 | p2 | a128 | a512 | B |
| s42_ALPHA_NOREPA_cast | p03 | s44 | p1 | a512 | a128 | A |
| s42_ALPHA_NOREPA_cast | p04 | s44 | p3 | a128 | a512 | A |
| s42_ALPHA_NOREPA_cast | p05 | s43 | p2 | a512 | a128 | B |
| s42_ALPHA_NOREPA_cast | p06 | s43 | p1 | a128 | a512 | A |
| s42_ALPHA_NOREPA_cast | p07 | s44 | p0 | a128 | a512 | A |
| s42_ALPHA_NOREPA_cast | p08 | s43 | p1 | a512 | a128 | B |
| s42_ALPHA_NOREPA_cast | p09 | s43 | p3 | a128 | a512 | B |
| s42_ALPHA_NOREPA_cast | p10 | s44 | p0 | a128 | a512 | A |
| s42_ALPHA_NOREPA_cast | p11 | s42 | p0 | a128 | a512 | B |
| s42_ALPHA_NOREPA_cast | p12 | s43 | p2 | a512 | a128 | A |
| s42_ALPHA_NOREPA_cast | p13 | s44 | p2 | a128 | a512 | A |
| s42_ALPHA_NOREPA_cast | p14 | s42 | p3 | a128 | a512 | B |
| s42_ALPHA_NOREPA_cast | p15 | s43 | p0 | a128 | a512 | A |
| s42_ALPHA_NOREPA_cast | p16 | s42 | p1 | a128 | a512 | A |
| s42_ALPHA_NOREPA_cast | p17 | s44 | p3 | a512 | a128 |  |
| s42_ALPHA_NOREPA_cast | p18 | s42 | p2 | a512 | a128 | A |
| s42_ALPHA_NOREPA_cast | p19 | s44 | p1 | a512 | a128 | B |
| s42_ALPHA_NOREPA_cast | p20 | s42 | p3 | a128 | a512 | B |
| s42_ALPHA_NOREPA_cast | p21 | s43 | p0 | a128 | a512 |  |
| s42_ALPHA_NOREPA_cast | p22 | s42 | p0 | a128 | a512 | A |
| s42_ALPHA_NOREPA_cast | p23 | s42 | p1 | a512 | a128 | A |
| s42_ALPHA_NOREPA_cast | p24 | s44 | p2 | a512 | a128 | A |
| s42_ALPHA_NOREPA_sweet | p01 | s42 | p0 | a512 | a128 | B |
| s42_ALPHA_NOREPA_sweet | p02 | s44 | p1 | a512 | a128 | B |
| s42_ALPHA_NOREPA_sweet | p03 | s44 | p0 | a128 | a512 | B |
| s42_ALPHA_NOREPA_sweet | p04 | s42 | p2 | a128 | a512 | B |
| s42_ALPHA_NOREPA_sweet | p05 | s44 | p0 | a128 | a512 |  |
| s42_ALPHA_NOREPA_sweet | p06 | s43 | p3 | a128 | a512 | A |
| s42_ALPHA_NOREPA_sweet | p07 | s44 | p1 | a128 | a512 | A |
| s42_ALPHA_NOREPA_sweet | p08 | s43 | p2 | a512 | a128 | B |
| s42_ALPHA_NOREPA_sweet | p09 | s42 | p3 | a512 | a128 | A |
| s42_ALPHA_NOREPA_sweet | p10 | s44 | p2 | a128 | a512 | B |
| s42_ALPHA_NOREPA_sweet | p11 | s43 | p3 | a512 | a128 | B |
| s42_ALPHA_NOREPA_sweet | p12 | s42 | p2 | a512 | a128 | A |
| s42_ALPHA_NOREPA_sweet | p13 | s43 | p1 | a128 | a512 | A |
| s42_ALPHA_NOREPA_sweet | p14 | s43 | p0 | a128 | a512 | B |
| s42_ALPHA_NOREPA_sweet | p15 | s42 | p1 | a128 | a512 | B |
| s42_ALPHA_NOREPA_sweet | p16 | s42 | p1 | a512 | a128 | A |
| s42_ALPHA_NOREPA_sweet | p17 | s43 | p0 | a128 | a512 | B |
| s42_ALPHA_NOREPA_sweet | p18 | s43 | p2 | a128 | a512 | A |
| s42_ALPHA_NOREPA_sweet | p19 | s44 | p3 | a512 | a128 | B |
| s42_ALPHA_NOREPA_sweet | p20 | s44 | p3 | a128 | a512 | B |
| s42_ALPHA_NOREPA_sweet | p21 | s42 | p3 | a512 | a128 | A |
| s42_ALPHA_NOREPA_sweet | p22 | s42 | p0 | a128 | a512 | A |
| s42_ALPHA_NOREPA_sweet | p23 | s43 | p1 | a128 | a512 | A |
| s42_ALPHA_NOREPA_sweet | p24 | s44 | p2 | a128 | a512 | A |
