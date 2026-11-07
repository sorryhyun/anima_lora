# blind A/B — s43_NODC

svd = the shipped lora.toml (weight_svd, channel scaling α 0.5); nodc = lora_down seeded from W(I − ûûᵀ), û the calibrated input mean, plus the channel-scaling floor 0.01 (render-neutral by construction: it only moves near-dead channels); nodc rescaled to svd's median ||dW||

| arm | wins |
|---|---:|
| svd | 25 |
| nodc | 23 |

Binomial sign test, nodc wins 23/48: one-sided p = 0.6673

Cluster sign-flip over 6 (artist, seed) pairs, nodc − svd = -2: one-sided p = 0.7344

| artist | seed | svd | nodc |
|---|---|---:|---:|
| cast | s42 | 3 | 5 |
| cast | s43 | 6 | 2 |
| cast | s44 | 3 | 5 |
| sweet | s42 | 3 | 5 |
| sweet | s43 | 5 | 3 |
| sweet | s44 | 5 | 3 |

| artist | prompt | svd | nodc |
|---|---|---:|---:|
| cast | p0 | 5 | 1 |
| cast | p1 | 5 | 1 |
| cast | p2 | 0 | 6 |
| cast | p3 | 2 | 4 |
| sweet | p0 | 5 | 1 |
| sweet | p1 | 4 | 2 |
| sweet | p2 | 2 | 4 |
| sweet | p3 | 2 | 4 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s43_NODC_cast | p01 | s42 | p1 | svd | nodc | A |
| s43_NODC_cast | p02 | s44 | p1 | nodc | svd | B |
| s43_NODC_cast | p03 | s42 | p3 | nodc | svd | A |
| s43_NODC_cast | p04 | s44 | p2 | svd | nodc | B |
| s43_NODC_cast | p05 | s44 | p1 | svd | nodc | A |
| s43_NODC_cast | p06 | s44 | p0 | svd | nodc | B |
| s43_NODC_cast | p07 | s44 | p3 | svd | nodc | B |
| s43_NODC_cast | p08 | s43 | p1 | nodc | svd | B |
| s43_NODC_cast | p09 | s43 | p2 | svd | nodc | B |
| s43_NODC_cast | p10 | s44 | p0 | svd | nodc | A |
| s43_NODC_cast | p11 | s43 | p2 | nodc | svd | A |
| s43_NODC_cast | p12 | s42 | p2 | nodc | svd | A |
| s43_NODC_cast | p13 | s44 | p3 | svd | nodc | B |
| s43_NODC_cast | p14 | s42 | p3 | svd | nodc | B |
| s43_NODC_cast | p15 | s43 | p0 | svd | nodc | A |
| s43_NODC_cast | p16 | s42 | p0 | nodc | svd | B |
| s43_NODC_cast | p17 | s43 | p3 | svd | nodc | A |
| s43_NODC_cast | p18 | s44 | p2 | nodc | svd | A |
| s43_NODC_cast | p19 | s42 | p1 | nodc | svd | A |
| s43_NODC_cast | p20 | s43 | p1 | svd | nodc | A |
| s43_NODC_cast | p21 | s43 | p3 | nodc | svd | B |
| s43_NODC_cast | p22 | s42 | p2 | nodc | svd | A |
| s43_NODC_cast | p23 | s43 | p0 | svd | nodc | A |
| s43_NODC_cast | p24 | s42 | p0 | nodc | svd | B |
| s43_NODC_sweet | p01 | s44 | p3 | svd | nodc | A |
| s43_NODC_sweet | p02 | s43 | p1 | svd | nodc | B |
| s43_NODC_sweet | p03 | s43 | p0 | svd | nodc | A |
| s43_NODC_sweet | p04 | s42 | p3 | nodc | svd | A |
| s43_NODC_sweet | p05 | s43 | p3 | nodc | svd | B |
| s43_NODC_sweet | p06 | s43 | p0 | nodc | svd | B |
| s43_NODC_sweet | p07 | s43 | p1 | svd | nodc | A |
| s43_NODC_sweet | p08 | s42 | p3 | svd | nodc | B |
| s43_NODC_sweet | p09 | s44 | p2 | svd | nodc | B |
| s43_NODC_sweet | p10 | s42 | p2 | svd | nodc | B |
| s43_NODC_sweet | p11 | s44 | p0 | svd | nodc | A |
| s43_NODC_sweet | p12 | s42 | p1 | nodc | svd | B |
| s43_NODC_sweet | p13 | s44 | p1 | svd | nodc | A |
| s43_NODC_sweet | p14 | s43 | p2 | nodc | svd | A |
| s43_NODC_sweet | p15 | s44 | p2 | svd | nodc | A |
| s43_NODC_sweet | p16 | s42 | p1 | svd | nodc | B |
| s43_NODC_sweet | p17 | s42 | p0 | nodc | svd | B |
| s43_NODC_sweet | p18 | s42 | p2 | svd | nodc | B |
| s43_NODC_sweet | p19 | s42 | p0 | nodc | svd | B |
| s43_NODC_sweet | p20 | s44 | p1 | svd | nodc | A |
| s43_NODC_sweet | p21 | s43 | p3 | svd | nodc | B |
| s43_NODC_sweet | p22 | s43 | p2 | svd | nodc | A |
| s43_NODC_sweet | p23 | s44 | p0 | nodc | svd | A |
| s43_NODC_sweet | p24 | s44 | p3 | nodc | svd | A |
