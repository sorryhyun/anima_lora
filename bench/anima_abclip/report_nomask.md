# blind A/B — s31_NOMASK

band = T-LoRA as published: the shipped lora.toml (B + σ-band clip, use_timestep_mask, min_rank 1) rendered with the same mask per step (--lora_timestep_mask); nomask = the same recipe with use_timestep_mask off, full rank in training and inference (static merge); nomask rescaled to band's median ||dW||

| arm | wins |
|---|---:|
| band | 11 |
| nomask | 13 |

Binomial sign test, nomask wins 13/24: one-sided p = 0.4194

Cluster sign-flip over 3 (artist, seed) pairs, nomask − band = 2: one-sided p = 0.5000

| artist | seed | band | nomask |
|---|---|---:|---:|
| sweet | s42 | 4 | 4 |
| sweet | s43 | 1 | 7 |
| sweet | s44 | 6 | 2 |

| artist | prompt | band | nomask |
|---|---|---:|---:|
| sweet | p0 | 4 | 2 |
| sweet | p1 | 3 | 3 |
| sweet | p2 | 1 | 5 |
| sweet | p3 | 3 | 3 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s31_NOMASK_sweet | p01 | s44 | p1 | band | nomask | A |
| s31_NOMASK_sweet | p02 | s43 | p0 | nomask | band | A |
| s31_NOMASK_sweet | p03 | s43 | p0 | band | nomask | B |
| s31_NOMASK_sweet | p04 | s42 | p3 | nomask | band | B |
| s31_NOMASK_sweet | p05 | s44 | p3 | nomask | band | B |
| s31_NOMASK_sweet | p06 | s42 | p2 | band | nomask | B |
| s31_NOMASK_sweet | p07 | s44 | p0 | nomask | band | B |
| s31_NOMASK_sweet | p08 | s44 | p0 | band | nomask | A |
| s31_NOMASK_sweet | p09 | s43 | p3 | nomask | band | A |
| s31_NOMASK_sweet | p10 | s43 | p1 | band | nomask | B |
| s31_NOMASK_sweet | p11 | s44 | p3 | nomask | band | B |
| s31_NOMASK_sweet | p12 | s43 | p2 | band | nomask | B |
| s31_NOMASK_sweet | p13 | s42 | p1 | band | nomask | B |
| s31_NOMASK_sweet | p14 | s44 | p2 | nomask | band | A |
| s31_NOMASK_sweet | p15 | s43 | p1 | band | nomask | B |
| s31_NOMASK_sweet | p16 | s42 | p3 | band | nomask | B |
| s31_NOMASK_sweet | p17 | s44 | p2 | band | nomask | B |
| s31_NOMASK_sweet | p18 | s44 | p1 | nomask | band | B |
| s31_NOMASK_sweet | p19 | s42 | p0 | nomask | band | B |
| s31_NOMASK_sweet | p20 | s42 | p2 | nomask | band | A |
| s31_NOMASK_sweet | p21 | s43 | p2 | band | nomask | A |
| s31_NOMASK_sweet | p22 | s42 | p1 | nomask | band | B |
| s31_NOMASK_sweet | p23 | s43 | p3 | nomask | band | A |
| s31_NOMASK_sweet | p24 | s42 | p0 | band | nomask | A |
