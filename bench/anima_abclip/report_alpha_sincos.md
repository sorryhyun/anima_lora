# blind A/B — s36_ALPHASINCOS

s35 on a 354-image set (~1416 updates against ~120): a128 = the shipped lora.toml (rank 32 / alpha 128, lr 5e-5); a512 = alpha 512, lr 1.25e-5; a512 rescaled to a128's median ||dW||

| arm | wins |
|---|---:|
| a128 | 18 |
| a512 | 13 |
| tie | 1 |

Binomial sign test, a512 wins 13/31: one-sided p = 0.8595

Cluster sign-flip over 2 (artist, seed) pairs, a512 − a128 = -5: one-sided p = 1.0000

| artist | seed | a128 | a512 |
|---|---|---:|---:|
| sincos | s42 | 8 | 8 |
| sincos | s43 | 10 | 5 |

| artist | prompt | a128 | a512 |
|---|---|---:|---:|
| sincos | p0 | 6 | 1 |
| sincos | p1 | 4 | 4 |
| sincos | p2 | 4 | 4 |
| sincos | p3 | 4 | 4 |

| set | pair | seed | prompt | A | B | verdict |
|---|---|---|---|---|---|---|
| s36_ALPHASINCOS_sincos | p01 | s42 | p3 | a512 | a128 | A |
| s36_ALPHASINCOS_sincos | p02 | s42 | p2 | a512 | a128 | B |
| s36_ALPHASINCOS_sincos | p03 | s42 | p3 | a128 | a512 | A |
| s36_ALPHASINCOS_sincos | p04 | s43 | p1 | a512 | a128 | B |
| s36_ALPHASINCOS_sincos | p05 | s42 | p1 | a128 | a512 | B |
| s36_ALPHASINCOS_sincos | p06 | s43 | p2 | a512 | a128 | B |
| s36_ALPHASINCOS_sincos | p07 | s43 | p3 | a128 | a512 | B |
| s36_ALPHASINCOS_sincos | p08 | s42 | p3 | a512 | a128 | B |
| s36_ALPHASINCOS_sincos | p09 | s42 | p0 | a128 | a512 | A |
| s36_ALPHASINCOS_sincos | p10 | s43 | p0 | a512 | a128 | TIE |
| s36_ALPHASINCOS_sincos | p11 | s43 | p3 | a128 | a512 | B |
| s36_ALPHASINCOS_sincos | p12 | s43 | p0 | a512 | a128 | B |
| s36_ALPHASINCOS_sincos | p13 | s43 | p1 | a128 | a512 | B |
| s36_ALPHASINCOS_sincos | p14 | s42 | p1 | a128 | a512 | B |
| s36_ALPHASINCOS_sincos | p15 | s43 | p2 | a512 | a128 | B |
| s36_ALPHASINCOS_sincos | p16 | s43 | p2 | a128 | a512 | A |
| s36_ALPHASINCOS_sincos | p17 | s43 | p0 | a512 | a128 | B |
| s36_ALPHASINCOS_sincos | p18 | s42 | p2 | a128 | a512 | B |
| s36_ALPHASINCOS_sincos | p19 | s42 | p3 | a128 | a512 | B |
| s36_ALPHASINCOS_sincos | p20 | s42 | p2 | a512 | a128 | A |
| s36_ALPHASINCOS_sincos | p21 | s42 | p1 | a128 | a512 | A |
| s36_ALPHASINCOS_sincos | p22 | s42 | p1 | a128 | a512 | A |
| s36_ALPHASINCOS_sincos | p23 | s43 | p1 | a128 | a512 | A |
| s36_ALPHASINCOS_sincos | p24 | s43 | p0 | a512 | a128 | B |
| s36_ALPHASINCOS_sincos | p25 | s43 | p3 | a512 | a128 | B |
| s36_ALPHASINCOS_sincos | p26 | s42 | p0 | a128 | a512 | A |
| s36_ALPHASINCOS_sincos | p27 | s42 | p2 | a128 | a512 | B |
| s36_ALPHASINCOS_sincos | p28 | s43 | p3 | a128 | a512 | A |
| s36_ALPHASINCOS_sincos | p29 | s43 | p2 | a128 | a512 | B |
| s36_ALPHASINCOS_sincos | p30 | s42 | p0 | a128 | a512 | A |
| s36_ALPHASINCOS_sincos | p31 | s42 | p0 | a512 | a128 | A |
| s36_ALPHASINCOS_sincos | p32 | s43 | p1 | a128 | a512 | B |
