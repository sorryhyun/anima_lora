# OCR eval — `vl16_sfxp_1k_ocr` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 115 s for 5117 crops (44.3 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 63 | 2.5 | 0.032 | 2.9 % | 0 |
| speech | 2559 | 2167 | 84.7 | 0.967 | 96.4 % | 8 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 631 | 0.6 | 0.014 |
| square | 490 | 1.4 | 0.020 |
| vertical | 1437 | 3.6 | 0.043 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 9.1 | 0.125 |
| 2 | 1017 | 1.4 | 0.015 |
| 3 | 730 | 1.9 | 0.020 |
| 4 | 315 | 1.9 | 0.030 |
| 5 | 167 | 2.4 | 0.045 |
| 6 | 135 | 9.6 | 0.117 |
| 7 | 48 | 4.2 | 0.058 |
| 8+ | 58 | 3.4 | 0.085 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| ParaisoRoad 078 1000a2cb | キィィイイィ |  | 0.00 |
| ParaisoRoad 074 1000a2c2 | ぷ |  | 0.00 |
| ParaisoRoad 074 1000a2c3 | ぷる |  | 0.00 |
| ParaisoRoad 074 1000a2c4 | うが |  | 0.00 |
| ParaisoRoad 075 1000a2c5 | ゴト |  | 0.00 |
| ParaisoRoad 076 1000a2c6 | オオオオォ |  | 0.00 |
| ParaisoRoad 077 1000a2c7 | バシッ |  | 0.00 |
| ParaisoRoad 077 1000a2c8 | キッ |  | 0.00 |
| ParaisoRoad 077 1000a2c9 | ギリッ |  | 0.00 |
| ParaisoRoad 077 1000a2ca | ピ~ッ |  | 0.00 |
| ParaisoRoad 074 1000a2c1 | うが |  | 0.00 |
| ParaisoRoad 078 1000a2cc | パッ |  | 0.00 |
| ParaisoRoad 078 1000a2cd | パッ |  | 0.00 |
| ParaisoRoad 078 1000a2ce | パッ |  | 0.00 |
| ParaisoRoad 078 1000a2cf | ズンズン |  | 0.00 |
| ParaisoRoad 078 1000a2d0 | ズン |  | 0.00 |
| ParaisoRoad 078 1000a2d1 | パッ |  | 0.00 |
| ParaisoRoad 078 1000a2d2 | ウウウゥ |  | 0.00 |
| ParaisoRoad 078 1000a2d3+1000a2d4 | バン | きゃあほのかちゃん!! | 0.00 |
| ParaisoRoad 079 1000a2d5 | パク |  | 0.00 |
| ParaisoRoad 070 1000a2b7 | きしっ |  | 0.00 |
| ParaisoRoad 069 1000a2ae | ぐっ |  | 0.00 |
| ParaisoRoad 069 1000a2af | ふらっ |  | 0.00 |
| ParaisoRoad 069 1000a2b0 | ふおん |  | 0.00 |
| ParaisoRoad 069 1000a2b1 | かくん |  | 0.00 |
