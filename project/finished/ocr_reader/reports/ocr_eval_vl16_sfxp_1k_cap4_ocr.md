# OCR eval — `vl16_sfxp_1k_cap4_ocr` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 112 s for 5117 crops (45.8 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 69 | 2.7 | 0.030 | 2.9 % | 0 |
| speech | 2559 | 2175 | 85.0 | 0.969 | 96.4 % | 7 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 652 | 0.6 | 0.008 |
| square | 417 | 1.7 | 0.024 |
| vertical | 1489 | 3.9 | 0.041 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 9.1 | 0.125 |
| 2 | 1017 | 1.9 | 0.020 |
| 3 | 730 | 2.1 | 0.021 |
| 4 | 315 | 1.6 | 0.016 |
| 5 | 167 | 3.6 | 0.051 |
| 6 | 135 | 9.6 | 0.101 |
| 7 | 48 | 4.2 | 0.042 |
| 8+ | 58 | 1.7 | 0.026 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| ParaisoRoad 078 1000a2cd | パッ |  | 0.00 |
| ParaisoRoad 074 1000a2c4 | うが |  | 0.00 |
| ParaisoRoad 075 1000a2c5 | ゴト |  | 0.00 |
| ParaisoRoad 076 1000a2c6 | オオオオォ |  | 0.00 |
| ParaisoRoad 077 1000a2c7 | バシッ |  | 0.00 |
| ParaisoRoad 077 1000a2c8 | キッ |  | 0.00 |
| ParaisoRoad 077 1000a2c9 | ギリッ |  | 0.00 |
| ParaisoRoad 077 1000a2ca | ピ~ッ |  | 0.00 |
| ParaisoRoad 078 1000a2cb | キィィイイィ |  | 0.00 |
| ParaisoRoad 078 1000a2cc | パッ |  | 0.00 |
| ParaisoRoad 074 1000a2c3 | ぷる |  | 0.00 |
| ParaisoRoad 078 1000a2ce | パッ |  | 0.00 |
| ParaisoRoad 078 1000a2cf | ズンズン |  | 0.00 |
| ParaisoRoad 078 1000a2d0 | ズン |  | 0.00 |
| ParaisoRoad 078 1000a2d1 | パッ |  | 0.00 |
| ParaisoRoad 078 1000a2d2 | ウウウゥ |  | 0.00 |
| ParaisoRoad 078 1000a2d3+1000a2d4 | バン |  | 0.00 |
| ParaisoRoad 079 1000a2d5 | パク |  | 0.00 |
| ParaisoRoad 079 1000a2d6 | ぶん |  | 0.00 |
| ParaisoRoad 079 1000a2d7 | パク |  | 0.00 |
| ParaisoRoad 071 1000a2b9 | べしっ |  | 0.00 |
| ParaisoRoad 069 1000a2b0 | ふおん |  | 0.00 |
| ParaisoRoad 069 1000a2b1 | かくん |  | 0.00 |
| ParaisoRoad 069 1000a2b2 | ぶる |  | 0.00 |
| ParaisoRoad 069 1000a2b3 | ぶる |  | 0.00 |
