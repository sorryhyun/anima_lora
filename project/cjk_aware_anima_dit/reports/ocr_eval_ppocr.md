# OCR eval — `ppocr` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 14 s for 5117 crops (354.8 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 77 | 3.0 | 0.093 | 5.1 % | 0 |
| speech | 2559 | 23 | 0.9 | 0.082 | 3.9 % | 0 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 1292 | 1.4 | 0.082 |
| square | 486 | 3.5 | 0.070 |
| vertical | 780 | 5.4 | 0.124 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 19.3 | 0.193 |
| 2 | 1017 | 3.0 | 0.081 |
| 3 | 730 | 2.7 | 0.079 |
| 4 | 315 | 2.5 | 0.099 |
| 5 | 167 | 0.0 | 0.096 |
| 6 | 135 | 0.0 | 0.151 |
| 7 | 48 | 0.0 | 0.121 |
| 8+ | 58 | 1.7 | 0.122 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| ParaisoRoad 068 1000a2ad | ぶん |  | 0.00 |
| ParaisoRoad 067 1000a2a4 | ばっ | は | 0.00 |
| ParaisoRoad 067 1000a2a5 | ざっ | ど | 0.00 |
| ParaisoRoad 067 1000a2a6 | ゴオオオオォォォ | SAC | 0.00 |
| ParaisoRoad 068 1000a2a7 | ば | 1 | 0.00 |
| ParaisoRoad 068 1000a2a8 | ゴオオオオ |  | 0.00 |
| ParaisoRoad 068 1000a2a9 | だん | th | 0.00 |
| ParaisoRoad 068 1000a2aa | ふんっ | K | 0.00 |
| ParaisoRoad 068 1000a2ab | ぶお |  | 0.00 |
| ParaisoRoad 068 1000a2ac | パリッ |  | 0.00 |
| ParaisoRoad 067 1000a2a3 | ゴォ |  | 0.00 |
| ParaisoRoad 069 1000a2ae | ぐっ | 4 | 0.00 |
| ParaisoRoad 069 1000a2af | ふらっ | 0.5 | 0.00 |
| ParaisoRoad 069 1000a2b0 | ふおん |  | 0.00 |
| ParaisoRoad 069 1000a2b1 | かくん | XZ | 0.00 |
| ParaisoRoad 069 1000a2b2 | ぶる | 33 | 0.00 |
| ParaisoRoad 069 1000a2b3 | ぶる | 3 | 0.00 |
| ParaisoRoad 069 1000a2b4 | しゅぅぅぅ | 事局 | 0.00 |
| ParaisoRoad 070 1000a2b5 | うっ |  | 0.00 |
| ParaisoRoad 070 1000a2b6 | ハァ | 1.4 | 0.00 |
| ParaisoRoad 064 1000a298 | あはは | 喜 | 0.00 |
| ParaisoRoad 062 1000a28d | ひゅー | (y | 0.00 |
| ParaisoRoad 062 1000a28e | ドゴォォオ | —9 | 0.00 |
| ParaisoRoad 062 1000a28f | たっ | E | 0.00 |
| ParaisoRoad 062 1000a291 | びゅっ | WHE | 0.00 |
