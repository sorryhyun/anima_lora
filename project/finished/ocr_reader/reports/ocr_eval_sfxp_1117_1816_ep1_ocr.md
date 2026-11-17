# OCR eval — `sfxp_1117_1816_ep1_ocr` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 135 s for 5117 crops (37.8 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 162 | 6.3 | 0.073 | 6.7 % | 2 |
| speech | 2559 | 2185 | 85.4 | 0.975 | 96.7 % | 34 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 631 | 0.5 | 0.009 |
| square | 490 | 4.1 | 0.047 |
| vertical | 1437 | 9.7 | 0.110 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 8.0 | 0.114 |
| 2 | 1017 | 7.3 | 0.083 |
| 3 | 730 | 6.2 | 0.069 |
| 4 | 315 | 7.0 | 0.079 |
| 5 | 167 | 1.8 | 0.022 |
| 6 | 135 | 6.7 | 0.070 |
| 7 | 48 | 2.1 | 0.035 |
| 8+ | 58 | 1.7 | 0.037 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| ParaisoRoad 069 1000a2af | ふらっ |  | 0.00 |
| ParaisoRoad 067 1000a2a6 | ゴオオオオォォォ |  | 0.00 |
| ParaisoRoad 068 1000a2a7 | ば |  | 0.00 |
| ParaisoRoad 068 1000a2a8 | ゴオオオオ |  | 0.00 |
| ParaisoRoad 068 1000a2a9 | だん |  | 0.00 |
| ParaisoRoad 068 1000a2aa | ふんっ |  | 0.00 |
| ParaisoRoad 068 1000a2ab | ぶお |  | 0.00 |
| ParaisoRoad 068 1000a2ac | パリッ |  | 0.00 |
| ParaisoRoad 068 1000a2ad | ぶん |  | 0.00 |
| ParaisoRoad 069 1000a2ae | ぐっ |  | 0.00 |
| ParaisoRoad 067 1000a2a5 | ざっ |  | 0.00 |
| ParaisoRoad 069 1000a2b0 | ふおん |  | 0.00 |
| ParaisoRoad 069 1000a2b1 | かくん |  | 0.00 |
| ParaisoRoad 069 1000a2b2 | ぶる |  | 0.00 |
| ParaisoRoad 069 1000a2b3 | ぶる |  | 0.00 |
| ParaisoRoad 069 1000a2b4 | しゅぅぅぅ |  | 0.00 |
| ParaisoRoad 070 1000a2b5 | うっ |  | 0.00 |
| ParaisoRoad 070 1000a2b6 | ハァ |  | 0.00 |
| ParaisoRoad 070 1000a2b7 | きしっ |  | 0.00 |
| ParaisoRoad 070 1000a2b8 | ハァ |  | 0.00 |
| ParaisoRoad 065 1000a29c | ぞっ |  | 0.00 |
| ParaisoRoad 062 1000a292 | ゴオオオォォォ |  | 0.00 |
| ParaisoRoad 062 1000a293 | ぶわっ |  | 0.00 |
| ParaisoRoad 063 1000a294 | はっ |  | 0.00 |
| ParaisoRoad 064 1000a295 | ぶん |  | 0.00 |
