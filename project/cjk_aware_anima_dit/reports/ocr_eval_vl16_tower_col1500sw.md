# OCR eval — `vl16_tower_col1500sw` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 152 s for 5117 crops (33.7 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2151 | 84.1 | 0.939 | 89.4 % | 26 |
| speech | 2559 | 2111 | 82.5 | 0.986 | 98.0 % | 168 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 71.5 | 0.904 |
| square | 486 | 86.0 | 0.930 |
| vertical | 1448 | 88.9 | 0.957 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 78.4 | 0.803 |
| 2 | 1017 | 91.9 | 0.950 |
| 3 | 730 | 86.0 | 0.950 |
| 4 | 315 | 78.7 | 0.941 |
| 5 | 167 | 74.9 | 0.927 |
| 6 | 135 | 76.3 | 0.951 |
| 7 | 48 | 56.2 | 0.904 |
| 8+ | 58 | 27.6 | 0.834 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b831 | プァ | フア | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ツ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | ガガガ | 0.00 |
| SaladDays_vol18 070 1000b7cf | ザザ | ギギ | 0.00 |
| SaladDays_vol18 082 1000b806 | ダッ | ゴ | 0.00 |
| MAD_STONE 087 100070af+100070b0 | ゴワッ | ドアアア | 0.00 |
| ParaisoRoad 004 1000a12a | ォォ | オオ | 0.00 |
| ParaisoRoad 018 1000a19b | ィ | イ | 0.00 |
| ParaisoRoad 018 1000a199 | ィ | イ | 0.00 |
| ParaisoRoad 018 1000a198 | ィィ | イイ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| MukoukizuNoChonbo 047 100090b6 | ラ~ | う~ | 0.00 |
| ParaisoRoad 059 1000a276 | さ | ざ | 0.00 |
| ParaisoRoad 059 1000a27a | さわ | ざ | 0.00 |
| ParaisoRoad 043 1000a239 | だっ | ザッ | 0.00 |
| ParaisoRoad 036 1000a20d | ドゴ | ぐぐ | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | ム | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | ムム | 0.00 |
| ParaisoRoad 103 1000a369 | チン | ヌこ | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | ム | 0.00 |
| ParaisoRoad 103 1000a358 | わー | キー | 0.00 |
| ParaisoRoad 059 1000a27d | ぁ | あ | 0.00 |
| ParaisoRoad 052 1000a25b | バリ | ぶり | 0.00 |
