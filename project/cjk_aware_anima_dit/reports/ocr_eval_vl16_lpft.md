# OCR eval — `vl16_lpft` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 149 s for 5117 crops (34.3 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2157 | 84.3 | 0.932 | 88.7 % | 26 |
| speech | 2559 | 2242 | 87.6 | 0.987 | 98.1 % | 162 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 76.1 | 0.898 |
| square | 486 | 84.6 | 0.913 |
| vertical | 1448 | 87.8 | 0.953 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 77.3 | 0.784 |
| 2 | 1017 | 90.7 | 0.945 |
| 3 | 730 | 85.1 | 0.942 |
| 4 | 315 | 80.0 | 0.937 |
| 5 | 167 | 76.0 | 0.916 |
| 6 | 135 | 85.2 | 0.949 |
| 7 | 48 | 56.2 | 0.882 |
| 8+ | 58 | 43.1 | 0.824 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 079 1000b7ff | ザ | バ | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ツ | 0.00 |
| MAD_STONE 085 100070a5+100070a6 | フ゛ン | ゴッ! | 0.00 |
| MAD_STONE 007 10006fee+10006fef | バツ | グッ | 0.00 |
| SaladDays_vol18 059 1000b796 | カン | わっ | 0.00 |
| ParaisoRoad 008 1000a146 | ギリ | ザッ | 0.00 |
| SaladDays_vol18 044 1000b74a | キャー | きゃー | 0.00 |
| SaladDays_vol18 043 1000b742 | キィ~・・・ | えぃ~・・・ | 0.00 |
| MukoukizuNoChonbo 047 100090b6 | ラ~ | う~ | 0.00 |
| MukoukizuNoChonbo 041 1000907a | リー | ソ・・・ | 0.00 |
| SaladDays_vol18 082 1000b806 | ダッ | ゴ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | バキッ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| ParaisoRoad 059 1000a276 | さ | ざ | 0.00 |
| SaladDays_vol18 009 1000b6a8 | ギャー | ぎゃー | 0.00 |
| ParaisoRoad 033 1000a1f8 | ぁぁ | ああ | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コン | 0.00 |
| SaladDays_vol18 005 1000b694 | ッ | ハー | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | イ | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | ムム | 0.00 |
| ParaisoRoad 059 1000a277 | あ | きゃ | 0.00 |
| ParaisoRoad 037 1000a218 | じ~ん | ビー | 0.00 |
| ParaisoRoad 018 1000a198 | ィィ | イイ | 0.00 |
