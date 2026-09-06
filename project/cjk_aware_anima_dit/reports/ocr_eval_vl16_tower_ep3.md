# OCR eval — `vl16_tower_ep3` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 148 s for 5117 crops (34.6 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2191 | 85.7 | 0.948 | 91.1 % | 25 |
| speech | 2559 | 2132 | 83.3 | 0.987 | 98.1 % | 168 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 76.1 | 0.925 |
| square | 486 | 87.0 | 0.935 |
| vertical | 1448 | 89.3 | 0.963 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 75.0 | 0.769 |
| 2 | 1017 | 93.0 | 0.959 |
| 3 | 730 | 88.2 | 0.959 |
| 4 | 315 | 80.0 | 0.958 |
| 5 | 167 | 75.4 | 0.933 |
| 6 | 135 | 77.8 | 0.960 |
| 7 | 48 | 58.3 | 0.908 |
| 8+ | 58 | 41.4 | 0.887 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| LoveHina_vol14 062 10006f38 | き | オ | 0.00 |
| SaladDays_vol18 085 1000b81d | イ | ィ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ッ | 0.00 |
| SaladDays_vol18 085 1000b824 | イ | ィ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| ParaisoRoad 018 1000a19b | ィ | イ | 0.00 |
| SaladDays_vol18 024 1000b70a | ォ | オ | 0.00 |
| ParaisoRoad 033 1000a1f8 | ぁぁ | ああ | 0.00 |
| ParaisoRoad 018 1000a199 | ィ | イ | 0.00 |
| ParaisoRoad 018 1000a198 | ィィ | イイ | 0.00 |
| ParaisoRoad 004 1000a12a | ォォ | オオ | 0.00 |
| MukoukizuNoChonbo 047 100090b6 | ラ~ | う~ | 0.00 |
| ParaisoRoad 017 1000a190 | くわん | ちゅ | 0.00 |
| ParaisoRoad 093 1000a322 | ザザザザザザ | ガササササ | 0.00 |
| ParaisoRoad 066 1000a2a1 | かっ | カン | 0.00 |
| ParaisoRoad 084 1000a2eb | ふっ | ひー | 0.00 |
| ParaisoRoad 085 1000a2f2 | がー | ガー | 0.00 |
| ParaisoRoad 085 1000a2f0 | きゃー | はぁー | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | ヶ | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | ムム | 0.00 |
| SaladDays_vol18 005 1000b694 | ッ | ガー | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | ん | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コソ | 0.00 |
