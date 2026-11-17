# OCR eval — `hayai27_auto` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 9 s for 5117 crops (548.0 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2044 | 79.9 | 0.916 | 86.1 % | 26 |
| speech | 2559 | 2214 | 86.5 | 0.987 | 98.2 % | 22 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 631 | 70.5 | 0.879 |
| square | 490 | 81.6 | 0.896 |
| vertical | 1437 | 83.4 | 0.940 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 75.0 | 0.750 |
| 2 | 1017 | 85.8 | 0.925 |
| 3 | 730 | 81.9 | 0.932 |
| 4 | 315 | 75.2 | 0.914 |
| 5 | 167 | 67.1 | 0.924 |
| 6 | 135 | 75.6 | 0.925 |
| 7 | 48 | 66.7 | 0.899 |
| 8+ | 58 | 41.4 | 0.812 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SyabondamaKieta 091 1000c353 | バサ | 随着... | 0.00 |
| SyabondamaKieta 078 1000c32a+1000c32b | ウオン! | ド | 0.00 |
| SyabondamaKieta 044 1000c2cf | びく | ベラッ | 0.00 |
| LoveHina_vol14 062 10006f38 | き | ブ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア... | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 085 1000b822 | リリ | ニニ | 0.00 |
| SaladDays_vol18 044 1000b748 | ドド | お茶ますぅ、♡ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| SaladDays_vol18 079 1000b7ff | ザ | ぶ | 0.00 |
| SaladDays_vol18 085 1000b81b | リ | り | 0.00 |
| SaladDays_vol18 085 1000b81d | イ | ス | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ッ | 0.00 |
| SaladDays_vol18 085 1000b824 | イ | ト | 0.00 |
| SaladDays_vol18 085 1000b821 | プア | ペイ | 0.00 |
| ParaisoRoad 103 1000a35d | わー | ねー | 0.00 |
| SaladDays_vol18 009 1000b6ac+1000b6ad | ギラッ | キ | 0.00 |
| SaladDays_vol18 008 1000b69a | ゴ | コ | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コソ | 0.00 |
| SaladDays_vol18 008 1000b69e | ゴゴ | コッ | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | イ | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | ムイ | 0.00 |
| ParaisoRoad 104 1000a373 | ガオーッ | チャ | 0.00 |
| ParaisoRoad 103 1000a366 | チン! | ケレ | 0.00 |
| ParaisoRoad 103 1000a361 | パ | ぷ | 0.00 |
