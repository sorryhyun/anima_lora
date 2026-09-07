# OCR eval — `hayai_v2_1` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 79 s for 5117 crops (64.8 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1899 | 74.2 | 0.917 | 86.5 % | 28 |
| speech | 2559 | 2006 | 78.4 | 0.987 | 98.1 % | 177 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 61.4 | 0.870 |
| square | 486 | 82.9 | 0.925 |
| vertical | 1448 | 76.9 | 0.934 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 71.6 | 0.769 |
| 2 | 1017 | 85.9 | 0.931 |
| 3 | 730 | 81.1 | 0.934 |
| 4 | 315 | 67.6 | 0.909 |
| 5 | 167 | 49.1 | 0.913 |
| 6 | 135 | 38.5 | 0.902 |
| 7 | 48 | 25.0 | 0.869 |
| 8+ | 58 | 19.0 | 0.811 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 089 1000b84c | プシーッ | SALAD BA | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SyabondamaKieta 078 1000c32a+1000c32b | ウオン! | ドドドドド | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 085 1000b824 | イ | ト | 0.00 |
| SaladDays_vol18 082 1000b806 | ダッ | ゴ | 0.00 |
| SaladDays_vol18 085 1000b81d | イ | ス | 0.00 |
| LoveHina_vol14 051 10006ed3 | ゴ | グ | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ッ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| SaladDays_vol18 049 1000b769 | イ | ィ | 0.00 |
| MukoukizuNoChonbo 049 100090cb | は | ば! | 0.00 |
| MukoukizuNoChonbo 049 100090d2 | は | ば | 0.00 |
| LoveHina_vol14 017 10006dc5 | つるっ | Mar | 0.00 |
| ParaisoRoad 004 1000a12a | ォォ | オオ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | スハハハハ | 0.00 |
| SaladDays_vol18 059 1000b796 | カン | ャフ | 0.00 |
| SaladDays_vol18 032 1000b721 | ズドドド | スジガトン | 0.00 |
| SaladDays_vol18 024 1000b70a | ォ | オ | 0.00 |
| ParaisoRoad 016 1000a183 | じ~ | で~ | 0.00 |
| ParaisoRoad 103 1000a361 | パ | ぴ | 0.00 |
| LoveHina_vol14 025 10006e06 | もがっ | メガ | 0.00 |
| ParaisoRoad 098 1000a337 | ひゅん | バ | 0.00 |
| ParaisoRoad 098 1000a335 | だんっ | ザン | 0.00 |
| ParaisoRoad 104 1000a374 | カリカリ | ガッガッ | 0.00 |
