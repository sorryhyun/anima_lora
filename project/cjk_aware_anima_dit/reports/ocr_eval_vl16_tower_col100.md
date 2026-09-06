# OCR eval — `vl16_tower_col100` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 155 s for 5117 crops (33.0 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2128 | 83.2 | 0.936 | 88.9 % | 29 |
| speech | 2559 | 2115 | 82.6 | 0.986 | 98.0 % | 167 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 71.3 | 0.904 |
| square | 486 | 85.8 | 0.930 |
| vertical | 1448 | 87.4 | 0.952 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 76.1 | 0.795 |
| 2 | 1017 | 90.9 | 0.947 |
| 3 | 730 | 85.1 | 0.946 |
| 4 | 315 | 76.5 | 0.937 |
| 5 | 167 | 75.4 | 0.940 |
| 6 | 135 | 73.3 | 0.952 |
| 7 | 48 | 60.4 | 0.910 |
| 8+ | 58 | 36.2 | 0.804 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| LoveHina_vol14 003 10006d70 | んにゃー | ヒヂー | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ツ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | ギオオン | 0.00 |
| LoveHina_vol14 087 10006fc0 | へへへっ | フフフフ | 0.00 |
| SaladDays_vol18 082 1000b806 | ダッ | ゴ | 0.00 |
| SaladDays_vol18 024 1000b70a | ォ | オ | 0.00 |
| MukoukizuNoChonbo 041 1000907a | リー | ソ・・・ | 0.00 |
| ParaisoRoad 004 1000a12a | ォォ | オオ | 0.00 |
| MukoukizuNoChonbo 047 100090b6 | ラ~ | う~ | 0.00 |
| MukoukizuNoChonbo 049 100090d8 | は | ぼ | 0.00 |
| MukoukizuNoChonbo 041 10009084 | リー | ソー | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| SaladDays_vol18 005 1000b694 | ッ | シ | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | ム | 0.00 |
| ParaisoRoad 036 1000a20d | ドゴ | ぐぐ | 0.00 |
| ParaisoRoad 037 1000a212 | じ~ん | ど~ | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | イ | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | んん | 0.00 |
| SaladDays_vol18 001 1000b677 | ガヤ | がや | 0.00 |
| ParaisoRoad 103 1000a369 | チン | ヌこ | 0.00 |
| ParaisoRoad 057 1000a26a | ど | と | 0.00 |
| ParaisoRoad 018 1000a19b | ィ | イ | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コソ | 0.00 |
