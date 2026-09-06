# OCR eval — `vl16_tower_lr1e-5` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 152 s for 5117 crops (33.7 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2089 | 81.7 | 0.927 | 87.6 % | 25 |
| speech | 2559 | 2120 | 82.8 | 0.986 | 97.9 % | 164 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 71.0 | 0.892 |
| square | 486 | 83.7 | 0.915 |
| vertical | 1448 | 85.6 | 0.946 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 75.0 | 0.769 |
| 2 | 1017 | 90.1 | 0.942 |
| 3 | 730 | 83.0 | 0.933 |
| 4 | 315 | 75.2 | 0.931 |
| 5 | 167 | 70.7 | 0.916 |
| 6 | 135 | 76.3 | 0.949 |
| 7 | 48 | 58.3 | 0.887 |
| 8+ | 58 | 25.9 | 0.789 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| MAD_STONE 008 10006ff2+10006ff3 | ガシイ | ザッス | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ツ | 0.00 |
| LoveHina_vol14 041 10006e9b | ぺたん | ペキシ | 0.00 |
| MAD_STONE 007 10006fee+10006fef | バツ | グッ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| MukoukizuNoChonbo 041 1000907a | リー | ソ・・・ | 0.00 |
| MukoukizuNoChonbo 041 10009084 | リー | ソー | 0.00 |
| ParaisoRoad 004 1000a12a | ォォ | オオ | 0.00 |
| ParaisoRoad 009 1000a151+1000a152 | じ~ーっ | ビーン | 0.00 |
| SaladDays_vol18 043 1000b742 | キィ~・・・ | えぃ~・・・ | 0.00 |
| SaladDays_vol18 059 1000b796 | カン | ヤフ | 0.00 |
| SaladDays_vol18 050 1000b76d | キィ~・・・ | ギー・・・ | 0.00 |
| MAD_STONE 085 100070a5+100070a6 | フ゛ン | バシ! | 0.00 |
| SaladDays_vol18 001 1000b677 | ガヤ | がや | 0.00 |
| ParaisoRoad 033 1000a1f8 | ぁぁ | ああ | 0.00 |
| ParaisoRoad 033 1000a1f0 | おう | むぅ | 0.00 |
| ParaisoRoad 018 1000a199 | ィ | イ | 0.00 |
| ParaisoRoad 018 1000a198 | ィィ | イイ | 0.00 |
| SaladDays_vol18 009 1000b6a8 | ギャー | ぎゅー | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コン | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | ィィ | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | イ | 0.00 |
| SaladDays_vol18 020 1000b6ec | ワー | アー | 0.00 |
