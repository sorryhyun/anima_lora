# OCR eval — `vl16_p2b_kozh_nl_swaprelax_ep2` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 119 s for 5117 crops (43.0 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2181 | 85.3 | 0.938 | 89.4 % | 24 |
| speech | 2559 | 2275 | 88.9 | 0.988 | 98.3 % | 7 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 631 | 76.1 | 0.906 |
| square | 490 | 84.7 | 0.916 |
| vertical | 1437 | 89.5 | 0.959 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 77.3 | 0.792 |
| 2 | 1017 | 91.6 | 0.948 |
| 3 | 730 | 85.9 | 0.948 |
| 4 | 315 | 79.0 | 0.934 |
| 5 | 167 | 78.4 | 0.933 |
| 6 | 135 | 83.7 | 0.953 |
| 7 | 48 | 68.8 | 0.911 |
| 8+ | 58 | 48.3 | 0.868 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b831 | プァ | アア | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ッ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 079 1000b7ff | ザ | バ | 0.00 |
| SaladDays_vol18 024 1000b70a | ォ | オ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | ドギッ | 0.00 |
| MAD_STONE 075 10007090+10007091 | スラ | ドク | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | ん | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コソ | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | んん | 0.00 |
| SaladDays_vol18 005 1000b694 | ッ | オン | 0.00 |
| MAD_STONE 066 1000707a+1000707b | カツ | ガッ | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | ん | 0.00 |
| SaladDays_vol18 021 1000b6fa | カッ | ん | 0.00 |
| SaladDays_vol18 001 1000b67f | ガヤ | がや | 0.00 |
| ParaisoRoad 103 1000a369 | チン | ざ | 0.00 |
| SaladDays_vol18 001 1000b677 | ガヤ | がや | 0.00 |
| MAD_STONE 085 100070a5+100070a6 | フ゛ン | ブッ | 0.00 |
| ParaisoRoad 098 1000a332 | かっ | ガコ | 0.00 |
| ParaisoRoad 103 1000a350 | わー | あー | 0.00 |
| ParaisoRoad 103 1000a34c | チン | も | 0.00 |
| ParaisoRoad 103 1000a358 | わー | キー | 0.00 |
| ParaisoRoad 004 1000a12a | ォォ | オオ | 0.00 |
