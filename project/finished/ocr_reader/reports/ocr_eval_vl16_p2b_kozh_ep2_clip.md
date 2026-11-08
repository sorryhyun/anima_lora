# OCR eval — `vl16_p2b_kozh_ep2_clip` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 125 s for 5117 crops (41.0 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2177 | 85.1 | 0.932 | 89.0 % | 17 |
| speech | 2559 | 2272 | 88.8 | 0.988 | 98.2 % | 9 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 631 | 77.8 | 0.900 |
| square | 490 | 84.7 | 0.915 |
| vertical | 1437 | 88.4 | 0.953 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 76.1 | 0.788 |
| 2 | 1017 | 90.9 | 0.941 |
| 3 | 730 | 85.6 | 0.945 |
| 4 | 315 | 80.6 | 0.939 |
| 5 | 167 | 76.6 | 0.917 |
| 6 | 135 | 85.9 | 0.957 |
| 7 | 48 | 72.9 | 0.911 |
| 8+ | 58 | 48.3 | 0.820 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 085 1000b81d | イ | ィ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b831 | プァ | アア | 0.00 |
| SaladDays_vol18 079 1000b7ff | ザ | !! | 0.00 |
| SaladDays_vol18 081 1000b802 | ぱぁ | ぽあ | 0.00 |
| SaladDays_vol18 085 1000b824 | イ | ィ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | ドオ | 0.00 |
| SaladDays_vol18 082 1000b806 | ダッ | ゴ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コソ | 0.00 |
| SaladDays_vol18 005 1000b694 | ッ | カン | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | イ | 0.00 |
| SaladDays_vol18 021 1000b6fa | カッ | ム | 0.00 |
| SaladDays_vol18 024 1000b70a | ォ | オ | 0.00 |
| LoveHina_vol14 043 10006e9d | ダッ | タリ | 0.00 |
| ParaisoRoad 103 1000a369 | チン | ぇこ | 0.00 |
| SaladDays_vol18 001 1000b67f | ガヤ | がや | 0.00 |
| SaladDays_vol18 001 1000b677 | ガヤ | がや | 0.00 |
| SaladDays_vol18 009 1000b6ac+1000b6ad | ギラッ | キン | 0.00 |
| SaladDays_vol18 013 1000b6ba | カア | わっ | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | ムム | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | ム | 0.00 |
| MAD_STONE 075 10007090+10007091 | スラ | トッ | 0.00 |
| MAD_STONE 085 100070a7+100070a8 | バッ | ゴゴゴ | 0.00 |
