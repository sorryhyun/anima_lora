# OCR eval — `v6_cap4_ocr` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 115 s for 5117 crops (44.6 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2180 | 85.2 | 0.937 | 89.5 % | 17 |
| speech | 2559 | 2262 | 88.4 | 0.987 | 98.2 % | 7 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 652 | 78.5 | 0.911 |
| square | 417 | 85.4 | 0.915 |
| vertical | 1489 | 88.1 | 0.955 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 76.1 | 0.780 |
| 2 | 1017 | 91.3 | 0.947 |
| 3 | 730 | 85.5 | 0.946 |
| 4 | 315 | 81.3 | 0.940 |
| 5 | 167 | 74.9 | 0.926 |
| 6 | 135 | 83.7 | 0.960 |
| 7 | 48 | 75.0 | 0.914 |
| 8+ | 58 | 51.7 | 0.894 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 086 1000b831 | プァ | アア | 0.00 |
| SaladDays_vol18 085 1000b824 | イ | ィ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 081 1000b802 | ぱぁ | はあ | 0.00 |
| SaladDays_vol18 079 1000b7ff | ザ | バ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | ドギッ | 0.00 |
| SaladDays_vol18 024 1000b70a | ォ | オ | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | ん | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | ん | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | んん | 0.00 |
| SaladDays_vol18 005 1000b694 | ッ | オン | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コソ | 0.00 |
| SaladDays_vol18 021 1000b6fa | カッ | ん | 0.00 |
| SaladDays_vol18 027 1000b718 | ゴ | ドカン | 0.00 |
| SaladDays_vol18 001 1000b67f | ガヤ | がや | 0.00 |
| ParaisoRoad 098 1000a337 | ひゅん | バリリ | 0.00 |
| SaladDays_vol18 001 1000b677 | ガヤ | がや | 0.00 |
| ParaisoRoad 103 1000a369 | チン | ざ | 0.00 |
| ParaisoRoad 103 1000a34c | チン | も | 0.00 |
| ParaisoRoad 103 1000a358 | わー | キー | 0.00 |
| MAD_STONE 075 10007090+10007091 | スラ | トン | 0.00 |
| MAD_STONE 085 100070a5+100070a6 | フ゛ン | ゴッ | 0.00 |
| ParaisoRoad 092 1000a31f | たっ | オハ | 0.00 |
