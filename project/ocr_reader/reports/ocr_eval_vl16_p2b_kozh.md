# OCR eval — `vl16_p2b_kozh` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 126 s for 5117 crops (40.5 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2174 | 85.0 | 0.938 | 89.0 % | 19 |
| speech | 2559 | 2265 | 88.5 | 0.986 | 98.0 % | 14 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 76.4 | 0.908 |
| square | 486 | 86.8 | 0.926 |
| vertical | 1448 | 88.1 | 0.955 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 80.7 | 0.818 |
| 2 | 1017 | 90.8 | 0.948 |
| 3 | 730 | 85.8 | 0.948 |
| 4 | 315 | 80.0 | 0.936 |
| 5 | 167 | 75.4 | 0.917 |
| 6 | 135 | 88.9 | 0.966 |
| 7 | 48 | 66.7 | 0.882 |
| 8+ | 58 | 41.4 | 0.874 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 082 1000b806 | ダッ | ゴ | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ツ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | ドギャ | 0.00 |
| SaladDays_vol18 044 1000b74c | ズキ | ベチ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| ParaisoRoad 103 1000a369 | チン | ヌこ | 0.00 |
| SaladDays_vol18 008 1000b69b | ゴ | コソ | 0.00 |
| SaladDays_vol18 005 1000b694 | ッ | カン | 0.00 |
| SaladDays_vol18 003 1000b689 | ケケ | ィィ | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | イ | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | イ | 0.00 |
| ParaisoRoad 103 1000a358 | わー | キー | 0.00 |
| SaladDays_vol18 020 1000b6ec | ワー | アー | 0.00 |
| SaladDays_vol18 024 1000b70a | ォ | オ | 0.00 |
| LoveHina_vol14 022 10006de8 | つつ・・ | フフ… | 0.00 |
| ParaisoRoad 018 1000a198 | ィィ | イイ | 0.00 |
| ParaisoRoad 018 1000a199 | ィ | イ | 0.00 |
| ParaisoRoad 018 1000a19b | ィ | イ | 0.00 |
| ParaisoRoad 085 1000a2f2 | がー | わかー | 0.00 |
| LoveHina_vol14 019 10006ddf | ぽー | ぼー | 0.00 |
| ParaisoRoad 092 1000a31f | たっ | オ | 0.00 |
| ParaisoRoad 092 1000a31e | ザザザザザ | ギギギギギ… | 0.00 |
| MukoukizuNoChonbo 041 1000907a | リー | ソー | 0.00 |
