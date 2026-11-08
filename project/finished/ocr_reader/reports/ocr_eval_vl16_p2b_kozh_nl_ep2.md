# OCR eval — `vl16_p2b_kozh_nl_ep2` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 123 s for 5117 crops (41.8 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 2172 | 84.9 | 0.934 | 89.2 % | 22 |
| speech | 2559 | 2262 | 88.4 | 0.987 | 98.1 % | 8 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 631 | 76.4 | 0.900 |
| square | 490 | 85.7 | 0.916 |
| vertical | 1437 | 88.4 | 0.955 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 76.1 | 0.780 |
| 2 | 1017 | 91.6 | 0.947 |
| 3 | 730 | 85.8 | 0.949 |
| 4 | 315 | 78.7 | 0.933 |
| 5 | 167 | 75.4 | 0.915 |
| 6 | 135 | 83.0 | 0.946 |
| 7 | 48 | 75.0 | 0.922 |
| 8+ | 58 | 43.1 | 0.795 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| LoveHina_vol14 062 10006f39 | オ... | キ… | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b831 | プァ | アア | 0.00 |
| SaladDays_vol18 081 1000b802 | ぱぁ | はあ | 0.00 |
| SaladDays_vol18 079 1000b7ff | ザ | バ | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ツ | 0.00 |
| SaladDays_vol18 085 1000b824 | イ | ィ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | ドギャ | 0.00 |
| MAD_STONE 008 10006ff2+10006ff3 | ガシイ | ドン | 0.00 |
| MAD_STONE 050 10007050 | グワシィ | バァンイ | 0.00 |
| SaladDays_vol18 024 1000b70a | ォ | オ | 0.00 |
| MAD_STONE 075 10007090+10007091 | スラ | トン | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| MAD_STONE 085 100070a5+100070a6 | フ゛ン | ゴ | 0.00 |
| ParaisoRoad 098 1000a335 | だんっ | ザワ | 0.00 |
| ParaisoRoad 016 1000a183 | じ~ | ご~ | 0.00 |
| ParaisoRoad 035 1000a1fe | ん | ス! | 0.00 |
| ParaisoRoad 033 1000a1f8 | ぁぁ | ああ | 0.00 |
| ParaisoRoad 026 1000a1cc | ヘェ | フエ | 0.00 |
| ParaisoRoad 018 1000a198 | ィィ | イイ | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | ん | 0.00 |
| ParaisoRoad 018 1000a19b | ィ | イ | 0.00 |
| ParaisoRoad 018 1000a199 | ィ | イ | 0.00 |
| SaladDays_vol18 003 1000b685 | ケ | ん | 0.00 |
