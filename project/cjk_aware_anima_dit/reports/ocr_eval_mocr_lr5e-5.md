# OCR eval — `mocr_lr5e-5` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 19 s for 5117 crops (269.7 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1879 | 73.5 | 0.884 | 82.1 % | 0 |
| speech | 2559 | 1596 | 62.4 | 0.975 | 96.7 % | 0 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 53.4 | 0.795 |
| square | 486 | 79.6 | 0.881 |
| vertical | 1448 | 80.0 | 0.924 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 70.5 | 0.720 |
| 2 | 1017 | 85.5 | 0.906 |
| 3 | 730 | 79.5 | 0.903 |
| 4 | 315 | 61.9 | 0.880 |
| 5 | 167 | 52.1 | 0.856 |
| 6 | 135 | 49.6 | 0.895 |
| 7 | 48 | 33.3 | 0.792 |
| 8+ | 58 | 3.4 | 0.675 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| LoveHina_vol14 004 10006d73 | えう | さっ | 0.00 |
| SyabondamaKieta 055 1000c2e7 | ァァ | アア | 0.00 |
| LoveHina_vol14 003 10006d70 | んにゃー | ヒャー | 0.00 |
| SaladDays_vol18 085 1000b81e | ア | イ | 0.00 |
| SaladDays_vol18 085 1000b81d | イ | オ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b831 | プァ | オ | 0.00 |
| LoveHina_vol14 062 10006f38 | き | オ | 0.00 |
| LoveHina_vol14 062 10006f34 | オオ | キキ | 0.00 |
| SaladDays_vol18 085 1000b821 | プア | ぺっ | 0.00 |
| SaladDays_vol18 085 1000b820 | タ | ? | 0.00 |
| SaladDays_vol18 081 1000b802 | ぱぁ | はは | 0.00 |
| MukoukizuNoChonbo 011 10008f9e | うおおおおおおおおお | だからあああぁぁぁ | 0.00 |
| LoveHina_vol14 062 10006f3a | オオ | キキ | 0.00 |
| LoveHina_vol14 062 10006f39 | オ... | キ・・・ | 0.00 |
| SaladDays_vol18 079 1000b7ff | ザ | バ | 0.00 |
| LoveHina_vol14 059 10006f26 | ガサ | カチャ | 0.00 |
| MukoukizuNoChonbo 034 1000903b | パパパパパパパ | PIPIRIPA | 0.00 |
| MukoukizuNoChonbo 041 10009080 | リー・・・ | りー | 0.00 |
| LoveHina_vol14 019 10006ddf | ぽー | ばー | 0.00 |
| SaladDays_vol18 024 1000b705+1000b706 | ゴッ | バシュウウウン | 0.00 |
| SaladDays_vol18 021 1000b6ff+1000b700 | ブワッ | グバ | 0.00 |
| SaladDays_vol18 044 1000b748 | ドド | ピ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
