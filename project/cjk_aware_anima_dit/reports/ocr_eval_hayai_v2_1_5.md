# OCR eval — `hayai_v2_1_5` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 74 s for 5117 crops (69.3 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1398 | 54.7 | 0.826 | 71.8 % | 25 |
| speech | 2559 | 1793 | 70.1 | 0.982 | 97.3 % | 277 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 42.5 | 0.794 |
| square | 486 | 64.0 | 0.825 |
| vertical | 1448 | 56.8 | 0.841 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 64.8 | 0.710 |
| 2 | 1017 | 63.5 | 0.804 |
| 3 | 730 | 64.7 | 0.857 |
| 4 | 315 | 47.3 | 0.852 |
| 5 | 167 | 25.1 | 0.856 |
| 6 | 135 | 14.8 | 0.847 |
| 7 | 48 | 10.4 | 0.811 |
| 8+ | 58 | 12.1 | 0.751 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| LoveHina_vol14 058 10006f14 | バッ | ば...要っ | 0.00 |
| SyabondamaKieta 042 1000c2b4 | ザァァァァァ | ゴワアアアア | 0.00 |
| SyabondamaKieta 050 1000c2e0 | ザアアアアアー・ | キュッ | 0.00 |
| SyabondamaKieta 044 1000c2cf | びく | キュッ | 0.00 |
| SyabondamaKieta 055 1000c2e7 | ァァ | アア | 0.00 |
| LoveHina_vol14 092 10006fd5 | ゴゴゴ... | つっつ... | 0.00 |
| SaladDays_vol18 047 1000b75c | キャー | きゃー | 0.00 |
| SyabondamaKieta 062 1000c2fa | ザク | キュッ | 0.00 |
| MAD_STONE 008 10006ff2+10006ff3 | ガシイ | ズンッ | 0.00 |
| SaladDays_vol18 051 1000b778 | ぁ | あ | 0.00 |
| SaladDays_vol18 050 1000b76d | キィ~・・・ | ギ〜〜... | 0.00 |
| SaladDays_vol18 076 1000b7ee | ザワ | パロ | 0.00 |
| SaladDays_vol18 074 1000b7e1+1000b7e2 | ザワ | アビクトリー | 0.00 |
| SaladDays_vol18 070 1000b7d0 | ザ | #" | 0.00 |
| MAD_STONE 015 10007007 | グワチャ | ギギギ | 0.00 |
| MAD_STONE 009 10006ffa | バラバラバラ | ドクンッ | 0.00 |
| SaladDays_vol18 062 1000b7b8 | ザワ | げっ | 0.00 |
| SaladDays_vol18 068 1000b7c9 | ズズー | スメー | 0.00 |
| SaladDays_vol18 072 1000b7d8 | ワー | ア | 0.00 |
| SaladDays_vol18 072 1000b7d7 | ワー | 7 | 0.00 |
| SaladDays_vol18 072 1000b7d4 | ワー | アッ | 0.00 |
| SaladDays_vol18 072 1000b7d6 | キャー | きゃー | 0.00 |
| SaladDays_vol18 059 1000b796 | カン | わっ | 0.00 |
| SaladDays_vol18 062 1000b7b9 | ザワ | ギロ | 0.00 |
| SaladDays_vol18 072 1000b7d3 | ワー | ア | 0.00 |
