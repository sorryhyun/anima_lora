# OCR eval — `mocr_lr2e-5` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 796 s for 5117 crops (6.4 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1821 | 71.2 | 0.872 | 80.1 % | 0 |
| speech | 2559 | 1579 | 61.7 | 0.975 | 96.5 % | 0 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 51.4 | 0.783 |
| square | 486 | 77.8 | 0.877 |
| vertical | 1448 | 77.5 | 0.908 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 69.3 | 0.708 |
| 2 | 1017 | 84.2 | 0.901 |
| 3 | 730 | 75.8 | 0.886 |
| 4 | 315 | 59.0 | 0.855 |
| 5 | 167 | 50.9 | 0.839 |
| 6 | 135 | 47.4 | 0.867 |
| 7 | 48 | 29.2 | 0.791 |
| 8+ | 58 | 3.4 | 0.674 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SyabondamaKieta 089 1000c34c | ぎゃははは | ざわばな | 0.00 |
| SyabondamaKieta 044 1000c2cf | びく | ズッ | 0.00 |
| SyabondamaKieta 055 1000c2e7 | ァァ | アア | 0.00 |
| SyabondamaKieta 055 1000c2e6 | ザァァァァ | ガアアア | 0.00 |
| MAD_STONE 009 10006ffa | バラバラバラ | ドスッ | 0.00 |
| MAD_STONE 015 10007007 | グワチャ | ボオオオ | 0.00 |
| LoveHina_vol14 091 10006fcb | キャー | もー | 0.00 |
| SyabondamaKieta 042 1000c2b9 | カッ | ゴゴゴ | 0.00 |
| MAD_STONE 050 10007050 | グワシィ | ガタン! | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| MAD_STONE 048 1000704c+1000704d | ズン | ガッ | 0.00 |
| MAD_STONE 045 10007045+10007046 | ガシャ | ザッ | 0.00 |
| MAD_STONE 060 1000706a | バギャ | ボキ | 0.00 |
| MAD_STONE 057 10007066+10007067 | バンッ | ドドド | 0.00 |
| MAD_STONE 056 10007061 | ウィイィ | わんんん | 0.00 |
| LoveHina_vol14 083 10006f94 | ドドドド | ヒュン! | 0.00 |
| MAD_STONE 064 10007079 | キゥイィイ | きゃんん | 0.00 |
| MAD_STONE 087 100070af+100070b0 | ゴワッ | ドドド | 0.00 |
| MAD_STONE 077 10007095+10007096 | ズッ | アアン | 0.00 |
| SaladDays_vol18 086 1000b831 | プァ | オオ | 0.00 |
| SaladDays_vol18 081 1000b804 | やあぁああ | ドキュキュウウ | 0.00 |
| LoveHina_vol14 062 10006f34 | オオ | キキ | 0.00 |
| SaladDays_vol18 071 1000b7d2 | ザ・・・ | ド・・・ | 0.00 |
| SaladDays_vol18 085 1000b81d | イ | ど | 0.00 |
