# OCR eval — `manga_ocr` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 23 s for 5117 crops (224.7 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 670 | 26.2 | 0.478 | 34.6 % | 0 |
| speech | 2559 | 1588 | 62.1 | 0.975 | 97.1 % | 0 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 10.7 | 0.332 |
| square | 486 | 30.0 | 0.423 |
| vertical | 1448 | 31.6 | 0.560 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 39.8 | 0.430 |
| 2 | 1017 | 35.0 | 0.480 |
| 3 | 730 | 29.7 | 0.508 |
| 4 | 315 | 13.0 | 0.467 |
| 5 | 167 | 6.0 | 0.449 |
| 6 | 135 | 5.9 | 0.488 |
| 7 | 48 | 6.2 | 0.469 |
| 8+ | 58 | 0.0 | 0.280 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| MAD_STONE 039 10007035 | シュー | うんっ、うん | 0.00 |
| MAD_STONE 034 10007028 | カッ | ジェ | 0.00 |
| MAD_STONE 032 10007025 | ガシッ | それは、 | 0.00 |
| MAD_STONE 032 10007022+10007023+10007024 | バァン | それは、 | 0.00 |
| MAD_STONE 031 10007021 | ジャキ | いやっ、 | 0.00 |
| ParaisoRoad 015 1000a179 | しん. | くくっ | 0.00 |
| ParaisoRoad 015 1000a178 | あ | メ | 0.00 |
| MAD_STONE 047 10007048+10007049 | グッ | ... | 0.00 |
| MAD_STONE 045 10007047 | グッ | 11/ | 0.00 |
| MAD_STONE 028 1000701c | ボン | それでも | 0.00 |
| MAD_STONE 027 1000701a | ヒュ | ... | 0.00 |
| MAD_STONE 026 10007019 | パァン | フリーソー | 0.00 |
| MAD_STONE 040 1000703c | ガシッ | わあっ | 0.00 |
| MAD_STONE 039 1000703a | ゴッ | ... | 0.00 |
| MAD_STONE 039 10007037 | カッ | が | 0.00 |
| MAD_STONE 036 1000702a+1000702b | バン | それは、 | 0.00 |
| MAD_STONE 038 10007032 | ビィン | それ | 0.00 |
| MAD_STONE 038 10007031 | ゴン | それは | 0.00 |
| ParaisoRoad 103 1000a36a | わー | ゃー | 0.00 |
| ParaisoRoad 103 1000a369 | チン | えっ | 0.00 |
| ParaisoRoad 014 1000a16b | ドッ | そして、 | 0.00 |
| ParaisoRoad 013 1000a168 | ズン | それは、 | 0.00 |
| MAD_STONE 008 10006ff6 | ポッ | それは... | 0.00 |
| SaladDays_vol18 037 1000b728 | ザワ | ボク | 0.00 |
| SaladDays_vol18 032 1000b721 | ズドドド | あ、あああっ | 0.00 |
