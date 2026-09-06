# OCR eval — `ppocr` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 19 s for 5117 crops (275.4 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 183 | 7.2 | 0.194 | 11.8 % | 0 |
| speech | 2559 | 332 | 13.0 | 0.297 | 21.5 % | 0 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 5.1 | 0.209 |
| square | 486 | 3.7 | 0.074 |
| vertical | 1448 | 9.2 | 0.227 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 21.6 | 0.216 |
| 2 | 1017 | 7.4 | 0.136 |
| 3 | 730 | 9.3 | 0.218 |
| 4 | 315 | 5.7 | 0.226 |
| 5 | 167 | 0.6 | 0.246 |
| 6 | 135 | 0.0 | 0.326 |
| 7 | 48 | 2.1 | 0.226 |
| 8+ | 58 | 1.7 | 0.217 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| MukoukizuNoChonbo 055 10009114 | カタ | の | 0.00 |
| MukoukizuNoChonbo 057 10009122 | どっぽーん・・・・ |  | 0.00 |
| SaladDays_vol18 020 1000b6eb | ザワ | 苦 | 0.00 |
| SaladDays_vol18 020 1000b6ec | ワー | S | 0.00 |
| SaladDays_vol18 020 1000b6ed | ザワ | 苦 | 0.00 |
| SaladDays_vol18 020 1000b6ee | ザワ | 苦 | 0.00 |
| SaladDays_vol18 020 1000b6ef | ワー | X | 0.00 |
| MukoukizuNoChonbo 055 1000911c | カタ | の | 0.00 |
| MukoukizuNoChonbo 055 1000911b | カタ | たっ | 0.00 |
| SaladDays_vol18 020 1000b6f0 | ワー | V | 0.00 |
| SaladDays_vol18 020 1000b6f1 | ワー | X | 0.00 |
| MukoukizuNoChonbo 055 10009118 | ぎくっ | しレの | 0.00 |
| SaladDays_vol18 020 1000b6f2 | ザワ | 苦 | 0.00 |
| SaladDays_vol18 020 1000b6f3 | ザワ | 苦 | 0.00 |
| SaladDays_vol18 020 1000b6f4 | ザワ | 苦 | 0.00 |
| SaladDays_vol18 019 1000b6ea | ザワ | 十 | 0.00 |
| SaladDays_vol18 020 1000b6f5 | ザワ | 第 | 0.00 |
| SaladDays_vol18 021 1000b6f6 | ザザッ | 4 | 0.00 |
| SaladDays_vol18 021 1000b6f8 | ザッ | # | 0.00 |
| SaladDays_vol18 021 1000b6f9 | ザッ |  | 0.00 |
| SaladDays_vol18 021 1000b6fa | カッ | C | 0.00 |
| MukoukizuNoChonbo 055 1000910e | カタ | り | 0.00 |
| SaladDays_vol18 021 1000b6fb | スッ | 又 | 0.00 |
| SaladDays_vol18 021 1000b6fc | スッ | 否 | 0.00 |
| SaladDays_vol18 021 1000b6fd | ザザッ | h | 0.00 |
