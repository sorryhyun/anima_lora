# OCR eval — `vl16` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 127 s for 5117 crops (40.3 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 772 | 30.2 | 0.545 | 39.5 % | 91 |
| speech | 2559 | 1623 | 63.4 | 0.976 | 96.2 % | 240 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 20.8 | 0.485 |
| square | 486 | 27.4 | 0.464 |
| vertical | 1448 | 35.2 | 0.598 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 36.4 | 0.379 |
| 2 | 1017 | 39.6 | 0.547 |
| 3 | 730 | 30.1 | 0.550 |
| 4 | 315 | 20.0 | 0.545 |
| 5 | 167 | 10.2 | 0.536 |
| 6 | 135 | 21.5 | 0.597 |
| 7 | 48 | 12.5 | 0.613 |
| 8+ | 58 | 3.4 | 0.551 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| ParaisoRoad 091 1000a31a | オギャァ | ブザラ | 0.00 |
| LoveHina_vol14 083 10006f90 | ビー | ピー | 0.00 |
| LoveHina_vol14 083 10006f92 | ビー | ピー | 0.00 |
| LoveHina_vol14 083 10006f94 | ドドドド | 雪 | 0.00 |
| LoveHina_vol14 083 10006f95 | キキーッ | 静け | 0.00 |
| LoveHina_vol14 083 10006f96 | ああああっ | ぬぬぬぬ | 0.00 |
| LoveHina_vol14 084 10006f99 | ブォン.. | ぐに！？ | 0.00 |
| LoveHina_vol14 084 10006f9d+10006f9e | ドン | ヒトル | 0.00 |
| ParaisoRoad 088 1000a303 | ゴッゴッ | うっっっっっっっっっっっっっっっっっっっっっっっっっっっっっっっっっっっっっっっ | 0.00 |
| ParaisoRoad 089 1000a304 | ズザザザザザ | ふーふー
ふーふー
ふーふー
ふーふー
ふーふー | 0.00 |
| ParaisoRoad 091 1000a318 | オギャァ | ぜ気 | 0.00 |
| ParaisoRoad 091 1000a319 | オギャァ | 大出 | 0.00 |
| LoveHina_vol14 083 10006f8d | ドカーンッ | あー
だから
蛍走
でなんて | 0.00 |
| ParaisoRoad 083 1000a2e4 | ぶん | ふ～～～ | 0.00 |
| ParaisoRoad 089 1000a307 | ピー | ビー | 0.00 |
| ParaisoRoad 089 1000a308 | ひょっ | んちー | 0.00 |
| ParaisoRoad 089 1000a309 | ぐえ | ぇ | 0.00 |
| ParaisoRoad 089 1000a30b | ピー | ち | 0.00 |
| ParaisoRoad 090 1000a30d | ゲロゲロゲロ | ははは | 0.00 |
| ParaisoRoad 091 1000a311 | ハァ | い？ | 0.00 |
| ParaisoRoad 091 1000a312 | ハァ | ケーフ | 0.00 |
| LoveHina_vol14 085 10006fa2 | ブオオオ・・ | ぇぇぇぇぇぇぇぇぇぇぇぇぇぇぇぇ | 0.00 |
| LoveHina_vol14 085 10006fa3 | きゅ~ | キャ～～ | 0.00 |
| LoveHina_vol14 085 10006faa | ザザァ・・・ン | 牛排food | 0.00 |
| ParaisoRoad 092 1000a31e | ザザザザザ | 手
手手 | 0.00 |
