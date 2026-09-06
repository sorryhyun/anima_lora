# OCR eval — `manga_ocr` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 19 s for 5117 crops (270.3 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 419 | 16.4 | 0.336 | 21.7 % | 0 |
| speech | 2559 | 817 | 31.9 | 0.824 | 70.6 % | 0 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 1292 | 11.5 | 0.296 |
| square | 486 | 28.8 | 0.415 |
| vertical | 780 | 16.8 | 0.354 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 40.9 | 0.438 |
| 2 | 1017 | 24.4 | 0.369 |
| 3 | 730 | 14.2 | 0.328 |
| 4 | 315 | 6.7 | 0.296 |
| 5 | 167 | 3.6 | 0.302 |
| 6 | 135 | 3.0 | 0.302 |
| 7 | 48 | 0.0 | 0.270 |
| 8+ | 58 | 0.0 | 0.177 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| ParaisoRoad 103 1000a35f | パタパタ | 「おめのお兄さん | 0.00 |
| MukoukizuNoChonbo 020 10008fd8 | コチョ | ? | 0.00 |
| MukoukizuNoChonbo 021 10008fdb | ぬう~っ | なら... | 0.00 |
| MukoukizuNoChonbo 021 10008fdc | わーっ | ... | 0.00 |
| MukoukizuNoChonbo 021 10008fdd | カチーン!! | それは、それでも、 | 0.00 |
| MukoukizuNoChonbo 021 10008fde | コロ | ... | 0.00 |
| MukoukizuNoChonbo 021 10008fdf | ざ~っ | ... | 0.00 |
| MukoukizuNoChonbo 021 10008fe2 | コロ | ... | 0.00 |
| MukoukizuNoChonbo 021 10008fe4 | コロ | ... | 0.00 |
| ParaisoRoad 103 1000a35d | わー | ねっ | 0.00 |
| ParaisoRoad 103 1000a35e | パチパ | ... | 0.00 |
| MukoukizuNoChonbo 024 10008ffa | バターン!! | ... | 0.00 |
| ParaisoRoad 103 1000a360 | チン | そ | 0.00 |
| ParaisoRoad 103 1000a361 | パ | ... | 0.00 |
| ParaisoRoad 103 1000a364 | げーっ | ! | 0.00 |
| ParaisoRoad 103 1000a366 | チン! | ん | 0.00 |
| SaladDays_vol18 003 1000b68a | ケ | 6 | 0.00 |
| SaladDays_vol18 003 1000b68b | ガラッ | どう!! | 0.00 |
| MukoukizuNoChonbo 018 10008fc7 | ピシーッ | えー、えーーーっ | 0.00 |
| MukoukizuNoChonbo 018 10008fc8 | うひゃーっ | えええいやー | 0.00 |
| MukoukizuNoChonbo 018 10008fcd | ピシーッ | はっはぁ...うん.. | 0.00 |
| MukoukizuNoChonbo 018 10008fce | ピシーッ | フフーフー | 0.00 |
| MukoukizuNoChonbo 021 10008fea | コロ | っ? | 0.00 |
| ParaisoRoad 103 1000a351 | パタ | ... | 0.00 |
| ParaisoRoad 103 1000a354 | わー | ん | 0.00 |
