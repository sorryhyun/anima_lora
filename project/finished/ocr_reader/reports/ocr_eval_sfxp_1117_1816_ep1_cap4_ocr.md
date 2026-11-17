# OCR eval — `sfxp_1117_1816_ep1_cap4_ocr` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 127 s for 5117 crops (40.4 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 201 | 7.9 | 0.087 | 8.3 % | 0 |
| speech | 2559 | 2195 | 85.8 | 0.977 | 97.1 % | 31 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 652 | 0.5 | 0.006 |
| square | 417 | 5.5 | 0.062 |
| vertical | 1489 | 11.8 | 0.130 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 9.1 | 0.136 |
| 2 | 1017 | 8.6 | 0.095 |
| 3 | 730 | 7.9 | 0.086 |
| 4 | 315 | 8.6 | 0.097 |
| 5 | 167 | 3.0 | 0.030 |
| 6 | 135 | 10.4 | 0.104 |
| 7 | 48 | 2.1 | 0.021 |
| 8+ | 58 | 1.7 | 0.017 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| ParaisoRoad 044 1000a23d | うっ・・ |  | 0.00 |
| SyabondamaKieta 027 1000c297 | くすっ・・・ |  | 0.00 |
| SyabondamaKieta 025 1000c295 | たたっ |  | 0.00 |
| SyabondamaKieta 042 1000c2b4 | ザァァァァァ |  | 0.00 |
| SyabondamaKieta 040 1000c2b1 | ばっ |  | 0.00 |
| SyabondamaKieta 040 1000c2b0 | バタ |  | 0.00 |
| SyabondamaKieta 040 1000c2af | バタ |  | 0.00 |
| SyabondamaKieta 040 1000c2ae | バタ |  | 0.00 |
| SyabondamaKieta 039 1000c2ad | ガサ |  | 0.00 |
| ParaisoRoad 044 1000a23e | ぐすっ |  | 0.00 |
| SyabondamaKieta 027 1000c298 | きゅっ・・・ |  | 0.00 |
| ParaisoRoad 044 1000a23c | カシャン |  | 0.00 |
| ParaisoRoad 043 1000a23b | ピタッ |  | 0.00 |
| ParaisoRoad 043 1000a23a | ゴッ |  | 0.00 |
| ParaisoRoad 052 1000a259 | いっ?! |  | 0.00 |
| ParaisoRoad 051 1000a258 | ゴオン |  | 0.00 |
| ParaisoRoad 051 1000a257 | ゴオン |  | 0.00 |
| ParaisoRoad 051 1000a256 | ゴオン |  | 0.00 |
| ParaisoRoad 048 1000a254 | ぴゅ~ |  | 0.00 |
| ParaisoRoad 045 1000a247 | ビクン |  | 0.00 |
| ParaisoRoad 042 1000a231 | ヒソ |  | 0.00 |
| ParaisoRoad 041 1000a230 | ドン |  | 0.00 |
| ParaisoRoad 041 1000a22f | わー |  | 0.00 |
| ParaisoRoad 041 1000a22e | ドン |  | 0.00 |
| ParaisoRoad 041 1000a22c | ドン |  | 0.00 |
