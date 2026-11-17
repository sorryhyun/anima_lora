# OCR eval — `vl16_sfxp_1k_sfx` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 54 s for 2558 crops (47.4 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1651 | 64.5 | 0.828 | 71.8 % | 24 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 631 | 51.8 | 0.781 |
| square | 490 | 68.6 | 0.818 |
| vertical | 1437 | 68.8 | 0.853 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 59.1 | 0.635 |
| 2 | 1017 | 73.4 | 0.850 |
| 3 | 730 | 62.9 | 0.834 |
| 4 | 315 | 58.7 | 0.820 |
| 5 | 167 | 48.5 | 0.818 |
| 6 | 135 | 63.0 | 0.860 |
| 7 | 48 | 50.0 | 0.792 |
| 8+ | 58 | 32.8 | 0.706 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| MukoukizuNoChonbo 041 1000907a | リー | ソー | 0.00 |
| ParaisoRoad 099 1000a33b | べしっ | ゴ | 0.00 |
| ParaisoRoad 097 1000a32b | ぐんっ | ベンッ | 0.00 |
| ParaisoRoad 097 1000a32d | ぐっ | バク | 0.00 |
| ParaisoRoad 098 1000a332 | かっ | ガコ | 0.00 |
| ParaisoRoad 098 1000a335 | だんっ | ザコン | 0.00 |
| LoveHina_vol14 002 10006d67 | んあ~? |  | 0.00 |
| SyabondamaKieta 082 1000c339 | どき |  | 0.00 |
| SyabondamaKieta 082 1000c33a | どき |  | 0.00 |
| MukoukizuNoChonbo 011 10008f9f | ウウ | ゅゅ | 0.00 |
| LoveHina_vol14 004 10006d76 | あわっ・・ |  | 0.00 |
| LoveHina_vol14 036 10006e7f | ワイ | ア | 0.00 |
| ParaisoRoad 103 1000a34c | チン | くこ | 0.00 |
| ParaisoRoad 103 1000a34e | わー | カー | 0.00 |
| ParaisoRoad 103 1000a34f | わー | カー | 0.00 |
| ParaisoRoad 098 1000a337 | ひゅん | いそロード | 0.00 |
| SyabondamaKieta 042 1000c2b9 | カッ | ガシ | 0.00 |
| ParaisoRoad 090 1000a30d | ゲロゲロゲロ | ゾゾゾ | 0.00 |
| ParaisoRoad 091 1000a312 | ハァ | ケーっ | 0.00 |
| ParaisoRoad 086 1000a2f6+1000a2f7 | うー~ | へつー | 0.00 |
| ParaisoRoad 088 1000a2fb | ゴオオオオオオオオオ |  | 0.00 |
| ParaisoRoad 092 1000a31f | たっ | メン | 0.00 |
| ParaisoRoad 093 1000a322 | ザザザザザザ | ギャギャギャ | 0.00 |
| ParaisoRoad 095 1000a323 | ド | トカ | 0.00 |
| MukoukizuNoChonbo 047 100090b6 | ラ~ | う~う | 0.00 |
