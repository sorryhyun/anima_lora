# OCR eval — `sfxp_1117_1816_ep1_sfx` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 68 s for 2558 crops (37.8 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1691 | 66.1 | 0.836 | 74.2 % | 44 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 631 | 58.6 | 0.813 |
| square | 490 | 67.1 | 0.838 |
| vertical | 1437 | 69.0 | 0.845 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 54.5 | 0.655 |
| 2 | 1017 | 70.3 | 0.845 |
| 3 | 730 | 68.8 | 0.852 |
| 4 | 315 | 64.1 | 0.854 |
| 5 | 167 | 57.5 | 0.832 |
| 6 | 135 | 63.0 | 0.837 |
| 7 | 48 | 54.2 | 0.826 |
| 8+ | 58 | 29.3 | 0.663 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| LoveHina_vol14 004 10006d76 | あわっ・・ |  | 0.00 |
| LoveHina_vol14 004 10006d73 | えう |  | 0.00 |
| SyabondamaKieta 069 1000c311 | カラン・・・ |  | 0.00 |
| MukoukizuNoChonbo 005 10008f6b | トン |  | 0.00 |
| SyabondamaKieta 062 1000c2fa | ザク | ユユユ | 0.00 |
| MukoukizuNoChonbo 005 10008f67 | トン |  | 0.00 |
| MukoukizuNoChonbo 005 10008f61 | わい |  | 0.00 |
| SyabondamaKieta 061 1000c2f7 | コツン・・・ |  | 0.00 |
| MukoukizuNoChonbo 011 10008f9f | ウウ |  | 0.00 |
| SyabondamaKieta 050 1000c2e0 | ザアアアアアー・ | きこ
きこ | 0.00 |
| SyabondamaKieta 055 1000c2e7 | ァァ | ア
ア
ア
ア
ア | 0.00 |
| LoveHina_vol14 009 10006d9c | くす |  | 0.00 |
| SyabondamaKieta 044 1000c2cf | びく | ミニマッ | 0.00 |
| SyabondamaKieta 044 1000c2d1 | !! | ハハ | 0.00 |
| SyabondamaKieta 025 1000c294 | ことん |  | 0.00 |
| LoveHina_vol14 012 10006da8 | あら |  | 0.00 |
| LoveHina_vol14 012 10006dac | あわわ・・・ |  | 0.00 |
| ParaisoRoad 055 1000a261 | う | ワ… | 0.00 |
| ParaisoRoad 056 1000a263+1000a264 | どごっ | ゴゴゴ | 0.00 |
| ParaisoRoad 043 1000a239 | だっ | ゴルッ | 0.00 |
| ParaisoRoad 036 1000a20d | ドゴ | ビブ | 0.00 |
| ParaisoRoad 052 1000a25a | バリ | ぶぅ | 0.00 |
| ParaisoRoad 052 1000a25b | バリ | ぶり | 0.00 |
| ParaisoRoad 035 1000a1fe | ん | ハスッ | 0.00 |
| ParaisoRoad 062 1000a28e | ドゴォォオ | フフフフフフフフフフフフフフフフフフフフフフフフフフフフフフフフフフフフフフフフ | 0.00 |
