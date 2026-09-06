# OCR eval — `vl16_lr1e-4` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 152 s for 5117 crops (33.7 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1656 | 64.7 | 0.816 | 72.8 % | 24 |
| speech | 2559 | 2041 | 79.8 | 0.981 | 97.2 % | 170 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 48.4 | 0.759 |
| square | 486 | 66.9 | 0.783 |
| vertical | 1448 | 71.1 | 0.852 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 63.6 | 0.652 |
| 2 | 1017 | 76.6 | 0.842 |
| 3 | 730 | 64.4 | 0.819 |
| 4 | 315 | 55.2 | 0.786 |
| 5 | 167 | 44.3 | 0.795 |
| 6 | 135 | 50.4 | 0.832 |
| 7 | 48 | 43.8 | 0.791 |
| 8+ | 58 | 24.1 | 0.793 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| SyabondamaKieta 092 1000c357 | ばっ | ぶつ | 0.00 |
| SyabondamaKieta 091 1000c353 | バサ | シュ | 0.00 |
| LoveHina_vol14 004 10006d73 | えう | ざぅ | 0.00 |
| LoveHina_vol14 004 10006d72 | ワー | ス | 0.00 |
| MAD_STONE 087 100070af+100070b0 | ゴワッ | ドン! | 0.00 |
| LoveHina_vol14 003 10006d70 | んにゃー | ドキー | 0.00 |
| MukoukizuNoChonbo 011 10008f9f | ウウ | ひら | 0.00 |
| SyabondamaKieta 044 1000c2cf | びく | ヒュッ | 0.00 |
| SyabondamaKieta 043 1000c2c8 | クスッ | カチャ | 0.00 |
| SyabondamaKieta 043 1000c2c0 | クス | シュウ | 0.00 |
| SyabondamaKieta 017 1000c28c | パン | バリ | 0.00 |
| ParaisoRoad 033 1000a1f8 | ぁぁ | ああ | 0.00 |
| ParaisoRoad 033 1000a1ef | おう | なっ | 0.00 |
| LoveHina_vol14 078 10006f77 | ず・・・ | ポ・・・・ | 0.00 |
| ParaisoRoad 036 1000a20d | ドゴ | ビク | 0.00 |
| ParaisoRoad 036 1000a208 | わ | ハ | 0.00 |
| SaladDays_vol18 086 1000b831 | プァ | ブン | 0.00 |
| SyabondamaKieta 016 1000c285 | ふわっ | ひゅん | 0.00 |
| SaladDays_vol18 089 1000b84d | プァ | ペラ | 0.00 |
| SaladDays_vol18 089 1000b850+1000b851 | アァン | ふっ | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア | 0.00 |
| ParaisoRoad 033 1000a1f1 | あ | ぱち | 0.00 |
| ParaisoRoad 034 1000a1f9 | イイイイイイ | バン | 0.00 |
| ParaisoRoad 034 1000a1fc | キ | オ | 0.00 |
| SaladDays_vol18 087 1000b83e | ゴオオ | ボォォ | 0.00 |
