# OCR eval — `vl16_sfxp_1k_cap4_sfx` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 54 s for 2558 crops (47.0 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 1722 | 67.3 | 0.837 | 73.9 % | 19 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 652 | 55.4 | 0.790 |
| square | 417 | 70.3 | 0.830 |
| vertical | 1489 | 71.7 | 0.859 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 60.2 | 0.636 |
| 2 | 1017 | 74.5 | 0.848 |
| 3 | 730 | 67.0 | 0.852 |
| 4 | 315 | 62.2 | 0.839 |
| 5 | 167 | 52.1 | 0.812 |
| 6 | 135 | 69.6 | 0.870 |
| 7 | 48 | 52.1 | 0.802 |
| 8+ | 58 | 34.5 | 0.763 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| LoveHina_vol14 002 10006d67 | んあ~? |  | 0.00 |
| MAD_STONE 085 100070a5+100070a6 | フ゛ン | シャッ | 0.00 |
| MAD_STONE 084 100070a4 | グイ | ズズ | 0.00 |
| SyabondamaKieta 042 1000c2b9 | カッ | ギシ | 0.00 |
| SaladDays_vol18 086 1000b835 | ァ | ア | 0.00 |
| SaladDays_vol18 086 1000b831 | プァ | ワア | 0.00 |
| SaladDays_vol18 086 1000b838 | ァ | ア… | 0.00 |
| SaladDays_vol18 085 1000b821 | プア | ペペ | 0.00 |
| ParaisoRoad 035 1000a1fe | ん |  | 0.00 |
| SaladDays_vol18 085 1000b822 | リリ | シシ | 0.00 |
| LoveHina_vol14 064 10006f54 | みゅー | チャー | 0.00 |
| ParaisoRoad 034 1000a1fc | キ | メ | 0.00 |
| ParaisoRoad 035 1000a201 | す~ | む | 0.00 |
| ParaisoRoad 092 1000a31f | たっ | メン | 0.00 |
| ParaisoRoad 092 1000a31e | ザザザザザ | ギギギギ | 0.00 |
| ParaisoRoad 086 1000a2f6+1000a2f7 | うー~ | くつー | 0.00 |
| ParaisoRoad 037 1000a212 | じ~ん | ぎっ | 0.00 |
| SaladDays_vol18 085 1000b824 | イ | 人 | 0.00 |
| MAD_STONE 060 1000706b | ドン | ズ | 0.00 |
| ParaisoRoad 089 1000a304 | ズザザザザザ | ギギギギ | 0.00 |
| ParaisoRoad 090 1000a30d | ゲロゲロゲロ | ぱぱぱ | 0.00 |
| LoveHina_vol14 003 10006d70 | んにゃー | ヒヒー | 0.00 |
| LoveHina_vol14 004 10006d76 | あわっ・・ |  | 0.00 |
| LoveHina_vol14 004 10006d73 | えう |  | 0.00 |
| SaladDays_vol18 084 1000b80f | リ | ソ | 0.00 |
