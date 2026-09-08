# OCR eval — `hunyuan_ja_prompt` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 321 s for 5117 crops (16.0 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 333 | 13.0 | 0.320 | 18.8 % | 35 |
| speech | 2559 | 1107 | 43.3 | 0.857 | 74.8 % | 17 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 10.4 | 0.350 |
| square | 486 | 10.3 | 0.209 |
| vertical | 1448 | 15.1 | 0.344 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 23.9 | 0.246 |
| 2 | 1017 | 17.8 | 0.302 |
| 3 | 730 | 10.3 | 0.290 |
| 4 | 315 | 7.3 | 0.337 |
| 5 | 167 | 5.4 | 0.337 |
| 6 | 135 | 11.9 | 0.450 |
| 7 | 48 | 8.3 | 0.489 |
| 8+ | 58 | 6.9 | 0.524 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| ParaisoRoad 099 1000a33c | ゴゴッ | ユニコン | 0.00 |
| ParaisoRoad 041 1000a230 | ドン | トイレ | 0.00 |
| ParaisoRoad 042 1000a231 | ヒソ | はし | 0.00 |
| ParaisoRoad 066 1000a29f | ダンッ | あっ | 0.00 |
| ParaisoRoad 066 1000a2a0 | ゴォッ | すみません | 0.00 |
| MukoukizuNoChonbo 024 10008ffe | ぎくっ | みん | 0.00 |
| MukoukizuNoChonbo 032 10009021 | パチッ | バイ三 | 0.00 |
| MukoukizuNoChonbo 032 10009023 | ガチャーン | 炎！ | 0.00 |
| ParaisoRoad 098 1000a337 | ひゅん | いそロード | 0.00 |
| ParaisoRoad 099 1000a338 | ボウッ | 木羊 | 0.00 |
| ParaisoRoad 099 1000a33a | バカッ | いい | 0.00 |
| ParaisoRoad 099 1000a33b | べしっ | あ | 0.00 |
| ParaisoRoad 041 1000a22c | ドン | これは、ある種類の植物の葉であることを示しています。 | 0.00 |
| ParaisoRoad 099 1000a33d | ゴゴゴゴ | フッフッコ | 0.00 |
| ParaisoRoad 100 1000a33e | カッ | 力 | 0.00 |
| ParaisoRoad 100 1000a33f | ズババババ | でさが田 か | 0.00 |
| ParaisoRoad 100 1000a340 | すっ | あ | 0.00 |
| ParaisoRoad 093 1000a320 | ザッ | 兄老い | 0.00 |
| ParaisoRoad 093 1000a322 | ザザザザザザ | 三矢 拜 拜 | 0.00 |
| ParaisoRoad 095 1000a323 | ド | 人 | 0.00 |
| ParaisoRoad 095 1000a324 | ほっ | は、 | 0.00 |
| ParaisoRoad 103 1000a351 | パタ | IPA | 0.00 |
| ParaisoRoad 097 1000a330+1000a331 | バンッ | なんの ダチ 塁舞 | 0.00 |
| MukoukizuNoChonbo 038 10009051 | パタ | ロンドン | 0.00 |
| ParaisoRoad 066 1000a2a2 | どっ | とん | 0.00 |
