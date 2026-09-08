# OCR eval — `hunyuan` on Manga109-s `test` (official COO split ∩ Manga109-s)

Reader wall 288 s for 5117 crops (17.8 crops/s).

| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |
|---|---|---|---|---|---|---|
| sfx | 2558 | 242 | 9.5 | 0.251 | 14.3 % | 18 |
| speech | 2559 | 966 | 37.7 | 0.814 | 70.2 % | 13 |

## SFX by orientation

| orient | n | exact % | sim |
|---|---|---|---|
| horizontal | 624 | 9.5 | 0.318 |
| square | 486 | 5.8 | 0.131 |
| vertical | 1448 | 10.7 | 0.262 |

## SFX by length

| len | n | exact % | sim |
|---|---|---|---|
| 1 | 88 | 17.0 | 0.170 |
| 2 | 1017 | 11.2 | 0.197 |
| 3 | 730 | 8.8 | 0.248 |
| 4 | 315 | 7.0 | 0.309 |
| 5 | 167 | 5.4 | 0.312 |
| 6 | 135 | 8.9 | 0.370 |
| 7 | 48 | 6.2 | 0.349 |
| 8+ | 58 | 5.2 | 0.510 |

## Worst 25 SFX (by sim)

| book / page / id | gt | pred | sim |
|---|---|---|---|
| ParaisoRoad 010 1000a15a | パンッ | ノロノロ | 0.00 |
| ParaisoRoad 016 1000a183 | じ~ | 二 | 0.00 |
| ParaisoRoad 016 1000a184 | ゲッ | 妙 | 0.00 |
| ParaisoRoad 013 1000a166 | ゔー | ぴー | 0.00 |
| ParaisoRoad 013 1000a167 | ちょっ | x16, | 0.00 |
| ParaisoRoad 013 1000a168 | ズン | せんで | 0.00 |
| ParaisoRoad 014 1000a16b | ドッ | 呸 | 0.00 |
| ParaisoRoad 014 1000a16c | あぁぁ | Otoha | 0.00 |
| ParaisoRoad 014 1000a16d | おああ | 16Aph | 0.00 |
| ParaisoRoad 014 1000a16e | ゴッ | 砲 | 0.00 |
| ParaisoRoad 014 1000a16f | すっ | □父乙□ | 0.00 |
| ParaisoRoad 014 1000a171 | ぎっ | Xu | 0.00 |
| ParaisoRoad 014 1000a172 | むっ | □戈 | 0.00 |
| ParaisoRoad 015 1000a173 | あ | 女 | 0.00 |
| ParaisoRoad 015 1000a174 | あ | d | 0.00 |
| ParaisoRoad 016 1000a182 | ガッ | 77/1''4 | 0.00 |
| ParaisoRoad 011 1000a15c | ひゅん | 乙空 | 0.00 |
| ParaisoRoad 011 1000a15e | ぷっ | 亞鳥 | 0.00 |
| ParaisoRoad 012 1000a160 | あっ | 女， | 0.00 |
| ParaisoRoad 012 1000a161 | あっ | も | 0.00 |
| ParaisoRoad 013 1000a163 | おー | ざー | 0.00 |
| ParaisoRoad 013 1000a164 | きっ | 卅七 | 0.00 |
| ParaisoRoad 013 1000a165 | ちゃい | 志 | 0.00 |
| ParaisoRoad 008 1000a145 | ギリギリ | 七亖分□吉□ | 0.00 |
| ParaisoRoad 008 1000a146 | ギリ | □三□11 | 0.00 |
