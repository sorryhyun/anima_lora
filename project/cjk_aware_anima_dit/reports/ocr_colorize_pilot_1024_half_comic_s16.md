# O3 colorize pilot — `1024_half_comic_s16`, reader `manga_ocr`

5 pages, 115 crops; reader wall 2 s for 230 crops.

## (a) stroke-mask IoU, source vs colorized crop

| kind | n | IoU mean | IoU p10 | IoU d1 mean | IoU d1 p10 | d1 ≥ thr | ink src | ink col |
|---|---|---|---|---|---|---|---|---|
| sfx | 42 | 0.840 | 0.690 | 0.895 | 0.798 | 88.1 % | 0.378 | 0.391 |
| speech | 73 | 0.842 | 0.787 | 0.919 | 0.896 | 98.6 % | 0.129 | 0.121 |

(thr = 0.8; d1 = both masks dilated 1 px)

## (b) read agreement, source vs colorized

| kind | n | src exact | col exact | src sim | col sim | reads agree | agree given src exact | col exact given src exact |
|---|---|---|---|---|---|---|---|---|
| sfx | 42 | 35.7 % | 31.0 % | 0.563 | 0.525 | 52.4 % | 73.3 % | 73.3 % |
| speech | 73 | 53.4 % | 53.4 % | 0.978 | 0.977 | 90.4 % | 100.0 % | 100.0 % |

## Worst 20 crops by IoU (d1)

| book / page / id | kind | gt | src read | col read | IoU d1 |
|---|---|---|---|---|---|
| HanzaiKousyouninMinegishiEitarou 072 100041a0 | sfx | ゴ | ガ | ガ | 0.49 |
| HanzaiKousyouninMinegishiEitarou 072 10004192 | sfx | オ | オ | え | 0.58 |
| LoveHina_vol01 060 10006bde | sfx | ぼてっ | ぼてっ | ぼてっ | 0.58 |
| LoveHina_vol01 060 10006bdb | sfx | どく | ど | ☆ | 0.75 |
| HanzaiKousyouninMinegishiEitarou 072 0001ffd9 | speech | ・・・・ | ・・・ | ... | 0.79 |
| LoveHina_vol01 060 10006be7 | sfx | パサッ | パサッ | パサッ | 0.80 |
| LoveHina_vol01 060 10006be3 | sfx | どく | ど | どく | 0.82 |
| HanzaiKousyouninMinegishiEitarou 072 10004193 | sfx | ゴ | ゴヘ | ピペ | 0.83 |
| HanzaiKousyouninMinegishiEitarou 072 1000419a | sfx | オ | え | オ | 0.83 |
| HanzaiKousyouninMinegishiEitarou 072 1000419f | sfx | オ | オ | オ | 0.86 |
| HanzaiKousyouninMinegishiEitarou 072 10004199 | sfx | はぁ | はぁ | はぁ | 0.86 |
| LoveHina_vol01 060 00037b26 | speech | あ あれ あれれ？ | ああれあれれ? | ああれあれれ? | 0.87 |
| HanzaiKousyouninMinegishiEitarou 072 0001ffdb | speech | 見つけたぞ・・・・・ | 見つけたぞ... | 見つけたぞ... | 0.87 |
| LoveHina_vol01 060 10006bdc | sfx | ゴッ | いいのいい | いい | 0.87 |
| GinNoChimera 052 10003a1e | sfx | ザアーー・・・ | ガテー... | ガテー... | 0.88 |
| LoveHina_vol01 060 10006be9 | sfx | しゅる・・・ | じゃる... | しっろ... | 0.88 |
| HanzaiKousyouninMinegishiEitarou 072 100041a4 | sfx | ゴ | はい | パッ | 0.88 |
| LoveHina_vol01 060 00037b2b | speech | いっしょに東大行こーね | いっしょに東大行こーね | いっしょに東大行こーね | 0.88 |
| SonokiDeABC 052 00066a90 | speech | あきらめがつくのに...... | あきらめがつくのに... | あきらめがつくのに... | 0.89 |
| HanzaiKousyouninMinegishiEitarou 072 10004198 | sfx | オ | ・・・ | ここ | 0.89 |

## Source right, colorized wrong

| book / page / id | kind | gt | col read | IoU d1 |
|---|---|---|---|---|
| HanzaiKousyouninMinegishiEitarou 072 10004192 | sfx | オ | え | 0.58 |
| HanzaiKousyouninMinegishiEitarou 072 10004194 | sfx | ゴ | ゴロ | 0.98 |
| LoveHina_vol01 060 10006be5 | sfx | とっ・・ | とっ | 0.99 |
| LoveHina_vol01 060 10006be8 | sfx | ぷる | ぷる。 | 0.94 |

(4 such crops; 2 the other way round)

