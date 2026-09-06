# O3 colorize pilot — tier 1024, reader `manga_ocr`

20 pages, 466 crops; reader wall 5 s for 932 crops.

## (a) stroke-mask IoU, source vs colorized crop

| kind | n | IoU mean | IoU p10 | IoU d1 mean | IoU d1 p10 | d1 ≥ thr | ink src | ink col |
|---|---|---|---|---|---|---|---|---|
| sfx | 126 | 0.794 | 0.653 | 0.868 | 0.778 | 87.3 % | 0.290 | 0.296 |
| speech | 340 | 0.785 | 0.691 | 0.886 | 0.853 | 98.8 % | 0.157 | 0.157 |

(thr = 0.8; d1 = both masks dilated 1 px)

## (b) read agreement, source vs colorized

| kind | n | src exact | col exact | src sim | col sim | reads agree | agree given src exact | col exact given src exact |
|---|---|---|---|---|---|---|---|---|
| sfx | 126 | 34.9 % | 31.7 % | 0.540 | 0.490 | 47.6 % | 79.5 % | 79.5 % |
| speech | 340 | 67.1 % | 41.8 % | 0.950 | 0.870 | 55.9 % | 62.3 % | 62.3 % |

## Worst 20 crops by IoU (d1)

| book / page / id | kind | gt | src read | col read | IoU d1 |
|---|---|---|---|---|---|
| TensiNoHaneToAkumaNoShippo 008 1000cb4d | sfx | ゴー | お | お | 0.20 |
| HanzaiKousyouninMinegishiEitarou 072 100041a0 | sfx | ゴ | ガ | ゴ | 0.49 |
| TensiNoHaneToAkumaNoShippo 008 1000cb4c | sfx | ゴー | 一 | __ | 0.54 |
| LoveHina_vol01 060 10006bde | sfx | ぼてっ | ぼてっ | ぼてっ | 0.57 |
| TapkunNoTanteisitsu 007 1000c490 | sfx | くふ | くふ | くふ | 0.58 |
| MisutenaideDaisy 105 10008062 | sfx | ふっ | ふっ | ふっ | 0.62 |
| Hamlet 128 10003c91 | sfx | パーン | パーン | パーン | 0.71 |
| TapkunNoTanteisitsu 007 1000c48d | sfx | わく | わく | わく | 0.72 |
| LoveHina_vol01 060 10006bdb | sfx | どく | ど | ピャ | 0.72 |
| LancelotFullThrottle 022 00036121 | speech | 執事の操作しだいで吹き飛ぶ千春の社会的生 | この事を探したいときは第4巻を支えて | この時の状況にしたいでは京都千歩み会のま | 0.76 |
| PrayerHaNemurenai 025 1000a6a7 | sfx | ゴッ | ゴッ | ゴッ | 0.76 |
| TapkunNoTanteisitsu 007 1000c48e | sfx | くふ | くふ | くふ | 0.76 |
| HanzaiKousyouninMinegishiEitarou 072 1000419a | sfx | オ | え | オ | 0.77 |
| SamayoeruSyonenNiJunaiWo 082 00060e54 | speech | 両親亡くしてからは兄貴というより親父代わ | あああ | うん...部屋亡くしてからは兄貴というよ | 0.77 |
| Akuhamu 021 10000793 | sfx | きゅう | きゃっ | きゃぁ | 0.77 |
| TapkunNoTanteisitsu 007 1000c489 | sfx | ほいっ | ぼいっ | はいっ | 0.79 |
| AisazuNihaIrarenai 076 10000319 | sfx | がばっ!! | がばっ!! | がばっ!! | 0.79 |
| AosugiruHaru 097 000055b2 | speech | お前を傷つける奴は地獄に落ちるから | お前を傷つける奴は地獄に落ちるから | お前を構っける訳は絶賛にもらるなら | 0.80 |
| HanzaiKousyouninMinegishiEitarou 072 10004193 | sfx | ゴ | ゴヘ | ゴへ | 0.80 |
| AisazuNihaIrarenai 076 00002220 | speech | 成島さんっ‼ | 成島さんっ!! | 成島さんっ!! | 0.80 |

## Source right, colorized wrong

| book / page / id | kind | gt | col read | IoU d1 |
|---|---|---|---|---|
| AisazuNihaIrarenai 076 10000317 | sfx | ぱっ | ばっ | 0.89 |
| AisazuNihaIrarenai 076 0000221b | speech | 「愛しあわずにはいられない」 | 「愛液の方にはいられない」 | 0.99 |
| AisazuNihaIrarenai 076 00002222 | speech | 歩ちゃん待っててくれるっていったじゃない | 歩ちゃん待っててくれるっていったじゃないか!!そ | 0.86 |
| Akuhamu 021 10000790 | sfx | ビビ | ビど | 0.97 |
| Akuhamu 021 000042c5 | speech | それにこの世界も
意外と楽しいもの
なの | それにこの世界も置外と差しいものなのだ | 0.89 |
| Akuhamu 021 000042c6 | speech | 私は語尾に
「～だゾウ」とか
つけないか | 私は悪魔にでくたソウことかつけないからナ | 0.89 |
| Akuhamu 021 000042c7 | speech | ばかを
言うな | ぱかを言うな | 0.90 |
| Akuhamu 021 000042c9 | speech | 私は悪魔
エルファン | 私は義親エルファン | 0.90 |
| Akuhamu 021 000042cb | speech | ハムル様
会いたかった
デス | ハムル様会いたかったテス | 0.92 |
| Akuhamu 021 000042cc | speech | 奥さんも「許すから
帰ってこい」って
言 | 微さんも「殺すから揃ってこい」って言ってましたヨ | 0.88 |
| Akuhamu 021 000042cf | speech | 暴れないでぇ | っ~暴れないでぇ | 0.88 |
| Akuhamu 021 000042d0 | speech | ひとつだけ
言っておきまス | ひとつだけ言っておきまえ | 0.89 |
| Akuhamu 021 000042d1 | speech | 我は召喚された
悪魔として
使命は果たさ | 我は日限された聖書として健霊は里たさなばならん | 0.90 |
| Akuhamu 021 000042d7 | speech | その比類なき魔力は
他に並ぶものなしデス | その比較なき魅力は他に追いものなしテス | 0.89 |
| Akuhamu 021 000042d8 | speech | あぁ　ハムル様
スイマセン～ | あぁハムル櫻スイマセン~ | 0.92 |
| Akuhamu 021 000042d9 | speech | 私の最も尊敬
する方デス | 私の鍵も鬱殺する方デス | 0.88 |
| Akuhamu 021 000042de | speech | その尊敬する人
踏んでるよ | その警巻する人護んでるよ | 0.86 |
| AosugiruHaru 097 10000b54 | sfx | ツカッ | !! | 0.87 |
| AosugiruHaru 097 10000b56 | sfx | カッ | ん | 0.90 |
| AosugiruHaru 097 000055a1 | speech | はあ | にあ | 0.87 |
| AosugiruHaru 097 000055aa | speech | 自由通りを雪谷方面へ | 日出通りを懸念方国へ | 0.88 |
| AosugiruHaru 097 000055ad | speech | 徒歩で南下中 | 後歩で勝下中 | 0.89 |
| AosugiruHaru 097 000055af | speech | 女性は白のワンピースを着用 | 次はは自のワンピースを鷹用 | 0.86 |
| AosugiruHaru 097 000055b0 | speech | なおこの女性はタレントの青木永遠さん―― | なおこの女性はタレントの贅木沢遽さん―― | 0.88 |
| AosugiruHaru 097 000055b2 | speech | お前を傷つける奴は地獄に落ちるから | お前を構っける訳は絶賛にもらるなら | 0.80 |
| DollGun 011 000122db | speech | 貴様 | 吉様 | 0.90 |
| GinNoChimera 052 0001c134 | speech | えっ？図書館だけど | えっ?回警察だけど | 0.88 |
| GinNoChimera 052 0001c135 | speech | 君　どっち行くんだ？ | 愛どっち行くんだ? | 0.88 |
| GinNoChimera 052 0001c136 | speech | 一見ごく普通の高校生だが | 一見ごく警速の高校生だが | 0.88 |
| GinNoChimera 052 0001c137 | speech | 実は父親が異星人というぶっとんだ出生の秘 | 実は父親が栗湿人という不っとんだ出生の秘密がある | 0.88 |

(95 such crops; 5 the other way round)

