# O3 colorize pilot — `1024_half_comic`, reader `vl16` (`output/ocr/vl16_tower_lr1e-5/best`)

20 pages, 466 crops; reader wall 31 s for 932 crops.

## (a) stroke-mask IoU, source vs colorized crop

| kind | n | IoU mean | IoU p10 | IoU d1 mean | IoU d1 p10 | d1 ≥ thr | ink src | ink col |
|---|---|---|---|---|---|---|---|---|
| sfx | 126 | 0.843 | 0.729 | 0.898 | 0.808 | 90.5 % | 0.290 | 0.290 |
| speech | 340 | 0.848 | 0.789 | 0.919 | 0.892 | 98.8 % | 0.157 | 0.149 |

(thr = 0.8; d1 = both masks dilated 1 px)

## (b) read agreement, source vs colorized

| kind | n | src exact | col exact | src sim | col sim | reads agree | agree given src exact | col exact given src exact |
|---|---|---|---|---|---|---|---|---|
| sfx | 126 | 89.7 % | 84.9 % | 0.959 | 0.957 | 88.9 % | 92.9 % | 92.9 % |
| speech | 340 | 84.7 % | 75.6 % | 0.985 | 0.973 | 84.1 % | 87.8 % | 87.8 % |

## Worst 20 crops by IoU (d1)

| book / page / id | kind | gt | src read | col read | IoU d1 |
|---|---|---|---|---|---|
| HanzaiKousyouninMinegishiEitarou 072 100041a0 | sfx | ゴ | ゴ | ゴ | 0.49 |
| HanzaiKousyouninMinegishiEitarou 072 10004192 | sfx | オ | オ | オ | 0.58 |
| LoveHina_vol01 060 10006bde | sfx | ぼてっ | ぼてっ | ぼてっ | 0.58 |
| TensiNoHaneToAkumaNoShippo 008 1000cb45 | sfx | バリ | バリ | バリ | 0.59 |
| TapkunNoTanteisitsu 007 1000c490 | sfx | くふ | くふ | くふ | 0.63 |
| TensiNoHaneToAkumaNoShippo 008 1000cb4e+1000cb4f | sfx | ドン | ドン | ドン | 0.64 |
| MisutenaideDaisy 105 10008062 | sfx | ふっ | ふっ | ふっ | 0.65 |
| Ningyoushi 077 10009bea | sfx | トン | トン | トン | 0.70 |
| LoveHina_vol01 060 10006bdb | sfx | どく | どく | どく | 0.74 |
| SamayoeruSyonenNiJunaiWo 082 00060e54 | speech | 両親亡くしてからは兄貴というより親父代わ | 両親亡くしてからは兄貴というより親父代わ | 両親亡くしてからは兄貴というより親父代わ | 0.76 |
| PrismHeart 042 1000a90c | sfx | ピンポンパンポン | ピンポンパンポン | ピンポンパンポン | 0.77 |
| HanzaiKousyouninMinegishiEitarou 072 0001ffd9 | speech | ・・・・ | ・・・・ | ・・・・・・ | 0.77 |
| AisazuNihaIrarenai 076 00002221 | speech | そ…それはちがうわ‼ | そ...それはちがうわ!! | そ...それはちがうわ!! | 0.78 |
| AisazuNihaIrarenai 076 00002220 | speech | 成島さんっ‼ | 成島さんっ!! | 成島さんっ!! | 0.78 |
| Akuhamu 021 10000793 | sfx | きゅう | きゅう | きゅう | 0.79 |
| LoveHina_vol01 060 10006be7 | sfx | パサッ | パサッ | パサッ | 0.79 |
| Hamlet 128 10003c92 | sfx | バン | バン | バン | 0.80 |
| UltraEleven 101 1000e368+1000e369 | sfx | ワーーッ | ウーーーッ | ワーーーッ | 0.81 |
| HanzaiKousyouninMinegishiEitarou 072 1000419a | sfx | オ | オ | オ | 0.81 |
| Ningyoushi 077 0004dcd2 | speech | うーん | うーん | うーん | 0.82 |

## Source right, colorized wrong

| book / page / id | kind | gt | col read | IoU d1 |
|---|---|---|---|---|
| AisazuNihaIrarenai 076 0000221b | speech | 「愛しあわずにはいられない」 | 「愛しあわずこぱいられな!」 | 0.99 |
| Akuhamu 021 000042c6 | speech | 私は語尾に
「～だゾウ」とか
つけないか | 私は語属に「〜だゾウ」とかつけないからナ | 0.93 |
| Akuhamu 021 000042c9 | speech | 私は悪魔
エルファン | 私は悪隠エルファン | 0.96 |
| Akuhamu 021 000042ca | speech | オマエがハムル様を
呼び出した人間
です | オマエがハムル様を呼び出した人間です力? | 0.93 |
| Akuhamu 021 000042de | speech | その尊敬する人
踏んでるよ | その雑敬する人踏んでるよ | 0.91 |
| Akuhamu 021 000042e2 | speech | このハンマーで
私を罰してください | このハンマーで私を剛してください | 0.93 |
| AosugiruHaru 097 10000b54 | sfx | ツカッ | リガッ | 0.93 |
| AosugiruHaru 097 10000b5d | sfx | カッ | ガ | 0.95 |
| AosugiruHaru 097 0000559e | speech | なければ関空に飛んで...... | なければ閥空に飛んで...... | 0.93 |
| AosugiruHaru 097 000055b2 | speech | お前を傷つける奴は地獄に落ちるから | お前を傷つける奴は地獄に渇ちるから | 0.91 |
| DollGun 011 000122e0 | speech | ぐ…ク…ククク俺にＹＥＳ以外の回答を求め | ぐ・・・ク・・・ククク俺にYES以外の回答を求め | 0.93 |
| Hamlet 128 10003c91 | sfx | パーン | パリン | 0.88 |
| Hamlet 128 0001f038 | speech | 敵艦隊の出現に備えて臨戦態勢をとっておけ | 敵艦隊の出現に偽えて臨戦態勢をとっておけ | 0.91 |
| Hamlet 128 0001f042 | speech | この旅は呑木氏にとって生まれて初めての宇 | この旅は香木氏にとって生まれて初めての宇宙旅行で | 0.92 |
| Hamlet 128 0001f048 | speech | 旅慣れた者でも | 旅憤れた者でも | 0.89 |
| HanzaiKousyouninMinegishiEitarou 072 10004195 | sfx | ガチン・・ | ガチン | 0.95 |
| HanzaiKousyouninMinegishiEitarou 072 10004196 | sfx | はぁ | はあ | 0.92 |
| HanzaiKousyouninMinegishiEitarou 072 0001ffd9 | speech | ・・・・ | ・・・・・・ | 0.77 |
| LancelotFullThrottle 022 0003610c | speech | 気明鳴如汝自分のテンションがすごく上がる | 気明鳴如汝自分のテンションがすごく上がるんだ符来 | 0.94 |
| LancelotFullThrottle 022 00036110 | speech | 相手を金縛りにする技だ股間のあたりがキュ | 相手を金鏡りにする抜だ股間のあたりがキュッとする | 0.95 |
| LancelotFullThrottle 022 0003611b | speech | 高速腰振りコマンド | 高速願振りコマンド | 0.95 |
| LancelotFullThrottle 022 00036121 | speech | 執事の操作しだいで吹き飛ぶ千春の社会的生 | 執事の操作しだいで吹き飛び千春の社会的生命! | 0.94 |
| LancelotFullThrottle 022 00036122 | speech | 全裸コマンド | 全様コマンド | 0.95 |
| LoveHina_vol01 060 10006be4 | sfx | リー・・ | リー | 0.95 |
| LoveHina_vol01 060 10006be6 | sfx | ガタン | ガクン | 0.99 |
| MisutenaideDaisy 105 00043974 | speech | わ…わけわかんねーあらわれかたすんじゃね | わ...わけわかんねーもらわれかたすんじゃねーよ | 0.94 |
| MisutenaideDaisy 105 00043975 | speech | スタンドバイミー | スタンドベイミー | 0.93 |
| PrismHeart 042 00056f59 | speech | 生徒の呼び出し
をします | 生徒の呼び出しなします | 0.89 |
| PrismHeart 042 00056f5e | speech | この4人は
至急～・・・ | このそんは至急~・・・ | 0.93 |
| PrismHeart 042 00056f5f | speech | 別に悪い事とは言わないけど | 別に癒い事とは言わないけど | 0.90 |

(43 such crops; 6 the other way round)

