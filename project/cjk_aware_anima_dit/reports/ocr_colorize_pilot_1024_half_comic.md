# O3 colorize pilot — `1024_half_comic`, reader `manga_ocr`

20 pages, 466 crops; reader wall 5 s for 932 crops.

## (a) stroke-mask IoU, source vs colorized crop

| kind | n | IoU mean | IoU p10 | IoU d1 mean | IoU d1 p10 | d1 ≥ thr | ink src | ink col |
|---|---|---|---|---|---|---|---|---|
| sfx | 126 | 0.843 | 0.729 | 0.898 | 0.808 | 90.5 % | 0.290 | 0.290 |
| speech | 340 | 0.848 | 0.789 | 0.919 | 0.892 | 98.8 % | 0.157 | 0.149 |

(thr = 0.8; d1 = both masks dilated 1 px)

## (b) read agreement, source vs colorized

| kind | n | src exact | col exact | src sim | col sim | reads agree | agree given src exact | col exact given src exact |
|---|---|---|---|---|---|---|---|---|
| sfx | 126 | 34.9 % | 31.0 % | 0.540 | 0.500 | 52.4 % | 77.3 % | 77.3 % |
| speech | 340 | 67.1 % | 60.9 % | 0.950 | 0.939 | 77.9 % | 89.9 % | 89.9 % |

## Worst 20 crops by IoU (d1)

| book / page / id | kind | gt | src read | col read | IoU d1 |
|---|---|---|---|---|---|
| HanzaiKousyouninMinegishiEitarou 072 100041a0 | sfx | ゴ | ガ | ガ | 0.49 |
| HanzaiKousyouninMinegishiEitarou 072 10004192 | sfx | オ | オ | え | 0.58 |
| LoveHina_vol01 060 10006bde | sfx | ぼてっ | ぼてっ | ぼてっ | 0.58 |
| TensiNoHaneToAkumaNoShippo 008 1000cb45 | sfx | バリ | んーーー | いや、 | 0.59 |
| TapkunNoTanteisitsu 007 1000c490 | sfx | くふ | くふ | くふ | 0.63 |
| TensiNoHaneToAkumaNoShippo 008 1000cb4e+1000cb4f | sfx | ドン | うん、ん | それは | 0.64 |
| MisutenaideDaisy 105 10008062 | sfx | ふっ | ふっ | ぶっ | 0.65 |
| Ningyoushi 077 10009bea | sfx | トン | トン | トン | 0.70 |
| LoveHina_vol01 060 10006bdb | sfx | どく | ど | ぐ | 0.74 |
| SamayoeruSyonenNiJunaiWo 082 00060e54 | speech | 両親亡くしてからは兄貴というより親父代わ | あああ | ハハ・・・両親亡くしてからは兄貴というよ | 0.76 |
| PrismHeart 042 1000a90c | sfx | ピンポンパンポン | ピンポンパンポール | ピンポンパンポート | 0.77 |
| HanzaiKousyouninMinegishiEitarou 072 0001ffd9 | speech | ・・・・ | ・・・ | ... | 0.77 |
| AisazuNihaIrarenai 076 00002221 | speech | そ…それはちがうわ‼ | そ...それはちがうわ!! | そ...それはちがうわ!! | 0.78 |
| AisazuNihaIrarenai 076 00002220 | speech | 成島さんっ‼ | 成島さんっ!! | 成島さんっ!! | 0.78 |
| Akuhamu 021 10000793 | sfx | きゅう | きゃっ | きゅっ | 0.79 |
| LoveHina_vol01 060 10006be7 | sfx | パサッ | パサッ | パサッ | 0.79 |
| Hamlet 128 10003c92 | sfx | バン | バン | バッ | 0.80 |
| UltraEleven 101 1000e368+1000e369 | sfx | ワーーッ | はっかりしたのは、 | 「はっ、はぁ、はっはぁ... | 0.81 |
| HanzaiKousyouninMinegishiEitarou 072 1000419a | sfx | オ | え | オ | 0.81 |
| Ningyoushi 077 0004dcd2 | speech | うーん | うーん | うーん | 0.82 |

## Source right, colorized wrong

| book / page / id | kind | gt | col read | IoU d1 |
|---|---|---|---|---|
| AisazuNihaIrarenai 076 10000319 | sfx | がばっ!! | がぼっ!! | 0.90 |
| AisazuNihaIrarenai 076 0000221b | speech | 「愛しあわずにはいられない」 | 「愛知県の方がいられない」 | 0.99 |
| Akuhamu 021 10000790 | sfx | ビビ | いい | 0.98 |
| Akuhamu 021 000042c5 | speech | それにこの世界も
意外と楽しいもの
なの | それにこの世界も是外と楽しいものなのだ | 0.94 |
| Akuhamu 021 000042c6 | speech | 私は語尾に
「～だゾウ」とか
つけないか | 私は語尾に「~だソウ」とかつけないからナ | 0.93 |
| Akuhamu 021 000042de | speech | その尊敬する人
踏んでるよ | その嫌敬する人踏んでるよ | 0.91 |
| AosugiruHaru 097 10000b54 | sfx | ツカッ | ツチッ | 0.93 |
| AosugiruHaru 097 000055b0 | speech | なおこの女性はタレントの青木永遠さん―― | なおこの女性はタレントの青木永遠さん― | 0.93 |
| DollGun 011 000122db | speech | 貴様 | 害様 | 0.92 |
| Hamlet 128 10003c92 | sfx | バン | バッ | 0.80 |
| Hamlet 128 0001f032 | speech | 我々はこれから月と地球の間にある中継ステ | 我々はこれから月と想球の間にある中継ステーション | 0.89 |
| Hamlet 128 0001f038 | speech | 敵艦隊の出現に備えて臨戦態勢をとっておけ | 戯艦隊の出現に備えて臨戦態勢をとっておけ | 0.91 |
| Hamlet 128 0001f042 | speech | この旅は呑木氏にとって生まれて初めての宇 | この旅は香木氏にとって生まれて初めての宇宙旅行で | 0.92 |
| HanzaiKousyouninMinegishiEitarou 072 10004192 | sfx | オ | え | 0.58 |
| HanzaiKousyouninMinegishiEitarou 072 10004194 | sfx | ゴ | ゴロ | 0.97 |
| LancelotFullThrottle 022 0003610c | speech | 気明鳴如汝自分のテンションがすごく上がる | 気明鳴如汝自分のデンションがすごく上がるんだ将来 | 0.94 |
| LancelotFullThrottle 022 00036110 | speech | 相手を金縛りにする技だ股間のあたりがキュ | 相手を金縛りにする技だ船間のあたりがキュッとする | 0.95 |
| LancelotFullThrottle 022 0003611b | speech | 高速腰振りコマンド | 高速標振りコマンド | 0.95 |
| LoveHina_vol01 060 10006be5 | sfx | とっ・・ | とっ | 0.99 |
| LoveHina_vol01 060 10006be8 | sfx | ぷる | ぷる。 | 0.93 |
| MisutenaideDaisy 105 10008062 | sfx | ふっ | ぶっ | 0.65 |
| MisutenaideDaisy 105 00043974 | speech | わ…わけわかんねーあらわれかたすんじゃね | わ...わけわかんねーおらわれかたすんじゃねーよ | 0.94 |
| MisutenaideDaisy 105 0004397e | speech | どおおっせオレにはともだちがいねーやいっ | どわおっせオレにはともだちがいねーやいっ | 0.95 |
| Ningyoushi 077 10009be9 | sfx | トン | トッ | 0.87 |
| PrayerHaNemurenai 025 00055c37 | speech | ほんっとにハンパなんだから！！ | ほんっとにハンバなんだから!! | 0.92 |
| PrismHeart 042 00056f57 | speech | あなた達　昨日
コンサート会場近くで
楽 | あなた達昨日コンサート会場近くで楽器清奏してたそ | 0.93 |
| PrismHeart 042 00056f5a | speech | 3-A
早瀬秋生
長谷部哲也 | 31人早瀬秋生長谷部哲也 | 0.89 |
| PrismHeart 042 00056f70 | speech | うん　そう
それそれ | うんそうそれまあ | 0.89 |
| SamayoeruSyonenNiJunaiWo 082 00060e4e | speech | だから...その　お兄さんが贈賄で捕まっ | だから...そのお兄さんが購購で捕まっちゃったん | 0.91 |
| SamayoeruSyonenNiJunaiWo 082 00060e51 | speech | さっきの「刑務所」で思い出したけど... | さっきの「刑務所」で思い出したけど...久須見一 | 0.92 |

(33 such crops; 7 the other way round)

