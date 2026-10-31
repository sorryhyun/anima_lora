# MOSS-Image LoRA ガイドブック

GUI で MOSS-Image 用の LoRA を学習・テストするためのガイドです。`make gui` で開き、上部バーの **モデル** ドロップダウンを **MOSS-Image** に切り替えます（`make gui-moss` は同じパネルを別ウィンドウで開きます）。パネルは英語で表示され、GUI の言語が中国語のときだけ中国語で表示されます。

MOSS ラインは **アルファ** 段階です。スタイルのセットはこれまで 1 つ（1 人の絵師の画像 30 枚）だけですが、それで多くの A/B を行っており、フラグやデフォルト値は今後変わる可能性があります。以下の数値は、その A/B と 16 GB の RTX 5070 Ti でのベンチマークによるものです。

---

## 目次

1. [MOSS-Image とは](#1-moss-image-とは)
2. [セットアップ](#2-セットアップ)
3. [前処理・学習・テスト](#3-前処理学習テスト)
4. [推奨設定](#4-推奨設定)
5. [テスト](#5-テスト)
6. [トラブルシューティング](#6-トラブルシューティング)

- [付録. CLI](#付録-cli)

---

## 1. MOSS-Image とは

[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)（OpenMOSS、Apache-2.0）は Anima とは別のモデルで、このリポジトリでは独立したラインとして学習します。

| | Anima | MOSS-Image |
|---|---|---|
| 拡散モデル | Anima DiT | 5B single-stream DiT（42 ブロック） |
| テキストエンコーダー | Qwen3 0.6B | Qwen3.5-9B |
| VAE | QwenImage VAE | FLUX.2 VAE |
| キャッシュ | `post_image_dataset/lora/` | `post_image_dataset/moss/` |
| LoRA | `output/ckpt/` | `output/ckpt_moss/` |
| 設定 | variant TOML + Hardware プリセット | パネルの入力欄 |

- キャッシュ、設定、LoRA は互いに共有できません。MOSS の LoRA は Anima では読み込めず、Anima の LoRA も MOSS では読み込めません。Anima のバリアント、プリセット、`make lora` もここでは使いません。
- MOSS には DiT が 2 つあります。**Base** は事前学習モデル、**SFT** はそれをファインチューニングしたもので、光沢の強いソシャゲ風の絵になります。学習は Base で行ってください。Base で学習した LoRA は SFT でも動作し、SFT の光沢をほぼ取り除いて Base + LoRA に近い結果になります。

---

## 2. セットアップ

> まず共通の[インストールガイド](../installation/インストール.md)でリポジトリをインストールします。

| パック | 内容 | サイズ |
|---|---|---|
| `moss-base` | DiT、テキストエンコーダー、VAE、トークナイザー | 約 27 GB |
| `moss-sft`（任意） | SFT の DiT のみ。テキストエンコーダーと VAE は Base のものを使用 | 約 10 GB |

GUI の「モデルのダウンロード」画面から取得するか、次のコマンドを実行します。

```bash
make download-model ARGS=moss-base
make download-model ARGS=moss-sft
```

重みは `models/moss_image/MOSS-Image-Base/` と `models/moss_image/MOSS-Image-SFT/` に保存されます。別の場所に置く場合は `.env` に `ANIMA_MOSS_MODEL_DIR` を設定するか、パネルの **Advanced** で `model_dir`（2 つの `MOSS-Image-*` フォルダを含むフォルダ）を指定します。フォルダが揃うまで、パネルには `Model not found: …` と表示されます。

デフォルトのキャプション形式（`layout_json`、§3.1）には danbooru タグ CSV も必要です。`make download-models` で取得済みで、単独では `make download-model ARGS=danbooru_tags` で取得できます。

---

## 3. 前処理・学習・テスト

パネルには **Preprocess** と **Train** の 2 つのタブがあり、**Test** ボックスは Train タブの下部にあります。あまり使わない項目は **Advanced** の中に折りたたまれています。各ボタンは学習デーモンにジョブを投入します。モデルのドロップダウンを Anima に戻しても GUI を閉じてもジョブは動き続け、再度開くと実行中のジョブに再接続し（`Re-attached to running job …`）、**Stop** でキャンセルします。

### 3.1 データセットとキャプション

同名の `.txt` キャプションが付いた画像のフォルダで、サブフォルダも含まれます。デフォルトのソースは Anima のリサイズ済みフォルダ `post_image_dataset/resized`（修正済みキャプション付き）なので、Anima 用に前処理したデータセットをそのまま使えます。1 つのセットだけを学習するには、`src` をそのサブフォルダに向けます。

- **Anima 形式のタグキャプションで学習もプロンプトも動作します。** MOSS 自体は自然言語で学習されており、長いキャプションやキャラクターの多いキャプションは未検証です。
- **`layout_json`（デフォルト）**: Anima のキャプションを MOSS の layout-JSON プロンプトに変換します。rating/meta、キャラクター、作品、絵師はそれぞれ専用のキーに入り、残りのタグは caption に、位置節は位置情報付きの subject に、OCR のテキスト行は text 項目になります。30 枚のセットでは、通常のタグと同じスタイル強度を保ちつつ、セットの属性が他のキャラクターに漏れることが少なくなりました。
- **学習とプロンプトは同じ形式で**: `caption_format`（Preprocess）と `prompt_format`（Test）を一致させます。両方 `layout_json`、または通常のキャプションなら両方 `text` です。どちらの場合もプロンプトは Anima のタグ形式で入力し、`layout_json` が変換します。
- **トリガーはキャプションに書かれているとおりの絵師タグです。** 例: `@channel (caststation)`。プロンプトにも同じ書き方で入れます。
- 自然言語のプロンプトは `layout_json` を通っても 1 つの caption 文字列として渡され、テキストのままのプロンプトと同じように動作します。
- ComfyUI では、`task = layout` を指定した **MOSS Caption Helper** ノード（[MOSS Toolkit](https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit) パック）がテキストエンコーダーの前で同じ変換を行います。

### 3.2 前処理（Preprocess）

全画像の Qwen3.5-9B テキスト埋め込みと FLUX.2 VAE 潜在を事前にキャッシュし、学習中はこの 2 つのモデルを読み込みません。

| 項目 | デフォルト | 役割 |
|---|---|---|
| `src` | `post_image_dataset/resized` | 画像 + キャプションのフォルダ |
| `out` | `post_image_dataset/moss` | キャッシュフォルダ: 画像 1 枚につき 2 ファイル、フラットに保存 |
| `resolution` | `1024` | ピクセル面積の上限: 元の比率のまま最大約 1024² ピクセル、両辺は 16 の倍数、アップスケールはしない |
| `caption_format` | `layout_json` | `layout_json` または `text`（§3.1） |

**Run preprocessing** を押します。ステータス行に画像、キャプション、キャッシュ済みファイルの数が表示されます。再実行は差分処理です。新しい画像は追加され、キャッシュ後に編集されたキャプションは再エンコードされ（件数がステータス行に表示されます）、`caption_format` を変更するとテキストキャッシュが再エンコードされます。サブフォルダ間でファイル名が重複する画像は、キャッシュファイル名の先頭にフォルダ名が付きます。

### 3.3 学習（Train）

| 項目 | デフォルト | 役割 |
|---|---|---|
| `cache` | `post_image_dataset/moss` | 前処理の出力 |
| `output` | `output/ckpt_moss/moss_lora.safetensors` | LoRA の保存先。隣に `.json` の実行レポート（ステップごとの loss）が保存される |
| `model` | `base` | 学習に使う DiT |
| `epochs` | `8` | キャッシュを回る回数（1 ステップに画像 1 枚） |
| `save_every_epochs` | `0` | N エポックごとに最終ファイルの隣の `<name>/<name>_e10.safetensors` などとして追加保存（最後のエポックは最終ファイルそのもの） |

ランク、alpha、lr、アダプターの種類は §4 で扱います。**Train** は `cache` を読み込み、**Preprocess, then train** は先に前処理を実行してから `cache` をその `out` に合わせて学習します。

### 3.4 テスト（Test）

同じシードでプロンプトを 2 回レンダリングします。1 回は `lora_multiplier` の強さで LoRA を適用し、もう 1 回は LoRA なしです。2 枚はパネルに並べて表示され、違いは LoRA だけによるものです。

- `lora` を空欄にすると Train タブの `output` を使います。別のファイル（例: `_e10` チェックポイント）をテストするにはここで指定します。
- `model` を `sft` にすると、Base の LoRA を SFT でテストします。
- 出力: `output/tests/<日付-時刻>/lora/` と `…/base/`、それぞれ `manifest.json` 付き。テストごとに新しいフォルダが作られるので、以前の結果は残ります。

---

## 4. 推奨設定

比較は 1 人の絵師（`@channel (caststation)`）の画像 30 枚、Base、20 エポック = 600 ステップ（約 18–21 分）で行いました。

| 項目 | デフォルト | メモ |
|---|---|---|
| `algo` | `lora` | `lokr` / `loha`: §4.1 |
| `rank` | `32` | ランク 16 でパラメータ 37.2 M（以前のデフォルト） |
| `alpha` | `lora` は `128`（ランク 32 で scale 4）、`lokr` / `loha` は = rank | scale = alpha / rank |
| `lr` | `5e-5` | 5 % の線形ウォームアップ |
| `optimizer` | `muon` | `adamw` は以前のデフォルト。`lokr` / `loha` は常に AdamW を使う |
| `down_init` | `kaiming` | `kaiming` または `weight_svd` |
| `epochs` | `8` | 計測した学習は 20 |
| `plain_caption_rate` | `0.2` | `layout_json` キャッシュのみ: layout JSON の代わりに通常の Anima キャプションで学習するステップの割合 |

- **lr、チャンネルスケーリング、DC 分離**: FFN の入力では 1 つの入力チャンネルだけが他よりはるかに大きくなっています。AdamW では、LoRA がそのチャンネルを通じて全体的なシフトに更新を使ってしまうのを防ぐ 2 つの補正（チャンネルスケーリングと `dc_isolate`）を使っていました。Muon のデフォルトで kaiming の `down` 初期化なら、どちらも学習後の LoRA を変えないため、両方とも削除しました。以前のデフォルトと各変更の根拠となった比較: `docs/moss/channel_scaling.md`。
- **optimizer**: Muon（MOSS の事前学習に使われたオプティマイザ）は、同じ lr で AdamW と同じ大きさだけ LoRA を動かしつつ、より多くの方向に分散させ、同じシードでより良く見えました。全体的なシフトに更新を使わないことも、それ自体で果たします。`docs/moss/findings/muon_lora.md`。
- **down_init**: `kaiming` はランダム初期化で、`weight_svd` は LoRA の down 行列をベース重みの上位特異ベクトルから始めます。これらのベクトルは入力の定数オフセットを強く読み取り、Muon では `weight_svd` の LoRA が全トークンの出力を `kaiming` の約 3 倍も大きくシフトさせます。画像の差はシードを変えた程度です。`docs/moss/findings/muon_channel_init.md`。
- **epochs**: 20 エポック中 10 エポック時点の結果はゆるく多様で、20 エポックではより整って均一になる一方、正面・中央の上半身構図へ少し寄りました。`save_every_epochs` を設定すれば 1 回の学習で両方を比べられます。
- **sigma_\***（Advanced）: デフォルトは MOSS の最終事前学習段階のノイズ分布です。Anima の分布（`sigma_mu 0`、`sigma_s 1`、`sigma_cutoff 1.0`、`sigma_uniform 0`）は全設定の中で転写が最も弱くなりました。

### 4.1 LoKr と LoHa

`algo` = `lokr`（クロネッカー積）または `loha`（ランク r のペア 2 組のアダマール積）で、LyCORIS の実装です。ファイルは LyCORIS のテンソル名のままで、同じ `lora` 欄から読み込めます。`down_init` は `lora` にのみ適用されます。

| 構成 | 項目 | 速度（16 GB、1024²） | 絵師セット、lr 1e-4 での結果 |
|---|---|---|---|
| LoKr f8 | `rank 100000`、`factor 8` | 2.26 s/step、空き 3.6 GB | LoRA に近い。顔の個性はやや弱い |
| LoHa r16 | `rank 16` | 2.49 s/step、空き 1.7 GB | 行き過ぎ: 灰青緑へのパレット変化、セットの衣装が他のキャラクターに出る |

- この数値は bf16 構成（§4.2）で測定したものです。LoHa は bf16 が必要で、デフォルトの int8 ベースは LoHa を拒否します。LoKr は int8 でも動きますが、そのときのピークは未測定です。
- bf16 構成では、どちらも LoRA よりメモリを使います。**Advanced** で `activation_memory_budget 0.1` を設定し、`blocks_to_swap` は 24 以上にしてください。swap 12・budget 0.15 では 1024² でメモリ不足になりました。
- LoKr の `rank` が大きい因子の辺の半分以上（上の `rank 100000`）だと、その因子はフル行列になり、alpha は無視されます（scale 1）。
- LoHa: 次に試す値は lr 5e-5 で、まだ計測していません。

### 4.2 メモリ構成（16 GB）

デフォルト: `grad_checkpointing` オフ、`compile` オン、**Advanced** の `int8 stream` と GPTQ ファイル（DiT のブロック重みを int8 で実行。ファイルは初回使用時にダウンロード）、`stream fp32`（ブロックの残りは fp32 で実行）、`blocks_to_swap 0`（全ブロックをカードに置く）、`activation_memory_budget 0.1`（forward の結果のうち backward 用に保持する割合。残りは再計算）、`ema_card` オン、`expandable_segments` オン（シーケンス長が変わっても CUDA メモリプールが断片化しないようにする）。

この構成は 1024² のサンプルで 1 step あたり 1.07 秒、ピーク 10.78 GB で、システムメモリも 4.4 GiB 固定（pin）します。この構成では 0.1–0.3 は同じ速度で、0.1 上げるごとにピークが約 1 GB 増えます。0.05 ではピークが 9.98 GB に下がりますが、attention を再計算するため step は 1.16 秒に遅くなります。この構成で最後まで学習した LoRA はまだありません（[詳細](../moss/optimization/int8.md)）。

**bf16 構成。** int8 ベースは LoHa を拒否し、i2i のステップも実行したことがないため、i2i のタスク設定は bf16 のままです。`--config configs/moss/lora_bf16.toml`、または **Advanced** で `int8 off`、`stream bf16`、`ema_card` オフ、`blocks_to_swap 36`（CPU に置き、必要なときに転送する DiT ブロック数）、`activation_memory_budget 0.5`。

| 構成 | 項目 | s/step | 空き VRAM |
|---|---|---|---|
| 以前の bf16 デフォルト | swap 12、budget 0.1（ランク 32） | 1.85 | 約 1.7 GiB |
| さらに前の bf16 デフォルト | swap 12、budget 0.15（ランク 16） | 1.77 | 約 0.7 GiB |
| 余裕あり | swap 24、budget 0.2 | 1.78 | 2.26 GiB |
| チェックポイント | `grad_checkpointing` オン、swap 0 | 2.09 | 1.60 GiB |

bf16 構成（swap 36、budget 0.5、`swap_h2d_only`、`expandable_segments`、ランク 32）は、caststation セットで 1 step あたり 1.66 秒、最大のサンプル（画像 4.2k + テキスト 0.4k トークン）で空き 0.85 GB です。budget は約 0.45 を超えると上げても効果がなく（0.3: 2.05 s/step）、swap 28～36 は速度が同じなので、swap を下げても空き VRAM を手放すだけです。i2i のサンプルは参照画像も含むため、i2i のタスク設定では swap 38、budget 0.175 を使います。swap 28 では budget 0.2 でメモリ不足になり、swap 38 にすると高い budget の余地ができます。約 8.3k の結合トークン（sanitize）で、swap 38 / budget 0.175 は 1 step あたり 3.79 秒で、swap 28 / budget 0.15 の 4.17 秒より速くなります。`swap_h2d_only` はスワップされる全ブロックの page-locked コピーを RAM に保持し（最大 8.8 GB）、スワップがカードへのコピーだけで済み、戻しのコピーが発生しないようにします。結果と VRAM はオフのときと同じで、swap 38 までは速度も同じ、swap 40 では 8% 速くなります。デフォルトでオンで、ブロックをスワップするときだけ働き、`--no-swap_h2d_only` でオフにできます。

表の条件: 合成 1024² 画像 + テキスト 300 トークン（合計約 4.4k トークン）、LoRA ランク 16、RTX 5070 Ti。以前のデフォルト（ランク 32、budget 0.1、swap 12）は約 4.5k トークン（画像 4096 + テキスト 455）で 1.85 s/step、空き約 1.7 GB、ピーク reserved 約 13.1 GB。ランク 16、budget 0.1 では 1.82 s/step、空き約 2.0 GB でした。実際のサンプルは最大約 4.9k トークンで、キャプションが長いほどメモリを使いますが、`swap 24` なら同じ速度でその分の余裕が残ります。

- swap 8 から 36 の間は速度がほとんど変わらず（1.75–1.84 s/step）、budget 0.1–0.3 も速度にほぼ影響しません。swap 40 では PCIe 転送が計算の裏に隠れきれず遅くなります（2.34 s/step）。それより下では、swap 数は速度ではなく VRAM を決める値です。
- 最適な組み合わせは PCIe リンクと GPU によって変わります。`scripts/moss/tune_swap.py`（CLI、[付録](#付録-cli)、約 4 分）がキャッシュ内の最大サンプルで bf16 構成の swap 数ごとの速度を測り、学習に使う項目の値を表示します。
- `blocks_to_swap -1` は空き VRAM から swap 数を決めます。LoRA 基準で調整されているため、LoKr/LoHa では確保量が足りません。
- `compile` をオフにするには `grad_checkpointing` をオンにする必要があります。compile ありの学習は 3.1 に対して 2.0 s/step で、開始時に約 30 秒のコンパイルが入ります。

---

## 5. テスト

サンプラーのデフォルトは MOSS のリファレンスパイプラインに従います。Anima で使っていたサンプラー設定はそのままでは当てはまりません。

| 項目 | デフォルト | 役割 |
|---|---|---|
| `cfg` | `3.0` | CFG スケール。`1.0` = uncond パスなし |
| `uncond_skip` | `22` | uncond パスは 22 番ブロックを飛ばしたモデル（1 ステップのパスは 2 回のまま）。`''` = 通常の CFG |
| `shift` | `5.0`（Advanced） | スケジューラーの shift |
| `steps` | `50` | |
| `sampler` | `er_sde` | `er_sde` または `euler` |
| `width` / `height` | `1024` | 16 の倍数 |
| `negative_prompt` | 空 | パイプラインのデフォルト |
| `dave` | 空 | `auto` = DAVE: 同じプロンプトでシードごとのバリエーションが増える（構図、ポーズ、レイアウト）代わりに、手が少し甘くなる。`docs/moss/dave.md` |
| `dave_strength` | `0.3` | DAVE が取り除く量。`dave_tau`（Advanced、`0.10`）= DAVE を適用する序盤ステップの割合 |

- **er_sde と euler**: `euler` はリファレンスパイプラインのサンプラーです。`er_sde` は同じコストで線がよりシャープになり、ディテールも密になります。LoRA 適用時は手もきれいに出ました。一方で、プロンプトの一部のディテール（夕焼けの光、髪色、"sitting on a bench"）から外れ、セットの絵柄も少し失われました。同じシードでも 2 つのサンプラーは別の画像を出すので、LoRA とベースの比較は同じサンプラー内で行ってください。
- **Spectrum**（チェックボックス、デフォルトでオン）: ほとんどのステップで 42 ブロックすべてを実行する代わりにモデル出力を予測します。50 ステップ、デフォルトの `spectrum_flex_window 0.5`（Advanced）で 2.6 倍速く、VRAM は 0.85 GiB 増え、グリッド表示では 50 ステップ全実行と見分けがつきにくい結果です。`0.25` は 2.2 倍でより保守的、`1.0` は 3.1 倍ですが細部が崩れます。
- **compile**（Advanced、オン）: Spectrum の有無にかかわらずさらに約 1.37 倍速くなり、ジョブごとに数秒コンパイルします。構図はコンパイルなしと同じですが、細部（髪の毛の筋、模様）は変わるので、コンパイルした画像同士で比較してください。

| Base、1024²、50 ステップ | 1 枚あたりの時間 |
|---|---|
| 通常、コンパイルなし | 72 秒 |
| 通常、コンパイルあり | 52 秒 |
| Spectrum、コンパイルあり | 20 秒 |

- **ComfyUI**: MOSS の LoRA は、`moss-image` ノードで作ったモデルに ComfyUI の LoRA ローダーで読み込めます（2026-10-17 より前に保存したファイルは読み込めません）。**KSampler (Spectrum for MOSS)**（Spectrum ノードパック）がこれらのサンプラーのデフォルトと Spectrum を備え、`ComfyUI-Anima-DAVE` 0.2.0 以降が MOSS で DAVE を実行します。

**比較のしかた**: Test ボタンは最初から LoRA ありとなしを同じシードでレンダリングします。2 つの LoRA やエポック同士を比べるときは、プロンプト、シード、サンプラー、compile 設定を固定し、`lora` だけを変えてください。

---

## 6. トラブルシューティング

| 症状 | 対処 |
|---|---|
| 学習中にメモリ不足 | Advanced: `blocks_to_swap 20`（int8 構成でピークが 2.1 GB 下がり、速度は同じ）、`activation_memory_budget 0.05`（1 GB 減、8% 遅い）、または `grad_checkpointing` オン（int8 構成でピーク 9.61 GB、16% 遅い）。それでも駄目なら `blocks_to_swap 24` の bf16 構成（§4.2）、または `tune_swap.py` を実行。この構成では LoKr/LoHa は 24 / 0.1 が必要 |
| `int8 base: loha builds W + ΔW` | int8 ベースは LoHa を扱えません。bf16 構成（§4.2）を使ってください |
| スタイルが弱すぎる | エポックを増やす（計測した学習は 20）。プロンプトの絵師タグがキャプションと同じ表記か、`prompt_format` が `caption_format` と一致しているかを確認 |
| スタイルが強すぎる、セットのサインやウォーターマークが出る、どの画像も同じ構図になる | 早いエポックの `_e<N>` チェックポイントをテストするか、`lora_multiplier` を下げる。サインを写し始めたのは実効 lr を上げた設定。`lr 2e-5` を試してください |
| セットの衣装や色が他のキャラクターに漏れる | `caption_format layout_json` を使う（デフォルト。通常のタグより漏れが少なかった）。LoHa は LoRA か LoKr に戻す |
| `caption_format` を変更した | 前処理を再実行: テキストキャッシュは再エンコードされ、潜在は再利用される。ステータス行の件数には形式変更が反映されない。`prompt_format` を合わせてから再学習 |
| `layout_json needs the danbooru tag CSV` | `make download-model ARGS=danbooru_tags` |
| Test で `No LoRA at …` | 先に学習するか、Test ボックスで `lora` を指定 |

---

## 付録. CLI

GUI が投入するジョブと同じもので、デーモン経由で直接実行することもできます。各スクリプトの `--help` にすべてのフラグが表示されます。 `train.py --config configs/moss/lora.toml` でフラグを TOML ファイルから読み込めます（同梱のデフォルト値。コピーしてレシピとして使えます）。コマンドラインで指定したフラグが優先されます。

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/cache.py --src 'post_image_dataset/resized/<set>' --out 'post_image_dataset/moss/<set>'"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/lora.toml"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/lora_bf16.toml"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --lora output/ckpt_moss/moss_lora.safetensors"
make daemon-run ARGS="scripts/moss/tune_swap.py --cache post_image_dataset/moss/<set>"
```

LoKr / LoHa:

```bash
make daemon-run ARGS="scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20 --algo lokr --rank 100000 --factor 8 --config configs/moss/lora_bf16.toml --blocks_to_swap 24 --activation_memory_budget 0.1"
make daemon-run ARGS="scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20 --algo loha --rank 16 --lr 5e-5 --config configs/moss/lora_bf16.toml --blocks_to_swap 24 --activation_memory_budget 0.1"
```

詳しい計測結果: [`docs/moss/README.md`](../moss/README.md)。
