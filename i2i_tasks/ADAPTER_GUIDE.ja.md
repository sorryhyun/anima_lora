# 自前の EasyControl アダプターを作る

このガイドでは、Anima に**新しい EasyControl コントロールタスク**を自分用に追加する方法を説明します。git へのコントリビューションとしてではなく、`i2i_tasks/<your_task>/` の下に置くローカルアダプターとして追加します。正準的な例である **colorize** (`i2i_tasks/colorization/`) をステップごとに辿り、自分のタスク向けに何を変えればよいかを正確に示します。

一つだけ覚えておくことがあります:

> **モデルコードを書くわけではありません。** ネットワーク、フォワードパス、`b_cond` ゲート、推論キャッシュ — これらはすべて出荷済みで、あらゆるコントロールタスクで共有されています。タスクごとに変わるのは **参照画像をどう作るか** だけです。このガイドの残りはすべてその周りの配管です。

---

## 0. アイデア — EasyControl アダプターとは何か

EasyControl は **参照画像** を使って生成を導きます。参照は VAE を通じて *cond トークン* に変換され、生成中の画像と並んで流れていきます。各ステップでモデルは両方を見ます。(完全なアーキテクチャは `docs/anima/experimental/easycontrol.md`。)

デフォルトの EasyControl は**同じ画像**を参照とターゲットの両方に使います — つまりただコピーすることを学習します。コントロールタスクはこれを変えます: 各ターゲットを**別の**参照とペアにすることで、モデルがコピーではなく `reference → target` を学習するようになります。

| | デフォルト EasyControl | コントロールタスク (例: colorize) |
|---|---|---|
| ターゲット (ほしいもの) | 画像 X | カラー画像 X |
| 参照 (ヒント) | 画像 X (同一) | X の*変換*バージョン (X の白黒マンガ) |
| 学習すること | コピー | manga → color |
| テキスト | 完全なキャプション | (任意) 短いキャプション |

colorize のやり方 — 再利用できるアイデアです: **実際の白黒マンガにはカラーバージョンがないため**、`(白黒, カラー)` ペアを集めて作ることはできません。そこで方向を反転させます — すでに手元にあるカラー画像をターゲットとし (それが欲しい結果なので)、各画像から白黒の参照を**作ります** (線画 + スクリーントーン、アルゴリズムで)。肝心なのは、作った白黒が推論時に実際に入力するものに見えるようにすることです。`depth → image` や `pose → image` を構築するなら、「参照を作る」ステップは学習画像に対して実行する深度推定器やポーズ検出器になります。

つまりあなたの仕事は: **関数 `target_image → reference_image` を書き、その出力をキャッシュし、データセットをそれに向け、ディスクリプターでデータセットを記述し、タスク名を登録すること** です。

---

## 1. 触れる四つのもの

`<task>` という名前のアダプターを追加するには、以下の四つを正確に作成/編集します:

| # | もの | colorize バージョン | 役割 |
|---|---------|-------------------|--------------|
| 1 | `i2i_tasks/<task>/` プロジェクト | `colorization/` (`mangafy*.py`, `prep.py`) | 参照画像を作ってキャッシュする (任意で短いテキストキャッシュも) |
| 2 | `i2i_tasks/configs/<task>.toml` + `configs/easycontrol/<task>.toml` | `i2i_tasks/configs/colorize.toml` + `configs/easycontrol/colorize.toml` | タスクファイル (`name` スラッグ、任意の `stage`、`[staging]`) と EasyControl ディスクリプター (`[preprocess]` / `[training]` テーブル、`[[datasets]]` 設計図、任意の `[variant]`) |
| 3 | `scripts/tasks/training.py` の `_I2I_STAGES` | `"colorize": {"stage": _colorize_stage, "preprocess": _colorize_preprocess}` | `make i2i-stage` で `TASK=<task>` を、`easycontrol-preprocess` / `easycontrol` で `EASYADAPTER=<task>` を動くようにする |
| 4 | `cmd_test_easycontrol` の `_ADAPTERS` (`scripts/tasks/inference.py`) | `"colorize"` の行 | `make test-easycontrol` 用のチェックポイント接頭辞、出力フォルダ、参照のフォールバックフォルダ、空プロンプトのデフォルト |

順番に見ていきます。どれも `networks/` を触りません。

---

## 2. もの 1 — アダプタープロジェクト (`i2i_tasks/<task>/`)

本当の作業はここにあります。二つの仕事をします: **参照画像を作る**ことと、**キャッシュする**ことです。(任意で三つ目として、タスク固有のテキストキャッシュを構築します。)

### 2a. 参照を作る関数

普通の関数です: カラー画像を RGB `uint8 (H,W,3)` と seed で受け取り、**同じサイズ**の参照画像を RGB `uint8 (H,W,3)` で返します。(同じサイズが重要です — §3 参照。) 同じ seed に対して同じ出力を返すことで、再実行と並列ワーカーがすべて一致します。

colorize ではこれが `mangafy.py::mangafy_array` (と GPU バージョン `mangafy_gpu.py::mangafy_array_gpu`) です:

```python
# i2i_tasks/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

コピーする価値のある四つのポイント:

- **ファイル名から決定論的に seed を作ってください。** colorize は `zlib.crc32(stem)` を使います — プロセスごとに変わって並列ワーカーが食い違う Python の `hash()` *ではありません*。バリエーションを加えてもかまいません (colorize はスクリーントーンの角度をページごとに変えています) — seed から導出するだけで再現性が保たれます。
- **重いものは遅延インポートしてください。** colorize には三つのエンジン (`cv2` / `gpu` / `sd`) があり、ページが実際にそちらへルーティングされた場合にのみ 3.5 GB の SD モデルをロードします。ビルダーがモデル (深度ネット、線抽出器) を必要とするなら、使うときだけインポートしてください。
- **ダウンロードなしのフォールバックは大きな利点です。** colorize の `cv2`/`gpu` エンジンはダウンロードが一切不要で、クリーンなチェックアウトで追加ダウンロードなしに prep + 学習ができます。可能ならこのようなフォールバックを用意してください。
- **ファイルは原子的に書き込んでください。** colorize の `_save_png_atomic` は一時ファイルに書いてから名前を変えます。これがないと中断された実行が半分書かれた PNG を残す可能性があり、「あれば飛ばす」チェックがそのファイルを永遠に信じてしまいます。実際に起きるバグです — このパターンをコピーしてください。

参照が**すでにディスクにある場合** (本物の深度マップ、本物のスケッチ)、ビルドステップをまるごとスキップしてそれらを直接キャッシュするだけで済みます。ターゲットから参照を導出しなければならない場合にのみビルドが必要です。

### 2b. (任意) 短いテキストキャッシュ

colorize は参照を変えるだけでなく、**キャプションを色の単語だけに絞り込み**もします (`anime_tools.captions.color_tags.filter_to_colors`)。その理由は理解する価値があります: 参照 (線画 + スクリーントーン) がすでに *形とレイアウトのすべて* をエンコードしているので、テキストがまだ言うべきことは白黒では表せない一つのもの — **色** — だけです。キャプションを色の単語に絞ると、残ったすべての単語がモデルが参照から得られない情報になり、長いキャプションに埋もれた色の単語による弱いステアリングの代わりに、強い `prompt → color` の結びつきが生まれます。

自分のタスクについて同じ問いを立ててください: **参照がすでに決めることは何で、テキストが決めるべき曖昧なことは何か?** `pose → image` 参照はポーズを固定しますが、服装や設定は固定しません — おそらく *完全な* キャプションを保持するでしょう。`depth → image` 参照はレイアウトを固定しますが、アイデンティティや色は固定しません。colorize の「まだ曖昧なことへキャプションを絞り込む」やり方は考慮すべき *パターン* であってルールではありません — 多くのアダプターはキャプションをそのままにしてテキストキャッシュを完全にスキップします (データセットから `text_cache_dir` を省くだけです — §4)。

キャプションを絞り込む場合、colorize の二つの独立したノブに注意してください (混同しないでください):

- **`caption_dropout_rate`** — 自動着色の *フロア (下限)*。約 5% の学習ステップでキャプションをまるごとドロップし、空プロンプトのときの動作を学習させます。**低く** 保ってください (`0.05`); 高い値は無条件パスを過学習させてプロンプトを弱くします。
- **`use_shuffled_caption_variants`** — 完全対部分の *バランス*。テキストキャッシュは複数のバージョンを持ちます (v0 = 完全な色のセット、v1+ = 各単語が約半分の確率でドロップされたシャッフル版)。ローダーは v0 を 20%、v1+ を 80% 引くので、"pink hair" だけのような部分プロンプトでも機能します。

### 2c. `prep.py` — キャッシュビルダー

三つのステージ、それぞれ **べき等 (idempotent)** (完了済みの作業はスキップするので、再実行しても安全です):

1. **ビルド** — `--src` (`post_image_dataset/resized`) 以下のすべてのカラー画像を走査し、ビルダー関数を実行して、ソースのレイアウトをミラーリングした `--staging` フォルダに参照 PNG を書き込みます。
2. **エンコード** — ステージングされた参照を `library.preprocess.cache_latents` で `--cond_cache_dir` へ VAE エンコードします。各画像の **ネイティブサイズ** で行うことで、参照 latent がターゲット latent と同じ形になります。通常のキャッシュと同じ `{stem}_{WxH}_anima.npz` フォーマットです。
3. **(任意) テキスト** — キャプションをフィルターを通して `--text_cache_dir` へ再エンコードします。`library.preprocess.cache_text_embeddings` に `caption_transform=` (と `caption_shuffle_variants` / `caption_tag_dropout_rate`) を与えて行います。

既存のライブラリヘルパーを使ってください — `library.preprocess.{cache_latents, cache_text_embeddings, tqdm_progress}` と `library.preprocess._dataset.walk_images`。エンコードループを自前で書かないこと; `prep.py` は `scripts/preprocess/*.py` と同じく、これらの上の薄いシェルです。

colorize が対処している、その構造をコピーすれば無償で受け継げる二つの正確性の落とし穴:

- **Stem が一致していなければなりません。** テキストステージはキャプションマスター (`image_dataset/`、`resized/` と同じレイアウト) から `.txt` キャプションを読むので、生成されるテキストキャッシュのファイル名がローダーの `image_dir=post_image_dataset/resized` を参照する形と一致します。キャッシュのファイル名がターゲットの stem と合わなければ、ローダーは黙ってペアにしません。
- **uncond サイドカー。** colorize のテキストステージは共有の `T5("")` 空プロンプトサイドカーがなければ再作成します。テキストキャッシュを構築してキャプションドロップアウトを使うなら同じことをしてください (`library.preprocess.uncond.stage_uncond_sidecar_with_models`)。

---

## 3. 破ってはいけない唯一のルール — 参照とターゲットのトークン数が一致していなければならない

DiT は Anima の **ネイティブ形状バケッティング** (二つのトークン数ファミリー、4032 と 4200; CLAUDE.md と `docs/anima/experimental/easycontrol.md` の "Cond token count" を参照) で動作します。**パディングのノブはありません** — 参照は latent の実際のトークン数で実行されます。

これこそが §2a で参照を入力と **同じサイズ** にするよう言い、§2c で **ネイティブサイズ** でエンコードすると言う理由です: 両方を守れば参照 latent が自動的にターゲットと同じバケットに入り、すべてが機能します。より小さい参照が欲しければ (まったく問題ありません — 小さいほどメモリと速度が節約できます)、**画像** レベルでエンコードの前に縮小してください。そうすれば latent は依然として実際のバケットに入ります。ネットワーク内でトークン数を制限しようとしないでください。

---

## 4. もの 2 — タスクファイルとディスクリプター

2 つのファイルがタスク固有の設定をすべて保持します (`i2i_tasks/configs/colorize.toml` と
`configs/easycontrol/colorize.toml` を参照)。モデルに依存しないタスクファイル
`i2i_tasks/configs/<task>.toml` が `name`、`stage`、`[staging]` を持ち、
残りは EasyControl ディスクリプター `configs/easycontrol/<task>.toml` が持ちます:

- タスクファイルのトップレベル **`name`** — スラッグ。ツリーは
  `post_image_dataset/i2i/<name>/` 以下に作られ、`output_name` のデフォルトは
  `anima_easycontrol_<name>` です。
- タスクファイルの **`stage`** — `_I2I_STAGES` のキー (デフォルト: タスク id)。
- **`[staging]`** (タスクファイル) / **`[preprocess]`** (ディスクリプター) — stage/preprocess
  関数が `prep.py` のフラグに変換するフラットなテーブル (`_toml_table_to_argv`: `--key value`、
  リストは展開、`true` → 値なしの `--flag`、`false` → 省略)。
- **`[training]`** — ベースの `easycontrol` メソッドに対する `train.py` の CLI
  オーバーライドとしてマージされます。
- **`[general]` / `[[datasets]]`** — データセット設計図。`train.py` のデータセットバリデーターは
  他のトップレベルキーを拒否するため、`make easycontrol` はこれを (`{name}` を展開して)
  スラッグディレクトリ以下の生成された `dataset_config.toml` に書き出します。
- **`[variant]`** (任意) — `family = "easycontrol"`、`label`、`description`、
  `order`; このブロックを持つディスクリプターは GUI の EasyControl タブに表示されます。

colorize の設計図のサブセット:

```toml
  [[datasets.subsets]]
  image_dir = 'post_image_dataset/resized'                         # the color targets
  cache_dir = 'post_image_dataset/lora'                            # shared TE/PE cache, reused
  cond_cache_dir = 'post_image_dataset/i2i/{name}/cond'    # reference latents (prep.py)
  text_cache_dir = 'post_image_dataset/i2i/{name}/text'    # color-only text cache (prep.py)
  latent_cache_dir = 'post_image_dataset/i2i/{name}/target' # WB target latents (prep.py)
  recursive = true
  flip_aug = false
  num_repeats = 1
```

- **`cond_cache_dir`** — これがコントロールタスクにします。ローダーは各ターゲットを
  stem で参照レイテントとマッチさせ、参照があるターゲットだけを残します。
- **`text_cache_dir`** — テキストキャッシュのみをリダイレクトします。省略すると
  共有テキストキャッシュを使い、テキストステージをスキップします。
- **`latent_cache_dir`** — ターゲットレイテントのみをリダイレクトします。省略すると
  `cache_dir` を使います。ここに無いレイテントは、補正前の元画像から (警告付きで)
  再エンコードされるので、学習前に埋めておいてください。
- **`flip_aug = false`** — 必須です。cond キャッシュには反転レイテントがなく、
  反転が有効だとローダーがエラーを出します。

検討すべき `[training]` のノブ:

- **`network_args`** — `b_cond_init` は step 0 で参照がどれだけ寄与するかを決めます
  (`-10` ≈ step 0 でベース DiT とほぼ同じ; `docs/anima/experimental/easycontrol.md` の
  "Step-0 baseline equivalence" を参照; colorize は参照が早く効くよう緩めています;
  どちらでも学習可能)、`cond_scale`、`apply_ffn_lora` (`0` で FFN LoRA を外し、
  学習パラメータが約半分に)、`cond_res_scale`。
- **`easycontrol_cond_noise_max`** — 学習中に参照へ加えるノイズ
  (σ ~ `U(0, max)`、`cond + σ·ε`)。`0` は参照を正確な設計図として扱い、
  大きい値は大まかなヒントにしてディテールをテキストに任せます。
- **`easycontrol_drop_p`** — image-CFG のために参照全体をドロップする頻度。
  常に参照があるタスクは `0` にします。
- **`output_name`** — 推論セレクターはこの接頭辞で最新のチェックポイントを探します (§5)。
- **`blocks_to_swap = 0`** — 同梱のディスクリプターが使っている値です。

---

## 5. もの 3 と 4 — タスクを登録する

**`scripts/tasks/training.py`:** シグネチャ `(adapter, cfg, base, extra)` の関数を 2 つ書き、
`_I2I_STAGES` (stage の種類をキーとする) に追加します:

```python
_I2I_STAGES = {
    ...
    "colorize": {"stage": _colorize_stage, "preprocess": _colorize_preprocess},
    "<task>": {"stage": _task_stage, "preprocess": _task_preprocess},
}
```

`_colorize_stage` / `_colorize_preprocess` が手本です: それぞれ stage スキップフラグ、
スラッグ由来のパス (`_colorize_prep_paths(base)`)、テーブルのノブを付けて
`i2i_tasks/colorization/prep.py` を実行し、`extra` を最後に付けてユーザーの `ARGS` が優先されるようにします。
`_easyadapter()` は `i2i_tasks/configs/<task>.toml` が無いタスクを拒否し、タスクファイルの `stage` がこの dict に無い場合は stage 時に失敗します。
`make easycontrol EASYADAPTER=<task>` にはこれ以上の編集は不要です。

**`scripts/tasks/inference.py`** (`cmd_test_easycontrol`): `_ADAPTERS` に行を追加します:

```python
"<task>": {
    "weight": "anima_easycontrol_<task>",   # prefix of output_name
    "out": "<task>",                         # output/tests/<out>/
    "ref_dir": ROOT / "post_image_dataset" / "resized",  # random ref if none given
    "empty_prompt": False,                   # True → default to --prompt ""
},
```

---

## 6. 実行する

```bash
# First time only: shared target latents + text cache in post_image_dataset/lora.
make preprocess

# 1. Build the reference tree. QA a few first, then eyeball the PNGs under
#    post_image_dataset/i2i/<name>/staging/.
make i2i-stage TASK=<task> ARGS="--limit 8"
make i2i-stage TASK=<task>

# 2. Encode the caches. Idempotent.
make easycontrol-preprocess EASYADAPTER=<task>

# 3. Train (DiT frozen, adapter only).
make easycontrol EASYADAPTER=<task>

# 4. Inference — give it a real, in-distribution reference image.
REF_IMAGE=path/to/condition.png make test-easycontrol EASYADAPTER=<task>
#    Steer with text:  ... ARGS='--prompt "..."'
```

### 推論のメモ

- 推論時、参照はビルドステップなしでそのまま VAE エンコードされます。ビルダーが
  真似ようとしたものを入力してください (colorize: グレースケール写真ではなく、
  実際にスクリーントーンが入ったページ)。
- `make test-easycontrol` は常に `--easycontrol_image_match_size` を渡し、
  参照のアスペクト比に合うトークンバケットを選びます。
- `--easycontrol_scale` (`EC_SCALE=`) は出力が参照にどれだけ従うかを決めます;
  `1.0` が学習時のデフォルトです。
- colorize で調整済みの `--guidance_scale` / scale / ステップ範囲は
  `colorization/README.md` ("Inference settings") にあります。

---

## 7. チェックリスト

- [ ] `i2i_tasks/<task>/` に決定論的でアトミックに書き込むビルダー
      (`(img, seed) → reference`、同じサイズ) と冪等な `prep.py`。
- [ ] 参照を**元のサイズ**でエンコード (§3)。
- [ ] `i2i_tasks/configs/<task>.toml`: `name`、`[staging]`; そして
      `configs/easycontrol/<task>.toml`: `[preprocess]` / `[training]`、`cond_cache_dir` (+ 任意の
      `text_cache_dir` / `latent_cache_dir`) と `flip_aug = false` を持つ設計図。
- [ ] `_I2I_STAGES` エントリ (training.py) と `_ADAPTERS` の行 (inference.py)。
- [ ] 本番実行の前に `--limit 8` のステージングバッチを目視確認。

---

## 8. さらに読む

- **`i2i_tasks/colorization/README.md`** — colorize のリファレンス実装:
  キャプション方針、スクリーントーンのバンド、タグスライス、推論設定。
- **`docs/anima/experimental/easycontrol.md`** — ネットワーク自体: 2 ストリームのフォワード、
  `b_cond` step-0 ベンチ、`cond_res_scale`、推論キャッシュ、メモリ、制限。
- **`networks/methods/easycontrol.py`** — `EasyControlNetwork` とパッチされた
  `Block.forward`。
- **`networks/CLAUDE.md`** — モジュール別マップとディスパッチルール。
