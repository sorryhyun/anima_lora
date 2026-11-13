"""Japanese strings for the MOSS-Image GUI; assembled in ``gui.moss.strings``."""

from __future__ import annotations

UI: dict[str, str] = {
    "window_title": "MOSS-Image LoRA（alpha）",
    "model_missing": "モデルが見つかりません: {path}\n`make download-model ARGS=moss-base`"
    " を実行する（SFT DiT は ARGS=moss-sft）か、詳細 → model_dir を指定するか、"
    ".env に ANIMA_MOSS_MODEL_DIR を設定してください。",
    "scan_stale": "{stale} 件のキャプションがテキストキャッシュ以降に変更されています — "
    "次の前処理で再エンコードされます。",
    "scan_dupes": "{n} 枚の画像が別のサブフォルダの画像と同じファイル名です — "
    "それらのキャッシュファイル名にはフォルダ名が接頭辞として付きます。",
    "tab_i2i_preprocess": "i2i 前処理",
    "tab_i2i_train": "i2i 学習",
    "i2i_task": "タスク",
    "i2i_no_tasks": "configs/moss/i2i/ に i2i タスク設定がありません。",
    "i2i_guide": "i2i LoRA は参照画像をターゲット画像に変換することを学習します: "
    "参照画像はクリーンな潜在変数トークンとして毎回の forward に加わります。"
    "ステージングがペアを作り、前処理が各ターゲットを参照画像（とキャプション）と一緒に"
    "キャッシュし、学習が LoRA を学習し、テストがその LoRA で新しい参照画像を変換します。"
    "各フォームは configs/moss/i2i/<task>/ から始まり、変更した値だけが保存されます。",
    "i2i_task_colorize": "colorize: 白黒マンガのページ → カラー。ステージングでカラー"
    "コーパスをマンガ風に変換します。ページは原寸の参照画像になり、キャプションには"
    "OCR テキストを含まない制御用の項目だけが入ります（ビジョンタワーは使いません）。"
    "マンガページのテストのプロンプトは「comic」にします（ボックスのみの text 項目と"
    "キャラクターごとの色の項目は任意）。イラストはカラー + 作品タグでテストします。",
    "i2i_task_sanitize": "sanitize: 文字と吹き出しを消します。ステージングで同じ作者の"
    "ほぼ同じ画像ペア（文字あり / なし）を探します。文字ありのページが原寸の参照画像に"
    "なります。テストのプロンプトは、そのページのキャプションから文字関連のタグを除いた"
    "ものです。",
    "i2i_stage_group": "1 · ステージング",
    "i2i_stage_hint": "データセットから post_image_dataset/i2i/ にタスクのペアツリーを"
    "作ります（make i2i-stage TASK={task}。設定は i2i_tasks/configs/{task}.toml の "
    "[staging] テーブル）。EasyControl も同じツリーを読みます。",
    "run_stage": "ステージング",
    "i2i_cache_group": "2 · 前処理",
    "i2i_train_group": "3 · 学習",
    "i2i_chain": "ステージング、前処理してから学習",
    "i2i_test_group": "4 · テスト",
    "i2i_test_hint": "各参照画像を LoRA で変換し、横に並べて表示します。ref_image は"
    "画像1枚または画像フォルダです。ref_scale と ref_vision は LoRA を学習したキャッシュと"
    "一致させてください。",
    "i2i_staged": "ステージング済みペア {n} 組。",
    "i2i_not_staged": "まだステージングされていません: {path} がありません。"
    "先にステージングを実行してください。",
    "i2i_stage_next": "ステージング完了 — 前処理を開始します。",
    "i2i_need_ref": "ref_image に画像1枚または画像フォルダを指定してください。",
    "i2i_need_prompt": "ref_vision がオフの場合、テストにはプロンプトが必要です: "
    "ページのキャプション。",
    "test_ref": "参照画像",
    "model_help_test": "生成に使う DiT(base / sft)とその動かし方。int8(デフォルト): "
    "ブロックの重みを int8、残りを fp32 stream で動かします。1024² で 7.8 GiB、Base の"
    "ファイルは初回使用時に取得し、sft は最近傍丸めを使います。loha とブロックスワップは"
    "使えません。bf16: 1024² で 10.7 GiB。切り替えると int8、stream、blocks_to_swap がまとめて変わり、"
    "これらは詳細で引き続き調整できます。",
    "model_help_i2i": "学習する DiT(base / sft)とその動かし方。bf16(ここでのデフォルト): "
    "このタスクで測定済みの設定。int8: swap 20、budget 0.1。sanitize の最大サンプルは 6 "
    "ステップの間ピーク 12.3 GB に収まりましたが、この設定で i2i LoRA を最後まで学習した"
    "ことはまだありません。loha は使えません。切り替えると int8、stream、ema_card、"
    "blocks_to_swap、swap_h2d_only、activation_memory_budget がまとめて変わり、これらは"
    "詳細で引き続き調整できます。",
    "model_help": "学習する DiT(base / sft)とその動かし方。int8(デフォルト): ブロックの"
    "重みを int8 で動かします。約 1.1 s/step、ピーク 13 GB で、システムメモリ 4.4 GiB が"
    "固定され、Base のファイルは初回使用時に取得し、sft は最近傍丸めを使います。loha は"
    "使えません。bf16: 36 ブロックのスワップで同じピークのまま約 1.6 s/step で、loha を"
    "使えます。切り替えると int8、stream、ema_card、blocks_to_swap、swap_h2d_only、"
    "activation_memory_budget がまとめて変わり、これらは詳細で引き続き調整できます。",
    "recipe": "レシピ",
    "recipe_lora": "LoRA",
    "recipe_trigger_registers": "トリガーレジスタ",
    "recipe_selfdistill": "自己蒸留",
    "recipe_help": "configs/moss/ の学習 config を読み込みます: その値と、他のレシピが設定するキーはデフォルト値に。cache、output、model、trigger と int8 / bf16 プランはそのままです。LoRA: 標準レシピ(lora.toml)。トリガーレジスタ: 差分をトリガーの行だけに置き、その後に学習行を加えます。trigger を設定してください。自己蒸留: flow loss に加え、σ ≥ 0.9 で学習画像をテキストエンコーダに入れた LoRA の EMA 0.97 コピーの velocity に合わせます。16 epoch、lr 1e-4。teacher 埋め込み(画像 + キャプションを TE に通す、30 枚で約 1 分)は前処理で distill_teacher をオンにすると作られ、そうでなければ Train が欠けた・古いものを先に作ります。一般的なプロンプトや複数キャラのプロンプトで標準 LoRA より安定し、画風はやや弱くなります。docs/moss/experimental/self_distill.md。",
    "distill_teacher": "自己蒸留: teacher 埋め込みを先にキャッシュします(画像 + キャプションをテキストエンコーダに通す。変わったファイルだけ再エンコード)。",
}

FIELDS: dict[str, tuple[str, str]] = {
    "src": (
        "画像フォルダ",
        "画像と同名の .txt キャプションを含むフォルダ、サブフォルダを含む"
        "（既定: Anima の resized ツリー。そのキャプションは修正版です。"
        "そこにキャプションだけあって画像がない場合は image_dataset/ の元画像を使います）",
    ),
    "out": (
        "キャッシュフォルダ",
        "画像ごとにキャッシュファイルを2つ（テキスト埋め込み + 潜在変数）書き込みます",
    ),
    "resolution": (
        "解像度",
        "ピクセル面積の上限を正方形の辺で指定 — 1024 は元のアスペクト比のまま最大約 "
        "1024² ピクセル、両辺は 16 の倍数。拡大はしません",
    ),
    "caption_format": (
        "キャプション形式",
        "テキストエンコーダがエンコードする内容: layout_json = Anima キャプションを MOSS の "
        "layout JSON プロンプトに変換（danbooru タグ CSV が必要）、text = .txt をそのまま、"
        "fixed = すべての画像で fixed_caption をエンコード。生成時も同じ形式を使ってください",
    ),
    "overwrite": ("上書き", "キャッシュが最新でも再エンコードする"),
    "skip_text": ("テキストをスキップ", "テキストエンコーダの処理をスキップ"),
    "skip_latents": ("潜在変数をスキップ", "VAE エンコードをスキップ"),
    "caption_variants": (
        "キャプション変形",
        "Anima キャプション段の変形({stem}.variants.txt の v1, v2, …: 先頭タグの後ろを並べ替え、"
        "タグドロップアウトをかけたキャプション)も embeds_v1… としてエンコードします。それぞれキャプション形式で変換。"
        "学習では caption_variant_rate で使います",
    ),
    "distill_teacher": (
        "自己蒸留 teacher キャッシュ",
        "自己蒸留(Recipe → 自己蒸留)の学習が読む teacher 埋め込みも作ります: 各画像 + キャプションを TE の vision tower に通し {stem}.moss_vis.safetensors に保存。変わったファイルだけ再エンコードし、TE 読み込み後 30 枚で約 1 分。オフでも自己蒸留の Train が欠けた・古い teacher を先に作ります",
    ),
    "model_dir": (
        "モデルフォルダ",
        "MOSS-Image-Base/ と MOSS-Image-SFT/ を含むフォルダ"
        "（既定: $ANIMA_MOSS_MODEL_DIR または models/moss_image）",
    ),
    "cache": ("キャッシュフォルダ", "前処理が書き込んだキャッシュフォルダ"),
    "output": (
        "出力",
        "LoRA の出力パス（同じフォルダに .json の実行レポートが書かれます）",
    ),
    "model": ("モデル", "学習に使う DiT（base / sft）"),
    "epochs": ("エポック数", "キャッシュを周回する回数"),
    "max_steps": ("最大ステップ数", "このステップ数で停止（0 = エポック数で決まる）"),
    "algo": (
        "アダプター",
        "lora、または LyCORIS の lokr（クロネッカー積）/ loha（ランク r 行列2組のアダマール積）",
    ),
    "rank": ("ランク", "LoRA ランク（lokr: 大きいクロネッカー因子のランク）"),
    "alpha": (
        "alpha",
        "LoRA alpha（lora の既定は 128、lokr / loha の既定はランクと同じ）。"
        "スケール = alpha / ランク",
    ),
    "factor": (
        "lokr 因子",
        "lokr のみ: 小さいクロネッカー因子の辺の上限（-1 = 正方形に近い分割）",
    ),
    "down_init": (
        "down 初期化",
        "lora のみ: kaiming = 既定のランダム初期化。weight_svd = ベース重みの上位 r 個の"
        "右特異ベクトル / √3 で初期化（Anima の SVD-Down）。weight_svd_nodc = 同じ SVD "
        "基底から較正済みの入力 DC 方向（channel_mean）を除いたもの、初期 down · x̄ = 0",
    ),
    "lr": ("学習率", "学習率（既定は Muon、lokr / loha は AdamW）"),
    "warmup_ratio": (
        "ウォームアップ比率",
        "線形ウォームアップが全ステップに占める割合",
    ),
    "grad_accum": (
        "勾配累積",
        "オプティマイザ更新1回あたりに累積するサンプル数。ステップ数、エポック数、"
        "損失ログは引き続きサンプル単位、ウォームアップは更新単位で数えます",
    ),
    "save_every_epochs": (
        "N エポックごとに保存",
        "N エポックごとに追加で保存（0 = 最終結果のみ）",
    ),
    "caption_dropout": (
        "キャプションドロップアウト",
        "空プロンプトの埋め込み（CFG の無条件側）で1ステップ学習する確率",
    ),
    "plain_caption_rate": (
        "Anima キャプション比率",
        "layout JSON の代わりに元の Anima 形式キャプションで1ステップ学習する確率。"
        "元のキャプションを持つのは caption_format=layout_json のキャッシュだけです",
    ),
    "lora_dtype": (
        "LoRA 精度",
        "保存する LoRA ファイルの精度。学習中は fp32 のマスター重みを保持し、"
        "ランク行列の積は bf16 で計算します",
    ),
    "grad_checkpointing": (
        "勾配チェックポイント",
        "ブロック単位の活性チェックポイント",
    ),
    "seed": ("シード", "乱数シード"),
    "n": (
        "イラストペア数",
        "学習に使うイラストペア数（ステージング済みのマンガ風ページ + resized のカラーターゲット）",
    ),
    "manga": (
        "マンガページ数",
        "追加する manga_color のマンガページ数（0 = イラストのみ）",
    ),
    "holdout": ("ホールドアウトイラスト数", "評価用に取り分けるイラストペア数"),
    "manga_holdout": (
        "ホールドアウトマンガ作品数",
        "評価用に取り分けるマンガ作品数、各1ページ。その作品のどの版も学習に使われません",
    ),
    "ref_scale": (
        "参照画像の縮小",
        "参照潜在変数の辺 = ターゲットの辺 ÷ この値、16 の倍数（生成時も同じ値を使う）",
    ),
    "variants": (
        "キャプションバリアント数",
        "captions control: ページごとの抽選数（embeds + embeds_v1 … "
        "embeds_v{variants-1}）。captions ocr: 疎キャプション数 + 1。embeds_v1 … "
        "embeds_v{variants-1} は OCR レイアウト付きのカラーキャプションで、奇数 k は"
        "タグの tag_dropout 割合を、偶数 k は全部を除きます（マンガページはマンガタグを"
        "1つ残す）。奇数なら2種類が半々。最小 3",
    ),
    "tag_dropout": (
        "タグドロップアウト率",
        "captions ocr: 奇数 k の疎キャプションがカラーキャプションから除くタグの割合。"
        "captions control: 各抽選がプールから除く割合",
    ),
    "captions": (
        "キャプション方式",
        "control（v5）: イラストはカラー + 作品タグから抽選します（bare: 空のプロンプト）。"
        "マンガページは常に comic と、swap_text の keep_line が残す OCR 行ごとのボックスのみの"
        " text 項目を持ち、キャラクターごとのボックスとその髪 / 目 / 服の色から抽選します"
        "（カラーのターゲットに SAM3 + Anima Tagger。bare: キャラクターなし）。embeds と"
        "各 embeds_v{k} はプールの tag_dropout を除いた抽選、embeds_plain は "
        "plain_caption_rate 用の bare キャプション。vision オフが必要。ocr: 完全 / 疎 / "
        "bare のカラーキャプションに OCR レイアウトを重ねたもの（v4）",
    ),
    "vision": (
        "ビジョンタワーでページをエンコード",
        "ページがビジョンタワー経由で、各キャプションの前にテキストエンコーダへ入ります。"
        "オフ = キャプションのみエンコード、embeds_plain は書かない",
    ),
}

FIELDS_GENERATE: dict[str, tuple[str, str]] = {
    "prompt": (
        "プロンプト",
        "プロンプト（prompts_file または example を指定すると無視）",
    ),
    "negative_prompt": (
        "ネガティブプロンプト",
        "ネガティブプロンプト（パイプラインの既定は空）",
    ),
    "prompt_format": (
        "プロンプト形式",
        "layout_json = Anima 形式のプロンプトを MOSS の layout JSON に変換（キャッシュの"
        "キャプション形式と同じ）。text = 入力どおり。すでに JSON のプロンプトや、text 形式の"
        "キャッシュで学習した LoRA 向け",
    ),
    "lora": (
        "LoRA",
        "テストする LoRA（空 = 学習出力）。Base で学習した LoRA は SFT でも使えます",
    ),
    "lora_multiplier": ("LoRA 強度", "LoRA の強度。0.0 でベースモデルそのもの"),
    "width": ("幅", "16 の倍数"),
    "height": ("高さ", "16 の倍数"),
    "steps": ("ステップ数", "デノイズのステップ数"),
    "cfg": ("CFG", "CFG スケール。1.0 = 無条件 forward なし"),
    "sampler": (
        "サンプラー",
        "er_sde = 同じ σ グリッド上の ER-SDE 確率的サンプリング（線がよりシャープ、"
        "forward の追加なし）。euler = パイプライン既定の ODE",
    ),
    "uncond_skip": (
        "無条件側でスキップするブロック",
        "カンマ区切りの DiT ブロック番号（'' = 通常の CFG）。パイプラインの既定は "
        "22 番ブロックをスキップ",
    ),
    "seed": ("シード", "最初のプロンプトのシード（プロンプトごとに +1）"),
    "out_dir": ("出力フォルダ", "画像と manifest.json の保存先"),
    "spectrum": (
        "Spectrum",
        "ほとんどのステップで 42 ブロックすべてを実行する代わりに、最後のブロックの出力を予測",
    ),
    "dave": (
        "DAVE",
        "同一プロンプトの多様性: 最初の数ステップの間、ブロック群で画像トークンのチャネルごとの"
        "平均を弱めます。'' = オフ、'auto' = 内蔵の MOSS マスク、'lo-hi' = lo..hi 番ブロック、"
        "または dave_alpha.npz のパス",
    ),
    "dave_strength": (
        "DAVE 強度",
        "群内のブロックで取り除く平均の割合（0 = オフ、1 = すべて）",
    ),
    "cns": (
        "CNS",
        "er_sde 専用: サンプラーが注入するノイズを CNS で再着色します。'auto' = バリアントと "
        "cfg に合わせて較正した内蔵 γ、または npz のパス、'' = オフ。ComfyUI の er_sde_cns と同じです",
    ),
    "ref_image": (
        "参照画像",
        "i2i の参照: 画像1枚または画像フォルダ（画像 × プロンプトごとに1ジョブ）。"
        "出力は参照画像のアスペクト比を保ち、面積の上限は 幅 × 高さ。LoRA に記録された "
        "ref_timestep / ref_isolate が適用されます",
    ),
    "ref_scale": (
        "参照画像の縮小",
        "ref_image 指定時: 参照潜在変数の辺 = 出力の辺 ÷ この値、16 の倍数。LoRA を"
        "学習したキャッシュと一致させてください（sanitize と colorize は 1、colorize v2 は 2）",
    ),
    "ref_vision": (
        "参照画像をビジョンタワーへ",
        "ref_image 指定時: 参照画像がビジョンタワー経由でプロンプトの前にテキストエンコーダ"
        "にも入ります（colorize v2 キャッシュのページのエンコードと同じ）。このときプロンプトを"
        "空にするとページだけを入力します",
    ),
}
