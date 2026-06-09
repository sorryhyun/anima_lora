"""Japanese strings for the Anima LoRA GUI."""

from __future__ import annotations

STRINGS: dict[str, str] = {
    # Window / tabs
    "window_title": "Anima LoRA",
    "tab_config": "学習設定",
    "tab_ip_adapter": "IP-Adapter",
    "tab_easycontrol": "EasyControl",
    "tab_spd": "SPD",
    "tab_turbo": "Turbo",
    "tab_methods": "手法",
    "tab_images": "データセット",
    "tab_merge": "マージ",
    "tab_queue": "キュー状況",
    "tab_preprocess": "前処理",
    "tab_tensorboard": "TensorBoard",
    # PreprocessingTab
    "preprocess_intro": (
        "キャプションのシャッフルやテキストバブルのマスキングを設定し、"
        "各ステップを個別に実行できます。学習設定タブの「学習」ボタンは、"
        "キャッシュが存在しない場合にデフォルト設定で前処理を自動実行します。"
        "このタブは設定の調整や個別ステップの再実行に使用します。"
    ),
    "preprocess_image_prep": "画像前処理 (リサイズ / フィルター)",
    "preprocess_source_image_dir": "ソース画像フォルダー:",
    "preprocess_source_image_dir_tip": (
        "学習用元画像が入ったフォルダー (.txt キャプションサイドカーを含む)。"
        "キャッシュステップで post_image_dataset/ にリサイズされます。"
        "configs/preprocess.toml に保存されます。"
    ),
    "preprocess_drop_lowres": "低解像度画像を除外",
    "preprocess_drop_lowres_tip": (
        "下のピクセル閾値を下回るソース画像をスキップし、"
        "リサイズ / VAE / テキストキャッシュに含まれないようにします。"
        "チェックを外すとサイズに関わらずすべての画像を保持します。"
    ),
    "preprocess_min_pixels": "最小ピクセル数 (フィルター閾値):",
    "preprocess_min_pixels_tip": (
        "低解像度フィルターのピクセル数閾値。500000 = 0.5MP。"
        "「低解像度画像を除外」がオフの場合は無視されます。"
    ),
    "preprocess_text_caching": "キャッシュ (VAE + テキスト)",
    "preprocess_caption_shuffle_variants": "キャプションあたりのシャッフルバリアント数 (N):",
    "preprocess_caption_shuffle_variants_tip": (
        "1枚の画像につきNバリアントのキャプションを生成します。v0はオリジナル; "
        "v1..v(N-1)はスマートシャッフルされ、タグドロップアウト > 0 の場合は "
        "プレフィックス以外のタグが独立してドロップされます。"
        "use_shuffled_caption_variants=true の場合、データローダーは20%の確率でv0を、"
        "それ以外ではv1..v(N-1)を均一にサンプリングします。"
        "0 に設定するとオリジナルキャプション1件のみをキャッシュします。"
    ),
    "preprocess_caption_tag_dropout_rate": "タグドロップアウト率 (0.0–1.0):",
    "preprocess_caption_tag_dropout_rate_tip": (
        "v1..v(N-1)に適用されるタグごとのドロップアウト確率。"
        "最初の @artist マーカー以前のタグはドロップされません。"
        "シャッフルバリアント ≤ 0 の場合は無視されます。"
    ),
    "preprocess_run_te": "キャッシュ実行 (VAE + テキスト)",
    "preprocess_masking_sam": "SAM3 マスキング (テキストバブル)",
    "preprocess_masking_mit": "MIT マスキング (漫画テキスト)",
    "preprocess_sam_prompts": "SAM プロンプト (1行1件):",
    "preprocess_sam_prompts_tip": (
        "SAM3 が検索するテキストプロンプト。1行1件。"
        "デフォルトは 'speech bubble' と 'text bubble'。"
    ),
    "preprocess_sam_focus_prompts": "SAM フォーカスプロンプト (1行1件):",
    "preprocess_sam_focus_prompts_tip": (
        "逆極性: 残したい被写体を指定します。設定すると、マスクはその被写体のみを"
        "学習対象とし、それ以外はすべて無視されます (例: 'girl' を指定すると背景全体が"
        "無視されます)。上のプロンプトと合成され、最終的な学習領域はフォーカスした"
        "被写体から無視領域を除いた部分になります。空欄にするとデフォルトの"
        "無視専用の動作になります。"
    ),
    "preprocess_sam_rule": "マスクルール",
    "preprocess_sam_add_rule": "+ ルール追加",
    "preprocess_sam_add_rule_tip": (
        "マスクルールをもう一つ追加します。各ルールはパスパターンで画像の"
        "サブセットを対象とし、パターンが一致するルールは互いに合成されます。"
    ),
    "preprocess_sam_remove_rule": "ルール削除",
    "preprocess_sam_rule_path_pattern": "パスパターン (このルール):",
    "preprocess_sam_rule_path_pattern_tip": (
        "このルールを適用する画像を指定します — データセットルート基準の各画像"
        "パスに対する fnmatch グロブ ('|' で OR 結合)。例: 'character_a/*'。"
        "空欄または '*' はすべての画像にマッチするキャッチオール規則です。"
    ),
    "preprocess_sam_threshold": "SAM しきい値 (0.0–1.0):",
    "preprocess_sam_threshold_tip": (
        "SAM3 の検出結果を採用するための最小信頼度。"
        "低いほど多くのマスクを生成 (誤検出が増える可能性あり)、"
        "高いほど厳しくなります。デフォルト 0.5。"
    ),
    "preprocess_dilate": "膨張 (px):",
    "preprocess_dilate_tip": (
        "バイナリマスクに適用するピクセル膨張量。"
        "大きい値ほどマスクのエッジが外側に広がります。"
        "デフォルト 5。0 で無効化。"
    ),
    "preprocess_mit_threshold": "MIT テキストしきい値 (0.0–1.0):",
    "preprocess_mit_threshold_tip": (
        "MIT/ComicTextDetector テキストセグメンタの信頼度しきい値。デフォルト 0.8。"
    ),
    "preprocess_mask_path_pattern": "マスクパスフィルター:",
    "preprocess_mask_path_pattern_tip": (
        "マスク対象のリサイズ済み画像を絞り込む fnmatch glob パターン。"
        "post_image_dataset/resized を基準とした各パスに対してマッチングされます。"
        "SAM と MIT の両方に適用されます。学習用 path_pattern と同じ文法: "
        "'*'(または空欄) で全件マスク; 'char_a/*' で 1 サブフォルダー; "
        "'char_a/*|char_b/*' で OR 結合。"
    ),
    "preprocess_run_mask": "マスキング実行",
    "preprocess_run_sam_mask": "SAM マスキング実行",
    "preprocess_run_sam_mask_tip": (
        "マスク生成時に SAM3 バブルセグメンテーションを実行します。"
        "チェックを外すと SAM をスキップし、MIT のみ (または有効な他のバックエンド) を使用します。"
    ),
    "preprocess_run_mit_mask": "MIT マスキング実行",
    "preprocess_run_mit_mask_tip": (
        "マスク生成時に MIT/ComicTextDetector テキストセグメンテーションを実行します。"
        "チェックを外すと MIT をスキップし、SAM のみを使用します。"
    ),
    "preprocess_mask_nothing_enabled": (
        "SAM または MIT のどちらか一方を有効にしてください。"
    ),
    "preprocess_status_resized": "リサイズ済み画像: {n}",
    "preprocess_status_caches": "キャッシュ — 潜在変数: {lat}, テキスト: {te}, PE: {pe}",
    "preprocess_status_masks": "マスク: {masks}",
    "preprocess_status_no_resized": "リサイズ済み画像がありません — まず学習設定タブの前処理を実行してください。",
    "preprocess_log_placeholder": "前処理の出力がここに表示されます...",
    "preprocess_save_settings": "保存",
    "preprocess_save_settings_tip": "設定を保存します (configs/preprocess.toml + configs/sam_mask.yaml + GUI設定に書き込みます)。",
    "preprocess_settings_saved": "前処理設定を保存しました。",
    "preprocess_invalid_float": "{field} の値が不正です: {value}",
    "preprocess_already_running": "前処理ステップが既に実行中です。",
    # ConfigTab
    "preset": "プリセット:",
    "save": "保存",
    "save_dirty_tooltip": "未保存の編集があります。「保存」をクリックしてバリアントファイルに書き込んでください (学習/前処理実行時にスキップした場合は自動保存されます)。",
    "train": "学習",
    "queue": "キューに追加",
    "queue_tooltip": "現在のバリアントをこのタブに紐付けずにデーモンキューに追加します。続けて別のバリアントをキューに追加できます。",
    "test": "テスト",
    "stop": "停止",
    "log_placeholder": "学習の出力がここに表示されます...",
    "from_base": "base.toml から",
    "saved": "保存済み",
    "saved_file": "{name} を保存しました",
    "invalid_toml": "TOML が不正です",
    "config_bad_keys_header": "不明なデータセットキー — これらを削除するまで学習は失敗します:",
    "config_remove_keys_btn": "削除",
    "config_remove_keys_confirm": "これら {n} 個の古いキーを設定ファイルから削除しますか?\n\n{keys}",
    "config_remove_keys_none": "削除されたキーはありません (ディスク上の該当行が変更された可能性があります)。",
    "error": "エラー",
    "accelerate_not_found": "PATH に accelerate が見つかりません",
    "preprocess": "前処理",
    "preprocess_required": "学習前に前処理を実行してください。",
    "preprocess_existing_caches_title": "既存のキャッシュを再利用します",
    "preprocess_existing_caches_body": (
        "次のディレクトリにキャッシュファイルが既に存在します:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "前処理はこれらを再利用します — 削除・再生成はされません。"
        "不足しているエントリのみ処理されます。\n\n"
        "完全な再構築を強制したい場合 (例: キャプション編集後やトークナイザー設定変更後) は、"
        "キャンセルしてキャッシュディレクトリを削除してから再実行してください。"
    ),
    "preprocess_cache_count_latents": "{n} 件の VAE 潜在変数 (.npz)",
    "preprocess_cache_count_te": "{n} 件のテキスト埋め込み (_te.safetensors)",
    "preprocess_cache_count_pe": "{n} 件の PE 特徴量 (_pe.safetensors)",
    "train_using_cache_title": "キャッシュ済みデータセットを使用しますか?",
    "train_using_cache_body": (
        "次の場所に前処理済みデータセットキャッシュが存在します:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "学習はこのキャッシュをそのまま再利用します。新しい画像を追加したり、"
        "キャプションを編集した場合は、キャンセルして前処理を実行してください。\n\n"
        "既存のキャッシュで続行しますか?"
    ),
    "train_autopreprocess_log": (
        "前処理済みキャッシュが見つかりません — 前処理を実行してから自動的に学習を開始します。\n"
    ),
    "train_preprocessing": "前処理中…",
    "no_lora_for_test": "output/ckpt/ に LoRA が見つかりません。先に学習を実行してください。",
    "test_output_title": "最新のテスト出力",
    "test_output_empty": "output/tests/ が空です。",
    "finished": "--- 完了 (終了コード {code}) ---",
    "starting": "起動中… (torch / accelerate を読み込んでいます)",
    "queue_submitting": "{variant} を学習デーモンキューに追加中…",
    "queue_added_train": "{variant} を学習ジョブ {job_id} としてキューに追加しました。\n",
    "queue_added_preprocess": "{variant} を前処理ジョブ {job_id} としてキューに追加しました。完了後に学習が連続して実行されます。\n",
    "queue_refresh": "更新",
    "queue_stop_selected": "選択項目を停止",
    "queue_copy_output": "出力をコピー",
    "queue_status": "実行中/待機中 {live} 件 / 合計 {total} 件",
    "queue_daemon_unavailable": "デーモンに接続できません",
    "queue_detail_placeholder": "キュー項目を選択すると詳細が表示されます。",
    "queue_log_placeholder": "選択したジョブの出力がここに表示されます...",
    "queue_log_missing": "(まだ出力ログがありません。)",
    "queue_log_read_failed": "(出力ログを読み込めませんでした: {err})",
    "queue_log_truncated": "--- 最後の {mb} MB の出力を表示中 ---\n",
    "queue_detail_id": "id: {job_id}",
    "queue_detail_state": "状態: {state}",
    "queue_detail_kind": "種別: {kind}",
    "queue_detail_method": "対象: {method}",
    "queue_detail_submitted": "追加日時: {time}",
    "queue_detail_started": "開始日時: {time}",
    "queue_detail_ended": "終了日時: {time}",
    "queue_detail_from_chain": "from_chain: true",
    "queue_detail_chain": "連続学習: {method}",
    "queue_detail_chained_id": "連結ジョブ: {job_id}",
    "queue_detail_pid": "pid: {pid}",
    "queue_detail_error": "エラー: {error}",
    "queue_detail_status_detail": "詳細: {detail}",
    "queue_detail_config": "設定スナップショット: {path}",
    "queue_detail_stdout": "stdout: {path}",
    "daemon_job_failed": "--- Job {job_id} {state}: {error} ---",
    "daemon_error_cause": "↳ 推定される原因: {summary}",
    "update_success_title": "更新完了",
    "update_success_message": (
        "anima_lora が {v} に更新されました。\n\n"
        "GUI を閉じて再起動すると新しいコードが読み込まれます。"
    ),
    "update_success_badge": "更新済み → {v} (適用するには再起動してください)",
    "update_dryrun_done_title": "ドライラン完了",
    "update_dryrun_done_message": (
        "ドライランが完了しました — ファイルは書き込まれていません。"
        "ログを確認して実際の更新内容を確認してください。"
    ),
    "update_failed_title": "更新失敗",
    "update_failed_message": (
        "更新がコード {code} で終了しました。"
        "ログを確認してください。作業ツリーが一部変更されている可能性があります。"
    ),
    "resume_checkpoint_title": "学習を再開しますか?",
    "resume_checkpoint_question": (
        "ステップ {step} で再開可能なチェックポイントが見つかりました。\n\n"
        "• はい — ステップ {step} から学習を再開\n"
        "• いいえ — チェックポイントを破棄して最初から開始\n"
        "• キャンセル — 学習を開始しない"
    ),
    "resume_checkpoint_delete_failed": "古いチェックポイント状態を削除できませんでした:\n{error}",
    "locked_by_preset": "プリセットによりロックされています (このVRAMプロファイルではパフォーマンス設定は固定されています)",
    "lora_variants": "LoRA バリアント",
    "variant": "バリアント:",
    "apply_variant": "適用",
    "apply_variant_tooltip": "このバリアントのプリセット値をフォームに反映します。「保存」をクリックするまで保存されません。",
    "show_guide": "ガイド",
    "show_guide_tooltip": "バリアントガイドと適用時の注意を右パネルに表示します。",
    "click_field_for_help": "フィールドラベルをクリックすると説明が表示されます。",
    "no_help_available": "このフィールドのヘルプはありません。",
    "extra_args_toggle": "+ 追加引数",
    "extra_args_placeholder": "フォームにないフィールドを TOML 形式で記述してください。例:\nmy_new_flag = true\nsome_value = 5e-5",
    "extra_args_tooltip": "フォームに表示されていない設定キーを追加します。保存時に TOML として解析され、現在のバリアントファイルにマージされます。フォームが再読み込みされ、新しいキーがウィジェットとして表示されます。同一キーがフォームと両方に存在する場合、こちらが優先されます。",
    "new_variant": "+ 新規",
    "new_variant_tooltip": "configs/gui-methods/custom/<name>.toml に新しいカスタムバリアントを作成します。",
    "new_variant_prompt": "新しいバリアントの名前 (configs/gui-methods/custom/<name>.toml に保存されます)。\n英数字、_、- のみ使用できます。",
    "new_variant_invalid": "名前が不正です。英数字、_、- のみ使用してください。",
    "new_variant_exists": "バリアント '{name}' は既に存在します。",
    "basic_section": "基本",
    "advanced_section": "詳細 (クリックして展開)",
    # AdapterTab (IP-Adapter / EasyControl)
    "adapter_source_dir": "ソースデータセット:",
    "adapter_cache_dir": "キャッシュディレクトリ:",
    "adapter_n_pairs": "{n} 枚の画像 / {c} 件のキャプションペア",
    "adapter_n_caches": "{n} 件キャッシュ済み",
    "adapter_preprocess": "前処理 (リサイズ + VAE + テキスト)",
    "adapter_preprocess_pe": "前処理 (リサイズ + VAE + テキスト + PE)",
    "adapter_train": "学習",
    "adapter_stop": "停止",
    # SPD / Turbo 蒸留設定タブ (gui/tabs/distill_tab.py)
    "distill_general_section": "全般",
    "distill_job_running": "このタブでは既にジョブが実行中です。",
    "distill_config_missing": "設定ファイルを読み込めませんでした: {err}",
    "adapter_log_placeholder": "実行出力がここに表示されます...",
    "adapter_no_dataset": "ソースデータセットのディレクトリが存在しません。ディレクトリを作成して画像とキャプションのペアを配置してください。",
    "adapter_open_dir": "ディレクトリを開く",
    "n_images": "{n} 枚の画像",
    # ImageViewerTab
    "directory": "ディレクトリ:",
    "dataset_reload": "再読み込み",
    "dataset_reload_tooltip": "現在のディレクトリを再スキャンして画像リストと選択を更新します。",
    "dataset_open_dir": "開く",
    "dataset_open_dir_tooltip": "現在のディレクトリをシステムのファイルマネージャーで開きます。",
    "dataset_add_dir": "ディレクトリを追加…",
    "dataset_add_dir_tooltip": "別のディレクトリを選択してこのセッションのドロップダウンに追加します。",
    "dataset_add_dir_picker": "追加するディレクトリを選択",
    "dataset_add_dir_already": "ディレクトリ '{name}' は既にリストにあります。",
    "dataset_search_placeholder": "ファイル名を検索…",
    "dataset_sort_asc_tooltip": "A→Z 順 (クリックで逆順)",
    "dataset_sort_desc_tooltip": "Z→A 順 (クリックで逆順)",
    "dataset_mask_overlay": "マスクオーバーレイを表示",
    "dataset_view_list_tooltip": "フラットリスト表示 (クリックでツリー表示に切り替え)",
    "dataset_view_tree_tooltip": "フォルダーツリー表示 (クリックでリスト表示に切り替え)",
    "n_images_filtered": "{shown} / {total} 枚の画像",
    "caption": "キャプション:",
    "no_caption": "(キャプションなし)",
    "caption_save": "保存",
    "caption_revert": "元に戻す",
    "caption_versions": "履歴…",
    "caption_dirty_marker": " *",
    "caption_diff_stats": "(+{add} / −{rem})",
    "caption_diff_clean": "(変更なし)",
    "caption_save_failed": "キャプションの保存に失敗しました: {err}",
    "caption_unsaved_title": "未保存のキャプション",
    "caption_unsaved_body": "キャプションに未保存の編集があります。切り替える前に保存しますか?",
    "caption_versions_title": "キャプション履歴 — {name}",
    "caption_versions_empty": "(過去のバージョンなし)",
    "caption_versions_restore": "選択したバージョンを復元",
    "caption_versions_close": "閉じる",
    "caption_no_history": "このキャプションにはまだ履歴がありません。",
    "caption_guideline_html": (
        "<b>順序:</b> レーティング → カウント → キャラクター (シリーズ) → シリーズ → "
        "<span style='color:#c9a227;'>@artist</span> → コンテンツタグ。"
        "リージョンごとのサブセクション: 前のタグを <code>.</code> で終了し、"
        "次を <span style='color:#5e8eb0;'>On the&nbsp;…,</span> "
        "または <span style='color:#5e8eb0;'>In the&nbsp;…,</span> で開始します。"
        "最初の <code>@artist</code> 以前のタグは固定されます;"
        "それ以降はセクション内でシャッフルされます。"
        "<b>アーティストがいない場合は</b> "
        "<span style='color:#c9a227;'>@no-artist</span> をプレースホルダーとして使用してください — "
        "同じようにシャッフル境界を固定し、トークン化前に除去されるためモデルには届きません。"
    ),
    # Language
    "language": "言語:",
    # Guidebook
    "guidebook": "📖 ガイドブック",
    "guidebook_tooltip": "日本語総合ガイドを開きます (docs/guidelines/ガイドブック.md)",
    "guidebook_missing": "{path} にガイドが見つかりません",
    "guidebook_open_external": "システムビューアで開く",
    "guidebook_close": "閉じる",
    # EasyControl アダプターガイド (自作コントロールタスク)
    "adapter_guide": "📘 アダプターガイド",
    "adapter_guide_tooltip": "独自の EasyControl アダプターの作り方 (easycontrol_adapters/ADAPTER_GUIDE.md)",
    "easycontrol_descriptor_note": "このコントロールタスクは、複数テーブル構造を持つ独立したディスクリプターで、左側で生の TOML として編集します:<br><br>• <b>name</b> — 出力スラッグ; 派生するすべてのキャッシュ/出力パスを再ルーティングします。<br>• <code>[staging]</code> — 条件ツリーを実体化するデータ生成ステップ。<br>• <code>[preprocess]</code> — ステージング済みツリーへの VAE/TE キャッシュ設定。<br>• <code>[training]</code> — ベース EasyControl 手法にマージされるオーバーライド。<br>• <code>[general]</code> / <code>[[datasets]]</code> — train.py が読み込むデータセット設計図。<br>• <code>[variant]</code> — このドロップダウン項目の GUI メタデータ。<br><br><b>前処理</b>ボタンは条件ツリーを合成してキャッシュします; <b>学習</b>はこのディスクリプターの <code>[training]</code> オーバーライドをマージしてベース EasyControl 手法を学習します。どちらも GUI を閉じても続行されます。",
    "easycontrol_descriptor_form_header": "ディスクリプター <b>{path}</b> を編集中。以下の設定テーブルはフォームとして編集します; 保存時に変更された値を書き戻し、コメントと <code>[[datasets]]</code> 設計図は保持されます。設計図と <code>[variant]</code> メタデータはここには表示されません — それらはファイルを直接編集してください。フィールド名をクリックするとヘルプが表示されます。",
    "ec_desc_group_top": "ディスクリプター",
    # Top-bar buttons (models / update / report issue)
    "models_btn": "モデル",
    "models_btn_tooltip": "モデルチェックポイントをダウンロードまたは再ダウンロードします (Anima ベース、SAM3、MIT、IP-Adapter エンコーダー)",
    "update_btn": "更新",
    "update_btn_tooltip": "GitHub から最新の anima_lora リリースを取得して uv sync を実行します",
    "update_btn_available": "更新 ●",
    "update_btn_available_tooltip": "新しいリリース {v} があります — クリックしてリリースノートを確認",
    "report_issue": "問題を報告",
    "report_issue_tooltip": "ブラウザで GitHub Issue トラッカーを開きます",
    "experimental_features": "🧪 実験的機能",
    "experimental_features_tooltip": "Postfix および IP-Adapter / EasyControl タブを開きます (画像条件付け手法)",
    "experimental_features_title": "実験的機能",
    # Models dialog
    "models_title": "モデルのダウンロード",
    "models_intro": "以下からモデルグループを選択するか、「すべてダウンロード」で標準セット "
    "(Anima + SAM3 + MIT + PE) をダウンロードします。ファイルは models/ に保存されます。",
    "models_download_all": "すべてダウンロード (Anima + SAM3 + MIT + PE)",
    "models_download": "ダウンロード",
    "models_redownload": "再ダウンロード",
    "models_installed": "✓ インストール済み",
    "models_missing": "✗ 未インストール",
    "model_anima": "Anima — DiT + テキストエンコーダー + VAE",
    "model_sam3": "SAM3 — テキストバブルマスキング",
    "model_mit": "MIT — 漫画テキストマスキング",
    "model_pe": "PE-Core-L14-336 — IP-Adapter ビジョンエンコーダー",
    # HuggingFace 認証 (モデルダイアログ)
    "models_hf_token_placeholder": "HuggingFace トークンを貼り付けてください (hf_…)",
    "models_hf_authenticate": "認証",
    "models_hf_token_hint": "ゲート付き/レート制限のあるダウンロード(SAM3 など)に必要です。"
    '<a href="https://huggingface.co/settings/tokens">'
    "huggingface.co/settings/tokens</a> でトークンを作成し、"
    '<a href="https://huggingface.co/facebook/sam3">huggingface.co/facebook/sam3</a> で SAM3 のアクセスを申請してください。',
    "models_hf_token_present": "✓ HuggingFace トークンは既に保存されています。",
    "models_hf_not_authenticated": "未認証 — トークンを貼り付けてゲート付きダウンロードを有効にしてください。",
    "models_hf_token_empty": "先にトークンを貼り付けてください。",
    "models_hf_authenticating": "認証中…",
    "models_hf_logged_in": "✓ {name} としてログインしました。",
    "models_hf_login_failed": "認証に失敗しました: {err}",
    # Update dialog
    "update_title": "anima_lora の更新",
    "update_warning": "更新により GitHub から最新リリースが取得され、作業ツリーが上書きされます "
    "(datasets、output/、models/ は保持されます)。configs/methods/ と configs/gui-methods/ については、"
    "自分の編集を維持するか上流で上書きするかを選択できます (バックアップが先に作成されます)。"
    "「ドライラン」で変更内容をプレビューできます。",
    "update_dry_run": "ドライラン",
    "update_run": "更新を実行",
    "update_run_keep": "更新 — 自分の設定を維持",
    "update_run_overwrite": "更新 — 設定を上書き (バックアップあり)",
    "update_confirm": "anima_lora のソースファイルが書き換えられます。続行しますか?",
    "update_check_now": "今すぐ確認",
    "update_view_release": "GitHub で表示",
    "update_current_version": "現在: {v}",
    "update_latest_version": "最新: {v}",
    "update_no_baseline": "不明 (マニフェストなし)",
    "update_status_checking": "確認中…",
    "update_status_uptodate": "✓ 最新です",
    "update_status_available": "● 更新があります",
    "update_status_unknown": "? 比較不可 (ローカルマニフェストなし)",
    "update_status_failed": "✗ 確認失敗",
    "update_release_notes": "リリースノート:",
    "update_no_release_notes": "(このリリースには説明がありません)",
    "update_check_error": "GitHub に到達できませんでした: {err}",
    # MergeTab
    "n_files": "{n} ファイル",
    "merge_no_adapter": "アダプターが見つかりません",
    "merge_no_adapter_msg": "アダプターが選択されていないか、ファイルが存在しません。",
    "merge_no_selection": "リストからチェックポイントを選択してスキャンしてください。",
    "merge_verdict_ready": "✓ ベイク可能",
    "merge_verdict_hydra": "✗ HydraLoRA moe — レイヤーローカルルーターはベイクできません",
    "merge_verdict_postfix_only": "✗ Postfix/prefix のみ — 重み差分ではありません",
    "merge_verdict_unknown": "? 認識できるアダプターキーがありません",
    "merge_options": "マージオプション",
    "merge_base_dit": "ベース DiT:",
    "merge_multiplier": "乗数:",
    "merge_multiplier_tip": "ベイクする LoRA の強度 (1.0 = フル強度)。",
    "merge_dtype": "保存データ型:",
    "merge_out": "出力:",
    "merge_out_placeholder": "(自動: <adapter>_merged.safetensors)",
    "merge_allow_partial": "部分マージを許可 (Hydra / postfix キーをドロップ)",
    "merge_allow_partial_tip": "アダプターにベイクできないコンポーネントが含まれていても続行します。ドロップされたコンポーネントはマージ済み DiT には含まれません。",
    "merge_button": "DiT にマージ",
    "merge_log_placeholder": "マージの出力がここに表示されます...",
    "merge_pick_dir": "アダプターディレクトリを選択",
    "merge_pick_file": "アダプター .safetensors を選択",
    "merge_pick_dit": "ベース DiT .safetensors を選択",
    "merge_pick_out": "マージ済み DiT を名前を付けて保存...",
    "browse": "参照…",
    # Multi-scale target_res tiers
    "target_res_danger_tooltip": "高コストなティア：{edge}px は画像あたり約 {tokens} トークンを使用し、コンパイル済みブロックグラフを 1 つ追加します（コンパイルが遅くなり、VRAM が増加）。この解像度が本当に必要な場合のみ有効にしてください。",
    # TensorBoard panel
    "tb_panel_title": "TensorBoard 実行一覧",
    "tb_open": "TensorBoard を開く",
    "tb_stop": "サーバーを停止",
    "tb_remove": "削除",
    "tb_view": "表示",
    "tb_view_tip": "この実行のみを TensorBoard で開きます。",
    "tb_no_runs": "まだ実行記録がありません。学習を開始するとリストが表示されます。",
    "tb_status_running": "ポート {port} で実行中",
    "tb_status_stopped": "",
    "tb_not_installed": "tensorboard がインストールされていません。実行: pip install tensorboard",
    "tb_current_run_label": "（現在）",
    "tb_open_current": "現在の学習を表示",
    "tb_open_current_tip": "進行中の学習実行のみを TensorBoard で開きます。",
    "tb_open_current_idle_tip": "学習が進行中のときに使用できます。",
    "tb_appear_hint": "実行がリストに表示されない場合は、TensorBoard の再読み込み（アップデート）ボタンを押してみてください。",
}
