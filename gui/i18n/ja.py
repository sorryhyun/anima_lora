"""Japanese strings for the Anima LoRA GUI."""

from __future__ import annotations

STRINGS: dict[str, str] = {
    # Window / tabs
    "window_title": "Anima LoRA",
    "tab_config": "学習設定",
    "tab_easycontrol": "EasyControl",
    "tab_turbo": "Turbo",
    "tab_experimental": "実験機能",
    "tab_merge": "マージ",
    "tab_queue": "キュー状況",
    "tab_anime_tools": "anime_tools",
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
        "選択中の GUI method のベース元画像ルートです (デフォルトは configs/preprocess.toml; "
        "編集内容は該当 variant に保存されます)。実行時に path_scope がこの上に付加されるため、"
        "ここに表示されるのはスコープ後の最終パスではなくスコープなしのルートです。"
        "ファイルの保存先を変えずにツリーの一部だけ前処理する場合は、下の前処理パスフィルターを使用してください。"
    ),
    "preprocess_path_pattern": "前処理パスフィルター:",
    "preprocess_path_pattern_tip": (
        "path_scope が先に実効ソース画像ルートを決めます。"
        "たとえば path_scope=data_group1 の場合、前処理ルートは "
        "image_dataset/data_group1 になります。このフィルターはそのルートからの"
        "相対パスに適用されます。'*'(または空欄) は全件、'1/*' は "
        "data_group1/1 のみ、'1/*|2/*' は両方のサブフォルダーを処理します。"
    ),
    "preprocess_target_res": "解像度ティア (target_res):",
    "preprocess_freefit_max_ratio": "最大アスペクト比:",
    "preprocess_freefit_max_ratio_tip": (
        "フリーフィット用クランプ: 1:R / R:1 を超えるアスペクト比の画像は、"
        "上で設定したクロップ位置に従いカバークロップされます。"
        "デフォルト 4.0 は旧バケットテーブルの最長の縦横比に合わせており、"
        "1:5 / 1:6 など極端な比率の入力をブロックします。"
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
    "preprocess_caption_editing": "キャプション編集",
    "preprocess_caption_correct_order": "キャプション順序補正",
    "preprocess_caption_correct_order_tip": (
        "補正済み .txt キャプションをリサイズ画像の隣に保存し、"
        "テキストエンコーダーキャッシュに使用します。元のソースキャプションは変更しません。"
    ),
    "preprocess_caption_insert_no_artist": "@no-artist を挿入",
    "preprocess_caption_insert_no_artist_tip": (
        "キャプション順序補正が有効で、アーティストマーカーがない場合に "
        "artist 位置へ @no-artist を挿入します。トリガー語が artist 位置にある場合は"
        "トリガーが優先され、@no-artist は挿入されません。"
    ),
    "preprocess_caption_trigger_word": "トリガー語:",
    "preprocess_caption_trigger_word_tip": (
        "補正済みキャプションに配置する任意のトリガータグです。先頭固定がオフの場合、"
        "既存アーティストタグより前の artist 位置に置かれます。シャッフル/ドロップアウト"
        "保護の artist マーカーとして使う場合は @ を含めてください。"
        "トリガー語のアンダースコアは保持されます。"
    ),
    "preprocess_caption_trigger_at_front": "トリガーを先頭に固定",
    "preprocess_caption_trigger_at_front_tip": (
        "トリガー語をキャプションの最初に置きます。このモードでは @no-artist 挿入は、"
        "既存アーティストタグの有無と no-artist オプションに従って別に動作します。"
    ),
    "preprocess_run_te": "キャッシュ実行 (VAE + テキスト)",
    "preprocess_run_pe": "PE キャッシュ実行",
    "preprocess_add_to_queue": "キューに追加",
    "preprocess_queued": "{label} をキューに追加しました (ジョブ {job_id}) — キュータブで確認できます。",
    "preprocess_invalid_stage": "{stage}: {err}",
    "preprocess_status_resized": "リサイズ済み画像: {n}",
    "preprocess_status_caches": "キャッシュ — 潜在変数: {lat}, テキスト: {te}, PE: {pe}",
    "preprocess_status_masks": "マスク: {masks}",
    "preprocess_status_no_resized": "リサイズ済み画像がありません。",
    "preprocess_open_dataset_dir": "cacheフォルダを開く",
    "preprocess_open_dataset_dir_tooltip": "post_image_dataset/ フォルダ（リサイズ済み画像 + キャッシュ）をファイルマネージャーで開きます。",
    "preprocess_status_export_stale": "anime_tools に最後の Export より新しい編集があります",
    "preprocess_resize_preview": "リサイズプレビュー",
    "preprocess_resize_preview_tooltip": "元画像ごとに、下の設定でリサイズが選ぶ bucket と tier、そしてクロップ領域を表示します。元ファイルは変更しません。",
    "preprocess_resize_preview_title": "リサイズプレビュー",
    "preprocess_resize_preview_refresh": "更新",
    "preprocess_resize_preview_col_image": "画像",
    "preprocess_resize_preview_col_source": "元サイズ",
    "preprocess_resize_preview_col_bucket": "Bucket",
    "preprocess_resize_preview_col_tier": "Tier",
    "preprocess_resize_preview_col_kept": "保持",
    "preprocess_resize_preview_summary": "画像 {n} 枚、bucket {buckets} 個",
    "preprocess_resize_preview_skipped": "{n} 枚除外",
    "preprocess_resize_preview_unreadable": "{n} 枚読み込み不可",
    "preprocess_resize_preview_empty": "{path} に画像がありません",
    "anime_tools_failed": "anime_tools を起動できません: {err}",
    "anime_tools_root_warning": "パネルに保存されたルートが post_image_dataset/ 内を指しています（{roots}）。パネルのステージがトレーナーのツリーを上書きします。パネルの ⚙ 設定（または {path}）で空にしてください。",
    "anime_tools_starting": "anime_tools を起動中…（ログ: {log}）",
    "anime_tools_timeout": "anime_tools が起動しませんでした。ログを確認してください: {log}",
    "anime_tools_no_webengine": "Qt WebEngine がインストールされていないため、ここにパネルを表示できません。ブラウザで {url} を開いてください。",
    "preprocess_clear_scope_cache": "現在scopeのキャッシュ削除",
    "preprocess_clear_scope_cache_tooltip": "現在の path_scope に対応するリサイズ画像と LoRA キャッシュフォルダーを削除します。",
    "preprocess_clear_scope_cache_all_scope": "scopeなし / 全体",
    "preprocess_clear_scope_cache_empty": "削除するリサイズ画像または LoRA キャッシュファイルがありません。",
    "preprocess_clear_scope_cache_outside_root": "プロジェクトフォルダー外のパスは GUI から削除しません:\n{path}",
    "preprocess_clear_scope_cache_confirm": (
        "現在の scope の前処理ファイルを削除しますか?\n\n"
        "scope: {scope}\n"
        "resize: {resized}\n  {resized_count} ファイル\n"
        "lora: {lora}\n  {lora_count} ファイル\n\n"
        "削除後は前処理を再実行して再生成してください。"
    ),
    "preprocess_clear_scope_cache_done": "前処理ファイル {count} 件を削除しました。",
    "preprocess_invalid_path_scope": "path_scope の値が不正です: {value}",
    "preprocess_log_placeholder": "前処理の出力がここに表示されます...",
    "preprocess_save_settings": "保存",
    "preprocess_save_settings_tip": "設定を選択中の GUI method プロファイルに保存します。マスキング実行時は現在のプロファイルのマスク設定がジョブに渡されます。",
    "preprocess_settings_saved": "前処理設定を保存しました。",
    "preprocess_invalid_float": "{field} の値が不正です: {value}",
    "preprocess_already_running": "前処理ステップが既に実行中です。",
    # ConfigTab
    "preset": "プリセット:",
    "save": "保存",
    "save_dirty_tooltip": "未保存の編集があります。「保存」をクリックしてバリアントファイルに書き込んでください (学習/前処理実行時にスキップした場合は自動保存されます)。",
    "train": "学習",
    "train_tooltip": "現在のバリアントを今すぐ学習します。ドロップダウンを開くと、今すぐ開始せずデーモンキューに追加できます。",
    "train_busy_use_queue": "すでにこのタブにジョブが紐付いています。Train のドロップダウンで別のジョブをキューに追加するか、先に現在のジョブを停止してください。",
    "queue": "キューに追加",
    "queue_tooltip": "現在のバリアントをこのタブに紐付けずにデーモンキューに追加します。続けて別のバリアントをキューに追加できます。",
    "queue_train_preprocess": "キューに追加: 学習 + 前処理",
    "queue_train_only": "キューに追加: 学習のみ",
    "queue_preprocess_only": "キューに追加: 前処理のみ",
    "test": "テスト",
    "stop": "停止",
    "log_placeholder": "学習の出力がここに表示されます...",
    "copy_log": "コピー",
    "copy_log_tooltip": "学習ログ全体をクリップボードにコピー",
    "gpu_probing": "GPU: 確認中…",
    "gpu_stat": "GPU{i}: {util}%  ·  {used}/{total} GiB  ·  {temp}°C",
    "copy_log_done": "コピーしました",
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
    "preprocess_required": "学習開始前に前処理が先に実行されます。",
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
        "前処理済みキャッシュが見つかりません — 学習開始前に前処理を先に実行します。\n"
    ),
    "train_preprocessing": "前処理中…",
    "no_lora_for_test": "output/ckpt/ に LoRA が見つかりません。先に学習を実行してください。",
    "test_output_title": "最新のテスト出力",
    "test_output_empty": "output/tests/ が空です。",
    "sample_output_title": "最新の学習サンプル",
    "sample_output_empty": "サンプルはまだありません — 学習が生成するにつれて出力ディレクトリの sample/ フォルダに表示されます。",
    "sample_prompt_edit_button": "サンプルプロンプトを編集…",
    "sample_prompt_dialog_title": "サンプルプロンプト",
    "sample_prompt_summary_none": "サンプルプロンプトなし",
    "sample_prompt_summary_count": "{n} 件のプロンプト · {first}",
    "finished": "--- 完了 (終了コード {code}) ---",
    "starting": "起動中… (torch / accelerate を読み込んでいます)",
    "queue_submitting": "{variant} を学習デーモンキューに追加中…",
    "queue_added_train": "{variant} を学習ジョブ {job_id} としてキューに追加しました。\n",
    "queue_added_preprocess": "{variant} を前処理ジョブ {job_id} としてキューに追加しました。完了後に学習が連続して実行されます。\n",
    "queue_refresh": "更新",
    "queue_start": "キューを開始",
    "queue_pause": "キューを一時停止",
    "queue_start_tooltip": "待機中のジョブ（キューのドロップダウンで追加したもの）を実行します。1 件ずつ処理します。",
    "queue_pause_tooltip": "キューを保留します — 実行中のジョブは続行しますが、「キューを開始」を押すまで次の待機ジョブは始まりません。",
    "queue_stop_selected": "選択項目を停止",
    "queue_copy_output": "出力をコピー",
    "queue_status": "実行中/待機中 {live} 件 / 合計 {total} 件",
    "queue_status_paused": "実行中/待機中 {live} 件 / 合計 {total} 件 — キュー一時停止中",
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
    "hardware_preset": "ハードウェア:",
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
    # Turbo 蒸留設定タブ (gui/tabs/distill_tab.py)
    "distill_general_section": "全般",
    "distill_job_running": "このタブでは既にジョブが実行中です。",
    "distill_config_missing": "設定ファイルを読み込めませんでした: {err}",
    # Soup パイプラインタブ (gui/tabs/soup_tab.py) — 実行専用フィールド。プール /
    # 投与量 / シード / ランクの値は下の [soup] セクションフォームで編集します。
    "soup_run_section": "実行",
    "soup_path_pattern": "パスパターン",
    "soup_path_pattern_tip": "ファインチューニング画像を選択する fnmatch グロブ。各画像のパス（サブセットの image_dir を基準とした相対パス）と照合します。'|' で複数パターンを区切ります（例: 'sincos/*' や 'art_a/*|art_b/*'）。必須です。",
    "soup_name": "出力名",
    "soup_name_tip": "出力スラッグ → output/ckpt/anima_soup_<name>.safetensors。空欄ならパスパターンから導出します（単純な '<dir>/*' は '<dir>' になります）。",
    "soup_ft_args": "ファインチューニング引数",
    "soup_ft_args_tip": "各ファインチューニング実行に渡す追加 CLI 引数。例: --network_dim 32 --max_train_epochs 8。",
    "soup_path_pattern_required": "Soup にはパスパターンが必要です（例: 'sincos/*' や 'art_a/*|art_b/*'）。",
    # Shared labels
    "directory": "ディレクトリ:",
    # Language
    "language": "言語:",
    # Settings dialog
    "settings_btn": "⚙ 設定",
    "settings_btn_tooltip": "アプリ設定 — 言語、環境設定、MCP サーバー登録",
    "settings_title": "設定",
    "settings_prefs_header": "環境設定",
    "settings_theme": "テーマ:",
    "settings_theme_tooltip": (
        "インターフェース全体のカラーテーマです。即時に反映され、設定画面を閉じると"
        "ウィンドウが再描画されて完全に適用されます。"
    ),
    "settings_font_size": "フォントサイズ:",
    "settings_font_size_tooltip": (
        "インターフェースフォントのポイントサイズです。即時に反映され、設定画面を"
        "閉じるとウィンドウが再描画され各パネルが再配置されます。既定値 10。"
    ),
    "settings_theme_dark": "ダーク",
    "settings_theme_light": "ライト",
    "settings_theme_sepia": "セピア",
    "settings_debug_mode": "デバッグモード",
    "settings_debug_mode_tooltip": (
        "学習デーモンを DEBUG レベルで記録し、ジョブが固まった原因をバグ報告に含め "
        "られるようにします。次にデーモンが起動したときに反映されます（アプリを閉じ "
        "るか、デーモンを停止してから開き直してください）。"
    ),
    "settings_debug_report_desc": (
        "無限ローディングで止まっていますか？ デバッグモードを有効にして症状を再現し、"
        "「デバッグレポートをコピー」を押して結果をバグ報告に貼り付けてください。デー "
        "モンのログと最近のジョブ状態がまとめられます。"
    ),
    "settings_debug_copy_report": "デバッグレポートをコピー",
    "settings_mcp_header": "MCP サーバー（エージェント連携）",
    "settings_mcp_desc": "ローカル学習デーモンを MCP クライアント（Claude Code、Claude Desktop "
    "など）に公開します。以下のコマンドをターミナルで実行すると Claude Code に登録されます:",
    "settings_mcp_desc_json": "他の MCP クライアント（Claude Desktop、OpenClaw など）には、"
    "同等の JSON 設定を使用します:",
    "settings_mcp_copy": "コピー",
    "settings_mcp_copied": "コピーしました ✓",
    "settings_close": "閉じる",
    "settings_lang_apply_title": "言語",
    "settings_lang_apply_question": "今すぐインターフェースを再読み込みして言語を適用しますか？\n\n"
    "タブの未保存の編集内容は失われます。待機中・実行中の学習ジョブはデーモンで"
    "動いているため影響ありません。\n\n「いいえ」を選ぶと次回起動時に適用されます。",
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
    "models_btn_tooltip": "モデルチェックポイントをダウンロード / 再ダウンロード — Anima の重み(CJK 語彙パック含む)と anime_tools のキュレーションパックをまとめて",
    "update_btn": "更新",
    "update_btn_tooltip": "GitHub から最新の anima_lora リリースを取得して uv sync を実行します",
    "update_btn_available": "更新 ●",
    "update_btn_available_tooltip": "新しいリリース {v} があります — クリックしてリリースノートを確認",
    "report_issue": "問題を報告",
    "report_issue_tooltip": "ブラウザで GitHub Issue トラッカーを開きます",
    "visit_github": "GitHub ページを開く",
    # Models dialog
    "models_title": "モデルのダウンロード",
    "models_intro": "学習 / 推論の実行に必要な重みをパック単位で表示します。「初回セットをダウンロード」で Anima の重み 3 点、PE-Spatial (REPA)、CJK 語彙パック(v2 から既定で有効)、タガーのチェックポイント、タグ DB を取得します。SAM3(マスキング)と OCR は「キュレーション」タブの任意パックです。ファイルは models/ に保存されます。",
    "models_download_all": "初回セットをダウンロード",
    "models_download": "ダウンロード",
    "models_redownload": "再ダウンロード",
    "models_installed": "✓ インストール済み",
    "models_missing": "✗ 未インストール",
    "model_anima": "Anima — DiT + テキストエンコーダー + VAE",
    "model_sam3": "SAM3 — テキストバブルマスキング",
    "model_pe": "PE-Core-L14-336 — ビジョンエンコーダー (CMMD 検証)",
    "model_anima_dit": "Anima ベース DiT — 学習対象のモデル",
    "model_anima_te": "Qwen3-0.6B テキストエンコーダー — プロンプト埋め込み",
    "model_anima_vae": "Qwen-Image VAE — 潜在表現のエンコード / デコード",
    "model_vocab_pack": "CJK 語彙パック — 日本語 / 韓国語 / 中国語のキャプション行 (既定で有効)",
    "model_pe_spatial": "PE-Spatial-B16-512 — REPA + 類似画像のグルーピング",
    "models_used_by": "使用箇所: {what}",
    "models_tab_anima": "Anima",
    "models_tab_curation": "キュレーション (anime_tools)",
    "curation_models_intro": "キュレーション各段階の重みをパック単位で表示します — タガー、タグ DB、マスキング(SAM3、任意)、OCR(任意)、グルーピング。anime_tools のカタログを直接読みます。各段階は初回実行時に自分で取得もするため、ここでは待ち時間を移すだけです。",
    "models_download_missing": "未取得をすべてダウンロード ({n} 件)",
    "models_all_installed": "✓ すべてインストール済み",
    "models_download_pack": "パックをダウンロード",
    "models_redownload_pack": "パックを再ダウンロード",
    "models_pack_anima": "Anima ベース",
    "models_pack_anima_desc": "DiT、Qwen3-0.6B テキストエンコーダー、Qwen-Image VAE — すべての学習 / 推論の実行に 3 点とも必要です。",
    "models_pack_pe": "PE-Core",
    "models_pack_pe_desc": "PE-Core-L14-336: CMMD 検証メトリック専用 (use_cmmd、既定で無効)。REPA の特徴キャッシュは PE-Spatial を使います — 「キュレーション」タブの「グルーピング」パック。",
    "models_pack_cjk": "CJK 語彙パック",
    "models_pack_cjk_desc": "日本語 / 韓国語 / 中国語のキャプション・プロンプト区間向けの追加テキストエンコーダー行。v2 から既定で有効で、英語テキストはどちらでもビット単位で同一です。",
    "models_pack_tagger": "タガー",
    "models_pack_tagger_desc": "Anima Tagger: チェックポイント、ゲート付きの dbv4 バックボーン、そこからトレースした ONNX グラフ。",
    "models_pack_tags": "Danbooru タグ DB",
    "models_pack_tags_desc": "キャプション補正が照合する約 114k 行のタグ表と、その英語説明。",
    "models_pack_masking": "マスキング",
    "models_pack_masking_desc": "SAM3 被写体マスク(ゲート付き; v2 からマスキングは任意)と、位置段階が検出に使う被写体ソフトプロンプト。",
    "models_pack_ocr": "OCR",
    "models_pack_ocr_desc": "AnimeText テキストブロック検出器と漫画 VL リーダー(PaddleOCR-VL-1.6 ベース + SFX ファインチューン)。任意。",
    "models_pack_grouping": "グルーピング",
    "models_pack_grouping_desc": "PE-Spatial-B16-512: REPA の特徴キャッシュ (preprocess-pe) と類似画像グルーピング用タワー。",
    "model_danbooru_tags": "Danbooru タグ DB — キャプション順序補正",
    "model_tagger": "Anima Tagger — caformer_b36 バックボーン (ゲート付き)",
    # HuggingFace 認証 (モデルダイアログ)
    "models_hf_token_placeholder": "HuggingFace トークンを貼り付けてください (hf_…)",
    "models_hf_authenticate": "認証",
    "models_hf_token_hint": "ゲート付き/レート制限のあるダウンロード(SAM3 など)に必要です。"
    '<a href="https://huggingface.co/settings/tokens">'
    "huggingface.co/settings/tokens</a> でトークンを作成し、"
    '<a href="https://huggingface.co/facebook/sam3">huggingface.co/facebook/sam3</a> で SAM3 のアクセスを申請してください。'
    "Anima Tagger のバックボーンもゲート付きです。"
    '<a href="https://huggingface.co/animetimm/caformer_b36.dbv4-full">animetimm/caformer_b36.dbv4-full</a> で利用規約に同意してください。',
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
    # LoRA ⊕ LoRA → LoRA マージ (merge_loras.py)
    "merge_mode": "モード:",
    "merge_mode_dit": "DiT にマージ",
    "merge_mode_loras": "LoRA をマージ",
    "merge_lora_options": "LoRA マージオプション",
    "merge_lora_select_hint": "2 つ以上のアダプターを選択（Ctrl/Shift クリック）して 1 つの LoRA に統合します。通常の --lora_weight パスで使用できます。",
    "merge_lora_selected": "選択中:",
    "merge_lora_need_two": "LoRA を 2 つ以上選択してください。",
    "merge_weights": "重み:",
    "merge_weights_placeholder": "（任意: 例 1.0,0.7,0.7 — リスト順）",
    "merge_weights_tip": "カンマ区切りの LoRA ごとの強度で、リスト順に選択した各アダプターに 1 つずつ指定します。空欄ですべて 1.0。",
    "merge_weights_mismatch": "重みが {n} 個指定されましたが、LoRA は {m} 個選択されています。",
    "merge_normalize": "正規化:",
    "merge_normalize_tip": "マージしたデルタを単一 LoRA の大きさに再スケーリングします。'global'（既定）はグローバル RMS を平均単一 LoRA ノルムに合わせ、'per_module' はモジュールごとに合わせますがレイヤー重みが変わり、'off' は生の厳密連結和（DiT を過剰駆動する可能性があります）です。",
    "merge_lora_button": "LoRA をマージ",
    "merge_analyze_button": "干渉を分析",
    "merge_analyze_tip": "ドライラン: 選択した LoRA が重み空間でどう干渉するか（ペアごと・層ごとの強め合い対打ち消し合い）を、マージファイルを書き出さずにレポートします。",
    "merge_analysis_safe": "✓ マージ安全 — LoRA はほぼ直交 · 最大ペア |cos| {cos} · エネルギー比 {ratio} · 共有モジュール {shared}/{modules}",
    "merge_analysis_reinforce": "⚠ {strength}強め合い — {a} ↔ {b} cos {cos} · エネルギー比 {ratio}（normalize=global が合算量を再スケール）· 共有モジュール {shared}/{modules}",
    "merge_analysis_cancel": "⚠ {strength}打ち消し合い — {a} ↔ {b} cos {cos} · これらの LoRA は互いに部分的に消し合う · エネルギー比 {ratio} · 共有モジュール {shared}/{modules}",
    "merge_analysis_overlap": "⚠ 部分空間の衝突 — {a} ↔ {b} は方向はほぼ直交だが同じ重み部分空間に書き込む（重なり {overlap}、ランダム比 {x}倍）· 推論時にスタイルが競合する可能性 · エネルギー比 {ratio} · 共有モジュール {shared}/{modules}",
    "merge_analysis_moderate": "中程度の",
    "merge_analysis_strong": "強い",
    "merge_lora_out_placeholder": "（自動: <最初の LoRA>_merged<N>.safetensors）",
    "merge_mode_extract": "LoRA 抽出",
    "merge_extract_options": "LoRA 抽出オプション",
    "merge_extract_hint": "2 つのフル DiT チェックポイント間の重みデルタ（ΔW = チューニング − ベース）から素の LoRA を抽出し、選択したランクに SVD 切り詰めします。結果はすべての素の LoRA 経路（--lora_weight、DiT へのマージ、ComfyUI、ターボのウォームスタート）で読み込めます。両チェックポイントとも net. プレフィックスの Anima レイアウトである必要があります。",
    "merge_extract_base": "ベース DiT:",
    "merge_extract_tuned": "チューニング済みチェックポイント:",
    "merge_extract_tuned_placeholder": "（必須: 左のリストからチェックポイントをクリック、または参照）",
    "merge_extract_rank": "ランク:",
    "merge_extract_rank_tip": "ΔW を SVD 切り詰めするランク（出力 LoRA の network_dim）。96 で一般的なフルモデルデルタのほとんどを捉えます。",
    "merge_extract_include_adaln": "adaln アッププロジェクションを含める",
    "merge_extract_include_adaln_tip": "adaln_up_{branch} デルタも抽出します（公式ターボデルタで最も大きく動く部分）。デフォルトはオフ: 素の生徒モデルは adaln を対象とせず、標準ローダーはこれらのキーをスキップします。",
    "merge_extract_adaln_layout": "adaln レイアウト:",
    "merge_extract_adaln_layout_tip": "adaln LoRA のキー命名: 'runtime' = リポジトリ内モジュール名（ターボのウォームスタート経路用）、'comfy' = ComfyUI state-dict パス（コアでネイティブに読み込み）。adaln を含める場合のみ使用されます。",
    "merge_extract_out_placeholder": "（必須: 出力 LoRA .safetensors）",
    "merge_extract_pick_ckpt": "フル DiT .safetensors を選択",
    "merge_extract_no_tuned": "ベースと差分を取るチューニング済みチェックポイントを選択してください。",
    "merge_extract_no_out": "抽出した LoRA の出力パスを指定してください。",
    "merge_extract_button": "LoRA 抽出",
    "browse": "参照…",
    # Multi-scale target_res tiers
    "target_res_bucket_tooltip": "{edge}px ティアでこの bucket 解像度だけを前処理に使用します。すべて未選択なら選択ティアの全 bucket を使用します。",
    "target_res_danger_tooltip": "高コストなティア：{edge}px は画像あたり約 {tokens} トークンを使用し、コンパイル済みブロックグラフを 1 つ追加します（コンパイルが遅くなり、VRAM が増加）。この解像度が本当に必要な場合のみ有効にしてください。",
    "resize_crop_anchor": "リサイズクロップ位置:",
    "resize_crop_anchor_tip": "cover resize 後に target bucket へクロップするとき、どの方向を保持するか選択します。",
    "resize_crop_anchor_top_left": "左上",
    "resize_crop_anchor_top": "上",
    "resize_crop_anchor_top_right": "右上",
    "resize_crop_anchor_left": "左",
    "resize_crop_anchor_center": "中央",
    "resize_crop_anchor_right": "右",
    "resize_crop_anchor_bottom_left": "左下",
    "resize_crop_anchor_bottom": "下",
    "resize_crop_anchor_bottom_right": "右下",
    "resize_crop_margins": "リサイズ余白:",
    "resize_crop_margin_top": "上",
    "resize_crop_margin_right": "右",
    "resize_crop_margin_bottom": "下",
    "resize_crop_margin_left": "左",
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
    "preprocess_current_tooltip": "現在のバリアントに対し、GUI のパス範囲で前処理を実行します。",
    "sample_prompt_col_prompt": "プロンプト",
    "sample_prompt_col_width": "W",
    "sample_prompt_col_height": "H",
    "sample_prompt_col_steps": "ステップ",
    "sample_prompt_col_seed": "シード",
    "sample_prompt_col_cfg": "CFG",
    "sample_prompt_col_guidance": "ガイダンス",
    "sample_prompt_col_shift": "シフト",
    "sample_prompt_col_negative": "ネガティブ",
    "sample_prompt_col_extra": "追加",
    "sample_prompt_add": "プロンプトを追加",
    "sample_prompt_select_all": "すべて選択",
    "sample_prompt_remove": "選択を削除",
    "sample_prompt_remove_confirm_title": "サンプルプロンプトの削除",
    "sample_prompt_remove_confirm_body": "選択した {n} 件のサンプルプロンプトを削除しますか?",
    "sample_prompt_expand": "エディタを展開",
    "sample_prompt_collapse": "エディタを折りたたむ",
    "sample_prompt_select": "選択",
    "sample_prompt_prompt_placeholder": "プロンプト本文。改行はここでは表示され、保存時にはスペースになります。",
    "sample_prompt_hint": "空欄/デフォルト値はプロンプト行に書き込まれません。",
    "sample_prompt_default_width": "デフォルト 512",
    "sample_prompt_default_height": "デフォルト 512",
    "sample_prompt_default_steps": "デフォルト 30",
    "sample_prompt_default_seed": "自動シード",
    "sample_prompt_default_cfg": "デフォルト 7.5",
    "sample_prompt_default_guidance": "デフォルト 1.0",
    "sample_prompt_default_shift": "デフォルト 3.0",
    "sample_prompt_default_negative": "デフォルト: なし",
    "sample_prompt_tip_width": "画像の幅 (`--w`)。空欄なら train.py のデフォルト 512 を使用します。",
    "sample_prompt_tip_height": "画像の高さ (`--h`)。空欄なら train.py のデフォルト 512 を使用します。",
    "sample_prompt_tip_steps": "サンプリングステップ数 (`--s`)。空欄なら train.py のデフォルト 30 を使用します。",
    "sample_prompt_tip_seed": "シード (`--d`)。自動シードにすると、エポックをまたいで各プロンプトを比較できます。",
    "sample_prompt_tip_cfg": "CFG スケール (`--l`)。空欄なら train.py のデフォルト 7.5 を使用します。",
    "sample_prompt_tip_guidance": "ガイダンススケール (`--g`)。空欄なら train.py のデフォルト 1.0 を使用します。",
    "sample_prompt_tip_shift": "サンプリングのシグマスケジュール用 Flow shift (`--fs`)。空欄なら train.py のデフォルト 3.0 を使用します。",
    "sample_prompt_tip_negative": "このサンプルだけに使うネガティブプロンプト (`--n`)。",
    "sample_prompt_tip_extra": "追加のサンプル引数。入力したまま保持されます。",
    "daemon_submitting": "学習デーモンにジョブを送信中…",
    "daemon_submit_failed": "学習デーモンに接続できませんでした: {err}",
    "daemon_queued": "ジョブ {job_id} を学習デーモンのキューに追加しました。\n",
    "queue_submitting_train_preprocess": "{variant} の前処理 + 学習を学習デーモンのキューに追加中…",
    "queue_submitting_preprocess": "{variant} の前処理を学習デーモンのキューに追加中…",
    "queue_added_preprocess_only": "{variant} を前処理ジョブ {job_id} としてキューに追加しました。\n",
    "daemon_reattached": "実行中のジョブ {job_id} に再接続しました (前のセッションで開始)。\n",
    "daemon_next_queued": "学習デーモンの次の待機ジョブ {job_id} を追跡します。\n",
    "daemon_job_finished": "--- Job {job_id} {state} ---",
    "train_queued": "学習 (キュー登録済み)",
    "train_running_daemon": "学習 (実行中…)",
    "models_done_title": "ダウンロード完了",
    "models_done_message": "モデルのダウンロードが完了しました。ファイルは models/ 以下に保存されます。",
    "models_failed_title": "ダウンロード失敗",
    "models_failed_message": "ダウンロードが終了コード {code} で終了しました。詳細はログを確認してください。",
}
