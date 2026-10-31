"""Strings shared by the sidecar-line windows (``gui.lines.window``).

Kept out of ``gui.i18n``: their keys never reach the main GUI. A line offers
the languages its own ``LineSpec.ui`` has (Qwen: en/cn, MOSS: en/ko/ja/cn). Each line adds its own keys (``window_title``,
``model_missing``, ``scan_stale``, ``scan_dupes``) through ``LineSpec.ui``.
"""

from __future__ import annotations

LANGUAGES = {"en": "English", "ko": "한국어", "ja": "日本語", "cn": "中文"}

UI: dict[str, dict[str, str]] = {
    "en": {
        "tab_preprocess": "Preprocess",
        "tab_train": "Train",
        "run_cache": "Run preprocessing",
        "run_train": "Train",
        "run_chain": "Preprocess, then train",
        "stop": "Stop",
        "advanced": "Advanced",
        "browse": "Browse…",
        "auto": "auto",
        "language": "Language",
        "model": "Model folder: {path}",
        "guide_preprocess": "Preprocessing encodes every image in the image folder "
        "once — its .txt caption through the text encoder, the image through the "
        "VAE — and writes both to the cache folder. Images that are already cached "
        "are skipped, so after adding images only the new ones are encoded. "
        "Training reads the cache folder only.",
        "guide_train": "Training fits a LoRA on the cached samples; the base model "
        'stays frozen. "Preprocess, then train" runs both in order and starts '
        "training only if preprocessing succeeds. Test renders the prompt with the "
        "LoRA and without it at the same seed, so the pair differs by the adapter "
        "only.",
        "click_field": "Click a field label to see its explanation here.",
        "scan": "{pairs}/{images} images have captions · text cached {text} · "
        "latents cached {latents}",
        "scan_no_src": "Pick a folder of images with .txt captions.",
        "cache_counts": "Cache folder: {samples} samples ready to train "
        "(text {text} · latents {latents})",
        "train_ready": "{n} cached samples ready.",
        "train_empty": "No cached samples in this folder yet — run preprocessing first.",
        "train_subfolders": "No cached samples directly in this folder, and training "
        "does not read subfolders. Set the cache folder to one of: {folders}",
        "bad_value": "{field}: '{value}' is not a valid number.",
        "running": "Running: {label} ({job})",
        "finished": "{label} finished: {state}",
        "chain_next": "Preprocessing done — starting training.",
        "chain_cache": "Training will read the preprocessing output: {path}",
        "reattached": "Re-attached to running job {job}.",
        "idle": "Idle",
        "test_group": "Test",
        "run_test": "Test",
        "test_lora_placeholder": "train output: {path}",
        "no_lora": "No LoRA at {path}.\nTrain first, or set lora in the Test panel.",
        "test_lora": "LoRA ×{m}",
        "test_base": "base model",
        "test_hint": "Renders the prompt with the LoRA and without it (same seed), "
        "so the pair differs by the adapter only.",
        "test_next": "LoRA pass done — rendering the base model.",
    },
    "ko": {
        "tab_preprocess": "전처리",
        "tab_train": "학습",
        "run_cache": "전처리 실행",
        "run_train": "학습 시작",
        "run_chain": "전처리 후 학습",
        "stop": "중지",
        "advanced": "고급",
        "browse": "찾아보기…",
        "auto": "자동",
        "language": "언어",
        "model": "모델 폴더: {path}",
        "guide_preprocess": "전처리는 이미지 폴더의 모든 이미지를 한 번씩 인코딩합니다 — "
        ".txt 캡션은 텍스트 인코더로, 이미지는 VAE로 — 그리고 결과를 캐시 폴더에 씁니다. "
        "이미 캐시된 이미지는 건너뛰므로, 이미지를 추가한 뒤에는 새 이미지만 인코딩됩니다. "
        "학습은 캐시 폴더만 읽습니다.",
        "guide_train": "학습은 캐시된 샘플로 LoRA를 학습하며, 베이스 모델은 고정됩니다. "
        '"전처리 후 학습"은 두 단계를 순서대로 실행하고, 전처리가 성공해야만 학습을 '
        "시작합니다. 테스트는 같은 시드로 LoRA를 적용한 이미지와 적용하지 않은 이미지를 "
        "생성하므로, 두 이미지의 차이는 어댑터에서만 옵니다.",
        "click_field": "필드 이름을 클릭하면 여기에 설명이 표시됩니다.",
        "scan": "이미지 {images}장 중 {pairs}장에 캡션 있음 · 텍스트 캐시 {text} · "
        "잠재변수 캐시 {latents}",
        "scan_no_src": ".txt 캡션이 있는 이미지 폴더를 선택하세요.",
        "cache_counts": "캐시 폴더: 학습 가능한 샘플 {samples}개 "
        "(텍스트 {text} · 잠재변수 {latents})",
        "train_ready": "캐시된 샘플 {n}개, 학습 준비 완료.",
        "train_empty": "이 폴더에 아직 캐시된 샘플이 없습니다 — 먼저 전처리를 실행하세요.",
        "train_subfolders": "이 폴더 바로 아래에는 캐시된 샘플이 없고, 학습은 하위 폴더를 "
        "읽지 않습니다. 캐시 폴더를 다음 중 하나로 지정하세요: {folders}",
        "bad_value": "{field}: '{value}'은(는) 올바른 숫자가 아닙니다.",
        "running": "실행 중: {label} ({job})",
        "finished": "{label} 종료: {state}",
        "chain_next": "전처리 완료 — 학습을 시작합니다.",
        "chain_cache": "학습은 전처리 출력을 읽습니다: {path}",
        "reattached": "실행 중인 작업 {job}에 다시 연결했습니다.",
        "idle": "대기 중",
        "test_group": "테스트",
        "run_test": "테스트",
        "test_lora_placeholder": "학습 출력: {path}",
        "no_lora": "LoRA가 없습니다: {path}\n먼저 학습하거나, 테스트 패널에서 lora를 "
        "지정하세요.",
        "test_lora": "LoRA ×{m}",
        "test_base": "베이스 모델",
        "test_hint": "같은 시드로 LoRA를 적용한 이미지와 적용하지 않은 이미지를 생성하므로, "
        "두 이미지의 차이는 어댑터에서만 옵니다.",
        "test_next": "LoRA 생성 완료 — 베이스 모델을 생성합니다.",
    },
    "ja": {
        "tab_preprocess": "前処理",
        "tab_train": "学習",
        "run_cache": "前処理を実行",
        "run_train": "学習開始",
        "run_chain": "前処理してから学習",
        "stop": "停止",
        "advanced": "詳細",
        "browse": "参照…",
        "auto": "自動",
        "language": "言語",
        "model": "モデルフォルダ: {path}",
        "guide_preprocess": "前処理は画像フォルダ内の各画像を一度だけエンコードします — "
        ".txt キャプションはテキストエンコーダで、画像は VAE で — そして結果をキャッシュ"
        "フォルダに書き込みます。キャッシュ済みの画像はスキップされるので、画像を追加した後は"
        "新しい画像だけがエンコードされます。学習はキャッシュフォルダだけを読みます。",
        "guide_train": "学習はキャッシュ済みサンプルで LoRA を学習します。ベースモデルは"
        "固定されたままです。「前処理してから学習」は両方を順に実行し、前処理が成功した"
        "場合にのみ学習を始めます。テストは同じシードで LoRA ありとなしの画像を生成するので、"
        "2枚の違いはアダプターだけによるものです。",
        "click_field": "フィールド名をクリックすると、ここに説明が表示されます。",
        "scan": "画像 {images} 枚中 {pairs} 枚にキャプションあり · テキストキャッシュ "
        "{text} · 潜在変数キャッシュ {latents}",
        "scan_no_src": ".txt キャプション付きの画像フォルダを選んでください。",
        "cache_counts": "キャッシュフォルダ: 学習可能なサンプル {samples} 件"
        "（テキスト {text} · 潜在変数 {latents}）",
        "train_ready": "キャッシュ済みサンプル {n} 件、学習できます。",
        "train_empty": "このフォルダにはまだキャッシュ済みサンプルがありません — "
        "先に前処理を実行してください。",
        "train_subfolders": "このフォルダの直下にはキャッシュ済みサンプルがなく、学習は"
        "サブフォルダを読みません。キャッシュフォルダを次のいずれかに設定してください: "
        "{folders}",
        "bad_value": "{field}: 「{value}」は有効な数値ではありません。",
        "running": "実行中: {label}（{job}）",
        "finished": "{label} 終了: {state}",
        "chain_next": "前処理完了 — 学習を開始します。",
        "chain_cache": "学習は前処理の出力を読みます: {path}",
        "reattached": "実行中のジョブ {job} に再接続しました。",
        "idle": "待機中",
        "test_group": "テスト",
        "run_test": "テスト",
        "test_lora_placeholder": "学習出力: {path}",
        "no_lora": "LoRA が見つかりません: {path}\n先に学習するか、テストパネルで lora を"
        "指定してください。",
        "test_lora": "LoRA ×{m}",
        "test_base": "ベースモデル",
        "test_hint": "同じシードで LoRA ありとなしの画像を生成するので、2枚の違いは"
        "アダプターだけによるものです。",
        "test_next": "LoRA の生成完了 — ベースモデルを生成します。",
    },
    "cn": {
        "tab_preprocess": "预处理",
        "tab_train": "训练",
        "run_cache": "开始预处理",
        "run_train": "开始训练",
        "run_chain": "预处理后训练",
        "stop": "停止",
        "advanced": "高级",
        "browse": "浏览…",
        "auto": "自动",
        "language": "语言",
        "model": "模型文件夹：{path}",
        "guide_preprocess": "预处理会把图片文件夹中的每张图片编码一次——.txt 标注经过"
        "文本编码器，图片经过 VAE——并把结果写入缓存文件夹。已有缓存的图片会被跳过，"
        "所以新增图片后只会编码新图片。训练只读取缓存文件夹。",
        "guide_train": "训练在缓存样本上拟合一个 LoRA，基础模型保持冻结。"
        "「预处理后训练」会依次运行两步，只有预处理成功才开始训练。"
        "测试会用同一个种子分别在加载和不加载 LoRA 的情况下生成，"
        "两张图的差别只来自适配器。",
        "click_field": "点击字段标签可在此处查看说明。",
        "scan": "{images} 张图片中 {pairs} 张有标注 · 已缓存文本 {text} · "
        "已缓存潜变量 {latents}",
        "scan_no_src": "请选择包含图片和 .txt 标注的文件夹。",
        "cache_counts": "缓存文件夹：{samples} 个样本可用于训练"
        "（文本 {text} · 潜变量 {latents}）",
        "train_ready": "已缓存 {n} 个样本，可以训练。",
        "train_empty": "此文件夹中还没有缓存样本——请先运行预处理。",
        "train_subfolders": "此文件夹下没有直接存放的缓存样本，训练也不会读取子文件夹。"
        "请把缓存文件夹设为以下之一：{folders}",
        "bad_value": "{field}：「{value}」不是有效的数字。",
        "running": "运行中：{label}（{job}）",
        "finished": "{label} 已结束：{state}",
        "chain_next": "预处理完成——开始训练。",
        "chain_cache": "训练将读取预处理输出：{path}",
        "reattached": "已重新连接到正在运行的任务 {job}。",
        "idle": "空闲",
        "test_group": "测试",
        "run_test": "测试",
        "test_lora_placeholder": "训练输出：{path}",
        "no_lora": "找不到 LoRA：{path}\n请先训练，或在测试面板中设置 lora。",
        "test_lora": "LoRA ×{m}",
        "test_base": "基础模型",
        "test_hint": "用同一个种子分别在加载和不加载 LoRA 的情况下生成，"
        "两张图的差别只来自适配器。",
        "test_next": "LoRA 生成完成——开始生成基础模型。",
    },
}
