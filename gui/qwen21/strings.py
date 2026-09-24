"""en / cn strings for the Qwen-Image-2.1 GUI.

Kept out of ``gui.i18n``: this window is en/cn only and its keys never reach the
main GUI. Field labels/help: English falls back to the request field's own
``help`` metadata (``library.qwen21.requests``), so a new flag shows up in
English with no entry here.
"""

from __future__ import annotations

LANGUAGES = {"en": "English", "cn": "中文"}

UI: dict[str, dict[str, str]] = {
    "en": {
        "window_title": "Qwen-Image-2.1 LoRA",
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
        "model": "Model: {path}",
        "model_missing": "Model folder not found: {path}\nSet it under Advanced → "
        "model_dir, or set ANIMA_QWEN21_MODEL_DIR in .env.",
        "scan": "{pairs}/{images} images have captions · text cached {text} · "
        "latents cached {latents}",
        "scan_stale": "{stale} captions changed after their text cache — tick "
        "overwrite (or delete those .te files) to re-encode them.",
        "scan_no_src": "Pick a folder of images with .txt captions.",
        "cache_counts": "Cache folder: {samples} samples ready to train "
        "(text {text} · latents {latents})",
        "scan_dupes": "{n} file names appear in more than one subfolder — caching "
        "will refuse until they are renamed (the cache is keyed by file name).",
        "train_ready": "{n} cached samples ready.",
        "train_empty": "No cached samples in this folder yet — run preprocessing first.",
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
    },
    "cn": {
        "window_title": "Qwen-Image-2.1 LoRA",
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
        "model": "模型：{path}",
        "model_missing": "找不到模型文件夹：{path}\n请在「高级 → model_dir」中设置，"
        "或在 .env 中设置 ANIMA_QWEN21_MODEL_DIR。",
        "scan": "{images} 张图片中 {pairs} 张有标注 · 已缓存文本 {text} · "
        "已缓存潜变量 {latents}",
        "scan_stale": "{stale} 条标注在文本缓存之后被修改——勾选「覆盖」"
        "（或删除对应的 .te 文件）以重新编码。",
        "scan_no_src": "请选择包含图片和 .txt 标注的文件夹。",
        "cache_counts": "缓存文件夹：{samples} 个样本可用于训练"
        "（文本 {text} · 潜变量 {latents}）",
        "scan_dupes": "有 {n} 个文件名在多个子文件夹中重复——缓存按文件名存放，"
        "重命名之前预处理会拒绝运行。",
        "train_ready": "已缓存 {n} 个样本，可以训练。",
        "train_empty": "此文件夹中还没有缓存样本——请先运行预处理。",
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
    },
}

# Field labels + help, keyed by request field name. English is not listed:
# its label is the field name and its help the request's own metadata.
FIELDS_CN: dict[str, tuple[str, str]] = {
    "src": (
        "图片文件夹",
        "包含图片和同名 .txt 标注的文件夹，包括子文件夹"
        "（默认：Anima 的 resized 目录，其中是修订后的标注）",
    ),
    "out": ("缓存文件夹", "每张图片写入两个缓存文件（文本嵌入 + 潜变量）"),
    "resolution": (
        "分辨率",
        "目标像素面积的边长——1024 表示按图片原始比例缩放到约 1024² 像素，"
        "而不是裁成 1024×1024",
    ),
    "overwrite": ("覆盖", "即使缓存文件已存在也重新编码"),
    "skip_text": ("跳过文本", "跳过文本编码器"),
    "skip_latents": ("跳过潜变量", "跳过 VAE 编码"),
    "save_crops": ("保存缩放图", "同时保存用于生成潜变量的缩放后图片"),
    "te_blocks_to_swap": ("文本编码器交换块数", "默认根据可用显存自动计算"),
    "model_dir": (
        "模型文件夹",
        "diffusers 格式的文件夹（默认：$ANIMA_QWEN21_MODEL_DIR 或 "
        "models/qwen_image_2.1）",
    ),
    "cache": ("缓存文件夹", "预处理写入的缓存文件夹"),
    "output": ("输出", "LoRA 输出路径（同目录下会写入 .json 运行报告）"),
    "epochs": ("轮数", "遍历缓存的次数"),
    "rank": ("秩", "LoRA 秩"),
    "alpha": ("alpha", "LoRA alpha（默认等于秩）"),
    "lr": ("学习率", "AdamW 学习率"),
    "save_every_epochs": ("每 N 轮保存", "每 N 轮额外保存一次（0 = 只保存最终结果）"),
    "warmup_ratio": ("预热比例", "线性预热占总步数的比例"),
    "max_grad_norm": ("梯度裁剪", "最大梯度范数"),
    "lora_dtype": ("LoRA 精度", "适配器主权重的数据类型（矩阵乘法始终按模型的 bf16 进行）；bf16 省一半优化器状态但会丢失小更新"),
    "targets": ("目标层", "对线性层名称做完整匹配的正则表达式"),
    "logit_mean": ("logit 均值", "logit-normal σ 采样的均值"),
    "logit_std": ("logit 标准差", "logit-normal σ 采样的标准差"),
    "blocks_to_swap": (
        "交换块数",
        "交换到内存的 transformer 块数（默认根据 activation_reserve_gb 计算）",
    ),
    "activation_reserve_gb": (
        "激活预留 (GB)",
        "计算交换块数时为激活保留的显存（默认按缓存中最大的图像+文本 token 数计算：0.3 GB + 0.6 MB × token 数 + 一个块的余量）",
    ),
    "grad_checkpointing": ("梯度检查点", "激活检查点——节省显存的主要手段"),
    "compile": ("torch.compile", "对每个块编译（默认开启）；计算瓶颈（1024²、交换 7 块、检查点）下每步快 11%，PCIe 瓶颈（512²、交换 12 块）下无收益；启动时约 40 秒编译"),
    "compile_seq": (
        "序列符号化方式",
        "dynamic = torch.compile(dynamic=True)，一张图覆盖所有样本；bounded = 自动动态形状 + 按缓存的 [最小, 最大] 联合 token 数 mark_dynamic（隐藏维保持静态）。速度相同，只是编译时间的分摊方式不同",
    ),
    "compile_mode": ("编译模式", "torch.compile 模式"),
    "seed": ("随机种子", "随机数种子"),
}

# GenerateRequest fields whose name also exists on another request with a
# different meaning.
FIELDS_CN_GENERATE: dict[str, tuple[str, str]] = {
    "prompt": ("提示词", "提示词（设置 prompts_file 时忽略）"),
    "lora": ("LoRA", "要测试的 LoRA（留空 = 训练输出）"),
    "multipliers": ("强度", "逗号分隔的适配器强度；0.0 即基础模型"),
    "width": ("宽度", "32 的倍数（默认：分辨率，正方形）"),
    "height": ("高度", "32 的倍数（默认：分辨率，正方形）"),
    "steps": ("步数", "去噪步数"),
    "seed": ("随机种子", "第一个提示词的种子（每个提示词 +1）"),
    "out_dir": ("输出文件夹", "图片和 manifest.json 保存在这里"),
    "prompts_file": ("提示词文件", "每行一个提示词；会覆盖上面的提示词"),
    "resolution": ("分辨率", "未设置宽高时的正方形边长"),
    "true_cfg_scale": (
        "CFG",
        "管线默认值——2.1 没有引导嵌入，大于 1 时每步多一次前向计算",
    ),
    "negative_prompt": ("反向提示词", "仅在 CFG > 1 时使用"),
    "blocks_to_swap": (
        "交换块数",
        "交换到内存的 transformer 块数（默认按可用显存计算）",
    ),
}
