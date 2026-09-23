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
    "lora_dtype": ("LoRA 精度", "适配器权重的数据类型"),
    "targets": ("目标层", "对线性层名称做完整匹配的正则表达式"),
    "logit_mean": ("logit 均值", "logit-normal σ 采样的均值"),
    "logit_std": ("logit 标准差", "logit-normal σ 采样的标准差"),
    "blocks_to_swap": (
        "交换块数",
        "交换到内存的 transformer 块数（默认根据 activation_reserve_gb 计算）",
    ),
    "activation_reserve_gb": ("激活预留 (GB)", "计算交换块数时为激活保留的显存"),
    "grad_checkpointing": ("梯度检查点", "激活检查点——节省显存的主要手段"),
    "compile": ("torch.compile", "对每个块动态编译；在 PCIe 瓶颈下实测无收益"),
    "compile_mode": ("编译模式", "torch.compile 模式"),
    "seed": ("随机种子", "随机数种子"),
}
