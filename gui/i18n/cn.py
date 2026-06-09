"""Simplified Chinese strings for the Anima LoRA GUI.

Best-effort machine translation — please proofread before relying on it.
Missing keys fall back to English via the `t()` lookup in `__init__.py`.
"""

from __future__ import annotations

STRINGS: dict[str, str] = {
    # Window / tabs
    "window_title": "Anima LoRA",
    "tab_config": "训练配置",
    "tab_ip_adapter": "IP-Adapter",
    "tab_easycontrol": "EasyControl",
    "tab_spd": "SPD",
    "tab_turbo": "Turbo",
    "tab_methods": "方法",
    "tab_images": "数据集",
    "tab_merge": "合并",
    "tab_queue": "队列状况",
    "tab_preprocess": "预处理",
    "tab_tensorboard": "TensorBoard",
    # PreprocessingTab
    "preprocess_intro": (
        "配置标注随机化和气泡蒙版,然后按需运行每个步骤。"
        "训练配置选项卡的「训练」按钮在没有缓存时会用默认设置自动运行预处理 —— "
        "本选项卡用于细调和单独重跑某个步骤。"
    ),
    "preprocess_image_prep": "图像预处理 (调整大小 / 过滤)",
    "preprocess_source_image_dir": "源图像目录:",
    "preprocess_source_image_dir_tip": (
        "存放原始训练图像的文件夹 (含 .txt 标注附属文件)，"
        "缓存步骤会将其调整大小后写入 post_image_dataset/。"
        "保存至 configs/preprocess.toml。"
    ),
    "preprocess_drop_lowres": "丢弃低分辨率图像",
    "preprocess_drop_lowres_tip": (
        "跳过小于下方像素阈值的源图像，使其不进入调整大小 / VAE / 文本缓存。"
        "取消勾选可保留所有图像，无论大小。"
    ),
    "preprocess_min_pixels": "最小像素数 (过滤阈值):",
    "preprocess_min_pixels_tip": (
        "低分辨率过滤器的像素数阈值。500000 = 0.5MP。"
        "当「丢弃低分辨率图像」未勾选时忽略。"
    ),
    "preprocess_text_caching": "缓存 (VAE + 文本)",
    "preprocess_caption_shuffle_variants": "每条标注的随机变体数 (N):",
    "preprocess_caption_shuffle_variants_tip": (
        "为每张图像生成 N 个标注变体。v0 是原始标注;"
        "v1..v(N-1) 经过智能打乱,且 (若标签 dropout > 0) 非前缀标签会独立丢弃。"
        "当 use_shuffled_caption_variants=true 时,数据加载器以 20% 概率选 v0,"
        "其余以均匀分布从 v1..v(N-1) 中选。"
        "设为 0 时仅缓存一条原始标注。"
    ),
    "preprocess_caption_tag_dropout_rate": "标签 dropout 比率 (0.0–1.0):",
    "preprocess_caption_tag_dropout_rate_tip": (
        "适用于 v1..v(N-1) 的每标签 dropout 概率。"
        "直到首个 @artist 标记之前的标签 (含该标记) 永不丢弃。"
        "当随机变体数 ≤ 0 时忽略。"
    ),
    "preprocess_run_te": "运行缓存 (VAE + 文本)",
    "preprocess_masking_sam": "SAM3 蒙版 (对话气泡)",
    "preprocess_masking_mit": "MIT 蒙版 (漫画文字)",
    "preprocess_sam_prompts": "SAM 提示词 (每行一个):",
    "preprocess_sam_prompts_tip": (
        "SAM3 要查找的文本提示词,每行一个。默认值: 'speech bubble' 和 'text bubble'。"
    ),
    "preprocess_sam_focus_prompts": "SAM 焦点提示词 (每行一个):",
    "preprocess_sam_focus_prompts_tip": (
        "反转极性: 指定要保留的主体。设置后,蒙版将仅对该主体进行训练,"
        "其余全部忽略 (例如 'girl' 会使背景全部被忽略)。"
        "与上方提示词合成 —— 最终可训练区域为焦点主体去除忽略区域后的部分。"
        "留空则使用默认的仅忽略模式。"
    ),
    "preprocess_sam_rule": "蒙版规则",
    "preprocess_sam_add_rule": "+ 添加规则",
    "preprocess_sam_add_rule_tip": (
        "再添加一条蒙版规则。每条规则通过路径模式定位图像子集,"
        "模式匹配同一图像的多条规则将相互合成。"
    ),
    "preprocess_sam_remove_rule": "删除规则",
    "preprocess_sam_rule_path_pattern": "路径模式 (此规则):",
    "preprocess_sam_rule_path_pattern_tip": (
        "指定此规则适用的图像 —— 以数据集根目录为基准对各图像路径进行"
        "fnmatch 匹配 ('|' 作 OR 组合)。例: 'character_a/*'。"
        "留空或 '*' 可匹配所有图像 (通配默认规则)。"
    ),
    "preprocess_sam_threshold": "SAM 阈值 (0.0–1.0):",
    "preprocess_sam_threshold_tip": (
        "保留 SAM3 检测结果的最低置信度。越低 = 蒙版越多 "
        "(可能包含误报),越高 = 越严格。默认 0.5。"
    ),
    "preprocess_dilate": "膨胀 (px):",
    "preprocess_dilate_tip": (
        "对二值蒙版应用的膨胀像素数。值越大蒙版边缘越往外扩。默认 5。设为 0 表示禁用。"
    ),
    "preprocess_mit_threshold": "MIT 文字阈值 (0.0–1.0):",
    "preprocess_mit_threshold_tip": (
        "MIT/ComicTextDetector 文字分割器的置信度阈值。默认 0.8。"
    ),
    "preprocess_mask_path_pattern": "蒙版路径过滤器:",
    "preprocess_mask_path_pattern_tip": (
        "限制哪些已缩放图像参与蒙版生成的 fnmatch glob 模式，"
        "以 post_image_dataset/resized 为基准对每个路径进行匹配。"
        "同时作用于 SAM 和 MIT。与训练用 path_pattern 语法相同："
        "'*'（或空白）遮罩全部；'char_a/*' 限定单个子文件夹；"
        "'char_a/*|char_b/*' 进行 OR 组合。"
    ),
    "preprocess_run_mask": "运行蒙版生成",
    "preprocess_run_sam_mask": "运行 SAM 蒙版",
    "preprocess_run_sam_mask_tip": (
        "在蒙版生成阶段运行 SAM3 气泡分割。"
        "取消勾选则跳过 SAM,仅使用 MIT (或其他已启用的后端)。"
    ),
    "preprocess_run_mit_mask": "运行 MIT 蒙版",
    "preprocess_run_mit_mask_tip": (
        "在蒙版生成阶段运行 MIT/ComicTextDetector 文字分割。"
        "取消勾选则跳过 MIT,仅使用 SAM。"
    ),
    "preprocess_mask_nothing_enabled": ("SAM 和 MIT 蒙版至少需启用一项。"),
    "preprocess_status_resized": "已调整大小的图像: {n}",
    "preprocess_status_caches": "缓存 — latents: {lat}, text: {te}, PE: {pe}",
    "preprocess_status_masks": "蒙版: {masks}",
    "preprocess_status_no_resized": "尚无已调整大小的图像 —— 请先在训练配置选项卡运行预处理。",
    "preprocess_log_placeholder": "预处理输出将显示在此处……",
    "preprocess_save_settings": "保存",
    "preprocess_save_settings_tip": "持久化这些设置 (写入 configs/preprocess.toml + configs/sam_mask.yaml + GUI 设置)。",
    "preprocess_settings_saved": "预处理设置已保存。",
    "preprocess_invalid_float": "{field} 的数字无效: {value}",
    "preprocess_already_running": "已有预处理步骤在运行。",
    # ConfigTab
    "preset": "预设:",
    "save": "保存",
    "save_dirty_tooltip": "表单有未保存的编辑。点击保存将写入 variant 文件 (训练 / 预处理会在跳过时自动保存)。",
    "train": "训练",
    "queue": "加入队列",
    "queue_tooltip": "将当前变体加入守护进程队列而不绑定此标签页，以便继续排队更多变体。",
    "test": "测试",
    "stop": "停止",
    "log_placeholder": "训练输出将显示在此处……",
    "from_base": "继承自 base.toml",
    "saved": "已保存",
    "saved_file": "已保存 {name}",
    "invalid_toml": "无效的 TOML",
    "config_bad_keys_header": "未知的数据集键 — 删除这些键之前训练将会失败:",
    "config_remove_keys_btn": "删除",
    "config_remove_keys_confirm": "从配置文件中删除这 {n} 个过时的键?\n\n{keys}",
    "config_remove_keys_none": "未删除任何键(磁盘上对应的行可能已更改)。",
    "error": "错误",
    "accelerate_not_found": "在 PATH 中找不到 accelerate",
    "preprocess": "预处理",
    "preprocess_required": "训练前请先运行预处理。",
    "preprocess_existing_caches_title": "将复用现有缓存",
    "preprocess_existing_caches_body": (
        "以下路径已存在缓存文件:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "预处理将复用这些缓存 —— 不会删除或重新生成。"
        "只会处理缺失的条目。\n\n"
        "若想强制完全重建 (例如修改了标注或更改了 tokenizer 设置),"
        "请取消并先手动删除缓存目录。"
    ),
    "preprocess_cache_count_latents": "{n} 个 VAE 隐变量 (.npz)",
    "preprocess_cache_count_te": "{n} 个文本嵌入 (_te.safetensors)",
    "preprocess_cache_count_pe": "{n} 个 PE 特征 (_pe.safetensors)",
    "train_using_cache_title": "使用现有缓存数据集?",
    "train_using_cache_body": (
        "以下路径已存在预处理过的数据集缓存:\n  {cache_dir}\n\n"
        "{items}\n\n"
        "训练将原样复用此缓存。若你添加了新图像或修改了标注并希望生效,"
        "请取消并先运行预处理。\n\n"
        "用现有缓存继续训练吗?"
    ),
    "train_autopreprocess_log": (
        "未找到预处理缓存 —— 将先运行预处理,然后自动开始训练。\n"
    ),
    "train_preprocessing": "预处理中……",
    "no_lora_for_test": "output/ckpt/ 中没有可测试的 LoRA。请先运行训练。",
    "test_output_title": "最新测试输出",
    "test_output_empty": "output/tests/ 为空。",
    "finished": "--- 完成 (退出码 {code}) ---",
    "starting": "启动中…… (加载 torch / accelerate)",
    "queue_submitting": "正在将 {variant} 加入训练守护进程队列……",
    "queue_added_train": "已将 {variant} 作为训练任务 {job_id} 加入队列。\n",
    "queue_added_preprocess": "已将 {variant} 作为预处理任务 {job_id} 加入队列；完成后将自动链接训练。\n",
    "queue_refresh": "刷新",
    "queue_stop_selected": "停止所选",
    "queue_copy_output": "复制输出",
    "queue_status": "运行/等待 {live} 个 / 共 {total} 个",
    "queue_daemon_unavailable": "守护进程不可用",
    "queue_detail_placeholder": "选择一个队列项目以查看详情。",
    "queue_log_placeholder": "所选任务的输出将显示在此处……",
    "queue_log_missing": "(暂无输出日志。)",
    "queue_log_read_failed": "(无法读取输出日志: {err})",
    "queue_log_truncated": "--- 仅显示最后 {mb} MB 的输出 ---\n",
    "queue_detail_id": "id: {job_id}",
    "queue_detail_state": "状态: {state}",
    "queue_detail_kind": "类型: {kind}",
    "queue_detail_method": "对象: {method}",
    "queue_detail_submitted": "添加时间: {time}",
    "queue_detail_started": "开始时间: {time}",
    "queue_detail_ended": "结束时间: {time}",
    "queue_detail_from_chain": "from_chain: true",
    "queue_detail_chain": "链接训练: {method}",
    "queue_detail_chained_id": "链接任务: {job_id}",
    "queue_detail_pid": "pid: {pid}",
    "queue_detail_error": "错误: {error}",
    "queue_detail_status_detail": "详情: {detail}",
    "queue_detail_config": "配置快照: {path}",
    "queue_detail_stdout": "stdout: {path}",
    "daemon_job_failed": "--- Job {job_id} {state}: {error} ---",
    "daemon_error_cause": "↳ 可能原因: {summary}",
    "update_success_title": "更新已应用",
    "update_success_message": (
        "anima_lora 已更新至 {v}。\n\n请关闭并重新启动 GUI 以加载新代码。"
    ),
    "update_success_badge": "已更新 → {v} (需重启生效)",
    "update_dryrun_done_title": "试运行结束",
    "update_dryrun_done_message": (
        "试运行已完成 —— 未写入任何文件。查看日志可了解真实更新会改动什么。"
    ),
    "update_failed_title": "更新失败",
    "update_failed_message": (
        "更新以退出码 {code} 退出。详情请查看日志;工作树可能已部分修改。"
    ),
    "resume_checkpoint_title": "继续训练?",
    "resume_checkpoint_question": (
        "在第 {step} 步检测到可恢复的检查点。\n\n"
        "• 是 —— 从第 {step} 步继续训练\n"
        "• 否 —— 丢弃检查点,从头开始\n"
        "• 取消 —— 不启动训练"
    ),
    "resume_checkpoint_delete_failed": "无法删除旧检查点状态:\n{error}",
    "locked_by_preset": "由预设锁定 (此 VRAM 档位的性能设置是固定的)",
    "lora_variants": "LoRA 变体",
    "variant": "变体:",
    "apply_variant": "应用",
    "apply_variant_tooltip": "用此变体的预设值填充下面的表单。点击「保存」前不会落盘。",
    "show_guide": "指南",
    "show_guide_tooltip": "在右侧面板显示变体指南和「应用」语义说明。",
    "click_field_for_help": "点击字段标签可在此处查看说明。",
    "no_help_available": "此字段无可用说明。",
    "extra_args_toggle": "+ 额外参数",
    "extra_args_placeholder": "表单中没有的字段用 TOML 行表示,例如:\nmy_new_flag = true\nsome_value = 5e-5",
    "extra_args_tooltip": "添加表单中未显示的配置键。保存时按 TOML 解析并合并到当前变体文件,表单会重新加载以将新键显示为控件。若同一键同时出现在表单和此处,此处优先。",
    "new_variant": "+ 新建",
    "new_variant_tooltip": "在 configs/gui-methods/custom/<name>.toml 下创建新的自定义变体。",
    "new_variant_prompt": "新变体的名称 (将保存到 configs/gui-methods/custom/<name>.toml)。\n仅允许字母、数字、_ 和 -。",
    "new_variant_invalid": "名称无效。仅允许字母、数字、_、-。",
    "new_variant_exists": "变体 '{name}' 已存在。",
    "basic_section": "基本",
    "advanced_section": "高级 (点击展开)",
    # AdapterTab (IP-Adapter / EasyControl)
    "adapter_source_dir": "源数据集:",
    "adapter_cache_dir": "缓存目录:",
    "adapter_n_pairs": "{n} 张图像 / {c} 条标注配对",
    "adapter_n_caches": "已缓存 {n} 项",
    "adapter_preprocess": "预处理 (调整大小 + VAE + 文本)",
    "adapter_preprocess_pe": "预处理 (调整大小 + VAE + 文本 + PE)",
    "adapter_train": "训练",
    "adapter_stop": "停止",
    # SPD / Turbo 蒸馏配置标签页 (gui/tabs/distill_tab.py)
    "distill_general_section": "通用",
    "distill_job_running": "此标签页已有任务正在运行。",
    "distill_config_missing": "无法读取配置文件: {err}",
    "adapter_log_placeholder": "运行输出将显示在此处……",
    "adapter_no_dataset": "源数据集目录不存在。请创建该目录并放入图像 + 标注配对。",
    "adapter_open_dir": "打开目录",
    "n_images": "{n} 张图像",
    # ImageViewerTab
    "directory": "目录:",
    "dataset_reload": "重新加载",
    "dataset_reload_tooltip": "重新扫描当前目录并刷新图像列表和选择。",
    "dataset_open_dir": "打开",
    "dataset_open_dir_tooltip": "在系统文件管理器中打开当前目录。",
    "dataset_add_dir": "添加目录……",
    "dataset_add_dir_tooltip": "选择另一个目录并在本次会话中加入下拉框。",
    "dataset_add_dir_picker": "选择要添加的目录",
    "dataset_add_dir_already": "目录 '{name}' 已在列表中。",
    "dataset_search_placeholder": "搜索文件名……",
    "dataset_sort_asc_tooltip": "升序 A→Z (点击反转)",
    "dataset_sort_desc_tooltip": "降序 Z→A (点击反转)",
    "dataset_mask_overlay": "显示蒙版覆盖",
    "dataset_view_list_tooltip": "平铺列表视图 (点击切换为树状视图)",
    "dataset_view_tree_tooltip": "文件夹树视图 (点击切换为列表视图)",
    "n_images_filtered": "{shown} / {total} 张图像",
    "caption": "标注:",
    "no_caption": "(无标注)",
    "caption_save": "保存",
    "caption_revert": "还原",
    "caption_versions": "历史……",
    "caption_dirty_marker": " *",
    "caption_diff_stats": "(+{add} / −{rem})",
    "caption_diff_clean": "(无变化)",
    "caption_save_failed": "保存标注失败: {err}",
    "caption_unsaved_title": "未保存的标注",
    "caption_unsaved_body": "标注编辑尚未保存。切换前先保存吗?",
    "caption_versions_title": "标注历史 — {name}",
    "caption_versions_empty": "(无历史版本)",
    "caption_versions_restore": "恢复所选版本",
    "caption_versions_close": "关闭",
    "caption_no_history": "此标注尚无历史记录。",
    "caption_guideline_html": (
        "<b>顺序:</b> 评级 → 人数 → 角色 (作品) → 作品 → "
        "<span style='color:#c9a227;'>@艺术家</span> → 内容标签。"
        "区域子分节: 在前一个标签末尾加上 <code>.</code>,然后用 "
        "<span style='color:#5e8eb0;'>On the&nbsp;…,</span> 或 "
        "<span style='color:#5e8eb0;'>In the&nbsp;…,</span> 开始下一节。"
        "首个 <code>@艺术家</code> 标签 (含) 之前的顺序保持固定,"
        "其后的标签在各分节内打乱。"
        "<b>没有艺术家?</b> 用 "
        "<span style='color:#c9a227;'>@no-artist</span> 作为占位符 —— "
        "它仅起到锚定打乱边界的作用,会在 tokenize 之前剥离,因此不会进入模型。"
    ),
    # Language
    "language": "语言:",
    # Guidebook
    "guidebook": "📖 指南书",
    "guidebook_tooltip": "打开中文综合指南 (docs/guidelines/指南书.md)",
    "guidebook_missing": "在 {path} 找不到指南",
    "guidebook_open_external": "用系统查看器打开",
    "guidebook_close": "关闭",
    # EasyControl 适配器指南（自建控制任务）
    "adapter_guide": "📘 适配器指南",
    "adapter_guide_tooltip": "如何构建你自己的 EasyControl 适配器 (easycontrol_adapters/ADAPTER_GUIDE.md)",
    "easycontrol_descriptor_note": "此控制任务是一个具有多表结构的独立描述符，在左侧以原始 TOML 形式编辑:<br><br>• <b>name</b> — 输出标识符；重路由所有衍生的缓存/输出路径。<br>• <code>[staging]</code> — 用于生成条件树的数据生成阶段。<br>• <code>[preprocess]</code> — 针对已暂存树的 VAE/TE 缓存参数。<br>• <code>[training]</code> — 叠加到基础 EasyControl 方法的覆盖项。<br>• <code>[general]</code> / <code>[[datasets]]</code> — train.py 读取的数据集蓝图。<br>• <code>[variant]</code> — 此下拉项的 GUI 元数据。<br><br><b>预处理</b>按钮会合成条件树并缓存；<b>训练</b>会将此描述符的 <code>[training]</code> 覆盖项叠加后训练基础 EasyControl 方法。两个操作在 GUI 关闭后仍会继续执行。",
    "easycontrol_descriptor_form_header": "正在编辑描述符 <b>{path}</b>。下方的参数表以表单形式编辑；保存时将变更值写回，同时保留注释和 <code>[[datasets]]</code> 蓝图。蓝图与 <code>[variant]</code> 元数据不在此处显示 —— 如需编辑请直接修改文件。点击字段名称可查看帮助。",
    "ec_desc_group_top": "描述符",
    # Top-bar buttons (models / update / report issue)
    "models_btn": "模型",
    "models_btn_tooltip": "下载或重新下载模型检查点 (Anima 基础、SAM3、MIT、IP-Adapter 编码器)",
    "update_btn": "更新",
    "update_btn_tooltip": "从 GitHub 拉取最新 anima_lora 版本并运行 uv sync",
    "update_btn_available": "更新 ●",
    "update_btn_available_tooltip": "有新版本 {v} 可用 — 点击查看发布说明",
    "report_issue": "提交问题",
    "report_issue_tooltip": "在浏览器中打开 GitHub 问题追踪",
    "experimental_features": "🧪 实验功能",
    "experimental_features_tooltip": "打开 Postfix 和 IP-Adapter / EasyControl 选项卡 (图像条件方法)",
    "experimental_features_title": "实验功能",
    # Models dialog
    "models_title": "下载模型",
    "models_intro": "在下方选择模型组,或使用「全部下载」获取标准套件 "
    "(Anima + SAM3 + MIT + PE)。文件保存于 models/ 下。",
    "models_download_all": "全部下载 (Anima + SAM3 + MIT + PE)",
    "models_download": "下载",
    "models_redownload": "重新下载",
    "models_installed": "✓ 已安装",
    "models_missing": "✗ 缺失",
    "model_anima": "Anima — DiT + 文本编码器 + VAE",
    "model_sam3": "SAM3 — 对话气泡蒙版",
    "model_mit": "MIT — 漫画文字蒙版",
    "model_pe": "PE-Core-L14-336 — IP-Adapter 视觉编码器",
    # HuggingFace 认证（模型对话框）
    "models_hf_token_placeholder": "粘贴你的 HuggingFace 令牌 (hf_…)",
    "models_hf_authenticate": "认证",
    "models_hf_token_hint": "用于受限/限速下载（例如 SAM3）。"
    '请在 <a href="https://huggingface.co/settings/tokens">'
    "huggingface.co/settings/tokens</a> 创建令牌，并在 "
    '<a href="https://huggingface.co/facebook/sam3">huggingface.co/facebook/sam3</a> 申请 SAM3 访问权限。',
    "models_hf_token_present": "✓ 已保存 HuggingFace 令牌。",
    "models_hf_not_authenticated": "未认证 — 粘贴令牌以启用受限下载。",
    "models_hf_token_empty": "请先粘贴令牌。",
    "models_hf_authenticating": "正在认证…",
    "models_hf_logged_in": "✓ 已登录为 {name}。",
    "models_hf_login_failed": "认证失败：{err}",
    # Update dialog
    "update_title": "更新 anima_lora",
    "update_warning": "更新会从 GitHub 拉取最新版本并覆盖工作树 "
    "(datasets、output/、models/ 会保留)。对于 configs/methods/ "
    "和 configs/gui-methods/,可选择保留你的修改或用上游覆盖 "
    "(原文件会先备份)。可先用「试运行」预览改动。",
    "update_dry_run": "试运行",
    "update_run": "执行更新",
    "update_run_keep": "更新 —— 保留我的配置",
    "update_run_overwrite": "更新 —— 覆盖配置 (备份原文件)",
    "update_confirm": "这将重写 anima_lora 源文件。继续吗?",
    "update_check_now": "立即检查",
    "update_view_release": "在 GitHub 上查看",
    "update_current_version": "当前: {v}",
    "update_latest_version": "最新: {v}",
    "update_no_baseline": "未知 (无 manifest)",
    "update_status_checking": "检查中……",
    "update_status_uptodate": "✓ 已是最新",
    "update_status_available": "● 有可用更新",
    "update_status_unknown": "? 无法比较 (无本地 manifest)",
    "update_status_failed": "✗ 检查失败",
    "update_release_notes": "版本说明:",
    "update_no_release_notes": "(本次发布无说明)",
    "update_check_error": "无法连接 GitHub: {err}",
    # MergeTab
    "n_files": "{n} 个文件",
    "merge_no_adapter": "未找到适配器",
    "merge_no_adapter_msg": "未选择适配器或文件不存在。",
    "merge_no_selection": "从列表中选择一个检查点以扫描它。",
    "merge_verdict_ready": "✓ 可合并",
    "merge_verdict_hydra": "✗ HydraLoRA moe —— 层级局部路由无法合并",
    "merge_verdict_postfix_only": "✗ 仅 postfix/prefix —— 不是权重增量",
    "merge_verdict_unknown": "? 未识别的适配器键",
    "merge_options": "合并选项",
    "merge_base_dit": "基础 DiT:",
    "merge_multiplier": "乘数:",
    "merge_multiplier_tip": "要烘焙的 LoRA 强度 (1.0 = 全强度)。",
    "merge_dtype": "保存 dtype:",
    "merge_out": "输出:",
    "merge_out_placeholder": "(自动: <adapter>_merged.safetensors)",
    "merge_allow_partial": "允许部分合并 (丢弃 Hydra / postfix 键)",
    "merge_allow_partial_tip": "即使适配器包含不可合并的组件也继续。被丢弃的组件不会出现在合并后的 DiT 中。",
    "merge_button": "合并到 DiT",
    "merge_log_placeholder": "合并输出将显示在此处……",
    "merge_pick_dir": "选择适配器目录",
    "merge_pick_file": "选择适配器 .safetensors",
    "merge_pick_dit": "选择基础 DiT .safetensors",
    "merge_pick_out": "另存为合并后的 DiT...",
    "browse": "浏览……",
    # Multi-scale target_res tiers
    "target_res_danger_tooltip": "高开销档位：{edge}px 每张图像约使用 {tokens} 个 token，并额外增加一张已编译的块图（编译更慢、显存更高）。仅在确实需要该分辨率时才启用。",
    # TensorBoard panel
    "tb_panel_title": "TensorBoard 运行列表",
    "tb_open": "打开 TensorBoard",
    "tb_stop": "停止服务器",
    "tb_remove": "删除",
    "tb_view": "查看",
    "tb_view_tip": "仅打开此运行的 TensorBoard。",
    "tb_no_runs": "暂无运行记录，开始训练后列表将自动填充。",
    "tb_status_running": "正在端口 {port} 上运行",
    "tb_status_stopped": "",
    "tb_not_installed": "未安装 tensorboard，请运行: pip install tensorboard",
    "tb_current_run_label": "（当前）",
    "tb_open_current": "查看当前训练",
    "tb_open_current_tip": "仅打开正在进行的训练运行的 TensorBoard。",
    "tb_open_current_idle_tip": "训练进行中时可用。",
    "tb_appear_hint": "如果运行未显示在列表中，请尝试点击 TensorBoard 的刷新（更新）按钮。",
}
