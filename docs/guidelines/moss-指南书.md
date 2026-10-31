# MOSS-Image LoRA 指南书

在 GUI 中为 MOSS-Image 训练和测试 LoRA 的指南。用 `make gui` 打开，把顶栏的 **模型** 下拉框切换到 **MOSS-Image** 即可（`make gui-moss` 会在独立窗口中打开同一个面板）。GUI 语言为中文时面板显示中文，其他语言时显示英文。下文的界面名称按中文面板书写，括号中是字段名；面板未翻译的字段直接显示字段名。

MOSS 线目前处于 **alpha** 阶段：迄今只有一个风格数据集（同一画师的 30 张图），并在其上做了大量 A/B，参数和默认值仍可能变化。下文的数字来自这些 A/B 以及在 16 GB RTX 5070 Ti 上的基准测试。

---

## 目录

1. [MOSS-Image 是什么](#1-moss-image-是什么)
2. [安装](#2-安装)
3. [预处理、训练、测试](#3-预处理训练测试)
4. [推荐设置](#4-推荐设置)
5. [测试](#5-测试)
6. [常见问题](#6-常见问题)

- [附录. CLI](#附录-cli)

---

## 1. MOSS-Image 是什么

[MOSS-Image](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)（OpenMOSS，Apache-2.0）是与 Anima 不同的模型，本仓库把它作为一条独立的线来训练。

| | Anima | MOSS-Image |
|---|---|---|
| 扩散模型 | Anima DiT | 5B single-stream DiT（42 个块） |
| 文本编码器 | Qwen3 0.6B | Qwen3.5-9B |
| VAE | QwenImage VAE | FLUX.2 VAE |
| 缓存 | `post_image_dataset/lora/` | `post_image_dataset/moss/` |
| LoRA | `output/ckpt/` | `output/ckpt_moss/` |
| 设置 | variant TOML + Hardware 预设 | 面板中的字段 |

- 缓存、设置和 LoRA 互不通用：MOSS LoRA 不能加载到 Anima 上，Anima LoRA 也不能加载到 MOSS 上。Anima 的变体、预设和 `make lora` 在这里都不适用。
- MOSS 提供两个 DiT。**Base** 是预训练模型；**SFT** 是在其基础上的微调，画风偏光泽感强的手游风。请在 Base 上训练。在 Base 上训练的 LoRA 也能用于 SFT：它会去掉 SFT 的大部分光泽感，效果接近 Base + LoRA。

---

## 2. 安装

> 请先按通用[安装指南](../installation/安装.md)安装本仓库。

| 包 | 内容 | 大小 |
|---|---|---|
| `moss-base` | DiT、文本编码器、VAE、分词器 | 约 27 GB |
| `moss-sft`（可选） | 仅 SFT 的 DiT，文本编码器和 VAE 复用 Base 的 | 约 10 GB |

在 GUI 的「下载模型」窗口中下载，或运行：

```bash
make download-model ARGS=moss-base
make download-model ARGS=moss-sft
```

权重保存在 `models/moss_image/MOSS-Image-Base/` 和 `models/moss_image/MOSS-Image-SFT/`。若要放在其他位置，可在 `.env` 中设置 `ANIMA_MOSS_MODEL_DIR`，或在面板的 **高级** 中设置 模型文件夹（`model_dir`，即包含两个 `MOSS-Image-*` 文件夹的目录）。文件夹就位之前，面板会显示「找不到模型：…」。

默认的标注格式（`layout_json`，§3.1）还需要 danbooru 标签 CSV。`make download-models` 已经包含它；单独下载用 `make download-model ARGS=danbooru_tags`。

---

## 3. 预处理、训练、测试

面板有 **预处理** 和 **训练** 两个标签页，**测试** 区域位于训练页底部。不常用的字段折叠在 **高级** 中。每个按钮都会向训练守护进程提交一个任务：把模型下拉框切回 Anima 或关闭 GUI，任务都会继续运行；重新打开时会自动重新连接（「已重新连接到正在运行的任务 …」）；**停止** 用于取消任务。

### 3.1 数据集与标注

图片加同名 `.txt` 标注的文件夹，包括子文件夹。默认来源是 Anima 的 resized 目录 `post_image_dataset/resized`（含修订后的标注），所以已经为 Anima 预处理过的数据集可以直接使用。只想训练其中一组时，把 图片文件夹（`src`）指向对应的子文件夹。

- **Anima 风格的标签标注可以直接用于训练和提示词。** MOSS 本身是用自然语言训练的；长标注或角色很多的标注尚未验证。
- **`layout_json`（默认）**：把 Anima 标注转换成 MOSS 的 layout-JSON 提示词。rating/meta、角色、作品和画师各占一个键，其余标签成为 caption，每个位置子句变成带位置信息的 subject，每行 OCR 文字变成 text 项。在 30 张图的数据集上，它与普通标签的风格强度相同，而数据集属性泄漏到其他角色上的情况更少。
- **训练和提示词使用同一格式**：`caption_format`（预处理）与 `prompt_format`（测试）必须一致——都用 `layout_json`，或普通标注都用 `text`。无论哪种，提示词都按 Anima 标签格式输入，`layout_json` 会自动转换。
- **触发词就是标注中原样出现的画师标签**，例如 `@channel (caststation)`。在提示词中按同样的写法使用。
- 自然语言提示词经过 `layout_json` 后作为一整段 caption 字符串传入，效果与纯文本提示词相同。
- 在 ComfyUI 中，设为 `task = layout` 的 **MOSS Caption Helper** 节点（[MOSS Toolkit](https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit) 包）会在文本编码器前做同样的转换。

### 3.2 预处理

预先缓存每张图片的 Qwen3.5-9B 文本嵌入和 FLUX.2 VAE 潜变量，训练时不再加载这两个模型。

| 字段 | 默认值 | 作用 |
|---|---|---|
| 图片文件夹（`src`） | `post_image_dataset/resized` | 图片 + 标注文件夹 |
| 缓存文件夹（`out`） | `post_image_dataset/moss` | 缓存文件夹：每张图片两个文件，平铺存放 |
| 分辨率（`resolution`） | `1024` | 像素面积上限：按原始比例最多约 1024² 像素，两边均为 16 的倍数，不放大 |
| `caption_format` | `layout_json` | `layout_json` 或 `text`（§3.1） |

点击 **开始预处理**。状态栏会显示图片、标注和已缓存文件的数量。重复运行是增量的：新增图片会被加入，缓存之后修改过的标注会重新编码（状态栏会显示数量），更改 `caption_format` 后文本缓存会重新编码。不同子文件夹中同名的图片，缓存文件名会加上文件夹前缀。

### 3.3 训练

| 字段 | 默认值 | 作用 |
|---|---|---|
| 缓存文件夹（`cache`） | `post_image_dataset/moss` | 预处理的输出 |
| 输出（`output`） | `output/ckpt_moss/moss_lora.safetensors` | LoRA 路径；同目录下写入 `.json` 运行报告（逐步 loss） |
| 模型（`model`） | `base` | 在哪个 DiT 上训练 |
| 轮数（`epochs`） | `8` | 遍历缓存的次数（每步一张图） |
| 每 N 轮保存（`save_every_epochs`） | `0` | 每 N 轮额外保存到最终文件旁的 `<name>/<name>_e10.safetensors` 等（最后一轮即最终文件本身） |

秩、alpha、学习率和适配器类型见 §4。**开始训练** 读取 缓存文件夹（`cache`）；**预处理后训练** 先运行预处理，再把 `cache` 指向其输出文件夹并开始训练。

### 3.4 测试

用同一个种子渲染两次提示词：一次加载强度为 LoRA 强度（`lora_multiplier`）的 LoRA，一次不加载。两张图在面板中并排显示，差别只来自 LoRA。

- `lora` 留空 = 训练页的 输出（`output`）。要测试其他文件（例如 `_e10` 检查点），在这里指定。
- 模型（`model`）设为 `sft` 即可在 SFT 上测试 Base LoRA。
- 输出：`output/tests/<日期-时间>/lora/` 和 `…/base/`，各带一个 `manifest.json`。每次测试新建一个文件夹，之前的结果会保留。

---

## 4. 推荐设置

对比在同一画师（`@channel (caststation)`）的 30 张图上进行，Base，20 轮 = 600 步（约 18–21 分钟）。

| 字段 | 默认值 | 说明 |
|---|---|---|
| 适配器（`algo`） | `lora` | `lokr` / `loha`：§4.1 |
| 秩（`rank`） | `32` | 秩 16 时参数量 37.2 M（此前的默认值） |
| `alpha` | `lora` 为 `128`（秩 32 时缩放 4）；`lokr` / `loha` 为 = 秩 | 缩放 = alpha / 秩 |
| 学习率（`lr`） | `5e-5` | 5 % 线性预热 |
| 优化器（`optimizer`） | `muon` | `adamw` 是此前的默认值；`lokr` / `loha` 始终使用 AdamW |
| down 初始化（`down_init`） | `kaiming` | `kaiming` 或 `weight_svd` |
| 轮数（`epochs`） | `8` | 实测训练用 20 |
| `plain_caption_rate` | `0.2` | 仅 `layout_json` 缓存：用普通 Anima 标注而非 layout JSON 训练的步数占比 |

- **学习率、通道缩放、DC 隔离**：FFN 输入处有一个输入通道比其余通道强得多。在 AdamW 下曾用两种修正（通道缩放和 `dc_isolate`）避免 LoRA 通过它把更新浪费在全局平移上。在 Muon 默认设置并使用 kaiming `down` 初始化时，两者都不改变训练出的 LoRA，因此都已移除。此前的默认值以及每次改动背后的对比见 `docs/moss/channel_scaling.md`。
- **optimizer**：Muon（MOSS 预训练所用的优化器）在相同学习率下让 LoRA 的移动幅度与 AdamW 相同，但分散到更多方向上，同一随机种子下看起来更好。它自身也能避免把更新浪费在全局平移上。`docs/moss/findings/muon_lora.md`。
- **down 初始化**：`kaiming` 是随机初始化，`weight_svd` 用基础权重的前几个奇异向量初始化 LoRA 的 down 矩阵。这些向量对输入的恒定偏移读取很强，搭配 Muon 时，`weight_svd` 的 LoRA 让每个 token 的输出平移幅度约为 `kaiming` 的三倍。图像差异约等于换一个种子。`docs/moss/findings/muon_channel_init.md`。
- **轮数**：20 轮训练中，第 10 轮的结果更松散、更多样；第 20 轮更干净、更统一，但略微偏向正面居中的半身构图。设置 每 N 轮保存（`save_every_epochs`）即可在一次训练中比较两者。
- **sigma_\***（高级）：默认值是 MOSS 最终预训练阶段的噪声分布。Anima 的分布（`sigma_mu 0`、`sigma_s 1`、`sigma_cutoff 1.0`、`sigma_uniform 0`）在所有设置中迁移最弱。

### 4.1 LoKr 与 LoHa

适配器（`algo`）= `lokr`（Kronecker 积）或 `loha`（两组秩 r 矩阵的 Hadamard 积），来自 LyCORIS。文件保留 LyCORIS 的张量名，通过同一个 `lora` 字段加载。`down_init` 只对 `lora` 生效。

| 配置 | 字段 | 速度（16 GB，1024²） | 画师数据集、学习率 1e-4 的结果 |
|---|---|---|---|
| LoKr f8 | `rank 100000`，`factor 8` | 2.26 s/step，剩余 3.6 GB | 接近 LoRA，面部的个性略少 |
| LoHa r16 | `rank 16` | 2.49 s/step，剩余 1.7 GB | 过头：色调偏灰青，数据集的服装出现在其他角色身上 |

- 这些数据是在 bf16 方案（§4.2）上测得的，LoHa 需要该方案：默认的 int8 基座会拒绝 LoHa。LoKr 可在 int8 上运行，但其峰值未测量。
- 在 bf16 方案上，两者都比 LoRA 占用更多显存：在 **高级** 中设置 `activation_memory_budget 0.1`，并让 `blocks_to_swap` 保持在 24 或更高。swap 12 配 budget 0.15 在 1024² 下会显存不足。
- LoKr 的 `rank` 达到较大因子边长的一半及以上时（如上面的 `rank 100000`），该因子为满矩阵，alpha 被忽略（缩放为 1）。
- LoHa：下一个要尝试的是学习率 5e-5，尚未实测。

### 4.2 显存方案（16 GB）

默认：梯度检查点（`grad_checkpointing`）关闭，`compile` 开启，**高级** 中 `int8 stream` 配合 GPTQ 文件（DiT 的块权重以 int8 运行，文件首次使用时下载）、`stream fp32`（块的其余部分以 fp32 运行）、`blocks_to_swap 0`（所有块都留在显卡上）、`activation_memory_budget 0.2`（前向结果中为反向保留的比例，其余重新计算）、`ema_card` 开启以及 `expandable_segments` 开启（序列长度变化时避免 CUDA 显存池碎片化）。

该方案在 1024² 样本上为 1.30 s/step，峰值 12.04 GB，并且还会锁定 4.4 GiB 系统内存。在此方案上 budget 买不到速度：0.05–0.4 速度相同，每提高 0.1 峰值约增加 1 GB，所以显存不足时请调低。目前还没有用它完整训练出 LoRA（[详情](../moss/optimization/int8.md)）。

**bf16 方案。** int8 基座拒绝 block swap 和 LoHa，也没有跑过 i2i 步骤，所以 i2i 任务配置仍用 bf16。使用 `--config configs/moss/lora_bf16.toml`，或在 **高级** 中设置 `int8 off`、`stream bf16`、`ema_card` 关闭、`blocks_to_swap 36`（放在 CPU 上、用时再传入的 DiT 块数）、`activation_memory_budget 0.5`。

| 方案 | 字段 | s/step | 剩余显存 |
|---|---|---|---|
| 此前的 bf16 默认 | swap 12，budget 0.1（秩 32） | 1.85 | 约 1.7 GiB |
| 更早的 bf16 默认 | swap 12，budget 0.15（秩 16） | 1.77 | 约 0.7 GiB |
| 留更多余量 | swap 24，budget 0.2 | 1.78 | 2.26 GiB |
| 检查点 | 梯度检查点开启，swap 0 | 2.09 | 1.60 GiB |

bf16 方案（swap 36，budget 0.5，`swap_h2d_only`，`expandable_segments`，秩 32）在 caststation 数据集上为 1.9 s/step，最大样本（图像 4.2k + 文本 0.4k token）剩余 0.85 GB。budget 超过约 0.45 后再提高也没有帮助（0.3：2.05 s/step），swap 28–36 速度相同，所以调低 swap 只会损失剩余显存。i2i 样本还带有参考图，因此 i2i 任务配置使用 swap 38、budget 0.175：swap 28 时 budget 0.2 会显存不足，而 swap 38 能为更高的 budget 腾出空间；在约 8.3k 联合 token（sanitize）下为 3.79 s/step，swap 28 / budget 0.15 则为 4.17 s/step。`swap_h2d_only` 会在 RAM 中为每个被换出的块保留一份锁页副本（最多 8.8 GB），这样换出只需复制到显卡，无需再复制回来。结果和显存与不开启时相同；swap 38 以内速度相同，swap 40 时快 8%。默认开启，且只在有块被换出时起作用，可用 `--no-swap_h2d_only` 关闭。

表格测试条件：合成 1024² 图像 + 300 个文本 token（共约 4.4k token），LoRA 秩 16，RTX 5070 Ti。此前的默认值（秩 32，budget 0.1，swap 12）在约 4.5k token（图像 4096 + 文本 455）下实测 1.85 s/step，剩余约 1.7 GB，峰值 reserved 约 13.1 GB；秩 16、budget 0.1 为 1.82 s/step，剩余约 2.0 GB。真实样本最多约 4.9k token，标注越长显存占用越多，`swap 24` 在相同速度下留出了这部分余量。

- swap 在 8 到 36 之间速度几乎不变（1.75–1.84 s/step），budget 0.1–0.3 对速度也几乎没有影响；swap 40 时 PCIe 传输无法被计算掩盖，变慢（2.34 s/step）。在此之下，swap 数决定的是显存，不是速度。
- 最佳组合取决于 PCIe 链路和 GPU。`scripts/moss/tune_swap.py`（CLI，[附录](#附录-cli)，约 4 分钟）会用缓存中最大的样本在 bf16 方案上测试各个 swap 数的速度，并输出训练应使用的字段值。
- `blocks_to_swap -1` 会根据剩余显存决定 swap 数；它按 LoRA 校准，对 LoKr/LoHa 预留不足。
- 关闭 `compile` 需要开启梯度检查点。开启 compile 的训练为 2.0 s/step（关闭时 3.1），开始时约编译 30 秒。

---

## 5. 测试

采样默认值来自 MOSS 的参考管线，Anima 的采样习惯在这里不适用。

| 字段 | 默认值 | 作用 |
|---|---|---|
| CFG（`cfg`） | `3.0` | CFG 强度；`1.0` = 不做无条件前向 |
| 无条件分支跳过的块（`uncond_skip`） | `22` | 无条件分支是跳过第 22 块的模型（每步仍是两次前向）；`''` = 普通 CFG |
| `shift` | `5.0`（高级） | 调度器 shift |
| 步数（`steps`） | `50` | |
| 采样器（`sampler`） | `er_sde` | `er_sde` 或 `euler` |
| 宽度 / 高度（`width` / `height`） | `1024` | 16 的倍数 |
| 反向提示词（`negative_prompt`） | 空 | 管线默认值 |
| `dave` | 空 | `auto` = DAVE：同一提示词在不同种子下变化更多（构图、姿势、布局），代价是手部略松；见 `docs/moss/dave.md` |
| `dave_strength` | `0.3` | DAVE 去除的程度；`dave_tau`（高级，`0.10`）= 其生效的前期步数占比 |

- **er_sde 与 euler**：`euler` 是参考管线的采样器。`er_sde` 耗时相同，线条更锐利、细节更密；加载 LoRA 时手部也更干净。但它会偏离部分提示词细节（夕阳光、发色、"sitting on a bench"），也会损失一点数据集的画风。两者在同一种子下生成不同的图，所以 LoRA 与基础模型的比较要在同一采样器内进行。
- **Spectrum**（复选框，默认开启）：大部分步数预测模型输出，而不是运行全部 42 个块。50 步、默认 `spectrum_flex_window 0.5`（高级）时快 2.6 倍，多用 0.85 GiB 显存，在网格尺寸下与完整 50 步难以区分。`0.25` 为 2.2 倍，更保守；`1.0` 为 3.1 倍，但会丢失细节。
- **compile**（高级，默认开启）：无论是否开启 Spectrum，都再快约 1.37 倍，每个任务编译几秒。构图与未编译时相同，细节（发丝、花纹）会有差异，所以编译过的图只与编译过的图比较。

| Base，1024²，50 步 | 每张耗时 |
|---|---|
| 普通，未编译 | 72 秒 |
| 普通，编译 | 52 秒 |
| Spectrum，编译 | 20 秒 |

- **ComfyUI**：MOSS LoRA 通过 ComfyUI 的 LoRA 加载器加载到由 `moss-image` 节点得到的模型上（2026-10-17 之前保存的文件无法加载）。**KSampler (Spectrum for MOSS)**（Spectrum 节点包）带有这些采样默认值和 Spectrum，`ComfyUI-Anima-DAVE` 0.2.0 及以上版本可在 MOSS 上运行 DAVE。

**比较方法**：测试按钮本身就会用同一种子生成加载与不加载 LoRA 的两张图。比较两个 LoRA 或不同轮数时，固定提示词、种子、采样器和 compile 设置，只改 `lora`。

---

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| 训练时显存不足 | 高级：`activation_memory_budget 0.1`，或开启梯度检查点（int8 方案上峰值 9.36 GB，稍慢）。再不行用 `blocks_to_swap 24` 的 bf16 方案（§4.2），或运行 `tune_swap.py`；在该方案上 LoKr/LoHa 需要 24 / 0.1 |
| `int8 base: --blocks_to_swap 0 is required` 或 `loha builds W + ΔW` | int8 基座不支持 swap 和 LoHa：请使用 bf16 方案（§4.2） |
| 风格太弱 | 增加轮数（实测训练用 20）。确认提示词中的画师标签与标注写法一致，`prompt_format` 与 `caption_format` 一致 |
| 风格太强、出现数据集的签名或水印、所有图构图相同 | 测试更早的 `_e<N>` 检查点，或降低 LoRA 强度（`lora_multiplier`）。开始复制签名的正是提高了有效学习率的那一组；可尝试 `lr 2e-5` |
| 数据集的服装或颜色泄漏到其他角色 | 使用 `caption_format layout_json`（默认值，泄漏比普通标签少）。LoHa 改回 LoRA 或 LoKr |
| 更改了 `caption_format` | 重新运行预处理：文本缓存会重新编码，潜变量复用。状态栏的计数不会反映格式变化。把 `prompt_format` 改成一致后重新训练 |
| `layout_json needs the danbooru tag CSV` | `make download-model ARGS=danbooru_tags` |
| 测试时提示「找不到 LoRA：…」 | 先训练，或在测试区域中设置 `lora` |

---

## 附录. CLI

与 GUI 提交的任务相同，也可以直接通过守护进程运行。各脚本的 `--help` 列出全部参数。 `train.py --config configs/moss/lora.toml` 可从 TOML 文件读取参数（内含默认值，可复制后作为配方保存）；命令行直接给出的参数优先。

```bash
make daemon-run ARGS="--stall-timeout 900 scripts/moss/cache.py --src 'post_image_dataset/resized/<set>' --out 'post_image_dataset/moss/<set>'"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/lora.toml"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/lora_bf16.toml"
make daemon-run ARGS="--stall-timeout 900 scripts/moss/generate.py --lora output/ckpt_moss/moss_lora.safetensors"
make daemon-run ARGS="scripts/moss/tune_swap.py --cache post_image_dataset/moss/<set>"
```

LoKr / LoHa：

```bash
make daemon-run ARGS="scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20 --algo lokr --rank 100000 --factor 8 --config configs/moss/lora_bf16.toml --blocks_to_swap 24 --activation_memory_budget 0.1"
make daemon-run ARGS="scripts/moss/train.py --cache 'post_image_dataset/moss/<set>' --epochs 20 --algo loha --rank 16 --lr 5e-5 --config configs/moss/lora_bf16.toml --blocks_to_swap 24 --activation_memory_budget 0.1"
```

更多实测数据：[`docs/moss/README.md`](../moss/README.md)。
