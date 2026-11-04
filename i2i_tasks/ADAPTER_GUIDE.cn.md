# 构建你自己的 EasyControl 适配器

本指南说明如何为 Anima 添加一个**新的 EasyControl 控制任务**，供你自己使用。这是一个本地适配器，存放在 `i2i_tasks/<your_task>/` 下，不提交到 git。我们以示例 **colorize** (`i2i_tasks/colorization/`) 为例逐步讲解，并明确指出针对你自己的任务需要修改哪些地方。

记住一件事：

> **你不是在写模型代码。** 网络、前向传播、`b_cond` 门控、推理缓存——这些都已内置，由所有控制任务共享。每个任务唯一不同的地方是**如何构建参考图像**。本指南的其余内容都只是围绕这一点的配管工作。

---

## 0. 思路——EasyControl 适配器是什么

EasyControl 使用一张**参考图像**来引导生成。参考图像经过 VAE 编码成 *cond token*，与正在生成的图像并行流动。每一步，模型同时看到两者。（完整架构见 `docs/anima/experimental/easycontrol.md`。）

默认的 EasyControl 将**同一张图像**同时用作参考和目标——因此它学到的只是复制。控制任务打破了这一点：将每个目标与**不同的**参考配对，让模型学习 `reference → target`，而不是复制。

| | 默认 EasyControl | 控制任务（如 colorize） |
|---|---|---|
| 目标（你想要的） | 图像 X | 彩色图像 X |
| 参考（提示） | 图像 X（相同） | X 的*变换*版本（X 的黑白漫画） |
| 学到什么 | 复制 | manga → color |
| 文本 | 完整标注 | （可选）较短的标注 |

colorize 的做法——你可以复用的思路：**真实的黑白漫画没有彩色版本可供学习**，所以你无法收集 `(黑白, 彩色)` 配对。因此反转方向——把你已有的彩色图像作为目标（那是你想要的结果），并从每张图像**生成**黑白参考（线稿 + 网点，用算法实现）。关键是生成的黑白图像要看起来像你在推理时实际输入的黑白图像。如果你要构建 `depth → image` 或 `pose → image`，那么"生成参考"这一步就是在训练图像上运行深度估计器或姿态检测器。

所以你的全部工作是：**编写一个函数 `target_image → reference_image`，缓存其输出，让数据集指向它，在描述符中描述数据集，并注册任务名称。**

---

## 1. 你需要改动的四个地方

添加一个名为 `<task>` 的适配器，需要创建或编辑以下四个地方：

| # | 内容 | colorize 版本 | 作用 |
|---|---------|-------------------|--------------|
| 1 | `i2i_tasks/<task>/` 项目 | `colorization/`（`mangafy*.py`、`prep.py`） | 构建并缓存参考图像（以及可选的较短文本缓存） |
| 2 | `i2i_tasks/configs/<task>.toml` + `configs/easycontrol/<task>.toml` | `i2i_tasks/configs/colorize.toml` + `configs/easycontrol/colorize.toml` | 任务文件（`name` 标识符、可选的 `stage`、`[staging]`）和 EasyControl 描述符（`[preprocess]` / `[training]` 表、`[[datasets]]` 蓝图、可选的 `[variant]`） |
| 3 | `scripts/tasks/training.py` 中的 `_I2I_STAGES` | `"colorize": {"stage": _colorize_stage, "preprocess": _colorize_preprocess}` | 让 `TASK=<task>` 在 `make i2i-stage` 中、`EASYADAPTER=<task>` 在 `easycontrol-preprocess` / `easycontrol` 中正常工作 |
| 4 | `cmd_test_easycontrol` 中的 `_ADAPTERS`（`scripts/tasks/inference.py`） | `"colorize"` 那一行 | `make test-easycontrol` 用的检查点前缀、输出文件夹、参考图回退文件夹、空提示默认值 |

我们按顺序逐一讲解。这里没有任何内容需要改动 `networks/`。

---

## 2. 第一件事——适配器项目（`i2i_tasks/<task>/`）

真正的工作在这里。它做两件事：**构建参考图像**和**缓存它**。（可选地，第三件事：构建任务专属的文本缓存。）

### 2a. 构建参考的函数

这是一个普通函数：接受彩色图像 RGB `uint8 (H,W,3)` 和一个种子，返回**相同尺寸**的参考图像 RGB `uint8 (H,W,3)`。（相同尺寸很重要——见 §3。）对相同种子必须给出相同输出，这样重跑和并行 worker 才能保持一致。

在 colorize 中这是 `mangafy.py::mangafy_array`（及其 GPU 版本 `mangafy_gpu.py::mangafy_array_gpu`）：

```python
# i2i_tasks/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

值得照搬的四点：

- **从文件名确定性地生成种子。** colorize 使用 `zlib.crc32(stem)`——*不是* Python 的 `hash()`，后者每个进程不同，会导致并行 worker 结果不一致。你仍然可以增加多样性（colorize 按页面抖动网点角度）——只要从种子派生即可保持可复现性。
- **延迟导入重量级依赖。** colorize 有三种引擎（`cv2` / `gpu` / `sd`），只有当某页实际路由到该引擎时才加载 3.5 GB 的 SD 模型。如果你的构建器需要某个模型（深度网络、线条提取器），请只在使用时才导入。
- **有无需下载的备用路径是很大的优势。** colorize 的 `cv2`/`gpu` 引擎不需要任何下载，因此可以在全新检出的仓库上直接进行 prep 和训练，无需额外的下载步骤。如果可以，请为你的任务也提供这样的备用路径。
- **原子写入文件。** colorize 的 `_save_png_atomic` 先写临时文件再重命名。没有这一步，被中断的运行可能留下半写的 PNG，而"如果存在则跳过"的检查会永远信任那个文件。这是真实会出现的 bug——请复制这个模式。

如果你的参考图像**已经在磁盘上**（真实的深度图、真实的草图），可以完全跳过构建步骤，直接缓存它们。只有在需要从目标派生参考时才需要构建。

### 2b.（可选）较短的文本缓存

colorize 不仅改变参考——还**将标注精简为颜色词**（`anime_tools.captions.color_tags.filter_to_colors`）。这背后的道理值得理解：参考（线稿 + 网点）已经编码了*关于形状和布局的所有信息*，所以文本还需要说的只是黑白图无法表达的那一件事——**颜色**。将标注精简为颜色词，使每个剩余的词都是模型无法从参考中获取的信息，从而形成强有力的 `prompt → color` 关联，而不是靠埋在长标注中的几个颜色词进行微弱引导。

对你的任务问同样的问题：**参考已经确定了什么，还有什么留给文本来决定？** `pose → image` 参考固定了姿态但没有固定服装或场景——你可能会保留*完整*标注。`depth → image` 参考固定了布局但没有固定身份或颜色。colorize 的"将标注精简为仍然模糊的内容"是一种可以考虑的*模式*，不是规则——很多适配器保留原始标注并完全跳过文本缓存（只需在数据集中省略 `text_cache_dir`——见 §4）。

如果你确实要精简标注，注意 colorize 的两个独立旋钮（不要混淆它们）：

- **`caption_dropout_rate`** — 自动着色的*下限（floor）*。约 5% 的训练步骤会完全丢弃标注，训练模型在没有提示时的行为。保持它**较低**（`0.05`）；过高的值会过度训练无条件路径，使提示效果变弱。
- **`use_shuffled_caption_variants`** — 完整对部分的*平衡*。文本缓存包含多个版本（v0 = 完整颜色集合，v1+ = 每个词以约一半概率丢弃后的打乱版本），加载器以 20% 的概率选 v0、80% 的概率选 v1+，这样像"pink hair"这样的局部提示也能生效。

### 2c. `prep.py`——缓存构建器

三个阶段，每个都是**幂等的**（跳过已完成的工作，重跑安全）：

1. **构建** — 遍历 `--src`（`post_image_dataset/resized`）下的所有彩色图像，运行构建函数，将参考 PNG 写入一个镜像源目录结构的 `--staging` 文件夹。
2. **编码** — 通过 `library.preprocess.cache_latents` 将暂存的参考图像 VAE 编码到 `--cond_cache_dir`，按每张图像的**原生尺寸**编码，使参考 latent 与目标 latent 形状一致。与普通缓存相同的 `{stem}_{WxH}_anima.npz` 格式。
3. **（可选）文本** — 通过 `library.preprocess.cache_text_embeddings`（带 `caption_transform=`，以及 `caption_shuffle_variants` / `caption_tag_dropout_rate`）将标注经过过滤器重新编码到 `--text_cache_dir`。

使用现有的库辅助函数——`library.preprocess.{cache_latents, cache_text_embeddings, tqdm_progress}` 和 `library.preprocess._dataset.walk_images`。不要自己写编码循环；`prep.py` 和 `scripts/preprocess/*.py` 一样，只是这些函数上方的薄壳。

colorize 处理了两个正确性陷阱，照搬它的结构即可免费获得：

- **Stem 必须匹配。** 文本阶段从标注主目录（`image_dataset/`，与 `resized/` 布局相同）读取 `.txt` 标注，使生成的文本缓存文件名与加载器查找的内容（基于 `image_dir=post_image_dataset/resized`）匹配。如果缓存文件名与目标 stem 不匹配，加载器会静默地不进行配对。
- **uncond sidecar。** colorize 的文本阶段在共享的 `T5("")` 空提示 sidecar 缺失时会重新创建它。如果你构建了文本缓存并使用了 caption dropout，也要这样做（`library.preprocess.uncond.stage_uncond_sidecar_with_models`）。

---

## 3. 不能违反的规则——参考和目标的 token 数必须匹配

DiT 运行在 Anima 的**原生形状分桶**上（两个 token 数家族，4032 和 4200；见 CLAUDE.md 和 `docs/anima/experimental/easycontrol.md` 中的"Cond token count"）。**没有填充旋钮**——参考以其 latent 实际的 token 数运行。

这正是 §2a 要求参考与输入**尺寸相同**、§2c 要求按**原生尺寸**编码的原因：两者都做到，参考 latent 就会自动落入与目标相同的桶，一切正常工作。如果你想要较小的参考（完全可以——越小越省内存、越快），请在**图像**层面、编码之前缩小，这样 latent 仍然落在真实的桶上。不要试图在网络内部限制 token 数。

---

## 4. 第二件事——任务文件与描述符

两个文件承载所有任务相关的设置（参见 `i2i_tasks/configs/colorize.toml` 和
`configs/easycontrol/colorize.toml`）。与模型无关的任务文件
`i2i_tasks/configs/<task>.toml` 包含 `name`、`stage` 和 `[staging]`；
其余内容在 EasyControl 描述符 `configs/easycontrol/<task>.toml` 中：

- 任务文件顶层 **`name`**——标识符（slug）。目录树位于
  `post_image_dataset/i2i/<name>/` 下，`output_name` 默认为
  `anima_easycontrol_<name>`。
- 任务文件 **`stage`**——`_I2I_STAGES` 中的键（默认：任务 id）。
- **`[staging]`**（任务文件）/ **`[preprocess]`**（描述符）——扁平表，由你的 stage/preprocess
  函数转换为 `prep.py` 的参数（`_toml_table_to_argv`：`--key value`，
  列表展开，`true` → 不带值的 `--flag`，`false` → 省略）。
- **`[training]`**——作为 CLI 覆盖项叠加到基础 `easycontrol` 方法上传给 `train.py`。
- **`[general]` / `[[datasets]]`**——数据集蓝图。由于 `train.py` 的数据集校验器会拒绝其他顶层键，
  `make easycontrol` 会将其（代入 `{name}`）写入标识符目录下生成的 `dataset_config.toml`。
- **`[variant]`**（可选）——`family = "easycontrol"`、`label`、`description`、
  `order`；带有此块的描述符会出现在 GUI 的 EasyControl 标签页中。

colorize 蓝图的子集：

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

- **`cond_cache_dir`**——这使它成为控制任务。加载器按 stem 将每个目标与参考 latent 匹配，
  并只保留有参考的目标。
- **`text_cache_dir`**——只重定向文本缓存。省略则使用共享文本缓存并跳过文本阶段。
- **`latent_cache_dir`**——只重定向目标 latent。省略则使用 `cache_dir`。
  此处缺失的 latent 会从未校正的原图重新编码（并给出警告），因此训练前请先生成好。
- **`flip_aug = false`**——必须设置。cond 缓存中没有翻转后的 latent，
  开启翻转时加载器会报错。

`[training]` 中需要考虑的参数：

- **`network_args`**——`b_cond_init` 决定第 0 步时参考的贡献程度
  （`-10` ≈ 第 0 步时等同于基础 DiT；参见 `docs/anima/experimental/easycontrol.md` 中的
  "Step-0 baseline equivalence"；colorize 放宽了该值，让参考更早起作用；
  两种方式都可学习）、`cond_scale`、`apply_ffn_lora`（`0` 会去掉 FFN LoRA，
  约减少一半可训练参数）、`cond_res_scale`。
- **`easycontrol_cond_noise_max`**——训练时加到参考上的噪声
  （σ ~ `U(0, max)`，`cond + σ·ε`）。`0` 把参考当作精确蓝图；
  值越大，参考越像粗略提示，细节更多交给文本。
- **`easycontrol_drop_p`**——为 image-CFG 而整体丢弃参考的频率。
  始终带参考的任务设为 `0`。
- **`output_name`**——推理选择器按此前缀查找你最新的检查点（§5）。
- **`blocks_to_swap = 0`**——随附的描述符使用的值。

---

## 5. 第三、四件事——注册任务

**`scripts/tasks/training.py`：** 编写两个签名为 `(adapter, cfg, base, extra)` 的函数，
并将它们加入 `_I2I_STAGES`（以 stage 类型为键）：

```python
_I2I_STAGES = {
    ...
    "colorize": {"stage": _colorize_stage, "preprocess": _colorize_preprocess},
    "<task>": {"stage": _task_stage, "preprocess": _task_preprocess},
}
```

`_colorize_stage` / `_colorize_preprocess` 是范例：它们各自带上 stage 跳过参数、
由标识符派生的路径（`_colorize_prep_paths(base)`）和表中的参数来运行
`i2i_tasks/colorization/prep.py`，并把 `extra` 放在最后，使用户的 `ARGS` 优先。
`_easyadapter()` 会拒绝没有 `i2i_tasks/configs/<task>.toml` 的任务；任务文件的 `stage` 若不在此 dict 中，则在 stage 时报错。
`make easycontrol EASYADAPTER=<task>` 无需其他改动。

**`scripts/tasks/inference.py`**（`cmd_test_easycontrol`）：在 `_ADAPTERS` 中添加一行：

```python
"<task>": {
    "weight": "anima_easycontrol_<task>",   # prefix of output_name
    "out": "<task>",                         # output/tests/<out>/
    "ref_dir": ROOT / "post_image_dataset" / "resized",  # random ref if none given
    "empty_prompt": False,                   # True → default to --prompt ""
},
```

---

## 6. 运行它

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

### 推理说明

- 推理时参考图按原样做 VAE 编码，没有构建步骤。请输入你的构建器所模仿的对象
  （colorize：真实带网点的页面，而不是灰度照片）。
- `make test-easycontrol` 总是传入 `--easycontrol_image_match_size`，
  它会选择与参考图宽高比匹配的 token 桶。
- `--easycontrol_scale`（`EC_SCALE=`）决定输出对参考的遵循程度；
  `1.0` 是训练时的默认值。
- colorize 调好的 `--guidance_scale` / scale / 步数范围见
  `colorization/README.md`（"Inference settings"）。

---

## 7. 检查清单

- [ ] `i2i_tasks/<task>/` 中有确定性、原子写入的构建器
      （`(img, seed) → reference`，尺寸相同）和幂等的 `prep.py`。
- [ ] 参考图按**原始尺寸**编码（§3）。
- [ ] `i2i_tasks/configs/<task>.toml`：`name`、`[staging]`；以及
      `configs/easycontrol/<task>.toml`：`[preprocess]` / `[training]`，带有 `cond_cache_dir`（+ 可选的
      `text_cache_dir` / `latent_cache_dir`）和 `flip_aug = false` 的蓝图。
- [ ] `_I2I_STAGES` 条目（training.py）和 `_ADAPTERS` 行（inference.py）。
- [ ] 完整运行前，先目检一批 `--limit 8` 的暂存结果。

---

## 8. 延伸阅读

- **`i2i_tasks/colorization/README.md`**——colorize 参考实现：
  caption 策略、网点（screentone）分档、标签切片、推理设置。
- **`docs/anima/experimental/easycontrol.md`**——网络本身：双流前向、
  `b_cond` 第 0 步基准、`cond_res_scale`、推理缓存、内存、限制。
- **`networks/methods/easycontrol.py`**——`EasyControlNetwork` 与被 patch 的
  `Block.forward`。
- **`networks/CLAUDE.md`**——按模块的映射与分发规则。
