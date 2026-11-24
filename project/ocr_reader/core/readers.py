"""OCR readers behind one surface: ``Reader(ckpt, device)``, ``.read(crops, orients, bs)`` →
one string per BGR crop, ``.max_tokens`` the decode cap. :data:`READERS` maps the ``--reader``
names of the eval scripts to them.

The Ovis, PP-OCRv6 and ``anime_tools`` package readers are in ``_archive/ocr_reader/code/``
(their dependencies left the tree).
"""

from __future__ import annotations

import math
from pathlib import Path

from .paths import REPO, VL16_BASE

# Decode cap, ``--max_new_tokens``. manga-ocr's WordPiece is ~1 token / kana, so
# the old 48 truncated every speech line past 47 chars — the sincos labels reach
# 60 (``label_sheet.py``, user pass 2026-09-07). 96 = the training-side
# ``crop_dataset.MAX_TARGET_CHARS`` of the time; greedy stops at EOS, so the headroom is
# free except on a runaway, which ``scoring.is_runaway`` already counts.
MAX_NEW_TOKENS = 96

VL16_TASKS = ("OCR:", "OCR2:", "SFX:")
"""``Vl16Reader.task`` values: ``OCR:`` main text, ``OCR2:`` sub text (v7 main / sub);
``SFX:`` is the two-prompt reader's (``v7_preview``)."""

MANGA_OCR_MODEL = "kha-white/manga-ocr-base"


class MangaOCR:
    """manga-ocr (kha-white) with a vocab-only decoder.

    The published tokenizer is ``BertJapaneseTokenizer``, whose *encoder* needs
    MeCab/fugashi -- but generation only ever **decodes**, and decoding is plain
    WordPiece over ``vocab.txt``. Reading the vocab directly keeps two native
    dependencies out of the project for a pilot that never tokenises Japanese
    input.
    """

    def __init__(self, model_id: str = MANGA_OCR_MODEL, device: str = "cuda") -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoImageProcessor, VisionEncoderDecoderModel

        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.model.to(device).eval()

        local = Path(model_id) / "vocab.txt"  # a fine-tuned dir (plan_ocr O2)
        vp = local if local.is_file() else Path(hf_hub_download(model_id, "vocab.txt"))
        vocab = vp.read_text(encoding="utf-8")
        self.vocab = vocab.splitlines()
        self.specials = {
            i for i, t in enumerate(self.vocab) if t.startswith("[") and t.endswith("]")
        }

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i in self.specials or i >= len(self.vocab):
                continue
            tok = self.vocab[i]
            out.append(tok[2:] if tok.startswith("##") else tok)
        return "".join(out).replace(" ", "")

    def read(self, crops: list, batch_size: int = 32, max_new_tokens: int = 48):
        """Return ``[(text, mean_token_logprob), ...]``.

        The logprob is the whole point of running greedy with scores: it is the
        only per-region quality signal available without ground truth, and the
        ``hard_neg`` control tells us whether it actually separates.
        """
        import numpy as np
        from PIL import Image

        torch = self.torch
        results: list[tuple[str, float]] = []
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            images = [Image.fromarray(c[:, :, ::-1]).convert("RGB") for c in batch]
            pixel = self.processor(images, return_tensors="pt").pixel_values.to(
                self.device
            )
            with torch.no_grad():
                out = self.model.generate(
                    pixel,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                scores = self.model.compute_transition_scores(
                    out.sequences, out.scores, normalize_logits=True
                )
            seqs = out.sequences[:, -scores.shape[-1] :]
            for row, sc, seq in zip(out.sequences, scores, seqs):
                text = self.decode(row.tolist())
                keep = [
                    float(s)
                    for s, t in zip(sc.tolist(), seq.tolist())
                    if int(t) not in self.specials and not math.isinf(s)
                ]
                results.append((text, float(np.mean(keep)) if keep else float("-inf")))
        return results


class MangaOcrReader:
    name = "manga_ocr"
    max_tokens = MAX_NEW_TOKENS

    def __init__(self, ckpt: str | None, device: str):
        self.m = MangaOCR(ckpt or MANGA_OCR_MODEL, device=device)

    def read(self, crops, orients, bs):
        return [t for t, _ in self.m.read(crops, bs, max_new_tokens=self.max_tokens)]


class Vl16Reader:
    """PaddleOCR-VL 1.6, the line's reader. ``ckpt`` is a full model dir, or a peft
    adapter dir merged onto :data:`VL16_BASE` (plus its ``tower.safetensors`` /
    ``lm.safetensors`` full-finetuned weights when present). ``task`` is the prompt
    (:data:`VL16_TASKS`); callers set it per read."""

    name = "vl16"
    max_tokens = MAX_NEW_TOKENS
    LONGEST = 1280 * 28 * 28
    """The processor's ``longest_edge`` (pixels)."""

    def __init__(self, ckpt: str | None, device: str, task: str = "OCR:"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        base = str(VL16_BASE)
        adapter = (
            ckpt if ckpt and (Path(ckpt) / "adapter_config.json").is_file() else None
        )
        path = base if adapter else (ckpt or base)
        self.torch = torch
        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        if adapter:  # O2 arm B: a peft LoRA on the LM, merged for the read
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
            # O2b: the full-finetuned tower + projector; ocr_reader P2b: LM layers
            for name in ("tower.safetensors", "lm.safetensors"):
                full = Path(adapter) / name
                if not full.is_file():
                    continue
                from safetensors.torch import load_file

                sd = load_file(str(full))
                unexpected = model.load_state_dict(sd, strict=False).unexpected_keys
                assert not unexpected, unexpected[:5]
                print(f"loaded {full} ({len(sd)} tensors)")
        self.model = model.to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        self.device = device
        self.min_edge = self.proc.image_processor.size["shortest_edge"]
        self.task = task

    def read(
        self,
        crops,
        orients=None,
        bs: int = 32,
        *,
        caps: list[int] | None = None,
        budget: float | None = None,
        prefetch: int = 0,
        min_pixels: int = 0,
        progress: bool = False,
    ) -> list[str]:
        """Greedy reads of ``crops`` (BGR) with :attr:`task`, batched in crop-area order.

        A batch closes at ``bs`` crops or, with ``budget``, at that many tower patches
        (bs 64 at the stock size OOMed on the largest crops). ``caps`` (one per crop)
        caps ``max_new_tokens`` per batch at its largest entry instead of the flat
        :attr:`max_tokens`, so a batch of short columns does not run the full cap for one
        runaway. ``prefetch`` batches are prepared on a thread while the GPU generates.
        ``min_pixels`` (0 = the processor's shortest edge, 112,896) is the floor a crop
        is upscaled to; the prefill is the tower's GEMMs, so the patch count sets the
        speed.
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        from PIL import Image

        proc, tok = self.proc, self.proc.tokenizer
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.task}],
            }
        ]
        text = proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        minpx = min_pixels or self.min_edge
        maxpx = self.LONGEST

        def patches(i):
            h, w = crops[i].shape[:2]
            return min(max(h * w, minpx), maxpx) / 196

        order = sorted(
            range(len(crops)), key=lambda i: crops[i].shape[0] * crops[i].shape[1]
        )
        batches, cur, tot = [], [], 0.0
        for i in order:
            p = patches(i) if budget else 0.0
            if cur and (len(cur) >= bs or (budget and tot + p > budget)):
                batches.append(cur)
                cur, tot = [], 0.0
            cur.append(i)
            tot += p
        if cur:
            batches.append(cur)

        def prep(idx):
            images = [
                Image.fromarray(np.ascontiguousarray(crops[i][:, :, ::-1])) for i in idx
            ]
            return proc(
                text=[text] * len(images),
                images=images,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                images_kwargs={"size": {"shortest_edge": minpx, "longest_edge": maxpx}},
            )

        # one prep thread: the fast tokenizer is not safe to call from two threads, and
        # decoding waits for the end for the same reason
        ids_out: list = [None] * len(crops)
        t0 = time.time()
        with ThreadPoolExecutor(1) as pool:
            futs = [pool.submit(prep, b) for b in batches[: prefetch + 1]]
            for bi, idx in enumerate(batches):
                # BatchFeature.to moves in place: drop the future so the GPU copy goes
                inputs, futs[bi] = futs[bi].result().to(self.device), None
                if bi + prefetch + 1 < len(batches):
                    futs.append(pool.submit(prep, batches[bi + prefetch + 1]))
                n = inputs["input_ids"].shape[-1]
                cap = max(caps[i] for i in idx) if caps else self.max_tokens
                with self.torch.inference_mode():
                    o = self.model.generate(
                        **inputs, max_new_tokens=cap, do_sample=False, use_cache=True
                    )
                for i, row in zip(idx, o[:, n:].tolist()):
                    ids_out[i] = row
                if progress and bi % 50 == 0:
                    done = sum(len(b) for b in batches[: bi + 1])
                    print(
                        f"  {done}/{len(crops)} crops, {done / (time.time() - t0):.1f}/s",
                        flush=True,
                    )
        drop = {tok.eos_token_id, tok.pad_token_id}
        out = [tok.decode([t for t in ids if t not in drop]).strip() for ids in ids_out]
        if progress:
            print(f"read {len(crops)} crops in {time.time() - t0:.0f} s", flush=True)
        return out


class HayaiReader:
    """The community `hayai-ocr` VLM (SigLIP2-NaFlex tower + a 12-layer causal
    decoder, ~150 M, `trust_remote_code`) — scored on request from its author
    (`sorryhyun/paddleocr-vl-1.6-manga-lora` discussion #1). ``--ckpt`` is
    ``<repo_id>[@<revision>]`` or a local dir; the default is the **v2.1**
    branch (the revision the request names — `main` is v2.0). Decode follows
    the card's own recipe (`num_beams=4`, `repetition_penalty=1.0`,
    `max_num_patches=256` for single lines; `ANIMA_HAYAI_PATCHES` overrides)
    and the crop goes in unrotated, as for `manga_ocr` — both read vertical
    Japanese natively."""

    name = "hayai"
    max_tokens = MAX_NEW_TOKENS
    DEFAULT = "JustANormalTinkerer/hayai-ocr-v2@v2.1"

    def __init__(self, ckpt: str | None, device: str):
        import os

        import torch
        from transformers import AutoModel, AutoProcessor, PreTrainedTokenizerFast

        repo, _, rev = (ckpt or self.DEFAULT).partition("@")
        kw = {"revision": rev} if rev else {}
        self.torch = torch
        self.model = (
            AutoModel.from_pretrained(repo, trust_remote_code=True, **kw)
            .to(device)
            .eval()
        )
        self.tok = PreTrainedTokenizerFast.from_pretrained(repo, **kw)
        # the card's processor: the stock SigLIP2 NaFlex one, not a repo file
        self.proc = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")
        self.patches = int(os.environ.get("ANIMA_HAYAI_PATCHES", 256))
        self.device = device

    def prepare(self, crops):
        """CPU half of a batch: BGR crops → SigLIP2 NaFlex tensors (on CPU),
        so a caller can run it on a thread while the GPU decodes the previous
        batch (``screen_shards.py``)."""
        from PIL import Image

        images = [Image.fromarray(c[:, :, ::-1]) for c in crops]
        return self.proc(
            images=images, max_num_patches=self.patches, return_tensors="pt"
        )

    def decode(self, inputs) -> list[str]:
        """GPU half of a batch."""
        import os
        from functools import partial

        from . import hayai_beam

        inputs = inputs.to(self.device)
        with self.torch.no_grad():
            # hayai_beam.generate = upstream's beam search with the
            # per-candidate Python loop vectorised and a per-row early stop
            # (the upstream path is launch-bound at ~25 % GPU). ``upstream``
            # keeps the model's own generate for an A/B. ``ANIMA_HAYAI_AMP``
            # = bf16 (default) | fp16 (upstream's autocast) | 0 (fp32).
            amp = os.environ.get("ANIMA_HAYAI_AMP", "bf16").lower()
            gen = (
                self.model.generate
                if os.environ.get("ANIMA_HAYAI_BEAM_IMPL") == "upstream"
                else partial(
                    hayai_beam.generate,
                    self.model,
                    amp=amp not in ("0", "fp32", "") and amp,
                )
            )
            texts = gen(
                pixel_values=inputs["pixel_values"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
                spatial_shapes=inputs["spatial_shapes"],
                tokenizer=self.tok,
                max_new_tokens=self.max_tokens,
                num_beams=4,
                repetition_penalty=1.0,
            )
        return [(t or "").strip() for t in texts]

    def read(self, crops, orients, bs):
        out = []
        for s in range(0, len(crops), bs):
            out.extend(self.decode(self.prepare(crops[s : s + bs])))
        return out


class Hayai27Reader:
    """hayai v2.7 Supernova — scored on request (discussion #1, 2026-11-16).
    The card's own entry point ``model.ocr``: ``max_patches="auto"`` (per-crop
    routing over {128, 192, 256, 384}), ``escalate=False``, greedy, its own
    processor (``trust_remote_code``), ``batch_size=64`` so the auto planner
    buckets the list. Crops go in unrotated, as for every hayai row."""

    name = "hayai27"
    max_tokens = MAX_NEW_TOKENS
    DEFAULT = (
        "JustANormalTinkerer/hayai-ocr-v2.7-supernova"
        "@5289b2cfdfd375dd651611b27ea87d21b5877008"
    )

    def __init__(self, ckpt: str | None, device: str):
        import os

        from transformers import AutoModel, AutoProcessor

        repo, _, rev = (ckpt or self.DEFAULT).partition("@")
        kw = {"revision": rev} if rev else {}
        self.model = (
            AutoModel.from_pretrained(repo, trust_remote_code=True, **kw)
            .to(device)
            .eval()
        )
        self.proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True, **kw)
        p = os.environ.get("ANIMA_HAYAI_PATCHES", "auto")
        self.patches = p if p == "auto" else int(p)

    def read(self, crops, orients, bs):
        from PIL import Image

        images = [Image.fromarray(c[:, :, ::-1]) for c in crops]
        texts = self.model.ocr(
            images,
            self.proc,
            max_new_tokens=self.max_tokens,
            max_patches=self.patches,
            batch_size=64,
            escalate=False,
        )
        return [(t or "").strip() for t in texts]


class HunyuanReader:
    """`tencent/HunyuanOCR` — HunyuanOCR-1.5 (1 B: 24 × 1024 LM + a 16-px-patch
    tower; native ``HunYuanVLForConditionalGeneration`` since transformers
    5.13, no remote code), the second general OCR VLM scored against
    PaddleOCR-VL-1.6 here. ``--ckpt`` overrides the local weights dir.

    Upstream **locks the prompt per task** (`inference/utils/tasks.py`: "users
    pick a task, not a prompt" — hand-edited instructions were observed to
    silently degrade quality), so a crop is read with ``structured_parse`` =
    ``提取图中的文字。`` verbatim; ``ANIMA_HUNYUAN_TASK`` picks another key of
    :data:`TASKS`. Decode is the card's locked recipe — greedy,
    ``repetition_penalty=1.08`` (``ANIMA_HUNYUAN_REP_PENALTY``, ``1.0`` = off).
    That guard is *shipped* here, where ``vl16``'s runaway count is part of its
    stock row, so the two stock rows are each the model as its authors ship it.

    Its image processor's ``min_pixels`` is 262144 (= 512²), so every crop is
    upscaled to ≥ 256 visual tokens however small the box — wall per crop is
    near-flat in crop size, unlike ``vl16``'s shortest-edge rule.
    """

    name = "hunyuan"
    max_tokens = MAX_NEW_TOKENS
    DEFAULT = "models/hunyuan_ocr"
    TASKS = {  # verbatim from inference/utils/tasks.py @ main, 2026-09-08
        "structured_parse": "提取图中的文字。",
        "doc_parse": (
            "提取文档图片中正文的所有信息用markdown格式表示，其中页眉、页脚部分忽略，"
            "表格用html格式表达，文档中公式用latex格式表示，按照阅读顺序组织进行解析。"
        ),
        "spotting_json": (
            "检测并识别图中所有的文字行，请按从上到下、从左到右的阅读顺序进行识别。 "
            "输出格式为 JSON 数组，每个元素必须包含："
            '"box": [xmin, ymin, xmax, ymax]（坐标需归一化到 [0, 1000] 范围内）；'
            '"text": "识别出的文字内容"。 '
            "注意：请直接输出 JSON 数组，不要包含任何多余的描述性文字。"
        ),
        "spotting_hunyuan": "检测并识别图片中的文字，将文本坐标格式化输出。",
    }

    def __init__(self, ckpt: str | None, device: str):
        import os

        import torch
        from transformers import AutoProcessor, HunYuanVLForConditionalGeneration

        path = ckpt or str(REPO / self.DEFAULT)
        self.torch = torch
        # ``ANIMA_HUNYUAN_PROMPT`` is off-recipe by construction (upstream ships
        # no free-form prompt) — it exists only to test whether the Chinese
        # task wording, not the tower, is what puts Han where the kana is.
        self.prompt = (
            os.environ.get("ANIMA_HUNYUAN_PROMPT")
            or self.TASKS[os.environ.get("ANIMA_HUNYUAN_TASK", "structured_parse")]
        )
        self.rep_penalty = float(os.environ.get("ANIMA_HUNYUAN_REP_PENALTY", 1.08))
        self.model = (
            HunYuanVLForConditionalGeneration.from_pretrained(
                path, dtype=torch.bfloat16, attn_implementation="sdpa"
            )
            .to(device)
            .eval()
        )
        self.proc = AutoProcessor.from_pretrained(path, use_fast=False)
        self.device = device

    def read(self, crops, orients, bs):
        from PIL import Image

        order = sorted(
            range(len(crops)), key=lambda i: crops[i].shape[0] * crops[i].shape[1]
        )
        out = [""] * len(crops)
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": self.prompt}],
            }
        ]
        text = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            images = [Image.fromarray(crops[i][:, :, ::-1]) for i in idx]
            inputs = self.proc(
                text=[text] * len(images),
                images=images,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(self.device)
            n = inputs["input_ids"].shape[-1]
            with self.torch.inference_mode():
                o = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    use_cache=True,
                    repetition_penalty=self.rep_penalty,
                )
            for i, row in zip(idx, o):
                out[i] = self.proc.tokenizer.decode(
                    row[n:], skip_special_tokens=True
                ).strip()
        return out


READERS = {
    "manga_ocr": MangaOcrReader,
    "vl16": Vl16Reader,
    "hayai": HayaiReader,
    "hayai27": Hayai27Reader,
    "hunyuan": HunyuanReader,
}
