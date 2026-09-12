#!/usr/bin/env python3
"""O0 scorer (``plan_ocr.md``): any reader over the Manga109-s crop manifest.

    ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="project/cjk_aware_anima_dit/ocr/eval_manga109.py --reader manga_ocr"
    … --reader ppocr        # PP-OCRv6 rec ONNX (anime_tools), crop rotated per its own rule
    … --reader vl16         # PaddleOCR-VL-1.6 crop ``OCR:`` (batched, left-padded, use_cache)
    … --reader manga_ocr --ckpt output/ocr/<run>/best    # a tuned model, same report
    … --reader hayai        # the community hayai-ocr VLM (--ckpt <repo>[@<rev>])
    … --reader hunyuan      # tencent/HunyuanOCR 1.5, official per-task prompt

Metrics per ``kind`` (``sfx`` = COO test crops, ``speech`` = the matched
``<text>`` control) on ``--split`` (default ``test``):

* **exact** — NFKC + whitespace-stripped string equality (hearts, ``ー``,
  small kana all count);
* **sim** — ``build_ocr_records.sim`` (NFKC, punctuation / symbols / spaces
  gone, ``SequenceMatcher`` ratio) after ``normalize_ja`` (vendored below)
  on the prediction (vertical if the crop is) — the A/B's comparison key;
* **runaway** — ``is_runaway`` count (the VL failure class).

Writes ``reports/ocr_eval_<name>.md`` (summary table + worst 25 SFX lines)
and ``output/ocr/eval/<name>_<split>.jsonl`` (every prediction).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402
import textnorm  # noqa: E402

REPORTS = m109.LINE / "reports"
OUT = m109.REPO / "output/ocr/eval"


HEART_FOLD = textnorm.HEART_FOLD
ELLIPSIS_RE = textnorm.ELLIPSIS_RE
"""Every dot run is one ``…`` on both sides of a comparison — the hand labels
spell a pause three ways (sincos: ``・・・`` 57 rows, ``...`` 23, ``…`` 73) and
so did the reader until ``anime_tools.ocr.sfx.normalize_read`` folded it
(8ebaf58, 2026-09-08). Since 2026-09-12 the table lives in :mod:`textnorm`,
shared with the training target, and also folds the long dash (``─`` / ``—`` →
``―``) and a lone ``‥`` — a third, small re-base (``rescore_eval.py``)."""

# Decode cap, ``--max_new_tokens``. manga-ocr's WordPiece is ~1 token / kana, so
# the old 48 truncated every speech line past 47 chars — the sincos labels reach
# 60 (``label_sheet.py``, user pass 2026-09-07). 96 = the training-side
# ``crop_dataset.MAX_TARGET_CHARS``; greedy stops at EOS, so the headroom is free
# except on a runaway, which ``is_runaway`` already counts.
MAX_NEW_TOKENS = 96


def exact_key(s: str) -> str:
    """:func:`textnorm.exact_key` — NFKC + whitespace-blind; heart / wave / dash
    variants folded (the hand labels write ``♡`` and ``〜``, readers emit ``♥``
    / ``~`` for the same glyph) and every dot run one ``…``."""
    return textnorm.exact_key(s)


# --------------------------------------------------------------------------- readers


class MangaOcrReader:
    name = "manga_ocr"
    max_tokens = MAX_NEW_TOKENS

    def __init__(self, ckpt: str | None, device: str):
        mt = m109.pilot_manga_text()
        self.m = mt.MangaOCR(ckpt or mt.OCR_MODEL, device=device)

    def read(self, crops, orients, bs):
        return [t for t, _ in self.m.read(crops, bs, max_new_tokens=self.max_tokens)]


class PpocrReader:
    name = "ppocr"
    max_tokens = MAX_NEW_TOKENS  # CTC — no decode budget, kept for the uniform surface

    def __init__(self, ckpt: str | None, device: str):
        from anime_tools.ocr._onnx import TextRecognizer

        self.r = TextRecognizer.load(
            Path(ckpt) if ckpt else None, device=device, batch_size=32
        )

    def read(self, crops, orients, bs):
        # upstream's rule (crop_quad): taller than 1.5× wide → quarter turn
        rot = [
            np.rot90(c).copy() if c.shape[0] / max(c.shape[1], 1) >= 1.5 else c
            for c in crops
        ]
        return [t for t, _ in self.r.recognize(rot)]


class Vl16Reader:
    name = "vl16"
    max_tokens = MAX_NEW_TOKENS

    def __init__(self, ckpt: str | None, device: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        base = str(m109.REPO / "models/paddleocr_vl_1.6")
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
            tower = Path(adapter) / "tower.safetensors"
            if tower.is_file():  # O2b: the full-finetuned vision tower + projector
                from safetensors.torch import load_file

                sd = load_file(str(tower))
                unexpected = model.load_state_dict(sd, strict=False).unexpected_keys
                assert not unexpected, unexpected[:5]
                print(f"loaded tower {tower} ({len(sd)} tensors)")
        self.model = model.to(device).eval()
        self.proc = AutoProcessor.from_pretrained(path)
        self.device = device
        self.min_edge = self.proc.image_processor.size["shortest_edge"]

    def read(self, crops, orients, bs):
        from PIL import Image

        order = sorted(
            range(len(crops)), key=lambda i: crops[i].shape[0] * crops[i].shape[1]
        )
        out = [""] * len(crops)
        tok = self.proc.tokenizer
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            images = [Image.fromarray(crops[i][:, :, ::-1]) for i in idx]
            msgs = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}],
                }
            ]
            text = self.proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            inputs = self.proc(
                text=[text] * len(images),
                images=images,
                padding=True,
                padding_side="left",
                return_tensors="pt",
                images_kwargs={
                    "size": {
                        "shortest_edge": self.min_edge,
                        "longest_edge": 1280 * 28 * 28,
                    }
                },
            ).to(self.device)
            n = inputs["input_ids"].shape[-1]
            with self.torch.inference_mode():
                o = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            for i, row in zip(idx, o):
                ids = [
                    t
                    for t in row[n:].tolist()
                    if t not in (tok.eos_token_id, tok.pad_token_id)
                ]
                out[i] = tok.decode(ids).strip()
        return out


class SfxPkgReader:
    """O4: the shipped reader — ``anime_tools.ocr.sfx.SfxReader`` (B′ weights
    from the catalog rows, decode guard built in). ``--ckpt`` overrides the
    adapter dir; a guarded-out read scores as an empty string."""

    name = "sfx"
    max_tokens = MAX_NEW_TOKENS

    def __init__(self, ckpt: str | None, device: str):
        try:
            from anime_tools.ocr import sfx
        except ImportError:  # dev loop before the pinned rev carries ocr/sfx.py
            sys.path.insert(0, str(m109.REPO.parent / "anime_tools"))
            for k in [k for k in sys.modules if k.startswith("anime_tools")]:
                del sys.modules[k]
            from anime_tools.ocr import sfx
        self.sfx = sfx
        self.r = sfx.SfxReader.load(
            device=device,
            base_dir=m109.REPO / "models/paddleocr_vl_1.6",
            adapter_dir=Path(ckpt) if ckpt else None,
        )

    def read(self, crops, orients, bs):
        self.r.batch_size = bs
        self.sfx.MAX_NEW_TOKENS = self.max_tokens  # module global, read per call
        return [t or "" for t in self.r.read(crops)]


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

        import hayai_beam

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

        path = ckpt or str(m109.REPO / self.DEFAULT)
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
    "ppocr": PpocrReader,
    "vl16": Vl16Reader,
    "sfx": SfxPkgReader,
    "hayai": HayaiReader,
    "hunyuan": HunyuanReader,
}


# --------------------------------------------------------------------------- ja normaliser
# Vendored verbatim from anime_tools.ocr._text @ 62f6fc3^ (the ``ー`` put-back that
# PP-OCRv6 needed). The package retired it with the CTC recognizer on 2026-09-07
# (anime-tools 62f6fc3); it stays here so every eval row scores on the same
# normaliser as O0–O3 — a re-based scorer would not be comparable.

CHOON = "ー"
"""The prolonged sound mark. PP-OCRv6's vocabulary is Chinese-first: set
vertically the mark is a bare stroke it reads as ``1`` / ``|`` / ``l``, set
horizontally a dash — so ``おちんぼの時間だぞ1`` and ``でるッリ一チ`` are one
misread each, not two words."""

_CHOON_VERTICAL = frozenset("1|lI１｜丨")
"""What a vertical ``ー`` comes back as."""

_CHOON_HORIZONTAL = frozenset("-—–ｰ")
"""What a horizontal ``ー`` comes back as (``ｰ`` is its own halfwidth form)."""

_NI_AS_EQUALS = frozenset("=＝")
"""What ``ニ`` comes back as between katakana — ``メ=ュー`` — two strokes the
vocabulary has as an equals sign."""


def is_kana(ch: str) -> bool:
    """Hiragana or katakana, ``ー`` included — the scripts a ``ー`` follows."""
    return "\u3041" <= ch <= "\u3096" or "\u30a1" <= ch <= "\u30fc"


def _is_katakana(ch: str) -> bool:
    return "\u30a1" <= ch <= "\u30fc"


_KANA_COUNTERS = frozenset("かつヶケ")
"""Counters written in kana — ``1か月`` / ``あと1つ`` — after which a digit is a
digit even though kana precedes it."""


def _counts_something(nxt: str) -> bool:
    """Whether the glyph after a would-be ``ー`` makes it a numeral instead:
    a kanji (``もう1回``), another digit or Latin, or a kana counter."""
    if not nxt:
        return False
    if "\u4e00" <= nxt <= "\u9fff" or nxt.isascii() and nxt.isalnum():
        return True
    return "\uff10" <= nxt <= "\uff19" or nxt in _KANA_COUNTERS


def normalize_ja(text: str, *, vertical: bool) -> str:
    """Put back the ``ー`` the recognizer spelled as a digit or a dash.

    Only after kana, and only when nothing counted follows: ``1`` after a kanji
    is a number, ``-`` after Latin a hyphen, and ``もう1回`` / ``1か月`` keep
    their digit (:func:`_counts_something`). A horizontal ``一`` (the kanji *one*) is a ``ー`` only *between* katakana —
    ``リ一チ`` — since ``一杯`` after a katakana word is a real ``一``. Vertical
    text never confuses the two (the strokes cross), so ``一`` is left alone
    there. The mark chains: ``ーー`` is written as such, and a fixed ``ー`` is
    kana for the next glyph. Likewise ``=`` between katakana is ``ニ``
    (``メ=ュー``), in either orientation.
    """
    out: list[str] = []
    for i, ch in enumerate(text):
        prev = out[-1] if out else ""
        if prev and is_kana(prev):
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if (
                ch in _CHOON_HORIZONTAL or (vertical and ch in _CHOON_VERTICAL)
            ) and not _counts_something(nxt):
                out.append(CHOON)
                continue
            between_katakana = _is_katakana(prev) and bool(nxt) and _is_katakana(nxt)
            if not vertical and ch == "一" and between_katakana:
                out.append(CHOON)
                continue
            if ch in _NI_AS_EQUALS and between_katakana:
                out.append("ニ")
                continue
        out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------- scoring


def score(df: pd.DataFrame, preds: list[str]) -> pd.DataFrame:
    rec = m109.pilot_records()
    rows = []
    for (_, r), p in zip(df.iterrows(), preds):
        pn = normalize_ja(p, vertical=(r.orient == "vertical"))
        rows.append(
            dict(
                pred=p,
                pred_norm=pn,
                exact=exact_key(pn) == exact_key(r.text),
                sim=rec.sim(pn, r.text),
                runaway=rec.is_runaway(p),
            )
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def summary(scored: pd.DataFrame, name: str, split: str, wall: float) -> str:
    lines = [
        f"# OCR eval — `{name}` on Manga109-s `{split}` (official COO split ∩ Manga109-s)\n",
        f"Reader wall {wall:.0f} s for {len(scored)} crops ({len(scored) / max(wall, 1e-9):.1f} crops/s).\n",
        "| kind | n | exact | exact % | sim (mean) | sim ≥ 0.8 | runaway |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, g in scored.groupby("kind"):
        lines.append(
            f"| {k} | {len(g)} | {int(g.exact.sum())} | {100 * g.exact.mean():.1f} | "
            f"{g.sim.mean():.3f} | {100 * (g.sim >= 0.8).mean():.1f} % | {int(g.runaway.sum())} |"
        )
    sfx = scored[scored.kind == "sfx"]
    if len(sfx):
        lines.append(
            "\n## SFX by orientation\n\n| orient | n | exact % | sim |\n|---|---|---|---|"
        )
        for o, g in sfx.groupby("orient"):
            lines.append(
                f"| {o} | {len(g)} | {100 * g.exact.mean():.1f} | {g.sim.mean():.3f} |"
            )
        lines.append(
            "\n## SFX by length\n\n| len | n | exact % | sim |\n|---|---|---|---|"
        )
        L = sfx.text.str.len().clip(upper=8)
        for n, g in sfx.groupby(L):
            lines.append(
                f"| {n}{'+' if n == 8 else ''} | {len(g)} | {100 * g.exact.mean():.1f} | {g.sim.mean():.3f} |"
            )
        lines.append(
            "\n## Worst 25 SFX (by sim)\n\n| book / page / id | gt | pred | sim |\n|---|---|---|---|"
        )
        for _, r in sfx.sort_values("sim").head(25).iterrows():
            pred = r.pred.replace("|", "\\|")[:40]
            lines.append(
                f"| {r.book} {r.page:03d} {r.id} | {r.text} | {pred} | {r.sim:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reader", choices=sorted(READERS), required=True)
    ap.add_argument(
        "--ckpt", help="model dir overriding the stock weights (same reader class)"
    )
    ap.add_argument(
        "--name", help="report name (default: reader, or reader-<ckpt stem>)"
    )
    ap.add_argument("--split", default="test", choices=m109.SPLITS)
    ap.add_argument("--kind", choices=["sfx", "speech"], action="append")
    ap.add_argument("--limit", type=int, help="first N crops per kind (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"decode cap per crop (default {MAX_NEW_TOKENS}); a long speech bubble "
        "needs one token per kana on manga-ocr",
    )
    a = ap.parse_args()
    name = a.name or (f"{a.reader}-{Path(a.ckpt).stem}" if a.ckpt else a.reader)

    derived = m109.derived_root()
    df = pd.read_parquet(derived / "manifest.parquet")
    df = df[df.split == a.split]
    if a.kind:
        df = df[df.kind.isin(a.kind)]
    if a.limit:
        df = df.groupby("kind", group_keys=False).head(a.limit)
    df = df.sort_values(["kind", "book", "page", "id"]).reset_index(drop=True)
    crops = [cv2.imread(str(derived / p)) for p in df.path]
    assert all(c is not None for c in crops), (
        "missing crop png — rerun build_manga109_crops"
    )

    reader = READERS[a.reader](a.ckpt, a.device)
    reader.max_tokens = a.max_new_tokens
    t0 = time.time()
    preds = reader.read(crops, list(df.orient), a.bs)
    wall = time.time() - t0
    scored = score(df, preds)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}_{a.split}.jsonl").open("w", encoding="utf-8") as f:
        for _, r in scored.iterrows():
            f.write(
                json.dumps(
                    {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()},
                    ensure_ascii=False,
                )
                + "\n"
            )
    md = summary(scored, name, a.split, wall)
    if not a.limit:
        REPORTS.mkdir(exist_ok=True)
        (REPORTS / f"ocr_eval_{name}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
