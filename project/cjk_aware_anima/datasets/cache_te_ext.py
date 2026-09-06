#!/usr/bin/env python3
"""Arm-C TE cache: OCR-quoted captions encoded through the ext vocab pack.

Builds a temp caption mirror (symlinked resized images + ``.txt`` /
``.variants.txt`` with the OCR quote tags appended to every variant's flat
bag), then runs the standard ``cache_text_embeddings`` stage with a tokenize
strategy whose T5 side always goes through ``HybridT5Encoder`` (the trained
synthjako2 rows appended to the frozen adapter embed, exactly the run_bench
``*_ext`` arm). Output caches land in a sidecar dir for the arm-C dataset's
``text_cache_dir`` redirect — production captions, caches and masters are
untouched. Captions with no CJK encode bit-identically to the stock strategy,
so copied-through images double as an in-run control.

Usage (GPU -> daemon)::

    make daemon-run ARGS="project/cjk_aware_anima/datasets/cache_te_ext.py \
        --shard sincos"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))  # ocr_sfx (sibling, torch-free)

import torch  # noqa: E402

from anima_lora import default_checkpoints  # noqa: E402
from bench.cjk_adapter import ext_vocab  # noqa: E402

TEXT_TAG_RE = re.compile(r"\b(text|speech bubble|sound effects)\b")


class ExtTokenizeStrategy:
    """AnimaTokenizeStrategy with the T5 side routed through the ext encoder."""

    def __init__(self, base, ext_encoder):
        self.base = base
        self.ext_encoder = ext_encoder
        # cache_text_embeddings reads these for the erasure-token pool.
        self.qwen3_tokenizer = base.qwen3_tokenizer
        self.t5_tokenizer = base.t5_tokenizer

    def tokenize(self, text):
        texts = [text] if isinstance(text, str) else list(text)
        t5_rows = []
        for t in texts:
            ids, mask = self.ext_encoder.encode(t, self.base.t5_max_length)
            t5_rows.append((torch.tensor(ids), torch.tensor(mask)))
        enc = self.base.qwen3_tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.base.qwen3_max_length,
        )
        return [
            enc["input_ids"],
            enc["attention_mask"],
            torch.stack([r[0] for r in t5_rows]),
            torch.stack([r[1] for r in t5_rows]),
        ]


OCR_FORMATS = ("sentence", "order", "tags", "presence")
"""``sentence`` — a trailing text clause after the tag bag and any position
clauses, ``Japanese text reads as "…", "…".`` (the anime_tools grammar's
``TEXT_PREFIXES`` clause kind, plan_base1 B2), speech lines in reading order,
then ``Japanese SFX reads as "…".`` with the ``kind: sfx`` lines, deduplicated
per sound unit (plan_ocr O4 / O4c — the SFX reader reads hand-lettered
onomatopoeia; before 2026-09-06 the SFX lines were skipped, ``--drop_sfx``
reproduces that); ``order`` — one phrase, ``Japanese text in following order:
"…", "…"``, the lines in the records' (reading) order; ``tags`` — the C2–C6
form, a ``japanese text`` presence tag plus one ``「…」`` flat tag per line;
``presence`` — the ``japanese text`` tag alone (text presence with no ext-row
address: the text-binding probe's control arm)."""

ORDER_PREFIX = "Japanese text in following order: "


DROP_KINDS: frozenset[str] = frozenset({"chrome"})
"""Record ``kind`` values no caption format sees (hybrid records, plan_base1
B0). ``chrome`` is UI text on a screenshot page — neither speech nor SFX.
``sfx`` was in this set 2026-09-05 → 2026-09-06 (the stock readers garbled
hand-lettered onomatopoeia, ``でくv`` for びくっ); it left once the SFX reader
(``anime_tools.ocr.sfx``, plan_ocr O4) could read them and arm C11 passed
its gate (spam = C10, blind s15 11–9 flat inside the floor). ``--drop_sfx``
restores the C10 caption for reproductions. Records without ``kind``
(pre-hybrid files) are untouched."""

SFX_DROPPED: frozenset[str] = DROP_KINDS | {"sfx"}
"""``DROP_KINDS`` as it stood for arms C2–C10 (``--drop_sfx``)."""


def ocr_records_by_stem(
    records: Path, max_lines: int, drop_kinds: frozenset[str] = DROP_KINDS
) -> dict[str, list[tuple[str, str]]]:
    """``(text, kind)`` per stem, in file order (the records are already
    sorted into reading order by the OCR pass), minus the ``drop_kinds``.
    A record without ``kind`` (pre-hybrid file) is ``speech``."""
    by_stem: dict[str, list[tuple[str, str]]] = {}
    with records.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            kind = r.get("kind", "speech")
            if kind in drop_kinds:
                continue
            by_stem.setdefault(r["stem"], []).append((r["text"], kind))
    return {k: v[:max_lines] for k, v in by_stem.items()}


def ocr_lines_by_stem(
    records: Path, max_lines: int, drop_kinds: frozenset[str] = DROP_KINDS
) -> dict[str, list[str]]:
    """Raw OCR lines per stem (:func:`ocr_records_by_stem` without the kind)."""
    return {
        k: [t for t, _ in v]
        for k, v in ocr_records_by_stem(records, max_lines, drop_kinds).items()
    }


def _quote_safe(text: str) -> str:
    """A line as it may sit inside ``"…"``: only an inner ASCII double quote
    is rewritten (fullwidth), since it would close the pair early. Commas
    stay — the caption grammar is quote-aware (anime_tools ≥ the D1 rev:
    a comma inside an open ``「『"`` is not a tag separator), and rewriting
    them leaked into OCR CER."""
    return text.replace('"', "”")


def ocr_text_clauses(
    lines: list[str], kinds: list[str] | None = None, sfx_sentence: bool = False
) -> list:
    """The ``sentence`` format's clauses: one ``Japanese text reads as`` text
    clause with the speech lines in reading order, then — with
    ``sfx_sentence`` (arm C11, plan_ocr O4) — one ``Japanese SFX reads as``
    clause with the SFX lines; neither when its lines are empty.

    ``kinds`` (the records' ``kind``, hand labels + rule, plan_ocr O4) decides
    which is which; without it the :mod:`ocr_sfx` text rule does
    (:func:`ocr_sfx.split_lines`), as C10 trained. Without ``sfx_sentence``
    the SFX lines are skipped (decision 2 amended, see ``OCR_FORMATS``).
    ``text_clause`` quotes each line and defuses an inner ASCII quote. The SFX
    lines are deduplicated by :func:`ocr_sfx.dedupe_sfx` (one per sound unit,
    first in reading order kept; C11 trained without this)."""
    from anime_tools.captions.position_clauses import text_clause
    from ocr_sfx import dedupe_sfx, split_lines

    if kinds is None:
        speech, sfx = split_lines(lines)
    else:
        speech = [ln for ln, k in zip(lines, kinds, strict=True) if k == "speech"]
        sfx = [ln for ln, k in zip(lines, kinds, strict=True) if k == "sfx"]
    out = [text_clause(speech)] if speech else []
    # a repeated SFX (じゅぽ, じゅぽ, じゅぽじゅぽ) is one sound, not three lines
    # (2026-09-06, user's call); speech repeats are content and stay
    sfx = dedupe_sfx(sfx)
    if sfx_sentence and sfx:
        out.append(text_clause(sfx, sfx=True))
    return out


def ocr_tags(lines: list[str], fmt: str) -> list[str]:
    """The tag(s) that carry ``lines`` under ``fmt`` (see ``OCR_FORMATS``)."""
    if fmt == "sentence":
        raise ValueError("'sentence' is not tag-shaped; use ocr_text_clauses()")
    if fmt == "tags":
        return [f"「{ln}」" for ln in lines]
    if fmt == "presence":
        return []
    if fmt == "order":
        quoted = ", ".join(f'"{_quote_safe(ln)}"' for ln in lines)
        return [ORDER_PREFIX + quoted]
    raise ValueError(f"unknown OCR format {fmt!r}")


def append_tags(
    caption: str,
    lines: list[str],
    fmt: str = "order",
    *,
    kinds: list[str] | None = None,
    sfx_sentence: bool = False,
) -> str:
    """Append the OCR lines to the caption's flat bag, grammar-safe.

    ``tags`` / ``presence`` add a ``japanese text`` presence tag first unless
    the caption already carries a ``* text`` tag; ``order`` needs none — the
    phrase opens with it. ``kinds`` / ``sfx_sentence`` are the ``sentence``
    format's (:func:`ocr_text_clauses`).
    """
    from anime_tools.captions.position_clauses import compose_caption, parse_caption

    parsed = parse_caption(caption)
    if fmt == "sentence":
        # Grammar-native (anime_tools ≥ b453cc2): the text clauses compose
        # last, after the position clauses, and re-parse to the same string.
        clauses = ocr_text_clauses(lines, kinds, sfx_sentence)
        if not clauses:
            return caption
        return compose_caption(parsed.flat_tags, (*parsed.clauses, *clauses))
    extra = []
    if fmt in ("tags", "presence") and not TEXT_TAG_RE.search(caption):
        extra = ["japanese text"]
    return compose_caption(
        tuple(parsed.flat_tags) + tuple(extra) + tuple(ocr_tags(lines, fmt)),
        parsed.clauses,
    )


def _link_image(link: Path, target: Path) -> None:
    """Symlink ``target`` at ``link``; on the ntfs3-mounted dataset volume
    ``os.symlink`` fails sporadically (``FileNotFoundError`` on a path that
    exists — `ln -s` by hand works), so retry once and then fall back to a
    hard link (same filesystem), saying which happened."""
    try:
        link.symlink_to(target)
        return
    except OSError as first:
        try:
            link.symlink_to(target)
            print(f"symlink {link.name}: succeeded on retry ({first!r})")
            return
        except OSError:
            pass
        try:
            os.link(target, link)
            print(f"symlink {link.name} failed twice ({first!r}); hard-linked instead")
        except OSError as hard:
            raise OSError(
                f"could not link {link} -> {target}: symlink {first!r}, hardlink {hard!r}"
            ) from hard


def build_mirror(
    resized: Path,
    mirror: Path,
    tags: dict[str, list[str]] | dict[str, list[tuple[str, str]]],
    fmt: str = "order",
    stems: set[str] | None = None,
    sfx_sentence: bool = False,
) -> tuple[int, int]:
    """``stems`` restricts the mirror to those images (the single-image
    text-binding probe); ``None`` mirrors the whole shard. ``tags`` values are
    lines, or ``(line, kind)`` pairs (:func:`ocr_records_by_stem`) when the
    records' kind should drive the sentence split."""
    from anime_tools.captions.variants import read_variants_sidecar

    mirror.mkdir(parents=True, exist_ok=True)
    n_text = n_plain = 0
    for img in sorted(resized.glob("*.png")):
        if stems is not None and img.stem not in stems:
            continue
        if not img.resolve().exists():
            print(f"skip {img.name}: dangling resized link")
            continue
        link = mirror / img.name
        if not link.exists():
            _link_image(link, img.resolve())
        stem_tags = list(tags.get(img.stem, []))
        kinds = None
        if stem_tags and isinstance(stem_tags[0], tuple):
            kinds = [k for _, k in stem_tags]
            stem_tags = [t for t, _ in stem_tags]
        cap_src = resized / f"{img.stem}.txt"
        caption = (
            cap_src.read_text(encoding="utf-8").strip() if cap_src.exists() else ""
        )
        var_src = resized / f"{img.stem}.variants.txt"
        rows = read_variants_sidecar(var_src) if var_src.exists() else []
        if stem_tags:
            kw = dict(kinds=kinds, sfx_sentence=sfx_sentence)
            caption = append_tags(caption, stem_tags, fmt, **kw)
            rows = [
                (label, append_tags(text, stem_tags, fmt, **kw)) for label, text in rows
            ]
            n_text += 1
        else:
            n_plain += 1
        (mirror / f"{img.stem}.txt").write_text(caption + "\n", encoding="utf-8")
        if rows:
            out = ["# anima caption variants — auto-generated, do not hand-edit"]
            out += [f"{label}\t{text}" for label, text in rows]
            (mirror / f"{img.stem}.variants.txt").write_text(
                "\n".join(out) + "\n", encoding="utf-8"
            )
    return n_text, n_plain


def main() -> None:
    ckpt = default_checkpoints()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default="sincos")
    ap.add_argument("--records", type=Path, default=None)
    ap.add_argument(
        "--ext_prefix",
        type=Path,
        default=REPO / "output" / "ckpt" / "cjk_vocab_pack_synthjako2",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--mirror",
        type=Path,
        default=None,
        help="Mirror dir (default mirror_<shard>); pass a fresh one when re-caching "
        "with a different OCR engine so the arm-C mirror stays as trained.",
    )
    ap.add_argument("--dit", default=ckpt.dit)
    ap.add_argument("--qwen3", default=ckpt.text_encoder)
    ap.add_argument("--max_lines", type=int, default=8)
    ap.add_argument(
        "--ocr_format",
        choices=OCR_FORMATS,
        default="order",
        help="how OCR lines enter the caption: 'sentence' = a trailing "
        '\'Japanese text reads as "…", "…".\' speech clause + a '
        "'Japanese SFX reads as' clause (the default caption); 'order' = one 'Japanese "
        "text in following order: ...' phrase (reading order); 'tags' = the "
        "C2–C6 japanese text + 「…」 flat tags.",
    )
    ap.add_argument(
        "--keep_sfx",
        action="store_true",
        help="no-op since 2026-09-06 (the default): keep kind=sfx records and, "
        "with --ocr_format sentence, add the 'Japanese SFX reads as \"…\"' "
        "clause from the records' kind (hand labels + rule) after the speech "
        "clause. Was arm C11 (plan_ocr O4); kept so old argv still parse.",
    )
    ap.add_argument(
        "--drop_sfx",
        action="store_true",
        help="arms C2–C10 (pre-O4): drop kind=sfx records and emit no SFX "
        "clause — the caption default before the C11 gate passed.",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--stems",
        default=None,
        help="comma-separated image stems; restricts the mirror + cache to "
        "those images (default: the whole shard).",
    )
    opts = ap.parse_args()
    stems = (
        {s.strip() for s in opts.stems.split(",") if s.strip()} if opts.stems else None
    )

    base_dir = REPO / "post_image_dataset" / "cjk_unmask"
    # default = plan_det D1/D3 (2026-09-06): the AnimeText detector's boxes read
    # by the fine-tuned VL reader (`ocr/animetext_records.py`), kind from the
    # hand labels + rule. The O4c `_hybrid_vl` file (PP-OCRv6 DB → Spotting →
    # mask components, all-VL re-read) is what C10/C11 trained on — pass
    # --records to reproduce those.
    records = opts.records or base_dir / f"ocr_records_{opts.shard}_animetext.jsonl"
    out = opts.out or base_dir / "te" / opts.shard
    mirror = opts.mirror or base_dir / f"mirror_{opts.shard}"
    resized = REPO / "post_image_dataset" / "resized" / opts.shard

    keep_sfx = not opts.drop_sfx
    if keep_sfx:
        tags = ocr_records_by_stem(records, opts.max_lines, DROP_KINDS)
    else:
        tags = ocr_lines_by_stem(records, opts.max_lines, SFX_DROPPED)
    if stems is not None:
        missing = stems - set(tags)
        if missing:
            sys.exit(f"no OCR lines in {records} for stems: {sorted(missing)}")
        tags = {k: v for k, v in tags.items() if k in stems}
    n_text, n_plain = build_mirror(
        resized, mirror, tags, opts.ocr_format, stems, sfx_sentence=keep_sfx
    )
    print(
        f"mirror ({opts.ocr_format}): {n_text} captions carry OCR lines, "
        f"{n_plain} plain -> {mirror}"
    )

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy
    from library.preprocess.text import cache_text_embeddings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        opts.qwen3, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(None)
    llm_adapter = anima_utils.load_llm_adapter(
        opts.dit, dtype=torch.bfloat16, device=str(device)
    )

    ext_table, mapping = ext_vocab.load_ext_assets(opts.ext_prefix)
    emb = llm_adapter.embed
    new_w = torch.cat(
        [emb.weight.data, ext_table.to(emb.weight.dtype).to(emb.weight.device)]
    )
    llm_adapter.embed = torch.nn.Embedding.from_pretrained(new_w)
    base_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    hybrid = ext_vocab.HybridT5Encoder.from_mapping(
        t5_tokenizer, qwen3_tokenizer, mapping
    )
    strategy = ExtTokenizeStrategy(base_strategy, hybrid)
    print(f"ext vocab: +{ext_table.shape[0]} rows from {opts.ext_prefix}")

    # Sanity: one OCR'd caption must actually hit ext rows.
    for stem, stem_tags in list(tags.items())[:1]:
        cap = (mirror / f"{stem}.txt").read_text(encoding="utf-8").strip()
        ids, mask = hybrid.encode(cap, 512)
        n_ext = sum(1 for i in ids[: sum(mask)] if i >= ext_vocab.T5_TABLE_SIZE)
        print(f"sanity {stem}: {sum(mask)} t5 tokens, {n_ext} ext — {stem_tags[:2]}")
        if opts.ocr_format != "presence":
            assert n_ext > 0, "OCR caption produced no ext rows — pack/encoder mismatch"

    out.mkdir(parents=True, exist_ok=True)
    stats = cache_text_embeddings(
        mirror,
        strategy,
        AnimaTextEncodingStrategy(),
        text_encoder,
        llm_adapter=llm_adapter,
        device=device,
        cache_dir=out,
        batch_size=opts.batch_size,
        overwrite=opts.overwrite,
    )
    print(f"cached -> {out}: {stats}")


if __name__ == "__main__":
    main()
