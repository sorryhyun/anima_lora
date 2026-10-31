"""The colorize cache for a MOSS i2i LoRA: the B&W page enters through Qwen3.5's
vision tower and as a VAE reference; the colour original is the target.

The condition is ``<staging>/<dir>/<stem>.png`` (``make i2i-stage TASK=colorize``),
the target the same path under ``<resized>``; ``manga`` adds that many
``manga_color`` pages as ``manga__{anchor}_{lang}_{page}``. Per pair, in a flat
cache ``train.py`` reads unchanged:

    {stem}_{W}x{H}.moss_latent.safetensors   the colour target
    {stem}.moss_ref.safetensors              the page VAE-encoded at 1/``ref_scale``
                                             of the target's sides (multiples of 16)
    {stem}.moss_te.safetensors               the page, then one of three captions
                                             (``vision`` off: the caption alone):
        embeds        OCR caption: the colour caption's layout JSON plus one
                      ``{"label":"text",…,"bbox":…}`` item per OCR line
        embeds_v{k}   sparse caption, OCR items as ``embeds``: odd k drops
                      ``tag_dropout`` of the colour caption's tags (comic tags
                      kept), even k all of them; a comic page (``manga__`` or a
                      comic tag) keeps one comic tag, ``comic`` when its caption
                      has none (k = 1 … ``variants`` - 1; ``caption_variant_rate``
                      at training)
        embeds_plain  the page alone, ``vision`` only (``plain_caption_rate``
                      at training); with ``vision`` off there is no empty-prompt
                      row

The OCR reads the condition, never the target. Its floors are looser than the
stage's (``LINE_MIN_DET`` 0.6, glyph 16): the detector scores horizontal English
balloons 0.3–0.55 while the reader reads them right, so a box down to the
detector's 0.25 is kept when the read scores ``LOW_DET_SCORE``, and an all-ASCII
line's glyph floor is ``LATIN_MIN_GLYPH``. Boxes are MOSS's ``abs_centralized``
``yxyx`` in the canvas's pixels.

Every encode, ``<out>/uncond.safetensors`` included, uses ``text_encoder`` (fp8 by
default; ``te_file`` overrides the catalog's file). Holdout (``holdout``
illustration pairs, one page each of ``manga_holdout`` works, no edition of which
trains; ``holdout_json`` names the pages instead; ``train_json`` names the
training pages instead of the draw) lands in ``<out>/holdout/`` with
``holdout.json``; OCR reads in ``<out>/ocr.json``. A rerun reuses every file
already written.

    make i2i-preprocess TASK=colorize MODEL=moss
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.loader import empty_cache, load_tokenizer
from library.moss.scan import REF_SUFFIX, TE_SUFFIX

# ComfyUI-MOSS-Toolkit's MOSSCaptionHelper copies these floors, readable, dedupe and
# ocr_caption to build the same prompt at inference: change both together.
LOW_DET_SCORE = 0.9  # reader score a box under LINE_MIN_DET needs
LATIN_MIN_GLYPH = 12.0


def find_pairs(staging: Path, resized: Path) -> list[tuple[str, Path, Path]]:
    out = []
    for cond in sorted(staging.rglob("*.png")):
        rel = cond.relative_to(staging)
        target = resized / rel
        if target.exists() and target.with_suffix(".txt").exists():
            out.append((f"{rel.parent.name}__{rel.stem}", cond, target))
    return out


def find_manga_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    """``manga_color/{staging,resized}/{ja,en,ko}/{anchor}_{lang}_{page}.png``."""
    out = []
    for cond in sorted((root / "staging").rglob("*.png")):
        target = root / "resized" / cond.relative_to(root / "staging")
        if target.exists() and target.with_suffix(".txt").exists():
            out.append((f"manga__{cond.stem}", cond, target))
    return out


def split_manga(pairs, n, n_holdout, rng):
    """Hold out whole works (every edition of them), one page each; draw ``n`` from
    the rest."""
    works = {}
    for p in pairs:
        works.setdefault(p[2].stem.split("_")[0], []).append(p)
    anchors = sorted(works)
    rng.shuffle(anchors)
    holdout = [rng.choice(works[a]) for a in anchors[:n_holdout]]
    rest = [p for a in anchors[n_holdout:] for p in works[a]]
    rng.shuffle(rest)
    return rest[:n], holdout


def readable(ln) -> bool:
    from anime_tools.captions.ocr_sidecar import DEFAULT_MIN_GLYPH
    from anime_tools.ocr.reread import LINE_MIN_DET

    latin = all(ch.isascii() for ch in ln.text if ch.isalpha())
    floor = LATIN_MIN_GLYPH if latin else DEFAULT_MIN_GLYPH
    sure = ln.det >= LINE_MIN_DET or ln.score >= LOW_DET_SCORE
    return bool(ln.text) and sure and ln.glyph_px >= floor


def read_pages(conditions: list[Path], device: str) -> list[list]:
    """Per condition, the readable lines as ``[text, [x0, y0, x1, y1]]`` in reading
    order, in the condition's pixels."""
    from anime_tools.ocr import load_ocr
    from anime_tools.ocr.animetext import DEFAULT_CONF
    from anime_tools.ocr.reread import RereadEngine
    from anime_tools.ocr.sfx import SfxReader

    reader = SfxReader.load(device=device)
    engine = RereadEngine(
        engine=load_ocr(device=device),
        read_boxes=reader.read_boxes_scored,
        resized_dir=Path("/"),
        skip_en=False,
        min_det=DEFAULT_CONF,
    )
    out, t0 = [], time.time()
    for i, lines in enumerate(engine.read_iter(conditions), 1):
        out.append([[ln.text, list(ln.box)] for ln in lines if readable(ln)])
        if i % 50 == 0 or i == len(conditions):
            print(f"  ocr {i}/{len(conditions)}, {time.time() - t0:.0f}s", flush=True)
    del engine, reader
    empty_cache()
    return out


def dedupe(lines: list[list]) -> list[tuple[str, list]]:
    """``[(text, box)]``, ``"`` → ``”``, first of each text kept."""
    seen, keep = set(), []
    for text, box in lines:
        text = text.replace('"', "”").strip()
        if text and text not in seen:
            seen.add(text)
            keep.append((text, box))
    return keep


def ocr_caption(caption, lines, size, kb) -> str:
    from anime_tools.captions.layout_json import caption_to_layout, centered_box

    layout = caption_to_layout(caption, kb)
    items = [
        {"label": "text", "caption": "text", "text": t, "bbox": centered_box(b, size)}
        for t, b in lines
    ]
    if items:
        layout["items"] = layout.get("items", []) + items
        layout["bbox_coord_space"] = "abs_centralized"
        layout["bbox_format"] = "yxyx"
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def sparse_captions(colour, n, dropout, comic_page, is_comic_tag) -> list[str]:
    """The ``n`` sparse colour captions, partial (``dropout`` of the tags, comic
    tags kept) and bare (one comic tag on a comic page, else nothing) in turn."""
    from anime_tools.captions.position_clauses import parse_caption
    from anime_tools.captions.variants import generate_caption_variants

    partial = generate_caption_variants(colour, (n + 1) // 2 + 1, dropout, is_comic_tag)
    comic = [t for t in parse_caption(colour).flat_tags if is_comic_tag(t)]
    if not comic and comic_page:
        comic = ["comic"]
    return [
        partial[k // 2 + 1] if k % 2 == 0 else random.choice(comic) if comic else ""
        for k in range(n)
    ]


def _color_caption():
    """``i2i_tasks/colorization/color_caption.py`` (not an installed package)."""
    sys.path.insert(0, str(resolve_under_home("i2i_tasks/colorization")))
    import color_caption

    return color_caption


def run_colorize_cache(req) -> Path:
    """Build ``req.out``; returns it."""
    from library.moss.vision_te import encode, encode_variants, load_te

    if req.variants < 3:
        raise SystemExit(
            "--variants must be at least 3: embeds_v1 is partial, embeds_v2 bare"
        )
    out = resolve_under_home(req.out)
    (out / "holdout").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    rng = random.Random(req.seed)
    staging = resolve_under_home(req.staging)
    pairs = find_pairs(staging, resolve_under_home(req.resized))
    if not pairs:
        raise SystemExit(
            f"no staged pages under {staging} (make i2i-stage TASK=colorize)"
        )
    rng.shuffle(pairs)
    holdout, train = pairs[: req.holdout], pairs[req.holdout :][: req.n]
    m_train, m_holdout = split_manga(
        find_manga_pairs(resolve_under_home(req.manga_dir)) if req.manga else [],
        req.manga,
        req.manga_holdout,
        rng,
    )
    train += m_train
    holdout += m_holdout
    if req.holdout_json:
        rows = json.loads(resolve_under_home(req.holdout_json).read_text())
        holdout = [(r["stem"], Path(r["condition"]), Path(r["target"])) for r in rows]
        held = {stem for stem, _, _ in holdout}
        works = {
            s.split("__")[1].split("_")[0] for s in held if s.startswith("manga__")
        }
        train = [
            p
            for p in train
            if p[0] not in held
            and not (p[0].startswith("manga__") and p[2].stem.split("_")[0] in works)
        ]
    if req.train_json:
        held = {stem for stem, _, _ in holdout}
        stems = set(json.loads(resolve_under_home(req.train_json).read_text())) - held
        manga_dir = resolve_under_home(req.manga_dir)
        pool = pairs + (find_manga_pairs(manga_dir) if manga_dir.is_dir() else [])
        train = [p for p in pool if p[0] in stems]
        if missing := stems - {p[0] for p in train}:
            raise SystemExit(
                f"{len(missing)} train_json stems have no staged pair, e.g. "
                f"{sorted(missing)[:3]}"
            )
        m_train = [p for p in train if p[0].startswith("manga__")]
    print(f"{len(train)} train ({len(m_train)} manga), {len(holdout)} held out")

    def te_path(stem, held):
        return out / ("holdout" if held else "") / f"{stem}{TE_SUFFIX}"

    def ref_path(stem, held):
        return out / ("holdout" if held else "") / f"{stem}{REF_SUFFIX}"

    # (stem, condition, target, held out, (w, h))
    jobs = []
    for held, rows in ((False, train), (True, holdout)):
        for stem, cond, target in rows:
            with Image.open(target) as im:
                size = moss_cache.target_size(*im.size, req.resolution)
            jobs.append((stem, cond, target, held, size))

    moss_cache.cache_latents(
        [moss_cache.Pair(stem, target, "", "") for stem, _, target in train],
        out,
        req.model_dir,
        device,
        req.resolution,
        "fp32",
        False,
        False,
    )

    todo = [j for j in jobs if not ref_path(j[0], j[3]).exists()]
    if todo:
        from library.moss.loader import load_vae
        from library.moss.model import encode as vae_encode

        def load(job):
            _, cond, _, _, (w, h) = job
            page = moss_cache.fit(moss_cache.load_rgb(cond), w, h)
            rw, rh = moss_cache.ref_size(w, h, req.ref_scale)
            return moss_cache.to_pixels(moss_cache.fit(page, rw, rh))

        vae = load_vae(req.model_dir, device=device)
        for i, ((stem, cond, _, held, (w, h)), pixels) in enumerate(
            moss_cache.prefetch(todo, load), 1
        ):
            rw, rh = moss_cache.ref_size(w, h, req.ref_scale)
            latent = vae_encode(pixels, vae)
            save_file(
                {"latent": latent[0].cpu().contiguous()},
                str(ref_path(stem, held)),
                metadata={
                    "width": str(rw),
                    "height": str(rh),
                    "target": f"{w}x{h}",
                    "ref_scale": str(req.ref_scale),
                    "condition": str(cond),
                },
            )
            print(f"  ref {i}/{len(todo)} {stem}: {w}x{h} -> {rw}x{rh}", flush=True)
        del vae
        empty_cache()

    ocr_path = out / "ocr.json"
    reads = json.loads(ocr_path.read_text()) if ocr_path.exists() else {}
    missing = [j for j in jobs if j[0] not in reads]
    if missing:
        for j, lines in zip(missing, read_pages([j[1] for j in missing], "cuda")):
            reads[j[0]] = lines
        ocr_path.write_text(json.dumps(reads, ensure_ascii=False, indent=1))
    print(
        f"{sum(bool(reads[j[0]]) for j in jobs)} of {len(jobs)} pages carry OCR lines",
        flush=True,
    )

    cc = _color_caption()
    copyright_tags = cc.load_copyright_tags()
    kb = moss_cache.tag_knowledge_base()
    # stem: (colour caption, OCR caption, sparse OCR captions)
    captions = {}
    for stem, cond, target, _, (w, h) in jobs:
        colour = cc.filter_to_colors_and_protected(
            target.with_suffix(".txt").read_text(encoding="utf-8").strip(),
            copyright_tags,
            keep_copyright=True,
            keep_comic=True,
        )
        with Image.open(cond) as im:
            cw, ch = im.size
        lines = [
            (t, [b[0] * w / cw, b[1] * h / ch, b[2] * w / cw, b[3] * h / ch])
            for t, b in dedupe(reads[stem])
        ]
        random.seed(f"{req.seed}:{stem}")
        sparse = sparse_captions(
            colour,
            req.variants - 1,
            req.tag_dropout,
            stem.startswith("manga__"),
            cc.is_comic_tag,
        )
        captions[stem] = (
            colour,
            ocr_caption(colour, lines, (w, h), kb),
            [ocr_caption(s, lines, (w, h), kb) for s in sparse],
        )

    todo = [j for j in jobs if not te_path(j[0], j[3]).exists()]
    uncond_path = out / "uncond.safetensors"
    if todo or not uncond_path.exists():
        tok = load_tokenizer(req.model_dir)
        te = load_te(
            tok,
            device,
            req.model_dir,
            precision=req.text_encoder,
            fp8_path=req.te_file or None,
        )
        encoder = f"Qwen3.5-9B {req.text_encoder}" + (
            f" ({req.te_file})" if req.te_file else ""
        )
        t0 = time.time()
        for i, (stem, cond, target, held, (w, h)) in enumerate(todo, 1):
            colour, ocr, sparse = captions[stem]
            texts = [ocr, *([""] if req.vision else []), *sparse]
            if req.vision:
                page = moss_cache.fit(moss_cache.load_rgb(cond), w, h)
                encoded = encode_variants(tok, te, page, texts, w * h, device)
            else:
                # No shared page rows: each variant is stored whole (prefix 0).
                encoded = [(*encode(tok, te, None, t, 0, device), 0) for t in texts]
            (embeds, n_img, prefix), *rest = encoded
            plain = rest.pop(0)[0].to("cpu", torch.bfloat16) if req.vision else None
            var, var_meta = moss_cache.variant_entries(
                [
                    (t, e.to("cpu", torch.bfloat16))
                    for t, (e, _, _) in zip(sparse, rest)
                ],
                prefix,
            )
            save_file(
                {
                    "embeds": embeds.to("cpu", torch.bfloat16).contiguous(),
                    **({"embeds_plain": plain.contiguous()} if req.vision else {}),
                    **var,
                },
                str(te_path(stem, held)),
                metadata={
                    "caption": ocr,
                    "colour_caption": colour,
                    "n_text": str(embeds.shape[0]),
                    **({"n_text_plain": str(plain.shape[0])} if req.vision else {}),
                    "n_image_tokens": str(n_img),
                    "condition": str(cond),
                    "source": str(target),
                    "encoder": f"{encoder} + vision bf16, image then caption"
                    if req.vision
                    else f"{encoder}, caption only",
                    **var_meta,
                },
            )
            print(
                f"  te {i}/{len(todo)} {stem}: {embeds.shape[0]} rows ({n_img} image), "
                f"{len(sparse)} sparse, {time.time() - t0:.0f}s",
                flush=True,
            )
        uncond, _ = encode(tok, te, None, "", 0, device)
        save_file(
            {"embeds": uncond.to("cpu", torch.bfloat16).contiguous()},
            str(uncond_path),
            metadata={"encoder": encoder},
        )
        del te, tok
        empty_cache()

    records = [
        {
            "stem": stem,
            "condition": str(cond),
            "target": str(target),
            "caption": captions[stem][1],
            "caption_v1": captions[stem][2][0],
            "caption_v2": captions[stem][2][1],
            "width": w,
            "height": h,
            "te": str(te_path(stem, True)),
            "ref": str(ref_path(stem, True)),
        }
        for stem, cond, target, held, (w, h) in jobs
        if held
    ]
    (out / "holdout.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"wrote {out}", flush=True)
    return out
