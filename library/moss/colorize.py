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

``captions = "control"`` replaces the three: ``embeds`` and each ``embeds_v{k}`` are
draws keeping ``1 - tag_dropout`` of a pool, and ``embeds_plain`` is the bare
caption. No OCR. An illustration's pool is its colour + series tags; its bare
caption is empty. A manga page always carries ``comic`` and a box-only ``text`` item
per OCR line of the condition that ``keep_line`` keeps, swap_text's caption floor
(``text_boxes.json``, the reads kept); its pool is each
character's hair / eye / clothing colours (SAM3 + Anima Tagger on the target, a
``character`` item with its box; ``characters.json``), and its bare caption is
``comic`` and the text lines alone.

The OCR reads the condition, never the target. Its floors are looser than the
stage's (``LINE_MIN_DET`` 0.6, glyph 16): the detector scores horizontal English
balloons 0.3–0.55 while the reader reads them right, so a box down to the
detector's 0.25 is kept when the read scores ``LOW_DET_SCORE``, and an all-ASCII
line's glyph floor is ``LATIN_MIN_GLYPH``. Boxes are MOSS's ``abs_centralized``
``yxyx`` in the canvas's pixels.

Every encode, ``<out>/uncond.safetensors`` included, uses ``text_encoder`` (int8 by
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
import time
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from anime_tools.ocr.floors import keep_line, load_reread_engine, readable

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.loader import empty_cache, load_tokenizer
from library.moss.scan import REF_SUFFIX, TE_SUFFIX

# The OCR floors and the colour tags live in anime_tools (ComfyUI-MOSS-Toolkit's
# MOSSCaptionHelper reads the same ones); the caption index holds the series tags.
CAPTION_INDEX = "post_image_dataset/captions/caption_index.json"


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


def read_pages(conditions: list[Path], device: str, keep=readable) -> list[list]:
    """Per condition, the lines ``keep`` keeps as ``[text, [x0, y0, x1, y1]]`` in
    reading order, in the condition's pixels."""
    engine = load_reread_engine(device)
    out, t0 = [], time.time()
    for i, lines in enumerate(engine.read_iter(conditions), 1):
        out.append([[ln.text, list(ln.box)] for ln in lines if keep(ln)])
        if i % 50 == 0 or i == len(conditions):
            print(f"  ocr {i}/{len(conditions)}, {time.time() - t0:.0f}s", flush=True)
    del engine
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


def keep_draw(atoms: list, dropout: float, rng: random.Random) -> list:
    """Each atom kept with probability ``1 - dropout``; one at least when there are
    any (the bare caption is the empty draw)."""
    kept = [a for a in atoms if rng.random() >= dropout]
    if atoms and not kept:
        kept = [rng.choice(atoms)]
    return kept


def illustration_draws(pool: list[str], n, dropout, rng, kb) -> list[str]:
    """``n`` layout captions over a draw of the colour + series tags each."""
    from anime_tools.captions.layout_json import layout_json

    if not pool:
        return [""] * n
    return [layout_json(", ".join(keep_draw(pool, dropout, rng)), kb) for _ in range(n)]


def manga_layout(characters: list, text_boxes: list, size) -> str:
    """``comic``, a ``character`` item with its box per character (its colours),
    then a box-only ``text`` item per text line."""
    from anime_tools.captions.layout_json import centered_box

    items = [
        {
            "label": "character",
            "caption": ", ".join(tags),
            "bbox": centered_box(b, size),
        }
        for b, tags in characters
        if tags
    ]
    items += [{"label": "text", "bbox": centered_box(b, size)} for b in text_boxes]
    layout: dict = {"summary": {"caption": "comic"}}
    if items:
        layout.update(
            items=items, bbox_coord_space="abs_centralized", bbox_format="yxyx"
        )
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))


def manga_draws(characters: list, text_boxes: list, size, n, dropout, rng) -> list[str]:
    """``n`` layout captions: ``comic`` and every text line always, each
    character's colours a draw dropping ``dropout`` of them (one kept at least;
    the bare caption is :func:`manga_layout` with no characters)."""
    atoms = [(i, t) for i, (_, tags) in enumerate(characters) for t in tags]
    out = []
    for _ in range(n):
        kept = keep_draw(atoms, dropout, rng)
        chars = [
            (b, [t for c, t in kept if c == i]) for i, (b, _) in enumerate(characters)
        ]
        out.append(manga_layout(chars, text_boxes, size))
    return out


def detect_characters(targets: list[Path], device: str) -> list[list]:
    """Per colour target, ``[[x0, y0, x1, y1], [colour tags]]`` for each character
    with a colour tag, in the target's pixels, top to bottom: the position stage's
    SAM3 subject detector, its mask-blanked crop, the Anima Tagger's kept tags that
    are colours (unfiltered: ``is_character_colour`` applies at caption time)."""
    from anime_tools.stages._models import load_tagger, release_models
    from anime_tools.stages.detector import build_detect_fn
    from anime_tools.stages.instance_detection import crop_instance
    from anime_tools.stages.position_captions import detect_subjects
    from anime_tools.stages.requests import PositionRequest

    cc = _color_caption()
    preq = PositionRequest(device=device)
    options = preq.options()
    detect_fn, _, sam_model, sam_processor = build_detect_fn(
        preq.detection, device=device
    )
    tagger, _, _ = load_tagger(preq)
    out, t0 = [], time.time()
    for i, path in enumerate(targets, 1):
        image = moss_cache.load_rgb(path)
        chars = []
        for d in sorted(
            detect_subjects(image, detect_fn, options, None),
            key=lambda d: (d.box[1], d.box[0]),
        ):
            crop = crop_instance(image, d, pad=options.pad, blank=options.blank_crops)
            kept = tagger.predict(crop).get("kept") or {}
            tags = [t for t in kept if cc.is_color_tag(t)]
            if tags:
                chars.append([[float(v) for v in d.box], tags])
        out.append(chars)
        if i % 50 == 0 or i == len(targets):
            print(
                f"  characters {i}/{len(targets)}, {time.time() - t0:.0f}s", flush=True
            )
    del sam_model, sam_processor, tagger, detect_fn
    release_models()
    empty_cache()
    return out


def _color_caption():
    from anime_tools.captions import color_tags

    return color_tags


def run_colorize_cache(req) -> Path:
    """Build ``req.out``; returns it."""
    from library.moss.vision_te import encode, encode_variants, load_te

    if req.variants < 3:
        raise SystemExit(
            "--variants must be at least 3: embeds_v1 is partial, embeds_v2 bare"
        )
    control = req.captions == "control"
    if control and req.vision:
        raise SystemExit("--captions control encodes the caption alone: --no-vision")
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

    cc = _color_caption()
    copyright_tags = cc.load_copyright_tags(resolve_under_home(CAPTION_INDEX))
    kb = moss_cache.tag_knowledge_base()
    # stem: (colour caption, caption, variant captions, bare caption or None)
    captions = {}
    if control:
        manga_jobs = [j for j in jobs if j[0].startswith("manga__")]
        chars_path, text_path = out / "characters.json", out / "text_boxes.json"
        found = json.loads(chars_path.read_text()) if chars_path.exists() else {}
        blocks = json.loads(text_path.read_text()) if text_path.exists() else {}
        if missing := [j for j in manga_jobs if j[0] not in blocks]:
            reads = read_pages([j[1] for j in missing], "cuda", keep=keep_line)
            for j, lines in zip(missing, reads):
                blocks[j[0]] = lines
            text_path.write_text(json.dumps(blocks, ensure_ascii=False, indent=1))
        if missing := [j for j in manga_jobs if j[0] not in found]:
            for j, chars in zip(
                missing, detect_characters([j[2] for j in missing], "cuda")
            ):
                found[j[0]] = chars
            chars_path.write_text(json.dumps(found, ensure_ascii=False, indent=1))
        for stem, cond, target, _, (w, h) in jobs:
            rng = random.Random(f"{req.seed}:{stem}")
            if stem.startswith("manga__"):
                with Image.open(target) as im:
                    tw, th = im.size
                with Image.open(cond) as im:
                    cw, ch = im.size
                chars = [
                    (
                        [b[0] * w / tw, b[1] * h / th, b[2] * w / tw, b[3] * h / th],
                        [x for x in t if cc.is_character_colour(x)],
                    )
                    for b, t in found[stem]
                ]
                texts = [
                    [b[0] * w / cw, b[1] * h / ch, b[2] * w / cw, b[3] * h / ch]
                    for _, b in blocks[stem]
                ]
                pool = "; ".join(", ".join(t) for _, t in chars) + (
                    f" + {len(texts)} text lines"
                )
                draws = manga_draws(
                    chars, texts, (w, h), req.variants, req.tag_dropout, rng
                )
                bare = manga_layout([], texts, (w, h))
            else:
                pool = cc.filter_to_colors_and_protected(
                    target.with_suffix(".txt").read_text(encoding="utf-8").strip(),
                    copyright_tags,
                    keep_copyright=True,
                    keep_comic=False,
                )
                tags = [t.strip() for t in pool.split(",") if t.strip()]
                draws = illustration_draws(tags, req.variants, req.tag_dropout, rng, kb)
                bare = ""
            captions[stem] = (pool, draws[0], draws[1:], bare)
        print(
            f"of {len(manga_jobs)} manga pages, {sum(bool(found[j[0]]) for j in manga_jobs)}"
            f" carry a character colour, {sum(bool(blocks[j[0]]) for j in manga_jobs)}"
            " a text line",
            flush=True,
        )
    else:
        ocr_path = out / "ocr.json"
        reads = json.loads(ocr_path.read_text()) if ocr_path.exists() else {}
        missing = [j for j in jobs if j[0] not in reads]
        if missing:
            for j, lines in zip(missing, read_pages([j[1] for j in missing], "cuda")):
                reads[j[0]] = lines
            ocr_path.write_text(json.dumps(reads, ensure_ascii=False, indent=1))
        print(
            f"{sum(bool(reads[j[0]]) for j in jobs)} of {len(jobs)} pages carry OCR "
            "lines",
            flush=True,
        )
    for stem, cond, target, _, (w, h) in [] if control else jobs:
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
            None,
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
            colour, ocr, sparse, bare = captions[stem]
            has_plain = req.vision or bare is not None
            texts = [ocr, *([bare or ""] if has_plain else []), *sparse]
            if req.vision:
                page = moss_cache.fit(moss_cache.load_rgb(cond), w, h)
                encoded = encode_variants(tok, te, page, texts, w * h, device)
            else:
                # No shared page rows: each variant is stored whole (prefix 0).
                encoded = [(*encode(tok, te, None, t, 0, device), 0) for t in texts]
            (embeds, n_img, prefix), *rest = encoded
            plain = rest.pop(0)[0].to("cpu", torch.bfloat16) if has_plain else None
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
                    **({"embeds_plain": plain.contiguous()} if has_plain else {}),
                    **var,
                },
                str(te_path(stem, held)),
                metadata={
                    "caption": ocr,
                    "colour_caption": colour,
                    "n_text": str(embeds.shape[0]),
                    **({"n_text_plain": str(plain.shape[0])} if has_plain else {}),
                    **({"caption_plain": bare} if bare is not None else {}),
                    "captions": req.captions,
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
            **(
                {"caption_plain": captions[stem][3]}
                if captions[stem][3] is not None
                else {}
            ),
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
