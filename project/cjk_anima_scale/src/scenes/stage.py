"""Stage ``scenes`` — self-generated in-domain scenes with one EN-anchored
speech bubble (plan_synth, S line).

The base model draws the scene itself from a combinatorial tag prompt plus
``{bubble tag}, english text. English text reads as "{anchor}"`` (short
Latin anchor, every piece pretrained, no ext row touched). The detector
finds the text box; an image is kept only when **exactly one** box is
found, a reader reads the anchor back, and the box is at least
``--scene_min_box`` px on its short side. The data stage later erases the
box, draws JA text into it and swaps the EN clause for the JA one — so the
scene is explained by its own tags and the only thing a row can learn is
the glyph.

Writes ``<OUT>/scenes_<tag>/{img/, scenes.jsonl, scenes_all.jsonl,
sheet_kept.png, sheet_rejected.png, report.md}``. ``scenes.jsonl`` rows:
``file, tags, prompt, anchor, box, shape, seed, read``.

The prompt stream is deterministic in ``--seed``, so a pool grows by re-running
its own argv with a larger ``--scene_n``: stored rows are kept verbatim, only
the new indices are rendered and judged. ``--scene_prune 1`` deletes the
rejected renders (rows stay); ``--scene_n 0 --scene_prune 1`` is that sweep
alone. ``--scene_rejudge 1`` re-applies the judge from the stored reads.

The 8 blind-pairs prompts (``native``'s held-out eval) are kept out of the
vocabulary: no setting / action / style token that identifies one of them
appears below, so ``native`` stays held out of the training scenes.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

from common.models import decode_image, gen_args, load_generator, load_vae
from common.paths import OUT
from common.readers import load_bgr
from common.shapes import parse_shapes

# vocabulary — held-out prompt tokens NOT here: classroom, sitting at desk,
# bedroom, on bed, hug, curtains, park bench, eating ice cream, greyscale,
# sitting on a windowsill, film grain, maid, holding a tray, cafe interior,
# upper body, portrait, simple background, comic, 2koma, surprised expression
# single-character prompts dominate: two characters → the base draws the
# anchor once per speaker (fine, every anchor bubble is swapped) but also
# far more stray text
COUNTS = ["1girl"] * 5 + ["1boy"] * 3 + ["2girls", "1boy, 1girl", "2boys"]
APPEARANCE = [
    "short hair",
    "long hair",
    "twintails",
    "ponytail",
    "glasses",
    "black hair",
    "brown hair",
    "blonde hair",
    "blue eyes",
    "brown eyes",
    "school uniform",
    "hoodie",
    "dress",
    "jacket",
    "t-shirt",
    "hat",
]
SETTINGS = [
    ["library"],
    ["rooftop"],
    ["kitchen"],
    ["train interior"],
    ["beach"],
    ["city", "street"],
    ["shrine"],
    ["hallway"],
    ["bus stop"],
    ["garden"],
    ["convenience store"],
    ["living room"],
    ["gym"],
    ["stairs"],
    ["night", "city lights"],
    ["office"],
    ["restaurant"],
    ["hospital"],
    ["balcony"],
    ["forest"],
    ["river"],
    ["school"],
    ["bookstore"],
    ["rain", "umbrella"],
    ["snow", "winter"],
    ["festival", "lantern"],
    ["pool"],
    ["locker room"],
    ["shopping mall"],
    ["rooftop", "sunset"],
]
ACTIONS = [
    "standing",
    "walking",
    "running",
    "waving",
    "holding book",
    "holding phone",
    "pointing at viewer",
    "crossed arms",
    "hands on hips",
    "sitting",
    "leaning",
    "holding cup",
    "reading",
    "cooking",
    "hand on own chin",
    "v",
    "crossed legs",
    "hands in pockets",
    "looking back",
    "stretching",
    "holding bag",
    "playing guitar",
]
EXPRESSIONS = [
    "smile",
    "grin",
    "open mouth",
    "angry",
    "crying",
    "closed eyes",
    "one eye closed",
    "embarrassed",
    "serious",
    "laughing",
    "pout",
    "sweatdrop",
    "shouting",
    "sleepy",
    "blush",
    "nervous",
]
STYLES = [
    ["screentone"],
    ["monochrome"],
    ["sketch"],
    ["anime coloring"],
    ["flat color"],
    ["watercolor (medium)"],
    ["lineart"],
    ["halftone"],
    [],
    [],
]
FRAMING = ["cowboy shot", "full body", "from side", ""]
ANCHORS = ["hi", "ok", "no", "yes", "wow", "hey", "oh", "huh", "yay", "wait"]
# In-domain identity (user, 2026-09-14: artist-less / character-less prompts
# are off-distribution for the base — the first smoke sheet was a generic
# average). Caption form is `rating, count, character, copyright, @artist,
# generals sorted alphabetically`, exactly as the dataset captions are
# written; ratings safe / sensitive; (character, copyright) pairs frequent
# in the dataset for 1girl counts, else `original`; artists are the
# dataset's 83 `@name` rows.
RATINGS = ["safe", "sensitive"]
CHARACTERS = [
    ("hatsune miku", "vocaloid"),
    ("asuna (blue archive)", "blue archive"),
    ("gotoh hitori", "bocchi the rock!"),
    ("fujita kotone", "gakuen idolmaster"),
    ("the herta (honkai: star rail)", "honkai: star rail"),
    ("ellen joe", "zenless zone zero"),
    ("nakano ichika", "go-toubun no hanayome"),
    ("iino miko", "kaguya-sama wa kokurasetai ~tensai-tachi no renai zunousen~"),
    ("zuikaku (kancolle)", "kantai collection"),
    ("phoebe (wuthering waves)", "wuthering waves"),
    ("pavolia reine", "hololive"),
    ("nakiri ayame", "hololive"),
    ("holo", "spice and wolf"),
    ("takagi-san", "karakai jouzu no takagi-san"),
]


def artist_pool() -> list[str]:
    """``@artist`` tags as the dataset spells them (one per artist dir under
    ``post_image_dataset/resized``); empty when the tree is absent."""
    from common.paths import REPO

    root = REPO / "post_image_dataset" / "resized"
    out = []
    for d in sorted(root.iterdir()) if root.exists() else []:
        if not d.is_dir():
            continue
        key = d.name.lower().replace("_", " ")
        found = None
        for t in list(d.rglob("*.txt"))[:20]:
            for ln in t.read_text(errors="ignore").splitlines():
                if ln.startswith("#"):
                    continue
                ln = ln.split("\t", 1)[-1]
                for x in [y.strip() for y in ln.split(".")[0].split(",")][:6]:
                    if x.startswith("@") and key in x.lower():
                        found = x
                        break
                if found:
                    break
            if found:
                break
        if found:
            out.append(found)
    return out


TPL_SCENE = '{tags}. English text reads as "{anchor}".'

# Text frames (user, 2026-09-15 night): how the prompt asks for the anchor.
# Every s0 composite was a `reads as` bubble, so the rows learned the bubble
# as their canvas (flat-0 arm) and stayed bound to the one clause (swap
# clause: か 0/16). Each frame = (generals it adds, clause template with
# `{a}` = anchor, `{pro}` = She/He from the count). Pronoun frames are drawn
# for solo counts only. The data stage swaps the anchor for the JA text in
# the *same* frame (`clause_tpl` on the record) — so the composite caption
# carries the frame the base drew the scene under.
# `sfx` (user, 2026-09-15): the trainer's OCR clause grammar is `Japanese SFX
# reads as "…"`, so the EN side mirrors it and the JA swap *is* the product
# caption. SFX are drawn bubble-less, so the frame keeps its own anchors
# (onomatopoeia the base letters big) and passes an open fill.
FRAMES = {
    "reads_as": (["{bubble}", "english text"], 'English text reads as "{a}".'),
    "bubble_reads": (
        ["{bubble}", "english text"],
        'There is a speech bubble that reads "{a}".',
    ),
    "saying": (["{bubble}", "english text"], '{pro} is saying "{a}".'),
    "sign": (
        ["holding sign", "sign", "english text"],
        '{pro} is holding a sign that reads "{a}".',
    ),
    "bare_quotes": (["{bubble}", "english text"], '"{a}".'),
    "sfx": (["sound effects", "english text"], 'English SFX reads as "{a}".'),
    # JA frames (user, 2026-09-16): the base writes EN horizontally and so
    # draws wide bubbles (sl1w: 22 % taller than wide, height median 136 px —
    # not moved by anchor length, frame or canvas). The one lever that can
    # change the bubble's shape is asking for *Japanese* text: the letters
    # come out garbled and are erased anyway; what is kept is the bubble.
    # The judge takes every detector box as an anchor box (no read match),
    # and the data stage's EN→JA swap is the identity on these frames.
    "ja_reads_as": (["{bubble}", "japanese text"], 'Japanese text reads as "{a}".'),
    "ja_bubble_reads": (
        ["{bubble}", "japanese text"],
        'There is a speech bubble that reads "{a}".',
    ),
    "ja_saying": (["{bubble}", "japanese text"], '{pro} is saying "{a}".'),
}
JA_FRAMES = {f for f in FRAMES if f.startswith("ja_")}
# short manga lines (3–8 glyphs) — long enough that the base draws a real
# bubble, short enough to stay one utterance; --scene_ja_anchors overrides
JA_ANCHORS = [
    "ちょっとまって",
    "なんだそれ",
    "だいじょうぶ？",
    "いってきます",
    "ただいま",
    "ごめんなさい",
    "ありがとう",
    "しらないよ",
    "そうなんだ",
    "まあいいか",
    "やめてよ",
    "おはよう",
    "またあした",
    "どうしたの",
    "うそでしょ",
    "わかった",
    "いやだ",
    "なにこれ",
    "やったあ",
    "おなかすいた",
    "ねえきいて",
    "もういいよ",
    "たすけて",
    "いくよ",
]
SFX_ANCHORS = [
    "BAM",
    "BOOM",
    "BANG",
    "WHAM",
    "THUD",
    "CRASH",
    "POW",
    "ZAP",
    "WHOOSH",
    "SLAM",
]
FRAME_ANCHORS = {"sfx": SFX_ANCHORS, **{f: JA_ANCHORS for f in JA_FRAMES}}
FRAME_OPEN_OK = {"sfx"}  # no bubble expected: an open fill is not a reject
PRONOUN = {"1girl": "She", "1boy": "He"}


def frame_clause(frame: str, count: str, anchor: str, bubble_tag: str):
    """``(generals, clause_tpl, clause)`` for one frame; ``clause_tpl`` keeps
    ``{a}`` so the data stage can re-fill it with the JA text."""
    gens, tpl = FRAMES[frame]
    gens = [bubble_tag if g == "{bubble}" else g for g in gens]
    tpl = tpl.replace("{pro}", PRONOUN.get(count, "She"))
    return gens, tpl, tpl.format(a=anchor)


def compose(head: list[str], generals: list[str]) -> str:
    """``rating, count, character, copyright, @artist, <generals sorted>``."""
    return ", ".join(head + sorted(set(generals)))


class ScenePool:
    """``--scene_shapes`` canvas draws (own rng stream)."""

    def __init__(self, spec: str, seed: int):
        self.shapes = parse_shapes(spec) or [(512, 512, 1.0)]
        self.rng = random.Random(seed + 31)

    def draw(self):
        W, H, _ = self.rng.choices(self.shapes, weights=[x[2] for x in self.shapes])[0]
        return (W, H)


def scene_items(a) -> list[dict]:
    """``--scene_n`` distinct prompts, one anchor + canvas each; deterministic
    in ``--seed`` (stream seed+29; shapes seed+31)."""
    rng = random.Random(a.seed + 29)
    pool = ScenePool(a.scene_shapes, a.seed)
    anchors = [x for x in a.scene_anchors.split(",") if x] or ANCHORS
    frames = [x for x in a.scene_frames.split(",") if x] or ["reads_as"]
    unknown = [f for f in frames if f not in FRAMES]
    assert not unknown, f"--scene_frames: unknown {unknown}; have {list(FRAMES)}"
    frame_anchors = dict(FRAME_ANCHORS)
    ja_anchors = [x for x in a.scene_ja_anchors.split(",") if x]
    if ja_anchors:
        frame_anchors.update({f: ja_anchors for f in JA_FRAMES})
    extra = [x.strip() for x in a.scene_extra_tags.split(",") if x.strip()]
    artists = artist_pool() if a.scene_artist_frac > 0 else []
    # curated well-known artists (user, 2026-09-14: sincos, hews) at 4× weight
    curated = [f"@{x.strip()}" for x in a.scene_artists.split(",") if x.strip()]
    artists = artists + curated * 4
    print(
        f"scenes: {len(artists)} artist draws ({len(curated)} curated ×4)", flush=True
    )
    seen: set = set()
    items = []
    tries = 0
    while len(items) < a.scene_n and tries < 50 * a.scene_n:
        tries += 1
        count = rng.choice(COUNTS)
        if count == "1girl" and rng.random() < a.scene_char_frac:
            ident = list(rng.choice(CHARACTERS))
        else:
            ident = ["original"]
        if artists and rng.random() < a.scene_artist_frac:
            ident.append(rng.choice(artists))
        head = [rng.choice(RATINGS), count, *ident]
        solo = count in ("1girl", "1boy")
        frame = rng.choice(frames)
        if not solo and "{pro}" in FRAMES[frame][1]:
            frame = rng.choice(
                [f for f in frames if "{pro}" not in FRAMES[f][1]] or ["reads_as"]
            )
        anchor = rng.choice(frame_anchors.get(frame, anchors))
        fgens, clause_tpl, clause = frame_clause(
            frame, count, anchor, a.scene_bubble_tag
        )
        action = rng.choice(ACTIONS)
        if frame == "sign" and action.startswith("holding"):
            action = "standing"  # one held object per prompt
        generals = [
            *rng.sample(APPEARANCE, rng.choice([1, 2, 2])),
            *rng.choice(SETTINGS),
            action,
            rng.choice(EXPRESSIONS),
            *rng.choice(STYLES),
            *fgens,
            *extra,
        ]
        if solo:
            generals.append("solo")
        if rng.random() < 0.5:
            generals.append("looking at viewer")
        framing = rng.choice(FRAMING)
        if framing:
            generals.append(framing)
        tags = compose(head, generals)
        if tags in seen:
            continue
        seen.add(tags)
        i = len(items)
        items.append(
            {
                "i": i,
                "head": head,
                "generals": sorted(set(generals)),
                "tags": tags,
                "anchor": anchor,
                "frame": frame,
                "clause_tpl": clause_tpl,
                "prompt": f"{tags}. {clause}",
                "shape": list(pool.draw()),
                "seed": a.seed * 100_000 + i,
            }
        )
    return items


def stage_scenes(a):
    import torch

    # judge reads FRAMES from this module, so it imports here, not at the top
    from .judge import filter_scenes, report_scenes

    out = OUT / f"scenes_{a.scene_tag}"
    (out / "img").mkdir(parents=True, exist_ok=True)
    if a.scene_rejudge:
        # CPU only: re-apply the filter to an existing run from its stored
        # detector boxes + reads (new size bar / open-bubble rule / stray
        # rule) — no generation, no readers
        import os
        from multiprocessing import Pool

        items = [
            json.loads(ln)
            for ln in (out / "scenes_all.jsonl").read_text().splitlines()
            if ln
        ]
        # flood fills are CPU-bound: one worker per core
        with Pool(os.cpu_count()) as pool:
            items = pool.starmap(_rejudge_one, [(a, it) for it in items], chunksize=8)
        report_scenes(a, out, items)
        _prune(a, items)
        return
    if a.scene_n == 0:
        # maintenance on the stored rows only (no prompts, no model): the
        # --scene_prune sweep
        items = [
            json.loads(ln)
            for ln in (out / "scenes_all.jsonl").read_text().splitlines()
            if ln
        ]
        _prune(a, items)
        return
    items = scene_items(a)
    cs = Counter("x".join(map(str, it["shape"])) for it in items)
    print(
        f"scenes: {len(items)} prompts, shapes {dict(sorted(cs.items()))}, "
        f"anchors {Counter(it['anchor'] for it in items).most_common()}, "
        f"frames {Counter(it['frame'] for it in items).most_common()}",
        flush=True,
    )
    for it in items:
        it["file"] = str(out / "img" / f"scene_{it['i']:05d}.png")
    # Growing a pool: the prompt stream is deterministic in --seed, so the same
    # argv with a larger --scene_n reproduces the stored prompts and appends.
    # A stored row (same index, same prompt) is taken verbatim — its render is
    # neither regenerated nor re-read, and a pruned reject stays a reject.
    stored = {}
    if (out / "scenes_all.jsonl").exists():
        for ln in (out / "scenes_all.jsonl").read_text().splitlines():
            if ln:
                row = json.loads(ln)
                stored[row["i"]] = row
    todo = []
    for k, it in enumerate(items):
        row = stored.get(it["i"])
        if row and row.get("prompt") == it["prompt"] and "reason" in row:
            items[k] = row
        else:
            todo.append(it)
    if stored:
        print(
            f"scenes: {len(items) - len(todo)} stored rows kept, {len(todo)} new",
            flush=True,
        )
    if not todo:
        report_scenes(a, out, items)
        _prune(a, items)
        return
    # prompts first, so a killed run still has one row per image index
    (out / "prompts.jsonl").write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items)
    )
    # one loaded model; per-shape args (size lives in args only); batches of
    # --scene_batch same-shape prompts through text encoder → DiT → VAE at
    # once; --scene_gen_scale renders at k× the pool shape and downsamples
    # (the base's native resolution is ~1024, 512² alone draws crude scenes)
    args, gen, device, shared = load_generator(
        tuple(todo[0]["shape"]), a.steps, a.cfg, out / "img"
    )
    shared["model"].eval()
    vae = load_vae(device)
    groups: dict = {}
    for it in todo:
        if not Path(it["file"]).exists():
            groups.setdefault(tuple(it["shape"]), []).append(it)
    t0 = time.time()
    n = 0
    for shp, its in groups.items():
        gshape = tuple(int(round(v * a.scene_gen_scale / 16)) * 16 for v in shp)
        sargs = gen_args(gshape, a.steps, a.cfg, out / "img", a.scene_negative)
        for j in range(0, len(its), a.scene_batch):
            chunk = its[j : j + a.scene_batch]
            generate_batch(sargs, shared, vae, device, chunk, shp)
            shared["conds_cache"].clear()  # every prompt is unique; no reuse
            n += len(chunk)
            if n % 48 < len(chunk) or n == sum(len(v) for v in groups.values()):
                print(
                    f"scenes gen {n} new / {len(items)} {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    del shared, vae
    torch.cuda.empty_cache()
    filter_scenes(a, out, items, todo)
    _prune(a, items)


def _prune(a, items: list[dict]):
    if a.scene_prune:
        from .judge import prune_rejected

        print(f"scenes: pruned {prune_rejected(items)} rejected renders", flush=True)


def _rejudge_one(a, it: dict) -> dict:
    """``--scene_rejudge`` worker: one stored row through the current judge."""
    from .judge import judge

    if not Path(it["file"]).exists():
        return it  # rejected render pruned from disk: stored reason stands
    reads = [{"box": b, **r} for b, r in zip(it["boxes"], it["reads"])]
    for k in (
        "box",
        "region",
        "bubble",
        "boxes_anchor",
        "regions",
        "bubbles",
        "read",
        "residual",
        "open_uniform",
        "open_lost",
        "region_offset",
        "boxes_speck",
        "speck_regions",
        "speck_bubbles",
    ):
        it.pop(k, None)
    it["reason"] = judge(a, it, reads, load_bgr(Path(it["file"])))
    return it


def generate_batch(args, shared, vae, device, chunk: list[dict], out_shape):
    """One text-encoder → DiT → VAE pass for ``chunk`` (same canvas): each
    prompt encoded through the shared cache, embeds stacked on the batch
    axis, per-item seeds drawn as one noise tensor each (bit-identical to
    the single-image path's noise), decoded and resized to ``out_shape``."""
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from PIL import Image

    from library.inference.generation import anima_models, generate_body
    from library.inference.text import prepare_text_inputs

    anima = shared["model"]
    ctxs, null = [], None
    for it in chunk:
        c, null = prepare_text_inputs(args, device, anima, shared, prompt=it["prompt"])
        ctxs.append(c)
    context = {
        **ctxs[0],
        "embed": [torch.cat([c["embed"][0] for c in ctxs])],
        "prompt": [it["prompt"] for it in chunk],
    }
    H, W = args.image_size
    C = anima_models.Anima.LATENT_CHANNELS
    lats = []
    for it in chunk:
        g = torch.Generator(device="cpu").manual_seed(it["seed"])
        lats.append(
            randn_tensor(
                (1, C, 1, H // 8, W // 8),
                generator=g,
                device=device,
                dtype=torch.bfloat16,
            )
        )
    with torch.no_grad():
        out = generate_body(
            args,
            anima,
            context,
            null,
            device,
            [it["seed"] for it in chunk],
            latents=torch.cat(lats),
        )
    for it, lat in zip(chunk, out):
        im = decode_image(vae, lat.unsqueeze(0), device)
        if im.size != tuple(out_shape):
            im = im.resize(tuple(out_shape), Image.LANCZOS)
        im.save(it["file"])
