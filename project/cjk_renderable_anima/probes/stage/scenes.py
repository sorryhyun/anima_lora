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

from wake.common import OUT, norm, parse_shapes
from wake.models import decode_image, gen_args, load_generator, load_vae
from wake.readers import Readers, contact_sheet, load_bgr

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
    from wake.common import REPO

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
        generals = [
            *rng.sample(APPEARANCE, rng.choice([1, 2, 2])),
            *rng.choice(SETTINGS),
            rng.choice(ACTIONS),
            rng.choice(EXPRESSIONS),
            *rng.choice(STYLES),
            a.scene_bubble_tag,
            "english text",
        ]
        if count in ("1girl", "1boy"):
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
        anchor = rng.choice(anchors)
        i = len(items)
        items.append(
            {
                "i": i,
                "head": head,
                "generals": sorted(set(generals)),
                "tags": tags,
                "anchor": anchor,
                "prompt": TPL_SCENE.format(tags=tags, anchor=anchor),
                "shape": list(pool.draw()),
                "seed": a.seed * 100_000 + i,
            }
        )
    return items


def stage_scenes(a):
    import torch

    out = OUT / f"scenes_{a.scene_tag}"
    (out / "img").mkdir(parents=True, exist_ok=True)
    items = scene_items(a)
    cs = Counter("x".join(map(str, it["shape"])) for it in items)
    print(
        f"scenes: {len(items)} prompts, shapes {dict(sorted(cs.items()))}, "
        f"anchors {Counter(it['anchor'] for it in items).most_common()}",
        flush=True,
    )
    # one loaded model; per-shape args (size lives in args only); batches of
    # --scene_batch same-shape prompts through text encoder → DiT → VAE at
    # once; --scene_gen_scale renders at k× the pool shape and downsamples
    # (the base's native resolution is ~1024, 512² alone draws crude scenes)
    args, gen, device, shared = load_generator(
        tuple(items[0]["shape"]), a.steps, a.cfg, out / "img"
    )
    shared["model"].eval()
    vae = load_vae(device)
    for it in items:
        it["file"] = str(out / "img" / f"scene_{it['i']:05d}.png")
    groups: dict = {}
    for it in items:
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
    _filter_scenes(a, out, items)


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


def _filter_scenes(a, out: Path, items: list[dict]):
    """Detector + readers: keep exactly-one-box images whose box reads the
    anchor and is large enough; write both jsonl files, sheets, report."""
    rd = Readers(a.device)
    t0 = time.time()
    for n, it in enumerate(items, 1):
        bgr = load_bgr(Path(it["file"]))
        reads = rd.read_image(bgr, whole=False)
        it["boxes"] = [r["box"] for r in reads]
        it["reads"] = [{"sfx": r["sfx"], "vl": r["vl"]} for r in reads]
        it["reason"] = _judge(a, it, reads, bgr)
        if n % 100 == 0 or n == len(items):
            print(
                f"scenes read {n}/{len(items)} {(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
    del rd
    kept = [it for it in items if it["reason"] == "pass"]
    (out / "scenes_all.jsonl").write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items)
    )
    keys = (
        "i",
        "file",
        "head",
        "generals",
        "tags",
        "prompt",
        "anchor",
        "box",
        "region",
        "bubble",
        "boxes_anchor",
        "regions",
        "bubbles",
        "shape",
        "seed",
        "read",
    )
    (out / "scenes.jsonl").write_text(
        "\n".join(
            json.dumps({k: it[k] for k in keys}, ensure_ascii=False) for it in kept
        )
    )
    reasons = Counter(it["reason"] for it in items)
    n = len(items)
    short = [
        min(it["region"][2] - it["region"][0], it["region"][3] - it["region"][1])
        for it in kept
    ]
    lines = [
        f"# scenes `{a.scene_tag}` — {n} generated, **{len(kept)} kept ({len(kept) / max(1, n):.0%})**",
        "",
        f"prompt: `{TPL_SCENE.format(tags='<rating, count, character, copyright, @artist, generals sorted>', anchor='<anchor>')}`; "
        f"shapes `{a.scene_shapes}` × gen scale {a.scene_gen_scale}; batch {a.scene_batch}; "
        f"{a.steps} steps cfg {a.cfg}; negative `{a.scene_negative}`; min box {a.scene_min_box} px",
        "",
        "| reason | n | share |",
        "|---|---|---|",
    ]
    for k in ("pass", "no_box", "multi_box", "read_miss", "small_box", "open_bubble"):
        lines.append(
            f"| {k} | {reasons.get(k, 0)} | {reasons.get(k, 0) / max(1, n):.0%} |"
        )
    if kept:
        short.sort()
        lines += [
            "",
            f"kept usable-region short side (px): min {short[0]} p10 {short[len(short) // 10]} "
            f"median {short[len(short) // 2]} max {short[-1]}; "
            f"anchor bubbles per kept image {sum(len(it['regions']) for it in kept) / len(kept):.2f}",
            "",
            "kept per anchor: "
            + ", ".join(
                f"{k} {v}/{c}"
                for (k, v), c in zip(
                    sorted(Counter(it["anchor"] for it in kept).items()),
                    [
                        Counter(it["anchor"] for it in items)[k]
                        for k in sorted(Counter(it["anchor"] for it in kept))
                    ],
                )
            ),
            "kept per shape: "
            + ", ".join(
                f"{k} {v}"
                for k, v in sorted(
                    Counter("x".join(map(str, it["shape"])) for it in kept).items()
                )
            ),
        ]
    lines += [
        "",
        "Sheets: sheet_kept.png (text box lime, region cyan, bubble yellow), sheet_rejected.png (reason / reads).",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    rng = random.Random(0)
    _sheet(rng.sample(kept, min(40, len(kept))), out / "sheet_kept.png", True)
    rej = [it for it in items if it["reason"] != "pass"]
    _sheet(rng.sample(rej, min(40, len(rej))), out / "sheet_rejected.png", False)


def _sheet(its, path: Path, kept: bool):
    from PIL import Image, ImageDraw

    if not its:
        return
    rows = []
    for it in its:
        im = Image.open(it["file"]).convert("RGB")
        d = ImageDraw.Draw(im)
        for b in it["boxes"]:
            d.rectangle(b, outline="red" if not kept else "lime", width=4)
        if kept:
            for bub, reg in zip(it["bubbles"], it["regions"]):
                if bub:
                    d.rectangle(bub, outline="yellow", width=3)
                d.rectangle(reg, outline="cyan", width=3)
        r0 = it["reads"][0] if it["reads"] else {"sfx": "", "vl": ""}
        rows.append(
            (
                im,
                [
                    f"{it['i']:05d} {it['anchor']} {it['reason']}",
                    f"vl {r0['vl'] or ''} / sfx {r0['sfx'] or ''}",
                    it["tags"][:40],
                    it["tags"][40:80],
                ],
            )
        )
    contact_sheet(rows, path, thumb=192, cols=8)


# ----------------------------------------------------------------------------
# judging one image


def _overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _judge(a, it: dict, reads: list, bgr) -> str:
    """Set ``boxes_anchor`` / ``regions`` / ``bubbles`` (one per anchor
    bubble; ``box`` / ``region`` / ``bubble`` = the largest) and ``read`` on
    ``it``; return the reason. The detector boxes the *text*, not the bubble.
    Every box that reads the anchor is an anchor bubble (the base draws one
    per speaker; the data stage swaps all of them); boxes overlapping an
    anchor box are merged into it; any other box is stray text and rejects
    the image unless it is a detector speck (under a quarter of the anchor
    box's area). Each anchor bubble must be closed (flood fill from a ring
    outside the text box stays off the border) and its inscribed region at
    least ``--scene_min_box`` on the short side."""
    if not reads:
        return "no_box"
    anchor = norm(it["anchor"])
    hits = [r for r in reads if anchor in {norm(r["vl"] or ""), norm(r["sfx"] or "")}]
    if not hits:
        return "read_miss"
    boxes = []
    for r in hits:
        box = list(r["box"])
        for o in reads:
            if o is not r and o not in hits and _overlap(box, o["box"]):
                box = [
                    min(box[0], o["box"][0]),
                    min(box[1], o["box"][1]),
                    max(box[2], o["box"][2]),
                    max(box[3], o["box"][3]),
                ]
        boxes.append(box)
    area = max(1, max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes))
    stray = [
        o
        for o in reads
        if o not in hits
        and not any(_overlap(b, o["box"]) for b in boxes)
        and (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]) >= 0.25 * area
    ]
    if stray:
        return "multi_box"
    r = hits[0]
    it["read"] = r["vl"] if norm(r["vl"] or "") == anchor else r["sfx"]
    it["boxes_anchor"], it["bubbles"], it["regions"] = boxes, [], []
    for b in boxes:
        bubble, region = bubble_region(bgr, b)
        it["bubbles"].append(bubble)
        it["regions"].append(region)
    # the largest bubble is the headline record
    k = max(
        range(len(boxes)),
        key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
    )
    it["box"], it["bubble"], it["region"] = boxes[k], it["bubbles"][k], it["regions"][k]
    if any(b is None for b in it["bubbles"]) and not a.scene_allow_open:
        return "open_bubble"
    if any(min(g[2] - g[0], g[3] - g[1]) < a.scene_min_box for g in it["regions"]):
        return "small_box"
    return "pass"


def bubble_region(bgr, box, pad: int = 5, tol: int = 24):
    """``(bubble bbox or None, usable region)``. Flood-fills from 8 seeds on
    a ring ``pad`` px outside the text box (fixed range ± ``tol`` per
    channel from each seed's colour), unions the fills, and takes the
    rectangle inscribed in the union's bbox (0.72 × — an ellipse's inscribed
    rectangle is 1/√2 of its axes). A union touching the image border or
    covering > 35 % of the image is an open background (``None``); the
    region is then the text box grown 1.8× and clamped."""
    import cv2
    import numpy as np

    H, W = bgr.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    seeds = []
    for sx in (x0 - pad, cx, x1 + pad):
        for sy in (y0 - pad, cy, y1 + pad):
            if (sx, sy) == (cx, cy):
                continue
            seeds.append((int(min(W - 1, max(0, sx))), int(min(H - 1, max(0, sy)))))
    union = np.zeros((H, W), dtype=np.uint8)
    for sx, sy in seeds:
        mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
        cv2.floodFill(
            bgr.copy(),
            mask,
            (sx, sy),
            0,
            (tol, tol, tol),
            (tol, tol, tol),
            cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8) | 4,
        )
        m = mask[1:-1, 1:-1]
        if m.sum() / 255 > 0.35 * H * W:
            continue
        union |= m
    ys, xs = np.nonzero(union)
    open_bg = (
        len(xs) == 0
        or len(xs) > 0.35 * H * W
        or xs.min() == 0
        or ys.min() == 0
        or xs.max() == W - 1
        or ys.max() == H - 1
    )
    if open_bg:
        w, h = (x1 - x0) * 1.8, (y1 - y0) * 1.8
        region = [
            int(max(0, cx - w / 2)),
            int(max(0, cy - h / 2)),
            int(min(W, cx + w / 2)),
            int(min(H, cy + h / 2)),
        ]
        return None, region
    bx0, by0 = int(xs.min()), int(ys.min())
    bx1, by1 = int(xs.max()) + 1, int(ys.max()) + 1
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    bw, bh = (bx1 - bx0) * 0.72, (by1 - by0) * 0.72
    region = [
        int(max(0, bcx - bw / 2)),
        int(max(0, bcy - bh / 2)),
        int(min(W, bcx + bw / 2)),
        int(min(H, bcy + bh / 2)),
    ]
    # the region must still hold the text box (else the fill was a
    # neighbouring patch, not the bubble): grow to cover it
    region = [
        min(region[0], x0),
        min(region[1], y0),
        max(region[2], x1),
        max(region[3], y1),
    ]
    return [bx0, by0, bx1, by1], region
