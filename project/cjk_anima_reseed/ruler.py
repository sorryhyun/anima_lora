"""The dialogue ruler (``criteria.md``): real JA dialogue in a speech bubble,
read per render against a floor rendered once.

``build`` (CPU) draws the string set from the training set's own bubble
dialogue — captions tagged ``speech bubble`` / ``comic`` / ``dialogue``,
their ``Japanese text reads as`` clause (not SFX) — up to two strings an image,
``PER_BIN`` per length bin, and gives each string its own image's caption
with the text clauses out and the other-language text tags out as its
prompt (user, 10-05). The short bin is half interjections, half lexical.
Every string carries ``cov3``: the share of its trigrams in the text the
arms of record trained on (the reseed windows came from these same lines),
so a read reports all strings and the low-coverage ones apart.

The EN reference (user, 10-05): each line's EN rendering, hand-written in
``output/cjk_anima_reseed/ruler/en.json`` (JA text → EN), under the same
prompt with ``english text`` for ``japanese text`` and ``English text reads
as`` — the base drawing this line's bubble in EN, the page Axis 2 reads
against.

    .venv/bin/python project/cjk_anima_reseed/ruler.py build
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/ruler.py run"
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/ruler.py run --arms kana_up,ball_rk_bubble,stick_nlg_high --label arms"

Writes ``output/cjk_anima_reseed/ruler/ruler.json`` (+ ``ruler.tsv`` to read
by eye). The dataset's text stays out of the repo.

``render`` (GPU) draws each string at its source image's aspect (``AREA``
pixels), seed 0, 28 steps, cfg 4, routed, one arm at a time through one
``ExtDelta`` over the union of the arms' ext ids; a render on disk is not
drawn again, so the floor (``FLOOR``: the EN refs, retrain_kana,
seed_retrain_0930) is rendered once and every later arm reads against it.
``read`` scores every render (readers cached per arm) — text: official
(both readers exact), exact (either), contained, ≤ 1 / ≤ 2 edits, CER, dup
(a doubled glyph the string lacks, or a read longer than it), a bubble's
boxes also joined in column and line order; glyphs (``score_page``):
``g_p`` / ``g_r`` / ``g_f1`` and the rest; page: ``en_cls`` / ``en_match``
(``AtSim``), EN-ref token cos outside every text box of both images
(``en_tok_out``), flat-white share over the EN ref's — and pairs each arm
against the floor arms per bin and on the unseen strings →
``results/<ts>-ruler-<label>/`` (+ sheets). ``run`` = both. ``sample``
draws random strings of a read into larger sheets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reseed import OUT, REPO, bootstrap  # noqa: E402

RULER = OUT / "ruler"
CAPTIONS = REPO / "post_image_dataset" / "resized"
SEED = 1005
PER_BIN = 32
PER_IMAGE = 2
BINS = {"short": (2, 4), "mid": (5, 9), "long": (10, 20)}
BUBBLE_TAGS = ("speech bubble", "comic", "dialogue")
# render size: the source image's aspect at this many pixels — manga-size
# dialogue needs more than the banner grid's 512²
AREA = 768 * 768
# the text the arms of record trained on — cov3 is read against their union
TRAINED = (
    "output/cjk_anima_reseed/kana_up/data",  # kana_up, stick_nlg_high (stick_from)
    "output/cjk_anima_reseed/ball_rk_bubble/data",
    "output/cjk_anima_scale/retrain_kana/data",
)
# tags naming text the ruler's clause does not carry
DROP_TAGS = {
    "sound effects",
    "translated",
    "translation request",
    "commentary request",
    "signature",
    "watermark",
    "twitter username",
    "artist name",
    "web address",
    "patreon username",
    "copyright notice",
}
# the `sensitive` prompts (user, 10-05: "explicit 태그만 빼고", sensitive allowed):
# the rating to sensitive, the sex-act subtree and these out; underwear,
# swimwear, cleavage and the rest stay
SEXUAL_PATH = "행동 > 성적행위"
EXPLICIT = re.compile(
    r"nipple|areola|penis|pussy|vulva|vagin|clitoris|\banus\b|\banal\b|pubic|"
    r"testic|\bcum\b|semen|erect|nud|naked|bottomless|topless|breasts? out|"
    r"censor|mosaic|condom|\bsex\b|orgasm|ahegao|cameltoe|bulge|uterus|"
    r"cross-section|x-ray|\bloli\b|shota|incest|netorare|ugly bastard|aftersex|lube|"
    r"dildo|vibrator|maebari|crotch|cupless|used tissue|casting couch|"
    r"\bass focus\b|upskirt|upshorts|hetero|threesome|bdsm|slave|mind control|"
    r"penis awe|\bm legs\b|proving gender|cleft of venus|gluteal fold|imminent|"
    r"stained panties"
)
# a source image tagged as a child (or a child in sex) leaves the pool whole:
# stripping its explicit tags would still prompt that scene (10-05)
CHILD = re.compile(
    r"\b(loli|shota|onee-shota|straight shota|kodomo doushi|child|aged down|"
    r"toddler|randoseru|elementary)\b"
)
RATINGS = {"explicit", "nsfw", "questionable", "sensitive", "general", "safe"}
# the prompt sets; renders under RULER / mode. `captioned` (the image's own
# rating and tags) was the 10-05 first floor / arms read, on a string set
# that still held the child-tagged images — its renders under RULER/<arm> no
# longer match ruler.json's indices and are kept as that read's record only
MODES = ("sensitive",)
PUNCT = set("…‥、。，．・！？!?～〜~♡♥❤☆★♪「」『』（）()・ 　.,")
SOKUON = set("っッー")
# an interjection is a string whose glyphs are all of these (punct and っ / ー aside)
INTERJ = set("あぁいぃうぅえぇおぉんはひふへほアァイィウゥエェオォンハヒフヘホ")
# read by eye (10-05): OCR misreads (出未 = 出来, 働考 = 参考, ヒヶ, ヌロ, おばん),
# SFX filed as dialogue, and fragments that are not a word
DROP = {
    "さくらぎまの",
    "イクぅヒヶ",
    "ヘロヌロヘロヌロ",
    "おっぱりすっげぇ…ドブドブ",
    "試乗車の準備が出未ました",
    "コノハちゃん、いつもありがとねー働考になるよー",
    "アギちゃんのおばんっ!ザーメンかけたい…",
    "ぶぽっぶぶぶぎゅっ",
    "ーー下、はいてるかって?",
    "おぶぅ",
    "ちゃ",
    "ひく",
    "わろ",
    "イラ",
    "うづ…",
    "ぬぎ",
    "ぬぷ…",
    "ませんが",
    "とーっー",
    "ファサァ…",
    "ビュクビュクッ",
    "どんどん挿へってろよ",
    "つ撮影ターゲットを発見",
    "つー…",
    "ヒョコ",
    "ビュッビュルルッ",
    "ハ奈見さん!?",
    "結ばれた二人は甘いロゾけを交わすんです…",
    "ほんとに毎日もねを使うとは思わなかった…",
    "…私…その…ざ…",
    "フリ",
    "ざちゃしむちゃし",
    "なで",
    "ほか",
    "出さ",
    "すぎるせいで…",
    "うず…",
    "くちょ…",
    "フゴッ",
    "くっさスゥ…",
    "アキ…ミリア助ナ…",
    "ぬぽん",
}


def glyphs(s: str) -> list:
    return [c for c in s if c not in PUNCT]


def length(s: str) -> int:
    return len(glyphs(s))


def is_interjection(s: str) -> bool:
    core = [c for c in glyphs(s) if c not in SOKUON]
    return bool(core) and all(c in INTERJ for c in core)


def is_kanji(c: str) -> bool:
    return "一" <= c <= "鿿"


def _key(*parts: str) -> str:
    return hashlib.sha1("\t".join((str(SEED),) + parts).encode()).hexdigest()


def trigram_set(texts) -> set:
    out = set()
    for t in texts:
        t = t.replace(" ", "")
        out.update(t[i : i + 3] for i in range(len(t) - 2))
    return out


def cov3(s: str, tri: set) -> float | None:
    g = [s[i : i + 3] for i in range(len(s) - 2)]
    return round(sum(x in tri for x in g) / len(g), 3) if g else None


def trained_texts() -> set:
    out = set()
    for d in TRAINED:
        p = REPO / d / "train.jsonl"
        assert p.is_file(), f"{p}: an arm of record's data is gone"
        for ln in p.open(encoding="utf-8"):
            out.add(json.loads(ln)["text"])
    return out


def explicit(t: str, kb) -> bool:
    info = kb.describe(t)
    if info is not None and (info.category_path or "").startswith(SEXUAL_PATH):
        return True
    return bool(EXPLICIT.search(t))


def prompt_of(parsed, kb=None) -> str:
    """The image's caption with its text clauses out, the other-language text
    tags out, ``speech bubble`` / ``japanese text`` in — the clause is added
    by the caller. With ``kb`` (the tag taxonomy): the ``sensitive`` prompt —
    the rating ``sensitive``, every explicit tag out (``explicit``), in the
    tag bag and the position clauses alike."""
    from anime_tools.captions.position_clauses import (
        TEXT_PREFIXES,
        PositionClause,
        compose_caption,
    )

    tags = [
        t
        for t in parsed.flat_tags
        if t not in DROP_TAGS and not (t.endswith(" text") and t != "japanese text")
    ]
    pos = [cl for cl in parsed.clauses if cl.prefix not in TEXT_PREFIXES]
    if kb is not None:
        tags = ["sensitive" if t in RATINGS else t for t in tags if not explicit(t, kb)]
        pos = [
            PositionClause(position=cl.position, tags=kept, prefix=cl.prefix)
            for cl in pos
            if (kept := tuple(t for t in cl.tags if not explicit(t, kb)))
        ]
    for t in ("speech bubble", "japanese text"):
        if t not in tags:
            tags.append(t)
    return compose_caption(tags, pos)


def pool() -> tuple[dict, Counter]:
    """stem → (prompt, [its JA dialogue strings], caption file); and every JA glyph's count
    over the whole dataset's text clauses (a kanji seen once is OCR noise)."""
    from anime_tools.captions.position_clauses import TEXT_PREFIXES, parse_caption
    from cjk_scale.config import is_ja_text

    text_prefix = TEXT_PREFIXES[0]  # `Japanese text reads as`, not SFX
    assert "SFX" not in text_prefix, TEXT_PREFIXES
    out, freq = {}, Counter()
    for f in sorted(CAPTIONS.rglob("*.txt")):
        if f.name.endswith(".variants.txt"):
            continue
        cap = f.read_text(encoding="utf-8")
        p = parse_caption(cap)
        lines = []
        for cl in p.clauses:
            if cl.prefix not in TEXT_PREFIXES:
                continue
            for t in cl.tags:
                s = t.strip().rstrip(".").strip('"').strip()
                if s and is_ja_text(s):
                    freq.update(glyphs(s))
                    if cl.prefix == text_prefix:
                        lines.append(s)
        if CHILD.search(", ".join(p.flat_tags)):
            continue
        if lines and any(k in p.flat_tags for k in BUBBLE_TAGS):
            out[f.stem] = (prompt_of(p), lines, f, p)
    return out, freq


def shape_of(caption_file: Path) -> list:
    """``[W, H]`` to render at: the source image's aspect (clamped to 1:2 –
    2:1) at ``AREA`` pixels, both sides multiples of 32."""
    from PIL import Image

    src = next(
        f
        for f in sorted(caption_file.parent.glob(caption_file.stem + ".*"))
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
    )
    w, h = Image.open(src).size
    ar = min(2.0, max(0.5, w / h))
    H = round((AREA / ar) ** 0.5 / 32) * 32
    W = round(AREA / H / 32) * 32
    return [W, H]


def core(s: str) -> str:
    """The string less punctuation and trailing っ / ー: はぁっ… and はぁ are one."""
    return "".join(glyphs(s)).rstrip("".join(SOKUON))


def clean(s: str, freq: Counter) -> bool:
    """Not in ``DROP``; no Latin, no digit, no kanji the dataset has only once
    (OCR noise)."""
    if s in DROP:
        return False
    if re.search(r"[A-Za-z0-9Ａ-Ｚａ-ｚ０-９]", s):
        return False
    return all(freq[c] > 1 for c in glyphs(s) if is_kanji(c))


def build() -> dict:
    bootstrap()
    from anime_tools.captions.position_clauses import compose_caption, text_clause

    from anime_tools.captions.correction import find_tag_csv, load_tag_knowledge_base

    kb = load_tag_knowledge_base(find_tag_csv(REPO))
    imgs, freq = pool()
    en_file = RULER / "en.json"
    en = json.loads(en_file.read_text(encoding="utf-8")) if en_file.is_file() else {}
    tri = trigram_set(trained_texts())
    # bin → kind → [(stem, string)]
    cands: dict = defaultdict(lambda: defaultdict(list))
    for stem, (_, lines, *_) in imgs.items():
        for s in dict.fromkeys(lines):
            n = length(s)
            b = next((k for k, (lo, hi) in BINS.items() if lo <= n <= hi), None)
            if b is None or not clean(s, freq):
                continue
            kind = (
                ("interj" if is_interjection(s) else "lexical")
                if b == "short"
                else "all"
            )
            cands[b][kind].append((stem, s))
    quota = {
        ("short", "interj"): PER_BIN // 2,
        ("short", "lexical"): PER_BIN // 2,
        ("mid", "all"): PER_BIN,
        ("long", "all"): PER_BIN,
    }
    # the quota's slots matched to images (augmenting paths — a greedy pass by
    # pool leaves the long bin short), up to PER_IMAGE strings an image: every
    # image's first copy is offered before any second (116 images after the
    # child-tagged ones left, short 82 / mid 52 / long 61 — 96 do not fit one
    # each); a second copy takes a pool only where the image holds two of its
    # strings. A core (はぁっ… = はぁ) stays at one image, so no two slots draw it
    by_img: dict = defaultdict(lambda: defaultdict(list))  # stem → pool → strings
    # every order keyed per string, so a string dropped moves only its own slot
    flat = sorted(
        ((pk, stem, s) for pk in quota for stem, s in cands[pk[0]][pk[1]]),
        key=lambda x: _key(x[1], x[2]),
    )
    core_at: dict = {}
    for pk, stem, s in flat:
        if core_at.setdefault(core(s), stem) == stem:
            by_img[stem][pk].append(s)
    slots = [pk for pk, n in quota.items() for _ in range(n)]
    nodes = [
        (st, k)
        for k in range(PER_IMAGE)
        for st in sorted(by_img, key=lambda st: _key(st))
    ]
    owner: dict = {}  # (stem, copy) → slot index

    def augment(j: int, seen: set) -> bool:
        for nd in nodes:
            st, k = nd
            if len(by_img[st].get(slots[j], ())) <= k or nd in seen:
                continue
            seen.add(nd)
            if nd not in owner or augment(owner[nd], seen):
                owner[nd] = j
                return True
        return False

    for j in range(len(slots)):
        assert augment(j, set()), f"slot {slots[j]}: no image left"
    used_str, items = set(), []
    for nd in sorted(owner, key=lambda nd: owner[nd]):
        st = nd[0]
        b, kind = pk = slots[owner[nd]]
        xs = [s for s in by_img[st][pk] if core(s) not in used_str]
        assert xs, f"{st} {pk}: every string's core already drawn"
        s = min(xs, key=lambda x: _key(st, x))
        used_str.add(core(s))
        prompts = {"captioned": imgs[st][0], "sensitive": prompt_of(imgs[st][3], kb)}
        items.append(
            {
                "bin": b,
                "kind": kind,
                "text": s,
                "len": length(s),
                "stem": st,
                "shape": shape_of(imgs[st][2]),
                "cov3": cov3(s, tri),
                "en": en.get(s),
                "prompts": {
                    mode: {
                        "prompt": pr,
                        "caption": compose_caption(
                            [pr.rstrip(".")], [text_clause([s])]
                        ),
                        "en_caption": re.sub(
                            r"\bjapanese text\b", "english text", pr
                        ).rstrip(".")
                        + f'. English text reads as "{en.get(s)}".',
                    }
                    for mode, pr in prompts.items()
                },
            }
        )
    missing = [m["text"] for m in items if m["en"] is None]
    order = list(BINS)
    items.sort(key=lambda m: (order.index(m["bin"]), m["kind"], m["len"], m["text"]))
    for i, m in enumerate(items):
        m["i"] = i
    stats = {
        "images": len(imgs),
        "candidates": {b: {k: len(v) for k, v in d.items()} for b, d in cands.items()},
        "items": len(items),
        "en_missing": len(missing),
        "cov3_median": {
            b: sorted(m["cov3"] or 0 for m in items if m["bin"] == b)[PER_BIN // 2]
            for b in BINS
        },
    }
    RULER.mkdir(parents=True, exist_ok=True)
    (RULER / "ruler.json").write_text(
        json.dumps(
            {"seed": SEED, "bins": BINS, "stats": stats, "items": items},
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    with (RULER / "ruler.tsv").open("w", encoding="utf-8") as f:
        f.write("i\tbin\tkind\tlen\tcov3\ttext\tstem\n")
        for m in items:
            f.write(
                f"{m['i']}\t{m['bin']}\t{m['kind']}\t{m['len']}\t{m['cov3']}\t{m['text']}\t{m['stem']}\n"
            )
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    if missing:
        print(
            f"{len(missing)} strings without an EN line in {en_file}:",
            *missing,
            sep="\n  ",
        )
    return stats


# ---------------------------------------------------------------- render


def arm_dirs() -> dict:
    """The arms the ruler renders: name → run dir with a finished ``trained.pt``."""
    from cjk_scale.paths import OUT as SCALE_OUT

    dirs = {
        "retrain_kana": SCALE_OUT / "retrain_kana",
        "seed_retrain_0930": SCALE_OUT / "seed_retrain_0930",
        "kana_up": OUT / "kana_up",
        "ball_rk_bubble": OUT / "ball_rk_bubble",
        "stick_nlg_high": OUT / "stick_nlg_high",
        # preview51's rows (_archive/sent_plan.md's floor), read on the punct pack
        "seed_fixed_1005_stick080": SCALE_OUT / "seed_fixed_1005_stick080",
        **PACK_ARMS.get(PACK, {}),
    }
    return {a: d for a, d in dirs.items() if (d / "trained.pt").exists()}


FLOOR = ("en", "retrain_kana", "seed_retrain_0930")  # rendered once (criteria.md)
# ``--pack``: the base pack a render sits on (``reseed.config.PACKS``), its own
# arms (a run with ``pack``); ``X@<pack>`` = arm X's rows on that pack (its
# routing), rendered beside X's on the raw pack
PACK = ""
PACK_ARMS = {
    "punct": {
        "punct": OUT / "punct",
        "sent_ball": OUT / "sent_ball",
        "sent_ball_lr2": OUT / "sent_ball_lr2",
        "sent_whole": OUT / "sent_whole",
        "sent_stick": OUT / "sent_stick",
        "sent_kanji": OUT / "sent_kanji",
        "sent_kanji_f0": OUT / "sent_kanji_f0",
        "sent_kanji_pres": OUT / "sent_kanji_pres",
        "seed_1008": OUT / "seed_1008",  # pres + sent_kanji_225's 225 (transplant.py)
        "kozh16": OUT / "kozh16",  # 8 Hangul + 8 hanzi cold on seed_1008 (probes/kozh_render.py)
    }
}


def base_arm(a: str) -> str:
    return a.split("@")[0]


EN = "en"  # the EN references: no ext row in the caption, the delta off
STEPS, CFG, SEED_RENDER = 28, 4.0, 0  # the reads of record's sampler


# ``--marks``: only the strings holding a punct run's mark row (``〜`` / ``～`` /
# ``~``, a dot run, ``♡♥``, ``、。，``) — the rest encode as on the floor
MARKS_ONLY = False
ONLY: set = set()  # ``--only``: these ruler indices only (a look, not a read)
MARK_CHARS = set("～〜~…‥♡♥、。，")


def has_mark(t: str) -> bool:
    return any(c in MARK_CHARS for c in t) or "・・" in t or ".." in t


def items() -> list:
    its = json.loads((RULER / "ruler.json").read_text(encoding="utf-8"))["items"]
    if ONLY:
        its = [m for m in its if m["i"] in ONLY]
    return [m for m in its if has_mark(m["text"])] if MARKS_ONLY else its


MODE = MODES[0]  # the prompt set rendered / read (``--prompts``)


def mode_dir() -> Path:
    return RULER / MODE


def render_file(arm: str, i: int) -> Path:
    # the EN refs keep eval.enref's naming (pi = the ruler index)
    if arm == EN:
        return mode_dir() / EN / f"enref_p{i:02d}_s{SEED_RENDER}.png"
    return mode_dir() / arm / f"r{i:02d}_s{SEED_RENDER}.png"


def rows_pt(path: Path) -> dict:
    """A probe's ``rows.pt`` (``_archive/probes/probe_pres_train.py``: the live rows'
    ``start`` / ``raw`` at its ``row_scale``) on preview51's rows
    (``seed_fixed_1005_stick080``, the probe's start): those rows replaced, the
    rest as stick080 has them."""
    import torch
    from cjk_scale.paths import OUT as SCALE_OUT
    from common.models import load_trained

    base = load_trained(SCALE_OUT / "seed_fixed_1005_stick080")["delta"]
    R = torch.load(path, map_location="cpu", weights_only=False)
    scale, rs = float(base["row_scale"]), float(R["row_scale"])
    at = {int(e): i for i, e in enumerate(base["ext_ids"])}
    raw = base["raw"].float().clone()
    for j, e in enumerate(R["ext_ids"]):
        s = R["start"][j].float() * rs
        assert torch.allclose(raw[at[e]] * scale, s, atol=1e-3 * float(s.norm())), (
            f"{path}: row {e}'s start is not stick080's"
        )
        raw[at[e]] = R["raw"][j].float() * rs / scale
    return {"ext_ids": base["ext_ids"], "raw": raw, "row_scale": scale}


# arms built from other runs' rows, not trained: name → its delta state
# (``--rows_pt name=path`` adds a probe's rows.pt)
DERIVED: dict = {}


def tables(names) -> tuple[list, dict]:
    """The union of the arms' ext ids, and each arm's table over it in
    effective units (``raw × row_scale``; a row an arm lacks is 0 = the pack
    row, what that arm renders it as)."""
    import torch

    from common.models import load_trained

    deltas = {a: load_trained(d)["delta"] for a, d in arm_dirs().items()}
    for a in map(base_arm, names):
        if a in DERIVED:
            deltas[a] = DERIVED[a]()
    ids = sorted({int(e) for d in deltas.values() for e in d["ext_ids"]})
    pos = {e: i for i, e in enumerate(ids)}
    out = {}
    for a in names:
        d = deltas[base_arm(a)]
        t = torch.zeros(len(ids), d["raw"].shape[1])
        t[[pos[int(e)] for e in d["ext_ids"]]] = d["raw"].float() * float(
            d["row_scale"]
        )
        out[a] = t
    return ids, out


class Renderer:
    def __init__(self):
        import torch

        from common.hooks import ExtDelta
        from common.models import load_generator, load_vae

        dyn = torch._dynamo.config  # 13 ruler shapes + the check's, one graph each
        for k in ("cache_size_limit", "recompile_limit"):
            if hasattr(dyn, k):
                setattr(dyn, k, 64)
        self.torch = torch
        self.args, self.gen, self.device, self.shared = load_generator(
            512, STEPS, CFG, RULER / "_tmp"
        )
        self.anima = self.shared["model"]
        self.anima.eval()
        self.vae = load_vae(self.device)
        self.ids, t = tables(["retrain_kana"])
        dim = t["retrain_kana"].shape[1]
        self.delta = ExtDelta(self.anima, self.ids, dim, self.device, row_scale=1.0)

    def set_arm(self, table) -> None:
        self.shared["conds_cache"].clear()
        if table is None:
            self.delta.scale = 0.0
            return
        self.delta.scale = 1.0
        self.delta.raw.data.copy_(table.to(self.delta.raw.device))

    def render(self, fn: Path, caption: str, seed: int, wh) -> None:
        import copy

        from common.models import decode_image
        from library.inference.generation import generate

        if fn.exists():
            return
        a2 = copy.deepcopy(self.args)
        a2.prompt, a2.seed = caption, seed
        a2.image_size = (wh[1], wh[0])
        with self.torch.no_grad():
            lat = generate(a2, self.gen, self.shared)
        fn.parent.mkdir(parents=True, exist_ok=True)
        decode_image(self.vae, lat, self.device).save(fn)


def check(r: Renderer, table) -> float:
    """retrain_kana's first cached plain render, re-rendered on this path."""
    import numpy as np
    from PIL import Image

    m = json.loads(
        (
            arm_dirs()["retrain_kana"] / "native_r4_plain" / "native_reads.json"
        ).read_text("utf-8")
    )[0]
    ref = Image.open(m["file"])
    fn = RULER / "_tmp" / "check_rk.png"
    fn.unlink(missing_ok=True)
    r.set_arm(table)
    r.render(fn, m["caption"], m["seed"], ref.size)
    d = float(
        np.abs(
            np.asarray(Image.open(fn), np.float32)
            - np.asarray(ref.convert("RGB"), np.float32)
        ).mean()
    )
    print(f"check retrain_kana {Path(m['file']).name}: mean |Δpx| {d:.3f}", flush=True)
    assert d < 12.0, "the ruler path does not reproduce retrain_kana's render"
    return d


def render(names: list) -> None:
    import time

    its = sorted(items(), key=lambda m: (m["shape"], m["i"]))  # one compile per shape
    _, tabs = tables([a for a in names if a != EN] + ["retrain_kana"])
    r = Renderer()
    info = {"check_rk": check(r, tabs["retrain_kana"])}
    t0 = time.time()
    for a in names:
        todo = [m for m in its if not render_file(a, m["i"]).exists()]
        r.set_arm(None if a == EN else tabs[a])
        for n, m in enumerate(todo):
            pr = m["prompts"][MODE]
            cap = pr["en_caption"] if a == EN else pr["caption"]
            r.render(render_file(a, m["i"]), cap, SEED_RENDER, m["shape"])
            print(
                f"  {a}: {n + 1} / {len(todo)} r{m['i']:02d} {m['shape']} "
                f"({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
        info[a] = {"rendered": len(todo)}
    mode_dir().mkdir(parents=True, exist_ok=True)
    (mode_dir() / "render_log.json").write_text(json.dumps(info, indent=1))
    print(f"render: {info} in {(time.time() - t0) / 60:.1f} min", flush=True)


# ------------------------------------------------------------------ read

UNSEEN = 0.15  # cov3 at or below: the string's trigrams the arms barely trained on
BOOL = ("official", "exact", "contained", "le1", "le2", "dup")
# the page scores (user, 10-06: what is drawn, not only whether the string
# reads whole) — ``score_page``; a real is ``None`` where it has no meaning
# (kanji recall on a kana-only string) and is left out of a mean / a pair
PAGE = (
    "g_p",
    "g_r",
    "g_f1",
    "g_r_kanji",
    "g_r_kana",
    "drawn",
    "a_p",
    "text_area",
    "iou_en",
)
REAL = ("cer", *PAGE, "en_cls", "en_match", "en_tok_out", "fw_over_en")


def candidates(reads: list, reader: str) -> list:
    """One reader's reads: every box (the whole page too), and the boxes
    joined in column order (right to left) and in line order (top to bottom)
    — a bubble set in two columns is one string (user, 10-05)."""
    from common.text import norm

    boxes = [r for r in reads if not r.get("whole")]
    out = [norm(r.get(reader) or "") for r in reads]
    if len(boxes) > 1:
        cx = lambda r: (r["box"][0] + r["box"][2]) / 2  # noqa: E731
        for key in (
            lambda r: (-cx(r), r["box"][1]),
            lambda r: (r["box"][1], r["box"][0]),
        ):
            out.append(
                "".join(norm(r.get(reader) or "") for r in sorted(boxes, key=key))
            )
    return [c for c in out if c]


def score_text(text: str, reads: list) -> dict:
    from common.text import lev, norm

    t = norm(text)
    by = {x: candidates(reads, x) for x in ("sfx", "vl")}
    allc = by["sfx"] + by["vl"]
    best = min(allc, key=lambda c: (lev(c, t), len(c)), default="")
    d = lev(best, t) if best else len(t)
    doubled = lambda s: {s[i] for i in range(len(s) - 1) if s[i] == s[i + 1]}  # noqa: E731
    return {
        "best": best,
        "official": t in by["sfx"] and t in by["vl"],
        "exact": t in allc,
        "contained": any(t in c for c in allc),
        "le1": len(t) > 2 and d <= 1,
        "le2": len(t) > 2 and d <= 2,
        "cer": min(1.0, d / max(1, len(t))),
        "dup": bool(best) and (bool(doubled(best) - doubled(t)) or len(best) > len(t)),
    }


LETTER_RE = re.compile(r"[ぁ-ゟァ-ヺー一-鿿]")  # kana, ー, kanji (NFKC)
KANJI = re.compile(r"[一-鿿]")
READERS = ("sfx", "vl")
ON_SHARE = 0.5  # a box is on the target: this share of its letters are the target's


def letters(s: str | None) -> list:
    import unicodedata

    return [c for c in unicodedata.normalize("NFKC", s or "") if LETTER_RE.match(c)]


def box_mask(boxes: list, hw: tuple):
    import numpy as np

    H, W = hw
    m = np.zeros((H, W), dtype=bool)
    for b in boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        m[max(0, y0) : min(H, y1), max(0, x0) : min(W, x1)] = True
    return m


def score_page(text: str, reads: list, en_reads: list) -> dict:
    """What the page draws against the string (user, 10-06), every text box
    read, each reader on its own and the two averaged:

    - glyphs (kana, ー, kanji; a bag — order and box free): ``g_p`` = the
      string's letters among all letters drawn (low: much text that is not
      the string), ``g_r`` = the string's letters drawn, ``g_f1``;
      ``g_r_kanji`` / ``g_r_kana`` the recall over its kanji / kana only;
      ``drawn`` = letters drawn;
    - regions: a box is on the string when ``ON_SHARE`` of its letters are
      the string's (and it holds two of them, one for a one-letter string);
      ``a_p`` = the on boxes' area over all text area, ``text_area`` = text
      area over the page, ``iou_en`` = the text area's IoU with the EN ref's
      (its layout — the EN page letters its other bubbles too, so not the
      string's place)."""
    from collections import Counter

    whole = next(r for r in reads if r.get("whole"))
    hw = (int(whole["box"][3]), int(whole["box"][2]))
    boxes = [r for r in reads if not r.get("whole")]
    G = Counter(letters(text))
    nG = sum(G.values())
    gk = Counter({c: n for c, n in G.items() if KANJI.match(c)})
    ga = G - gk
    text_m = box_mask([r["box"] for r in boxes], hw)
    en_m = box_mask([r["box"] for r in en_reads if not r.get("whole")], hw)
    t_area = int(text_m.sum())
    per = []
    for rd in READERS:
        D, on = Counter(), []
        for r in boxes:
            b = Counter(letters(r.get(rd)))
            D += b
            hit, nb = sum((b & G).values()), sum(b.values())
            if nb and hit >= max(min(2, nG), ON_SHARE * nb):
                on.append(r["box"])
        hit, nD = sum((D & G).values()), sum(D.values())
        pr = hit / nD if nD else 0.0
        rc = hit / nG if nG else 0.0
        on_area = int((box_mask(on, hw) & text_m).sum())
        per.append(
            {
                "g_p": pr,
                "g_r": rc,
                "g_f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
                "g_r_kanji": sum((D & gk).values()) / sum(gk.values()) if gk else None,
                "g_r_kana": sum((D & ga).values()) / sum(ga.values()) if ga else None,
                "drawn": nD,
                "a_p": on_area / t_area if t_area else 0.0,
            }
        )
    out = {
        k: None if per[0][k] is None else sum(x[k] for x in per) / len(per)
        for k in per[0]
    }
    union = int((text_m | en_m).sum())
    return out | {
        "text_area": t_area / (hw[0] * hw[1]),
        "iou_en": int((text_m & en_m).sum()) / union if union else 0.0,
    }


def flat_white(file: str) -> float:
    """16² patches with std < 6 and mean > 225 (sigma_split's ``placement``)."""
    import numpy as np
    from PIL import Image

    im = np.asarray(Image.open(file).convert("L"), np.float32)
    H, W = im.shape
    p = im[: H // 16 * 16, : W // 16 * 16].reshape(H // 16, 16, W // 16, 16)
    return float(((p.std(axis=(1, 3)) < 6) & (p.mean(axis=(1, 3)) > 225)).mean())


def outside_mask(pe, n: int, hw, boxes):
    """``EnRef._outside_mask`` on the encoder's own patch grid: PE buckets
    the aspect (``pick_bucket``), so the grid is not ``√(n·W/H)`` off the
    square (800×736 → 1024 tokens)."""
    import math

    import torch

    from library.vision.encoder import pick_bucket

    H, W = hw
    gh, gw = pick_bucket(H, W, pe.bundle.bucket_spec)
    assert gh * gw == n, f"{n} tokens is no {gh}×{gw} grid for {W}x{H}"
    keep = torch.ones(gh, gw, dtype=torch.bool)
    for b in boxes:
        x0, y0 = int(b[0] / W * gw), int(b[1] / H * gh)
        x1, y1 = math.ceil(b[2] / W * gw), math.ceil(b[3] / H * gh)
        keep[max(0, y0) : min(gh, y1), max(0, x0) : min(gw, x1)] = False
    return keep.flatten()


class AtSim:
    """EN-ref similarity as ``anime_tools.grouping`` scores a near-twin pair
    (user, 10-05): the page stretched to PE-Spatial's 512² bucket, the CLS
    cosine (``en_cls``) and the dense grid match — mutual NN + ratio test over
    G×G pooled cells, the inlier fraction (``en_match``) — at the package's
    defaults."""

    def __init__(self, device="cuda"):
        from anime_tools.grouping import groups
        from anime_tools.grouping.embedder import pe_spatial_embedder

        self.emb = pe_spatial_embedder(device)
        self.g = groups.DEFAULT_GRID
        self.cell_min = groups.DEFAULT_CELL_MATCH_MIN
        self.ratio = groups.DEFAULT_RATIO
        self._memo: dict = {}

    def feats(self, path: Path):
        import torch
        from anime_tools.grouping.features import _load_512
        from anime_tools.grouping.matching import pool_cells_batch

        if path not in self._memo:
            cls, g16 = self.emb(_load_512(path)[None])
            cells = pool_cells_batch(
                torch.from_numpy(g16.astype("float32")).to(self.emb.device), self.g
            )
            self._memo[path] = (torch.from_numpy(cls[0]), cells)
        return self._memo[path]

    def __call__(self, f: Path, ref: Path) -> dict:
        from anime_tools.grouping.matching import match_fracs

        (ca, ga), (cb, gb) = self.feats(f), self.feats(ref)
        return {
            "en_cls": float(ca @ cb),
            "en_match": float(match_fracs(ga, gb, self.cell_min, self.ratio)[0]),
        }


def read_renders(names: list) -> dict:
    """``{arm: {i: record}}``; the reads are cached in ``<arm>/reads.json``,
    the page scores recomputed (cheap) every time."""
    import torch
    import torch.nn.functional as F

    from common.readers import Readers, load_bgr
    from eval.enref import EnRef

    its = items()
    rd = None
    raw: dict = {}
    for a in [EN, *names]:
        f = mode_dir() / a / "reads.json"
        got = json.loads(f.read_text("utf-8")) if f.exists() else {}
        todo = [m for m in its if str(m["i"]) not in got]
        missing = [m["i"] for m in todo if not render_file(a, m["i"]).exists()]
        assert not missing, f"{a}: not rendered: {missing[:8]}…"
        if todo:
            rd = rd or Readers("cuda")
            for m in todo:
                got[str(m["i"])] = rd.read_image(
                    load_bgr(render_file(a, m["i"])), whole=True
                )
            f.write_text(
                json.dumps(got, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            print(f"read {a}: {len(todo)}", flush=True)
        raw[a] = got
    del rd
    torch.cuda.empty_cache()
    pe = EnRef("cuda", mode_dir() / EN, {})
    at = AtSim("cuda")
    recs: dict = {}
    for a in names:
        recs[a] = {}
        for m in its:
            i = m["i"]
            reads, en_reads = raw[a][str(i)], raw[EN][str(i)]
            f, ref = render_file(a, i), render_file(EN, i)
            fi, hw = pe.tokens(f)
            fr, _ = pe.tokens(ref)
            boxes = [r["box"] for r in reads + en_reads if not r.get("whole")]
            keep = outside_mask(pe, fi.shape[0], hw, boxes)
            if keep.sum() < 4:
                keep[:] = True
            recs[a][i] = {
                "i": i,
                "bin": m["bin"],
                "unseen": (m["cov3"] or 0) <= UNSEEN,
                "text": m["text"],
                "file": str(f),
                **score_text(m["text"], reads),
                **score_page(m["text"], reads, en_reads),
                **at(f, ref),
                "en_tok_out": float(
                    F.cosine_similarity(fi[keep], fr[keep], dim=1).mean()
                ),
                "fw_over_en": flat_white(str(f)) - flat_white(str(ref)),
                "reads": reads,
            }
    return recs


def sign_p(g: int, lo: int) -> float:
    from math import comb

    n = g + lo
    if not n:
        return 1.0
    return float(
        f"{min(1.0, 2 * sum(comb(n, k) for k in range(min(g, lo) + 1)) / 2**n):.2g}"
    )


def groups(recs_a: dict) -> dict:
    out = {"all": list(recs_a)}
    for b in BINS:
        out[b] = [i for i, r in recs_a.items() if r["bin"] == b]
    out["unseen"] = [i for i, r in recs_a.items() if r["unseen"]]
    out["unseen_short"] = [i for i in out["unseen"] if recs_a[i]["bin"] == "short"]
    return out


def tally(recs: dict) -> dict:
    out = {}
    for a, ra in recs.items():
        out[a] = {}
        for g, ks in groups(ra).items():
            c = {"n": len(ks)} | {k: sum(ra[i][k] for i in ks) for k in BOOL}
            for k in REAL:
                xs = [ra[i][k] for i in ks if ra[i][k] is not None]
                c[k] = round(sum(xs) / len(xs), 4) if xs else None
            out[a][g] = c
    return out


def paired(ra: dict, rb: dict) -> dict:
    """Per group: McNemar on the booleans (gained / lost / p), the sign test
    and mean difference on the reals (a − b; ``cer`` lower is better)."""
    out = {}
    for g, ks in groups(ra).items():
        c = {"n": len(ks)}
        for k in BOOL:
            gn = sum(ra[i][k] and not rb[i][k] for i in ks)
            ls = sum(rb[i][k] and not ra[i][k] for i in ks)
            c[k] = [gn, ls, sign_p(gn, ls)]
        for k in REAL:
            dif = [
                ra[i][k] - rb[i][k]
                for i in ks
                if ra[i][k] is not None and rb[i][k] is not None
            ]
            up, dn = sum(x > 1e-9 for x in dif), sum(x < -1e-9 for x in dif)
            c[k] = [round(sum(dif) / max(1, len(dif)), 4), up, dn, sign_p(up, dn)]
        out[g] = c
    return out


def sheets(recs: dict, out: Path, per: int = 8) -> None:
    """Per bin, ``per`` strings a sheet: a row per string, EN ref | the arms."""
    from PIL import Image

    from common.readers import contact_sheet

    out.mkdir(parents=True, exist_ok=True)
    names = list(recs)
    its = items()
    for b in BINS:
        xs = [m for m in its if m["bin"] == b]
        for k in range(0, len(xs), per):
            rows = []
            for m in xs[k : k + per]:
                i = m["i"]
                rows.append(
                    (
                        Image.open(render_file(EN, i)).convert("RGB"),
                        [f"r{i:02d} {m['text']}", f"EN {m['en']}"],
                    )
                )
                for a in names:
                    r = recs[a][i]
                    mark = "✓" if r["exact"] else "≤1" if r["le1"] else ""
                    rows.append(
                        (
                            Image.open(r["file"]).convert("RGB"),
                            [
                                f"{a} {mark}",
                                f"{r['best'][:22]}",
                                f"cer {r['cer']:.2f} tok {r['en_tok_out']:.3f}",
                                f"P {r['g_p']:.2f} R {r['g_r']:.2f} "
                                f"F1 {r['g_f1']:.2f} on {r['a_p']:.2f}",
                            ],
                        )
                    )
            contact_sheet(
                rows, out / f"sheet_{b}_{k // per}.png", thumb=256, cols=1 + len(names)
            )


def read(names: list, label: str) -> Path:
    from bench._common import make_run_dir, write_result

    from reseed import HOME

    recs = read_renders(names)
    t = tally(recs)
    for a, gs in t.items():
        for g, c in gs.items():
            print(f"  {a:<18} {g:<13} {c}", flush=True)
    pairs = {}
    for k, a in enumerate(names):
        # every arm against the floor and every arm named before it
        for b in dict.fromkeys([*FLOOR[1:], *names[:k]]):
            if a != b and b in recs:
                pairs[f"{a} vs {b}"] = pr = paired(recs[a], recs[b])
                print(f"  {a} vs {b}: {pr['all']}", flush=True)
    run_dir = make_run_dir(
        "cjk_anima_reseed", label=f"ruler-{MODE}-{label}", root=HOME / "results"
    )
    sheets(recs, run_dir / "sheets")
    (run_dir / "renders.json").write_text(
        json.dumps(
            {a: list(r.values()) for a, r in recs.items()}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )
    write_result(
        run_dir,
        script=__file__,
        args={
            "arms": names,
            "label": label,
            "prompts": MODE,
            "unseen": UNSEEN,
            "pack": PACK,
            "marks_only": MARKS_ONLY,
        },
        label=label,
        metrics={"tally": t, "paired": pairs},
        artifacts=[str(RULER)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)
    return run_dir


def sample(
    run_dir: Path, n: int, seed: int, has: str = "", per: int = 4, thumb: int = 384
) -> Path:
    """``n`` random strings of a read (its ``renders.json``; ``has``: only
    those holding one of these chars), ``per`` a sheet: a row per string, EN
    ref | every arm read, larger than the read's sheets →
    ``<run_dir>/random_s<seed>[_<has>]/``."""
    import random

    from PIL import Image

    from common.readers import contact_sheet

    recs = json.loads((run_dir / "renders.json").read_text(encoding="utf-8"))
    names = list(recs)
    by = {a: {r["i"]: r for r in rs} for a, rs in recs.items()}
    its = {
        m["i"]: m
        for m in items()
        if all(m["i"] in b for b in by.values())
        and (not has or set(has) & set(m["text"]))
    }
    n = min(n, len(its))
    pick = sorted(random.Random(seed).sample(sorted(its), n))
    out = run_dir / (f"random_s{seed}" + (f"_{has}" if has else ""))
    out.mkdir(parents=True, exist_ok=True)
    for k in range(0, n, per):
        rows = []
        for i in pick[k : k + per]:
            m = its[i]
            rows.append(
                (
                    Image.open(render_file(EN, i)).convert("RGB"),
                    [f"r{i:02d} {m['bin']} {m['text']}", f"EN {m['en']}"],
                )
            )
            for a in names:
                r = by[a][i]
                mark = (
                    "✓"
                    if r["exact"]
                    else "≤1"
                    if r["le1"]
                    else "≤2"
                    if r["le2"]
                    else ""
                )
                rows.append(
                    (
                        Image.open(r["file"]).convert("RGB"),
                        [f"{a} {mark}", r["best"][:24], f"cer {r['cer']:.2f}"],
                    )
                )
        contact_sheet(
            rows, out / f"random_{k // per}.png", thumb=thumb, cols=1 + len(names)
        )
    print(f"{n} strings {pick} → {out}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=["build", "render", "read", "run", "sample"])
    p.add_argument(
        "--arms",
        default=",".join(FLOOR),
        help="comma list; `en` = the EN references (default: the floor)",
    )
    p.add_argument("--label", default="floor")
    p.add_argument("--prompts", choices=MODES, default=MODES[0])
    p.add_argument("--pack", default="", help="render on this base pack (reseed PACKS)")
    p.add_argument(
        "--marks", action="store_true", help="only the strings holding a mark"
    )
    p.add_argument(
        "--only", default="", help="render: these ruler indices only (comma list)"
    )
    p.add_argument(
        "--rows_pt",
        default="",
        help="name=path,…: a probe's rows.pt on stick080's rows as arm `name`",
    )
    p.add_argument("--from", dest="src", help="sample: a read's results dir")
    p.add_argument("--n", type=int, default=12, help="sample: strings drawn")
    p.add_argument("--seed", type=int, default=0, help="sample: the draw's seed")
    p.add_argument(
        "--has", default="", help="sample: only strings holding one of these chars"
    )
    a = p.parse_args()
    if a.verb == "build":
        build()
        return
    if a.verb == "sample":
        bootstrap()
        sample(Path(a.src), a.n, a.seed, a.has)
        return
    global MODE, PACK, MARKS_ONLY, ONLY
    MODE, PACK, MARKS_ONLY = a.prompts, a.pack, a.marks
    ONLY = {int(x) for x in a.only.split(",") if x}
    assert not (ONLY and a.verb in ("read", "run")), "--only is a look: render"
    if PACK:
        from reseed.config import PACKS

        os.environ["ANIMA_VOCAB_PACK"] = PACKS[PACK]
    bootstrap()
    os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the runs trained routed, read routed
    for spec in filter(None, a.rows_pt.split(",")):
        k, v = spec.split("=", 1)
        assert k not in DERIVED and k not in arm_dirs(), f"arm {k} exists"
        DERIVED[k] = lambda v=v: rows_pt(
            REPO / v if not Path(v).is_absolute() else Path(v)
        )
    names = [x for x in a.arms.split(",") if x]
    plain = {EN, *arm_dirs(), *DERIVED}
    known = plain | ({f"{x}@{PACK}" for x in plain - {EN}} if PACK else set())
    assert set(names) <= known, f"unknown arms {set(names) - known}"
    if a.verb in ("render", "run"):
        # on a pack, render only its own arms: the raw pack's are on disk
        bad = [x for x in names if PACK and "@" not in x and x not in PACK_ARMS[PACK]]
        assert not bad, f"--pack {PACK} renders X@{PACK} or its own arms, not {bad}"
        render(names)
    if a.verb in ("read", "run"):
        # the floor arms always ride along: every read pairs against them
        rd = [x for x in dict.fromkeys([*FLOOR[1:], *names]) if x != EN]
        read(rd, a.label)


if __name__ == "__main__":
    main()
