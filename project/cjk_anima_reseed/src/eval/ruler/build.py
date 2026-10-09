"""The ruler's string set (CPU): the training set's own bubble dialogue —
captions tagged ``speech bubble`` / ``comic`` / ``dialogue``, their
``Japanese text reads as`` clause (not SFX) — up to two strings an image,
``PER_BIN`` per length bin, and each string given its own image's caption
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

Writes ``output/cjk_anima_reseed/ruler/ruler.json`` (+ ``ruler.tsv`` to read
by eye). The dataset's text stays out of the repo.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from reseed import REPO, bootstrap

from eval.ruler import BINS, RULER

CAPTIONS = REPO / "post_image_dataset" / "resized"
SEED = 1005
PER_BIN = 32
PER_IMAGE = 2
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

    from reseed.pools import is_ja_text

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
