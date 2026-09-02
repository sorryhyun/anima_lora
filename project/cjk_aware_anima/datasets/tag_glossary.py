"""EN→JA glossary for every tag our captions actually use (Phase 2a, D1 backbone).

Our captions are tag strings, so the JA side of the corpus is only as good as
the JA wording of the tags — and the *wording* is the whole point: ext rows are
only trained where the corpus visits them, so a JA caption saying 二本の髪 where
users type ツインテール trains the wrong rows. That is the same
distribution argument the proper-noun lexicon makes (``../findings.md``), applied to
the general vocabulary.

Machine translation is the wrong primary tool for this: it renders meaning, not
community register. Measured on Hy-MT2-1.8B, a held-out idiom probe scored 8/28
bare and 9/28 with few-shot exemplars (``mt.py --probe``) — the misses are
knowledge (ツインテール, カメラ目線, 割座), not instruction following.

So the glossary is sourced in priority order:

1. **Danbooru wiki ``other_names``** — the tagging community's own Japanese for
   its own tags, which is exactly the register users type. Covers **85% of our
   tag occurrences** and 1,130/1,314 character tags.
2. **Wikidata lexicon** (``wikidata_lexicon.json``) — preferred for proper
   nouns where it has an entry: CC0, spot-checked, and it carries ko/zh for the
   deferred phases.
3. **Hy-MT2** (``--mt``, needs a GPU) — the residue, which is mostly
   compositional tags (``black skirt``, ``open jacket``) where literal
   translation is the correct answer anyway.
4. A small built-in map for the rating band, which is ours and not Danbooru's.

Script detection matters: ``other_names`` mixes ja/zh/ko. Entries containing
kana are unambiguously Japanese; Han-only entries are accepted only if they
encode in Shift-JIS, which rejects simplified Chinese (小鸟游星野 → out,
火宮チナツ → in) at zero dependency cost.

    python project/cjk_aware_anima/datasets/tag_glossary.py            # CPU, sources 1/2/4
    make daemon-run ARGS="project/cjk_aware_anima/datasets/tag_glossary.py --mt"
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
REPO = ROOT.parents[2]
sys.path.insert(0, str(ROOT))

import build_pairs  # noqa: E402  (sibling module — caption roots + loader)
import kanji_allow  # noqa: E402  (sibling — the allowed-kanji set)
from anime_tools.captions.position_clauses import parse_caption  # noqa: E402
from mt import (  # noqa: E402  (sibling — MT exemplars)
    TAG_FEWSHOT,
    TAG_FEWSHOT_KO,
    TAG_FEWSHOT_ZH,
)

WIKI_REPO = "kierarkia/danbooru-wiki-2026"
WIKI_FILE = "danbooru_wiki_dataset_2026-04-28.jsonl"
DEFAULT_INDEX = REPO / "post_image_dataset/captions/caption_index.json"
DEFAULT_CAPTIONS = REPO / "image_dataset"
DEFAULT_OUT = ASSETS / "tag_glossary_ja.json"

KANA = re.compile(r"[぀-ゟ゠-ヿー]")
HAN = re.compile(r"[一-鿿]")
# Guard-width Han: ext-A/compat too (䌷 U+4337 slipped past the narrow class).
HAN_WIDE = re.compile(r"[㐀-鿿豈-﫿]")

# The rating band is ours (safe/sensitive/nsfw/explicit), not a Danbooru tag
# whose wiki would carry a Japanese name.
# Sources the arbitration may re-open. Everything else (override / rating /
# wikidata / artist passthrough) is pinned: letting MT touch those overwrote the
# rating band with デリケート instead of センシティブ.
ARBITRATED = {"wiki", "wiki_han", "kb", "unresolved"}

# Candidate provenance, best first at equal evidence: the 2026 wiki dump is the
# freshest reading of the community field; the tag-pair set is the same field at
# Oct-2024 partly LLM-filled; MT renders the *string*, not the booru sense
# (`bow` → お辞儀 "bowing" where the tag means 蝶結び the ribbon), so at equal
# back-translation F1 a community-attested name outranks it.
SRC_RANK = {"wiki": 0, "tagpair": 1, "kb": 1, "kbmt": 2, "mt": 2}
TAGPAIR_VERIFIED_VIA = "tagpair_verified"

# r5 (2026-09-01): below this occurrence count, an unverified KB keyword
# outranks the unverified MT rendering. Back-translation F1 against the tag
# *name* cannot verify booru jargon (백합→"lily"≠yuri, 파이즈리→"titjob"≠
# paizuri), so in the tail — where nobody reviews and MT degrades into
# semantic howlers (improvised gag→즉흥 개그) — the community field wins by
# default. Above the floor the r1–r4-reviewed MT wording stands (blanket
# KB-first regresses it: swimsuit→비키니, grey hair→은발). Evidence:
# reports/0901_ko_phase_k3.md; decision file tag_glossary_review_ko_r5_kb.md.
KB_UNVERIFIED_FLOOR = 100

RATING_JA = {
    "safe": "全年齢",
    "sensitive": "センシティブ",
    "nsfw": "成人向け",
    "explicit": "露骨な性描写",
}

HANGUL = re.compile(r"[가-힣ㄱ-ㅎㅏ-ㅣ]")

# user-reviewed 2026-08-31 — Arca Live register (후방주의 = community NSFW slang)
RATING_KO = {
    "safe": "건전",
    "sensitive": "약후방",
    "nsfw": "후방주의",
    "explicit": "성인용",
}

# zh (plan_zh.md Z1): mainland community register, simplified. 全年龄 is the
# Danbooru-zh wiki's own word for the safe band; the three graded bands are
# the fandom's (微涩 = "slightly spicy", 成人向 mirrors 成人向け). Review item.
RATING_ZH = {
    "safe": "全年龄",
    "sensitive": "微涩",
    "nsfw": "成人向",
    "explicit": "露骨性描写",
}

RATING = {"ja": RATING_JA, "ko": RATING_KO, "zh": RATING_ZH}

# ---- Chinese script routing (plan_zh.md "Script reality") -------------------
# Three Han inventories overlap only partly: JA shinjitai, zh-Hans, zh-Hant.
# OpenCC gives a per-character answer for two of the three: a char that s2t
# changes is simplified-only (发 → 髮), one that t2s changes is traditional-only
# (髮 → 发); everything else is *shared* between at least two inventories —
# and that "shared" class is where JA-only words hide (髪 / 顔 / 獣 are
# untouched by both converters and still not Chinese). The learned zh
# inventory (``zh_han_inventory``: chars seen inside hans-class wiki names)
# is the tie-breaker for shared wordings, mirroring ``ja_kanji_inventory``.
_OPENCC: dict[str, object] = {}
_HAN_CLASS_CACHE: dict[str, str] = {}


def _opencc(direction: str):
    if direction not in _OPENCC:
        from opencc import OpenCC  # noqa: PLC0415  (pure-python reimplementation)

        _OPENCC[direction] = OpenCC(direction)
    return _OPENCC[direction]


def han_char_class(ch: str) -> str:
    """'hans' | 'hant' | 'variant' | 'shared' | '' (non-Han).

    Round-trip rule: OpenCC's s2t table also knows Japanese shinjitai (対 →
    對, 広 → 廣, 観 → 觀) — a plain "s2t changes it" test called those
    simplified Chinese and let 左右対称 / 国広一 seed the zh inventory
    (measured 2026-09-02). A char is *hans* only if it is the canonical
    simplified form of what it converts to (``t2s(s2t(ch)) == ch``: 发 ✓,
    国 ✓, 対 ✗ since 對 → 对), *hant* only if it is the canonical traditional
    form (``s2t(t2s(ch)) == ch``: 髮 ✓), *variant* if a converter touches it
    but the round trip lands elsewhere (JA shinjitai, rare variants —
    never Chinese), *shared* if neither converter touches it.
    """
    if not HAN_WIDE.match(ch):
        return ""
    c = _HAN_CLASS_CACHE.get(ch)
    if c is None:
        s2t, t2s = _opencc("s2t"), _opencc("t2s")
        up, down = s2t.convert(ch), t2s.convert(ch)
        if up != ch and t2s.convert(up) == ch:
            c = "hans"
        elif down != ch and s2t.convert(down) == ch:
            c = "hant"
        elif up != ch or down != ch:
            c = "variant"
        else:
            c = "shared"
        _HAN_CLASS_CACHE[ch] = c
    return c


def han_class(s: str) -> str:
    """Wording-level class: any variant char ⇒ variant (not Chinese); else any
    simplified-only char ⇒ hans; else any traditional-only ⇒ hant; else
    shared (or '' with no Han at all)."""
    classes = {han_char_class(c) for c in HAN_WIDE.findall(s)}
    if "variant" in classes:
        return "variant"
    if "hans" in classes:
        return "hans"
    if "hant" in classes:
        return "hant"
    return "shared" if classes else ""


def to_hant(s: str) -> str:
    return _opencc("s2t").convert(s)


def to_hans(s: str) -> str:
    return _opencc("t2s").convert(s)


# Set by ``build`` for --lang zh (``zh_han_inventory``); None = accept every
# shared-class wording (tests / callers without the dump).
ZH_INVENTORY: set[str] | None = None


def zh_wording_ok(s: str) -> bool:
    """Veto-only zh analog of ``han_allowed``: kana or hangul means the
    candidate is Japanese/Korean; latin/digit wordings pass."""
    return not KANA.search(s) and not HANGUL.search(s)


def is_chinese(s: str, inventory: set[str] | None = None) -> bool:
    """Han-bearing, no kana/hangul, and either script-marked (hans/hant) or —
    for a shared-class wording — built only from chars the zh inventory has
    seen (棕毛 passes on 棕; 髪飾り fails on kana; 巨乳 passes shared)."""
    if not zh_wording_ok(s) or not HAN_WIDE.search(s):
        return False
    cls = han_class(s)
    if cls in ("hans", "hant"):
        return True
    inv = ZH_INVENTORY if inventory is None else inventory
    if inv is None:
        return True
    return all(c in inv for c in HAN_WIDE.findall(s))


def zh_han_inventory(
    wiki_path: Path, pack_paths: list[Path] | None = None, wiki_min_count: int = 5
) -> set[str]:
    """Han chars that really occur in Chinese, for the shared-class check.

    Primary census: the community tag packs (human-written zh_CN wordings —
    Chinese by construction, every char counts once). The wiki's hans-class
    entries were tried first and leak: a JA word whose shinjitai coincides
    with a simplified form (左右対称 on 称, 国広一 on 国) is classed hans and
    seeds 対 / 広 into the inventory (2026-09-02). They are kept only as a
    high-count supplement (``wiki_min_count``) for rare name chars the packs
    never spell. Traditional entries are folded through t2s so the inventory
    is simplified-normalized.
    """
    import csv  # noqa: PLC0415

    inv: set[str] = set()
    for path in pack_paths if pack_paths is not None else ZH_KB_DEFAULT:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                for cell in row[1:]:
                    if KANA.search(cell) or HANGUL.search(cell):
                        continue
                    inv.update(c for c in cell if HAN_WIDE.match(c))
    freq: collections.Counter = collections.Counter()
    if wiki_path.exists():
        for line in wiki_path.open(encoding="utf-8"):
            for name in json.loads(line).get("other_names") or []:
                if KANA.search(name) or HANGUL.search(name):
                    continue
                cls = han_class(name)
                if cls == "hans":
                    freq.update(c for c in name if HAN_WIDE.match(c))
                elif cls == "hant":
                    freq.update(c for c in to_hans(name) if HAN_WIDE.match(c))
    inv |= {c for c, n in freq.items() if n >= wiki_min_count}
    return inv


def han_allowed(s: str) -> bool:
    """Veto only disallowed Han — a wording with no Han at all passes.

    This is the guard for slots that never required Japaneseness (the MT
    fallback, stored-candidate re-checks, the wikidata wording): `:d`, `OL`,
    `3D` are legitimate wordings there and must not be rejected for merely
    not being Japanese. ``is_japanese`` (below) is the stricter predicate for
    the *name* paths, where Japaneseness is the point.
    """
    return all(c in kanji_allow.ALLOWED for c in HAN_WIDE.findall(s))


def is_japanese(s: str) -> bool:
    """Kana ⇒ Japanese — but every Han char must be in ``kanji_allow.ALLOWED``.

    The char-set guard runs before the kana short-circuit on purpose: zh
    wordings with decorative kana (结月ゆかり, 歌爱ユキ) used to pass on their
    kana, and Shift-JIS — the old Han-only test — happily encodes 崩坏's 坏
    and 海梦's 梦 (JIS level 2 carries some simplified forms). The allowed set
    is joyo + jinmeiyo + a reviewed hyogai whitelist; census in
    ``kanji_allow.py``.
    """
    han = HAN_WIDE.findall(s)
    if any(c not in kanji_allow.ALLOWED for c in han):
        return False
    if KANA.search(s):
        return True
    return bool(han)


# Correctly-bounded wide Han for the KO veto. NB ``HAN_WIDE`` above has a
# latent quirk: its compat range starts at the *unified* 豈 (U+8C48), not the
# compatibility one (U+F900), so it also swallows hangul/Yi/private-use —
# harmless in the JA path (hangul is not Japanese either way; kept as-is to
# leave JA behavior bit-identical), fatal for KO.
HAN_KO = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")


def ko_wording_ok(s: str) -> bool:
    """Veto-only KO analog of ``han_allowed``: hanja is not what users type
    (plan_ko.md K1 — Han-only ``other_names`` are not Korean) and kana means
    the candidate is Japanese; latin/digit wordings pass, as in the JA path."""
    return not HAN_KO.search(s) and not KANA.search(s)


def is_korean(s: str) -> bool:
    """Any hangul syllable/jamo ⇒ Korean — unless ``ko_wording_ok`` vetoes."""
    return bool(HANGUL.search(s)) and ko_wording_ok(s)


def wording_ok(s: str, lang: str = "ja") -> bool:
    """Lang-dispatched veto (never *requires* nativeness — see the JA note)."""
    if lang == "ja":
        return han_allowed(s)
    if lang == "zh":
        return zh_wording_ok(s)
    return ko_wording_ok(s)


def is_native(s: str, lang: str = "ja") -> bool:
    if lang == "ja":
        return is_japanese(s)
    if lang == "zh":
        return is_chinese(s)
    return is_korean(s)


# proof-of-nativeness script per language (kana for JA, hangul for KO, Han for
# ZH — where the inventory, not the script, carries the evidence)
NATIVE_RE = {"ja": KANA, "ko": HANGUL, "zh": HAN_WIDE}


def rank_names(names: list[str], lang: str = "ja") -> list[str]:
    """Native candidates; for JA, kana-bearing first (least ambiguous evidence).

    For ZH: hans-marked first (unambiguous), then shared, then hant (kept —
    the ``tags_zh_hant`` register and the alt pool want them), each by length.
    """
    if lang == "zh":
        # hans and inventory-passed shared tie (妃咲 over 妃咲会长); hant last
        order = {"hans": 0, "shared": 0, "hant": 1, "variant": 2}
        ranked = sorted(
            (n for n in names if is_chinese(n)),
            key=lambda n: (order[han_class(n)], len(n)),
        )
        # simplified-normalized (the Hant register is derived from the composed
        # caption by OpenCC in build_pairs, not carried per wording)
        keep: list[str] = []
        for n in ranked:
            n = to_hans(n)
            if n not in keep:
                keep.append(n)
        return keep
    if lang != "ja":
        keep = [n for n in names if is_native(n, lang)]
        keep.sort(key=len)
        return keep
    ja = [n for n in names if is_japanese(n)]
    ja.sort(key=lambda n: (0 if KANA.search(n) else 1, len(n)))
    return ja


_KB_KEYWORDS = re.compile(r"키워드\s*:\s*(.+)$", re.S)


def load_kr_kb(path: Path, lang: str = "ko") -> dict[str, list[str]]:
    """EN tag -> hangul keyword list from ``danbooru_tags_classified.csv``.

    The Korean community KB (Localsmile/danbooru_KR_wiki_tag_search, fetched by
    ``make download-danbooru-tags``) carries a ``키워드:`` field per tag — the
    Korean names users actually search with (트윈테일/양갈래, 갑옷/아머). It is
    the KO analog of the JA tag-pair set: 83.6%% of our general tag types
    (95.7%% of occurrences) measured 2026-08-31. Search keywords are noisier
    than a curated pair set, so entries compete through the same
    back-translation arbitration (``src: "kb"``), never win unexamined.
    """
    import csv  # noqa: PLC0415

    out: dict[str, list[str]] = {}
    if not path.exists():
        print(f"  [glossary] KR KB missing at {path} — kb tier disabled")
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 4 or not row[0].strip():
                continue
            m = _KB_KEYWORDS.search(row[3])
            if not m:
                continue
            name = row[0].strip().replace("_", " ").lower()
            kws = [k.strip() for k in m.group(1).split(",")]
            kws = [k for k in kws if k and is_native(k, lang)]
            if kws:
                out[name] = kws
    return out


# zh community tag packs (plan_zh.md §Sources 1): the tagcomplete ecosystem's
# EN→zh_CN tables, i.e. the wording Chinese users actually type (双马尾, 傲娇).
# Priority order (user call 2026-09-02: centre on the NGA translation):
#   HalfMAI  danbooru-0-zh-nga.csv   5.4k tags, human-translated NGA-community
#            register (39 → 初音未来), `|`-separated alternates; gist with NO
#            stated licence (source thread ngabbs.com/read.php?tid=33869519)
#            — provenance recorded per entry (``src: kb``), review the
#            licence before a public release of the glossary itself.
#   byzod    Tags-zh-full-pack.csv   10.6k tags, one curated wording each (MIT)
#   ChinaGPT danbooru-10w-zh_cn.csv  100k tags, space-separated machine
#            renderings (短毛短毛猫 for `short hair`, MIT) — a candidate pool
#            only, ranked with the MT tier (``src: kbmt``), never a sub-floor
#            default.
# Yellow-Rush/zh_CN-Tags carries no licence and is not read.
NAME_AXES_ZH = {"character", "copyright"}
_ZH_QUALIFIER = re.compile(r"\s*[（(][^（）()]*[）)]\s*$")
ZH_KB_DEFAULT = [
    ASSETS / ".zh" / "HalfMAI_danbooru-0-zh-nga.csv",
    ASSETS / ".zh" / "byzod_Tags-zh-full-pack.csv",
    ASSETS / ".zh" / "ChinaGPT_danbooru-10w-zh_cn.csv",
]


def load_zh_kb(
    paths: list[Path], max_per_tag: int = 3
) -> dict[str, list[tuple[str, str]]]:
    """EN tag -> [(zh wording, src)] from the community packs, curated first.

    Row shapes: ``tag,zh`` (byzod), ``tag,zh1 zh2 …`` (10w, space-packed
    alternates), ``tag,category,zh|alt,`` (tagcomplete extra-file layout,
    the NGA gist). The wording column is the first Han-bearing cell after
    the tag; ``|`` splits alternates in every layout.
    """
    import csv  # noqa: PLC0415

    out: dict[str, list[tuple[str, str]]] = {}
    for path in paths:
        if not path.exists():
            print(f"  [glossary] zh pack missing at {path} — skipped")
            continue
        src = "kbmt" if "10w" in path.name else "kb"
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 2 or not row[0].strip():
                    continue
                name = row[0].strip().replace("_", " ").lower()
                cell = next((c for c in row[1:] if HAN_WIDE.search(c)), "")
                raw = cell.replace("\t", " ").replace("\n", " ")
                # the 10w file packs alternates space-separated; the curated
                # files use `|` (a curated wording may itself contain a space)
                cands = raw.split() if src == "kbmt" else raw.split("|")
                seen = {w for w, _ in out.get(name, [])}
                n_src = sum(1 for _, s_ in out.get(name, []) if s_ == src)
                for w in cands:
                    w = w.strip(" ,;")
                    if not w or w in seen or w.lower() == name or not is_chinese(w):
                        continue
                    if src == "kb" and han_class(w) == "hant":
                        w = to_hans(w)  # the packs are nominally zh_CN
                    out.setdefault(name, []).append((w, src))
                    seen.add(w)
                    n_src += 1
                    if n_src >= max_per_tag:
                        break
    return out


def ja_kanji_inventory(path: Path, min_count: int = 3) -> set[str]:
    """Kanji that really occur in Japanese text, learned from the dump itself.

    Shift-JIS rejects simplified Chinese but happily encodes *traditional*
    Chinese (棕毛, 藍眼睛 — both scored a perfect back-translation F1 and are
    still not Japanese). Every ``other_names`` entry containing kana is
    Japanese by construction, so the kanji inside those entries are a free
    inventory of the JA repertoire; a Han-only candidate built from characters
    that never show up there is almost certainly Chinese.
    """
    freq: collections.Counter = collections.Counter()
    for line in path.open(encoding="utf-8"):
        for name in json.loads(line).get("other_names") or []:
            if KANA.search(name):
                freq.update(c for c in name if HAN.match(c))
    return {c for c, n in freq.items() if n >= min_count}


# Danbooru wiki ``category_name`` → our axis. Only consulted for tags the
# caption index does not classify: the index covers ``image_dataset`` (~3k
# images) while the glossary is built over the D1-wide roots (16k captions), so
# every character that never occurs in the small set used to fall through to
# ``general`` — and then went through MT, which renders a *name* as words
# (``ame (mignon)`` → 雨（可愛い）; 6,149 of 14,959 `names` pairs were affected
# on 2026-08-30). Names are never MT-able; the wiki knows which tags are names.
WIKI_AXES = {"Character": "character", "Copyright": "copyright", "Artist": "artist"}


def load_wiki_axes(path: Path) -> dict[str, str]:
    """``title -> axis`` for every wiki entry whose category is a name axis."""
    out: dict[str, str] = {}
    for line in path.open(encoding="utf-8"):
        d = json.loads(line)
        axis = WIKI_AXES.get(d.get("category_name") or "")
        title = (d.get("title") or "").replace("_", " ").strip().lower()
        if axis and title:
            out[title] = axis
    return out


_OC_RE = re.compile(r"^.+ \(([^()]+)\)$")


def resolve_axis(
    tag: str,
    axis_of: dict[str, str],
    wiki_axes: dict[str, str],
    artists: frozenset[str] = frozenset(),
) -> tuple[str, str]:
    """(axis, source): caption index → wiki category → artist-OC form →
    ``@`` handle → general.

    ``name (handle)`` where ``@handle`` is an artist in the corpus is an
    original character of that artist (``akiyama fumika (pepper0)``); such OCs
    rarely have a wiki page, so without this rule they fall to ``general`` and
    get MT-rendered as words."""
    if tag in axis_of:
        return axis_of[tag], "index"
    if tag.lower() in wiki_axes:
        return wiki_axes[tag.lower()], "wiki"
    m = _OC_RE.match(tag)
    if m and f"@{m.group(1)}" in artists:
        return "character", "artist_oc"
    return ("artist" if tag.startswith("@") else "general"), "default"


# ---------------------------------------------------------------------------
# Candidate contamination (2026-09-02). Two leak paths were found in the live
# glossary: Hy-MT2 echoing its own few-shot list at tags it finds meaningless
# (`:t` → `:女の子1人の全身写真、シンプルな背景、肩出し、制服、ニーハイ、金�`,
# `\m/` → the whole list) or substituting an exemplar word the tag never
# licensed (`pantyhose only` → ニーハイのみ, `chick` → 女の子1人); and the wiki
# `other_names` field carrying fujoshi-community titles (`fire emblem` →
# FE腐向け). All of these are vetoes: a rejected candidate falls to the next
# one or to `unresolved` — nothing here invents a translation.

FEWSHOT_NATIVE = {
    "ja": [ja for _, ja in TAG_FEWSHOT],
    "ko": [ko for _, ko in TAG_FEWSHOT_KO],
    "zh": [zh for _, zh in TAG_FEWSHOT_ZH],
}

# JA exemplar word → EN stems that license it in the source tag. An MT
# rendering carrying the word for a tag with none of the stems copied it from
# the prompt, not from the tag. Substring match on the lowercased tag, so
# `full-length mirror` keeps 全身鏡 and `2girls` keeps 女の子.
EXEMPLAR_LICENCE: dict[str, tuple[str, ...]] = {
    "ニーハイ": ("thigh", "knee", "legwear", "stocking", "sock"),
    "肩出し": ("shoulder",),
    "制服": ("uniform", "school", "serafuku", "seifuku", "sailor", "gakuran", "blazer"),
    "ショートヘア": ("hair", "bob"),
    "女の子1人": ("1girl", "girl", "female"),
    "金髪": ("blonde", "gold", "yellow"),
    "巨乳": ("breast", "bust", "busty", "oppai"),
    "シンプルな背景": ("background",),
    "全身": ("body", "full-length"),
    "修正あり": ("censor",),
}

# Community-register markers that are only right when the tag is about them.
# The wiki `other_names` of mainstream titles carry BL-community puns
# (`fire emblem` → FE腐向け / ガチホモエムブレム, `granblue fantasy` → グラ腐ル,
# `genshin impact` → 原神BL), and a name-axis alt is exactly what
# ``build_pairs.alt_pool`` samples into captions. Marker regex → licensing EN
# regex over the lowercased tag (word-bounded: `bl` must not match `black`).
_BL = r"yaoi|fujoshi|boys'? love|shounen[- ]ai|\bbl\b"
REGISTER_LICENCE: list[tuple[re.Pattern, re.Pattern]] = [
    (re.compile("腐"), re.compile(_BL + r"|tofu|\brot|decay|zombie|corrupt")),
    (re.compile("ホモ"), re.compile(_BL + r"|gay|homo|bara")),
    (re.compile("やおい|ヤオイ"), re.compile(r"yaoi")),
    (re.compile(r"(?<![A-Za-z])BL(?![A-Za-z])"), re.compile(_BL)),
]


def contaminated(cand: str, tag: str, src: str | None, lang: str = "ja") -> bool:
    """Veto a candidate wording that is prompt/community contamination.

    Classes (all veto-only — never require nativeness of a candidate):

    a. broken or list-shaped text, any source: U+FFFD, or 2+ *distinct*
       few-shot exemplar words (the echoed prompt); MT source only: the JA
       enumeration/sentence punctuation ``、`` / ``。`` — real wiki titles carry
       it legitimately (探偵はもう、死んでいる).
    b. exemplar leak, MT source only (wiki/tag-pair names are real community
       terms): an ``EXEMPLAR_LICENCE`` word whose stems are all absent from
       the tag.
    c. community-register markers, any source (``REGISTER_LICENCE``): 腐 /
       ホモ / やおい / standalone ``BL`` — unless the tag itself is about BL
       (or tofu, rot, …).
    """
    if "\ufffd" in cand:
        return True
    shots = FEWSHOT_NATIVE[lang]
    if sum(1 for w in shots if w in cand) >= 2:
        return True
    if src == "mt" and ("、" in cand or "。" in cand):
        return True
    if src == "mt" and lang == "zh" and any(p in cand for p in "，；："):
        return True  # list/sentence-shaped rendering (zh enumeration comma etc.)
    low = tag.lower()
    if src == "mt" and lang == "ja":
        for word, stems in EXEMPLAR_LICENCE.items():
            if word in cand and not any(stem in low for stem in stems):
                return True
    for marker, licence in REGISTER_LICENCE:
        if marker.search(cand) and not licence.search(low):
            return True
    return False


# Danbooru emoticon / symbol tags. `:d`, `^^^`, `\m/` are typed latin in every
# language ([[feedback_emoticon_tags_stay_latin]]): MT fullwidth-converts them
# (`!` → `！`) or echoes the exemplar list at them, and the community field
# offers *descriptions* (びっくりマーク), not the tag. The rules are shape-based
# so a new emoticon needs no override.
_EMOTICON_RES = (
    re.compile(r"^[:;]\S{1,3}$"),  # :d :t ;) :<=
    re.compile(r"^\S{1,3}[:;]$"),  # c: d: 3:
    re.compile(r"^\S_\S$"),  # >_< o_o ^_^
)
_ALNUM = re.compile(r"[a-z0-9]", re.I)
_FACE_CHARS = re.compile(r"[:;^<>|/\\_]")


def is_symbol_tag(tag: str) -> bool:
    """True for a tag whose surface is its own wording in every language."""
    if not _ALNUM.search(tag):
        return True  # ^^^  \||/  ...  !?  ??  + +
    if any(r.match(tag) for r in _EMOTICON_RES):
        return True
    return (  # \m/  ^o^  \o/  3:<  m/
        len(tag) <= 4
        and " " not in tag
        and len(_ALNUM.findall(tag)) <= 1
        and _FACE_CHARS.search(tag) is not None
    )


def load_wiki(path: Path, lang: str = "ja") -> dict[str, list[str]]:
    if not path.exists():
        print(f"  [glossary] fetching {WIKI_REPO} …", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "hf",
                "download",
                WIKI_REPO,
                WIKI_FILE,
                "--repo-type",
                "dataset",
                "--local-dir",
                str(path.parent),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    out: dict[str, list[str]] = {}
    for line in path.open(encoding="utf-8"):
        d = json.loads(line)
        title = (d.get("title") or "").replace("_", " ").strip().lower()
        names = rank_names(d.get("other_names") or [], lang)
        if title and names:
            out[title] = names
    return out


def tag_counts(
    caption_roots: list[tuple[Path, bool]],
    rules: Path,
    known: collections.abc.Container[str] = frozenset(),
) -> collections.Counter:
    """Tag occurrences over the same caption roots ``build_pairs.py`` composes.

    The glossary must span whatever D1 spans, or the widened captions compose
    with latin passthrough on every tag the narrow build never saw — so this
    shares ``build_pairs.load_captions`` (multi-root, raw roots normalized
    through gelcrawl's rules) rather than re-globbing one directory.

    ``known`` (lowercased tag names — wiki titles, overrides, lexicon) repairs
    the one ambiguity the clause grammar cannot resolve: a caption-final ``.``
    is the terminator (``unworn panties.``) unless the tag itself ends in one
    (``c.c.``, ``nanashi inc.``, ``takt op.``), so a stripped form that is not
    a known tag while its dotted form is gets the dot back.
    """
    from build_pairs import load_captions  # noqa: PLC0415

    counts: collections.Counter = collections.Counter()
    for _, text in load_captions(caption_roots, rules):
        # Position clauses (`<bag>. On the left, akita neru, yellow eyes.`)
        # are period-delimited, so a comma split minted `weight. On the left`
        # and bare `On the left` as tags; the grammar yields the bag plus each
        # clause's own tags, header excluded.
        parsed = parse_caption(text)
        for seg in (*parsed.flat_tags, *(t for c in parsed.clauses for t in c.tags)):
            counts[seg] += 1
    for tag in list(counts):
        if tag.lower() not in known and f"{tag.lower()}." in known:
            counts[f"{tag}."] += counts.pop(tag)
    return counts


def build(args: argparse.Namespace) -> dict:
    index = json.loads(Path(args.caption_index).read_text())
    overrides = (
        json.loads(Path(args.overrides).read_text())
        if Path(args.overrides).exists()
        else {}
    )
    lexicon = json.loads(Path(args.lexicon).read_text())["characters"]
    wiki = load_wiki(Path(args.wiki), args.lang)
    wiki_axes = load_wiki_axes(Path(args.wiki))
    global ZH_INVENTORY
    if args.lang == "zh":
        ZH_INVENTORY = zh_han_inventory(Path(args.wiki), args.zh_kb)
        print(f"  [glossary] zh Han inventory: {len(ZH_INVENTORY)} chars", flush=True)
        wiki = load_wiki(Path(args.wiki), args.lang)  # re-rank with the inventory live
    kr_kb = (
        load_kr_kb(Path(args.kr_kb))
        if args.lang == "ko"
        else {
            t: [w for w, s_ in ws if s_ == "kb"]
            for t, ws in load_zh_kb(args.zh_kb).items()
            if any(s_ == "kb" for _, s_ in ws)
        }
        if args.lang == "zh"
        else {}
    )
    roots = [(Path(p), False) for p in args.captions]
    roots += [(Path(p), True) for p in (args.raw_captions or [])]
    known = {t.lower() for t in (*wiki, *overrides, *lexicon)}
    counts = tag_counts(roots, Path(args.tag_rules), known)

    axis_of: dict[str, str] = {}
    for axis in ("character", "copyright", "artist"):
        for tag in index["groups"][axis]:
            axis_of[tag] = axis

    artists = frozenset(t for t in counts if t.startswith("@"))
    tags: dict[str, dict] = {}
    for tag, count in counts.most_common():
        axis, axis_src = resolve_axis(tag, axis_of, wiki_axes, artists)
        entry = {"count": count, "axis": axis, "axis_src": axis_src, "alts": []}

        wiki_names = [
            n
            for n in wiki.get(tag.lower(), [])
            if not contaminated(n, tag, "wiki", args.lang)
        ]
        lex = lexicon.get(tag)

        if tag in overrides:
            entry |= {"ja": overrides[tag], "via": "override"}
            entry["alts"] = [n for n in wiki_names if n != overrides[tag]][
                : args.max_alts
            ]
        elif is_symbol_tag(tag):
            # emoticons stay latin (`:d`, `^^^`) — pinned like artist handles
            entry |= {"ja": tag, "via": "passthrough"}
        elif axis == "artist":
            # pixiv/danbooru handles are latin identity — users type them latin,
            # and translating them would train rows nobody prompts with.
            entry |= {"ja": tag, "via": "passthrough"}
        elif tag in RATING[args.lang]:
            entry |= {"ja": RATING[args.lang][tag], "via": "rating"}
        elif lex and lex.get(args.lang) and wording_ok(lex[args.lang], args.lang):
            # Wikidata's `zh` label is frequently traditional (初音未來) — the
            # zh corpus is simplified-primary, the Hant register is derived.
            lex_name = to_hans(lex[args.lang]) if args.lang == "zh" else lex[args.lang]
            entry |= {"ja": lex_name, "via": "wikidata", "qid": lex.get("qid")}
            entry["alts"] = [n for n in wiki_names if n != lex_name][: args.max_alts]
        elif args.lang == "zh" and any(
            not contaminated(n, tag, "kb", args.lang) for n in kr_kb.get(tag, [])
        ):
            # zh: the curated community packs (NGA first) outrank the wiki —
            # user call 2026-09-02 — and the wiki's own names become the alts.
            # The wiki field for zh is where JA leaks in (尻 / 歯 / 半袖 pass
            # the shared-class inventory) and where bogus alts sit first
            # (1girl → 武装少女); the packs are the register users type.
            kb_names = [
                n for n in kr_kb[tag] if not contaminated(n, tag, "kb", args.lang)
            ]
            if axis in NAME_AXES_ZH and "(" in tag:
                # the packs mirror danbooru's qualifier (甘雨（原神）); users
                # type the bare name, like the JA/KO lexicon tiers carry it
                kb_names = [_ZH_QUALIFIER.sub("", n).strip() or n for n in kb_names]
            entry |= {"ja": kb_names[0], "via": "kb"}
            entry["alts"] = [
                n for n in (*kb_names[1:], *wiki_names) if n != kb_names[0]
            ][: args.max_alts]
        elif wiki_names:
            entry |= {
                "ja": wiki_names[0],
                "via": "wiki"
                if NATIVE_RE[args.lang].search(wiki_names[0])
                else "wiki_han",
            }
            entry["alts"] = wiki_names[1 : 1 + args.max_alts]
        elif any(not contaminated(n, tag, "kb", args.lang) for n in kr_kb.get(tag, [])):
            # general entries stay ARBITRATED (the --mt pass re-opens them);
            # name axes keep the KB wording directly, like the wiki tier
            kb_names = [
                n for n in kr_kb[tag] if not contaminated(n, tag, "kb", args.lang)
            ]
            entry |= {"ja": kb_names[0], "via": "kb"}
            entry["alts"] = kb_names[1 : 1 + args.max_alts]
        else:
            entry |= {"ja": None, "via": "unresolved"}
        tags[tag] = entry

    if args.reselect:
        n = reselect(tags, Path(args.reselect), args.accept_f1, args.lang)
        print(f"  [glossary] re-selected {n} tags from {args.reselect} (no GPU)")
    elif args.mt:
        _mt_pass(tags, args)

    total = sum(counts.values())
    by_via: collections.Counter = collections.Counter()
    occ_via: collections.Counter = collections.Counter()
    for tag, e in tags.items():
        by_via[e["via"]] += 1
        occ_via[e["via"]] += e["count"]
    codepoints = {
        c
        for e in tags.values()
        if e["ja"]
        for c in e["ja"]
        if is_native(c, args.lang) or NATIVE_RE[args.lang].search(c)
    }

    return {
        "meta": {
            "lang": args.lang,
            "caption_index": str(args.caption_index),
            "wiki": f"{WIKI_REPO}/{WIKI_FILE}",
            "lexicon": str(args.lexicon),
            "mt_model": args.model if args.mt else None,
            "n_tags": len(tags),
            "n_occurrences": total,
            "types_by_via": dict(by_via),
            "occurrences_by_via": dict(occ_via),
            "occurrence_coverage": round(
                100 * (total - occ_via.get("unresolved", 0)) / max(total, 1), 2
            ),
            "unique_cjk_codepoints": len(codepoints),
        },
        "tags": tags,
    }


_STOP = {
    "a",
    "an",
    "the",
    "of",
    "with",
    "in",
    "on",
    "at",
    "is",
    "are",
    "to",
    "and",
    "her",
    "his",
    "their",
    "its",
    "one",
    "s",
}


def _norm_tokens(s: str) -> set[str]:
    """Lowercase content tokens, lightly stemmed — enough to compare short tags."""
    out = set()
    for w in re.split(r"[^a-z0-9]+", s.lower()):
        if not w or w in _STOP:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def _f1(a: str, b: str) -> float:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    hit = len(ta & tb)
    if not hit:
        return 0.0
    p, r = hit / len(tb), hit / len(ta)
    return 2 * p * r / (p + r)


def choose(
    tag: str,
    entry: dict,
    cands: list[dict],
    mt: str,
    accept_f1: float,
    lang: str = "ja",
) -> None:
    """Pick the wording for one tag from its scored candidates.

    Pure post-processing over stored data — no GPU. Ranking is: recovered
    meaning first (back-translation F1), then *demonstrably* Japanese, then
    source, then brevity. Kana is the only proof of Japaneseness; the
    per-character inventory merely rules out characters Japanese never uses, so
    a Chinese word built from JA-common characters (珠宝, 拉下衣服) still gets
    through and brevity tie-breaking would hand it the win over ジュエリー at
    the same F1. Hence kana > the JA-targeting model's own rendering >
    everything else, and inside a kana tie the community field (which knows the
    booru *sense*) beats the MT rendering (which only knows the string).

    ``contaminated`` runs first on every candidate and on the MT fallback
    string (both the fresh ``--mt`` pass and ``--reselect`` land here), so a
    prompt echo can neither win nor be kept as ``mt_unverified``. The raw
    rendering still goes to ``mt_ja`` for the record.
    """
    bad = {c["ja"] for c in cands if contaminated(c["ja"], tag, c.get("src"), lang)}
    keep = [c for c in cands if c.get("ja_ok", True) and c["ja"] not in bad]
    keep.sort(
        key=lambda c: (
            -c["f1"],
            0 if c.get("kana") else 1 if c.get("mt") else 2,
            SRC_RANK.get(c.get("src"), 1),
            len(c["ja"]),
        )
    )
    entry["mt_ja"] = mt or None
    entry["candidates"] = [
        {k: c[k] for k in ("ja", "back", "f1", "kana", "src") if k in c} for c in keep
    ]
    entry["rejected_non_ja"] = [
        c["ja"] for c in cands if not c.get("ja_ok", True) and c["ja"] not in bad
    ]
    if bad:
        entry["rejected_contaminated"] = [c["ja"] for c in cands if c["ja"] in bad]
    mt_use = mt if mt and not contaminated(mt, tag, "mt", lang) else ""

    best = keep[0] if keep else None
    if lang == "zh":
        # gist-centred (user call 2026-09-02): a curated community wording
        # that back-translates to the tag wins outright, even when a literal
        # MT rendering scores higher — the pack is the register users type.
        kb_best = next((c for c in keep if c.get("src") == "kb"), None)
        if kb_best and kb_best["f1"] >= accept_f1:
            best = kb_best
    if best and best["f1"] >= accept_f1:
        if best["ja"] == mt:
            via = "mt_verified"
        elif best.get("src") == "tagpair":
            via = TAGPAIR_VERIFIED_VIA
        elif best.get("src") == "kb":
            via = "kb_verified"
        else:
            via = "wiki_verified"
        entry |= {
            "ja": best["ja"],
            "via": via,
            "f1": best["f1"],
        }
    elif (
        lang in ("ko", "zh")
        and entry.get("count", 0) < KB_UNVERIFIED_FLOOR
        and (kb := next((c for c in keep if c.get("src") == "kb"), None))
    ):
        # Sub-floor tail: the community keyword beats the unverified MT string
        # (see KB_UNVERIFIED_FLOOR). Marked kb_unverified so the review file
        # and the trust policy see the class.
        entry |= {"ja": kb["ja"], "via": "kb_unverified", "f1": kb["f1"]}
    elif mt_use and wording_ok(mt_use, lang):
        # No candidate recovers the tag — `closed mouth` came back as 目を閉じる
        # (closed *eyes*). Keep it, but mark it so the review file surfaces the
        # class instead of shipping it silently. (A rendering that fails the
        # kanji guard — 僵尸 — is not kept: better unresolved than Chinese.)
        entry |= {"ja": mt_use, "via": "mt_unverified"}
    elif best:
        entry |= {"ja": best["ja"], "via": "wiki_unverified", "f1": best["f1"]}
    else:
        entry |= {"ja": None, "via": "unresolved"}


def reselect(
    tags: dict[str, dict], prior_path: Path, accept_f1: float, lang: str = "ja"
) -> int:
    """Re-derive every choice from a previous build's stored candidates.

    Selection is pure post-processing, so a ranking or threshold fix must never
    re-buy the translations: a prior ``tag_glossary_ja.json`` already holds each
    candidate's back-translation and F1 plus the MT rendering. Changing
    ``--accept-f1`` or the ranking is then seconds of CPU, not an hour of GPU.
    """
    prior = json.loads(prior_path.read_text(encoding="utf-8"))["tags"]
    n = 0
    for tag, e in tags.items():
        if e["axis"] != "general" or e["via"] not in ARBITRATED:
            continue  # never re-open a pinned source (override / rating / wikidata)
        p = prior.get(tag)
        if not p or not (p.get("candidates") or p.get("mt_ja")):
            continue
        mt = (p.get("mt_ja") or "").strip()
        cands = [
            {
                **c,
                # Re-check the Han veto instead of trusting the prior build:
                # the guard may have tightened since (kanji_allow), and
                # reselect exists precisely to apply such fixes without the
                # GPU. Veto-only — latin candidates (`:d`, `OL`) stay eligible
                # exactly as the prior build treated them.
                "ja_ok": wording_ok(c["ja"], lang),
                "kana": NATIVE_RE[lang].search(c["ja"]) is not None,
                "mt": c["ja"] == mt,
                # pre-src builds only banked wiki candidates + the MT rendering
                "src": c.get("src") or ("mt" if c["ja"] == mt else "wiki"),
            }
            for c in (p.get("candidates") or [])
        ]
        choose(tag, e, cands, mt, accept_f1, lang)
        n += 1
    return n


def _mt_pass(tags: dict[str, dict], args: argparse.Namespace) -> None:
    """Render every general tag with MT, then arbitrate MT vs wiki idiom (GPU).

    Neither source is trustworthy alone. MT is literal and safe on compositional
    tags (``black skirt``) but does not know community idiom — measured 46%/54%
    on the held-out idiom probe even at 7B. The wiki knows the idiom but its
    ``other_names`` is a free-text field that also carries narrower compounds
    (``underwear`` → 下着コート) and outright different concepts
    (``multiple girls`` → 群像).

    The arbiter is back-translation: a wiki candidate is accepted only if
    translating it *back* to English recovers the tag (token F1 ≥
    ``--accept-f1``). Idiom that verifiably means the tag wins; the rest falls
    back to the literal MT rendering. Whatever the two sources disagree on lands
    in the review file for the user sign-off the phase gate asks for.

    The tag-pair set (``tag_pairs.py``'s source) competes on the same terms
    (D1-pairs item 2): its names enter the pool guarded exactly like the fill
    (latin drop, ``is_japanese``, kanji inventory), get back-translated, and win
    only on evidence — ``via: tagpair_verified``. It knows the booru sense MT
    cannot (`bow` → 蝶結び the ribbon, not お辞儀 *bowing*), which is why the
    tie-break in ``choose`` puts community sources ahead of the MT rendering.
    """
    sys.path.insert(0, str(ROOT))
    from mt import (  # noqa: PLC0415
        TAG_BACKGROUND,
        TAG_BACKGROUND_KO,
        TAG_BACKGROUND_ZH,
        TAG_FEWSHOT,
        TAG_FEWSHOT_KO,
        TAG_FEWSHOT_ZH,
        MTEngine,
        Request,
    )

    background = {
        "ja": TAG_BACKGROUND,
        "ko": TAG_BACKGROUND_KO,
        "zh": TAG_BACKGROUND_ZH,
    }[args.lang]
    fewshot = {"ja": TAG_FEWSHOT, "ko": TAG_FEWSHOT_KO, "zh": TAG_FEWSHOT_ZH}[args.lang]

    general = [
        t for t, e in tags.items() if e["axis"] == "general" and e["via"] in ARBITRATED
    ]
    if args.limit:
        general = sorted(general, key=lambda t: -tags[t]["count"])[: args.limit]
    todo = set(general)

    # Exemplars must be hand-checked: drawing them from the unverified wiki head
    # fed junk back into the prompts (`underwear` → 下着コート became a shot and
    # MT then reproduced it).
    shots = fewshot[: args.n_shots]
    if args.lang == "ja":
        inventory = ja_kanji_inventory(Path(args.wiki))
        print(f"  [glossary] JA kanji inventory: {len(inventory)} chars", flush=True)
    else:
        inventory = set()  # hangul is unambiguous; the veto is ko_wording_ok

    pair_names: dict[str, list[str]] = {}
    pair_src: str | dict[tuple[str, str], str] = "tagpair"
    if args.lang == "ko" and not args.no_tag_pairs:
        # the KO analog of the tag-pair set: the KR KB's keyword field
        kr_kb = load_kr_kb(Path(args.kr_kb))
        pair_names = {t: kr_kb.get(t, [])[: args.n_pair_candidates] for t in todo}
        pair_src = "kb"
        n_with = sum(1 for v in pair_names.values() if v)
        print(f"  [glossary] KR-KB candidates for {n_with}/{len(todo)} tags")
    if args.lang == "zh" and not args.no_tag_pairs:
        # the zh analog: the community tag packs, curated (kb) + 10w (kbmt)
        zh_kb = load_zh_kb(args.zh_kb)
        pair_names = {
            t: [w for w, _ in zh_kb.get(t, [])][: args.n_pair_candidates] for t in todo
        }
        pair_src = {(t, w): s_ for t, ws in zh_kb.items() for w, s_ in ws}
        n_with = sum(1 for v in pair_names.values() if v)
        print(f"  [glossary] zh-pack candidates for {n_with}/{len(todo)} tags")
    # the HF tag-pair set is JA-only (no KO analog on HF, verified 2026-08-31)
    if args.lang == "ja" and not args.no_tag_pairs:
        import tag_pairs  # noqa: PLC0415  (sibling — the shared source + guards)

        print(f"  [glossary] fetching {tag_pairs.PAIR_REPO} …", flush=True)
        raw_pairs = tag_pairs.load_pairs(args.tag_pairs_file)
        pair_names = {
            t: tag_pairs.japanese_names(raw_pairs.get(t, []), inventory)[
                : args.n_pair_candidates
            ]
            for t in todo
        }
        n_with = sum(1 for v in pair_names.values() if v)
        print(f"  [glossary] tag-pair candidates for {n_with}/{len(todo)} tags")

    engine = MTEngine(args.model, greedy=True, gpu_budget=args.gpu_budget)

    print(
        f"  [glossary] EN→{args.lang.upper()} over {len(todo)} general tags",
        flush=True,
    )
    order = sorted(todo)
    mt_ja = dict(
        zip(
            order,
            engine.translate(
                [Request(text=t, background=background, terms=shots) for t in order],
                target_lang=args.lang,
                batch_size=args.batch_size,
                progress_every=20,
                max_new_tokens=args.max_new_tokens,
                cache=args.mt_cache / f"glossary_en2{args.lang}.jsonl",
            ),
        )
    )

    # Back-translate the candidates: wiki (primary + alts), tag-pair names, MT.
    pairs: list[tuple[str, str]] = []
    src_of: dict[tuple[str, str], str] = {}
    for tag in order:
        e = tags[tag]
        primary_src = "kb" if e["via"] == "kb" else "wiki"
        wiki_c = (
            [e["ja"]]
            if (e["via"].startswith("wiki") or e["via"] == "kb") and e["ja"]
            else []
        ) + e["alts"]
        sourced = [(c, primary_src) for c in wiki_c[: args.n_candidates]]
        sourced += [
            (c, pair_src if isinstance(pair_src, str) else pair_src.get((tag, c), "kb"))
            for c in pair_names.get(tag, [])
        ]
        if mt_ja.get(tag):  # the literal rendering competes on the same terms
            sourced.append((mt_ja[tag].strip().splitlines()[0].strip(), "mt"))
        seen: dict[str, str] = {}
        for cand, src in sourced:
            if cand and cand not in seen:
                seen[cand] = src
        pairs += [(tag, c) for c in seen]
        src_of |= {(tag, c): s for c, s in seen.items()}
    print(
        f"  [glossary] JA→EN back-translation over {len(pairs)} candidates", flush=True
    )
    back = engine.translate(
        [Request(text=c) for _, c in pairs],
        target_lang="en",
        batch_size=args.batch_size,
        progress_every=20,
        max_new_tokens=args.max_new_tokens,
        cache=args.mt_cache / f"glossary_{args.lang}2en.jsonl",
    )

    scored: dict[str, list[tuple[float, str, str, bool]]] = collections.defaultdict(
        list
    )
    for (tag, cand), en in zip(pairs, back):
        if args.lang == "ja":
            ja_ok = han_allowed(cand) and (
                KANA.search(cand) is not None
                or all(c in inventory for c in cand if HAN.match(c))
            )
        elif args.lang == "zh":
            # veto-only: latin candidates stay eligible; Han-bearing ones must
            # be Chinese by script mark or inventory (髪飾り / 棕毛 both settle)
            ja_ok = zh_wording_ok(cand) and (
                not HAN_WIDE.search(cand) or is_chinese(cand)
            )
        else:
            ja_ok = ko_wording_ok(cand)
        scored[tag].append(
            {
                "f1": round(_f1(tag, en), 3),
                "ja": cand,
                "back": en.strip(),
                "ja_ok": ja_ok,
                "kana": NATIVE_RE[args.lang].search(cand) is not None,
                "src": src_of[(tag, cand)],
                "mt": cand == (mt_ja.get(tag) or "").strip().splitlines()[:1][0]
                if mt_ja.get(tag)
                else False,
            }
        )

    for tag in order:
        mt = (mt_ja.get(tag) or "").strip().splitlines()
        choose(
            tag,
            tags[tag],
            scored.get(tag, []),
            mt[0].strip() if mt else "",
            args.accept_f1,
            args.lang,
        )


def kanji_review_rows(payload: dict, min_f1: float = 0.75) -> list[tuple]:
    """The D1-words axis: a katakana-only choice with a Han-only rival at
    tie-or-better F1.

    Back-translation cannot arbitrate these and the kana-first ranking must not
    be relaxed for them: the rival is either native Japanese wrongly rejected
    (鎧, 俯瞰, 接写) or Chinese that survived the character guards — and
    `bed` → 床 (*floor* in Japanese, *bed* in Chinese) shows only Japanese
    knowledge separates the two. Human review axis; fixes go to
    ``tag_overrides.json``.
    """
    rows = []
    for tag, e in payload["tags"].items():
        ja = e.get("ja")
        if e["axis"] != "general" or not ja or not KANA.search(ja) or HAN.search(ja):
            continue
        floor = max(float(e.get("f1") or 0.0), min_f1)
        rivals = [
            c
            for c in e.get("candidates") or []
            if HAN.search(c["ja"])
            and not KANA.search(c["ja"])
            and c.get("f1", 0) >= floor
        ]
        if rivals:
            rows.append((e["count"], tag, e, rivals[0]))
    rows.sort(reverse=True, key=lambda r: r[0])
    return rows


def write_review(payload: dict, path: Path, top: int) -> int:
    """Dump the highest-traffic MT-vs-wiki disagreements for human sign-off.

    Ordered by occurrence count, so reading down the list buys coverage as fast
    as possible: the top 200 rows here decide the wording of most of the corpus.
    Corrections go into ``tag_overrides.json`` (committed) and win over every
    automatic source on the next build.
    """
    rows = [
        (e["count"], t, e)
        for t, e in payload["tags"].items()
        if e["axis"] == "general"
        and e.get("mt_ja")
        and (
            e["via"] == "mt_unverified"
            or (e.get("candidates") and e["candidates"][0]["ja"] != e.get("mt_ja"))
            # MT round-tripped its own sense, but the community field disagrees
            # with real recovered meaning — the polysemy class the arbiter is
            # structurally blind to (`bow`: お辞儀 "bowing" verifies at 1.0
            # while 蝶結び the ribbon comes back "bow tie knot" at 0.5). Only a
            # human can pick the booru sense; surface it.
            or (
                e["via"] == "mt_verified"
                and any(
                    c.get("src") != "mt" and c["ja"] != e["ja"] and c["f1"] >= 0.3
                    for c in e.get("candidates") or []
                )
            )
        )
    ]
    rows.sort(reverse=True, key=lambda r: r[0])
    rows = rows[:top]

    lines = [
        "# Tag glossary — review the disagreements",
        "",
        "Rows where the Danbooru-wiki idiom and the MT rendering differ, most",
        "frequent first. **chosen** is what the corpus will use; `wiki_verified`",
        "means back-translation recovered the tag, `mt` means it did not.",
        "",
        'Fix anything wrong by adding `"<en tag>": "<ja>"` to',
        "`tag_overrides.json`, then rebuild — overrides beat every source.",
        "",
        "| n | tag | chosen | via | MT | wiki candidates (back-translation, F1) |",
        "|--:|---|---|---|---|---|",
    ]
    for count, tag, e in rows:
        cands = " · ".join(
            f"{c['ja']} ({c['back']}, {c['f1']}"
            + (f", {c['src']}" if c.get("src") and c["src"] != "wiki" else "")
            + ")"
            for c in e["candidates"][:3]
        )
        lines.append(
            f"| {count} | {tag} | **{e['ja']}** | {e['via']} | {e.get('mt_ja')} | {cands} |"
        )

    kanji = (
        kanji_review_rows(payload)[:top]
        if payload["meta"].get("lang", "ja") == "ja"
        else []
    )
    if kanji:
        lines += [
            "",
            "## Katakana primary vs native-kanji rival (D1-words)",
            "",
            "The chosen wording is katakana-only and a Han-only candidate ties",
            "or beats it on back-translation. The kana-first ranking is kept on",
            "purpose (`bed` → 床 is *floor* in Japanese, *bed* in Chinese — no",
            "guard separates them), so a kanji wording here only ships through",
            "an override. Judge each rival: native Japanese → override it in;",
            "Chinese → leave it.",
            "",
            "| n | tag | chosen | kanji rival (back-translation, F1, src) |",
            "|--:|---|---|---|",
        ]
        for count, tag, e, c in kanji:
            lines.append(
                f"| {count} | {tag} | **{e['ja']}** | {c['ja']} "
                f"({c['back']}, {c['f1']}, {c.get('src', 'wiki')}) |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows) + len(kanji)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--lang",
        default="ja",
        choices=["ja", "ko", "zh"],
        help="student-side language; entry keys stay `ja` ('student text' — "
        "plan_ko.md K1) so every consumer works unchanged; zh = plan_zh.md Z1 "
        "(simplified primary, OpenCC-routed wiki names, community packs)",
    )
    ap.add_argument("--caption-index", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--captions", type=Path, nargs="+", default=[DEFAULT_CAPTIONS])
    ap.add_argument(
        "--raw-captions",
        type=Path,
        nargs="*",
        default=[build_pairs.GELCRAWL_RETRIEVED],
        help="raw crawler roots, normalized before counting (pass empty to disable)",
    )
    ap.add_argument("--tag-rules", type=Path, default=build_pairs.GELCRAWL_RULES)
    ap.add_argument("--lexicon", type=Path, default=ASSETS / "wikidata_lexicon.json")
    ap.add_argument("--wiki", type=Path, default=ASSETS / ".wiki" / WIKI_FILE)
    ap.add_argument(
        "--kr-kb",
        type=Path,
        default=REPO / "models" / "danbooru_tags_classified.csv",
        help="KR community KB (--lang ko candidate source; make download-danbooru-tags)",
    )
    ap.add_argument(
        "--zh-kb",
        type=Path,
        nargs="*",
        default=ZH_KB_DEFAULT,
        help="zh community tag packs (--lang zh candidate source; assets/.zh/)",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-alts", type=int, default=4)
    ap.add_argument("--overrides", type=Path, default=ROOT / "tag_overrides.json")
    ap.add_argument("--review", type=Path, default=ASSETS / "tag_glossary_review.md")
    ap.add_argument("--review-top", type=int, default=200)
    ap.add_argument("--accept-f1", type=float, default=0.75)
    ap.add_argument("--n-candidates", type=int, default=3)
    ap.add_argument(
        "--no-tag-pairs",
        action="store_true",
        help="drop the tag-pair set from the --mt candidate pool (D1-pairs item 2)",
    )
    ap.add_argument(
        "--tag-pairs-file", type=Path, default=None, help="local parquet override"
    )
    ap.add_argument("--n-pair-candidates", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="MT only the top-N tags")
    ap.add_argument("--mt", action="store_true", help="translate the residue (GPU)")
    ap.add_argument(
        "--reselect",
        type=Path,
        default=None,
        help="re-derive choices from a previous build's stored candidates (no GPU)",
    )
    ap.add_argument("--model", default="tencent/Hy-MT2-7B")
    ap.add_argument("--gpu-budget", default="13GiB", help="offload past this")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-shots", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32, help="tags are short")
    ap.add_argument(
        "--mt-cache",
        type=Path,
        default=ASSETS / ".mtcache",
        help="per-batch translation cache — a killed run resumes from here",
    )
    args = ap.parse_args()
    if args.lang != "ja":  # per-language artifact defaults, JA names untouched
        if args.out == DEFAULT_OUT:
            args.out = ASSETS / f"tag_glossary_{args.lang}.json"
        if args.review == ASSETS / "tag_glossary_review.md":
            args.review = ASSETS / f"tag_glossary_review_{args.lang}.md"
        if args.overrides == ROOT / "tag_overrides.json":
            args.overrides = ROOT / f"tag_overrides_{args.lang}.json"

    payload = build(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    n_review = write_review(payload, args.review, args.review_top)

    m = payload["meta"]
    print(f"\nwrote {args.out}")
    print(f"wrote {args.review} ({n_review} rows to review)")
    print(f"  tags {m['n_tags']} / occurrences {m['n_occurrences']}")
    print(f"  occurrence coverage: {m['occurrence_coverage']}%")
    print(f"  unique CJK codepoints: {m['unique_cjk_codepoints']}")
    for via, n in sorted(m["types_by_via"].items(), key=lambda kv: -kv[1]):
        occ = m["occurrences_by_via"][via]
        print(
            f"    {via:12s} {n:5d} types  {occ:7d} occ ({100 * occ / m['n_occurrences']:.1f}%)"
        )


if __name__ == "__main__":
    main()
