"""Invariants for the CJK Phase 2a glossary/corpus builders.

The subtle part of `project/cjk_aware_anima/datasets/` is deciding *which
string is Japanese* and *which alternate wording is safe to swap in* — both
silently corrupt the corpus when wrong, and neither shows up as a crash. These
guard the two rules that were established by measurement:

* script detection has to reject Chinese in both directions (simplified via
  Shift-JIS, traditional via a JA-kanji inventory), because a Chinese string
  scored a *perfect* back-translation and would otherwise have been chosen;
* the alternate-wording pool for general tags must come from verified
  candidates, never from the raw wiki list, which carries 女スパイ under
  ``1girl``.
"""

from __future__ import annotations

import collections
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

DATASETS = (
    Path(__file__).resolve().parents[1] / "project" / "cjk_aware_anima" / "datasets"
)


def _load(name: str):
    """Import a `datasets/` script by path — they are entry points, not a package."""
    sys.path.insert(0, str(DATASETS))
    spec = importlib.util.spec_from_file_location(name, DATASETS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve types via sys.modules
    spec.loader.exec_module(module)
    return module


tag_glossary = _load("tag_glossary")
build_pairs = _load("build_pairs")
mt = _load("mt")


def test_translation_cache_survives_a_killed_job(tmp_path):
    """A pass is hours of GPU, so results are appended per batch, not held.

    The last line of a killed job's cache can be torn mid-write; that must cost
    one entry, not the whole file.
    """
    engine = object.__new__(mt.MTEngine)
    engine.model_path, engine.greedy, engine.seed = "m", True, 0
    key = engine._cache_key("hello", "ja", 32)

    cache = tmp_path / "c.jsonl"
    cache.write_text(
        json.dumps({"key": key, "out": "こんにちは"}) + '\n{"torn line\n',
        encoding="utf-8",
    )
    loaded = mt.load_cache(cache)
    assert loaded == {key: "こんにちは"}


def test_cache_key_separates_runs_that_would_differ():
    engine = object.__new__(mt.MTEngine)
    engine.model_path, engine.greedy, engine.seed = "m", True, 0
    base = engine._cache_key("hello", "ja", 32)
    assert engine._cache_key("hello", "ja", 32) == base  # stable
    assert engine._cache_key("hello", "en", 32) != base  # target language
    assert engine._cache_key("other", "ja", 32) != base  # prompt
    engine.greedy = False
    assert engine._cache_key("hello", "ja", 32) != base  # decode mode


@pytest.mark.parametrize(
    "text",
    ["ツインテール", "カメラ目線", "割座", "火宮チナツ", "へそ", "赤面"],
)
def test_japanese_accepted(text):
    assert tag_glossary.is_japanese(text)


@pytest.mark.parametrize(
    "text", ["小鸟游星野", "双马尾", "多色发", "阿黑颜", "看向观众"]
)
def test_simplified_chinese_rejected(text):
    """Shift-JIS cannot encode simplified forms — the zero-dependency filter."""
    assert not tag_glossary.is_japanese(text)


def test_ja_kanji_inventory_rejects_traditional_chinese(tmp_path):
    """Traditional Chinese survives Shift-JIS, so the inventory has to catch it.

    棕毛 and 藍眼睛 both encode fine *and* back-translated perfectly ("brown
    hair" / "blue eyes"); only the fact that their characters never appear in
    kana-bearing Japanese entries separates them from 茶髪.
    """
    dump = tmp_path / "wiki.jsonl"
    dump.write_text(
        "\n".join(
            [
                '{"other_names": ["茶髪ロング", "黒髪ボブ", "目線カメラ"]}',
                '{"other_names": ["髪型のこと", "茶色い髪の毛"]}',
                '{"other_names": ["棕毛", "藍眼睛"]}',
            ]
        ),
        encoding="utf-8",
    )
    inventory = tag_glossary.ja_kanji_inventory(dump, min_count=1)
    assert "髪" in inventory  # seen inside kana-bearing entries
    assert "棕" not in inventory  # only ever seen in a Han-only entry
    assert "睛" not in inventory


def test_back_translation_f1_prefers_exact_recovery():
    assert tag_glossary._f1("blush", "Blush") == pytest.approx(1.0)
    assert tag_glossary._f1("blue eyes", "Blue eyes") == pytest.approx(1.0)
    # extra content is penalised — this is what demotes 黒髪ボブ under `black hair`
    assert tag_glossary._f1("black hair", "Black Bob") < 0.75
    assert tag_glossary._f1("solo", "Alone") == 0.0


def test_alt_pool_uses_verified_candidates_for_general_tags():
    entry = {
        "axis": "general",
        "ja": "貧乳",
        "alts": ["女スパイ", "ガキ巨乳"],  # raw wiki junk, must not be swapped in
        "candidates": [
            {"ja": "貧乳", "f1": 1.0, "kana": False},  # the chosen primary
            {"ja": "ちっぱい", "f1": 1.0, "kana": True},
            {"ja": "極上の貧乳", "f1": 0.4, "kana": False},  # below the gate
        ],
    }
    pool = build_pairs.alt_pool(entry, min_f1=0.75)
    assert pool == ["貧乳", "ちっぱい"]


def test_alt_pool_drops_verified_chinese():
    """Verified is not proof of Japanese — 汗液/短袖 back-translate perfectly.

    They were being swapped into the `tags_alt` register, so the pool demands
    kana unless the candidate is the primary wording itself.
    """
    entry = {
        "axis": "general",
        "ja": "汗",
        "candidates": [
            {"ja": "汗", "f1": 1.0, "kana": False},
            {"ja": "汗水", "f1": 1.0, "kana": False},
            {"ja": "汗液", "f1": 1.0, "kana": False},
        ],
    }
    assert build_pairs.alt_pool(entry, min_f1=0.75) == ["汗"]


def test_alt_pool_keeps_name_variants_for_proper_nouns():
    """Character nicknames are real alternates users type — no F1 gate there."""
    entry = {"axis": "character", "alts": ["ホシノ", "暁のホルス"]}
    assert build_pairs.alt_pool(entry, min_f1=0.75) == ["ホシノ", "暁のホルス"]


def test_compose_passes_unmapped_segments_through_and_counts_them():
    glossary = {
        "1girl": {"axis": "general", "ja": "女の子1人", "candidates": []},
        "@aak": {"axis": "artist", "ja": "@aak"},
    }
    ja, missing, spans = build_pairs.compose(
        ["1girl", "@aak", "unmapped tag"],
        glossary,
        alt=False,
        rng=random.Random(0),
        min_f1=0.75,
    )
    assert ja == ["女の子1人", "@aak", "unmapped tag"]
    assert missing == ["unmapped tag"]
    # The EN↔JA span alignment the distillation side consumes: one entry per
    # segment, in order, carrying the provenance of the wording actually used.
    assert [s["en"] for s in spans] == ["1girl", "@aak", "unmapped tag"]
    assert [s["ja"] for s in spans] == ja
    assert spans[-1]["via"] == "unmapped"


def test_names_register_swaps_only_names_and_skips_nameless_captions():
    """The name-swap contract: character/copyright go JA, everything else — the
    general tags *and* an unswappable name — stays EN with `via: en_pinned`,
    and a caption with no resolvable name emits no pair (it would visit no ext
    rows). The student joins with the record's ``joiner`` (``", "`` when no rng
    is given) and stores it, because the distill side's `_ja_span_chars`
    derives span offsets from that field.
    """
    glossary = {
        "acheron (honkai: star rail)": {"axis": "character", "ja": "黄泉"},
        "honkai: star rail": {"axis": "copyright", "ja": "崩壊：スターレイル"},
        "1girl": {"axis": "general", "ja": "女の子1人"},
        "no-ja name": {"axis": "character", "ja": None},
    }
    caption = "1girl, acheron (honkai: star rail), honkai: star rail, no-ja name"
    pairs = build_pairs.build_names([("img", caption)], glossary)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["register"] == "names"
    assert p["en"] == caption
    assert p["joiner"] == ", "
    assert p["ja"] == "1girl, 黄泉, 崩壊：スターレイル, no-ja name"
    assert p["n_missing"] == 1  # the name the glossary could not swap
    assert [s["via"] for s in p["spans"]] == [
        "en_pinned",
        "unknown",  # glossary entry carries no `via` in this fixture
        "unknown",
        "en_pinned",
    ]
    # `en_pinned` must never inherit default trust silently.
    from scripts.distill_cjk.config import TRUST_POLICIES

    assert all("en_pinned" in pol for name, pol in TRUST_POLICIES.items() if pol)

    assert build_pairs.build_names([("img2", "1girl, solo")], glossary) == []


def test_joiner_is_recorded_and_span_offsets_follow_it():
    """`, ` is primary (native T5 piece shared with the teacher), 、 stays a
    minority variant so its ext row keeps visits; the distill side must read
    the joiner off the record rather than assume either."""
    from scripts.distill_cjk.data import _ja_span_chars

    rng = random.Random(0)
    picks = collections.Counter(build_pairs.pick_joiner(rng) for _ in range(2000))
    assert set(picks) == {", ", "、"}
    assert 0.1 < picks["、"] / 2000 < 0.3
    assert build_pairs.pick_joiner(None) == ", "

    segs = ["女の子1人", "黒髪", "笑顔"]
    for joiner in (", ", "、"):
        text = joiner.join(segs)
        for seg, (lo, hi) in zip(segs, _ja_span_chars(segs, joiner=joiner)):
            assert text[lo:hi] == seg


def test_axis_falls_back_to_the_wiki_category_for_tags_outside_the_index():
    """The caption index only classifies tags of `image_dataset`; a D1-wide
    character it never saw must still be a name (never MT-translated), and the
    index wins where both know the tag."""
    axis_of = {"kanna (swimsuit) (blue archive)": "character", "arknights": "copyright"}
    wiki_axes = {
        "grani (arknights)": "character",
        "arknights": "general_would_be_wrong",
    }
    assert tag_glossary.resolve_axis("grani (arknights)", axis_of, wiki_axes) == (
        "character",
        "wiki",
    )
    assert tag_glossary.resolve_axis("arknights", axis_of, wiki_axes) == (
        "copyright",
        "index",
    )
    assert tag_glossary.resolve_axis("@pepper0", axis_of, wiki_axes) == (
        "artist",
        "default",
    )
    artists = frozenset({"@pepper0"})
    assert tag_glossary.resolve_axis(
        "akiyama fumika (pepper0)", axis_of, wiki_axes, artists
    ) == ("character", "artist_oc")
    assert tag_glossary.resolve_axis(
        "shrug (clothing)", axis_of, wiki_axes, artists
    ) == ("general", "default")
    assert tag_glossary.resolve_axis("smile", axis_of, wiki_axes) == (
        "general",
        "default",
    )


tag_pairs = _load("tag_pairs")


def test_tag_pairs_fills_only_what_the_glossary_left_unresolved():
    """The fill contract: a resolved wording is never re-opened on a CPU pass.

    Every wording the glossary already carries is either pinned or chosen by
    back-translation, and this source has neither behind it — overwriting one
    would reproduce the CPU-rebuild regression `datasets/README.md` records. An
    *unresolved* tag has nothing to lose: it composes as latin passthrough at
    span weight 0.
    """
    tags = {
        "1girl": {"count": 9, "axis": "general", "ja": "女の子", "via": "mt_verified"},
        "kizuato": {"count": 5, "axis": "general", "ja": None, "via": "unresolved"},
    }
    counts = {"1girl": 9, "kizuato": 5, "socks": 3, "@aak": 7}
    pairs = {
        "1girl": ["女子"],  # a competing wording — must be ignored
        "kizuato": ["痕"],
        "socks": ["靴下"],
        "@aak": ["アーク"],  # artist handles stay latin identity
    }
    inventory = set("痕靴下")
    stats = tag_pairs.fill(
        tags,
        __import__("collections").Counter(counts),
        pairs,
        inventory,
        {"@aak": "artist"},
    )

    assert tags["1girl"]["ja"] == "女の子"  # untouched
    assert tags["1girl"]["via"] == "mt_verified"
    assert tags["kizuato"]["ja"] == "痕"  # unresolved → filled
    assert tags["kizuato"]["via"] == tag_pairs.TAGPAIR_VIA
    assert tags["socks"]["ja"] == "靴下"  # absent → added
    assert "@aak" not in tags
    assert stats["filled"] == 3 - 1  # kizuato + socks


@pytest.mark.parametrize(
    "name",
    [
        "单色调",  # simplified Chinese — Shift-JIS rejects it
        "百褶裙",  # traditional Chinese — Shift-JIS accepts, the inventory does not
        "ウィンクX東方",  # latin contamination in the field
    ],
)
def test_tag_pairs_guards_reject_non_japanese(name):
    """The source is unfiltered by its own admission, so the guards re-run here."""
    inventory = set("東方髪目")
    assert tag_pairs.japanese_names([name], inventory) == []


def test_tagpair_via_has_an_explicit_trust_weight():
    """`apply_trust` defaults an unknown `via` to 1.0 — a new source must not
    inherit full trust by omission."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.distill_cjk.config import TRUST_POLICIES

    for policy, weights in TRUST_POLICIES.items():
        if policy == "all":
            continue  # `all` is empty by construction — every span at 1.0
        assert tag_pairs.TAGPAIR_VIA in weights, policy
        assert tag_glossary.TAGPAIR_VERIFIED_VIA in weights, policy
    assert (
        weights["mt_unverified"] <= TRUST_POLICIES["provenance"][tag_pairs.TAGPAIR_VIA]
    )


def test_choose_prefers_community_sense_over_mt_at_equal_f1():
    """`bow` — お辞儀 (MT, *bowing*) and 蝶結び (community, the ribbon) both
    back-translate to "bow", so F1 cannot separate the senses. Only provenance
    knows which one the booru tag means; at equal evidence the community field
    must beat the MT rendering (D1-pairs item 2)."""
    entry = {"count": 1582, "axis": "general", "alts": []}
    cands = [
        {
            "ja": "蝶結び",
            "back": "bow",
            "f1": 1.0,
            "kana": True,
            "mt": False,
            "src": "tagpair",
            "ja_ok": True,
        },
        {
            "ja": "お辞儀",
            "back": "bow",
            "f1": 1.0,
            "kana": True,
            "mt": True,
            "src": "mt",
            "ja_ok": True,
        },
    ]
    tag_glossary.choose("bow", entry, cands, "お辞儀", 0.75)
    assert entry["ja"] == "蝶結び"
    assert entry["via"] == tag_glossary.TAGPAIR_VERIFIED_VIA
    assert entry["candidates"][0]["src"] == "tagpair"  # provenance persisted


def test_choose_keeps_kana_over_kanji_and_surfaces_the_rival():
    """鎧 back-translates perfectly, but kana stays the proof of Japaneseness —
    a Han-only rival can be Chinese (`bed` → 床 is *floor* in JA), so it never
    wins a tie automatically. It must land in the D1-words review section
    instead, where a human can override it in."""
    entry = {"count": 39, "axis": "general", "alts": []}
    cands = [
        {
            "ja": "アーマー",
            "back": "armor",
            "f1": 1.0,
            "kana": True,
            "mt": False,
            "src": "wiki",
            "ja_ok": True,
        },
        {
            "ja": "鎧",
            "back": "armor",
            "f1": 1.0,
            "kana": False,
            "mt": False,
            "src": "tagpair",
            "ja_ok": True,
        },
    ]
    tag_glossary.choose("armor", entry, cands, "", 0.75)
    assert entry["ja"] == "アーマー"
    assert entry["via"] == "wiki_verified"

    rows = tag_glossary.kanji_review_rows({"tags": {"armor": entry}})
    assert [(r[1], r[3]["ja"]) for r in rows] == [("armor", "鎧")]


def _cand(ja, src, f1=0.0, back=""):
    return {"ja": ja, "back": back, "f1": f1, "kana": True, "src": src, "ja_ok": True}


@pytest.mark.parametrize(
    "tag, cand, src",
    [
        # (a) the echoed few-shot prompt, with the truncation glyph
        (
            ":t",
            ":女の子1人の全身写真、シンプルな背景、肩出し、制服、ニーハイ、金\ufffd",
            "mt",
        ),
        (
            "\\m/",
            "女の子1人の全身写真、シンプルな背景、肩出し、制服、ニーハイ、金髪",
            "mt",
        ),
        # (b) an exemplar word the tag never licensed
        ("pantyhose only", "ニーハイのみ", "mt"),
        ("backless panties", "肩出しパンティー", "mt"),
        ("holding shorts", "ショートヘアを持つ", "mt"),
        ("chick", "女の子1人", "mt"),
        ("jojifuku", "制服", "mt"),
        # (c) fujoshi-community titles from the wiki field
        ("re:zero kara hajimeru isekai seikatsu", "腐ゼロ", "wiki"),
        ("fire emblem", "FE腐向け", "wiki"),
        ("bungou stray dogs", "文スト【腐】", "wiki"),
        ("fire emblem", "ガチホモエムブレム", "wiki"),
        ("granblue fantasy", "グラ腐ル", "wiki"),
        ("genshin impact", "原神BL", "wiki"),
    ],
)
def test_contaminated_rejects_prompt_echo_exemplar_leak_and_fujoshi(tag, cand, src):
    assert tag_glossary.contaminated(cand, tag, src)


@pytest.mark.parametrize(
    "tag, cand, src",
    [
        ("full-length mirror", "全身鏡", "tagpair"),  # 全身 licensed by full-length
        ("full-length mirror", "全身鏡", "mt"),
        ("1girl", "女の子1人", "mt"),
        ("2girls", "女の子2人", "mt"),
        ("school uniform", "制服", "mt"),
        ("tantei wa mou shindeiru", "探偵はもう、死んでいる", "wiki"),  # real title
        ("yaoi", "腐向け", "wiki"),  # BL tag may keep its own register
        ("yaoi", "ゲイ向け", "wiki"),
        ("tofu", "豆腐", "mt"),
        ("black hair", "黒髪", "mt"),  # `bl` is word-bounded
        ("bleach", "BLEACH", "wiki"),
        ("bow", "お辞儀", "mt"),
    ],
)
def test_contaminated_keeps_licensed_and_community_wordings(tag, cand, src):
    assert not tag_glossary.contaminated(cand, tag, src)


def test_choose_never_keeps_a_contaminated_mt_rendering():
    """`chick`: MT echoed 女の子1人 and it used to ship as `mt_unverified`
    because the wiki rivals (ひよこ) scored F1 0. The veto must fall through to
    the next candidate — and record what it dropped — not invent anything."""
    entry = {"count": 3, "axis": "general", "alts": []}
    cands = [_cand("ひよこ", "wiki", back="chick"), _cand("女の子1人", "mt")]
    tag_glossary.choose("chick", entry, cands, "女の子1人", 0.75)
    assert entry["ja"] == "ひよこ"
    assert entry["via"] == "wiki_unverified"
    assert entry["mt_ja"] == "女の子1人"  # the record survives
    assert entry["rejected_contaminated"] == ["女の子1人"]
    assert all(c["ja"] != "女の子1人" for c in entry["candidates"])

    only = {"count": 2, "axis": "general", "alts": []}
    tag_glossary.choose(
        "\\m/",
        only,
        [_cand("女の子1人の全身写真、肩出し", "mt")],
        "女の子1人の全身写真、肩出し",
        0.75,
    )
    assert only["ja"] is None and only["via"] == "unresolved"


@pytest.mark.parametrize(
    "tag",
    [
        ":d",
        ":t",
        ";)",
        ":<=",
        "^^^",
        "\\m/",
        ">_<",
        "\\||/",
        "...",
        "!?",
        "??",
        "!",
        "...!",
        "c:",
        "3:",
        "^o^",
        "@ @",
        "+ +",
    ],
)
def test_symbol_tags_detected(tag):
    assert tag_glossary.is_symbol_tag(tag)


@pytest.mark.parametrize(
    "tag", ["1girl", "3d", "c.c.", "em-2", "smile", "blue eyes", "@aak", "xd", "e.t."]
)
def test_word_tags_are_not_symbols(tag):
    assert not tag_glossary.is_symbol_tag(tag)


def test_tag_counts_follow_the_clause_grammar(tmp_path):
    """A comma split minted `white socks. On the left` and bare `On the left`
    as tags (60 of them in the live glossary); the grammar counts the bag plus
    each clause's own tags, header excluded, and a clause-free caption is
    unchanged."""
    root = tmp_path / "artist"
    root.mkdir()
    (root / "a.txt").write_text(
        "safe, 2girls, white socks. On the left, akita neru, yellow eyes. "
        "On the right, kasane teto.",
        encoding="utf-8",
    )
    (root / "b.txt").write_text("safe, 1girl, :d, smile, white socks", encoding="utf-8")
    counts = tag_glossary.tag_counts([(tmp_path, False)], tmp_path / "no_rules.yaml")
    assert counts == collections.Counter(
        {
            "safe": 2,
            "2girls": 1,
            "white socks": 2,
            "akita neru": 1,
            "yellow eyes": 1,
            "kasane teto": 1,
            "1girl": 1,
            ":d": 1,
            "smile": 1,
        }
    )
    assert not any("On the" in t or t.endswith(".") for t in counts)


def test_tag_counts_restores_the_dot_of_period_final_tag_names(tmp_path):
    """`c.c.` / `nanashi inc.` end in a period by name; caption-final they look
    like a terminated `unworn panties.` to the grammar. Known titles disambiguate."""
    root = tmp_path / "artist"
    root.mkdir()
    (root / "a.txt").write_text("safe, code geass, c.c.", encoding="utf-8")
    (root / "b.txt").write_text(
        "safe, c.c., code geass, unworn panties.", encoding="utf-8"
    )
    counts = tag_glossary.tag_counts(
        [(tmp_path, False)], tmp_path / "no_rules.yaml", known={"c.c.", "code geass"}
    )
    assert counts["c.c."] == 2 and "c.c" not in counts
    assert counts["unworn panties"] == 1 and "unworn panties." not in counts


# ---- zh (plan_zh.md Z1) ------------------------------------------------------


@pytest.mark.parametrize(
    "text, cls",
    [
        ("发", "hans"),  # canonical simplified: t2s(s2t) round-trips
        ("国", "hans"),  # shinjitai == simplified: still hans
        ("髮", "variant"),  # OpenCC-touched but no round trip
        ("國", "hant"),
        ("対", "shared"),  # JA shinjitai OpenCC never touches
        ("双马尾", "hans"),
        ("雙馬尾", "hant"),
        ("制服", "shared"),
        ("ツインテール", ""),
    ],
)
def test_han_class_round_trip_rule(text, cls):
    assert tag_glossary.han_class(text) == cls


def test_is_chinese_uses_the_inventory_for_shared_wordings():
    inv = set("制服巨乳棕毛")
    assert tag_glossary.is_chinese("双马尾", inv)  # script-marked: no inventory needed
    assert tag_glossary.is_chinese("藍眼睛", inv)  # hant is Chinese too
    assert tag_glossary.is_chinese("制服", inv)  # shared, every char attested
    assert not tag_glossary.is_chinese("泥酔", inv)  # shared, 酔 never seen in zh
    assert not tag_glossary.is_chinese("髪飾り", inv)  # kana veto
    assert not tag_glossary.is_chinese("트윈테일", inv)  # hangul veto
    assert not tag_glossary.is_chinese(":d", inv)  # no Han at all
    assert tag_glossary.zh_wording_ok(":d")  # …but the veto-only guard passes it


def test_rank_names_zh_prefers_hans_and_simplifies_hant():
    ranked = tag_glossary.rank_names(
        ["ツインテール", "雙馬尾", "双马尾", "트윈테일", "二つ結い"], "zh"
    )
    assert ranked == ["双马尾"]  # kana/hangul out; 雙馬尾 folds into 双马尾


def test_zh_han_inventory_comes_from_the_packs_not_mixed_wiki_entries(tmp_path):
    pack = tmp_path / "pack.csv"
    pack.write_text("twintails,0,双马尾|双尾,\nschool uniform,校服\n", encoding="utf-8")
    dump = tmp_path / "wiki.jsonl"
    dump.write_text('{"other_names": ["左右対称"]}\n', encoding="utf-8")
    inv = tag_glossary.zh_han_inventory(dump, [pack])
    assert {"双", "马", "尾", "校", "服"} <= inv
    assert "対" not in inv  # a single mixed wiki entry cannot seed the census


def test_load_zh_kb_reads_every_pack_layout_and_ranks_curated_first(tmp_path):
    nga = tmp_path / "HalfMAI_nga.csv"
    nga.write_text(
        "39,0,初音未来,\nlooking_at_viewer,0,看向阅图者|看着你,\n", encoding="utf-8"
    )
    byzod = tmp_path / "byzod.csv"
    byzod.write_text("looking_at_viewer,看着观众\nbed,床\n", encoding="utf-8")
    tenw = tmp_path / "danbooru-10w-zh_cn.csv"
    tenw.write_text(
        "looking_at_viewer,pov 目光接触\nshort_hair,短毛短毛猫\n", encoding="utf-8"
    )
    kb = tag_glossary.load_zh_kb([nga, byzod, tenw])
    assert kb["39"] == [("初音未来", "kb")]
    assert kb["looking at viewer"] == [
        ("看向阅图者", "kb"),
        ("看着你", "kb"),
        ("看着观众", "kb"),
        ("目光接触", "kbmt"),
    ]
    assert kb["short hair"] == [("短毛短毛猫", "kbmt")]  # pool only, ranked with MT
    assert kb["bed"] == [("床", "kb")]


def test_choose_zh_lets_a_verified_pack_wording_beat_a_higher_f1_mt_rendering():
    entry = {"count": 500, "axis": "general", "alts": []}
    cands = [
        {
            "ja": "看着观众",
            "back": "looking at the viewer",
            "f1": 1.0,
            "src": "mt",
            "mt": True,
            "ja_ok": True,
        },
        {
            "ja": "看向阅图者",
            "back": "looking at viewer",
            "f1": 1.0,
            "src": "kb",
            "ja_ok": True,
        },
    ]
    tag_glossary.choose("looking at viewer", entry, cands, "看着观众", 0.75, "zh")
    assert entry["ja"] == "看向阅图者" and entry["via"] == "kb_verified"


def test_build_pairs_adds_a_hant_sibling_register_for_zh():
    pairs = [
        {
            "id": "D1/x/tags_zh",
            "register": "tags_zh",
            "ja": "1个女性, 双马尾, 校服",
            "spans": [{"en": "twintails", "ja": "双马尾", "via": "kb"}],
        },
        {"id": "D1/x/tags_alt_zh", "register": "tags_alt_zh", "ja": "…", "spans": []},
    ]
    hant = build_pairs.add_hant_register(pairs)
    assert len(hant) == 1
    assert hant[0]["register"] == "tags_zh_hant"
    assert hant[0]["id"] == "D1/x/tags_zh_hant"
    assert hant[0]["ja"] == "1個女性, 雙馬尾, 校服"
    assert hant[0]["spans"][0]["ja"] == "雙馬尾"
    assert pairs[0]["ja"] == "1个女性, 双马尾, 校服"  # source untouched
