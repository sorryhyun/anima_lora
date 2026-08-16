# `datasets/` — corpus builders for the CJK distillation

Data-side tooling for Phase 2a of [`../plan.md`](../plan.md). This file is the
home of the measured results for everything below; [`../done.md`](../done.md)
only lists what is finished.
Everything writes into `assets/` (gitignored — regenerate rather than commit)
except `tag_overrides.json`, which is hand-authored and committed. Run from the
repo root. The two builders that need a GPU (`mt.py`, and the `--mt` passes)
must go through the daemon — `make daemon-run ARGS="<script> …"` — never a
background shell.

The distillation teacher is computed at **embedding level** from an EN string,
so nothing here needs human-quality parallel text; what it does need is *JA that
is right*, in the *registers users actually prompt in*, hitting *ext rows the
corpus will really visit*.

| Script | Source | Licence | Output |
|---|---|---|---|
| `wikidata_lexicon.py` | Wikidata SPARQL, entity list from our `caption_index.json` | CC0 | `assets/wikidata_lexicon.json` — EN↔JA(↔KO/ZH) proper nouns |
| `lovehina.py` | [PLippmann/multimodal-manga-translation](https://github.com/PLippmann/multimodal-manga-translation) (COLING 2025, arXiv:2411.02589) | MIT | `assets/lovehina_ja.json` — 3,705 native JA speech-bubble lines |
| `mt.py` | `tencent/Hy-MT2-1.8B` / `-7B` | Apache-2.0 | library (`MTEngine`) + `--probe` / `--smoke` self-checks |
| `commentary.py` | Danbooru `artist_commentaries.json` (gelcrawl route) | site ToS | `assets/commentary_ja.jsonl` (73,015 native-JA records) → `--mt` → `commentary_pairs*.jsonl` (D2 pairs) |
| `tag_glossary.py` | Danbooru wiki dump + the two sources above + MT | WTFPL (dump) | `assets/tag_glossary_ja.json`, `assets/tag_glossary_review.md` |
| `build_pairs.py` | the glossary | — | `post_image_dataset/cjk_distill/{pairs.jsonl,coverage.json,spotcheck.md}` |

## Why MT is the *last* source, not the first

The obvious plan — "translate the captions with a local MT model" — fails on
register, and register is not cosmetic here: ext rows only train where the
corpus visits them, so JA that says 二本の髪 where users type ツインテール
trains the wrong rows. That is the proper-noun distribution argument
(below) applied to the whole general vocabulary.

Measured on a held-out 28-tag idiom probe (`mt.py --probe`, greedy):

| arm | hit rate |
|---|---|
| Hy-MT2-1.8B, plain prompt | 8/28 (29%) |
| Hy-MT2-1.8B + 16 few-shot exemplars | 9/28 (32%) |
| Hy-MT2-7B, plain prompt | 13/28 (46%) |
| Hy-MT2-7B + few-shot | 15/28 (54%) |

Prompting barely moves it and size only halves the gap: these are *knowledge*
misses (ツインテール, カメラ目線, 割座, セーラー服), not instruction-following
misses. **Do not re-propose fixing tag register with a better prompt or a
bigger MT model** — the fix is a source that already knows the community's
words, i.e. the Danbooru wiki.

The 7B needs `--gpu-budget 13GiB` to fit a 16 GB card (bf16 weights are 15.2 GB
against ~15.5 GB usable); accelerate then keeps the overflow blocks on CPU and
streams them, costing ~4.2 gen/s. Keep `--max-new-tokens` tight: a batch runs
until every sequence finishes, so the 512 default made tag translation ~14×
slower than necessary.

## `tag_glossary.py` — the EN→JA tag table

Sources in priority order, per tag: `tag_overrides.json` (hand fixes) →
artist passthrough (pixiv handles stay latin — users type them latin) → rating
band → Wikidata lexicon → Danbooru wiki `other_names` → MT.

`other_names` mixes languages, so script detection matters twice over. Kana
proves Japanese. Han-only strings are kept only if Shift-JIS can encode them,
which rejects simplified Chinese (小鸟游星野 out, 火宮チナツ in) — but *not*
traditional Chinese (棕毛, 藍眼睛 both encode fine and both scored a perfect
back-translation), so Han-only candidates are additionally checked against a
JA-kanji inventory learned from the dump itself: every `other_names` entry
containing kana is Japanese by construction, so the kanji inside those entries
are a free census of the JA repertoire.

The wiki is authoritative on idiom but *not* a synonym list — it also carries
narrower compounds (`underwear` → 下着コート) and different concepts
(`multiple girls` → 群像). So every candidate, **including the MT rendering**,
is back-translated to English and scored (token F1) against the original tag;
the best scorer above `--accept-f1` wins, and ties break toward brevity so
`黒髪ロング` loses to `黒髪`. Exemplars for the MT prompt come from a
hand-checked list in `mt.py`, never from the unverified wiki head — drawing
them automatically fed 下着コート back into the prompts and MT reproduced it.

Disagreements between the two sources land in `assets/tag_glossary_review.md`,
ordered by occurrence count, for the sign-off the phase gate asks for. Fixes go
in `tag_overrides.json` and beat every automatic source on the next build.

## `build_pairs.py` — the corpus

Captions are tag strings, so the JA side is **composed from the glossary**
rather than translated wholesale: idiom and pinned proper nouns are then true
by construction instead of by hope. Registers: `tags` (primary wording),
`tags_alt` (verified synonyms swapped in — unverified wiki alternates would
inject 女スパイ for `1girl`), `natural` (`--mt`, prose rewrite with
character/copyright terms pinned), plus D6 quoted-text templates seeded from
the LoveHina lines.

Coverage is reported in **ext rows**, encoded with the same
`HybridT5Encoder` training will use — the histogram is what says which rows a
run can actually move, and never-visited rows are the ones the plan says to
flag rather than ship as noise.

## Network note

A plain request to `danbooru.donmai.us` from this machine is reset at the
network level, which is why the wiki arrives via the HF mirror. That is **not**
a hard block: the gelcrawl route (`curl_cffi` Chrome impersonation through
SpoofDPI on `127.0.0.1:8080`, gelcrawl's `.env` credentials) reaches the API
fine, and `commentary.py` uses it for D2. No HF metadata dump is needed.

## `commentary.py` — D2, native JA at volume

Two halves. The **crawl** runs under gelcrawl's interpreter (that is where
`curl_cffi` lives) and appends every page to `assets/.commentary/raw.jsonl`, so
a kill costs one page. The **`--mt` pass** is GPU work under this repo's
interpreter and goes through the daemon; `--from-cache` rebuilds the pair file
from whatever a stopped pass already translated, with no model load at all.

Both halves write JSONL — a `{"stats": …}` header line, then one record per
line (`write_records` / `read_records`). At 73k records a single-line JSON blob
is unopenable and unstreamable.

Two things the MT pass does that a plain "translate every row" would not:
**names are pinned JA→EN** from the Wikidata lexicon (D5's argument backwards —
if 黄泉 comes back "Yomi" the teacher embedding lands on an entity the model
does not know), and **batches are length-bucketed**, because a batch runs until
every sequence in it finishes and 68% of records are ≤ 64 chars. Bucket batch
size scales down with length: batch 32 in the ≤128-char bucket OOMs the 7B.

Model choice was measured, not assumed — 64 records, 1.8B vs 7B. The 7B is
~4× the wall clock and wins on exactly the axis that matters, entity names
(十時愛梨 "Jūji Aira" → **"Toji Airi"**; 虹ヶ咲 "Neko no Hikari" → **"Nijigasaki"**;
澪ちゃん "Sori-chan" → **"Mio-chan"**). Note this is the *opposite* direction from
the tag-glossary finding above: JA→EN prose is a translation task, where model
size pays; EN→JA tag idiom is a knowledge lookup, where it does not.

## `wikidata_lexicon.py` — proper nouns

MT transliterates names, and our captions are made of names: unassisted,
`acheron (honkai: star rail)` comes out アケロン instead of **黄泉**,
`silver wolf` シルバーウルフ instead of **銀狼**.

This does **not** corrupt the training pairs — teacher and student share the
Qwen side and the teacher's T5 side carries the original EN caption, so
アケロン→Acheron is a self-consistent thing to learn. The cost is
*distribution*: a JA user types 黄泉, whose ext rows the corpus then never
visited, so the failure lands on the token that identifies the character. Plus
coverage — name kanji go unvisited entirely. Applied as a **pre-substitution on
the EN side with the JA target pinned**.

Entity list is ours, not the world's — `caption_index.json` (`make caption-index`)
already types every caption tag, so we only query names that actually occur in
training data (1,314 character / 150 copyright / 89 artist tags over 3,008 images).

**Two routes, because names come in two shapes.** Qualified names
(`aru (blue archive)`) go through the **franchise roster**: resolve the
copyright tag to candidate QIDs, keep only candidates that have a character
roster (the precision filter — it rejects `chimera` → キマイラ), union their
rosters, then token-subset match the paren-stripped name
(`aru` ⊆ {rikuhachima, aru} → **陸八魔アル**). Unqualified full names
(`aisaka taiga`, 607 of the tags) go through a **global label match** guarded
by ≥2 tokens plus `P31/P279* Q95074` (fictional character) — the type
constraint is what makes flat matching safe here.

An unguarded flat lookup on *single-token given names* is the trap: it looks
like 67.6% recall and mostly returns the wrong entity (`aru` → アランデル駅,
a railway station; `ann` → Anime News Network).

Measured 2026-08-15: 57/150 franchises resolved; **499/1314 characters**
(257 via roster, 242 via full name) = 38.0% of types and **42.9% of tag
occurrences**; **0/89 artists** — pixiv handles are not encyclopedia entities,
expect nothing there. Contributes **556 unique CJK codepoints**, including
rare name kanji (鈎 錠 霄 梔 棗 楯 芬) that ordinary caption prose never reaches.
Rows carry `via` provenance so the two routes stay separable.

Precision evidence is 25/25 correct on a spot-check of each route. The
`character_ambiguous: 0` stat is **not** independent evidence — ambiguity is
only flagged when two candidates carry different JA strings at equal EN-token
length, so a shorter label wins silently.

`--langs ja ko zh` pulls the other labels from the same rows, so the deferred
zh/ko phases inherit their proper-noun layer for free.

## `lovehina.py` — native register

Professional JA→PL translations of Love Hina vols 1 and 14. We keep the
**Japanese side only** — the Polish target is useless (our teacher needs EN),
but the JA is the one thing no other source supplies: native, professionally
written manga text that no MT system produced.

Its primary job is as a **held-out native-register eval set**. The proposal's
register-mismatch risk (student learns translated-sounding JA better than real
JA) is otherwise unmeasurable, because every other corpus source is either
MT output or MT input. Secondary job: 40 corner-bracket-quoted lines seed the
D6 quoted-text constructions, and reading order is preserved so consecutive
bubbles can be concatenated into multi-utterance samples.

Measured: 3,705 lines, 51,575 chars, **904 unique kanji + 161 kana**. The two
sources are complementary rather than redundant — each holds codepoints the
other misses, the lexicon's being name kanji (梔 棗 楯 鈎 錠 霄 芬).

Volume is small and it is **not** bulk training data — treat it as eval plus
seed. Only annotation JSON is fetched; the pages live in Manga109-s, which
forbids redistribution and needs a separate application. That matters only if
Phase 4 (OCR + glyph rendering) ever wants images — the text-only distillation
does not.

## `manga_text.py` — the danbooru text-detection corpus. **Measured: not corpus material**

Source: a local 8.4 GB text-*detection* set — 73,725 danbooru images (post ids
as filenames), `test-00000-of-00001.parquet` (images + merged axis-aligned
boxes) plus `polys_test.json` (COCO, **one segmentation polygon per text
line**: 602,127 `text` + 16,446 `hard_neg`). The polygon file's image ids are a
subset of the parquet's; the 1,417 missing are exactly the zero-annotation
images. No transcriptions anywhere — the labels are geometry only, so any text
has to be OCR'd out.

The pilot (`--sample 1000`, 60 evenly-spaced row groups; the parquet is sorted
by post id so a clustered draw samples one era of the site) crops each polygon
with orientation-preserving deskew — manga-ocr reads vertical Japanese
natively, and taking `minAreaRect`'s angle verbatim would transpose a bubble
column into a horizontal strip — reads it with manga-ocr, and translates the
dialogue with Hy-MT2 JA→EN. Measured on 986 text regions + 114 `hard_neg`
controls (`assets/manga_text_pilot/report.md`):

| question | measured |
|---|---|
| register | `line` 34.5%, `short` 11.8% → 46.2% nominally translatable; `sfx` 21.5%, punctuation-only 21.0%, latin/digit 10.3% |
| confidence gate | exists but overlaps: at logprob > −0.05 it keeps **35% of lines** and still admits 4.4% of `hard_neg` and 9.4% of SFX |
| end-to-end yield | 986 regions → 340 lines → **119 gated lines = 12%** |
| MT | not the bottleneck, and that is the problem — it launders OCR noise into fluent English (`ブスターSEMIT`→"Bustar SEMIT", `人を作る前に、セック`→"Before creating people, Sek", `鎌蛋`→"Sickle egg") |
| tag vocabulary in-image | **3.0%** of regions contain a glossary surface; 3/986 are a whole-region tag word |

**Verdict: do not add to the mix.** Two independent reasons, either sufficient:

1. **Register.** Clean reads are speech fragments ("What's going on here?",
   "Uh...", "Like a set?"), i.e. exactly D4's register — and JESC ships 2.8 M
   human-aligned pairs of it with no OCR and no MT. Paying an OCR pass to
   manufacture a noisier version of a free corpus is backwards.
2. **Undetectable corruption.** MT is fluent on garbage input, so a bad read
   arrives as a confident, plausible EN teacher target. Nothing downstream
   flags it, which is the one failure mode a distillation corpus cannot absorb.

Yield could be raised — crop size predicts usability strongly (min side ≥ 64 px
→ 65.8% usable vs 18.6% in [16,32)), so a size prefilter is the obvious lever —
but yield was never the deciding term; register is, and a prefilter does not
move it.

What the dataset *is* good for is the geometry, which needs no OCR at all:
**text-mask validation** (the MIT detector behind `make mask`, whose
`MIT_TEXT_THRESHOLD` is unmeasured, and whose masks Phase 2 depends on staying
ON), and **Phase 4 glyph rendering**, where per-line polygons on domain-matched
images are the asset nothing else supplies. The tag-hit path also shows the
flat-match trap a third time: unguarded substring matching fired `たな` (shelf)
inside `やがったなこのやろ` and two-mora katakana given names (`カイ` = irida
(pokemon), `アイ` = hoshino ai) inside plain dialogue — hence `_substring_safe`,
which trusts kanji at length 2 and demands 4 characters of everything else.
`--recount` re-cuts every bucket from `pilot.jsonl` without a second GPU pass.

## Caption roots — D1's width, and why the raw pool needs a rules pass

`build_pairs.py` and `tag_glossary.py` both read **multiple caption roots**:
`--captions` for roots already in Anima's caption convention, `--raw-captions`
for raw crawler output, `--tag-rules` for the rule file. The default pair is
`image_dataset` (curated, 3,008) + `~/gelcrawl/retrieved` (raw, 16,053), which
is what D1-wide means: this corpus is **text-only**, so an image that failed
curation still has a caption saying what a JA user would type, and the crawl
pool is 5.4× the curated set.

The two roots are not interchangeable strings. `retrieved/` is raw crawler
output — `&#039;` HTML entities, booru rating words (`general`/`questionable`)
instead of Anima's band (`safe`/`nsfw`), `highres`/`absurdres` meta tags, and
undeduped clothing bases (`breasts` alongside `large breasts`). Left alone it
would split the rating rows and train `questionable`'s ext rows separately
from `nsfw`'s. So raw roots go through gelcrawl's own `tag_rules.yaml` via
`library.captioning.tag_rules` — deliberately the *same* rule set that
produced `image_dataset`, not a reimplementation. Verified segment-for-segment
against a curated caption.

Dedup is on the **artist-relative path**, not the bare stem: `image_dataset`
was curated out of `retrieved/` so 2,933 of 3,008 overlap, and the stem alone
is not unique because both boorus share each artist directory (`dan_` prefix =
danbooru id space). First root wins, so the curated copy beats its own source.

**What widening does and does not buy** — measured in
[`../report_0816_phase2.md`](../report_0816_phase2.md#d1-wide--the-gelcrawl-widening-measured-2026-08-16).
It buys **visits** (500+ band 381 → 756, 4.3× total) and **not vocabulary**
(ext rows visited flat at ~6,400), because the JA side is composed from a
glossary that did not grow. Corollary worth keeping: the render grid's
zero-visit tokens are **not** a corpus-size problem. `armor` occurs 39× in the
widened corpus and `鎧` still has 0 visits — the glossary bound the tag to
`アーマー`. Do not answer a `v=0` token by crawling more captions.

## Do not rebuild the glossary without `--mt`

Tried and reverted. The back-translation scoring is what *selects* among
candidates, so a CPU-only rebuild drops every `mt_verified`/`wiki_verified`
verdict and the `candidates` field with it (5,920 entries → 0), re-picking
straight from the wiki head — which is exactly the failure this file warns
about above. It reproduced `underwear` → 下着コート, `censored` → 遮盖
(Chinese), `large breasts` → デカ乳, regressing 1,991 wordings and unresolving
2,948 tags. The widened rebuild is a daemon job; the prompt-keyed MT cache
makes the 7,514 existing tags nearly free.

### The kana guard cuts both ways (the D1-words review axis)

Kana proves Japaneseness, which is why `alt_pool` and the candidate selector
require it for general-axis tags — it is what keeps 汗液 and 短袖 out. But it
also **excludes native kanji words**, from the primary *and* from the alt
pool, so the kanji wording is reachable from neither register:

| tag | chosen | native candidate, same f1 |
|---|---|---|
| `armor` | アーマー | 鎧 |
| `from above` | 上から | 俯瞰 |
| `close-up` | クローズアップ | 接写 |
| `portrait` | ポートレート | 肖像 |

119 general tags / 1,735 occurrences sit in this class. It is **not**
automatically fixable: roughly half the Han-only rivals are native Japanese
wrongly rejected (上着 翼 靴下 腕輪 眼鏡 砂浜 刺青 逆光 直毛 漫画 扉 果物
指輪 化粧 提灯 水筒) and half are Chinese correctly rejected (指甲油 智能手机
杯子 牛仔布 毛巾 背包 手提包 特写 影子 睡衣) — and `bed` → 床 is the case
that settles it, a Han-only string of perfectly ordinary JA kanji that means
*bed* in Chinese and **floor** in Japanese. No codepoint inventory separates
those; only Japanese knowledge does. It is a **human review axis**, and a
different one from `tag_glossary_review.md`'s `mt_unverified` ordering.

Measured against `tag_pairs.py`'s source (below), the class is **5× larger than
this estimate**: 626 tags / 55,821 occurrences (8.6% of all tag mass) where our
choice is katakana-only and the community field offers a kanji reading.

## `tag_pairs.py` — filling the tail from the community's own pairs

[`p1atdev/danbooru-ja-tag-pair-20241015`](https://huggingface.co/datasets/p1atdev/danbooru-ja-tag-pair-20241015)
(CC0) is the danbooru wiki `other_names` field at an Oct-2024 snapshot with
`calm3-22b-chat` filling tags that had no Japanese name: 151,431 rows —
93,393 **character**, 22,330 **copyright**, 35,708 general.

It resolves **74.8%** of the tags our glossary leaves unresolved (71.3% of that
occurrence mass); the residue is almost all artist handles, which are latin by
design. Applied: **5,248 tags filled, unmapped segments 42,530 → 13,714**.

**The module only fills.** Every wording the glossary already carries is pinned
or back-translation-chosen, and a CPU pass cannot re-verify what it overwrites
(the section above). An *unresolved* tag has nothing to lose — it composes as
latin passthrough at span weight 0 — so filling regresses nothing. Filled spans
carry `via: tagpair`, weighted 0.6 in `TRUST_POLICIES` (above `mt_unverified`,
below `wiki`); note `apply_trust` defaults an *unknown* `via` to 1.0, so a new
source must be declared there or it silently inherits full trust.

The source is unfiltered by its own admission, so the fill re-runs our guards
rather than trusting the field — `is_japanese`, plus the kanji-inventory check
for Han-only names (Shift-JIS accepts traditional Chinese; the inventory does
not), plus a latin drop. On the full table those reject 8,372 non-JA, 5,033
Han-only-outside-inventory (`裙`, `百褶裙`), and 28,234 latin-bearing
(`ウィンクX東方`) names.

**What it does not do yet**: re-select *existing* wordings. That is the bigger
prize and it is GPU work — the source knows the booru *sense* that MT cannot
(`bow` → 蝶結び the ribbon, not our お辞儀 *bowing*; `on back` → 仰向け supine,
not our 後ろ姿 *from behind*), agreeing with only 35% of our 5,435 MT-derived
wordings. Feed its names as candidates to the existing back-translation arbiter
and let them win on evidence. See `plan.md` (D1-pairs, item 2).
