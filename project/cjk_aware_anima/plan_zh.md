# CJK-aware Anima — Chinese extension plan (plan_zh, 2026-09-02)

*Appends `zh` to the JA (+KO) line. Drafted the day the adapter probes
(`findings.md` §9, `probes/`) reframed what an ext row is: a **key into a
near-context-free per-token code** the DiT reads, trained only through a
shared map. Everything below cites a measured number or says it is unmeasured.
Read `findings.md` §9 first — it is the reason ZH is worth more than KO was.*

## Why ZH, and why now

Korean lives in its own 12k rows; KO data never touched a row Japanese uses.
Chinese is the one language whose corpus lands **on the JA row set**:

| fact (measured 2026-09-02) | number | pointer |
|---|---|---|
| JA tag-corpus kanji occurrences whose character also appears in zh `other_names` | **83.3%** (1,253 / 1,788 distinct) | this file's build probe |
| ext rows visited by a JA-only distill / JA+KO | 3,521 / 5,553 of 58,968 | `bench/cjk_distill/results/*` |
| surviving key dimensions after training, JA-only / JA+KO packs | PR 55 / 84 (init 236, native T5 267) | `probes/spread_probe.py`, `probes/map_probe.py` |
| Danbooru wiki entries with a simplified-Chinese `other_name` | **10,693** (kana 73,452; hangul 3,760) | wiki dump 2026-04-28 |
| occurrence-weighted coverage of our captions by those zh names | 20.4% (494 tags) vs JA kana 32.9% (906) | same, `tag_counts` over `image_dataset` |
| community EN→zh_CN tag tables (tagcomplete ecosystem) | ~100k tags | §Sources |

Training keeps only the directions the teacher rewards (init PR 236 →
55). KO added a second script and the kept dims went 55 → 84. ZH adds
Han-space directions, which is where JA's own kanji sit — the first corpus
lever that can widen the JA subspace rather than a disjoint one.

**Premise unchanged from KO:** the encoder does not change. ZH is a corpus job
plus one joint retrain of the same pack, gated on JA not regressing.

## What ZH cannot do (so the gate is honest)

- **Cannot separate degenerate rows.** `param=global` has zero per-row
  freedom; two near-identical keys stay near-identical under any shared map.
  The char-fallback layer is that case (random char-row pairs: 60% above cos
  0.5). Its heaviest members are load-bearing JA tag kanji — 髪 (45k tag
  occurrences), 顔, 獣, 緑, 歯, 観, 団, 広 — and **those are exactly the
  shinjitai forms simplified Chinese does not share** (发/脸/兽/绿/齿/观/团/广).
  ZH data will not visit them. The char-row init fix (§Phase Z0) is a
  separate, cheap item and must not be folded into the ZH readout.
- **Not "more data".** Sentence-register corpora were closed under span loss
  (JESC/STAIR, `findings.md` §1); widening JA captions 5× bought visits not
  vocabulary (§D1-wide). Only the **tag register** has ever moved metrics. ZH
  enters as a tag register or not at all.
- **Does not touch arm C.** The unmask line reads on renders whose prompts
  carry no CJK; the pack acts only through what the LoRA learned (§9 tier
  2). ZH is a tier-1 (prompting) experiment and is read on distill metrics +
  the JA/ZH render grids, never on `cjk_unmask_*`.

## Script reality — which rows a ZH corpus visits

Three Han inventories overlap only partially: JA shinjitai, zh-Hans
(simplified), zh-Hant (traditional). Measured on the JA tag corpus vs the
wiki's zh names: 83% of JA kanji occurrences are shared by character (子 人
目 女 乳 赤 黒 服 白 色 男 口 首 …); the 17% miss is the shinjitai tail above.
Traditional forms recover part of it (開 間 後 閉 are zh-Hant-identical;
髮/顏/獸/綠/齒/觀/團/廣 are not). So:

- source both **zh-Hans and zh-Hant** surfaces; normalise with OpenCC in
  *both* directions and keep both as `tags_zh` / `tags_zh_hant` registers
  (a Hant register is cheap and visits the 開/間/後/閉 class).
- the 25k Chinese-word subword rows JA never visits become live for the
  first time (制服 / 学校 / 男性 are single Qwen tokens shared by JA and ZH;
  严格执行-class rows stay dead — fine).
- `is_symbol_tag` / latin passthrough rules carry over unchanged
  ([[feedback_emoticon_tags_stay_latin]]).

## Sources (priority order, mirrors `tag_glossary.py`)

1. **Community EN→zh_CN tag tables** — the tagcomplete ecosystem: byzod
   `Tags-zh-full-pack.csv`, Yellow-Rush `zh_CN-Tags/danbooru.csv`, the
   ChinaGPT "10w" pack, the discussion-thread Danbooru.csv translation
   (links in §References). These are the Chinese community's *own register*
   for the tags users type (双马尾, 傲娇, 长筒袜) — the zh analogue of
   `other_names`, and far wider (~100k tags vs the wiki's 10.7k). **Licence
   check per file before use**; keep provenance per entry (`via`).
2. **Danbooru wiki `other_names`, zh subset** — invert the current guard:
   Han-only entries that *fail* Shift-JIS are zh-Hans; Han-only entries that
   pass are ambiguous (JA kanji or zh-Hant) → route by OpenCC round-trip +
   the JA glossary (if the JA build already claimed it as Japanese, it is
   not a zh surface). 10,693 entries, 20.4% of our occurrences.
3. **Wikidata lexicon** with `--langs zh` — proper nouns (characters,
   franchises); asset must be rebuilt (shipped one is ja-only).
4. **Hy-MT2 `--mt` residue** — compositional tags only, same few-shot
   pattern (`TAG_FEWSHOT_ZH`), same contamination guards, and the zh
   false-friend veto list (§Risks).
5. Rating band built-in map.

Not a source: image-caption datasets with zh captions (prose register,
closed), and zh↔ja MT of the JA glossary (would re-import the exact
contamination the r1/r2 rounds removed).

## Phases

**Z0 — char-row init fix (independent of ZH, do first, CPU).** The byte-
fragment mean init makes char rows near-identical. Replace with a
Procrustes-mixed anchor map (`ridge + 0.6·Procrustes`: key PR 236 → 373,
collisions 16% → 1%, held-out cos 0.75 → 0.70; `probes/map_probe.py`) and
initialise char rows from the *mapped Qwen contextual embedding* of the
character rather than the fragment mean. Gate: char-row pairwise cos among
the top-200 JA tag kanji ≤ the qwen-token-row baseline (0.23 in key space,
`probes/char_probe.py`); JA distill metrics at parity on a v5-recipe rerun.
Ship as `ext_embed` v2; every later pack inherits it.

**Z1 — zh glossary + corpus (CPU, ~1 day incl. review).**
`tag_glossary.py --lang zh` (new: `is_chinese`, OpenCC normalisation, the
inverted SJIS route, `TAG_FEWSHOT_ZH`), `build_pairs.py --lang zh` →
`tags_zh / tags_alt_zh / names_zh / names_synth_zh` (+ `tags_zh_hant`),
`synth_tags.py --lang zh`. One review round on the occ>100 head (the JA r1
pattern; expect the same ~300-fix scale). Deliverable: `tag_glossary_zh.json`,
`pairs_*_zh.jsonl`, coverage report (`coverage_zh.json`).

**Z2 — cache + joint distill (GPU, ~65 + 45 min).** Cache at KO scale
(cache_ko was 43 G / 73k pairs). ~~disk: 105 G free, synth4 alone is 173 G~~
— resolved 2026-09-02: `cache_synth4` deleted with the synth-name register
(277 G free; the JA cache is now ~49 G). Distill = the shipped recipe (JA
`pairs_tags` + KO + ZH registers, 12k steps) trained **cold and joint** →
`synthjakozh1`. Controls = `synthjako3` (warm chain, on disk) and a cold
JA+KO run (isolates the warm start).

**Z3 — read.** Three readouts, in order of authority:
1. **JA no-regression gate** (hard): JA holdout `recovery / cos(s,t) /
   disc_far` within noise of `synthjako3`; the JA render grids
   (`ja_eval_prompts*.json`, `run_bench --arms en,ja_t5en,ja_ext`) at parity
   on the 20 prompts. Any JA regression → ZH ships as a *separate* pack, not
   joint.
2. **Row geometry**: kept dims (PR) and collision rate on the JA-visited
   rows before/after (`probes/spread_probe.py`); the claim "ZH widens the JA
   subspace" is either a number here or dropped.
3. **ZH itself**: `zh_eval_prompts.json` (20 prompts, mixed register like
   r1 — zh name + EN tags is what users type), arms `en / zh_t5en / zh_ext`.
   Bar = the JA bar at the same stage (tags + katakana-class names transfer;
   rare-Han names are not expected to compose — plan3 verdict applies).

**Z4 — ship.** Vocab pack on HF (`anima-vocab-pack-cjk`), Adapter node
loader unchanged (the id space is tokenizer-deterministic, so a ZH-trained
table is a drop-in). README states the tiering: EN-aligned rows for
tags/names, substitution for known surfaces, trained per-char rows for free
text.

## Status log

**2026-09-02 (Z0 done, Z1 tooling done, corpus decision).**

- **synthja dropped, cache_synth4 deleted (user call).** The 175k
  `names_synth_ja` pairs were 73 % of the JA corpus and ~126 G of the 173 G
  cache, drawn at weight 0.2 (< 1 pass in 12k steps); their stated purpose
  (rare-kanji name rendering) was falsified in `findings.md` §4 and §9 hands
  names to substitution. The JA training corpus is now
  `pairs_tags.jsonl` = `pairs.jsonl` (63,241) + `tags_synth.jsonl` (4,814,
  re-allocated without the synth-name visits) = **68,055 pairs**
  (`synth_tags.py --pairs pairs.jsonl --out pairs_tags.jsonl`). Cache
  `cache_tags` (~49 G). What the removal costs is measured, not assumed:
  rows_visited / char-row collisions (`probes/z0_probe.py`) and the JA
  holdout + grid vs `synthja_v5` (jobs below). Also dropped from the recipe:
  `--register_sampling names_synth:0.5` was inert (no such register in the
  file) — do not carry it forward.
- **Z0 shipped as `ext_embed` v2** (`build_ext.py`, v1 kept at
  `ext_embed_v1.*`; id mapping identical, so caches and old packs are
  unaffected). Recipe: `--map procrustes-mix` (ridge + 0.6·Procrustes, qwen
  rows PR 231 → 365, collisions 17 % → 1.5 %, held-out 0.756 → 0.713) and
  `--char-init contextual`. The plan's "mapped Qwen contextual embedding"
  needed three corrections before it cleared the bar: (1) EN anchors are the
  wrong fit set for the contextual map (held-out 0.46, PR 6 — Qwen hidden
  states share one anisotropic direction; mapped raw, the char layer
  collapsed to PR 10 / 57 % collisions); (2) adding the T5 mean back made it
  worse; (3) the fit set that works is the **11.6k CJK chars that are clean
  single Qwen tokens** — standardized contextual state → their own token-row
  key, ridge 0.1 + 0.6·Procrustes, held-out cos→key 0.45, char layer
  row-normalized. Gate: top-200 JA tag kanji char rows **cos 0.137 / 1.4 %
  > 0.5** vs the token-row bar 0.188 / 0 % (plan bar 0.23); v1 was 0.589 /
  99.9 %. The rare tail (ext-A, unseen ideographs, most hangul *syllables* in
  the char layer) stays clustered (random char pairs 8.8 %) — Qwen reads
  those as one "unknown" state. **KO note:** hangul syllables that are not
  clean Qwen tokens live in that tail; whether v2 helps or hurts KO is read
  on the joint distill, not assumed.
- **Z1 glossary, CPU tier** (`tag_glossary.py --lang zh`): 15,097 tags /
  678k occurrences, **95.9 % covered before MT** — packs (`kb`) 86.4 %, wiki
  3.6 %, wikidata 0.6 %, passthrough 2.9 %, rating 2.5 %, unresolved 4.1 %.
  Source order for zh differs from JA (user call: centre on the NGA
  translation): override → symbol/artist passthrough → rating → Wikidata
  (hans-normalized) → **curated packs** (HalfMAI NGA gist 5.4k, then byzod
  10.6k MIT) → wiki `other_names` (OpenCC-routed) → unresolved; the ChinaGPT
  10w pack (MT renderings) is a candidate pool only (`src: kbmt`, ranks with
  MT). In arbitration a curated pack wording that back-translates to the
  tag (`f1 ≥ 0.75`) wins outright (`kb_verified`) — the literal MT rendering
  never outranks the community register on F1 alone. **Licence:** the NGA
  gist states none (source ngabbs.com thread); byzod/ChinaGPT are MIT;
  Yellow-Rush is unlicensed and unread. Recorded per entry via `src`.
  Script routing lessons (all measured today): a plain "OpenCC s2t changes
  it ⇒ simplified" test is wrong (the table also converts JA shinjitai) —
  the round-trip rule in `han_char_class` fixes that; and the wiki's
  hans-class entries are a contaminated census (左右対称 on 称 seeds 対), so
  the zh inventory comes from the packs (4,058 chars) plus wiki chars at
  count ≥ 5. Wikidata zh labels arrive traditional (初音未來) and are
  hans-normalized; pack names drop the franchise qualifier (甘雨（原神）→
  甘雨) on name axes. Remaining JA leaks are in the wiki tail (泥酔, 畳,
  令和6年 — 324 shared-class wiki picks, 3.6 % of occurrences) and are the
  MT arbitration's + the r1 review's job.
- **Z1 corpus tooling:** `build_pairs.py --lang zh` (KO layout + the
  `tags_zh_hant` sibling register = OpenCC s2t of every `tags_zh` record),
  `synth_tags.py --lang zh` (+ `--out`), `assets/zh_eval_prompts.json`
  (agent-drafted, **needs the native review**), `mt.py TAG_BACKGROUND_ZH /
  TAG_FEWSHOT_ZH`, Wikidata lexicon rebuilt `--langs ja ko zh` and the zh
  labels merged into the shipped file (313 characters; ja/ko bit-identical,
  backup `wikidata_lexicon.pre_zh.json`). `names_synth_zh` **not built**
  (synth names are out of the recipe; the name tier is substitution).
  Tests: `tests/test_cjk_glossary.py` zh block (han class round trip,
  inventory, pack loader, arbitration rule, hant register).
- **Jobs queued (daemon, FIFO):** `zh-glossary-mt` 20260902-183801-172ea4
  (EN→ZH over 10,568 general tags + back-translation; ~2–3 h) →
  `z1-cache-ja-tags` 20260902-184212-622c9c (`cache_tags`) →
  `2c-synthja-v6-tags-extv1` 20260902-184212-64ec0b and
  `2c-synthja-v6-tags-extv2` 20260902-184212-316a90 (same recipe as v5 minus
  the synth registers; the pair isolates Z0, and extv2 vs v5 reads the synth
  removal). After the MT job: `--reselect` (the job started before the
  wikidata/qualifier/rank patches landed), r1 review on the occ>100 head,
  `build_pairs.py --lang zh`, zh cache, then the **cold joint** JA+KO+ZH
  distill (no warm start — every jako pack so far was warm-started from a
  JA pack; a cold JA+KO control isolates that) with `synthjako3` as the
  warm-chain control.

## Risks

1. **False friends on shared rows.** 娘 (daughter / girl), 勉強, 湯, 手紙,
   床 (`bed` → 床 = floor already caught in the JA review), 汽車. The map is
   shared so rows do not fight per row, but the teacher becomes bimodal for
   those characters. Mitigation: a veto list seeded from the JA review's
   rejected-Chinese rows (`rejected_contaminated`), and the Z3 JA gate.
2. **Register drift inside zh.** Mainland vs Taiwan wordings (视频/影片,
   萝莉 is shared, 兽耳/獸耳 differ only in form). Treat as alternates
   (`tags_alt_zh`), not collisions.
3. **The r2 lesson.** A glossary edit alone moved arm-C renders (C3/C4,
   `findings.md` §9). Do not read ZH on arm C, and do not re-cut the JA
   glossary in the same round — one corpus change per distill.
4. **Disk.** See Z2. Delete `cache_synth4` only after v5/jako3 are declared
   the JA baseline.
5. **Licence** of the community CSVs is per-repo and often unstated; the
   wiki dump and Wikidata are the licence-clean floor (20.4% coverage).

## Deliverables

- `datasets/tag_glossary.py --lang zh`, `build_pairs.py --lang zh`,
  `synth_tags.py --lang zh`, `mt.py TAG_FEWSHOT_ZH`, OpenCC dep
  (`opencc-python-reimplemented`, pure Python).
- `assets/zh_eval_prompts.json`, `tag_glossary_review_zh_r1.md`.
- `bench/cjk_adapter/build_ext.py --map procrustes-mix` (+ char init) → Z0.
- `findings.md` §10 with the Z3 numbers, whichever way they fall.

## References

- Yellow-Rush/zh_CN-Tags — https://github.com/Yellow-Rush/zh_CN-Tags
- byzod/a1111-sd-webui-tagcomplete-CN (Tags-zh-full-pack) — https://github.com/byzod/a1111-sd-webui-tagcomplete-CN
- ChinaGPT/a1111-sd-webui-tagcomplete-10w — https://github.com/ChinaGPT/a1111-sd-webui-tagcomplete-10w
- tagcomplete zh translation thread — https://github.com/DominikDoom/a1111-sd-webui-tagcomplete/discussions/23
- HalfMAI tagcomplete zh gist — https://gist.github.com/HalfMAI/e20a974a8b87bbb63d8da8051442b6b2
- Danbooru wiki dump — `kierarkia/danbooru-wiki-2026` (local HF cache)
- Probes — `project/cjk_aware_anima/probes/` (CPU, no daemon needed)
