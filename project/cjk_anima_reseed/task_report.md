# task_report — KO / ZH rows beside the 1008 seed (2026-10-09, session 12:19–13:36)

**Status: discarded.** The work ran on `main`'s stale copy of this line
(`reseed/` as of `cf5f51f7`), not on `cjk-reseed`: none of the branch's data
mix, data build or ruler eval was used, so the trained rows, their geometry
and their renders do not compare with anything on the branch. Every code,
config and output change was reverted or deleted; this file records what was
done and what is still worth keeping.

## 1. Question (user)

- Is the trained row geometry really a stick and a ball ("burr on a
  stick", `reports/stick_2026_10_03.md`), or a shuriken / maple leaf?
- How do Hangul and hanzi rows sit relative to the existing ball?

## 2. Reads that do not depend on the branch (valid)

CPU only, on the delta rows (`raw × row_scale`) of seed files.

**Shape of the kana ball** (`retrain_kana` / `recap_hp`, 166 kana rows):
- not a few-blade planar shape: the spike cloud's PC1 holds 6 % of the
  energy (an isotropic cloud with the same norms: 1.4 %), centered pairwise
  cos ≈ 0 (p95 |cos| 0.16), PR ≈ 70;
- the spikes are ⟂ the stick (cos to the stick: mean ≈ 0, sd 0.08–0.12), so
  the "ball" is a spray perpendicular to the stick at its tip; the 56° of
  each row off the stick is atan(210 / 140);
- branching is weak: the hiragana / katakana split axis holds 3 % of the
  energy (sub-means 31 against spikes 206); same-sound hira ↔ kata centered
  cos 0.08–0.11; dakuten ↔ base 0.23, small ↔ full 0.22–0.28 (the
  `row_geometry_2026_09_28` pairs again).
- User's call: keep the name "stick + ball".

**1008 seed, kana (165 rows) against kanji (1 410 rows)** (and 0930 for reference):

| | 1008 | 0930 |
|---|---|---|
| stick length kana / kanji | 119 / 125 | 148 / 143 |
| stick cos kana ↔ kanji | 0.757 | 0.784 |
| ball radius kana / kanji | 209 / 211 | 206 / 196 |
| kanji spikes in kana top-40 | 0.107 | 0.107 |
| kana half → other half (calibration) | 0.237 | 0.233 |
| random top-40 / 1024 | 0.039 | 0.039 |
| kanji spikes in the whole kana span (rank 165) | 0.278 (random 0.161) | 0.284 |
| kanji ↔ kana centered cos | p95 0.10, max 0.39 | p95 0.10, max 0.37 |

1008 vs 0930: sticks cos 0.989 (kana) / 0.982 (kanji), same-row cos 0.98 /
0.94; 1008's shorter kana stick comes from the `stick080` lineage. So: two
sticks about 40° apart, equal balls, partly shared axes (2.7× random, under
half of kana's own overlap), no one-to-one glyph pairs.

**Earlier KO / ZH read** (`finished/cjk_renderable_anima/reports/krzh16_2026_09_16.md`,
53k table, old recipe): Hangul and hanzi rows rebuild the shared direction
(cos +0.41 vs the 53k rows' +0.46), have kanji-sized norms, and sit outside
the kana subspace (0.17 of their energy in the 53k top-40 vs 0.45).

## 3. What was built, then reverted (the logic only)

Main-tree `project/cjk_anima_reseed/` edits, all reverted:
- `reseed/config.py`: a `"1008"` seed (`output/cjk_anima_reseed/seed_1008/trained.pt`)
  and a `lang = { korean = "…", chinese = "…" }` key (glyph → language).
- `reseed/pools.py`: for a run with `lang`, the faces in
  `cjk_anima_scale/assets/fonts/kozh/` join `pools.fonts`, and a face whose
  cmap maps a row glyph to an empty outline leaves the draw; `relang()`
  swaps `japanese text` / `Japanese text reads as` in a scene caption for
  the item's language.
- `reseed/recipes.py`: scene captions go through `relang` by the first
  non-Japanese glyph of the text.
- `reseed/builder.py`: with no window for any row (no dialogue line spells
  these glyphs), the `bubbleN` tiers drop and the other shares scale back to
  the table's Σ (× 1.515), so the items per row stay.
- `configs/kozh16.toml`: rows `가힝ㄹ몹감없양한` + `你这个们说东时为`
  (krzh16's set, 国 → 东 because 1008 already trains 国), seed 1008,
  225 steps / row.
- `probes/kozh_geometry.py` (§ 2's reads for an arm) and
  `probes/kozh_render.py` (the 1008 pack against a baked kozh16 pack, one
  seed, bubble / sign / plain prompts at 512²).

Run (all outputs deleted): 1 600 items (grid / lone / bubble1 tiers only),
3 600 steps, 25.8 min, final loss 0.095. Its geometry read (old recipe —
indicative only): Hangul and hanzi rows carry about half of a seed row's
stick component (row cos to either stick 0.29–0.32 vs ≈ 0.5); their spikes
sit in the kanji ball's top-40 nearly as much as held-out kanji rows (Hangul
0.244, hanzi 0.205, kanji 0.294) and in kana's like kanji do (0.14 / 0.11 vs
0.13); nearest seed rows at cos 0.2–0.3 (`为 → 為`, `ㄹ → 己` the only
shape-like ones).

A first attempt on `cjk_anima_scale` (lone group of `builder.TABLE`, captions
still `japanese text`) was killed at step 1 100 / 2 880 and deleted.

## 4. Worth keeping for a redo on this branch

- **Fonts** (re-fetch; binaries gitignored, deleted with the rest). All
  SIL OFL 1.1; coverage checked with fontTools (KS X 1001 = 2 350 Hangul
  syllables, GB2312 = 6 763 hanzi). Several ZH faces also cover kana, so they
  must not sit in the top-level `assets/fonts/` that `find_fonts()` globs
  for every JA run.

| face | role | coverage | source |
|---|---|---|---|
| Nanum Gothic | KO dialogue | KS X 1001 | google/fonts `ofl/nanumgothic` |
| Do Hyeon | KO emphasis | KS X 1001 | `ofl/dohyeon` |
| Jua | KO rounded / playful | KS X 1001 | `ofl/jua` |
| Black Han Sans | KO poster | KS X 1001 | `ofl/blackhansans` |
| Nanum Pen Script | KO marker | KS X 1001 | `ofl/nanumpenscript` |
| Nanum Myeongjo | KO serif | KS X 1001 | `ofl/nanummyeongjo` |
| Noto Sans SC Medium | ZH dialogue (+ kana) | GB2312 | notofonts/noto-cjk `Sans/SubsetOTF/SC` |
| Smiley Sans 得意黑 | ZH emphasis (+ kana) | GB2312 | atelier-anchor/smiley-sans |
| ZCOOL KuaiLe 站酷快乐体 | ZH playful | GB2312 | `ofl/zcoolkuaile` |
| Douyin Sans 抖音美好体 | ZH poster (+ kana) | GB2312 | bytedance/fonts `DouyinSans/` |
| LXGW Marker Gothic 霞鹜漫黑 | ZH marker (+ JIS) | GB2312 | lxgw/LxgwMarkerGothic |
| LXGW WenKai 霞鹜文楷 | handwriting; KO + ZH + JA in one face | all 11 172, GB2312, JIS | lxgw/LxgwWenKai |

- **`TanukiMagic.ttf` maps 你 to an empty outline**: `font_covers` passes
  it and `render_grid` divides by zero width (`ZeroDivisionError` in
  `src/data/grid.py`). Any run drawing 你 must drop that face.
- **In-word windows need a KO / ZH corpus and enough rows.** A window is a
  2–6 glyph run whose every glyph is a row, so 8 Hangul syllables yield no
  window from any corpus; the Hangul row set has to cover common syllables
  (다 이 요 해 …) first. Corpus candidates: LCCC (ZH, MIT, Weibo turns),
  songys Chatbot_data (KO, MIT, 11.8k pairs), OpenSubtitles v2018 ko / zh_cn
  (large; unclear copyright), SmileStyle (KO, CC BY-NC). The JA pool format
  is `line\tbook\tn_pieces` (2–10 Qwen pieces, `norm_phrase`).
- **Captions**: an item lettered in Korean / Chinese needs `korean text` /
  `chinese text` and `Korean / Chinese text reads as`, not the JA words.
- **All 16 glyphs encode to one ext row**, routed and unrouted alike
  (`ext_encoder` check). `个` is in Shift-JIS, so script detection by
  encoding mislabels it; name the language per row instead.
- **Baking against 1008**: `seed_1008`'s pack is baked on
  `anima_cjk_vocab_pack_punct`; `bake_vocab_pack.py` without `--base` takes
  the configured `jp_v1`. Bake with `--base …_punct` so the packs differ in
  the new rows only. The branch's ruler renders arms through `ExtDelta`
  directly, so it does not need a bake.

## 5. Process notes

- Check out `cjk-reseed` before touching this line; `main` holds a stale
  snapshot of it.
- Use the branch's ruler for renders. Render at the training px (≈ 512),
  one seed, and work out the wall time before queuing.
