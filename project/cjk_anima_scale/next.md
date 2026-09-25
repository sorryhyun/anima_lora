# next — the 72-row joint read on the product rulers (2026-09-25)

What comes after `reports/conflict_joint_2026_09_25.md`, judged by
`product_criteria.md` (Axis 1 `native_sent` on はい / おしい / やったネ /
ちょっと来い / こんにちは + `target`; Axis 2 the page, on the sheet).

## 0. What the last probe closed (`conflict_run0923_micro_b30_piece_hi`)

Piece items only (`scene_piece`, `grid_string`), 800 per spec, gradients at
the `b30` 0709 table (= the seed on piece rows):

| data / σ | ‖ḡ‖ | coh | half | vs 0507 @ 0.5–0.7 |
|---|---|---|---|---|
| 0507 @ 0.5–0.7 | 0.069 | 0.24 | +0.81 | – |
| **0507 @ 0.7–0.9** | 0.089 | **0.16** | +0.58 | cos **+0.38**, keep 0.83; Δ0507·−g **+0.23** |
| 0305 @ 0.3–0.5 | 0.041 | 0.24 | +0.68 | cos +0.57, keep 0.89 |

- **High σ is not dead for warm pieces, but it is the same pull, noisier**:
  aligned with 0.5–0.7 (+0.38, keep 0.83), continues what 0507 bought
  (+0.23), coherence 0.16 vs 0.24. No separate "count" direction to buy; a
  0.7–0.9 piece share is 0.5–0.7 at 1.3 × the price. Closed — the doubling
  (ププロ / アアン) is not a missing-band problem.
- **The price table** (coherent movement per draw, `‖ḡ‖·coh`):

  | recipe | 0507 | 0507 @ 0.7–0.9 | 0305 |
  |---|---|---|---|
  | scene_piece | 0.048 | 0.037 | **0.070** |
  | grid_string | 0.0025 | 0.0040 | 0.0011 |

  **`grid_string` buys a piece row 10–60 × less per draw than `scene_piece`**
  (the `micro_warmg_0923` verdict, now as a number) — and it is 15 % of
  0507's items and 50 % of 0305's. Every one of those draws is nearly free
  of signal for the row. `scene_piece` at 12–24 px in small bubbles (0305,
  `fill_min` 0.5) is at least as good per draw as 35–48 px.

  **Superseded 2026-09-25** (`reports/grid_box_2026_09_25.md`): that price was
  the plain-MSE grid; under `grid_box = 1` `grid_string` reads 0.019 / 0.021
  (0507 / 0305), a third of `scene_piece` per draw. Shares to re-decide.

  Decision (yours, on the plain price): set `grid_string` share to 0 in `stage0507` / `stage0305`
  and give it to `scene_piece` (0507: scene_piece 0.55; 0305: scene_piece
  0.6, scene_sentence 0.4). Grid cells stay for singles (`grid_single`;
  identity — exposure ledger).

## 0b. Before the run: `stagev2` — the grid recipes re-priced (user, 2026-09-25)

`reports/grid_box_2026_09_25.md` re-priced `grid_string` at a quarter to a
third of `scene_piece` per draw (parity per item, and level with it at
0.7–0.9), so the § 0 shares are to be re-decided. **Not by editing the
stage files**: the band stages stay as the chain of record ran them, and the
new mix goes in as a second generation — `configs/stage0507v2.toml`,
`stage0305v2.toml` (and a `jointv2.toml` whose `joint_from` names them),
copied from the v1 files with only the `[[data.mix]]` shares changed and the
read that set each share in its comment. `config.stage_names()` globs
`stage*.toml`, so a v2 is a stage like any other (`--stage stage0507v2`,
`warm_from = "stage0709"` still resolves under the run's tag);
`tests/test_line.py::test_stage_configs_load_and_chain` pins the v1 set of
four and gets the v2 names added when they land. What v2 changes, to
decide from the report:

- `grid_string` share in 0507 / 0305 — kept, not zeroed; how much is the
  scene-pool question (grid draws cost no scene), not a price question now.
- `grid_string` band — **stays 0.5–0.7 (0305: 0.3–0.5)**, no `windows.py`
  change. The direction probe (report § 4, `conflict_run0923_micro_b30_gridstr_dir`)
  read 0.3–0.7 as one pull (0507 × 0305 cos 0.91 at a 0.95 floor) and
  0.7–0.9 as a different one (cos 0.27; Δ0507 · −ḡ 0.06) — more signal,
  unread on any ruler, and scene pieces at 0.7–0.9 lost in training.
- `grid_single` in 0507 (24–32 px) — unchanged by `grid_box` on warm singles
  (report § 3); stays out until a cold-single read.

`run0925_72` then runs on the v2 stages.

## 1. The run: `run0925_72`

72 rows, all warm in `rows_step1_0921_merged`, chosen so every
`native_sent` / `target` string is covered (checked against the tokenizer,
2026-09-25; norms are the seed's):

- **24 kana** — the micro 8 `あ い お な ア ナ ラ ル` + `は や ネ え う ん に
  だ め か す つ ま も て で` (は/い/お/や/ネ/来 are the target strings'
  singles; all at norm 1.2–2.0).
- **24 kanji** — the micro 8 `人 日 口 女 精 聞 動 願` + `来 大 中 上 下 手 目
  心 気 言 行 見 生 出 入` (23 verified warm; `小` is **cold** in the seed —
  excluded; pick the 24th from `kanji:200` and check it has a row).
- **24 pieces** — required `しい った ちょっと こんにちは ありがとう` + the
  micro 8 `それを はじ やはり すご メン アン プロ ファ` + `そう すごい して
  ます です ない から まで って でも ダメ` (all one-token rows, norm
  0.57–1.1 — the half-trained regime the pieces read is about).

```toml
# configs/runs/run0925_72.toml
run = "run0925_72"
seed_table = "output/cjk_anima_scale/rows_step1_0921_merged/trained.pt"
seed = 0
[data]
units = ["chars:あいおなアナラルはやネえうんにだめかすつまもてで人日口女精聞動願来大中上下手目心気言行見生出入<+1>",
         "list:しい,った,ちょっと,こんにちは,ありがとう,それを,はじ,やはり,すご,メン,アン,プロ,ファ,そう,すごい,して,ます,です,ない,から,まで,って,でも,ダメ"]
pieces = ""
phrase_file = ""                 # no corpus lines: the strings are the ruler, not the data
n_items = 10000                  # per stage dir: 72 rows × 90 × 4 = 26 k draws over 20–30 k items
[budget]
joint = 90
[eval]
groups = "single,word,en"
native_chars = "あ,い,日,願"
sent_strings = "はい,おしい,やったネ,ちょっと来い,こんにちは"
target = true
```

Data: `stage0709` / `stage0507` / `stage0305` `--steps data` under this run
(CPU, ~10 k items each), then two joint dirs:

- `joint` (`joint_from` = 0709 + 0507 + 0305) — the **30/30/30** composition;
- `joint0507_0305` (new stage file, `joint_from` = 0507 + 0305, otherwise
  `joint.toml`) — the **0/45/45** composition.

Both at 90 steps/row (72 rows → ~6 500 steps, ~50 min a train).

## 2. The arms (all joint, same data, `--submit --queue` in this order)

| arm | composition | μ | lr | question |
|---|---|---|---|---|
| A | 0/45/45 | 0 | 1e-3 | pieces move (drift ≈ 0.5); what do the strings and the page pay — **note** A trains the singles too, so "A up → pieces-only movement is the design" (§ 3) does not follow from A alone; the outside review (2026-09-25, `idea.md`) would run **A-F** first: same data, μ 0, lr 1e-3, `grid_box = 1`, the 24 pieces trainable and the 48 singles frozen as `context = "seed"` (はい = two frozen singles is then a same-as-seed control) |
| B | 30/30/30 | 0 | 1e-3 | does the 0709 share hold the singles inside A's regime |
| C | 0/45/45 | 0.01 | 1e-3 | the spring at the piece-moving lr — A with a leash |

```
S="project/cjk_anima_scale/scale.py --run run0925_72"
$S --stage joint0507_0305 --steps train eval --eval_only eval native sent target --init_anchor 0    --lr_rows 1e-3 --tag run0925_72_A --submit --queue
$S --stage joint          --steps train eval --eval_only eval native sent target --init_anchor 0    --lr_rows 1e-3 --tag run0925_72_B --submit --queue
$S --stage joint0507_0305 --steps train eval --eval_only eval native sent target --init_anchor 0.01 --lr_rows 1e-3 --tag run0925_72_C --submit --queue
```

(data dirs under `run0925_72_{A,B,C}` are symlinks to the one build, as the
`b100` arms were.) Plus the seed baseline once: `--seed_only` with
`sent target native` on the joint stage, so Axis 1 has its floor on the same
prompts. ~3.5 h for the three arms + rulers.

## 3. Read

One sheet per `product_criteria.md` § The sheet: `native_sent` blocks
first (5 strings × 16), `target` (はい 8, こんにちは 6), then あ い 日 願;
arms A / B / C / seed as rows. Accept by § Accept. What each outcome means:

- A up on strings, page holds → pieces-only movement is the design; B tells
  whether singles need the 0709 share or the freeze.
- A up on strings, page wipes (`uu` / 目 / fake words on the singles) → the
  freeze (`context = "seed"` on the singles, pieces the inventory) is the
  next run, at A's settings.
- A flat on strings at drift ≈ 0.5 → ~~rerun A at 200/row~~ **not a
  supported branch** (`reports/conflict_joint_2026_09_25.md` § 6, 2026-09-25):
  the μ 0 / lr 1e-3 joint's *pieces* were already at 0.89 vs the seed (the
  0.55 was the mixed-table mean); the displacement was bought and the hits
  were not. A flat A means the inventory / mix / loss differ from
  `micro_warm_0923`, not the path length — read the freeze arm (A-F below)
  before adding steps.
- C ≈ A → the spring costs nothing at 0.01 and is kept; C ≈ pinned → μ stays 0.

Not in this round: the token-wise outside-box EN-ref cosine (Axis 2 by eye
until it lands in `eval/enref.py`); sentence draws (`stage0309` data, the
one unread conflict cell); anything on the full 2 274-row inventory.

## 4. Launched instead: `run0925_300f` — the freeze arm at 300 pieces (2026-09-25)

Decided over § 1–2 after the § 6 re-read of the conflict report (the
displacement was bought, the hits were not; singles paid) and the outside
review (`idea.md`): one arm, not A / B / C.

- **Inventory**: `assets/units/ja_pieces_0925_300.txt` — the 5 ruler pieces +
  `run0925_72`'s 24 + `ja_cold_0001_1900.txt` by corpus count to 300, all
  warm in `rows_step1_0921_merged`; glyphs 2 / 3 / 4 / 5 = 199 / 82 / 16 / 3.
  `context = "seed"`: every kana / kanji single rides frozen (はい = a
  same-as-seed control; あ い `native` too). No kanji in the inventory — a
  single trained here would sit in the 0.3–0.7 bands off the law, on 0507's
  single recipes only; kanji refinement is a separate 0709 pass.
- **Data**: `stage0507` + `stage0305` under the run, 10 000 items each
  (disk-bound: 1.4 MB / item, 43 GB free), merged by `configs/joint0507_0305.toml`
  (0 / 45 / 45). stage0507's `scene_single` / `grid_single` drop under a
  pieces-only inventory (`recipes.missing_source`, new `scene_single` case;
  the `assert singles` became a notice) → scene_piece 0.53 / grid_string
  0.2 / scene_short 0.27. `phrase_file` on: 0305 `scene_sentence` 0.4 draws
  MANGA109S lines with ≥ 1 inventory piece, the rest context rows.
- **Train**: μ 0, lr 1e-3, `grid_box` 1, 90 steps/row × 300 = 27 000 steps
  ≈ 3.2 h at 2.33 it/s; eval `word,en` + native あ い + `native_sent` (5) +
  `target`.

```
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack
S="project/cjk_anima_scale/scale.py --run run0925_300f"
$S --stage stage0507 --steps data --workers 10 && $S --stage stage0305 --steps data --workers 10 && $S --stage joint0507_0305 --steps data   # CPU, launched 2026-09-25 14:05
$S --stage joint0507_0305 --steps train eval --eval_only eval native sent target --submit --queue                                          # after the build
```

Read by `product_criteria.md`: Axis 1 on the five strings (こんにちは / おしい /
やったネ / ちょっと来い carry trainable pieces; はい does not), Axis 2 on the
sheet against the seed. Dev vs acceptance: this run's strings *are* the
acceptance set — the first freeze read is a go / no-go on the design, not a
share choice; the `word` group (18 of the 300) is the dev-side exact ruler.

### 4a. Result (2026-09-25 evening) — nothing bought, at any displacement

Job `20260925-144054-bcf21e` (the first launch, `…-141129-9171b5`, trained
35 rows — `inventory_ext` read the inventory off the eval sample; fixed,
`e1ff8719`). 300 rows, 27 000 steps, 2.32 it/s; drift 0.79 (2 500) →
1.39 (5 000) → 1.74 (12 500, plateau) → 1.66 (end), cos 0.49, norm ×1.38.
Every 5 k table kept under `rows_scale_joint0507_0305_run0925_300f/intermediate/`.

| ruler | step 5 000 (drift 1.39) | step 27 000 (drift 1.66) |
|---|---|---|
| `native_sent` はい / おしい / やったネ / ちょっと来い / こんにちは (of 16) | 2 / 2 / 0 / 0 / 0 = 4/80 | 3 / 2 / 0 / 0 / 0 = 5/80 |
| `target` (verbatim, 14) | – | 0/14 |
| `word` exact (18 × 2) | 3/36 | 4/36 |
| `en` | 24/24 | 24/24 |
| frozen あ / い native en·swap (of 16) | – | 14·14 / 9·7 (seed 14·15 / 11·8) |

The reads are the doubling signature grown into **fake dialogue lines**
(こんにちは → こんはこんちは / がほがはいくにはしてまは。/ スムリにたの歩音は;
やったネ → ややったたたーう), already at 5 000 — not an overshoot. On the
sheet the page and the bubble hold (no new paste beyond the seed's flat
backgrounds) and the bubble carries a 14–22 px line of pseudo-Japanese:
the **wipe** shape of `product_criteria.md`. The frozen singles read as
the seed (±2 = the render noise floor). No seed floor exists for the five
strings yet (`--seed_only --eval_only sent target`, never run).

Read: with the singles frozen, piece displacement on **this mix** buys no
word at 1.4 or 1.7. The comparator that did move pieces
(`micro_warm_0923`, 2 → 13/32) was `scene_piece` only at 0.5–0.7; here a
piece row's draws were mostly a fragment of a `scene_sentence` /
`scene_short` line or a `grid_string` cell, and the rows learned "a small
line of text in the bubble". Suspect first: the sentence / short / grid
share, not the freeze and not the budget. Next, in order: (1) the seed
floor on the five strings; (2) the same 300 rows and freeze on
`scene_piece`-only data (0507 + 0305 tiers, no sentence / short / grid).
