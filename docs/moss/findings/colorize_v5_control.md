# MOSS colorize v5: control captions in place of the OCR caption

Status: shipped, 2026-11-05. The arm (accepted 2026-11-03, renders in
`output/moss_i2i_eval/colorize/results/control_1k/`) settled the recipe; v5 trained on it
and is the default in `configs/moss/i2i/colorize/{cache,train,generate}.toml` and
`ColorizeCacheRequest` (`captions = "control"`, `variants = 4`). Code:
`library/moss/colorize.py`, tests `tests/test_moss_colorize_control.py`. Results:
[§ Results](#results).

## Why

v4 (`moss_colorize_3k_textref1_iso_muon_ema`, Hub `colorize/moss_colorize_3k_v4`) trained
every step on an OCR caption: the colour caption's layout JSON plus one text item per OCR
line, text and box (`project/finished/moss_lora/README.md` § colorize v3). Its lettering comes
from the full-size isolated reference, not from the caption:

- With `comic` alone (the shipped generate prompt, `configs/moss/i2i/colorize/generate.toml`),
  v4 draws all 7 lines of the two Korean eval targets from the reference. With the right text
  as layout items, ` 이건` becomes `어건` on 1 of 3 seeds; `벌거` in the prompt does not
  override the reference's `씻겨` ([ko_token_split.md](ko_token_split.md)).
- v3 draws "H-HEY," where the OCR read "H-HE".

So the OCR text costs a reader pass at inference, puts Hangul through the TE's token split, and
can only override the reference with a misread. The caption's job can move from naming the
lettering to control: which colours go where, and where the lettering sits.

## Captions

Each page's TE file holds four draws (`embeds`, `embeds_v1..v3`) and a bare caption
(`embeds_plain`). Training takes the bare caption at `plain_caption_rate 0.2` and one of the
four draws evenly otherwise (`caption_variant_rate 0.75`). No OCR text anywhere.

| | always | pool (each draw keeps 20 %, one at least) | bare (20 % of steps) |
|---|---|---|---|
| illustration | — | colour + series tags of the target's caption (`filter_to_colors_and_protected`, no comic tag) | empty prompt |
| manga page | `comic`; a box-only `text` item per OCR line of the B&W page | a `character` item per detected character, box + its hair / eye / clothing colours | `comic` + the text items |

- **Text boxes** are swap_text's caption lines without the text: the OCR engine
  (`load_reread_engine`: AnimeText blocks, SFX reader re-reading them) on the condition,
  filtered by `keep_line` (moved from `i2i_tasks/swap_text/ocr.py` into
  `library/moss/colorize.py`, so both tasks share it). Cached as `text_boxes.json` with the
  reads, for inspection.
- **Character boxes**: the position stage's SAM3 subject detector (`girl` soft prompt) on the
  colour target, mask-blanked crop, Anima Tagger kept tags that are colours (stored
  unfiltered in `characters.json`), minus skin (`dark-skinned …` too), background, sky,
  border at caption time (`is_character_colour`). The arm's cache predates the
  `-skinned` rule, so `dark-skinned female` sits in some of its manga draws.
- Items are layout JSON in MOSS's `abs_centralized` `yxyx` on the target canvas:
  `{"label":"character","caption":"pink hair, white shirt","bbox":[…]}`,
  `{"label":"text","bbox":[…]}`.

Since no Hangul reaches the TE, `token_split` does not apply to this cache.

## The arm (deciding run)

| | |
|---|---|
| cache | today's `configs/moss/i2i/colorize/cache.toml` at `n = 500`, `manga = 500` (the arm's own config was dropped after v5): 500 illustrations + 500 manga pages, holdout 8 + 4 works, seed 0 → `post_image_dataset/moss/i2i_colorize_control_1k` (job `20261103-170540-020fa4`) |
| train | 1250 steps (the 1-hour arm `moss_colorize_1k_textref1_iso`'s length), Muon 5e-5, rank 32, `ref_isolate`, int8 stream + fp32, swap 36 / budget 0.1 (job `20261103-170540-389002`) |
| control | v4 with `comic`, same seed and pages; v4's cache is gone, so its holdout cannot be reused |
| renders | `output/moss_i2i_eval/colorize/results/control_1k/`: `{v4,arm}_holdout*`, `{v4,arm}_wild2_comic_s1` (v4: `v4_wild2_s1`), `arm_wild2_boxes_s1` (comic + text boxes, `prompts.json`), `arm_eval` (`colorize_eval.py`, draw 0 / bare) |

Confounds: 1250 steps against v4's 10000, and the int8 base against v4's bf16. Colour will be
weaker at this length (the 1k arm was); the arm decides lettering, not colour.

**Gate (per glyph, sheets viewed):** on the 4 held-out manga pages and `wild2/` (10 real Korean
pages), with `comic` + text boxes and with `comic` alone, the arm keeps every balloon v4 keeps,
with no new dropped or swapped glyph. Informative, not gating: holdout chroma error, and whether
a `character` item's colour lands on that character (holdout draw 0 against the target).

## v5

- Cache: v4's scale (1667 illustrations + 1667 manga, `variants 4`, seed 0). The arm's pages
  are a prefix of that draw (same shuffle, `rest[:n]`) and its holdout is the same 12, so an
  `out` of `post_image_dataset/moss/i2i_colorize_control_1k` reuses the arm's latents, refs,
  OCR reads and SAM3 tags. Delete its `manga__*.moss_te.safetensors` first so the `-skinned`
  rule reaches those draws (TE only; `characters.json` holds the unfiltered tags).
- Train: v4's schedule (3 epochs, 10000 steps, Muon 5e-5, EMA window 0.1) on the int8 base.
  Planned at `blocks_to_swap 8`, budget 0.1 (the arm's fit report: 10.9 GB peak at swap 36
  with 3.15 GB free); ran at swap 34, budget 0.15. The arm ran 2.4–2.5 s/step at swap 36;
  v5 ran 2.2 s/step, 374 min, peak 13.8 GB, against v4's 11.2 h.
- Text rows: up to 905 on a manga page (37 lines × ~23 tokens per box item), median 326.
- Eval: `bench/moss_i2i_eval/colorize_eval.py` (arms `lora_ocr` = draw 0, `lora_page` = bare)
  plus `wild/` and `wild2/`, against v4.
- Stamp the caption mode in the LoRA (`colorize_captions = control`) so the caption helper can
  pick the format from the file, as `ref_isolate` / `prompt_tail` are read today.

## Results

`moss_colorize_v5` (`output/ckpt_moss_i2i/moss_colorize_v5.safetensors`), renders and sheets in
`output/moss_i2i_eval/colorize/results/v5/`. v4 has no score on this holdout: its cache is
gone.

| holdout (`colorize_eval.py`) | chroma_err (all) | chroma_err (manga) | luma_r |
|---|---|---|---|
| v5 | 11.4 | 9.5 | 0.994 |
| the 1k arm | 14.6 | 16.2 | |

- **Lettering.** Korean holdout page `manga__821938_ko_0013`, prompt `comic`, seed 1: v5 has
  1 wrong syllable (정말 → 청말, which v4 and the arm share); v4 has 3 (싸버렸 → 싸비렸,
  정 → 청, 깊숙히 → 깊슥히), the arm 5–6.
- **Target colours.** v5 matches the target where v4 does not on `ko_0013` (blonde target; v4
  pink) and `2550520_ja` (grey stockings).
- **Character items** (draw 0) put their colours on the character: `en_0001` blue bikini,
  `3140189` silver hair.
- **Real B&W pages** (`wild2/`, 10 Korean pages): with `comic` alone v5 adds little colour,
  less than v4. Random `character` colour items per character on SAM3 boxes add colour, with
  some bleed between overlapping boxes (`v5_wild2_randcolour_s1`, `random_colours.json`).

## Shipping

### anime_tools

The OCR floors are copied in three places today: `readable` in `library/moss/colorize.py`,
`keep_line` beside it (swap_text's Latin / Hangul rescues), and `readable` in
ComfyUI-MOSS-Toolkit's `moss_layout.py` ("change both together"). The Toolkit's copy lacks
`keep_line`'s rescues, so it would box fewer lines than v5 trained on.

1. Move `readable`, `keep_line` and their constants (`LOW_DET_SCORE`, `LATIN_MIN_GLYPH`,
   `LATIN_MIN_SCORE`) into anime_tools (`anime_tools.ocr`), with the tests in
   `tests/test_swap_text_regions.py`.
2. Move the colour-tag filter (the trainer's `i2i_tasks.colorization.color_caption`:
   `is_color_tag`, `filter_to_colors_and_protected`, `is_character_colour`) into
   `anime_tools.captions`, so the Toolkit can build the illustration caption the same way.
3. Cut a tag, move the trainer's pin, import from there in the trainer and in swap_text.

### ComfyUI-MOSS-Toolkit

1. `MOSSCaptionHelper`: a `colorize_v5` task (or `colorize` keyed off the LoRA stamp).
   Manga: `comic` + a box-only text item per `keep_line` line of the page, no text, no dedupe.
   Illustration: the caption's colour + series tags as layout JSON. Optional character
   colours: a STRING of `box: tags` rows, or SAM3 boxes numbered on a preview with colours given
   per number (open: which UI; the first ships with the text boxes alone).
2. The reference node needs no change (`ref_scale 1`, `ref_isolate` from the stamp).
3. `example_workflows/moss-i2i-colorize-v5.json`, README, `_vendor/` via `make vendor-sync`.

### Hub and this repo

- **Done** (2026-11-05): `colorize/moss_colorize_v5.safetensors` on `sorryhyun/MOSS-I2I-LORAs`
  with the workflow `colorize/moss-i2i-colorize-v5.json`.
- **Done** (2026-11-05): `configs/moss/i2i/colorize/{cache,train,generate}.toml` on the control
  recipe (generate's prompt stays `comic`), `ColorizeCacheRequest` defaults,
  `docs/moss/methods/i2i.md` colorize row, the GUI's colorize task text (`gui/moss/strings_*.py`).
- Open: text boxes at generate time need the OCR there (a `generate.py` option); the
  `colorize_captions` stamp.
