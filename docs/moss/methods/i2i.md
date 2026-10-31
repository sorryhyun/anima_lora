# MOSS i2i (image to image)

An i2i LoRA learns a reference → target edit on MOSS-Image. The reference image is
VAE-encoded and goes into every forward (both CFG branches) as clean latent tokens on RoPE's
reserved `r = 1` axis, after the image and text rows. The flags below control it:

| Flag | Meaning | Stamped in the LoRA |
|---|---|---|
| `ref_scale` | reference side = target side / `ref_scale` (1 = full size, 2 = ¼ area) | checked against the cache |
| `ref_timestep` | the ref rows are modulated at this timestep in every block (0 = clean) instead of the step's | yes |
| `ref_isolate` | ref queries attend only to ref keys; image and text still read the ref | yes |
| `ref_vision` | the page also enters the text encoder through Qwen3.5's vision tower, ahead of the caption | no (cache recipe) |

`generate.py` and the ComfyUI Reference node read `ref_timestep` / `ref_isolate` from the
LoRA. `ref_scale`, `ref_vision` and the prompt format must match the cache the LoRA trained on.
With `ref_isolate` and a `ref_timestep`, the ref rows depend only on the page, so ComfyUI runs
them through the blocks once per page and caches their K/V (`ref_cache`).

## Tasks

Recipes: `configs/moss/i2i/<task>/{cache,train,generate}.toml`. The task → cache-request map
is `library.moss.requests.I2I_CACHE`.

| Task | Reference | Text encoder input | Status |
|---|---|---|---|
| `colorize` | B&W page, full size, isolated | the colour caption's layout JSON plus one text item per OCR line (no vision tower) | v4 recipe in the configs (`moss_colorize_3k_textref1_iso_muon_ema`, on the Hub as `colorize/moss_colorize_3k_v4`). v2 (`moss_colorize_3k_ocr`: half size, joint, vision tower) needs `ref_scale = 2`, `vision = true`: `project/finished/moss_lora/README.md` |
| `sanitize` | texted image, full size, isolated | one fixed caption for every pair: `no sfx, no dialogue, no watermark` | v3 on the Hub (`sanitize/moss_i2i_sanitize_v3`): the tone term, `docs/moss/methods/tone_loss.md` |
| `swap_text` | JA manga page, full size, isolated | layout JSON, one text item per line to letter (translated text + bbox) | 1,300-pair run (`moss_i2i_swap_text_1300_ema5e5`): English clean on the held-out pages, Korean down to one-syllable swaps |

## Running

```bash
make i2i-stage TASK=<task>                      # pair tree under post_image_dataset/i2i/<task>/
make i2i-preprocess TASK=<task> MODEL=moss      # cache (colorize: scripts/moss/colorize_cache.py; sanitize, swap_text: cache.py)
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/i2i/<task>/train.toml"
make daemon-run ARGS="scripts/moss/generate.py --config configs/moss/i2i/<task>/generate.toml \
    --lora output/ckpt_moss_i2i/moss_i2i_<task>.safetensors --ref_image <page or folder>"
```

`--ref_image` takes one image or a folder (one job per image and prompt). The output keeps
the reference's aspect ratio, with the area capped at `width × height`. The GUI's **i2i Preprocess**
(stage, preprocess) and **i2i Train** (train, test) tabs (`make gui-moss`, or MOSS mode in
`make gui`) run the same four steps. Staging is shared with
EasyControl: `i2i_tasks/README.md`.

## sanitize

Removes text, SFX, speech-bubble text and watermarks and repaints what was underneath.

**Data.** `make i2i-stage TASK=sanitize` mines in-artist near-twins
(`i2i_tasks/tools/near_twins/`, knobs in the `[staging]` table of
`i2i_tasks/configs/sanitize.toml`): a texted image (`{id}_tags`, the condition) and its
clean twin (`{id}_no_tags`, the target), listed in `staging/pairs.jsonl`. `add_identity_pairs`
adds some clean → clean pairs.

**Cache.** `caption_format = "fixed"` encodes `fixed_caption` for every pair instead of the
pair's own `.txt`. The reference carries the edit, so the caption is just a constant. Changing
it re-encodes only the TE files. The reference is full size (`ref_scale = 1`). Copies of
`fixed_caption` live in `generate.toml` (as `prompt`, with `prompt_format = "text"`) and in
the ComfyUI MOSS Toolkit (`FIXED_CAPTIONS["sanitize"]`); change all three together.

**Train.** `ref_timestep = 0`, `ref_isolate = true`, other fields at the `TrainRequest`
defaults. The first run (`moss_i2i_sanitize`) used 49 pairs from 17 artist folders, rank 32,
12 epochs = 588 steps, 37 min on 16 GB (12.6 GB peak reserved, swap 28).

**Result.** On four test images from outside the staging tree (not checked against the
training pairs), the first run removes title text, handwritten
annotations and their arrows, unbubbled dialogue, a large stamped "reproduction prohibited"
watermark and corner signature watermarks. It repaints the background, clothing and legs
underneath, and leaves glyph-like art in place (a halo's red symbols). Line work, faces and
colour come through unchanged: outside the text regions, the output is ~31 dB PSNR against
the input. Small doodle hearts next to annotations go with the text. The prompt barely
matters: the same page with the layout JSON of `no sfx, no text` (not the training caption)
came out nearly identical. Still, use the fixed caption, since it is what the LoRA saw.

**v3 (2026-10-28).** Later Muon runs edited better but brightened and cooled the whole
image. v3 adds the tone term late in training: `tone_weight 0.003`, `tone_start 0.5`, on
277 pairs (39 of them clean → clean, `post_image_dataset/moss/i2i_sanitize_idt`), Muon
lr 5e-5, 8 epochs = 2,216 steps. On the five held-out pages it averages ΔL −0.03 /
Δb −0.23 against the input outside the edit. Method and numbers: `docs/moss/methods/tone_loss.md`.
`configs/moss/i2i/sanitize/{cache,train}.toml` carry this recipe.

**ComfyUI.** [ComfyUI-MOSS-Toolkit](https://github.com/sorryhyun/ComfyUI-MOSS-Toolkit) 0.7.0
or later (0.6.x: Reference `task custom`, `ref_scale 1`), `example_workflows/moss-i2i-sanitize.json`:

```
MOSS-Image Loader ─ MODEL ─ MOSS i2i Reference (LoRA, task auto) ─────────────── MODEL ─┐
                  ─ CLIP ─┬ CLIP Text Encode (Caption Helper, task sanitize) ── positive ┤
                          └ CLIP Text Encode ("") ──────────────────────────── negative ┤ MOSS Spectrum KSampler
                  ─ VAE ──── MOSS i2i Reference ────────────────────────────── LATENT ──┘ (30 steps, CFG 4, skip 22)
Load Image → Reference.image
```

The Caption Helper needs no image. It registers only when `anime-tools` is installed in
ComfyUI's environment. LoRA + workflow per task: `sorryhyun/MOSS-I2I-LORAs` on the Hub
(`colorize/`, `sanitize/`, `decensor/`).

## swap_text

Re-letters a Japanese manga page in a translated edition's text, inside the balloons the
prompt names.

**Data.** `make i2i-stage TASK=swap_text` (`i2i_tasks/swap_text/prep.py`, knobs in
`i2i_tasks/configs/swap_text.toml`) registers hitocrawl edition twins (the JA page and
its `en` / `ko` edition) into a pool, `pool/<lang>/{stem}.{src,tgt}.webp` (symlinks to
the built pair) with `pool.jsonl`: a group of lettering is clean when it was read, would
paste without tearing, and the twin keeps the page's look around it. AnimeText blocks on
both sides plus the OCR lines mask the text, and outside that mask the twin must match
(`--max_nontext`: no darker fill or screentone, censor bar or mosaic one edition lacks,
translator's end card), while the lettering keeps its colour (`--max_ink_shift`, no
dark ↔ light flip). `assemble` samples `--sample` pool pages (0: all) into `staging/`.
Each target is the JA page with clean text groups pasted from
the twin. `full` pairs paste every clean group; `partial` pairs (`partial_ratio`) paste a
proper subset. The config sets `partial_ratio = 0` (2026-10-28): the unclean groups
(44 %) already stay Japanese and uncaptioned in a full pair. The caption is layout JSON with one text item
per pasted line, `abs_centralized` `yxyx` pixels of the staged page, read off the twin
(`--caption_ocr` reads the staged target instead). The current tree: 7,744 pairs from 254
works (en 5,990, ko 1,754; all full), with 2,374 lines the OCR misread. The runs below
up to `1300_ema5e5` trained on the earlier tree, 3,887 of whose pairs were partial.

**Cache.** `caption_format = "layout_json"` passes a caption that already is layout JSON
through as written (`library/moss/cache.py::is_layout_json`), so no `embeds_plain` is
stored and `plain_caption_rate` has no effect. The reference is full size
(`ref_scale = 1`).

**Train.** `ref_timestep = 0`, `ref_isolate = true`. The first run
(`moss_i2i_swap_text_muon_lr1e4`) used `pairs_100.jsonl`: 100 pairs, one page per work
(en 60 / ko 40, full 54 / partial 46). It trained rank 32, Muon at lr 1e-4,
`dc_isolate` off, 6 epochs = 600 steps, `blocks_to_swap = 28`, at ~4 s/step (40 min) and
12.2 GB peak on 16 GB, with joint sequences up to 8.7k tokens. The global loss does not track the
lettering. A median 3 % of latent tokens differ between target and reference (p90 7 %), so
once the copy is learned the loss falls toward 0 (0.25 in epoch 1, 0.03–0.04 from epoch 3)
and what remains is the text regions.

**Generate.** `generate.toml`, with the page's own staged caption (or a hand-written one) as
`--prompt` and `--width` / `--height` set to the page size, so the bboxes land where they
were measured.

**Result.** Final checkpoint, seed 0, on three held-out pages from works outside the 100
(a fourth, `ko` partial, was generated but not reviewed):

- English (full and partial): every line lands in its bbox, legible, broken to the balloon
  shape, in the page's lettering style; the Japanese underneath is gone and the balloon
  art kept. Misses: one wrong glyph (`MOVE` → `MOWE`); a short exclamation drawn with its
  own outline lost the outline along with the Japanese; `♡` marks absent from the OCR
  caption are not drawn.
- Korean (full, 3 lines): one exact; one with a syllable dropped and others swapped
  (`처음 오시는 분인가요?` → `처음 시는 문이니요?`); one drew the caption's OCR misread
  (`벌거` for `씻겨`) faithfully, in a typeset font and colour unlike the page's hand
  lettering. 40 Korean pages leave most Hangul syllables seen once or never.
- Outside the text, 22–25 dB PSNR against the target. The difference is the screentone's
  dot phase (the VAE round trip), not drift.

**Second run (`moss_i2i_swap_text_1300_ema5e5`).** `pairs_en300_ko1000.jsonl`: 1,300
pairs drawn at random (seed 0) from the staging tree, the four review works held out.
Muon at lr 5e-5, EMA (`ema_window 0.1`), plain MSE, 2 epochs = 2,600 steps, 173 min at
13.7 GB peak; cache 7 GB, 11 min. Between the two, `600_r2x05` (600 pairs, lr 1e-4,
`region_weight 2`, `rest_weight 0.5`, 3,600 steps) wrote ~5 wrong Korean syllables on the
same lines; its edit-region loss stalled at 0.15–0.18 from step 400 while the rest kept
falling. `region05` (`region_weight 0.5`, 100 pairs) was not reviewed.
Seed 0 on the four held-out pages (`output/tests/moss_i2i_swap_text_region/`), against
the first run:

- English (4 lines): all exact in e1, EMA and raw. EMA and raw keep the outline of the
  short exclamation (`Uwah…`).
- Korean (6 lines, the misread caption line aside): 3–4 wrong syllables against ~7 for the
  first run, all one-syllable swaps (no drops or reordering). All three checkpoints write
  `이건` → `이전` and `싼다` → `씬다`. The misread caption line is still drawn as captioned,
  in magenta type.
- e1 / EMA / raw differ by one syllable at most; raw letters heavier and slanted. The EMA
  final is the default.

**Full pairs only (`moss_i2i_swap_text_1300_full_ema5e5`).** The same 1,300 stems
restaged with `partial_ratio = 0` (`pairs_en300_ko1000_full.jsonl`), same recipe, 2,600
steps at 13.7 GB peak. Seed 0 on the four held-out pages, against `1300_ema5e5`:

- English: EMA exact on all 4 lines. e1 draws `Uwah…`'s outline faint; raw writes
  `YOU'RE TE BEST`.
- Korean (5 lines, the misread caption line aside): wrong syllables 1300 EMA 3, full e1 2,
  full EMA 2, full raw 3 plus a doubled syllable (`뭔뭔가요`). `싼다` → `씬다` stays in
  every checkpoint of both runs. The errors move: full EMA gets `분인가요` right
  (1300: `분인나요`) and writes `이건` as `이컨` (1300: `이전`), so one seed on five lines
  cannot separate the two. The misread line is still drawn as captioned, in magenta.

**Gradient accumulation (`moss_i2i_swap_text_en200_ko800_rw2_mod16_ga4_lr1e4`).**
`en200_ko800_rw2_mod16_ema5e5` (`pairs_en200_ko800.jsonl`, `region_weight 2`,
`mod_rank 16`, Muon 5e-5, batch 1) retrained with `grad_accum 4` and lr 1e-4 (√batch),
the rest of the request identical: 2,000 samples = 500 updates, 134 min. Same seed, so
both runs draw the same samples at the same σ (per-sample region loss correlates 0.95
between them); the logged loss is per sample at any `grad_accum`.

- Loss. The copy is learned ~2× later in samples (rest loss at reference level by ~500
  against ~200) and faster per update. The edit region never drops below the reference:
  +0.003 to +0.010 through epoch 2, both flat at 0.15–0.16. Epoch means 0.126 / 0.034
  against 0.088 / 0.031.
- Seed 0 on the four held-out pages, EMA: English exact on all 4 lines in both. Korean
  wrong syllables 3 against 2: `분인나요` (reference `분이가요`), `이건` → `이큰`
  (reference exact), `싼다` → `씬다` in both. Line breaks follow the balloon a little
  better, letters heavier, `Uwah…`'s outline nearly gone.

No gain. The errors that move between checkpoints (`이건` → `이전` / `이컨` / `이큰`,
`분인가요`) did not shrink with a 4× averaged gradient, and `싼다` → `씬다` held.

**Syllable coverage and tokenisation.** In `pairs_en300_ko1000_full.jsonl` the 959 `ko`
pairs (99 works) hold 30,180 Hangul syllables, 852 distinct, 138 seen once. The syllables
the runs keep missing are not rare: `건` 76, `인` 70, `분` 184, `싼` 21 (`씬` 2). The
Qwen3.5 tokenizer splits `싼` into two byte tokens while `씬` is one token, and writes
`인가요` as a single token. `건` → `전` is a single-syllable token drawn as a near
glyph. The whole pool holds 1,684 `ko` pairs (107 works).

Next: a larger held-out `ko` set scored by OCR, the errors split by token kind
(byte-split / multi-syllable / single-syllable), to choose between exposing the
syllables to the TE and a glyph-level loss; drop misread lines from the captions
(Hangul-for-Hangul misreads such as `벌거` / `씻겨` pass the JA-script check).

**ComfyUI.** Toolkit 0.7.0 or later: the sanitize wiring, with the layout-JSON prompt as
the positive (typed into `CLIP Text Encode`, or through Caption Helper `task layout`, which
passes layout JSON through unchanged).

## More

- Probes and design notes: `project/finished/moss_lora/done/i2i.md`. Colorize runs: `project/finished/moss_lora/`
  (`plan.md`, `colorize_ocr.md`).
- Ref forward: `library/moss/model.py` (`ref=`, `_ref_only`). The ComfyUI copy is
  `reference.py` in the toolkit.
