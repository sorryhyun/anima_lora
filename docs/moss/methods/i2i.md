# MOSS i2i (image to image)

An i2i LoRA learns a reference → target edit on MOSS-Image. The reference image is
VAE-encoded and goes into every forward (both CFG branches) as clean latent tokens on RoPE's
reserved `r = 1` axis, after the image and text rows.

| Flag | Meaning | Stamped in the LoRA |
|---|---|---|
| `ref_scale` | reference side = target side / `ref_scale` (1 = full size, 2 = ¼ area) | checked against the cache |
| `ref_aspect` | `target`: the reference is cropped to the target's aspect; `native`: it keeps its own, at the target's area / `ref_scale`² (cache and generate; the ComfyUI Reference node crops) | no (cache recipe) |
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
| `colorize` | B&W page, full size, isolated | control captions, no OCR text (no vision tower). Manga page: `comic`, optionally a box-only `text` item per lettered line and `character` items (box + hair / eye / clothing colours). Illustration: its colour + series tags | v5 recipe in the configs (`moss_colorize_v5`, int8 base; on the Hub as `colorize/moss_colorize_v5` with the workflow `colorize/moss-i2i-colorize-v5.json`): [colorize_v5_control.md](../proposal/colorize_v5_control.md). v4 (`colorize/moss_colorize_3k_v4`: OCR text items in the caption) is `captions = "ocr"`, `variants = 3`; v2 (`moss_colorize_3k_ocr`: half size, joint, vision tower) adds `ref_scale = 2`, `vision = true`: `project/finished/moss_lora/README.md` |
| `sanitize` | texted image, full size, isolated | one fixed caption for every pair: `no sfx, no dialogue, no watermark` | v3 on the Hub (`sanitize/moss_i2i_sanitize_v3`): the tone term, `docs/moss/methods/tone_loss.md` |
| `character_ref` | a solo image of the character by the same artist (the target: solo, or with an unnamed `1boy`, captioned `add 1boy`), own aspect (`ref_aspect native`), full or half size by pair (`ref_scale_alt 2`), isolated | the target's caption without character, series, artist and meta tags, and without the tags the reference's caption shares | 1,000 pairs (308 with `add 1boy`), 88 artists, any rating. First run `moss_i2i_character_ref_e1` (1 epoch = 1,000 steps, int8, swap 36, budget 0.15, 27.6 min, 12.7 GB): on two held-out OCs neither the character nor the style carries over yet; the speech-bubble text follows the prompt (`output/moss_i2i_eval/character_ref/`) |
| `swap_text` | JA manga page, full size, isolated | layout JSON, one text item per line to letter (translated text + bbox) | 1,300-pair run (`moss_i2i_swap_text_1300_ema5e5`): English clean on the held-out pages, Korean down to one-syllable swaps |

## Running

```bash
make i2i-stage TASK=<task>                      # pair tree under post_image_dataset/i2i/<task>/
make i2i-preprocess TASK=<task> MODEL=moss      # cache (colorize: scripts/moss/colorize_cache.py; sanitize, swap_text: cache.py)
make daemon-run ARGS="--stall-timeout 900 scripts/moss/train.py --config configs/moss/i2i/<task>/train.toml"
make daemon-run ARGS="scripts/moss/generate.py --config configs/moss/i2i/<task>/generate.toml \
    --lora output/ckpt_moss_i2i/moss_i2i_<task>.safetensors --ref_image <page or folder>"
```

`--ref_image` takes one image, a folder or a quoted glob (one job per image and prompt). The output keeps
the reference's aspect ratio, with the area capped at `width × height`. For an eval set in one
job, `--ref_prompts` gives each reference the `.txt` beside it (`<page>.src.webp` →
`<page>.txt`) and `--seeds 0,1,2` renders every page once per seed
(`project/lora_blocks/src/hangul_eval.sh`). The GUI's **i2i Preprocess**
(stage, preprocess) and **i2i Train** (train, test) tabs (`make gui`, Model set to MOSS-Image) run the same four steps. Staging is shared with
EasyControl: `i2i_tasks/README.md`.

## sanitize

Removes text, SFX, speech-bubble text and watermarks and repaints what was underneath.

**Data.** `make i2i-stage TASK=sanitize` mines in-artist near-twins
(`i2i_tasks/tools/near_twins/`, knobs in the `[staging]` table of
`i2i_tasks/configs/sanitize.toml`): a texted image (`{id}_tags`, the condition) and its
clean twin (`{id}_no_tags`, the target), listed in `staging/pairs.jsonl`. `add_identity_pairs`
adds some clean → clean pairs.

**Cache.** `caption_format = "fixed"` encodes `fixed_caption` for every pair instead of the
pair's own `.txt`; the reference carries the edit. Changing it re-encodes only the TE files.
The reference is full size (`ref_scale = 1`). Copies of `fixed_caption` live in `generate.toml` (as `prompt`, with `prompt_format = "text"`) and in
the ComfyUI MOSS Toolkit (`FIXED_CAPTIONS["sanitize"]`); change all three together.

**Train.** `ref_timestep = 0`, `ref_isolate = true`, other fields at the `TrainRequest`
defaults. `configs/moss/i2i/sanitize/{cache,train}.toml` carry the v3 recipe (2026-10-28): 277
pairs (39 of them clean → clean, `post_image_dataset/moss/i2i_sanitize_idt`), Muon lr 5e-5, 8
epochs = 2,216 steps, `ema_window 0.1`, with the tone term late in training (`tone_weight 0.003`,
`tone_start 0.5`), since earlier Muon runs edited better but brightened and cooled the whole image.

**Result.** On the five held-out pages v3 averages ΔL −0.03 / Δb −0.23 against the input outside
the edit: `docs/moss/methods/tone_loss.md`.

The first run (`moss_i2i_sanitize`: 49 pairs from 17 artist folders, rank 32, 12 epochs = 588
steps, 37 min on 16 GB, 12.6 GB peak reserved, swap 28), on four test images from outside the
staging tree (not checked against the training pairs), removes title text, handwritten
annotations and their arrows, unbubbled dialogue, a large stamped "reproduction prohibited"
watermark and corner signature watermarks. It repaints what was underneath and leaves
glyph-like art in place (a halo's red symbols); small doodle hearts next to annotations go with
the text. Outside the text regions the output is ~31 dB PSNR against the input. The same page
with the layout JSON of `no sfx, no text` (not the training caption) came out nearly identical;
use the fixed caption anyway, since it is what the LoRA saw.

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
the built pair) with `pool.jsonl`. A group of lettering is clean when it was read, would
paste without tearing, and the lettering keeps its colour (`--max_ink_shift`, no
dark ↔ light flip of the inks themselves; outlined lettering keeps its ink). AnimeText
blocks on both sides plus the OCR lines mask the text. A group pastes **text-only**
(`--paste text`, the default since 2026-11-08): its diff region cut down to that mask, so
the art stays the JA page's even when the edition is another scan, checked by
`seam_text` (`--max_seam_text` 8: the tone step on the paper pixels around the mask; the
ring's line art is left out, since a box border a few px off on another scan is not a
tear, and with it counted resid-8 pages lost their dialogue at 8–11) and `leftover_text`.
When the box cuts through art and seams, it falls back to
the whole region, where the twin must also match outside the text (`--max_nontext`: no
darker fill or screentone, censor bar or mosaic one edition lacks, translator's end card)
and carry at most `--max_free` (3,000) non-text pixels: on a twin from another scan the
diff region spreads over redrawn balloons and boxes whose median |ΔLab| passes `nontext`,
and the fallback pasted the other edition's art (429 groups, up to 24 % of a page, before
the cap on 2026-11-08). A Japanese line (`ja_line`, by script majority: kana / CJK alone, or at least two of
them outnumbering the Hangul — a Korean line keeping a `成人向け` or a `凸` is Korean, since
2026-11-09) inside a region marks it misread only when its box changed between the
editions (`--retained_ja`, shift-tolerant change ≥ 0.05); an unchanged one is JA lettering
the twin kept beside the translation (`はっ`, `えへへ`), out of the caption but not dropping
the Korean line next to it.
With text-only pastes the page gates open to `max_resid = 12`, `max_uncovered = 0.05`
(config): on 2026-11-08 ko 1,781 → 2,302 pages, 9,336 → 11,695 pasted groups, 99 % of them
text-only. That tree is `post_image_dataset/i2i/swap_text_relax` (ko only; `work/` is a symlink
to `swap_text_v3/work/`; `prep.py`'s default `--root` is still `swap_text_v3`). Its `ocr.json`
was re-read with the v5 reader (line breaks kept as `\n` in the caption's text items),
proofread by `audit.py` and restaged on 2026-11-09: 5,660 pages from 175 works, 29,087 pasted
groups (28,899 text-only). `assemble` samples `--sample` pool pages (0: all) into `staging/`; it
wipes and re-pastes it, so a restage gives the targets new mtimes (MOSS latent caches invalidate
on image mtime) and `pairs.jsonl` is rewritten, while the `pairs_*.jsonl` subsets are derived from
it by hand afterwards. The cache for the dense Korean pages plus v3's en400 is
`configs/moss/i2i/swap_text/cache_v6.toml` (`i2i_swap_text_v6`, syllable split).
Each target is the JA page with clean text groups pasted from the twin: `full` pairs paste
every clean group, `partial` pairs (`partial_ratio`) a proper subset. The config sets
`partial_ratio = 0` (2026-10-28): the unclean groups (44 %) already stay Japanese and
uncaptioned in a full pair. The caption is layout JSON with one text item per pasted line,
`abs_centralized` `yxyx` pixels of the staged page, read off the twin (`--caption_ocr` reads
the staged target instead). The `swap_text_v3` tree (built 2026-11-01): 8,282 pairs from 241 works (en 6,501,
ko 1,781; all full), with 2,761 lines the OCR misread. The runs below up to `1300_ema5e5`
trained on the earlier tree, 3,887 of whose pairs were partial.

**Cache.** `caption_format = "layout_json"` passes a caption that already is layout JSON
through as written (`library/moss/caching/sources.py::is_layout_json`), so no `embeds_plain`
is stored and `plain_caption_rate` has no effect. The reference is full size
(`ref_scale = 1`).

**Train.** `ref_timestep = 0`, `ref_isolate = true`.

**Generate.** `generate.toml`, with the page's own staged caption (or a hand-written one) as
`--prompt` and `--width` / `--height` set to the page size, so the bboxes land where they
were measured.

**Runs.** Each review is seed 0 on the four held-out pages (first run: three of them, `ko`
partial unreviewed). English counts 4 lines from `1300_ema5e5` on. Korean counts leave out the
line the OCR misread (`벌거` for `씻겨`), which the 1300 runs still draw as captioned, in magenta:
6 lines for `1300_ema5e5`, 5 for `1300_full_ema5e5`. Names drop the `moss_i2i_swap_text_` prefix,
`…` is `en200_ko800_rw2_mod16`, and every row from `1300_ema5e5` on is Muon at the lr shown with
`ema_window 0.1`, EMA final. Renders: `output/tests/moss_i2i_swap_text_region/`,
`output/moss_i2i_eval/swap_text/results/`.

| run | pairs | recipe | steps | time, peak | English | Korean wrong syllables |
|---|---|---|---|---|---|---|
| `muon_lr1e4` | `pairs_100.jsonl`: 100 (en 60 / ko 40) | Muon lr 1e-4, swap 28 | 600 | 40 min at ~4 s/step, 12.2 GB | one wrong glyph (`MOVE` → `MOWE`) | ~7, with a drop |
| `600_r2x05` | 600 | lr 1e-4, `region_weight 2`, `rest_weight 0.5` | 3,600 | | | ~5 |
| `1300_ema5e5` | `pairs_en300_ko1000.jsonl`: 1,300 at random (seed 0), review works held out | lr 5e-5, plain MSE | 2,600 | 173 min, 13.7 GB; cache 7 GB, 11 min | exact | 3–4, one-syllable swaps |
| `1300_full_ema5e5` | the same stems, `partial_ratio = 0` (`pairs_en300_ko1000_full.jsonl`) | as above | 2,600 | 13.7 GB | exact | 2 (1300: 3 on these lines) |
| `…_ema5e5` | `pairs_en200_ko800.jsonl` | lr 5e-5, `region_weight 2`, `mod_rank 16`, batch 1 | 2,000 | | exact | 2 |
| `…_ga4_lr1e4` | as above | `grad_accum 4`, lr 1e-4 (√batch) | 2,000 samples = 500 updates | 134 min | exact | 3 |
| `…_ema5e5_fp32s` | as above | `…_ema5e5` + fp32 stream | 2,000 | 145 min at 4.34 s/step | 1 wrong line (drops `THE`) | 3 |
| `…_ema5e5_int8_e3` | as above | `fp32s` on the GPTQ int8 base, swap 38, budget 0.1, EMA on the card | 3,000 | 135 min at 2.71 s/step | exact on the bf16 base apart from `BEST…` losing a dot | 3 |

- **The global loss does not track the lettering.** A median 3 % of latent tokens differ between
  target and reference (p90 7 %), so once the copy is learned the loss falls toward 0 (0.25 in
  epoch 1, 0.03–0.04 from epoch 3 of the first run) and what remains is the text regions.
  `600_r2x05`'s edit-region loss stalled at 0.15–0.18 from step 400 while the rest kept falling.
- **Checkpoints.** e1 / EMA / raw differ by one syllable at most; raw letters heavier and slanted.
  The EMA final is the default.
- **Errors move between checkpoints.** `이건` comes out `이전`, `이컨`, `이큰`, `이긴`; `분인가요`
  comes out `분인나요`, `분이가요`, `분인머요`, and `1300_full_ema5e5` gets it right. `싼다` → `씬다`
  stays in every checkpoint of the 1300 runs and in `…_ga4_lr1e4` and its reference. One seed on
  five lines cannot separate the two 1300 runs.
- **`grad_accum 4` is no gain.** Same seed, so both runs draw the same samples at the same σ
  (per-sample region loss correlates 0.95; the logged loss is per sample at any `grad_accum`).
  The copy is learned ~2× later in samples (rest loss at reference level by ~500 against ~200)
  and faster per update; the edit region never drops below the reference, +0.003 to +0.010
  through epoch 2, both flat at 0.15–0.16. Epoch means 0.126 / 0.034 against 0.088 / 0.031. The
  moving errors did not shrink with a 4× averaged gradient.
- **The int8 base does not change training; the third epoch helps slightly.** Both runs draw the
  same σ, and over steps 1,000–2,000 their per-sample region loss correlates 0.989 with a mean
  difference of −0.0001. Region loss goes from 0.153 at steps 1,500–2,000 to 0.144–0.145 over the
  third epoch, rest loss from 0.012 to 0.006–0.007. `싼다` is written, as in `fp32s`; the three
  wrong syllables are `분인머요`, `이건` → `이긴`, `아타고씨` → `아타고싸`. With no second-epoch
  checkpoint, the epoch's effect on the letters is not separated. Generating on the int8 base
  gives the same Korean and draws `YOU'RE THE` run together.
- **Misses outside the syllables.** `♡` marks absent from the OCR caption are not drawn. `Uwah…`'s
  outline is lost or faint on some checkpoints. Outside the text, 22–25 dB PSNR against the target
  in the first run; the difference is the screentone's dot phase (the VAE round trip), not drift.

**Syllable coverage and tokenisation.** In `pairs_en300_ko1000_full.jsonl` the 959 `ko`
pairs (99 works) hold 30,180 Hangul syllables, 852 distinct, 138 seen once. The syllables
the runs keep missing are not rare: `건` 76, `인` 70, `분` 184, `싼` 21 (`씬` 2). After a
space the Qwen3.5 tokenizer splits `싼` and `씬` into two byte tokens, and it writes
`인가요` and ` 이건` as single tokens. Re-splitting only the tokens fixes `분인가요`, while
4 epochs at `grad_accum` 2 move nothing: [ko_token_split.md](../findings/ko_token_split.md).
`token_split = "syllable"` on the cache and on `generate` applies the split; the LoRA stamps
it and `generate` follows the stamp. On an unsplit LoRA it moves line breaks. Trained on the
split at 200 pairs, Korean wrong syllables go from 8 to 2, with a new `처음` -> `처즘`.

Next: a larger held-out `ko` set scored by OCR, the errors split by token kind
(byte-split / multi-syllable / single-syllable); the split at full size; drop misread lines from the captions
(Hangul-for-Hangul misreads such as `벌거` / `씻겨` pass the JA-script check).

**ComfyUI.** Toolkit 0.7.0 or later: the sanitize wiring, with the layout-JSON prompt as
the positive (typed into `CLIP Text Encode`, or through Caption Helper `task layout`, which
passes layout JSON through unchanged).

## More

- Probes and design notes: `project/finished/moss_lora/done/i2i.md`. Colorize runs: `project/finished/moss_lora/`
  (`plan.md`, `colorize_ocr.md`); v5 and its eval: [colorize_v5_control.md](../proposal/colorize_v5_control.md).
- Ref forward: `library/moss/model.py` (`ref=`, `_ref_only`). The ComfyUI copy is
  `reference.py` in the toolkit.
