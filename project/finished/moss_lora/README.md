# moss_lora — MOSS-Image on a 16 GB card

**Not Anima.** MOSS-Image (OpenMOSS, Apache-2.0, released 2026-10-11): 5B single-stream
DiT, frozen Qwen3.5-9B TE, frozen FLUX.2 VAE (`OpenMOSS-Team/MOSS-Image-Base`, `-SFT`).
Code map `library/moss/CLAUDE.md`; user doc `docs/moss/README.md`.

**Finished 2026-10-28** (opened 2026-10-16): the port, Spectrum, LoRA training, channel
scaling (later removed), DAVE, the GUI / ComfyUI surface, the INT8 TE, colorize v4 and
sanitize v3 all shipped. The A/B and eval scripts later work reuses moved to `bench/moss_ab/`
(LoRA / sampler grids, the Space prompts) and `bench/moss_i2i_eval/` (colorize eval, tone
drift); `src/` keeps the line's own probes and trainers, runnable by path. Open remainder:
compile P4 / P5 (`compile.md`), stamping `ref_scale` in the LoRA metadata (§ Deploy 2), and
the open questions at the end.

## Sub-lines

| Sub-line | State | Log / code |
|---|---|---|
| i2i / colorize (VAE reference on RoPE `r`) | v3 recipe shipped on the Hub as `moss_colorize_3k_v4` (§ below) and in the configs | `done/i2i.md`, `colorize_ocr.md`, `bench/moss_i2i_eval/colorize_*.py` |
| Correction training (arXiv 2610.14136), dual LoRA | joint D + C holds to K 8; CFG 1.5 + K 2 ≈ CFG 3 at 2× forwards (one seed) | `correction.md`, `src/correction*.py` |
| TE quantization | INT8 W8A8 (SmoothQuant + act-aware GPTQ) is the default TE | `src/te_*.py`; comparison runs + drivers in `_archive/bench/moss_te_fp8/` |
| Technical-report digest | reference | `report_notes.md` |
| Compile: FA2 + range-marked token axis (Anima's `dynamic=False` design) | P1 + P3 landed (FA2, strict range on `dynamic=True`, Anima pins): one graph, no stall; next P4 (generate), P5 (bench + tests) | `compile.md` |

## colorize v3 — the default i2i recipe from here on

v1 shared the image's timestep on the ref; v2 (`moss_colorize_3k_ocr`) gives it t = 0 and
sends the page through the vision tower as well. v3 drops the vision tower and gives the
ref the target's full size, isolated. This recipe becomes the default for i2i training
and for what ships; v3 is it trained long.

| | v2 (`moss_colorize_3k_ocr`) | v3 |
|---|---|---|
| TE input | page via vision tower (~1000 rows at 1 MP) + caption | caption only, the OCR layout over the full / sparse (80 % dropped) / bare colour caption, sparse and bare split evenly (`variants = 3`), bare keeping one comic tag on a comic page; no `embeds_plain` (no empty-prompt row) |
| Ref | 1/2 per side (`ref_scale 2`), joint attention | full size (`ref_scale 1`), `ref_isolate`: ref queries see only ref keys |
| Ref modulation | t = 0 | t = 0 |
| Uncond skip-22 | whole block, ref included | image + text only; the isolated ref still runs block 22, so cond and uncond read the same ref (`model.py::_ref_only`) |

### Basis: 1-hour arm `moss_colorize_1k_textref1_iso` (2026-10-23)

980 pairs (half manga) drawn as v2's cache, v2's 12 held-out pages (`--holdout_json`),
v2's OCR reads; rank 32, lr 1e-4, 1250 steps. 79.8 min at 3.83 s/step (v2 recipe 2.9),
swap 28 / budget 0.1 (0.15 OOMs on the 9.5k-token worst sample), peak 13.0 GB. Eval
seed 1, ER-SDE + Spectrum, 30 steps, CFG 3, against v2's eval (10000 steps):

| | v2 `lora_cap` | v3 arm `lora_ocr` | v3 arm, uncond skip on the ref too |
|---|---|---|---|
| luma r (128 px) | 0.977 | **0.989** | 0.989 |
| chroma (targets ~16) | 18.7 | 14.9 | 15.0 |
| chroma error, 8 illus. / 4 manga | **12.8 / 11.6** | 14.2 / 14.6 | 13.8 / 14.4 |

- **Lettering: every OCR'd balloon exact** on the 4 manga pages, per glyph. v2 garbles
  623751 ("ニューシネ未末", "くすぐわたいよ", ☆ dropped) and 403801 (", I" dropped, "ROOI",
  "SETACY WAE TOO MUC"). v3 also draws "H-HEY," where the OCR read "H-HE": the
  full-size ref carries the glyphs, not only the caption.
- **Colour: weaker than v2.** Desaturated; 2550520's warm lamp turns to a blue night
  cast, yomu's pink background goes grey-blue. Lettering colour is sometimes invented
  (623751: red where the target is black). The arm had 1/8 of v2's steps and no vision
  tower, and does not separate the two.
- Skip-on-ref vs image + text only: lettering identical, colour within noise. Image +
  text only stays the default; it is what lets one ref KV cache serve both branches.

### Training v3

- Cache: v2's draw (`n 1667`, `manga 1667`), v2's holdout (`--holdout_json`) and OCR
  reads (`--ocr_from`), `--ref_scale 1 --no-vision`.
- Train: v2's schedule (lr 5e-5, 3 epochs / 10000 steps) with `--ref_isolate`, swap 28 /
  budget 0.1. ~10.6 h at 3.83 s/step.
- Eval: `colorize_eval.py --arms lora_ocr,lora_sparse,lora_bare,lora_page,lora_ocr_refskip`
  against v2's `output/tests/colorize_moss_colorize_3k_ocr`, plus `colorize_wild.py` on
  the real pages.
- Ship if lettering holds per glyph and chroma error is not worse than v2's. If colour
  stays behind at 10000 steps, run the vision-on arm (full-size isolated ref + vision
  tower, 1-hour budget) before shipping, to see whether the vision tower carries the hue.

### Result: `moss_colorize_3k_textref1_iso_muon_ema` (2026-10-26), shipped as v4

Two long runs on v2's pairs and holdout, same seed (σ and sample order match step for step):

| | `moss_colorize_3k_textref1_iso` (Hub v3) | `…_muon_ema` (Hub v4) |
|---|---|---|
| optimizer / init | AdamW, `weight_svd`, channel scaling 0.5, `dc_isolate` | Muon, kaiming, `ema_window 0.1` (decay 0.999) |
| TE cache | fp8 | int8 (`…_iso_int8`, latents / refs hard-linked) |
| steps / where | 8000, pod, 1.0 s/step | 10000, local swap 32 / budget 0.15, 4.03 s/step, 11.2 h |

Training loss matches within ±0.001 per 500 steps from step 2k on (Muon + kaiming starts
higher and catches up by ~1.5k). Eval (`colorize_eval.py`, ER-SDE + Spectrum + RefCache, 30
steps, seed 1), `lora_ocr`:

| | luma r | chroma err | illus. 8 / manga 4 |
|---|---|---|---|
| v2 `lora_cap` | 0.977 | 12.4 | 12.8 / 11.6 |
| Hub v3 | 0.995 | 13.3 | 12.3 / 15.2 |
| **Hub v4** | 0.995 | **11.2** | **11.5 / 10.5** |

- Lettering: every OCR'd balloon exact on the 4 manga pages, per glyph; "H-HEY" drawn
  from the ref where the OCR read "H-HE".
- Still wrong: lettering and prop hues (403801's magenta lettering turns black / purple,
  its plaid green → blue; 2550520's globe blue → beige).
- Real Korean pages (`comfy/input/to_colorize2`, prompt `comic`, no OCR): every balloon
  kept, colour faint (cream tint, a few local hues) on that minimal line-art style.

### Deploy

1. **Done** (2026-10-27): `configs/moss/i2i/colorize/{cache,train,generate}.toml` and
   `ColorizeCacheRequest` (`vision = false`, `ref_scale = 1`, `ref_isolate = true`,
   `ref_vision = false`, `plain_caption_rate 0`, generate prompt `comic`), `docs/moss/methods/i2i.md`,
   the GUI task text. Swap stays at the i2i configs' 38 / 0.175 / H2D-only.
2. Stamp `ref_scale` and whether the TE saw the page into the LoRA metadata. Today only
   `ref_timestep` / `ref_isolate` are stamped, while ComfyUI-MOSS-Toolkit reads
   `ref_scale` from it.
3. ComfyUI-MOSS-Toolkit: **done in 0.5.0** (2026-10-23). `reference.py` reads
   `ref_isolate` and runs the ref rows through a `patches_replace`-skipped block;
   the Reference node's `task = colorize` picks the ref size from the recipe; prompt
   through a plain CLIP Text Encode (its template is byte-identical to
   `vision_te.encode(image=None)`). Example workflow `moss-i2i-colorize-v3.json`.
4. **Done** (2026-10-26): `colorize/moss_colorize_3k_v4.safetensors` on
   `sorryhyun/MOSS-I2I-LORAs`, the workflow pointed at it, new showcase.
5. Ref KV cache: **in the node** (0.5.0, `ref_cache`): the ref runs through the
   blocks once per page, ~2.1 GB bf16 at 1 MP; with the 1-hour arm an iteration
   costs ~1.25× t2i. Still to do in `generate.py`.

## Done (shipped or settled)

`done/`: `channel_scaling.md` (shipped, removed 2026-10-25 with `dc_isolate` once Muon made both no-ops), `dave.md` (shipped),
`sampler.md` (ER-SDE default; CNS bench-only), `soup.md` (3-seed soup keeps the style),
`i2i.md` (the i2i design + probes; shipped as `docs/moss/methods/i2i.md`),
`i2i_tone_loss.md` (the i2i tone term; shipped 2026-10-28 as sanitize v3, user doc `docs/moss/methods/tone_loss.md`).
Archived: REPA (parked), caption dropout, template mask → `_archive/moss_lora/`; the latent HF
budget + VR headroom proposals (closed 2026-10-23: the matched-σ loss gap is 72 % HF texture,
VR's control variate is uncorrelated with it at ρ² ≈ 0.01, so no VR port and no HF loss
weighting) → `_archive/proposals/`, their benches → `_archive/bench/`; the Anima-negatives
sweep (sinks, RFE, σ staircase measured; spectral guidance NO-GO 2026-10-25; Oscar / CTCal /
SSC / MIST and §4–§6 unmeasured) → `_archive/proposals/moss_anima_negatives_revisit.md`; the
template-mask code is on `moss-template-mask` in the private `preserve` mirror.

## src/

| Script | Role |
|---|---|
| `correction.py`, `correction_train.py`, `correction_ab.py` | corrector LoRA, its trainer, CFG × K grids |
| `te_gptq.py`, `te_calib.py`, `te_calib_eval.py` | calibrated TE rounding (fp8 / INT8), its calibration manifest, per-category eval |
| `swift_train.py` | SWIFT target extrapolation (`done/swift.md`) |
| `tone_probe.py`, `tone_dirs.py`, `dc_bias.py` | the i2i tone probes (`done/i2i_tone_loss.md`); `tone_dirs.py` built `library/moss/calibration/tone_dirs.safetensors` |

Moved to `bench/`: `_render.py`, `lora_ab.py`, `ab_cols.py`, `schedule_ab.py` and the Space
prompts (`space/004`, `017`) → `bench/moss_ab/`; `colorize_{eval,run,wild}.py`,
`tone_map.py` → `bench/moss_i2i_eval/`. `space/` here keeps the untracked tone probe outputs.

## Open questions

1. Character knowledge: does a name-only prompt (`akita neru`) work without the trait list?
2. Resolution ceiling: the report caps Base at ~1 MP, the Space presets run ~2.1 MP
   (`report_notes.md` § (a)). Keep LoRA buckets ≤ ~1 MP until Base is compared at 1 vs 2 MP.
