# release plan — `anima-jp-extended` (2026-10-08)

The user (10-08): once the 225 are trained and moved onto pres's rows, and
the result holds on the 96-string read and under a LoRA, publish it as a new
HF repo with a README (two usage paths + a base / jp-extended comparison),
and hand the blog a `content.md` for a post on the band law and L_pres.

**Two artifacts, one table.** jp_v1 = sent_kanji_pres's rows (kana 163,
kanji 1 185) + B's 225 kanji, on the punct pack; 1 573 trained rows, sha
`ce0ec15f7168…`.

- **`anima_cjk_vocab_pack_jp_v1`** → `sorryhyun/anima-vocab-pack-cjk` (§ 2.A, done).
- **`anima-jp-extended`** → new repo, one merged DiT for ComfyUI's
  `AnimaMergedLoader` (§ 2.1). Format: stock 32 128-row `embed` + the rows
  as `llm_adapter.ext_embed.weight` + the JSON in metadata; stock loaders
  load it as plain Anima. anima_lora doesn't read it (uses `vocab_pack`).

## 1. Gates

- **1.1 B (`sent_kanji_225`) — done 10-08.** 9 000 steps, job
  `20261008-181347-2c8d0d`. The new-kanji `g_r` read was not run (user:
  by eye in ComfyUI). Detail: `plan.md` § 3.
- **1.2 Transplant → jp_v1 — done 10-08.** `transplant.py` (plain; adding
  pres's Δstick turned the 225 < 1°). Ruler vs pres: every read n.s.
  (`results/20261008-2025-ruler-sensitive-seed_1008/`). Detail: `plan.md` § 3.
- **1.3 LoRA (`channel_(caststation)`) — open.** The user's run
  (`anima_lora_channel`, job `20261008-204941-265453`, GUI) looked good but
  trained on **preview51** (base.toml default, `lora/` TE caches). For
  jp_v1: the dataset's TE caches in `post_image_dataset/lora/` are now
  jp_v1's (30, stamped `ce0ec15f7168`), and `configs/gui-methods/custom/lora.toml`
  sets `vocab_pack` to jp_v1.

## 2. The HF repos

### 2.A The pack repo — done 10-08

- `old/`: preview … preview4 (HF `161ada4`). preview5 / preview51 stay at
  the root (anima_lora v2.x fetches preview51).
- jp_v1 pair + `_jp_v1_trained.json` (HF `df9797f`, `0ade7f5`); the
  re-downloaded pair reads `ce0ec15f7168`. Punctuation `～…♡♥、。「」` =
  preview51's retrained rows (max |Δ| ≤ 1.9e-6), listed as carried.
- Card (HF `7eb745e`): pack table, jp_v1 in the examples, limits.
  **Open:** the `anima-jp-extended` link once that repo exists.

### 2.0 Code — done 10-08; node publish open

- anima_lora: `merge_pack_into_dit` / `read_merged_pack`,
  `scripts/toolkits/merge_vocab_pack.py`, `load_anima_model` drops the ext
  key, tests (commit `b432b6ed`).
- Node `AnimaMergedLoader` (`~/ComfyUI-Anima_lora-Adapter`), 3.15.0,
  README + changelog. **Open:** commit + push, `comfy node publish` before
  the HF repo goes public.
- Built `models/diffusion_models/anima_jp_extended.safetensors` (base +
  jp_v1, 4.47 GB). Headless ComfyUI check (job `20261008-205913-aa29d2`):
  EN bitwise equal across stock / pair / merged / UNETLoader-on-merged; JA
  merged = pair bitwise; UNETLoader logs one `unet unexpected` line.

### 2.1 Files

New model repo `sorryhyun/anima-jp-extended`.

| file | what |
|---|---|
| `anima_jp_extended.safetensors` | the merged DiT (§ 2.0) built from jp_v1, ≈ base DiT + 285 MB |
| `README.md` | § 2.2 |
| `assets/` | the comparison grids (§ 2.3) |
| `workflows/anima_jp_extended.json` | a minimal ComfyUI workflow with `AnimaMergedLoader` wired |

No pack files here; the pack lives in `anima-vocab-pack-cjk` (§ 2.A).

Upload with `hf upload` to a **private** repo first; the user flips it
public after reading the rendered README.

### 2.2 README

1. **What it is** — one paragraph: Anima's T5-side tokenizer maps Japanese
   to `<unk>`; this checkpoint carries 1 573 trained rows (163 kana, 1 410
   kanji: 92.8 → 96.7 % of Manga109-s kanji occurrences covered) so a JA
   string in the prompt is drawn as that string. The DiT weights are base
   Anima's; EN prompts tokenize exactly as before. **Want the rows alone
   (another Anima derivative, your own loader)?** →
   `sorryhyun/anima-vocab-pack-cjk`, `anima_cjk_vocab_pack_jp_v1`.
2. **ComfyUI** — install *Anima Adapter Loader*
   (`comfyui-anima-lora-adapter`, the version that adds `AnimaMergedLoader`),
   put the file in `ComfyUI/models/diffusion_models/`, load it with
   `AnimaMergedLoader` (it takes the stock Qwen3 text encoder too) in place
   of the UNet and CLIP loaders, and type Japanese directly in the prompt.
   Say that a stock UNet loader loads the file as plain Anima (no JA), and
   that a LoRA trained in anima_lora on jp_v1 stacks on it as is. The
   workflow file, and how to quote the text (the caption grammar the rows
   were trained under: `japanese text`, `"…"` in a speech bubble).
3. **anima_lora** — uses the pack, not this checkpoint: with § 2.4's
   release, `make download-models` (or `make download-model vocab_pack`)
   fetches jp_v1 and `configs/base.toml`'s `vocab_pack` points at it, so
   inference, TE caching and LoRA training all use it with the stock base
   DiT. Upgrading from preview51: `make preprocess-te ARGS=--overwrite`
   (TE caches are mtime-keyed and won't notice the pack change).
4. **Comparison** (§ 2.3).
5. **Limits**, from the reads: long lines (10+ glyphs) are rarely exact;
   horizontal text is 6.8 % of the training items and unread; the 3.3 % of
   kanji occurrences without a row still draw as noise; pres draws text
   about a third smaller than preview51.
6. License: the base Anima model's terms; credits (Manga109-s for the
   dialogue lines — check its terms allow naming it in a model card).

### 2.3 The comparison

Two grids, base vs jp-extended, same seeds, same everything but the pack:

- **No Japanese in the caption** — 6 EN prompts × 2 seeds. An EN caption
  holds no ext id, so the forward should be the base's exactly. Measure it
  (max |Δ| per image) before writing the sentence: if bitwise identical,
  the README says identical; if not, it says by how much and why (the
  punct pack's folds touch `…` / `...` — keep those out of the EN prompts or
  report them).
- **Japanese in the caption** — 8 strings: short kana, mid mixed, a long
  line, 2 of the 225. Strings from the **unseen** set (the ruler's unseen 42
  or `b5_held`), not training captions. Base garbles, jp-extended letters
  it; cherry-picking stated (one seed each, or best of N said as such).

Rendered through `make gen` (daemon) into `output/cjk_anima_reseed/release/`.

### 2.4 Repo follow-ups (after the upload, each its own commit)

- **The default moves to jp_v1** (user, 10-08): `library/downloads.py`
  `VOCAB_PACK_STEM` → `anima_cjk_vocab_pack_jp_v1` (same repo), and
  `configs/base.toml` `vocab_pack` → its prefix. That also updates
  `scripts/tasks/downloads.py` (reads the stem); the literal `preview51`s
  are in `tests/test_vocab_pack.py:358`, `examples/09_cjk_vocab_pack.py`,
  `examples/10_cjk_vocab_pack_diffusers.py` (`PACK_STEM`), the four
  guidebooks (line 81; the translator agent for ko / ja / zh),
  `docs/methods/cjk_vocab_pack.md`, `docs/experimental/anima_cjk_vocab_ext.md`
  and the root README's v2 blurb. A release note says to re-cache CJK TE
  (`make preprocess-te ARGS=--overwrite`).
- `docs/methods/cjk_vocab_pack.md`: jp_v1, the recipe of record
  (A → B → transplant), the `anima-jp-extended` repo for ComfyUI.
- No catalog row for the merged DiT: anima_lora doesn't use it.
- The ComfyUI node is published (§ 2.0) **before** the HF repo goes
  public, since the README points at it.

## 3. `content.md` for the blog

`/home/sorryhyun/sorryhyunblog/content.md` — a brief for the agent there,
not the post. Written after § 1.2 (the post links the HF repo and quotes the
release's numbers). Already published in `waking-anima-up-to-read-japanese`
(09-17): identity is decided near σ 0.8 (the classifier read), scene
composites, the paired loss, warm-start erasure. The brief says so, so the
new post builds on it.

**Post 1 — how the effective bands were found.** Sources:
`../cjk_anima_scale/band_experiment_results.md`, the `cf_*` / `band_*`
reports in `../finished/cjk_renderable_anima/reports/`,
`_archive/reports/grad_identity_2026_10_02.md`,
`_archive/reports/grad_bands_2026_10_03.md`.

- The method: the frozen base as a classifier over captions per σ (where
  does a caption change the prediction?) on EN text at 12–128 px → the
  ceiling table (peak σ 0.4 at 12 px → 0.8 at 128, monotone).
- The training reads that turned the ceiling into a rule: singles at
  0.7–0.9, multi-glyph pieces at 0.5–0.7 — **the band is keyed on glyph
  count, px sets the floor, nothing above 0.9** (0.8–0.95 collapses 48 px
  kanji, 71 → 18 of 192). Ink does not move the band; dense kanji are a
  px / exposure question.
- The cross-check: bands read off the gradient (`grad_identity`) tie the
  rule.
- The transferable part: before training a token, measure where in σ its
  caption has leverage on the frozen model; train there.

**Post 2 — L_pres.** Sources: `reports/probe_pres_2026_10_06.md`,
`probes/probe_pres_train.py`'s h16 / h32 / p10c09 reads (commits
d8db5265, 92a80b1b, 3e81e605), `reports/sent_kanji_pres_2026_10_07.md`.

- The problem: longer JA lines spread across the page and the subject goes.
- The loss: the student under the JA caption against the frozen base under
  the same caption with the JA swapped for an EN line of its length, same
  x_σ and ε, outside the text box dilated by 2 latent cells; the teacher
  sees no row.
- What the probe showed: the most coherent of the three gradient terms,
  near zero at σ ≤ 0.7, against the text at 0.95 → band capped at 0.8–0.9
  (p10's text cost was σ 0.95's).
- The full run: the page up on every read (`en_match` +0.097, p 7e-8),
  the text held (cer n.s.); the cost — text area −33 %, kana recall
  −0.066 (p 0.09), training × 1.67.
- The transferable part: when fine-tuning a new concept in, distil
  "everything else" from the frozen model under a caption with the concept
  swapped out, at the σ band where layout is decided.
- Optional, short: the lenses that did not explain it (Jacobian lens, PE
  lens, twin difference — all stopped on their own rules).

Images: pick from the ruler result dirs (`results/*-ruler-sent_kanji_pres/`)
and § 2.3's grids; the brief lists paths, the blog agent copies.

## Order

Done: 1.1, 1.2, § 2.0 code + merged file, § 2.A. Left: 1.3 on jp_v1 →
§ 2.4 (anima_lora default → jp_v1) → § 2.3 renders → node published →
§ 2.1–2.2 private upload → user reads → public (+ the pack card's link) → § 3.
