# release plan — `anima-jp-extended` (2026-10-08)

The user (10-08): once the 225 are trained and moved onto pres's rows, and
the result holds on the 96-string read and under a LoRA, publish it as a new
HF repo with a README (two usage paths + a base / jp-extended comparison),
and hand the blog a `content.md` for a post on the band law and L_pres.

**Two artifacts, one table** (user, 10-08). The table is sent_kanji_pres's
1 185 kanji + 163 kana rows with B's 225 transplanted onto them, on the
punct pack: 1 573 trained rows (§ 1.2).

- **`anima_cjk_vocab_pack_jp_v1`**: the pack pair, uploaded to the existing
  `sorryhyun/anima-vocab-pack-cjk` first (§ 2.A). anima_lora keeps using
  the pack (`vocab_pack`), and anyone who wants the pack alone is sent here.
- **`anima-jp-extended`**: a new repo with one merged DiT checkpoint built
  from jp_v1, for ComfyUI through a new node (§ 2.0).

The merged checkpoint:

- `llm_adapter.embed.weight` stays at the stock 32 128 rows; the
  materialized ext rows go in under their own key
  (`llm_adapter.ext_embed.weight`, `[rows, 1024]`), and the pack's JSON
  sidecar (route / dots / row maps) and digest go into the safetensors
  metadata. A stock loader sees one unexpected key and still loads a working
  base Anima (EN fine, JA → `<unk>` as today). Widening `embed` instead
  would break stock loaders on the size mismatch.
- The Qwen3 text encoder and both tokenizers stay stock. The hybrid
  encoder is built at load time from the stock T5 + Qwen3 tokenizers and
  the mapping in the checkpoint (`VocabPack.build_encoder(t5, qwen3)`), so
  there is no tokenizer file to ship.
- To check before building: that ComfyUI's stock UNet loader tolerates
  the extra key (unexpected keys logged, not raised), and the metadata
  header size (~1.9 MB JSON; safetensors allows up to 100 MB).
- anima_lora does not load the merged file as a pack source; it runs on
  jp_v1 via `vocab_pack`.

## 1. Gates — nothing is uploaded until all three pass

### 1.1 The 225 trained (B, `sent_kanji_225`)

`plan.md` § 3 B, then § 4's read. Pass:

- **New-kanji strings** (`b5_held.tsv`: 160 lines covering 211 of the 225,
  plus the single read for the 14 it misses — 藩 鉛 緋 昴 眞 燈 柏 薩 帖 駕 彗
  兒 杖 詫): `g_r` on the 225 above preview51 and pres (no row → their
  floor), paired, p < 0.05.
- **B's own Δstick** (kana / the 1 185) against pres's, size and cos — says
  how close B's context is to the one the 225 land in (§ 1.2).

Fail → back to `plan.md` (A's budget or B's data), no release.

### 1.2 The transplant (the 225 onto pres)

A script (to write, `reseed/transplant.py` or a `run.py` verb): pres's pack
with B's 225 rows copied in, nothing else changed → `models/vocab_packs/anima_cjk_vocab_pack_jp_v1/`.
Before reading it, `probe_stick_move`'s read of the 225's mean against pres's
kanji stick (pres's sits 11.5° off stick080's, |stick| 114 → 125). If it
differs materially, two arms: plain transplant and + pres's kanji Δstick on
the 225; the read below picks.

Pass, against `sent_kanji_pres` on the **96-string dialogue ruler**
(`ruler.py run --pack punct --arms sent_kanji_pres,<jp_v1>`):

| read | rule |
|---|---|
| `g_f1`, `g_r_kanji`, `g_r_kana` | no paired loss (p ≥ 0.05 or better) |
| `en_match`, `en_tok_out`, `iou_en` | no paired loss — pres's page gain is the reason pres is the base |
| `cer`, exact, le2 | reported; no gate (every arm sits at 0–1 past a word) |

and on the new-kanji strings: `g_r` on the 225 within B's own (the move
onto pres costs them little), above pres's.

### 1.3 A LoRA on top (`channel_(caststation)`)

30 images; the revised captions (`post_image_dataset/resized/`, what TE
caching reads — the `image_dataset/` masters are EN only) carry an OCR text
clause on **12 of 30** (`Japanese text reads as "満足でしょうか"`, `"※危険なので
真似しないでください"`, `"後藤ひとり156cm50kg"` …; 120 strings with the
`.variants.txt`, 6–88 chars, median 21). So this is the case the pack is
for downstream: an artist LoRA on real pages whose captions quote their JA
text. It reads three things: training with the pack on is ordinary, the
LoRA does not take the text away on unseen strings, and the pack helps the
LoRA learn its own pages' text.

0. Coverage first (CPU): the 12 images' strings against the jp_v1 charset —
   the share of their glyphs with a trained row (鬱 and the like route to
   no row and draw as noise whatever the LoRA does). Reported, not a gate.
1. `vocab_pack` → the jp_v1 prefix (CLI or a local config; not `base.toml`
   yet, § 2.4), `make preprocess-te ARGS=--overwrite` on that dataset
   alone (TE caches carry the pack digest).
2. `make lora --queue` at the default preset; `log-analyst` on the run.
3. Render through `make gen`, same seeds:
   - (a) 8 EN prompts in the artist's style, LoRA on: the style lands.
   - (b) 16 of the ruler's unseen strings with their prompts, LoRA on vs
     off: the text survives (`g_f1` / `g_r` by the ruler's scorer).
   - (c) the 12 images' own captions, LoRA on: their quoted text read by
     the ruler's scorer against the caption string.
Pass: a normal run verdict, style visible, (b) LoRA-on not below LoRA-off
by more than the run-to-run noise on the 16. A collapse in (b) is a README
warning at minimum and blocks the release until understood. (c) is a read,
not a gate. No base-TE control arm (user, 10-08), so the README makes no
claim that the pack helps a LoRA learn its pages' text.

Run 1.3 on jp_v1 through `vocab_pack` with the stock base DiT. Which one
it trains on doesn't matter: the merged file is base's DiT weights plus
jp_v1's rows, both frozen under a LoRA, so the forward and the LoRA are the
same. A LoRA trained either way loads on either (§ 2.0's digest test is
what keeps that true).

## 2. The HF repos

### 2.A The pack repo (`sorryhyun/anima-vocab-pack-cjk`), first

Now at the root: `preview`, `preview2`–`preview5` and `preview51`
(`.safetensors` + `.json`, `_trained.json` for 3 / 4 / 5 / 51), `assets/`
(`preview3_hentai.webp`, `preview_hai.png`, `training_diagram.png`),
`README.md`.

1. Upload `anima_cjk_vocab_pack_jp_v1.{safetensors,json}` and
   `_jp_v1_trained.json` (the 1 573 trained rows) to the root. Check that
   the uploaded pair loads and prints the same `sha[:12]` as the local one.
2. Move `preview`, `preview2`, `preview3`, `preview4` (and the
   `_trained.json` of 3 / 4) to `old/`, in one commit (**done 10-08**,
   HF commit `161ada4`, with the card's "stay in the repo" line → "under
   `old/`"; v2.0.0.beta2's default `anima_cjk_vocab_pack_preview` is now
   only under `old/`)
   (`HfApi.create_commit` with `CommitOperationCopy` +
   `CommitOperationDelete`; the copy is server-side, no 285 MB re-upload).
   **`preview5` and `preview51` stay at the root** (user, 10-08), so
   released anima_lora (v2.x, whose `base.toml` and `_fetch_shipped_pack`
   fetch `anima_cjk_vocab_pack_preview51.*` from the root) keeps working.
3. The model card: jp_v1 as the current pack, preview5 / preview51 as the
   previous ones, the old/ files listed as history, a link to
   `anima-jp-extended` for ComfyUI users who want one checkpoint, and the
   `assets/` images either kept under a "previous versions" section or
   moved to `old/assets/`.

### 2.0 Code (built 10-08; the real-file check pending)

**anima_lora**

- `library/anima/vocab_pack.py`: `merge_pack_into_dit(dit, pack, out)`
  (every DiT tensor and metadata key copied, the pack's raw `ext_embed` as
  `<prefix>llm_adapter.ext_embed.weight`, the JSON text as
  `ss_ext_pack_mapping`, the `ss_ext_pack` / `ss_ext_pack_sha` stamp) and
  `read_merged_pack(path)` (refuses a file whose rows don't match its stamp).
- `scripts/toolkits/merge_vocab_pack.py`: the CLI (`--dit` defaults to the
  base DiT, `--pack`, `--out`), with a read-back digest check.
- `load_anima_model` drops the ext key with a warning (it raised on
  unexpected keys before), so a merged file still loads in anima_lora as a
  plain DiT; the rows come from `vocab_pack`.
- Tests (`tests/test_vocab_pack.py`): the round trip keeps every DiT
  tensor bit-exact, the digest equals the pack's, an already-merged file and
  a tampered stamp are refused. So a LoRA trained on jp_v1 (stamped
  `ss_ext_pack_sha`) meets the same digest on `AnimaMergedLoader` and the
  node's `check_pack_vs_adapter` stays silent.

**ComfyUI** (`~/ComfyUI-Anima_lora-Adapter`, symlinked into `../comfy`)

- **`AnimaMergedLoader`** ("Anima Merged Loader (CJK)"): `unet_name`
  (`diffusion_models/`), `clip_name` (`text_encoders/`, the stock Qwen3
  0.6B; it loads as Anima under the default CLIP type), `weight_dtype` as
  UNETLoader → MODEL + CLIP. `vocab_pack.load_merged_diffusion_model` splits
  the pack off the state dict before the DiT is built, registers itself as
  the patcher's `cached_patcher_init` (DiT only; clones keep the hooks),
  then `apply_vocab_pack` + `VocabPackTokenizer` as the pack loader does. A
  file without a pack loads as plain Anima, with a warning.
- Checked on CPU: the node's `split_merged` on a file written by
  anima_lora's merge gives the same digest and leaves a plain state dict.
- `AnimaVocabPackLoader` stays (pack pairs).
- **Still to do:** the real-file check after B frees the GPU (merge the
  base DiT + pres's pack; ComfyUI: EN prompt identical to UNETLoader +
  CLIPLoader, a JA prompt identical to AnimaVocabPackLoader on the pair,
  UNETLoader on the merged file loads with one "unet unexpected" line);
  then version bump, `make vendor-sync` (check `_vendor` drift first),
  `comfy node publish` at release.

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

§ 2.0 code (can start now, against pres's pack) ∥ 1.1 → 1.2 → jp_v1 →
1.3 on it (anima_lora, `vocab_pack` = jp_v1) → § 2.A (upload jp_v1, preview–preview4 → old/)
→ § 2.4 (anima_lora default → jp_v1) → merge the checkpoint → § 2.3
renders through `AnimaMergedLoader` and anima_lora both → node published →
§ 2.1–2.2 private upload → user reads → public → § 3.
