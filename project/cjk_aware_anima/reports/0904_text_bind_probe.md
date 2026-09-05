# Text-binding probe — can the DiT bind text pixels to ext-row keys? (2026-09-04)

*Instrument for findings §9 job (2): the ext rows as stable, separable
addresses for text pixels — the requirement of the reframed goal (unmasked
manga training stays healthy with text captioned), which no readout-cos
metric sees. Dreambooth-style: train a plain LoRA on 1–3 sincos images whose
caption carries the OCR line through the ext pack, then render prompt
variants and ask where the text goes. Code: `probes/text_bind_probe.py`
(cache → train → `run_bench.py --ext --lora` render), `probes/text_bind_judge.py`
(PP-OCRv6 CER + montage), `datasets/cache_te_ext.py` gained `--stems` and the
`presence` OCR format (the `japanese text` tag alone, no 「…」 address).*

## Setup

- Images (safe/sensitive, real OCR lines, PP-OCR v2 records): **9095721**
  「ちょっとだけだから」 (handwritten vertical, white bg), **11883943**
  「着けるんですか！？」 (speech bubble), **6067089** 「かわいいなー♡」+「ちら。」.
- Train: plain LoRA dim 32 / α 32, lr 1e-4 cosine, bs 1, caption dropout 0,
  shuffled variants on, `path_pattern` = the image(s), `num_repeats` = steps
  (per image), preset default, torch_compile on, 16 GB card. Pack for the
  TE cache **and** the render: `cjk_vocab_pack_synthjakozh1sym_r256` (arm C9's).
- Render: `run_bench.py --ext --lora`, the image's own bucket size, 28 steps
  cfg 4.0, seeds 42 / 7 / 1234, arms `ja_ext` (T5 through the pack — the
  trained-on encoding) and `ja_native` (stock T5, 「…」 → `<unk>`: the
  no-address render of the same LoRA). Six conditions per image, composed
  through the caption grammar from the training caption:

  | cond | edit | expect |
  |---|---|---|
  | same | training caption verbatim | GT line |
  | swap | hair / eyes / clothes / background tags swapped, text kept | GT line |
  | drop_quote | 「…」 removed, `japanese text` kept | no text |
  | drop_all | `japanese text` removed too (= original caption) | no text |
  | other_text | 「…」 → a different JA line | the other line |
  | swap_drop | swap + drop_quote | no text |

- Judge: PP-OCRv6 over every cell; CER = Levenshtein over NFKC-normalized,
  punctuation-stripped strings, best over OCR lines (+ their concat); hit =
  CER ≤ 0.5; present = any CJK line. Sanity: the judge reads the source
  9095721 as exactly `ちょっとだけだから` (0.89).

## Results

### Single image 9095721, 100 steps (`reports/textbind_9095721.md`)

Hit 0/36 (base pack without LoRA: also 0/36). Composition memorized
(character, pose, outfit ≈ source). Text: a vertical handwritten blob in the
source's position and colour, **content all pseudo-JA** (OCR `慢にてんみるに…`).
`ja_ext` ≡ `ja_native` pixel-for-pixel per seed. `drop_all` → no text 6/6;
every condition that keeps `japanese text` → text 100 %, including the
beach `swap`. So at 100 steps the address of the text is the **presence
tag**, not the 「…」 keys; non-diegetic leakage 0/6.

### Single image 9095721, 400 steps (`reports/textbind400_9095721.md`)

Glyphs appear: OCR reads `ちょっとちけだけだぢ` / `ちょっとかっとぢけだ…` (hit
12/36; `same` ja_ext 2/3, CER 0.59). **But in every condition**: `drop_all`
text 3/3 (CER 0.93), `other_text` writes ちょっとだけ… instead of
明日も晴れるかな (CER 1.8), `swap` spawns a speech bubble with the line on
the beach. `ja_ext` ≡ `ja_native` still. → At 400 steps the LoRA body is the
address (arm B's "spam" in miniature); the keys are not an address at any
budget on one image — no contrast forces attribution.

### Three images, 100 steps total (~34 / image) (`reports/textbind3img_*.md`)

Pre-glyph again: hit 0/108, all pseudo-JA, `ja_ext` ≡ `ja_native`, `drop_all`
→ no text for 9095721/6067089 (11883943 keeps its `speech bubble` tag, so it
still draws a bubble with pseudo-text). The three images' text styles blend
(pink 6067089 handwriting shows up on 9095721). Budget too small — see next.

### Three images, 1200 steps total (400 / image), trained pack vs presence control

`reports/textbind1200_*.md` (trained: 「…」 through the pack in the training
caption) vs `reports/textbindP1200_*.md` (presence: `japanese text` only, no
「…」 in training; same recipe, same pack at render). Final loss 0.027 both.

**Text binds to the image's visual identity, not to any caption token.**
Each image's renders write (garbled) versions of **its own** line in every
condition — 9095721 → `ちょっとだけ…`, 11883943 → a bubble with
`えっ!?着けるんですか…`, 6067089 → `ちら。`/`かわいいな` — with little
cross-contamination (own-line CER < other-lines CER in most cells).
The decisive detail: 11883943's caption carried only `着けるんですか！？`, but
the renders reproduce the source's **`えっ!?`** too — text pixels memorized as
image content, exactly like the gold bikini. `drop_all` (no text tag at all)
still writes the own line at 100 % presence (trained) / 67–100 % (presence);
`other_text` never renders the requested line (CER vs other ≈ 1.0) and writes
the own line instead. Trained vs presence: same pattern, hit rates in the same
band (trained `same` ja_ext 0 / 0 / 0.33 per stem, presence 0 / 0.33 / 0;
11883943 `swap` 0 vs 0.67 — single-seed noise). `ja_ext` ≡ `ja_native`
throughout.

Reading: with three images the text lines differ, but so does everything
else — the model attributes `ちょっとだけだから`'s pixels to "this brown-haired
girl" and needs no key. Same rule as tags (findings §3: identity-carrying
tokens want O(100+) visits): **a line that occurs once binds to the image,
not to its row sequence, and prompting that image's identity spawns it.**
That is arm B's spam in miniature, and the 「…」 keys cannot change it for
unique lines at any budget. The rows would only become the address for a
line that recurs across images with different content.

### 351-image scale: C9 recipe with `presence` captions (arm P)

`configs/gui-methods/custom/cjk_unmask_presence.toml` = C9 verbatim (same
pack, same PP-OCR records, same latents, 8 epochs, seeds 42/7/1234) except
the caption carries only the `japanese text` presence tag — no 「…」, so no
ext row is ever looked up in training. Job `20260904-175824-79bdab`; grid
`output/tests/cjk_unmask_eval2/armP_s*`; side-by-side
`reports/unmask_armP_vs_C9.png`.

**P ≈ C9 cell for cell.** Same composition, palette and style per seed on
all 8 rows (the two LoRAs differ only in the T5-side tokens of the ~97
OCR'd captions). Text: rows 1–7 carry no non-diegetic JA text in either arm
(the r6 s42 "CAFES MAID"-style latin banner appears in **both**); the comic
row (r8) is where they differ, and in P's favour — C9 s7 fills its bubbles
with pseudo-JA, P s7 leaves them near-empty, P s42 draws a legitimate `…?`
bubble. Automated count (PP-OCR, loose thresholds, rows 1–7): C2 1 · C9 2 ·
P 2 of 21 cells, all single stray glyphs (`快` 0.39, `姐` 0.45, `三小酒武`
0.79); comic-row lines C9 4 · P 0. (The automated count is weaker than the
eyeball tally — at default thresholds C2–C5 and C9 are all 0/21 in rows 1–7,
and the loose pass gives C3 0 where the eyeball saw 3 — so it is a floor,
not the readout.)

So at the arm-C scale the 「…」 rows are not what keeps the LoRA clean:
**the presence tag alone reproduces C9's grid**, leakage and quality
included. Arm B (plain captions, no `japanese text` tag) is the one that
spammed; the caption is load-bearing through the presence tag.

### 351-image scale: C9 recipe with a geometry-matched RANDOM pack (arm R)

Does the *learned* representation help the LoRA train, or is any
well-conditioned set of addresses as good? `probes/make_random_pack.py`
builds `cjk_vocab_pack_random_r256`: same routing json as C9's pack, rows =
independent Gaussian draws recoloured to the trained pack's covariance
spectrum in a random basis, random mean direction at the trained mean norm,
per-row norms permuted from the trained rows (8 alternating projections).
Match: PR 18.2 → 18.2, random-pair cos 0.055 ± 0.285 → 0.062 ± 0.220
(the Gaussian cannot reproduce the trained tails — 2.2 % vs 0.8 % pairs
> 0.5), norm 203.9 ± 25.8 exact, row-wise cos(trained, random) −0.000 ±
0.030. `configs/gui-methods/custom/cjk_unmask_random.toml` = C9 verbatim
with `text_cache_dir` → `te/sincos_random_r256`; arm `armR`, job
`20260904-185406-b2fddc`. Flat adapter-output cos vs EN on the
positive-control prompts is the same for both packs (ja_ext 0.63 / 0.57 —
EN tags dominate the sequence; §2's "control, not gate").

**R ≈ C9 ≈ P.** Sheets `reports/unmask_C9_P_R_s{42,7,1234}.png`. Per seed
the three LoRAs render the same composition and style on all 8 rows; R's
deviations are of the same size as C9-vs-P's (r7 s42 R lands on a different
girl than the seed-locked cat-ear one; r2 s7 R adds pink hair + striped
thighhighs; r6 s42 all three carry the latin "MAID" banner). Non-diegetic
text, loose PP-OCR count rows 1–7: C9 2 · P 2 · R 2 of 21 cells, all stray
glyphs; comic-row lines C9 4 · P 0 · R 4 (R s7 fills its bubbles with
pseudo-JA like C9 s7). Nothing separates the learned representation from a
random table with its geometry — the LoRA trains equally well on either,
and equally well with no ext row at all (P).

## Reading so far

1. On one image there is nothing for the key to do: at 100 steps the
   presence tag carries the text, at 400 the whole LoRA does. The probe
   needs contrast (several images, different lines) before "which key" is a
   question the loss can answer.
2. The presence tag is a real address at low budget (drop it → no text 6/6
   at 100 steps) and stops being one once the image is memorized. Arm B
   carries no text tag at all; arm C carries the tag **and** the rows — so
   "captions are load-bearing" (0901_unmask_ab) may be mostly the presence
   tag. At 3 images the trained-vs-presence pair cannot separate them (both
   memorize); the 351-image arm P is the test that can.
3. `ja_ext` ≡ `ja_native`, and trained ≡ presence, on every run: the pack's
   rows contributed nothing visible to these LoRAs, at render or at
   training. Whatever the OCR-quoted captions did in arm C (0 vs 3 leakage
   events), it was not through row identity at this scale.
4. Ext-vs-no-ext is **undecidable at 1–3 images and indistinguishable at
   351** (arm P ≈ C9 on leakage and on quality). For the manga-training goal
   the 「…」 encoding buys nothing over `japanese text`; the rows only become
   an address for a line that recurs across images, which OCR'd manga text
   never does. Text fidelity (rendering an asked-for JA line) is out of
   reach of any of these LoRAs — the base model cannot write JA and 400
   steps on a line teach the image, not the string.

## Verdict

- The ext pack's job (2) — separable addresses for text pixels — is not
  exercised by real manga data: every line is unique, so the DiT binds it to
  the image identity regardless of the T5-side key, and the presence tag is
  what prevents spawn when the image identity is *not* prompted. **Ship the
  `japanese text` tag; the 「…」 quote tags are inert for this purpose.**
- Arm C's advantage over B in `0901_unmask_ab.md` is re-attributed to the
  presence tag (P reproduces C9 without any ext row).
- The pack still matters for job (1) (users typing JA tags — the shipped
  vocab pack); nothing here touches that.
- **The learned representation does not help the LoRA train**: a
  geometry-matched random table (arm R) gives the same grid as the trained
  pack (C9) and as no table (P). Whatever the rows encode is not consumed
  by LoRA training at this scale; the T5-side content of a unique OCR span
  is inert either way.
- Left open: whether a *recurring* line (the same 「…」 on ≥ 100 images with
  different content) would bind to the rows — the tag-visit rule predicts
  yes, but no corpus has it, and the goal does not need it.

