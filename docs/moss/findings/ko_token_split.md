# Korean wrong syllables follow the tokenizer: the same letters, split differently, come out right

Measured 2026-11-03 on swap_text, MOSS-Image Base, with the v3 en400_ko1709 LoRAs
(`..._rw2_mod16_ema5e5_int8_e2` and `..._int8_ga2_e4`). Renders
`output/moss_i2i_eval/swap_text/results/tokprobe_{zwsp,nospace}_*`, sheet
`v3_en400_ko1709_ga2_e4_compare/tokprobe.png`.

**Verdict: on the two lines tested the error is in the token, not in the amount of training.**
Changing only how the line tokenizes takes `분인가요` from 0 of 6 renders to 6 of 6, and `싼다`
on the 4-epoch LoRA from 1 of 3 to 3 of 3. More epochs or more of the same data are not the fix.

## How Qwen3.5 tokenizes Hangul

Tokenized line by line over the 1,709 `ko` pairs of `pairs_en400_ko1709.jsonl` (10,429 lines,
92,659 syllables, 1,079 distinct):

| token kind | syllables | example |
|---|---|---|
| one syllable | about 59 % | `인`, `가`, `싸` |
| two or more syllables | about 41 % | `니까`, `가요`, `인가요` |
| split into byte fragments | about 1 % | ` 싼` -> ` \xec\x8b` + `\xbc` |

The multi-syllable tokens are 2,026 distinct, and 1,683 of them appear 11 times or fewer. A
split depends on context: `싼` and `씬` are single tokens at the start of a line and after `…`,
and both split after a space. ` 싸` stays one token. 191 distinct syllables are split somewhere.
The TE is frozen and gives each token its own vector, so the DiT has no way to read off that
`인가요` holds the glyphs it knows from `인`, `가` and `요`.

| token | times in training |
|---|---|
| `인가요` | 3 |
| `가요` | 53 |
| `나요` | 43 |
| byte-split ` 싼` | 0 |

## Probe

Same LoRA, sampler, page, seeds 0, 1, 2; one line of the prompt changed.

| line | LoRA | as captioned | changed | changed tokens |
|---|---|---|---|---|
| `분인가요` | e2 | 0 of 3 | 3 of 3 | `분인​가요`: `인`, ZWSP, `가요` |
| `분인가요` | ga2 e4 | 0 of 3 | 3 of 3 | as above |
| `싸… 싼다` | ga2 e4 | 1 of 3 | 3 of 3 | `싸…싼다`: `싼`, `다` |

The originals read `분이나요`, `분이가요` and `분인나요`, each a small stroke away, which is why the
errors look like a resolution limit. The probe rules that out for these two lines: the glyphs are
the same size in both columns.

The ZWSP draws as a space (`분인 가요`), so it is a probe, not a fix. Dropping the space in
`싸…싼다` costs nothing on this line, since the balloon still breaks there.

## Epochs and data did not move this

`..._ga2_e4` (4 epochs, `grad_accum` 2, 8,436 samples = 4,218 updates, 6.0 h) against `..._e2`
(2 epochs, 3.2 h), seeds 0, 1, 2 on the four held-out pages, as captioned:

| | e2 | ga2 e4 |
|---|---|---|
| Korean wrong syllables, `벌거` left out | 5 | 5 or 6 |
| broken English lines | 2 | 1 |
| outside-text \|ΔL\|, English pages | 5.6 / 7.9–8.0 | 5.6 / 8.0 |
| outside-text \|ΔL\|, Korean pages | 2.6 / 2.5 | 2.1–2.2 / 2.3–2.4 |

`벌거` is left out because the caption is itself a Hangul-for-Hangul misread. Seed spread is larger
than the gap between the two LoRAs. Region loss over the last quarter is 0.134 against 0.138 and
flat from epoch 3. The double exposure leaves a rare token rare: `인가요` is still 3 pairs of 2,109.

## Not tested

- **Two lines, nine renders.** The other Korean lines were not re-split.
- **Splitting token ids without a character** in the tokenizer, at cache and inference. That is
  the candidate fix, and it would avoid the ZWSP's space.
- **Whether every syllable should be its own token.** `가요` above is still a two-syllable token,
  so splitting only rare tokens may be enough. Full syllable splitting may also cost the TE context.
- **`e2` on `싸…싼다`.** It already gets `싼다` right as captioned, from a byte split it never saw
  in training, so a rare token does not always fail.
- `건` -> `전` ([i2i.md](../methods/i2i.md)) is a single-syllable token drawn as a near glyph, which
  this finding does not explain.
