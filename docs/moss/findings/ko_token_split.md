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
| split into byte fragments | about 2 % | ` 싼` -> ` \xec\x8b` + `\xbc` |

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
| byte-split ` 싼` | 39 (18 of them in ` 싼다`) |
| ` 이건` | 16 |
| ` 분` | 2 |

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

## Trained on the split

`train_v3_en50_ko150_ksplit.toml` against `train_v3_en50_ko150.toml`: same 200 pairs, 600 steps, seed
and recipe, 26.5 min at 2.6 s/step, 13.04 GB peak. EMA final, `589883_0004_ko` on seeds 0, 1, 2 and the
other three pages on seed 0, as the control has them. Sheets `ksplit_e3_compare/`.

| | control | trained on the split |
|---|---|---|
| Korean wrong syllables, `벌거` left out | 8 | 2 |
| `분인가요` | 0 of 3: `분인나요` ×2, `분이마요` | 3 of 3 |
| `이건` | 1 of 3: `이런`, `이전` | 3 of 3 |
| `싼다` (seed 0) | `씬다` | right |
| `가득해` (seed 0) | `가늑해` | right |
| `처음` | 3 of 3 | 1 of 3: `처즘` ×2 |
| English | exact | exact |

The split adds an error the control does not make: `음` -> `즘`. ` 처음` is a token seen 4 times in these
150 pages, and `음` as a syllable 28, so the more frequent form is the one that fails here. `뭔` reads
ambiguously on both arms and is not counted. `저기… / 이건 뭔가요?` breaks the same on both arms, so at
this size the rebreak does not come from the split. `Uwah…` keeps its balloon outline on the split and
loses it on the control, one seed.

## Not tested

- **Two lines, nine renders.** The other Korean lines were not re-split.
- **Training on the split at full size.** Only the 200-pair arm above is measured.
- **Whether every syllable should be its own token.** Splitting only rare tokens may be enough.
- **`e2` on `싸…싼다`.** It already gets `싼다` right as captioned: the byte split appears 39 times in
  training, more often than the one-token `싼` (23), so `싼다` is not a rare-token case.
- `건` -> `전` ([i2i.md](../methods/i2i.md)) sits in the two-syllable token ` 이건` (16 times), not a
  single-syllable token.
- **Rarity alone.** English draws rare tokens exactly on the same pages: ` MOVE` 7 times in training,
  `URA` 2, ` MMM` 0. The DiT can read a rare token, so the claim above that it has no way to is too
  strong; what differs for Hangul is not measured.

## Token ids split at inference

`token_split = "syllable"` (`library/moss/token_split.py`): every Hangul syllable tokenized alone, a
space before one as its own token. 843 of the pool's 1,079 distinct syllables have their own token,
98.8 % of occurrences; the rest stay byte fragments. The two eval prompts gain 9 % and 4 % tokens.

`..._e2`, not trained on the split, seeds 0, 1, 2. Renders `tokprobe_{split,placebo}_e2_s*`.

| arm | `분인가요` | wrong syllables, 7 lines, `벌거` left out |
|---|---|---|
| as captioned | 0 of 3 | 5 |
| placebo: ZWSP after `인가요`, token kept | 0 of 3 | |
| ids split | 3 of 3 | 0 |

All five wrong syllables are in that one word. The split also moves layout: `저기… 이건 뭔가요?` breaks
as `저기… / 이건 뭔가요?` in smaller type on 3 of 3 seeds, against the target's `저기… 이건 / 뭔가요?`, and
`어머?` rebreaks on seed 0. The en50_ko150 control breaks that line the same way (see "Trained on the split").

Colorize, `moss_colorize_3k_textref1_iso_muon_ema`, the two Korean eval targets as input: `comic`
draws all 7 lines from the reference. With the right text as layout items, ` 이건` becomes `어건` on 1
of 3 seeds, and the split gives 3 of 3 with no layout change. `벌거` in the prompt does not override
the reference's `씻겨`, and it stays black where `comic` turns it magenta. Renders
`output/moss_i2i_eval/colorize/results/ko_tokprobe/`.
