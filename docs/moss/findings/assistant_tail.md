# An assistant-turn tail after the template sharpens MOSS-Image renders

Measured 2026-11-03 on MOSS-Image Base, no LoRA. Script:
`project/moss_te_sae/src/base_te_render.py`; renders in `output/tests/base_te_render/`
(eval prompts) and `output/tests/base_te_render_kikuri2/` (explicit prompts).

## Setting

The shipped prompt ends at the generation prompt with thinking on:
`<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n<think>\n`
(3 prefix rows, the caption, 7 suffix rows). The TE is causal, so text appended after
`<think>\n` adds rows and leaves every earlier row bit-identical. Arms append a tail to
the cond prompt and, unless noted, the same tail to the empty uncond prompt:

| arm | appended after `<think>\n` | rows added |
|---|---|---|
| `instruct` | nothing (shipped) | 0 |
| `instruct_ok` | `\n</think>\n\nOK` (closed empty think block, Qwen's non-thinking form) | 4 |
| `instruct_sure` | `\n</think>\n\nSure,` | 5 |
| `instruct_think` | `Okay,` (think block left open) | 2 |
| `instruct_ok_cond` | `instruct_ok`'s tail on cond only; uncond shipped | 4 / 0 |
| `instruct_pad4` | four `\n` tokens (id 198): `instruct_ok`'s length, no content | 4 |

Shipped INT8 TE, layout-JSON prompts, ER-SDE 50, CFG 3, Spectrum, fp32 stream, 1024²,
seeds 1 / 2. Prompts: the three `bench/moss_ab/lora_ab.py` PROMPTS (Miku and Neru tag
bags, an NL schoolgirl prompt, caststation artist tag) and two explicit tag prompts of an
adult character (Hiroi Kikuri, solo and missionary). 10 prompt × seed pairs per arm.

## Measured

Mean |Δpixel| (0–255) against `instruct`, range over the 10 pairs: `ok` 12–31,
`ok_cond` 12–32, `sure` 11–32, `think` 9–24, `pad4` 13–27. `ok` against `ok_cond`:
3.5–15 (mean 8).

- **The uncond tail barely matters.** `ok_cond` renders close to `ok` (8 against 12–31
  from shipped), so the effect comes from the cond rows, not from a changed CFG direction.
- **Length alone changes the picture but not the quality.** `pad4` moves the image as
  far as `ok` and, by the user's review, gives a different picture at shipped quality.
  The tokens' content carries the gain.
- **By the user's review, the tail gains detail consistently and `sure` gains the most.**
  `ok` sharpened hands and backgrounds against shipped on the eval and the explicit
  prompts; `sure` was the most detailed of `ok` / `sure` / `think`, and showed less of
  the unrequested or unintended "preparing to refuse" look on the explicit prompts.

Unblinded and two seeds per prompt; no tagger or metric score backs the quality reading.

## Base TE arms (same run)

Qwen3.5-9B Base in place of the instruct TE: the shipped template (`base`), instruct's
template rows around Base's caption rows (`splice`), Base without the 7 suffix rows
(`base_nosuffix`), Base with `Describe this image: ` before the caption
(`base_describe`). None rendered better than `instruct`; `base_nosuffix` did not
collapse (the prefix keeps 3 of the 4 heavy sink rows) but rendered flatter, more
saturated, doll-like. Base's conditioning stays linearly close to instruct's (linear CKA
0.84 at the text-embedder output over 230 captions) with its variance on fewer
directions (participation ratio 72 vs 113); `project/moss_te_sae/src/te_stats.py`,
`output/tests/te_stats/20261103-1241/`.

## Hypotheses (untested)

- Instruct reads a user turn as a request (`project/moss_te_sae/hypothesis.md`, H1).
  Ending the prompt mid-answer, with an affirmative start (`Sure,`), puts the last rows
  in a state of complying with the caption rather than deciding whether and how to
  answer, and the DiT reads that state.
- Which rows carry it: the image's attention mass on the tail rows is unmeasured. The
  shipped suffix's caption-dependent rows get almost none (`text_read_path.md` §1); a
  tail row that becomes a heavy, caption-dependent key would be new.

## Next

- Attention mass and per-row value spread on the tail rows (`template_attn.py`).
- More seeds and a blind pick before any default change.
- A shipped tail changes the TE input, so TE caches would need to stamp it like
  `caption_format`; a LoRA trained on shipped caches and rendered with a tail is
  untested.
