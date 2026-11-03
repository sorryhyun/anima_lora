# An assistant-turn tail after the template sharpens MOSS-Image renders

Measured 2026-11-03 on MOSS-Image Base, no LoRA. Script:
`project/moss_te_sae/src/base_te_render.py`; renders in `output/tests/base_te_render/`
(eval prompts) and `output/tests/base_te_render_kikuri2/` (explicit prompts).

## Setting

The shipped prompt ends at the generation prompt with thinking on:
`<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n<think>\n`
(3 prefix rows, the caption, 7 suffix rows). The TE is causal, so a tail appended to
the prompt text leaves the earlier rows' states unchanged, except where the tokenizer
merges the template's last `\n` with a tail starting in `\n` into one `\n\n` token
(`ok`, `sure`): that one suffix row changes token. Arms append a tail to the cond prompt
and, unless noted, the same tail to the empty uncond prompt:

| arm | appended to the prompt text | tokens after `<think>` | rows added |
|---|---|---|---|
| `instruct` | nothing (shipped) | `\n` | 0 |
| `instruct_ok` | `\n</think>\n\nOK` (closed empty think block, Qwen's non-thinking form) | `\n\n` `</think>` `\n\n` `OK` | 3 |
| `instruct_sure` | `\n</think>\n\nSure,` | `\n\n` `</think>` `\n\n` `Sure` `,` | 4 |
| `instruct_think` | `Okay,` (think block left open) | `\n` `Okay` `,` | 2 |
| `instruct_ok_cond` | `instruct_ok`'s tail on cond only; uncond shipped | as `ok` | 3 / 0 |
| `instruct_pad4` | four `\n` token ids (198) appended to the ids, no content | `\n` + 4 × `\n` | 4 |

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
- **Length alone changes the picture but not the quality.** `pad4` (4 rows, as `sure`;
  one more than `ok`) moves the image as
  far as `ok` and, by the user's review, gives a different picture at shipped quality.
  The tokens' content carries the gain.
- **By the user's review, the tail gains detail consistently and `sure` gains the most.**
  `ok` sharpened hands and backgrounds against shipped on the eval and the explicit
  prompts; `sure` was the most detailed of `ok` / `sure` / `think`, and showed less of
  the unrequested or unintended "preparing to refuse" look on the explicit prompts.

Unblinded and two seeds per prompt; no tagger or metric score backs the quality reading.

## Attention on the tail

`project/moss_te_sae/src/tail_attn.py` (`template_attn.py`'s read with the tail rows as
a fifth key segment): Base, Euler 30 steps, CFG 3, no Spectrum, seed 1, 1024², the three
eval prompts, cond branch; `output/tests/tail_attn/20261103-1338/`. Mass and value
contribution `‖Σ_seg p·v‖ / ‖Σ p·v‖` of the image queries, mean over heads and blocks,
early / mid / late step thirds:

| tail | tail mass | tail contrib | suffix mass | `</think>` | answer rows | peak blocks |
|---|---|---|---|---|---|---|
| none | — | — | 0.11–0.13 | — | — | — |
| `ok` | 0.010–0.018 | 0.022–0.039 | 0.10–0.12 | 0.010–0.015 | `OK` 0.0003 | 19 / 15 / 12 / 10 / 17 |
| `sure` | 0.011–0.021 | 0.023–0.041 | 0.10–0.11 | 0.011–0.016 | `Sure` 0.0002, `,` 0.0002 | 19 / 15 / 12 / 17 / 20 |
| `pad4` | 0.007–0.010 | 0.019–0.037 | 0.11–0.12 | — | first `\n` 0.0065, rest ≤ 0.001 | 25 / 23 / 29 / 19 / 34 |

- **The image reads `</think>`, not the answer word.** `</think>` becomes the fifth
  heaviest key after the four sink rows (`user`, `assistant`, `<|im_start|>` ~0.07,
  `<|im_end|>` ~0.03). `OK` / `Sure` / `,` get 0.0002–0.0003, as the shipped suffix's
  caption-dependent rows do. The tail's mass comes out of the suffix (mostly
  `assistant`); the caption rows keep theirs.
- **It writes more than its mass.** Contribution 2–4 % on 1–2 % of the mass: not a
  sink that takes mass and returns little.
- **`ok` / `sure` are read in blocks 9–20, `pad4` in 23–34.** The extra `\n` rows act
  as a copy of the template's last `\n` (0.0065 vs 0.007) and are read late.
- **`</think>` is the first heavy key that varies with the caption.** Cross-prompt spread
  of each row's value (`mean_p ‖v_p − v̄‖ / ‖v̄‖`, mean over blocks, step 0 / 29): the
  four sink rows 0.02–0.03 in every arm; `</think>` 0.29 / 0.23 (`ok`), 0.23 / 0.19
  (`sure`); the answer rows 0.27–0.49. `pad4`'s one read row (the first extra `\n`) is
  prompt-constant, 0.02 / 0.02. Every shipped row the image reads heavily is constant
  (`text_read_path.md` §1), so `ok` / `sure` add a caption-dependent key at sink-adjacent
  mass, and `pad4` adds another constant one.
- **Caption queries read the tail too**: 2–3 % of their mass (`</think>` 0.012–0.017;
  `Sure` 0.0015 and `,` 0.003, against `OK` 0.002).

The TE is causal, so `</think>`'s TE row is the same under `ok` and `sure`. The text
refiner (`text_embedder`, 2 bidirectional blocks) and the joint blocks let the answer
rows write into it before the image reads it, which is the path by which `Sure,` and
`OK` can differ while the image barely reads either directly.

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
- The `</think>` row carries it, read in blocks 9–20, with the answer rows reaching it
  through the text refiner and the joint blocks. Untested: cutting the image's keys to
  `</think>`, or to the answer rows, per arm.

## Next

- A key cut on `</think>` and on the answer rows (`template_causal.py`'s cut arms).
- More seeds and a blind pick before any default change.
- A shipped tail changes the TE input, so TE caches would need to stamp it like
  `caption_format`; a LoRA trained on shipped caches and rendered with a tail is
  untested.
