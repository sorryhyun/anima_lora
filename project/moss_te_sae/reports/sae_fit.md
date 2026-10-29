# How well the Base SAEs fit the instruct TE, and what the DiT reads of it

Scripts: `src/probe.py` (20261027-1609,
`/media/sorryhyun/new/qwen35_sae/probe/20261027-1609/`), the SAE part of
`src/base_contrast.py` (20261029-2123). SAEs `Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100`
(TopK k=100, 64K features, `resid_post`), layers 15 / 31. 150 layout-JSON captions from
five `*_layout` caches + 80 pe_lora NL prompts; base_contrast adds the Base model and the
raw (no template) encodes of the same 230 captions.

## Claims

1. **The Base SAEs reconstruct the instruct TE's caption rows at FVU 0.25–0.46**, with
   the card encoder and no `b_dec` centring (the centred variant is far worse). The
   features change under the instruct template (top-100 code overlap 0.57 against 0.92
   under Base framing) but stay reconstructable (FVU +0.03).
2. **About a third of what the DiT reads sits in the SAE error term.** Energy into
   `text_embedder.proj`: reconstruction 59 %, error 31 % (TE side 73 / 26 %). This
   bounds any SAE-space edit.
3. **The DiT's projection suppresses the structural features and amplifies the
   tag-content ones.**

## Reconstruction

FVU on the caption body, layer 31 / 15:

| | layout body | NL body | NL raw (no template) | template prefix | template suffix |
|---|---|---|---|---|---|
| L31 | 0.34 | 0.25 | 0.21 | 0.31 | 0.31–0.36 |
| L15 | 0.46 | 0.41 | 0.24 | 0.07 | 0.49–0.60 |

base_contrast, head dropped (first 4 body tokens, the raw sink), L31 layout / NL: Base raw
0.30 / 0.20, Base template 0.30 / 0.21, instruct raw 0.31 / 0.23, instruct template 0.34 /
0.27 (L15 0.41–0.47 throughout). The first run's L15 raw-vs-template gap (0.24 vs 0.41)
does not reproduce with the head dropped (0.39 vs 0.43): it was the raw sink token
inflating the variance, not the template.

## DiT side

Layer 31 body tokens; `P` = `text_embedder.proj` linearised through the final norm (the
two text blocks after `P` are not modelled).

- Readability `‖P(g⊙d)‖ / ‖g⊙d‖`: random directions 2.08 (median), SAE features
  1.71–3.09 (1–99 %), actual body tokens 1.93.
- The top 100 features carry 49 % of feature energy into the DiT, mostly structural
  (JSON `":"`, the `@` artist marker, bbox digits, a near-always-on f13752).
- `P` suppresses the high-energy generic and structural features (read 0.8–1.0: f26383
  with 15 % of TE-side energy, the `@` marker, JSON delimiters, the first token of a
  quoted string) and amplifies low-energy tag-content features, mostly second subwords
  of multi-token tags (read 3.6–4.0: `pantyh|ose`, `thigh|high`, `tor|og|ao`,
  `rand|oser|u`, `speech| bubble`).

## Where the SAEs were used elsewhere

- Name knowledge: top-100 overlap at a name's last token (character_knowledge.md). The
  strongest features there are structural, so QA and cloze decided the question.
- LoRA attribution: the channel LoRA's gradient × input split over SAE features cancels
  heavily (Σ|c| = 4.6× the token-level total) and its top features are the generic and
  structural ones; no semantic feature stands out. The token-level read
  (text_read_path.md §4) is the usable one.

Not done: a causal check on the SAE side (ablate / boost single features in the TE,
generate with a fixed seed).
