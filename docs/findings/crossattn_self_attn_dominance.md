# Cross-attn vs self-attn drive across σ — text writes the low-σ plan early, self-attn + MLP render it; the dominant pathway is self/MLP at every σ

> **ARCHIVED (2026-07-12).** The measurement bench moved to
> `_archive/bench/cross_attn_drive/` — mechanism questions answered, the
> exploitation lever shipped as `--xattn_boost` (see
> `docs/inference/xattn_boost.md`, `bench/frontload_text_boost/`). The glyph
> scatter probe lives on in `bench/turbo/glyph_scatter_probe.py` (turbo
> text-loss diagnosis). Paths below are pre-archive.
>
> STATUS (2026-06-28). Measurement-only finding. Two probes added under
> `_archive/bench/cross_attn_drive/`: `attn_evolution.py` (cross-attn map re-routing —
> "where patches look") and `attn_contribution.py` (gated residual contribution
> per pathway — "what each pathway writes"). Extends the velocity-level result
> [[project_crossattn_drive_frontloaded]] and [[project_crossattn_map_evolution]].
> No code path changed.
>
> UPDATE (2026-06-29). Added Result 4 — `attn_evolution.py` run on literal
> text-in-image glyph strings (the text a speech bubble "reads"), EN + KO,
> tracked as their own token columns. Confirms the text wall for Korean and tests
> (and tempers) the idea that cross-attn mass could measure glyph-rendering
> difficulty: it is prompt-composition-sensitive and sink-confounded, not yet
> a portable metric. No code path changed.

## The question

[[project_crossattn_drive_frontloaded]] showed the velocity-level text drive is
front-loaded: `‖v_cond − v_uncond‖/‖v_cond‖` peaks at σ=1 and `cos(v_cond,v_uncond)`
→0.997 below σ≈0.85, so below mid-σ text only rescales the base velocity. Open
question: is the cross-attention pattern itself frozen early, and is the late-σ
detail formation driven by self-attn rather than cross-attn? If so, why, and what
does it imply for adapters?

Two complementary probes, both baseline-free (single generation, no full−drop diff,
so the tag-dropped-baseline quality confound that pollutes `tag_influence` cannot
touch them):

1. `attn_evolution.py` — WHERE. Per block per step, the eager softmax cross-attn
   map (image-patch × text-token, head-avg), reduced to drift (1−cos vs prev
   step = re-routing) and mass (attention on a tag's key columns). Tag→column
   mapping is exact via qwen3 token-id subsequence match against the padded
   `input_ids`.
2. `attn_contribution.py` — WHAT. Mirrors `Block._forward` (bit-identical;
   verified `max|Δlatent| = 0.0`) to log the L2 norm of each gated term added to
   the residual: `gate_self·self_attn`, `gate_cross·cross_attn`, `gate_mlp·mlp`.
   The gates are adaln-modulated by the timestep, so this captures the σ-scheduled
   down-weighting that the WHERE probe is blind to.

## Result 1 — the cross-attn map never locks, but its re-routing rate is front-loaded

Whole-map drift is meaningless — 512 text positions are mostly padding sinks
(pad-id 151643), which dominate the cosine and dilute drift to ~0.006→0.002. Scope
to the tag's token columns. On real captions (18 cap × 2 seed × 28 step), per-tag
Δσ-normalized re-routing rate (`drift/Δσ`):

| band (σ) | rate (speech bubble) |
|---|---|
| hi (>0.9) | ~1.65 |
| mid (0.45–0.7) | ~0.29 |
| low (<0.45) | ~0.23 |

The rate collapses ~8× from the high-σ burst to a flat non-zero floor (~0.2–0.3)
that persists to σ=0.1 — the map keeps churning but never re-intensifies (a raw
per-step "tail rebound" is a wide-Δσ artifact; it vanishes on normalization). Tag
attention mass declines monotonically and sits below per-token uniform
(2/512≈0.004) for the generic text tags — they are under-attended and the late
churn rides a shrinking budget. Reframe: the low-σ drift floor is not text
re-engaging — mass is falling — it is the *image-derived queries* drifting as
self-attn sharpens the patches, dragging the cross-attn map along mechanically.

## Result 2 — self-attn + MLP dominate the residual at every σ; cross-attn is a small, fading term

Gated contribution L2 norm (base model, 12 cap × 2 seed × 28 step):

| band (σ) | self | cross | mlp | cross_frac = cross/(self+cross) |
|---|---|---|---|---|
| hi (>0.9) | 248 653 | 28 498 | 254 358 | 0.151 |
| upper_mid | 281 332 | 25 264 | 274 210 | 0.101 |
| mid | 301 985 | 22 269 | 291 838 | 0.076 |
| low (<0.45) | 292 827 | 19 393 | 286 331 | 0.064 |

- self-attn dominates cross-attn at the highest σ already (`self_dominant_below_sigma
  = 1.0`): ~9× at σ=1, growing to ~15× at low σ. There is no σ where cross-attn
  "takes over"; the user's "self-attn dominant from σ<0.9" is an understatement —
  it dominates throughout.
- cross-attn's share is front-loaded: 15% of the attention update at high σ →
  6% at low σ. Including MLP, text is ~5%→3.5% of the total residual update.
- self and MLP GROW into mid/low σ (detail-formation phase) while cross
  shrinks — direct confirmation that low-σ detail is rendered by self-attn + MLP,
  not by cross-attn re-reading the prompt.

## Result 3 — the sincos style LoRA rides the self/MLP stream, not cross-attn

Same probe, base vs `anima_sincos2` LoRA (identical captions/seeds):

| band | Δself | Δcross | Δmlp | cross_frac base→lora |
|---|---|---|---|---|
| hi | +4.5% | +3.4% | (up) | 0.151 → 0.148 |
| mid | +4.6% | +5.1% | (up) | 0.076 → 0.075 |
| low | +5.4% | +2.9% | (up) | 0.064 → 0.060 |

The LoRA amplifies all three pathways ~uniformly (+3–5%) and leaves the
self/cross/MLP balance unchanged (cross_frac flat). Because self+MLP carry
85–94% of the update at every σ, the overwhelming majority of the LoRA's added
signal flows through self+MLP, not cross-attn. The companion WHERE probe agrees on
its own trigger: for `@sincos`, the base model already attends strongly and stably
(mass ≈0.0206 ≈ 2.6× uniform, roughly flat across σ — unlike the under-attended text
tags), and the LoRA changes that allocation by ~0 (slightly lowers it,
Δmass −3e-4…−6e-4). So the adapter does not steer by making patches look harder at
its trigger; the trigger is a switch, and the style is delivered downstream.

This cross-checks [[project_lora_crossattn_learns_labeled_only]] (text tags are a
data/capability limit, not a cross-attn-mass deficit the adapter can fix) and
explains why late cross-attn levers are inert ([[project_tag_boost_late_sigma_kill]]):
there is almost no cross-attn budget to lever below mid-σ.

## Result 4 — literal glyph tokens hit the same wall (EN + KO), but attention mass is not yet a portable glyph-rendering metric

Probed the actual text-in-image string a speech bubble "reads", tracked as its own
token columns alongside the `speech bubble` / `*text` category tags (base model,
8 cap × 2 seed, 1024/28, cfg 4). Two contexts:

- Run A (clean): `speech bubble` + EN glyph `this is anima image`.
- Run B (crowded): `speech bubble`, `korean text`, `english text`, EN glyph
  `this is anima image`, KO glyph `아니마 이미지 입니다`.

Mass is shown per token — raw mass is summed over a tag's columns, so a 5–6
token glyph inflates ~2.5–3× vs a 2-token tag purely by span; always divide out.
Per-token uniform = 1/512 = 0.00195.

| run | tag | kind | tok | rate hi→low | mass/tok hi→low (×unif) |
|---|---|---|---|---|---|
| A | speech bubble | category | 2 | 1.97→0.21 | 1.21× → 0.58× |
| A | this is anima image | EN glyph | 5 | 1.50→0.24 | 1.06× → 0.55× |
| B | speech bubble | category | 2 | 1.89→0.20 | 0.81× → 0.32× |
| B | 아니마 이미지 입니다 | KO glyph | 6 | 1.37→0.21 | 0.81× → 0.46× |
| B | english text | category (irrel) | 2 | 1.22→0.16 | 1.14× → 0.79× |
| B | korean text | category (match) | 3 | 0.90→0.11 | 2.40× → 2.06× |
| B | this is anima image | EN glyph | 5 | 0.48→0.04 | 2.08× → 1.99× |

Two robust signals:

1. Re-routing: every column is `LOCKS_EARLY` (english text borderline
   `PARTIAL_DECAY`). The KO glyph behaves exactly like the EN glyph and the
   categories — front-loaded burst, then floor. Adding a non-Latin script or a
   category tag changes nothing on the WHERE axis. The Korean glyph hits the same
   wall (Result 1).
2. category ≫ literal content. The matching category `korean text` sustains high
   mass (2.40×→2.06×) while the KO glyph it labels decays to under per-token uniform
   (0.81×→0.46×, same trajectory as `speech bubble`'s own content). Cross-attn handles
   "there is Korean text here" far better than the specific characters — the model
   holds no sustained token-level attention on the glyphs where they'd be drawn. Direct
   read of why the bubble's presence renders but its text garbles, and a multilingual
   confirmation of [[project_lora_crossattn_learns_labeled_only]].

Why mass is *not* (yet) a glyph-rendering metric — the context-flip. The same
EN glyph `this is anima image` reads clean-decaying in Run A (1.06×→0.55×, rate
1.50) but flat-high in Run B (2.08×→1.99×, rate 0.48) — identical tokens, the only
change is Run B's prompt carrying 3 more text tags. Flat-high mass + near-zero
re-route rate is the attention-sink signature (same family as the padding sink in
Result 1): the patches dump a constant, un-churning budget onto a few positions —
almost certainly the ordinary content-word subtokens (`image`, `text`), not the glyph
content — and softmax redistributes which positions sink as the prompt composition
changes. So absolute cross-attn mass is prompt-composition-sensitive and
sink-confounded: a high reading can be sink absorption rather than rendering
fidelity, and the value isn't portable across prompts. What is comparable is the
within-a-fixed-prompt re-routing verdict (all `LOCKS_EARLY`) and the
category-vs-glyph mass gap.

OPEN — is there a glyph-rendering proxy here at all? A usable one would need
(a) per-column mass to strip the sink subtokens (the recorder currently sums over
a tag's columns — a per-column variant is a small change), and (b) calibration against
by-eye legibility — which Anima has no quality reward to anchor
([[project_null_tta_phase0_bounded_nudge]]). Until then, read the glyph result as
"confirms the text wall (EN + KO), via re-routing + the category-vs-content gap", not
as a rendering-quality number.

## Result 5 — a glyph to render extends text drive into the feature-commit band; mass couldn't see it, the velocity delta can

Result 4 left "is there a glyph proxy here at all?" open on the mass axis. There is
one on the velocity axis. Same production CfgDelta probe as Result 1's companion
(`ANIMA_CFG_DELTA_LOG`), base DiT, 14 real captions × 2 seeds × 5 arms, 1024/28/cfg 4,
paired by (caption, seed). Probe: `bench/frontload_text_boost/glyph_delta_probe.py`,
run `results/20260716-2207-glyph-delta/`.

Each arm appends a clause to the same real caption. A vs C alone confounds three
things — a clause was appended, it is longer, it carries a glyph — so the controls
split them (T5 tokens of the clause in parens; the cond rows are T5 positions):

| arm | appended clause | tok |
|---|---|---|
| `A_bare` | — | 0 |
| `B_category` | `, there is a speech bubble` | 6 |
| `C_glyph` | `, … that reads "say hello from anima_lora"` | 20 |
| `D_glyph_id` | `, … that reads "But it was fast"` | 15 |
| `E_nonglyph` | `, … that is round and white and slightly tilted` | 16 |

Read the PAIRED delta, never the per-arm means. Cross-prompt std of `ratio` is
~0.078 at σ=1 and swamps a ~0.004 effect — which is exactly why the original probe
concluded "prompts are near-indistinguishable in drive below σ≈0.7". That was a
cross-prompt read; paired, the std drops to ~0.0025 and the effect resolves. Read
`rel_dDelta` (the numerator ‖v_cond − v_uncond‖), not `ratio`: a ratio move could
come from the ‖v_cond‖ denominator instead. It doesn't here — `rel_dVcond` stays
within ±0.6% on every pair below.

Band = σ ∈ [0.5, 0.8], the feature-commit band (n=28 cells):

| pair | what it isolates | rel Δ‖delta‖ | n_pos |
|---|---|---|---|
| `B − A` | a category clause at all | +1.71% | 15/28 — null |
| `E − B` | pure token count (+10 tok, no glyph) | −1.18% | 13/28 — null |
| `D − E` | length-matched glyph vs non-glyph (15 vs 16 tok) | +6.77% | 25/28 |
| `D − B` | plain-word glyph | +5.23% | 26/28 |
| `C − B` | OOD glyph | +8.55% | 25/28 |
| `C − D` | OOD token axis (both glyphs) | +3.14% | 22/28 |

- The front-loaded peak is untouched. At σ=1.0 every arm sits at ratio
  0.275–0.280, min cos 0.955–0.957. A glyph changes nothing at the top.
- A new mid-σ bump appears, σ≈0.8→0.45, peaking at σ≈0.66 (ratio: bare 0.0352 →
  `D` 0.0380 → `C` 0.0393). `C − A` is 28/28 sign-consistent at σ=0.66 and 0.625.
- It is not length. `E` adds 10 tokens over `B` and produces a coin flip
  (−1.18%, 13/28). The `D − E` contrast is therefore load-bearing: at matched
  length, only the glyph arm drives. Note `E` is a conservative control — "round
  and white and slightly tilted" carries visual-attribute demand of its own — so
  +6.77% understates the glyph effect.
- It is glyph-generic, not `anima_lora`-specific. `D` reproduces the bump with
  four plain English words. The effect decomposes near-additively:
  `C − B` (+8.55%) ≈ `D − B` (+5.23%, glyph-generic) + `C − D` (+3.14%, OOD). The
  OOD term is also not length (`E − B` is null at +10 tok while `C − D` is +3.14%
  at +5 tok) — `anima_lora` shredding into `▁/anim/a/_/lor/a` demands more drive than
  in-distribution words do.
- Still rescale, not rotate. Band cos is 0.9991–0.9992 across every arm. The
  glyph makes text drive harder in the band; it does not rotate the velocity away
  from the base prior, so CFG's linear extrapolation still spans it. The "rotational
  headroom lives only in the first 2–3 steps" verdict stands — a bigger collinear
  push is what CFG already does.

Why this matters for the localized lever. `bench/frontload_text_boost`'s Phase-1
verdict gated the localized feature-commit-band boost on finding "a regime that
demonstrably DROPS small features". Glyphs are that regime, and this says the model
is already spending extra drive there — the deficit is not that text goes quiet in
the band. Combined with the collinearity, that argues the missing piece is upstream
(the early low-freq plan never encodes the glyph), consistent with the Interpretation
below — not something a band-localized gain multiplies into existence.

Caveat — the probe is direction-blind *between arms*. It records ‖v_cond −
v_uncond‖ and cos(v_cond, v_uncond), both reads against uncond, never
cos(Δ_armX, Δ_armY). So "`B − A` is null" means the category clause doesn't change
how hard text drives — obviously it changes where (a bubble appears). Comparing
arm directions needs the delta vectors, not their norms.

## Interpretation — coarse-to-fine division of labor

Flow-matching denoising is coarse→fine in frequency: high σ sets low-frequency
structure (layout, identity, color blocks), low σ fills high-frequency detail
(texture, edges, small parts). The pathways split along this axis:

- Cross-attn writes the low-frequency plan early. Text influence (velocity
  delta, re-routing rate, and contribution share) all peak at high σ and fade. By
  mid-σ the cond/uncond velocities are near-parallel — text only rescales. Result 5
  refines the magnitude of that rescale without touching the direction: it is
  content-dependent, and a glyph to render buys ~+7% of extra drive through the
  feature-commit band (cos still 0.999). "Text goes quiet at mid-σ" is a
  cross-prompt average; paired, prompts are distinguishable there.
- Self-attn + MLP elaborate the plan into high-frequency detail late. Their
  contribution grows as detail forms; a late-emerging detail's cause was committed
  early by text, but its low-σ velocity is computed by self/MLP. Emergence-time ≠
  causation-time, and the late cross-attn map churn is a downstream consequence of
  self-attn sharpening the (image-derived) queries, not text re-asserting itself.

Adapter consequence: a style/identity LoRA cannot work by re-routing or
up-weighting cross-attn — there is no budget there. It must (and the sincos LoRA
does) ride the dominant self/MLP stream, triggered by an early, already-well-attended
token. Capabilities that genuinely need new text→pixel routing (readable glyphs,
speech-bubble text) can't be bought by amplifying late cross-attn; the deficit is
that the early low-freq plan never encodes them and self-attn has no prior to
elaborate them.

## Caveats

- The contribution probe measures magnitude, not direction. "cross_norm +3%"
  means the cross term's size is ~unchanged; the LoRA could rotate the cross
  contribution's direction (inject style via the cross value path) while keeping
  its norm flat. The where/what attribution rests on the magnitude argument
  (self+MLP adds dwarf cross adds) + cross_frac invariance, not on a claim that
  cross content is identical. A direction-resolved follow-up (cosine of the
  per-pathway LoRA delta-contribution) would close this fully.
- Norms are over the full flattened latent (4096 patches × 2048 dim); only ratios
  are meaningful, not absolute values.
- Probes run with `compile_blocks=False` (the eager hooks/mirror must run every
  step). Generation is otherwise unchanged.

## Reproduce

```
# NB the probes live in _archive/ now; their sys.path bootstrap assumes bench/<x>/,
# so run them with PYTHONPATH=. from the repo root:
PYTHONPATH=. uv run python _archive/bench/cross_attn_drive/attn_evolution.py   --captions 12 --seeds 2 --tags "speech bubble,japanese text,english text" [--lora_weight …]
PYTHONPATH=. uv run python _archive/bench/cross_attn_drive/attn_contribution.py --captions 12 --seeds 2 [--tags "@sincos"] [--lora_weight …]

# Result 4 — literal glyph columns. The tracked tags must appear as comma entries in
# each caption (caption_has_tag selection + exact token-id subsequence match), so pass
# a --prompts file of tag-list captions that each carry every tracked tag, e.g.
#   "1girl, …, speech bubble, korean text, english text, this is anima image, 아니마 이미지 입니다"
PYTHONPATH=. uv run python _archive/bench/cross_attn_drive/attn_evolution.py --prompts <glyph_captions.txt> \
  --tags "speech bubble,korean text,english text,this is anima image,아니마 이미지 입니다" \
  --captions 8 --seeds 2 --label glyph_ko_en
# NB read mass PER TOKEN (divide by tag token count); flat-high mass + low re-route
# rate = attention sink, not rendering fidelity (see Result 4 context-flip).

# Result 5 — glyph vs the CFG-delta. This one is LIVE (it feeds that bench's open
# localized-boost gate) and rides the production CfgDelta probe, no hooks:
uv run python bench/frontload_text_boost/glyph_delta_probe.py --label glyph-delta
uv run python bench/frontload_text_boost/glyph_delta_probe.py --check-tokens   # asserts D/E stay length-matched
# Re-analyze an existing log without regenerating (~28 min of GPU otherwise):
uv run python bench/frontload_text_boost/glyph_delta_probe.py --log <cfg_delta.json>
```

Results: `_archive/bench/cross_attn_drive/results/*-contrib_{base,sincos}/` (+ `attn_contribution.png`),
`*-tag_{base,sincos2}/`, `*-sincostag_{base,lora}/`,
`*-glyph_vs_bubble/` (Run A), `*-glyph_ko_en/` (Run B).
