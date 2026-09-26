# Opinion — observe and replay line control inside the adapter

2026-09-26. Advisory follow-up to [proposal_factorizedrows.md](proposal_factorizedrows.md).
**No experiment or implementation performed for this note.** Code and existing
reports were read; the mechanisms below are hypotheses unless identified as observations.

Recommendation: run a paired, adapter-only observation and replay study before
expanding FM training. Find whether the known line intervention becomes a simpler,
more transferable control inside the frozen adapter. Then render a small held-out
comparison. Replaying the existing effect and separating composition from doubling
are different outcomes.

## 1. What the evidence warrants

[Stage B](reports/stage_b_2026_09_26.md) finds a shared direction holding 24.4% of
donor delta energy, split-half cosine 0.82, and held-out composition improving
11 → 66 / 160 within one edit. This establishes transferable control, not that
the remaining energy is pure identity. [Transplant-line](reports/transplant_line_2026_09_26.md)
shows composition and learned transitions also live in the residual.

A plausible explanation is that common rendering requirements align gradients
across rows while identity-specific errors supply different updates. That does
not require a uniquely identifiable semantic factorization.

The adapter is not structurally context-free:
[LLMAdapterTransformerBlock](../../library/anima/models.py) has self-attention,
cross-attention to Qwen states, and an MLP, each with residual addition. The older
[manifold probe](../cjk_renderable_anima/reports/rows_manifold_2026_09_16.md)
found delta-output cosine 0.91–0.94 across quote frames, with frame-specific
residual magnitude 0.32–0.38 of the delta. It did not establish invariance to
neighbouring glyphs. A stable dominant component can coexist with useful contextual
interactions.

F1 remains plausible, but gate-on/off exposure only constrains the split; it does
not force it. Stage B's count examples cover 24–40 px while lines reach about
18 px, so failure on unchanged data would not close the parameterization.

## 2. The clean pair: same caption, different rows

For caption `c`, record residual states at each of the six block boundaries:

```text
H_S[l,c] = adapter state with seed rows
H_L[l,c] = adapter state with line-trained rows
D[l,c]   = H_L[l,c] - H_S[l,c]
```

Keep Qwen states, target IDs, positions and masks identical within each pair.
Cache Qwen once per caption. No DiT or image latents are needed for this stage.
Also record self-attention, cross-attention and MLP branch outputs to locate where
the change spreads or becomes context-sensitive. Inspect glyph and non-glyph
positions separately.

Use two teachers:

- **Seed + measured `u_S`:** follows an already demonstrated transferable
  intervention through the adapter.
- **Full line-trained rows:** tests whether their complicated per-row changes
  become a simpler shared effect downstream. This possibility is speculative.

Start with singles, preserving Stage B's 36 donor / 10 held-out split. Use novel
donor strings for validation. The full-row teacher provides no independently
trained target for the ten held-out rows: their eventual render transfer is a
separate test. A `u_S` teacher there tests replay of the existing intervention.

## 3. Extract, patch, and test replay

At each boundary, average `D` over donor glyph positions, balancing glyphs and
lengths, to obtain one candidate `v_l`. Add it at the corresponding positions in
the seed run and execute the remaining adapter blocks.

Compare the resulting final conditioning with the teacher, relative to the
unmodified seed error. A tiny absolute error is uninformative when the teacher
effect itself is tiny. Measure glyph and surrounding-position errors separately.
Check generalization across identities, strings, lengths and quote frames.

Controls:

- Unmodified seed and the existing input-space `u_S` intervention.
- Same-magnitude random activation additions and a small dose sweep.
- Full example-specific residual-state replacement, with identical source and
  masks, as a plumbing/replay ceiling. It should reproduce the teacher suffix;
  this is not evidence of a shared mode.

A donor-derived vector replaying an intervention on unseen identities is the
useful result. A large principal component or a line-versus-alone classifier is
only an observation. If glyph-only injection fails, check whether the effect
travels through surrounding positions before increasing parameter count. Preserve
the production padding/mask contract throughout.

## 4. Paired loss: distill the intervention without FM

If averaging fails, fit only `v_l` through the frozen adapter suffix:

```text
L_pair = E_c || A_suffix(H_S[l,c] + M(c) v_l) - stopgrad(A_teacher(c)) ||²
```

`M(c)` selects the injection positions and gate. Cache the teacher outputs and
seed boundary states. Balance losses over glyph and surrounding positions; many
unchanged tokens must not hide failure at the glyph positions. Gate-off examples
target the seed behaviour. Reserve a small low-rank conditional correction as a
fallback if one vector fails held-out replay.

This is activation distillation, not fitting new glyph identity rows from existing
embeddings, and not the closed correct-versus-doubled one-step FM margin. It offers
a cheap search objective for reproducing an established intervention.

**Its teacher carries doubling.** Perfect distillation can reproduce that defect
perfectly. Neither low loss nor good conditioning cosine certifies correct text.

## 5. What Qwen/context observation can add

Qwen is identical in the same-caption pair, so it cannot explain the difference
between the two row tables. Separately, compare the row intervention across
contexts for an aligned glyph:

```text
interaction[l] = D[l, line context] - D[l, lone context]
```

Vary neighbours, order and length while balancing position and caption frame.
This measures contextual modulation of the row intervention, rather than ordinary
contextual variation. Branch-output or source-state patches can localize that
interaction, but mismatched source/target combinations may be out of distribution.
A length probe in Qwen establishes available information, not a rendering control.

## 6. Decision and relation to factorized rows

Select one internal candidate using held-out adapter replay, then compare it with
gated input-space `u_S` in a small paired render panel. Read exact/≤1-edit text,
excess repetitions relative to the target, omissions, and scene fidelity; include
lone glyphs and legitimate repeated-character strings. Choose dose on separate
validation cases, not the final panel.

- **Same composition, less doubling, preserved scene:** evidence for a better
  intervention site; pursue a context-gated internal mode.
- **Same behaviour including doubling:** useful replay/control, no separation.
- **Good adapter replay, bad renders:** conditioning distance was an inadequate
  selection metric; do not call this successful mode extraction.
- **Only example-specific patches work:** no demonstrated universal additive mode
  at the tested sites. F1 remains open; do not infer an impossibility theorem.

The same conditioning can produce correct or doubled text across diffusion seeds.
Adapter-only observation therefore cannot resolve doubling on its own. This route
does not reopen geometry penalties, shape/IDS encoders, identity initialization
from existing embeddings, pinned-trigger step-saving, flat-heavy mixes, EN-quote
seeding, dropping layouts, or norm reduction as a cure. It tests where an existing
control is best expressed, not whether new identities can be obtained for free.

## References that directly apply

- [Contrastive Activation Addition](https://arxiv.org/abs/2312.06681): paired
  activation differences as steering candidates; an analogy, not evidence for
  Anima glyph/count disentanglement.
- [Activation-patching methodology](https://arxiv.org/abs/2309.16042): patch scope,
  contrast choice and measurement affect causal interpretations.
- [Textual Inversion](https://arxiv.org/abs/2208.01618): embedding-only control of
  a frozen generator; it does not guarantee identity/layout separation.
