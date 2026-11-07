# Opinion on the lora_blocks proposal

Written 2026-11-07 against [proposal.md](proposal.md) as committed in `1beec613`, before
anything in it was built. A reading of the idea and where it sits in the literature, not a
review of the numbers. The checkpoint read it asks for is at the end.

## Summary

The gate is the finding; the block system is a bet riding on it. A parameter-free,
monotonic, per-query handle on the reference that leaves the rest of the page at 5.9–6.1
luma L1 is a shippable thing on its own (an "edit here" / "edit strength" input on the
Reference node), and it addresses exactly the failure class the proposal cites for
sanitize: the faint stamp, the watermark that goes on some seeds only. P0 may carry most
of the value of the line, and the proposal half-says so ("then G alone is the result").
P0 is a result in its own right, not a gate for P1.

## Points

**The crux is mis-ordered.** "How much of an existing task LoRA's training goes into the
copy?" sits under Open questions, but it decides whether C saves anything. Prior: the copy
head forms early. Identity is the strongest and lowest-σ signal in every pair set, so most
of sanitize's 2,216 steps go to *where* and *what*, not to copy. If the checkpoint read
confirms that, C's saving in steps is small and the win should be restated as **data**: G
externalises "where", so a task needs fewer pairs, not fewer steps. Hence the read below,
before P1.

**Train-with-G, deploy-without-G will not work, and should not be a goal.** A block
trained under a gate has never had to decide where; asking it to at inference is a
train/test mismatch. Two consistent choices: always ship with a mask (cheap for sanitize
and swap_text), or run the one cheap experiment that could lift the restriction — train
with G *and* the bbox in the prompt, test with the bbox alone. Base already turns a layout
bbox into a location code over blocks ~14–35, so this distils the gate into a pathway the
model has. That is the version that reaches users without a mask.

**`diag` is the right inductive bias; give it a schedule.** The own-cell edge is the whole
copy signal, and own-cell is exactly what text removal must suppress, so cutting ±k while
the box still reads the rest of the reference (style, font, surroundings) is clean. The
Edit LoRA's own gate forms *during* sampling (copy inside the edit region 0.41 → 0.17),
which argues for strong-early / relax-late over σ: lay the balloon out without the text,
then borrow texture.

**Colorize does not belong in round one.** It is a global value transform, not a gated
edit; bending C to luma to fit it dilutes the thesis. Sanitize and swap_text fit the
frame; prove it there.

**Composition 1 (frozen stack) is the default; 2 is optional.** The 0.06–0.18 early/mid
subspace overlap between two runs of the same task says independent blocks are roughly
non-interfering *and* non-cooperative: a sum does each block's thing separately, which is
fine when their regions are disjoint (C outside, L inside the `diag`-cut box). On
concat-rank init: the shared seed-0 kaiming `down` is not an advantage. At step 0
`[B₁ B₂]·[A;A] = (B₁+B₂)A` has rank r, not 2r, and the fine-tune must break a symmetry.
Orthogonal Adaptation argues the opposite (orthogonal downs merge cleanly).

**Resolution.** The Edit LoRA's copy collapses at 1360 (outside its training size). C
trained across the native-aspect buckets should carry the copy at every size we render;
worth a read in P1.

## Related work

- **ACE++ (Mao et al., 2025) and FLUX.1 Fill.** C almost exactly: a fill / inpaint LoRA
  trained on self-supervised masked pairs as the shared base for local edit, subject and
  portrait tasks, each a short task LoRA on top. The strongest precedent for the idea and
  a source for the mask-sampling recipe (text-shaped holes from other pages is a twist
  they do not have).
- **KV-Edit (2025), MasaCtrl (Cao et al., 2023), regional attention masks for DiTs
  (instantX training-free regional prompting; RAG-Diffusion).** G is a per-query mask on
  a cached reference KV — KV-Edit's background preservation, with `ref_isolate` making the
  reference a literal KV cache. Blended Latent Diffusion and DiffEdit are the latent-blend
  ancestors.
- **Induction heads (Olsson et al., 2022).** "No slice carries the copy" is the vision
  analogue of induction heads needing a previous-token head upstream: b39 h19 is the
  induction head, earlier blocks write the matchable keys. It names a probe not yet run:
  ablate block ranges upstream and watch h19's own-cell mass, to find the other half of
  the circuit.
- **Task arithmetic (Ilharco et al., 2022), TIES-Merging, ZipLoRA, Multi-LoRA Composition
  (Zhong et al., 2024), Mix-of-Show, Orthogonal Adaptation (Po et al., 2024).** P4's
  interference question with its known mitigations; LoRA Switch / Composite (compose at
  denoising steps rather than in weights) if sums interfere.
- **GLIGEN, training-free layout guidance, BoxDiff.** The "where from text" path; the
  bbox → location-code finding is its DiT form.
- **manga-image-translator (comic-text-detector + LaMa).** The detector + inpaint
  pipeline that sanitize-as-C+G competes with. P2's pass should include beating it: that
  is what the DiT has to justify (style-consistent fill, SFX-aware, soft boundary).

## One more idea

G's monotonicity makes it a free training-time augmentation: a random bias on random
regions while training any task LoRA teaches "edit here, copy there" as a controllable
input instead of something inferred from the pairs. Cheaper than C, testable on the next
swap_text micro arm.

## Checkpoint read: when does the copy form?

Run the same day: [reports/copy_vs_step.md](reports/copy_vs_step.md). The copy head is a
phase change at steps ~175–225 of any pair set; most of the pixel copy (L1 15.4 → 8.0)
arrives before it through own-cell heads Base already has in the early blocks; after it,
what keeps training is breadth (44 → 80 heads over 0.1 by step 2216). So C saves pairs and
the transition, not steps of a long run, and P1 is 300 steps on the int8 plan plus the
hole / `diag` gate, which is the part still to test.
