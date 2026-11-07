# The copy: what it is, when it forms, what a copy block can save

Sources: [reports/copy_vs_step.md](../reports/copy_vs_step.md),
[`../moss_te_sae/reports/copy_protocol.md`](../../moss_te_sae/reports/copy_protocol.md),
[`../moss_te_sae/reports/i2i_copy_head.md`](../../moss_te_sae/reports/i2i_copy_head.md).

## Conclusions

- **Every i2i LoRA learns the same copy circuit.** An own-cell copy head at b39 h19, which
  Base lacks (own-cell mass 0.000), with b37 h20, b36 h21 and b40 h20 beside it. Our
  sanitize, swap_text and uncensor LoRAs all have it, and so does an outside edit LoRA
  trained on other data with another trainer. The i2i LoRAs share a late-block output
  subspace that t2i runs do not.
- **The copy is spread over the network, not held by a slice of it.** Transplanting b39 h19,
  its q / k, the ten top copy heads or blocks 36–41 leaves the copy at Base level. The head's
  match needs image queries and reference keys that the earlier blocks have shaped. A whole
  LoRA is the smallest unit that can be reused.
- **The copy is cheap and comes early.** The head forms in a phase change at steps ~175–225
  of any pair set (h19 0.06 → 0.42 → 0.60), roughly the same step count on 277 pairs and on
  66. Most of the pixel copy comes before it (L1 15.4 → 8.0 by step 100) through Base's
  early own-cell heads (b5 h3, b8 h14, b14 h15), which training leaves in place. Fine
  detail becomes exact only once the late head is on.
- **After the head forms, training adds breadth.** Heads over 0.1 go 44 → 80 between steps
  300 and 2216 while h19 gains 0.2 and L1 6.3 → 5.8.
- **The copy needs no identity pairs.** It is learned from the unedited part of any pair.

## What follows for the line

- A copy block saves 200–300 steps: 10–15 % of a sanitize run, 5 % of a swap_text run. It
  does not shorten a long run. What it changes is where a task fine-tune starts (past the
  phase change, every gradient going to the edit) and what the pairs have to teach (nothing
  about copying). Its win should be measured in pairs, not steps.
- Any 300-step run on any pair set already gives a plain copy. C is worth a file of its own
  only if training with the hole and the `diag` gate turns it into a fill block.

## Conjectures (not measured)

- **An induction-style circuit.** Base's early heads match by position (RoPE), and the late
  head copies content once training gives it matchable keys, as an induction head needs a
  previous-token head upstream (Olsson et al., 2022). Two tests: ablate the deltas of one
  upstream block range at a time and watch h19's own-cell mass; ablate b5 h3 / b8 h14 /
  b14 h15 on the step-100 weights.
- **The window is set by the optimiser, not the data.** Uncensor at 66 pairs an epoch
  crosses it at about the same step as sanitize at 277. Learning rate and rank would then
  move it, and pair count would not.
- **Breadth is what carries the copy to other sizes.** The outside edit LoRA's copy
  collapses at a 1360 edge, outside its training size. Training across the native-aspect
  buckets for longer may be what spreads the copy over enough heads to hold at every size.
