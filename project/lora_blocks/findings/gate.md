# The reference gate: "where" as an input

Sources: [reports/p0_gate.md](../reports/p0_gate.md),
[reports/colorize_inboost.md](../reports/colorize_inboost.md),
[`../moss_te_sae/reports/copy_protocol.md`](../../moss_te_sae/reports/copy_protocol.md)
(the edge bias on the Edit LoRA).

## Conclusions

- **The edges are a complete handle on the copy.** Under `ref_isolate` the reference reaches
  the main rows only through main-query → reference-key attention. A bias on those edges for
  the queries in a mask moves the copy monotonically (box luma L1 6 → 34 → 41 → 48 at bias
  0 / −2 / −4 / −∞) and leaves the rest of the page where it was (5.9–6.1 outside). It
  needs no parameters, and it overrides a wrong decision of the LoRA.
- **It fixes the "where" failures of shipped LoRAs with no training.**
  - sanitize v3 + `diag1` in AnimeText boxes removes `usual_case1`'s watermark on seeds 0–2.
    v3 alone keeps it on every seed. Outside the boxes ΔL / Δb stay within v3's own seed
    spread.
  - swap_text ksplit e2 + `boost1` outside the layout boxes stops the LoRA from rewriting
    regions no layout item names. It keeps about 95 % of `boost2`'s correction (user, every
    page), gives the lowest L1 outside on every page and keeps Δb within ±0.1. Letters are
    unchanged.
- **Each task has its own direction.** Removal cuts inside the region (`diag`). Lettering
  boosts outside it and leaves the inside alone. Cutting inside the layout boxes (`diag1`)
  breaks balloons.
- **The gate sets how much a region copies, not how exactly.** On colorize v5 a boost
  inside the text boxes makes the strokes heavier (edge strength 0.91–0.95 → 1.01–1.07 of
  the source's at +2) and leaves the glyphs no closer to the source: the same misshapen
  syllables, the same stray dot in `생`, on every seed. A glyph the LoRA draws wrong is
  drawn wrong at any copy strength.
- **`hard` invents content.** A region that cannot read the reference at all is filled with
  a new object (a grey wedge, a headband, a grid patch). It is not a removal mode.
- **`diag` has a size rule.** It removes what fits in its ±k window and re-reads anything
  larger from its own other cells. Text strokes fit. A flower is repainted at k ≥ 1, and a
  balloon border caught by a mask 16 px past the bbox is redrawn.
- **`boost` shifts tone.** It raises every reference key, not only the own cell: at +2 the
  page's ΔL flips sign and Δb moves +0.1 … +0.8. At +1 the shift is mostly gone.
- **The mask is the limit.** AnimeText at conf 0.05 boxes the watermark and both stamps, and
  misses most of `usual_case2`'s faint URL watermark. On pages v3 already cleans, G adds
  nothing, and `diag1` costs an occasional small mark.
- The faint-stamp ghost cited against sanitize v3 does not reproduce on the int8 generate
  plan, so the only sanitize miss left to fix is the watermark.

## What follows for the line

- G is a result in its own right, before any block is trained. The product is a mask input
  on the Reference node with two controls, a cut inside and a boost outside. For swap_text
  the layout boxes are already in the prompt, so `boost` needs no extra input.
- The assemblies take the mode of each task: sanitize = C + G(`diag`, detector boxes);
  swap_text = C + G(`boost` outside the layout boxes) + L.
- G takes "where" off the LoRA. What the blocks still have to earn is "what" from fewer
  pairs (P2, P3).

## Conjectures (not measured)

- **Why swap_text edits unasked regions.** An image query outside the boxes splits its mass
  between the image being drawn, the layout-JSON tokens and the reference. ctrl's stray
  edits are those queries reading a string where no item put one. `boost` multiplies the
  reference's share by eᵛ, so the copy wins there.
- **Why the copy cannot fix a glyph.** The reference is read cell by cell (16 px), and a
  manga glyph spans two or three cells; what lies inside a cell is drawn by the network
  from what it knows of glyphs. The copy hands it the cell's value, and a stronger copy
  hands the same value harder. Fixing a glyph means changing what it draws there.
- **An own-cell boost (+v on the own cell ± k only) would keep the correction without the
  tone shift**, since the shift comes from raising the non-local reference keys.
- **A σ schedule (strong early, relaxed late) would let `diag` remove larger objects.** The
  Edit LoRA's own gate forms during sampling (copy inside the edit region 0.41 → 0.17), and
  the layout is set at high σ.
- **A tighter mask (text pixels or SAM3, not padded boxes) would make `diag` usable on
  swap_text**, if the broken balloons come only from border cells inside the mask.
- **A random G on random regions while training any task LoRA** would teach "edit here,
  copy there" as a controllable input instead of something read from the pairs
  ([opinion.md](../opinion.md)).
