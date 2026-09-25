# influence_smoke — first contact for idea.md's validation influence (2026-09-25)

`experiments/influence_smoke/` (the line's new bench-style idea-validation
tree), run on run0925_300f's artifacts so the ground truth is a run with a
known verdict. Envelope:
`experiments/influence_smoke/results/20260925-1903-smoke1/`; bank tensors
`output/cjk_anima_scale/influence_smoke1/bank.pt`; 19 min on GPU
(6 cells × 192 items × 2 table points + 5 dev strings × ~24 held-out
scene_piece composites × 3 σ bins × 2 noises).

## Numbers

- **E1 (v_s sanity)**: 98.7–100 % of ‖v_s‖² sits on the string's own rows.
  v_s is effectively diagonal in row space.
- **E2 (linearization, seed → 300f table)**: pooled sign agreement 5/5 —
  the 300f delta *reduced* in-box dev FM loss on all five dev strings
  (measured +0.005…+0.018), linear prediction 2–5× over (drift 1.66 is far
  outside the linear regime; per-cell gradient drift cos(ḡ seed, ḡ moved)
  0.14–0.40 says the same). High-σ bin signs disagree on 3/5 strings.
- **E3 (cell ranking by I[c,s] = v_sᵀ ḡ_c)**: inverted vs the rulers —
  scene_piece cells last (I ≈ 0, several exactly 0), scene_short /
  grid_string / scene_sentence on top.

## Verdict, two layers

1. **Estimator flaw (fixable).** Because v_s is diagonal, I[c,s] reduces
   to the s-row component of ḡ_c — and an *unstratified* 192-item cell
   sample contains 0–2 items touching s (≈18/5334 scene_piece items per
   string), so E3 is touch-count noise; the exact zeros are cells whose
   sample never drew s at all. Any bank v2 must sample **row-conditioned**
   (items containing s, per cell) — the per-row-per-cell unit the
   exposure ledger already implied.
2. **Objective scope (the binding one).** The dev target (in-box FM loss
   on product-shaped composites) improved across a delta whose
   acceptance rulers read ≈ 0. With the piece read
   (`piece_2026_09_25.md`) this resolves precisely: the loss tracked
   the **real piece-native gains** the run bought (すごい, a dev string,
   went 1 → 6 official), but it cannot see what the acceptance axis
   measures — doubling and sentence assembly. So the surrogate as
   specced can price *piece-identity* movement, and **cannot choose
   shares for Axis 1** (`native_sent` full-string).

## Standing

**Superseded 2026-09-25 by `influence_target_2026_09_25.md`**: the layer-2
reading below — that the dev loss tracked the run's real piece-native gains —
is withdrawn. Measured per piece, the loss gain ranks the unrendered
3+-glyph pieces first (ちょっと +0.020, ありがとう +0.016) and the one piece
with a real official gain at zero (すごい −0.001); ρ vs piece +0.02. The
loss read the pseudo-text line, on rendered and unrendered pieces alike.
Loss-target influence is closed for both axes; (a) below is moot and (b)'s
certified-delta anchor is the only candidate left.


Full gradient bank: **on hold.** If revived, two changes are
prerequisites, in order: (a) row-conditioned stratified banking; (b) a
validation target that sees the acceptance axis — the candidate is
anchoring on certified deltas (Δ of ruler-verified tables, e.g.
micro_warm_0923's piece rows, against Δ_300f as the negative anchor)
instead of dev-loss gradients; untested. The calibration gate of
`idea.md` § Calibrate stays: rank the known tables or don't use it.

Repro: `run_exp.py --label smoke1` (defaults), `--dry_run` for the plan;
submitted via `make daemon-run`, `ANIMA_VOCAB_PACK` set.
