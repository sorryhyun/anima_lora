# PE towers have no patch-token outliers — test-time registers dies at Phase 0a

**Verdict: NO-GO (premise absent).** The load-bearing premise of the
test-time-registers port (`_archive/proposals/test_time_registers_pe.md`, paper
2506.08010) is that PE-Core / PE-Spatial, as ordinary ViTs trained without
registers, manufacture sparse high-norm outlier patch tokens. **They don't.**
Measured 2026-07-20; line closed at the first gate, as the proposal's own kill
criterion required.

## The probe

`outlier_probe.py` (now `_archive/bench/pe_registers/`) — 256 tagger-manifest
images through both towers (fp32, per-image aspect buckets, forward hooks on
every residual block; CLS tracked separately from patch tokens). Result
envelope: `_archive/bench/pe_registers/results/20260720-2014-phase0a/`.

## What the data showed

Across 256 images × every layer (24 in PE-Core-L14-336, 12 in
PE-Spatial-B16-512), **not one patch token ever reached even 5× the per-image
median norm** — the paper's outlier populations sit at 10×+. The patch-norm
distribution is tight everywhere:

| Tower | max/median ratio, mean (worst layer) | ratio p90 (worst layer) | patches ≥5× median, any layer, 256 imgs |
|---|---|---|---|
| PE-Core-L14-336 | 1.89 (layer 0) | 2.22 | **0** |
| PE-Spatial-B16-512 | 1.64 (layer 0) | 1.83 | **0** |

The ratio *decreases* with depth on both towers (mid-stack ~1.3) — the
opposite of outlier formation, which grows through the stack.

Meanwhile the CLS token does exactly what the outlier literature predicts a
scratch/sink token should: its norm explodes with depth while the patch grid
stays clean — PE-Core 34 → 923 (27× the patch median by the last layer),
PE-Spatial 17 → 425 mid-stack. Both PE variants ship `use_cls_token=True`,
and the CLS is evidently already absorbing the global/sink role that
register-free ViTs dump into patch tokens. (OpenCLIP/DINOv2 in the paper have
CLS too and still leak into patches, so CLS alone doesn't explain it — PE's
training regime (progressive resolution, RoPE-2D) presumably does. We only
claim the measurement, not the mechanism.)

## What this settles

- **The cached PE features are not contaminated by register-style outliers.**
  The proposal's one surviving argument against the tagger-ceiling probe —
  common-mode token damage invisible to head-vs-head comparison — is now
  directly measured and absent. The ceiling result
  ("tower fine, head/label-space is the headroom") stands *stronger*.
- **Phases 0b–2 are dead**: no outlier positions ⇒ no register neurons to
  find, nothing to relocate, no cache rebuild, no head retrain. Do not
  re-propose MLP register-neuron editing on these towers.
- The paper's mechanism was never tested against a tower like PE; this is a
  clean negative about **PE's regime**, not evidence against the paper (whose
  OpenCLIP/DINOv2 results we have no reason to doubt).

## Reusable

`_archive/bench/pe_registers/` keeps the harness (archived 2026-07-20, indexed
in `_archive/shelved_benches.md`): `outlier_probe.py` (norm census, gate) and
`find_neurons.py` (Algorithm 1 — register-neuron ranking with per-image /
per-bucket consistency + activation-map alignment gates, smoke-tested
end-to-end). If a future vision tower (a PE successor, a different aux
encoder) is suspected of carrying outliers, the whole Phase 0 harness is
already built — point `--manifest` at any stem/image-path manifest.
