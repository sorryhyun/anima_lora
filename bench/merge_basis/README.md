# merge_basis — per-artist input basis vs the shared `weight_svd` seed, read at the merge

Question: two closed lines say (a) artist LoRAs collide only on the **input**
side, because every adapter's `A` is seeded from the same `top-r(W₀)`
(`artist_lora_merge_interference_phase0`: in-overlap 0.589, 33× null), and
(b) the seed is render-neutral per artist (`bench/grad_init` E1: `kaiming` vs
`weight_svd` flat) while `A` stays at its seed (same-artist weight_svd-vs-kaiming
in-overlap = null exactly, 2026-09-12). So per-artist random seeds remove the
co-location for free. **Does the merge get better?** Nobody has measured the
behavioural side — phase 0 of the merge-interference line never ran.

`run.py` — one daemon command job:

1. trains Y = `channel_(caststation)` three times (`weight_svd` s42, `kaiming`
   s42, `kaiming` s7; shipped lora.toml recipe, `--deterministic
   --paired_step_rng`), reusing E1's `aak` checkpoints as X;
2. merges 2-way with `scripts/toolkits/merge_loras.py` (global normalize):
   `SHAREDWSVD` (shipped), `SHAREDKAI` (same random seed both sides — isolates
   "shared seed" from "weight_svd"; verified post hoc by its in-overlap),
   `DISTINCT` (different random seeds);
3. reports per-pair in/out-subspace overlap vs the random null (the archived
   phase-0 metric, vendored);
4. renders 24 tag-routed rows (12 `@aak` + the same 12 as `@channel
   (caststation)`, `prompts.txt`) per merged arm and builds two blind sets via
   `project/cjk_aware_anima/probes/blind_pairs.py`, each at its own seed:
   `s26 DISTINCT vs SHAREDWSVD` (shipping question), `s27 DISTINCT vs SHAREDKAI`
   (mechanism).

```bash
make daemon-run ARGS="--stall-timeout 0 --queue bench/merge_basis/run.py --push"
.venv/bin/python project/cjk_aware_anima/probes/blind_pairs.py score --set s26_MB_DISTINCT_vs_SHAREDWSVD
```

Kill / pass are the blind protocol's: inside the 15–9 seed-twin floor with no
row lean = flat. One training seed per arm, so a lean earns a second-seed
replicate before it becomes a verdict.

## Results — 2026-09-12 (`results/20260912-1644-mb/`, blind s26 / s27): FLAT

Mechanism gauge (X = aak, Y = channel_(caststation), 364 shared modules):

| merge arm | in-overlap (× null) | out-overlap (× null) | ΔW abs cos |
|---|---|---|---|
| SHAREDWSVD (shipped) | 0.999 (35.7×) | 0.092 (7.9×) | 0.092 |
| SHAREDKAI (same random seed) | 0.998 (35.7×) | 0.097 (8.3×) | 0.131 |
| DISTINCT (different seeds) | 0.028 (1.0× = null) | 0.094 (8.0×) | 0.008 |

- `A` does not move at this recipe (lr 2e-5, ~250 steps): two artists seeded
  alike keep **identical** input subspaces (0.999; phase 0's 0.589 was an
  older recipe), and the same `--seed` reproduces the kaiming basis across
  artists, so SHAREDKAI is a true shared-seed control.
- `B` converges to shared output directions regardless of `A` (~8× null in
  every arm); with distinct `A` the task vectors are near-orthogonal
  (|cos| 0.008).

Blind (24 tag-routed rows, one seed per set, graded 2026-09-12):

| set | pair | pairs (tie) | rows by artist (aak / channel) |
|---|---|---|---|
| s26 | DISTINCT vs SHAREDWSVD | 12–9 (3) | 5–6 / 7–3 |
| s27 | DISTINCT vs SHAREDKAI | 10–10 (4) | 4–7 / 6–3 |

Both inside the 15–9 seed-twin floor; the channel-rows lean toward DISTINCT and
the aak-rows lean the other way are each n = 12 and cancel. **Verdict: a fully
co-located input subspace (0.999) is behaviourally inert at a 2-way merge.**
Even though a shared-`A` merge collapses to `(B₁+B₂)·A` — 32 input directions
in total — the render does not suffer, so the input side is not the capacity
bottleneck at this scale. Distinct seeds cost nothing and buy nothing here;
`weight_svd` stays. What this does not cover: N-way merges (N ≥ 4) where the
shared-`A` collapse is N× harsher, and any recipe where `A` actually trains
(higher LR / longer runs) — both would need their own blind set.

**Does `A` ever train?** Yes, given steps: the soup line's uncond inter-train
(`anima_uncond_df58248c_r1_e4`, 12,032 steps at lr 2e-5 over 3,008 images) sits
at in-overlap **0.618** (22× null) against a 250-step weight_svd LoRA, i.e.
~40 % of the seed subspace rotates away — phase 0's 0.589 was such a run. So
the "identical `A`" fact above is a 250-step fact. It does not reopen the seed
lever for the soup path, though: every artist souped off one pool inherits the
same trained `A` and the 4-epoch fine-tunes leave it there, so artists are
again ~1.0 co-located with each other; separating them means a per-artist
uncond run (hours), not a different seed.

## N = 4 (`--variant n4`, `results/20260912-1733-mb_n4/`, blind s28) — FLAT, line closed

All four E0 held-out artists (aak / channel_(caststation) / sweetonedollar /
ootomo_takuji): `SHAREDWSVD4` (four weight_svd runs, one 32-dim input subspace
for all) vs `DISTINCT4` (kaiming seeds 42 / 7 / 11 / 13, 128 input dims).
6 base rows × 4 triggers (`prompts_n4.txt`), one seed, 24 pairs.

| arm | in-overlap (6-pair mean) | out-overlap | ΔW abs cos | global-normalize scale |
|---|---|---|---|---|
| SHAREDWSVD4 | 0.998 (35.7×) | 0.088 (7.5×) | 0.080 | 0.443 |
| DISTINCT4 | 0.028 (null) | 0.088 (7.5×) | 0.006 | 0.489 |

The shared arm's correlated `B`s sum ~10 % larger, so normalize dilutes each
artist ~10 % more — a real but small structural cost of the shared `A`.

Blind: **DISTINCT4 13 – SHAREDWSVD4 10, 1 tie**; per artist 3–3 / 4–2 / 3–2 / 3–3,
sides 11 / 12. Inside the seed-twin floor, no artist lean.

**Verdict (N = 2 and N = 4): the input-subspace co-location that the shared
`weight_svd` seed creates is behaviourally inert at merge.** A 4-artist merge
whose adapters all read the same 32 input directions renders as well as one
with 128. `weight_svd` stays; per-artist seeds are not worth a knob. The
remaining unexplored branch is a recipe where `A` actually trains (12k-step
runs rotate it ~40 %), which the soup path can't exploit anyway (shared uncond
init). Reports: `project/cjk_aware_anima/reports/blind_s2{6,7,8}_MB*.md`.

## Next: N = 8 with `svd_slice` (knob landed 2026-09-12, not yet run)

`down_init="weight_svd"` + `svd_slice = k` (lora.toml key, or `--network_args svd_slice=k`)
seeds artist k from W₀'s right singular vectors `[32k, 32k+32)` — exact SVD,
so slices are exactly orthogonal (the kaiming-distinct arms above were only
null-overlap, 0.028). r=32 allows k = 0…7 (the 256-row adaln `.1` layers).
Planned read: `SHAREDWSVD8` (all k=0) vs `SLICE8` (k = artist index), 4 rows ×
8 triggers, blind set with a reference strip of each artist's real images and
the instruction "which side looks more like this artist", plus a solo-LoRA
PE-cos fidelity gauge. `run.py` does not carry this variant yet.

