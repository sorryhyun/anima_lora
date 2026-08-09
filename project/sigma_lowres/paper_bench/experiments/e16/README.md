# E16 — placement vs dilution: demotion scheduling as a trajectory-propagator probe

| | |
|---|---|
| **Status** | **DONE 2026-08-03** — 16.0 verdict **AMPLIFICATION** (ΔW cos vs native: late **0.906** ≫ win768 0.395 > spread 0.281 > early 0.193; placement dominates the σ-window choice for off-map bias). 16.1 (3 seeds × 2 artists, 5 arms): **combo** (768 if σ∈(0.65,0.95), elif σ>0.5→896+yarnsig, via the new stacked router `--sigma_lowres_route2`) is the new throughput frontier — **−18.3% wall, inside the seed lottery on BOTH corpora** (0.9576/0.9580 vs yardsticks 0.9547/0.9541), beating shipped sigma896 (−14.7%, hews boundary tie reproduced). **win768late** = max-margin arm (0.9678/0.9728 at −6%). sigma896late ≈ sigma896 ⇒ scheduling unnecessary on certified routes; the 16.0 amplification result governs off-map bias. Full record: [`launch_20260802_160.md`](launch_20260802_160.md) (trainer deltas: `--sigma_lowres_span`, `--sigma_lowres_threshold_max`, stacked router — all pinned in `tests/test_sigma_lowres.py`). Renumbered from **E15** 2026-08-01. PROPOSED 2026-07-31 |
| **Question** | Is a protected span ("don't demote the first/last N% of steps") just η-weighted dilution of the demotion effect, or does *placement* matter? Equivalently: which regime does the training trajectory live in — washout, linear, or amplification? |
| **Depends on** | [E4](../e4/) (harness, the below-yardstick `sigma768` arm = the rescue target, seed-keyed σ stream, `claim_accumulated_bias.md`), [E15](../../../record/e15/) (the currency argument: schedules spend neither FLOPs nor variance), `--deterministic` ΔW twins (no chaos floor) |
| **Instrument** | trainer: a step-span gate on top of the σ-gate (few lines in `_maybe_sigma_demote`); readout: `bench/compare_ckpt_dw.py` (16.0), E4 yardstick/render harness (16.1) |
| **In the paper** | if placement matters: a scheduling corollary to §5; if not: one sentence closing the placement DOF that `claim_accumulated_bias.md` leaves open. *(Outcome: placement matters — the scheduling-corollary branch fired.)* |

## The theory (why this is one experiment, not a hyperparameter sweep)

To first order, the final-weight deviation from a clean run is

> Δθ_T ≈ Σ_t M_{t→T} · η_t · b_t,  M_{t→T} = Π_{s>t} (I − η_s H_s)

— each step's demotion bias b_t is propagated through the remainder of
training. The propagator's effective spectrum on the bias direction
picks one of three regimes, each with a different shipping rule:

| regime | M_{t→T} on b | early-placed bias | rule it licenses |
|---|---|---|---|
| **washout** (contraction, ‖M‖<1) | decays | forgiven | *demote early, finish clean* — progressive-resizing folklore (FixRes etc.) becomes a measured statement |
| **linear** (‖M‖≈1) | preserved | = late | placement is inert; only η-weighted demoted mass matters (the "9/10" intuition, exactly) |
| **amplification** (subspace selection, from-zero LoRA) | grows | worst | *protect the first epoch*; late demotion is the cheap region |

E15 killed the variance route to buying off-map routes; a schedule is
the zero-cost route — same FLOPs, same variance, it only moves biased
steps to where the propagator forgives them. Whether such a place
exists is exactly the accumulated-composition question
`claim_accumulated_bias.md` declares uncertified. (Answered 2026-08-03:
AMPLIFICATION — the forgiving place is *late*; 16.1's combo router
lands inside the seed lottery on both corpora.)

## 16.0 — the ΔW ordering probe (~1 h, decides everything)

Four deterministic twins at E4 tenth scale (480 steps, bs 1, hews,
stock recipe, `--deterministic --paired_step_rng`): **native** +
three 768-route arms with the *same* demoted-step budget on the *same*
seed-keyed eligible set (σ>0.5 steps, identical across arms per E4's
CRN property), differing only in placement:

- **early** — demote eligible steps in the first half only (~0.25·T);
- **late** — second half only (~0.25·T);
- **spread** — every other eligible step (seed-keyed p=0.5 coin, ~0.25·T).

Readout: cos(ΔW_arm, ΔW_native), global + depth profile
(`compare_ckpt_dw.py`). Identical mass ⇒ any ordering is pure
placement signal; `--deterministic` ⇒ no chaos floor.

**Pre-registered mapping** (frozen before the run): washout ⇒
early > spread > late (early's bias decayed the longest); linear ⇒
all three within the twin floor; amplification ⇒ early is worst.
**Decision rule**: an ordering beyond the twin resolution → 16.1 on
the favored schedule; all-within-floor → verdict "linear-dilution
regime — placement inert, only η-weighted mass matters", line closes
at zero further cost (a clean, useful closure). Discordant with all
three rows (e.g. spread strictly worst) ⇒ the σ-coupling of b_t is
doing work the scalar story misses — record, don't over-interpret.

## 16.1 — deployable-rule A/B (conditional, E4 protocol; ran on the amplification winner)

Only the regime winner, at full practical mass. E.g. if washout wins:
**hybrid768** = 768 on eligible steps for the first (1−f)·T, 896 (or
native) for the last f·T, f ∈ {0.1, 0.2} — targeting ~−21…−24% wall at
896-level fidelity. Arms: native / sigma896 (shipped) / sigma768
(uniform, already measured) / hybrid768 × 3 seeds × 2 artists, E4
yardstick + FLOPs + renders. **Gate**: hybrid768 at-or-inside the
yardstick on both corpora (the bar uniform sigma768 failed) at
net ≤ −20% wall. Kill: hybrid ≈ uniform ⇒ the 16.0 ordering does not
survive full-recipe scale; close with the 16.0 result as the finding.

Outcome (2026-08-03): amplification won, so the arms were late-window
variants rather than the hybrid768 example above — winner **combo**
(−18.3 % wall, inside the seed lottery on both corpora; win768late =
max-margin arm at −6 %). Full record: `launch_20260802_160.md`.

## Groundings

- `sigma768` uniform: gross −26.5%, below the yardstick on ≥1 corpus —
  the measured rescue target ([E4](../e4/) 5-arm addendum).
- Seed-keyed σ stream ⇒ the eligible set is arm-invariant (E4 in-vivo
  CRN check: identical 244/480 demote set) — placement arms partition
  the *same* steps.
- Deterministic paired ΔW has no chaos floor (`methods.md`,
  `--deterministic`; nondeterministic floor 0.413).
- Placement is precisely the DOF E4's certification left open
  (`claim_accumulated_bias.md`).
- Currency argument: E15 15.0 measured why variance is the expensive
  currency; schedules spend none of it
  (`../../../record/e15/`, `runs/20260731-2114-e14-pricing/`).
- Prior art being upgraded, not invented: progressive resizing /
  low-res pretrain + high-res finish (FixRes lineage) is the washout
  bet made by folklore.

## Cost

16.0: 4 × 480-step deterministic runs (~1.5× slower) ≈ **~1 h GPU**
total, daemon-queued. 16.1: E4-scale, ~2–3 h + evals. The realized
trainer deltas grew beyond the planned step-span gate + seed-keyed
coin: `--sigma_lowres_span`, `--sigma_lowres_threshold_max`, and the
stacked router `--sigma_lowres_route2` — all pinned in
`tests/test_sigma_lowres.py`.
