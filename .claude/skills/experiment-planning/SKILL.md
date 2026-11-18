---
name: experiment-planning
description: How to plan and record an experiment here — its design, gate and preconditions — before a GPU run is queued. Load before proposing an arm, writing a run config's header, or adding an arm to a project plan.
---

# Planning an experiment

An experiment is written down before it runs, in two places: the run config's header comment
(`project/<line>/configs/<handle>_mmdd_hhmm.toml`) and the arm's entry in the line's `plan.md`. The
record has three parts: design, gate, preconditions. Write it with the `report-writing` skill's
split: what is set and measured as facts, the reasoning for the arm as interpretation.

## Design

- One question whose answer changes what happens next: which arm runs next, which default ships,
  which line closes. Write the question, and what each answer would lead to, before the run.
- Name the parent run. The arm changes one variable from it; the header says "nothing else changed"
  and lists the variable's old and new value. Same data, cache, seeds, schedule, rank, init.
  Example: `turbo_1117_1020` / `1021` (student LR A/B, control named;
  `_archive/anima_turbo/reports/step0_depth.md` (local)).
- When two things must change together, say so in the header and name what the run cannot
  separate. Bundled arms read as wins that a matched rerun reverses
  (`docs/anima/findings/lora_recipe.md`, s29 → s33).
- Cheap first: a forward-only probe or a smoke (minutes), then a micro arm (about 25 min, or 250
  DM iterations), then the full run only on a win. A win at micro scale needs the longer check
  before it ships.
- A term's weight is set from a ratio measured on a run that matches the real one in everything
  that scales the term (learning rates, the disc's LR, delay, the data). A smoke on different
  settings sets nothing: `turbo_1118_0953` took λ from a smoke whose disc trained at 1e-5 and ran
  with 5e-5. Where the scale drifts during training, normalise it each step instead
  (`gan.paired_ratio`, since removed from the loop; `turbo_1118_1028`).
- State the cost (minutes, it/s) in the header. Hour-scale work is proposed to the user once and
  queued on their word.

## Gate

The gate says what is read, against what, and who decides.

- What is read: the render sheets (which prompts, which seeds, which regions — "guitarist seeds 1,
  2 and 4, the group lower halves"), and the log or probe numbers that will be reported beside
  them.
- Against what: the reference run and checkpoint (`1442` EMA 2048), rendered with the same script
  and settings.
- Who decides: the user, on the renders. Numbers are reported as numbers beside the reference.
- How the renders are read: where the read is the gate, the sheets are composed without arm labels
  and in a shuffled order per prompt (`project/finished/cjk_aware_anima/probes/blind_pairs.py`;
  `bench/grad_init/e1_blind.py` is a worked case). A labelled read is recorded as labelled.
- The prompts and seeds that arms are read on are the line's working set. A ship decision is read
  on a second set that no arm of the line has been read on; name it in the plan when the line
  opens.

Propose the gate from what already exists; do not invent one.

- A prior experiment on the same question: carry its read forward (same sheets, same seeds, same
  probe).
- A bar comes from a spread, and a spread is the gap between two runs of one config: a seed twin,
  or a rerun where the run is not deterministic (the control `turbo_1113_1105` and its seed-43
  twin `turbo_1114_1745`, `_archive/anima_turbo/reports/grid_cdm.md` (local)). The gap between the two arms
  of an A/B is an effect and sets no bar.
- An arm inside the spread is recorded as level. It does not become the parent of the next arm
  on that number; the parent stays, or the user's read picks.
- A shipped release: it is the reference to beat or hold (Hub v3 / `v4_preview` for turbo, Hub v6
  for the OCR reader), read on the eval it shipped on.
- A prior implementation or recipe: its gate and eval come along with it (the reference pipeline's
  sampler settings, the shipped eval set and its sizes).
- Write each bar with its source ("<number>: the <id> / <id> twin gap"). A bar with no source is not
  proposed: the header says "spread unknown", and the read stays on the renders and numbers
  beside them.
- No threshold on a quantity that is small by construction.
- A probe column that the user's read has overruled twice on one line (the probe says not passed,
  the read says go on, or the reverse) stops being a pass / fail bar on that line. It is reported
  as a number, and the plan's open list takes an item for a probe that follows the read.

## Preconditions

A precondition is something not yet known that the gate's read depends on. Check it before the
run, at the smallest scale that answers it.

- Do not re-verify, or list as a precondition, anything that already follows from an observed or
  finished experiment or a training record. Cite the record (run id, report path) and build on it.
  This holds while the record was measured on what the new run uses: the same teacher or base, LR
  range, data and recipe family. Where one differs, name the difference in the header; the
  carry-over is then the precondition, checked at the smallest scale that answers it.
- The meter discriminates: before a new metric or feature distance is trusted, read it against a
  null at equal norm (an equal-norm random residual, a permutation null, a rounding-noise arm).
  Examples: `probe_fm_floor` in `_archive/anima_turbo/reports/step0_depth.md` (local); the permutation null
  in `project/anima_turbo/findings/turbo_gan_dm_grad_orthogonal.md`.
- The noise level is known: a spread between seeds or a twin run, if no record gives one. Paired
  ΔW cosines read without a twin control were vacuous
  (`docs/anima/findings/paired_dw_chaos_floor_deterministic.md`).
- The probe runs where the real run runs: same LR range, same data distribution. A gate read at the
  tail LR flipped at the peak LR (`project/anima_turbo/findings/turbo_tau_critic_interference_lr_artifact.md`).
- The signal is large enough to move what is trained: when a smoke or P0 already shows the gradient
  near the optimizer's floor, fix that before the long run (`project/cjk_rl/reports/p1_2026_11_13.md`).
- New data or caches: their build is checked once when built (counts, keys, draws per item).

## Header shape

Finished configs move to the gitignored `_archive/`, so the shape is kept here:

```
# <what>: <parent id> with <key> <old> -> <new>, nothing else changed (<recipe in one line>).
# Question: <one question>. <answer A> -> <what follows>. <answer B> -> <what follows>.
# Records it rests on:
#   - <run id or report path>: <number, with its conditions>
# Read: <checkpoint> against <reference checkpoint>, <probe command>, <sheets: prompts, seeds,
#   regions>, blind | labelled; the user's read decides.
# Bars: <number> (<the twin pair it comes from>) | spread unknown
# Cost: <iterations>, <minutes>.
```

## Templates in the records

- `project/cjk_anima_reseed/reports/row_grad_bias_2026_11_06.md`: a 20-minute probe with a
  rounding-noise control that replaced two training runs.
- `project/cjk_ko/reports/micro_2026_10_11.md`: three micro arms from one data build, the shared
  start checked by the step-1 loss.
- `project/finished/ocr_reader/roadmap.md`: the eval set sized for the effect before training.
