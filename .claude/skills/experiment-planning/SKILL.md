---
name: experiment-planning
description: How to plan and record an experiment here — what to queue unasked, its design, gate and preconditions, run names — and how a result is read against its gate. Load before queuing GPU work the user did not spell out, proposing an arm, writing a run config's header, adding an arm to a project plan, or reading a run's result.
---

# Planning an experiment

An experiment is written down before it runs, in one place: the run config's header comment
(`project/<line>/configs/<handle>_mmdd_hhmm.toml`). The line's `plan.md`, reports, README and
memory name the run by its config file name and do not repeat its settings, question or gate. The
header holds the design, the gate and the preconditions, with what is set and measured kept apart
from the reasoning for the arm (`report-writing` skill).

A run is named `<handle>_mmdd_hhmm` (submit time, e.g. `turbo_1112_1512`): the config file's name
and its `output_name`, not passed on the CLI. The handle is one short word for the line or arm
(`turbo`, `swaptext`). Smokes may keep a `smoke_*` name.

## What to queue

- A step the line's plan already holds runs as planned, without asking again. A step over about
  10 minutes enters the plan by the user's word or by a proposal the user accepted.
- GPU work outside the plan (a check that came up, a forward-only probe, a quick measurement) is
  Claude's call up to about 10 minutes. Over that (a micro arm, a run, a retrained control, a
  control render pass) it is proposed once, with its cost, and queued on the user's word; later
  replies do not ask again.
- A request ends at what it names: a build / cache / preprocess request at that artifact, a
  compare request at the judge renders beside the reference run and the runs' own log numbers.
  What would follow goes by the two rules above.
- An existing checkpoint or render is the control unless the user says otherwise. A confound that
  makes it unfair (code drift, a recipe change) is noted beside the result, and a fresh control is
  offered as a follow-up after the treatment.
- When the user says run, or sets the plan: submit exactly the arms they named, back to back, and
  report any open check beside the submitted job. No smoke, seed twin or wait is added between
  their steps. The cheap-first ladder under Design is for arms Claude proposes.
- A throughput / utilisation / sync probe ends within 10 steps. "정적분석만" means read the code and
  the saved logs and queue nothing.
- No CPU-heavy side work while a build or GPU job runs.

## Design

- One question whose answer changes what happens next (which arm runs next, which default ships,
  which line closes), written with what each answer leads to.
- One variable changes from a named parent run; data, cache, seeds, schedule, rank and init stay.
- When two things must change together, the header names what the run cannot separate. A bundled
  win can reverse on a matched rerun (`docs/anima/findings/lora_recipe.md`, s29 → s33).
- Cheap first, for an arm Claude proposes: a forward-only probe or a smoke (minutes), then a micro
  arm (about 25 min; the first step for a mechanism question), then the full run only on a win. A
  win at micro scale needs the longer check before it ships.
- A term's weight is set from a ratio measured on a run that matches the real one in everything
  that scales the term (learning rates, delay, the data); a smoke on other settings sets nothing.
  Where the scale drifts during training, normalise it each step.

## Gate

The gate says what is read, against what, and who decides.

- What is read: the render sheets (which prompts, seeds and regions) and the log or probe numbers
  reported beside them, against the reference run's checkpoint rendered with the same script and
  settings.
- The user decides, on the renders. Numbers are reported as numbers beside the reference. An A/B
  verdict is the user's pick string on blind sheets (a zip sent with SendUserFile, keys left out,
  a `rows.txt` of prompts). Claude's blind picks are a second read, not the verdict; a fresh agent
  gives an unanchored one.
- Easy reads: where the call is natural to judge (does the head generalise to generated images),
  Claude reads it, sends the sheets and numbers, records it as Claude's read (`report-writing`
  skill) and goes on with the next step, kept easy to revert. The user's flag stops or reverts it.
  This covers reads and record updates, not GPU work the queue rules send to the user.
- Text and glyph renders are read per glyph with the sheets viewed, not as totals.
- Where the read is the gate, the sheets carry no arm labels and a shuffled order per prompt
  (`project/finished/cjk_aware_anima/probes/blind_pairs.py`; worked case
  `bench/grad_init/e1_blind.py`). A labelled read is recorded as labelled.
- A sheet holds the two arms the question compares, as two columns or two rows, one (prompt, seed)
  per file (`bench/turbo/blind_sheets.py`). Other probe columns stay off it; a third arm gets its
  own pairwise sheet.
- The prompts and seeds that arms are read on are the line's working set. A ship decision is read
  on a second set that no arm of the line has been read on, named in the plan when the line opens.

The gate is built from what already exists:

- A prior experiment on the same question: its read carries forward (same sheets, seeds, probe).
- A prior implementation or recipe: its gate and eval come with it (the reference pipeline's
  sampler settings, the shipped eval set and its sizes).
- A shipped release is the reference to beat or hold, read on the eval it shipped on.
- A bar is written with its source. The gap between the two arms of an A/B is an effect and sets
  no bar.
  - The floor on record, the default: what two runs differ by with no treatment.
    - Training: flash-attention's backward is not seedable (about 1e-4 a step), and two identical
      1,200-step runs ended at ΔW cosine 0.413. `--deterministic` (`train.py` only, about a third
      slower) makes the rerun identical
      (`docs/anima/findings/paired_dw_chaos_floor_deterministic.md`).
    - A forward: on the MOSS DiT any nonzero change of the input moves the velocity by
      1e-4…5e-4 on bf16 and 2e-4…1e-3 on int8 + fp32 stream, over σ 0.95…0.3
      (`docs/moss/findings/text_stream_page_state.md`).
  - A twin gap: the gap between two runs of one config, a seed twin or a rerun. It is prepared
    only where the read needs a bar stricter than the floor: a ship decision or a change of parent
    that rests on the size of a gap, or a render habit counted over seeds on a 30-image LoRA. It
    is proposed once, with its cost.
  - Neither: the header says "spread unknown", and the read stays on the renders and the numbers
    beside them.
- An arm at the floor or inside the twin gap is recorded as level. It does not become the parent
  of the next arm on that number; the parent stays, or the user's read picks.
- No threshold on a quantity that is small by construction (a one-step x̂₀ gap at low σ, where x_t
  already holds the image).
- Criteria are fixed before the result is read. A gate passes when its evidence does: if every
  metric and the renders favour the arm, it passed, however small the gap.
- A probe column that the user's read has overruled twice on one line stops being a pass / fail
  bar there. It is reported as a number, and the plan's open list takes an item for a probe that
  follows the read.

## Preconditions

A precondition is something not yet known that the gate's read depends on, checked before the run
at the smallest scale that answers it.

- What already follows from a finished experiment or a training record is cited (run id, report
  path), not re-verified, while the record was measured on what the new run uses: the same teacher
  or base, LR range, data and recipe family. Where one differs, the header names the difference
  and the carry-over becomes the precondition.
- The meter discriminates: a new metric or feature distance is read against a null at equal norm
  (an equal-norm random residual, a permutation null, a rounding-noise arm) before it is trusted.
  Worked case: `project/anima_turbo/findings/turbo_gan_dm_grad_orthogonal.md`.
- A probe that patches, swaps or masks activations on the MOSS DiT carries its `own`, `jitter` and
  equal-norm arms from its first run: `library/moss/CLAUDE.md`, Standing rules.
- The noise level is known: the floor on record (Gate), or a twin where the read needs more.
- The probe runs where the real run runs: same LR range, same data distribution
  (`project/anima_turbo/findings/turbo_tau_critic_interference_lr_artifact.md`).
- The signal is large enough to move what is trained: a gradient near the optimizer's floor in a
  smoke is fixed before the long run.
- New data or caches are checked once when built (counts, keys, draws per item).

## Header shape

A run over about 10 minutes takes the full form, in about 15 lines:

```
# <what>: <parent id> with <key> <old> -> <new>, nothing else changed (<recipe in one line>).
# Question: <one question>. <answer A> -> <what follows>. <answer B> -> <what follows>.
# Records it rests on:
#   - <run id or report path>: <number, with its conditions>
# Read: <checkpoint> against <reference checkpoint>, <probe command>, <sheets: prompts, seeds,
#   regions>, blind | labelled; the user's read decides.
# Bars: <number> (<the twin pair it comes from>) | floor <number> (<its record>) | spread unknown
# Cost: <iterations>, <minutes>.
```

A probe within the 10 minutes takes the short form, in about 6 lines:

```
# <what>: <script> on <the rows or checkpoint it reads>; <parent id> with <what differs>.
# Question: <one question>. <answer> -> <what follows>.
# Read: <what is read, against what>; Claude's read | the user's read decides.
# Cost: <minutes>.
```

The header holds what the run cannot be read without. A parent's numbers stay in the parent's
record and are cited by id; how one key's value was drawn (the strings, pages or seeds) is a
comment on that key; what the run found goes in its report.

Worked records: `project/cjk_anima_reseed/reports/row_grad_bias_2026_11_06.md` (a 20-minute probe
with a rounding-noise control), `project/cjk_ko/reports/micro_2026_10_11.md` (three micro arms
from one data build, the shared start checked by the step-1 loss).
