---
name: experiment-planning
description: How to plan and record an experiment here — what to queue unasked, its design, gate and preconditions, run names — and how a result is read against its gate. Load before queuing GPU work the user did not spell out, proposing an arm, writing a run config's header, adding an arm to a project plan, or reading a run's result.
---

# Planning an experiment

An experiment is written down before it runs, in one place: the run config's header comment
(`project/<line>/configs/<handle>_mmdd_hhmm.toml`). The line's `plan.md`, its reports, its README
and memory name the run by its config file name and do not repeat its settings, question or gate.
The record has three parts: design, gate, preconditions. Write it with the `report-writing` skill's
split: what is set and measured as facts, the reasoning for the arm as interpretation.

A run is named `<handle>_mmdd_hhmm` (submit time, e.g. `turbo_1112_1512`), written as `output_name`
in its config file, which takes the same name; it is not passed on the CLI. The handle is one short
word naming the line or arm (`turbo`, `swaptext`, `selfdistill`). Smokes may keep a `smoke_*` name.

## What to queue

- Unasked GPU work up to about 10 minutes (a forward-only probe, a quick measurement) is Claude's
  call. Hour-scale work (a run, a retrained control, a control render pass) is proposed once, with
  its cost, and queued on the user's word; later replies do not ask again.
- A request ends at what it names. A build / cache / preprocess request ends at that artifact; a
  compare request gets the judge renders beside the reference run and the runs' own log numbers.
  The run, probe or bench that would follow is proposed once, not queued.
- An existing checkpoint or render is the control unless the user says otherwise. A confound that
  makes it unfair (code drift, a recipe change) is noted beside the result, and the control is
  offered as a follow-up after the treatment.
- When the user says run, or sets the plan: submit exactly the arms they named, back to back, and
  report any open check beside the submitted job. No smoke, seed twin or wait between their steps
  is added; where a twin would set the bar, the header says "spread unknown". The cheap-first ladder
  under Design is for arms Claude proposes.
- A throughput / utilisation / sync probe ends within 10 steps. "정적분석만" means read the code and
  the saved logs and queue nothing.
- No CPU-heavy side work while a build or GPU job runs.

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
- Cheap first, for an arm Claude proposes: a forward-only probe or a smoke (minutes), then a micro
  arm (about 25 min, or 250 DM iterations; the first step for a mechanism question), then the full
  run only on a win. A win at micro scale needs the longer check before it ships.
- A term's weight is set from a ratio measured on a run that matches the real one in everything
  that scales the term (learning rates, the disc's LR, delay, the data). A smoke on different
  settings sets nothing: `turbo_1118_0953` took λ from a smoke whose disc trained at 1e-5 and ran
  with 5e-5. Where the scale drifts during training, normalise it each step instead
  (`gan.paired_ratio`, since removed from the loop; `turbo_1118_1028`).
- State the cost (minutes, it/s) in the header.

## Gate

The gate says what is read, against what, and who decides.

- What is read: the render sheets (which prompts, which seeds, which regions — "guitarist seeds 1,
  2 and 4, the group lower halves"), and the log or probe numbers that will be reported beside
  them.
- Against what: the reference run and checkpoint (`1442` EMA 2048), rendered with the same script
  and settings.
- Who decides: the user, on the renders. Numbers are reported as numbers beside the reference.
  An A/B verdict is the user's pick string on blind sheets (a zip sent with SendUserFile, keys left
  out, a `rows.txt` of prompts). Claude's blind picks are a second read, not the verdict: on
  caststation `clause_hw` they went 15:5 where the user's went 11:11. A fresh agent gives an
  unanchored second read. On a 30-image LoRA, a render habit counted over seeds needs a
  same-config retrain before it is called an effect.
- Easy reads: where the call is natural to judge (does the head generalise to generated images),
  Claude reads it, sends the sheets and numbers, records it as Claude's read (`report-writing`
  skill) and goes on with the next step, kept easy to revert. The user's flag stops or reverts it.
  This covers reads and record updates, not hour-scale GPU work.
- Text and glyph renders are read per glyph with the sheets viewed, not as totals.
- How the renders are read: where the read is the gate, the sheets are composed without arm labels
  and in a shuffled order per prompt (`project/finished/cjk_aware_anima/probes/blind_pairs.py`;
  `bench/grad_init/e1_blind.py` is a worked case). A labelled read is recorded as labelled.
- Sheet layout, blind or labelled: the two arms the question compares, as two columns or two rows,
  one (prompt, seed) per file (`bench/turbo/blind_sheets.py`). A column the question does not read
  stays off the sheet: when the question is T against S, the sheet holds T and S only, not `S>T`,
  `A>S` or the other probe columns beside them (a 4 × 3 grid of probe columns leaves the reader
  guessing which pair to compare). A third arm gets its own pairwise sheet.
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
- No threshold on a quantity that is small by construction (a one-step x̂₀ gap at low σ, where x_t
  already holds the image).
- Criteria are fixed before the result is read. A gate passes when its evidence does: if every
  metric and the renders favour the arm, it passed, however small the gap.
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
- On the MOSS DiT, a probe that patches, swaps or masks activations has an `own` arm (must read 0;
  store the fed rows at the stream's precision, since bf16 storage on an fp32 stream reads the
  floor), a `jitter` arm and an equal-norm isotropic arm from its first run, and arms are compared
  against those, not against 0. Any nonzero change re-draws the rounding inside the blocks: a 1e-3
  relative jitter of the text rows moves the velocity by 1e-4…2e-4 (σ 0.95) to 5e-4…1e-3 (σ 0.3),
  as much as a change 10–30 times larger, on bf16 and on int8 + fp32 stream alike
  (`docs/moss/findings/text_stream_page_state.md`). Probes run on generation's DiT (int8 + fp32
  stream).
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
