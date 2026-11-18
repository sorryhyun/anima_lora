---
name: report-writing
description: How to write research records here — run reports, probe write-ups, findings, plan entries, a project README's state section, run summaries in chat. Facts and interpretation in separate sections, at most one emphasis per report file. Load before writing or revising any of these.
---

# Writing reports

Applies to `project/*/reports/`, `project/*/findings/`, plan entries and state sections in
`project/*/`, `docs/*/findings/`, and to a summary of a run given in chat. Generated eval tables
(`reports/ocr_eval_*.md` and the like) are tool output: leave them to their generator.

## Facts and interpretation are separate sections

- **Facts:** what ran, settings, data, numbers, tables, probe and eval outputs, recorded verdicts,
  shipped artifacts, and the user's own reads and decisions, attributed to the user ("The user read
  1403's composition as better than 1020's").
- **Interpretation:** what the numbers suggest, mechanisms, hypotheses, conclusions, the reasoning
  for the next step.

Shape: a one-line header saying what the report covers, then `## Facts`, then
`## Interpretation`. A report covering several runs or probes has one pair per run section. A plan
entry keeps the plan's order and splits within each item.

A number or record line carries no interpretive clause:

```
wrong:  anchor MSE 0.131 against 0.070: the GAN pulled the student off the target.
right:  Facts           anchor MSE 0.131 against 0.070.
        Interpretation  The GAN term pulled the student away from the anchor target.
```

Recorded verdicts and the user's reads are part of the record. When restructuring an existing
report, keep them word for word in Facts; do not reword, soften or re-judge them.

Claude's own read of renders goes in Facts under its own label ("Claude's read"), with whether the
arm labels were visible when it was made. It is an observation to check, not a verdict: no plan
entry or later report gives it as the reason an arm passed or a line went on.

Where a probe verdict and the user's read differ, both are written on adjacent lines in Facts
("Probe: not passed. The user's read: composition better than 1020's, the line goes on."), and
the report's first line says they differ.

## What a number carries

- The run id and checkpoint (EMA step), the script or log tag it came from, and n: pairs, prompts,
  seeds.
- Its conditions when they differ from the report's stated setup (teacher, LR, data).
- For a comparison, the spread it is read against and that spread's source, or "spread unknown".
  A gap is called a gain, a loss or a tie only against a spread.
- Where the output lives. `output/`, `results/`, `_archive/` and paths outside the repo are local:
  mark them "(local)".

Proposals (`idea.md`, `docs/*/proposal/`) argue by nature; split out only the measured evidence they
cite.

## Emphasis

At most one bold or other emphasis per report file, on the one thing a reader must not miss. No
emphasis for the writer's own stress.

## Index files

`README.md` maps, `reports/README.md`, `configs/README.md`, `findings/README.md` and run tables stay
lists and tables: each entry says what the file or run is, plus a number or a recorded verdict where
one exists. Readings go in a short note under the table, not in its cells.

A verdict cell uses one of: baseline, passed, not passed, level, adopted on the user's read,
shipped, stopped at <step>, failed (<what was observed>). A cell that explains ("not the cause",
"GAN is not the walk") is a reading and goes in the note.
