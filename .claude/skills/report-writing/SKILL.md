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

Proposals (`idea.md`, `docs/*/proposal/`) argue by nature; split out only the measured evidence they
cite.

## Emphasis

At most one bold or other emphasis per report file, on the one thing a reader must not miss. No
emphasis for the writer's own stress.

## Index files

`README.md` maps, `reports/README.md`, `configs/README.md`, `findings/README.md` and run tables stay
lists and tables: each entry says what the file or run is, plus a number or a recorded verdict where
one exists. Readings go in a short note under the table, not in its cells.
