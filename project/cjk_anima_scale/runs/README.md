# runs — the ledger

`ledger.jsonl`: one line per `scale.py <run> <verb> --submit`, written at
submit time — timestamp, job id, label, run, verb, the child argv, and the
`ANIMA_VOCAB_PACK` the submit shell named. Rows before 2026-09-25 carry
`stage` / `tag` / `steps` instead of `run` / `verb`. `scale.py ledger`
prints both.

Reads go in the run's dir under `output/cjk_anima_scale/<run>/`
(`data/build.json`, `train_log.json`, `train_record.json`, `reads.json`,
`sheet.png`, the arms' `eval_reads.json` / `native*/` / `target/`) and, once
digested, in `../reports/` like every other read.
