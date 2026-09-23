# runs — the ledger

`ledger.jsonl`: one line per `scale.py --submit`, written at submit time —
timestamp, job id, label, stage, tag, steps, the child argv, and the
`ANIMA_VOCAB_PACK` the submit shell named. `scale.py ledger` prints it.

Reads go in the stage's own dirs under `output/cjk_anima_scale/` (`build.json`,
`train_log.json`, `train_record.json`, `eval_reads.json`, `regress.json`,
`native/`, `cf_sense_ja*/`) and, once digested, in
`../../cjk_renderable_anima/reports/` like every other read.
