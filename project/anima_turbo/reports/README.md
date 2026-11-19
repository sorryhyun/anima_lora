# Reports

Reports of the line since its restart on 2026-11-18. Earlier ones are in
`_archive/anima_turbo/reports/` (local) and carry no verdict into this line.

| Report | Covers | Verdict |
|---|---|---|
| [step0_k7of28.md](step0_k7of28.md) | v1.1 untrained `T>S@1` read; `turbo_1118_1818` (step 0 alone on 7 of 28, 1000 pairs) and its step 0 on the untouched v1.1 (`P>S@1`); v1.1 `T>S@1` against its own S; teacher T against v1.1 S on `judge_hard` and on three random dataset captions | v1.1 handoff: adopted on the user's read (8 / 4 near tie); `1818`: not passed (S 0 / 12, `T>S@1` 2 / 10, `P>S@1` 2 / 10); v1.1 `T>S@1` against its S: level (6 / 6); T against v1.1 S: level (5 / 7) on `judge_hard`, T 3 / S 9 on the random captions |
