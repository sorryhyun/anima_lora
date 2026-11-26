# Issues — the OCR reader line

Things found that are not fixed and not yet an item of a plan. One entry each: what was
seen, where it was counted, what it touches. An entry leaves this file when a plan takes it
up or it is fixed.

## Dot runs keep three spellings in the train and val targets (2026-11-26)

- Seen from the cjk_rl sentence run, which draws its strings from the main labels
  (`project/cjk_rl/probes/sent_strings.py`).
- `load_mainsub`, metric `main`, the target as `normalize_target` writes it
  (`TARGET_NORM = 10`):

  | split | main labels | a run of two or more `・` | `…` | a run of two or more `.` |
  |---|---|---|---|---|
  | train | 115 529 | 2 343 | 21 818 | 167 |
  | val | 11 634 | 294 | 2 146 | 6 |

- Since `TARGET_NORM = 7` the training target keeps a dot run as labelled; the scoring keys
  fold every run to one `…` (`core/textnorm.py`, fold 6). So the reader is trained to
  write `・・・`, `…` and `...` apart, in the train split as in val.
- The user (11-26) took the train set to be folded already and only the val labels left.
  The loader's train targets are not folded.
- Touches: the reader's training target (a fold of `・{2,}` → `…` would be a new
  `TARGET_NORM`), and whatever reads the targets as strings. cjk_rl folds `・{2,}` → `…` in
  its own draw from the draw after `sent_strings_s1126` (the user's call, 11-26).
