# Step 0 on the 7 of 28 anchor: the v1.1 handoff read and `turbo_1118_1818` (2026-11-18)

Every probe number favours `1818` over the v1.1 delta untrained; the user's blind reads favour the
untrained delta (S 0 / 12, `T>S@1` 2 / 10).

Setup throughout: merged15 teacher (CFG 4, 28 steps, shift 3, int8 `stream`, fp32 stream), the
static 4-step grid at shift 3 (σ 1 → 0.9 → 0.75 → 0.5 → 0), judge set `bench/turbo/judge_hard.txt`
(group, café, guitarist × seeds 1–4, 12 cells). Sheets are blind, two columns, one row per file;
picks are the user's preference.

## 1. Can the v1.1 delta finish from the teacher's σ 0.9 state?

### Facts

- Arm: `T>S@1` of `bench.turbo.probe_handoff`, the v1.1 delta (rank 96 ASVD, adaln) untrained,
  handed the teacher's own state at σ 0.9 (7 of 28). Run 2026-11-16:
  `bench/turbo/results/20261116-1550-handoff_m15_v11_untrained/` (local).
- Cos to the teacher's image (lowpass 8) 0.932 (group 0.914, café 0.946, guitarist 0.936);
  sameness 0.301 against the teacher's 0.314.
- Claude's read (guitarist grid only, arm labels visible): hard to tell from the teacher's on all
  four seeds.
- The user's read (blind, `T` against `T>S@1`, `output/blind/turbo/v11_untrained_T_vs_TtoS/`,
  local): picks 001010011000, `T` preferred 8 / `T>S@1` 4 (group 1 / 3, café 4 / 0,
  guitarist 3 / 1). The user: "선호긴 한데 꽤 tie이긴 해서 그대로 가도 될듯". Answered yes.

### Interpretation

- `T>S@1` uses the teacher's real σ 0.9 state and no training of ours, so the anchor mismatch that
  voided the earlier students does not touch it. On the yes, step 0 on 7 of 28 was taken as the
  next job.

## 2. `turbo_1118_1818`: step 0 alone on 7 of 28

### Facts

- Cache `post_image_dataset/turbo_anchor_cache/m15_k7of28_p500` (local): 500 prompts of
  `post_image_dataset/lora` drawn at seed 42, 2 ε draws each, 1000 pairs, `--k_anchor 7
  --teacher_anchor_steps 28`. Built in 23 min (1.51 s per pair at the end), job
  `20261118-181541-5ea263`.
- Run `turbo_1118_1818`: `dpdmd.anchor_only`, v1.1 delta init truncated to rank 64 / alpha 180
  (adaln 16 / 90), `student_lr` 2e-5, `grad_accum` 2, 1000 iterations (500 student updates, one
  pass over the pairs), EMA window 0.2. 6:43 at 2.59 it/s. Config:
  `_archive/configs/anima_turbo/turbo_1118_1818.toml` (local); checkpoint
  `output/ckpt/turbo_1118_1818_ema.safetensors` (local).
- Probe `step0_1818_vs_v11` (`--anchor_k 7 --anchor_steps 28`), `1818` EMA against the v1.1 delta
  untrained, `bench/turbo/results/20261118-1850-step0_1818_vs_v11/` (local). Spread unknown (no
  seed twin on this chord).

  | | `1818` EMA | v1.1 untrained |
  |---|---|---|
  | anchor MSE (7 of 28) | 0.038 | 0.071 |
  | S cos (group / café / guitarist) | 0.767 (0.783 / 0.799 / 0.720) | 0.672 (0.708 / 0.707 / 0.600) |
  | S sameness (teacher 0.314) | 0.343 | 0.397 |
  | `T>S@1` cos | 0.940 | 0.932 |
  | `T>S@1` sameness | 0.294 | 0.302 |

- The user's read (blind, S against S, `output/blind/turbo/step0_1818_vs_v11_S/`, local): picks
  000001110000, the v1.1 delta untrained preferred on 12 of 12 rows. Not passed.
- Claude's read (café seed 1 only, labels known): `1818`'s frame is busier, with more garbled text
  blocks (a sign and a counter panel) and harder contrast; the untrained frame is cleaner.
- The user's read (blind, `T>S@1` against `T>S@1`, `output/blind/turbo/step0_1818_vs_v11_TtoS/`,
  local): picks 100000110110, the v1.1 delta untrained preferred 10 / 2 (group 4 / 0, café 3 / 1,
  guitarist 3 / 1).

### Interpretation

- The probe numbers did not predict the user's read on this run: every column moved toward the
  teacher while the renders lost.
- The 4 steps share one LoRA, so fitting step 0 alone moved steps 1–3 as well; from the teacher's
  own σ 0.9 state they lose 2 / 10. The S read (0 / 12) carries that loss and does not say whether
  the fitted step 0 is itself better. The `P>S@1` probe (`1818` draws step 0, the untouched v1.1
  delta finishes) separates the two (section 3).

## 3. `1818`'s step 0 on the untouched v1.1 delta (`P>S@1`)

### Facts

- Probe `step0lora_1818_on_v11`: `--step0_lora` `1818` EMA on the v1.1 delta untrained, so `1818`
  draws step 0 and the untouched delta draws steps 1–3.
  `bench/turbo/results/20261118-1856-step0lora_1818_on_v11/` (local). Spread unknown.

  | | `P>S@1` | v1.1 S |
  |---|---|---|
  | cos (group / café / guitarist) | 0.753 (0.758 / 0.795 / 0.705) | 0.672 (0.708 / 0.707 / 0.600) |
  | sameness (teacher 0.314) | 0.355 | 0.397 |
  | step-0 anchor MSE (7 of 28) | 0.038 | 0.071 |

- The user's read (blind, `P>S@1` against v1.1 S, `output/blind/turbo/step0lora_1818_P_vs_v11_S/`,
  local): picks 001010001110, v1.1 S preferred 10 / 2 (group 3 / 1, café 4 / 0, guitarist 3 / 1).
  Not passed.

### Interpretation

- With steps 1–3 untouched, `1818`'s step 0 still loses: the fitted step 0 is worse in itself, not
  only through the shared LoRA.
- Whether moving step 0 toward the teacher can pay at all depends on the teacher's real σ 0.9
  state finishing better than v1.1's own S (section 4).

## 4. The v1.1 delta untrained: `T>S@1` against its own S

### Facts

- Renders from `bench/turbo/results/20261118-1850-step0_1818_vs_v11/` (local), no new run.
- The user's read (blind, `output/blind/turbo/v11_TtoS_vs_S/`, local): picks 100001001111,
  `T>S@1` 6 / S 6 (group 2 / 2, café 2 / 2, guitarist 2 / 2). Level.

### Interpretation

- Handed the teacher's real σ 0.9 state, v1.1's steps 1–3 finish no better than from v1.1's own
  step 0, so pulling step 0 toward the teacher does not pay with them; the 7 of 28 target is not
  the step-0 objective for v1.1's steps 1–3. Teacher T against v1.1 S has not been read.
