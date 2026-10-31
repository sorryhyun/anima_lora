# Anima `wq_t` streaming and budget 1.0 (2026-10-31)

**Verdict: `--int8 stream` takes 1.42 GB off the Anima plain-LoRA peak for 17 ms/step, and
the loss matches `full`. Budget 1.0 does not fit 16 GB on this set, and where it fits it buys
no speed: 333 ms/step at both 1.0 and 0.99, 12.90 against 8.84 GB. The budget stays 0.99.**

`library/anima/int8.py` now takes `--int8 stream`, on the same `WqtStream` as MOSS.
`stream_compacted_` (the move off and back after `wq_t` leaves the card) went from
`library/moss/dit_int8/base.py` to `library/int8/linear.py`, so both lines share it.

## Setup

`ama mitsuki`, method `lora`, preset `default`, seed 1408198296, GPTQ forward copies, fp32
kernel input, bf16 stream, compiled. All arms ran `src/train_blockmem.py`: `train.py` plus
`memory_allocated` after each block of step 1 and time between step starts, with the first ten
intervals dropped. One run per arm.

    make daemon-run ARGS="--queue --stall-timeout 600 project/int8_train/src/train_blockmem.py \
        --method lora --preset default --artist_filter 'ama mitsuki' --seed 1408198296 \
        --max_train_epochs 1 --int8 stream --activation_memory_budget 0.99"

`ANIMA_BM_SMALL=1` keeps the 13 samples of 2990–3080 tokens, run for 4 epochs (41 intervals).
The rest of the set sits at 4032–4200 tokens, so nothing lies between the two groups.

## Results

| arm | samples | job | median | mean | peak |
|---|---|---|---|---|---|
| `stream`, budget 1.0 | 13 small | `20261031-194358-68d8f3` | 333.1 ms | 335.0 ms | 12.90 GB |
| `stream`, budget 0.99 | 13 small | `20261031-194358-df6a87` | 333.2 ms | 334.9 ms | 8.84 GB |
| `stream`, budget 0.99 | all 106 | `20261031-194358-4855b7` | 498.9 ms | 480.4 ms | 10.96 GB |
| `full`, budget 0.99 | all 106 | `20261031-194358-b4e149` | 481.9 ms | 463.5 ms | 12.38 GB |
| `stream`, budget 1.0 | all 106 | `20261031-193929-63ff07`, `20261031-194104-d9fdfc` | OOM at step 0 | – | 14.07 GB allocated |

The peak is `max_memory_allocated`.

## Budget 1.0

Step 1 memory, GB:

| budget | tokens | after block 0 | per block | after block 27 |
|---|---|---|---|---|
| 1.0 | 2990 | 3.08 | 0.332 | 12.03 |
| 0.99 | 2990 | 2.94 | 0.190 | 8.05 |
| 1.0 | 4081 | 3.20 | 0.449 | OOM past block 20 |
| 0.99 | 4081 | 3.01 | 0.257 | 9.95 |

- The slope is linear in tokens: 0.110 GB per block per 1000 tokens at 1.0, 0.063 at 0.99. So
  1.0 saves 1.75x as much, and the gap to 0.99 is 1.3 GB per 1000 tokens over 28 blocks.
- 1.0 is a different partition from 0.99, not a slightly larger cap.
  `choose_saved_values_set` returns the default min-cut at exactly 1 and takes the more
  aggressive cut below it.
- Extrapolated, not run: 4200 tokens at 1.0 ends the forward near 15.6 GB and peaks near
  16.5 GB. The largest sample that fits 16 GB is about 3400 tokens, below everything in the
  set's 4032–4200 group. These figures already include `stream`'s saving.
- The two speed arms match to 0.1 ms, on one run each. What each saved set holds and how many
  attention calls each step makes were not profiled, so why they tie is not traced.

## `stream` against `full`

- Peak 12.38 → 10.96 GB. The same 1.42 GB shows at block 0 (4.43 → 3.01 GB), so it is the
  resident `wq_t` less the two slots, as on MOSS.
- 17 ms/step (3.6%) on the median and on the mean. MOSS measured at most 0.03 s.
- One cold block per step: 106 in 106 steps. "Waited" reads 20.8 s, or 196 ms per step. Against
  a 17 ms step cost it counts time blocked on the GPU's queue, as `moss_wqt_stream.md` found.
- Loss, `stream` against `full`: first logged `loss/current` 0.08497 / 0.08496, epoch average
  0.07342 / 0.07351.
- `stream` adds about 1.5 GB of pinned host memory (`wq_t`), not measured here.

The 482 ms median is the 4100-token samples. The mean of 463 ms sits above the 424 ms/step of
the 848-step `full` run in `docs/anima/optimizations/int8.md`. That run covered 8 epochs and
these cover the first one; the gap was not traced.

## Not done

- Turbo. `scripts/distill_turbo` takes `--int8 off|fwd|full|stream`, converting through
  `TurboDMDNetwork`'s `before_apply`, stamping `ss_int8_base` and isolating its compile cache.
  Only config parsing and the hook's order are unit-tested. No distill step has run on it. It
  refuses `dmad` and `dmad_probe`, and its teacher would run on the int8 base too.
- A complete LoRA on `stream`, and samples from one.
- Budgets below 0.99 on `stream`. The peak of 10.96 GB already leaves about 3 GB on 16 GB.
