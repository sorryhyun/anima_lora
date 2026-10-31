# MOSS-Image `wq_t` streaming (2026-10-31)

**Verdict: the backward copy streams from the host at no measurable cost, and the
training peak drops about 4.2 GB. Raising the activation budget with that room buys no
speed: budgets 0.05–0.4 all run at 1.33–1.37 s/step.**

`--int8 stream` (then `src/train_int8_moss.py`, since moved to `library/moss/dit_int8/`):
`full` with `wq_t` in pinned host buffers and two block-sized slots on the card
(`WqtStream`, 0.21 GiB). The first dX
of a block takes its slot and queues the block below on a worker thread. Block ids reach
the op as a CPU tensor: an int would be a graph constant, and the 42 blocks share one
graph (`library/moss/loader.py::compile_blocks`). After `wq_t` leaves the card,
`stream_targets` moves `wq`, `s` and `s_t` off and back. Without that, the pages of the
dropped copies stay pinned: free VRAM before training was 6.39 GB instead of 8.64, and
budget 0.4 OOMed with 12.79 GiB allocated of 15.17 in use.

## Speed and peak

`hews_layout`, GPTQ file, fp32 stream, swap 0, 30 steps, median of steps 11–30; jobs
`20261031-1122*`, one run per arm. The peak is `max_memory_allocated`.

| arm | s/step | peak | reserved |
|---|---|---|---|
| `full`, checkpointing | 1.42 | 13.58 GB | – |
| `full`, budget 0.05 | 1.33 | 13.95 GB | – |
| `full`, budget 0.1 | OOM at step 1 | – | – |
| `stream`, checkpointing | 1.45 | 9.36 GB | 9.86 GB |
| `stream`, budget 0.1 | 1.37 | 10.73 GB | 11.03 GB |
| `stream`, budget 0.2 | 1.35 | 11.77 GB | 12.09 GB |
| `stream`, budget 0.3 | 1.33 | 12.80 GB | 13.20 GB |
| `stream`, budget 0.4 | 1.35 | 13.83 GB | 14.18 GB |
| `stream`, budget 0.5 | OOM at step 2 | 14.60 GB | 14.73 GB |

- The speed spread across budgets is the size of run noise. `moss_scoping.md` found FA2's
  forward recomputed in almost every block at budget 0.5, so this range would not reach
  attention's saved set. Attention call counts were not profiled here.
- `stream` costs at most 0.03 s/step (checkpointing, one run each). Each step has one cold
  block (block 41). The "waited" counter reads about 0.84 s per step, but it counts time
  blocked on the GPU's own queue, which the step time shows is not lost.
- `stream` adds 4.4 GiB of pinned host memory.

## EMA shadow on the card

`--ema_card` (`ema_step_card`, now `library/moss/training/ema.py`) lerps the shadow on the card instead of
copying the parameters to the host each update. `stream`, budget 0.2, one shadow, 60 steps,
median of steps 11–60, arms alternating (jobs `20261031-113552-{567084,4750a8,014cc0,02f197}`):

| shadow | s/step, runs 1 / 2 | p90 | peak |
|---|---|---|---|
| CPU | 1.368 / 1.365 | 1.441 / 1.450 | 11.77 GB |
| card | 1.300 / 1.303 | 1.336 / 1.335 | 12.04 GB |

0.065 s/step (4.8%) faster for 0.27 GB. A synthetic bench had predicted about 0.04 s. The
saved EMA sits at the same distance from the raw weights as the CPU shadow's (1.17% against
1.12%, sampled `up` matrices), and the lerp matches the CPU's to 1e-6 in the unit test.

## Same numbers as `full`

- `test_stream_dx_is_full_dx_and_prefetches`: dX bit-identical to `full` through four
  blocks, over two steps.
- Eager, checkpointing (jobs `20261031-111516-*`): step 1 loss 0.350093 in both.
- Compiled: `full` and `stream` differ at step 1 (0.35949 against 0.35962, each repeated
  exactly) and from then on by no more than two runs of one mode differ. The forward uses
  the same weights, so this reads as a different graph's bf16 rounding; not traced.
  Gradient norms cannot settle it, as below.

The two compiled arms' step 1 sits 0.0095 from the eager one because they trained a
different sample order, not a different prediction: the compile line's rich log record drew
from the global `random` the shuffle used, and importing `library.anima` had turned that
logging on. So every compiled int8 run from `train_int8_moss.py` shuffled differently from
bf16 runs of the same seed. Fixed with its own RNG in `library.moss.train` (2026-10-31); same
order as a bf16 run now, and step losses within 5e-4 of it on `hews_layout`.

## Same-seed gradient norms

`|g|` moves between runs of one seed and one mode, while the loss stays within ~3e-4:

| arm | step | run a | run b |
|---|---|---|---|
| bf16, fp32 stream, eager (`20261031-111713-{1ab630,98e163}`) | 2 | 0.077 | 0.378 |
| `full`, eager | 2 | 0.125 | 0.077 |
| `full`, compiled | 7 | 0.313 | 0.100 |

So this is not an int8 effect, and it fits the hot tokens of
[moss_gptq.md](moss_gptq.md#single-token-gradient-outliers), which change from arm to arm:
they also change from run to run. The source of the nondeterminism is not known.

## Not done

- i2i, where the full-size reference is what the room was meant for.
- A complete LoRA on `stream`.
- Budgets above 0.5 on a larger card, or swapping `wq` as well.
