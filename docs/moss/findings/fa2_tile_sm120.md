# FA2's forward takes the wrong tile on sm_120, and fixing it is worth 0.7%

Measured 2026-11-02 on an RTX 5070 Ti, for MOSS-Image and Anima. Numbers, setup and job ids:
[`project/int8_train/reports/fa2_tile_sm120.md`](../../../project/int8_train/reports/fa2_tile_sm120.md)
(script `project/int8_train/src/fa2_tile_bench.py`). Started as candidate 1 of
`_archive/proposals/step_headroom.md`.

## Setting

`run_mha_fwd_hdim128` picks its tile from the compute capability alone, and FA2 has no option for
it. Compute capability 12.0 falls to the A100 / H100 branch, 128×64. The sm86 / 89 branch takes
128×32 on non-causal calls. Both MOSS (`dim 3072 / 24 heads`) and Anima run head dim 128, so both
land on 128×64. MOSS's padded batches go to SDPA and never reach this kernel.

## Measured

| | 128×64 | 128×32 | forward | forward + backward |
|---|---|---|---|---|
| Anima self | 1.582 ms | 1.499 ms | 1.056x | 1.017x |
| Anima cross | 0.209 ms | 0.187 ms | 1.118x | 1.029x |
| MOSS self, random input | 2.831 ms | 2.684 ms | 1.055x | 1.018x |

- **The tile is faster and costs no accuracy.** Median distance from fp32 goes from 0.00170 to
  0.00169 on self-attention and from 0.00181 to 0.00175 on cross-attention. A seed may render
  slightly differently; no render was compared.
- **The gain is small.** 3.1 ms of Anima's 430 ms step and 7.5 ms of MOSS's 1059 ms, 0.7% each, and
  about 1.6% of a MOSS generation forward. The proposal's 1.5–2.6% assumed a 1.2–1.4 ms forward.
- **CUDNN attention (9.24) is no alternative.** Its forward is 1.07x, but its backward puts forward
  + backward at 0.84x on self-attention.
- **A patched build matches the wheel's 128×32 path.** The wheel's own 128×32 binary was timed by
  interposing `cudaDeviceGetAttribute` (`cc_spoof.c`); a build from the 2.8.3 sdist reproduces it.

## Build

The tile ships in the patched sm_120 build together with the fp16-accumulate backward:
[fa2_sm120.md](../optimization/fa2_sm120.md) has the install, head-dim limits and the ComfyUI caveat.
`attention_dispatch.py` and `library/moss/model.py` turn an ImportError into SDPA silently, so a
torch bump past the wheel's minor slows both lines without an error.

## Untested

Tiles neither binary holds (non-causal 64×64, 8 warps). MOSS's real inputs, whose error was not
measured. A sm_120 card other than this one.
