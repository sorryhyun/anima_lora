# FA2 forward tile on sm_120 (2026-11-02)

**Verdict: fails the 1.15x gate.** FA2's sm86 / 89 tile, 128×32, runs 1.060x on
self-attention and 1.117x on cross-attention at FA2's error, which is about 0.7% of an Anima
step. CUDNN attention, the other forward on this card, reaches 1.072x and its backward is
slower. Candidate 1 of [step_headroom.md](../../../docs/moss/proposal/step_headroom.md) is closed. It costs no
accuracy, so a local build is free to take; a patched build below reproduces the speed.

Script: [`src/fa2_tile_bench.py`](../src/fa2_tile_bench.py), shim
[`src/cc_spoof.c`](../src/cc_spoof.c), numbers [`fa2_tile.json`](../assets/fa2_tile.json), job
`20261102-125641-d3634a`.

## Setup

- The wheel holds both tiles and picks one at run time from `cudaDeviceGetAttribute`, so no
  build was needed. The shim, in `LD_PRELOAD`, returns compute capability 8.6 while a flag is
  set. It reaches `is_sm8x`, and the profiled kernel name changed to
  `Flash_fwd_kernel_traits<128, 128, 32, 4, …>` in that arm.
- Bf16 captured calls from `output/int8_train/attn_capture.pt`, shapes q `[1, 4131, 16, 128]`, Lk
  4131 (self) or 512 (cross). Six timed rounds of 50 forwards or 20 forward + backward, arms
  alternating, after one warm-up round; medians. Error is the median over the 48 calls per
  kind against fp32 SDPA.
- CUDNN 9.24 through `sdpa_kernel(SDPBackend.CUDNN_ATTENTION)`. Its forward + backward is
  CUDNN's own backward; both FA2 arms use FA2's.

## Results

ms, speedup against 128×64 in brackets:

| arm | self fwd | self fwd + bwd | self err | cross fwd | cross fwd + bwd | cross err |
|---|---|---|---|---|---|---|
| FA2 128×64 | 1.575 | 5.327 | 0.00170 | 0.208 | 0.808 | 0.00181 |
| FA2 128×32 | 1.486 (1.060x) | 5.266 (1.012x) | 0.00169 | 0.186 (1.117x) | 0.786 (1.029x) | 0.00175 |
| CUDNN | 1.470 (1.072x) | 6.347 (0.839x) | 0.00170 | 0.203 (1.025x) | 1.070 (0.755x) | 0.00190 |

Over 28 blocks 128×32 saves 3.1 ms of Anima's 430 ms step, 0.7%. The proposal assumed a
1.2–1.4 ms forward, which neither arm reaches.

## MOSS and the patched build

FA2 2.8.3 sdist, `is_sm8x` extended with `|| cc_major == 12` in `run_mha_fwd_hdim128`, `-std=c++20`
for torch 2.14's headers, sm_120 only, head dim 128 only (`HEADDIM_SWITCH` raises on any other).
13 objects in about 4 minutes at `MAX_JOBS=6`; 73 at `MAX_JOBS=10` is about 50 GB of `cicc`.
Source and `.so` in `new/anima_output/int8_train/fa2_build/`; wheel in the private repo
[sorryhyun/flash-attention-fp16](https://github.com/sorryhyun/flash-attention-fp16); not installed in the venv.

MOSS has one self-attention per block at q `[1, 4600, 24, 128]`, timed on random bf16, so no error
figure. Jobs `20261102-131059-3bf34f` (wheel), `20261102-131121-47a965` (build):

| | self 128×64 | self 128×32 | cross 128×64 | cross 128×32 | MOSS 128×64 | MOSS 128×32 |
|---|---|---|---|---|---|---|
| wheel, fwd | 1.583 | 1.491 | 0.209 | 0.187 | 2.831 | 2.674 |
| build, fwd | 1.582 | 1.499 | 0.209 | 0.187 | 2.831 | 2.684 |
| wheel, fwd + bwd | 5.354 | 5.245 | 0.809 | 0.786 | 9.832 | 9.653 |
| build, fwd + bwd | 5.332 | 5.244 | 0.808 | 0.785 | 9.848 | 9.670 |

The build matches the wheel within the rounds' spread, and takes 128×32 with no spoof. MOSS gains
1.055x on the forward and 1.018x with the backward: 7.5 ms of 1059 ms over 42 blocks, 0.7%, and
about 1.6% of a generation forward at attention's 30% share.

The venv's FA2 also serves ComfyUI (`comfy/ldm/modules/attention.py`) and Qwen's optional
`flash_attention_2` text encoder, and `attention_dispatch` and `model.py` turn an ImportError
into SDPA silently. An install therefore needs every head dim, or a route limited to repo jobs.

## Not tried

Tiles the wheel does not hold (64×64 non-causal, 8 warps), which need a build. CUDNN's 1.072x
suggests little is left there, but that is not a measurement of those tiles.
