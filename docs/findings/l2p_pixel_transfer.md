# L2P latent→pixel transfer — shelved on the Anima budget

<!-- check-docs: ignore-flags (historical record of shelved work; the flags below never shipped) -->

This records why L2P (turn the frozen latent DiT into a pixel-space DiT by
swapping the VAE for large-patch RGB tokenization) was not promoted past its
Phase-0 probe on Anima. The short version: the paper's load-bearing premise —
that a frozen DiT core trained on VAE-latent token statistics stays "within its
native optimization manifold" when fed RGB-patch tokens through re-trained shells
— **does not transplant to Anima's 2048-dim / 28-block scale on a single-GPU
budget**. Every cheap config we could afford (shells-only, shallow LoRA, real DiP
input-skip detailer) plateaus at the predict-noise floor and generates monochrome
noise from noise, even when the scalar FM loss says it succeeded.

The single most reusable lesson is methodological: single-step teacher-forced FM
loss is a false positive for "is this a generator." A run dropped FM loss −84%
while iterative sampling still produced pure noise. Gate latent→pixel (and similar
frozen-core distill) work on montages, not loss. This is a sharp instance of
[[project_fm_val_loss_uninformative]].

Method reference: Chen et al., *Unlocking Latent Potential for Pixel Generation*,
arXiv:2605.12013. Proposal + module-swap table: `_archive/proposals/l2p_pixel_anima.md`
(the Phase-0 bench code, staged plan, and reference-code delta backlog are no longer
in-tree).

## What L2P is (the part we tested)

Discard the VAE; tokenize raw RGB with a large patch (16² at 1K so the grid stays
64² = ~4096 tokens — already Anima's `CONSTANT_TOKEN_BUCKETS` invariant); freeze
the DiT core; re-train only the input projection + first-n ∪ last-n blocks + a
DiP-style "Detailer Head" (U-Net on the noisy input, skip-connected, frozen-core
features fused at the bottleneck). Train on self-generated synthetic images, same
FM objective Anima already uses (`noisy = (1−σ)x0 + σε`, `target = ε−x0`). Payoff
is native 4K without the VAE-decode memory wall at flat transformer cost — a
*separate pixel model*, not a foldable adapter, not a 1K speedup.

The FM convention, the input-shell linear tokenizer, and (absence of) refiner
blocks were all confirmed identical to the reference — those are not the gap
("Confirmed NON-gaps").

## What the probe does

The Phase-0 probe (since removed from the tree) was the cheapest falsifier: freeze the
entire DiT, swap `x_embedder` → fresh RGB patch-embed and `final_layer`/`unpatchify` →
fresh decoder, overfit ≤64 images at 1024²/bs=1 with the exact Anima FM objective,
Euler-sample montages. Phase-0 gate: loss must drop >30% to <0.85 and montages
must show recognizable colored structure. Decoder is selectable (`--dip_skip` =
real DiP detailer); shallow blocks via `--lora_blocks N` / `--train_blocks N`
(N per-end); `--flow_shift` (default 3) matches the reference 1K pixel schedule.

## Results (all 2026-05-26, 64 imgs @ 1024², bs=1, `anima-base-v1.0`)

| Config | Trainable | FM loss | Montage | Verdict |
|---|---|---|---|---|
| shells-only (pure token decoder, 0 blocks) | shells | 1.425 → 1.052 (−26%), plateau ~1.05 from step ~850 | monochrome noise-blobs, no color | WEAK |
| `--lora_blocks 2` (r32, both I/O ends) | +13.41M | ma50 ~1.05–1.06, tracks shells-only | unchanged | PLATEAU HELD |
| `--lora_blocks 2 --dip_skip --flow_shift 3` | +~19M | 1.317 → 0.207 (−84%); per-σ 0.6–0.8: 0.174, 0.8–1.0: 0.197 | still monochrome noise | LOSS PASS, GEN FAIL |
| above, post padding-mask fix | same | (killed early) | still monochrome noise @ step 750 | FAIL holds |

`~1.05` is essentially the `‖ε−x0‖²` predict-pure-noise floor: shells-only learned
only the low-frequency mean velocity field. Adapting first-n ∪ last-n blocks bought
~nothing (falsifies the shallow-block-capacity hypothesis → the decoder, not block
count, is the bottleneck). The DiP input-skip decoder then drove loss far below the
floor — but generated noise anyway.

## Interpretation

The frozen-core premise doesn't survive the modality swap at this scale. The
core is *reachable* (loss moves) but does not become a pixel generator through
affordable shells. The reference proves the transfer on a 3840/30 source LDM with
8 GPUs / 20k images; Anima's 2048/28 on a single 16 GB GPU is a different regime,
and the transfer doesn't carry.

Why the −84% loss is a mirage. The objective is single-step and teacher-forced:
it scores the velocity field around `x_t = (1−σ)x0 + σε` for a given clean `x0`.
The DiP input-skip lets the head trivially copy the near-clean input at low σ
(paper §3.4 under-corruption "cheat," benign at 1K), so the field is easy to fit
locally. But iterative sampling from noise walks off that teacher-forced manifold
— there is no `x0` to skip-copy — and the learned field doesn't compose into a
trajectory that lands on an image. Low FM loss ⇏ generator. We rewrote the Phase-0
gate to generation-first after this; the scalar criterion alone would have green-lit
a non-generator into Phase 1.

## Scope — what this does not claim

- Not "L2P is wrong." It reproduces at the paper's scale. The negative is
  specific to Anima's width/depth × single-GPU budget.
- We tested the paper-endorsed frozen-core + shallow path, not the released
  full-DiT tune. The authors' released `train_run.sh` full-tunes the entire DiT
  (contradicting their own §3.3 / Fig-9b, which says shallow > full). We
  deliberately kept the frozen-core/LoRA path because it's the only single-GPU
  affordable recipe and the paper's ablation says it should win. Full-DiT tune is
  out of scope here — not falsified, just unaffordable. So the honest framing is:
  the affordable transfer recipe fails on Anima, not "L2P fails on Anima."
- The padding-mask fix (probe was feeding an all-ones mask vs. Anima's zeros for
  fully-valid images) did not change the verdict — the post-fix rerun still
  generated noise.

## What would reopen it

A working transfer would need either the full-DiT tune (needs multi-GPU /
gradient budget we don't have, and the paper's own ablation predicts it degrades)
or evidence that the noise output is an exposure/off-manifold problem fixable with
a sampling-time fix (e.g. consistency-style or multi-step training that isn't
single-step teacher-forced). Neither is cheap; both are speculative. Shelved
pending a reason to pay for one of them.

## Reproduce

The Phase-0 probe is no longer in-tree. Three configs were run in order — shells-only
(→ WEAK), `--lora_blocks 2` (→ plateau held), and `--lora_blocks 2 --dip_skip
--flow_shift 3` (→ loss pass, gen fail) — each judged on the `sample_*.png` montages,
NOT the loss. Open question never
reached: whether existing mid-stack identity/style LoRAs carry over to a pixel
model unchanged (the core is bit-frozen) — moot until a Phase-1 model exists.
Related frozen-DiT + shallow-train shape: [[project_spd_finetune_lora_proposal]].
