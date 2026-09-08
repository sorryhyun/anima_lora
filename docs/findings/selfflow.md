# Self-Flow rep-loss — falsified on the frozen Anima backbone

This records why Self-Flow's representation-distillation objective was not
ported into Anima's LoRA fine-tuning pipeline. The short version: the method
exploits an information gap between clean-noised and heavy-noised views of a
latent that a pretraining-scale, still-learning backbone hasn't closed yet.
Anima's frozen DiT is a finished denoiser — it has already closed that gap, so
the rep loss has nothing to transfer and exerts ~zero pressure on a rank-r
adapter. The cheap pre-flight probe (since removed from the tree) fails both of
its necessary conditions with clear margin.

Method reference: Chefer, Esser et al., Self-Supervised Flow Matching for
Scalable Multi-Modal Synthesis (BFL, arXiv 2603.06507).

## What Self-Flow is (the part we tested)

Self-Flow is a from-scratch / pretraining-scale flow-matching objective that
strengthens a DiT's internal representations instead of importing them from
a frozen DINO/SigLIP encoder (REPA-style). Two mechanisms:

1. Dual-Timestep Scheduling (DTS) — noise each token to one of two sampled
   levels, so the model must infer the noisier tokens from the cleaner ones.
2. Self-Flow proper — an EMA teacher sees the cleaner view; a representation
   loss aligns an early student layer `l` to a later teacher layer `k` (`l < k`)
   via cosine similarity through an MLP head `h`, `sg` on the teacher:

   ```
   L_rep = −E cos( h_θ^(l)(x_τ, τ),  sg[ f_θ'^(k)(x_{τ_min}, τ_min) ] )
   ```

The proposed cheap adaptation for our regime was an EMA-LoRA teacher (only
the adapter is EMA'd; frozen base shared) + a tiny `h` MLP + a `selfflow_rep`
loss entry. This finding is about the rep-loss half only — the DTS-as-
augmentation half is untested and orthogonal (see "What this does not kill").

## What the probe measures

On the frozen base DiT, no adapter, for each latent and each sampled
timestep pair `(τ_lo, τ_hi) = sort(t, s)` sharing one noise sample `ε`, it
builds two in-distribution views — teacher `x_{τ_lo}` (cleaner) and student
`x_{τ_hi}` (noisier) — taps the residual stream of block `l` and block `k` via
forward hooks, and reports two necessary-condition metrics:

1. Information asymmetry `asym_k = median cos(f^k(student), f^k(teacher))`.
   If `≈1`, the frozen backbone produces the same layer-`k` feature regardless of
   input noise level → nothing for the teacher to transfer. *Make-or-break.*
2. Alignment-target headroom — fit only the head `h` (backbone frozen) to
   maximize `cos(h(f^l(student)), f^k(teacher))` on held-out images. If it
   saturates near 1, the target is satisfiable without the adapter.

## Results

Clean run — 24 latents × 8
timestep pairs (192 pairs), bucket `150x112`, `layer_l=6`, `layer_k=18`,
28-block backbone, `anima-base-v1.0`:

| Metric | Value | Threshold | Reads as |
|---|---|---|---|
| `asym_k_median` | 0.984 | ≥0.97 ⇒ no asymmetry | noise gap already collapsed by block 18 |
| `asym_k` favorable (σ≥0.7, gap≥0.4; n=40) | 0.911 | — | even the most generous corner shows only modest asymmetry |
| `asym_l_median` (block 6) | 0.973 | — | lower than `asym_k` — see depth trend below |
| `raw_cos_floor` (no head) | 0.746 | — | early-noisy already predicts late-clean before any learning |
| `headroom_eval_cos` (with head) | 0.986 | ≥0.95 ⇒ trivial | head saturates the target alone |
| `headroom_lift_vs_raw` | +0.240 | — | the lift is real but the ceiling is the problem |
| verdict | NO-ASYMMETRY | | both necessary conditions fail |

## Interpretation

**Anima's frozen DiT is a mature, noise-robust denoiser.** `asym_k = 0.984`
means a token's block-18 representation barely depends on how much noise was on
the input — the network has inferred the clean content and that inference is
robust across noise levels. That is what a well-trained flow backbone is
supposed to do; it is also exactly the capability Self-Flow's rep loss tries
to install.

The depth trend is the mechanism. `asym_l` (block 6) = 0.973 < `asym_k`
(block 18) = 0.984: representations get more noise-invariant with depth. The
network progressively resolves "how noisy was this?" into "what is this?" The
rep loss wants to distill the teacher's late-layer (block 18) feature — but
that feature is ~identical between the clean and noisy views, so there is no
extra information to move. Self-Flow exploits a gap a half-trained backbone
hasn't closed; Anima already closed it.

The headroom says the same thing from the other side. `raw_cos_floor =
0.746` → with head `0.986`: the student's early-layer noisy feature already
predicts the teacher's late-layer clean feature, and a trivial frozen-backbone
MLP finishes the job. The alignment target is met without touching the
backbone, so a rank-r LoRA delta receives ~zero gradient from `L_rep`.

The one sign of life doesn't rescue it. Asymmetry survives only in the
high-σ wide-gap corner (`asym_k` favorable = 0.911, n=40) — where the noisier
view genuinely lacks information. But it's still 0.91, it's a small slice of the
schedule, and a rank-r adapter can't meaningfully move *deep-layer*
representations there. A usable lever needs asymmetry in the bulk; the bulk is
collapsed.

## What this does not kill

The probe tests the rep-loss / EMA-teacher half. It says nothing about
DTS used purely as a training augmentation — heterogeneous per-token noise
on the plain flow-matching loss, no teacher, no second forward (and so it runs
under block swap). The paper reports DTS alone slightly helps vanilla flow
matching (Fig 11b); that is a regularization effect on the training signal,
independent of backbone asymmetry and untested here. If anything from Self-Flow
is worth a run on Anima, it's DTS-as-augmentation — not the teacher path.

## Confidence and caveats

Two independent metrics fail with margin, and the depth trend gives a
mechanistic *why* rather than a bare number, so the rep-loss conclusion is
solid. The probe tests necessary, not sufficient conditions — but here they
fail, which is dispositive in the negative direction. Layer choice (6/18) is
the only real degree of freedom; the deep teacher tap at 18 is where the target
lives and that's the collapsed one, so shallower taps won't revive the rep loss.

Two probe bugs were fixed before these numbers were trusted (both would have
silently corrupted the verdict):

- Timestep scale. The DiT forward takes timesteps in `[0, 1]` (training
  divides by 1000; inference's `*1000` in `get_timesteps_sigmas` is undone by
  `/= 1000` at every call site). The probe was conditioning at `t*1000`,
  driving the backbone out of distribution. Fixed to pass `t` directly.
- fp16 storage overflow. Late-block residual-stream values exceed fp16's
  65504 ceiling; storing captures via `.half()` produced `inf → nan` headroom.
  Fixed to store fp32.

## Reproduce

The probe is no longer in-tree. It was run on the frozen base DiT over 24 samples
× 8 timestep pairs at `layer_l=6` / `layer_k=18`, emitting `result.json` +
`per_pair.csv`; the training A/B that a pass would have unlocked was never run.
