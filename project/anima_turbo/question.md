# Question: turbo + SnapRef as a pseudo-video model

Written 2026-11-16. A question, not an arm: nothing here is built or run.

## The question

Put SnapRef (a character-reference adapter for Anima) on our 4-step turbo and render a character
through a sequence of action prompts with one reference. If the frames come out fast and close enough
to each other — not GIF-smooth, but a recognisable motion in steps — the stack behaves like a cheap
pseudo-video model.

## What SnapRef is (read from its code, `github.com/ArcAnima/SnapRef` @ `87bdacd`)

- **Asymmetric attention with a K/V cache.** The reference runs once through its own stream at σ = 0
  with its own LoRA (v1a rank 128) and attends only to itself; every block's self-attention K/V are
  kept. The target stream (its own LoRA, rank 64) reads `[K_ref; K_t]`. The reference is encoded once
  per generation and reused at every step and in both CFG branches; generation is about 15 % slower
  than plain Anima.
- **RoPE: the reference sits on the time axis.** Target at frame t = 0, reference at t = 10 (`REF_T`),
  its (h, w) grid centred on the target. Training jitters the reference (h, w) uniformly in ±32
  patches per axis (inference: 0) "so it is read for appearance and not copied as a layout". No
  ablation of the t value is in the repo.
- **Training pairs:** the same character in another scene and pose (v1a: 58,300 Anima renders,
  2,367 characters, one artist style per character). The target prompt leaves out the character,
  appearance and artist tags; the reference stream gets an identity caption (70 %) or empty text.
  A learned mode embedding on the reference's adaLN input switches "character & style" against
  "character only".
- **Inference knobs:** per-reference weight = `log w` on that reference's attention logits; several
  references are encoded separately and their K/V concatenated. By default only the cond branch
  reads the reference.

## Why it might work

- **Speed.** One reference, encoded once, serves every frame of the sequence, so a frame costs four
  target forwards plus about 15 %. The `turbo_1116_0958` render log saved 768 × 1344 images about
  0.3 s apart (4 steps, int8, compiled, decode included).
- **cfg 1 suits it.** The turbo runs without an uncond branch, which is where SnapRef reads the
  reference by default anyway.
- **Identity across frames** is what SnapRef is trained for.
- **Step 0 carries the layout** on our students (`reports/handoff.md`): sharing the step-0 output of
  one seed across frames and changing the action prompt for steps 1–3 is a ready lever for frame-
  to-frame consistency, without the time axis.

## Why it might not

- **SnapRef is trained against continuity.** The ±32-patch jitter and the different-scene pairs teach
  it to drop the reference's layout, camera and background. A sequence needs those kept; only the
  identity carries over.
- **One time value.** It saw t = 10 only. Nothing in it orders t = 1, 2, 3 or knows what a next frame
  looks like; that it reads the time axis at all may be its LoRA's doing, not a video prior in Anima.
- **σ = 0 references only.** Caching a partly denoised state (e.g. σ 0.9) as the reference is outside
  its training. A finished previous frame as the next reference is in distribution, but chains drift.
- **Base mismatch.** SnapRef was trained on Anima base v1.0 with the 28-step teacher velocity. Our
  turbo runs on a merged soup (merged14 / 15) with a distilled LoRA in the target stream; whether
  the target LoRA's reading of `K_ref` survives the turbo LoRA is the first unknown. Open choice:
  whether the turbo LoRA is on in the reference stream.

## Probes, in order (no training)

1. **Does it load and hold identity?** SnapRef on turbo (Hub v3 on base, and v4_preview on
   merged14), 4 steps, cfg 1, a few references × `judge_hard`-style scene prompts; against SnapRef on
   the 28-step teacher.
2. **A fixed-seed action sequence.** One reference, one seed, 4–8 prompts that differ only in the
   action. Read identity, layout drift and pose change between neighbours.
3. **Shared step 0.** As 2, with step 0 run once on a neutral prompt and steps 1–3 per action
   prompt. Does the layout lock while the pose still moves?
4. **Previous frame as reference** (autoregressive): adds continuity of detail, at the cost of drift
   and of the layout jitter working against it.

If 2 or 3 gives recognisable motion, the next question is training: consecutive-frame pairs with
aligned positions (t = 1, same (h, w), no jitter) in place of SnapRef's different-scene pairs.
Cosmos-Predict2 has a video sibling (2B Video2World) of the same architecture, but its weights are
not Anima's, so no turbo or SnapRef LoRA moves across.
