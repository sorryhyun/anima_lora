# anima_turbo — a few-step DP-DMD student on the merged14 soup

Opened as a project 2026-11-12. The goal is a 4-step turbo whose teacher is the soup
`output/ckpt/anima_merged14.safetensors` (8-shard concat_exact LoRA, rank 256, adaln averaged). It
ships as **one LoRA that loads on plain anima-base-v1.0**: `scripts/toolkits/merge_loras.py
anima_merged14.safetensors <turbo>.safetensors --normalize off --adaln sum` gives exact ΔW14 + ΔT.
The defaults (global normalise, adaln average) break exactness. The turbo LoRA alone means nothing
on base, and the merged LoRA is used at multiplier 1.0 only.

The shipped base-teacher turbo (`huggingface.co/sorryhyun/anima-turbo-4step`) is documented in
`docs/anima/methods/turbo.md`; this line is the work after it.

## Layout

- [plan.md](plan.md): the open plan (A/B protocol, next runs). The closed 2-step plan and its
  results are `91d476e0:scripts/distill_turbo/plan.md`.
- [1112_v1.md](1112_v1.md): the recipe of the first 1000-iteration run (`1112_1512`, shipped as
  `v4_preview`); the 250-iteration control of the later arms is this recipe.
- [idea.md](idea.md): candidates that are not arms.
- [question.md](question.md): turbo + SnapRef (character reference) as a pseudo-video model; probes, none run.
- [vatdm.md](vatdm.md): VA-TDM (arXiv:2611.06796) against the loop and MFD.
- [reports/](reports/README.md): one report per run or probe.
- [proposal/](proposal/README.md): turbo proposals, open and closed.
- [findings/](findings/README.md): settled findings.
- [configs/](configs/README.md): one TOML per run, and the run table.
- [deploy/](deploy/): `deploy.sh` writes the release forms of a student: one LoRA for base, the
  baked DiT, and its 4-step / cfg 1 int8 GPTQ.
- [pref_rl/](pref_rl/): the preference-round window and the head scripts (next section).

## Preference rounds (`make exp-turbo-pref`)

The window of [proposal/pref_rl.md](proposal/pref_rl.md): the user picks renders, and the picks
are the labels of a preference head.

1. `make exp-turbo-pref`, then **New…** creates a session (`output/turbo_pref/<name>/`). The
   defaults are the proposal's policy: merged14 + Hub v3, int8, compiled, 4 steps, cfg 1, 16
   renders a round, at most 8 picks, captions from `post_image_dataset/resized`.
2. **Start server** submits one daemon job that keeps the policy loaded (a warm round renders in
   about 15 s). It holds the GPU until **Stop server**, or 20 minutes with no request. Closing
   the window leaves it running, and a reopened window attaches to it again.
3. Pick and submit. Rounds have no end: the window keeps the next ones queued.

| Action | Input |
|---|---|
| pick / unpick | click, or `1234` / `qwer` / `asdf` / `zxcv` (one row each) |
| open a render large | right-click, or Space over it; ←/→ move, P or a click picks, Esc closes |
| submit | Enter (zero picks is an answer) |
| skip | Ctrl+K (logged, left out of every read) |

- **Next caption…** types the caption of an upcoming round. Otherwise captions are drawn at
  random; no pick-count predictor exists.
- **Blind A/B**: put the LoRA to compare in **Policy…** as arm B, restart the server and switch
  Mode. Each grid is 8 + 8 at matched seeds with the arms hidden; the side panel shows B's share
  of the picks and a sign test over grids.
- `picks.jsonl` in the session holds one row per answered round: caption, seeds, picks, arms and
  the policy that rendered it.

The head (P0 and P1 of the proposal) runs on cached features:

```bash
make daemon-run ARGS="--stall-timeout 900 project/anima_turbo/pref_rl/features.py --set gelcrawl"
make daemon-run ARGS="--stall-timeout 1800 project/anima_turbo/pref_rl/head.py gelcrawl"
make daemon-run ARGS="--stall-timeout 600 project/anima_turbo/pref_rl/features.py --set <session>"
.venv/bin/python project/anima_turbo/pref_rl/head.py picks --session <session>
```

`features.py` caches four bases per image (DINOv3 CLS, PE-Spatial pooled, tagger logits and
embedding) and extracts only the rows it does not hold. `head.py gelcrawl` scores each basis on
held-out artists and saves a head; `head.py picks` reads that head zero-shot, a head fitted on
the picks alone and the gelcrawl head refitted on the picks, each against the sign-test gate.
The DiT-head basis is not built. First read: [reports/pref_p1.md](reports/pref_p1.md).

## Code and data

- Loop: `scripts/distill_turbo/` (`make turbo`); method doc `docs/anima/methods/turbo.md`,
  structure `docs/anima/structure/turbo.md`.
- Benches and probes: `bench/turbo/`, including `dw_overlap.py` (ΔW share between runs),
  `probe_grad_noise.py` (2-step only), `probe_handoff.py` (teacher / student handoff per step
  boundary), `moment_gap.py` and `judge_hard.txt`.
- Teacher: `models/diffusion_models/anima_merged14.safetensors` (baked at 1.0, bf16) with its own GPTQ
  `anima_merged14_int8_gptq.safetensors`. The base GPTQ is refused on the merged DiT.
- Checkpoints in `output/ckpt/<id>/` (125-step saves, `_ema`, `resume.pt`). TB and
  `grad_steps.jsonl` in `output/logs/turbo/<stamp>/`. Renders in `output/tests/turbo4/` (2-step:
  `output/tests/superturbo/`).
- Older material: `_archive/turbo_dmad/` (the removed DMAD path), retired proposals under
  `_archive/proposals/turbo_*`.

## State (2026-11-14)

`v4_preview` (`1112_1512`, 1k EMA on the `1112_v1.md` recipe) is shipped. The 2-step line is closed
at the level of `1111_1115` step 250.

The open gap is composition: every merged14 student draws one frontal layout per prompt. It sits in
step 0 (`reports/handoff.md`), and the layout comes back in proportion to the step-0 anchor fit
(`reports/step0_fit.md`). The 1k students fit at 0.85 of the teacher's cond velocity; Hub v3, after
10k iterations, at 0.30, with the view from below in its renders, and its step 0 carries over to
merged14 untrained (0.063 against 0.146–0.149). Arm 1 (MFD) failed: one image per prompt, mostly
through steps 1–3 (`reports/mfd.md`).

The warm start from Hub v3 (`turbo_1113_1221`) regressed from v3 itself: Hub v3 on merged14 with no
training draws seeds as varied as the teacher's (sameness 0.326 against 0.328) and is the best
4-step result on merged14; the v1 recipe pulls any start toward the control, with `dm_first_step`
off (`turbo_1113_1309`) and with the fake seeded from v3 (`turbo_1113_1438`) alike
(`reports/warm_v3.md`).

Preference RL (`proposal/pref_rl.md`) is at its P1 gate, not passed: on 25 rounds of picks no
head on the four pooled bases ranks picked above unpicked, though each ranks gelcrawl's selection
(`reports/pref_p1.md`). The DiT-head basis is the one not read.
