# Release plan: `anima_turbo_4step_v4_longer`

The merged15 soup with the official turbo v1.1 delta on top, shipped as one LoRA for plain Anima
base. Written 2026-11-18; nothing below has been built or uploaded yet.

## Facts

- The user's call (2026-11-18): the 4-step ships as it stands, to
  `huggingface.co/sorryhyun/anima-turbo-4step` as `anima_turbo_4step_v4_longer.safetensors`. The
  name follows `old/anima_turbo_4step_v2_longer.safetensors`; no longer training is involved.
- Contents: soup `output/ckpt/anima_merged15.safetensors` (rank 256, adaln 128) and
  `models/extracted/anima_turbo_v1.1_delta_r96_asvd_adaln.safetensors` (rank 96 on every module,
  adaln included; an ASVD extraction of the official `anima-turbo-v1.1` release against
  anima-base-v1.0, fp32). No training of this line is in the file.
- The reads it rests on ([../reports/step0_k7of28.md](../reports/step0_k7of28.md), sections 5 and
  6): the user's blind reads of this stack at 4 steps against merged15 at 28 steps and CFG 4, T 5 /
  S 7 on `judge_hard` and T 3 / S 9 on three random dataset captions, T 8 / S 16 over 24 rows.
- How those renders were made: merged15 DiT (its int8 GPTQ file, fp32 stream) with the delta as a
  LoRA at 1.0, 4 Euler steps, cfg 1, shift 3, 768×1344 (`bench/turbo/probe_handoff.py`).
- On the Hub now: `anima_turbo_4step_v4.safetensors` (merged14 with the v3 turbo, rank 256 + 64,
  1.07 GB, the card's top section), `anima_turbo_4step_v4_preview.safetensors`,
  `anima_turbo_4step_v3.safetensors`, `old/`, `onnx/` (int8 exports of v4 and v3).
- Not read: this stack against Hub v4; the v3 turbo against the v1.1 delta on merged15; seed
  diversity (sameness 0.397 / 0.572 against the teacher's 0.314 / 0.504 on the two sets).

## Steps

1. Build the one LoRA (CPU):

   ```bash
   project/anima_turbo/deploy/deploy.sh \
       models/extracted/anima_turbo_v1.1_delta_r96_asvd_adaln.safetensors \
       anima_turbo_4step_v4_longer lora
   ```

   Output `output/ckpt/release/anima_turbo_4step_v4_longer.safetensors`: an exact concat at rank
   256 + 96 (adaln 128 + 96, summed), bf16. Check the header (`merged_from`, `merge_kind =
   concat_exact`, `adaln_merge = sum`, `normalize = off`) and the down ranks 352 / 224. Size
   expected near 1.2 GB (v4 at rank 320 is 1.07 GB).
2. Check that the file renders what was read: `probe_handoff` on plain anima-base-v1.0 with the one
   LoRA as the student, `--prompts_file bench/turbo/judge_random_1118.txt`, and its S column beside
   the S PNGs of `bench/turbo/results/20261118-1922-random3_v11/png/` (local). The base's int8 GPTQ
   file differs from merged15's and the ups are rounded to bf16 (rel err 8e-4,
   `deploy/README.md`), so the pair is read for the same picture per seed, not equal pixels. About
   5 min on the daemon.
3. Card image: one 4-step sheet in the form of `assets/v4_bluearchive.jpg`, saved as
   `assets/v4_longer_<name>.jpg`.
4. Card text: a `## v4_longer` section above `## v4`, naming the file, the soup (merged15), the
   turbo part (the official turbo v1.1 delta, extracted at rank 96), rank 256 + 96 and the size,
   strength 1.0 only, CFG 1, 4 steps, shift 3, no further training.
5. Upload, on the user's word:

   ```bash
   hf upload sorryhyun/anima-turbo-4step \
       output/ckpt/release/anima_turbo_4step_v4_longer.safetensors \
       anima_turbo_4step_v4_longer.safetensors
   ```

   then the card image and `README.md` in one commit.
6. Afterwards: the "Shipped" line of [../README.md](../README.md), and `docs/anima/methods/turbo.md`
   where it names the shipped file.

## Open, the user's calls

- Forms: the one LoRA only, or also the baked DiT and its int8 GPTQ (`deploy.sh … bake` /
  `gptq`, local files) and an ONNX export beside `onnx/int8/dit_int8_v4`.
- Whether `v4_longer` becomes the card's top section or sits under `v4`.
- The turbo part is an extraction of the official turbo release, not a student trained here, while
  the card's first line and `NOTICE.md` describe the repo's weights as DP-DMD distillation. The
  card section should say what the file holds, and the official turbo v1.1 release's licence is to
  be read for redistributing its delta before the upload.

## Interpretation

- Hub v4 was built the same way (a soup with a turbo on top, no further training), so this release
  changes two things from it at once: the soup (merged14 to merged15) and the turbo part (v3 to
  the v1.1 delta). The reads above compare the new stack with its own teacher, not with v4, so the
  card can say it reads level with merged15 at 28 steps and cannot say it is better than v4.
