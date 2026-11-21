# pixel_readout

Code for `project/pixel_decoder/`: a pixel head on the frozen Anima DiT's pre-head features (the
input of `final_layer.linear`) in place of the VAE decoder (FastFrames, arXiv:2611.23226 §3.2,
App. C). Run records, numbers and reads: `project/pixel_decoder/reports/` (index in its
`README.md`). Plan: `project/pixel_decoder/plan.md`; current recipe:
`project/pixel_decoder/reports/phases_p1_p4_2026_11_19.md`.

The model code (`PixelHead`, `PixelTap`, `load_pixel_head`, `lat_patches`, `from_patches`,
`PATCH`) lives in `library/anima/pixel_head.py` since 2026-11-19; `convnext_head.py` and
`linear_probe.py` re-export it for the scripts here.

×2 training of the settled recipe (the real 2× image as the target, PE-GAN from a start
rollout) is `scripts/pixel_head/train_x2.py` on `library/anima/pixel_head_train.py`
(config `configs/pixel_head/x2.toml`) since 2026-11-21. `x2_head.py` here keeps the earlier
arms (teacher target, ×1 loss in a ×2 file) and the whole-image reads with the teacher / chain
columns; it imports the loss, window, discriminator and sheet helpers from the library module.

## Scripts

| script | what it does | records |
|---|---|---|
| `linear_probe.py` | P1: ridge per-token linear readout; writes `readout_linear.safetensors` | `p1_readout_2026_11_18.md` |
| `convnext_head.py` | head training (rollout → DiT forward → VAE target → replay updates); `--points` draws the decode point per rollout; evals on test / text / manga | `p2_skip`, `upsample`, `replay_recipe`, `pagemix`, `p4_t2`, `p4_mix` |
| `p3_generated.py` | the head on sampled images (`generate_body`), base and / or an adapter; reads cfg / cond / teacher-forced | `p3_2026_11_19.md`, `p4_t2_2026_11_19.md` |
| `render_sheets.py` | VAE \| P1 \| head sheets on chosen cached items or raw images (raw: original \| VAE \| head) | `p2_skip`, `p3`, `upsample` |
| `build_pages.py` | precaches manga page latents into `post_image_dataset/pixel_pages/{train,eval}/` | `pagemix_2026_11_19.md` |
| `e2e_speed.py` | decode cost on the `make test SPECTRUM=1` sampler, or turbo with `--adapter --no_spectrum` (head vs VAE resident / swap / disk) | `replay_recipe_2026_11_19.md`, `release_2026_11_19.md` |
| `tone_probe.py` | tone of a head against the VAE on sheets (CPU): block mean shift in code values and in linear light, by texture | `p4_mix_2026_11_19.md` |
| `x2_head.py` | ×2 head: a second 2× stage on a released head, trained on windows against ResShift ×2 of VAE(x̂0); `--rollouts 0 --chain` reads the teacher on the ×1 output; `--resume_x2` continues a trained ×2 head; `--x2flex … --x2_target real` is arm B, `--pegan*` its PE-GAN term (`--rollouts 0 --x2flex <manifest>` is the safe-set read) | `x2_arm_a_2026_11_19.md`, `x2_lpips_dc_2026_11_19.md`, `x2_critic_2026_11_20.md` |
| `lpips_grad_probe.py` | ∂LPIPS/∂v through a head (row-space projection) against the VAE decode of a token window: cosines, a step check, floor and bf16-VAE references, backward cost | `lpips_grad_2026_11_21.md` |
| `lora_ab.py` | a LoRA A/B's renders at the `make test` recipe: labelled pair sheets, blind sets in the `anima-blind-pairs` layout, median ‖dW‖ per file | `lpips_lora_2026_11_21.md` |
| `x2_overfit.py` | can the ×2 stage fit N train windows, per arm (depth, width, frozen base) | `x2_arm_a_2026_11_19.md` |
| `x2_pair.py` | one column of two runs' sheets as an `A \| B` sheet per image (CPU) | `x2_critic_2026_11_20.md` |
| `x2_masters.py` | real 2× targets from the masters: supply manifest and the alignment check (`target_2x`; CPU) | `x2_plan.md`, arm B |

### Archived (2026-11-21)

Finished probes and speed benches with no ×2 path, and the closed ×2 branches (the last three
rows; their flags left `x2_head.py` with them), moved to `_archive/bench/pixel_readout/` (local; in
git up to `370ae506`, the last three up to `cfc75c5d`). They import this package as it was then.

| script | what it did | records |
|---|---|---|
| `rollout_profile.py` | where a training rollout's time goes | `replay_recipe_2026_11_19.md` |
| `ceiling_probe.py` | which target the features reach: VAE(x_t), VAE(x̂0) vs VAE(x0), head vs both | `p2_skip_2026_11_19.md` |
| `overfit_probe.py` | can a fresh head fit 16 images (~70 s per arm) | `p2_skip_2026_11_19.md` |
| `grad_probe.py` | gradient shares and per-image gradient cosines of a head | `p2_skip_2026_11_19.md` |
| `feat_probe.py` | P1's ridge readout on other inputs: last-step feature, previous step, a mid block | `feat_probe_2026_11_20.md` |
| `lpips_speed.py` | LPIPS-VGG step time: channels_last, compile, crop sizes | `x2_arm_b_2026_11_20.md` |
| `pegan_speed.py` | PE-GAN update time: token reuse, batched noised crops, compiled tower | `x2_gan_micro_2026_11_21.md` |
| `x2_dmd.py` | `X2Critic`: the SR line's RSD critic on the ×2 output (frozen ResShift ×2 teacher, fake critic, GAN head, VQ-f4 encoder); a module, no CLI | `x2_critic_2026_11_20.md` |
| `x2_dprobe.py` | discriminator probes behind `x2_head.py --dprobe`: one PE-Spatial discriminator per fake source or frequency band, the head frozen | `x2_disc_probes_2026_11_21.md` |
| `x2_pegan.py` | the PE-Spatial discriminator's first copy; now `PEDisc` in `library/anima/pixel_head_train.py` | `x2_arm_b_2026_11_20.md` |

Every script compiles through `_compile.py::compile_blocks_isolated` (a compile cache per seq-band
signature; a shared cache replayed head training's 4032–4200 guard into the 3952–4096 P3 renders).

All run through the daemon, e.g.:

```bash
make daemon-run ARGS="--queue --stall-timeout 900 bench/pixel_readout/convnext_head.py --compile --label <name>_mmdd_hhmm …"
```

## `convnext_head.py` flags that matter

- Head shape: `--width 1024 --depth 3 --up 1 --no_skip --ls_init 1.0` (current recipe).
- Training: `--rollouts`, `--opt_steps 16 --replay 64`, `--lr 2e-4`, `--target x0|xhat0`,
  `--pages <dir> --page_share p`, `--stream fp32|bf16`.
- Where it reads: `--sigma s` (default: last σ of `--infer_steps`/`--flow_shift`, 0.0577),
  `--adapter <lora>` (hooks after `stream_fp32_`; `attach_adapter`), `--init <run or label>`.
- `load_head` / `resolve_run` take a results path or a bare label (newest `results/*-<label>`).

## Outputs

`results/<YYYYMMDD-HHMM>-<label>/` (local): `result.json` (envelope with per-image PSNR and eval
history), `head.safetensors` (shape, σ, target, adapter in its metadata), `sheet_*.png`,
`text_*.png`, `manga_*.png`.
