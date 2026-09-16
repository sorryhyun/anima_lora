"""RSD distillation training loop (reconstruction of paper 22490, Algorithm 1).

Two-timescale: K fake-critic updates per generator update. Frozen v2 teacher, 1-step
stochastic student, fake ResShift critic + GAN head, image-space LPIPS. See DESIGN.md.

Run in the root venv (`make sr-*` targets removed 2026-08-22 — line finished):
  make daemon-run ARGS="project/finished/sr/distill_rsd/train.py --iters 3000
                        --src project/finished/sr/data/rsd_hr_cap4096 [--bs 6 --compile] [--no-amp]"
(perf defaults benched 2026-07-02 — see DESIGN.md "Throughput".)
"""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library.training.ema import ema_update, make_ema

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsd_models as M  # noqa: E402
from data import ArtSRDataset  # noqa: E402
from rsd_models import make_eps, predict_x0  # noqa: E402


def dc_loss(a, b, k=32):
    """L1 on the low-frequency (DC + coarse tone) band of the decoded image.

    VGG-LPIPS is ~invariant to a uniform color/tone shift and the DMD gradient is
    per-sample magnitude-normalized, so the global DC (mean color) is the objective's
    unconstrained null-space — the 1-step student settles into a small systematic tint
    (measured ~-0.027 on blue, -0.012 luma vs GT; the VQGAN roundtrip is color-faithful,
    so it's the student, not the decoder). Avg-pooling to a coarse map and matching L1
    pins that band without touching high-freq detail. Both a,b are image-space [-1,1].
    """
    return F.l1_loss(F.avg_pool2d(a, k), F.avg_pool2d(b, k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=18000, help="generator updates")
    ap.add_argument(
        "--K", type=int, default=5, help="fake updates per generator update"
    )
    ap.add_argument(
        "--bs",
        type=int,
        default=4,
        help="batch size (default 4: benched 2026-07-02 on the 16GB 5070 Ti — "
        "bs4 no-ckpt eager 2.47 rel-throughput vs bs2-ckpt 1.31; bs8 OOM)",
    )
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lambda_lpips", type=float, default=2.0)
    ap.add_argument("--lambda_gan", type=float, default=3e-3)
    ap.add_argument(
        "--lambda_dc",
        type=float,
        default=1.0,
        help="low-freq DC/color match on the 1-step decoded image — pins the "
        "global tone LPIPS+DMD leave free (fixes the student color drift). "
        "0 disables; raise toward ~5 if a tint persists.",
    )
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[4, 8, 12, 14],
        help="N multistep timestep nodes (0-indexed in [0,T-1])",
    )
    ap.add_argument(
        "--sid_denom",
        choices=["student", "input"],
        default="student",
        help="SiD per-sample normalization denominator for the DMD gradient. "
        "'student' = |teacher_x0 - student_x0| + 1e-8 (matches official RSD "
        "release, trainer.py:1836); 'input' = |z_t - teacher_x0| floored 0.05 "
        "(this reconstruction's original form).",
    )
    ap.add_argument(
        "--noise_mode",
        choices=["add", "concat"],
        default="concat",
        help="student/fake noise injection. 'concat' = widen the first conv by "
        "noise_channels and concat eps (default; matches the official RSD "
        "release, sampler.py:79 — won the 500-iter A/B, MUSIQ 65.8 vs add 64.0); "
        "'add' = zero-init 3ch conv added after block0 (the original "
        "reconstruction; back-compatible with pre-concat checkpoints, which "
        "won't load into the widened concat conv).",
    )
    ap.add_argument(
        "--noise_channels",
        type=int,
        default=None,
        help="injected-noise channel count (default 3 for add, 1 for concat/official)",
    )
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bf16 autocast on the nets + NATIVE-bf16 VQGAN (matches infer.py; "
        "autocast alone would keep the VQGAN GroupNorms fp32 and materialize "
        "the full-res norm activations). --no-amp = full fp32.",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="block-compile the Swin/Res blocks at the CLASS level (shared across "
        "student/fake/teacher; glue stays eager, checkpoint() stays outside "
        "the compiled region — see rsd_models.compile_swin_blocks). Per-step "
        "it/s is a wash vs eager, but the ~1.2GB VRAM save is what lets "
        "`--bs 6` run without grad-ckpt (best measured throughput, ~14.8GB). "
        "Whole-graph compile was measured strictly worse — don't revive it.",
    )
    ap.add_argument(
        "--grad_ckpt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Swin gradient checkpointing on student/fake (bit-exact activation "
        "save; default OFF — at 256-px crops the recompute costs ~25%% "
        "throughput and bs4 fits without it; turn on to fit bigger bs "
        "on smaller cards)",
    )
    ap.add_argument(
        "--no_grad_ckpt", dest="grad_ckpt", action="store_false", help=argparse.SUPPRESS
    )  # legacy spelling (pre-2026-07 runs)
    ap.add_argument(
        "--src",
        default=None,
        help="HR source dir (default image_dataset; pass the prep_rsd_cache "
        "4096-capped cache for faster decode at the right 1024->4096 scale)",
    )
    ap.add_argument("--num_workers", type=int, default=6, help="DataLoader workers")
    ap.add_argument(
        "--version",
        choices=list(M.VERSIONS),
        default="x4",
        help="teacher family: x4 = released 15-step v2; x2 / x4ft = our sr-train "
        "finetunes (x4ft = the art-finetuned x4 teacher, same 15-step schedule)",
    )
    ap.add_argument(
        "--cond_lq",
        choices=["pixel", "latent"],
        default=None,
        help="what to feed the UNet's `lq` conditioning input (default: the "
        "version's — pixel for x4/x4ft, latent for the legacy x2 line). The "
        "teacher, student and fake critic all use it, so it also lands in the "
        "student ckpt meta for infer.py. See rsd_models.make_cond_lq: feeding "
        "the VQ latent to a released teacher wrecks its x0 prediction "
        "(img-MSE 0.004 -> 1.50 at t=14) and was the pre-2026-07-11 default.",
    )
    ap.add_argument(
        "--config", default=None, help="override the version's ResShift config yaml"
    )
    ap.add_argument(
        "--teacher",
        default="",
        help="override the version's teacher checkpoint (empty = the version's "
        "default — do NOT hardcode one here: a non-empty default wins over "
        "--version and silently distills the wrong teacher)",
    )
    ap.add_argument(
        "--save_dir",
        default=None,
        help="default: output/sr/rsd (x4) or output/sr/rsd_<version>",
    )
    ap.add_argument("--log_every", type=int, default=40)
    ap.add_argument("--save_every", type=int, default=4500)
    ap.add_argument(
        "--max_steps", type=int, default=0, help="smoke cap on gen updates (0=off)"
    )
    args = ap.parse_args()

    dev = M.DEVICE
    config_path, teacher_ckpt = M.resolve_version(
        args.version, args.config, args.teacher
    )
    default_sub = "rsd" if args.version == "x4" else f"rsd_{args.version}"
    save_dir = (
        Path(args.save_dir) if args.save_dir else M.REPO / "output" / "sr" / default_sub
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    cfg = M.load_configs(config_path)
    sf_scale = cfg.diffusion.params.scale_factor
    cond_mode = args.cond_lq or M.cond_mode_for(args.version)
    print(
        f"version={args.version} sf={cfg.diffusion.params.sf} teacher={teacher_ckpt.name} "
        f"config={config_path.name} cond_lq={cond_mode}"
    )

    print("building nets...")
    gen_kw = dict(noise_mode=args.noise_mode, noise_channels=args.noise_channels)
    tck = str(teacher_ckpt)
    teacher = M.build_teacher(cfg, tck, dev)
    student = M.build_generator(cfg, tck, dev, grad_ckpt=args.grad_ckpt, **gen_kw)
    fake = M.build_generator(cfg, tck, dev, grad_ckpt=args.grad_ckpt, **gen_kw)
    disc = M.DiscHead().to(dev)
    vqgan = M.build_autoencoder(cfg, dev)
    if args.amp:
        # native bf16 (mirrors infer.py): the frozen VQGAN is the full pixel-res conv
        # stack; under autocast its GroupNorms stay fp32 and dominate encode VRAM/time.
        vqgan = vqgan.to(torch.bfloat16)
    vdt = next(vqgan.parameters()).dtype
    diff = M.build_diffusion(cfg)
    T = diff.num_timesteps
    ema = make_ema(student)
    # provenance for inference (rebuilds the matching noise-injection arch) + A/B tracking
    ckpt_meta = {
        "noise_mode": student.noise_mode,
        "noise_channels": student.noise_channels,
        "sid_denom": args.sid_denom,
        "cond_lq": cond_mode,
    }

    import lpips

    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95))
    opt_f = torch.optim.AdamW(
        list(fake.parameters()) + list(disc.parameters()), lr=args.lr, betas=(0.9, 0.95)
    )

    if args.compile:
        # Block-compile (class-level, state_dict-transparent). All shape/grad-mode
        # variants share each class's ONE code object, so the recompile budget must be
        # raised — and pinned via its ContextVar .default or it silently reverts to 8 in
        # the backward-compile context and spills to eager (same trap as distill_mod).
        sys.path.insert(0, str(M.REPO))
        from library.runtime.dynamo import pin_dynamo_limit  # dependency-free helper

        for knob in ("recompile_limit", "cache_size_limit"):
            pin_dynamo_limit(knob, 64)
        M.compile_swin_blocks()

    loader = DataLoader(
        ArtSRDataset(
            src=args.src,
            gt_size=256,
            scale=cfg.diffusion.params.sf,
            length=args.iters * (args.K + 1) * args.bs * args.grad_accum + 1000,
        ),
        batch_size=args.bs,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    it = iter(loader)
    nodes = torch.tensor(args.nodes, device=dev)

    def autocast():
        return (
            torch.autocast("cuda", dtype=torch.bfloat16) if args.amp else nullcontext()
        )

    def next_batch():
        nonlocal it
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader)
            b = next(it)
        return b["gt"].to(dev, non_blocking=True), b["lq"].to(dev, non_blocking=True)

    def encode(gt, lq):
        # one fused 2B encode (native bf16 under --amp); latents come back fp32 so all
        # downstream diffusion math is unchanged (they're tiny — 3ch at /4 res).
        with torch.no_grad():
            z = vqgan.encode(torch.cat([gt, lq]).to(vdt)).float() * sf_scale
        return z.chunk(2)

    def rand_t(B, src):
        return (
            src[torch.randint(0, len(src), (B,), device=dev)]
            if torch.is_tensor(src)
            else torch.randint(0, src, (B,), device=dev)
        )

    log_path = save_dir / "progress.jsonl"
    t0 = time.time()
    print(
        f"training: iters={args.iters} K={args.K} bs={args.bs}x{args.grad_accum} "
        f"T={T} nodes={args.nodes} amp={args.amp} compile={args.compile}"
    )

    for step in range(args.iters):
        # ===== K fake-critic updates =====
        for k in range(args.K):
            opt_f.zero_grad(set_to_none=True)
            gt, lq = next_batch()
            B = gt.shape[0]
            z0, z_y = encode(gt, lq)
            c_y = M.make_cond_lq(
                lq, z_y, cond_mode
            )  # `lq` conditioning != residual base z_y
            t_n = rand_t(B, nodes)
            eps = make_eps(student, z0)
            with torch.no_grad(), autocast():
                z_tn = diff.q_sample(z0, z_y, t_n)
                z0_hat = predict_x0(diff, student, z_tn, c_y, t_n, eps)
            t = rand_t(B, T)
            z_t = diff.q_sample(z0_hat, z_y, t)
            with autocast():
                x0_fake = predict_x0(diff, fake, z_t, c_y, t, eps)
                L_fake = F.mse_loss(x0_fake.float(), z0_hat.float())
                # real + fake GAN features in ONE 2B encoder pass (was two serial passes)
                feats = fake.encode_features(
                    torch.cat([z0, z0_hat.detach().float()]),
                    c_y.repeat(2, 1, 1, 1),
                    eps=eps.repeat(2, 1, 1, 1),
                )
                d_real, d_fake = disc(feats).chunk(2)
                L_gan_d = (F.softplus(-d_real) + F.softplus(d_fake)).mean()
            (L_fake + args.lambda_gan * L_gan_d).backward()
            opt_f.step()

        # ===== 1 generator update =====
        opt_g.zero_grad(set_to_none=True)
        g_logs = {}
        for _ in range(args.grad_accum):
            gt, lq = next_batch()
            B = gt.shape[0]
            z0, z_y = encode(gt, lq)
            c_y = M.make_cond_lq(lq, z_y, cond_mode)
            t_n = rand_t(B, nodes)
            eps = make_eps(student, z0)
            z_tn = diff.q_sample(z0, z_y, t_n)
            # single-step LPIPS path from z_T ~ N(z_y, kappa^2), stacked into the SAME
            # student forward as the node path (t is per-sample, so the two paths batch
            # cleanly) — halves the most expensive fwd+bwd of the generator phase.
            z_TT = z_y + diff.kappa * torch.randn_like(z_y)
            tT = torch.full((B,), T - 1, device=dev, dtype=torch.long)
            with autocast():
                out = predict_x0(
                    diff,
                    student,
                    torch.cat([z_tn, z_TT]),
                    c_y.repeat(2, 1, 1, 1),
                    torch.cat([t_n, tT]),
                    torch.cat([eps, make_eps(student, z_y)]),
                )
            z0_hat, z0_single = out[:B], out[B:]
            t = rand_t(B, T)
            z_t = diff.q_sample(z0_hat.detach().float(), z_y, t)
            with torch.no_grad(), autocast():
                x0_teacher = predict_x0(diff, teacher, z_t, c_y, t)
                x0_fakep = predict_x0(diff, fake, z_t, c_y, t, eps)
                # DMD/SiD per-sample normalization (App C, "loss normalization … SiD"):
                # divide the (fake−teacher) push by a TEACHER-derived magnitude — NOT by the
                # (fake−teacher) diff's own norm (self-normalizing flattens the
                # distribution-matching signal so every sample gets a unit push regardless of
                # how wrong it is, and amplifies critic noise once fake≈teacher).
                grad = (x0_fakep - x0_teacher).float()
                if args.sid_denom == "student":
                    # official RSD release (trainer.py:1836): |teacher_x0 − student_x0| + 1e-8
                    denom = (x0_teacher.float() - z0_hat.detach().float()).abs().mean(
                        dim=[1, 2, 3], keepdim=True
                    ) + 1e-8
                else:
                    # original reconstruction: ‖z_t − f*_x0‖ floored 0.05 (turbo_dmd norm_floor)
                    denom = (
                        (z_t.float() - x0_teacher.float())
                        .abs()
                        .mean(dim=[1, 2, 3], keepdim=True)
                        .clamp_min(0.05)
                    )
                grad = grad / denom
            L_theta = 0.5 * F.mse_loss(z0_hat.float(), (z0_hat.float() - grad).detach())
            # decode runs native bf16 outside autocast (grad flows through the frozen
            # decoder; matches infer.py's VAE handling), image back to fp32 for the losses
            x0_img = (
                vqgan.decode(z0_single.to(vdt), force_not_quantize=True)
                .float()
                .clamp(-1, 1)
            )
            # reference the roundtrip-consistent decode(z0), NOT raw gt: the frozen VQ roundtrip
            # is color-tinted on this data, so comparing decode(pred) against raw gt puts the
            # LPIPS/dc color-minimum at the anti-tint and injects a color drift (the 1-step
            # student can't compound it like the x2 rollout does, but the residual is real —
            # ~-0.03 blue). decode(z0) coincides with the latent objective's zero. See the
            # matching fix + full derivation in sr/train_sr/train.py.
            with torch.no_grad():
                gt_img = (
                    vqgan.decode(z0.to(vdt), force_not_quantize=True)
                    .float()
                    .clamp(-1, 1)
                )
            with autocast():
                L_lpips = lp(x0_img, gt_img).mean()
                L_dc = dc_loss(x0_img, gt_img.float())
                L_gan_g = F.softplus(
                    -disc(fake.encode_features(z0_hat, c_y, eps=eps))
                ).mean()
            loss = (
                L_theta
                + args.lambda_lpips * L_lpips
                + args.lambda_dc * L_dc
                + args.lambda_gan * L_gan_g
            ) / args.grad_accum
            loss.backward()
            g_logs = {
                k: v.detach().item()
                for k, v in {
                    "L_theta": L_theta,
                    "L_lpips": L_lpips,
                    "L_dc": L_dc,
                    "L_gan_g": L_gan_g,
                    "L_fake": L_fake,
                    "L_gan_d": L_gan_d,
                }.items()
            }
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt_g.step()
        ema_update(ema, student, args.ema)

        if step % args.log_every == 0:
            rate = (step + 1) / (time.time() - t0)
            line = {
                "step": step,
                **{k: round(v, 4) for k, v in g_logs.items()},
                "it_s": round(rate, 3),
                "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            }
            print(line)
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")
        if step > 0 and step % args.save_every == 0:
            torch.save(
                {
                    "ema": ema.state_dict(),
                    "student": student.state_dict(),
                    "step": step,
                    **ckpt_meta,
                },
                save_dir / f"rsd_student_{step}.pth",
            )
        if args.max_steps and step + 1 >= args.max_steps:
            print(f"max_steps {args.max_steps} hit — stopping (smoke).")
            break

    torch.save(
        {
            "ema": ema.state_dict(),
            "student": student.state_dict(),
            "step": args.iters,
            **ckpt_meta,
        },
        save_dir / "rsd_student_final.pth",
    )
    print(f"done. saved to {save_dir}")


if __name__ == "__main__":
    main()
