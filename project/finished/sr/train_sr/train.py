"""ResShift domain-finetune loop (art SR sidecar) — ×2 and ×4.

The upstream ResShift trainer + basicsr data pipeline are NOT vendored (sr/resshift/
is config/ldm/models/utils only), so this is our own loop — same shape as
distill_rsd/train.py but far simpler: ONE model, ONE optimizer, the canonical
residual-shift latent MSE objective (predict_type=xstart).

`--version` (default x2) picks the (config, warm-start ckpt, output dir) triple — see
VERSIONS below. The three scales share every line of this loop; the config only moves
`sf` (and, for x4s4, the noise schedule to match the v3 weights).

  x2    sf=2, 15-step schedule, warm-start = released x4 v2 (strict load — the x2
        config IS the x4 config with sf flipped; the schedule is sf-independent).
  x4    sf=4, 15-step, warm-start = released x4 v2. Domain-finetune of the teacher the
        shipped 1-step RSD student was distilled from — the student has ~saturated it
        (68.0 vs 68.4 MUSIQ), so lifting the teacher is the only way up. Feeds
        `sr-rsd-train --version x4ft`.
  x4s4  sf=4, **4-step** v3 schedule, warm-start = released x4 v3. v3 scores higher on
        our art eval (68.4) and is 4x cheaper to roll out, but the distiller needs an
        explicit `--config sr/configs/realsr_x4_s4_art.yaml` to use it.

What it does:
  - builds UNetModelSwin from the version's config (lq_size=64 throughout)
  - WARM-STARTS from a released ResShift checkpoint (strict load — see warm_start).
    The model already knows ResShift denoising; finetuning adapts it to our art domain
    (and, at x2, to the easier 2x gap).
  - ArtSRDataset over the HR pool: gt = 256² HR crop, lq = bicubic ÷sf + light JPEG/
    blur, upsampled back to 256² (our LR-to-HR-size convention -> z_y at 64 = z0).
    Text-fidelity knobs ON by default (both no-op gracefully if the box json is
    missing): --text_crop_prob 0.25 biases crops onto CTD text boxes
    (sr/scripts/detect_text_boxes.py) and --scale_jitter_prob 0.25 degrades at a
    random scale in [sf, --scale_jitter_max] (auto = 2·sf) so tiny glyphs appear with
    GT supervision instead of being hallucinated at inference.
  - per step: t ~ U[0,T), z_t = q_sample(z0, z_y, t), pred = model(scale_input(z_t),
    t, lq=c_y), loss = MSE(pred, z0) + decoded-image LPIPS (--lambda_lpips, default ON
    since 2026-07-06: the latent-MSE-only teacher was confirmed as the student quality
    ceiling — edge detail regresses to the mean) + low-freq DC color anchor
    (--lambda_dc, ported from distill_rsd: LPIPS opens the color null-space that pure
    MSE kept pinned, so the two ship together). The image losses compare decode(pred)
    against decode(z0), NOT raw gt — the frozen VQ roundtrip is color-tinted on this
    data, so referencing raw gt disagrees with the latent MSE zero and bakes a green
    cast into the model (fixed 2026-07-06; see the loss-branch comment).
  - EMA, bf16 AMP + native-bf16 VQGAN (default ON), optional block-compile /
    Swin grad-checkpointing, periodic p_sample_loop montage eval.
    Perf defaults follow the 2026-07-02 distill_rsd pass (same net, same 256² crops):
    grad-ckpt OFF (recompute costs ~25%), bf16 ON, --compile to trade warmup for VRAM.

Run in the root venv (`make sr-*` targets removed 2026-08-22 — line finished):
  make daemon-run ARGS="project/finished/sr/train_sr/train.py --iters 30000 [--bs 8 --compile] [--no-amp]"
or ×4: append --version x4.
"""

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from library.training.ema import ema_update, make_ema

HERE = Path(__file__).resolve().parent
SR = HERE.parent
sys.path.insert(0, str(SR / "distill_rsd"))  # reuse rsd_models + data
import rsd_models as M  # noqa: E402
from data import ArtSRDataset  # noqa: E402
from rsd_models import predict_x0  # noqa: E402
from models.unet import UNetModelSwin  # noqa: E402  (vendored ResShift, on sys.path via rsd_models)

# version -> (config, warm-start checkpoint, output subdir under output/sr/, cond_lq mode)
# cond_lq: "pixel" = the released conditioning convention (LR pixels at latent resolution),
# so the warm-start prior actually transfers. "latent" only for x2, whose already-shipped
# teacher + 1-step student were both trained under it — see rsd_models.make_cond_lq.
VERSIONS = {
    "x2": (
        SR / "configs" / "realsr_x2_art.yaml",
        M.WEIGHTS / "resshift_realsrx4_s15_v2.pth",
        "x2_lpips_30k",
        "latent",
    ),
    "x4": (
        SR / "configs" / "realsr_x4_art.yaml",
        M.WEIGHTS / "resshift_realsrx4_s15_v2.pth",
        "x4_art",
        "pixel",
    ),
    "x4s4": (
        SR / "configs" / "realsr_x4_s4_art.yaml",
        M.WEIGHTS / "resshift_realsrx4_s4_v3.pth",
        "x4_s4_art",
        "pixel",
    ),
}


def dc_loss(a, b, k=32):
    """L1 on the low-frequency (DC + coarse tone) band of the decoded image.

    Same rationale as distill_rsd/train.py: VGG-LPIPS is ~invariant to a uniform
    color/tone shift, so once LPIPS joins the objective the global DC becomes an
    unconstrained null-space (the pure-MSE teacher measured ~0 drift — it's LPIPS
    that opens it). Avg-pooling to a coarse map and matching L1 pins that band
    without touching high-freq detail. Both a,b are image-space [-1,1].
    """
    return F.l1_loss(F.avg_pool2d(a, k), F.avg_pool2d(b, k))


def warm_start(model, ckpt_path):
    """Load a released ResShift checkpoint into the model, shape-tolerant.

    All three version configs differ from the released ones only in `sf` and the noise
    schedule (diffusion args, not weights), so EVERY tensor should match — each pairing
    in VERSIONS is a strict load. We still filter by shape and report, so a future config
    tweak that does change a shape degrades to a partial warm-start with a loud warning
    instead of a crash.
    """
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    sd = {
        k[len("module.") :] if k.startswith("module.") else k: v for k, v in sd.items()
    }
    tgt = model.state_dict()
    take = {k: v for k, v in sd.items() if k in tgt and tgt[k].shape == v.shape}
    skipped_src = [k for k in sd if k not in take]
    missing_tgt = [k for k in tgt if k not in take]
    model.load_state_dict(take, strict=False)
    print(
        f"warm-start from {Path(ckpt_path).name}: loaded {len(take)}/{len(tgt)} tensors"
    )
    if skipped_src or missing_tgt:
        print(
            f"  WARNING partial warm-start — skipped(src)={skipped_src[:6]}"
            f"{'…' if len(skipped_src) > 6 else ''} missing(tgt)={missing_tgt[:6]}"
            f"{'…' if len(missing_tgt) > 6 else ''}"
        )
    else:
        print("  (strict match — all tensors warm-started)")
    return model


@torch.no_grad()
def sample_sr(diff, model, z_y, c_y, vqgan):
    """Latent-space ResShift reverse sampling, decoded to pixels in [-1,1].

    NOT diff.p_sample_loop: that one takes y as a PIXEL LR and re-encodes it with an
    internal x{sf} bicubic upsample (encode_first_stage up_sample=True). We already
    hold z_y as a LATENT (the same z_y training uses), so feeding it to p_sample_loop
    would upsample-then-encode the latent (64->128->32) and break the residual base. Here
    we drive p_sample on latents directly — identical math, our encoding convention.

    z_y = residual-shift base (latent); c_y = the `lq` conditioning map (make_cond_lq).
    """
    z = diff.prior_sample(z_y)
    for i in reversed(range(diff.num_timesteps)):
        t = torch.full((z_y.shape[0],), i, device=z_y.device, dtype=torch.long)
        # clip_denoised=False — these are VQ latents (range ~±3.4), not bounded to [-1,1]
        z = diff.p_sample(
            model, z, z_y, t, clip_denoised=False, model_kwargs={"lq": c_y}
        )["sample"]
    return diff.decode_first_stage(z.float(), first_stage_model=vqgan).clamp(-1, 1)


def to_uint8(t):
    """[-1,1] CHW float -> HWC uint8."""
    a = ((t.clamp(-1, 1).float().cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    return np.transpose(a, (1, 2, 0))


def save_montage(path, lq, sr, gt, n=4):
    """[lq | sr | gt] rows for the first n samples, saved as one PNG."""
    rows = []
    for i in range(min(n, lq.shape[0])):
        rows.append(
            np.concatenate([to_uint8(lq[i]), to_uint8(sr[i]), to_uint8(gt[i])], axis=1)
        )
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=45000, help="optimizer steps")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument(
        "--lr_min", type=float, default=2e-5, help="cosine floor (set ==lr to disable)"
    )
    ap.add_argument("--warmup", type=int, default=500, help="linear LR warmup steps")
    ap.add_argument(
        "--lambda_lpips",
        type=float,
        default=0.5,
        help="decoded-image VGG-LPIPS on the per-step x0 prediction. Default ON "
        "(0.5) since 2026-07-06 — pure latent MSE was confirmed as the "
        "detail ceiling the students inherit. 0 = canonical ResShift "
        "latent MSE; costs a VQGAN decode per step when on.",
    )
    ap.add_argument(
        "--lambda_dc",
        type=float,
        default=1.0,
        help="low-freq DC/color L1 on the same decoded x0 (dc_loss, k=32) — "
        "pins the global tone LPIPS leaves free. Inert extra decode cost "
        "only when --lambda_lpips 0; set 0 to disable.",
    )
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="bf16 autocast on the net + NATIVE-bf16 VQGAN (matches distill_rsd; "
        "autocast alone would keep the VQGAN GroupNorms fp32 and materialize "
        "the full-res norm activations). --no-amp = full fp32.",
    )
    ap.add_argument(
        "--compile",
        action="store_true",
        help="block-compile the Swin/Res blocks at the CLASS level (glue stays "
        "eager — see rsd_models.compile_swin_blocks). Per-step it/s is ~a "
        "wash vs eager but saves VRAM; the lever for a bigger --bs without "
        "grad-ckpt. Whole-graph compile measured strictly worse — don't revive.",
    )
    ap.add_argument(
        "--grad_ckpt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Swin gradient checkpointing (bit-exact activation save; default OFF — "
        "at 256-px crops the recompute costs ~25%% throughput, and one net + "
        "one optimizer fits without it; turn on to fit bigger bs on smaller "
        "cards)",
    )
    ap.add_argument(
        "--no_grad_ckpt", dest="grad_ckpt", action="store_false", help=argparse.SUPPRESS
    )  # legacy spelling (pre-2026-07 runs)
    ap.add_argument(
        "--version",
        choices=list(VERSIONS),
        default="x2",
        help="scale/schedule family — picks config + warm-start ckpt + output "
        "dir (see VERSIONS). x2 = 2x finetune; x4 = 15-step x4 teacher "
        "finetune (feeds sr-rsd-train --version x4ft); x4s4 = 4-step v3.",
    )
    ap.add_argument(
        "--cond_lq",
        choices=["pixel", "latent"],
        default=None,
        help="what to feed the UNet's `lq` conditioning input (default: the "
        "version's — pixel for x4/x4s4, latent for the legacy x2 line). "
        "pixel = LR pixels at latent resolution, the convention the released "
        "weights were trained with; latent = the VQ latent, which is "
        "off-manifold for them. See rsd_models.make_cond_lq.",
    )
    ap.add_argument(
        "--init",
        default=None,
        help="warm-start checkpoint (default: the version's released teacher)",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="ResShift config yaml (default: the version's art config)",
    )
    ap.add_argument(
        "--src",
        default=None,
        help="HR pool dir (default: sr/data/hr_pool if present, else image_dataset)",
    )
    ap.add_argument(
        "--text_boxes",
        default=str(SR / "data" / "text_boxes.json"),
        help="CTD text-box json from sr/scripts/detect_text_boxes.py; missing "
        "file just disables text oversampling (dataset warns)",
    )
    ap.add_argument(
        "--text_crop_prob",
        type=float,
        default=0.25,
        help="fraction of samples biased to crops covering a text box — random "
        "crops almost never hit text, so without this the model learns "
        "tiny glyphs as texture and hallucinates strokes. 0 = off",
    )
    ap.add_argument(
        "--scale_jitter_prob",
        type=float,
        default=0.25,
        help="fraction of samples degraded at a random scale in "
        "[sf, scale_jitter_max] instead of exactly sf — puts sub-native "
        "text/detail sizes in-distribution WITH GT supervision. 0 = off",
    )
    ap.add_argument(
        "--scale_jitter_max",
        type=float,
        default=0.0,
        help="upper degradation scale for --scale_jitter_prob draws "
        "(0 = auto: 2·sf, i.e. 4 at x2 and 8 at x4)",
    )
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument(
        "--save_dir", default=None, help="default: output/sr/<version's subdir>"
    )
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument(
        "--sample_every",
        type=int,
        default=2500,
        help="render a p_sample_loop montage every N steps (0=off)",
    )
    ap.add_argument("--max_steps", type=int, default=0, help="smoke cap (0=off)")
    args = ap.parse_args()

    dev = M.DEVICE
    cfg_path, init_ckpt, out_sub, default_cond = VERSIONS[args.version]
    config = args.config or str(cfg_path)
    init = args.init or str(init_ckpt)
    cond_mode = args.cond_lq or default_cond
    save_dir = (
        Path(args.save_dir) if args.save_dir else M.REPO / "output" / "sr" / out_sub
    )
    (save_dir / "samples").mkdir(parents=True, exist_ok=True)
    cfg = M.load_configs(config)
    sf = cfg.diffusion.params.sf
    sf_scale = cfg.diffusion.params.scale_factor
    # 0 = auto: match the dataset's own default (2·sf) rather than the x2-era literal 4
    jitter_max = args.scale_jitter_max or 2 * sf

    # default --src to the filtered HR pool when it exists
    src = args.src
    if src is None:
        pool = SR / "data" / "hr_pool"
        src = str(pool) if pool.is_dir() else None
        print(
            f"[sr-train] --src defaulting to {src or 'image_dataset (no hr_pool — run build_hr_pool)'}"
        )

    print(f"building model ({args.version}: warm-start from {Path(init).name})...")
    model = M._build_unet(cfg, UNetModelSwin)
    warm_start(model, init)
    if args.grad_ckpt:
        n = M.enable_grad_checkpointing(model)
        print(f"  grad-checkpointing on {n} Swin layers")
    model = model.to(dev).train()
    vqgan = M.build_autoencoder(cfg, dev)
    if args.amp:
        # native bf16 (mirrors distill_rsd + infer.py): the frozen VQGAN is the full
        # pixel-res conv stack; under autocast its GroupNorms stay fp32 and dominate
        # encode VRAM/time. Also matches what inference feeds the finetuned model.
        vqgan = vqgan.to(torch.bfloat16)
    vdt = next(vqgan.parameters()).dtype
    diff = M.build_diffusion(cfg)
    T = diff.num_timesteps

    ema = make_ema(model)
    for m in (
        ema.modules()
    ):  # ema is sample-only — grad-ckpt has no benefit and warns under no_grad
        if hasattr(m, "use_checkpoint"):
            m.use_checkpoint = False

    lp = None
    if args.lambda_lpips > 0:
        import lpips

        lp = lpips.LPIPS(net="vgg").to(dev).eval()
        for p in lp.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    if args.compile:
        # Block-compile (class-level, state_dict-transparent). All shape/grad-mode
        # variants share each class's ONE code object, so the recompile budget must be
        # raised — and pinned via its ContextVar .default or it silently reverts to 8 in
        # the backward-compile context and spills to eager (same trap as distill_rsd).
        sys.path.insert(0, str(M.REPO))
        from library.runtime.dynamo import pin_dynamo_limit  # dependency-free helper

        for knob in ("recompile_limit", "cache_size_limit"):
            pin_dynamo_limit(knob, 64)
        M.compile_swin_blocks()

    loader = DataLoader(
        ArtSRDataset(
            src=src,
            gt_size=256,
            scale=sf,
            length=args.iters * args.bs * args.grad_accum + 1000,
            text_boxes=args.text_boxes,
            text_crop_prob=args.text_crop_prob,
            scale_jitter_prob=args.scale_jitter_prob,
            scale_jitter_max=jitter_max,
        ),
        batch_size=args.bs,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    it = iter(loader)

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

    def set_lr(step):
        if step < args.warmup:
            f = (step + 1) / args.warmup
        elif args.lr_min >= args.lr:
            f = 1.0
        else:
            prog = (step - args.warmup) / max(1, args.iters - args.warmup)
            cos = 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))
            f = (args.lr_min + (args.lr - args.lr_min) * cos) / args.lr
        for g in opt.param_groups:
            g["lr"] = args.lr * f
        return args.lr * f

    log_path = save_dir / "progress.jsonl"
    snap = {
        "method": f"resshift_{args.version}",
        "version": args.version,
        "config": config,
        "sf": sf,
        "init": init,
        "cond_lq": cond_mode,
        "iters": args.iters,
        "bs": args.bs,
        "lr": args.lr,
        "lambda_lpips": args.lambda_lpips,
        "lambda_dc": args.lambda_dc,
        "amp": args.amp,
        "compile": args.compile,
        "grad_ckpt": args.grad_ckpt,
        "text_boxes": args.text_boxes,
        "text_crop_prob": args.text_crop_prob,
        "scale_jitter_prob": args.scale_jitter_prob,
        "scale_jitter_max": jitter_max,
    }
    (save_dir / "snapshot.toml").write_text(
        "\n".join(f"{k} = {json.dumps(v)}" for k, v in snap.items()) + "\n"
    )

    t0 = time.time()
    print(
        f"training: version={args.version} cond_lq={cond_mode} iters={args.iters} "
        f"bs={args.bs}x{args.grad_accum} sf={sf} T={T} amp={args.amp} compile={args.compile} grad_ckpt={args.grad_ckpt} "
        f"lpips={args.lambda_lpips} dc={args.lambda_dc} src={src}"
    )

    for step in range(args.iters):
        lr_now = set_lr(step)
        opt.zero_grad(set_to_none=True)
        logs = {}
        for _ in range(args.grad_accum):
            gt, lq = next_batch()
            B = gt.shape[0]
            z0, z_y = encode(gt, lq)
            c_y = M.make_cond_lq(
                lq, z_y, cond_mode
            )  # `lq` conditioning != residual base
            t = torch.randint(0, T, (B,), device=dev)
            noise = torch.randn_like(z0)
            z_t = diff.q_sample(z0, z_y, t, noise=noise)
            with autocast():
                pred = predict_x0(diff, model, z_t, c_y, t)  # xstart -> pred IS z0
                L_mse = F.mse_loss(pred.float(), z0.float())
            L = L_mse
            if lp is not None or args.lambda_dc > 0:
                # decode runs native bf16 outside autocast (grad flows through the frozen
                # decoder; matches distill_rsd), image back to fp32 for the losses
                img = (
                    vqgan.decode(pred.to(vdt), force_not_quantize=True)
                    .float()
                    .clamp(-1, 1)
                )
                # reference the roundtrip-consistent target decode(z0), NOT raw gt. The frozen
                # VQ-f4 roundtrip is color-tinted on this art data (measured ~[+2.5 R, +0.9 B]
                # u8 on content crops), so comparing decode(pred) against raw gt puts the
                # LPIPS/dc color-minimum at the *anti*-tint while the latent MSE(pred, z0) zero
                # sits at the tint — the two references disagree by exactly the roundtrip color,
                # and the model bakes in the anti-tint (measured ~[-1.8 R, -1.1 B] green-cast
                # drift, compounded through the 15-step rollout). decode(z0) coincides with the
                # MSE zero at pred==z0, so there is no color tug-of-war and the perceptual/detail
                # signal is untouched. Kept as a separate no_grad decode (not fused into the pred
                # decode) so no decoder activations are stored for it.
                with torch.no_grad():
                    gt_img = (
                        vqgan.decode(z0.to(vdt), force_not_quantize=True)
                        .float()
                        .clamp(-1, 1)
                    )
                if lp is not None:
                    with autocast():
                        L_lpips = lp(img, gt_img).mean()
                    L = L + args.lambda_lpips * L_lpips
                    logs["L_lpips"] = L_lpips.detach().item()
                if args.lambda_dc > 0:
                    L_dc = dc_loss(img, gt_img)
                    L = L + args.lambda_dc * L_dc
                    logs["L_dc"] = L_dc.detach().item()
            (L / args.grad_accum).backward()
            logs["L_mse"] = L_mse.detach().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema, model, args.ema)

        if step % args.log_every == 0:
            rate = (step + 1) / (time.time() - t0)
            line = {
                "step": step,
                **{k: round(v, 5) for k, v in logs.items()},
                "lr": round(lr_now, 7),
                "it_s": round(rate, 3),
            }
            print(line)
            with open(log_path, "a") as f:
                f.write(json.dumps(line) + "\n")

        if args.sample_every and step % args.sample_every == 0:
            gt, lq = next_batch()
            _, z_y = encode(gt, lq)
            sr = sample_sr(diff, ema, z_y, M.make_cond_lq(lq, z_y, cond_mode), vqgan)
            save_montage(save_dir / "samples" / f"step_{step:06d}.png", lq, sr, gt)

        if step > 0 and step % args.save_every == 0:
            torch.save(
                {"ema": ema.state_dict(), "model": model.state_dict(), "step": step},
                save_dir / f"resshift_{args.version}_{step}.pth",
            )
        if args.max_steps and step + 1 >= args.max_steps:
            print(f"max_steps {args.max_steps} hit — stopping (smoke).")
            break

    torch.save(
        {"ema": ema.state_dict(), "model": model.state_dict(), "step": args.iters},
        save_dir / f"resshift_{args.version}_final.pth",
    )
    print(f"done. saved to {save_dir}")


if __name__ == "__main__":
    main()
