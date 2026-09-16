"""RSD feasibility dry-run: build teacher/student/fake/disc/VQGAN/LPIPS, run one
real fake-update and one generator-update backward (+optimizer step) and report peak
VRAM. Answers 'does the RSD loop fit on 16 GB' before writing the full training loop.

Uses random tensors (shapes only) — not real data. Run in the root venv.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsd_models as M  # noqa: E402
from rsd_models import make_eps, predict_x0  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--version", choices=["x4", "x2"], default="x4")
    args = ap.parse_args()
    dev = M.DEVICE
    torch.cuda.reset_peak_memory_stats()

    config_path, teacher_ckpt = M.resolve_version(args.version)
    cfg = M.load_configs(config_path)
    tck = str(teacher_ckpt)
    print("building nets...")
    teacher = M.build_teacher(cfg, tck, dev)
    student = M.build_generator(cfg, tck, dev)
    fake = M.build_generator(cfg, tck, dev)
    disc = M.DiscHead().to(dev)
    vqgan = M.build_autoencoder(cfg, dev)
    diff = M.build_diffusion(cfg)
    T = diff.num_timesteps
    print(
        f"  T={T} kappa={diff.kappa} params/net={sum(p.numel() for p in student.parameters()) / 1e6:.0f}M"
    )

    import lpips

    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)

    opt_g = torch.optim.AdamW(student.parameters(), lr=5e-5, betas=(0.9, 0.95))
    opt_f = torch.optim.AdamW(
        list(fake.parameters()) + list(disc.parameters()), lr=5e-5, betas=(0.9, 0.95)
    )

    B = args.bs
    N_NODES = [4, 8, 12, 14]  # provisional N=4 multistep set (0-indexed t in [0,T-1])
    # NOTE: --amp is a placeholder; real bf16 autocast is wired in train.py, not here.

    def sample_latents():
        gt = torch.rand(B, 3, 256, 256, device=dev) * 2 - 1
        lq = torch.rand(B, 3, 256, 256, device=dev) * 2 - 1
        z0 = vqgan.encode(gt) * cfg.diffusion.params.scale_factor
        z_y = vqgan.encode(lq) * cfg.diffusion.params.scale_factor
        # c_y = the `lq` conditioning map (pixel convention); z_y stays the residual base
        return z0, z_y, M.make_cond_lq(lq, z_y, "pixel")

    def report(tag):
        torch.cuda.synchronize()
        print(
            f"  [{tag}] peak alloc {torch.cuda.max_memory_allocated() / 1e9:.2f} GB | "
            f"reserved {torch.cuda.max_memory_reserved() / 1e9:.2f} GB"
        )

    # ---- one FAKE-critic update ----
    print("fake-critic update...")
    z0, z_y, c_y = sample_latents()
    t_n = torch.tensor([N_NODES[0]] * B, device=dev).long()
    eps = make_eps(student, z0)
    with torch.no_grad():
        z_tn = diff.q_sample(z0, z_y, t_n)
        z0_hat = predict_x0(
            diff, student, z_tn, c_y, t_n, eps
        )  # student output (frozen here)
    t = torch.randint(0, T, (B,), device=dev).long()
    z_t = diff.q_sample(z0_hat, z_y, t)
    x0_fake = predict_x0(diff, fake, z_t, c_y, t, eps)
    L_fake = F.mse_loss(x0_fake, z0_hat)
    # GAN-D: real=z0, fake=z0_hat features off fake critic bottleneck
    d_real = disc(fake.encode_features(z0, c_y, eps=eps))
    d_fake = disc(fake.encode_features(z0_hat.detach(), c_y, eps=eps))
    L_gan_d = (F.softplus(-d_real) + F.softplus(d_fake)).mean()
    (L_fake + 3e-3 * L_gan_d).backward()
    opt_f.step()
    opt_f.zero_grad(set_to_none=True)
    report("after fake update")

    # ---- one GENERATOR update ----
    print("generator update...")
    z0, z_y, c_y = sample_latents()
    t_n = torch.tensor([N_NODES[0]] * B, device=dev).long()
    eps = make_eps(student, z0)
    z_tn = diff.q_sample(z0, z_y, t_n)
    z0_hat = predict_x0(diff, student, z_tn, c_y, t_n, eps)  # grad
    t = torch.randint(0, T, (B,), device=dev).long()
    z_t = diff.q_sample(z0_hat.detach(), z_y, t)
    with torch.no_grad():  # DMD: teacher & fake no-grad
        x0_teacher = predict_x0(diff, teacher, z_t, c_y, t)
        x0_fakep = predict_x0(diff, fake, z_t, c_y, t, eps)
        grad = x0_fakep - x0_teacher
        grad = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8)
    L_theta = 0.5 * F.mse_loss(z0_hat, (z0_hat - grad).detach())
    # LPIPS single-step path from t=T-1
    z_TT = z_y + diff.kappa * torch.randn_like(z_y)
    tT = torch.tensor([T - 1] * B, device=dev).long()
    z0_single = predict_x0(diff, student, z_TT, c_y, tT, make_eps(student, z_y))
    x0_img = vqgan.decode(z0_single, force_not_quantize=True).clamp(-1, 1)
    gt_img = torch.rand(B, 3, 256, 256, device=dev) * 2 - 1
    L_lpips = lp(x0_img, gt_img).mean()
    # GAN-G through fake bottleneck
    L_gan_g = F.softplus(-disc(fake.encode_features(z0_hat, c_y, eps=eps))).mean()
    (L_theta + 2.0 * L_lpips + 3e-3 * L_gan_g).backward()
    opt_g.step()
    opt_g.zero_grad(set_to_none=True)
    report("after generator update")

    print(
        f"\n=== RSD dry-run OK at bs={B}{' (bf16 amp)' if args.amp else ' (fp32)'} ==="
    )
    report("FINAL PEAK")


if __name__ == "__main__":
    main()
