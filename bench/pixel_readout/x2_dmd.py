"""RSD-style critic for the ×2 head: a DMD term and a GAN term in the teacher's VQ-f4 latent.

The SR line distilled its 1-step ResShift student with a distribution-matching loss between
the frozen multi-step teacher and a fake critic trained on the student's outputs, plus a GAN
head on the critic's bottleneck (``project/finished/sr/distill_rsd/``, DESIGN.md Algorithm 1).
Here the generator is the ×2 pixel head: a crop of its ×2 output is encoded by the VQ-f4
encoder (the gradient reaches the head through it), noised on ResShift's residual-shift
forward and read by both nets, conditioned on the VQ latent of the bicubic ×2 of VAE(x̂0)
(the ×2 line's ``latent`` conditioning).

The models come from the ComfyUI node's vendored ResShift (``x2_head.load_teacher``), the
multi-step ×2 teacher from ``project/finished/sr/weights/resshift_x2_final.pth``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

X2_TEACHER = "project/finished/sr/weights/resshift_x2_final.pth"


class DiscHead(nn.Module):
    """The SR line's GAN head on the critic's bottleneck (B, 640, h, w) → (B,)."""

    def __init__(self, in_ch: int = 640):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 256, 4, 2, 1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, 1, 2, 1, 0),
        )

    def forward(self, feats):
        return self.net(feats).flatten(1).mean(1)


class X2Critic:
    """Frozen teacher, trainable fake critic + GAN head, and a fp32 VQ-f4 encoder.

    ``gen_terms`` returns the generator's DMD and GAN losses for one crop and keeps the
    detached latents; ``step`` then makes one critic update on them (``update`` makes one
    from fresh crops). ``use_dmd`` / ``use_gan`` drop a term from the generator's graph
    (it comes back as None); without the GAN term the discriminator is not trained.
    ``R`` is the node's ``resshift_infer`` module."""

    def __init__(
        self,
        R,
        cfg,
        diff,
        teacher_ckpt: Path,
        device,
        lr: float = 5e-5,
        lambda_gan_d: float = 3e-3,
        seed: int = 0,
        use_dmd: bool = True,
        use_gan: bool = True,
    ):
        self.use_dmd, self.use_gan = use_dmd, use_gan
        self.R, self.diff, self.device = R, diff, device
        self.T = int(diff.num_timesteps)
        self.sf = float(cfg.diffusion.params.scale_factor)
        self.lambda_gan_d = lambda_gan_d
        raw = torch.load(str(teacher_ckpt), map_location="cpu", weights_only=False)
        sd = R._strip_module(raw["ema"] if "ema" in raw else raw)
        params = R._unet_params(cfg)
        self.teacher = R.UNetModelSwin(**params)
        self.teacher.load_state_dict(sd, strict=True)
        self.teacher.to(device).eval().requires_grad_(False)
        self.fake = R.UNetModelSwin(**params)
        self.fake.load_state_dict(sd, strict=True)
        self.fake.to(device).train()
        self.disc = DiscHead().to(device)
        self.vq = R.build_autoencoder(cfg, device, dtype=torch.float32)
        self._params = list(self.fake.parameters()) + list(self.disc.parameters())
        self.opt = torch.optim.AdamW(
            self._params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0
        )
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.last: tuple | None = None

    # ------------------------------------------------------------------ pieces

    def enc(self, img: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) in [-1, 1], sides multiples of 256 → VQ-f4 latent."""
        return self.vq.encode(img.float()) * self.sf

    def _noised(self, z0: torch.Tensor, z_y: torch.Tensor):
        b = z0.shape[0]
        t = torch.randint(0, self.T, (b,), device=self.device, generator=self.gen)
        noise = torch.randn(z0.shape, device=self.device, generator=self.gen)
        return self.diff.q_sample(z0, z_y, t, noise=noise), t

    def _x0(self, model, z_t, c_y, t):
        return model(self.diff._scale_input(z_t, t), t, lq=c_y)

    def _feats(self, z: torch.Tensor, c_y: torch.Tensor) -> torch.Tensor:
        """The fake critic's encoder half at t = 0: input blocks + the middle ResBlock."""
        from resshift.models.unet import timestep_embedding

        m = self.fake
        t = torch.zeros(z.shape[0], device=z.device, dtype=torch.long)
        emb = m.time_embed(timestep_embedding(t, m.model_channels)).type(m.dtype)
        h = torch.cat([z, m.feature_extractor(c_y.type(m.dtype))], dim=1).type(m.dtype)
        for module in m.input_blocks:
            h = module(h, emb)
        return m.middle_block[0](h, emb)

    def _trainable(self, on: bool):
        for q in self._params:
            q.requires_grad_(on)

    # ------------------------------------------------------------------ generator

    def gen_terms(self, x: torch.Tensor, lq_up: torch.Tensor, real: torch.Tensor):
        """``x`` the head's ×2 crop (1, 3, c, c) with its graph, ``lq_up`` the bicubic ×2
        of VAE(x̂0) and ``real`` the teacher's ×2 at the same crop → (DMD, GAN) losses."""
        with torch.no_grad():
            z_y, z_real = self.enc(torch.cat([lq_up, real])).chunk(2)
        z0 = self.enc(x)
        dmd = gan = None
        self._trainable(False)
        if self.use_dmd:
            z_t, t = self._noised(z0.detach(), z_y)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                x0_t = self._x0(self.teacher, z_t, z_y, t).float()
                x0_f = self._x0(self.fake, z_t, z_y, t).float()
            # the official RSD normalisation (SiD form): |teacher x0 − generator x0|
            denom = (x0_t - z0.detach()).abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-8
            g = (x0_f - x0_t) / denom
            dmd = 0.5 * F.mse_loss(z0, (z0 - g).detach())
        if self.use_gan:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gan = F.softplus(-self.disc(self._feats(z0, z_y)).float()).mean()
        self._trainable(True)
        self.last = (z0.detach(), z_y, z_real)
        return dmd, gan

    # ------------------------------------------------------------------ critic

    def step(self) -> dict[str, float]:
        """One critic update on the latents the last ``gen_terms`` call kept."""
        z0, z_y, z_real = self.last
        self.opt.zero_grad(set_to_none=True)
        z_t, t = self._noised(z0, z_y)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l_fake = F.mse_loss(self._x0(self.fake, z_t, z_y, t).float(), z0)
        if not self.use_gan:
            l_fake.backward()
            self.opt.step()
            return {"c_fake": l_fake.item()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d_real, d_fake = (
                self.disc(self._feats(torch.cat([z_real, z0]), z_y.repeat(2, 1, 1, 1)))
                .float()
                .chunk(2)
            )
            l_d = (F.softplus(-d_real) + F.softplus(d_fake)).mean()
        (l_fake + self.lambda_gan_d * l_d).backward()
        self.opt.step()
        return {
            "c_fake": l_fake.item(),
            "c_gan_d": l_d.item(),
            "d_real": d_real.mean().item(),
            "d_fake": d_fake.mean().item(),
        }

    @torch.no_grad()
    def _stash(self, x, lq_up, real):
        z0, z_y, z_real = self.enc(torch.cat([x, lq_up, real])).chunk(3)
        self.last = (z0, z_y, z_real)

    def update(self, x, lq_up, real) -> dict[str, float]:
        """One critic update from fresh crops (``x`` detached)."""
        self._stash(x, lq_up, real)
        return self.step()
