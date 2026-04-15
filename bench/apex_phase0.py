#!/usr/bin/env python
"""APEX Phase 0 sanity check on a 2D Gaussian mixture toy.

Standalone (no Anima deps). Runs three tests end to end:

    T1  ConditionShift differentiates model predictions under real vs. fake c
        once the network is past random init (meaningful adversarial branch).

    T2  Per-sample gradient equivalence (Theorem 1, Eq. 63):
            grad_theta L_mix (Eq. 24) == grad_theta G_APEX (Eq. 22)
        Loss *values* differ by a theta-independent constant; gradients must not.

    T3  Cold-start APEX training reduces NFE=1 sliced-Wasserstein distance to
        the data distribution vs. pure flow-matching, when both are trained for
        the same number of optimizer steps on the same data.

Usage:
    python bench/apex_phase0.py
    python bench/apex_phase0.py --steps 2000 --device cuda
    python bench/apex_phase0.py --lam 0.5 --a -1.0 --b 0.5 --plot
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def build_gmm(n_modes: int, radius: float, std: float, device: str):
    angles = torch.linspace(0, 2 * math.pi, n_modes + 1, device=device)[:-1]
    centers = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1)
    return centers, std


def sample_gmm(centers: torch.Tensor, std: float, n: int):
    n_modes = centers.size(0)
    classes = torch.randint(0, n_modes, (n,), device=centers.device)
    mu = centers[classes]
    x = mu + std * torch.randn(n, 2, device=centers.device)
    return x, classes


class TinyDiT(nn.Module):
    """Minimal velocity-field network F_theta(x_t, t, c) -> R^2.

    Condition is a per-class embedding to stand in for post-T5 text features.
    """

    def __init__(self, n_classes: int = 8, cond_dim: int = 16, hidden: int = 128):
        super().__init__()
        self.n_classes = n_classes
        self.cond_dim = cond_dim
        self.cond_embed = nn.Embedding(n_classes, cond_dim)
        self.x_proj = nn.Linear(2, hidden)
        self.t_proj = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.c_proj = nn.Linear(cond_dim, hidden)
        self.trunk = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def encode(self, classes: torch.Tensor) -> torch.Tensor:
        return self.cond_embed(classes)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.x_proj(x_t) + self.t_proj(t.view(-1, 1)) + self.c_proj(c)
        return self.trunk(h)


class ConditionShift(nn.Module):
    """APEX condition-space affine: c_fake = a*c + b (scalar variant, 2 params)."""

    def __init__(self, init_a: float = -1.0, init_b: float = 0.5):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(float(init_a)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.a * c + self.b


def sample_t(n: int, device: str) -> torch.Tensor:
    return torch.rand(n, device=device)


def ot_interp(x: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """OT path: x_t = t*z + (1-t)*x  (Eq. 1)."""
    return t.view(-1, 1) * z + (1 - t.view(-1, 1)) * x


def v_target_fn(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """v_data = z - x under OT path."""
    return z - x


def fm_step(model: TinyDiT, x: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    cond = model.encode(classes)
    z = torch.randn_like(x)
    t = sample_t(x.size(0), x.device)
    x_t = ot_interp(x, z, t)
    v_pred = model(x_t, t, cond)
    return ((v_pred - v_target_fn(x, z)) ** 2).mean()


def apex_step(
    model: TinyDiT,
    shift: ConditionShift,
    x: torch.Tensor,
    classes: torch.Tensor,
    lam: float,
    lam_p: float,
    lam_c: float,
    lam_f: float,
) -> tuple[torch.Tensor, dict]:
    """One APEX training step (velocity-space L_mix form).

    Three forwards per step:
      (1) real branch at (x_t, t, c)       -> F_real      for L_sup + L_mix
      (2) fake branch at (x_t, t, c_fake)  -> v_fake (sg) target for L_mix
      (3) fake branch at (x_fake_t, t_f, c_fake) -> L_fake  (trains shift + model)
    """
    cond = model.encode(classes)
    bsz = x.size(0)

    # Real branch forward (differentiable)
    z = torch.randn_like(x)
    t = sample_t(bsz, x.device)
    x_t = ot_interp(x, z, t)
    v_real_target = v_target_fn(x, z)
    F_real = model(x_t, t, cond)

    # Build x_fake from real branch with stop gradient (Eq. 11)
    with torch.no_grad():
        x_fake = x_t - t.view(-1, 1) * F_real

    # Fake condition (gradients flow into shift via L_fake below, not via v_fake_sg)
    c_fake = shift(cond)

    # Fake-branch query at real (x_t, t) with SG — target for L_mix (Eq. 24)
    with torch.no_grad():
        v_fake_sg = model(x_t, t, c_fake.detach())

    # L_sup (velocity space, Eq. 20 with t^2 absorbed into unit weight)
    L_sup = ((F_real - v_real_target) ** 2).mean()

    # L_mix in velocity space.
    # Endpoint form:  f^x(F) - T_mix where T_mix = (1-lam)x + lam*f^x(v_fake)
    # Velocity form (after t^2 factoring via Prop. 3):
    #   ||F - ((1-lam)*v_data + lam*v_fake)||^2
    T_mix_v = (1 - lam) * v_real_target + lam * v_fake_sg
    L_mix = ((F_real - T_mix_v) ** 2).mean()

    # L_fake (Eq. 12): train fake branch to fit its own OT trajectory from x_fake
    z_f = torch.randn_like(x_fake)
    t_f = sample_t(bsz, x.device)
    x_fake_t = ot_interp(x_fake, z_f, t_f)
    F_fake_on_traj = model(x_fake_t, t_f, c_fake)
    L_fake = ((F_fake_on_traj - v_target_fn(x_fake, z_f)) ** 2).mean()

    total = lam_p * L_sup + lam_c * L_mix + lam_f * L_fake
    return total, {
        "L_sup": L_sup.item(),
        "L_mix": L_mix.item(),
        "L_fake": L_fake.item(),
        "total": total.item(),
    }


@torch.no_grad()
def sample_euler(model: TinyDiT, classes: torch.Tensor, nfe: int) -> torch.Tensor:
    """Euler integration t=1 -> t=0 under OT path. dt = -1/nfe per step."""
    device = classes.device
    n = classes.size(0)
    cond = model.encode(classes)
    x = torch.randn(n, 2, device=device)
    ts = torch.linspace(1.0, 0.0, nfe + 1, device=device)
    for i in range(nfe):
        t_cur = ts[i].expand(n)
        dt = ts[i + 1] - ts[i]  # negative
        v = model(x, t_cur, cond)
        x = x + dt * v
    return x


def sliced_wasserstein(p: torch.Tensor, q: torch.Tensor, n_proj: int = 128) -> float:
    """1D-sliced W1 over 2D samples. Same-size inputs required."""
    assert p.shape == q.shape, f"{p.shape} vs {q.shape}"
    device = p.device
    theta = torch.randn(n_proj, 2, device=device)
    theta = theta / theta.norm(dim=1, keepdim=True)
    pp = (p @ theta.T).sort(dim=0).values
    qq = (q @ theta.T).sort(dim=0).values
    return (pp - qq).abs().mean().item()


@dataclass
class TrainResult:
    model: TinyDiT
    final_loss: float


def _fm_loop(model, opt, centers, std, steps, bsz):
    last = 0.0
    for _ in range(steps):
        x, classes = sample_gmm(centers, std, bsz)
        loss = fm_step(model, x, classes)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = loss.item()
    return last


def _apex_loop(model, shift, opt, centers, std, steps, bsz, lam, lam_p, lam_c, lam_f):
    last = 0.0
    for _ in range(steps):
        x, classes = sample_gmm(centers, std, bsz)
        loss, _ = apex_step(model, shift, x, classes, lam, lam_p, lam_c, lam_f)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = loss.item()
    return last


def train_warm_fm(centers, std, warm_steps, bsz, lr, seed, device):
    """Warm-start the velocity field with pure flow matching."""
    torch.manual_seed(seed)
    model = TinyDiT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = _fm_loop(model, opt, centers, std, warm_steps, bsz)
    return model, last


def clone_model(src: TinyDiT) -> TinyDiT:
    dst = TinyDiT(n_classes=src.n_classes, cond_dim=src.cond_dim).to(next(src.parameters()).device)
    dst.load_state_dict(src.state_dict())
    return dst


def refine_fm(model, centers, std, refine_steps, bsz, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = _fm_loop(model, opt, centers, std, refine_steps, bsz)
    return TrainResult(model=model, final_loss=last)


def refine_apex(model, centers, std, refine_steps, bsz, lr, lam, lam_p, lam_c, lam_f, init_a, init_b):
    shift = ConditionShift(init_a=init_a, init_b=init_b).to(next(model.parameters()).device)
    opt = torch.optim.Adam(list(model.parameters()) + list(shift.parameters()), lr=lr)
    last = _apex_loop(model, shift, opt, centers, std, refine_steps, bsz, lam, lam_p, lam_c, lam_f)
    return TrainResult(model=model, final_loss=last), shift


# -------------------- tests --------------------

def test_condition_shift_differentiates(
    model: TinyDiT, shift: ConditionShift, centers: torch.Tensor, std: float, bsz: int = 1024
) -> dict:
    """T1: after training, F(c) and F(c_fake) should differ meaningfully.

    We evaluate at real (x_t, t) pairs drawn from the GMM OT path, not at
    arbitrary zero-centered inputs — the fake branch is only trained to
    disagree with the real branch on the data manifold.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x, classes = sample_gmm(centers, std, bsz)
        c = model.encode(classes)
        c_fake = shift(c)
        z = torch.randn_like(x)
        t = sample_t(bsz, device)
        x_t = ot_interp(x, z, t)
        F_real = model(x_t, t, c)
        F_fake = model(x_t, t, c_fake)
        diff = (F_real - F_fake).norm(dim=1)
        base = F_real.norm(dim=1).clamp(min=1e-6)
        rel = (diff / base).mean().item()
    # Threshold 0.05: anything above 5% of the real prediction magnitude is a
    # clear, non-noise divergence between branches on a 2D toy.
    return {"mean_rel_shift": rel, "pass": rel > 0.05}


def test_gradient_equivalence(seed: int = 0, bsz: int = 64, lam: float = 0.5) -> dict:
    """T2: per-sample grad_theta L_mix == grad_theta G_APEX (Theorem 1).

    Loss *values* differ by a theta-independent constant (expected and verified);
    gradients on the model parameters must match to within fp32 numerical noise.
    """
    torch.manual_seed(seed)
    device = "cpu"
    model = TinyDiT().to(device)
    shift = ConditionShift().to(device)

    # Warm the network a few steps so F is not at init-degenerate
    centers, std = build_gmm(8, 4.0, 0.3, device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(100):
        x, classes = sample_gmm(centers, std, 256)
        loss = fm_step(model, x, classes)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Shared (x_t, t, c, v_fake_sg) batch for both formulations.
    # Detach everything that is not the model parameters so the two backward
    # passes don't share any autograd graph state.
    x, classes = sample_gmm(centers, std, bsz)
    with torch.no_grad():
        cond = model.encode(classes)
        z = torch.randn_like(x)
        t = sample_t(bsz, device)
        x_t = ot_interp(x, z, t)
        v_data = v_target_fn(x, z)
        c_fake = shift(cond)
        v_fake_sg = model(x_t, t, c_fake)
    cond = cond.detach()
    x_t = x_t.detach()
    t = t.detach()
    v_data = v_data.detach()
    v_fake_sg = v_fake_sg.detach()

    # Only compare gradients for params actually reached by the forward at
    # (x_t, t, cond_detached). The Embedding layer isn't reached when `cond`
    # is a detached leaf — exclude it from the comparison.
    params = [p for n, p in model.named_parameters() if not n.startswith("cond_embed")]

    # --- Formulation A: L_mix (velocity-space, Eq. 24 after t^2 cancellation)
    model.zero_grad(set_to_none=True)
    F_real_a = model(x_t, t, cond)
    T_mix_v = (1 - lam) * v_data + lam * v_fake_sg
    L_mix = ((F_real_a - T_mix_v) ** 2).mean()
    grads_mix = torch.autograd.grad(L_mix, params, retain_graph=False)

    # --- Formulation B: G_APEX = (1-lam)*L_sup + lam*L_cons (Eq. 22, velocity-space)
    model.zero_grad(set_to_none=True)
    F_real_b = model(x_t, t, cond)
    L_sup = ((F_real_b - v_data) ** 2).mean()
    L_cons = ((F_real_b - v_fake_sg) ** 2).mean()
    G_APEX = (1 - lam) * L_sup + lam * L_cons
    grads_g = torch.autograd.grad(G_APEX, params, retain_graph=False)

    # Compare element-wise
    max_abs = 0.0
    max_rel = 0.0
    total_cos = 0.0
    n_params = 0
    for gm, gg in zip(grads_mix, grads_g):
        abs_err = (gm - gg).abs().max().item()
        denom = gg.abs().max().clamp(min=1e-8).item()
        rel_err = abs_err / denom
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)
        # cosine on the flattened param slab
        fm = gm.flatten()
        fg = gg.flatten()
        cos = torch.dot(fm, fg) / (fm.norm() * fg.norm()).clamp(min=1e-12)
        total_cos += cos.item()
        n_params += 1

    loss_value_diff = (L_mix.item() - G_APEX.item())
    return {
        "L_mix": L_mix.item(),
        "G_APEX": G_APEX.item(),
        "loss_value_diff": loss_value_diff,  # expected non-zero (constant in theta)
        "max_abs_grad_err": max_abs,
        "max_rel_grad_err": max_rel,
        "mean_grad_cosine": total_cos / n_params,
        "pass": (max_rel < 1e-4) and (total_cos / n_params > 1 - 1e-6),
    }


def test_training_improves_nfe1(
    centers: torch.Tensor,
    std: float,
    warm_steps: int,
    refine_steps: int,
    bsz: int,
    lr: float,
    device: str,
    lam: float,
    lam_p: float,
    lam_c: float,
    lam_f: float,
    init_a: float,
    init_b: float,
    eval_n: int = 4096,
) -> dict:
    """T3: warm-start then branch.

    Protocol matches the paper's actual use case:
      1. Warm-start with pure FM for `warm_steps` (stand-in for pretrained base).
      2. Clone into two copies.
      3. Continue one with FM for `refine_steps` (fair baseline: same total budget).
      4. Continue the other with APEX for `refine_steps` (distillation phase).
      5. Compare NFE=1 sliced-Wasserstein.
    """
    base, warm_loss = train_warm_fm(centers, std, warm_steps, bsz, lr, seed=1234, device=device)
    fm_cont = clone_model(base)
    apex_init = clone_model(base)

    fm_res = refine_fm(fm_cont, centers, std, refine_steps, bsz, lr)
    apex_res, shift = refine_apex(
        apex_init, centers, std, refine_steps, bsz, lr,
        lam=lam, lam_p=lam_p, lam_c=lam_c, lam_f=lam_f,
        init_a=init_a, init_b=init_b,
    )

    # Evaluation batch (matched classes across both models)
    torch.manual_seed(9999)
    eval_classes = torch.randint(0, 8, (eval_n,), device=device)
    data, _ = sample_gmm(centers, std, eval_n)

    def eval_w(model: TinyDiT):
        w1 = sliced_wasserstein(sample_euler(model, eval_classes, nfe=1), data)
        w2 = sliced_wasserstein(sample_euler(model, eval_classes, nfe=4), data)
        w20 = sliced_wasserstein(sample_euler(model, eval_classes, nfe=20), data)
        return w1, w2, w20

    fm_w1, fm_w4, fm_w20 = eval_w(fm_res.model)
    apex_w1, apex_w4, apex_w20 = eval_w(apex_res.model)

    return {
        "warm_loss": warm_loss,
        "fm_final_loss": fm_res.final_loss,
        "apex_final_loss": apex_res.final_loss,
        "fm_W1_nfe1": fm_w1,
        "apex_W1_nfe1": apex_w1,
        "fm_W1_nfe4": fm_w4,
        "apex_W1_nfe4": apex_w4,
        "fm_W1_nfe20": fm_w20,
        "apex_W1_nfe20": apex_w20,
        "nfe1_improvement_pct": 100.0 * (fm_w1 - apex_w1) / max(fm_w1, 1e-8),
        "nfe20_gap_pct": 100.0 * (fm_w20 - apex_w20) / max(fm_w20, 1e-8),
        "learned_shift_a": shift.a.item(),
        "learned_shift_b": shift.b.item(),
        "pass": apex_w1 < fm_w1,  # APEX must beat FM at NFE=1
    }, (fm_res.model, apex_res.model, shift)


def maybe_plot(centers, std, fm_model, apex_model, device, out_path):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print(f"  [plot skipped — matplotlib not installed]")
        return
    n = 2048
    torch.manual_seed(77)
    eval_classes = torch.randint(0, 8, (n,), device=device)
    data, _ = sample_gmm(centers, std, n)
    fm1 = sample_euler(fm_model, eval_classes, nfe=1).cpu()
    ap1 = sample_euler(apex_model, eval_classes, nfe=1).cpu()
    fm20 = sample_euler(fm_model, eval_classes, nfe=20).cpu()
    ap20 = sample_euler(apex_model, eval_classes, nfe=20).cpu()
    data_cpu = data.cpu()

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax in axes.flat:
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect("equal")
    axes[0, 0].scatter(data_cpu[:, 0], data_cpu[:, 1], s=2, alpha=0.4)
    axes[0, 0].set_title("data")
    axes[0, 1].scatter(fm1[:, 0], fm1[:, 1], s=2, alpha=0.4)
    axes[0, 1].set_title("FM @ NFE=1")
    axes[0, 2].scatter(fm20[:, 0], fm20[:, 1], s=2, alpha=0.4)
    axes[0, 2].set_title("FM @ NFE=20")
    axes[1, 0].scatter(data_cpu[:, 0], data_cpu[:, 1], s=2, alpha=0.4)
    axes[1, 0].set_title("data")
    axes[1, 1].scatter(ap1[:, 0], ap1[:, 1], s=2, alpha=0.4)
    axes[1, 1].set_title("APEX @ NFE=1")
    axes[1, 2].scatter(ap20[:, 0], ap20[:, 1], s=2, alpha=0.4)
    axes[1, 2].set_title("APEX @ NFE=20")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"  saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--warm_steps", type=int, default=2000, help="pure-FM warm-start before branching")
    p.add_argument("--refine_steps", type=int, default=2000, help="APEX vs FM refine phase (equal budget)")
    p.add_argument("--bsz", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--lam", type=float, default=1.0, help="mixing coefficient in T_mix (Eq. 23)")
    p.add_argument("--lam_p", type=float, default=1.0, help="weight on L_sup (Eq. 25)")
    p.add_argument("--lam_c", type=float, default=1.0, help="weight on L_mix (Eq. 25)")
    p.add_argument("--lam_f", type=float, default=1.0, help="weight on L_fake (Eq. 12)")
    p.add_argument("--a", type=float, default=-1.0, help="ConditionShift init_a (Table 7 optimum ~ -1)")
    p.add_argument("--b", type=float, default=0.5, help="ConditionShift init_b (Table 7 range [0.1, 1.0])")
    p.add_argument("--plot", action="store_true", help="save sample scatter plot")
    p.add_argument("--plot_path", type=str, default="bench/results/apex_phase0.png")
    args = p.parse_args()

    device = args.device
    centers, std = build_gmm(8, 4.0, 0.3, device)

    def hrule():
        print("=" * 72)

    hrule()
    print(
        f"APEX Phase 0 sanity check  "
        f"(warm={args.warm_steps} + refine={args.refine_steps}, bsz={args.bsz}, device={device})"
    )
    hrule()

    print("\n[T2] Gradient equivalence: grad L_mix vs grad G_APEX")
    t2 = test_gradient_equivalence(seed=0, bsz=64, lam=0.5)
    for k, v in t2.items():
        if k == "pass":
            continue
        print(f"  {k:22s} = {v:.6g}" if isinstance(v, float) else f"  {k:22s} = {v}")
    print(f"  RESULT: {'PASS' if t2['pass'] else 'FAIL'}")

    print("\n[T3] Warm-start + equal-refine: FM vs APEX, measured at NFE=1 / 4 / 20")
    t3, (fm_model, apex_model, shift) = test_training_improves_nfe1(
        centers, std, args.warm_steps, args.refine_steps, args.bsz, args.lr, device,
        args.lam, args.lam_p, args.lam_c, args.lam_f, args.a, args.b,
    )
    for k, v in t3.items():
        if k == "pass":
            continue
        print(f"  {k:22s} = {v:.4g}" if isinstance(v, float) else f"  {k:22s} = {v}")
    print(f"  RESULT: {'PASS' if t3['pass'] else 'FAIL'}")

    print("\n[T1] ConditionShift differentiates trained model predictions")
    t1 = test_condition_shift_differentiates(apex_model, shift, centers, std)
    for k, v in t1.items():
        if k == "pass":
            continue
        print(f"  {k:22s} = {v:.4g}" if isinstance(v, float) else f"  {k:22s} = {v}")
    print(f"  RESULT: {'PASS' if t1['pass'] else 'FAIL'}")

    hrule()
    all_pass = t1["pass"] and t2["pass"] and t3["pass"]
    print(f"\nPhase 0 overall: {'PASS — green-light Phase 1' if all_pass else 'FAIL — debug before proceeding'}")
    hrule()

    if args.plot:
        from pathlib import Path
        Path(args.plot_path).parent.mkdir(parents=True, exist_ok=True)
        maybe_plot(centers, std, fm_model, apex_model, device, args.plot_path)


if __name__ == "__main__":
    main()
