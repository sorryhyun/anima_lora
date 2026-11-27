"""Whitened-PCA ``lora_A`` init for PaddleOCR-VL-1.6 training (``finetune_vl16_lora.py
--eva``), after EVA (Paischer et al., arXiv:2410.07170).

``init(model, batches)`` takes the peft model with fresh adapters (``lora_B`` zero, so the
model's function is the base's) and an iterable of model inputs. Each LoRA Linear's input
is read over those batches as a second moment and a mean; q / k / v of one attention share
theirs. Per input, with covariance C and mean m:

* the directions are the top-r eigenvectors of C restricted to the complement of m, so a
  row of ``lora_A`` is orthogonal to the mean and the latent carries no constant;
* each row is scaled by ``lambda^-1/2``, so the r latents are uncorrelated with one
  variance on the calibration inputs;
* that variance is ``tr(C) / (3 d)``, a kaiming-uniform row's on the centred inputs, so a
  step on ``lora_B`` moves the output as far as it does from the stock init.

``lora_B`` is not touched. peft's own ``initialize_lora_eva_weights`` divides by the square
root of the singular value (a scale that moves with the token count) and loops until every
component's cosine settles; this computes the eigenvectors once from the full moments.
"""

from __future__ import annotations

from collections import defaultdict

import torch

SHARED = ("q_proj", "k_proj", "v_proj")
"""Leaves of one attention module that read the same tensor."""


def _group(name: str) -> str:
    parent, _, leaf = name.rpartition(".")
    return f"{parent}.qkv" if leaf in SHARED else name


def _eff_rank(cov: torch.Tensor) -> float:
    ev = torch.linalg.eigvalsh(cov).clamp_min(0)
    return float(ev.sum() ** 2 / (ev**2).sum())


@torch.no_grad()
def init(model, batches, adapter: str = "default") -> dict:
    """Overwrite every LoRA Linear's ``lora_A`` in place; returns the per-leaf medians it
    printed (latent effective rank before and after, the eigenvalue range)."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    layers = {
        n: m
        for n, m in model.named_modules()
        if isinstance(m, LoraLinear) and adapter in m.lora_A
    }
    groups: dict[str, list[str]] = defaultdict(list)
    for n in layers:
        groups[_group(n)].append(n)
    mom, mean, count = {}, {}, defaultdict(int)

    def hook(g: str):
        def f(_m, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            if g not in mom:
                d = x.shape[-1]
                mom[g] = torch.zeros(d, d, dtype=torch.float64, device=x.device)
                mean[g] = torch.zeros(d, dtype=torch.float64, device=x.device)
            mom[g] += (x.T @ x).double()
            mean[g] += x.sum(0).double()
            count[g] += x.shape[0]

        return f

    handles = [
        layers[ns[0]].register_forward_pre_hook(hook(g)) for g, ns in groups.items()
    ]
    was_training = model.training
    model.eval()
    n_batches = 0
    for inputs in batches:
        model(**inputs)
        n_batches += 1
    model.train(was_training)
    for h in handles:
        h.remove()

    stats = defaultdict(list)
    for g, names in groups.items():
        n = count[g]
        assert n > 0, f"{g}: no input seen"
        m = mom.pop(g).div_(n)
        mu = mean.pop(g) / n
        d = m.shape[0]
        cov = m - torch.outer(mu, mu)
        var = float(cov.diagonal().sum())
        u = mu / mu.norm().clamp_min(1e-30)
        cu = cov @ u
        # P C P with P = I - u u^T
        perp = (
            cov - torch.outer(u, cu) - torch.outer(cu, u) + (u @ cu) * torch.outer(u, u)
        )
        ev, evec = torch.linalg.eigh(perp)
        del cov, perp
        for name in names:
            a = layers[name].lora_A[adapter].weight
            r = a.shape[0]
            assert n > r, f"{name}: {n} tokens for rank {r}"
            lam, v = ev[-r:].flip(0), evec[:, -r:].flip(1)
            assert lam[-1] > 1e-12 * lam[0], f"{name}: input covariance below rank {r}"
            new = (var / (3 * d)) ** 0.5 * (v / lam.sqrt()).T
            leaf = name.rsplit(".", 1)[-1]
            a0 = a.double()
            stats[leaf].append(
                (
                    _eff_rank(a0 @ m @ a0.T),
                    _eff_rank(new @ m @ new.T),
                    float(lam[-1] / lam[0]),
                    float(lam.sum() / var),
                )
            )
            a.copy_(new.to(a.dtype))
        del m, ev, evec
    torch.cuda.empty_cache()

    out = {}
    print(
        f"eva init: {len(layers)} lora_A over {len(groups)} inputs, {n_batches} batches, "
        f"{max(count.values())} tokens at most per input",
        flush=True,
    )
    for leaf, rows in sorted(stats.items()):
        t = torch.tensor(rows).median(0).values
        out[leaf] = dict(
            n=len(rows),
            latent_rank_kaiming=float(t[0]),
            latent_rank_eva=float(t[1]),
            lam_last_over_first=float(t[2]),
            var_share=float(t[3]),
        )
        print(
            f"  {leaf:9s} n {len(rows):3d}: latent eff. rank {t[0]:.1f} -> {t[1]:.1f}, "
            f"lambda_r / lambda_1 {t[2]:.1e}, centred variance in the rows {t[3]:.2f}",
            flush=True,
        )
    return out
