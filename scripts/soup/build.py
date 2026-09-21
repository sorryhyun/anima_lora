#!/usr/bin/env python3
"""ΔW-level LoRA soups: exact rank-concat average + rank-r SVD truncation.

Souping LoRAs *parameterwise* (averaging A's and B's) is wrong twice over:
``avg(B) @ avg(A) != avg(B @ A)``, and with ``down_init="weight_svd"`` the
randomized SVD's per-vector sign ambiguity means row-wise A averaging can
actively cancel rows. So we soup at the ΔW level, which is invariant to any
per-ingredient ``(A, B)`` reparameterization — and a weighted average of
rank-r deltas is *exactly* representable as one rank-``sum(r_i)`` LoRA:

    dW_soup = sum_i w_i * scale_i * up_i @ down_i'
            = [w_1*scale_1*up_1 | w_2*scale_2*up_2 | ...] @ [down_1' ; down_2' ; ...]

(block concatenation — no approximation), where ``down_i' = down_i *
inv_scale_i`` folds the persisted channel-scaling buffer back in so the soup
applies to raw inputs (mirrors ``LoRAModule.get_weight``), and the soup's
alpha is set to its rank (scale = 1). No ``inv_scale`` keys are carried, so
the loaded soup module takes the raw-input path.

``truncated_soup`` additionally truncates each module's averaged ΔW to its
top-r singular directions (Eckart–Young: the best rank-r Frobenius
approximation) so the artifact stays single-adapter-sized. The full ΔW is
never materialized: QR on the concatenated factors reduces the SVD to an
(R x R) core where R = sum of ingredient ranks. Per-module retained energy
(sum sigma_kept^2 / sum sigma^2) is returned so the truncation loss is
measured, not assumed — on shared weight_svd-init ingredients it is ~99.9%
(bench/memorization/report.md, Act 5).

Plain-LoRA checkpoints only (the lora.toml weight_svd / T-LoRA family —
T-LoRA is training-only so its checkpoints are plain). Hydra key shapes are
refused loudly.

Usage
-----
    # exact rank-concat soup (rank = sum of ingredient ranks)
    python -m scripts.soup.build --ckpts a.safetensors b.safetensors --out soup.safetensors

    # rank-16 SVD-truncated soup (same size as a single ingredient)
    python -m scripts.soup.build --ckpts a.st b.st c.st --rank 16 --out soup16.safetensors

    # lambda=0.25 interpolation of a pair (LMC probe point)
    python -m scripts.soup.build --ckpts a.st b.st --weights 0.75 0.25 --out mid.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# Suffixes a plain-LoRA checkpoint may carry per module. Anything else means a
# non-plain variant (or a new format) — fail loudly rather than soup garbage.
_ALLOWED_SUFFIXES = ("lora_down.weight", "lora_up.weight", "alpha", "inv_scale")
# Non-module top-level keys we refuse (registers / routers / adapters change
# inference behavior and cannot be averaged this way).
_REFUSED_TOP_LEVEL = ("register_tokens",)


def _split_key(key: str) -> tuple[str, str]:
    """``lora_unet_..._q_proj.lora_down.weight`` -> (module, suffix)."""
    head, _, _ = key.partition(".")
    return head, key[len(head) + 1 :]


def _group_modules(sd: dict, path: str) -> dict[str, dict[str, torch.Tensor]]:
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in sd.items():
        if key in _REFUSED_TOP_LEVEL or "." not in key:
            raise ValueError(
                f"{path}: key {key!r} is not a plain-LoRA module key — this "
                "checkpoint carries state the ΔW soup cannot represent."
            )
        module, suffix = _split_key(key)
        if suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(
                f"{path}: unrecognized key suffix {suffix!r} on {module!r} — "
                "only plain LoRA checkpoints (lora_down/lora_up/alpha/inv_scale) "
                "can be souped; Hydra is refused."
            )
        modules.setdefault(module, {})[suffix] = value
    return modules


def _effective_factors(
    mod: dict[str, torch.Tensor], module: str, path: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """(down', up') in float32 such that ΔW = up' @ down' exactly.

    Folds alpha/r scaling and the channel-scaling ``inv_scale`` buffer
    (mirrors ``LoRAModule.get_weight``). Linear (2-D) only.
    """
    try:
        down = mod["lora_down.weight"].to(torch.float32)
        up = mod["lora_up.weight"].to(torch.float32)
    except KeyError as exc:
        raise ValueError(f"{path}: {module} is missing {exc} — truncated file?")
    if down.dim() != 2 or up.dim() != 2:
        raise ValueError(
            f"{path}: {module} has non-Linear factors "
            f"(down {tuple(down.shape)}, up {tuple(up.shape)}) — Conv LoRA is "
            "not supported by the soup (the DiT LoRA family is Linear-only)."
        )
    rank = down.shape[0]
    alpha = mod.get("alpha")
    scale = (float(alpha) / rank) if alpha is not None else 1.0
    inv_scale = mod.get("inv_scale")
    if inv_scale is not None:
        down = down * inv_scale.to(torch.float32).unsqueeze(0)
    return down, up * scale


def _grouped_factors(
    sds: list[dict],
    weights: list[float] | None,
    paths: list[str] | None,
) -> tuple[list[dict], list[float], list[str], set[str], torch.dtype]:
    """Shared ingredient validation for both soup flavors."""
    n = len(sds)
    if n < 2:
        raise ValueError("Need >= 2 ingredient checkpoints to soup.")
    if weights is None:
        weights = [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weights for {n} checkpoints.")
    paths = paths or [f"<sd {i}>" for i in range(n)]
    grouped = [_group_modules(sd, p) for sd, p in zip(sds, paths)]
    module_names = set(grouped[0])
    for g, p in zip(grouped[1:], paths[1:]):
        if set(g) != module_names:
            missing = module_names.symmetric_difference(g)
            raise ValueError(
                f"Ingredient module sets differ ({p}): {sorted(missing)[:5]}... — "
                "all ingredients must come from the same recipe."
            )
    out_dtype = sds[0][next(iter(sds[0]))].dtype
    return grouped, weights, paths, module_names, out_dtype


def soup_state_dicts(
    sds: list[dict],
    weights: list[float] | None = None,
    *,
    paths: list[str] | None = None,
    out_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Build the exact rank-concatenated soup state dict.

    ``weights`` default to uniform 1/N. They are used as-is (not normalized),
    so ``[0.5, 0.5]`` on a pair is both the uniform soup and the lambda=0.5
    LMC midpoint.
    """
    grouped, weights, paths, module_names, dtype = _grouped_factors(sds, weights, paths)
    out_dtype = out_dtype or dtype

    soup: dict[str, torch.Tensor] = {}
    for module in sorted(module_names):
        downs, ups = [], []
        for g, w, p in zip(grouped, weights, paths):
            down, up = _effective_factors(g[module], module, p)
            downs.append(down)
            ups.append(up * w)
        soup_down = torch.cat(downs, dim=0)
        soup_up = torch.cat(ups, dim=1)
        rank = soup_down.shape[0]
        soup[f"{module}.lora_down.weight"] = soup_down.to(out_dtype)
        soup[f"{module}.lora_up.weight"] = soup_up.to(out_dtype)
        # alpha = rank -> scale 1 (per-ingredient scales already folded into up).
        soup[f"{module}.alpha"] = torch.tensor(float(rank), dtype=torch.float32)
    return soup


def truncated_soup(
    sds: list[dict],
    rank: int,
    weights: list[float] | None = None,
    *,
    paths: list[str] | None = None,
    out_dtype: torch.dtype | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Rank-r SVD truncation of the ΔW-average soup.

    Returns ``(state_dict, retained_energy_per_module)``.
    """
    grouped, weights, paths, module_names, dtype = _grouped_factors(sds, weights, paths)
    out_dtype = out_dtype or dtype

    soup: dict[str, torch.Tensor] = {}
    energy: dict[str, float] = {}
    for module in sorted(module_names):
        downs, ups = [], []
        for g, w, p in zip(grouped, weights, paths):
            down, up = _effective_factors(g[module], module, p)
            downs.append(down)
            ups.append(up * w)
        # dW = U_cat @ V_cat, U_cat: (out, R), V_cat: (R, in)
        u_cat = torch.cat(ups, dim=1)
        v_cat = torch.cat(downs, dim=0)
        # Exact SVD of dW via QR reduction to the (R x R) core.
        qu, ru = torch.linalg.qr(u_cat)  # (out,R),(R,R)
        qv, rv = torch.linalg.qr(v_cat.T)  # (in,R),(R,R)
        core = ru @ rv.T  # (R,R)
        cu, s, cvh = torch.linalg.svd(core)
        r = min(rank, s.shape[0])
        total = float((s**2).sum())
        kept = float((s[:r] ** 2).sum())
        energy[module] = kept / total if total > 0 else 1.0
        sqrt_s = s[:r].sqrt()
        up_t = (qu @ cu[:, :r]) * sqrt_s  # (out, r)
        down_t = (cvh[:r] @ qv.T) * sqrt_s.unsqueeze(1)  # (r, in)
        soup[f"{module}.lora_up.weight"] = up_t.to(out_dtype)
        soup[f"{module}.lora_down.weight"] = down_t.to(out_dtype)
        # alpha = rank -> scale 1, matching soup_state_dicts' convention.
        soup[f"{module}.alpha"] = torch.tensor(float(r), dtype=torch.float32)
    return soup, energy


def _print_energy(energy: dict[str, float], rank: int) -> dict[str, float]:
    es = torch.tensor(list(energy.values()))
    print(
        f"retained energy @rank {rank}: mean {es.mean():.4f}  "
        f"min {es.min():.4f}  p10 {es.quantile(0.1):.4f}  max {es.max():.4f}"
    )
    print("worst modules:")
    for name, e in sorted(energy.items(), key=lambda kv: kv[1])[:8]:
        print(f"  {e:.4f}  {name}")
    return {
        "retained_energy_mean": round(float(es.mean()), 6),
        "retained_energy_min": round(float(es.min()), 6),
    }


def build_soup_file(
    ckpts: list[str],
    out: str,
    weights: list[float] | None = None,
    *,
    rank: int | None = None,
) -> Path:
    """Load ingredient .safetensors, soup, write ``out`` with updated metadata.

    ``rank=None`` writes the exact rank-concat soup; an int writes the rank-r
    SVD truncation (and prints per-module retained-energy stats).
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    sds = [load_file(c) for c in ckpts]
    with safe_open(ckpts[0], framework="pt") as f:
        metadata = dict(f.metadata() or {})

    ingredients: dict = {
        "ckpts": [Path(c).name for c in ckpts],
        "weights": weights or [1.0 / len(ckpts)] * len(ckpts),
    }
    if rank is None:
        soup = soup_state_dicts(sds, weights, paths=list(ckpts))
        ranks = {v.shape[0] for k, v in soup.items() if k.endswith(".lora_down.weight")}
        rank_str = str(ranks.pop()) if len(ranks) == 1 else "Dynamic"
    else:
        soup, energy = truncated_soup(sds, rank, weights, paths=list(ckpts))
        ingredients["svd_rank"] = rank
        ingredients.update(_print_energy(energy, rank))
        rank_str = str(rank)

    metadata["ss_network_dim"] = rank_str
    metadata["ss_network_alpha"] = rank_str
    metadata["ss_soup_ingredients"] = json.dumps(ingredients)

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(soup, str(out_path), metadata=metadata)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Per-ingredient weights (default uniform 1/N). Used as-is.",
    )
    ap.add_argument(
        "--rank",
        type=int,
        default=None,
        help="SVD-truncate the averaged ΔW to this rank (default: exact "
        "rank-concat, rank = sum of ingredient ranks).",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = build_soup_file(args.ckpts, args.out, args.weights, rank=args.rank)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
