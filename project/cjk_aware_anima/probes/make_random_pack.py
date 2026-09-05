#!/usr/bin/env python3
"""Geometry-matched *random* ext pack — the control for "does the learned
vocab representation help a LoRA train?" (arm R of the unmask line).

Reads a trained pack, measures its row geometry (per-row norms, mean vector,
centered covariance spectrum → participation ratio, random-pair cosine), and
writes a pack whose rows are **independent Gaussian draws with the same
spectrum, the same norm distribution and the same mean-norm** but a random
basis and a random mean direction — so PR / orthogonality / scale match while
every row's content, and every row-to-row relation, is noise. Routing json
is copied verbatim (same ids, same rows, same symbol block), so the
HybridT5Encoder tokenizes identically and only the table differs.

    .venv/bin/python project/cjk_aware_anima/probes/make_random_pack.py \
        --src output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256 \
        --out output/ckpt/cjk_vocab_pack_random_r256 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def geometry(x: torch.Tensor, n_pairs: int = 50_000, gen=None) -> dict:
    x = x.float()
    norms = x.norm(dim=1)
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / (x.shape[0] - 1)
    lam = torch.linalg.eigvalsh(cov).clamp_min(0)
    pr = float(lam.sum() ** 2 / (lam**2).sum())
    n = x.shape[0]
    i = torch.randint(0, n, (n_pairs,), generator=gen)
    j = torch.randint(0, n, (n_pairs,), generator=gen)
    keep = i != j
    cos = torch.nn.functional.cosine_similarity(x[i[keep]], x[j[keep]], dim=1)
    xn = x / norms[:, None].clamp_min(1e-8)
    common = float((xn @ (mu / mu.norm())).mean())
    return {
        "rows": int(n),
        "dim": int(x.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "mean_norm": float(mu.norm()),
        "pr": pr,
        "pair_cos_mean": float(cos.mean()),
        "pair_cos_std": float(cos.std()),
        "pair_cos_gt_0.5": float((cos > 0.5).float().mean()),
        "common_direction_cos": common,
        "spectrum_top5": [float(v) for v in lam.flip(0)[:5]],
    }


def build_mode(opts, x: torch.Tensor, gen) -> torch.Tensor:
    n, d = x.shape
    norms = x.norm(dim=1)
    if opts.mode == "hot":
        z = torch.randn(n, d, generator=gen)
        return z / z.norm(dim=1, keepdim=True) * (norms.mean() * opts.scale)
    if opts.mode == "cold":
        z = torch.randn(n, d, generator=gen)
        return z / z.norm(dim=1, keepdim=True) * (norms.mean() * opts.scale)
    if opts.mode == "collapse":
        mu = x.mean(0)
        return (mu / mu.norm() * norms.mean()).expand(n, d).clone()
    if opts.mode == "rotate":
        q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))
        return x @ q
    if opts.mode == "collide":
        from anima_lora import default_checkpoints
        from library.anima import weights as anima_utils

        dit = opts.dit or default_checkpoints().dit
        adapter = anima_utils.load_llm_adapter(dit, dtype=torch.float32, device="cpu")
        native = adapter.embed.weight.detach().float()
        # ordinary sentencepiece rows only: skip pad/eos/unk and the 28 extra ids
        lo, hi = 3, min(32000, native.shape[0])
        idx = torch.randint(lo, hi, (n,), generator=gen)
        rows = native[idx].clone()
        print(
            f"collide: {n} rows drawn from native ids [{lo},{hi}); native norm {float(native[lo:hi].norm(dim=1).mean()):.2f}"
        )
        return rows
    raise ValueError(opts.mode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="trained pack prefix")
    ap.add_argument("--out", type=Path, required=True, help="output pack prefix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_pairs", type=int, default=50_000)
    ap.add_argument(
        "--iters", type=int, default=8, help="spectrum/norm alternating projections"
    )
    ap.add_argument(
        "--mode",
        default="matched",
        choices=("matched", "hot", "cold", "collapse", "collide", "rotate"),
        help="matched = spectrum/norm/mean-matched Gaussian (arm R); hot/cold = "
        "isotropic Gaussian at --scale x the trained mean norm; collapse = every row "
        "the trained mean direction (PR 1); collide = rows copied from random native "
        "T5 rows of the adapter embed (real addresses, wrong content); rotate = the "
        "trained table under one random orthogonal rotation.",
    )
    ap.add_argument(
        "--scale", type=float, default=5.0, help="norm multiplier for hot/cold"
    )
    ap.add_argument("--dit", default=None, help="DiT checkpoint for --mode collide")
    opts = ap.parse_args()

    from safetensors.torch import load_file, save_file

    src = REPO / opts.src if not opts.src.is_absolute() else opts.src
    out = REPO / opts.out if not opts.out.is_absolute() else opts.out
    table = load_file(str(src.with_suffix(".safetensors")))["ext_embed"]
    mapping = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
    dtype = table.dtype
    x = table.float()
    n, d = x.shape
    gen = torch.Generator().manual_seed(opts.seed)

    g_src = geometry(x, opts.n_pairs, gen)
    print("source :", json.dumps(g_src))

    if opts.mode != "matched":
        xr = build_mode(opts, x, gen)
        g_out = geometry(xr, opts.n_pairs, gen)
        print(f"{opts.mode} :", json.dumps(g_out))
        match = torch.nn.functional.cosine_similarity(x, xr, dim=1)
        print(
            f"row-wise cos(source, {opts.mode}): mean {float(match.mean()):+.4f} std {float(match.std()):.4f}"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"ext_embed": xr.to(dtype).contiguous()},
            str(out.with_suffix(".safetensors")),
        )
        mapping["training"] = {
            "mode": f"geometry_{opts.mode}",
            "source": str(opts.src),
            "seed": opts.seed,
            "scale": opts.scale,
            "geometry_source": g_src,
            "geometry_out": g_out,
            "row_cos_to_source_mean": float(match.mean()),
        }
        out.with_suffix(".json").write_text(
            json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
        )
        print(f"-> {out}.{{safetensors,json}}")
        return

    # --- matched-spectrum Gaussian rows in a random basis --------------------
    mu = x.mean(0)
    xc = x - mu
    cov = xc.T @ xc / (n - 1)
    lam, _ = torch.linalg.eigh(cov)
    lam = lam.clamp_min(0)
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen))  # random orthonormal basis
    z = torch.randn(n, d, generator=gen)
    xr = z * lam.sqrt()[None, :] @ q.T
    mu_r = torch.randn(d, generator=gen)
    mu_r = mu_r / mu_r.norm() * mu.norm()
    xr = xr + mu_r
    # match the per-row norm distribution exactly (permuted assignment); the
    # row rescale bends the spectrum, so alternate "recolor to the target
    # spectrum in the current eigenbasis" / "re-match norms" a few times.
    norms_src = x.norm(dim=1)
    perm = torch.randperm(n, generator=gen)
    target = lam.clamp_min(1e-6)
    for it in range(opts.iters):
        xr = (
            xr / xr.norm(dim=1, keepdim=True).clamp_min(1e-8) * norms_src[perm][:, None]
        )
        if it == opts.iters - 1:
            break
        m = xr.mean(0)
        c = xr - m
        lam_r, v_r = torch.linalg.eigh(c.T @ c / (n - 1))
        scale = (target / lam_r.clamp_min(1e-6)).sqrt()
        xr = (c @ v_r) * scale[None, :] @ v_r.T + m

    g_out = geometry(xr, opts.n_pairs, gen)
    print("random :", json.dumps(g_out))
    print(
        f"PR {g_src['pr']:.1f} -> {g_out['pr']:.1f} | pair cos {g_src['pair_cos_mean']:.3f}±{g_src['pair_cos_std']:.3f}"
        f" -> {g_out['pair_cos_mean']:.3f}±{g_out['pair_cos_std']:.3f} | norm {g_src['norm_mean']:.3f} -> {g_out['norm_mean']:.3f}"
    )
    # sanity: content is random — cosine between matched rows ≈ 0
    match = torch.nn.functional.cosine_similarity(x, xr, dim=1)
    print(
        f"row-wise cos(source, random): mean {float(match.mean()):+.4f} std {float(match.std()):.4f}"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"ext_embed": xr.to(dtype).contiguous()}, str(out.with_suffix(".safetensors"))
    )
    mapping["training"] = {
        "mode": "random_geometry_match",
        "source": str(opts.src),
        "seed": opts.seed,
        "geometry_source": g_src,
        "geometry_random": g_out,
        "row_cos_to_source_mean": float(match.mean()),
    }
    out.with_suffix(".json").write_text(
        json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
    )
    print(f"-> {out}.{{safetensors,json}}")


if __name__ == "__main__":
    main()
