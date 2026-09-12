#!/usr/bin/env python3
"""Gradient noise scale (McCandlish B_simple) per LoRA-target Linear, per σ-bin.

Answers "is raising the effective batch (gradient accumulation) worth it?" for
this DiT at init (LoRA = 0 ⇒ base model), without training. Per sample
``i`` (one image, one σ, one ε) the sketched gradient ``s_i = Ωᵀ G_i`` of every
target Linear is accumulated as a running sum and a running Σ‖s_i‖², so

    E‖s_i‖²  = ‖g‖² + tr Σ            ‖ŝ‖² = ‖Σ s_i / N‖² = ‖g‖² + tr Σ / N
    tr Σ     = (E‖s_i‖² − ‖ŝ‖²) · N / (N − 1)
    B_simple = tr Σ / ‖g‖²             (Gaussian sketch: norms preserved in expectation)

``B_simple`` is the batch size at which the gradient's SNR reaches 1 — beyond
a few × B_simple, extra samples per step buy almost nothing. Reported overall
(σ random per sample, what training sees) and within σ quartile bins (σ
heterogeneity removed). Also per module kind and per block.

Usage::

    make daemon-run ARGS="bench/grad_init/probe_noise_scale.py --artists aak --gradient_checkpointing --passes 4"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from bench.grad_init.probe_subspace import (  # noqa: E402
    _BLOCK_RE,
    enumerate_targets,
    latent_tokens,
    module_kind,
    stratified_logit_normal,
)
from library.env import resolve_under_home  # noqa: E402
from library.io.cache import (  # noqa: E402
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)

# σ quartile edges of the trainer's logit-normal(0,1): sigmoid(Φ⁻¹(.25/.5/.75))
SIGMA_EDGES = (0.338, 0.5, 0.662)
N_BINS = len(SIGMA_EDGES) + 1


def sigma_bin(sigma: float) -> int:
    return sum(sigma >= e for e in SIGMA_EDGES)


class NoiseAccumulator:
    def __init__(self, targets, q, device, seed):
        self.q = q
        self.omega, self.sum_s, self.sumsq, self.n = {}, {}, {}, {}
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for lora_name, _o, mod in targets:
            om = torch.randn(mod.out_features, q, generator=gen, dtype=torch.float32)
            self.omega[lora_name] = om.to(device)
            self.sum_s[lora_name] = torch.zeros(
                N_BINS, q, mod.in_features, dtype=torch.float32, device=device
            )
            self.sumsq[lora_name] = torch.zeros(
                N_BINS, dtype=torch.float64, device=device
            )
            self.n[lora_name] = [0] * N_BINS
        self.bin = 0
        self._handles = []

    def attach(self, targets):
        for lora_name, _o, mod in targets:
            self._handles.append(mod.register_forward_pre_hook(self._pre_hook))
            self._handles.append(mod.register_forward_hook(self._fwd_hook(lora_name)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @staticmethod
    def _pre_hook(_mod, inputs):
        x = inputs[0]
        if torch.is_grad_enabled() and not x.requires_grad:
            return (x.detach().requires_grad_(True),) + tuple(inputs[1:])
        return None

    def _fwd_hook(self, lora_name):
        def hook(_mod, inputs, out):
            if not out.requires_grad:
                return
            x, om, b = inputs[0], self.omega[lora_name], self.bin

            def on_grad(delta):
                d = delta.reshape(-1, delta.shape[-1]).float()
                xx = x.reshape(-1, x.shape[-1]).float()
                s_i = (d @ om).T @ xx  # (q, in): this sample's sketched gradient
                self.sum_s[lora_name][b].add_(s_i)
                self.sumsq[lora_name][b] += s_i.double().pow(2).sum()
                self.n[lora_name][b] += 1

            out.register_hook(on_grad)

        return hook


def noise_stats(
    sum_s: torch.Tensor, sumsq: torch.Tensor, n: int
) -> tuple[float, float, float]:
    """→ (‖g‖², tr Σ, B_simple) from Σ s_i and Σ‖s_i‖² over n samples."""
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean_sq = float(sumsq) / n  # E‖s_i‖²
    shat_sq = float(sum_s.double().pow(2).sum()) / (n * n)  # ‖ŝ‖²
    tr_sigma = max(mean_sq - shat_sq, 0.0) * n / (n - 1)
    g_sq = max(shat_sq - tr_sigma / n, 1e-30)
    return g_sq, tr_sigma, tr_sigma / g_sq


def run_artist(anima, acc, cache_dir, args, device, seed):
    pairs = discover_cached_pairs(str(cache_dir))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pairs = [pairs[i] for i in torch.randperm(len(pairs), generator=gen).tolist()]
    if args.max_samples:
        pairs = pairs[: args.max_samples]
    noise_gen = torch.Generator(device=device).manual_seed(seed + 1)
    schedule = []
    for p in range(args.passes):
        sig = stratified_logit_normal(len(pairs), gen)
        schedule += [(j, sig[j]) for j in range(len(pairs))]
    used, t0 = 0, time.perf_counter()
    for j, sigma_cpu in schedule:
        ci = pairs[j]
        latents = load_cached_latents(ci.npz_path)[0].unsqueeze(0)
        if latent_tokens(latents) > args.max_tokens:
            continue
        crossattn, _ = load_cached_text_features(ci.te_path, variant=0)
        if crossattn is None:
            continue
        latents = latents.to(device)
        crossattn = crossattn.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        noise = torch.randn(
            latents.shape, generator=noise_gen, device=device, dtype=latents.dtype
        )
        sigma = sigma_cpu.view(1).to(device)
        s4 = sigma.view(-1, 1, 1, 1)
        noisy_5d = ((1.0 - s4) * latents + s4 * noise).unsqueeze(2).to(torch.bfloat16)
        noisy_5d.requires_grad_(True)
        target = noise - latents
        pad = torch.zeros(
            1, 1, *latents.shape[-2:], dtype=torch.bfloat16, device=device
        )
        acc.bin = sigma_bin(float(sigma_cpu))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = anima(noisy_5d, sigma, crossattn, padding_mask=pad)
        loss = torch.nn.functional.mse_loss(pred.squeeze(2).float(), target)
        loss.backward()
        used += 1
        del pred, loss, noisy_5d, latents, crossattn, noise
        if used % 16 == 0:
            print(
                f"  [{cache_dir.name}] {used}/{len(schedule)} {(time.perf_counter() - t0) / used:.2f}s/img",
                flush=True,
            )
    return {
        "n_pairs": len(pairs),
        "n_used": used,
        "passes": args.passes,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser, vae=False, text_encoder=False)
    add_common_args(parser, include_compile=False)
    parser.add_argument("--artists", type=str, required=True)
    parser.add_argument("--cache_root", type=str, default="post_image_dataset/lora")
    parser.add_argument("--sketch_q", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=4608)
    parser.add_argument("--passes", type=int, default=4)
    args = parser.parse_args()
    start_heartbeat(label="grad_noise")
    artists = [a.strip() for a in args.artists.split(",") if a.strip()]
    cache_root = Path(resolve_under_home(args.cache_root))

    bundle = build_anima(args, adapter=None, train_mode=True)
    anima, device = bundle.anima, bundle.device
    targets = enumerate_targets(anima)
    kinds = {ln: module_kind(o) for ln, o, _ in targets}
    blocks = {
        ln: int(m.group(1)) if (m := _BLOCK_RE.match(o)) else -1 for ln, o, _ in targets
    }

    rows, per_artist = [], {}
    for a in artists:
        acc = NoiseAccumulator(targets, args.sketch_q, device, seed=args.seed)
        acc.attach(targets)
        per_artist[a] = run_artist(
            anima, acc, cache_root / a, args, device, seed=args.seed
        )
        acc.detach()
        print(f"[{a}] {per_artist[a]}", flush=True)
        for ln, _o, mod in targets:
            S, Q, n = acc.sum_s[ln], acc.sumsq[ln], acc.n[ln]
            g_all, tr_all, b_all = noise_stats(S.sum(0), Q.sum(), sum(n))
            row = {
                "artist": a,
                "lora_name": ln,
                "kind": kinds[ln],
                "block": blocks[ln],
                "token_layer": int(not kinds[ln].startswith("adaln")),
                "n": sum(n),
                "g_sq": g_all,
                "tr_sigma": tr_all,
                "B_all": b_all,
            }
            for b in range(N_BINS):
                gb, tb, bb = noise_stats(S[b], Q[b], n[b])
                row[f"n_bin{b}"] = n[b]
                row[f"g_sq_bin{b}"] = gb
                row[f"tr_bin{b}"] = tb
                row[f"B_bin{b}"] = bb
            rows.append(row)
        acc.sum_s.clear()
        acc.omega.clear()
        del acc
        torch.cuda.empty_cache()

    # aggregates: whole-parameter-vector B = Σ tr Σ / Σ ‖g‖² (McCandlish), plus median per-layer
    def pooled(sel, gk, tk):
        rs = [r for r in rows if sel(r) and r[gk] == r[gk]]
        return (
            sum(r[tk] for r in rs) / max(sum(r[gk] for r in rs), 1e-30)
            if rs
            else float("nan")
        )

    def median(sel, key):
        v = sorted(r[key] for r in rows if sel(r) and r[key] == r[key])
        return v[len(v) // 2] if v else float("nan")

    summary = {
        "artists": artists,
        "per_artist": per_artist,
        "sigma_edges": SIGMA_EDGES,
        "by_artist": {},
    }
    for a in artists:

        def tok(r, a=a):
            return r["artist"] == a and r["token_layer"] and r["block"] >= 0

        d = {
            "B_pooled_all_sigma": pooled(tok, "g_sq", "tr_sigma"),
            "B_median_layer_all_sigma": median(tok, "B_all"),
            "B_pooled_by_bin": [
                pooled(tok, f"g_sq_bin{b}", f"tr_bin{b}") for b in range(N_BINS)
            ],
            "B_median_layer_by_bin": [median(tok, f"B_bin{b}") for b in range(N_BINS)],
            "g_sq_share_by_bin": None,
            "by_kind": {},
            "by_block": {},
        }
        tot = sum(r["g_sq"] for r in rows if tok(r))
        d["g_sq_share_by_bin"] = [
            sum(r[f"g_sq_bin{b}"] * r[f"n_bin{b}"] for r in rows if tok(r))
            / max(sum(r["g_sq"] * r["n"] for r in rows if tok(r)), 1e-30)
            for b in range(N_BINS)
        ]
        for k in sorted({r["kind"] for r in rows if tok(r)}):
            d["by_kind"][k] = pooled(
                lambda r, k=k: tok(r) and r["kind"] == k, "g_sq", "tr_sigma"
            )
        for blk in range(0, 28, 4):
            d["by_block"][blk] = pooled(
                lambda r, blk=blk: tok(r) and r["block"] == blk, "g_sq", "tr_sigma"
            )
        summary["by_artist"][a] = d
        _ = tot

    run_dir = make_run_dir("grad_init", label=args.label or "noise")
    with (run_dir / "per_layer_noise.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    for a in artists:
        d = summary["by_artist"][a]
        print(
            f"\n=== {a}: gradient noise scale B_simple (block token layers, LoRA=0) ==="
        )
        print(
            f"  pooled, σ random per sample : {d['B_pooled_all_sigma']:.2f}   (median layer {d['B_median_layer_all_sigma']:.2f})"
        )
        for b in range(N_BINS):
            lo = 0.0 if b == 0 else SIGMA_EDGES[b - 1]
            hi = 1.0 if b == N_BINS - 1 else SIGMA_EDGES[b]
            print(
                f"  σ∈[{lo:.2f},{hi:.2f}) pooled {d['B_pooled_by_bin'][b]:6.2f}  median layer {d['B_median_layer_by_bin'][b]:6.2f}"
            )
        print("  by kind :", {k: round(v, 2) for k, v in d["by_kind"].items()})
        print("  by block:", {k: round(v, 2) for k, v in d["by_block"].items()})
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=summary,
        label=args.label,
        artifacts=["per_layer_noise.csv", "summary.json"],
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
