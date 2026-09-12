#!/usr/bin/env python3
"""Gradient-SVD init probe — do two artists' task gradients live in different
input subspaces, and how does either compare to ``down_init="weight_svd"``?

No training. For every LoRA-target Linear (same enumeration as
``LoRANetwork.create_modules``: Linear children of Block / PatchEmbed /
TimestepEmbedding / FinalLayer) we accumulate a one-pass randomized sketch of
the flow-matching gradient over one artist's cached dataset::

    G  = Σ_tokens δ xᵀ            (out × in, never materialised)
    S  = Ωᵀ G = Σ (Ωᵀ δ) xᵀ       (q × in, fp32, Ω ~ N(0,1) out×q, fixed per layer)

The top-r right singular vectors of ``S`` estimate the top-r row space of
``G`` — exactly the subspace a LoRA-GA / LoRA-One ``lora_down`` init would use
(B = 0, first step = rank-r truncated full-FT step). Each artist is split into
two halves with separate accumulators so the within-artist split-half overlap
is the reliability floor for every between-artist number.

Overlap metric between orthonormal bases ``V1, V2 ∈ R^{in×r}``:
``‖V1ᵀV2‖_F² / r`` ∈ [0, 1]; the random-subspace null is ``r / in``.

σ is drawn stratified from the trainer's default logit-normal
(``timestep_sampling="sigmoid"``, scale 1, bias 0) so a 64-image artist covers
the σ marginal evenly instead of by luck (per-sample gradients are heavy-tailed
in σ on this model).

Usage::

    make daemon-run ARGS="bench/grad_init/probe_subspace.py --artists aak,abmayo --gradient_checkpointing"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
# the per-layer sketches + a 4k-token backward sit close to 16 GB; avoid
# fragmentation-driven OOM (1.7 GB reserved-unallocated was observed).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from bench._anima import add_common_args, add_model_args, build_anima  # noqa: E402
from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.io.cache import (  # noqa: E402
    discover_cached_pairs,
    load_cached_latents,
    load_cached_text_features,
)
from networks.lora_anima.network import LoRANetwork  # noqa: E402

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


# --------------------------------------------------------------------------- #
# target enumeration (mirrors LoRANetwork.create_modules, Linear-only)
# --------------------------------------------------------------------------- #
def enumerate_targets(anima: torch.nn.Module) -> list[tuple[str, str, torch.nn.Linear]]:
    """→ [(lora_name, original_name, module)] in named_modules() order."""
    wanted = set(LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
    out: list[tuple[str, str, torch.nn.Linear]] = []
    seen: set[int] = set()
    for name, module in anima.named_modules():
        if module.__class__.__name__ not in wanted:
            continue
        for child_name, child in module.named_modules():
            if not isinstance(child, torch.nn.Linear) or id(child) in seen:
                continue
            seen.add(id(child))
            original = (name + "." if name else "") + child_name
            original = original.replace("_orig_mod.", "")
            lora_name = f"{LoRANetwork.LORA_PREFIX_ANIMA}.{original}".replace(".", "_")
            out.append((lora_name, original, child))
    return out


def module_kind(original_name: str) -> str:
    """Coarse bucket for per-type aggregation."""
    leaf = original_name.split(".")[-1]
    if leaf.isdigit():  # nn.Sequential leaves (adaln_lora / t_embedder / proj)
        leaf = ".".join(original_name.split(".")[-2:])
    if not _BLOCK_RE.match(original_name):
        return "non_block." + leaf
    parent = original_name.split(".")[2] if original_name.count(".") >= 3 else ""
    return f"{parent}.{leaf}" if parent else leaf


# --------------------------------------------------------------------------- #
# sketch accumulator
# --------------------------------------------------------------------------- #
class SketchAccumulator:
    """One (Ω, S[half]) pair per target Linear; hooks feed it during backward."""

    def __init__(
        self,
        targets: list[tuple[str, str, torch.nn.Linear]],
        q: int,
        device: torch.device,
        seed: int,
    ):
        self.q = q
        self.device = device
        self.omega: dict[str, torch.Tensor] = {}
        self.sketch: dict[str, torch.Tensor] = {}
        self.n_tokens: dict[str, list[int]] = {}
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for lora_name, _orig, mod in targets:
            om = torch.randn(mod.out_features, q, generator=gen, dtype=torch.float32)
            self.omega[lora_name] = om.to(device)
            self.sketch[lora_name] = torch.zeros(
                4, q, mod.in_features, dtype=torch.float32, device=device
            )
            self.n_tokens[lora_name] = [0, 0, 0, 0]
        self.slot = 0  # image_half * 2 + pass_half
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, targets):
        for lora_name, _orig, mod in targets:
            self._handles.append(mod.register_forward_pre_hook(self._pre_hook))
            self._handles.append(
                mod.register_forward_hook(self._make_fwd_hook(lora_name))
            )

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @staticmethod
    def _pre_hook(_mod, inputs):
        # Frozen DiT: a Linear whose input carries no grad (t-embedding path,
        # cached crossattn context, …) would never produce δ. Give it a leaf
        # that requires grad — nothing upstream was receiving grad through it
        # anyway, so this changes no other layer's δ.
        x = inputs[0]
        if torch.is_grad_enabled() and not x.requires_grad:
            x = x.detach().requires_grad_(True)
            return (x,) + tuple(inputs[1:])
        return None

    def _make_fwd_hook(self, lora_name: str):
        def hook(_mod, inputs, out):
            if not out.requires_grad:  # no-grad pass of the checkpointer
                return
            x = inputs[0]
            om = self.omega[lora_name]
            slot = self.slot

            def on_grad(delta: torch.Tensor):
                d = delta.reshape(-1, delta.shape[-1]).float()  # (T, out)
                xx = x.reshape(-1, x.shape[-1]).float()  # (T, in)
                proj = d @ om  # (T, q)
                self.sketch[lora_name][slot].add_(proj.T @ xx)
                self.n_tokens[lora_name][slot] += d.shape[0]

            out.register_hook(on_grad)

        return hook


# --------------------------------------------------------------------------- #
# subspace math
# --------------------------------------------------------------------------- #
def top_right_basis(S: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor]:
    """S (q×in) → (V_r (in×r), singular values (q,))."""
    _u, s, vh = torch.linalg.svd(S, full_matrices=False)
    return vh[:r].T.contiguous(), s


def weight_svd_basis(W: torch.Tensor, r: int) -> torch.Tensor:
    """The shipped ``down_init="weight_svd"`` basis (lora.py::_init_down_weight_svd)."""
    q = min(r + 6, min(W.shape))
    _, _, V = torch.svd_lowrank(W.float(), q=q, niter=2)
    return V[:, :r].contiguous()


def capture(S: torch.Tensor, V: torch.Tensor) -> float:
    """Fraction of the sketched gradient energy ‖S‖² that lies in span(V).

    ‖Ωᵀ G V‖² / ‖Ωᵀ G‖² estimates ‖G V‖² / ‖G‖² (Gaussian sketches preserve
    Frobenius norms in expectation) — i.e. how much of the first-step full-FT
    gradient a rank-r ``lora_down`` seeded with V would pass through.
    """
    return float((S @ V).pow(2).sum() / S.pow(2).sum().clamp_min(1e-30))


def random_basis(n_in: int, r: int, gen: torch.Generator) -> torch.Tensor:
    Q, _ = torch.linalg.qr(torch.randn(n_in, r, generator=gen))
    return Q


def overlap(V1: torch.Tensor, V2: torch.Tensor) -> float:
    r = V1.shape[1]
    return float((V1.T @ V2).pow(2).sum() / r)


def stratified_logit_normal(n: int, gen: torch.Generator) -> torch.Tensor:
    """n σ values covering the logit-normal(0,1) marginal by quantile strata."""
    u = (torch.randperm(n, generator=gen).float() + torch.rand(n, generator=gen)) / n
    u = u.clamp(1e-4, 1 - 1e-4)
    return torch.sigmoid(torch.special.ndtri(u))


# --------------------------------------------------------------------------- #
# one artist pass
# --------------------------------------------------------------------------- #
def latent_tokens(latents: torch.Tensor) -> int:
    h, w = latents.shape[-2:]
    return (h // 2) * (w // 2)


def run_artist(
    anima, acc: SketchAccumulator, cache_dir: Path, args, device, seed: int
) -> dict:
    pairs = discover_cached_pairs(str(cache_dir))
    if not pairs:
        raise SystemExit(f"no cached pairs under {cache_dir}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(pairs), generator=gen).tolist()
    pairs = [pairs[i] for i in order]
    if args.max_samples:
        pairs = pairs[: args.max_samples]
    noise_gen = torch.Generator(device=device).manual_seed(seed + 1)

    used, skipped_tokens, skipped_te = 0, 0, 0
    losses = []
    t0 = time.perf_counter()
    schedule = []  # (pass_idx, image_idx, sigma)
    for p in range(args.passes):
        sig = stratified_logit_normal(len(pairs), gen)
        schedule += [(p, j, sig[j]) for j in range(len(pairs))]
    total = len(schedule)
    for p, j, sigma_cpu in schedule:
        ci = pairs[j]
        npz_path, te_path = ci.npz_path, ci.te_path
        latents = load_cached_latents(npz_path)[0].unsqueeze(0)
        if latent_tokens(latents) > args.max_tokens:
            skipped_tokens += 1
            continue
        crossattn, _pooled = load_cached_text_features(te_path, variant=0)
        if crossattn is None:
            skipped_te += 1
            continue
        latents = latents.to(device)
        crossattn = crossattn.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        noise = torch.randn(
            latents.shape, generator=noise_gen, device=device, dtype=latents.dtype
        )
        sigma = sigma_cpu.view(1).to(device)
        # trainer default: timestep_sampling="sigmoid"; σ is the DiT time arg
        noisy = (1.0 - sigma.view(-1, 1, 1, 1)) * latents + sigma.view(
            -1, 1, 1, 1
        ) * noise
        target = noise - latents
        noisy_5d = noisy.unsqueeze(2).to(torch.bfloat16).requires_grad_(True)
        padding_mask = torch.zeros(
            1,
            1,
            latents.shape[-2],
            latents.shape[-1],
            dtype=torch.bfloat16,
            device=device,
        )
        acc.slot = (j % 2) * 2 + (p % 2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = anima(noisy_5d, sigma, crossattn, padding_mask=padding_mask)
        pred = pred.squeeze(2).float()
        loss = torch.nn.functional.mse_loss(pred, target)  # l2, uniform weighting
        loss.backward()
        losses.append(loss.item())
        used += 1
        del pred, loss, noisy_5d, noisy, latents, crossattn, noise
        if used % 8 == 0:
            print(
                f"  [{cache_dir.name}] {used}/{total} "
                f"loss={sum(losses[-8:]) / 8:.4f} "
                f"{(time.perf_counter() - t0) / used:.2f}s/img",
                flush=True,
            )
    return {
        "n_pairs": len(pairs),
        "n_used": used,
        "passes": args.passes,
        "skipped_tokens": skipped_tokens,
        "skipped_te": skipped_te,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "seconds": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_args(parser, vae=False, text_encoder=False)
    add_common_args(parser, include_compile=False)
    parser.add_argument(
        "--artists",
        type=str,
        required=True,
        help="comma-separated artist subdirs under --cache_root (2+).",
    )
    parser.add_argument(
        "--cache_root",
        type=str,
        default="post_image_dataset/lora",
        help="cache tree (repo-relative), artists are subdirs.",
    )
    parser.add_argument(
        "--rank", type=int, default=32, help="r (lora.toml network_dim)."
    )
    parser.add_argument(
        "--oversample", type=int, default=32, help="sketch width q = r + this."
    )
    parser.add_argument(
        "--max_samples", type=int, default=0, help="per artist; 0 = all."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4608, help="skip larger latents."
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=1,
        help="revisit every image this many times with fresh σ strata + noise; "
        "even/odd passes form the pass-split halves.",
    )
    parser.add_argument(
        "--save_bases", action="store_true", help="write per-artist V_r safetensors."
    )
    args = parser.parse_args()
    if not args.gradient_checkpointing:
        print(
            "note: --gradient_checkpointing off; a 4k-token image may OOM on 16 GB",
            flush=True,
        )

    start_heartbeat(label="grad_init")
    artists = [a.strip() for a in args.artists.split(",") if a.strip()]
    if len(artists) < 2:
        raise SystemExit("--artists needs at least two entries")
    cache_root = Path(resolve_under_home(args.cache_root))
    for a in artists:
        if not (cache_root / a).is_dir():
            raise SystemExit(f"missing cache dir: {cache_root / a}")

    bundle = build_anima(args, adapter=None, train_mode=True)
    anima, device = bundle.anima, bundle.device
    targets = enumerate_targets(anima)
    print(f"targets: {len(targets)} Linear modules", flush=True)
    q = args.rank + args.oversample

    # ---- per-artist sketches ------------------------------------------------
    per_artist_meta: dict[str, dict] = {}
    bases: dict[str, dict[str, torch.Tensor]] = {}  # artist -> {name: V (in×r)} (full)
    half_bases: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    pass_bases: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    sketches: dict[str, dict[str, torch.Tensor]] = {}  # artist -> {name: S (q×in)}
    sketch_halves: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    spectra: dict[str, dict[str, torch.Tensor]] = {}
    for a in artists:
        acc = SketchAccumulator(
            targets, q, device, seed=args.seed
        )  # same Ω every artist
        acc.attach(targets)
        meta = run_artist(anima, acc, cache_root / a, args, device, seed=args.seed)
        acc.detach()
        per_artist_meta[a] = meta
        print(f"[{a}] {meta}", flush=True)
        bases[a], half_bases[a], pass_bases[a], spectra[a] = {}, {}, {}, {}
        sketches[a], sketch_halves[a] = {}, {}
        for lora_name, _orig, _mod in targets:
            S = acc.sketch[lora_name]  # (4, q, in): slot = img_half*2 + pass_half
            sketches[a][lora_name] = S.sum(0).cpu()
            sketch_halves[a][lora_name] = ((S[0] + S[1]).cpu(), (S[2] + S[3]).cpu())
            V_full, s = top_right_basis(S.sum(0), args.rank)
            V_i0, _ = top_right_basis(S[0] + S[1], args.rank)  # image half 0
            V_i1, _ = top_right_basis(S[2] + S[3], args.rank)  # image half 1
            V_p0, _ = top_right_basis(S[0] + S[2], args.rank)  # pass half 0
            V_p1, _ = top_right_basis(S[1] + S[3], args.rank)  # pass half 1
            bases[a][lora_name] = V_full.cpu()
            half_bases[a][lora_name] = (V_i0.cpu(), V_i1.cpu())
            pass_bases[a][lora_name] = (V_p0.cpu(), V_p1.cpu())
            spectra[a][lora_name] = s.cpu()
        acc.sketch.clear()
        acc.omega.clear()
        del acc
        torch.cuda.empty_cache()

    # ---- weight_svd bases ---------------------------------------------------
    wsvd: dict[str, torch.Tensor] = {}
    for lora_name, _orig, mod in targets:
        wsvd[lora_name] = weight_svd_basis(mod.weight.data, args.rank).cpu()

    # ---- per-layer table ----------------------------------------------------
    a1, a2 = artists[0], artists[1]
    rand_gen = torch.Generator(device="cpu").manual_seed(args.seed + 7)
    rows = []
    for lora_name, orig, mod in targets:
        n_in = mod.in_features
        r = min(args.rank, n_in)
        s1, s2 = spectra[a1][lora_name], spectra[a2][lora_name]
        row = {
            "lora_name": lora_name,
            "kind": module_kind(orig),
            # adaln inputs are one vector per image (rank-1 per sample) — not a
            # token-level subspace; keep them out of the headline aggregates
            "token_layer": int(not module_kind(orig).startswith("adaln")),
            "block": int(m.group(1)) if (m := _BLOCK_RE.match(orig)) else -1,
            "in": n_in,
            "out": mod.out_features,
            "null": r / n_in,
            "grad_a1_vs_a2": overlap(bases[a1][lora_name], bases[a2][lora_name]),
            # same-n comparison: half of a1 vs half of a2 (symmetric to split_*)
            "grad_a1h_vs_a2h": overlap(
                half_bases[a1][lora_name][0], half_bases[a2][lora_name][0]
            ),
            "split_a1": overlap(*half_bases[a1][lora_name]),
            "split_a2": overlap(*half_bases[a2][lora_name]),
            "split_pass_a1": overlap(*pass_bases[a1][lora_name]),
            "split_pass_a2": overlap(*pass_bases[a2][lora_name]),
            # drop the shared dominant direction (top-1 right singular vector)
            "grad_a1_vs_a2_no1": overlap(
                bases[a1][lora_name][:, 1:], bases[a2][lora_name][:, 1:]
            ),
            "split_a1_no1": overlap(
                half_bases[a1][lora_name][0][:, 1:], half_bases[a1][lora_name][1][:, 1:]
            ),
            # energy capture of a1's gradient by each candidate basis
            "cap_a1_by_a1half": capture(
                sketch_halves[a1][lora_name][0], half_bases[a1][lora_name][1]
            ),
            "cap_a1_by_a2": capture(sketches[a1][lora_name], bases[a2][lora_name]),
            "cap_a1half_by_a2half": capture(
                sketch_halves[a1][lora_name][0], half_bases[a2][lora_name][0]
            ),
            "cap_a1_by_wsvd": capture(sketches[a1][lora_name], wsvd[lora_name]),
            "cap_a1_by_random": capture(
                sketches[a1][lora_name], random_basis(n_in, r, rand_gen)
            ),
            "cap_a2_by_a2half": capture(
                sketch_halves[a2][lora_name][0], half_bases[a2][lora_name][1]
            ),
            "cap_a2_by_a1": capture(sketches[a2][lora_name], bases[a1][lora_name]),
            "cap_a2_by_wsvd": capture(sketches[a2][lora_name], wsvd[lora_name]),
            "a1_vs_wsvd": overlap(bases[a1][lora_name], wsvd[lora_name]),
            "a2_vs_wsvd": overlap(bases[a2][lora_name], wsvd[lora_name]),
            "topr_energy_a1": float(s1[:r].pow(2).sum() / s1.pow(2).sum()),
            "topr_energy_a2": float(s2[:r].pow(2).sum() / s2.pow(2).sum()),
            "top1_frac_a1": float(s1[0].pow(2) / s1.pow(2).sum()),
            "top1_frac_a2": float(s2[0].pow(2) / s2.pow(2).sum()),
        }
        # extra artists (if any): pairwise vs a1
        for a in artists[2:]:
            row[f"grad_a1_vs_{a}"] = overlap(bases[a1][lora_name], bases[a][lora_name])
        rows.append(row)

    def agg(sel, key):
        vals = [r[key] for r in rows if sel(r)]
        return sum(vals) / len(vals) if vals else float("nan")

    keys = [
        "null",
        "grad_a1_vs_a2",
        "grad_a1h_vs_a2h",
        "split_a1",
        "split_a2",
        "split_pass_a1",
        "split_pass_a2",
        "grad_a1_vs_a2_no1",
        "split_a1_no1",
        "a1_vs_wsvd",
        "a2_vs_wsvd",
        "topr_energy_a1",
        "topr_energy_a2",
        "top1_frac_a1",
        "cap_a1_by_a1half",
        "cap_a1_by_a2",
        "cap_a1half_by_a2half",
        "cap_a1_by_wsvd",
        "cap_a1_by_random",
        "cap_a2_by_a2half",
        "cap_a2_by_a1",
        "cap_a2_by_wsvd",
    ]
    by_kind = defaultdict(dict)
    for k in sorted({r["kind"] for r in rows}):
        for key in keys:
            by_kind[k][key] = agg(lambda r, k=k: r["kind"] == k, key)
        by_kind[k]["n"] = sum(1 for r in rows if r["kind"] == k)
    overall = {key: agg(lambda r: True, key) for key in keys}
    blocks_only = {
        key: agg(lambda r: r["block"] >= 0 and r["token_layer"], key) for key in keys
    }
    # excess over null, block layers only
    excess = {
        key: blocks_only[key] / blocks_only["null"]
        for key in (
            "grad_a1_vs_a2",
            "grad_a1h_vs_a2h",
            "split_a1",
            "split_a2",
            "a1_vs_wsvd",
            "a2_vs_wsvd",
        )
    }

    metrics = {
        "artists": artists,
        "rank": args.rank,
        "sketch_q": q,
        "n_targets": len(targets),
        "per_artist": per_artist_meta,
        "overall_mean": overall,
        "block_token_layers_mean": blocks_only,
        "block_token_layers_x_null": excess,
        "by_kind": dict(by_kind),
    }

    run_dir = make_run_dir("grad_init", label=args.label)
    with (run_dir / "per_layer.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    artifacts = ["per_layer.csv"]
    if args.save_bases:
        for a in artists:
            fn = f"grad_basis_{a}_r{args.rank}.safetensors"
            save_file(
                {
                    k: v.to(torch.float16).cpu().contiguous()
                    for k, v in bases[a].items()
                },
                str(run_dir / fn),
                metadata={"artist": a, "rank": str(args.rank), "layout": "in x r"},
            )
            artifacts.append(fn)
            fn_s = f"grad_spectrum_{a}_r{args.rank}.safetensors"
            save_file(
                {k: v.contiguous() for k, v in spectra[a].items()}, str(run_dir / fn_s)
            )
            artifacts.append(fn_s)
    with (run_dir / "summary.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    artifacts.append("summary.json")

    print(
        "\n=== block token layers (adaln excluded), mean over modules (overlap; null = r/in) ==="
    )
    for key in keys:
        print(f"  {key:16s} {blocks_only[key]:.4f}")
    print("=== energy capture of a1's gradient (block token layers) ===")
    for key in (
        "cap_a1_by_a1half",
        "cap_a1_by_a2",
        "cap_a1half_by_a2half",
        "cap_a1_by_wsvd",
        "cap_a1_by_random",
    ):
        print(f"  {key:22s} {blocks_only[key]:.4f}")
    print("=== × null ===")
    for key, v in excess.items():
        print(f"  {key:16s} {v:6.1f}×")
    print("=== by kind: grad_a1_vs_a2 / split_a1 / split_a2 / a1_vs_wsvd / null ===")
    for k, d in by_kind.items():
        print(
            f"  {k:36s} n={d['n']:3d}  {d['grad_a1_vs_a2']:.3f} / {d['split_a1']:.3f} / "
            f"{d['split_a2']:.3f} / {d['a1_vs_wsvd']:.3f} / {d['null']:.3f}"
        )

    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=artifacts,
        device=device,
    )
    print(f"envelope: {out}")


if __name__ == "__main__":
    main()
