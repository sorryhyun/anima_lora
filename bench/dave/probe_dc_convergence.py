#!/usr/bin/env python
"""DAVE premise probe — is the per-block DC the shared/locked component vs the AC?

DAVE (DC Attenuation for diVersity Enhancement, ICML'26, training-free) claims
that the *spatial average* (DC component) of intermediate Transformer-block
features `h^l` converges across seeds early in generation, pinning the global
layout and collapsing same-prompt diversity. This read-only probe tests that
diagnosis on Anima.

What it does
------------
Generates the SAME prompt across N seeds (real inference via ``generate()``),
and for each (denoising step, block) captures from the **cond** forward:

  * DC vector      μ = mean over spatial/token axes (per channel)   -> (D,)
  * AC residual    (h − μ) avg-pooled to GxG per channel            -> (D·G·G,)
  * power ratio    ||μ||² · (T·H·W) / ||h||²                        -> scalar

DC and AC are **disjoint by construction** (the AC residual has the DC removed),
so their cross-seed cosine similarities measure genuinely different components —
unlike a plain coarse pool, which contains the DC and understates the gap.

Per (step, block) it measures the cross-seed cosine similarity of DC and of AC.
The DAVE premise predicts, *in the blocks where the DC carries real energy*
(high power ratio): DC sim ≫ AC sim — the DC is the shared, non-diversifying
part while the AC carries the seed-specific structure. The verdict is therefore
restricted to the **DC-heavy blocks** (block-averaging smears the bimodal
per-block power structure — see the power-ratio heatmap).

If DC sim ≈ AC sim even in DC-heavy blocks, the seeds don't diversify via the AC
and attenuating the DC won't help.

Read-only: post-hooks only, eager 5D forwards, compile off. Outputs a
``result.json``, three PNGs, and a ``per_block.npz`` of the (S,L) matrices.

Arms (MOD / DAVE compose)
-------------------------
``--mod`` (mod-guidance via pooled_text_proj, stock step_i8_skip27 schedule) and
``--dave`` (DC attenuation) turn the probe into the compose-mechanism experiment:
does MOD deepen the early DC lock, and does DAVE restore the AC gap under MOD?
Run the same seeds/prompt across arms and compare ``per_block.npz``.

Capture semantics under ``--dave``: the probe's hooks are registered *before*
``generate()`` arms DAVE's hooks, so each block's capture is the block's own
(pre-attenuation) output — while its *input* already carries the propagated
effect of attenuation at earlier blocks and diverged latents from earlier steps.
That is deliberate: the probe measures the **emergent** unlock of the network
state, not the mechanical subtraction DAVE itself applies.

Usage
-----
    .venv/bin/python bench/dave/probe_dc_convergence.py --seeds 8 --steps 24 --cfg 4.0
    .venv/bin/python bench/dave/probe_dc_convergence.py --seeds 12 --label teddy \
        --prompt "a girl in a plain dress next to a white teddy bear"
    # compose arms (same seeds, then diff the result.json / per_block.npz):
    .venv/bin/python bench/dave/probe_dc_convergence.py --lora latest --label base
    .venv/bin/python bench/dave/probe_dc_convergence.py --lora latest --mod auto
    .venv/bin/python bench/dave/probe_dc_convergence.py --lora latest --mod auto --dave auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from anima_lora import (
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_dit_model,
    prepare_text_inputs,
)

# Mirror the `make test` prompt (scripts/tasks/_common.py::INFERENCE_BASE) so the
# probe reads on the same distribution we eyeball test images on.
from bench._anima import DEFAULT_NEG as DEFAULT_NEGATIVE
from bench._anima import DEFAULT_PROMPT
from bench._common import make_run_dir, write_result


# --------------------------------------------------------------------------- #
# Capture state machine                                                        #
# --------------------------------------------------------------------------- #
class DCProbe:
    """Per-block post-hook capture, keyed to (step, block) of the cond forward.

    Block firing order is deterministic (0..L-1) per forward; under CFG each
    sampler step is two forwards (cond then uncond — see generation.py:786/809).
    We tick a forward counter when block 0 fires and capture only the even
    (cond) forwards. No edits to the sampler loop.
    """

    def __init__(self, num_blocks: int, num_steps: int, do_cfg: bool, ac_grid: int):
        self.L = num_blocks
        self.S = num_steps
        self.do_cfg = do_cfg
        self.G = ac_grid
        self.capturing = False
        self.forward_idx = -1
        # store[kind][seed] -> dict[(step, block)] = tensor
        self._dc: list[dict] = []
        self._ac: list[dict] = []
        self._ratio: list[dict] = []

    def start_seed(self):
        self.forward_idx = -1
        self.capturing = True
        self._dc.append({})
        self._ac.append({})
        self._ratio.append({})

    def stop(self):
        self.capturing = False

    def make_hook(self, bidx: int):
        def hook(_module, _inputs, output):
            if bidx == 0:
                self.forward_idx += 1
            if not self.capturing:
                return
            fi = self.forward_idx
            if self.do_cfg:
                if fi % 2 != 0:  # uncond forward — skip
                    return
                step = fi // 2
            else:
                step = fi
            if step >= self.S:
                return
            h = output.detach().float()  # (B, T, H, W, D)
            B, T, H, W, D = h.shape
            dc = h.mean(dim=(1, 2, 3))  # (B, D) — the DC component
            total_p = (h * h).sum(dim=(1, 2, 3, 4)).clamp_min(1e-12)  # (B,)
            dc_p = (dc * dc).sum(dim=1) * (T * H * W)  # (B,)
            ratio = dc_p / total_p  # (B,)
            # AC residual = h − μ (DC removed), avg-pooled to GxG per channel.
            ac = h - dc.view(B, 1, 1, 1, D)
            ac = ac.permute(0, 4, 1, 2, 3).reshape(B, D * T, H, W)
            ac = F.adaptive_avg_pool2d(ac, (self.G, self.G)).reshape(B, -1)
            # B==1 for single-image generation; average defensively if not.
            self._dc[-1][(step, bidx)] = dc.mean(0).cpu()
            self._ac[-1][(step, bidx)] = ac.mean(0).cpu()
            self._ratio[-1][(step, bidx)] = float(ratio.mean())

        return hook

    # ---- aggregation ----
    def _stack(self, store, s, bl):
        """(N, dim) over seeds that captured this (step, block)."""
        rows = [d[(s, bl)] for d in store if (s, bl) in d]
        return torch.stack(rows, 0) if rows else None

    @staticmethod
    def _mean_pairwise_cos(X: torch.Tensor) -> float:
        """Mean off-diagonal cosine similarity across the N rows."""
        if X is None or X.shape[0] < 2:
            return float("nan")
        Xn = F.normalize(X.float(), dim=1, eps=1e-8)
        G = Xn @ Xn.t()  # (N, N)
        n = X.shape[0]
        off = (G.sum() - torch.diagonal(G).sum()) / (n * (n - 1))
        return float(off)

    def curves(self):
        """(S,L) matrices: dc_sim, ac_sim, power_ratio."""
        dc_sim = torch.full((self.S, self.L), float("nan"))
        ac_sim = torch.full((self.S, self.L), float("nan"))
        pratio = torch.full((self.S, self.L), float("nan"))
        for s in range(self.S):
            for bl in range(self.L):
                dc_sim[s, bl] = self._mean_pairwise_cos(self._stack(self._dc, s, bl))
                ac_sim[s, bl] = self._mean_pairwise_cos(self._stack(self._ac, s, bl))
                rr = [d[(s, bl)] for d in self._ratio if (s, bl) in d]
                if rr:
                    pratio[s, bl] = sum(rr) / len(rr)
        return dc_sim, ac_sim, pratio


def _nan_std_over_blocks(M: torch.Tensor) -> np.ndarray:
    """Per-step std across blocks, ignoring NaNs."""
    return np.array(
        [
            float(torch.std(row[~torch.isnan(row)]))
            if (~torch.isnan(row)).sum() > 1
            else 0.0
            for row in M
        ]
    )


# --------------------------------------------------------------------------- #
# Plotting                                                                     #
# --------------------------------------------------------------------------- #
def _plot(run_dir: Path, dc_sim, ac_sim, pratio, heavy_blocks) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable — skipping plots")
        return []

    steps = list(range(dc_sim.shape[0]))

    def band(ax, M, cols, color, label):
        sub = M[:, cols] if len(cols) else M
        m = torch.nanmean(sub, dim=1).numpy()
        sd = _nan_std_over_blocks(sub)
        ax.plot(steps, m, color=color, label=label, lw=2)
        ax.fill_between(steps, m - sd, m + sd, color=color, alpha=0.15)

    all_cols = list(range(dc_sim.shape[1]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, cols, title in (
        (axes[0], all_cols, "all blocks"),
        (axes[1], heavy_blocks, f"DC-heavy blocks {heavy_blocks}"),
    ):
        band(ax, dc_sim, cols, "tab:blue", "DC (spatial mean)")
        band(ax, ac_sim, cols, "tab:orange", "AC residual (h−μ)")
        ax.set_xlabel("denoising step")
        ax.set_title(title)
        ax.axhline(0.9, ls="--", c="gray", lw=0.8)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("cross-seed cosine similarity")
    fig.suptitle("Cross-seed convergence: DC vs AC (block-avg ± std)")
    p1 = run_dir / "convergence_curves.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(
        pratio.t().numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    for bl in heavy_blocks:
        ax.axhline(bl, c="cyan", lw=0.4, alpha=0.5)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("block index")
    ax.set_title("DC power ratio  ||DC||² / ||h||²  (cyan = DC-heavy)")
    fig.colorbar(im, ax=ax)
    p2 = run_dir / "power_ratio_heatmap.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=130)
    plt.close(fig)

    # DC−AC gap heatmap: where (block × step) is DC most ahead of AC?
    fig, ax = plt.subplots(figsize=(7, 4.2))
    gap = (dc_sim - ac_sim).t().numpy()
    im = ax.imshow(
        gap,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-0.3,
        vmax=0.3,
        interpolation="nearest",
    )
    ax.set_xlabel("denoising step")
    ax.set_ylabel("block index")
    ax.set_title("DC sim − AC sim  (red = DC more cross-seed-locked than AC)")
    fig.colorbar(im, ax=ax)
    p3 = run_dir / "dc_minus_ac_gap.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=130)
    plt.close(fig)

    return [p1.name, p2.name, p3.name]


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE)
    p.add_argument("--seeds", type=int, default=8, help="number of seeds (N)")
    p.add_argument(
        "--seed0", type=int, default=1000, help="first seed; uses seed0..seed0+N-1"
    )
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument(
        "--ac_grid", type=int, default=4, help="GxG avg-pool grid for the AC residual"
    )
    p.add_argument(
        "--heavy_ratio",
        type=float,
        default=0.30,
        help="block counts as DC-heavy if its early-step mean power ratio ≥ this",
    )
    p.add_argument(
        "--lora",
        default=None,
        help="LoRA .safetensors to arm ('latest' = newest non-pooled ckpt in "
        "output/ckpt/; default: bare DiT, the original Phase-0 setup)",
    )
    p.add_argument(
        "--mod",
        default=None,
        help="MOD arm: pooled_text_proj .safetensors ('auto' = newest "
        "output/ckpt/pooled_text_proj*). Stock per-block schedule (step_i8_skip27).",
    )
    p.add_argument("--mod_w", type=float, default=3.0, help="mod-guidance strength w")
    p.add_argument(
        "--dave",
        default=None,
        help="DAVE arm: mask npz path or 'auto' (shipped flat 8-18 pool)",
    )
    p.add_argument("--dave_strength", type=float, default=0.5)
    p.add_argument("--dave_tau", type=float, default=0.10)
    p.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="verbatim inference.py argv passthrough (e.g. --extra --mod_taper 4)",
    )
    p.add_argument("--label", default=None)
    opts = p.parse_args()

    # Resolve arm checkpoints + auto-label so result dirs are self-describing.
    if opts.lora == "latest":
        from scripts.tasks._common import latest_lora

        opts.lora = str(latest_lora())
    if opts.mod == "auto":
        from scripts.tasks._common import latest_output

        opts.mod = str(latest_output("pooled_text_proj"))
    if opts.label is None:
        arm = [s for s, on in (("mod", opts.mod), ("dave", opts.dave)) if on]
        opts.label = "-".join(arm) if arm else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a canonical inference args Namespace (every getattr() knob populated).
    extra_argv = []
    if opts.mod:
        extra_argv += ["--mod_w", str(opts.mod_w)]
    if opts.dave:
        extra_argv += [
            "--dave",
            opts.dave,
            "--dave_strength",
            str(opts.dave_strength),
            "--dave_tau",
            str(opts.dave_tau),
        ]
    extra_argv += opts.extra
    ckpts = default_checkpoints()
    req = GenerationRequest(
        dit=ckpts.dit,
        vae=ckpts.vae,
        text_encoder=ckpts.text_encoder,
        prompt=opts.prompt,
        negative_prompt=opts.negative_prompt,
        save_path="output/tests/_dave_probe.png",  # unused (we never decode)
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=tuple(opts.size),
        seed=opts.seed0,
        lora_weight=[opts.lora] if opts.lora else None,
        pooled_text_proj=opts.mod,
        extra_argv=extra_argv,
    )
    args = req.to_args()
    args.device = device
    args.compile = False  # eager 5D blocks — hooks see (B,T,H,W,D), no recompiles
    args.compile_blocks = False

    gen_settings = get_generation_settings(args)

    print("[dave-probe] loading DiT…")
    anima = load_dit_model(args, device, torch.bfloat16)
    num_blocks = len(anima.blocks)
    do_cfg = opts.cfg != 1.0
    print(
        f"[dave-probe] {num_blocks} blocks, steps={opts.steps}, cfg={opts.cfg} "
        f"(do_cfg={do_cfg}), seeds={opts.seeds}, ac_grid={opts.ac_grid}"
    )

    probe = DCProbe(num_blocks, opts.steps, do_cfg, opts.ac_grid)
    handles = [
        blk.register_forward_hook(probe.make_hook(i))
        for i, blk in enumerate(anima.blocks)
    ]

    shared = {"model": anima}
    if opts.mod:
        # setup_mod_guidance runs per generate() call; keep the text encoder in
        # shared_models so it's loaded once (it parks itself on CPU between calls).
        from library.inference.models import load_text_encoder

        print("[dave-probe] loading text encoder (shared, MOD arm)…")
        shared["text_encoder"] = load_text_encoder(
            args, dtype=torch.bfloat16, device=device
        )
        shared["text_encoder"].eval()

    # Encode text once (prompt identical across seeds) — reused via precomputed.
    print("[dave-probe] encoding text…")
    context, context_null = prepare_text_inputs(args, device, anima, shared)
    text_data = {"context": context, "context_null": context_null}
    try:
        for k in range(opts.seeds):
            seed = opts.seed0 + k
            args.seed = seed
            probe.start_seed()
            print(f"[dave-probe] seed {k + 1}/{opts.seeds} (seed={seed})")
            generate(
                args,
                gen_settings,
                shared_models=shared,
                precomputed_text_data=text_data,
            )
            probe.stop()
    finally:
        for h in handles:
            h.remove()

    dc_sim, ac_sim, pratio = probe.curves()

    early = max(1, opts.steps // 3)  # early window = first 1/3 of steps

    # DC-heavy blocks: those whose early-step mean power ratio clears the bar.
    # These are the only blocks where attenuating the DC could matter.
    early_block_ratio = torch.nanmean(pratio[:early], dim=0)  # (L,)
    heavy_blocks = [
        bl
        for bl in range(num_blocks)
        if float(early_block_ratio[bl]) >= opts.heavy_ratio
    ]

    def col_avg(M, cols, rows):
        sub = M[:rows][:, cols] if cols else M[:rows]
        return float(torch.nanmean(sub))

    dc_heavy = col_avg(dc_sim, heavy_blocks, early)
    ac_heavy = col_avg(ac_sim, heavy_blocks, early)
    dc_all = col_avg(dc_sim, [], early)
    ac_all = col_avg(ac_sim, [], early)

    arm = (
        "+".join(s for s, on in (("mod", opts.mod), ("dave", opts.dave)) if on)
        or "vanilla"
    )
    metrics = {
        "arm": arm,
        "lora": opts.lora,
        "mod": opts.mod,
        "mod_w": opts.mod_w if opts.mod else None,
        "dave": opts.dave,
        "dave_strength": opts.dave_strength if opts.dave else None,
        "dave_tau": opts.dave_tau if opts.dave else None,
        "num_blocks": num_blocks,
        "num_steps": opts.steps,
        "num_seeds": opts.seeds,
        "cfg": opts.cfg,
        "ac_grid": opts.ac_grid,
        "heavy_ratio_thr": opts.heavy_ratio,
        "heavy_blocks": heavy_blocks,
        "n_heavy_blocks": len(heavy_blocks),
        # all-block (reference — smears the bimodal power structure)
        "dc_sim_all_early": dc_all,
        "ac_sim_all_early": ac_all,
        "dc_minus_ac_all_early": dc_all - ac_all,
        # DC-heavy blocks (the verdict-bearing numbers)
        "dc_sim_heavy_early": dc_heavy,
        "ac_sim_heavy_early": ac_heavy,
        "dc_minus_ac_heavy_early": dc_heavy - ac_heavy,
        "power_ratio_heavy_early": col_avg(pratio, heavy_blocks, early),
        # per-step block-averaged curves
        "dc_sim_per_step": [float(x) for x in torch.nanmean(dc_sim, dim=1).tolist()],
        "ac_sim_per_step": [float(x) for x in torch.nanmean(ac_sim, dim=1).tolist()],
        "power_ratio_per_step": [
            float(x) for x in torch.nanmean(pratio, dim=1).tolist()
        ],
    }

    # Premise verdict (DC-heavy blocks only): the DC is highly cross-seed-locked
    # AND clearly more locked than the AC residual that carries seed structure.
    holds = len(heavy_blocks) > 0 and dc_heavy > 0.85 and (dc_heavy - ac_heavy) > 0.05
    metrics["premise_holds"] = bool(holds)

    run_dir = make_run_dir("dave", label=opts.label)
    artifacts = _plot(run_dir, dc_sim, ac_sim, pratio, heavy_blocks)
    np.savez(
        run_dir / "per_block.npz",
        dc_sim=dc_sim.numpy(),
        ac_sim=ac_sim.numpy(),
        power_ratio=pratio.numpy(),
        heavy_blocks=np.array(heavy_blocks),
    )
    artifacts.append("per_block.npz")
    write_result(
        run_dir,
        script=__file__,
        args=opts,
        metrics=metrics,
        label=opts.label,
        artifacts=artifacts,
        device=device,
    )

    print("\n" + "=" * 66)
    print(f"  arm: {arm}" + (f"  (lora={Path(opts.lora).name})" if opts.lora else ""))
    print(f"  DC-heavy blocks (ratio≥{opts.heavy_ratio}): {heavy_blocks}")
    print(
        f"  -- all blocks (early) --   DC {dc_all:.3f} | AC {ac_all:.3f} "
        f"| gap {dc_all - ac_all:+.3f}"
    )
    print(
        f"  -- DC-heavy   (early) --   DC {dc_heavy:.3f} | AC {ac_heavy:.3f} "
        f"| gap {dc_heavy - ac_heavy:+.3f}   <-- verdict-bearing"
    )
    print(f"  DC power ratio (heavy, early): {metrics['power_ratio_heavy_early']:.3f}")
    print(f"  PREMISE HOLDS:                 {holds}")
    print("=" * 66)
    print(f"→ {run_dir}")


if __name__ == "__main__":
    main()
