#!/usr/bin/env python3
"""P0b of turbo_tau_split_critic — the τ-specialization-headroom (interference) probe.

Measures whether the single fake/critic LoRA's per-τ-band error is dominated by
CROSS-BAND GRADIENT INTERFERENCE — the only mechanism a τ-split critic
(`fake_tau_banks = 2`) could harvest. Offline, zero training-loop changes: loads
a crash-resume bundle (`scripts/distill_turbo/resume.py`, which carries the fake
weights + optimizer moments), freezes the student, and runs a controlled
fine-tune of the critic four ways from the identical restore point:

    lo   — τ drawn only from [0, boundary)
    hi   — τ drawn only from [boundary, 1]
    ctl  — τ from the run's full t_distribution (the status quo)
    ctl2 — same as ctl, different RNG seed → the noise floor

Each arm takes the same number of fake-only updates at the bundle's current
post-warmup LR (no schedule), on the same data stream (separate RNG generators
for data vs τ, so lo/hi/ctl share batches, ε draws, and student x_pred exactly).
Per-τ-bin loss on a fixed held-out probe set is measured before and after.

Pre-registered gate (see _archive/proposals/turbo_tau_split_critic.md):

  G1 (headroom): lo beats ctl on the lo band AND hi beats ctl on the hi band,
      each by more than the ctl-vs-ctl2 spread on that band.
  G2 (sanity):   lo degrades on the hi band and vice versa — otherwise the
      bands share one solution and a numerically-passing G1 is spurious.

FIRE (G1 && G2) unlocks Phase 1; G1 fail closes the line.

Optional capacity arm: ``--fake_rank 48`` SVD-truncates the bundle's fake to
r48 before the arms run (fresh Adam moments, scale-preserving alpha) — r48 ≈
r96 per-band means capacity is slack (negative G1 unsurprising); r48 clearly
worse means capacity binds elsewhere.

    python bench/turbo/tau_critic_probe.py \\
        --bundle output/ckpt/anima_superturbo_B/anima_superturbo_B_resume.pt
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from tqdm import tqdm  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.config.resolved import load_toml  # noqa: E402
from library.datasets.cache import CachedDataset  # noqa: E402
from library.inference.sampling import get_timesteps_sigmas  # noqa: E402
from library.runtime.dynamo import pin_dynamo_limit  # noqa: E402
from library.runtime.harness import (  # noqa: E402
    compile_dit_blocks,
    compile_signature,
    enable_training_grad_ckpt,
    isolate_compile_cache,
    place_dit_for_training,
)
from library.training.distill_runtime import (  # noqa: E402
    ensure_dynamic_seq_for_freefit,
    resolve_device_dtype,
)
from networks.lora_anima.factory import create_network  # noqa: E402
from networks.methods.turbo_dmd import (  # noqa: E402
    TurboDMDNetwork,
    warm_start_plain_lora,
)
from scripts.distill_turbo.setup import _cached_token_counts  # noqa: E402
from scripts.distill_turbo.primitives import (  # noqa: E402
    PadCache,
    make_collate,
    renoise,
)
from scripts.distill_turbo.resume import load_resume_state  # noqa: E402

logger = logging.getLogger("tau_critic_probe")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

CANONICAL_ARMS = ("lo", "hi", "ctl", "ctl2")


def default_snapshot_for(bundle: Path) -> Path:
    """``<dir>/<name>/<name>_resume.pt`` → ``<dir>/<name>.snapshot.toml``."""
    return bundle.parent.parent / f"{bundle.parent.name}.snapshot.toml"


def state_checksum(sd: dict) -> float:
    """Cheap weight-identity fingerprint: Σ|w| over every tensor."""
    return float(sum(v.detach().float().abs().sum().item() for v in sd.values()))


def band_bins(nbins: int, boundary: float) -> tuple[list[int], list[int]]:
    """Partition bin indices by whether the bin CENTER falls below the boundary."""
    lo = [i for i in range(nbins) if (i + 0.5) / nbins < boundary]
    hi = [i for i in range(nbins) if i not in lo]
    if not lo or not hi:
        raise ValueError(f"boundary={boundary} leaves an empty band at nbins={nbins}")
    return lo, hi


def band_mean(per_bin: list[float], bins: list[int]) -> float:
    return sum(per_bin[i] for i in bins) / len(bins)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--bundle",
        required=True,
        help="Crash-resume bundle ({name}_resume.pt) carrying student+fake+opts.",
    )
    p.add_argument(
        "--snapshot",
        default=None,
        help="The run's .snapshot.toml (default: derived from the bundle path). "
        "Source of the run's exact config — data_dir, t_distribution, grids.",
    )
    p.add_argument("--probe_size", type=int, default=256)
    p.add_argument("--nbins", type=int, default=8)
    p.add_argument("--updates", type=int, default=400)
    p.add_argument("--boundary", type=float, default=0.5)
    p.add_argument(
        "--arms",
        default=",".join(CANONICAL_ARMS),
        help="Comma list from {lo,hi,ctl,ctl2}. The gate needs all four.",
    )
    p.add_argument(
        "--fake_rank",
        type=int,
        default=0,
        help="0 = bundle rank. Else SVD-truncate the bundle's fake to this rank "
        "(capacity arm; fresh Adam moments, scale-preserving alpha).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.0,
        help="0 = the bundle fake optimizer's current (post-warmup) LR.",
    )
    p.add_argument("--data_dir", default=None, help="Override the snapshot's pool.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no_compile",
        action="store_true",
        help="Force eager (default: follow the snapshot's torch_compile).",
    )
    p.add_argument("--label", default="tau-critic-probe")
    args = p.parse_args()

    bundle_path = Path(args.bundle)
    snapshot_path = (
        Path(args.snapshot) if args.snapshot else default_snapshot_for(bundle_path)
    )
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"run snapshot not found: {snapshot_path} — pass --snapshot explicitly."
        )
    snap = load_toml(str(snapshot_path))
    state = load_resume_state(bundle_path)
    fp = state.get("fingerprint", {})
    if snap.get("output_name") != state.get("output_name"):
        logger.warning(
            f"snapshot output_name={snap.get('output_name')!r} != bundle "
            f"output_name={state.get('output_name')!r} — mixed-run inputs?"
        )
    # The snapshot is the same run's resolved dump, so it must agree with the
    # bundle fingerprint on topology. A mismatch means a wrong --snapshot.
    for k in ("student_rank", "fake_rank", "base_loss"):
        if k in fp and k in snap and fp[k] != snap[k]:
            raise SystemExit(
                f"bundle fingerprint {k}={fp[k]!r} != snapshot {k}={snap[k]!r} "
                "— snapshot is not from the bundle's run."
            )

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in CANONICAL_ARMS:
            raise SystemExit(f"unknown arm {a!r}; expected one of {CANONICAL_ARMS}")

    torch.manual_seed(args.seed)
    device, dtype = resolve_device_dtype()

    bundle_fake_rank = int(snap["fake_rank"])
    bundle_fake_alpha = float(snap.get("fake_alpha", bundle_fake_rank))
    fake_rank = args.fake_rank or bundle_fake_rank
    if fake_rank > bundle_fake_rank:
        raise SystemExit(
            f"--fake_rank {fake_rank} > bundle rank {bundle_fake_rank}: the "
            "capacity arm only truncates."
        )
    # Scale-preserving alpha: keep alpha/rank equal to the bundle's so the
    # truncated critic's effective ΔW scaling (and factor-space LR geometry)
    # matches the r-full arms.
    fake_alpha = bundle_fake_alpha * fake_rank / bundle_fake_rank

    torch_compile = bool(snap.get("torch_compile", False)) and not args.no_compile
    if torch_compile:
        pin_dynamo_limit("recompile_limit", int(snap.get("dynamo_recompile_limit", 64)))
    torch.set_num_threads(torch.get_num_threads())

    logger.info(f"loading DiT: {snap['dit_path']}")
    blocks_to_swap = int(snap.get("blocks_to_swap", 0))
    model = anima_utils.load_anima_model(
        device,
        snap["dit_path"],
        attn_mode=snap.get("attn_mode", "flash"),
        loading_device="cpu" if blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )
    place_dit_for_training(model, device, blocks_to_swap=blocks_to_swap)
    enable_training_grad_ckpt(model, enabled=bool(snap.get("grad_ckpt", False)))

    # Capacity-arm export: reconstruct the bundle's fake on the PRISTINE DiT
    # (constructed, never applied — a post-apply_to create_network would see the
    # LoRA modules' own internal Linears as wrap targets), save it as a plain
    # LoRA file for the SVD-truncating warm start below, then drop it.
    run_dir = make_run_dir("turbo", label=args.label)
    fake_export = None
    if fake_rank != bundle_fake_rank:
        donor = create_network(
            multiplier=1.0,
            network_dim=bundle_fake_rank,
            network_alpha=bundle_fake_alpha,
            vae=None,
            text_encoders=[],
            unet=model,
            channel_scaling_alpha=float(snap.get("channel_scaling_alpha", 0.0)),
        )
        donor.load_state_dict(state["fake"])
        fake_export = run_dir / "fake_bundle_export.safetensors"
        donor.save_weights(str(fake_export), torch.float32, None)
        del donor
        logger.info(
            f"capacity arm: exported bundle fake (r{bundle_fake_rank}) → "
            f"{fake_export}; probe runs at r{fake_rank} (alpha {fake_alpha:g})"
        )

    turbo = TurboDMDNetwork(
        unet=model,
        student_rank=int(snap["student_rank"]),
        fake_rank=fake_rank,
        student_alpha=float(snap.get("student_alpha", snap["student_rank"])),
        fake_alpha=fake_alpha,
        channel_scaling_alpha=float(snap.get("channel_scaling_alpha", 0.0)),
    )
    turbo.freeze_dit()
    turbo.student.to(device=device, dtype=dtype)
    turbo.fake.to(device=device, dtype=dtype)

    turbo.student.load_state_dict(state["student"])
    if fake_export is None:
        turbo.fake.load_state_dict(state["fake"])
    else:
        warm_start_plain_lora(turbo.fake, str(fake_export), f"fake-r{fake_rank}")
    # The probe never trains the student — hard-freeze so a wiring mistake
    # errors instead of silently drifting the x_pred distribution.
    for p_ in turbo.student.parameters():
        p_.requires_grad_(False)

    # COMPILE LAST (harness invariant) — same self-describing seq-budget dance
    # as scripts/distill_turbo/distill.py.
    data_dir = args.data_dir or snap["data_dir"]
    counts = _cached_token_counts(data_dir)
    n_token_families = len(counts) or None
    seq_range = (min(counts), max(counts)) if counts else None
    dynamic_seq = bool(snap.get("compile_dynamic_seq", True))
    if torch_compile and not dynamic_seq:
        dynamic_seq = ensure_dynamic_seq_for_freefit(counts, dynamic_seq, logger=logger)
    if torch_compile:
        budget = float(snap.get("activation_memory_budget", 1.0))
        if budget < 1.0 and not bool(snap.get("grad_ckpt", False)):
            import torch._functorch.config as _functorch_config

            _functorch_config.activation_memory_budget = budget
        isolate_compile_cache(
            compile_signature(
                n_token_families=n_token_families,
                seq_range=seq_range,
                dynamic_seq=dynamic_seq,
                mode="",
            )
        )
    compile_dit_blocks(
        model,
        enabled=torch_compile,
        mode="",
        dynamic_seq=dynamic_seq,
        n_token_families=n_token_families,
        seq_range=seq_range,
    )
    if torch_compile:
        lim = int(snap.get("dynamo_recompile_limit", 64))
        pin_dynamo_limit("recompile_limit", lim)
        pin_dynamo_limit("accumulated_recompile_limit", len(model.blocks) * lim)

    # Probe LR: the bundle optimizer's live (post-warmup, post-cosine-progress)
    # value — "the run continues, only τ is restricted".
    probe_lr = args.lr or float(state["fake_opt"]["param_groups"][0]["lr"])
    grad_clip = float(snap.get("grad_clip", 0.0))
    t_distribution = str(snap.get("t_distribution", "uniform"))
    sigmoid_scale = float(snap.get("sigmoid_scale", 1.0))
    student_steps = int(snap["student_steps"])
    base_loss = str(snap.get("base_loss", "dpdmd"))
    dmd_grad_step = str(snap.get("dmd_grad_step", "all"))
    logger.info(
        f"probe: lr={probe_lr:.3g} updates={args.updates} probe_size="
        f"{args.probe_size} nbins={args.nbins} boundary={args.boundary} "
        f"t_distribution={t_distribution} arms={arms}"
    )

    dataset = CachedDataset(
        data_dir,
        batch_size=1,
        sample_ratio=float(snap.get("sample_ratio", 1.0)),
        mask_dir=None,  # the fake loss is unmasked in the training loop too
        need_pooled=False,
    )

    def make_loader():
        """Fresh loader per arm → identical batch order for every arm."""
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=dataset.make_batch_sampler(shuffle=True, seed=args.seed),
            num_workers=2,
            pin_memory=True,
            collate_fn=make_collate(),
        )

    pad_cache = PadCache(dtype)

    def _forward(view: str, x, t_b, c, *, no_grad: bool):
        turbo.set_view(view)
        if model.blocks_to_swap:
            model.prepare_block_swap_before_forward(free_cache=False)
        pad = pad_cache.get(x)
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx, torch.autocast("cuda", dtype=dtype):
            return model.forward_mini_train_dit(
                x.unsqueeze(2), t_b, c, padding_mask=pad, skip_pooled_text_proj=True
            )

    sigmas = get_timesteps_sigmas(student_steps, float(snap["flow_shift"]), "cpu")[
        1
    ].tolist()

    # x_pred exactly as the training-loop critic sees it: backward-simulate the
    # frozen student to grid step g, take the one-step x0-prediction there
    # (x_g − σ_g·v_g; at g = N−1 this equals the Euler endpoint). g's
    # distribution mirrors the run's grad_step policy — those are the grid
    # points whose x_pred the fake actually trained against.
    g_low = (1 if base_loss == "dpdmd" else 0) if dmd_grad_step == "random" else None

    def draw_g(gen: torch.Generator) -> int:
        if g_low is None or g_low == student_steps - 1:
            return student_steps - 1
        return int(torch.randint(g_low, student_steps, (1,), generator=gen).item())

    @torch.no_grad()
    def student_x_pred(eps0: torch.Tensor, c: torch.Tensor, g: int) -> torch.Tensor:
        x = eps0
        for i in range(g):
            t_b = torch.full((1,), sigmas[i], device=device, dtype=dtype)
            v = _forward("student", x, t_b, c, no_grad=True).squeeze(2)
            x = x - (sigmas[i] - sigmas[i + 1]) * v
        t_b = torch.full((1,), sigmas[g], device=device, dtype=dtype)
        v_g = _forward("student", x, t_b, c, no_grad=True).squeeze(2)
        return (x - sigmas[g] * v_g).detach()

    def draw_tau_band(lo: float, hi: float, gen: torch.Generator) -> torch.Tensor:
        """One τ from the run's t_distribution restricted to [lo, hi).

        Rejection sampling keeps the within-band density identical to the ctl
        arm's (a plain U[lo,hi) would silently change it under "sigmoid").
        """
        if t_distribution == "uniform":
            return lo + (hi - lo) * torch.rand(1, generator=gen)
        while True:
            cand = torch.sigmoid(sigmoid_scale * torch.randn(64, generator=gen))
            hit = cand[(cand >= lo) & (cand < hi)]
            if hit.numel():
                return hit[:1]

    # --- fixed held-out probe set: probe_size × (x_pred, c, τ, ε), τ stratified ---
    logger.info(f"building probe set ({args.probe_size} tuples)...")
    gen_probe = torch.Generator().manual_seed(args.seed * 9973 + 1)
    probe_set = []
    loader_iter = iter(make_loader())
    while len(probe_set) < args.probe_size:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(make_loader())
            batch = next(loader_iter)
        c = batch["crossattn_emb"][:1].to(device, dtype=dtype)
        lat = batch["latents"][:1]
        eps0 = torch.randn(lat.shape, generator=gen_probe).to(device, dtype=dtype)
        x_pred = student_x_pred(eps0, c, draw_g(gen_probe))
        j = len(probe_set)
        bin_i = j % args.nbins
        u = torch.rand(1, generator=gen_probe)
        tau = (bin_i + u) / args.nbins
        eps_p = torch.randn(lat.shape, generator=gen_probe)
        probe_set.append(
            {
                "x_pred": x_pred.cpu(),
                "c": c.cpu(),
                "tau": tau,
                "eps": eps_p,
                "bin": bin_i,
            }
        )
    del loader_iter

    @torch.no_grad()
    def eval_probe() -> list[float]:
        """Per-bin mean fake loss over the fixed probe set."""
        sums = [0.0] * args.nbins
        cnts = [0] * args.nbins
        for item in probe_set:
            x_pred = item["x_pred"].to(device, dtype=dtype)
            c = item["c"].to(device, dtype=dtype)
            tau = item["tau"].to(device, dtype=dtype)
            eps_p = item["eps"].to(device, dtype=dtype)
            x_t = renoise(x_pred, tau, eps_p)
            t_b = tau.reshape(1)
            v = _forward("fake", x_t, t_b, c, no_grad=True).squeeze(2)
            loss = nn.functional.mse_loss(v.float(), (eps_p - x_pred).float())
            sums[item["bin"]] += float(loss.item())
            cnts[item["bin"]] += 1
        return [s / max(n, 1) for s, n in zip(sums, cnts)]

    fake_ref_checksum = state_checksum(state["fake"]) if fake_export is None else None
    trunc_init_sd = None
    if fake_export is not None:
        # r-truncated restore point: every arm resets to the truncated critic.
        trunc_init_sd = {
            k: v.detach().clone() for k, v in turbo.fake.state_dict().items()
        }

    logger.info("evaluating the bundle critic (pre-fine-tune baseline)...")
    before = eval_probe()
    logger.info(f"before (per-bin): {['%.4f' % v for v in before]}")

    bands = {
        "lo": (0.0, args.boundary),
        "hi": (args.boundary, 1.0),
        "ctl": (0.0, 1.0),
        "ctl2": (0.0, 1.0),
    }
    after: dict[str, list[float]] = {}
    for arm in arms:
        # Restore the identical starting point (weights + moments), then verify.
        if trunc_init_sd is not None:
            turbo.fake.load_state_dict(trunc_init_sd)
        else:
            turbo.fake.load_state_dict(state["fake"])
            got = state_checksum(turbo.fake.state_dict())
            if abs(got - fake_ref_checksum) > 1e-3 * max(abs(fake_ref_checksum), 1.0):
                raise RuntimeError(
                    f"arm {arm}: fake restore checksum mismatch "
                    f"({got} vs {fake_ref_checksum}) — clone reset is broken."
                )
        fake_opt = torch.optim.AdamW(
            turbo.fake_params(),
            lr=probe_lr,
            weight_decay=float(snap.get("weight_decay", 0.0)),
            fused=torch.cuda.is_available(),
        )
        if trunc_init_sd is None:
            # Warm Adam moments: exactly "the run continues, τ restricted".
            fake_opt.load_state_dict(copy.deepcopy(state["fake_opt"]))
            for group in fake_opt.param_groups:
                group["lr"] = probe_lr
        else:
            logger.info(f"arm {arm}: truncated rank → fresh Adam moments")

        # Shared data stream across arms (same seed → same batches, ε, x_pred);
        # τ rides its OWN generator so band restriction never desyncs the data
        # stream. ctl2 shifts BOTH → the run-to-run noise floor.
        off = 1 if arm == "ctl2" else 0
        gen_data = torch.Generator().manual_seed(args.seed * 7919 + 101 + off)
        gen_tau = torch.Generator().manual_seed(args.seed * 104729 + 211 + off)
        lo_b, hi_b = bands[arm]

        data_iter = iter(make_loader())
        tau_hist = [0] * args.nbins
        for _ in tqdm(range(args.updates), desc=f"arm:{arm}"):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(make_loader())
                batch = next(data_iter)
            c = batch["crossattn_emb"][:1].to(device, dtype=dtype)
            lat = batch["latents"][:1]
            torch.compiler.cudagraph_mark_step_begin()
            eps0 = torch.randn(lat.shape, generator=gen_data).to(device, dtype=dtype)
            x_pred_d = student_x_pred(eps0, c, draw_g(gen_data))

            tau_cpu = draw_tau_band(lo_b, hi_b, gen_tau)
            t_val = float(tau_cpu.item())
            if not (lo_b <= t_val < hi_b or (hi_b == 1.0 and t_val == 1.0)):
                raise RuntimeError(
                    f"arm {arm}: τ={t_val} outside its band [{lo_b}, {hi_b})"
                )
            tau_hist[min(int(t_val * args.nbins), args.nbins - 1)] += 1
            tau = tau_cpu.to(device, dtype=dtype)
            eps_f = torch.randn(lat.shape, generator=gen_data).to(device, dtype=dtype)
            x_t = renoise(x_pred_d, tau, eps_f).requires_grad_()
            v_fake = _forward("fake", x_t, tau.reshape(1), c, no_grad=False).squeeze(2)
            fake_loss = nn.functional.mse_loss(
                v_fake.float(), (eps_f - x_pred_d).float()
            )
            fake_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(turbo.fake_params(), max_norm=grad_clip)
            fake_opt.step()
            fake_opt.zero_grad(set_to_none=True)

        # The drawn-τ histogram is the arm's own audit that the band held
        # (the per-draw range assert above is the hard guarantee).
        logger.info(f"arm {arm}: τ histogram per bin {tau_hist}")
        after[arm] = eval_probe()
        logger.info(f"after[{arm}] (per-bin): {['%.4f' % v for v in after[arm]]}")
        del fake_opt

    # --- gate evaluation -----------------------------------------------------
    metrics: dict = {
        "probe_lr": probe_lr,
        "updates": args.updates,
        "probe_size": args.probe_size,
        "nbins": args.nbins,
        "boundary": args.boundary,
        "fake_rank": fake_rank,
        "bundle_fake_rank": bundle_fake_rank,
        "bundle_step": int(state["step"]),
        "t_distribution": t_distribution,
        "before_per_bin": before,
        "after_per_bin": after,
    }
    verdict = "INCOMPLETE (need all of lo/hi/ctl/ctl2 for the gate)"
    if all(a in after for a in CANONICAL_ARMS):
        lo_bins, hi_bins = band_bins(args.nbins, args.boundary)
        bm = band_mean
        spread_lo = abs(bm(after["ctl"], lo_bins) - bm(after["ctl2"], lo_bins))
        spread_hi = abs(bm(after["ctl"], hi_bins) - bm(after["ctl2"], hi_bins))
        imp_lo = bm(after["ctl"], lo_bins) - bm(after["lo"], lo_bins)
        imp_hi = bm(after["ctl"], hi_bins) - bm(after["hi"], hi_bins)
        deg_lo_on_hi = bm(after["lo"], hi_bins) - bm(after["ctl"], hi_bins)
        deg_hi_on_lo = bm(after["hi"], lo_bins) - bm(after["ctl"], lo_bins)
        g1 = imp_lo > spread_lo and imp_hi > spread_hi
        g2 = deg_lo_on_hi > 0 and deg_hi_on_lo > 0
        verdict = (
            "FIRE — Phase 1 unlocked"
            if g1 and g2
            else (
                "G1_FAIL — close the line"
                if not g1
                else "G2_FAIL — bands share one solution"
            )
        )
        metrics["gate"] = {
            "improvement_lo_band": imp_lo,
            "improvement_hi_band": imp_hi,
            "noise_floor_lo_band": spread_lo,
            "noise_floor_hi_band": spread_hi,
            "degradation_lo_arm_on_hi_band": deg_lo_on_hi,
            "degradation_hi_arm_on_lo_band": deg_hi_on_lo,
            "G1_headroom": g1,
            "G2_forgetting": g2,
        }
    metrics["verdict"] = verdict
    logger.info(f"verdict: {verdict}")
    if "gate" in metrics:
        for k, v in metrics["gate"].items():
            logger.info(f"  {k}: {v}")

    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=[fake_export] if fake_export else [],
        device=device,
        extra={"bundle": str(bundle_path), "snapshot": str(snapshot_path)},
    )
    logger.info(f"wrote {out}")


if __name__ == "__main__":
    main()
