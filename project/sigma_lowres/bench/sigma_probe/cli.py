"""CLI surface of ``run_sigma_probe.py``: the flags, and the cross-flag
validation that turns them into a :class:`RunConfig`.

``resolve_run_config`` holds every rule that reads more than one flag (mode
forcing, incompatibilities, the seed-block budget) so the driver's ``main``
stays orchestration. It **mutates** ``args`` where a mode forces a grid
(``--draw_sweep`` → endpoint-only, ``--probe_list`` → num_images), exactly as
the single-file version did — the mutated values are what lands in
``result.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project.sigma_lowres.bench.tier_routing.run_grad_probe import DIT, VAE  # noqa: E402

log = logging.getLogger(__name__)


def parse_args(
    doc: str | None = None, argv: list[str] | None = None
) -> argparse.Namespace:
    """The probe's argparse surface. ``doc`` supplies the ``--help`` summary
    line (the driver passes its own module docstring)."""
    p = argparse.ArgumentParser(description=(doc or __doc__).split("\n", 1)[0])
    p.add_argument("--adapter", required=True, help="trained plain-LoRA checkpoint")
    p.add_argument("--dit", default=DIT)
    p.add_argument("--vae", default=VAE)
    p.add_argument("--num_images", type=int, default=40)
    p.add_argument("--bins", type=int, default=8, help="uniform σ bins on (0,1)")
    p.add_argument("--draws_per_bin", type=int, default=8)
    p.add_argument(
        "--sigma_window",
        default="0,1",
        help="LO,HI sub-interval the uniform bins cover (e.g. 0.5,1.0 puts "
        "every bin in the high-σ crossover region); --endpoint_bin unaffected. "
        "Segmented form (E13): ':'-separated LO,HI,BINS segments — "
        "'0,0.1,4 : 0.1,0.9,6 : 0.9,1.0,4' resolves both ends of the curve "
        "densely in ONE process. Segments must be sorted, non-overlapping and "
        "inside [0,1]; --bins is then ignored (it is set to the total) and "
        "draws-per-bin stays global (the estimator grid is rectangular)",
    )
    p.add_argument("--tier", type=int, default=1024)
    p.add_argument("--demote_edges", default="896,768")
    p.add_argument(
        "--data_root",
        default=None,
        help="alternate dataset root holding lora/ + resized/ (e.g. the "
        "probe-local 1280 cache from prep_1280_probe.py); default = "
        "post_image_dataset",
    )
    p.add_argument("--artists", default=None, help="csv restriction on the corpus")
    p.add_argument(
        "--probe_list",
        default=None,
        help="E7 instrument delta: JSON file naming the EXACT probe images "
        "plus per-image tags — either a list of objects or {'images': [...]}, "
        "each {'artist': ..., 'stem': ..., <tag>: <value>, ...}. Replaces "
        "stratified selection (--num_images/--artists/--max_per_artist/"
        "--score_limit are ignored); images run in file order, and every key "
        "besides artist/stem is copied verbatim into that image's "
        "per_image.jsonl row (cell/membership/style tags). A listed stem "
        "that is missing, incomplete, or off-tier is a hard error — the "
        "probe set is a frozen design object, not a best-effort filter.",
    )
    p.add_argument("--max_per_artist", type=int, default=None)
    p.add_argument("--score_limit", type=int, default=None)
    p.add_argument("--no_reenc_control", action="store_true")
    p.add_argument(
        "--per_group",
        action="store_true",
        help="additionally report per-parameter-group gaps (Q2 J-decomposition): "
        "module types (incl. lora_up row-splits of the fused qkv/kv projs — "
        "the RoPE q/k-vs-v discriminator) x 28 blocks. Bookkeeping only — "
        "same forwards/backwards, per-slice cosines of the same flat "
        "gradient vectors.",
    )
    p.add_argument(
        "--endpoint_bin",
        action="store_true",
        help="append an exact sigma=1.0 bin (input = pure noise; any gap there "
        "is the target x graph floor — the S2/S3 term of the two-term account)",
    )
    p.add_argument(
        "--x_zero",
        action="store_true",
        help="zero the image in BOTH input and target on every grid (input = "
        "sigma*eps, target = eps; captions and latent shapes kept). Isolates "
        "pure graph-shape gradient sensitivity — no content anywhere. Implies "
        "--no_reenc_control (re-encode of nothing = the floor arm).",
    )
    p.add_argument(
        "--target_alpha",
        default=None,
        metavar="A1,A2,...",
        help="E2 target-strength sweep: run every arm at each alpha with "
        "target = noise - alpha*lat (input untouched — at sigma=1 it stays "
        "pure eps for every alpha). alpha=0 is the graph-only target "
        "(== x-zero-in-target at the endpoint), alpha=1 the standard target "
        "(must be included; its keys stay unsuffixed). Other alphas suffix "
        "every per-arm/native key with @a<alpha>. Draw seeds are shared "
        "across alphas (same noise draws -> the alpha-slope carries no draw "
        "noise) and all alphas live in ONE run (cross-process cosines are "
        "kernel-path chaotic). Cost: x len(alphas) forwards/backwards. "
        "Incompatible with --pool/--per_group/--draw_sweep/--x_zero. "
        "e.g. --target_alpha 0,0.25,0.5,0.75,1",
    )
    p.add_argument(
        "--pi_align",
        action="store_true",
        help="add a '<edge>pi' arm per demote edge: the SAME demoted latent, "
        "but RoPE generated at PI-stretched fractional positions (patch i at "
        "i*H_nat/H_dem per axis via generate_embeddings_scaled) so the demoted "
        "grid's relative phase geometry matches the native grid's. DyPE-style "
        "(arXiv:2510.20766) discriminator for the RoPE share of the Floor: "
        "gap_<e>pi << gap_<e> at sigma=1 => PE-geometry share is real; "
        "~= => Floor lives in softmax-N / normalization (G4 confirmed from "
        "the origin side). gap_896pi ~ gap_896 ~ 0 is the off-manifold "
        "control (fractional positions harmless where the Floor is 0).",
    )
    p.add_argument(
        "--yarn_align",
        default=None,
        metavar="ALPHA,BETA",
        help="add a '<edge>yarn' arm per demote edge: frequency-SELECTIVE "
        "position alignment (YaRN/NTK-by-parts, arXiv via DyPE 2510.20766 "
        "Eq.7): spatial RoPE bands completing < ALPHA rotations across the "
        "demoted grid extent get the full PI stretch toward native "
        "coordinates (global extent aligned), bands > BETA rotations keep "
        "native integer spacing (trained local content lobes preserved), "
        "linear ramp between. Tests whether the RoPE-mediated share of the "
        "gap can be removed WITHOUT G11's uniform-stretch off-manifold "
        "penalty. e.g. --yarn_align 1,4",
    )
    p.add_argument(
        "--yarn_sigma_gate",
        default=None,
        metavar="CENTER,GAMMA",
        help="with --yarn_align, add a '<edge>yarnsig' arm: same banded "
        "rescale but with SigMa-style dynamic boundary gating (SigMa Eq.21, "
        "github.com/bxuanz/SigMa): both band thresholds scaled per-draw by "
        "mu(sigma) = sigmoid(GAMMA*(logit(sigma)-logit(CENTER))), so the "
        "alignment self-attenuates toward native RoPE at low sigma (ramp "
        "bands leave the stretch zone -> the measured low-sigma liability "
        "mechanism is removed) and approaches static yarn at high sigma "
        "(mu->1 at the endpoint). Functional form only — the paper's scale "
        "laws (t_c=1/s, gamma=sqrt(s)) are inference-side and off-scale at "
        "s=8/7; CENTER comes from the measured improvement crossover. "
        "e.g. --yarn_sigma_gate 0.35,2",
    )
    p.add_argument(
        "--repromote",
        action="store_true",
        help="add an '<e>rp' arm per demote edge: the demoted pixels resized "
        "straight back UP to the native bucket and re-encoded — the same band "
        "destruction as the demote arm but on the NATIVE grid/graph. This is "
        "the operational data intervention B = g_rp - g_native of the "
        "interventional split (graph intervention C = g_demote - g_rp), so "
        "gap_<e>rp reads the data branch with Floor = 0 by construction and "
        "C(sigma) reads the graph share per bin without assuming it "
        "sigma-flat. Pair with --keep_arm_sums to retain the mean vectors "
        "the B/C ledger needs.",
    )
    p.add_argument(
        "--per_image_ledger",
        action="store_true",
        help="E22 amendment: per-image debiased B/C scalar reductions inside "
        "the per-image arm loop — B_i = g_rp - g_reenc, C_i = g_dem - g_rp, "
        "perp against the image's own native direction, second moments from "
        "cross-set products only, ref-noise subtracted from the image's own "
        "reenc set-diff, rho_i = I_i / 2*sqrt(S_i*F_i) — at three "
        "granularities (global / E21's four type bands / depth-block x "
        "core-type cells), appended per image to per_image_ledger.jsonl. "
        "No new forwards: the same arm gradients, reduced per image before "
        "accumulation. Requires --repromote --self_floor + the reenc "
        "control; forces non-streaming arm retention (every arm's per-bin "
        "vectors stay resident for one image — ~arms x bins x 311 MB CPU "
        "RAM, so keep the grid small).",
    )
    p.add_argument(
        "--keep_arm_sums",
        action="store_true",
        help="retain the cross-image SUM of every arm's per-bin flat LoRA "
        "gradient (both draw sets, every target-alpha) as fp32 memmaps under "
        "paper_bench/arm_sums/<run-name>/ + manifest.json (the central store "
        "root; <run_dir>/arm_sums is a symlink to it). Turns scalar-cosine runs into "
        "vector-ledger runs (B/C split, kappa_par/kappa_perp, exact "
        "counterfactual angles) at ~311 MB x arms x bins of disk and zero "
        "extra forwards. Vectors are sums over images of per-image "
        "draw-summed gradients — divide by n_images*draws for means.",
    )
    p.add_argument(
        "--arm_sums_dtype",
        choices=("fp32", "fp16"),
        default="fp32",
        help="--keep_arm_sums store dtype. fp16 halves the disk footprint "
        "(a repromote x self-floor 15-bin store is ~75 GB fp32 — over this "
        "box's headroom) at ~1e-3 relative accumulation rounding, well "
        "under the vector ledger's read precision; manifest records it.",
    )
    p.add_argument(
        "--target_kappa",
        action="store_true",
        help="with --target_alpha 0,1 (endpoint-only: --bins 0 "
        "--endpoint_bin): per image, form the EXACT target-content gradient "
        "t_arm = g_arm(alpha=1) - g_arm(alpha=0) (the forward pass is "
        "alpha-independent, so the difference is exact at shared seeds) and "
        "report per-arm kappa_par_<k> = ghat_src . (t_k - t_src) / G and "
        "kappa_perp_<k> = |P_perp (t_k - t_src)| / G, plus the a-vs-b "
        "null and |t|/G observability norms. Resolves whether an "
        "unresolvable alpha-slope means parallel landing (rescaling, "
        "invisible to the angular gap) or genuine J^T attenuation of the "
        "destroyed band.",
    )
    p.add_argument(
        "--pool",
        type=int,
        default=0,
        help="stratified gradient-pooling: sort the probe set by redundancy, "
        "chunk into strata of N images, and ADDITIONALLY report pooled gap "
        "curves — per stratum the per-image bin-gradients are summed across "
        "images (gradient accumulation = the batch-SGD aggregate object) "
        "before cosines, in two variants: unweighted (training-realistic, "
        "large-gnorm images dominate) and per-image-normalized (side-channel), "
        "plus an all-images aggregate. Pooled cosines are dominated by the "
        "shared cross-image gradient component, so pooled floors/gaps are NOT "
        "comparable to per-image gaps or the ±0.04 instrument band — each "
        "stratum carries its own noise-redraw floor and an image-split-half "
        "floor. Per-image rows are still written unchanged.",
    )
    p.add_argument(
        "--pool_spill",
        action="store_true",
        help="back the per-stratum pool accumulator with disk memmaps under "
        "the run dir (the all-images aggregate already is). Required when "
        "arms x bins x 311 MB won't fit in RAM twice — an E1b-shaped grid "
        "(10 self-floor arm lists x 9 bins) is ~28 GB per accumulator set, "
        "which OOM-killed the first E3 launch on this 46 GB box. Spill is "
        "transient (~2x accumulator size on disk) and deleted at the end.",
    )
    p.add_argument(
        "--pool_no_norm",
        action="store_true",
        help="skip the per-image-normalized pooled side-channel (norm_* "
        "keys): halves each accumulator's footprint. The unweighted "
        "(batch-SGD) pooled object — the a + b/B fit's input — is unaffected.",
    )
    p.add_argument(
        "--self_floor",
        action="store_true",
        help="E1 debiasing: run a SECOND independent draw set for every arm "
        "(reenc + each demote/pi/yarn arm) and report cos_self_<key> per bin "
        "plus the split-half attenuation-corrected cosine "
        "c_hat = cos(a+b, d+d') / sqrt(cos(a,b) * cos(d,d')) and "
        "debiased_gap_<key> = 1 - c_hat. Raw gaps unchanged (first draw set "
        "only). Doubles the per-arm forward/backward cost. With --pool, "
        "pooled arms get pooled self-floors + debiased gaps too.",
    )
    p.add_argument(
        "--draw_sweep",
        default=None,
        metavar="D1,D2,...,DMAX",
        help="E1 draw-count extrapolation: endpoint-only mode (forces "
        "--bins 0 --endpoint_bin --draws_per_bin DMAX); one pass at DMAX "
        "draws with NESTED seeds — the accumulated gradient is snapshotted "
        "at each prefix D, so the D=DMAX estimate contains the D=D1 draws "
        "(no extra forwards). Every per-bin array in rows/headline is "
        "indexed by prefix D instead of sigma-bin; headline adds per-arm "
        "fits gap(D) = gap_inf + c/D with a bootstrap CI over images.",
    )
    p.add_argument(
        "--draw_batch_tokens",
        type=int,
        default=0,
        help="GPU-efficiency: batch noise draws into one forward/backward, "
        "B = pow2-floor(BUDGET // grid_tokens) per arm (small demoted grids "
        "batch hardest; native may stay at B=1). Statistically identical to "
        "B=1 — per-draw seeds/noise unchanged, loss = B * batch-mean-MSE "
        "which equals the accumulated sum of per-draw mean losses exactly "
        "(up to float reduction order). Draw chunks never cross --draw_sweep "
        "snapshot boundaries; σ-gated rope arms (yarnsig) fall back to B=1 "
        "within any chunk whose σ values differ. Each distinct B adds one "
        "compile graph per token family (one-time warmup). ~8400 batches "
        "2x native / 8x at 512; 0 = off.",
    )
    p.add_argument(
        "--grad_ckpt",
        action="store_true",
        help="use gradient checkpointing INSTEAD of block compile (fallback)",
    )
    p.add_argument(
        "--activation_memory_budget",
        type=float,
        default=0.99,
        help="partitioner knapsack cap under compile (freefit knee; 1.0 = off)",
    )
    p.add_argument(
        "--partitioner_aggressive",
        action="store_true",
        help="opt back into partitioner aggressive recomputation (the "
        "issue-58 bench: −2.25 GB for +12.6% s/it). Historical probe runs "
        "(≤ E13) had it hardwired ON; at B=1 the probe peaks well under "
        "16 GB without it, so the default is now the faster ship_base "
        "partitioner.",
    )
    p.add_argument(
        "--cond_sigma",
        type=float,
        default=None,
        metavar="SIGMA",
        help="E28 frozen-conditioning probe: pin the sigma fed to the DiT "
        "forward (timestep embedding -> adaln) at this value on EVERY draw of "
        "every arm, while the NOISING sigma still sweeps the grid. Replaces "
        "only the conditioning argument at the single DiT call site — the "
        "noised input, the flow-matching target (eps - x, conditioning-"
        "independent), the sigma-gated rope handle (keeps the noising sigma), "
        "and every arm/seed path are untouched. Off-manifold at distant "
        "noising sigma (a (z, sigma_cond) pair the network never trained on) "
        "— readability is gated by E28's 28-A. Recorded in the arm_sums "
        "manifest so a frozen-conditioning store can never be mistaken for a "
        "native one.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="mirror train.py's deterministic mode (flash-attn deterministic "
        "backward, use_deterministic_algorithms warn_only, cudnn "
        "deterministic, CUBLAS_WORKSPACE_CONFIG): kills the atomics-order "
        "run-to-run noise (measured |Δcos| ≤ 0.015 at D=2) so runs sharing "
        "a warm inductor cache are bit-comparable. Use for any CROSS-RUN "
        "read (E7 cells, adapter bridging). NB does NOT pin the kernel set "
        "itself — a cold /tmp inductor cache (e.g. post-reboot) re-autotunes "
        "and can still shift results by ~0.3 at D=2; within-run pairings "
        "never need this flag.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quant_k", type=int, default=4)
    p.add_argument(
        "--results_root",
        default=None,
        help="run-dir root override (default: <script dir>/results). Paper-"
        "bench runs pass project/sigma_lowres/paper_bench/runs — a name the "
        "global 'results' gitignore does NOT match, so verdict runs are "
        "committable (repro deliverable).",
    )
    p.add_argument("--label", default=None)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.num_images = 4
        args.bins = 4
        args.draws_per_bin = 2
        args.demote_edges = "896"
        args.score_limit = args.score_limit or 120
    return args


@dataclass
class RunConfig:
    """Everything ``main`` needs that is derived from more than one flag."""

    edges: list[int]
    reenc_control: bool
    sigma_lo: float
    sigma_hi: float
    # (lo, hi, bins) per --sigma_window segment; single-entry for the plain form
    segments: list[tuple[float, float, int]] = field(default_factory=list)
    alphas: list[float] = field(default_factory=lambda: [1.0])
    sweep: list[int] | None = None
    yarn_bands: tuple[float, float] | None = None
    yarn_gate: tuple[float, float] | None = None
    probe_order: list[tuple[str, str]] | None = None
    probe_tags: dict[tuple[str, str], dict] | None = None


def parse_sigma_window(spec: str, fallback_bins: int) -> list[tuple[float, float, int]]:
    """``--sigma_window`` → sorted, non-overlapping ``(lo, hi, bins)`` segments.

    ``"0.5,1.0"`` (one segment, no BINS) keeps the historical meaning and takes
    its bin count from ``--bins``. ``"0,0.1,4 : 0.1,0.9,6 : 0.9,1.0,4"`` is the
    segmented form — per-segment bin *density*, global draws-per-bin.
    """
    segs: list[tuple[float, float, int]] = []
    parts = [s for s in spec.split(":") if s.strip()]
    if not parts:
        raise SystemExit("--sigma_window is empty")
    for part in parts:
        fields = [f for f in part.split(",") if f.strip()]
        if len(fields) == 2 and len(parts) == 1:
            lo, hi, bins = float(fields[0]), float(fields[1]), fallback_bins
        elif len(fields) == 3:
            lo, hi = float(fields[0]), float(fields[1])
            bins = int(fields[2])
            if bins < 1:
                raise SystemExit(f"--sigma_window segment '{part}' needs BINS >= 1")
        else:
            raise SystemExit(
                f"--sigma_window segment '{part}' must be LO,HI,BINS "
                "(LO,HI only allowed for the single-segment form)"
            )
        if not 0.0 <= lo < hi <= 1.0:
            raise SystemExit(
                f"--sigma_window must satisfy 0 <= LO < HI <= 1, got {lo},{hi}"
            )
        if segs and lo < segs[-1][1]:
            raise SystemExit(
                "--sigma_window segments must be sorted and non-overlapping, "
                f"got [{segs[-1][0]},{segs[-1][1]}] then [{lo},{hi}]"
            )
        segs.append((lo, hi, bins))
    return segs


def resolve_run_config(args: argparse.Namespace) -> RunConfig:
    """Validate the flag combination and derive the run's shape. Mutates
    ``args`` where a mode forces the grid (draw-sweep, probe-list)."""
    edges = [int(e) for e in args.demote_edges.split(",") if e]

    yarn_bands = None
    if args.yarn_align:
        a, b = (float(v) for v in args.yarn_align.split(","))
        yarn_bands = (a, b)
    yarn_gate = None
    if args.yarn_sigma_gate:
        if not args.yarn_align:
            raise SystemExit("--yarn_sigma_gate requires --yarn_align")
        gc, gg = (float(v) for v in args.yarn_sigma_gate.split(","))
        if not (0.0 < gc < 1.0 and gg > 0.0):
            raise SystemExit(
                f"--yarn_sigma_gate needs 0<CENTER<1 and GAMMA>0, got {gc},{gg}"
            )
        yarn_gate = (gc, gg)

    if args.cond_sigma is not None and not (0.0 < args.cond_sigma <= 1.0):
        raise SystemExit(f"--cond_sigma must be in (0, 1], got {args.cond_sigma}")

    if args.x_zero:
        args.no_reenc_control = True
    reenc_control = not args.no_reenc_control

    sweep: list[int] | None = None
    if args.draw_sweep:
        sweep = sorted({int(v) for v in args.draw_sweep.split(",") if v})
        if len(sweep) < 2:
            raise SystemExit("--draw_sweep needs >= 2 draw counts")
        if args.pool or args.per_group:
            raise SystemExit("--draw_sweep is incompatible with --pool/--per_group")
        args.bins = 0
        args.endpoint_bin = True
        args.draws_per_bin = sweep[-1]
        log.info(f"draw-sweep mode: endpoint-only, nested prefixes D={sweep}")

    alphas: list[float] = [1.0]
    if args.target_alpha:
        alphas = sorted({round(float(v), 4) for v in args.target_alpha.split(",") if v})
        if not all(0.0 <= a <= 1.0 for a in alphas):
            raise SystemExit("--target_alpha values must be in [0, 1]")
        if 1.0 not in alphas:
            raise SystemExit(
                "--target_alpha must include 1 (the standard-target anchor)"
            )
        if args.pool or args.per_group or sweep or args.x_zero:
            raise SystemExit(
                "--target_alpha is incompatible with "
                "--pool/--per_group/--draw_sweep/--x_zero"
            )
        log.info(f"target-alpha sweep: alphas={alphas} (alpha=1 keys unsuffixed)")
    if args.target_kappa:
        if not (0.0 in alphas and 1.0 in alphas):
            raise SystemExit("--target_kappa needs --target_alpha including 0 and 1")
        if args.bins != 0 or not args.endpoint_bin:
            raise SystemExit(
                "--target_kappa is endpoint-only (--bins 0 --endpoint_bin): "
                "per-image cross-alpha vector retention is sized for one bin"
            )

    if args.per_image_ledger:
        if not (args.repromote and args.self_floor and reenc_control):
            raise SystemExit(
                "--per_image_ledger needs --repromote, --self_floor and the "
                "reenc control (its estimand is the cross-set debiased "
                "B_i/C_i split with the image's own reenc reference)"
            )
        if args.pool or args.target_kappa or args.draw_sweep or args.x_zero:
            raise SystemExit(
                "--per_image_ledger is incompatible with "
                "--pool/--target_kappa/--draw_sweep/--x_zero"
            )
        if args.target_alpha:
            raise SystemExit("--per_image_ledger is incompatible with --target_alpha")

    probe_tags: dict[tuple[str, str], dict] | None = None
    probe_order: list[tuple[str, str]] | None = None
    if args.probe_list:
        entries = json.loads(Path(args.probe_list).read_text())
        if isinstance(entries, dict):
            entries = entries["images"]
        probe_order = [(e["artist"], e["stem"]) for e in entries]
        if len(set(probe_order)) != len(probe_order):
            raise SystemExit("--probe_list contains duplicate artist/stem entries")
        probe_tags = {
            (e["artist"], e["stem"]): {
                k: v for k, v in e.items() if k not in ("artist", "stem")
            }
            for e in entries
        }
        args.num_images = len(probe_order)
        args.score_limit = None  # a truncated scoring pool could drop listed stems
        log.info(f"probe list: {len(probe_order)} images from {args.probe_list}")

    if args.self_floor and args.num_images > 50:
        # the alt seed base (+500_000) sits above i*10_000 only for i < 50
        raise SystemExit("--self_floor seed spacing supports --num_images <= 50")

    segments = parse_sigma_window(args.sigma_window, args.bins)
    if len(segments) > 1:
        # the segmented form owns the bin count; --bins is only the
        # single-segment fallback. result.json and the seed-budget check both
        # read args.bins, so it has to become the total.
        args.bins = sum(b for _, _, b in segments)
        log.info(
            "segmented σ window: "
            + " : ".join(f"[{lo},{hi}]×{b}" for lo, hi, b in segments)
            + f" = {args.bins} bins"
        )

    return RunConfig(
        edges=edges,
        reenc_control=reenc_control,
        sigma_lo=segments[0][0],
        sigma_hi=segments[-1][1],
        segments=segments,
        alphas=alphas,
        sweep=sweep,
        yarn_bands=yarn_bands,
        yarn_gate=yarn_gate,
        probe_order=probe_order,
        probe_tags=probe_tags,
    )


def build_arm_keys(
    args: argparse.Namespace, cfg: RunConfig, total_draws: int
) -> list[str]:
    """Arm keys in run order (reenc first, then each edge's demote/rp/pi/yarn/
    yarnsig), with the per-image seed-block budget checked."""
    arm_keys = ["reenc"] if cfg.reenc_control else []
    for e in cfg.edges:
        arm_keys.append(str(e))
        if args.repromote:
            arm_keys.append(f"{e}rp")
        if args.pi_align:
            arm_keys.append(f"{e}pi")
        if args.yarn_align:
            arm_keys.append(f"{e}yarn")
            if args.yarn_sigma_gate:
                arm_keys.append(f"{e}yarnsig")
    # seeds() spaces arms 1_000 apart inside a 10_000-wide per-image block;
    # a/b take indices 0/1 and each arm key one more, so the last arm's
    # draw range must stay inside the block
    if (1 + len(arm_keys)) * 1_000 + total_draws > 10_000:
        raise SystemExit(
            f"{len(arm_keys)} arms x {total_draws} draws overflows the "
            "per-image seed block (arm_idx*1000 spacing) — drop arms or draws"
        )
    return arm_keys
