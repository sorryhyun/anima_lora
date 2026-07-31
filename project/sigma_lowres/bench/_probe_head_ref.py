#!/usr/bin/env python
"""sigma_lowres Measurement B — per-σ-bin gradient-equivalence probe.

The tier_routing Phase 3a instrument (redraw-floor null, re-encode confound
control, demote arms) with one change: gradients are accumulated into
**per-σ-bin buckets** instead of one pooled estimate, so the demotion gap
becomes a curve gap_e(σ) instead of a scalar. Phase 3a marginalized σ out;
SwD's spectral analysis (arXiv:2503.16397 §3) pre-registers the hypothesis
that the gap concentrates below a tier-specific σ* and collapses above it
(``project/sigma_lowres/initial_proposal.md`` H2/H3).

σ bins are uniform on (0, 1) — the mechanism axis. Per-bin means across
images are the verdict quantity (the estimator class that was reliable in
3a); per-image per-bin rows land in ``per_image.jsonl`` for split-half
reliability analysis.

Usage::

    uv run python project/sigma_lowres/bench/run_sigma_probe.py \
        --adapter output/ckpt/anima_soup_sincos.safetensors --label phase0
    uv run python project/sigma_lowres/bench/run_sigma_probe.py \
        --adapter <ckpt> --smoke --label smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from project.sigma_lowres.bench.tier_routing.redundancy import (  # noqa: E402
    score_corpus,
    select_probe_set,
)
from project.sigma_lowres.bench.tier_routing.run_grad_probe import (  # noqa: E402
    DIT,
    VAE,
    cosine,
    encode_probe_latents,
    spearman,
)
from library.io.cache import (  # noqa: E402
    load_cached_latents,
    load_cached_text_features,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
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
        "every bin in the high-σ crossover region); --endpoint_bin unaffected",
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
        "--keep_arm_sums",
        action="store_true",
        help="retain the cross-image SUM of every arm's per-bin flat LoRA "
        "gradient (both draw sets, every target-alpha) as fp32 memmaps under "
        "<run_dir>/arm_sums/ + manifest.json. Turns scalar-cosine runs into "
        "vector-ledger runs (B/C split, kappa_par/kappa_perp, exact "
        "counterfactual angles) at ~311 MB x arms x bins of disk and zero "
        "extra forwards. Vectors are sums over images of per-image "
        "draw-summed gradients — divide by n_images*draws for means.",
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
    args = p.parse_args()
    if args.smoke:
        args.num_images = 4
        args.bins = 4
        args.draws_per_bin = 2
        args.demote_edges = "896"
        args.score_limit = args.score_limit or 120
    return args


def bin_sigmas(bins: int, draws: int, lo: float = 0.0, hi: float = 1.0) -> torch.Tensor:
    """(bins, draws) σ grid: uniform bins on (lo, hi), stratified midpoints
    inside each bin. Uniform (not training-density) — per-bin means make the
    marginal density irrelevant, and σ is the axis under test. The window
    concentrates all bins in a sub-interval (crossover localization)."""
    b = torch.arange(bins, dtype=torch.float64).view(-1, 1)
    j = (torch.arange(draws, dtype=torch.float64) + 0.5).view(1, -1)
    u = (b + j / draws) / bins
    return (lo + (hi - lo) * u).to(torch.float32)


def build_sigmas(
    bins: int, draws: int, endpoint: bool, lo: float = 0.0, hi: float = 1.0
) -> torch.Tensor:
    """Uniform-bin grid over the (lo, hi) window, optionally with an exact
    σ=1.0 bin appended. ``--bins 0 --endpoint_bin`` gives an endpoint-only
    grid."""
    parts = []
    if bins > 0:
        parts.append(bin_sigmas(bins, draws, lo, hi))
    if endpoint:
        parts.append(torch.ones(1, draws, dtype=torch.float32))
    if not parts:
        raise SystemExit("need --bins > 0 and/or --endpoint_bin")
    return torch.cat(parts, dim=0)


GROUP_RE = re.compile(r"^lora_unet_blocks_(\d+)_(.+)$")


def build_groups(network) -> dict[str, list[tuple[int, int]]]:
    """Group -> flat-vector ranges (sorted-name order — must match
    ``grad_estimate_binned``'s flatten), at two levels: ``type:<module minus
    block prefix>`` and ``block:<idx>``.

    Fused projections additionally get row-block sub-groups on ``lora_up``
    (rows are contiguous in the row-major flatten): ``self_attn_qkv_proj`` →
    ``type:self_attn_up_{q,k,v}`` and ``cross_attn_kv_proj`` →
    ``type:cross_attn_up_{k,v}``. RoPE touches self-attn q/k only, so the
    q/k-vs-v contrast is the RoPE discriminator; ``lora_down`` is shared
    across the fused heads and stays only in the module-level group."""
    named = [(n, p) for n, p in sorted(network.named_parameters()) if p.requires_grad]
    groups: dict[str, list[tuple[int, int]]] = {}
    pos = 0
    for name, p in named:
        s, e = pos, pos + p.numel()
        pos = e
        m = GROUP_RE.match(name.split(".")[0])
        keys = (
            (f"type:{m.group(2)}", f"block:{int(m.group(1)):02d}")
            if m
            else ("type:other", "block:other")
        )
        for k in keys:
            groups.setdefault(k, []).append((s, e))
        if m and name.endswith(".lora_up.weight"):
            typ = m.group(2)
            if typ == "self_attn_qkv_proj":
                third = p.numel() // 3  # rows are [q; k; v] blocks
                for j, sub in enumerate(("q", "k", "v")):
                    groups.setdefault(f"type:self_attn_up_{sub}", []).append(
                        (s + j * third, s + (j + 1) * third)
                    )
            elif typ == "cross_attn_kv_proj":
                half = p.numel() // 2  # rows are [k; v] blocks
                for j, sub in enumerate(("k", "v")):
                    groups.setdefault(f"type:cross_attn_up_{sub}", []).append(
                        (s + j * half, s + (j + 1) * half)
                    )
    return groups


def grouped_cosine(
    a: torch.Tensor,
    b: torch.Tensor,
    groups: dict[str, list[tuple[int, int]]],
) -> dict[str, float]:
    """Per-group cosine of two flat gradient vectors over each group's
    flat-vector ranges."""
    out = {}
    for g, ranges in groups.items():
        d = na = nb = 0.0
        for s, e in ranges:
            va, vb = a[s:e], b[s:e]
            d += float(va.dot(vb))
            na += float(va.dot(va))
            nb += float(vb.dot(vb))
        out[g] = d / (na**0.5 * nb**0.5) if na > 0 and nb > 0 else 0.0
    return out


class PoolAccumulator:
    """Cross-image gradient accumulator for one stratum (or the aggregate).

    Holds, per arm and per σ-bin, the running sum of per-image bin-gradient
    vectors (``sums``, unweighted — the batch-SGD object) and of per-image
    L2-normalized vectors (``nsums`` — the equal-weight side-channel), plus a
    parity split of the native (a+b) sum for the image-split-half floor.
    Everything is float32 CPU; memory is O(arms x bins) vectors regardless of
    stratum size — but each vector is a full flat LoRA gradient (~311 MB at
    77M params), so one accumulator is ~19 GB at 5 arms x 5 bins. With
    ``backing_dir`` set, every accumulator vector lives in a disk memmap
    instead of RAM (pages are cache-evictable) — used for the all-images
    aggregate, which is written 10x but read once at the end. Without it the
    aggregate + stratum accumulators together OOM a 46 GB box. ``release()``
    additionally closes the memmap handles between merges: mapped file pages
    count against process RSS while a handle is open even though they're
    reclaimable, so a released aggregate costs ~zero RSS outside merges.
    """

    def __init__(self, backing_dir: Path | None = None) -> None:
        self.backing_dir = backing_dir
        self._numel: int | None = None
        self.sums: dict[str, list[torch.Tensor]] = {}
        self.nsums: dict[str, list[torch.Tensor]] = {}
        self.halves: dict[int, list[torch.Tensor]] = {}
        self.n = 0
        self.redundancy: list[float] = []

    def _stores(self) -> tuple[tuple[str, dict], ...]:
        return ("sums", self.sums), ("nsums", self.nsums), ("halves", self.halves)

    def _open(self, name: str, key, idx: int, mode: str) -> torch.Tensor:
        mm = np.memmap(
            self.backing_dir / f"{name}_{key}_{idx}.f32",
            dtype=np.float32,
            mode=mode,
            shape=(self._numel,),
        )
        return torch.from_numpy(mm)

    def _materialize(self, name: str, key, idx: int, v: torch.Tensor) -> torch.Tensor:
        if self.backing_dir is None:
            return v
        self.backing_dir.mkdir(parents=True, exist_ok=True)
        self._numel = v.numel()
        t = self._open(name, key, idx, "w+")
        t.copy_(v)
        return t

    def _add(
        self,
        name: str,
        store: dict,
        key,
        vecs: list[torch.Tensor],
        scales: list[float] | None = None,
    ) -> None:
        if scales is None:
            scales = [1.0] * len(vecs)
        if key not in store:
            store[key] = [
                self._materialize(name, key, i, v * s)
                for i, (v, s) in enumerate(zip(vecs, scales))
            ]
        else:
            for acc, v, s in zip(store[key], vecs, scales):
                acc += v * s

    def release(self) -> None:
        """Backed mode: replace each vector list with its length, dropping the
        memmap handles (data is on disk). Reopened by ``ensure_open``."""
        if self.backing_dir is None:
            return
        for _, store in self._stores():
            for key, vecs in store.items():
                if not isinstance(vecs, int):
                    store[key] = len(vecs)

    def ensure_open(self) -> None:
        if self.backing_dir is None:
            return
        for name, store in self._stores():
            for key, val in store.items():
                if isinstance(val, int):
                    store[key] = [self._open(name, key, i, "r+") for i in range(val)]

    def add_image(self, arms: dict[str, list[torch.Tensor]], redundancy: float) -> None:
        for key, vecs in arms.items():
            self._add("sums", self.sums, key, vecs)
            self._add(
                "nsums",
                self.nsums,
                key,
                vecs,
                [1.0 / (float(v.norm()) + 1e-12) for v in vecs],
            )
        native = [a + b for a, b in zip(arms["a"], arms["b"])]
        self._add("halves", self.halves, self.n % 2, native)
        self.n += 1
        self.redundancy.append(redundancy)

    def merge(self, other: "PoolAccumulator") -> None:
        self.ensure_open()
        for name, mine in self._stores():
            theirs = getattr(other, name)
            for key, vecs in theirs.items():
                self._add(name, mine, key, vecs)
        self.n += other.n
        self.redundancy.extend(other.redundancy)


class ArmSumAccumulator:
    """Cross-image sums of per-bin flat LoRA gradients, one fp32 disk memmap
    per (arm key, bin) under ``dir/``. Unlike :class:`PoolAccumulator` this
    keeps EVERY arm (native a/b, alt draw sets, every target-alpha suffix)
    and survives the run — the raw material for the interventional
    vector ledger (B/C split, kappa components, exact angles). Keys are
    sanitized for filenames (``@`` → ``~``); ``manifest.json`` records the
    mapping plus the scale convention (sum over images of per-image
    draw-summed gradients).
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.maps: dict[tuple[str, int], np.memmap] = {}
        self.n_images: dict[str, int] = {}

    @staticmethod
    def _fname(key: str, bi: int) -> str:
        return f"{key.replace('@', '~')}__b{bi}.npy"

    def add(self, key: str, vecs: list[torch.Tensor]) -> None:
        from numpy.lib.format import open_memmap

        for bi, v in enumerate(vecs):
            mm = self.maps.get((key, bi))
            if mm is None:
                path = self.root / self._fname(key, bi)
                mm = open_memmap(path, mode="w+", dtype=np.float32, shape=(v.numel(),))
                self.maps[(key, bi)] = mm
            mm += v.numpy()
        self.n_images[key] = self.n_images.get(key, 0) + 1

    def finalize(self, meta: dict) -> None:
        for mm in self.maps.values():
            mm.flush()
        keys = sorted({k for k, _ in self.maps})
        manifest = {
            **meta,
            "scale": "sum over images of per-image draw-summed gradients "
            "(divide by n_images * draws_per_bin for the mean gradient)",
            "keys": {
                k: {
                    "n_images": self.n_images[k],
                    "files": [
                        self._fname(k, bi)
                        for bi in sorted(bi for kk, bi in self.maps if kk == k)
                    ],
                }
                for k in keys
            },
        }
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def pool_stats(acc: PoolAccumulator, arm_keys: list[str]) -> dict:
    """Pooled-cosine curves for one accumulator: noise-redraw floor
    (pooled-a vs pooled-b over the same images), per-arm cos/gap, the
    normalized variant (``norm_`` prefix), and the image-split-half floor
    (pooled native over even- vs odd-indexed images — includes image-sampling
    variance, which the redraw floor does not)."""
    acc.ensure_open()
    out: dict = {
        "n_images": acc.n,
        "redundancy_mean": round(float(np.mean(acc.redundancy)), 4),
        "redundancy_range": [
            round(min(acc.redundancy), 4),
            round(max(acc.redundancy), 4),
        ],
    }
    for prefix, store in (("", acc.sums), ("norm_", acc.nsums)):
        a, b = store["a"], store["b"]
        floor = [cosine(x, y) for x, y in zip(a, b)]
        out[f"{prefix}cos_floor"] = [round(v, 5) for v in floor]
        for key in arm_keys:
            d = store[key]
            c = [0.5 * (cosine(x, g) + cosine(y, g)) for x, y, g in zip(a, b, d)]
            out[f"{prefix}cos_{key}"] = [round(v, 5) for v in c]
            out[f"{prefix}gap_{key}"] = [round(f - v, 5) for f, v in zip(floor, c)]
            d2 = store.get(f"{key}__2")
            if d2 is None:  # no --self_floor second draw set pooled
                continue
            selfc = [cosine(x, y) for x, y in zip(d, d2)]
            out[f"{prefix}cos_self_{key}"] = [round(v, 5) for v in selfc]
            dg = []
            for f_, s_, x, y, u, v in zip(floor, selfc, a, b, d, d2):
                if f_ <= 0 or s_ <= 0:
                    dg.append(None)
                else:
                    dg.append(round(1.0 - cosine(x + y, u + v) / math.sqrt(f_ * s_), 5))
            out[f"{prefix}debiased_gap_{key}"] = dg
    out["gnorm_pooled"] = [
        round(0.5 * (float(x.norm()) + float(y.norm())), 3)
        for x, y in zip(acc.sums["a"], acc.sums["b"])
    ]
    if len(acc.halves) == 2:
        out["imgsplit_floor"] = [
            round(cosine(h0, h1), 5) for h0, h1 in zip(acc.halves[0], acc.halves[1])
        ]
    return out


@contextmanager
def pi_rope(anima, h_scale: float, w_scale: float):
    """PI-stretch the main-stream RoPE for the duration: every spatial patch
    ``i`` sits at fractional position ``i * scale`` (EasyControl's
    ``generate_embeddings_scaled`` — exact at fractional positions, distinct
    cache key). Instance-level ``forward`` patch on the pos_embedder; RoPE is
    built OUTSIDE the compiled block graph (``prepare_embedded_sequence``), so
    there is no dynamo interaction — the blocks just receive different cos/sin
    input tensors at the same token count. Scaled cache entries are dropped on
    exit (per-image scales would otherwise accrete VRAM across the run)."""
    pe = anima.pos_embedder

    def fwd(x_B_T_H_W_C, fps=None):
        return pe.generate_embeddings_scaled(
            x_B_T_H_W_C.shape, h_scale=h_scale, w_scale=w_scale, fps=fps
        )

    pe.forward = fwd
    try:
        yield
    finally:
        del pe.forward  # restore the class-level forward
        for k in [k for k in pe._cos_sin_cache if k and k[0] == "scaled"]:
            del pe._cos_sin_cache[k]


@contextmanager
def yarn_rope(
    anima,
    h_scale: float,
    w_scale: float,
    alpha: float,
    beta: float,
    gate: tuple[float, float] | None = None,
):
    """Frequency-banded position alignment (YaRN/NTK-by-parts style).

    Per spatial RoPE frequency ``f_d`` (rad/patch), the rotation count across
    the demoted extent is ``r_d = N * f_d / 2π``. Bands with ``r_d < alpha``
    (global-extent carriers) get the full PI stretch toward native
    coordinates; ``r_d > beta`` (local content-precision carriers) keep the
    native integer spacing; linear ramp between. Implemented as a per-dim
    frequency rescale (phase ``i·f_d·s_d`` ≡ position scale per band), built
    probe-locally from the pos_embedder's own buffers — models.py untouched.
    Same instance-``forward`` patch + no-dynamo-interaction reasoning as
    ``pi_rope``; local per-shape cache, dropped on exit.

    With ``gate=(center, gamma)`` (the ``<edge>yarnsig`` arm), both band
    thresholds are scaled by ``μ(σ) = sigmoid(gamma·[logit(σ) −
    logit(center)])`` — SigMa's dynamic boundary gating (Eq. 21) adapted to
    demotion: at low σ the thresholds shrink toward 0, every band lands above
    ``beta·μ`` and keeps native spacing (intervention off); μ→1 recovers the
    static yarn arm. Yields a handle with ``set_sigma`` that the estimator
    calls per draw; cos/sin cache keyed on (shape, μ)."""
    from einops import repeat as _repeat

    pe = anima.pos_embedder
    cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
    state = {"mu": 1.0}

    def s_vec(freqs: torch.Tensor, n: int, scale: float) -> torch.Tensor:
        mu = state["mu"]
        if mu < 1e-9:
            return torch.ones_like(freqs)
        r = n * freqs / (2 * math.pi)
        g = ((r - alpha * mu) / ((beta - alpha) * mu)).clamp(0.0, 1.0)
        return (1.0 - g) * scale + g

    def fwd(x_B_T_H_W_C, fps=None):
        B, T, H, W, _ = x_B_T_H_W_C.shape
        key = (T, H, W, round(state["mu"], 6))
        hit = cache.get(key)
        if hit is not None:
            return hit
        h_freqs = 1.0 / ((10000.0 * pe.h_ntk_factor) ** pe.dim_spatial_range)
        w_freqs = 1.0 / ((10000.0 * pe.w_ntk_factor) ** pe.dim_spatial_range)
        t_freqs = 1.0 / ((10000.0 * pe.t_ntk_factor) ** pe.dim_temporal_range)
        half_h = torch.outer(pe.seq[:H], h_freqs * s_vec(h_freqs, H, h_scale))
        half_w = torch.outer(pe.seq[:W], w_freqs * s_vec(w_freqs, W, w_scale))
        half_t = torch.outer(pe.seq[:T], t_freqs)
        em = torch.cat(
            [
                _repeat(half_t, "t d -> t h w d", h=H, w=W),
                _repeat(half_h, "h d -> t h w d", t=T, w=W),
                _repeat(half_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )
        freqs = em.flatten(0, 2).unsqueeze(1).unsqueeze(1).float()
        out = (torch.cos(freqs), torch.sin(freqs))
        cache[key] = out
        return out

    handle = None
    if gate is not None:
        center, gamma = gate
        logit_c = math.log(center / (1.0 - center))

        def set_sigma(sigma: float) -> None:
            s = min(max(sigma, 1e-6), 1.0 - 1e-6)
            state["mu"] = 1.0 / (
                1.0 + math.exp(-gamma * (math.log(s / (1.0 - s)) - logit_c))
            )

        handle = argparse.Namespace(set_sigma=set_sigma)
    pe.forward = fwd
    try:
        yield handle
    finally:
        del pe.forward
        cache.clear()


def grad_estimate_binned(
    bundle,
    latents: torch.Tensor,
    crossattn: torch.Tensor,
    sigmas: torch.Tensor,  # (bins, draws)
    seeds: list[int],  # len == bins * draws
    rope_patch=None,  # no-arg callable returning a rope-patch context manager
    prefix_draws: list[int] | None = None,
    batch_draws: int = 1,
    target_alpha: float = 1.0,
) -> tuple[list[torch.Tensor], list[float]]:
    """Per-σ-bin accumulated-gradient estimates.

    Returns ``(vecs, norms)``: per bin, the flattened LoRA gradient summed
    over that bin's draws (float32, CPU) and its L2 norm. Same forward/
    backward cost as the pooled 3a estimator at equal total draws — only the
    accumulator is split.

    ``prefix_draws`` (draw-sweep mode, single-bin grids only): instead of one
    vector per bin, snapshot the accumulating gradient after each listed draw
    count — nested-seed estimates at D ∈ prefix_draws from one pass; the
    returned lists are indexed by prefix.

    ``batch_draws`` > 1 runs that many draws per forward/backward: per-draw
    noise is generated from the same per-draw seeds and stacked, σ rides the
    batch axis, and the loss is ``B * mse_mean`` — exactly the accumulated
    sum of per-draw mean losses (same-shape elements), so the resulting bin
    gradient is unchanged up to float reduction order. Chunks are clamped at
    prefix-snapshot boundaries; a σ-gated rope arm falls back to per-draw
    stepping inside any chunk whose σ values differ.
    """
    device = bundle.device
    params = [
        p for _, p in sorted(bundle.network.named_parameters()) if p.requires_grad
    ]
    lat = latents.unsqueeze(0).to(device)  # (1, 16, H, W) float32
    pad = torch.zeros(
        1, 1, lat.shape[-2], lat.shape[-1], dtype=torch.bfloat16, device=device
    )
    vecs: list[torch.Tensor] = []
    norms: list[float] = []
    n_bins, n_draws = sigmas.shape
    snap = set(prefix_draws) if prefix_draws else None
    if snap is not None:
        assert n_bins == 1, "prefix_draws needs a single-bin (endpoint-only) grid"
        assert max(snap) == n_draws, "largest prefix must equal draws_per_bin"

    def flat_grad() -> torch.Tensor:
        return torch.cat([p.grad.detach().float().flatten().cpu() for p in params])

    ctx = rope_patch() if rope_patch else nullcontext()
    with ctx as rope_handle:
        for b in range(n_bins):
            for p in params:
                p.grad = None
            j = 0
            while j < n_draws:
                nb = min(batch_draws, n_draws - j)
                if snap is not None:  # never straddle a snapshot boundary
                    nb = min(nb, min(s for s in snap if s > j) - j)
                sig = sigmas[b, j : j + nb]
                if rope_handle is not None and nb > 1 and len(set(sig.tolist())) > 1:
                    nb, sig = 1, sigmas[b, j : j + 1]  # σ-gated rope: uniform σ only
                noise = torch.cat(
                    [
                        torch.randn(
                            lat.shape,
                            generator=torch.Generator(device=device).manual_seed(
                                seeds[b * n_draws + j + k]
                            ),
                            device=device,
                            dtype=lat.dtype,
                        )
                        for k in range(nb)
                    ],
                    dim=0,
                )
                sigma_b = sig.to(device)  # (nb,)
                if rope_handle is not None:  # σ-gated rope (yarnsig arm)
                    rope_handle.set_sigma(float(sig[0]))
                sview = sigma_b.view(-1, 1, 1, 1)
                noisy = (1.0 - sview) * lat + sview * noise
                # target_alpha scales the image in the TARGET only (input
                # untouched); 1.0 * lat is exact, so the default is
                # bit-identical to the historical target
                target = noise - target_alpha * lat
                noisy_5d = noisy.unsqueeze(2).to(torch.bfloat16)
                # nb == 1 must hand the compiled blocks the EXACT tensors the
                # pre-batching code did — an expand() view changes strides,
                # which changes dynamo guards → re-autotuned kernels → fp
                # drift vs. earlier runs' cached graphs
                ca = crossattn if nb == 1 else crossattn.expand(nb, -1, -1)
                pm = pad if nb == 1 else pad.expand(nb, -1, -1, -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = bundle.anima(noisy_5d, sigma_b, ca, padding_mask=pm)
                pred = pred.squeeze(2).float()
                loss = torch.nn.functional.mse_loss(pred, target)
                if nb > 1:  # recover the accumulated sum of per-draw means
                    loss = loss * nb
                loss.backward()
                j += nb
                if snap is not None and j in snap:
                    vec = flat_grad()
                    vecs.append(vec)
                    norms.append(float(vec.norm()))
            if snap is None:
                vec = flat_grad()
                vecs.append(vec)
                norms.append(float(vec.norm()))
    for p in params:
        p.grad = None
    return vecs, norms


def main() -> None:
    args = parse_args()
    if args.deterministic:  # must precede any CUDA/cublas init
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        from networks import attention_dispatch

        attention_dispatch.set_deterministic(True)
        log.info("deterministic mode on (train.py-mirrored knobs)")
    start_heartbeat()
    edges = [int(e) for e in args.demote_edges.split(",") if e]
    if args.yarn_sigma_gate:
        if not args.yarn_align:
            raise SystemExit("--yarn_sigma_gate requires --yarn_align")
        gc, gg = (float(v) for v in args.yarn_sigma_gate.split(","))
        if not (0.0 < gc < 1.0 and gg > 0.0):
            raise SystemExit(
                f"--yarn_sigma_gate needs 0<CENTER<1 and GAMMA>0, got {gc},{gg}"
            )
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
    device = torch.device("cuda")
    lo, hi = (float(v) for v in args.sigma_window.split(","))
    if not 0.0 <= lo < hi <= 1.0:
        raise SystemExit(
            f"--sigma_window must satisfy 0 <= LO < HI <= 1, got {lo},{hi}"
        )
    sigmas = build_sigmas(args.bins, args.draws_per_bin, args.endpoint_bin, lo, hi)
    total_draws = int(sigmas.numel())

    log.info("scoring corpus + selecting probe set (3a-compatible)…")
    artists = args.artists.split(",") if args.artists else None
    if probe_order is not None:
        artists = sorted({a for a, _ in probe_order})
    records = score_corpus(
        artists=artists,
        k=args.quant_k,
        limit=args.score_limit,
        data_root=Path(args.data_root).resolve() if args.data_root else None,
    )
    if probe_order is not None:
        by_key = {(r.artist, r.stem): r for r in records}
        missing = [k for k in probe_order if k not in by_key]
        bad = [
            k
            for k in probe_order
            if k in by_key
            and not (by_key[k].tier == args.tier and by_key[k].complete())
        ]
        if missing or bad:
            raise SystemExit(
                f"--probe_list stems not usable as frozen: "
                f"missing={missing[:5]} incomplete-or-off-tier={bad[:5]}"
            )
        probe = [by_key[k] for k in probe_order]
    else:
        probe = select_probe_set(
            records, args.num_images, tier=args.tier, max_per_artist=args.max_per_artist
        )
    if not probe:
        raise SystemExit(f"no complete tier-{args.tier} records in the scored pool")
    if args.pool:
        probe = sorted(probe, key=lambda r: r.redundancy)
        log.info(
            f"pool mode: {args.pool} images/stratum, sorted by redundancy "
            f"({probe[0].redundancy:.3f}..{probe[-1].redundancy:.3f})"
        )
    log.info(
        f"probe set: {len(probe)} images, {len({r.artist for r in probe})} artists; "
        f"{args.bins} σ-bins × {args.draws_per_bin} draws × "
        f"{2 + int(reenc_control) + len(edges)} arms"
    )

    log.info(
        f"encoding demoted arms ({edges}) + reenc={reenc_control}"
        f" + repromote={args.repromote}…"
    )
    extra_latents = encode_probe_latents(
        probe, edges, args.vae, device, reenc_control, repromote=args.repromote
    )
    if args.x_zero:
        # keep the exact demoted grid shapes, drop all content
        extra_latents = {k: torch.zeros_like(v) for k, v in extra_latents.items()}

    from library.runtime.harness import build_anima, compile_blocks_for_training

    args.gradient_checkpointing = args.grad_ckpt
    args.compile = False  # compile is wired below (needs dynamic-seq marks)
    bundle = build_anima(args, dit_path=args.dit, adapter=args.adapter, train_mode=True)

    if not args.grad_ckpt:
        counts = {r.tokens for r in probe}
        counts.update(
            (t.shape[-2] // 2) * (t.shape[-1] // 2) for t in extra_latents.values()
        )
        compile_blocks_for_training(
            bundle.anima,
            bundle.network,
            backend="inductor",
            mode=None,
            n_token_families=len(counts),
            seq_range=(min(counts), max(counts)),
            dynamic_seq=True,
            activation_memory_budget=args.activation_memory_budget,
            partitioner_aggressive_recomputation=True,
            grad_ckpt=False,
        )

    groups: dict[str, list[tuple[int, int]]] | None = None
    if args.per_group:
        groups = build_groups(bundle.network)
        n_type = sum(1 for g in groups if g.startswith("type:"))
        n_block = sum(1 for g in groups if g.startswith("block:"))
        log.info(f"per-group: {n_type} type groups + {n_block} block groups")

    centers = [round(float(s), 4) for s in sigmas.mean(dim=1)]
    run_dir = make_run_dir(
        "sigma_lowres",
        args.label,
        root=Path(args.results_root).resolve()
        if args.results_root
        else Path(__file__).resolve().parent / "results",
    )
    rows_path = run_dir / "per_image.jsonl"
    rows: list[dict] = []
    arm_keys = ["reenc"] if reenc_control else []
    for e in edges:
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
    arm_sums = ArmSumAccumulator(run_dir / "arm_sums") if args.keep_arm_sums else None
    pool_strata: list[dict] = []
    pool_spill = run_dir / "pool_agg_spill"
    pool_agg = PoolAccumulator(backing_dir=pool_spill)
    cur_pool = PoolAccumulator()

    def finalize_stratum() -> None:
        if not args.pool or cur_pool.n == 0:
            return
        stats = pool_stats(cur_pool, arm_keys)
        pool_strata.append(stats)
        pool_agg.merge(cur_pool)
        pool_agg.release()  # drop memmap handles — RSS-free between merges
        cur_pool.__init__()
        gaps = " ".join(f"gap_{k}@last={stats[f'gap_{k}'][-1]:+.4f}" for k in arm_keys)
        log.info(
            f"[pool s{len(pool_strata) - 1}] n={stats['n_images']} "
            f"redundancy {stats['redundancy_range'][0]:.2f}"
            f"-{stats['redundancy_range'][1]:.2f} "
            f"floor@last={stats['cos_floor'][-1]:.4f} {gaps}"
        )

    t0 = time.time()
    # single worker: per-arm stat jobs are serial among themselves (cheap
    # relative to an arm's forwards) — the point is overlap with the GPU
    stats_pool = ThreadPoolExecutor(max_workers=1)

    for i, r in enumerate(probe):
        crossattn, _ = load_cached_text_features(r.te_path, variant=0)
        if crossattn is None:
            log.info(f"  [{i}] {r.artist}/{r.stem}: no crossattn_emb — skipped")
            continue
        crossattn = crossattn.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        native = load_cached_latents(r.npz_path)[0]
        if args.x_zero:
            native = torch.zeros_like(native)

        def seeds(arm_idx: int, alt: bool = False) -> list[int]:
            # alt (self-floor second draw set): +500_000 keeps the base
            # disjoint from every primary base (i*10_000 < 500_000 for
            # i < 50) without shifting any primary seed vs. earlier runs
            base = (
                args.seed * 1_000_000
                + (500_000 if alt else 0)
                + i * 10_000
                + arm_idx * 1_000
            )
            return [base + d for d in range(total_draws)]

        def estimate(lat_, seed_list, rope_patch=None, alpha=1.0):
            bd = 1
            if args.draw_batch_tokens:
                grid_tokens = (lat_.shape[-2] // 2) * (lat_.shape[-1] // 2)
                bd = max(1, args.draw_batch_tokens // grid_tokens)
                bd = 1 << (bd.bit_length() - 1)  # pow2 → bounded graph count
            return grad_estimate_binned(
                bundle,
                lat_,
                crossattn,
                sigmas,
                seed_list,
                rope_patch=rope_patch,
                prefix_draws=sweep,
                batch_draws=bd,
                target_alpha=alpha,
            )

        row = {
            "artist": r.artist,
            "stem": r.stem,
            "redundancy": round(r.redundancy, 4),
            "tokens_native": r.tokens,
            "sigma_centers": centers,
        }
        if probe_tags is not None:
            row.update(probe_tags[(r.artist, r.stem)])
        if sweep:
            row["draw_prefixes"] = sweep

        # --target_alpha: one full pass per alpha. arm_idx (and therefore the
        # seeds) resets per alpha, so every alpha sees the SAME noise draws —
        # the alpha-slope is free of draw noise, and the seed-space bounds
        # (arm_idx * 1_000 < i * 10_000 spacing) are preserved.
        kap0: dict[str, list[torch.Tensor]] = {}
        for alpha_ in alphas:
            sfx = "" if alpha_ == 1.0 else f"@a{alpha_:g}"
            g_a, n_a = estimate(native, seeds(0), alpha=alpha_)
            g_b, n_b = estimate(native, seeds(1), alpha=alpha_)
            floor = [cosine(a, b) for a, b in zip(g_a, g_b)]
            row[f"cos_floor{sfx}"] = [round(c, 5) for c in floor]
            row[f"gnorm_native{sfx}"] = [
                round(0.5 * (x + y), 3) for x, y in zip(n_a, n_b)
            ]

            def debias(g_d, g_d2) -> tuple[list[float], list[float]]:
                """cos_self per bin + debiased gap 1 − cos(a+b, d+d′)/√(floor·self).
                NaN where either floor is ≤ 0 (attenuation correction undefined —
                only plausible in the lowest-σ bins, never near the endpoint)."""
                selfc, dgap = [], []
                for a, b, fl, d, d2 in zip(g_a, g_b, floor, g_d, g_d2):
                    sc = cosine(d, d2)
                    selfc.append(round(sc, 5))
                    if fl <= 0 or sc <= 0:
                        dgap.append(float("nan"))
                    else:
                        dgap.append(
                            round(1.0 - cosine(a + b, d + d2) / math.sqrt(fl * sc), 5)
                        )
                return selfc, dgap

            floor_g: list[dict[str, float]] = []
            if args.per_group:
                floor_g = [grouped_cosine(a, b, groups) for a, b in zip(g_a, g_b)]
                row["cosg_floor"] = {
                    g: [round(fb[g], 5) for fb in floor_g] for g in groups
                }

            def grouped_gaps(g_arm: list[torch.Tensor]) -> dict[str, list[float]]:
                out: dict[str, list[float]] = {g: [] for g in groups}
                for bi, (a, b, d) in enumerate(zip(g_a, g_b, g_arm)):
                    ca = grouped_cosine(a, d, groups)
                    cb = grouped_cosine(b, d, groups)
                    for g in groups:
                        out[g].append(round(floor_g[bi][g] - 0.5 * (ca[g] + cb[g]), 5))
                return out

            def arm_stats(key, g_d, n_d, g_d2):
                """Per-arm CPU bookkeeping — runs on the stats worker thread so
                the 77M-element dots overlap the NEXT arm's GPU forwards (torch
                CPU ops release the GIL). Reads g_a/g_b/floor/floor_g read-only;
                results merged into ``row`` when the image's futures resolve."""
                out = {}
                c = [
                    0.5 * (cosine(a, g) + cosine(b, g))
                    for a, b, g in zip(g_a, g_b, g_d)
                ]
                out[f"cos_{key}"] = [round(v, 5) for v in c]
                out[f"gap_{key}"] = [round(f - v, 5) for f, v in zip(floor, c)]
                if n_d is not None:
                    out[f"gnorm_{key}"] = [round(v, 3) for v in n_d]
                if args.per_group:
                    out[f"gapg_{key}"] = grouped_gaps(g_d)
                if g_d2 is not None:
                    out[f"cos_self_{key}"], out[f"debiased_gap_{key}"] = debias(
                        g_d, g_d2
                    )
                return out

            stat_futs = []
            arms: dict[str, list[torch.Tensor]] = {"a": g_a, "b": g_b}
            arm_idx = 2
            if reenc_control:
                re_lat = extra_latents[(r.stem, "reenc")]
                g_re, _ = estimate(re_lat, seeds(arm_idx), alpha=alpha_)
                g_re2 = None
                if args.self_floor:
                    g_re2, _ = estimate(re_lat, seeds(arm_idx, alt=True), alpha=alpha_)
                    arms["reenc__2"] = g_re2
                stat_futs.append(
                    stats_pool.submit(arm_stats, "reenc" + sfx, g_re, None, g_re2)
                )
                arm_idx += 1
                arms["reenc"] = g_re
            demote_arms: list = []  # (key, latent, rope-patch factory | None)
            for e in edges:
                lat = extra_latents[(r.stem, f"demote{e}")]
                demote_arms.append((str(e), lat, None))
                if args.repromote:
                    demote_arms.append(
                        (f"{e}rp", extra_latents[(r.stem, f"repromote{e}")], None)
                    )
                # per-axis stretch in patch units: demoted patch i sits at
                # i * (native_patches / demoted_patches), spanning the native
                # coordinate range so relative RoPE phases match native's
                hs = (native.shape[-2] // 2) / (lat.shape[-2] // 2)
                ws = (native.shape[-1] // 2) / (lat.shape[-1] // 2)
                if args.pi_align:
                    demote_arms.append(
                        (f"{e}pi", lat, partial(pi_rope, bundle.anima, hs, ws))
                    )
                if args.yarn_align:
                    a, b_ = (float(v) for v in args.yarn_align.split(","))
                    demote_arms.append(
                        (
                            f"{e}yarn",
                            lat,
                            partial(yarn_rope, bundle.anima, hs, ws, a, b_),
                        )
                    )
                    if args.yarn_sigma_gate:
                        c_, g_ = (float(v) for v in args.yarn_sigma_gate.split(","))
                        demote_arms.append(
                            (
                                f"{e}yarnsig",
                                lat,
                                partial(
                                    yarn_rope, bundle.anima, hs, ws, a, b_, (c_, g_)
                                ),
                            )
                        )
            for key, lat, patch in demote_arms:
                g_d, n_d = estimate(lat, seeds(arm_idx), rope_patch=patch, alpha=alpha_)
                g_d2 = None
                if args.self_floor:
                    g_d2, _ = estimate(
                        lat, seeds(arm_idx, alt=True), rope_patch=patch, alpha=alpha_
                    )
                    arms[f"{key}__2"] = g_d2
                stat_futs.append(
                    stats_pool.submit(arm_stats, key + sfx, g_d, n_d, g_d2)
                )
                arm_idx += 1
                arms[key] = g_d
            for fut in stat_futs:  # join stats before pooling/freeing this image
                row.update(fut.result())
            if args.pool:
                cur_pool.add_image(arms, r.redundancy)
                if cur_pool.n == args.pool:
                    finalize_stratum()
            if arm_sums is not None:
                for k, v in arms.items():
                    arm_sums.add(k + sfx, v)
            if args.target_kappa and alpha_ == 0.0:
                # tensors survive the free below (only the LISTS are cleared)
                kap0 = {k: list(v) for k, v in arms.items()}
            if args.target_kappa and alpha_ == 1.0 and kap0:
                # exact target-content gradients: the forward pass is
                # alpha-independent and seeds are shared across alphas, so
                # t = g(1) - g(0) = E_draws[J^T x] with zero draw noise
                inv2d = 1.0 / (2.0 * args.draws_per_bin)
                invd = 1.0 / args.draws_per_bin
                kap: dict[str, list[float]] = {}
                for bi in range(len(arms["a"])):
                    a1, b1 = arms["a"][bi].double(), arms["b"][bi].double()
                    a0, b0 = kap0["a"][bi].double(), kap0["b"][bi].double()
                    src = (a1 + b1) * inv2d
                    g_norm = float(src.norm())
                    ghat = src / g_norm
                    t_src = ((a1 - a0) + (b1 - b0)) * inv2d
                    t_null = ((a1 - a0) - (b1 - b0)) * inv2d
                    del src

                    def kap_of(dt: torch.Tensor, name: str) -> None:
                        par = float(torch.dot(ghat, dt))
                        perp = math.sqrt(max(0.0, float(dt.norm()) ** 2 - par * par))
                        kap.setdefault(f"kappa_par_{name}", []).append(
                            round(par / g_norm, 6)
                        )
                        kap.setdefault(f"kappa_perp_{name}", []).append(
                            round(perp / g_norm, 6)
                        )

                    kap.setdefault("tnorm_src", []).append(
                        round(float(t_src.norm()) / g_norm, 6)
                    )
                    kap_of(t_null, "null")  # a-vs-b draw-noise floor of t_src
                    for k in arms:
                        if k in ("a", "b"):
                            continue
                        t_k = (arms[k][bi].double() - kap0[k][bi].double()) * invd
                        kap.setdefault(f"tnorm_{k}", []).append(
                            round(float(t_k.norm()) / g_norm, 6)
                        )
                        kap_of(t_k - t_src, k)
                        del t_k
                row.update(kap)
                kap0 = {}
                log.info(
                    "  [kappa] "
                    + " ".join(
                        f"{k[10:]}=({row[k][-1]:+.4f},{row['kappa_perp_' + k[10:]][-1]:.4f})"
                        for k in row
                        if k.startswith("kappa_par_")
                    )
                )
            # free this image's ~8 GB of flat gradient vectors now — otherwise
            # the locals keep them resident through the next image's compute
            for vecs in arms.values():
                vecs.clear()
            g_a = g_b = arms = None  # noqa: F841
        rows.append(row)
        with rows_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        gap_str = " ".join(
            f"{k}={[f'{v:+.3f}' for v in row[k]]}" for k in row if k.startswith("gap_")
        )
        log.info(f"  [{i + 1}/{len(probe)}] {r.artist}/{r.stem} {gap_str}")

    if not rows:
        raise SystemExit("no per-image rows produced")
    finalize_stratum()  # remainder stratum (may be smaller than --pool)

    def bin_stats(key: str) -> dict:
        # nan-aware: debiased gaps are NaN where a floor was <= 0
        m = np.array([r[key] for r in rows], dtype=np.float64)  # (n_images, bins)
        n_fin = np.isfinite(m).sum(axis=0).clip(min=1)
        mean = np.nanmean(m, axis=0)
        sem = np.nanstd(m, axis=0, ddof=1) / np.sqrt(n_fin)
        # split-half reliability of the bin-mean curve (odd/even image split)
        h1, h2 = np.nanmean(m[0::2], axis=0), np.nanmean(m[1::2], axis=0)
        return {
            "mean": [round(float(v), 5) for v in mean],
            "sem": [round(float(v), 5) for v in sem],
            "spearman_sigma": round(spearman(np.arange(len(mean)), mean), 4)
            if len(mean) > 1
            else None,
            "splithalf_pearson": round(float(np.corrcoef(h1, h2)[0, 1]), 4)
            if len(mean) > 2
            else None,
        }

    headline: dict = {
        "n_images": len(rows),
        "bins": args.bins,
        "draws_per_bin": args.draws_per_bin,
        "sigma_centers": centers,
        "adapter": args.adapter,
        "cos_floor": bin_stats("cos_floor"),
        "gnorm_native": bin_stats("gnorm_native"),
        "wall_time_s": round(time.time() - t0, 1),
    }
    for k in arm_keys:  # reenc + demote edges + any <edge>pi arms
        headline[f"gap_{k}"] = bin_stats(f"gap_{k}")
        if args.self_floor:
            headline[f"cos_self_{k}"] = bin_stats(f"cos_self_{k}")
            headline[f"debiased_gap_{k}"] = bin_stats(f"debiased_gap_{k}")

    if len(alphas) > 1:
        headline["target_alphas"] = alphas
        for alpha_ in alphas:
            if alpha_ == 1.0:
                continue
            sfx = f"@a{alpha_:g}"
            headline[f"cos_floor{sfx}"] = bin_stats(f"cos_floor{sfx}")
            headline[f"gnorm_native{sfx}"] = bin_stats(f"gnorm_native{sfx}")
            for k in arm_keys:
                headline[f"gap_{k}{sfx}"] = bin_stats(f"gap_{k}{sfx}")
                if args.self_floor:
                    headline[f"cos_self_{k}{sfx}"] = bin_stats(f"cos_self_{k}{sfx}")
                    headline[f"debiased_gap_{k}{sfx}"] = bin_stats(
                        f"debiased_gap_{k}{sfx}"
                    )
        if args.self_floor:
            for k in arm_keys:  # E2 readout: last bin = endpoint under --endpoint_bin
                pts = [
                    (
                        a,
                        headline[
                            f"debiased_gap_{k}" + ("" if a == 1.0 else f"@a{a:g}")
                        ]["mean"][-1],
                    )
                    for a in alphas
                ]
                slope = float(
                    np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]
                )
                headline[f"alpha_slope_{k}"] = round(slope, 5)
                log.info(
                    f"[alpha-sweep] debiased_gap_{k}@last: "
                    + " ".join(f"a{a:g}={v:+.4f}" for a, v in pts)
                    + f" slope={slope:+.4f}"
                )

    if args.target_kappa:
        for key in sorted(
            k for k in rows[0] if k.startswith(("kappa_par_", "kappa_perp_", "tnorm_"))
        ):
            headline[key] = bin_stats(key)
        for key in sorted(k for k in headline if k.startswith("kappa_par_")):
            arm = key[len("kappa_par_") :]
            log.info(
                f"[target-kappa] {arm}: par={headline[key]['mean'][-1]:+.5f}"
                f"±{headline[key]['sem'][-1]:.5f} "
                f"perp={headline[f'kappa_perp_{arm}']['mean'][-1]:+.5f}"
                f"±{headline[f'kappa_perp_{arm}']['sem'][-1]:.5f}"
            )

    if sweep:
        headline["draw_prefixes"] = sweep
        x = 1.0 / np.asarray(sweep, dtype=np.float64)

        def sweep_fit(key: str) -> dict:
            """Least-squares fit y(D) = y_inf + c/D on the per-image-mean
            curve; 95% CI on y_inf from a 2000-resample image bootstrap."""
            m = np.array([r[key] for r in rows], dtype=np.float64)

            def fit(y: np.ndarray) -> tuple[float, float]:
                msk = np.isfinite(y)
                if msk.sum() < 2:
                    return float("nan"), float("nan")
                a_mat = np.stack([np.ones(int(msk.sum())), x[msk]], axis=1)
                coef, *_ = np.linalg.lstsq(a_mat, y[msk], rcond=None)
                return float(coef[0]), float(coef[1])

            y_inf, c_ = fit(np.nanmean(m, axis=0))
            rng = np.random.default_rng(args.seed)
            n = m.shape[0]
            boots = [
                fit(np.nanmean(m[rng.integers(0, n, n)], axis=0))[0]
                for _ in range(2000)
            ]
            lo_b, hi_b = np.nanpercentile(boots, [2.5, 97.5])
            return {
                "y_inf": round(y_inf, 5),
                "c": round(c_, 5),
                "y_inf_ci95": [round(float(lo_b), 5), round(float(hi_b), 5)],
            }

        headline["drawfit_floor"] = sweep_fit("cos_floor")
        for k in arm_keys:
            headline[f"drawfit_gap_{k}"] = sweep_fit(f"gap_{k}")
            if args.self_floor:
                headline[f"drawfit_debiased_{k}"] = sweep_fit(f"debiased_gap_{k}")
        for k in arm_keys:
            f_ = headline[f"drawfit_gap_{k}"]
            log.info(
                f"[drawfit] gap_{k}: gap_inf={f_['y_inf']:+.4f} "
                f"ci95=[{f_['y_inf_ci95'][0]:+.4f},{f_['y_inf_ci95'][1]:+.4f}] "
                f"c={f_['c']:+.4f}"
            )

    if args.pool and pool_strata:
        # stratum-level redundancy trend: spearman of stratum redundancy mean
        # vs pooled gap at the last (highest-σ) bin, per demote edge
        trend = {}
        if len(pool_strata) > 2:
            red = np.array([s["redundancy_mean"] for s in pool_strata])
            for e in edges:
                g = np.array([s[f"gap_{e}"][-1] for s in pool_strata])
                trend[f"spearman_redundancy_gap_{e}"] = round(spearman(red, g), 4)
        headline["pool"] = {
            "size": args.pool,
            "strata": pool_strata,
            "aggregate": pool_stats(pool_agg, arm_keys),
            **trend,
        }
        del pool_agg  # release memmap handles before removing the spill
        shutil.rmtree(pool_spill, ignore_errors=True)

    if args.per_group:

        def group_stats(key: str) -> dict:
            out = {}
            for g in rows[0][key]:
                m = np.array([r[key][g] for r in rows])
                mean = m.mean(axis=0)
                sem = m.std(axis=0, ddof=1) / np.sqrt(m.shape[0])
                h1, h2 = m[0::2].mean(axis=0), m[1::2].mean(axis=0)
                out[g] = {
                    "mean": [round(float(v), 5) for v in mean],
                    "sem": [round(float(v), 5) for v in sem],
                    "splithalf_pearson": round(float(np.corrcoef(h1, h2)[0, 1]), 4)
                    if len(mean) > 2
                    else None,
                }
            return out

        for key in [k for k in rows[0] if k.startswith("gapg_")]:
            headline[key] = group_stats(key)
        for e in edges:
            stats = headline.get(f"gapg_{e}")
            if not stats:
                continue
            tg = {
                g[5:]: v["mean"][-1] for g, v in stats.items() if g.startswith("type:")
            }
            ranked = sorted(tg.items(), key=lambda kv: -kv[1])
            log.info(
                f"[per-group] {e} type gaps @ last bin: "
                + ", ".join(f"{n}={v:+.3f}" for n, v in ranked)
            )

    if arm_sums is not None:
        arm_sums.finalize(
            {
                "adapter": args.adapter,
                "sigma_centers": centers,
                "bins": args.bins,
                "endpoint_bin": args.endpoint_bin,
                "draws_per_bin": args.draws_per_bin,
                "target_alphas": alphas,
                "arm_keys": arm_keys,
                "self_floor": args.self_floor,
                "repromote": args.repromote,
                "demote_edges": args.demote_edges,
                "n_images": len(rows),
                "seed": args.seed,
            }
        )
        log.info(f"arm sums → {run_dir / 'arm_sums'} ({len(arm_sums.maps)} vectors)")

    log.info(json.dumps(headline, indent=2))
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=headline,
        label=args.label,
        artifacts=[rows_path],
        extra={"probe_set": [f"{r['artist']}/{r['stem']}" for r in rows]},
    )
    log.info(f"result → {run_dir}")


if __name__ == "__main__":
    main()
