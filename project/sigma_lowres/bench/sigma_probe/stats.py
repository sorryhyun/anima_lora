"""Every reduction ``run_sigma_probe.py`` performs on flat gradient vectors.

Three layers, none of which touch the GPU:

* **accumulators** — :class:`PoolAccumulator` (the batch-SGD aggregate object,
  optionally disk-backed) and :class:`ArmSumAccumulator` (the vector-ledger
  retention behind ``--keep_arm_sums``).
* **per-image** — :class:`ArmStatter` (cos/gap/debias/per-group for one arm
  against one image's native draw pair) and :func:`kappa_row` (the exact
  target-content decomposition behind ``--target_kappa``).
* **run-level** — :func:`build_headline`, which turns the per-image rows into
  the result envelope's ``metrics``.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from project.sigma_lowres.bench.tier_routing.run_grad_probe import (  # noqa: E402
    cosine,
    spearman,
)

log = logging.getLogger(__name__)


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

    def __init__(self, backing_dir: Path | None = None, keep_norm: bool = True) -> None:
        self.backing_dir = backing_dir
        self.keep_norm = keep_norm
        self._numel: int | None = None
        self.sums: dict[str, list[torch.Tensor]] = {}
        self.nsums: dict[str, list[torch.Tensor]] = {}
        self.halves: dict[int, list[torch.Tensor]] = {}
        self.n = 0
        self.redundancy: list[float] = []

    def reset(self) -> None:
        """Empty the accumulator for the next stratum, keeping the backing
        config (``__init__`` would drop ``backing_dir``). Backed mode reuses
        the same on-disk files — ``_materialize`` reopens them ``w+``."""
        self.sums, self.nsums, self.halves = {}, {}, {}
        self.n = 0
        self.redundancy = []

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

    def add_arm(self, key: str, vecs: list[torch.Tensor]) -> None:
        """Add one arm's per-bin vectors for the current image. Safe to call
        as each arm finishes (streaming) — per-key accumulation order is
        identical to a whole-image ``add_image``, so results are bit-equal."""
        self.ensure_open()
        self._add("sums", self.sums, key, vecs)
        if self.keep_norm:
            self._add(
                "nsums",
                self.nsums,
                key,
                vecs,
                [1.0 / (float(v.norm()) + 1e-12) for v in vecs],
            )

    def add_native(
        self,
        g_a: list[torch.Tensor],
        g_b: list[torch.Tensor],
        redundancy: float,
    ) -> None:
        """Add the native draw pair and close out the current image (halves
        parity split + image count). Call once per image, after ``add_arm``."""
        self.add_arm("a", g_a)
        self.add_arm("b", g_b)
        native = [a + b for a, b in zip(g_a, g_b)]
        self._add("halves", self.halves, self.n % 2, native)
        self.n += 1
        self.redundancy.append(redundancy)

    def add_image(self, arms: dict[str, list[torch.Tensor]], redundancy: float) -> None:
        for key, vecs in arms.items():
            if key not in ("a", "b"):
                self.add_arm(key, vecs)
        self.add_native(arms["a"], arms["b"], redundancy)

    def merge(self, other: "PoolAccumulator") -> None:
        self.ensure_open()
        other.ensure_open()
        for name, mine in self._stores():
            theirs = getattr(other, name)
            for key, vecs in theirs.items():
                self._add(name, mine, key, vecs)
        self.n += other.n
        self.redundancy.extend(other.redundancy)


class ArmSumAccumulator:
    """Cross-image sums of per-bin flat LoRA gradients, one disk memmap
    per (arm key, bin) under ``dir/``. Unlike :class:`PoolAccumulator` this
    keeps EVERY arm (native a/b, alt draw sets, every target-alpha suffix)
    and survives the run — the raw material for the interventional
    vector ledger (B/C split, kappa components, exact angles). Keys are
    sanitized for filenames (``@`` → ``~``); ``manifest.json`` records the
    mapping plus the scale convention (sum over images of per-image
    draw-summed gradients).

    ``dtype`` fp16 halves the store (a full repromote×self-floor 15-bin
    grid is ~75 GB fp32 — over this machine's disk headroom); the fp16
    accumulation rounds at ~1e-3 relative per add, well under the
    ledger's read precision (cosines/κ at two significant figures).
    """

    def __init__(self, root: Path, dtype: str = "fp32") -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.dtype = np.float16 if dtype == "fp16" else np.float32
        self.dtype_name = dtype
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
                mm = open_memmap(path, mode="w+", dtype=self.dtype, shape=(v.numel(),))
                self.maps[(key, bi)] = mm
            mm += v.numpy().astype(self.dtype, copy=False)
        self.n_images[key] = self.n_images.get(key, 0) + 1

    def finalize(self, meta: dict) -> None:
        from bench._common import boot_fingerprint

        for mm in self.maps.values():
            mm.flush()
        keys = sorted({k for k, _ in self.maps})
        manifest = {
            **meta,
            # T0 (roadmap §3): stamped here, not by the caller, so no
            # entry point can produce an unfingerprinted store
            "boot_fingerprint": boot_fingerprint(),
            "dtype": self.dtype_name,
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
        if not store:  # norm side-channel skipped (keep_norm=False)
            continue
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


class ArmStatter:
    """Per-arm cosine/gap bookkeeping for ONE image (and one target-alpha),
    against that image's native draw pair.

    ``stats()`` is what the driver hands to the stats worker thread, so the
    77M-element dots overlap the NEXT arm's GPU forwards (torch CPU ops
    release the GIL). It reads the native vectors read-only; results are
    merged into the image's row when its futures resolve.
    """

    def __init__(
        self,
        g_a: list[torch.Tensor],
        g_b: list[torch.Tensor],
        floor: list[float],
        groups: dict[str, list[tuple[int, int]]] | None = None,
        floor_g: list[dict[str, float]] | None = None,
    ) -> None:
        self.g_a, self.g_b, self.floor = g_a, g_b, floor
        self.groups, self.floor_g = groups, floor_g

    def _debias(self, g_d, g_d2) -> tuple[list[float], list[float]]:
        """cos_self per bin + debiased gap 1 − cos(a+b, d+d′)/√(floor·self).
        NaN where either floor is ≤ 0 (attenuation correction undefined —
        only plausible in the lowest-σ bins, never near the endpoint)."""
        selfc, dgap = [], []
        for a, b, fl, d, d2 in zip(self.g_a, self.g_b, self.floor, g_d, g_d2):
            sc = cosine(d, d2)
            selfc.append(round(sc, 5))
            if fl <= 0 or sc <= 0:
                dgap.append(float("nan"))
            else:
                dgap.append(round(1.0 - cosine(a + b, d + d2) / math.sqrt(fl * sc), 5))
        return selfc, dgap

    def _grouped_gaps(self, g_arm: list[torch.Tensor]) -> dict[str, list[float]]:
        from .kernel import grouped_cosine

        out: dict[str, list[float]] = {g: [] for g in self.groups}
        for bi, (a, b, d) in enumerate(zip(self.g_a, self.g_b, g_arm)):
            ca = grouped_cosine(a, d, self.groups)
            cb = grouped_cosine(b, d, self.groups)
            for g in self.groups:
                out[g].append(round(self.floor_g[bi][g] - 0.5 * (ca[g] + cb[g]), 5))
        return out

    def stats(self, key: str, g_d, n_d=None, g_d2=None) -> dict:
        out = {}
        c = [
            0.5 * (cosine(a, g) + cosine(b, g))
            for a, b, g in zip(self.g_a, self.g_b, g_d)
        ]
        out[f"cos_{key}"] = [round(v, 5) for v in c]
        out[f"gap_{key}"] = [round(f - v, 5) for f, v in zip(self.floor, c)]
        if n_d is not None:
            out[f"gnorm_{key}"] = [round(v, 3) for v in n_d]
        if self.groups:
            out[f"gapg_{key}"] = self._grouped_gaps(g_d)
        if g_d2 is not None:
            out[f"cos_self_{key}"], out[f"debiased_gap_{key}"] = self._debias(g_d, g_d2)
        return out


def kappa_row(
    arms: dict[str, list[torch.Tensor]],
    kap0: dict[str, list[torch.Tensor]],
    draws_per_bin: int,
) -> dict[str, list[float]]:
    """``--target_kappa``: exact target-content gradients.

    The forward pass is alpha-independent and seeds are shared across alphas,
    so ``t = g(1) − g(0) = E_draws[J^T x]`` with zero draw noise. Reports, per
    arm, the component of ``t_k − t_src`` parallel to the source gradient
    direction and the perpendicular magnitude, both in units of ‖g_src‖, plus
    the a-vs-b null and the ‖t‖/G observability norms."""
    inv2d = 1.0 / (2.0 * draws_per_bin)
    invd = 1.0 / draws_per_bin
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
            kap.setdefault(f"kappa_par_{name}", []).append(round(par / g_norm, 6))
            kap.setdefault(f"kappa_perp_{name}", []).append(round(perp / g_norm, 6))

        kap.setdefault("tnorm_src", []).append(round(float(t_src.norm()) / g_norm, 6))
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
    log.info(
        "  [kappa] "
        + " ".join(
            f"{k[10:]}=({kap[k][-1]:+.4f},{kap['kappa_perp_' + k[10:]][-1]:.4f})"
            for k in kap
            if k.startswith("kappa_par_")
        )
    )
    return kap


# --------------------------------------------------------------------------
# E22 per-image B/C ledger (--per_image_ledger)
#
# The E14/E19 interventional ledger with the IMAGE as the slice: same legs
# (B = g_rp − g_reenc, C = g_dem − g_rp), same noise handling (second moments
# from CROSS-SET products only, ref-noise subtracted from the reenc set-diff,
# same-set values as bias checks), computed from ONE image's arm gradients
# before any cross-image accumulation. Granularities mirror E21: global
# (verdict), the four type bands (secondary), depth-block × core-type cells
# (exploratory). Slice rows carry both the slice-local ledger (perp against
# the slice's own native direction) and the additive global-perp partition
# (Sp/Fp/Ip — core cells resum to the global row exactly). Conventions are
# frozen in experiments/e22/README.md; the math mirrors
# paper_bench/vector_ledger.bc_ledger and e21_cells.cell_row verbatim.

E21_CORE_TYPES = (
    "adaln_up_self_attn",
    "adaln_up_cross_attn",
    "adaln_up_mlp",
    "self_attn_qkv_proj",
    "self_attn_output_proj",
    "cross_attn_q_proj",
    "cross_attn_kv_proj",
    "cross_attn_output_proj",
    "mlp_layer1",
    "mlp_layer2",
)
E21_BANDS = {
    "adaln": ("adaln_up_self_attn", "adaln_up_cross_attn", "adaln_up_mlp"),
    "cross_attn": (
        "cross_attn_q_proj",
        "cross_attn_kv_proj",
        "cross_attn_output_proj",
    ),
    "self_attn": ("self_attn_qkv_proj", "self_attn_output_proj"),
    "mlp": ("mlp_layer1", "mlp_layer2"),
}


def _intersect_ranges(a, b) -> list[tuple[int, int]]:
    a, b = sorted(a), sorted(b)
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def build_ledger_slices(groups: dict) -> dict[str, list[tuple[int, int]]]:
    """``band:<name>`` + ``cell:bXX|<type>`` → flat-vector ranges, from the
    ``build_groups`` map (same sorted-name flatten as the gradient vectors)."""
    slices: dict[str, list[tuple[int, int]]] = {}
    for band, types in E21_BANDS.items():
        slices[f"band:{band}"] = sorted(
            tuple(r) for t in types for r in groups[f"type:{t}"]
        )
    for bk in sorted(k for k in groups if k.startswith("block:")):
        bi = int(bk.split(":")[1])
        for t in E21_CORE_TYPES:
            rr = _intersect_ranges(groups[bk], groups[f"type:{t}"])
            if rr:
                slices[f"cell:b{bi:02d}|{t}"] = rr
    return slices


def _gather(v: np.ndarray, ranges) -> np.ndarray:
    if len(ranges) == 1:
        s, e = ranges[0]
        return v[s:e]
    return np.concatenate([v[s:e] for s, e in ranges])


def _np_cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(a @ b) / (na * nb)


def _ledger_global_row(g0, G, ghat, B1, B2, C1, C2, rd) -> dict:
    """bc_ledger (reenc ref) for one image's (bin, route) + sig_rho (the
    cross-pairing half-spread, e21's twin-based per-row noise scale)."""

    def perp(v):
        return v - float(ghat @ v) * ghat

    B1p, B2p, C1p, C2p = perp(B1), perp(B2), perp(C1), perp(C2)
    rdp = perp(rd)
    G2 = G * G
    ref_noise = float(rdp @ rdp) / 4.0
    S = (float(B1p @ B2p) - ref_noise) / (2 * G2)
    F = float(C1p @ C2p) / (2 * G2)
    i1, i2 = float(B1p @ C2p) / G2, float(B2p @ C1p) / G2
    I_x = 0.5 * (i1 + i2)
    I_same = (float(B1p @ C1p) + float(B2p @ C2p)) / (2 * G2)
    denom = 2.0 * math.sqrt(max(S * F, 0.0))
    Bm, Cm = 0.5 * (B1 + B2), 0.5 * (C1 + C2)

    def h(u):
        return 1.0 - _np_cos(g0, g0 + u)

    return {
        "G": round(G, 5),
        "b_perp_rawnorm": round(float(np.linalg.norm(0.5 * (B1p + B2p))) / G, 5),
        "c_perp_rawnorm": round(float(np.linalg.norm(0.5 * (C1p + C2p))) / G, 5),
        "kappa_par_B": round(float(ghat @ Bm) / G, 5),
        "kappa_par_C": round(float(ghat @ Cm) / G, 5),
        "rel_cos_B": round(_np_cos(B1p, B2p), 5),
        "rel_cos_C": round(_np_cos(C1p, C2p), 5),
        "ref_noise_over_2G2": round(ref_noise / (2 * G2), 6),
        "S": round(S, 6),
        "F": round(F, 6),
        "I": round(I_x, 6),
        "I_sameset_biascheck": round(I_same, 6),
        "rho": round(I_x / denom, 5) if denom > 0.0 else float("nan"),
        "sig_rho": round((abs(i1 - i2) / 2.0) / denom, 5)
        if denom > 0.0
        else float("nan"),
        "amp_ratio": round(math.sqrt(S / F), 5) if S > 0 and F > 0 else float("nan"),
        "quad_pred_gap": round(S + F + I_x, 6),
        "h_B": round(h(Bm), 6),
        "h_C": round(h(Cm), 6),
        "h_B_plus_C": round(h(Bm + Cm), 6),
    }


def _ledger_slice_row(g0l, B1l, B2l, C1l, C2l, rdl, scal: dict) -> dict:
    """e21_cells.cell_row math (no pi extension), one slice of one image:
    slice-local ledger + the additive global-perp partition (Sp/Fp/Ip)."""
    gh_g = g0l / scal["G"]
    G2g = 2.0 * scal["G"] ** 2
    B1pg, B2pg = B1l - scal["cB1"] * gh_g, B2l - scal["cB2"] * gh_g
    C1pg, C2pg = C1l - scal["cC1"] * gh_g, C2l - scal["cC2"] * gh_g
    rdpg = rdl - scal["cRd"] * gh_g
    S_part = (float(B1pg @ B2pg) - float(rdpg @ rdpg) / 4.0) / G2g
    F_part = float(C1pg @ C2pg) / G2g
    I_part = (float(B1pg @ C2pg) + float(B2pg @ C1pg)) / G2g

    Gl = float(np.linalg.norm(g0l))
    ghl = g0l / Gl
    B1p, B2p = B1l - float(ghl @ B1l) * ghl, B2l - float(ghl @ B2l) * ghl
    C1p, C2p = C1l - float(ghl @ C1l) * ghl, C2l - float(ghl @ C2l) * ghl
    rdp = rdl - float(ghl @ rdl) * ghl
    G2 = 2.0 * Gl * Gl
    S = (float(B1p @ B2p) - float(rdp @ rdp) / 4.0) / G2
    F = float(C1p @ C2p) / G2
    i1, i2 = float(B1p @ C2p) / G2 * 2.0, float(B2p @ C1p) / G2 * 2.0
    I_x = 0.5 * (i1 + i2)
    I_same = (float(B1p @ C1p) + float(B2p @ C2p)) / G2
    denom = 2.0 * math.sqrt(max(S * F, 0.0))
    return {
        "G_l": round(Gl, 5),
        "S": round(S, 6),
        "F": round(F, 6),
        "I": round(I_x, 6),
        "I_same": round(I_same, 6),
        "rho": round(I_x / denom, 5) if denom > 0.0 else float("nan"),
        "sig_rho": round((abs(i1 - i2) / 2.0) / denom, 5)
        if denom > 0.0
        else float("nan"),
        "relB": round(_np_cos(B1p, B2p), 5),
        "relC": round(_np_cos(C1p, C2p), 5),
        "amp": round(math.sqrt(S / F), 5) if S > 0 and F > 0 else float("nan"),
        "Sp": round(S_part, 6),
        "Fp": round(F_part, 6),
        "Ip": round(I_part, 6),
    }


def image_ledger(
    arms: dict[str, list[torch.Tensor]],
    edges: list[str],
    slices: dict[str, list[tuple[int, int]]],
    draws: int,
) -> dict:
    """Per-image scalar B/C reductions for every (route, bin) — the E22
    instrument. ``arms`` is one image's retained arm dict (draw-summed fp32
    vectors); vectors are scaled to per-draw means (uniform 1/draws — all
    arms share the grid) so h() magnitudes match the pooled ledger's
    convention. Returns {route: [per-bin {global, bands, cells}]}."""
    n_bins = len(arms["a"])
    inv = 1.0 / draws
    out: dict = {e: [] for e in edges}
    for bi in range(n_bins):

        def f64(key: str) -> np.ndarray:
            return arms[key][bi].numpy().astype(np.float64) * inv

        a, b = f64("a"), f64("b")
        g0 = 0.5 * (a + b)
        del a, b
        G = float(np.linalg.norm(g0))
        ghat = g0 / G
        re1, re2 = f64("reenc"), f64("reenc__2")
        ref, rd = 0.5 * (re1 + re2), re1 - re2
        del re1, re2
        cRd = float(ghat @ rd)
        for e in edges:
            rp1, rp2 = f64(f"{e}rp"), f64(f"{e}rp__2")
            dem1, dem2 = f64(e), f64(f"{e}__2")
            B1, B2 = rp1 - ref, rp2 - ref
            C1, C2 = dem1 - rp1, dem2 - rp2
            del rp1, rp2, dem1, dem2
            entry = {
                "global": _ledger_global_row(g0, G, ghat, B1, B2, C1, C2, rd),
                "bands": {},
                "cells": {},
            }
            scal = {
                "G": G,
                "cB1": float(ghat @ B1),
                "cB2": float(ghat @ B2),
                "cC1": float(ghat @ C1),
                "cC2": float(ghat @ C2),
                "cRd": cRd,
            }
            for name, rr in slices.items():
                row = _ledger_slice_row(
                    _gather(g0, rr),
                    _gather(B1, rr),
                    _gather(B2, rr),
                    _gather(C1, rr),
                    _gather(C2, rr),
                    _gather(rd, rr),
                    scal,
                )
                kind, key = name.split(":", 1)
                entry[kind + "s"][key] = row
            out[e].append(entry)
            del B1, B2, C1, C2
        del g0, ghat, ref, rd
    return out


def bin_stats(rows: list[dict], key: str) -> dict:
    """Per-bin mean/SEM across images + the σ-trend and split-half reliability
    of the bin-mean curve (odd/even image split). nan-aware: debiased gaps are
    NaN where a floor was <= 0."""
    m = np.array([r[key] for r in rows], dtype=np.float64)  # (n_images, bins)
    n_fin = np.isfinite(m).sum(axis=0).clip(min=1)
    mean = np.nanmean(m, axis=0)
    sem = np.nanstd(m, axis=0, ddof=1) / np.sqrt(n_fin)
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


def group_stats(rows: list[dict], key: str) -> dict:
    """Same reduction as :func:`bin_stats`, per parameter group."""
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


def sweep_fit(rows: list[dict], key: str, sweep: list[int], seed: int) -> dict:
    """Least-squares fit y(D) = y_inf + c/D on the per-image-mean curve; 95% CI
    on y_inf from a 2000-resample image bootstrap."""
    x = 1.0 / np.asarray(sweep, dtype=np.float64)
    m = np.array([r[key] for r in rows], dtype=np.float64)

    def fit(y: np.ndarray) -> tuple[float, float]:
        msk = np.isfinite(y)
        if msk.sum() < 2:
            return float("nan"), float("nan")
        a_mat = np.stack([np.ones(int(msk.sum())), x[msk]], axis=1)
        coef, *_ = np.linalg.lstsq(a_mat, y[msk], rcond=None)
        return float(coef[0]), float(coef[1])

    y_inf, c_ = fit(np.nanmean(m, axis=0))
    rng = np.random.default_rng(seed)
    n = m.shape[0]
    boots = [fit(np.nanmean(m[rng.integers(0, n, n)], axis=0))[0] for _ in range(2000)]
    lo_b, hi_b = np.nanpercentile(boots, [2.5, 97.5])
    return {
        "y_inf": round(y_inf, 5),
        "c": round(c_, 5),
        "y_inf_ci95": [round(float(lo_b), 5), round(float(hi_b), 5)],
    }


def build_headline(
    rows: list[dict],
    args,
    *,
    arm_keys: list[str],
    edges: list[int],
    alphas: list[float],
    sweep: list[int] | None,
    centers: list[float],
    wall_time_s: float,
    pool_strata: list[dict] | None = None,
    pool_agg: PoolAccumulator | None = None,
) -> dict:
    """The run's ``metrics`` envelope: per-bin curves for every arm, plus the
    alpha-sweep / kappa / draw-fit / pooled / per-group sections each mode
    contributes. Logs the same per-section summaries the single-file version
    did."""
    headline: dict = {
        "n_images": len(rows),
        "bins": args.bins,
        "draws_per_bin": args.draws_per_bin,
        "sigma_centers": centers,
        "adapter": args.adapter,
        "cos_floor": bin_stats(rows, "cos_floor"),
        "gnorm_native": bin_stats(rows, "gnorm_native"),
        "wall_time_s": round(wall_time_s, 1),
    }
    for k in arm_keys:  # reenc + demote edges + any <edge>pi arms
        headline[f"gap_{k}"] = bin_stats(rows, f"gap_{k}")
        if args.self_floor:
            headline[f"cos_self_{k}"] = bin_stats(rows, f"cos_self_{k}")
            headline[f"debiased_gap_{k}"] = bin_stats(rows, f"debiased_gap_{k}")

    if len(alphas) > 1:
        headline["target_alphas"] = alphas
        for alpha_ in alphas:
            if alpha_ == 1.0:
                continue
            sfx = f"@a{alpha_:g}"
            headline[f"cos_floor{sfx}"] = bin_stats(rows, f"cos_floor{sfx}")
            headline[f"gnorm_native{sfx}"] = bin_stats(rows, f"gnorm_native{sfx}")
            for k in arm_keys:
                headline[f"gap_{k}{sfx}"] = bin_stats(rows, f"gap_{k}{sfx}")
                if args.self_floor:
                    headline[f"cos_self_{k}{sfx}"] = bin_stats(
                        rows, f"cos_self_{k}{sfx}"
                    )
                    headline[f"debiased_gap_{k}{sfx}"] = bin_stats(
                        rows, f"debiased_gap_{k}{sfx}"
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
            headline[key] = bin_stats(rows, key)
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
        headline["drawfit_floor"] = sweep_fit(rows, "cos_floor", sweep, args.seed)
        for k in arm_keys:
            headline[f"drawfit_gap_{k}"] = sweep_fit(rows, f"gap_{k}", sweep, args.seed)
            if args.self_floor:
                headline[f"drawfit_debiased_{k}"] = sweep_fit(
                    rows, f"debiased_gap_{k}", sweep, args.seed
                )
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

    if args.per_group:
        for key in [k for k in rows[0] if k.startswith("gapg_")]:
            headline[key] = group_stats(rows, key)
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

    return headline
