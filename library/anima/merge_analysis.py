"""Weight-space interference analysis for multi-LoRA merges.

When ``scripts/toolkits/merge_loras.py`` fuses N adapters it sums their weight-deltas
``ΔW_i = (α_i/r_i)·up_i @ down_i``.  The energy of that sum is **not** just the
sum of energies — it carries pairwise cross-terms::

    ‖Σ_i ΔW_i‖²_F = Σ_i ‖ΔW_i‖²_F  +  2·Σ_{i<j} ⟨ΔW_i, ΔW_j⟩_F
                                       └──────── interference ────────┘

- ``⟨ΔW_i, ΔW_j⟩ > 0`` → **constructive**: the two LoRAs push the same
  direction in weight space (reinforcing — risk of overdrive into high-freq
  noise once summed).
- ``< 0`` → **destructive**: they partially cancel — one LoRA erodes the
  other's learned effect.
- ``≈ 0`` → orthogonal/independent — the ``--normalize global`` √N-quadrature
  assumption holds.

The merge script derives its normalization scale *assuming* orthogonality;
this module measures the actual interference exactly and cheaply via the
Gram-trace identity — no ``out×in`` ΔW is ever materialized
(mirrors ``merge_loras.fro2``)::

    ⟨ΔW_i, ΔW_j⟩_F = Σ( (up_iᵀ up_j) ⊙ (down_i down_jᵀ) )

The per-LoRA scale ``(α/r)·w`` is folded into the up factor before measuring, so
the energy-weighted aggregate reflects what the merge actually writes (pairwise
*cosine* is scale-invariant, but the per-module weighting of the global cosine
is not).

**Subspace overlap** complements the signed cosine. Two independent low-rank
deltas in a huge matrix space have vanishing expected Frobenius cosine
(measured on independent artist LoRAs), so the cosine reads
"orthogonal" for essentially every real merge — including ones that visibly
clash. The precursor to a visible clash is the LoRAs *occupying the same
functional subspace*: both writing into the same output directions (column
space of ``up``) and/or reading the same input features (row space of
``down``), with ⟨ΔW_i, ΔW_j⟩_F ≈ 0 all the while. We measure it as::

    overlap(U, V) = ‖QᵤᵀQᵥ‖²_F / min(rᵤ, rᵥ)   ∈ [0, 1]

(mean squared cosine of the principal angles; Q = orthonormal basis of the
*effective* — rank-truncated — subspace, so T-LoRA-masked dead ranks don't
dilute it). Raw overlap is scale-trapped: random rank-r subspaces in d dims
land at ≈ r/d, which varies per module, so each pair is also reported as a
**×random factor** (observed / chance). ~1× is coincidence-level;
``OVERLAP_ELEVATED_X`` / ``OVERLAP_COLLIDING_X`` band the magnitude.

Overlap magnitude is **sign-blind**, though, so the band is finished off with
the pair's signed cosine (``COS_ALIGNED``): a high overlap only reads
"colliding" when the directions are ≈ orthogonal (cos ≈ 0) — the blind spot the
metric exists for. A high overlap with cos ≫ 0 is **reinforcing** (both LoRAs
push the same way — common when the inputs share an init basin, e.g. soup
outputs from one uncond init), and with cos ≪ 0 is **cancelling**.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import torch

# Loaded-LoRA tuple as returned by ``scripts/toolkits/merge_loras.load_lora``:
# (downs, ups, alphas) keyed by module stem.
LoadedLoRA = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float]]

_EPS = 1e-12

# ×random severity bands for subspace overlap (observed / chance-level).
# Chance itself is ≈ rank/dim per module (~0.01 for r=32 in 3072-d), so raw
# overlap thresholds would not transfer across module shapes — the ×random
# factor does.
OVERLAP_ELEVATED_X = 3.0
OVERLAP_COLLIDING_X = 8.0

# Above this |global energy-weighted cosine| a high subspace overlap is
# *explained by the sign* of the interference — the pair reinforces (cos ≫ 0)
# or cancels (cos ≪ 0) rather than genuinely colliding. Only a same-subspace /
# orthogonal-direction pair (cos ≈ 0) is a real "collision" the cosine can't
# see. 0.5 sits well above the near-orthogonal regime independent adapters
# occupy (≈0–0.15 on artist LoRAs) yet below the blind-spot
# construction's <0.5, so both regimes classify correctly. (The GUI banner uses
# a separate, coarser _SAFE_COS=0.15 gate for the same reinforce-vs-collide
# decision — see gui/tabs/merge_tab.py::_apply_analysis_marker.)
COS_ALIGNED = 0.5


def _scaled_up(up: torch.Tensor, down: torch.Tensor, alpha: float | None, w: float):
    """``up`` with ``(α/r)·w`` folded in — matches ``merge_loras`` line ``ups_cat``."""
    r = down.shape[0]
    s = (alpha if alpha is not None else float(r)) / r
    return up * (s * w)


def _gram_inner(up_i, down_i, up_j, down_j) -> float:
    """⟨up_i@down_i, up_j@down_j⟩_F via the Gram-trace identity (no out×in)."""
    return ((up_i.T @ up_j) * (down_i @ down_j.T)).sum().item()


def _column_basis(mat: torch.Tensor, rtol: float = 1e-4) -> torch.Tensor:
    """Orthonormal basis ``(d, r_eff)`` of ``mat``'s column space.

    SVD with a relative cutoff rather than QR: T-LoRA masking zeroes whole
    rank columns, and QR would replace those with arbitrary noise directions,
    inflating the subspace dimension.
    """
    if mat.numel() == 0:
        return mat.new_zeros((mat.shape[0], 0))
    u, s, _ = torch.linalg.svd(mat, full_matrices=False)
    if s.numel() == 0 or s[0] <= 0:
        return mat.new_zeros((mat.shape[0], 0))
    return u[:, s > s[0] * rtol]


def _pair_overlap(b_i: torch.Tensor, b_j: torch.Tensor) -> tuple[float, float]:
    """(observed, chance) subspace overlap between two orthonormal bases.

    Observed = ‖b_iᵀ b_j‖²_F / min(r_i, r_j) ∈ [0, 1]; chance is the expected
    value for independent random subspaces of the same ranks in the same
    ambient dim (r_i·r_j / d / min = max(r_i, r_j)/d).
    """
    r_i, r_j = b_i.shape[1], b_j.shape[1]
    m = min(r_i, r_j)
    obs = (b_i.T @ b_j).pow(2).sum().item() / m
    return obs, r_i * r_j / (b_i.shape[0] * m)


@dataclass(frozen=True)
class SubspaceOverlap:
    """Energy-weighted subspace overlap for one input pair, out + in sides."""

    out: float  # output (up column-space) overlap ∈ [0, 1] — write collision
    out_xrandom: float  # out / chance-level; ~1 = coincidence, ≫1 = collision
    inp: float  # input (down row-space) overlap ∈ [0, 1] — read overlap
    in_xrandom: float
    # The pair's global energy-weighted Frobenius cosine, so ``band`` can tell a
    # sign-explained overlap (reinforcing / cancelling) apart from a genuine
    # same-subspace / orthogonal-direction collision. Defaults to 0 (⇒ sign
    # can't rescue a high overlap) for constructions that omit it.
    cosine: float = 0.0

    @property
    def band(self) -> str:
        """'chance' | 'reinforcing' | 'cancelling' | 'elevated' | 'colliding'.

        Overlap magnitude alone is sign-blind: two LoRAs can share a subspace
        while reinforcing (cos ≫ 0), cancelling (cos ≪ 0), or genuinely
        colliding (cos ≈ 0 — the blind spot this metric exists to catch). A
        strongly-aligned pair is NOT a collision even at 200× random overlap, so
        the sign of the interference gates the high-overlap tier. Graded on the
        output side."""
        if self.out_xrandom < OVERLAP_ELEVATED_X:
            return "chance"
        if self.cosine >= COS_ALIGNED:
            return "reinforcing"
        if self.cosine <= -COS_ALIGNED:
            return "cancelling"
        # Shared subspace, direction ≈ orthogonal: the collision cosine misses.
        return "colliding" if self.out_xrandom >= OVERLAP_COLLIDING_X else "elevated"


# How alarming each overlap band is — drives ``worst_overlap`` so the summary /
# GUI banner surface a real problem over a benign-but-huge reinforcing overlap.
_BAND_CONCERN = {
    "chance": 0,
    "reinforcing": 0,
    "elevated": 1,
    "cancelling": 2,
    "colliding": 3,
}


@dataclass
class InterferenceReport:
    """Pairwise + aggregate weight-space interference for a multi-LoRA merge."""

    names: list[str]
    weights: list[float]
    n_inputs: int
    n_modules: int  # union of module stems
    n_shared_modules: int  # stems present in ≥2 inputs (interference only defined here)
    # Global energy-weighted cosine per input pair, keyed by (i, j) with i < j.
    pair_cosine: dict[tuple[int, int], float]
    # ‖Σ ΔW‖² / Σ‖ΔW_i‖² over the whole weight space. 1 = orthogonal,
    # >1 = net constructive, <1 = net destructive.
    overall_energy_ratio: float
    # Per-module energy-excess index (2·Σ_{i<j}⟨⟩ / Σ‖·‖²): <0 destructive layer.
    module_index: dict[str, float] = field(default_factory=dict)
    # Energy-weighted subspace overlap per input pair — catches "same functional
    # subspace, orthogonal directions" collisions the signed cosine cannot see.
    pair_overlap: dict[tuple[int, int], SubspaceOverlap] = field(default_factory=dict)
    # Per-module worst-pair output-overlap ×random factor: which layers collide.
    module_overlap: dict[str, float] = field(default_factory=dict)

    @property
    def worst_pair(self) -> tuple[tuple[int, int], float] | None:
        """The input pair with the most negative (destructive) cosine, if any."""
        if not self.pair_cosine:
            return None
        return min(self.pair_cosine.items(), key=lambda kv: kv[1])

    @property
    def strongest_pair(self) -> tuple[tuple[int, int], float] | None:
        """The input pair with the largest |cosine| — the dominant interference,
        whether reinforcing (+) or cancelling (−). Grades the GUI banner
        severity (the energy ratio inflates with N)."""
        if not self.pair_cosine:
            return None
        return max(self.pair_cosine.items(), key=lambda kv: abs(kv[1]))

    @property
    def worst_overlap(self) -> tuple[tuple[int, int], SubspaceOverlap] | None:
        """The most *concerning* overlap pair, if any. A genuine collision /
        cancellation outranks a merely-large but sign-explained (reinforcing)
        overlap — a 200× reinforcing overlap is benign, a 20× cos≈0 one is not;
        ties break on the raw ×random factor."""
        if not self.pair_overlap:
            return None
        return max(
            self.pair_overlap.items(),
            key=lambda kv: (_BAND_CONCERN[kv[1].band], kv[1].out_xrandom),
        )

    def _verdict(self, ratio: float) -> str:
        if ratio > 1.05:
            return "constructive"
        if ratio < 0.95:
            return "destructive"
        return "orthogonal"

    @property
    def verdict(self) -> str:
        """'orthogonal' | 'constructive' | 'destructive' for the aggregate ratio."""
        return self._verdict(self.overall_energy_ratio)

    def marker_payload(self) -> dict:
        """Machine-readable summary for the GUI banner (one compact JSON line)."""
        payload: dict = {
            "verdict": self.verdict,
            "ratio": round(self.overall_energy_ratio, 4),
            "shared": self.n_shared_modules,
            "modules": self.n_modules,
        }
        # The dominant pair (largest |cos|) grades the banner severity; the
        # most-negative pair stays the CLI report's "most opposed".
        strongest = self.strongest_pair
        if strongest is not None:
            (i, j), c = strongest
            payload["strongest"] = [self.names[i], self.names[j], round(c, 4)]
        # Worst subspace collision — lets the GUI escalate a near-orthogonal
        # cosine (the common case) when the pair still shares weight subspaces.
        worst_ov = self.worst_overlap
        if worst_ov is not None:
            (i, j), ov = worst_ov
            payload["overlap"] = [
                self.names[i],
                self.names[j],
                round(ov.out, 4),
                round(ov.out_xrandom, 1),
            ]
        return payload

    def summary_line(self) -> str:
        v = self._verdict(self.overall_energy_ratio)
        worst = self.worst_pair
        tail = ""
        if worst is not None:
            (i, j), c = worst
            tail = f" — most opposed: {self.names[i]}↔{self.names[j]} cos={c:+.3f}"
        worst_ov = self.worst_overlap
        if worst_ov is not None:
            (i, j), ov = worst_ov
            tail += (
                f" — worst subspace overlap: {self.names[i]}↔{self.names[j]} "
                f"{ov.out_xrandom:.1f}× random ({ov.band})"
            )
        return (
            f"interference: energy ratio {self.overall_energy_ratio:.3f} ({v}), "
            f"{self.n_shared_modules}/{self.n_modules} shared modules{tail}"
        )


def analyze(
    loaded: list[LoadedLoRA],
    names: list[str],
    weights: list[float] | None = None,
) -> InterferenceReport:
    """Compute weight-space interference across ``loaded`` LoRAs.

    ``loaded`` / ``names`` / ``weights`` are parallel lists (one entry per input
    adapter); ``loaded[k]`` is the ``(downs, ups, alphas)`` tuple from
    ``merge_loras.load_lora``. ``weights`` defaults to all-1.0.
    """
    n = len(loaded)
    if n < 2:
        raise ValueError("interference analysis needs at least 2 LoRAs")
    if weights is None:
        weights = [1.0] * n

    # Pre-scale every module's up factor by (α/r)·w once.
    scaled: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
    for (downs, ups, alphas), w in zip(loaded, weights):
        per: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for stem, down in downs.items():
            if stem not in ups:
                continue
            per[stem] = (_scaled_up(ups[stem], down, alphas.get(stem), w), down)
        scaled.append(per)

    stems = sorted({s for per in scaled for s in per})

    # Global accumulators: per-input energy N_i, per-pair inner product S_ij,
    # and per-pair subspace-overlap sums (energy weight, out/in observed+chance).
    norm_sq = [0.0] * n
    inner = {pair: 0.0 for pair in combinations(range(n), 2)}
    ov_acc = {pair: [0.0] * 5 for pair in combinations(range(n), 2)}
    module_index: dict[str, float] = {}
    module_overlap: dict[str, float] = {}
    n_shared = 0

    for stem in stems:
        present = [k for k in range(n) if stem in scaled[k]]
        # Per-module energy bookkeeping for the local interference index.
        m_norm = {}
        for k in present:
            up_k, down_k = scaled[k][stem]
            nrm = _gram_inner(up_k, down_k, up_k, down_k)
            m_norm[k] = nrm
            norm_sq[k] += nrm
        # Effective-subspace bases, computed once per input at this module.
        bases = {}
        if len(present) >= 2:
            for k in present:
                up_k, down_k = scaled[k][stem]
                bases[k] = (_column_basis(up_k), _column_basis(down_k.T))
        m_cross = 0.0
        for i, j in combinations(present, 2):
            up_i, down_i = scaled[i][stem]
            up_j, down_j = scaled[j][stem]
            s = _gram_inner(up_i, down_i, up_j, down_j)
            inner[(i, j)] += s
            m_cross += s
            bo_i, bi_i = bases[i]
            bo_j, bi_j = bases[j]
            # Zero effective rank (zero --weights entry, all-masked T-LoRA):
            # no subspace to compare — skip; the wgt would be ~0 anyway.
            if min(bo_i.shape[1], bo_j.shape[1], bi_i.shape[1], bi_j.shape[1]) == 0:
                continue
            obs_out, exp_out = _pair_overlap(bo_i, bo_j)
            obs_in, exp_in = _pair_overlap(bi_i, bi_j)
            wgt = (m_norm[i] * m_norm[j]) ** 0.5
            acc = ov_acc[(i, j)]
            acc[0] += wgt
            acc[1] += wgt * obs_out
            acc[2] += wgt * exp_out
            acc[3] += wgt * obs_in
            acc[4] += wgt * exp_in
            x_out = obs_out / (exp_out + _EPS)
            if x_out > module_overlap.get(stem, 0.0):
                module_overlap[stem] = x_out
        if len(present) >= 2:
            n_shared += 1
            denom = sum(m_norm.values()) + _EPS
            module_index[stem] = 2.0 * m_cross / denom

    pair_cosine = {}
    for (i, j), s in inner.items():
        denom = (norm_sq[i] * norm_sq[j]) ** 0.5 + _EPS
        pair_cosine[(i, j)] = s / denom

    pair_overlap = {}
    for pair, (wsum, o_out, e_out, o_in, e_in) in ov_acc.items():
        if wsum <= 0.0:
            continue
        pair_overlap[pair] = SubspaceOverlap(
            out=o_out / wsum,
            out_xrandom=o_out / (e_out + _EPS),
            inp=o_in / wsum,
            in_xrandom=o_in / (e_in + _EPS),
            cosine=pair_cosine.get(pair, 0.0),
        )

    total_norm = sum(norm_sq)
    total_cross = sum(inner.values())
    overall_energy_ratio = (total_norm + 2.0 * total_cross) / (total_norm + _EPS)

    return InterferenceReport(
        names=list(names),
        weights=list(weights),
        n_inputs=n,
        n_modules=len(stems),
        n_shared_modules=n_shared,
        pair_cosine=pair_cosine,
        overall_energy_ratio=overall_energy_ratio,
        module_index=module_index,
        pair_overlap=pair_overlap,
        module_overlap=module_overlap,
    )


def format_report(report: InterferenceReport, *, top_modules: int = 8) -> str:
    """Human-readable multi-line interference report for CLI / GUI log output."""
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("LoRA merge — weight-space interference analysis")
    lines.append("=" * 64)
    for k, (nm, w) in enumerate(zip(report.names, report.weights)):
        lines.append(f"  [{k}] {nm}  (weight {w:g})")
    lines.append("")
    lines.append(report.summary_line())
    lines.append(
        "  energy ratio = ‖Σ ΔW‖² / Σ‖ΔW_i‖²  "
        "(1.0 = orthogonal, >1 reinforce, <1 cancel)"
    )
    lines.append("")

    lines.append("Pairwise interference (global energy-weighted cosine):")
    for (i, j), c in sorted(report.pair_cosine.items()):
        if c > 0.05:
            tag = "constructive"
        elif c < -0.05:
            tag = "destructive"
        else:
            tag = "orthogonal"
        lines.append(f"  {report.names[i]} ↔ {report.names[j]}: {c:+.3f}  ({tag})")
    lines.append("")

    if report.pair_overlap:
        lines.append(
            "Subspace overlap (do the LoRAs occupy the same weight subspaces?):"
        )
        lines.append(
            "  chance ≈ rank/dim; severity grades on ×random AND interference sign —"
        )
        lines.append(
            "  a high overlap only 'collides' at cos≈0 (shared subspace, orthogonal"
        )
        lines.append(
            "  directions); an aligned overlap reinforces, an opposed one cancels."
        )
        for (i, j), ov in sorted(report.pair_overlap.items()):
            lines.append(
                f"  {report.names[i]} ↔ {report.names[j]}: "
                f"out {ov.out:.3f} ({ov.out_xrandom:.1f}× random) · "
                f"in {ov.inp:.3f} ({ov.in_xrandom:.1f}×)  ({ov.band})"
            )
        # Sign-aware: a module whose shared subspace is sign-explained (its
        # energy-excess index is strongly positive ⇒ reinforcing) is not a
        # collision, so exclude it — otherwise every reinforcing merge lists its
        # highest-energy layers as "colliding".
        colliding = sorted(
            (
                kv
                for kv in report.module_overlap.items()
                if kv[1] >= OVERLAP_ELEVATED_X
                and report.module_index.get(kv[0], 0.0) < COS_ALIGNED
            ),
            key=lambda kv: -kv[1],
        )[:top_modules]
        if colliding:
            lines.append(
                "  most colliding modules (shared subspace, not sign-aligned, ×random):"
            )
            for stem, x in colliding:
                lines.append(f"    {x:6.1f}×  {stem}")
        lines.append("")

    if report.module_index:
        worst = sorted(report.module_index.items(), key=lambda kv: kv[1])[:top_modules]
        lines.append(f"Most destructive modules (of {report.n_shared_modules} shared):")
        any_neg = False
        for stem, idx in worst:
            if idx < 0:
                any_neg = True
                lines.append(f"  {idx:+.3f}  {stem}")
        if not any_neg:
            lines.append("  (none — no module has net cancellation)")
    lines.append("=" * 64)
    return "\n".join(lines)
