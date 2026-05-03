"""Per-LoRA DCW calibration selector — LL-only, multi-anchor.

Given the late-half LL gap from a baseline reverse rollout (λ=0) and N
anchor rollouts (DCW with λ=anchor_i on band LL), pick the anchor that
**actually measured** the lowest weighted late-half ``Σ w(σ_i) · gap(i)²``.
Baseline (λ=0) is always treated as a candidate so flat-style LoRAs that
DCW hurts can land on "no correction".

This replaces an earlier closed-form linear-response solver
(``archive/dcw/measure_bias.py::compute_optimal_lambda``) which
extrapolated from a single ε-probe under
``gap_corrected(i) ≈ gap_baseline(i) + λ·s(i)``. LL is a strongly
nonlinear lever (per ``docs/methods/dcw.md`` §"Re-tuning λ for LL-only");
the extrapolation systematically over-shot — it picked λ ≈ −0.033 on
bench (vs the empirical safe optimum at −0.015) and could blow up to
λ ≈ −0.1 on LoRAs with weaker probe response, producing visibly drifting
samples. Picking from a small set of measured anchors avoids the
extrapolation entirely.

The result is one number plus a fixed schedule string, written to the
LoRA's safetensors metadata as ``ss_dcw_recipe``. See
``docs/proposal/lora-dcw-proposal.md`` for the original design and
``docs/methods/dcw.md`` §LL-only correction for the empirical motivation.

This module is dependency-free — no torch tensors come in, only Python
floats / lists. The trainer reduces ``gap_LL`` from the per-batch
trajectory dicts before calling here.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Iterable, Literal, Sequence

logger = logging.getLogger(__name__)

Schedule = Literal["one_minus_sigma", "sigma_i", "const", "none"]

# Global fallbacks when neither a CLI override nor a per-LoRA recipe is
# available. These match the post-magnitude-sweep default in
# docs/methods/dcw.md §Re-tuning λ for LL-only.
DEFAULT_LAMBDA_LL = -0.015
DEFAULT_SCHEDULE_LL = "one_minus_sigma"
DEFAULT_BAND_MASK = "LL"


def _w(sigma_i: float, schedule: Schedule) -> float:
    if schedule == "one_minus_sigma":
        return 1.0 - sigma_i
    if schedule == "sigma_i":
        return sigma_i
    if schedule == "const":
        return 1.0
    return 0.0


def _weighted_late_loss(
    gap: Sequence[float], sigmas: Sequence[float], schedule: Schedule, late_start: int
) -> float:
    """``Σ_{i ≥ late_start} w(σ_i) · gap(i)²`` — the loss the selector
    minimizes across candidates."""
    return sum(
        _w(sigmas[i], schedule) * gap[i] * gap[i] for i in range(late_start, len(gap))
    )


def select_best_anchor_LL(
    gap_LL_baseline: Iterable[float],
    anchor_results: Sequence[tuple[float, Iterable[float]]],
    sigmas: Iterable[float],
    *,
    schedule: Schedule = "one_minus_sigma",
    late_fraction: float = 0.5,
) -> dict:
    """Pick the λ_LL anchor with the lowest measured late-half weighted
    ``Σ w(σ_i) · gap_LL(i)²``. Baseline (λ=0) is always a candidate.

    Args:
        gap_LL_baseline: per-step LL gap at λ=0 (no correction), length
            ``num_steps``. Defines the ``λ=0`` candidate.
        anchor_results: list of ``(lambda_value, gap_LL_at_lambda)``
            pairs, one per probe rollout. Each ``gap_LL_at_lambda`` must
            have the same length as ``gap_LL_baseline``.
        sigmas: σ_i schedule used during measurement, same length.
        schedule: σ-shape that weights the late-half loss. Must match the
            schedule the recipe will be applied with at inference.
        late_fraction: tail fraction of the schedule that the loss is
            integrated over. Default 0.5 = late half, per the 2026-05-03
            finding that bias is concentrated at low σ.

    Returns:
        dict with keys:
            ``lambda_LL`` — winning anchor (may be 0.0 if baseline won).
            ``schedule_LL`` — passed through.
            ``baseline_gap_LL_late`` — λ=0 weighted loss.
            ``best_gap_LL_late`` — winning anchor's weighted loss.
            ``num_late_steps`` — number of steps in the late-half sum.
            ``candidates`` — list of ``{"lambda": float, "loss": float}``
                in input order with baseline first; for logging.
            ``baseline_was_best`` — bool.

    Raises:
        ValueError: on length mismatches or empty inputs.
    """
    gap_b = list(map(float, gap_LL_baseline))
    sig = list(map(float, sigmas))
    n = len(gap_b)
    if n == 0:
        raise ValueError("empty inputs")
    if len(sig) != n:
        raise ValueError(f"sigmas length {len(sig)} != gap_baseline length {n}")

    late_start = int(n * (1.0 - late_fraction))

    candidates: list[dict] = [
        {
            "lambda": 0.0,
            "loss": _weighted_late_loss(gap_b, sig, schedule, late_start),
        }
    ]
    for lam, gap_at_lam in anchor_results:
        gap_a = list(map(float, gap_at_lam))
        if len(gap_a) != n:
            raise ValueError(
                f"anchor λ={lam} gap length {len(gap_a)} != baseline length {n}"
            )
        candidates.append(
            {
                "lambda": float(lam),
                "loss": _weighted_late_loss(gap_a, sig, schedule, late_start),
            }
        )

    best = min(candidates, key=lambda c: c["loss"])

    return {
        "lambda_LL": float(best["lambda"]),
        "schedule_LL": schedule,
        "baseline_gap_LL_late": float(candidates[0]["loss"]),
        "best_gap_LL_late": float(best["loss"]),
        "num_late_steps": int(n - late_start),
        "candidates": candidates,
        "baseline_was_best": best is candidates[0],
    }


def read_recipe_from_safetensors(path: str) -> dict | None:
    """Read ss_dcw_recipe metadata from a safetensors file.

    Returns the parsed recipe dict (``{"lambda_LL": float, "schedule_LL":
    str}``) or ``None`` when the file has no recipe key, the metadata is
    unreadable, or the JSON is malformed. Header-only read via safe_open
    — no tensors materialized.
    """
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
    except Exception as exc:
        logger.warning("dcw-recipe: failed to read metadata from %s: %s", path, exc)
        return None

    raw = meta.get("ss_dcw_recipe")
    if not raw:
        return None
    try:
        recipe = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "dcw-recipe: malformed ss_dcw_recipe in %s (%s) — ignoring", path, exc
        )
        return None

    if not isinstance(recipe, dict) or "lambda_LL" not in recipe:
        logger.warning(
            "dcw-recipe: ss_dcw_recipe in %s missing 'lambda_LL' key — ignoring", path
        )
        return None
    recipe.setdefault("schedule_LL", DEFAULT_SCHEDULE_LL)
    try:
        recipe["lambda_LL"] = float(recipe["lambda_LL"])
    except (TypeError, ValueError):
        logger.warning(
            "dcw-recipe: lambda_LL in %s is not numeric — ignoring", path
        )
        return None
    return recipe


def _normalize_multipliers(
    multipliers: float | Sequence[float] | None, n: int
) -> list[float]:
    if multipliers is None:
        return [1.0] * n
    if isinstance(multipliers, (int, float)):
        return [float(multipliers)] * n
    out = [float(m) for m in multipliers]
    if len(out) == 1:
        return out * n
    if len(out) != n:
        raise ValueError(
            f"--lora_multiplier length {len(out)} does not match "
            f"--lora_weight length {n}"
        )
    return out


def combine_recipes(
    recipes: Sequence[dict | None],
    multipliers: float | Sequence[float] | None,
) -> dict | None:
    """Reduce per-LoRA recipes into a single inference-time recipe.

    Single LoRA: pass-through. Multiple LoRAs with recipes: λ_LL is the
    multiplier-weighted average across LoRAs that carry a recipe (LoRAs
    without one contribute nothing — neither weight nor λ). Schedule:
    modal across present recipes (ties resolved by position).

    Returns ``None`` when no input recipe is present. The averaging rule
    is explicitly heuristic (per proposal §"Stacked LoRAs") — callers
    should emit a stderr warning when ``len(present) > 1`` so the user
    knows they're in extrapolation territory.
    """
    n = len(recipes)
    mults = _normalize_multipliers(multipliers, n)
    present_lambdas: list[float] = []
    present_weights: list[float] = []
    present_schedules: list[str] = []
    for rec, m in zip(recipes, mults):
        if rec is None:
            continue
        present_lambdas.append(float(rec["lambda_LL"]))
        present_weights.append(float(m))
        present_schedules.append(str(rec.get("schedule_LL", DEFAULT_SCHEDULE_LL)))

    if not present_lambdas:
        return None

    total_w = sum(abs(w) for w in present_weights)
    if total_w == 0.0:
        # All LoRAs disabled — recipe is meaningless; fall back to global.
        return None
    lam = sum(lm * w for lm, w in zip(present_lambdas, present_weights)) / total_w
    schedule = Counter(present_schedules).most_common(1)[0][0]
    return {
        "lambda_LL": float(lam),
        "schedule_LL": schedule,
        "_n_present": len(present_lambdas),
        "_n_total": n,
    }


def resolve_dcw_args(args: Any) -> None:
    """In-place resolve ``args.dcw_lambda`` / ``dcw_schedule`` /
    ``dcw_band_mask`` against per-LoRA recipes.

    Resolution order (CLI explicit always wins, sentinel ``None`` on each
    arg means "not explicitly set"):
        1. CLI value (``args.dcw_lambda is not None`` etc.)
        2. ``--dcw_disable_per_lora_recipe`` → skip recipe, use defaults
        3. Combined recipe across loaded LoRAs (read from safetensors)
        4. Global default (``DEFAULT_*`` constants)

    Always populates the three args attributes (with global defaults if
    necessary) so downstream code can rely on them being non-None. Skips
    the safetensors I/O when ``args.dcw`` is False since the resolved
    values won't be consumed.
    """
    cli_lambda = getattr(args, "dcw_lambda", None)
    cli_schedule = getattr(args, "dcw_schedule", None)
    cli_band_mask = getattr(args, "dcw_band_mask", None)

    recipe: dict | None = None
    lora_paths = getattr(args, "lora_weight", None) or []
    disable = bool(getattr(args, "dcw_disable_per_lora_recipe", False))
    dcw_on = bool(getattr(args, "dcw", False))
    if dcw_on and lora_paths and not disable:
        recipes = [read_recipe_from_safetensors(p) for p in lora_paths]
        recipe = combine_recipes(recipes, getattr(args, "lora_multiplier", None))
        if recipe is not None and recipe.get("_n_total", 1) > 1:
            n_pres = recipe.get("_n_present", 0)
            n_tot = recipe.get("_n_total", 0)
            logger.warning(
                "dcw-recipe: combining %d/%d per-LoRA recipes via "
                "multiplier-weighted average — heuristic, not theory; "
                "pass --dcw_disable_per_lora_recipe or explicit "
                "--dcw_lambda to override.",
                n_pres,
                n_tot,
            )

    if cli_lambda is None:
        args.dcw_lambda = (
            float(recipe["lambda_LL"]) if recipe is not None else DEFAULT_LAMBDA_LL
        )
    if cli_schedule is None:
        args.dcw_schedule = (
            str(recipe["schedule_LL"]) if recipe is not None else DEFAULT_SCHEDULE_LL
        )
    if cli_band_mask is None:
        # Per-LoRA recipe is single-band (LL); if the user didn't set the
        # mask, force LL whether or not a recipe was found (the global
        # default is also LL post 2026-05-03 finding).
        args.dcw_band_mask = DEFAULT_BAND_MASK

    if recipe is not None and cli_lambda is None:
        logger.info(
            "dcw-recipe: applied λ_LL=%+.5f (%s) from per-LoRA metadata "
            "(%d/%d LoRAs carried a recipe)",
            args.dcw_lambda,
            args.dcw_schedule,
            recipe.get("_n_present", 0),
            recipe.get("_n_total", 1),
        )
