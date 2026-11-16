"""sigma_lowres σ-demote routing (``--sigma_lowres*``).

Parsers and gates are pure functions of ``args``; :class:`SigmaDemoter` holds
the per-run counters and swaps a train step onto its demote-tier sibling
latent. Skill: ``sigma-lowres``.
"""

from __future__ import annotations

import logging
import math
import random

from library.training import TrainCtx

logger = logging.getLogger(__name__)

# The probe's operating point, and the value `--sigma_lowres` implies.
YARNSIG_OPERATING_POINT = "1,4,0.35,2"
_YARNSIG_OFF = frozenset({"", "0", "off", "no", "none", "false"})


def route(args):
    """Parsed ``--sigma_lowres_route`` as ``(native_edge, demote_edge)``.

    Defaults to the measured-safe 1024:896 (``SIGMA_DEMOTE_ROUTE``).
    """
    raw = str(getattr(args, "sigma_lowres_route", None) or "1024:896")
    try:
        native, demote = (int(x) for x in raw.split(":"))
    except ValueError:
        raise ValueError(
            f"--sigma_lowres_route expects NATIVE:DEMOTE (e.g. 1024:896), got {raw!r}"
        ) from None
    if not (native > demote > 0):
        raise ValueError(f"--sigma_lowres_route needs NATIVE > DEMOTE > 0, got {raw!r}")
    return native, demote


def yarnsig_params(args):
    """Parsed ``--sigma_lowres_yarnsig`` as ``(alpha, beta, center, gamma)``,
    or None when off; cached on the args namespace. **Default-on with
    ``--sigma_lowres``** (docs/anima/optimizations/sigma_lowres.md): an unset
    flag resolves to the operating point, written back onto ``args`` so the
    snapshot records what ran. Opt out with ``--sigma_lowres_yarnsig off``.
    """
    raw = getattr(args, "sigma_lowres_yarnsig", None)
    if raw is None and getattr(args, "sigma_lowres", False):
        raw = YARNSIG_OPERATING_POINT
        args.sigma_lowres_yarnsig = raw
    if raw is None or str(raw).strip().lower() in _YARNSIG_OFF:
        return None
    cached = getattr(args, "_yarnsig_parsed", None)
    if cached is not None:
        return cached
    try:
        alpha, beta, center, gamma = (float(v) for v in raw.split(","))
    except ValueError:
        raise ValueError(
            f"--sigma_lowres_yarnsig expects ALPHA,BETA,CENTER,GAMMA, got {raw!r}"
        )
    if not (0.0 <= alpha < beta and 0.0 < center < 1.0 and gamma > 0.0):
        raise ValueError(
            "--sigma_lowres_yarnsig needs 0<=ALPHA<BETA, 0<CENTER<1, "
            f"GAMMA>0, got {raw!r}"
        )
    args._yarnsig_parsed = (alpha, beta, center, gamma)
    return args._yarnsig_parsed


def _rule_suffix(rule):
    if rule not in (1, 2):
        raise ValueError(f"sigma_lowres rule must be 1 or 2, got {rule!r}")
    return "" if rule == 1 else "2"


def rule_cfg(args, rule):
    """``(threshold, threshold_max, span_spec)`` for one rule.

    Written as literal ``getattr(args, "…")`` calls, one branch per rule:
    the drift guard (``tests/config_closure.py``) AST-scans for literal
    names only.
    """
    if rule == 1:
        return (
            getattr(args, "sigma_lowres_threshold", 0.5),
            getattr(args, "sigma_lowres_threshold_max", None),
            getattr(args, "sigma_lowres_span", None),
        )
    if rule == 2:
        return (
            getattr(args, "sigma_lowres_threshold2", None),
            getattr(args, "sigma_lowres_threshold2_max", None),
            getattr(args, "sigma_lowres_span2", None),
        )
    raise ValueError(f"sigma_lowres rule must be 1 or 2, got {rule!r}")


def route2(args):
    """Parsed ``--sigma_lowres_route2`` as ``(native, demote)``, or None
    when the secondary rule is off. Same shape/validation as :func:`route`
    (which always has its 1024:896 default)."""
    raw = getattr(args, "sigma_lowres_route2", None)
    if not raw:
        return None
    try:
        native, demote = (int(x) for x in str(raw).split(":"))
    except ValueError:
        raise ValueError(
            f"--sigma_lowres_route2 expects NATIVE:DEMOTE (e.g. 1024:768), got {raw!r}"
        ) from None
    if not (native > demote > 0):
        raise ValueError(
            f"--sigma_lowres_route2 needs NATIVE > DEMOTE > 0, got {raw!r}"
        )
    return native, demote


def span_params(args, rule=1):
    """Parsed ``--sigma_lowres_span[2]`` as ``(mode, frac)``, or None when
    off. ``mode`` ∈ early|late|spread, 0 < frac < 1 (default 0.5).
    Validates once; cached on the args namespace per rule."""
    sfx = _rule_suffix(rule)
    raw = rule_cfg(args, rule)[2]
    if not raw:
        return None
    cache_attr = f"_sigma_span{sfx}_parsed"
    cached = getattr(args, cache_attr, None)
    if cached is not None:
        return cached
    parts = str(raw).split(":")
    mode, frac = parts[0], 0.5
    ok = mode in ("early", "late", "spread") and len(parts) <= 2
    if ok and len(parts) == 2:
        try:
            frac = float(parts[1])
        except ValueError:
            ok = False
    if not ok or not (0.0 < frac < 1.0):
        raise ValueError(
            f"--sigma_lowres_span{sfx} expects early|late|spread[:FRAC] "
            f"with 0<FRAC<1, got {raw!r}"
        )
    setattr(args, cache_attr, (mode, frac))
    return getattr(args, cache_attr)


def gate_allows(args, sigmas_flat, rule=1):
    """σ gate: every σ in the batch above --sigma_lowres_threshold[2]
    and, when --sigma_lowres_threshold[2]_max is set, below it too — the
    measured demote-safe region is a per-route σ WINDOW, not a
    half-line."""
    threshold, threshold_max, _ = rule_cfg(args, rule)
    # Rule 2's bounds have no defaults — validate_rules() refuses a
    # route2 without both, so an unset bound here can only be rule 1's.
    threshold = 0.5 if threshold is None else float(threshold)
    if not bool((sigmas_flat > threshold).all()):
        return False
    if threshold_max is not None:
        return bool((sigmas_flat < float(threshold_max)).all())
    return True


def span_allows(args, step_idx, rule=1):
    """Step-span gate: may step ``step_idx`` (1-based train-forward index)
    demote? True when no span is set.

    early/late split total train forwards (max_train_steps × grad-accum)
    at the FRAC point; spread is a per-step coin seeded from
    (--seed, step_idx) alone — deterministic and touches neither the
    global nor the paired RNG streams.
    """
    span = span_params(args, rule)
    if span is None:
        return True
    mode, frac = span
    total = int(getattr(args, "max_train_steps", 0) or 0) * int(
        getattr(args, "gradient_accumulation_steps", 1) or 1
    )
    if total <= 0:
        raise ValueError("--sigma_lowres_span needs a finalized max_train_steps")
    if mode == "early":
        return step_idx <= round(frac * total)
    if mode == "late":
        # Boundary at (1-frac)·T, same rounding as early's — so
        # early:f and late:1-f partition [1, T] exactly for any T.
        return step_idx > round((1.0 - frac) * total)
    seed = int(getattr(args, "seed", 0) or 0)
    return random.Random(seed * 1_000_003 + step_idx).random() < frac


def validate_rules(args):
    """Validate the whole sigma_lowres surface at setup, not on the first
    train forward. Rejects ``--sigma_lowres_route2`` without both σ bounds:
    rule 2 has priority, so an unbounded gate would shadow the primary rule.
    """
    r1 = route(args)
    r2 = route2(args)
    if r2 is not None:
        lo, hi, _ = rule_cfg(args, 2)
        missing = [
            flag
            for flag, val in (
                ("--sigma_lowres_threshold2", lo),
                ("--sigma_lowres_threshold2_max", hi),
            )
            if val is None
        ]
        if missing:
            raise ValueError(
                f"--sigma_lowres_route2 {r2[0]}:{r2[1]} needs an "
                f"explicit σ WINDOW — missing {', '.join(missing)}. Rule 2 "
                "has priority over rule 1, so an unbounded gate demotes on "
                "the deep route everywhere it fires (shadowing the primary "
                "rule and silencing yarnsig, which is primary-only) — "
                "including σ below the route's certified window, where the "
                "measured demote gap is large. The shipped 1024:768 window "
                "is 0.65 < σ < 0.95."
            )
        if not (0.0 <= float(lo) < float(hi) <= 1.0):
            raise ValueError(
                f"--sigma_lowres_threshold2/_max must satisfy "
                f"0 <= lo < hi <= 1, got {lo} / {hi}"
            )
    hi1 = rule_cfg(args, 1)[1]
    if hi1 is not None and not (
        0.0 <= float(getattr(args, "sigma_lowres_threshold", 0.5)) < float(hi1)
    ):
        raise ValueError(
            f"--sigma_lowres_threshold_max ({hi1}) must exceed "
            f"--sigma_lowres_threshold "
            f"({getattr(args, 'sigma_lowres_threshold', 0.5)})"
        )
    span_params(args, 1)
    span_params(args, 2)
    yarnsig_params(args)
    return r1, r2


def demote_choice(args, sigmas_flat, step_idx):
    """Which demote rule fires this step: 2, 1, or None.

    The secondary rule (route2 + its own gate/span) takes precedence —
    stacked-router semantics: demote DEEPER (e.g. 768) when σ is inside
    the deep route's measured window, else the primary rule (e.g. σ>0.5 →
    896), else native.
    """
    if route2(args) is not None:
        if gate_allows(args, sigmas_flat, rule=2) and span_allows(
            args, step_idx, rule=2
        ):
            return 2
    if gate_allows(args, sigmas_flat) and span_allows(args, step_idx):
        return 1
    return None


def enable_on_datasets(args, train_group) -> None:
    """Activate the σ-demote sidecar on TRAIN datasets only — validation
    stays native so val loss is comparable across arms. Refuses a yarnsig
    spec without ``--sigma_lowres``."""
    if not getattr(args, "sigma_lowres", False):
        if yarnsig_params(args) is not None:
            # `off` (and friends) parse to None, so opting out in a config that
            # never enables sigma_lowres stays inert instead of hard-failing.
            raise ValueError(
                "--sigma_lowres_yarnsig requires --sigma_lowres (it only "
                "changes rope on demoted steps, which need the demote sidecar)."
            )
        return

    r1, r2 = validate_rules(args)
    enabled_on = 0
    for ds in getattr(train_group, "datasets", []):
        if hasattr(ds, "enable_sigma_demote"):
            ds.enable_sigma_demote(*r1)
            if r2 is not None:
                ds.enable_sigma_demote2(*r2)
            enabled_on += 1
    logger.info(
        "sigma_lowres ENABLED: %d→%d demote at σ > %s on %d train "
        "dataset(s); validation stays native.",
        r1[0],
        r1[1],
        getattr(args, "sigma_lowres_threshold", 0.5),
        enabled_on,
    )
    if r2 is not None:
        lo, hi, span2 = rule_cfg(args, 2)
        logger.info(
            "sigma_lowres rule2 ENABLED (priority): %d→%d demote at %s < σ < %s%s.",
            r2[0],
            r2[1],
            lo,
            hi,
            f", span {span2}" if span2 else "",
        )
    yarn = yarnsig_params(args)
    if yarn is not None:
        logger.info(
            "sigma_lowres yarnsig ENABLED: banded rope on demoted "
            "steps, α,β=%s,%s, μ=sigmoid(%s·[logit(σ)−logit(%s)]).",
            *(yarn[0], yarn[1], yarn[3], yarn[2]),
        )


def demoted_token_counts(args, resos) -> set:
    """Token counts demoted forwards run at — they must sit inside the
    compiled dynamic-seq range. Empty when sigma_lowres is off."""
    if not getattr(args, "sigma_lowres", False):
        return set()
    from library.datasets.buckets import demoted_token_counts as _counts

    counts = _counts(resos, *route(args))
    r2 = route2(args)
    if r2 is not None:
        counts |= _counts(resos, *r2)
    return counts


class SigmaDemoter:
    """Per-run σ-demote state: the train-forward counter the span gate reads,
    eligible/demoted tallies, warn-once flags, and this step's yarnsig
    params (``yarn_step``, consumed by the primary forward)."""

    def __init__(self):
        # Pre-seeded with the resume offset by the trainer — else a resumed run
        # would measure the span boundary from the resume point, not absolute T.
        self.span_step = 0
        self.yarn_step = None
        self._seen = 0
        self._demoted = 0
        self._no_flat_draw_warned = False
        self._dead_rule_warned: set[int] = set()

    def _warn_dead_rule(self, args, rule):
        """Warn once per rule when its gate passed but the batch has no
        sibling latent for that route (the rule would silently fall through).
        """
        if rule in self._dead_rule_warned:
            return
        self._dead_rule_warned.add(rule)
        r = (route2(args) if rule == 2 else route(args)) or ("?", "?")
        logger.warning(
            "sigma_lowres: rule %d (%s:%s) passed its σ gate but the batch "
            "carries no sibling latent for that route — those steps fall "
            "through to %s. Emit it with `make preprocess-demote "
            'ARGS="--sigma_demote %s:%s"` (or list both routes in '
            "configs/preprocess.toml's sigma_demote). Warned once per rule.",
            rule,
            r[0],
            r[1],
            "rule 1 / native" if rule == 2 else "native",
            r[0],
            r[1],
        )

    def maybe_demote(
        self, ctx: TrainCtx, batch, latents, is_train, adapters, generator=None
    ):
        """sigma_lowres σ-demote (σ > threshold → demote-tier latent).

        Returns ``(latents, sigmas_flat)``: possibly-swapped latents plus the
        pre-drawn flat σ (None → sampler draws internally). Active only when
        --sigma_lowres is on, the batch carries ``demoted_latents``, and this
        is a train step; ``--sigma_lowres_span`` further restricts by training
        progress. Side effect: sets :attr:`yarn_step`.
        """
        args = ctx.args
        self.yarn_step = None
        if not is_train or not getattr(args, "sigma_lowres", False):
            return latents, None
        step_idx = self.span_step + 1
        self.span_step = step_idx
        demoted = batch.get("demoted_latents")
        demoted2 = batch.get("demoted_latents2")
        if demoted is None and demoted2 is None:
            return latents, None
        unsafe = [
            a.name for a in adapters if not getattr(a, "sigma_demote_safe", False)
        ]
        if unsafe:
            raise ValueError(
                f"--sigma_lowres is unsupported under method adapter(s) "
                f"{unsafe} — fixed-grid cond/extra-forward streams need their "
                "own operating-point probe first (sigma_lowres Q5). "
                "Grid-agnostic adapters (repa) are allowed."
            )
        from library.runtime.noise import draw_flat_sigmas

        sigmas_flat = draw_flat_sigmas(
            args,
            latents.shape[0],
            latents.shape[-2],
            latents.shape[-1],
            ctx.accelerator.device,
            generator=generator,
        )
        if sigmas_flat is None:
            if not self._no_flat_draw_warned:
                self._no_flat_draw_warned = True
                logger.warning(
                    "sigma_lowres: timestep_sampling=%r has no flat-σ draw — "
                    "training native throughout (no demotion).",
                    getattr(args, "timestep_sampling", None),
                )
            return latents, None
        self._seen += 1
        total = self._seen
        # Priority router: rule 2 > rule 1 > native. A rule whose sibling
        # latent is missing falls through to the next.
        choice = demote_choice(args, sigmas_flat, step_idx)
        if choice == 2 and demoted2 is None:
            self._warn_dead_rule(args, 2)
            choice = (
                1
                if gate_allows(args, sigmas_flat) and span_allows(args, step_idx)
                else None
            )
        if choice == 1 and demoted is None:
            self._warn_dead_rule(args, 1)
            choice = None
        if choice is not None:
            native_hw = tuple(latents.shape[-2:])
            swapped = demoted2 if choice == 2 else demoted
            latents = swapped.to(device=latents.device, dtype=latents.dtype)
            # yarnsig belongs to the PRIMARY rule only — the deep route's
            # window was measured on plain demotion.
            yarn = yarnsig_params(args) if choice == 1 else None
            if yarn is not None:
                alpha, beta, center, gamma = yarn
                # Patch-grid units: demoted patch i sits at
                # i · (native_patches / demoted_patches). μ from the
                # batch-min σ (the sample nearest the gate).
                s = min(max(float(sigmas_flat.min()), 1e-6), 1.0 - 1e-6)
                mu = 1.0 / (
                    1.0
                    + math.exp(
                        -gamma
                        * (math.log(s / (1.0 - s)) - math.log(center / (1.0 - center)))
                    )
                )
                self.yarn_step = (
                    (native_hw[0] // 2) / (latents.shape[-2] // 2),
                    (native_hw[1] // 2) / (latents.shape[-1] // 2),
                    alpha,
                    beta,
                    mu,
                )
            self._demoted += 1
            if self._demoted == 1:
                logger.info(
                    "sigma_lowres: first demoted step — latent grid %s → %s at σ=%s%s",
                    native_hw,
                    tuple(latents.shape[-2:]),
                    [round(float(s), 3) for s in sigmas_flat],
                    (
                        f" (yarnsig μ={self.yarn_step[-1]:.3f})"
                        if self.yarn_step is not None
                        else ""
                    ),
                )
        if total % 500 == 0:
            logger.info(
                "sigma_lowres: demoted %d/%d eligible steps (%.1f%%)",
                self._demoted,
                total,
                100.0 * self._demoted / total,
            )
        return latents, sigmas_flat
