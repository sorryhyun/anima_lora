"""Internals of ``run_sigma_probe.py`` (the σ-binned gradient-equivalence probe).

Split out of the single-file instrument 2026-07-31; the CLI, flag semantics
and on-disk outputs are unchanged — ``run_sigma_probe.py`` remains the only
entry point.

| module | what |
|---|---|
| ``cli`` | the argparse surface + cross-flag validation, folded into a ``RunConfig`` |
| ``kernel`` | everything that touches the model: σ grids, RoPE-alignment patches, the per-bin gradient estimator |
| ``stats`` | accumulators + every cosine/gap reduction (per-arm, pooled, κ, headline) |
"""
