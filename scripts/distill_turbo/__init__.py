"""Turbo Anima — DP-DMD distillation.

Trains an N-step LoRA student against the CFG=4 Anima teacher with Diversity-Preserved
DMD (arXiv:2602.03139) and a co-LoRA fake score model. Knobs and ops:
``docs/anima/methods/turbo.md``; structure and math: ``docs/anima/structure/turbo.md``.

Config:   ``configs/methods/turbo.toml`` (CLI flags override TOML values).
Entry:    ``python -m scripts.distill_turbo.distill`` (``make turbo``).

Module map:

* :mod:`scripts.distill_turbo.distill`    — main loop (``run_loop``) and the
  per-iteration orchestration, logging, checkpoints.
* :mod:`scripts.distill_turbo.setup`      — ``build_run``: model, stacks, compile,
  optimizers, data, logging.
* :mod:`scripts.distill_turbo.context`    — ``RunContext`` (the built run state).
* :mod:`scripts.distill_turbo.steps`      — per-step DP-DMD loss terms and updates.
* :mod:`scripts.distill_turbo.config`     — ``TurboConfig`` field table → argparser,
  CLI/TOML resolver, validation.
* :mod:`scripts.distill_turbo.primitives` — CDM grid and extrapolation, GAN ramp,
  LR scheduler.
* :mod:`scripts.distill_turbo.warmup`     — fake (critic) head-start
  (``fake_warmup_steps``) before the main loop.
* :mod:`scripts.distill_turbo.anchor_cache` — anchor target cache: builder and
  dataset.
* :mod:`scripts.distill_turbo.metrics`    — GPU-side accumulators + single-sync
  log flush.
* :mod:`scripts.distill_turbo.resume`     — crash-resume state.
* :mod:`scripts.distill_turbo.diversity`  — same-prompt diversity validation.

Output: ``<output_dir>/<output_name>.safetensors``, a plain LoRA (``make test-turbo``).
"""
