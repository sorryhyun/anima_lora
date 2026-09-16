"""Turbo Anima — DP-DMD distillation.

Trains an N-step LoRA student against the CFG=4 Anima teacher using
Diversity-Preserved DMD (Wu et al., arXiv:2602.03139) on top of a co-LoRA fake
score model. One frozen DiT serves the teacher / student / fake views
(``TurboDMDNetwork.set_view``). The per-step algorithm, config keys and ops live in
``docs/methods/turbo.md``; module structure in ``docs/structure/turbo.md``.

Config:   ``configs/methods/turbo.toml`` (CLI flags override TOML values).
Entry:    ``python -m scripts.distill_turbo.distill`` (``make turbo``).

Module map:

* :mod:`scripts.distill_turbo.distill`    — main loop (``run_loop``).
* :mod:`scripts.distill_turbo.setup`      — run construction before the step loop.
* :mod:`scripts.distill_turbo.steps`      — per-step DP-DMD loss terms and updates.
* :mod:`scripts.distill_turbo.config`     — TOML loader, argparser, CLI/TOML
  precedence resolver, schema validation.
* :mod:`scripts.distill_turbo.primitives` — re-noising, τ samplers, scheduler
  factory, pad-tensor cache, dataloader collate.
* :mod:`scripts.distill_turbo.warmup`     — fake (critic) head-start loop
  (``fake_warmup_steps``) before the main loop.
* :mod:`scripts.distill_turbo.metrics`    — GPU-side accumulators + single-sync
  log flush.
* :mod:`scripts.distill_turbo.resume`     — crash-resume state.
* :mod:`scripts.distill_turbo.diversity`  — same-prompt diversity validation.
* :mod:`scripts.distill_turbo.softrank`   — soft-rank caption-discrimination
  auxiliary.

Output: ``<output_dir>/<output_name>.safetensors`` — a plain LoRA; infer with
``--infer_steps`` matched to ``student_steps`` and ``--cfg 1.0``
(``make test-turbo``).
"""
