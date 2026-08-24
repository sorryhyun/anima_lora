"""Modulation-guidance distillation scripts.

Two CLIs (formerly the ``distill-prep`` / ``distill-mod`` make targets,
removed when the line finished):

* ``python -m project.finished.mod_guidance.prep``    — Phase 1 (T5("") sidecar) + Phase 2
  (teacher-synthesized clean latents) staging.
* ``python -m project.finished.mod_guidance.distill`` — train ``pooled_text_proj`` against
  the frozen teacher.

Shared modules:

* :mod:`library.anima.uncond`              — T5("") sidecar encode/load helpers.
* :mod:`library.preprocess.uncond`         — T5("") sidecar staging (produce-to-disk).
* :mod:`project.finished.mod_guidance.synth`         — Phase 2 teacher-driven synthesis.
* :mod:`project.finished.mod_guidance.teacher_cache` — train + val teacher prediction caches.
* :mod:`project.finished.mod_guidance.validation`    — fixed-sigma teacher↔student MSE pass.
"""
