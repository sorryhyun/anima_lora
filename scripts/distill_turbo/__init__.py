"""Turbo Anima — DP-DMD distillation.

Trains an N-step LoRA student against the CFG=4 Anima teacher using
Diversity-Preserved DMD (Wu et al., arXiv:2602.03139) on top of a co-LoRA fake
score model. The student's first step is supervised toward a teacher K-step CFG
anchor (diversity) then detached; the remaining steps are refined by a standard
DMD loss (quality).

Docs:     ``docs/methods/turbo.md``.
Config:   ``configs/methods/turbo.toml`` (CLI flags override TOML values).

Entry point:

* ``python -m scripts.distill_turbo.distill`` — train the student LoRA.

Module map:

* :mod:`scripts.distill_turbo.distill`    — main loop (teacher anchor, student
  rollout, DMD + fake forwards, optimizer steps, save).
* :mod:`scripts.distill_turbo.config`     — TOML loader, argparser, CLI/TOML
  precedence resolver, schema validation.
* :mod:`scripts.distill_turbo.primitives` — re-noising, τ samplers, scheduler
  factory, pad-tensor cache, dataloader collate.
* :mod:`scripts.distill_turbo.warmup`     — fake (critic) head-start loop
  that runs before the main training loop.
* :mod:`scripts.distill_turbo.metrics`    — GPU-side accumulators + single-sync
  log flush.

One frozen DiT serves three roles via per-network ``set_enabled`` toggling:

    teacher view  — both LoRA stacks off (base velocity)
    student view  — student on, fake off (v_student for the rollout)
    fake view     — student off, fake on (v_fake_cond_dm)

Before the main loop, an optional fake (critic) head-start runs
``fake_warmup_steps`` fake-only updates (step 5 alone) against the student's
init ≈ teacher x_pred distribution, so the critic is calibrated before the
student LR warmup ramps to full strength — this removes the early
grad_signal_rms spike (~step 50). The student is untouched during it.

Per training step:

    1.  Teacher K-step CFG anchor (no grad), from shared noise ε:
        z_tk  = teacher_rollout(ε, K, cfg=teacher_cfg)      # diverse landing pt
        v_tgt = (ε − z_tk) / (1 − t_k)                       # first-step target

    2.  Student N-step Euler rollout from ε:
        v_first = student(ε, t=1, c)                         # grad to student
        div_loss = ‖v_first − v_tgt‖²       # diversity supervision (step 1)
        z_1 = stopgrad(euler_step(ε, v_first))               # detach step 1
        x_θ = student_rollout(z_1, N−1)                      # grad → x_θ

    3.  DMD on x_θ (steps 2..N) against teacher (CFG-guided) + fake:
        v_real = v_u + α·(v_c − v_u)        # CFG-guided real score at τ_DM
        v_fake = fake(x_τ_dm, τ_DM, c)
        grad_signal = τ_DM·(v_real − v_fake)   (optionally x0-norm normalized)
        loss = (grad_signal · x_θ).mean()
             + div_weight · div_loss          # folded in here unless detached
        loss.backward()  → student.step()

    4.  Fake update — flow-matching loss on student's x_θ distribution:
        τ_fake ~ U[0,1]
        x_t_fake = (1-τ_fake)·x_θ.detach() + τ_fake·ε_fake
        v_fake   = fake(x_t_fake, τ_fake, c)                # grad to fake params
        target   = ε_fake - x_θ.detach()                    # flow-matching target
        fake_loss = MSE(v_fake, target)  → fake.step()

Output: ``output/ckpt/anima_turbo.safetensors`` — a normal plain-LoRA file
loadable by the standard inference path with ``--infer_steps`` matched to
``student_steps`` and ``--cfg 1.0`` (``make test-turbo``).
"""
