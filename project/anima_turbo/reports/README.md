# Reports

One report per run or probe. Run ids are in [../configs/README.md](../configs/README.md).

- [arm0.md](arm0.md): arm 0 control and twin. The two runs share only 9–10 % of ΔW,
  both random-walk from step 125, and the σ_a norm profile gives arm 3 = adaptive clip alone.
  The control's raw 250 sits inside a divergence burst. The render noise floor is large, and
  composition is the gap both seeds share. GAN-off: GAN is not the walk.
- [handoff.md](handoff.md): teacher / student handoff at each step boundary on the arm 0
  students. Step 0 (σ 1 → 0.9) is the largest single loss of the teacher's layout; with the
  teacher drawing step 0 the student's steps 1–3 keep the view from below. Render read by Claude,
  the user's read pending.
- [step0.md](step0.md): the step-0 arms. The anchor target is right (S-a). At 250
  iterations no trained step 0 gets past the teacher's instantaneous velocity (S-b, S-b2, S-c.1),
  and four anchor draws buy a small gain (S-c.2). The fit follows the LR at the end of training
  and not the length (`1112_1336`, `1112_1410`); at a held-out MSE of 0.12 the layout is still
  the students' own.
- [step0_fit.md](step0_fit.md): how good the step-0 fit has to be, and Hub v3 as a
  fitted step 0. The layout is in the low band of the step-0 residual and comes back in
  proportion to the fit; Hub v3 (10k it) sits at 0.3 of the reference where the 1k students sit
  at 0.85, and its step 0 carries over to merged14 untrained (held-out MSE 0.063 against 0.146).
- [walk.md](walk.md): the measurements taken before arm 0. It holds what the 2-step line
  measured (the batch-1 random walk), the 4-step ΔW overlap and the 2-step gradient norms.
- [mfd.md](mfd.md): MFD in the DM slot (2-step 100 / 500-step arms, 2026-11-04), the
  anchor-free MFD collapse (4-step m14, 2026-11-10) and arm 1 on the v1 recipe (2026-11-13):
  cleaner, and the seeds of a prompt draw one image.
- [pref_p1.md](pref_p1.md): the preference head's P0 and P1 on four bases. Each ranks gelcrawl's
  selected images on held-out artists (AUROC 0.66–0.71); on 25 rounds of picks every head is at
  chance (0.41–0.55), zero-shot, fitted and refitted. The DiT-head basis is not built.
- [warm_v3.md](warm_v3.md): the control warm-started from Hub v3. It passes the gate
  against the control and regresses from v3 untrained on merged14, which sits at the teacher's
  seed diversity.
