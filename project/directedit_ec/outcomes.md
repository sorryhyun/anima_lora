# directedit_ec — outcomes

The line's shippable (practical) artifacts, as distinct from its open research
phases (roadmap.md) and open questions (questions.md). Three product-shaped
outcomes plus one paper.

## 1. In-place edit recipe — SHIPPED (zero-training)

The Phase-1a/1b winner: DirectEdit inversion + the stock inpaint EasyControl
adapter, hole punched in **both** preservation mechanisms (EC cond hole +
Δz anchor mask), `b_offset 0`, no per-image tuning. Beats V-injection on
every in-place edit type at lower cost (one KV prefill vs a parallel source
forward per injected step).

- Surface: `scripts/edit.py --easycontrol_weight … --easycontrol_mask m.png
  --mask m.png` (same file both flags). Component map: `methods.md`.
- ComfyUI: `custom_nodes/comfyui-anima-directedit/` ships DirectEdit;
  the EC-hole recipe is **not** in the node yet — it rides the EasyEdit ship
  proposal below as the "in-place mode" follow-up.
- Open polish item: automatic mask source (Q5 — cfgdelta subject localizer).

## 2. EasyEdit — reference + instruction editor — SHIPPED as alpha (2026-07-26)

**Naming.** Recommended: **EasyEdit** — states the lineage (EasyControl's
architecture doing DirectEdit's job) and what it does. "DirectControl" reads
as a control method and hides the edit semantics. One caveat for the *paper*:
"EasyEdit" collides with the LLM knowledge-editing framework (zjunlp) in
academic search — fine for a ComfyUI node, use a distinct method name in the
paper (e.g. "delta-caption instruction editing").

What it is: the Phase-2.5 `subject_edit` adapter used as a standalone editor.
cond = any image, prompt = a tag-delta instruction (`additions, -removals`),
one plain generation at the trained operating point — **no inversion, no
anchor, no mask, no offset hunting** (b_offset 0 *is* the engaged band; +2
already begins the copy regime). Feed-forward: 1 generation vs DirectEdit's
2 passes. The instruction probe passed on all three judged pairs
(`bench/report.md#phase-25`): identity retrieval + instructed changes land
simultaneously; the noec control proves both come from the adapter.

- Checkpoint: `output/ckpt/anima_easycontrol_subject_edit.safetensors`
  (7860 steps / 12 epochs, arm-2 open-gate recipe).
- Semantics: **"re-render with these changes"** — identity-preserving,
  composition-free. Complementary to outcome 1 (composition-preserving,
  in-place). Both belong in one node story: EasyEdit for "change the scene /
  outfit / state", DirectEdit(+EC hole) for "edit THIS image in place".
- Known limits (disclose, don't hide): object *removals* mostly fail (the
  base TE reads `-tag` as an attractor; negation is adapter-side and weaker
  than addition), train-pair probe = upper bound, single-seed n=3 judgment.
- Ship path: `docs/proposal/easyedit_comfy_node.md` — Phases 1 + 2 done
  2026-07-26. Published as `anima_subject_edit_alpha.safetensors`
  (`sorryhyun/anima-easycontrol-adapters`); ComfyUI side is
  `~/ComfyUI-EasyControl-KSamplerCompat` v0.3.0 — README section,
  `workflows/easyedit.json`, and two CPU-only instruction-builder nodes
  (add/remove fields, or caption-diff for the tagger loop) whose output is
  test-pinned equal to the training-time miner's.
- **The held-out gate (Phase 0) was skipped, not passed.** The `_alpha` name
  plus the disclosed-limits sections carry that; if a held-out probe later
  collapses, the response is unpublish, not re-word.

## In-place editing surface map (2026-07-26)

Where the line's recipes now sit, by prompt cost and failure class. Probe
evidence: `bench/results/20260726-*inplace*` (`run_inplace_probe.py`; the
`tar=caption` arm key feeds full-caption ψ_tar to edit passes that run the
base model, which does not speak the delta-instruction format). One-glance
side-by-side (same 3 pairs / seed / sizes, cells lifted from the probe runs):
**`bench/outcomes_comparison_20260726.png`** — source | target | EasyEdit ff |
src0+EC λ0.5 | base-inv+ψ_tar λ0.5 | EC-inv-only (refuted). The outcome-1
mask recipe is absent from the sheet only because its probes ran on different
images (phase-1a/1b mask sets).

| recipe | prompt cost | preservation | fails on |
|---|---|---|---|
| EasyEdit ff @ b0 (outcome 2) | delta instruction | identity only ("re-render") | not in-place; removals weak |
| DirectEdit + EC hole + mask (outcome 1) | full ψ_src + ψ_tar + mask | in-place, best outside-hole | needs mask + captions |
| src="" + EC both passes, λ0.5 | delta instruction | in-place | aligned-copy lock on copyable sources |
| base inversion (ψ_src="") + full ψ_tar, λ0.5 | full target caption | in-place, ~2–3× closer to source than ff (mse ~0.05 vs 0.10–0.15) | copyable sources stay at reconstruction |

Two findings behind the last row (`20260726-1826-inplace-ecinvonly-probe`):

- **EC-inversion-only is REFUTED.** Inverting with cond engaged (net b0 via
  `--easycontrol_invert_b_offset 0 --easycontrol_b_offset -12`) then editing
  under the ~base model buys nothing over promptless base inversion
  (fm_error 0.088 vs 0.102; renders indistinguishable on 2/3 pairs) and
  destabilizes the copyable class (yanami: wash-out — Δz increments recorded
  under the EC field don't replay under the base field exactly where cond
  dominated the trajectory). Model-mismatch absorption is one-directional:
  base-inversion→EC-edit (`--easycontrol_edit_only`) is safe; EC-inversion→
  base-edit is not.
- **The control was the discovery**: promptless base inversion + full-caption
  ψ_tar + λ0.5 lands edits in place with zero adapter and zero tagger at
  inversion time. The earlier `inv_noec` failure was ψ_tar=delta (the base
  TE reads the instruction format as attractors), not inversion itself.

The one class every inversion recipe fails is the trivially-copyable source
(flat bg + large structural instruction): base inversion reconstructs, EC
inversion washes out, EC-both copy-locks, and the twin_edit e3 checkpoint
locks hardest (trained on aligned pairs, it has *learned* copy-through —
yanami mse_vs_src 0.0010 vs subject_edit's 0.0069;
`20260726-1758-inplace-probe-twinedit-e3`). Feed-forward EasyEdit is the only
cell that lands it. Division of labor stands: EasyEdit for structural change,
inversion recipes for surgical in-place edits.

## 3. Subject descriptor — conditional (gates pending)

The Phase-2 `subject` adapter demonstrates position-free identity retrieval
at b_offset +2/+3 (gate (c) passed), but its DirectEdit-composition gates
(a)/(b) have not run against the arm-2 checkpoint. If they pass, it ships as
a "character reference" conditioning; if not, it stays a research artifact
that fed 2.5. Do not ship ahead of the gates — its engaged band is *off* the
trained point, which is exactly the UX the delta objective fixed.

## 4. Paper — GO (prep starting 2026-07-26)

Q7's bar is now met on the story side: two connected contributions
(training-free preservation dial composing with residual-anchored inversion
+ the trained delta-caption instruction editor that removes inversion
entirely), with the subject/subject_edit contrast as the ablation that shows
*why* the delta objective is the thing that opens the gate at the trained
point. What's missing is entirely evaluation: matched-NFE external baselines,
PIE-Bench, quantitative edit-success/identity metrics (tag-readback, Q6),
held-out splits. The matched-NFE baseline table is the **first** work item,
not the last (FSG lesson). Paper-facing question updates: `questions.md` Q7.

## Reusable infra this line produced

- `bench/run_bench.py` (phases smoke/0/0b/1a/1b/2), `run_subject_probe.py`
  (retrieval isolation), `run_edit_probe.py` (instruction gate; `--rating`
  band filter + `--max_delta` judgeability cap).
- Pair miners: `easycontrol_adapters/tools/subject_pairs.py` (symlink-only
  staging) and `subject_edit_pairs.py` (delta captions, min-delta partner
  policy) — both reusable for any future descriptor.
- Mechanism knowledge now load-bearing elsewhere: `b_cond` never trains
  (init = fixed hyperparameter), the aligned-cond copy path is architectural,
  EC and V-injection cannot stack (`methods.md`).
