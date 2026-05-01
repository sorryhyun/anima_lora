# Contributing

Thanks for considering a contribution. This repo welcomes targeted fixes and new adapter methods. Read the right tier below before opening a PR — the bar is different for each.

## Before you start

- Open or comment on a [GitHub issue](https://github.com/sorryhyun/anima_lora/issues) describing the change. For anything bigger than Tier 1, please get a thumbs-up on scope before writing code — saves a round-trip on direction.
- Read [`CLAUDE.md`](CLAUDE.md) end-to-end. It is the single source of truth for the config flow, training invariants, and architecture. Most "is this how things work?" questions are answered there.
- Use `uv` for everything (`uv sync`, `uv run python …`). Don't add `pip` instructions to docs or commit `requirements.txt` files.
- Run the linters before pushing:
  ```bash
  ruff check . --fix && ruff format .
  ```

## Priority areas

Two areas where outside contributions would have the biggest impact right now. Both are wired and training end-to-end — what's missing is productionization, ecosystem, and breadth. Tier annotations below map each item to the requirements in the rest of this document.

### IP-Adapter

Decoupled image cross-attention (Ye et al. 2023). DiT frozen; resampler + per-block `to_k_ip`/`to_v_ip` train end-to-end and inference works via `inference.py`. See [`docs/methods/ip-adapter.md`](docs/methods/ip-adapter.md).

What's missing:

- **Integration tests** — fixed reference images, recorded resampler outputs, per-block IP KV shape contract, end-to-end seeded SSIM range. *[Tier 1.5]*
- **Reference checkpoint on HuggingFace** — `anima-ip-adapter-v1` with a model card (training recipe, samples, numbers) so contributors have a baseline to compare against. *[ecosystem; no new code, treat as Tier 1.5 — paste training command + numbers]*
- **Pluggable vision encoder** — PE-Core-L14-336 is the install-friction tax. SigLIP-L or CLIP-L as a swappable lighter default; PE-Core stays as the high-quality option. *[Tier 1.5]*
- **ComfyUI parity** — verify `AnimaAdapterLoader` (in `custom_nodes/comfyui-hydralora/`) covers IP-Adapter (resampler + per-block IP KV) or file the gap. *[Tier 1]*
- **Image-only CFG, batched multi-reference, `--ip_scale` schedule** — small self-contained PRs. *[Tier 1]*

### EasyControl adapters

Per-block cond LoRA on self-attn + FFN with a logit-bias gate. The architecture is naturally contribution-friendly: each control type is one independent adapter. See [`docs/methods/easycontrol.md`](docs/methods/easycontrol.md). What's missing is the **adapter zoo** around it.

- **Trained adapters** — canny, depth, pose, lineart, scribble, segmentation, … each one a self-contained PR with model card, training config, and samples. Hosted under a HuggingFace collection (planned: `anima-easycontrol`). *[Tier 1.5 — bench numbers and side-by-side samples carry the PR; no new method code]*
- **Per-task dataset spec** — one doc per control type covering pair format, recommended size (~2k pairs), where to source signal images. Currently undocumented. *[Tier 1]*
- **Toy datasets** — 200-pair CC-licensed bundles per control type so a contributor can validate the pipeline before committing to a full dataset. *[Tier 1]*
- **One-command training aliases** — `make easycontrol-canny`, `make easycontrol-depth`, … as per-task preset configs in `configs/methods/easycontrol/`. *[Tier 1]*
- **Eval harness** — held-out ~100-pair sets per control type with a control-fidelity metric (re-extract signal from generation, compare to input). Lets adapter PRs be reviewed on numbers rather than vibes. *[Tier 1.5]*

Open a draft PR or issue early — happy to scope and review.

## Tier 1 — bug fixes, typos, UI, arg/CLI tweaks

Lightweight contributions. Examples: fixing a regex in a LoRA target list, a typo in a docstring, a confused error message, a GUI label, a missing CLI flag, a `tasks.py` argument-forwarding bug.

**Requirements:**
- Existing tests pass:
  ```bash
  make test-unit
  ```
- The change is minimal and scoped. No drive-by refactors, no new abstractions, no "while I'm here" reformatting in unrelated files.
- For GUI changes, actually launch the GUI (`make gui`) and exercise the affected tab before claiming the PR is done. Type-checking is not a substitute for clicking the button.
- For training-path changes, smoke-test one short run end-to-end (`PRESET=low_vram make lora` truncated to a few steps is fine) and paste the tail of the log into the PR description.

That's it. Open the PR.

## Tier 1.5 — efficiency improvement or algorithm revision

A change that touches an existing method's compute path, scheduling, or numerics — without introducing a new method. Examples: a faster kernel for an existing attention path, replacing an FP32 reduction with a lower-precision one, revising T-LoRA's mask schedule, tweaking HydraLoRA's router temperature handling, changing the LSE correction in `attention_dispatch.py`, swapping the optimizer step order for memory.

These sit between Tier 1 and Tier 2: no new paper or new docs page is required, but **the burden of proof is empirical** — you are claiming the existing method runs faster, uses less memory, or produces equivalent-or-better outputs under a revised algorithm. That claim has to be measurable.

**Requirements:**

1. **Bench script.** A runnable script that quantifies the change. Two acceptable shapes:
   - **Add to an existing `bench/<method>/`** if the change is scoped to one method (e.g. a HydraLoRA router tweak goes under `bench/hydralora/`). Append a new script and a new section to that bench's README.
   - **Add a small `bench/<topic>/`** for cross-cutting changes (e.g. an attention dispatch optimization belongs in something like `bench/attention/` or extends the existing `bench/rope_fusion/`).

   The script must report the headline number(s) it claims to move — wall-clock, peak VRAM, loss-at-N-steps, drift, whatever the change targets — for **both before and after**. A single-number claim ("20% faster") with no reproducible script does not clear the bar.

2. **New or extended tests.** At least one test that locks in the invariant the change is supposed to preserve. Examples:
   - For a kernel rewrite: a numerical-equivalence test against the previous path within a stated tolerance.
   - For a schedule revision: a test that the new schedule reduces to the old one under a documented config flag, so the change can be A/B'd.
   - For a memory optimization: an assertion on peak allocator usage on a small fixture, if feasible.

   Add the test to `tests/`, following the patterns in `test_network_registry.py` and `test_lora_custom_autograd.py`. If exact equivalence is impossible (e.g. a deliberately different algorithm), state the tolerance and what would constitute a regression.

3. **Documentation update.** Update the relevant `docs/methods/<name>.md`, `docs/optimizations/<name>.md`, or section of `CLAUDE.md` to reflect the new behavior. No new top-level doc unless the change introduces a user-visible flag that warrants one.

4. **Result in the PR description.** Paste the bench output (before/after) and the test results into the PR description. Link to the bench script that produced them. Reviewers should be able to reproduce the claim with one command.

5. **Backwards-compat statement.** If the change alters numerics (loss curves shift, output images change at fixed seed), say so explicitly. If it does not, say that and explain why — bit-equivalent refactors and behavior-changing optimizations get reviewed differently.

A paper citation is welcome but not required. If the revision is paper-derived, cite the paper as you would in Tier 2; if it's a hand-rolled improvement, the bench results stand on their own.

## Tier 2 — new LoRA / adapter method

A new entry in `networks/lora_modules/` or `networks/methods/`, or a new variant block in `configs/methods/lora.toml` / a new `configs/methods/<name>.toml`.

**Requirements:**

1. **Paper reference.** New methods exist because someone published a result that justifies the complexity. The PR description must cite the paper (title, authors, venue, arXiv id) and the upstream code if any. Method docs in `docs/methods/<name>.md` follow the same format as the existing ones — see `docs/methods/reft.md` and `docs/methods/easycontrol.md` for the shape.

   Hand-rolled methods without prior art are not categorically rejected, but the bar is higher: in the absence of a paper, the bench results have to carry the argument alone, and reviewers will be skeptical. If you are confident, propose the method in an issue first.

2. **Dedicated bench subdirectory.** Create `bench/<method_name>/` with the same shape as the existing ones (`bench/spectrum/`, `bench/dcw/`, `bench/easycontrol/`, `bench/img2emb/`, `bench/inversionv2/`):

   ```
   bench/<method_name>/
   ├── README.md              # what the bench measures, how to run, how to read output
   ├── proposal.md            # (optional) design framing — why this method, what it should beat
   ├── plan.md                # (optional) integration plan if the bench is an early diagnostic
   ├── <bench_script>.py      # a runnable script, not a notebook
   └── results/               # gitignored except for the timestamped run you cite in the PR
   ```

   The bench README must include:
   - **What it measures** — the headline number(s) and what "good" looks like.
   - **Run command** — copy-pasteable, defaults reasonable, runs on a single 12–16 GB GPU in under 30 minutes.
   - **Output layout** — what files land in `results/<timestamp>/`.
   - **Interpretation** — what the numbers mean, including what would falsify the method.
   - **Baseline run** — at least one results directory checked in (or linked from a release artifact if large), with the exact CLI used to produce it.

   `bench/dcw/README.md` is a good template — it documents the measurement, has an "Observed on Anima" section with a dated baseline, and a "Next actions" section. Aim for that.

3. **Documentation.** A method doc at `docs/methods/<name>.md` covering the algorithm, config knobs, training/inference flow, and known failure modes. Cross-link from the README's "Experimental features" table.

4. **Tests.** At least a smoke test that constructs the network and runs one forward pass on CPU/CUDA. Existing tests in `tests/` show the shape (`test_network_registry.py`, `test_loss_registry.py`, `test_smoke.py`).

5. **Make/`tasks.py` entry points.** A new method needs `make <name>` and matching `python tasks.py <name>` invocations, plus a `test-<name>` target that runs `inference.py` against a checkpoint produced by the method. Follow the patterns in the `Makefile`.

6. **Mergeability statement.** If the method produces weights that fold into the base DiT (LoRA family), confirm that `make merge` works and ship a merge-equivalence check in the bench. If it does not (ReFT / Hydra moe / postfix / prefix / IP-Adapter / EasyControl), say so explicitly in the doc and update `scripts/merge_to_dit.py`'s refusal list.

7. **Empirical result.** The PR must show the method works on Anima specifically. Cite a bench run from `bench/<method_name>/results/<timestamp>/` and link to a small set of side-by-side images (3–6 seeds is fine) demonstrating the claimed effect. "It compiles and trains without crashing" is not a result — both `LoRA + this` and `LoRA alone` need to be in the comparison.

## Tier 3 — new base-model support

**Currently not accepted.**

This repo is Anima-specific by design. Adding a second base model is a multi-week project that touches the trainer forward path, every adapter monkey-patch, the cache filename convention, and every `configs/methods/*.toml` LoRA target list. The blocker is `train.py::get_noise_pred_and_target` and the per-adapter Anima coupling, not the DiT class itself. See [`docs/multi_model_support.md`](docs/multi_model_support.md) for the full terrain map and effort estimate.

What is in scope:
- **Improving `docs/multi_model_support.md`** — sharper coupling map, more accurate effort estimates, concrete protocol sketches, a worked example of what a `ModelFamily` port would look like for a specific candidate model. Pure-doc PRs of this kind are welcome.
- **Decoupling work that has standalone value on Anima** — e.g. parameterizing cache suffixes, lifting the LoRA target regex into a `lora_target_spec()`, moving strategy base classes up. If a refactor makes the Anima code cleaner *and* incidentally reduces the multi-model blocker, propose it as its own PR with the Anima-side justification leading.

What is not in scope:
- A second `library/models/<family>/` namespace populated for a real second model.
- A new `forward_for_loss` slot on a hypothetical `ModelFamily` protocol that nothing else uses yet.
- Caches, configs, or test fixtures for a second model.

If you want to fork the repo to support a different base model, that's fine and encouraged — but the upstream stays Anima-only until a maintainer decides otherwise.

## PR checklist

Copy this into your PR description and tick what applies:

- [ ] Tier identified (1 / 1.5 / 2 / 3-eligible doc work).
- [ ] `make test-unit` passes locally.
- [ ] `ruff check` and `ruff format` clean.
- [ ] (Tier 1.5) Bench script added or extended; before/after numbers in the PR description.
- [ ] (Tier 1.5) New or extended test locking in the invariant the change preserves.
- [ ] (Tier 1.5) Backwards-compat statement: numerics-equivalent or behavior-changing.
- [ ] (Tier 2) Bench subdirectory present with README, runnable script, and a timestamped baseline run.
- [ ] (Tier 2) Paper citation in the PR description and method doc.
- [ ] (Tier 2) `docs/methods/<name>.md` added and cross-linked from `README.md`.
- [ ] (Tier 2) `make <name>` and `make test-<name>` work.
- [ ] (Tier 2) Merge story documented (folds into DiT? if not, why not?).
- [ ] No commented-out code, no `print(...)` debug leftovers, no unrelated formatting churn.

## License

By contributing you agree your changes are licensed under the same license as this repo (see `LICENSE`).
