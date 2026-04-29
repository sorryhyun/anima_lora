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

## Tier 1 — bug fixes, typos, UI, arg/CLI tweaks

Lightweight contributions. Examples: fixing a regex in a LoRA target list, a typo in a docstring, a confused error message, a GUI label, a missing CLI flag, a `tasks.py` argument-forwarding bug.

**Requirements:**
- Existing tests pass:
  ```bash
  make test-unit
  ```
- The change is minimal and scoped. No drive-by refactors, no new abstractions, no "while I'm here" reformatting in unrelated files.
- For GUI changes, actually launch the GUI (`make gui`) and exercise the affected tab before claiming the PR is done. Type-checking is not a substitute for clicking the button.
- For training-path changes, smoke-test one short run end-to-end (`make lora-low-vram` truncated to a few steps is fine) and paste the tail of the log into the PR description.

That's it. Open the PR.

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

- [ ] Tier identified (1 / 2 / 3-eligible doc work).
- [ ] `make test-unit` passes locally.
- [ ] `ruff check` and `ruff format` clean.
- [ ] (Tier 2) Bench subdirectory present with README, runnable script, and a timestamped baseline run.
- [ ] (Tier 2) Paper citation in the PR description and method doc.
- [ ] (Tier 2) `docs/methods/<name>.md` added and cross-linked from `README.md`.
- [ ] (Tier 2) `make <name>` and `make test-<name>` work.
- [ ] (Tier 2) Merge story documented (folds into DiT? if not, why not?).
- [ ] No commented-out code, no `print(...)` debug leftovers, no unrelated formatting churn.

## License

By contributing you agree your changes are licensed under the same license as this repo (see `LICENSE`).
