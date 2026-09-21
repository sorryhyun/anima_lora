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

Five areas where outside contributions would have the biggest impact right now. Each item below carries a tier annotation that maps to the requirements in the rest of this document. Open a draft PR or issue early on anything bigger than Tier 1 — happy to scope and review.

### 1. EasyControl adapters

Per-block cond LoRA on self-attn + FFN with a logit-bias gate. DiT frozen; trains a handful of cond LoRA blocks plus a scalar gate. The architecture is naturally contribution-friendly: each control type is one independent adapter, no method changes required. See [`docs/experimental/easycontrol.md`](docs/experimental/easycontrol.md). The wall right now is the **adapter zoo** around it.

**There is already one shipped control task to copy: colorize** (`easycontrol_adapters/colorization/`). It's the worked template for everything below — a per-task project that builds its own *condition* (mangafied B&W → color) while reusing the shipped network unchanged. A new control type follows the same mold rather than inventing structure:

- a **project dir** under `easycontrol_adapters/<task>/` (the condition builder + `prep.py` + a README), exactly like `easycontrol_adapters/colorization/`;
- a **method config** `configs/methods/<task>.toml` (+ a `configs/gui-methods/<task>.toml` GUI variant), selected at runtime by the **`EASYADAPTER=<task>`** env var — *not* a `make easycontrol-<task>` target. The existing `make easycontrol[-preprocess] EASYADAPTER=<task>` and `make test-easycontrol EASYADAPTER=<task>` targets already dispatch on it;
- a **dataset blueprint** `configs/datasets/<task>.toml` wiring the `cond_cache_dir` (and `text_cache_dir` if the task reshapes the text channel, as colorize does).

Read colorize's [README](easycontrol_adapters/colorization/README.md) before starting a new one — its caption policy, cond-noise, and inference-settings notes generalize. Concretely, the zoo still needs:

- **Trained adapters** — canny, depth, pose, lineart, scribble, segmentation, … each one a self-contained PR in colorize's shape (project dir + `configs/methods/<task>.toml` + model card + samples). Hosted under a HuggingFace collection (planned: `anima-easycontrol`). *[Tier 1.5 — bench numbers and side-by-side samples carry the PR; no new method code]*
- **Per-task dataset spec** — one doc per control type covering pair format, recommended size (~2k pairs), where to source signal images. colorize's README is the only one written so far. *[Tier 1]*
- **Toy datasets** — 200-pair CC-licensed bundles per control type so a contributor can validate the pipeline before committing to a full dataset. *[Tier 1]*
- **Control-fidelity eval harness** — held-out ~100-pair sets per control type that re-extract the signal from generation (canny→canny, depth→depth, …) and report a fidelity metric vs the input. Lets adapter PRs be reviewed on numbers rather than vibes. No EasyControl bench currently exists; the control-fidelity harness slot is empty. *[Tier 1.5]*

### 2. Turbo LoRA (Decoupled DMD distillation)

Distill 28-step Anima @ CFG=4 into a 4–8 step generator using **co-LoRA** (LoRA for both the student and the fake score model on the same frozen DiT). The deployment story is that `turbo_anima_lora.safetensors` stacks on top of any existing concept LoRA at inference, the same way LCM-LoRA composes with style LoRAs. See [`docs/methods/turbo.md`](docs/methods/turbo.md) (ops) and [`docs/structure/turbo.md`](docs/structure/turbo.md) (structure/math) — the shipped method is DP-DMD (Wu et al., arXiv:2602.03139); the CA-decoupled DMD2 it replaced is Liu et al., arXiv:2511.22677.

Status: **shipped** — `make turbo` / `make test-turbo`, with a published 4-step student at [huggingface.co/sorryhyun/anima-turbo-4step](https://huggingface.co/sorryhyun/anima-turbo-4step). The phased plan below is retained as a worked example of how a Tier 2 method PR is scoped and gated.

The shape of the Tier 2 PR (new method + paper + a turbo bench + docs/methods entry + `make turbo` / `make test-turbo`), split along phase boundaries:

- **Phase 0: single-prompt overfit (~1 day).** Implement `networks/methods/turbo_dmd.py` (two LoRA networks, attachment toggle), `scripts/distill_turbo/` (CA + DM gradient assembly, two optimizer states, the renoise primitive), `configs/methods/turbo.toml`, `make turbo`. Prove the loop converges on one prompt at batch 1, 2k iterations. *[Tier 2 — drop a `bench/turbo/results/<ts>-phase0/` with teacher@28 vs student@4 side-by-side on a fixed seed]*
- **Phase 1: 100-prompt sweep (~3 days).** Image Reward + HPS v2.1 + per-aspect breakdown (1024², 832×1248, 1248×832). Pass = student IR ≥ 80% of teacher, no aspect below 60%. *[Tier 2 continuation]*
- **Phase 2: full HPS bench (~1 week).** 1k COCO-prompt sample, all 4 schedule configs from the paper's Table 1 as an ablation, replicates the paper's Decoupled-Hybrid claim on Anima. *[Tier 1.5 once Phase 1 has landed]*
- **Phase 3: composition test (~2 days).** (turbo only) vs (concept LoRA @ 28) vs (turbo + concept @ 4) on three existing concept checkpoints. Validates the deployment story. *[Tier 1.5]*

If Phase 1 fails after one rank bump, the proposal explicitly says kill it — don't grind. The phase gates are there to bound the contributor's downside.

### 3. Filling the bench gaps

The `bench/<method>/` convention from Tier 2 below requires every method bench to ship a `README.md` (what it measures, run command, output layout, baseline run, interpretation). Several existing subdirs predate that requirement and are missing it:

| Subdir | Status | What it has | What's needed |
|---|---|---|---|
| `bench/dave/`, `bench/memorization/` | Has README | per-method probe ladder + `results/` | Use these as the shape template |

Each missing README is a self-contained Tier 1 PR. Use `_archive/spd/bench/README.md` as the model (the SPD bench is retired, but its README remains the shape to copy): headline, what each script does, a copy-pasteable run command, the headline number(s) and what "good" looks like, links to representative `results/<timestamp>/` runs, and an "Observed on Anima" section.

A second-order bench-gap contribution worth calling out:

- **Envelope conformance.** Older bench scripts predate `bench/_common.py` and don't drop a `result.json` via `make_run_dir` + `write_result`. Auditing each script and converting the holdouts (so cross-run indexing actually works) is a clean Tier 1 PR per script.
- **A dedicated turbo bench** lands as part of the Turbo LoRA contribution in (2) above (currently `bench/turbo/`).

### 4. Translations & localization

Translatable content lives in four places, each with its own contribution shape but all reviewed as Tier 1 (no bench, no test — `make gui` walkthrough screenshots in the PR description are the proof). Missing entries in every surface below **fall back to English**, so it's fine to ship an incomplete translation and grow it over time. Currently shipped: `en` (canonical), `ko` (mostly complete), `cn` (machine-translated stub, unproofread).

**(a) GUI strings — `gui/i18n/<code>.py`.** One module per language, each exporting `STRINGS: dict[str, str]`. `gui/i18n/__init__.py` assembles these into `TRANSLATIONS` and `t(key, **kwargs)` resolves keys against the current language. To **add a new language**, drop in `gui/i18n/<code>.py` mirroring `en.py`'s key set, register it in `TRANSLATIONS`, and add a friendly label to `LANG_NAMES` in `gui/app.py` (e.g. `"ja": "日本語"`). Every key you do include must use the same `{placeholder}` names as the English source — `t()` calls `.format(**kwargs)` and will raise at runtime on a typo. *[Tier 1]*

**(b) Per-field tooltips — `gui/explanations/__init__.py`.** Two dict-of-dicts power the form-field help: `FIELD_HELP` (config-form tooltips, ~50 keys) and `PREPROCESS_FIELD_HELP` (Preprocessing tab knobs, ~10 keys). Each entry is `{"en": "...", "ko": "..."}`. Add your language code as a sibling key in every entry you want translated. `field_help()` / `preprocess_field_help()` fall back to `"en"` for missing language keys. These are the strings users see when they click a form-row label, so translation quality matters more than for transient buttons — keep technical terms (LoRA, MoE, σ-bucket, VAE) untranslated. *[Tier 1]*

**(c) Long-form method guides — `gui/explanations/guides/<name>.<lang>.html`.** Right-panel HTML blocks for method variants and the Preprocessing tab. Filename convention is `<name>.<lang>.html`; the loader (`_read_guide` in `gui/explanations/__init__.py`) auto-falls back to `.en.html` when the language version is absent. Names currently present: `lora`, `tlora`, `hydralora`, `preprocess`, plus the shared snippets `_apply_note` and `_not_mergeable`. To translate, drop in `<name>.<code>.html` files alongside the English ones — no code change required. Preserve any `<a href="…">`, `<code>`, and color-coded `<span>` markup; the GUI's QTextBrowser renders these. *[Tier 1]*

**(d) Docs and structure images — `docs/`.**
- `docs/guidelines/가이드북.md` is the end-to-end onboarding doc and only exists in Korean. An English translation (or any other language) would significantly widen the audience. The `guidebook_tooltip` string in `gui/i18n/en.py` currently points users at the Korean file — once a translation lands, wire the Guidebook button (in `gui/app.py`) to pick the right file based on `current_language()`.
- `docs/structure_images_korean/` holds Korean-labeled versions of the architecture diagrams under `docs/structure_images/` (e.g. `animakor.png` ↔ `anima.png`). English/other-language equivalents are welcome under the natural sibling tree (`docs/structure_images/` is the English baseline; e.g. the existing `docs/structure_images_korean/`). Mention which markdown files reference the diagram so the reviewer can update the embed paths.
- Method docs under `docs/methods/`, `docs/experimental/`, `docs/proposal/`, and `docs/optimizations/` are **English-only by convention** — translations are welcome as `<name>.<code>.md` siblings, but nothing reads them at runtime yet. If you contribute one, also propose how it should surface (e.g. a language switcher in the README's docs table, or wiring it into a GUI "Open method doc" button). Don't translate `CLAUDE.md` — that file is consumed by Claude Code and is single-source-of-truth for project conventions.

**(e) Tag knowledge-base descriptions — `models/danbooru_tags_classified.csv` (KR) + `.en.csv` (EN).** The Dataset tab's tag-explanation view (click a tag in the caption editor → a KB tooltip with its category and a written description) is powered by these CSVs (`name,category,post_count,description`; loaded by `anime_tools.captions.correction.load_tag_knowledge_base`). The base CSV's `description` column is **Korean** (from `Localsmile/danbooru_KR_wiki_tag_search`); `download-danbooru-tags` also builds an **English** sibling `danbooru_tags_classified.en.csv` (`anime_tools.tagger.cli.build_english_tag_csv`, joining tag names against the `isek-ai/danbooru-wiki-2024` wiki mirror). `find_tag_csv(root, lang)` resolves per UI language: `ko` → Korean base; any other language → same-language sibling `danbooru_tags_classified.<code>.csv` if present, else the English `.en.csv`. `ImageViewerTab._on_tag_clicked` (`gui/tabs/image_tab.py`) shows the tooltip whenever the resolved file is **not** the Korean base — so `en`/`ja`/`cn` all show English today, and only a hypothetical language with neither sibling nor English would suppress it. (The autocomplete *helper* is never gated — tag names and categories are language-neutral.) To add a fully-translated language, ship `danbooru_tags_classified.<code>.csv` and it's picked up automatically. ~114k rows, so partial coverage is fine — untranslated rows fall back to the name+category head with no description line. *[Tier 1]*

**Parity check (covers all per-language surfaces):**
```bash
python -c "
from gui.i18n import TRANSLATIONS as T
from gui.explanations import FIELD_HELP, PREPROCESS_FIELD_HELP
en_keys = set(T['en'])
for lang in T:
    if lang == 'en': continue
    missing = sorted(en_keys - set(T[lang]))
    print(f'{lang} i18n missing  ({len(missing)}):', missing[:5], '…' if len(missing) > 5 else '')
for name, d in (('FIELD_HELP', FIELD_HELP), ('PREPROCESS_FIELD_HELP', PREPROCESS_FIELD_HELP)):
    en_entries = {k for k, v in d.items() if 'en' in v}
    for lang in ('ko', 'cn'):
        missing = sorted(k for k in en_entries if lang not in d[k])
        print(f'{lang} {name:24s} missing ({len(missing)}):', missing[:5], '…' if len(missing) > 5 else '')
"
```

If a translated string is too long for its widget, mention it in the PR — the fix is usually a layout tweak in the relevant `gui/tabs/*.py`, not a shorter translation.

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
   - **Add to an existing `bench/<method>/`** if the change is scoped to one method (e.g. a router tweak goes under that method's bench dir such as `bench/turbo/`). Append a new script and a new section to that bench's README.
   - **Add a small `bench/<topic>/`** for cross-cutting changes (e.g. a sampler-correction optimization belongs in a new dir alongside `bench/dave/`). Start from the copy-me skeleton `bench/_template/run_bench.py`; the directory conventions (envelope, layout, per-bench README) are in `bench/README.md`.

   The script must report the headline number(s) it claims to move — wall-clock, peak VRAM, loss-at-N-steps, drift, whatever the change targets — for **both before and after**. A single-number claim ("20% faster") with no reproducible script does not clear the bar. If the script loads the DiT, use `bench/_anima.py` (`add_common_args` + `build_anima`) — same rationale as Tier 2 §2 below: every DiT-loading bench needs to expose `--compile` and load the adapter in the right order, and the helper enforces both.

2. **New or extended tests.** At least one test that locks in the invariant the change is supposed to preserve. Examples:
   - For a kernel rewrite: a numerical-equivalence test against the previous path within a stated tolerance.
   - For a schedule revision: a test that the new schedule reduces to the old one under a documented config flag, so the change can be A/B'd.
   - For a memory optimization: an assertion on peak allocator usage on a small fixture, if feasible.
   - For a gated feature (off-by-default flag): both directions — an inertness test (flag off ⇒ identical training) **and** a liveness signal (flag on ⇒ measurably on). Aux losses get the latter for free from the `LossComposer` liveness ledger (the `LIVENESS:` audit in `library/training/losses.py`); any other producer that can silently skip should emit a `<name>/active` metric.

   Add the test to `tests/`, following the patterns in `test_network_registry.py` and `test_lora_dtype_policy.py`. If exact equivalence is impossible (e.g. a deliberately different algorithm), state the tolerance and what would constitute a regression.

3. **Documentation update.** Update the relevant `docs/methods/<name>.md`, `docs/optimizations/<name>.md`, or section of `CLAUDE.md` to reflect the new behavior. No new top-level doc unless the change introduces a user-visible flag that warrants one.

4. **Result in the PR description.** Paste the bench output (before/after) and the test results into the PR description. Link to the bench script that produced them. Reviewers should be able to reproduce the claim with one command.

5. **Backwards-compat statement.** If the change alters numerics (loss curves shift, output images change at fixed seed), say so explicitly. If it does not, say that and explain why — bit-equivalent refactors and behavior-changing optimizations get reviewed differently.

A paper citation is welcome but not required. If the revision is paper-derived, cite the paper as you would in Tier 2; if it's a hand-rolled improvement, the bench results stand on their own.

## Tier 2 — new LoRA / adapter method

A new entry in `networks/lora_modules/` or `networks/methods/`, or a new variant block in `configs/methods/lora.toml` / a new `configs/methods/<name>.toml`.

**Requirements:**

1. **Paper reference.** New methods exist because someone published a result that justifies the complexity. The PR description must cite the paper (title, authors, venue, arXiv id) and the upstream code if any. Method docs follow the same format as the existing ones — see `docs/methods/hydra-lora.md` (shipped) and `docs/experimental/easycontrol.md` (experimental) for the shape. Stable methods land in `docs/methods/<name>.md`; unstable / unmerged-into-shipped methods land in `docs/experimental/<name>.md`.

   Hand-rolled methods without prior art are not categorically rejected, but the bar is higher: in the absence of a paper, the bench results have to carry the argument alone, and reviewers will be skeptical. If you are confident, propose the method in an issue first.

2. **Dedicated bench subdirectory.** Create `bench/<method_name>/` with the same shape as the existing ones (`bench/dave/`, `bench/memorization/`):

   ```
   bench/<method_name>/
   ├── README.md              # what the bench measures, how to run, how to read output
   ├── proposal.md            # (optional) design framing — why this method, what it should beat
   ├── plan.md                # (optional) integration plan if the bench is an early diagnostic
   ├── <bench_script>.py      # a runnable script, not a notebook
   └── results/               # gitignored except for the timestamped run you cite in the PR
   ```

   Wire the script's output through `bench/_common.py` so it produces a standard `result.json` envelope (script path, git SHA, env, args, metrics, artifacts) under `results/<YYYYMMDD-HHMM>[-<label>]/`. The two helpers are:

   ```python
   from bench._common import make_run_dir, write_result

   out_dir = make_run_dir("<method_name>", label=args.label)
   # ... write CSVs / PNGs / etc. into out_dir ...
   write_result(out_dir, script=__file__, args=args,
                metrics={...}, artifacts=[...], device=device)
   ```

   **Benches that load the DiT must use `bench/_anima.py`.** It owns the model-side boilerplate — argparse surface, DiT + adapter loading in the correct order, bucketed sample discovery from `post_image_dataset/lora/`. `add_common_args(parser)` injects `--label`, `--seed`, `--device`, `--dtype`, `--attn_mode`, `--gradient_checkpointing`, `--cpu_offload_checkpointing`, `--compile`, `--compile_mode`. `build_anima(args, adapter=..., train_mode=...)` does the load in the right order — in particular, **`compile_blocks` runs after `apply_to` + `load_weights`** so adapter monkey-patches are part of the compiled graph. Open-coding this is a footgun (skipping `--compile` entirely, or compiling in the wrong order so the adapter is bypassed). Use the helper:

   ```python
   from bench._anima import add_common_args, build_anima, discover_bucketed_samples

   p = argparse.ArgumentParser()
   p.add_argument("--dit", required=True)
   p.add_argument("--adapter", default=None)
   add_common_args(p)
   args = p.parse_args()

   bundle = build_anima(args, adapter=args.adapter, train_mode=False)
   anima, network = bundle.anima, bundle.network
   ```

   Benches that don't load the DiT (analytical simulators, post-hoc result-aggregators) don't import `bench/_anima.py` — both modules are opt-in.

   The bench README must include:
   - **What it measures** — the headline number(s) and what "good" looks like.
   - **Run command** — copy-pasteable, defaults reasonable, runs on a single 12–16 GB GPU in under 30 minutes.
   - **Output layout** — what files land alongside `result.json` under `results/<YYYYMMDD-HHMM>[-<label>]/`.
   - **Interpretation** — what the numbers mean, including what would falsify the method.
   - **Baseline run** — at least one results directory checked in (or linked from a release artifact if large), with the exact CLI used to produce it.

   `_archive/spd/bench/README.md` is a good template (the SPD bench itself is retired — the README shape is what to copy): it documents the measurement, has an "Observed on Anima" section with a dated baseline, and a "Next actions" section. Aim for that.

3. **Documentation.** A method doc at `docs/methods/<name>.md` covering the algorithm, config knobs, training/inference flow, and known failure modes. Cross-link from the README's "Experimental features" table.

4. **Tests.** At least a smoke test that constructs the network and runs one forward pass on CPU/CUDA. Existing tests in `tests/` show the shape (`test_network_registry.py`, `test_loss_registry.py`, `test_smoke.py`).

5. **Make/`tasks.py` entry points.** A new method needs `make <name>` and matching `python tasks.py <name>` invocations, plus a `test-<name>` target that runs `inference.py` against a checkpoint produced by the method. Follow the patterns in the `Makefile`.

6. **Mergeability statement.** If the method produces weights that fold into the base DiT (LoRA family), confirm that `make merge` works and ship a merge-equivalence check in the bench. If it does not (Hydra moe / postfix / prefix / IP-Adapter / EasyControl), say so explicitly in the doc and update `scripts/toolkits/merge_to_dit.py`'s refusal list.

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
