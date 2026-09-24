# project/finished/ — completed lines

Lines that ran to a successful conclusion — the goal was reached or the
measured ceiling was hit — land here, one subdir each. This tier is distinct
from the gitignored `_archive/` tree on both axes:

- Outcome: `finished/` lines *worked* (shipped artifacts, ceiling reached);
  `_archive/` holds retired/killed/superseded material.
- Visibility: `finished/` is **tracked** — the record of what was achieved
  and where the ceiling sits stays in the public repo, so nobody re-proposes a
  lever that was already exhausted.

Each subdir is a digest home page (`README.md`): final verdicts with evidence
pointers, shipped artifacts, and any small open remainder. It does not
duplicate code or bench tables — canonical sources stay where they live.

A finished line's whole working tree may move here when its top-level surface
(make targets, repo-root dir) is no longer worth keeping — the code stays
runnable via direct script invocation. Each line's `STATUS.md` (or `README.md`
for digest-only homes) records the verdicts.

Finished lines:

- [`mod_guidance/`](mod_guidance/) — text-conditioned AdaLN steering via a
  distilled `pooled_text_proj` MLP (2026-08-24). Shipped (release asset + the
  Spectrum KSampler node); the bench was archived 2026-07-12 with every axis
  terminal — it's a global-tone/finishing lever, the σ and layer schedule axes
  are falsified, and the content-direction ceiling is architectural. The
  one-shot distillation loop moved here from its old `distill_mod` home under
  `scripts/`, and its two
  make targets were removed. Verdicts:
  [`mod_guidance/STATUS.md`](mod_guidance/STATUS.md); ops:
  [`mod_guidance/README.md`](mod_guidance/README.md).
- [`sr/`](sr/) — ResShift ×4/×2 super-resolution sidecar (full tree moved
  from repo-root `sr/`; `make sr-*` targets removed). ×2 and ×4 both at their
  measured ceilings (teacher-bound; distillation faithful). Open remainder:
  Korean-text training data. Verdicts: [`sr/STATUS.md`](sr/STATUS.md); ops:
  [`sr/README.md`](sr/README.md).
- [`cjk_aware_anima/`](cjk_aware_anima/) — the encoder-side CJK line: native
  JA / KO / ZH prompting through extra T5-side vocab rows distilled against the
  EN-translation teacher (frozen 2026-09-05, moved here 2026-09-24). Shipped:
  the `synthja_v4` tag tier and the `synthjakozh1sym_r256` pack that the
  public `anima-vocab-pack-cjk` repo is built on. Ceiling: rare kanji names
  never compose under any lever, coverage / geometry refinements are inert,
  content-free tables tie the trained pack for unmask training; the DiT-side
  question continued in `../cjk_aware_anima_dit/` → `../cjk_renderable_anima/`.
  The pack recipe stays operational in `scripts/distill_cjk/` (corpus builders
  under `corpus/`; `docs/methods/cjk_vocab_pack.md` § Rebuilding a pack).
  Verdicts: [`cjk_aware_anima/findings.md`](cjk_aware_anima/findings.md);
  digest: [`cjk_aware_anima/README.md`](cjk_aware_anima/README.md).
