---
name: model-catalog
description: The model catalog (library/downloads.py) — one Asset row per weight (repo, files, destination, installed probe), packs, resolve() name order, and the rule that loaders import their default paths from here. Load before adding or moving a weight, changing a loader's default path, adding a download target, or touching a GUI Models panel.
---

# Model catalog

`library/downloads.py` is the single catalog: **one `Asset` per weight** — repo · files ·
destination · offline installed-probe. It holds the rows only the trainer needs (the
Anima half) and concatenates `anime_tools.downloads` for the curation half. The
package's own catalog skill is `../anime_tools/.claude/skills/model-catalog/SKILL.md`.

**Add a weight by adding a row, not a command.** `make download-*`, `make download-list`
and both GUI Models panels read the catalog; `resolve()` takes a name as **legacy alias →
pack id → row id** (so `pe` keeps its historical meaning). **Loaders import their default
paths from here** rather than spelling them.

## Packs

A pack is what a "Download pack" button is a button *for*. Display order is
`TRAINER_PACKS` then the package's:

- trainer: `anima` (DiT + Qwen3-0.6B TE + Qwen-Image VAE) · `pe` (PE-Core-L14-336) ·
  `cjk` (vocab pack, on by default since v2)
- package: `tagger` · `tags` · `masking` · `ocr` · `grouping`

`HIDDEN_PACKS` is the seam for a package pack the trainer does not offer (not listed,
resolved or downloaded here). It is currently **empty** — keep it as the mechanism; don't
filter rows instead.

## Commands

```bash
make download-models        # first-run set: DiT, TE, VAE, PE, CJK vocab pack, tagger, tag DB
make download-list          # every row grouped by pack: installed / MISSING, repo, destination
make download-model ocr     # by pack, legacy alias, or row id
```

SAM3 (`masking`) and OCR are **opt-in** — not in the first-run set.
`make download-anima-variant ARGS=Anima-2.9B-preview-v1` fetches a depth variant.
