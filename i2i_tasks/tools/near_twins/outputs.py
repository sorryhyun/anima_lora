#!/usr/bin/env python3
"""near_twins outputs — pair-tree export + dataset blueprint.

Pure I/O over ``engine.PairRecord``s: the ``_tags`` / ``_no_tags`` pair tree
(``export_pairs``) and the EasyControl dataset blueprint (``write_dataset_config``).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .engine import (
    Member,
    PairRecord,
)

WEBP_LOSSLESS = {"lossless": True, "quality": 100, "method": 4}


def caption_text(txt_path: Path) -> str:
    return (
        txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if txt_path.is_file()
        else ""
    )


def _materialize_one(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def export_pairs(
    pairs: list[PairRecord], export_dir: Path, copy: bool, emit_mask: bool
) -> int:
    """Materialize the ``_tags`` / ``_no_tags`` pair tree (the training-shaped output).

    Symlinks keep the source extension (``_tags.webp``); ``--copy`` re-saves as
    lossless WebP. The EasyControl loader globs png+webp, so stem-matching is unaffected.
    """
    written = 0
    for p in pairs:
        holder, clean = p.holder_member(), p.clean_member()
        adir = export_dir / p.artist
        for member, side in ((holder, "_tags"), (clean, "_no_tags")):
            stem_ext = ".webp" if copy else member.image_path.suffix
            img_dst = adir / f"{p.pair_id}{side}{stem_ext}"
            if copy:
                with Image.open(member.image_path) as im:
                    img_dst.parent.mkdir(parents=True, exist_ok=True)
                    im.convert("RGB").save(img_dst, **WEBP_LOSSLESS)
            else:
                _materialize_one(member.image_path, img_dst, copy=False)
            txt_dst = adir / f"{p.pair_id}{side}.txt"
            txt_dst.write_text(caption_text(member.txt_path), encoding="utf-8")
        if emit_mask:
            _write_mask(p, adir / f"{p.pair_id}_mask.png")
        written += 1
    return written


def export_identity_members(members: list[Member], export_dir: Path, copy: bool) -> int:
    """Materialize identity (cond==target) pairs into the ``_tags`` / ``_no_tags`` tree.

    Each clean single is written as BOTH sides of a degenerate pair from the same
    source image — ``id_{stem}_tags`` (the condition) and ``id_{stem}_no_tags`` (the
    target) — so the preprocess cond-pairing symlinks an identical latent as the
    cond and the adapter sees a no-op edit. The ``id_`` prefix keeps these clear of
    the ``{a}-{b}`` mined-pair ids; both members share one image so they always
    land in the same bucket (the cond-pairing requires a same-bucket twin).
    Captions mirror to both sides (sanitize blanks them via caption_dropout anyway).
    """
    written = 0
    for m in members:
        adir = export_dir / m.artist
        pid = f"id_{m.stem}"
        ext = ".webp" if copy else m.image_path.suffix
        for side in ("_tags", "_no_tags"):
            img_dst = adir / f"{pid}{side}{ext}"
            if copy:
                with Image.open(m.image_path) as im:
                    img_dst.parent.mkdir(parents=True, exist_ok=True)
                    im.convert("RGB").save(img_dst, **WEBP_LOSSLESS)
            else:
                _materialize_one(m.image_path, img_dst, copy=False)
            txt_dst = adir / f"{pid}{side}.txt"
            txt_dst.write_text(caption_text(m.txt_path), encoding="utf-8")
        written += 1
    return written


PAIR_MANIFEST = "pairs.jsonl"
_IMAGE_SUFFIXES = (".png", ".webp", ".jpg", ".jpeg")


def write_pair_manifest(export_dir: Path) -> int:
    """``<export_dir>/pairs.jsonl``: one ``{stem, target, cond, caption}`` row per
    ``{id}_no_tags`` / ``{id}_tags`` pair in the tree (identity pairs included),
    paths relative to ``export_dir``. The model-agnostic i2i pair list a model
    line's cache reads (``library.moss.cache.manifest_pairs``)."""
    rows = []
    for target in sorted(export_dir.rglob("*_no_tags.*")):
        if target.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        pid = target.name[: -len("_no_tags" + target.suffix)]
        cond = next(
            (
                c
                for ext in _IMAGE_SUFFIXES
                if (c := target.with_name(f"{pid}_tags{ext}")).exists()
            ),
            None,
        )
        caption = target.with_suffix(".txt")
        if cond is None or not caption.exists():
            continue
        rows.append(
            {
                "stem": pid,
                "target": str(target.relative_to(export_dir)),
                "cond": str(cond.relative_to(export_dir)),
                "caption": str(caption.relative_to(export_dir)),
            }
        )
    (export_dir / PAIR_MANIFEST).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return len(rows)


def _write_mask(p: PairRecord, out: Path) -> None:
    G = p.match.G
    grid = np.zeros((G, G), dtype=np.uint8)
    for c in p.match.diff_cells:
        grid[c // G, c % G] = 255
    with Image.open(p.holder_member().image_path) as im:
        w, h = im.size
    Image.fromarray(grid, mode="L").resize((w, h), Image.NEAREST).save(out)


# The miner rewrites everything BELOW this sentinel (the output dataset
# blueprint) on every run, and leaves everything above it — the user's
# [preprocess] / [training] knobs + comments — untouched. The run knobs
# (`name` + [staging]) live in the i2i task file read via --config.
_BLUEPRINT_SENTINEL = (
    "# === generated dataset blueprint (rewritten by the miner; do not edit below) ==="
)

_DEFAULT_EASYCONTROL_HEADER = """\
# EasyControl descriptor for a near-twin pair tree, scaffolded by the miner. The
# task's `name` and [staging] knobs live in i2i_tasks/configs/<task>.toml; the
# pipeline lives under post_image_dataset/i2i/<name>/{staging,resized,cache,cond}
# and checkpoints save as anima_easycontrol_<name>.
#   make i2i-stage TASK=<task>                       # mine pair tree
#   make easycontrol-preprocess EASYADAPTER=<task>   # [preprocess] → VAE/TE caches
# CLI flags override these. Paths support {CAPTION_CORPUS_DIR}/$VARS/~ expansion.

# Resize + VAE/TE caching pass over the mined pair tree (read by
# `make easycontrol-preprocess EASYADAPTER=<task>`). Native-res staging is
# resized into constant-token buckets under resized_dir (the training image_dir),
# then VAE/TE-encoded into cache_dir. The image_dir/resized_dir/cache_dir/cond_dir
# default to i2i/<name>/{staging,resized,cache,cond} (`name` in the task file) —
# only add a path key here to point one elsewhere.
[preprocess]
target_res = [1024]  # bucket tiers (512 768 1024 1280 1536); must match training
recursive = true
vae = 'models/vae/qwen_image_vae.safetensors'
batch_size = 4
chunk_size = 64
qwen3 = 'models/text_encoders/qwen_3_06b_base.safetensors'
dit = 'models/diffusion_models/anima-base-v1.0.safetensors'
caption_shuffle_variants = 4
caption_tag_dropout_rate = 0.1

# Training overrides for `make easycontrol EASYADAPTER=<task>` (optional).
# Each key is folded in as a `--key value` CLI override on top of
# configs/easycontrol/easycontrol.toml; `--dataset_config` is injected automatically.
# output_name defaults to anima_easycontrol_<name>; omit the table to train with
# the plain easycontrol method defaults.
[training]
learning_rate = 3e-5
max_train_epochs = 8
# Removal task: the _tags reference is the structural source of truth, so keep
# the cond clean (no training-time perturbation) for faithful reconstruction.
easycontrol_cond_noise_max = 0.0
"""


def _blueprint_text(export_dir: Path) -> str:
    # Subset paths are written as `{name}` placeholders (not the resolved slug):
    # `make easycontrol EASYADAPTER=<task>` interpolates `{name}` from the
    # task file's `name` key when it generates the dataset-config sidecar,
    # so changing `name` reroutes everything without touching this tail.
    # Training reads the bucket-resized tree (the preprocess pass resizes the
    # native-res `staging/` export into `resized/` before VAE-encoding), so the
    # subset image_dir points at the resized sibling, not the export dir itself.
    return f"""{_BLUEPRINT_SENTINEL}
# Near-twin pair tree wired as an EasyControl *text-removal* control task. The
# mined staging tree is resized into constant-token buckets under `resized/`,
# then VAE/TE-encoded into `cache/` (nested <artist>/ mirror of the resized tree).
# Each accepted pair is a `{{id}}_tags` / `{{id}}_no_tags` couple, the `_tags` side
# holding the discriminator attribute (speech bubbles / text).
#
# Roles (EasyControl: cache_dir = denoising target, cond_cache_dir = condition):
#   target = the clean `_no_tags` member (+ its caption) — what the model generates
#   cond   = the paired `_tags` latent (the text-bearing reference fed via set_cond),
#            symlinked into `cond/` under the `_no_tags` stem by the preprocess step.
# So the adapter learns: given a text-bearing panel as the reference, regenerate
# the clean version. `path_pattern` keeps only the `_no_tags` members as targets;
# the `_tags` members enter solely as the condition via cond_cache_dir.
# `{{name}}` below interpolates from the i2i task file's `name` at train time.

[general]
caption_extension = '.txt'

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = 'post_image_dataset/i2i/{{name}}/resized'
  cache_dir = 'post_image_dataset/i2i/{{name}}/cache'        # denoising target: the clean _no_tags members
  cond_cache_dir = 'post_image_dataset/i2i/{{name}}/cond'    # condition: the paired _tags latent (keyed by the _no_tags stem)
  path_pattern = '*_no_tags.*'     # targets = clean members only (the _tags twin rides in as the cond)
  recursive = true                 # tree is nested <artist>/<pair_id>_{{tags,no_tags}}; caches mirror the nesting
  flip_aug = false                 # latents can't be flipped post-hoc; the cond cache has no flipped variant
  num_repeats = 1
"""


# Blueprint section headers ([general] / [[datasets]] / [[datasets.subsets]]).
# The head only carries [preprocess] / [training], so the first of these marks
# where the blueprint begins when the sentinel comment has been edited away.
_BLUEPRINT_HEADER_RE = re.compile(r"^\s*\[\[?(?:general|datasets)\b")


def _strip_blueprint(existing: str) -> str:
    """Return the user-owned head of ``existing``, dropping any prior blueprint.

    Preferred boundary is the sentinel comment. If it's missing (the user edited
    the head and lost the sentinel line), fall back to the first blueprint
    section header and rewind past the contiguous comment/blank block that
    introduces it — so the blueprint's own header comments don't accumulate
    across runs. Without this fallback a missing sentinel makes ``split`` keep
    the whole file (blueprint included) as head, and each run appends another
    blueprint copy.
    """
    if _BLUEPRINT_SENTINEL in existing:
        return existing.split(_BLUEPRINT_SENTINEL, 1)[0].rstrip()
    lines = existing.splitlines()
    for i, line in enumerate(lines):
        if _BLUEPRINT_HEADER_RE.match(line):
            j = i
            while j > 0 and (
                lines[j - 1].lstrip().startswith("#") or not lines[j - 1].strip()
            ):
                j -= 1
            return "\n".join(lines[:j]).rstrip()
    return existing.rstrip()


def write_dataset_config(export_dir: Path, config_path: Path) -> None:
    """Rewrite the blueprint tail of ``config_path``, preserving the head tables.

    First creation scaffolds a ``[preprocess]`` + ``[training]`` template; re-runs
    keep everything above the sentinel (the user's edited run knobs) verbatim.
    """
    blueprint = _blueprint_text(export_dir)
    if config_path.is_file():
        head = _strip_blueprint(config_path.read_text(encoding="utf-8"))
        content = f"{head}\n\n{blueprint}" if head else blueprint
    else:
        content = f"{_DEFAULT_EASYCONTROL_HEADER}\n{blueprint}"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")
