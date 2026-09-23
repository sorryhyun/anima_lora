"""Experimental inference entry-points (exp-test-* commands).

Covers the unstable methods kept under ``make exp-*``: soft tokens, BYG, plus
the DirectEdit probes. Reference-image variants accept
REF_IMAGE env or first positional arg, copy the ref alongside the generated
output.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from scripts.tasks._common import (
    INFERENCE_BASE,
    ROOT,
    _random_ref_image,
    _REF_IMAGE_EXTS,
    latest_output,
    run,
)

_TE_SUFFIX = "_anima_te.safetensors"


def _te_cache_candidates(ref_image: str | os.PathLike) -> list[Path]:
    """TE cache locations to probe for a resized reference image.

    Preprocessing writes caches under ``post_image_dataset/lora/`` mirroring
    the per-artist subdir layout of ``post_image_dataset/resized/`` (see
    ``resolve_cache_path`` with ``image_dir``). So for
    ``resized/mejikara_scene/10083096.png`` the cache lands at
    ``lora/mejikara_scene/10083096_anima_te.safetensors`` — not flat under
    ``lora/``. Probe, in order: the nested mirror, the flat ``lora/`` root,
    and the legacy sidecar next to the image.
    """
    from library.io.cache import resolve_cache_path  # noqa: PLC0415

    ref = Path(ref_image)
    stem = ref.stem
    resized_root = ROOT / "post_image_dataset" / "resized"
    lora_root = ROOT / "post_image_dataset" / "lora"
    nested = Path(
        resolve_cache_path(
            str(ref),
            _TE_SUFFIX,
            cache_dir=str(lora_root),
            image_dir=str(resized_root),
        )
    )
    # Deduplicate while preserving order (nested == flat when no subdir).
    candidates = [
        nested,
        lora_root / f"{stem}{_TE_SUFFIX}",
        ref.parent / f"{stem}{_TE_SUFFIX}",
    ]
    seen: set[Path] = set()
    return [p for p in candidates if not (p in seen or seen.add(p))]


def _resolve_te_cache(ref_image: str | os.PathLike) -> Path | None:
    """First existing TE cache for ``ref_image``, or ``None``."""
    return next((p for p in _te_cache_candidates(ref_image) if p.is_file()), None)


def _resolve_ref_image(ref_image: str) -> str:
    """Resolve a possibly-partial ``REF_IMAGE`` to a real file under ``resized/``.

    Accepts a path that already exists as given, or a partial path relative to
    ``post_image_dataset/resized/`` with or without an extension (e.g.
    ``sushispin/10186995`` → ``.../resized/sushispin/10186995.png``). Returning
    the full nested path is what lets ``_te_cache_candidates`` /
    ``resolve_cache_path`` mirror the per-artist subdir into the cache lookup —
    a bare ``artist/stem`` makes ``relpath`` escape the resized root and fall
    back to the (wrong) flat ``lora/stem`` candidate. Returns ``ref_image``
    untouched when nothing matches, so downstream "not found" messaging fires.
    """
    if Path(ref_image).is_file():
        return ref_image
    resized_root = ROOT / "post_image_dataset" / "resized"
    base = resized_root / ref_image
    if base.is_file():
        return str(base)
    for ext in _REF_IMAGE_EXTS:
        for cand in (Path(f"{base}{ext}"), base.with_suffix(ext)):
            if cand.is_file():
                return str(cand)
    return ref_image


def cmd_test_soft(extra):
    """Inference with latest soft_tokens weight (SoftREPA-style per-layer × per-t bank).

    Resolves the newest ``anima_soft_tokens*.safetensors`` under ``output/ckpt/``
    and passes it via ``--soft_tokens_weight``. The network is built in
    ``library/inference/generation.py``, ``apply_to`` monkey-patches the first
    ``n_layers`` ``Block.forward``s, and ``append_postfix(..., timesteps=t)``
    fires per CFG branch inside the denoising loop (mirrored in the Spectrum
    runner). Composes freely with ``--spectrum``; cached spectrum steps skip
    blocks so soft_tokens silently no-ops on those steps.
    """
    run(
        [
            *INFERENCE_BASE,
            "--soft_tokens_weight",
            str(latest_output("anima_soft_tokens")),
            *extra,
        ]
    )


def cmd_test_directedit(extra):
    """DirectEdit on a random source image, seeded by the Anima Tagger.

    Pipeline:
      1. Pick source image (REF_IMAGE env, first positional arg, or random
         from ``post_image_dataset/resized/``).
      2. Run the Anima Tagger on the source -> ``--prompt_src`` caption
         (checkpoint auto-fetched on first use).
      3. Pass the edit instruction as ``--edit_instruction`` (PROMPT env, else
         a ``--prompt`` extra arg, else a "double peace" default); edit.py's
         dispatcher derives ψ_tar from it.
      4. Call ``scripts/edit.py`` (DirectEdit invert + edit) using the same
         DiT/VAE/TE trio as the other inference targets.
      5. Save under ``output/tests/directedit/`` and copy the source image
         alongside as ``<name>_src.png``.

    Examples:
      make exp-test-directedit PROMPT='double peace'
      REF_IMAGE=foo.png make exp-test-directedit PROMPT='glasses'
      python tasks.py exp-test-directedit foo.png --prompt 'smile'
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-directedit [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-directedit\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)
    ref_image = _resolve_ref_image(ref_image)

    # Edit instruction: PROMPT env wins, then a --prompt flag in extra, then default.
    edit_prompt = os.environ.get("PROMPT", "").strip()
    cleaned_extra: list[str] = []
    skip_next = False
    for j, tok in enumerate(extra):
        if skip_next:
            skip_next = False
            continue
        if tok == "--prompt" and j + 1 < len(extra):
            if not edit_prompt:
                edit_prompt = extra[j + 1]
            skip_next = True
            continue
        cleaned_extra.append(tok)
    extra = cleaned_extra
    if not edit_prompt:
        edit_prompt = "double peace, v v. She is showing double peace"

    from PIL import Image  # noqa: PLC0415

    from anime_tools.tagger.tagger import (  # noqa: PLC0415
        DEFAULT_TAGGER_DIR,
        AnimaTagger,
        ensure_tagger_checkpoint,
    )

    # Auto-fetch the published checkpoint (sorryhyun/anima-tagger) when absent —
    # same path every other tagger entry point takes (autotag.py / autotag_server.py).
    ckpt_dir = ensure_tagger_checkpoint(ROOT / DEFAULT_TAGGER_DIR)
    print(f"  > tagging source: {ref_image}")
    tagger = AnimaTagger(ckpt_dir=ckpt_dir)

    src_caption = tagger.predict_caption(Image.open(ref_image))
    if not src_caption:
        print(
            "  ! tagger produced no tags above threshold; using empty source "
            "prompt — DirectEdit reconstruction will be weaker than usual.",
            file=sys.stderr,
        )
    print(
        f"  > src caption: {src_caption[:120]}{'...' if len(src_caption) > 120 else ''}"
    )

    # Hand the edit instruction to edit.py via --edit_instruction so its dispatcher
    # runs in-process — running it in this wrapper would load Qwen3 a second time.
    save_dir = ROOT / "output" / "tests" / "directedit"
    save_dir.mkdir(parents=True, exist_ok=True)

    base_iter = iter(INFERENCE_BASE)
    py = next(base_iter)
    next(base_iter)  # drop "inference.py"
    leftover_base = list(base_iter)
    args = [py, "scripts/edit.py", *_filter_inference_base_for_edit(leftover_base)]
    args += [
        "--image",
        str(ref_image),
        "--prompt_src",
        src_caption,
        "--edit_instruction",
        edit_prompt,
        "--save_path",
        str(save_dir),
    ]
    args += list(extra)
    run(args)

    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_src.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        src_dst = pngs[0].with_name(pngs[0].stem + "_src.png")
        shutil.copy(ref_image, src_dst)
        print(f"  > Source pasted: {src_dst}")


def cmd_test_directedit_dry(extra):
    """DirectEdit functional sanity check using preprocessed cross-emb variants.

    Bypasses the tagger and the text encoder. Auto-resolves the source image's
    `_anima_te.safetensors` cache (the file `cache_text_embeddings.py` writes
    — same format the trainer consumes) and runs one invert + edit pass per
    stored variant with ψ_tar == ψ_src. With `--caption_shuffle_variants N`
    caches, this sweeps v0 (pristine) + v1..v{N-1} (tag-shuffled). Each pass
    should reconstruct the source; divergence flags numeric drift in
    invert/edit_forward against that variant's cross-emb representation.

    Add ``--fm_score`` to also rank each variant's ψ_src by its intrinsic
    flow-matching error (AGSM-style reward; lower = more on-manifold) and
    correlate that ranking against each variant's reconstruction MSE — a
    quantitative replacement for eyeballing the side-by-side divergence.

    Examples:
      make exp-test-directedit-dry
      REF_IMAGE=foo.png make exp-test-directedit-dry
      python tasks.py exp-test-directedit-dry foo.png --seed 7
      python tasks.py exp-test-directedit-dry foo.png --fm_score
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-directedit-dry [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-directedit-dry\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)
    ref_image = _resolve_ref_image(ref_image)

    candidates = _te_cache_candidates(ref_image)
    cache_path = _resolve_te_cache(ref_image)
    if cache_path is None:
        looked = "\n".join(f"      {p}" for p in candidates)
        print(
            f"  ! No TE cache found for {ref_image}.\n"
            f"    Looked in:\n{looked}\n"
            "    Run `make preprocess-te` first (with --caption_shuffle_variants N "
            "to get a multi-variant cache).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  > TE cache: {cache_path}")

    save_dir = ROOT / "output" / "tests" / "directedit_dry"
    save_dir.mkdir(parents=True, exist_ok=True)

    base_iter = iter(INFERENCE_BASE)
    py = next(base_iter)
    next(base_iter)  # drop "inference.py"
    leftover_base = list(base_iter)
    args = [py, "scripts/edit.py", *_filter_inference_base_for_edit(leftover_base)]
    args += [
        "--image",
        str(ref_image),
        "--cached_embed",
        str(cache_path),
        "--save_path",
        str(save_dir),
    ]
    args += list(extra)
    run(args)

    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_src.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        src_dst = pngs[0].with_name(pngs[0].stem + "_src.png")
        shutil.copy(ref_image, src_dst)
        print(f"  > Source pasted: {src_dst}")


def _filter_inference_base_for_edit(args: list[str]) -> list[str]:
    """Drop INFERENCE_BASE flags that ``scripts/edit.py`` doesn't accept.

    INFERENCE_BASE bundles plenty of generation-only flags (--prompt, --seed,
    --image_size, --infer_steps, --sampler, etc.) that overlap with or
    conflict with edit.py's own. Keep only the model/path flags we actually
    want to forward; let edit.py supply its own defaults for the rest.
    """
    keep_flags = {
        "--dit",
        "--text_encoder",
        "--vae",
        "--vae_chunk_size",
        "--attn_mode",
    }
    boolean_flags = {"--vae_disable_cache"}
    out: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in keep_flags and i + 1 < len(args):
            out.extend([tok, args[i + 1]])
            i += 2
        elif tok in boolean_flags:
            out.append(tok)
            i += 1
        else:
            i += 1
    return out
