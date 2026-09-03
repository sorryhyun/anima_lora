"""Cache text-encoder (Qwen3) outputs.

Orchestration extracted from ``preprocess/cache_text_embeddings.py`` (see
``docs/proposal/tooling_architecture.md`` §A). The script keeps argparse + model
load + uncond staging; the caption-variant generation and the batched
tokenize→encode→(LLM-adapter)→save loop live here.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Collection
from pathlib import Path

import torch
from PIL import Image

from library.io.cache import TE_CACHE_SUFFIX, resolve_cache_path
from library.preprocess._dataset import PreprocessStats, walk_images
from library.preprocess._progress import ProgressFn

# generate_caption_variants + build_erasure_token_pool live in the torch-free
# caption_variants module so the caption-correction step (which materializes the
# variant sidecars before the encoder loads) and the GUI can reuse them. Re-export
# here for backward compatibility (existing callers import them off this module /
# the package façade).
from anime_tools.captions.variants import (  # noqa: F401
    build_erasure_token_pool,
    generate_caption_variants,
    read_variants_sidecar,
    variants_sidecar_path,
)

logger = logging.getLogger(__name__)


def _strip_no_artist_sentinel_from_caption(caption: str) -> str:
    from library.anima import training as anima_train_utils

    sentinel = anima_train_utils.NO_ARTIST_SENTINEL
    tags = [t.strip() for t in caption.split(",")]
    if sentinel not in tags:
        return caption
    return ", ".join(anima_train_utils.strip_no_artist_sentinel(tags))


def _encode_batch(
    captions: list[str],
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    llm_adapter,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Tokenize, encode through Qwen3, optionally run the LLM adapter. CPU tensors out."""
    tokens_and_masks = tokenize_strategy.tokenize(captions)
    with torch.no_grad():
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = (
            encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens_and_masks
            )
        )

        crossattn_emb = None
        if llm_adapter is not None:
            crossattn_emb = llm_adapter(
                source_hidden_states=prompt_embeds,
                target_input_ids=t5_input_ids.to(device, dtype=torch.long),
                target_attention_mask=t5_attn_mask.to(device),
                source_attention_mask=attn_mask,
            )
            crossattn_emb[~t5_attn_mask.to(device).bool()] = 0
            crossattn_emb = crossattn_emb.to(dtype=torch.bfloat16).cpu()

    return (
        prompt_embeds.to(dtype=torch.bfloat16).cpu(),
        attn_mask.to(dtype=torch.int32).cpu(),
        t5_input_ids.to(dtype=torch.long).cpu(),
        t5_attn_mask.to(dtype=torch.int32).cpu(),
        crossattn_emb,
    )


def _te_cache_path(image_path: Path, cache_dir: Path | None, image_dir: Path) -> Path:
    if cache_dir is None:
        return image_path.with_name(image_path.stem + TE_CACHE_SUFFIX)
    return Path(
        resolve_cache_path(
            str(image_path),
            TE_CACHE_SUFFIX,
            cache_dir=str(cache_dir),
            image_dir=str(image_dir),
        )
    )


def _cache_has_randomized(cache_path: Path) -> bool:
    """True iff an existing TE cache already carries the identity-randomized
    ``r``-family (``num_randomized`` marker). Reads only the safetensors header.

    Lets a re-run with ``--caption_tag_randomize_rate`` *upgrade in place* a cache
    written before randomization existed, instead of being skipped by the
    existence check (the writer otherwise never reopens a present cache)."""
    from safetensors import safe_open

    try:
        with safe_open(str(cache_path), framework="pt") as f:
            return "num_randomized" in f.keys()
    except Exception:
        # Unreadable/partial cache → treat as missing the family so it re-encodes.
        return False


def _cache_is_current(image_path: Path, cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    try:
        cache_mtime = cache_path.stat().st_mtime
    except OSError:
        return False
    # The cache must be newer than BOTH the caption and (if present) the variant
    # sidecar — the sidecar is the encoded source of truth, so an edit/regen to it
    # has to invalidate the cache just like a caption edit does.
    newest = 0.0
    for src in (image_path.with_suffix(".txt"), variants_sidecar_path(image_path)):
        if src.exists():
            try:
                newest = max(newest, src.stat().st_mtime)
            except OSError:
                return False
    if newest == 0.0:
        return True  # no caption and no sidecar → nothing to be stale against
    return cache_mtime >= newest


def _walk_te_candidates(
    data_dir: Path,
    *,
    recursive: bool,
    path_pattern: str | None,
    keep_stems: Collection[str] | None,
    keep_rel_stems: Collection[str] | None,
    min_pixels: int,
    verbose: bool,
) -> list[Path]:
    """Enumerate the images a TE cache pass would encode (caption-agnostic).

    Applies the same ``keep_stems`` + ``min_pixels`` filters as
    :func:`cache_text_embeddings`; an absent or empty ``.txt`` is *not* a
    filter (uncaptioned images are encoded with an empty caption). Shared by
    the encode loop and :func:`count_pending_text` so they agree on the set.
    """
    candidates = walk_images(data_dir, recursive=recursive, pattern=path_pattern)
    if keep_stems is not None:
        keep = frozenset(keep_stems)
        pre = len(candidates)
        candidates = [p for p in candidates if p.stem in keep]
        if verbose and len(candidates) != pre:
            print(
                f"Stem filter: keeping {len(candidates)}/{pre} captions "
                "(matched-subset only)."
            )

    if keep_rel_stems is not None:
        keep = frozenset(keep_rel_stems)
        pre = len(candidates)
        filtered: list[Path] = []
        for p in candidates:
            try:
                key = p.relative_to(data_dir).with_suffix("").as_posix()
            except ValueError:
                rel_dir = os.path.relpath(p.parent, data_dir)
                rel_dir = "" if rel_dir == "." else rel_dir
                key = (Path(rel_dir) / p.stem).as_posix() if rel_dir else p.stem
            if key in keep:
                filtered.append(p)
        candidates = filtered
        if verbose and len(candidates) != pre:
            print(
                f"Matched-image filter: keeping {len(candidates)}/{pre} captions "
                "(resized outputs only)."
            )

    # The per-image header open below exists only to mirror the resize-time
    # min_pixels drop. When a ``keep_*`` filter is active the candidate set is
    # already the resized/curated outputs — every survivor passed min_pixels at
    # resize — so re-opening each (large, original) source image just to re-derive
    # that fact is pure I/O waste. Skip it; the matched set is the authority.
    # (TE only needs the caption ``.txt``, never the image pixels.)
    already_filtered = keep_stems is not None or keep_rel_stems is not None
    check_pixels = min_pixels > 0 and not already_filtered

    kept: list[Path] = []
    skipped_small = 0
    for p in candidates:
        if check_pixels:
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception as e:
                logger.warning("could not read %s: %s", p.name, e)
                continue
            if w * h < min_pixels:
                skipped_small += 1
                continue
        kept.append(p)

    if skipped_small and verbose:
        print(
            f"Skipping {skipped_small} images below {min_pixels:,} pixels "
            f"({min_pixels / 1e6:.2f}MP) -- same filter as the resize stage."
        )
    return kept


def count_pending_text(
    data_dir: Path,
    *,
    cache_dir: Path | None = None,
    recursive: bool = False,
    path_pattern: str | None = None,
    keep_stems: Collection[str] | None = None,
    keep_rel_stems: Collection[str] | None = None,
    min_pixels: int = 500_000,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Return ``(pending, total)`` TE caches **without loading the encoder**.

    ``pending`` is the number of candidate images whose
    ``{stem}_anima_te.safetensors`` isn't on disk; ``total`` is every candidate
    (post ``keep_stems`` / ``min_pixels`` filtering). Mirrors the per-batch skip
    in :func:`cache_text_embeddings`, so the entry point can skip the (slow)
    Qwen3 + LLM-adapter load when ``pending == 0``. With ``overwrite`` every
    candidate counts as pending (the encoder always loads)."""
    candidates = _walk_te_candidates(
        data_dir,
        recursive=recursive,
        path_pattern=path_pattern,
        keep_stems=keep_stems,
        keep_rel_stems=keep_rel_stems,
        min_pixels=min_pixels,
        verbose=False,
    )
    if overwrite:
        return len(candidates), len(candidates)
    pending = sum(
        1
        for p in candidates
        if not _cache_is_current(p, _te_cache_path(p, cache_dir, data_dir))
    )
    return pending, len(candidates)


def cache_text_embeddings(
    data_dir: Path,
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
    *,
    llm_adapter=None,
    device: torch.device,
    cache_dir: Path | None = None,
    recursive: bool = False,
    path_pattern: str | None = None,
    keep_stems: Collection[str] | None = None,
    keep_rel_stems: Collection[str] | None = None,
    batch_size: int = 16,
    caption_shuffle_variants: int = 0,
    caption_tag_dropout_rate: float = 0.0,
    caption_tag_randomize_rate: float = 0.0,
    caption_transform: Callable[[str], str] | None = None,
    caption_protect_fn: Callable[[str], bool] | None = None,
    min_pixels: int = 500_000,
    overwrite: bool = False,
    verbose: bool = True,
    progress: ProgressFn | None = None,
) -> PreprocessStats:
    """Encode ``.txt`` captions for every image under ``data_dir``.

    Images with no ``.txt`` sidecar (or an empty one) are encoded with an empty
    caption ``""`` rather than dropped, so the cached TE set mirrors the
    training dataset — dreambooth loads uncaptioned images with an empty caption
    and the trainer's cache-completeness probe expects a TE cache for each.

    Strategies + encoder + (optional) ``llm_adapter`` are supplied loaded + on
    ``device``. Images below ``min_pixels`` are skipped (mirrors the resize
    filter). With ``caption_shuffle_variants > 0`` each cache holds N variants
    (v0 pristine, v1..v{N-1} shuffled + optionally tag-dropped + optionally
    identity-randomized via ``caption_tag_randomize_rate``). Returns counts;
    pass ``progress`` for a per-image bar.

    ``caption_transform`` (when given) is applied to each raw caption before
    tokenization / variant generation — used by task-specific re-encodes such
    as colorization's color-only caption filter. It runs *before* shuffle so
    the kept tags are what gets shuffled/dropped.

    ``caption_protect_fn`` (when given) is forwarded to
    :func:`generate_caption_variants` to exempt matching tags from tag-dropout
    (e.g. colorize copyright tags). No-op unless ``caption_shuffle_variants > 0``.

    ``keep_stems`` (when given) restricts encoding to images whose filename stem
    is in the set — the matched subset the VAE/cond stage already materialized,
    so the TE cache mirrors that set rather than re-encoding the whole caption
    master.

    ``keep_rel_stems`` is the path-safe variant for nested datasets. Each key is
    the image path relative to ``data_dir`` without its extension.
    """
    candidates = _walk_te_candidates(
        data_dir,
        recursive=recursive,
        path_pattern=path_pattern,
        keep_stems=keep_stems,
        keep_rel_stems=keep_rel_stems,
        min_pixels=min_pixels,
        verbose=verbose,
    )

    entries: list[tuple[Path, str]] = []
    for p in candidates:
        caption_path = p.with_suffix(".txt")
        # Missing/empty caption → encode "" (not drop): dreambooth loads uncaptioned
        # images with an empty caption and the cache-completeness probe expects a TE
        # cache for each, so dropping here would leave the cache incomplete.
        if caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip().split("\n")[0]
        else:
            caption = ""
        if caption_transform is not None:
            caption = caption_transform(caption)
        caption = _strip_no_artist_sentinel_from_caption(caption)
        entries.append((p, caption))

    stats = PreprocessStats(seen=len(entries))
    caption_dropout_rate = torch.tensor(0.0, dtype=torch.float32)
    n_variants = caption_shuffle_variants
    tag_dropout_rate = float(caption_tag_dropout_rate)
    tag_randomize_rate = float(caption_tag_randomize_rate)
    # The identity-randomized r-family rides alongside v0..v{N-1} (sharing the
    # pristine v0 as anchor), so one cache serves both the baseline-shuffle and
    # the lexinvariant arm — the consumer picks the family via
    # use_randomized_caption_variants. Needs >=2 variants (r1..r{N-1}).
    want_randomized = tag_randomize_rate > 0.0 and n_variants >= 2
    n_rand = (n_variants - 1) if want_randomized else 0
    # Dual-single erasure pool, built once: words that are exactly one token in
    # *both* Qwen3 and T5, minus this dataset's real tags (so a filler is never a
    # genuine tag). Built from the loaded strategy's two tokenizers.
    erasure_pool = None
    if want_randomized:
        real_tags = {
            t.strip().lower() for _, cap in entries for t in cap.split(",") if t.strip()
        }
        erasure_pool = build_erasure_token_pool(
            getattr(tokenize_strategy, "qwen3_tokenizer", None),
            getattr(tokenize_strategy, "t5_tokenizer", None),
            exclude=real_tags,
        )
        if not erasure_pool:
            raise ValueError(
                "Identity-randomize requested but the erasure-token pool is empty "
                "(strategy lacks qwen3_tokenizer/t5_tokenizer or no qualifying "
                "tokens) -- cannot erase tag identity without it."
            )
        if verbose:
            print(
                f"Identity-randomize: erasure pool of {len(erasure_pool)} "
                "dual-single tokens "
                f"(excluded {len(real_tags)} real tags)"
            )

    from safetensors.torch import save_file

    def _rows_for(img_path: Path, caption: str) -> list[tuple[str, str]]:
        """Ordered ``(label, text)`` variants to encode for one image.

        The ``{stem}.variants.txt`` sidecar — written upstream by the caption
        step — is the **source of truth**: when present we encode exactly its
        lines so the visible text matches what trains. Without a sidecar we fall
        back to in-process generation (the colorize ``caption_transform`` path,
        and any flow that skipped the caption step). A lone ``("", caption)`` row
        is the legacy single-caption (flat-key) layout.
        """
        if caption_transform is None:
            sidecar = variants_sidecar_path(img_path)
            if sidecar.exists():
                try:
                    rows = read_variants_sidecar(sidecar)
                except OSError:
                    rows = []
                if rows:
                    # Sentinel is already stripped by the generator; re-strip
                    # defensively against a hand-edited sidecar.
                    return [
                        (label, _strip_no_artist_sentinel_from_caption(text))
                        for label, text in rows
                    ]
        if n_variants > 0:
            v_list = generate_caption_variants(
                caption, n_variants, tag_dropout_rate, caption_protect_fn
            )
            rows = [(f"v{i}", text) for i, text in enumerate(v_list)]
            if n_rand:
                r_list = generate_caption_variants(
                    caption,
                    n_variants,
                    tag_dropout_rate,
                    caption_protect_fn,
                    tag_randomize_rate=tag_randomize_rate,
                    erasure_pool=erasure_pool,
                )
                rows += [(f"r{j}", text) for j, text in enumerate(r_list[1:], start=1)]
            return rows
        return [("", caption)]

    if progress is not None:
        progress(0, total=len(entries))

    for batch_start in range(0, len(entries), batch_size):
        batch = entries[batch_start : batch_start + batch_size]

        to_encode: list[tuple[Path, str, Path]] = []
        for img_path, caption in batch:
            cache_path = _te_cache_path(img_path, cache_dir, data_dir)
            # Re-encode an existing cache only to add a newly-requested r-family
            # (in-place upgrade); otherwise the existence check skips it.
            # ``overwrite`` forces a full re-encode (e.g. after changing the
            # randomize rate / variant count, which the existence check can't see).
            if (
                not overwrite
                and _cache_is_current(img_path, cache_path)
                and not (want_randomized and not _cache_has_randomized(cache_path))
            ):
                stats.skipped += 1
                if progress is not None:
                    progress(1, detail=f"skip {img_path.name}")
            else:
                to_encode.append((img_path, caption, cache_path))

        if not to_encode:
            continue

        # Build each image's variant rows, then flatten the whole batch into one
        # encode call. Row counts may differ per image (a sidecar vs the
        # fallback, mixed within a batch), so we track a running flat offset
        # instead of a uniform block stride.
        per_image_rows = [
            _rows_for(img_path, caption) for img_path, caption, _ in to_encode
        ]
        all_captions = [text for rows in per_image_rows for _, text in rows]
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
            _encode_batch(
                all_captions,
                tokenize_strategy,
                encoding_strategy,
                text_encoder,
                llm_adapter,
                device,
            )
        )

        flat = 0
        for (img_path, _, cache_path), rows in zip(to_encode, per_image_rows):
            labels = [label for label, _ in rows]
            if labels == [""]:
                # Legacy single-caption layout: flat (unsuffixed) keys.
                save_dict = {
                    "t5_attn_mask": t5_attn_mask[flat],
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if crossattn_emb is not None:
                    # Adapter-output cache: only crossattn_emb is consumed at
                    # train time (+ t5_attn_mask for postfix). The Qwen
                    # prompt_embeds / attn_mask / t5_input_ids are unused
                    # downstream (see library/training/forward/text_conds.py),
                    # so dropping them ~halves the file. The pad length is
                    # unchanged (512), so crossattn values are bit-identical.
                    save_dict["crossattn_emb"] = crossattn_emb[flat]
                else:
                    save_dict["prompt_embeds"] = prompt_embeds[flat]
                    save_dict["attn_mask"] = attn_mask[flat]
                    save_dict["t5_input_ids"] = t5_input_ids[flat]
                detail = img_path.name
            else:
                n_v = sum(1 for label in labels if label.startswith("v"))
                n_r = sum(1 for label in labels if label.startswith("r"))
                save_dict = {
                    "num_variants": torch.tensor(n_v, dtype=torch.int64),
                    # Marker: v0 is pristine; loaders switch on weighted 20%/80%
                    # sampling between v0 and v1..v{N-1}.
                    "v0_intact": torch.tensor(1, dtype=torch.int8),
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if n_r:
                    save_dict["num_randomized"] = torch.tensor(n_r, dtype=torch.int64)
                for off, (label, _) in enumerate(rows):
                    k = flat + off
                    save_dict[f"t5_attn_mask_{label}"] = t5_attn_mask[k]
                    if crossattn_emb is not None:
                        # Adapter-output cache: prune the unused Qwen
                        # prompt_embeds / attn_mask / t5_input_ids (~half the
                        # file). Only crossattn_emb (+ t5_attn_mask for postfix)
                        # is read at train time — see
                        # library/training/forward/text_conds.py. 512-pad kept,
                        # so crossattn is bit-identical to the legacy layout.
                        save_dict[f"crossattn_emb_{label}"] = crossattn_emb[k]
                    else:
                        save_dict[f"prompt_embeds_{label}"] = prompt_embeds[k]
                        save_dict[f"attn_mask_{label}"] = attn_mask[k]
                        save_dict[f"t5_input_ids_{label}"] = t5_input_ids[k]
                detail = f"{img_path.name} ({n_v}v" + (f"+{n_r}r)" if n_r else ")")

            save_file(save_dict, str(cache_path))
            stats.written += 1
            if progress is not None:
                progress(1, detail=detail)
            flat += len(rows)

    return stats
