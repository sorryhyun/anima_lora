"""Cache text-encoder (Qwen3) outputs.

Orchestration extracted from ``preprocess/cache_text_embeddings.py`` (see
``docs/proposal/tooling_architecture.md`` §A). The script keeps argparse + model
load + uncond staging; the caption-variant generation and the batched
tokenize→encode→(LLM-adapter)→save loop live here.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Callable, Collection
from pathlib import Path

import torch
from PIL import Image

from library.io.cache import TE_CACHE_SUFFIX, resolve_cache_path
from library.preprocess._dataset import PreprocessStats, walk_images
from library.preprocess._progress import ProgressFn

logger = logging.getLogger(__name__)


def build_erasure_token_pool(
    qwen3_tokenizer,
    t5_tokenizer,
    *,
    exclude: Collection[str] | None = None,
    min_len: int = 4,
    max_len: int = 9,
) -> list[str]:
    """Pool of tokens used to *erase a tag's identity* while keeping its slot
    (lexinvariant tag regularization, arXiv:2305.16349).

    A randomized slot is filled by a *different* pool token each draw, so the
    literal symbol stays unreliable — pushing the adapter to ground the concept
    in surrounding structure (image content + co-occurring tags) rather than the
    exact trigger vector (role-based, not vector-based, binding). Per the cited
    method the filler is a *real* vocab token, not gibberish: unreliability comes
    from the per-draw variation, not from the symbol being intrinsically
    meaningless. Distinct from tag-dropout (which removes the slot).

    **Dual-single** is the load-bearing constraint. Anima tokenizes captions
    twice — Qwen3 for the text encoder (``prompt_embeds``) *and* T5 for the LLM
    adapter's target ids (``crossattn_emb``). A filler must therefore be exactly
    one token in *both* vocabularies; otherwise it shreds into junk subword
    pieces on whichever side fragments it (random ASCII fragments in Qwen3; rare
    foreign tokens are clean in Qwen3 but explode in T5's English-centric
    sentencepiece). We pick lowercase ascii alphabetic words and keep only those
    that survive as a single token in both. ``exclude`` (the dataset's real tags)
    drops any word that is itself a genuine tag, so a filler is never mistakable
    for a true concept. Returns bare words (no leading space).

    Selection keys off the ``Ġ`` (leading-space) Qwen3 vocab form — exact for
    this token class (verified: heuristic == round-trip), so no Qwen3 re-encode
    is needed; only the T5 single-token property is checked at runtime.
    """
    excl = {t.lower() for t in exclude} if exclude else set()
    try:
        vocab = qwen3_tokenizer.get_vocab()
    except Exception:
        return []
    # Qwen3-single candidates (cheap, pure vocab): leading-space lowercase ascii.
    candidates = sorted(
        sym[1:]
        for sym in vocab
        if sym.startswith("Ġ")
        and (core := sym[1:]).isascii()
        and core.isalpha()
        and core.islower()
        and min_len <= len(core) <= max_len
        and core not in excl
    )
    # Keep only those that are also a single T5 token in a ``", "``-joined caption.
    try:
        t5_base = len(t5_tokenizer(", ", add_special_tokens=False)["input_ids"])
    except Exception:
        return []
    pool: list[str] = []
    for word in candidates:
        ids = t5_tokenizer(", " + word, add_special_tokens=False)["input_ids"]
        if len(ids) - t5_base == 1:
            pool.append(word)
    return pool


def generate_caption_variants(
    caption: str,
    num_variants: int,
    tag_dropout_rate: float,
    protect_fn: Callable[[str], bool] | None = None,
    tag_randomize_rate: float = 0.0,
    erasure_pool: Collection[str] | None = None,
) -> list[str]:
    """Generate ``num_variants`` caption variants for stochastic train-time sampling.

    v0 = pristine original caption. v1..v{N-1} are smart-shuffled (preserving
    the @artist prefix and section anchors), then every tag *after* the prefix
    is independently dropped with probability ``tag_dropout_rate``, then every
    surviving tag *after the prefix* has its identity erased (replaced by a fresh
    vocab token drawn from ``erasure_pool``) with probability
    ``tag_randomize_rate``. The ``@no-artist`` sentinel participates in the
    boundary but is stripped from every variant (including v0) before it is
    written.

    ``protect_fn`` (when given) marks tags that must survive tag-dropout *and*
    tag-randomization: a tag for which it returns True is always kept verbatim
    even past the @artist prefix. It is still subject to shuffling. Used by the
    colorize prep to keep copyright tags present/intact in every variant.

    Both axes are **prefix-protected**: tag-dropout is the *presence* axis and
    tag-randomize is the *identity* axis, and neither touches the @artist prefix
    (the trigger tag stays intact, only tags *after* ``split_idx`` are
    randomized). Section headers (``On the …`` / ``In the …``) and the sentinel
    are never randomized either.

    ``erasure_pool`` (see :func:`build_erasure_token_pool`) is the source of
    erasure symbols: each randomized slot draws a fresh dual-single vocab token
    (clean one-token in both Qwen3 and T5). It is **required** whenever
    ``tag_randomize_rate > 0`` (no random-ASCII fallback); ignored otherwise.
    """
    from library.anima import training as anima_train_utils

    sentinel = anima_train_utils.NO_ARTIST_SENTINEL
    if tag_randomize_rate > 0.0 and not erasure_pool:
        raise ValueError(
            "tag_randomize_rate > 0 requires a non-empty erasure_pool "
            "(build_erasure_token_pool); there is no random-ASCII fallback."
        )
    pool = list(erasure_pool) if erasure_pool else None

    tags = [t.strip() for t in caption.split(",")]
    split_idx = anima_train_utils.find_anima_prefix_end(tags)

    # v0 stays byte-identical to the source caption unless the sentinel is present
    # — re-joining would otherwise normalize whitespace around commas.
    if sentinel in tags:
        variants = [", ".join(anima_train_utils.strip_no_artist_sentinel(tags))]
    else:
        variants = [caption]

    for _ in range(max(0, num_variants - 1)):
        shuffled = anima_train_utils.anima_smart_shuffle_caption(tags.copy())
        if tag_dropout_rate > 0.0 and len(shuffled) > split_idx:
            kept = list(shuffled[:split_idx])
            for tag in shuffled[split_idx:]:
                if (protect_fn is not None and protect_fn(tag)) or (
                    random.random() >= tag_dropout_rate
                ):
                    kept.append(tag)
            if not kept:
                kept = shuffled[:1]
            shuffled = kept
        if tag_randomize_rate > 0.0:
            shuffled = [
                random.choice(pool)
                if (
                    i >= split_idx
                    and tag != sentinel
                    and not tag.startswith(("On the ", "In the "))
                    and not (protect_fn is not None and protect_fn(tag))
                    and random.random() < tag_randomize_rate
                )
                else tag
                for i, tag in enumerate(shuffled)
            ]
        shuffled = anima_train_utils.strip_no_artist_sentinel(shuffled)
        variants.append(", ".join(shuffled))
    return variants


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
    caption_path = image_path.with_suffix(".txt")
    if not caption_path.exists():
        return True
    try:
        return cache_path.stat().st_mtime >= caption_path.stat().st_mtime
    except OSError:
        return False


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
            f"({min_pixels / 1e6:.2f}MP) -- same filter as resize_images.py."
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

        if n_variants <= 0:
            captions = [c for _, c, _ in to_encode]
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
                _encode_batch(
                    captions,
                    tokenize_strategy,
                    encoding_strategy,
                    text_encoder,
                    llm_adapter,
                    device,
                )
            )

            for i, (img_path, _, cache_path) in enumerate(to_encode):
                save_dict = {
                    "prompt_embeds": prompt_embeds[i],
                    "attn_mask": attn_mask[i],
                    "t5_input_ids": t5_input_ids[i],
                    "t5_attn_mask": t5_attn_mask[i],
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if crossattn_emb is not None:
                    save_dict["crossattn_emb"] = crossattn_emb[i]
                save_file(save_dict, str(cache_path))
                stats.written += 1
                if progress is not None:
                    progress(1, detail=f"{img_path.name}")
        else:
            # Per image: N v-variants (v0 pristine + v1..v{N-1} shuffled/dropped)
            # then, when randomizing, N-1 r-variants (shuffled/dropped + identity
            # erasure; their own pristine v0 is dropped since it equals the
            # shared v0). Block size is uniform, so flat indexing stays simple.
            block = n_variants + n_rand
            all_captions: list[str] = []
            for _, caption, _ in to_encode:
                all_captions.extend(
                    generate_caption_variants(
                        caption,
                        n_variants,
                        tag_dropout_rate,
                        caption_protect_fn,
                    )
                )
                if n_rand:
                    r_list = generate_caption_variants(
                        caption,
                        n_variants,
                        tag_dropout_rate,
                        caption_protect_fn,
                        tag_randomize_rate=tag_randomize_rate,
                        erasure_pool=erasure_pool,
                    )
                    all_captions.extend(r_list[1:])  # share v0 as the anchor

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

            for i, (img_path, _, cache_path) in enumerate(to_encode):
                base = i * block
                save_dict = {
                    "num_variants": torch.tensor(n_variants, dtype=torch.int64),
                    # Marker: v0 is pristine; loaders switch on weighted 20%/80%
                    # sampling between v0 and v1..v{N-1}.
                    "v0_intact": torch.tensor(1, dtype=torch.int8),
                    "caption_dropout_rate": caption_dropout_rate,
                }
                if n_rand:
                    save_dict["num_randomized"] = torch.tensor(
                        n_rand, dtype=torch.int64
                    )
                for vi in range(n_variants):
                    flat_idx = base + vi
                    save_dict[f"prompt_embeds_v{vi}"] = prompt_embeds[flat_idx]
                    save_dict[f"attn_mask_v{vi}"] = attn_mask[flat_idx]
                    save_dict[f"t5_input_ids_v{vi}"] = t5_input_ids[flat_idx]
                    save_dict[f"t5_attn_mask_v{vi}"] = t5_attn_mask[flat_idx]
                    if crossattn_emb is not None:
                        save_dict[f"crossattn_emb_v{vi}"] = crossattn_emb[flat_idx]
                for rj in range(n_rand):
                    ri = rj + 1  # r1..r{n_rand}; v0 is the shared pristine anchor
                    flat_idx = base + n_variants + rj
                    save_dict[f"prompt_embeds_r{ri}"] = prompt_embeds[flat_idx]
                    save_dict[f"attn_mask_r{ri}"] = attn_mask[flat_idx]
                    save_dict[f"t5_input_ids_r{ri}"] = t5_input_ids[flat_idx]
                    save_dict[f"t5_attn_mask_r{ri}"] = t5_attn_mask[flat_idx]
                    if crossattn_emb is not None:
                        save_dict[f"crossattn_emb_r{ri}"] = crossattn_emb[flat_idx]
                save_file(save_dict, str(cache_path))
                stats.written += 1
                if progress is not None:
                    detail = f"{img_path.name} ({n_variants}v"
                    detail += f"+{n_rand}r)" if n_rand else ")"
                    progress(1, detail=detail)

    return stats
