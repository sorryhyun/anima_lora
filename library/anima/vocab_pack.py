"""CJK vocab pack as a first-class text-encoder asset.

A vocab pack is **not a LoRA**: it is a table of extra T5-side embedding rows
(``ext_embed [rows, 1024]``, ids ``>= T5_TABLE_SIZE``) plus a JSON sidecar
carrying the segmentation / row maps (``library.anima.ext_vocab`` owns the
primitives). Applying it means patching two places, and this module is the
one home for both so the trainer, the preprocess TE cache, the inference CLI
and the embedder front door all see the same table:

1. **Tokenizer** — :class:`VocabPackTokenizeStrategy` re-routes the T5 id
   stream of any prompt that carries a routed character through
   :class:`~library.anima.ext_vocab.HybridT5Encoder`. Prompts with no routed
   character take the stock path untouched, so pure-English captions and
   prompts are bit-identical with or without the pack.
2. **Embedding table** — :func:`attach_vocab_pack` hooks ``llm_adapter.embed``
   so ext ids resolve to pack rows. It is a hook pair (clamp pre-hook +
   row-substitution forward hook, the ComfyUI Adapter node's reference
   design), not a widened ``nn.Embedding``: the module's state dict stays at
   the stock 32128 rows, so ``make merge`` / checkpoint saves / ``ss_``
   metadata are unaffected and the pack composes with any DiT or LoRA.

The active pack is selected by the ``vocab_pack`` config key (``configs/
base.toml``; empty = off), ``--vocab_pack`` on ``train.py`` /
``inference.py`` / ``cache_text_embeddings.py``, or ``ANIMA_VOCAB_PACK``.
Packs are loaded once per process (:func:`load_vocab_pack` memoises by
resolved prefix) because the strategy and the DiT loader both need it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch

from library.anima import ext_vocab
from library.anima.ext_vocab import T5_TABLE_SIZE, HybridT5Encoder
from library.anima.strategy import AnimaTokenizeStrategy
from library.env import resolve_under_home

logger = logging.getLogger(__name__)

#: Public test release of the CJK pack (JA / KO / ZH tag rows + symbol block).
PACK_REPO = "sorryhyun/anima-vocab-pack-cjk"
PACK_STEM = "anima_cjk_vocab_pack"
#: Where ``make download-vocab-pack`` lands the pack (path prefix, no suffix).
DEFAULT_PACK_DIR = "models/vocab_packs"
DEFAULT_PACK_PREFIX = f"{DEFAULT_PACK_DIR}/{PACK_STEM}"

_PACK_SUFFIXES = (".safetensors", ".json")

# safetensors metadata keys stamped into TE caches encoded through a pack, and
# the checkpoint metadata keys train.py stamps on a LoRA trained through one
# (the ComfyUI Adapter node reads the same ``ss_`` names).
CACHE_META_NAME = "vocab_pack"
CACHE_META_SHA = "vocab_pack_sha"
CKPT_META_NAME = "ss_ext_pack"
CKPT_META_SHA = "ss_ext_pack_sha"


def resolve_pack_prefix(path: Union[str, Path, None]) -> Optional[Path]:
    """Path prefix of the ``.safetensors`` + ``.json`` pair, or ``None`` when off.

    Accepts the bare prefix, either file of the pair, or a directory holding
    exactly one pack. Repo-relative paths anchor on :func:`anima_home`. Raises
    ``FileNotFoundError`` (with the download hint) when either half is missing —
    a half-installed pack must never silently degrade to the stock tokenizer.
    """
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return None
    p = resolve_under_home(raw)
    if p.is_dir():
        candidates = sorted(p.glob("*.safetensors"))
        candidates = [c for c in candidates if c.with_suffix(".json").exists()]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"vocab pack dir {p} must hold exactly one .safetensors/.json pair "
                f"(found {len(candidates)}); point vocab_pack at the file prefix."
            )
        p = candidates[0]
    if p.suffix in _PACK_SUFFIXES:
        p = p.with_suffix("")
    missing = [s for s in _PACK_SUFFIXES if not p.with_suffix(s).exists()]
    if missing:
        raise FileNotFoundError(
            f"vocab pack {p} is missing {', '.join(missing)}. Fetch the shipped "
            f"pack with `make download-vocab-pack` (→ {DEFAULT_PACK_PREFIX}) or "
            "point vocab_pack at a local build's path prefix."
        )
    return p


@dataclass
class VocabPack:
    """A loaded pack: the row table, its JSON mapping and a stable identity."""

    prefix: Path
    table: torch.Tensor
    mapping: dict = field(repr=False)
    digest: str

    @property
    def name(self) -> str:
        return self.prefix.name

    @property
    def rows(self) -> int:
        return int(self.table.shape[0])

    @property
    def training(self) -> dict:
        """The pack builder's training label (empty for hand-built packs)."""
        t = self.mapping.get("training")
        return dict(t) if isinstance(t, dict) else {}

    def identity(self) -> dict[str, str]:
        """Stamp-ready identity for cache / checkpoint metadata."""
        return {
            "name": self.name,
            "sha": self.digest,
            "rows": str(self.rows),
        }

    def cache_metadata(self) -> dict[str, str]:
        return {CACHE_META_NAME: self.name, CACHE_META_SHA: self.digest}

    def checkpoint_metadata(self) -> dict[str, str]:
        return {CKPT_META_NAME: self.name, CKPT_META_SHA: self.digest}

    def build_encoder(self, t5_tokenizer, qwen3_tokenizer) -> HybridT5Encoder:
        return HybridT5Encoder.from_mapping(t5_tokenizer, qwen3_tokenizer, self.mapping)

    @classmethod
    def load(cls, prefix: Union[str, Path]) -> "VocabPack":
        prefix = resolve_pack_prefix(prefix)
        if prefix is None:
            raise ValueError("VocabPack.load needs a pack prefix")
        table, mapping = ext_vocab.load_ext_assets(prefix)
        digest = ext_vocab.pack_digest(table, mapping)
        logger.info(
            "vocab pack %s: %d ext rows (sha %s…)",
            prefix.name,
            table.shape[0],
            digest[:12],
        )
        return cls(prefix=prefix, table=table, mapping=mapping, digest=digest)


_LOADED: dict[Path, VocabPack] = {}


def load_vocab_pack(
    path: Union[str, Path, VocabPack, None],
) -> Optional[VocabPack]:
    """Resolve ``path`` to a loaded pack (memoised per process), ``None`` when off.

    The strategy install and the DiT load both need the pack; loading the
    ~285 MB table once and sharing it keeps the second call free.
    """
    if path is None or isinstance(path, VocabPack):
        return path
    prefix = resolve_pack_prefix(path)
    if prefix is None:
        return None
    pack = _LOADED.get(prefix)
    if pack is None:
        pack = VocabPack.load(prefix)
        _LOADED[prefix] = pack
    return pack


def default_vocab_pack() -> str:
    """The configured pack prefix (``ANIMA_VOCAB_PACK`` > base.toml ``vocab_pack``), or ``""``."""
    from library.env import default_checkpoints

    return default_checkpoints().vocab_pack


def resolve_active_pack(args) -> Optional[VocabPack]:
    """The pack an argparse namespace selects.

    ``--no_vocab_pack`` wins; then an explicit ``--vocab_pack``; then the
    config default (``vocab_pack`` in base.toml / ``ANIMA_VOCAB_PACK``). The
    inference parser leaves ``vocab_pack`` at ``None`` so the config default
    applies; the training chain fills it from the merged config (``""`` = off).
    """
    if getattr(args, "no_vocab_pack", False):
        return None
    explicit = getattr(args, "vocab_pack", None)
    if explicit is None:
        explicit = default_vocab_pack()
    return load_vocab_pack(explicit)


# --- Patch point 1: the tokenizer -------------------------------------------


class VocabPackTokenizeStrategy(AnimaTokenizeStrategy):
    """The stock dual tokenizer with the T5 stream re-routed through the pack.

    ``super().tokenize`` still produces the Qwen3 ids (the text-encoder side is
    untouched by the pack) and the stock T5 ids; rows whose text carries a
    routed character get their T5 ids / mask replaced by the hybrid encoding
    (eos-terminated, padded to ``t5_max_length`` — the same max-padding the
    pretrained model expects). Everything else is bit-identical to the parent.
    """

    def __init__(self, pack: VocabPack, **kwargs) -> None:
        super().__init__(**kwargs)
        self.pack = pack
        self.encoder = pack.build_encoder(self.t5_tokenizer, self.qwen3_tokenizer)

    def tokenize(self, text):
        texts = [text] if isinstance(text, str) else list(text)
        q_ids, q_mask, t5_ids, t5_mask = super().tokenize(texts)
        for i, t in enumerate(texts):
            if self.encoder.routes(t):
                ids, mask = self.encoder.encode(t, self.t5_max_length)
                t5_ids[i] = torch.tensor(ids, dtype=t5_ids.dtype)
                t5_mask[i] = torch.tensor(mask, dtype=t5_mask.dtype)
        return [q_ids, q_mask, t5_ids, t5_mask]


def make_tokenize_strategy(
    pack: Optional[VocabPack], **kwargs
) -> AnimaTokenizeStrategy:
    """``VocabPackTokenizeStrategy`` when a pack is active, else the stock one."""
    if pack is None:
        return AnimaTokenizeStrategy(**kwargs)
    return VocabPackTokenizeStrategy(pack, **kwargs)


def strategy_pack(strategy) -> Optional[VocabPack]:
    """The pack a tokenize strategy carries (``None`` for the stock strategy)."""
    return getattr(strategy, "pack", None)


# --- Patch point 2: the embedding table --------------------------------------

_HOOK_ATTR = "_vocab_pack_hooks"
_DIGEST_ATTR = "_vocab_pack_digest"


def _embed_module(model_or_adapter) -> torch.nn.Embedding:
    adapter = getattr(model_or_adapter, "llm_adapter", model_or_adapter)
    embed = getattr(adapter, "embed", None)
    if not isinstance(embed, torch.nn.Embedding):
        raise RuntimeError(
            "vocab pack needs an Anima DiT (with llm_adapter) or an LLMAdapter — "
            f"got {type(model_or_adapter).__name__}"
        )
    return embed


def attached_pack_digest(model_or_adapter) -> Optional[str]:
    """Digest of the pack hooked onto this model / adapter, ``None`` if none."""
    try:
        return getattr(_embed_module(model_or_adapter), _DIGEST_ATTR, None)
    except RuntimeError:
        return None


def attach_vocab_pack(model_or_adapter, pack: VocabPack) -> None:
    """Hook the pack rows onto ``llm_adapter.embed`` (idempotent per pack).

    The pre-hook clamps ext ids to ``<unk>`` so the stock table never sees an
    out-of-range index, remembers which positions were ext, and the forward
    hook overwrites those positions with the pack rows. The table stays on CPU
    in its stored dtype; only the rows a batch actually uses are gathered and
    moved (a few KB per encode — no resident VRAM cost). Prompts with no ext
    id pass through with zero overhead beyond the mask test.

    Re-attaching the same pack is a no-op; a different pack replaces the hooks
    (the previous handles are removed first).
    """
    embed = _embed_module(model_or_adapter)
    if getattr(embed, _DIGEST_ATTR, None) == pack.digest:
        return
    detach_vocab_pack(model_or_adapter)

    if embed.weight.shape[0] != T5_TABLE_SIZE:
        raise RuntimeError(
            f"llm_adapter.embed has {embed.weight.shape[0]} rows, expected "
            f"{T5_TABLE_SIZE} — was the table widened by hand?"
        )
    table = pack.table
    unk = ext_vocab.T5_UNK_ID
    state: dict = {}

    def _clamp_pre_hook(module, args):
        if not args or not torch.is_tensor(args[0]):
            state.pop("mask", None)
            return None
        ids = args[0]
        mask = ids >= T5_TABLE_SIZE
        if not bool(mask.any()):
            state.pop("mask", None)
            return None
        state["mask"] = mask
        state["ext"] = ids[mask] - T5_TABLE_SIZE
        return (ids.masked_fill(mask, unk),) + tuple(args[1:])

    def _rows_hook(module, args, output):
        mask = state.pop("mask", None)
        if mask is None:
            return None
        ext = state.pop("ext")
        rows = table[ext.to("cpu", torch.long)]
        out = output.clone()
        out[mask] = rows.to(device=output.device, dtype=output.dtype)
        return out

    handles = (
        embed.register_forward_pre_hook(_clamp_pre_hook),
        embed.register_forward_hook(_rows_hook),
    )
    setattr(embed, _HOOK_ATTR, handles)
    setattr(embed, _DIGEST_ATTR, pack.digest)
    logger.info(
        "vocab pack %s: %d ext rows hooked onto llm_adapter.embed", pack.name, pack.rows
    )


def detach_vocab_pack(model_or_adapter) -> None:
    """Remove the hook pair (no-op when nothing is attached)."""
    embed = _embed_module(model_or_adapter)
    for h in getattr(embed, _HOOK_ATTR, ()) or ():
        h.remove()
    if hasattr(embed, _HOOK_ATTR):
        delattr(embed, _HOOK_ATTR)
    if hasattr(embed, _DIGEST_ATTR):
        delattr(embed, _DIGEST_ATTR)


# --- Identity checks ---------------------------------------------------------


def read_checkpoint_stamp(path: Union[str, Path]) -> tuple[str, str]:
    """``(name, sha)`` a LoRA file was stamped with, ``("", "")`` when unstamped."""
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            md = f.metadata() or {}
    except Exception:
        return "", ""
    return str(md.get(CKPT_META_NAME, "") or ""), str(md.get(CKPT_META_SHA, "") or "")


def warn_checkpoint_pack_mismatch(
    path: Union[str, Path], active: Optional[VocabPack]
) -> None:
    """Log when a LoRA's stamped pack disagrees with the active one.

    A LoRA trained through a pack expects that pack's rows behind ids
    ``>= T5_TABLE_SIZE``; no pack (or a different one) means CJK / quoted
    prompt spans reach rows it never saw. EN prompts are unaffected either
    way, so this is a warning, not an error.
    """
    name, sha = read_checkpoint_stamp(path)
    if not sha:
        return
    if active is None:
        logger.warning(
            "%s was trained through vocab pack %s (sha %s…) but no vocab pack is "
            "active — CJK / quoted prompt spans will not reach the rows it was "
            "trained on (set vocab_pack in configs/base.toml or pass --vocab_pack; "
            "EN prompts are unaffected).",
            path,
            name or "?",
            sha[:12],
        )
    elif active.digest != sha:
        logger.warning(
            "%s was trained through vocab pack %s (sha %s…) but the active pack is "
            "%s (sha %s…) — ext rows differ; expect CJK drift.",
            path,
            name or "?",
            sha[:12],
            active.name,
            active.digest[:12],
        )


_warned_cache_stamps: set[str] = set()


def check_cache_stamp(
    metadata: Optional[dict], cache_path: str, active: Optional[VocabPack]
) -> None:
    """Warn once per (pack-state, kind) when a TE cache's stamp disagrees.

    TE caches skip on existence only (no content hash), so a cache encoded
    through one pack — or through none — silently trains against whatever the
    active pack now is. The stamp turns that into a visible warning; the fix is
    always ``make preprocess-te ARGS=--overwrite``.
    """
    md = metadata or {}
    cached_sha = str(md.get(CACHE_META_SHA, "") or "")
    active_sha = active.digest if active is not None else ""
    if cached_sha == active_sha:
        return
    key = f"{cached_sha}->{active_sha}"
    if key in _warned_cache_stamps:
        return
    _warned_cache_stamps.add(key)
    if cached_sha and not active_sha:
        what = f"was encoded through vocab pack {md.get(CACHE_META_NAME, '?')} but no pack is active"
    elif active_sha and not cached_sha:
        what = f"was encoded without a vocab pack but {active.name} is active"
    else:
        what = (
            f"was encoded through vocab pack {md.get(CACHE_META_NAME, '?')} "
            f"but {active.name} is active"
        )
    logger.warning(
        "TE cache %s %s — the cached T5 ids / crossattn no longer match the "
        "tokenizer. Re-run `make preprocess-te ARGS=--overwrite` (EN-only "
        "captions are unaffected). Further mismatches of this kind are not repeated.",
        cache_path,
        what,
    )
