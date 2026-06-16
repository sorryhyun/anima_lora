"""AnimaTagger — multi-label tagger trained on the Anima caption distribution.

The ψ_src provider for DirectEdit. Public surface: ``predict``,
``predict_caption``.

Checkpoint layout (produced by ``python -m scripts.anima_tagger.cli``):

::

    ckpt_dir/
      config.json              # model config + training metadata
      model.safetensors        # AnimaTaggerHead state dict
      thresholds.safetensors   # per-tag F1-optimal thresholds
      vocab.json               # tag list with category + median_pos + group info
      rules.yaml               # caption-normalization rules snapshot
      groups.yaml              # tag-group taxonomy (optional)

When ``groups.yaml`` is present, prediction is group-aware: ``softmax`` and
``softmax_when_solo`` (the latter gated on solo + no-escape) groups emit
exactly one tag per group (argmax over group logits), even when the
sigmoid threshold would have admitted several. Multi-label groups and
ungrouped tags fall back to the standard threshold path.

The head is always dual-encoder (PE-Core + PE-Spatial, hard-routed): PE-Core
(``--encoder``, default PE-Core-L14-336) drives rating / people-count /
identity tags; PE-Spatial (``--aux_encoder``, default PE-Spatial-B16-512)
drives localized tags. Both encoders are loaded lazily on first ``predict``
call and run per image. Pre-dual / v1 single-encoder checkpoints no longer
load (``config.json`` must carry ``aux_encoder`` + ``model.d_in_aux``).

Captions are emitted in Anima's canonical slot order:
``rating, count_tags, characters, copyrights, @artists, generals``, with
underscores replaced by spaces (matching how Anima's training-time T5 saw
the data).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as st_load

from library.captioning import tag_groups as tg
from library.captioning import tag_rules as tr
from library.captioning.anima_tagger_data import pil_resize_to_bucket
from library.captioning.anima_tagger_model import AnimaTaggerConfig, AnimaTaggerHead
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.vision.encoder import (
    VisionEncoderBundle,
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

logger = logging.getLogger(__name__)

# HF repo the dual-encoder tagger checkpoint is auto-fetched from when its
# directory is missing required files. Mirrors the ComfyUI loader's one-time
# auto-download (custom_nodes/comfyui-anima-tagger/nodes.py) so the GUI
# "Autotag" button and CLI entries work out of the box. The dual-encoder
# checkpoint lives at the repo root (no version subfolder).
TAGGER_HF_REPO = "sorryhyun/anima-tagger"
TAGGER_REQUIRED_FILES = ("config.json", "model.safetensors", "vocab.json", "rules.yaml")
TAGGER_OPTIONAL_FILES = ("thresholds.safetensors", "groups.yaml")
DEFAULT_TAGGER_DIR = "models/captioners/anima-tagger-v2"


def ensure_tagger_checkpoint(
    ckpt_dir: str | Path,
    repo: str = TAGGER_HF_REPO,
    subfolder: str = "",
) -> Path:
    """Fetch the tagger checkpoint into ``ckpt_dir`` if any required file is missing.

    One-time download from ``repo`` (default ``sorryhyun/anima-tagger``); files
    are flattened into ``ckpt_dir`` regardless of the source layout so the
    loader's directory contract (``ckpt_dir/config.json`` …) stays uniform.
    Optional files (thresholds / groups) are best-effort — a 404 just means the
    published checkpoint doesn't ship that file. Returns the resolved ``Path``.
    """
    ckpt_dir = Path(ckpt_dir)
    if all((ckpt_dir / f).exists() for f in TAGGER_REQUIRED_FILES):
        return ckpt_dir
    from huggingface_hub.utils import EntryNotFoundError

    from library.runtime.hf_download import hf_download

    logger.info(
        "AnimaTagger: %s missing required files — fetching %s%s (one-time).",
        ckpt_dir,
        repo,
        f"/{subfolder}" if subfolder else "",
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _fetch_flat(fname: str) -> Path:
        repo_path = f"{subfolder}/{fname}" if subfolder else fname
        downloaded = Path(
            hf_download(
                what="AnimaTagger weights",
                repo_id=repo,
                filename=repo_path,
                local_dir=str(ckpt_dir),
            )
        )
        dest = ckpt_dir / fname
        if downloaded.resolve() != dest.resolve():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), str(dest))
        return dest

    for fname in TAGGER_REQUIRED_FILES:
        _fetch_flat(fname)
    for fname in TAGGER_OPTIONAL_FILES:
        try:
            _fetch_flat(fname)
        except EntryNotFoundError:
            logger.debug("optional tagger file %s not present on %s", fname, repo)
    return ckpt_dir


# Matches "1girl", "2girls", …, "6+girls" (digit-prefixed girls counts).
# "multiple girls" is intentionally not matched — it carries no exact count
# so we leave the character head's output untouched in that case.
_GIRLS_COUNT_RE = re.compile(r"^(\d+)\+?girls?$")

# Trailing parenthesized suffix on a tag name, e.g. "nejet (kawakami rokkaku)".
# Booru OC naming convention: when the copyright is `original` (or a meta
# imprint), the parens content is the artist's name.
_OC_SUFFIX_RE = re.compile(r"\(([^()]+)\)\s*$")

# Copyrights that mean "no franchise" — `original` plus the meta-publisher /
# store imprints that booru tags alongside `original` on store-bonus and
# anthology art (melonbooks / toranoana store releases, comic kairakuten
# anthology submissions). All three appear in the trained vocab; other
# meta tags (comiket NN, dengeki <pub>, ...) are out-of-vocab and can't be
# predicted. Membership is exact (no regex) — vocab is small and stable.
_META_COPYRIGHTS = frozenset(
    {"original", "melonbooks", "toranoana", "comic kairakuten"}
)


# Canonical caption-format slot order (matches Anima training captions).
SLOT_ORDER: Tuple[str, ...] = (
    "rating",
    "count",
    "character",
    "copyright",
    "artist",
    "general",
)

# Booru-style tag-type integer → category name. Source of truth for the
# trainer's view of the corpus; written into vocab.json and read back by the
# inference wrapper, so changes here invalidate existing checkpoints.
TAG_TYPE_NAMES: Dict[int, str] = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "metadata",
    6: "deprecated",
}

# 3-class rating set (post-``questionable→sensitive`` collapse).
RATINGS: Tuple[str, ...] = ("general", "sensitive", "explicit")

# 8-class people-count bucket. Derived from parsed count tags
# (``scripts.anima_tagger.constants.classify_people``); trained as a dedicated
# softmax head separate from the multi-label tag head. Order is the canonical
# class index — do not reorder without rebuilding vocab.
PEOPLE_COUNT_LABELS: Tuple[str, ...] = (
    "no_people",  # 0 — no count tag at all
    "1girl",  # 1 — 1girl, no boy
    "1girl_1boy",  # 2 — exactly one of each
    "2girls",  # 3 — 2girls, no boy
    "2girls_1boy",  # 4 — 2girls + 1boy
    "2boys_1girl",  # 5 — 2boys + 1girl  (mirror of 2girls_1boy)
    "1boy",  # 6 — 1boy, no girl (solo male)
    "multi",  # 7 — 3+girls / 3+boys / 2g-2b+ / multiple_* / Nothers
)


@dataclass
class _TagEntry:
    name: str
    index: int
    category: str
    median_pos: float


def _underscore_to_space(s: str) -> str:
    """Anima caption format: tags with spaces, not underscores.

    The cache key uses underscores; the canonical caption uses spaces.
    Apply at emit time (not vocab-build) so tag indexing stays stable.
    """
    return s.replace("_", " ")


def _fix_artist_category(category: str, name: str) -> str:
    """Retype legacy mis-categorized "artist" entries shipped in vocab.json.

    Older vocab builds typed any ``@``-prefixed tag as ``artist``, which
    swept up booru emoticons like ``@_@`` (stored space-form as ``@ @``).
    The corrected rule (see ``scripts/anima_tagger/vocab.categorize``)
    requires ``@`` followed by non-whitespace; anything else falls back
    to ``general``. We patch loaded vocab here so existing checkpoints
    don't need to be rebuilt — the model's tag-level sigmoid is unchanged.
    """
    if category != "artist":
        return category
    if len(name) >= 2 and name[0] == "@" and not name[1].isspace():
        return "artist"
    return "general"


def _load_thresholds(path: Path, n_tags: int, default: float = 0.5) -> torch.Tensor:
    """Load per-tag thresholds; missing → uniform default."""
    if not path.exists():
        logger.warning(
            "no thresholds.safetensors at %s - using default=%.2f", path, default
        )
        return torch.full((n_tags,), default)
    d = st_load(str(path))
    t = d["thresholds"]
    if t.shape != (n_tags,):
        raise ValueError(f"thresholds shape {tuple(t.shape)} != ({n_tags},)")
    return t


class AnimaTagger:
    """Multi-label tagger over the Anima-distribution vocabulary."""

    def __init__(
        self,
        ckpt_dir: str | Path = "models/captioners/anima-tagger-v1",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        pe_ckpt: str | Path | None = None,
        character_floor: float = 0.5,
        pe_lora_path: str | Path | None = None,
        pe_lora_disabled: bool = False,
        pe_aux_ckpt: str | Path | None = None,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype
        self.pe_ckpt = Path(pe_ckpt) if pe_ckpt else None
        # Optional override for the auxiliary (PE-Spatial) encoder's weights
        # path. None → fall back to the encoder registry's default
        # (PE-Spatial-B16-512 at ``models/pe/PE-Spatial-B16-512.pt``).
        self.pe_aux_ckpt = Path(pe_aux_ckpt) if pe_aux_ckpt else None
        # ``pe_lora_path`` / ``pe_lora_disabled`` are accepted for ComfyUI-node
        # call-site compatibility but are no-ops — PE-LoRA was removed when the
        # tagger collapsed to the dual-encoder frozen-trunk architecture.
        del pe_lora_path, pe_lora_disabled
        # Absolute confidence floor for character predictions. Sits *above*
        # the per-tag F1-optimal threshold for the low-confidence end of the
        # character vocab (some F1 thresholds are as low as 0.05 — chasing
        # F1 there produces visible false positives on gender-ambiguous /
        # stylized art). Below the floor → suppress and fall back to the
        # `original` copyright tag.
        self._character_floor = float(character_floor)

        with open(self.ckpt_dir / "config.json", encoding="utf-8") as f:
            cfg_d = json.load(f)
        self.encoder_name: str = cfg_d.get("encoder", "pe")
        # Auxiliary (PE-Spatial) encoder — mandatory; the head is always
        # dual-encoder. ``AnimaTaggerConfig.from_dict`` already rejects
        # pre-dual / v1 configs (missing d_in_aux), so a clear failure there
        # covers the head side; here we make sure we know which encoder to load.
        self.aux_encoder_name: Optional[str] = cfg_d.get("aux_encoder")
        self.cfg = AnimaTaggerConfig.from_dict(cfg_d["model"])
        if not self.aux_encoder_name:
            raise ValueError(
                f"config.json has model.d_in_aux set but no top-level "
                f"'aux_encoder' field — can't determine which auxiliary "
                f'encoder to load. Re-train or hand-add `"aux_encoder": '
                f'"pe_spatial"` to {self.ckpt_dir / "config.json"}.'
            )
        self._cfg_d = cfg_d

        self.model = AnimaTaggerHead(self.cfg)
        self.model.load_state_dict(st_load(str(self.ckpt_dir / "model.safetensors")))
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        with open(self.ckpt_dir / "vocab.json", encoding="utf-8") as f:
            vocab = json.load(f)
        self.tag_entries: List[_TagEntry] = [
            _TagEntry(
                name=t["name"],
                index=int(t["index"]),
                category=_fix_artist_category(str(t["category"]), t["name"]),
                median_pos=float(t.get("median_pos", 0.0)),
            )
            for t in vocab["tags"]
        ]
        self.ratings: List[str] = list(vocab["ratings"])
        # Optional — older vocab.json builds didn't carry the people-count
        # labels, in which case the people head is also absent on the
        # checkpoint side (cfg.n_people_counts == 0). Empty list is the
        # legacy / disabled signal.
        self.people_count_labels: List[str] = list(
            vocab.get("people_count_labels") or []
        )
        # Vocab index of the canonical "original" copyright tag, or None
        # when absent. Used by predict() as the uncertainty-fallback when
        # a character was guessed but didn't clear `_character_floor`.
        self._original_idx: Optional[int] = next(
            (
                e.index
                for e in self.tag_entries
                if e.name == "original" and e.category == "copyright"
            ),
            None,
        )
        # Map category → list of (index, median_pos, name) sorted by median_pos.
        self._by_cat: Dict[str, List[Tuple[int, float, str]]] = {}
        for e in self.tag_entries:
            cat = e.category if e.category in SLOT_ORDER else "general"
            self._by_cat.setdefault(cat, []).append((e.index, e.median_pos, e.name))
        for cat in self._by_cat:
            self._by_cat[cat].sort(key=lambda triple: (triple[1], triple[2]))

        self.thresholds = _load_thresholds(
            self.ckpt_dir / "thresholds.safetensors", n_tags=self.cfg.n_tags
        )
        self.thresholds_dev = self.thresholds.to(self.device)

        self.rules = tr.load_rules(self.ckpt_dir / "rules.yaml")

        # Optional groups snapshot. Built per-group caches so predict()
        # doesn't reparse names every call. When the snapshot is missing
        # (older checkpoints / flat-vocab builds) self._groups is None.
        groups_path = self.ckpt_dir / "groups.yaml"
        self._groups: Optional[tg.TagGroups] = None
        # Per-group: name → {mode, tag_idx_tensor[K], escape_idx_tensor[E],
        #                    tag_names_set, escape_names_set}
        self._group_lookup: Dict[str, Dict] = {}
        # Vocab indices used to detect "single-subject" at inference. We
        # mirror the trainer's GroupRouter logic — `solo`/`1girl`/`1boy`/
        # `1other` are single-count, anything else matching the count
        # regex is multi-count.
        self._single_count_names = {"solo", "1girl", "1boy", "1other"}
        self._multi_count_names: set = set()
        if groups_path.exists():
            self._groups = tg.load_groups(groups_path)
            tag_to_idx = {e.name: e.index for e in self.tag_entries}
            for g in self._groups.groups:
                if g.mode not in ("softmax", "softmax_when_solo"):
                    continue
                tag_idx = [tag_to_idx[t] for t in g.tags if t in tag_to_idx]
                if not tag_idx:
                    continue
                self._group_lookup[g.name] = {
                    "mode": g.mode,
                    "tag_idx": torch.tensor(
                        tag_idx, dtype=torch.long, device=self.device
                    ),
                    "tag_names": tuple(g.tags),
                    "escape_names": tuple(g.escape),
                }
            # Detect multi-count tags by regex over the vocab.
            from re import compile as _re_compile

            count_re = _re_compile(
                r"^(?:\d+(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
            )
            for e in self.tag_entries:
                if e.name in self._single_count_names:
                    continue
                if count_re.match(e.name):
                    self._multi_count_names.add(e.name)
        self._encoder: Optional[VisionEncoderBundle] = None
        self._encoder_aux: Optional[VisionEncoderBundle] = None

    # ── Encoder lazy-load ──────────────────────────────────────────────

    def _bundle(self) -> VisionEncoderBundle:
        if self._encoder is None:
            self._encoder = load_pe_encoder(
                self.device,
                name=self.encoder_name,
                model_id=str(self.pe_ckpt) if self.pe_ckpt else None,
                dtype=self.dtype,
            )
        return self._encoder

    def _bundle_aux(self) -> VisionEncoderBundle:
        """Lazy-load the auxiliary (PE-Spatial) encoder."""
        if self._encoder_aux is None:
            # ``pe_aux_ckpt`` overrides the registry's default checkpoint
            # location when set; otherwise None lets ``load_pe_encoder``
            # resolve via the registry (e.g. ``models/pe/PE-Spatial-B16-512.pt``,
            # auto-fetched from HF when absent).
            self._encoder_aux = load_pe_encoder(
                self.device,
                name=self.aux_encoder_name,
                model_id=str(self.pe_aux_ckpt) if self.pe_aux_ckpt else None,
                dtype=self.dtype,
            )
        return self._encoder_aux

    @torch.no_grad()
    def _encode_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Image → main encoder feature on ``self.device``.

        Shape depends on ``cfg.pool_kind``:
          * ``mean`` → ``[d_enc]`` mean-pooled feature.
          * ``map`` → ``[T, d_enc]`` token sequence; head's MAPHead pools
            internally.
        """
        return self._encode_with(pil_img, self._bundle(), self.cfg.pool_kind)

    @torch.no_grad()
    def _encode_image_aux(self, pil_img: Image.Image) -> torch.Tensor:
        """Image → aux encoder feature, shape per ``cfg.pool_kind_aux``.

        Mirrors :meth:`_encode_image` but for the auxiliary encoder. When
        the aux side is mean-pool, returns ``[d_enc_aux]``; when map,
        returns ``[T_a, d_enc_aux]``."""
        return self._encode_with(
            pil_img,
            self._bundle_aux(),
            self.cfg.pool_kind_aux,
        )

    @torch.no_grad()
    def _encode_with(
        self,
        pil_img: Image.Image,
        bundle: VisionEncoderBundle,
        pool_kind: str,
    ) -> torch.Tensor:
        """Shared image → feature path used by both encoders.

        Each bundle has its own bucket spec so the same source image is
        re-bucketed independently per encoder. Returns ``[T, d_enc]`` tokens
        for ``pool_kind="map"``, or mean-pooled ``[d_enc]`` for ``"mean"``.
        """
        pil_resized = pil_resize_to_bucket(pil_img.convert("RGB"), bundle.bucket_spec)
        tensor = IMAGE_TRANSFORMS(np.array(pil_resized)).unsqueeze(0)
        feats_list = encode_pe_from_imageminus1to1(bundle, tensor, same_bucket=True)
        feats = feats_list[0]  # [T, d_enc]
        if pool_kind == "mean":
            return feats.mean(dim=0).to(torch.float32)
        return feats.to(torch.float32)  # [T, d_enc]

    # ── Public API ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> Dict[str, object]:
        """Run one image through the head; return raw + thresholded outputs.

        Returns a dict with:

        * ``rating``: predicted rating string (one of ``self.ratings``)
        * ``rating_scores``: dict ``{rating: prob}``
        * ``people_count`` / ``people_count_scores``: argmax label and
          ``{label: prob}`` distribution from the 8-class people-count head.
          Both are absent when the loaded checkpoint was trained without
          the people head (legacy ``cfg.n_people_counts == 0``).
        * ``scores``: dict ``{tag: prob}`` for *all* in-vocab tags
        * ``kept``: dict ``{tag: prob}`` for tags emitted as positives.
          When typed groups are loaded, softmax-group winners are picked
          by argmax (one per group) instead of by sigmoid threshold.
        * ``groups``: dict ``{group_name: predicted_tag_or_None}`` — only
          present when typed groups are loaded.
        """
        feat = self._encode_image(pil_img).unsqueeze(0).to(self.device)
        feat_aux = self._encode_image_aux(pil_img).unsqueeze(0).to(self.device)
        tag_logits, rating_logits, people_logits = self.model(feat, feat_aux)
        tag_logits_row = tag_logits[0]  # [n_tags]
        tag_probs = tag_logits_row.sigmoid()  # [n_tags]
        rating_probs = rating_logits.softmax(dim=-1)[0]  # [n_ratings]
        kept_mask = (tag_probs >= self.thresholds_dev).cpu()
        tag_probs_cpu = tag_probs.cpu()
        scores = {
            self.tag_entries[i].name: float(tag_probs_cpu[i])
            for i in range(self.cfg.n_tags)
        }
        kept = {
            self.tag_entries[i].name: float(tag_probs_cpu[i])
            for i in range(self.cfg.n_tags)
            if kept_mask[i]
        }
        rating_idx = int(rating_probs.argmax().item())
        out: Dict[str, object] = {
            "rating": self.ratings[rating_idx],
            "rating_scores": {
                r: float(rating_probs[i].cpu()) for i, r in enumerate(self.ratings)
            },
            "scores": scores,
            "kept": kept,
        }
        if people_logits is not None and self.people_count_labels:
            people_probs = people_logits.softmax(dim=-1)[0]
            people_idx = int(people_probs.argmax().item())
            out["people_count"] = self.people_count_labels[people_idx]
            out["people_count_scores"] = {
                lbl: float(people_probs[i].cpu())
                for i, lbl in enumerate(self.people_count_labels)
            }

        # Group-aware refinement of `kept`. Replaces softmax-group sigmoid
        # threshold output with one argmax winner per applicable group.
        if self._group_lookup:
            kept_names = set(kept.keys())
            is_solo = bool(kept_names & self._single_count_names) and not (
                kept_names & self._multi_count_names
            )
            group_preds: Dict[str, Optional[str]] = {}
            for name, info in self._group_lookup.items():
                mode = info["mode"]
                escape_fired = bool(kept_names & set(info["escape_names"]))
                if mode == "softmax_when_solo":
                    applicable = is_solo and not escape_fired
                else:  # "softmax"
                    applicable = not escape_fired
                if not applicable:
                    # Leave the group's tags exactly as the per-tag
                    # threshold decided. predict_caption can pull whatever
                    # set the sigmoid path admitted.
                    group_preds[name] = None
                    continue
                idx_t = info["tag_idx"]
                group_logits = tag_logits_row.index_select(0, idx_t)
                winner_local = int(group_logits.argmax().item())
                winner_idx = int(idx_t[winner_local].item())
                winner_name = self.tag_entries[winner_idx].name
                # Drop any sigmoid-admitted tags from this group, then add
                # the argmax winner back with its sigmoid probability so
                # downstream callers can still inspect a confidence.
                for t in info["tag_names"]:
                    kept.pop(t, None)
                kept[winner_name] = float(tag_probs_cpu[winner_idx])
                group_preds[name] = winner_name
            out["kept"] = kept
            out["groups"] = group_preds

        # Cap character-tag predictions by the largest digit-prefixed
        # girls-count tag in `kept`. The character head emits an independent
        # sigmoid per tag, so on gender-ambiguous art where both `1boy` and
        # `1girl` fire it can still admit several borderline character
        # matches for what is actually a single subject. Trim to top-N by
        # score where N is parsed from `1girl`/`2girls`/`6+girls`/...; if no
        # digit-prefixed girls tag is in `kept` (e.g. only `1boy`, or only
        # `multiple girls`) leave the character set alone.
        girl_caps = [
            int(m.group(1)) for name in kept if (m := _GIRLS_COUNT_RE.match(name))
        ]
        if girl_caps:
            cap = max(girl_caps)
            char_scored = sorted(
                (
                    (kept[e.name], e.name)
                    for e in self.tag_entries
                    if e.category == "character" and e.name in kept
                ),
                reverse=True,
            )
            for _, name in char_scored[cap:]:
                kept.pop(name, None)
            out["kept"] = kept

        # Suppress uncertain character predictions and fall back to the
        # `original` copyright tag. Any kept character whose score is below
        # `_character_floor` is treated as a guess and dropped. When that
        # empties the character slot AND no copyright tag is in `kept`,
        # add "original" (using its raw sigmoid score) so the caption has
        # a slot-filling copyright — booru convention for non-IP work.
        # Confident characters are preserved; the floor only fires on
        # borderline admits where the F1 threshold lets a noisy guess
        # through.
        dropped_any = False
        for e in self.tag_entries:
            if e.category != "character" or e.name not in kept:
                continue
            if kept[e.name] < self._character_floor:
                kept.pop(e.name, None)
                dropped_any = True
        if dropped_any and self._original_idx is not None:
            has_char = any(
                e.category == "character" and e.name in kept for e in self.tag_entries
            )
            has_copy = any(
                e.category == "copyright" and e.name in kept for e in self.tag_entries
            )
            if not has_char and not has_copy:
                kept["original"] = float(tag_probs_cpu[self._original_idx])

        # Cap artist and copyright slots to top-1 by score. Both heads emit
        # independent sigmoids, so multiple borderline tags can clear their
        # F1 thresholds on a single image — booru convention is one artist
        # / one copyright per work, and downstream callers expect that.
        for cat in ("artist", "copyright"):
            cat_scored = sorted(
                (
                    (kept[e.name], e.name)
                    for e in self.tag_entries
                    if e.category == cat and e.name in kept
                ),
                reverse=True,
            )
            for _, name in cat_scored[1:]:
                kept.pop(name, None)

        # When the surviving copyright is `original` (or a meta-publisher
        # imprint that rides alongside it — see `_META_COPYRIGHTS`), keep a
        # character only when its parens-suffix matches the surviving
        # artist tag (sans `@`). Booru convention: `original` means no
        # franchise, so the only legitimate co-tagged character is the
        # artist's named OC `<name> (<artist>)`. Empirically on
        # `image_dataset/` (n=148 vocab-character tags co-occurring with
        # `original`), 145/148 (98%) match this pattern. No-paren names
        # (`hatsune miku`, `frieren`) and franchise-suffixed names
        # (`kisaki (blue archive)`) paired with `original` are misfires.
        if any(c in kept for c in _META_COPYRIGHTS):
            artist_suffix = next(
                (
                    e.name[1:].lower()
                    for e in self.tag_entries
                    if e.category == "artist"
                    and e.name in kept
                    and e.name.startswith("@")
                ),
                None,
            )
            for e in self.tag_entries:
                if e.category != "character" or e.name not in kept:
                    continue
                m = _OC_SUFFIX_RE.search(e.name)
                if (
                    m is None
                    or artist_suffix is None
                    or m.group(1).strip().lower() != artist_suffix
                ):
                    kept.pop(e.name, None)

        out["kept"] = kept
        return out

    def predict_caption(self, pil_img: Image.Image, min_confidence: float = 0.0) -> str:
        """Image → canonical Anima caption string (rating + slotted tags).

        ``min_confidence`` (0–1) is an extra probability floor applied on top
        of the per-tag F1 thresholds the model was calibrated with: any kept
        tag whose sigmoid probability is below it is dropped. 0.0 (default)
        leaves the model's own threshold decisions untouched. The rating slot
        is always emitted regardless of this floor.
        """
        out = self.predict(pil_img)
        kept = out["kept"]
        if min_confidence > 0.0:
            kept = {name: p for name, p in kept.items() if p >= min_confidence}
        kept_idxs = {
            self.tag_entries[i].index
            for i, name in enumerate([e.name for e in self.tag_entries])
            if name in kept
        }
        # Slot tags by canonical category order, within-slot by median_pos.
        slotted: Dict[str, List[str]] = {cat: [] for cat in SLOT_ORDER}
        slotted["rating"].append(out["rating"])
        for cat, entries in self._by_cat.items():
            for idx, _, name in entries:
                if idx in kept_idxs:
                    slotted.setdefault(cat, []).append(name)
        # Re-apply tag rules at emit time as a safety net (the dedup map
        # already fired during training-data normalization, but the model
        # could in principle predict both ``bra`` and ``black bra``;
        # apply_rules drops ``bra`` in that case).
        flat: List[str] = []
        for cat in SLOT_ORDER:
            flat.extend(slotted.get(cat, []))
        rating_held = flat[:1]
        rest = tr.apply_rules(flat[1:], self.rules)
        out_tags = rating_held + rest
        return ", ".join(_underscore_to_space(t) for t in out_tags)
