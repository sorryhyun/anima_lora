"""DirectEdit intent dispatcher — turn (ψ_src, edit_instruction) → (ψ_tar, intent).

Three intents, with APPEND as the safe default:
  * REMOVE  — explicit ``-X`` syntax, or ``no X`` when X matches an existing tag.
              Strip the tag from the caption.
  * REPLACE — Qwen3 last-non-padding-token cosine geometry says the edit phrase
              is "near" exactly one source tag (top-1 above ``replace_threshold``
              AND gap-to-top-2 above ``replace_gap``). String-substitute that tag.
  * APPEND  — everything else. Append ``", " + edit_instruction`` to the caption.

The threshold/gap gate is the load-bearing piece: probe
``_archive/text_enc_probes/edit_nearest_tag.py`` shows that legitimate REPLACE
cases (e.g. "large breasts" into a caption with "medium breasts") have both
top-1 ≳ 0.95 AND gaps ≳ 0.07, while ambiguous or no-conflict cases (huge+large
both present, "medium hair" near "grey hair") have gaps < 0.01. Failing the
gate falls through to APPEND.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import torch
import torch.nn.functional as F


Intent = Literal["append", "remove", "replace", "noop"]

# Type alias for the encoder shim the dispatcher consumes. Takes a list of N
# short phrases (the edit + each comma-split src tag) and returns an (N, D)
# tensor of last-non-padding-token Qwen3 embeddings. The dtype must be float
# (cosine is computed in fp32) and any device is fine.
EncodeLastPooledFn = Callable[[list[str]], torch.Tensor]


@dataclass
class EditPlan:
    """What the dispatcher decided, plus diagnostics so the caller can log."""

    tar_caption: str
    intent: Intent
    detected_conflict_tag: Optional[str] = None
    detection_top1_sim: Optional[float] = None
    detection_gap: Optional[float] = None
    # The literal phrase the dispatcher matched against ψ_src (i.e. after we
    # stripped any "-" / "no " prefix). Useful when the REMOVE syntax was used
    # but the tag wasn't present.
    parsed_edit_phrase: str = ""

    def log_line(self) -> str:
        if self.intent == "replace":
            return (
                f"[dispatcher] intent=REPLACE src_tag={self.detected_conflict_tag!r} "
                f"top1={self.detection_top1_sim:.3f} gap={self.detection_gap:.3f}"
            )
        if self.intent == "remove":
            return f"[dispatcher] intent=REMOVE src_tag={self.detected_conflict_tag!r}"
        if self.intent == "noop":
            return (
                f"[dispatcher] intent=NOOP edit={self.parsed_edit_phrase!r} "
                f"already in ψ_src as {self.detected_conflict_tag!r}"
            )
        return f"[dispatcher] intent=APPEND edit={self.parsed_edit_phrase!r}"


def _split_tags(caption: str) -> list[str]:
    """Comma-split, strip, drop empties. Matches probe behaviour."""
    return [t.strip() for t in caption.split(",") if t.strip()]


def _find_tag_case_insensitive(tags: list[str], needle: str) -> Optional[int]:
    """Return the index of ``needle`` in ``tags`` (case-insensitive exact match)."""
    n = needle.strip().lower()
    for i, t in enumerate(tags):
        if t.lower() == n:
            return i
    return None


def _join_tags(tags: list[str]) -> str:
    return ", ".join(tags)


def _parse_remove_syntax(edit_instruction: str) -> tuple[str, Optional[str]]:
    """Detect explicit removal markers in the edit instruction.

    Returns ``(phrase, removal_kind)`` where ``removal_kind`` is:
      * ``"explicit"`` — ``-X`` form; treat as REMOVE regardless of whether X
        is present in ψ_src.
      * ``"soft"``     — ``no X`` form; only honour as REMOVE if X is actually
        a tag in ψ_src (otherwise the literal phrase stays the edit).
      * ``None``       — no removal syntax detected.
    """
    s = edit_instruction.strip()
    if s.startswith("-"):
        phrase = s.lstrip("-").strip()
        if phrase:
            return phrase, "explicit"
    # Case-insensitive "no X" — only if there's a phrase after it.
    lower = s.lower()
    if lower.startswith("no ") and len(s) > 3:
        phrase = s[3:].strip()
        if phrase:
            return phrase, "soft"
    return s, None


@torch.no_grad()
def encode_last_pooled_via_anima_strategy(
    phrases: list[str],
    text_encoder,
    tokenize_strategy,
    encoding_strategy,
    device: torch.device,
) -> torch.Tensor:
    """Anima-flavoured ``EncodeLastPooledFn``: tokenize + encode via the
    Anima strategy trio and return (N, D) last-non-padding-token vectors.

    Mirrors ``_archive/text_enc_probes/edit_nearest_tag.py::encode_phrases`` exactly — the
    probe is the regression set for the dispatcher thresholds, so the encoding
    path must match.
    """
    tokens = tokenize_strategy.tokenize(phrases)
    prompt_embeds, attn_mask, _t5_ids, _t5_mask = encoding_strategy.encode_tokens(
        tokenize_strategy, [text_encoder], tokens
    )
    lengths = attn_mask.to(prompt_embeds.device).sum(dim=1)
    last_idx = (lengths - 1).clamp(min=0).long()
    rows = torch.arange(prompt_embeds.size(0), device=prompt_embeds.device)
    last_pooled = prompt_embeds[rows, last_idx].float()
    return last_pooled.to(device)


def derive_target_caption(
    src_caption: str,
    edit_instruction: str,
    *,
    encode_last_pooled: EncodeLastPooledFn,
    replace_threshold: float = 0.92,
    replace_gap: float = 0.04,
) -> EditPlan:
    """Decide ψ_tar from (ψ_src, edit phrase) without a tag-families YAML.

    Default is APPEND (the edit phrase is appended). REPLACE only fires when
    Qwen3 geometry is confidently pointing at a single tag.

    The encoder shim is caller-supplied so the ComfyUI node (which goes through
    comfy's stock ``CLIP`` socket, not the AnimaTokenizeStrategy) and the
    standalone CLI (Anima strategy trio) can share this code path. For the
    Anima case, see ``encode_last_pooled_via_anima_strategy``.

    Thresholds tuned against ``_archive/text_enc_probes/edit_nearest_tag.py``
    (22 cases).
    """
    src_tags = _split_tags(src_caption)
    phrase, removal_kind = _parse_remove_syntax(edit_instruction)

    if removal_kind is not None:
        idx = _find_tag_case_insensitive(src_tags, phrase)
        if idx is not None:
            removed = src_tags.pop(idx)
            return EditPlan(
                tar_caption=_join_tags(src_tags),
                intent="remove",
                detected_conflict_tag=removed,
                parsed_edit_phrase=phrase,
            )
        if removal_kind == "explicit":
            # `-X` always means remove; if X isn't in ψ_src this is a no-op.
            return EditPlan(
                tar_caption=src_caption,
                intent="remove",
                detected_conflict_tag=None,
                parsed_edit_phrase=phrase,
            )
        # Soft `no X`: tag absent → user really did want to add the literal
        # phrase "no X" (it's a valid danbooru tag). Fall through to the
        # detection branch with the *original* edit string.
        phrase = edit_instruction.strip()

    # Edit phrase already literally present in ψ_src → NOOP. Skips the encoder
    # forward, prevents the "REPLACE its-own-tag" no-op outcome, and most
    # importantly stops the dispatcher from picking a *different* but
    # cosine-near tag (e.g. user re-types "smile" when caption already has it
    # AND has "grin" nearby — APPEND/REPLACE would corrupt; NOOP keeps it clean).
    existing_idx = _find_tag_case_insensitive(src_tags, phrase)
    if existing_idx is not None:
        return EditPlan(
            tar_caption=src_caption,
            intent="noop",
            detected_conflict_tag=src_tags[existing_idx],
            parsed_edit_phrase=phrase,
        )

    # No tags to compare against → trivially APPEND.
    if not src_tags:
        return EditPlan(
            tar_caption=phrase if not src_caption else f"{src_caption}, {phrase}",
            intent="append",
            parsed_edit_phrase=phrase,
        )

    embeds = encode_last_pooled([phrase] + src_tags)
    edit_vec = embeds[0:1].float()
    tag_vecs = embeds[1:].float()
    sims = F.cosine_similarity(edit_vec, tag_vecs, dim=-1)
    sorted_vals, sorted_idx = torch.sort(sims, descending=True)
    top1_sim = float(sorted_vals[0])
    top2_sim = float(sorted_vals[1]) if sorted_vals.numel() > 1 else 0.0
    gap = top1_sim - top2_sim
    top1_tag = src_tags[int(sorted_idx[0])]

    if top1_sim >= replace_threshold and gap >= replace_gap:
        # Confident REPLACE — string-substitute the top-1 tag.
        new_tags = list(src_tags)
        new_tags[int(sorted_idx[0])] = phrase
        return EditPlan(
            tar_caption=_join_tags(new_tags),
            intent="replace",
            detected_conflict_tag=top1_tag,
            detection_top1_sim=top1_sim,
            detection_gap=gap,
            parsed_edit_phrase=phrase,
        )

    # Otherwise: APPEND. Carry the diagnostics so callers can audit near-misses.
    return EditPlan(
        tar_caption=f"{src_caption}, {phrase}",
        intent="append",
        detected_conflict_tag=top1_tag,
        detection_top1_sim=top1_sim,
        detection_gap=gap,
        parsed_edit_phrase=phrase,
    )
