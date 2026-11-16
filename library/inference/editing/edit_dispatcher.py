"""DirectEdit intent dispatcher — turn (ψ_src, edit_instruction) → (ψ_tar, intent).

Three intents, APPEND by default:
  * REMOVE  — ``-X``, or ``no X`` when X is an existing tag. Strip the tag.
  * REPLACE — the edit phrase's Qwen3 last-token embedding is near exactly one
              source tag (top-1 ≥ ``replace_threshold`` and gap-to-top-2 ≥
              ``replace_gap``). Substitute that tag.
  * APPEND  — everything else.

The gap gate carries the decision: real REPLACE cases have gaps ≳ 0.07, while
ambiguous ones ("medium hair" near "grey hair") sit below 0.01.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

from anime_tools.captions.position_clauses import compose_caption, parse_caption

import torch
import torch.nn.functional as F


Intent = Literal["append", "remove", "replace", "noop"]

# N phrases (the edit + each src tag) -> (N, D) float tensor of
# last-non-padding-token Qwen3 embeddings, any device.
EncodeLastPooledFn = Callable[[list[str]], torch.Tensor]


@dataclass
class EditPlan:
    """What the dispatcher decided, plus diagnostics so the caller can log."""

    tar_caption: str
    intent: Intent
    detected_conflict_tag: Optional[str] = None
    detection_top1_sim: Optional[float] = None
    detection_gap: Optional[float] = None
    # The phrase matched against ψ_src, after stripping any "-" / "no " prefix.
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


def _split_tags(caption: str) -> tuple[list[str], tuple]:
    """The flat tag bag (stripped, empties dropped) and the position clauses,
    which the edit leaves in place."""
    parsed = parse_caption(caption)
    return [t.strip() for t in parsed.flat_tags if t.strip()], parsed.clauses


def _append(src_caption: str, phrase: str, tags: list[str], clauses: tuple) -> str:
    """Add ``phrase`` to the flat bag; a trailing append would land it inside
    the last position clause."""
    if not clauses:
        return phrase if not src_caption else f"{src_caption}, {phrase}"
    return compose_caption([*tags, phrase], clauses)


def _find_tag_case_insensitive(tags: list[str], needle: str) -> Optional[int]:
    """Return the index of ``needle`` in ``tags`` (case-insensitive exact match)."""
    n = needle.strip().lower()
    for i, t in enumerate(tags):
        if t.lower() == n:
            return i
    return None


def _join_tags(tags: list[str], clauses: tuple = ()) -> str:
    return compose_caption(tags, clauses)


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
    """``EncodeLastPooledFn`` via the Anima strategy trio: (N, D)
    last-non-padding-token vectors. The dispatcher thresholds were tuned on
    this encoding path.
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

    The encoder shim is caller-supplied so the ComfyUI node (comfy's ``CLIP``
    socket) and the CLI (``encode_last_pooled_via_anima_strategy``) share this.
    """
    src_tags, clauses = _split_tags(src_caption)
    phrase, removal_kind = _parse_remove_syntax(edit_instruction)

    if removal_kind is not None:
        idx = _find_tag_case_insensitive(src_tags, phrase)
        if idx is not None:
            removed = src_tags.pop(idx)
            return EditPlan(
                tar_caption=_join_tags(src_tags, clauses),
                intent="remove",
                detected_conflict_tag=removed,
                parsed_edit_phrase=phrase,
            )
        if removal_kind == "explicit":
            # `-X` with X absent from ψ_src: nothing to remove.
            return EditPlan(
                tar_caption=src_caption,
                intent="remove",
                detected_conflict_tag=None,
                parsed_edit_phrase=phrase,
            )
        # `no X` with X absent: "no X" is itself a danbooru tag, so treat the
        # original string as the edit.
        phrase = edit_instruction.strip()

    # Phrase already present → NOOP; otherwise a cosine-near neighbour ("grin"
    # beside "smile") could get replaced.
    existing_idx = _find_tag_case_insensitive(src_tags, phrase)
    if existing_idx is not None:
        return EditPlan(
            tar_caption=src_caption,
            intent="noop",
            detected_conflict_tag=src_tags[existing_idx],
            parsed_edit_phrase=phrase,
        )

    if not src_tags:
        return EditPlan(
            tar_caption=_append(src_caption, phrase, src_tags, clauses),
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
        new_tags = list(src_tags)
        new_tags[int(sorted_idx[0])] = phrase
        return EditPlan(
            tar_caption=_join_tags(new_tags, clauses),
            intent="replace",
            detected_conflict_tag=top1_tag,
            detection_top1_sim=top1_sim,
            detection_gap=gap,
            parsed_edit_phrase=phrase,
        )

    # APPEND, keeping the diagnostics for near-miss audits.
    return EditPlan(
        tar_caption=_append(src_caption, phrase, src_tags, clauses),
        intent="append",
        detected_conflict_tag=top1_tag,
        detection_top1_sim=top1_sim,
        detection_gap=gap,
        parsed_edit_phrase=phrase,
    )
