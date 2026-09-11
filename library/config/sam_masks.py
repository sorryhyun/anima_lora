"""The SAM mask stage's region list, as the trainer spells it.

anime_tools 0.6.4 replaced ``SamMaskRequest``'s ``prompts`` (masked out) /
``focus_prompts`` (kept) / ``prompt_embed`` with one ``masks`` list of
``ROLE:KIND:VALUE`` specs (``keep|ignore`` × ``text|soft``). A
``configs/sam_mask.yaml`` or a saved ``[[variant.stages.masks_sam]]`` card written
before then still carries the old pair; this module reads it into the new list.
Torch-free — the GUI imports it (lazily; ``anime_tools.downloads`` is the cost).
"""

from __future__ import annotations

from anime_tools.downloads import DEFAULT_SUBJECT_PROMPT_EMBED

SUBJECT_PROMPT = "girl"
"""The phrase the shipped soft prompt was learned from. Before 0.6.4 the stage
served this prompt — in either list — through ``--prompt_embed``, so a legacy
``girl`` becomes the soft entry, not a text one."""

LEGACY_KEYS = ("prompts", "focus_prompts")
_DROPPED = (*LEGACY_KEYS, "prompt_embed")


def _items(value) -> list[str]:
    """A legacy prompt list: a yaml list or the GUI's csv text, with ``none`` /
    ``off`` spelling "no prompts"."""
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = [str(p).strip() for p in items if str(p).strip()]
    return [] if [p.lower() for p in out] in (["none"], ["off"]) else out


def _spec(role: str, prompt: str) -> str:
    if prompt == SUBJECT_PROMPT:
        return f"{role}:soft:{DEFAULT_SUBJECT_PROMPT_EMBED}"
    return f"{role}:text:{prompt}"


def legacy_masks(prompts=None, focus_prompts=None) -> list[str]:
    """``prompts`` / ``focus_prompts`` → ``masks`` specs: kept regions first,
    then ignored ones."""
    return [_spec("keep", p) for p in _items(focus_prompts)] + [
        _spec("ignore", p) for p in _items(prompts)
    ]


def rule_masks(rule: dict) -> list[str]:
    """One ``sam_mask.yaml`` rule's ``masks`` list: its own ``masks:`` key, else
    its pre-0.6.4 ``prompts`` / ``focus_prompts`` pair (an absent list is empty —
    the yaml never took the stage's ``girl`` default)."""
    if "masks" in rule:
        return [str(m).strip() for m in (rule.get("masks") or ()) if str(m).strip()]
    return legacy_masks(rule.get("prompts"), rule.get("focus_prompts"))


def migrate_card(card: dict) -> dict:
    """A ``masks_sam`` form card in the 0.6.4 shape. A pre-0.6.4 card elided
    ``focus_prompts`` at its old default ``girl`` and ``prompts`` at ``none``, so
    a missing key reads as that default. A card with ``masks`` already, or with
    neither legacy key, keeps its values (minus a stale ``prompt_embed``)."""
    out = {k: v for k, v in card.items() if k not in _DROPPED}
    if "masks" not in card and any(k in card for k in LEGACY_KEYS):
        out["masks"] = legacy_masks(
            card.get("prompts"), card.get("focus_prompts", SUBJECT_PROMPT)
        )
    return out
