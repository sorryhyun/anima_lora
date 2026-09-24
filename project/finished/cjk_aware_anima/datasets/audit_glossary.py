"""Second-pass audit views over a built tag glossary.

``tag_glossary.py --review`` emits one view: the top-N *disagreements* between
the wiki idiom and the MT rendering, sorted by occurrence. Two blind spots
survive that view and both cost corpus quality:

1. **Collisions.** The review table is per-tag, so it structurally cannot show
   that two different EN tags were handed the *same* target wording. A collision
   is not a mistranslation — each row can look fine alone — but the pair teaches
   the ext rows of one word for two concepts (``bow``/``ribbon`` → 리본,
   ``school uniform``/``serafuku`` → 교복), which is exactly the discrimination
   the pack is trained to buy.
2. **The tail below the review floor.** 200 rows bottoms out around n=450 on the
   KO build (~44 % of occurrences). The unreviewed remainder is dominated by
   ``mt_unverified``, i.e. the tier with no back-translation support at all.

Both views are read-only over the built glossary — fixes go to
``tag_overrides_<lang>.json`` and ride the next ``--reselect`` like any other
review fix.

    python datasets/audit_glossary.py --lang ko
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

# Purely orthographic EN variants: same concept, different spelling. A group
# whose members all normalise to one string is a *correct* merge, not a
# collision, and folding them keeps the review table at eyeball length.
_SPELLING = {"gray": "grey", "colour": "color", "moustache": "mustache"}
# ``-es`` only after a sibilant (dresses → dress); a bare ``(?:s|es)$`` strips
# the earliest match and turns `eyes` into `ey`, which un-folds `eye`/`eyes`.
_SIBILANT = ("s", "x", "z", "ch", "sh")


def singular(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("es") and w[:-2].endswith(_SIBILANT):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def normalise_en(tag: str) -> str:
    words = re.sub(r"[-_]+", " ", tag.lower()).split()
    return " ".join(sorted(singular(_SPELLING.get(w, w)) for w in words))


def render_candidates(entry: dict, n: int = 3) -> str:
    return " · ".join(
        f"{c['ja']} ({c['back']}, {c['f1']}"
        + (f", {c['src']}" if c.get("src") and c["src"] != "wiki" else "")
        + ")"
        for c in (entry.get("candidates") or [])[:n]
    )


def collision_groups(
    tags: dict, noise_floor: int, accepted: dict
) -> tuple[list, list, list, list]:
    """(semantic, orthographic, long_tail, accepted) groups, occurrence-sorted."""
    inv: dict[str, list] = collections.defaultdict(list)
    for tag, e in tags.items():
        if e.get("axis") == "general":
            inv[e["ja"]].append((e["count"], tag, e))

    semantic, orthographic, long_tail, signed_off = [], [], [], []
    for target, members in inv.items():
        if len(members) < 2:
            continue
        members.sort(reverse=True, key=lambda m: m[0])
        group = (sum(m[0] for m in members), target, members)
        if target in accepted:
            signed_off.append(group)
        elif len({normalise_en(t) for _, t, _ in members}) == 1:
            orthographic.append(group)
        elif all(c < noise_floor for c, _, _ in members[1:]):
            # e.g. `1girl` (14754) vs `o o` (2) — a typo tag, not a real merge.
            long_tail.append(group)
        else:
            semantic.append(group)
    for bucket in (semantic, orthographic, long_tail, signed_off):
        bucket.sort(reverse=True, key=lambda g: g[0])
    return semantic, orthographic, long_tail, signed_off


def write_collisions(
    payload: dict, path: Path, overrides: dict, noise_floor: int, accepted: dict
) -> int:
    tags = payload["tags"]
    lang = payload["meta"].get("lang", "ja")
    semantic, orthographic, long_tail, signed_off = collision_groups(
        tags, noise_floor, accepted
    )
    total_occ = sum(g[0] for g in semantic)

    lines = [
        "# Tag glossary — wording collisions",
        "",
        "EN tags that were handed the **same** target wording. The disagreement",
        "review is per-tag and cannot surface these: each row can be a defensible",
        "translation on its own while the pair still collapses two concepts onto",
        "one set of ext rows.",
        "",
        "Ordered by combined occurrence. `*` marks a member whose wording comes",
        f"from `tag_overrides_{lang}.json` — a collision the review rounds created.",
        "",
        f"Fix by giving one member a distinct wording in `tag_overrides_{lang}.json`,",
        "then `--reselect`. Leaving a group alone is a valid outcome: some merges",
        "are what a Korean user would actually type.",
        "",
        f"**{len(semantic)} groups / {total_occ} occurrences** need a decision;",
        f"{len(orthographic)} orthographic + {len(long_tail)} long-tail +",
        f"{len(signed_off)} already-signed-off groups are folded into the",
        "appendices below.",
        "",
        "| n | target | EN tags (count, via) | notes |",
        "|--:|---|---|---|",
    ]
    for total, target, members in semantic:
        cells = " · ".join(
            f"**{t}**{'*' if t in overrides else ''} ({c}, {e['via']})"
            for c, t, e in members
        )
        # The biggest member keeps the target wording; it is the *rivals* that
        # need a distinct one, so surface their unused candidates.
        rival = " · ".join(
            f"**{t}** → {render_candidates(e, 2) or '—'}" for _, t, e in members[1:3]
        )
        lines.append(f"| {total} | {target} | {cells} | {rival} |")

    lines += [
        "",
        "## Appendix A — orthographic variants (no action expected)",
        "",
        "Every member normalises to the same EN string (plural, hyphen, "
        "gray/grey, typo).",
        "",
    ]
    for total, target, members in orthographic:
        lines.append(
            f"- {total} — {target} ← " + ", ".join(f"{t} ({c})" for c, t, _ in members)
        )

    lines += [
        "",
        f"## Appendix B — long-tail members (every rival under n={noise_floor})",
        "",
    ]
    for total, target, members in long_tail:
        lines.append(
            f"- {total} — {target} ← " + ", ".join(f"{t} ({c})" for c, t, _ in members)
        )

    lines += [
        "",
        "## Appendix C — merges signed off by the user (do not re-litigate)",
        "",
        f"Recorded in `collisions_accepted_{lang}.json`. These are real synonyms in",
        "the target language; the merge is the intended wording.",
        "",
    ]
    for total, target, members in signed_off:
        lines.append(
            f"- {total} — {target} ← "
            + ", ".join(f"{t} ({c})" for c, t, _ in members)
            + f" — {accepted.get(target, '')}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(semantic)


def write_tier2(
    payload: dict, path: Path, overrides: dict, floor: int, top: int
) -> int:
    """The `mt_unverified` tail *below* the disagreement review's floor."""
    tags = payload["tags"]
    lang = payload["meta"].get("lang", "ja")
    rows = [
        (e["count"], t, e)
        for t, e in tags.items()
        if e.get("axis") == "general"
        and e["via"] == "mt_unverified"
        and e["count"] < floor
        and t not in overrides
    ]
    rows.sort(reverse=True, key=lambda r: r[0])
    tail_occ = sum(r[0] for r in rows)
    n_tail = len(rows)
    rows = rows[:top]
    shown_occ = sum(r[0] for r in rows)

    lines = [
        "# Tag glossary — review tier 2 (below the disagreement floor)",
        "",
        f"`tag_glossary_review_{lang}.md` stops at n={floor}. Everything here sits",
        "below that line **and** is `mt_unverified` — the tier where back-translation",
        "recovered nothing, so no automatic source ever confirmed the wording.",
        "",
        f"The full unreviewed tail is **{n_tail} tags / {tail_occ} occurrences**;",
        f"the {len(rows)} rows below are its head and carry {shown_occ} of them.",
        "",
        f'Fix by adding `"<en tag>": "<{lang}>"` to `tag_overrides_{lang}.json`,',
        "then `--reselect`. Reading down this list buys coverage fastest.",
        "",
        "| n | tag | chosen | MT | wiki candidates (back-translation, F1) |",
        "|--:|---|---|---|---|",
    ]
    for count, tag, e in rows:
        lines.append(
            f"| {count} | {tag} | **{e['ja']}** | {e.get('mt_ja')} | "
            f"{render_candidates(e) or '—'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)


def review_floor(path: Path) -> int:
    """Lowest occurrence count present in an existing review table."""
    counts = [
        int(m.group(1))
        for m in (
            re.match(r"\|\s*(\d+)\s*\|", ln)
            for ln in path.read_text(encoding="utf-8").splitlines()
        )
        if m
    ]
    return min(counts) if counts else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", default="ko")
    ap.add_argument("--glossary", type=Path, default=None)
    ap.add_argument("--overrides", type=Path, default=None)
    ap.add_argument(
        "--review", type=Path, default=None, help="tier-1 review file (sets the floor)"
    )
    ap.add_argument(
        "--accepted", type=Path, default=None, help="signed-off merges ledger"
    )
    ap.add_argument("--floor", type=int, default=0, help="override the tier-1 floor")
    ap.add_argument("--top", type=int, default=200, help="tier-2 rows")
    ap.add_argument(
        "--noise-floor", type=int, default=5, help="collision long-tail cutoff"
    )
    args = ap.parse_args()

    glossary = args.glossary or ASSETS / f"tag_glossary_{args.lang}.json"
    overrides_path = args.overrides or HERE / f"tag_overrides_{args.lang}.json"
    review = args.review or ASSETS / f"tag_glossary_review_{args.lang}.md"
    accepted_path = args.accepted or HERE / f"collisions_accepted_{args.lang}.json"

    payload = json.loads(glossary.read_text(encoding="utf-8"))
    overrides = (
        json.loads(overrides_path.read_text(encoding="utf-8"))
        if overrides_path.exists()
        else {}
    )
    accepted = (
        json.loads(accepted_path.read_text(encoding="utf-8"))
        if accepted_path.exists()
        else {}
    )
    floor = args.floor or (review_floor(review) if review.exists() else 0)

    col_path = ASSETS / f"collisions_{args.lang}.md"
    t2_path = ASSETS / f"tag_glossary_review_{args.lang}_tier2.md"
    n_col = write_collisions(payload, col_path, overrides, args.noise_floor, accepted)
    n_t2 = write_tier2(payload, t2_path, overrides, floor, args.top)

    print(f"tier-1 floor: n={floor}   overrides: {len(overrides)}")
    print(f"{col_path.relative_to(HERE.parent)}: {n_col} groups needing a decision")
    print(f"{t2_path.relative_to(HERE.parent)}: {n_t2} rows")


if __name__ == "__main__":
    main()
