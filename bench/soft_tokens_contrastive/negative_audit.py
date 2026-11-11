#!/usr/bin/env python3
"""Soft-tokens contrastive hard-negative audit (Phase 0).

Consumes the method-agnostic typed-tag index
(``post_image_dataset/captions/caption_index.json``, built by
``make caption-index``) and answers the one question Phase 0 of
``docs/anima/proposal/soft_tokens_contrastive.md`` exists to answer:

  *Per training stem, does a **same-artist / different-character** sibling
  exist?* — the "option (c)" hard-negative pool that ``hard_negative()`` would
  draw from. Style is held fixed (same artist) so the only axis a contrastive
  loss can win on is content (different character).

This is the *negative* analogue of ``_archive/bench/ip_adapter/pair_audit.py`` (which
measures distinct **positives**). Where that script walks the
character→copyright→artist back-off looking for a *same*-identity partner, this
one fixes the artist and looks for a *different*-character partner — and reports
how often the genuine signal exists vs. how often ``hard`` mode would silently
fall back to ``shuffled``.

The note's kill threshold: <~50% genuine (strict) coverage means the hard pool
is too thin to bother building the sampler — Phase 1 should run ``shuffled``
only. Writes the standard ``result.json`` envelope + ``negative_audit.md`` into
``bench/soft_tokens_contrastive/results/<ts>[-label]/`` and prints a summary.
Pure data, no GPU.

Run from anima_lora/::

    uv run python -m bench.soft_tokens_contrastive.negative_audit [--label ...]
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bench._common import make_run_dir, write_result

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "post_image_dataset/captions/caption_index.json"

# Headline gate: the scratch note's kill line for the hard pool.
KILL_THRESHOLD_PCT = 50.0


def _same_artist_siblings(stem: str, artists: set[str], groups: dict) -> set[str]:
    """All distinct stems that share at least one artist tag with ``stem``."""
    sib: set[str] = set()
    for a in artists:
        sib.update(groups["artist"].get(a, ()))
    sib.discard(stem)
    return sib


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--label", default="phase0", help="run-dir label (default: phase0)"
    )
    args = parser.parse_args()

    if not INDEX_PATH.exists():
        sys.exit(f"missing {INDEX_PATH} — run `make caption-index` first")
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    image_meta = index["image_meta"]
    groups = index["groups"]
    n_images = len(image_meta)

    # Precompute character sets per stem (used for the disjointness test).
    char_of = {s: set(m.get("character", [])) for s, m in image_meta.items()}
    copy_of = {s: set(m.get("copyright", [])) for s, m in image_meta.items()}

    # ── per-stem classification ───────────────────────────────────────────
    # Mutually-exclusive buckets, tightest signal first:
    #   strict   — anchor has a character tag AND a same-artist sibling has a
    #              *non-empty* character set disjoint from the anchor's. The
    #              genuine option-(c) hard negative (both sides content-defined).
    #   lenient  — not strict, but hard_negative() would still return a non-self
    #              partner: a same-artist sibling whose character set is disjoint
    #              (sibling untagged, or anchor untagged → trivially disjoint).
    #   no_disjoint_sibling — has same-artist siblings, but none are character-
    #              disjoint (every sibling shares all the anchor's characters).
    #   no_artist_sibling   — singleton artist (or orphan): no same-artist peer
    #              at all → hard mode always falls back to shuffled.
    cat = Counter()
    strict_pool_depth = []  # # of strict hard candidates, for stems that have ≥1
    # Among strict candidates: do they also differ in copyright (cleanest), or
    # share a franchise (same copyright, diff char → softer / confound risk)?
    strict_diff_copyright = 0
    strict_same_copyright = 0
    n_with_artist = 0
    n_with_character = 0
    # Per-artist character diversity → which artists are the richest hard sources.
    artist_chars: dict[str, set[str]] = {}

    for stem, meta in image_meta.items():
        A = set(meta.get("artist", []))
        C = char_of[stem]
        if A:
            n_with_artist += 1
        if C:
            n_with_character += 1
        for a in A:
            artist_chars.setdefault(a, set()).update(C)

        siblings = _same_artist_siblings(stem, A, groups)
        if not siblings:
            cat["no_artist_sibling"] += 1
            continue

        # disjoint = sibling's characters share nothing with the anchor's.
        disjoint = [s for s in siblings if not (char_of[s] & C)]
        if not disjoint:
            cat["no_disjoint_sibling"] += 1
            continue

        # strict = anchor content-defined AND ≥1 disjoint sibling also content-defined.
        strict = [s for s in disjoint if C and char_of[s]]
        if strict:
            cat["strict"] += 1
            strict_pool_depth.append(len(strict))
            # copyright relationship of the strict candidates.
            if any(not (copy_of[s] & copy_of[stem]) for s in strict):
                strict_diff_copyright += 1
            else:
                strict_same_copyright += 1
        else:
            cat["lenient"] += 1

    strict = cat["strict"]
    lenient = cat["lenient"]
    no_disjoint = cat["no_disjoint_sibling"]
    no_sibling = cat["no_artist_sibling"]
    # hard_negative() returns a real (non-self) partner for strict ∪ lenient.
    hard_ok = strict + lenient
    fallback = no_disjoint + no_sibling

    def pct(x):
        return 100.0 * x / n_images

    # pool-depth distribution (for k>1 feasibility).
    depth_hist = Counter(min(d, 8) for d in strict_pool_depth)  # cap label at 8+

    # ── richest hard-negative artists (most distinct characters) ──────────
    artist_div = sorted(
        (
            (a, len(chars), len(groups["artist"].get(a, [])))
            for a, chars in artist_chars.items()
        ),
        key=lambda r: (-r[1], -r[2]),
    )
    top_artists = artist_div[:10]

    # ── artist group-size imbalance (singletons = forced fallback) ────────
    artist_sizes = sorted(
        ((a, len(s)) for a, s in groups["artist"].items()), key=lambda r: -r[1]
    )
    singleton_artists = sum(1 for _a, n in artist_sizes if n == 1)
    largest_artists = artist_sizes[:5]

    # ── render markdown ───────────────────────────────────────────────────
    L = []
    L.append("# Soft-tokens contrastive — hard-negative audit (Phase 0)\n")
    L.append(
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"from `{INDEX_PATH.relative_to(REPO_ROOT)}`\n"
    )
    L.append(
        f"(index built {index['meta']['generated']} · vocab "
        f"`{index['meta']['vocab_path']}`).\n"
    )
    L.append(f"\n**Dataset: {n_images} captioned images.**\n")
    L.append(
        "\nQuestion: per stem, is there a **same-artist / different-character** "
        "sibling — the option-(c) hard negative `hard_negative()` would draw "
        "(style held fixed, content differs)? This is the *negative* analogue of "
        "`_archive/bench/ip_adapter/pair_audit.py` (which counts distinct positives).\n"
    )

    L.append("\n## Hard-negative reachability\n")
    L.append(
        "Each stem assigned to the tightest bucket it qualifies for. "
        "**strict** = anchor *and* a same-artist sibling both carry character "
        "tags that are disjoint (genuine content contrast). **lenient** = "
        "`hard_negative()` still returns a non-self partner, but one side is "
        "untagged (trivially disjoint → degenerate contrast). **fallback** = "
        "`hard` mode silently falls through to `shuffled`.\n"
    )
    L.append("| Bucket | Images | % |")
    L.append("|---|---|---|")
    L.append(f"| **strict** (option-(c), genuine) | {strict} | {pct(strict):.1f}% |")
    L.append(f"| lenient (one side untagged) | {lenient} | {pct(lenient):.1f}% |")
    L.append(
        f"| no disjoint sibling (all siblings share a character) | "
        f"{no_disjoint} | {pct(no_disjoint):.1f}% |"
    )
    L.append(
        f"| no same-artist sibling (singleton artist) | "
        f"{no_sibling} | {pct(no_sibling):.1f}% |"
    )
    L.append(
        f"\n**Headline (strict): {strict} / {n_images} = {pct(strict):.1f}%** "
        f"have a genuine same-artist/different-character negative "
        f"(kill threshold {KILL_THRESHOLD_PCT:.0f}%) — "
        f"**{'PASS' if pct(strict) >= KILL_THRESHOLD_PCT else 'FAIL'}**.\n"
    )
    L.append(
        f"`hard_negative()` returns a non-self partner for {hard_ok} "
        f"({pct(hard_ok):.1f}%) of stems; falls back to `shuffled` for "
        f"{fallback} ({pct(fallback):.1f}%)."
    )

    L.append("\n## Coverage ladder (preconditions)\n")
    L.append("| Condition | Images | % |")
    L.append("|---|---|---|")
    L.append(f"| has artist tag | {n_with_artist} | {pct(n_with_artist):.1f}% |")
    L.append(
        f"| has character tag | {n_with_character} | {pct(n_with_character):.1f}% |"
    )
    L.append(
        f"| has same-artist sibling | {n_images - no_sibling} | "
        f"{pct(n_images - no_sibling):.1f}% |"
    )
    L.append(
        "\nThe option-(c) pool is the *intersection* of artist coverage (~100%) "
        "and character coverage — the unmeasured number the proposal flagged. "
        "The strict headline above is that intersection, further gated on a "
        "content-disjoint sibling actually existing."
    )

    if strict_pool_depth:
        L.append("\n## Strict hard-negative pool depth (k>1 feasibility)\n")
        L.append(
            "For stems with ≥1 strict candidate, how many distinct strict "
            "negatives exist (caps the usable `contrastive_k`).\n"
        )
        L.append("| # strict candidates | Stems |")
        L.append("|---|---|")
        for d in sorted(depth_hist):
            label = "8+" if d >= 8 else str(d)
            L.append(f"| {label} | {depth_hist[d]} |")
        median = sorted(strict_pool_depth)[len(strict_pool_depth) // 2]
        L.append(
            f"\nMedian strict pool depth: **{median}** "
            f"(max {max(strict_pool_depth)}). "
            f"k=1 is always safe where strict>0; k=2 needs depth≥2."
        )

    if strict:
        L.append("\n## Franchise confound among strict negatives\n")
        L.append(
            "B=1 false-negative risk: a same-artist/different-character negative "
            "from the *same franchise* (e.g. two Genshin characters) may still "
            "produce a similar velocity. Cleaner negatives also differ in "
            "copyright.\n"
        )
        L.append(
            f"- Strict stems with a **different-copyright** strict candidate "
            f"(cleanest): {strict_diff_copyright} "
            f"({100 * strict_diff_copyright / strict:.1f}% of strict)."
        )
        L.append(
            f"- Strict stems whose strict candidates **all share a franchise** "
            f"(confound risk): {strict_same_copyright} "
            f"({100 * strict_same_copyright / strict:.1f}% of strict)."
        )

    L.append("\n## Richest hard-negative sources (artist character diversity)\n")
    L.append(
        "Artists drawing many distinct characters are the deepest hard-negative "
        "wells — most option-(c) pairs concentrate here.\n"
    )
    L.append("| Artist | Distinct characters | Images |")
    L.append("|---|---|---|")
    for a, ndiv, nimg in top_artists:
        L.append(f"| `{a}` | {ndiv} | {nimg} |")

    L.append("\n## Artist group imbalance\n")
    L.append(
        f"- **{singleton_artists}** artists are singletons (1 image) → those "
        f"stems can never get a same-artist negative (forced `shuffled`)."
    )
    rows = ", ".join(f"`{a}` ({n})" for a, n in largest_artists)
    L.append(f"- Largest artist groups: {rows}")

    L.append("\n## Caveats\n")
    L.append(
        "- **Vocab-limited character classification.** Characters come from the "
        "Anima Tagger vocab (frequency-cutoff subset); untagged characters fall "
        "into the *lenient* bucket as trivially-disjoint, inflating it. Strict "
        "coverage is a floor — tagging more of the set lifts it."
    )
    L.append(
        "- **Artist is exact** (`@` prefix), so same-artist grouping is reliable; "
        "the uncertainty is all on the character axis."
    )
    L.append(
        "- **Whole-index audit.** Counts every captioned image; a training run "
        "with `restrict_stems` (no val leakage) narrows the pool slightly — "
        "re-run with the split if Phase 1 is close to the threshold."
    )
    L.append(
        "- **Structure only.** This says a hard negative *exists*, not that the "
        "contrastive term is load-bearing — that is Phase 1's λ=0 vs λ>0 A/B."
    )

    run_dir = make_run_dir("soft_tokens_contrastive", label=args.label)
    out_md = run_dir / "negative_audit.md"
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")

    metrics = {
        "n_images": n_images,
        "strict": strict,
        "strict_pct": round(pct(strict), 2),
        "lenient": lenient,
        "lenient_pct": round(pct(lenient), 2),
        "no_disjoint_sibling": no_disjoint,
        "no_artist_sibling": no_sibling,
        "hard_ok": hard_ok,
        "hard_ok_pct": round(pct(hard_ok), 2),
        "fallback": fallback,
        "fallback_pct": round(pct(fallback), 2),
        "has_artist": n_with_artist,
        "has_character": n_with_character,
        "singleton_artists": singleton_artists,
        "strict_pool_depth_median": (
            sorted(strict_pool_depth)[len(strict_pool_depth) // 2]
            if strict_pool_depth
            else 0
        ),
        "strict_pool_depth_max": max(strict_pool_depth) if strict_pool_depth else 0,
        "strict_diff_copyright": strict_diff_copyright,
        "strict_same_copyright": strict_same_copyright,
        "kill_threshold_pct": KILL_THRESHOLD_PCT,
        "verdict": "PASS" if pct(strict) >= KILL_THRESHOLD_PCT else "FAIL",
        "index_generated": index["meta"]["generated"],
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=["negative_audit.md"],
    )

    # ── console summary ───────────────────────────────────────────────────
    print(f"hard-negative audit → {run_dir.relative_to(REPO_ROOT)}/")
    print(f"  images: {n_images}")
    print(
        f"  strict (option-(c)) : {strict:5d} ({pct(strict):4.1f}%)  "
        f"[kill <{KILL_THRESHOLD_PCT:.0f}% → {metrics['verdict']}]"
    )
    print(f"  lenient (degenerate): {lenient:5d} ({pct(lenient):4.1f}%)")
    print(f"  no disjoint sibling : {no_disjoint:5d} ({pct(no_disjoint):4.1f}%)")
    print(f"  no artist sibling   : {no_sibling:5d} ({pct(no_sibling):4.1f}%)")
    print(
        f"  hard_negative() ok  : {hard_ok:5d} ({pct(hard_ok):4.1f}%), "
        f"fallback {fallback} ({pct(fallback):4.1f}%)"
    )
    if strict_pool_depth:
        median = sorted(strict_pool_depth)[len(strict_pool_depth) // 2]
        print(f"  strict pool depth   : median {median}, max {max(strict_pool_depth)}")


if __name__ == "__main__":
    main()
