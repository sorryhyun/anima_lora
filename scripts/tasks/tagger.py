"""Anima Tagger task entry-points: vocab build / curation modes (``tagger``),
predict (``test-tagger``), autotag, and the dbv4 checkpoint builder.

``make tagger`` / ``make test-tagger`` invoke ``python -m anime_tools.tagger.cli.main``
with the appropriate ``--mode``; extra args are forwarded verbatim. Sidecar
head training runs via
``make daemon-run ARGS="-m anime_tools.tagger.cli.train_sidecar"``.
"""

from __future__ import annotations

from ._common import PY, run


def _tagger(mode: str, extra):
    run([PY, "-m", "anime_tools.tagger.cli.main", "--mode", mode, *extra])


def _mode_in(extra):
    """Return ``(mode, normalized_extra)`` — the ``--mode`` value in ``extra``
    (hyphens→underscores in place), or ``(None, extra)`` when absent.

    Lets ``make tagger ARGS="--mode build-vocab"`` dispatch to any tagger mode
    through the one target, accepting the hyphenated spelling users reach for.
    """
    extra = list(extra)
    for i, a in enumerate(extra):
        if a == "--mode" and i + 1 < len(extra):
            extra[i + 1] = extra[i + 1].replace("-", "_")
            return extra[i + 1], extra
    return None, extra


def cmd_tagger(extra):
    """Run a vocab / curation mode of the Anima Tagger CLI.

    ``make tagger`` (no ``--mode``) runs ``build_vocab`` — scans caption
    sources, emits ``vocab.json`` + ``dataset.json`` and derives + bakes
    tag-groups by default (``--no-derive_groups`` to opt out). Any other mode
    (``predict`` / ``scan_role_markers`` / ``derive_groups``) forwards via
    ``ARGS="--mode <name>"``; hyphenated spellings (``build-vocab``) are
    accepted. Requires ``CAPTION_CORPUS_DIR`` in ``anima_lora/.env`` for
    ``build_vocab`` (or the relevant paths passed via flags).
    """
    mode, extra = _mode_in(extra)
    if mode is None:
        _tagger("build_vocab", extra)
    else:
        run([PY, "-m", "anime_tools.tagger.cli.main", *extra])


def cmd_test_tagger(extra):
    """Single-image debug entry — runs the trained head and prints the caption.

    Without ``--image``, samples a random stem from the val split for a
    side-by-side comparison against ground-truth tags. Pass ``--show_scores``
    to also print rating distribution + top-K kept tags.
    """
    _tagger("predict", extra)


def cmd_autotag(extra):
    """Autotag a single image (CLI one-shot).

    Thin wrapper over ``anime_tools.tagger.cli.autotag``: auto-downloads the
    tagger checkpoint on first use, runs it on ``--image``, and prints the
    predicted caption on one sentinel-prefixed stdout line. Handy for smoke-
    testing the tagger without the GUI (which runs a resident worker —
    ``anime_tools.tagger.cli.autotag_server`` — for fast consecutive tagging).
    Extra args (``--image``, ``--tagger_dir``, ``--device``) forwarded verbatim.
    """
    run([PY, "-m", "anime_tools.tagger.cli.autotag", *extra])


def cmd_tagger_dbv4(extra):
    """Build the dbv4-backed tagger checkpoint dir (external caformer backend).

    ``python -m anime_tools.tagger.cli.build_dbv4_ckpt`` — copies our vocab /
    rules / groups / split from ``--src`` (default ``anima-tagger-v5``) next to
    a ``config.json`` naming the upstream ``animetimm/*.dbv4-full`` repo and
    thresholds seeded from its card. No weights are vendored (GPL-3.0, gated
    — fetched under the user's HF token on first use). Then train the sidecar
    head (copyright / OC characters / people-count) on the GPU via
    ``make daemon-run ARGS="-m anime_tools.tagger.cli.train_sidecar"``.
    """
    run([PY, "-m", "anime_tools.tagger.cli.build_dbv4_ckpt", *extra])
