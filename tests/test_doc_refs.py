"""Guard that tracked docs don't drift from the live tree.

``scripts/release/check_docs.py`` is a standalone CLI linter, but nothing ran it
automatically — so a doc could reference a deleted file or a renamed ``make``
target and nobody would notice. This pins its **ERROR** classes (broken repo
paths + unknown ``make`` targets) as a hard test: the live tree must be clean.

CLI *flags* are deliberately WARN-only in the linter (the known-flag set is
permissive on purpose — see its module docstring), so this suite does not fail
on them. We only assert the two ERROR classes, plus a few unit checks so the
guard can't silently pass because the detector itself broke.
"""

from __future__ import annotations


from scripts.release.check_docs import (
    _check_path,
    collect_issues,
    known_make_targets,
    tracked_top_level,
)


def _format(issues) -> str:
    return "\n".join(f"  {x.path}:{x.line}: [{x.kind}] {x.message}" for x in issues)


def test_no_doc_reference_errors():
    """No tracked doc references a nonexistent file path or make target."""
    errors = [x for x in collect_issues() if x.level == "ERROR"]
    assert not errors, (
        f"{len(errors)} broken doc reference(s) — fix the doc or the path/target:\n"
        + _format(errors)
    )


def test_check_path_flags_missing_and_accepts_real():
    """The path detector actually fires — otherwise the guard is a no-op."""
    top = tracked_top_level()
    # A repo-rooted path that does not exist is flagged (returned verbatim).
    missing = "library/this_module_does_not_exist.py"
    assert _check_path(missing, top) == missing
    # A real tracked file is accepted (returns None).
    assert _check_path("scripts/release/check_docs.py", top) is None
    # An extensionless module reference resolves via the ``.py`` fallback.
    assert _check_path("scripts/release/check_docs", top) is None
    # Non-repo roots (URLs, data dirs) are skipped, never flagged.
    assert _check_path("https://example.com/x", top) is None
    assert _check_path("output/tests/foo.png", top) is None
    # A gitignored path under a *tracked* top-level dir is machine-local (a
    # user's own GUI preset) — its absence here is not doc drift.
    assert _check_path("configs/gui-methods/custom/not_generated.toml", top) is None


def test_known_make_targets_include_canonical_entries():
    """The make-target source-of-truth parse works — a couple of stable targets
    must show up, or the make-target ERROR check can't fire."""
    targets = known_make_targets()
    assert {"lora", "preprocess", "help"} <= targets
