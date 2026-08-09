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
    REPO_ROOT,
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


def test_error_issues_are_only_paths_and_targets():
    """ERROR level is reserved for the two machine-decidable classes; flags are
    WARN. Pins the contract so a future change can't quietly promote flags."""
    bad = {x.kind for x in collect_issues() if x.level == "ERROR"} - {
        "path",
        "make-target",
    }
    assert not bad, f"unexpected ERROR kinds: {sorted(bad)}"


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


def test_check_path_resolves_relative_to_the_referencing_doc():
    """Self-contained doc trees cite siblings relatively.

    Every ``project/<line>/`` digest writes ``bench/report.md`` meaning *that
    line's* bench dir (see ``project/README.md``). Root-only resolution read
    those as broken because ``bench/`` is also a real top-level dir.
    """
    top = tracked_top_level()
    line_home = REPO_ROOT / "project" / "sigma_lowres"
    # Broken from the root, correct from the doc's own directory.
    probe = "bench/run_sigma_probe.py"
    assert _check_path(probe, top) == probe
    assert _check_path(probe, top, base=line_home) is None
    # The fallback doesn't make the detector permissive: a path that exists
    # under neither root is still flagged.
    missing = "bench/this_bench_does_not_exist.py"
    assert _check_path(missing, top, base=line_home) == missing


def test_known_make_targets_include_canonical_entries():
    """The make-target source-of-truth parse works — a couple of stable targets
    must show up, or the make-target ERROR check can't fire."""
    targets = known_make_targets()
    assert {"lora", "preprocess", "help"} <= targets


def test_collect_issues_is_stable():
    """``collect_issues`` runs against the live tree without raising, and every
    returned record is a well-formed ``Issue`` rooted in the repo."""
    issues = collect_issues()
    for x in issues:
        assert x.level in {"ERROR", "WARN"}
        assert (REPO_ROOT / x.path).exists(), f"issue points at a missing doc: {x.path}"
