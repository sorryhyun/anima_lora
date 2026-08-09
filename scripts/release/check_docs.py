#!/usr/bin/env python3
"""Doc-reference linter — catch docs that drift from the implementation.

Scans tracked Markdown (excluding ``_archive/``) and verifies three classes of
machine-checkable reference against the live tree:

  * file paths   — ``library/config/io.py``, ``configs/methods/lora.toml``,
                   ``scripts/preprocess/`` … must exist on disk.        (ERROR)
  * make targets — ``make test-hydra``, ``python tasks.py turbo`` … must be
                   a real ``tasks.py`` COMMANDS key or a Makefile target. (ERROR)
  * CLI flags    — ``--infer_steps``, ``--dave_tau`` … should be a flag the code
                   actually declares.                  (WARN — see caveat below)

Source of truth, never hard-coded here:
  * make targets : the ``COMMANDS`` dict in ``tasks.py`` (AST-parsed) + Makefile
                   targets / ``.PHONY`` names.
  * cli flags    : every ``--flag`` literal appearing in any tracked ``.py``.
  * file paths   : the working tree itself.

Design choices that keep it low-noise:
  * Paths are only checked when their first segment is a git-tracked top-level
    entry, so URLs (``claude.ai/code``), repo slugs (``sorryhyun/anima_lora``)
    and runtime/data dirs (``output/…``, ``post_image_dataset/…``) are skipped,
    not flagged.
  * ``make <x>`` and flags are read only from inline-code spans and fenced
    blocks, so English prose ("make sure", "we make use of") never trips the
    target check.

CLI-flag caveat: the "known flags" set is every ``--x`` mentioned anywhere in
the ``.py`` sources — permissive on purpose (a noisy linter gets disabled). It
reliably catches a *fully removed* flag but won't notice one that lingers only
in a comment. Several benign mentions are suppressed so they don't read as
drift, since that's why flags are WARN, not ERROR:
  * foreign tool flags in shell snippets (uv / gh / ruff …) → ``FOREIGN_FLAGS``;
  * bare placeholders in example payloads (``--some_flag``) → ``PLACEHOLDER_FLAGS``;
  * truncated glob/prefix families (``--region-*`` → ``--region-``, ``--ddp_*``
    → ``--ddp_``) — any token ending in ``-`` or ``_`` is skipped;
  * a flag named on a line that *documents its removal* ("``--fp8`` was removed",
    "no ``--smc_cfg_k`` … retired") → ``_REMOVAL_RE``;
  * a doc that is a historical record of archived/shelved work, opted out with a
    ``<!-- check-docs: ignore-flags -->`` marker anywhere in the file.
Promote with ``--strict`` once the docs are clean.

Usage:
    python scripts/release/check_docs.py            # human report, exit 1 on any ERROR
    python scripts/release/check_docs.py --strict   # WARNs count as failures too
    python scripts/release/check_docs.py --json     # machine-readable

``collect_issues()`` returns the structured list for ``tests/test_doc_refs.py``.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# File extensions we treat as "a reference to a tracked file".
_EXTS = (
    "py",
    "toml",
    "md",
    "json",
    "yaml",
    "yml",
    "sh",
    "ps1",
    "cfg",
    "txt",
    "safetensors",
    "npz",
)

# External-tool long-options that legitimately appear in doc shell snippets;
# suppressed from the flag WARN pass so they don't read as drift. Extend as needed.
FOREIGN_FLAGS = {
    "--no-project",
    "--fix",
    "--clobber",
    "--generate-notes",
    "--title",
    "--token",
    "--name-only",
}

# Bare placeholders in example payloads, never meant to resolve to a real flag.
PLACEHOLDER_FLAGS = {
    "--some_flag",
}

# A flag named on a line whose prose documents its *removal* isn't drift — skip
# flag WARNs on any line matching this.
_REMOVAL_RE = re.compile(
    r"\b(removed|retired|gone|deprecated|no longer)\b", re.IGNORECASE
)

# Historical-record docs opt out of the flag check via this marker — their
# commands reference flags that only ever existed in now-removed/archived code.
_IGNORE_FLAGS_MARKER = "check-docs: ignore-flags"

# A path token: multi-segment (has a slash) or a bare file with a known
# extension. The char class excludes ':' '#' '(' ')' so line/symbol anchors
# (`models.py:1435`, `weights.py::foo`) and trailing punctuation fall off.
_MULTI_RE = re.compile(r"[\w.\-]+(?:/[\w.\-]+)+/?")
_SINGLE_RE = re.compile(r"\b[\w-]+\.(?:%s)\b" % "|".join(_EXTS))
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_FLAG_RE = re.compile(r"--[A-Za-z][A-Za-z0-9_-]+")
# `make <target>` / `python tasks.py <target>` inside a code span.
_MAKE_RE = re.compile(r"\b(?:make|tasks\.py)\s+([a-z][a-z0-9-]*)")

Issue = namedtuple("Issue", "level path line kind token message")


def _git(*args: str) -> str:
    # ``core.quotepath=false`` keeps non-ASCII paths (translated guidebooks)
    # as raw UTF-8, not C-style escapes — else they're unopenable + skipped.
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "-c", "core.quotepath=false", *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _tracked(*pathspec: str) -> list[Path]:
    out = _git("ls-files", *pathspec)
    return [REPO_ROOT / line for line in out.splitlines() if line]


def doc_files(include_scratch: bool = False) -> list[Path]:
    # _archive/ always skipped; bench/ + docs/proposal/ are forward-looking
    # scratchpads (reference to-be-built paths) — opt in with --include-scratch.
    exclude = [":!:_archive/*"]
    if not include_scratch:
        exclude += [":!:bench/*", ":!:docs/proposal/*"]
    return _tracked("*.md", *exclude)


def tracked_top_level() -> set[str]:
    return set(_git("ls-tree", "HEAD", "--name-only").split())


def known_flags() -> set[str]:
    flags: set[str] = set()
    for path in _tracked("*.py"):
        try:
            flags.update(
                _FLAG_RE.findall(path.read_text(encoding="utf-8", errors="ignore"))
            )
        except OSError:
            continue
    return flags


def known_make_targets() -> set[str]:
    targets: set[str] = set()

    # tasks.py COMMANDS dict keys — the canonical target list.
    tree = ast.parse((REPO_ROOT / "tasks.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "COMMANDS" for t in node.targets
        ):
            continue
        if isinstance(node.value, ast.Dict):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    targets.add(key.value)

    # Makefile explicit targets + .PHONY names (e.g. `help`, which forwards to
    # `tasks.py --help` and isn't a COMMANDS key).
    for line in (REPO_ROOT / "Makefile").read_text(encoding="utf-8").splitlines():
        if line.startswith(".PHONY:"):
            targets.update(line.split(":", 1)[1].split())
        m = re.match(r"([A-Za-z][\w-]*)\s*:", line)
        if m:
            targets.add(m.group(1))
    targets.discard("Makefile")
    return targets


def _doc_bases(doc: Path) -> list[Path]:
    """The roots a doc's relative refs may legitimately be written against.

    Always the doc's own directory, plus — for a doc nested inside a
    ``project/<line>/`` tree — that line's home. A line's digests cite
    ``bench/report.md`` / ``bench/run_*.py`` meaning the LINE's bench dir (see
    ``project/README.md``), and those digests are not all at the top of the
    line: an experiment write-up several levels down (e.g.
    ``paper_bench/experiments/e1/README.md``) cites the same line-relative
    paths, which ``doc.parent`` alone never reaches.
    """
    bases = [doc.parent]
    try:
        rel = doc.relative_to(REPO_ROOT / "project")
    except ValueError:
        return bases
    if len(rel.parts) >= 2:  # project/<line>/**/<doc>
        line_home = REPO_ROOT / "project" / rel.parts[0]
        if line_home != doc.parent:
            bases.append(line_home)
    return bases


def _check_path(
    tok: str, top: set[str], base: Path | Sequence[Path] | None = None
) -> str | None:
    """Return the token if it's a broken path reference, else None.

    Resolved against the repo root first, then — when ``base`` is given — each
    of the referencing doc's own roots (see ``_doc_bases``). Self-contained doc
    trees cite their siblings relatively: every ``project/<line>/`` digest says
    ``bench/report.md`` / ``bench/run_*.py`` meaning *that line's* bench dir
    (see ``project/README.md``), which is a correct reference that a
    root-only resolver reads as broken because ``bench/`` is also a real
    top-level dir. Root-first ordering keeps repo-rooted refs authoritative.
    """
    tok = tok.strip().rstrip(".")
    if not tok or tok.startswith(("http", "..", "/", "~", "mailto")):
        return None
    if "/" not in tok and "." not in tok:
        return None
    first = tok.split("/", 1)[0]
    if first not in top:  # not rooted in the repo (URL / data dir / bare name)
        return None
    if base is None:
        bases: list[Path] = []
    elif isinstance(base, Path):
        bases = [base]
    else:
        bases = list(base)
    roots = [REPO_ROOT, *bases]
    for root in roots:
        if (root / tok).exists():
            return None
        # Docs often cite a module without its extension (`bench/_common`).
        if (root / f"{tok}.py").exists():
            return None
    return tok


def _code_fragments(line: str, in_fence: bool) -> list[str]:
    """Code fragments on a line: the whole line inside a fence, else each inline
    `code` span separately. Kept separate (not joined) so adjacent spans can't
    fabricate a `make <next-span>` adjacency."""
    if in_fence:
        return [line]
    return _INLINE_CODE_RE.findall(line)


def collect_issues(include_bench: bool = False) -> list[Issue]:
    top = tracked_top_level()
    flags = known_flags()
    targets = known_make_targets()
    issues: list[Issue] = []

    for doc in doc_files(include_bench):
        rel = str(doc.relative_to(REPO_ROOT))
        doc_bases = _doc_bases(doc)
        in_fence = False
        try:
            text = doc.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.splitlines()
        skip_flags_doc = _IGNORE_FLAGS_MARKER in text

        for lineno, line in enumerate(lines, 1):
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            fragments = _code_fragments(line, in_fence)

            # paths — checked across the whole line (prose + code)
            seen: set[str] = set()
            for rx in (_MULTI_RE, _SINGLE_RE):
                for m in rx.finditer(line):
                    tok = m.group(0)
                    # A glob prefix (`results/20260607-*`) or a brace expansion
                    # (`results/20260801-{1226,1425}`) isn't a literal path
                    # claim — the regex truncates at the '*' / '{'. Anchor on the
                    # parent directory instead so the family ref still verifies
                    # its root but the partial stem doesn't read as broken.
                    if line[m.end() : m.end() + 1] in ("*", "{"):
                        head, sep, _ = tok.rpartition("/")
                        if not sep:
                            continue  # bare glob (`foo*`) — nothing to anchor
                        tok = head + "/"
                    bad = _check_path(tok, top, base=doc_bases)
                    if bad and bad not in seen:
                        seen.add(bad)
                        issues.append(
                            Issue(
                                "ERROR",
                                rel,
                                lineno,
                                "path",
                                bad,
                                f"path does not exist: {bad}",
                            )
                        )

            # make targets — code fragments only
            for frag in fragments:
                for m in _MAKE_RE.finditer(frag):
                    tgt = m.group(1)
                    # a trailing '-' is brace-expansion shorthand truncated at
                    # '{' (`make preprocess-{resize,vae,…}`), not a real target.
                    if tgt.endswith("-") or tgt in targets:
                        continue
                    issues.append(
                        Issue(
                            "ERROR",
                            rel,
                            lineno,
                            "make-target",
                            tgt,
                            f"unknown make target: make {tgt}",
                        )
                    )

            # cli flags — code fragments only, WARN level
            if skip_flags_doc or _REMOVAL_RE.search(line):
                continue
            low = line.lower()
            flagged: set[str] = set()
            for frag in fragments:
                for flag in _FLAG_RE.findall(frag):
                    if (
                        flag in flags
                        or flag in FOREIGN_FLAGS
                        or flag in PLACEHOLDER_FLAGS
                    ):
                        continue
                    # truncated glob / prefix family (`--region-*`, `--ddp_*`)
                    if flag.endswith(("-", "_")) or flag in flagged:
                        continue
                    # negated mention ("No `--vr_frozen_ref_dit` flag") — the doc
                    # is telling readers it doesn't exist, not using it.
                    fl = flag.lower()
                    if f"no {fl}" in low or f"no `{fl}" in low:
                        continue
                    flagged.add(flag)
                    issues.append(
                        Issue(
                            "WARN",
                            rel,
                            lineno,
                            "cli-flag",
                            flag,
                            f"flag not declared in any .py: {flag}",
                        )
                    )

    return issues


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Lint doc references against the live tree."
    )
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--strict", action="store_true", help="WARN-level issues also fail")
    ap.add_argument(
        "--include-bench", action="store_true", help="also scan bench/ scratch docs"
    )
    args = ap.parse_args(argv)

    issues = collect_issues(include_bench=args.include_bench)
    errors = [x for x in issues if x.level == "ERROR"]
    warns = [x for x in issues if x.level == "WARN"]

    if args.json:
        print(json.dumps([x._asdict() for x in issues], indent=2))
    elif not issues:
        print("✓ doc references OK — no broken paths/targets, no unknown flags")
    else:
        by_file: dict[str, list[Issue]] = {}
        for x in issues:
            by_file.setdefault(x.path, []).append(x)
        for path in sorted(by_file):
            print(f"\n{path}")
            for x in sorted(by_file[path], key=lambda y: (y.line, y.kind)):
                tag = "✗" if x.level == "ERROR" else "•"
                print(f"  {tag} {x.path}:{x.line}: [{x.kind}] {x.message}")
        print(f"\n{len(errors)} error(s), {len(warns)} warning(s)")

    return 1 if (errors or (warns and args.strict)) else 0


if __name__ == "__main__":
    sys.exit(main())
