"""Regression tests for caption sidecar reading.

An empty (or whitespace-only) ``.txt`` caption file is a valid *explicit
empty caption* — used deliberately for unconditional / style-LoRA training.
It must resolve to ``""`` (a real, empty caption) rather than raising or
being treated as a missing sidecar. A genuinely absent sidecar still returns
``None`` so the ``class_tokens`` fallback can kick in.
"""

import pytest

from library.datasets.dreambooth import read_caption


def _write(tmp_path, name, content):
    img = tmp_path / name
    img.write_bytes(b"")  # placeholder image
    (tmp_path / (img.stem + ".txt")).write_text(content, encoding="utf-8")
    return str(img)


@pytest.mark.parametrize("enable_wildcard", [False, True])
@pytest.mark.parametrize(
    "content",
    ["", "  \n  \n"],
    ids=["zero-byte", "blank-lines"],
)
def test_empty_caption_file_resolves_to_empty_string(
    tmp_path, content, enable_wildcard
):
    img = _write(tmp_path, "abc.png", content)
    caption = read_caption(img, ".txt", enable_wildcard)
    # Not None (sidecar exists) and not a crash — an explicit empty caption.
    assert caption == ""


def test_missing_sidecar_returns_none(tmp_path):
    img = tmp_path / "abc.png"
    img.write_bytes(b"")
    assert read_caption(str(img), ".txt", False) is None
