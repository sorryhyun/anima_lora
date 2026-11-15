"""MOSS-format captions: the tree build and the implied cut (no GPU, no tagger)."""

from __future__ import annotations

from pathlib import Path

import pytest

from library.moss.captions import (
    STAMP,
    build_tree,
    cut_implied,
    run_captions,
    write_variants,
)
from library.moss.requests import CaptionsRequest


def _set(root: Path) -> Path:
    src = root / "set"
    (src / "sub").mkdir(parents=True)
    (src / "a.png").write_bytes(b"png")
    (src / "a.txt").write_text(
        "safe, 2girls, @artist, smile, long hair. On the left, very long hair. "
        'On the right, bow, bowtie. Japanese text reads as "はい".\n'
    )
    (src / "sub" / "b.webp").write_bytes(b"webp")
    (src / "sub" / "b.txt").write_text("safe, 1girl, @artist, smile\n")
    (src / "c.png").write_bytes(b"no caption")
    return src


def test_build_tree_flattens_clauses_and_symlinks_images(tmp_path):
    src = _set(tmp_path)
    out = tmp_path / "out"
    assert build_tree(src, out) == 2
    assert (out / "a.png").resolve() == (src / "a.png").resolve()
    assert (out / "sub" / "b.webp").is_symlink()
    assert not (out / "c.png").exists()
    a = (out / "a.txt").read_text()
    assert "On the" not in a  # position clauses merged back for the position stage
    assert a.strip().endswith('Japanese text reads as "はい".')


def test_cut_implied_skips_sidecars(tmp_path):
    src = _set(tmp_path)
    out = tmp_path / "out"
    build_tree(src, out)
    (out / "a.history.txt").write_text("long hair, very long hair\n")
    assert cut_implied(out) == 2  # `long hair` and `bow`
    a = (out / "a.txt").read_text()
    assert "long hair," not in a.replace("very long hair", "")
    assert ", bow," not in a
    assert (out / "a.history.txt").read_text() == "long hair, very long hair\n"


def test_out_defaults_under_moss_captions_and_argv_round_trips():
    req = CaptionsRequest(src="post_image_dataset/resized/channel_(caststation)")
    assert req.out_dir().parts[-2:] == ("moss_captions", "channel_(caststation)")
    again = CaptionsRequest.from_argv([*req.to_argv(), "--no-implied"])
    assert again.src == req.src and again.implied is False


def test_refuses_to_remove_a_folder_it_did_not_write(tmp_path):
    src = _set(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit, match="exists"):
        run_captions(CaptionsRequest(src=str(src), out=str(out)))
    with pytest.raises(SystemExit, match=STAMP):
        run_captions(CaptionsRequest(src=str(src), out=str(out), overwrite=True))


def test_write_variants_keeps_head_and_text_clause(tmp_path):
    src = _set(tmp_path)
    out = tmp_path / "out"
    (out / "sub").mkdir(parents=True)
    (out / "a.txt").write_text((src / "a.txt").read_text())
    (out / "a.history.txt").write_text("x\n")
    assert write_variants(out, 4, 0.1) == 1
    from anime_tools.captions.variants import read_variants_sidecar

    rows = read_variants_sidecar(out / "a.variants.txt")
    assert [label for label, _ in rows] == ["v0", "v1", "v2", "v3"]
    assert rows[0][1] == (src / "a.txt").read_text().strip()
    for _, text in rows[1:]:
        assert text.startswith("safe, 2girls, @artist")
        assert text.endswith('Japanese text reads as "はい".')
    first = (out / "a.variants.txt").read_text()
    write_variants(out, 4, 0.1)
    assert (out / "a.variants.txt").read_text() == first  # seeded per caption
