"""The i2i pair list: near_twins staging writes it, the MOSS cache reads it."""

from __future__ import annotations

import json

from PIL import Image

from i2i_tasks.tools.near_twins.outputs import write_pair_manifest
from library.moss.cache import manifest_pairs


def _member(path, caption):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), "white").save(path)
    path.with_suffix(".txt").write_text(caption, encoding="utf-8")


def test_staging_tree_round_trips_into_moss_pairs(tmp_path):
    staging = tmp_path / "staging"
    _member(staging / "artist_a" / "1-2_tags.webp", "1girl, speech bubble")
    _member(staging / "artist_a" / "1-2_no_tags.webp", "1girl")
    _member(staging / "artist_b" / "id_7_tags.png", "1boy")
    _member(staging / "artist_b" / "id_7_no_tags.png", "1boy")
    _member(staging / "artist_b" / "9-3_no_tags.png", "orphan")  # no cond twin

    assert write_pair_manifest(staging) == 2
    rows = [
        json.loads(line) for line in (staging / "pairs.jsonl").read_text().splitlines()
    ]
    assert {r["stem"] for r in rows} == {"1-2", "id_7"}

    pairs, refs = manifest_pairs(staging / "pairs.jsonl", "text")
    by_stem = {p.stem: p for p in pairs}
    # the target is the clean twin, captioned with its own caption
    assert by_stem["1-2"].image.name == "1-2_no_tags.webp"
    assert by_stem["1-2"].caption == "1girl"
    assert refs["1-2"].name == "1-2_tags.webp"
    assert refs["id_7"].name == "id_7_tags.png"
