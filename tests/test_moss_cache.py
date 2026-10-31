"""MOSS cache contract: latent pack/unpack inverse, sizing, flat-cache reading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from library.moss.cache import (
    LATENT_SUFFIX,
    TE_SUFFIX,
    Pair,
    add_variants,
    cache_text,
    find_pairs,
    fit,
    _te_current,
    list_items,
    load_item,
    load_plain,
    load_variant,
    read_te,
    target_size,
    variant_entries,
)
from library.moss.model import pack_latents, unpack_latents
from library.moss.scan import caption_sha, scan


def _fake_vae(channels=32):
    g = torch.Generator().manual_seed(0)
    bn = SimpleNamespace(
        running_mean=torch.randn(channels * 4, generator=g),
        running_var=torch.rand(channels * 4, generator=g) + 0.5,
    )
    return SimpleNamespace(
        bn=bn, config=SimpleNamespace(batch_norm_eps=1e-4), dtype=torch.float32
    )


def test_pack_is_inverse_of_unpack():
    vae = _fake_vae()
    z = torch.randn(2, 32, 12, 18, dtype=torch.float64).float()
    packed = pack_latents(z, vae)
    assert packed.shape == (2, 128, 6, 9)
    torch.testing.assert_close(unpack_latents(packed, vae), z, rtol=1e-5, atol=1e-5)
    d = torch.randn(1, 128, 5, 7)
    torch.testing.assert_close(
        pack_latents(unpack_latents(d, vae), vae), d, rtol=1e-5, atol=1e-5
    )


def test_pack_matches_flux2_patchify():
    # Channel 4c + 2a + b of the packed latent holds pixel (2i + a, 2j + b) of channel c.
    vae = _fake_vae(channels=1)
    vae.bn.running_mean = torch.zeros(4)
    vae.bn.running_var = torch.ones(4) - 1e-4
    z = torch.arange(16.0).view(1, 1, 4, 4)
    packed = pack_latents(z, vae)
    for a in range(2):
        for b in range(2):
            torch.testing.assert_close(packed[0, 2 * a + b], z[0, 0, a::2, b::2])


@pytest.mark.parametrize(
    "size",
    [
        (896, 1152),
        (1024, 1024),
        (3000, 2000),
        (1200, 4000),
        (640, 480),
        (1023, 1023),
        (5000, 300),
        (17, 33),
    ],
)
def test_target_size_invariants(size):
    w, h = size
    tw, th = target_size(w, h, 1024)
    assert tw % 16 == 0 and th % 16 == 0 and tw >= 16 and th >= 16
    assert tw <= max(16, w) and th <= max(16, h)  # never upscaled
    assert tw * th <= 1024 * 1024 * 1.10
    if min(w, h) >= 256:
        assert abs((tw / th) / (w / h) - 1) < 0.05


def test_target_size_keeps_grid_images():
    assert target_size(960, 1120, 1024) == (960, 1120)  # 1.075 MP, already on the grid
    assert target_size(1536, 1536, 1024) == (1024, 1024)


def test_fit_exact_size():
    im = Image.new("RGB", (1000, 1500))
    assert fit(im, 832, 1248).size == (832, 1248)
    assert fit(im, 1000, 1500) is im


def test_list_and_load_items(tmp_path):
    save_file(
        {"latent": torch.randn(128, 72, 56)}, tmp_path / f"a_b_896x1152{LATENT_SUFFIX}"
    )
    save_file(
        {"embeds": torch.randn(37, 4096).bfloat16()}, tmp_path / f"a_b{TE_SUFFIX}"
    )
    save_file(
        {"latent": torch.randn(128, 64, 64)},
        tmp_path / f"lonely_1024x1024{LATENT_SUFFIX}",
    )
    save_file(
        {"embeds": torch.randn(5, 4096).bfloat16()}, tmp_path / f"textonly{TE_SUFFIX}"
    )
    items = list_items(tmp_path)
    assert [(i.stem, i.width, i.height, i.n_text) for i in items] == [
        ("a_b", 896, 1152, 37)
    ]
    latent, embeds = load_item(items[0])
    assert latent.dtype == torch.float32 and latent.shape == (128, 72, 56)
    assert embeds.dtype == torch.bfloat16 and embeds.shape == (37, 4096)
    assert items[0].n_text_plain == 0 and load_plain(items[0]) is None


def test_plain_embedding_rides_in_the_te_file(tmp_path):
    save_file({"latent": torch.randn(128, 4, 4)}, tmp_path / f"a_64x64{LATENT_SUFFIX}")
    save_file(
        {
            "embeds": torch.randn(40, 4096).bfloat16(),
            "embeds_plain": torch.randn(25, 4096).bfloat16(),
        },
        tmp_path / f"a{TE_SUFFIX}",
    )
    (item,) = list_items(tmp_path)
    assert (item.n_text, item.n_text_plain) == (40, 25)
    plain = load_plain(item)
    assert plain.dtype == torch.bfloat16 and plain.shape == (25, 4096)


def test_te_current_tracks_the_plain_caption(tmp_path):
    path = tmp_path / f"a{TE_SUFFIX}"
    meta = {"caption_sha256": caption_sha("{json}"), "max_text_tokens": "4096"}
    save_file({"embeds": torch.zeros(1, 4)}, path, metadata=meta)
    assert _te_current(path, "{json}", 4096)
    # A layout cache written before plain captions: re-encoded to add one.
    assert not _te_current(path, "{json}", 4096, plain="tags")
    save_file(
        {"embeds": torch.zeros(1, 4)},
        path,
        metadata={**meta, "plain_caption_sha256": caption_sha("tags")},
    )
    assert _te_current(path, "{json}", 4096, plain="tags")
    assert not _te_current(path, "{json}", 4096, plain="edited tags")


def test_find_pairs_prefixes_colliding_stems(tmp_path):
    for sub in ("x", "y"):
        (tmp_path / sub).mkdir()
        Image.new("RGB", (32, 32)).save(tmp_path / sub / "img.png")
        (tmp_path / sub / "img.txt").write_text(
            f" caption {sub}, tag \n", encoding="utf-8"
        )
    Image.new("RGB", (32, 32)).save(tmp_path / "x" / "solo.png")
    (tmp_path / "x" / "solo.txt").write_text("solo", encoding="utf-8")
    Image.new("RGB", (32, 32)).save(tmp_path / "x" / "nocap.png")
    pairs = {p.stem: p.caption for p in find_pairs(tmp_path)}
    assert pairs == {
        "x__img": "caption x, tag",
        "y__img": "caption y, tag",
        "solo": "solo",
    }


def test_find_pairs_falls_back_to_originals(tmp_path, monkeypatch):
    """A sidecars_only Export leaves captions in the resized tree with no PNG:
    those pair with the image_dataset/ original, minus the excluded ones."""
    import library.datasets.curation_actions as curation

    monkeypatch.setenv("ANIMA_HOME", str(tmp_path))
    monkeypatch.setattr(curation, "excluded_rels", lambda: ("set/gone.jpg",))
    monkeypatch.setattr(curation, "workspace_excluded_rels", lambda: ())
    resized = tmp_path / "post_image_dataset" / "resized" / "set"
    source = tmp_path / "image_dataset" / "set"
    resized.mkdir(parents=True)
    source.mkdir(parents=True)
    for stem in ("done", "orig", "gone", "nocap"):
        Image.new("RGB", (32, 32)).save(source / f"{stem}.jpg")
    for stem in ("done", "orig", "gone"):
        (resized / f"{stem}.txt").write_text(stem, encoding="utf-8")
    Image.new("RGB", (32, 32)).save(resized / "done.png")
    pairs = {p.stem: p.image for p in find_pairs(resized)}
    assert pairs == {"done": resized / "done.png", "orig": source / "orig.jpg"}
    s = scan(resized, None)
    assert (s.images, s.pairs) == (2, 2)
    # Outside the resized tree there is nothing to fall back to.
    assert [p.stem for p in find_pairs(source.parent)] == []


def test_find_pairs_layout_json(tmp_path, monkeypatch):
    pytest.importorskip("anime_tools.captions.layout_json")
    import anime_tools.captions.correction as correction

    csv = tmp_path / "tags.csv"
    csv.write_text(
        'name,category,post_count,description\nhatsune_miku,4,10,"[캐릭터 > x] c"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(correction, "find_tag_csv", lambda root=None, lang=None: csv)
    src = tmp_path / "src"
    src.mkdir()
    Image.new("RGB", (32, 32)).save(src / "a.png")
    (src / "a.txt").write_text("1girl, hatsune miku, long hair\n", encoding="utf-8")
    (pair,) = find_pairs(src, "layout_json")
    assert pair.caption == (
        '{"summary":{"characters":["hatsune miku"],"caption":"1girl, long hair"}}'
    )
    assert pair.source_sha == caption_sha("1girl, hatsune miku, long hair")
    assert pair.plain == "1girl, hatsune miku, long hair"
    assert find_pairs(src, "text")[0].plain == ""


def test_layout_json_passes_layout_captions_through(monkeypatch):
    from library.moss.caching import sources

    def no_kb():
        raise AssertionError("an already-converted caption needs no tag CSV")

    monkeypatch.setattr(sources, "tag_knowledge_base", no_kb)
    convert = sources.caption_converter("layout_json")
    done = '{"summary":{"characters":["hatsune miku"],"caption":"1girl"}}'
    assert convert(done) == done
    assert not sources.is_layout_json('{"items":[]}')
    assert not sources.is_layout_json("{broken, tags")


def test_find_pairs_prefers_a_current_layout_sidecar(tmp_path, monkeypatch):
    import os

    from library.moss.caching import sources

    monkeypatch.setattr(
        sources, "caption_converter", lambda fmt: lambda raw: f"converted:{raw}"
    )
    src = tmp_path / "src"
    src.mkdir()
    for stem in ("fresh", "edited", "bare"):
        Image.new("RGB", (32, 32)).save(src / f"{stem}.png")
        (src / f"{stem}.txt").write_text(f"{stem} tags", encoding="utf-8")
    (src / "fresh.layout.txt").write_text('{"summary":{}}', encoding="utf-8")
    edited = src / "edited.layout.txt"
    edited.write_text('{"old":1}', encoding="utf-8")
    t = (src / "edited.txt").stat().st_mtime
    os.utime(edited, (t - 10, t - 10))  # the caption was edited after the export
    pairs = {p.stem: p for p in find_pairs(src, "layout_json")}
    assert pairs["fresh"].caption == '{"summary":{}}'
    assert pairs["fresh"].source_sha == caption_sha("fresh tags")
    assert pairs["fresh"].plain == "fresh tags"
    assert pairs["edited"].caption == "converted:edited tags"
    assert pairs["bare"].caption == "converted:bare tags"
    # ``text`` ignores the sidecar.
    monkeypatch.undo()
    assert {p.caption for p in find_pairs(src, "text")} == {
        "fresh tags",
        "edited tags",
        "bare tags",
    }


def test_scan_reads_source_sha(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    out.mkdir()
    for stem, meta in (
        ("new", {"caption_sha256": "json-sha", "source_sha256": caption_sha("cap")}),
        ("old", {"caption_sha256": caption_sha("cap")}),
        ("stale", {"caption_sha256": "x", "source_sha256": "y"}),
    ):
        Image.new("RGB", (32, 32)).save(src / f"{stem}.png")
        (src / f"{stem}.txt").write_text("cap", encoding="utf-8")
        save_file(
            {"embeds": torch.zeros(1, 4)}, out / f"{stem}{TE_SUFFIX}", metadata=meta
        )
    s = scan(src, out)
    assert (s.text_cached, s.stale_text) == (2, 1)


def test_caption_variants_ride_in_the_te_file(tmp_path):
    """A variant stores its rows after the shared page prefix; reading it puts
    ``embeds``' first ``variant_prefix`` rows back in front."""
    save_file({"latent": torch.randn(128, 4, 4)}, tmp_path / f"a_64x64{LATENT_SUFFIX}")
    embeds = torch.randn(40, 4096).bfloat16()
    tail = torch.randn(4, 4096).bfloat16()
    save_file(
        {
            "embeds": embeds,
            "embeds_plain": torch.randn(25, 4096).bfloat16(),
            "embeds_v1": torch.randn(9, 4096).bfloat16(),
            "embeds_v2": tail,
        },
        tmp_path / f"a{TE_SUFFIX}",
        metadata={"variant_prefix": "30"},
    )
    (item,) = list_items(tmp_path)
    assert (item.n_variants, item.n_text_variants) == (2, 39)
    assert torch.equal(load_variant(item, 2), torch.cat((embeds[:30], tail)))
    assert torch.equal(read_te(item.te_path), embeds)


def test_variant_without_prefix_reads_whole(tmp_path):
    """``variant_prefix`` absent (or 0): the variant is stored whole."""
    whole = torch.randn(7, 4096).bfloat16()
    path = tmp_path / f"a{TE_SUFFIX}"
    save_file({"embeds": torch.randn(9, 4096).bfloat16(), "embeds_v1": whole}, path)
    assert torch.equal(read_te(path, "embeds_v1"), whole)


def test_variant_writers_round_trip(tmp_path):
    """``variant_entries`` stores the tails, ``add_variants`` replaces the set in
    place (through a symlink, at its target) and keeps everything else."""
    embeds = torch.randn(12, 4096).bfloat16()
    full = [
        ("a, b", torch.cat((embeds[:8], torch.randn(3, 4096).bfloat16()))),
        ("b", torch.cat((embeds[:8], torch.randn(2, 4096).bfloat16()))),
    ]
    target = tmp_path / "real" / f"a{TE_SUFFIX}"
    target.parent.mkdir()
    tensors, meta = variant_entries(full, 8)
    assert [t.shape[0] for t in tensors.values()] == [3, 2]
    save_file({"embeds": embeds} | tensors, target, metadata={"caption": "c"} | meta)
    link = tmp_path / f"a{TE_SUFFIX}"
    link.symlink_to(target)
    assert torch.equal(read_te(link, "embeds_v2"), full[1][1])

    add_variants(link, full[:1], 8)
    assert link.is_symlink()
    with safe_open(str(target), "pt") as f:
        assert sorted(f.keys()) == ["embeds", "embeds_v1"]
        assert f.metadata() == {
            "caption": "c",
            "variant_prefix": "8",
            "caption_v1": "a, b",
        }
    assert torch.equal(read_te(target, "embeds_v1"), full[0][1])


def _stub_te(monkeypatch, rows: dict[str, int]):
    """``cache_text`` on CPU: the TE returns ``rows[text]`` random rows."""
    import library.moss.loader as loader

    monkeypatch.setattr(loader, "load_tokenizer", lambda _d: None)
    monkeypatch.setattr(loader, "load_text_encoder", lambda *a, **k: (None, None))
    monkeypatch.setattr(
        loader,
        "prompt_ids",
        lambda _t, text, _m: SimpleNamespace(input_ids=torch.zeros(1, rows[text])),
    )
    monkeypatch.setattr(
        loader,
        "encode_prompt",
        lambda _t, _te, text, *_a: torch.randn(rows[text], 4096),
    )
    monkeypatch.setattr(loader, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)


@pytest.mark.parametrize("edited", [False, True])
def test_cache_text_keeps_variants_of_an_unchanged_caption(
    tmp_path, monkeypatch, capsys, edited
):
    """A re-encode (here ``overwrite``) keeps the file's caption variants while its
    caption is the one they were drawn from; an edited caption drops them."""
    _stub_te(monkeypatch, {"red hair": 6, "blue hair": 6})
    path = tmp_path / f"a{TE_SUFFIX}"
    tail = torch.randn(2, 4096).bfloat16()
    save_file(
        {"embeds": torch.randn(6, 4096).bfloat16(), "embeds_v1": tail},
        path,
        metadata={
            "caption_sha256": caption_sha("red hair"),
            "variant_prefix": "4",
            "caption_v1": "hair",
        },
    )
    caption = "blue hair" if edited else "red hair"
    pair = Pair("a", tmp_path / "a.png", caption, caption_sha(caption))
    cache_text([pair], tmp_path, "", "cpu", 4096, overwrite=True)
    with safe_open(str(path), "pt") as f:
        keys, meta = set(f.keys()), f.metadata()
    assert meta["caption"] == caption
    if edited:
        assert keys == {"embeds"}
        assert "caption_v1" not in meta and "variant_prefix" not in meta
        assert "stale caption variants" in capsys.readouterr().out
    else:
        assert keys == {"embeds", "embeds_v1"}
        assert (meta["caption_v1"], meta["variant_prefix"]) == ("hair", "4")
        assert torch.equal(read_te(path, "embeds_v1")[4:], tail)
