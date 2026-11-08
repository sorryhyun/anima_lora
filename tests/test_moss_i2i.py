"""MOSS i2i: the task recipes load against their requests, ``generate`` expands a
reference into sized jobs."""

from __future__ import annotations


import pytest
from PIL import Image

from library.moss.requests import (
    I2I_CACHE,
    GenerateRequest,
    TrainRequest,
    i2i_config,
)


@pytest.mark.core
@pytest.mark.parametrize("task", sorted(I2I_CACHE))
def test_task_recipes_load(task):
    """A task's cache, train and generate recipes agree on the cache folder and
    the reference scale, and run a swap plan (bf16, or the int8 base: colorize v5)."""
    cache = I2I_CACHE[task].load_config(i2i_config(task, "cache"))
    train = TrainRequest.load_config(i2i_config(task, "train"))
    gen = GenerateRequest.load_config(i2i_config(task, "generate"))
    out = cache.get("out", I2I_CACHE[task]().out)
    assert train["cache"] == out
    # The test reference matches what the cache encoded.
    ref_scale = cache.get("ref_scale", I2I_CACHE[task]().ref_scale)
    assert gen.get("ref_scale", GenerateRequest().ref_scale) == ref_scale
    assert train["int8"] in ("off", "stream") and train["blocks_to_swap"] > 0


def test_ref_jobs_follow_each_reference(tmp_path):
    from library.moss.generate import build_jobs, fingerprint

    Image.new("RGB", (1200, 1700)).save(tmp_path / "page one.png")
    Image.new("RGB", (800, 800)).save(tmp_path / "b.webp")
    (tmp_path / "notes.txt").write_text("x")
    req = GenerateRequest(prompt="", ref_image=str(tmp_path), ref_vision=True)
    jobs = build_jobs(req)
    assert [(j["key"], j["width"], j["height"]) for j in jobs] == [
        ("000_b", 800, 800),  # never upscaled
        ("000_page_one", 864, 1216),
    ]
    assert all(j["prompt"] == "" and j["ref_vision"] for j in jobs)
    # A reference edit invalidates the saved latent.
    assert fingerprint(req, jobs[0])["ref_mtime"] == jobs[0]["ref_mtime"]

    # Any reference may stand alone (sanitize's fixed caption), vision or not;
    # without one an empty prompt is an error.
    alone = build_jobs(GenerateRequest(prompt="", ref_image=str(tmp_path)))
    assert [j["prompt"] for j in alone] == ["", ""]
    assert not any(j["ref_vision"] for j in alone)
    with pytest.raises(SystemExit, match="no prompts"):
        build_jobs(GenerateRequest(prompt=""))


def test_ref_prompts_and_seeds_render_an_eval_set_in_one_job(tmp_path):
    """A glob of ``<page>.src.webp`` takes each page's caption from ``<page>.txt``
    at the page's native size, once per seed."""
    from library.moss.generate import build_jobs

    for page, size in (("p1", (848, 1232)), ("p2", (1360, 768))):
        Image.new("RGB", size).save(tmp_path / f"{page}.src.webp")
        Image.new("RGB", size).save(tmp_path / f"{page}.tgt.webp")
        (tmp_path / f"{page}.txt").write_text(f"caption {page}\n")
    req = GenerateRequest(
        ref_image=str(tmp_path / "*.src.webp"),
        ref_prompts=True,
        seeds="0,2",
        width=1536,
        height=1536,
        prompt="ignored",
        prompt_format="text",
    )
    jobs = build_jobs(req)
    assert [
        (j["key"], j["seed"], j["width"], j["height"], j["prompt"]) for j in jobs
    ] == [
        ("000_p1.src", 0, 848, 1232, "caption p1"),
        ("000_p1.src", 2, 848, 1232, "caption p1"),
        ("000_p2.src", 0, 1360, 768, "caption p2"),
        ("000_p2.src", 2, 1360, 768, "caption p2"),
    ]
    (tmp_path / "p2.txt").unlink()
    with pytest.raises(SystemExit, match="no p2.txt"):
        build_jobs(req)
    with pytest.raises(SystemExit, match="needs ref_image"):
        build_jobs(GenerateRequest(ref_prompts=True))
