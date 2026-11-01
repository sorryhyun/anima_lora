"""__random__ pool substitution in scripts/toolkits/comfy_batch.py."""

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    """comfy_batch lives in scripts/toolkits/, which is not an installed package."""
    spec = importlib.util.spec_from_file_location(
        "comfy_batch", ROOT / "scripts" / "toolkits" / "comfy_batch.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["comfy_batch"] = mod
    spec.loader.exec_module(mod)
    return mod


cb = _load_module()


def _wf(text):
    return {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": text}}}


def _text(wf):
    return wf["1"]["inputs"]["text"]


def _tags(entries):
    return [", ".join(e.tags) for e in entries]


@pytest.fixture
def pool(tmp_path):
    (tmp_path / "r.yaml").write_text(
        "default: [a]\na: [alpha, beta]\nb: [gamma, delta]\nunder_score: [eps]\n"
    )
    return cb.load_randoms(str(tmp_path / "r.yaml"))


def test_group_union_and_dedup(pool):
    assert _tags(pool.pool_for("a|b")) == ["alpha", "beta", "gamma", "delta"]
    assert _tags(pool.pool_for("a|a")) == ["alpha", "beta"]


def test_group_name_may_contain_underscore(pool):
    wf, draws = cb.substitute_random(_wf("__random:under_score__"), pool)
    assert _text(wf) == "eps" and draws == ["eps"]


def test_bare_random_draws_from_default_only(pool):
    for _ in range(20):
        wf, _d = cb.substitute_random(_wf("__random__"), pool)
        assert _text(wf) in ("alpha", "beta")


def test_draws_without_replacement_within_a_job(pool):
    for _ in range(20):
        wf, draws = cb.substitute_random(_wf("__random__ / __random__"), pool)
        assert sorted(draws) == ["alpha", "beta"], _text(wf)


def test_exhausted_pool_repeats_rather_than_starving(pool):
    wf, draws = cb.substitute_random(_wf("__random__ __random__ __random__"), pool)
    assert len(draws) == 3 and "__random__" not in _text(wf)


def test_apply_prompt_targets_a_random_only_node():
    wf = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "1girl, __random__"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "worst quality"}},
    }
    cb.apply_prompt(wf, "replaced")
    assert wf["1"]["inputs"]["text"] == "replaced"
    assert wf["2"]["inputs"]["text"] == "worst quality"


# --- shipped-pool invariants -------------------------------------------------
# workflows/ is gitignored: both the generated pool (randoms.yaml) and the
# build spec (randoms_build.yaml) exist only on the working machine. The spec
# also carries every corpus-specific expectation these tests assert, so the
# committed test file stays content-free. Without the files, skip.

SPEC_PATH = ROOT / "workflows" / "randoms_build.yaml"
SHIPPED_PATH = ROOT / "workflows" / "randoms.yaml"
HAVE_SHIPPED = SPEC_PATH.exists() and SHIPPED_PATH.exists()
shipped_only = pytest.mark.skipif(
    not HAVE_SHIPPED, reason="gitignored workflows/ pool + build spec not present"
)
if HAVE_SHIPPED:
    import yaml

    SPEC = yaml.safe_load(SPEC_PATH.read_text())
    SHIPPED = cb.load_randoms(str(SHIPPED_PATH))
else:
    SPEC, SHIPPED = {}, None


@shipped_only
def test_shipped_position_always_names_an_act():
    """The bug this pool was rebuilt for: a position draw that isn't an act.

    Every position cluster must carry a tag that says what is happening —
    otherwise the prompt renders a pose with no scene.
    """
    acts = set(SPEC["position_acts"]) | set(SPEC["core_companions"])
    for entry in SHIPPED.pool_for(SPEC["position_slot"]):
        assert acts & set(entry.tags), f"no act in position cluster {entry.tags}"


@shipped_only
def test_avoid_blocks_incompatible_draws():
    """An entry's avoid-list can never land in the same prompt as its anchor.

    The probe anchor must appear in exactly one entry across the whole pool,
    so seeing it drawn identifies the entry whose avoid-list applies.
    """
    from collections import Counter

    tag_count = Counter(
        t for entries in SHIPPED.groups.values() for e in entries for t in e.tags
    )
    probes = [
        e
        for e in SHIPPED.pool_for(SPEC["position_slot"])
        if e.avoid and tag_count[e.tags[0]] == 1
    ]
    assert probes, "no position entry with an avoid-list and a unique anchor"
    wf = _wf(", ".join(f"__random:{slot}__" for slot in SPEC["slots"]))
    for _ in range(120):
        out, _d = cb.substitute_random(wf, SHIPPED)
        tags = set(_text(out).split(", "))
        for probe in probes:
            if probe.tags[0] in tags:
                assert not tags & probe.avoid, (probe.tags[0], tags & probe.avoid)


@shipped_only
def test_pinned_prompt_tags_are_not_redrawn():
    """A tag the template already states is not repeated by a draw."""
    pinned = list(dict.fromkeys(SPEC["pinned"] + SPEC["context"]))
    wf = _wf(", ".join(pinned) + f", __random:{SPEC['position_slot']}__")
    for _ in range(40):
        out, draws = cb.substitute_random(wf, SHIPPED)
        rendered = _text(out).split(", ")
        assert len(rendered) == len(set(rendered)), rendered


def test_weighted_draw_biases_toward_heavy_entries(tmp_path):
    (tmp_path / "r.yaml").write_text(
        "a:\n  - tags: hot\n    weight: 9.0\n  - tags: cold\n    weight: 1.0\n"
    )
    p = cb.load_randoms(str(tmp_path / "r.yaml"))
    random.seed(3)
    hot = sum(
        cb.substitute_random(_wf("__random:a__"), p)[1] == ["hot"] for _ in range(400)
    )
    assert 320 <= hot <= 400, hot  # E[hot] = 360
