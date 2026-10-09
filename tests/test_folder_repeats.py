"""Kohya-style ``repeat_by_folder_name``: a ``{n}_...`` directory component
under a subset's image_dir overrides ``num_repeats`` with n for the images
inside it (``0_...`` drops them). Off by default; training pool only.
"""

import os
from dataclasses import asdict

from library.config.loader import ConfigSanitizer, DreamBoothSubsetParams
from library.datasets.dreambooth import DreamBoothDataset
from library.datasets.subsets import DreamBoothSubset, folder_repeat_count


# ── folder_repeat_count parsing ─────────────────────────────────────────────


def test_parses_immediate_parent():
    assert folder_repeat_count(os.path.join("root", "5_char", "a.png"), "root") == 5


def test_no_match_returns_none():
    assert folder_repeat_count(os.path.join("root", "char", "a.png"), "root") is None
    # no underscore after the digits
    assert folder_repeat_count(os.path.join("root", "5char", "a.png"), "root") is None


def test_image_directly_in_root_returns_none():
    assert folder_repeat_count(os.path.join("root", "a.png"), "root") is None


def test_nested_inherits_from_ancestor():
    path = os.path.join("root", "5_char", "extra", "a.png")
    assert folder_repeat_count(path, "root") == 5


def test_deepest_matching_component_wins():
    path = os.path.join("root", "5_char", "3_sub", "a.png")
    assert folder_repeat_count(path, "root") == 3


def test_zero_prefix_returns_zero():
    assert folder_repeat_count(os.path.join("root", "0_skip", "a.png"), "root") == 0


def test_multi_digit_and_extra_underscores():
    assert folder_repeat_count(os.path.join("root", "12_a_b c", "a.png"), "root") == 12


def test_image_outside_root_returns_none():
    assert folder_repeat_count(os.path.join("other", "5_x", "a.png"), "root") is None


# ── config plumbing ─────────────────────────────────────────────────────────


def test_sanitizer_accepts_toggle_in_general_and_subset():
    s = ConfigSanitizer(support_dropout=True)
    cfg = {
        "general": {"repeat_by_folder_name": True},
        "datasets": [{"subsets": [{"image_dir": "x", "repeat_by_folder_name": False}]}],
    }
    assert s.sanitize_user_config(cfg) == cfg


def test_params_default_off_and_forward_to_subset():
    params = DreamBoothSubsetParams(image_dir="x", caption_extension=".txt")
    assert params.repeat_by_folder_name is False
    subset = DreamBoothSubset(**asdict(params))
    assert subset.repeat_by_folder_name is False


def test_dataset_level_key_ascends_to_subset():
    """The GUI writes its override on the [[datasets]] table (subset-level
    overrides aren't merged by _apply_dataset_overrides) — the blueprint
    fallback chain must carry it down to the subset params."""
    import argparse

    from library.config.loader import BlueprintGenerator

    gen = BlueprintGenerator(ConfigSanitizer(support_dropout=True))
    user_config = {
        "datasets": [{"repeat_by_folder_name": True, "subsets": [{"image_dir": "x"}]}]
    }
    blueprint = gen.generate(user_config, argparse.Namespace())
    params = blueprint.dataset_group.datasets[0].subsets[0].params
    assert params.repeat_by_folder_name is True


# ── GUI virtual-key encoding (Qt-free helpers) ──────────────────────────────


def test_gui_base_default_reads_dataset_table_and_general():
    from gui.core.validation import _base_folder_repeats

    assert _base_folder_repeats({}) is False
    assert _base_folder_repeats({"datasets": [{"repeat_by_folder_name": True}]})
    # subset fallback for hand-edited configs
    assert _base_folder_repeats(
        {"datasets": [{"subsets": [{"repeat_by_folder_name": True}]}]}
    )
    assert _base_folder_repeats({"general": {"repeat_by_folder_name": True}})


def test_gui_apply_writes_and_strips_override():
    from gui.core.validation import apply_folder_repeats_choice

    out = {}
    apply_folder_repeats_choice(out, True, base_enabled=False)
    assert out == {"datasets": [{"repeat_by_folder_name": True}]}

    # flipping back to the base value strips the override entirely
    apply_folder_repeats_choice(out, False, base_enabled=False)
    assert out == {}

    # other keys on the [[datasets]] block are preserved
    out = {"datasets": [{"validation_split_num": 16}]}
    apply_folder_repeats_choice(out, True, base_enabled=False)
    assert out["datasets"][0] == {
        "validation_split_num": 16,
        "repeat_by_folder_name": True,
    }
    apply_folder_repeats_choice(out, False, base_enabled=False)
    assert out == {"datasets": [{"validation_split_num": 16}]}


# ── dataset integration ─────────────────────────────────────────────────────


def _make_tree(tmp_path):
    """root/3_a/one.png, root/b/two.png, root/0_skip/three.png (+ captions)."""
    root = tmp_path / "images"
    for folder, stem in (("3_a", "one"), ("b", "two"), ("0_skip", "three")):
        d = root / folder
        d.mkdir(parents=True)
        (d / f"{stem}.png").touch()
        (d / f"{stem}.txt").write_text("a caption", encoding="utf-8")
    return root


def _make_dataset(root, *, enabled, is_training=True, num_repeats=2):
    subset = DreamBoothSubset(
        **asdict(
            DreamBoothSubsetParams(
                image_dir=str(root),
                num_repeats=num_repeats,
                repeat_by_folder_name=enabled,
                recursive=True,
                caption_extension=".txt",
                caption_separator=",",
                keep_tokens_separator="",
                mask_dir="",  # suppress mask_dir auto-resolution from CWD
            )
        )
    )
    return DreamBoothDataset(
        subsets=[subset],
        is_training_dataset=is_training,
        batch_size=1,
        network_multiplier=1.0,
        prior_loss_weight=1.0,
        debug_dataset=False,
        validation_split=0.0,
        validation_seed=None,
        resize_interpolation=None,
        validation_split_num=0,
    )


def _repeats_by_stem(dataset):
    return {
        os.path.splitext(os.path.basename(info.absolute_path))[0]: info.num_repeats
        for info in dataset.image_data.values()
    }


def test_dataset_folder_repeats_enabled(tmp_path):
    dataset = _make_dataset(_make_tree(tmp_path), enabled=True)
    repeats = _repeats_by_stem(dataset)
    # 3_a → 3; plain folder falls back to subset num_repeats; 0_skip dropped.
    assert repeats == {"one": 3, "two": 2}
    assert dataset.num_train_images == 5


def test_dataset_folder_repeats_disabled(tmp_path):
    dataset = _make_dataset(_make_tree(tmp_path), enabled=False)
    repeats = _repeats_by_stem(dataset)
    assert repeats == {"one": 2, "two": 2, "three": 2}
    assert dataset.num_train_images == 6


def test_validation_pool_ignores_folder_repeats(tmp_path):
    dataset = _make_dataset(_make_tree(tmp_path), enabled=True, is_training=False)
    repeats = _repeats_by_stem(dataset)
    # Validation always runs at 1 repeat; 0_* images are not dropped there.
    assert repeats == {"one": 1, "two": 1, "three": 1}
