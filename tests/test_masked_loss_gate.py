"""``masked_loss = false`` must make a dataset maskless even when a mask tree
is on disk — the gate has to act before the datasets are built, because
construction bakes ``image_info.mask_path`` and ``make_buckets`` preloads the
PNGs, and ``__getitem__`` honours the preloaded mask whatever the subset
flags say afterwards. Caught by review on the v2 branch (2026-09-10)."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
from PIL import Image

from library.config.loader import (
    DatasetBlueprint,
    DatasetGroupBlueprint,
    DreamBoothDatasetParams,
    DreamBoothSubsetParams,
    SubsetBlueprint,
    disable_masks_in_blueprint,
    generate_dataset_group_by_blueprint,
)


def _write_png(path, size=(64, 64), value=255):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((size[1], size[0], 3), value, dtype=np.uint8)).save(path)


def _tree(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    for stem in ("a", "b"):
        _write_png(images / f"{stem}.png")
        (images / f"{stem}.txt").write_text("a caption", encoding="utf-8")
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(
            masks / f"{stem}_mask.png"
        ) if masks.mkdir(parents=True, exist_ok=True) is None else None
    return images, masks


def _blueprint(images, masks):
    return DatasetGroupBlueprint(
        datasets=[
            DatasetBlueprint(
                params=DreamBoothDatasetParams(batch_size=1),
                subsets=[
                    SubsetBlueprint(
                        params=DreamBoothSubsetParams(
                            image_dir=str(images),
                            caption_extension=".txt",
                            caption_separator=",",
                            keep_tokens_separator="",
                            mask_dir=str(masks),
                        )
                    )
                ],
            )
        ]
    )


def _build(blueprint):
    train, _val = generate_dataset_group_by_blueprint(blueprint)
    (dataset,) = train.datasets
    return dataset


def _infos(dataset):
    return list(dataset.image_data.values())


def _alpha_mask_of(dataset, info):
    """What the batch would carry for ``info`` (``__getitem__`` minus the
    tokenizer it would also need)."""
    subset = dataset.image_to_subset[info.image_key]
    _image, _latents, alpha_mask, _size, _crop = dataset._load_sample(
        subset, info, flipped=False
    )
    return alpha_mask


def test_mask_tree_on_disk_is_used_when_masked_loss_is_on(tmp_path):
    """The control: without the gate the mask tree is picked up and preloaded,
    which is exactly what would leak through when masked_loss is off."""
    images, masks = _tree(tmp_path)
    dataset = _build(_blueprint(images, masks))

    assert all(info.mask_path for info in _infos(dataset))
    assert all(info.preloaded_alpha_mask is not None for info in _infos(dataset))
    assert all(_alpha_mask_of(dataset, info) is not None for info in _infos(dataset))


def test_masked_loss_off_gates_before_construction(tmp_path):
    images, masks = _tree(tmp_path)
    blueprint = _blueprint(images, masks)

    assert disable_masks_in_blueprint(blueprint) == [str(masks)]
    (subset_params,) = [s.params for s in blueprint.datasets[0].subsets]
    assert subset_params.mask_dir == "" and subset_params.alpha_mask is False

    dataset = _build(blueprint)
    assert all(info.mask_path is None for info in _infos(dataset))
    assert all(info.preloaded_alpha_mask is None for info in _infos(dataset))
    assert not any(s.alpha_mask for s in dataset.subsets)
    assert all(_alpha_mask_of(dataset, info) is None for info in _infos(dataset))


def test_gate_is_idempotent_and_quiet_without_masks(tmp_path):
    images, masks = _tree(tmp_path)
    blueprint = _blueprint(images, masks)
    disable_masks_in_blueprint(blueprint)
    assert disable_masks_in_blueprint(blueprint) == []


@pytest.mark.parametrize("legacy", ["masks/merged", "post_image_dataset/masks"])
def test_empty_mask_dir_suppresses_legacy_auto_resolution(
    tmp_path, monkeypatch, legacy
):
    """``mask_dir=""`` (what the gate writes) must not fall through to the
    constructor's CWD-relative ``masks/{merged,sam}`` lookup."""
    from library.datasets.subsets import DreamBoothSubset

    images, _masks = _tree(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / legacy).mkdir(parents=True)
    subset = DreamBoothSubset(
        **asdict(
            DreamBoothSubsetParams(
                image_dir=str(images),
                caption_extension=".txt",
                caption_separator=",",
                keep_tokens_separator="",
                mask_dir="",
            )
        )
    )
    assert not subset.mask_dir and subset.alpha_mask is False
