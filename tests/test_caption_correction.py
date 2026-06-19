from __future__ import annotations

from pathlib import Path

from library.captioning.correction import (
    CaptionCorrectionOptions,
    correct_caption,
    default_tag_csv_candidates,
    find_tag_csv,
    load_tag_knowledge_base,
)


def _csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "name,category,post_count,description",
                '1girl,0,10,"[인물 > 인원수] count"',
                'solo,0,10,"[인물 > 인원수] count"',
                'hatsune_miku,4,10,"[캐릭터 > vocaloid] character"',
                'vocaloid,3,10,"[작품 > series] copyright"',
                'sincos,1,10,"[작가 > illustrator] artist"',
                'best_quality,5,10,"[메타 > 화질] quality"',
                'long_hair,0,10,"[머리카락 > 머리 길이] general"',
                'copyright_notice,0,10,"[메타 > 정보_요청] misleading description"',
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_correct_caption_orders_known_sections_and_preserves_general_order(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, vocaloid, @sincos, hatsune miku, 1girl, best quality",
        kb,
        options=CaptionCorrectionOptions(insert_no_artist=True),
    )

    assert result.text == (
        "best quality, 1girl, hatsune miku, vocaloid, @sincos, long hair"
    )
    assert result.changed
    assert not result.inserted_no_artist


def test_correct_caption_inserts_no_artist_at_artist_position(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption("long hair, vocaloid, hatsune miku, solo", kb)

    assert result.text == "solo, hatsune miku, vocaloid, @no-artist, long hair"
    assert result.inserted_no_artist


def test_trigger_defaults_to_artist_slot_and_suppresses_no_artist(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, @sincos, vocaloid, hatsune miku, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            trigger_word="@dataset-trigger",
        ),
    )

    assert result.text == (
        "1girl, hatsune miku, vocaloid, @dataset-trigger, @sincos, long hair"
    )
    assert not result.inserted_no_artist


def test_existing_trigger_is_moved_without_duplication(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, vocaloid, @dataset-trigger, hatsune miku, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            trigger_word="@dataset-trigger",
        ),
    )

    assert result.text == "1girl, hatsune miku, vocaloid, @dataset-trigger, long hair"


def test_front_trigger_allows_no_artist_when_no_artist_marker(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "long hair, vocaloid, hatsune miku, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            trigger_word="@dataset-trigger",
            trigger_at_front=True,
        ),
    )

    assert result.text == (
        "@dataset-trigger, 1girl, hatsune miku, vocaloid, @no-artist, long hair"
    )
    assert result.inserted_no_artist


def test_artist_validation_keeps_unknown_at_tag_in_general_tail(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "@trigger-word, hatsune miku, vocaloid, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            validate_artist_tags=True,
        ),
    )

    assert result.text == (
        "1girl, hatsune miku, vocaloid, @no-artist, @trigger-word"
    )


def test_artist_validation_does_not_reclassify_at_prefixed_character(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "@hatsune miku, vocaloid, 1girl",
        kb,
        options=CaptionCorrectionOptions(
            insert_no_artist=True,
            validate_artist_tags=True,
        ),
    )

    assert result.text == "1girl, vocaloid, @no-artist, @hatsune miku"


def test_numeric_category_wins_over_description_prefix(tmp_path):
    kb = load_tag_knowledge_base(_csv(tmp_path / "tags.csv"))
    result = correct_caption(
        "copyright notice, hatsune miku, vocaloid, 1girl",
        kb,
    )

    assert result.text == (
        "1girl, hatsune miku, vocaloid, @no-artist, copyright notice"
    )


def test_default_tag_csv_prefers_models_dir_over_env(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    model_csv = root / "models" / "danbooru_tags_classified.csv"
    env_csv = tmp_path / "env.csv"
    model_csv.parent.mkdir(parents=True)
    model_csv.write_text("name,category,post_count,description\n", encoding="utf-8")
    env_csv.write_text("name,category,post_count,description\n", encoding="utf-8")
    monkeypatch.setenv("ANIMA_DANBOORU_TAGS_CSV", str(env_csv))

    candidates = default_tag_csv_candidates(root)

    assert candidates[0] == model_csv
    assert root.parent / "danbooru_tags_classified.csv" not in candidates
    assert find_tag_csv(root) == model_csv
