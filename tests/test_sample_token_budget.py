"""Sample-prompt token counts feeding the torch.compile budget (issue #42).

A training run compiled for the buckets the dataset populates (e.g. the 1024
tier's 4032/4200 range) crashed mid-training when a sample prompt requested
1024x1536 — 6144 tokens, outside the dynamic-seq mark_dynamic range. The fix
folds the sample prompts' token counts into the compile budget
(library/anima/trainer/data.py::sample_prompt_token_counts → token_counts_for_sample_prompts);
these tests lock the pure count/snap math.
"""

from library.datasets.buckets import (
    snap_sample_size,
    token_counts_for_resos,
    token_counts_for_sample_prompts,
)


def test_union_with_bucket_counts_widens_range():
    # The train.py call site unions bucket counts with sample counts; the
    # issue's buckets (4032/4200 families) plus the 1024x1536 sample must yield
    # a range covering 6144.
    bucket_counts = token_counts_for_resos(
        [(768, 1344), (800, 1344), (896, 1200), (1344, 768), (1344, 800)]
    )
    assert bucket_counts == {4032, 4200}
    merged = bucket_counts | token_counts_for_sample_prompts(
        [{"width": 1024, "height": 1536}]
    )
    assert (min(merged), max(merged)) == (4032, 6144)
    assert len(merged) == 3


def test_snap_matches_inference_formula():
    # snap_sample_size is the shared definition of _sample_image_inference's
    # pre-sampling snap: dim → max(64, dim - dim % 16).
    assert snap_sample_size(1000, 1000) == (992, 992)
    assert snap_sample_size(1024, 1536) == (1024, 1536)
    assert snap_sample_size(10, 30) == (64, 64)
    counts = token_counts_for_sample_prompts([{"width": 1000, "height": 1000}])
    assert counts == {(992 // 16) * (992 // 16)}
