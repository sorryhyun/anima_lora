"""MOSS trainer: the per-step caption choice, the edit-region loss and the tone term (CPU, no model)."""

import random

import torch

from library.moss.train import (
    choose_embeds,
    edit_region,
    ema_decay,
    ema_step,
    region_mse,
    tone_loss,
    tone_projection,
    training_loss,
)


def _rows(n):
    return torch.zeros(n, 4)


FULL, PLAIN, EMPTY = _rows(10), _rows(14), _rows(3)
VARIANTS = {1: _rows(6), 2: _rows(17), 3: _rows(9)}


def _rngs(seed=0):
    return dict(
        plain_rng=random.Random(seed + 1),
        drop_rng=random.Random(seed + 2),
        variant_rng=random.Random(seed + 3),
    )


def test_worst_case_is_the_longest_caption_and_never_dropped():
    loaded = []

    def load(k):
        loaded.append(k)
        return VARIANTS[k]

    rngs = _rngs()
    x, kind = choose_embeds(
        FULL,
        PLAIN,
        3,
        load,
        EMPTY,
        plain_rate=1.0,
        variant_rate=1.0,
        dropout=1.0,
        worst_case=True,
        **rngs,
    )
    assert (kind, x.shape[0]) == ("v2", 17)
    assert loaded == [1, 2, 3]
    # The fit step draws nothing, so the run after it is the run without it.
    fresh = _rngs()
    assert [r.random() for r in rngs.values()] == [r.random() for r in fresh.values()]

    x, kind = choose_embeds(FULL, PLAIN, worst_case=True)
    assert (kind, x.shape[0]) == ("plain", 14)
    x, kind = choose_embeds(_rows(20), PLAIN, worst_case=True)
    assert kind == "full"


def _kinds(n=4000, **kw):
    rngs = _rngs()
    counts = {}
    for _ in range(n):
        _, kind = choose_embeds(
            FULL, PLAIN, 3, VARIANTS.__getitem__, EMPTY, **kw, **rngs
        )
        kind = "variant" if kind.startswith("v") else kind
        counts[kind] = counts.get(kind, 0) + 1
    return {k: v / n for k, v in counts.items()}


def test_rates_compose():
    """Variants draw only on steps the plain caption did not take, so their share
    is (1 - plain_rate) · variant_rate; dropout then takes its share of all."""
    share = _kinds(plain_rate=0.2, variant_rate=0.5)
    assert abs(share["plain"] - 0.2) < 0.03
    assert abs(share["variant"] - 0.8 * 0.5) < 0.03
    assert abs(share["full"] - 0.8 * 0.5) < 0.03
    share = _kinds(plain_rate=0.0, variant_rate=1.0, dropout=0.25)
    assert abs(share["empty"] - 0.25) < 0.03
    assert abs(share["variant"] - 0.75) < 0.03 and "plain" not in share


def test_missing_candidates_draw_nothing():
    rngs = _rngs()
    for _ in range(5):
        x, kind = choose_embeds(
            FULL, plain_rate=1.0, variant_rate=1.0, dropout=1.0, **rngs
        )
        assert kind == "full" and x is FULL
    fresh = _rngs()
    assert [r.random() for r in rngs.values()] == [r.random() for r in fresh.values()]


def test_edit_region_drops_specks_and_dilates():
    ref = torch.zeros(128, 12, 12)
    latent = ref.clone()
    latent[:, 2:5, 2:5] = 1.0  # a 9-token edit
    latent[:, 9, 9] = 1.0  # a 1-token speck
    latent[:, 7, 1] = 0.1  # below threshold
    m = edit_region(latent, ref, 0.3, min_tokens=5, dilate=0)
    assert m.sum() == 9 and not m[9, 9] and not m[7, 1]
    grown = edit_region(latent, ref, 0.3, min_tokens=5, dilate=1)
    assert grown.sum() == 25 and grown[1:6, 1:6].all()


def test_region_mse_multiplies_each_side():
    target = torch.zeros(1, 128, 10, 10)
    pred = torch.zeros(1, 128, 10, 10)
    pred[:, :, :2, :2] = 2.0  # 4 region tokens with error 4
    pred[:, :, 9, :] = 1.0  # 10 rest tokens with error 1
    region = torch.zeros(10, 10, dtype=torch.bool)
    region[:2, :2] = True
    parts = {}
    loss = region_mse(pred, target, region, 2.0, 0.5, parts)
    assert torch.isclose(loss, torch.tensor((2 * 4 * 4 + 0.5 * 10) / 100))
    assert parts["region"] == 4.0 and abs(parts["rest"] - 10 / 96) < 1e-6
    assert torch.isclose(region_mse(pred, target, region, 1.0, 1.0), (pred**2).mean())


def test_tone_loss_constant_offset_pool0():
    offset = torch.linspace(-1, 1, 128)
    err = offset.view(1, 128, 1, 1).expand(1, 128, 8, 8).clone()
    rest = torch.ones(8, 8, dtype=torch.bool)
    rest[:3, :3] = False
    err[:, :, :3, :3] = 50.0  # edit-region error: excluded
    assert torch.isclose(tone_loss(err, rest, 0), offset.pow(2).mean())


def test_tone_loss_projects_to_lab():
    dirs = torch.randn(128, 3, generator=torch.Generator().manual_seed(0))
    proj = torch.linalg.pinv(dirs)
    c = torch.tensor([1.5, -0.5, 2.0])
    err = (dirs @ c).view(1, 128, 1, 1).expand(1, 128, 16, 16)
    rest = torch.ones(16, 16, dtype=torch.bool)
    assert torch.isclose(tone_loss(err, rest, 8, proj), c.pow(2).sum(), rtol=1e-4)


def test_tone_projection_b_weight_scales_b_error(tmp_path):
    from safetensors.torch import save_file

    dirs = torch.randn(128, 3, generator=torch.Generator().manual_seed(0))
    path = tmp_path / "dirs.safetensors"
    save_file({k: dirs[:, i].contiguous() for i, k in enumerate("Lab")}, str(path))
    c = torch.tensor([1.5, -0.5, 2.0])
    err = (dirs @ c).view(1, 128, 1, 1).expand(1, 128, 16, 16)
    rest = torch.ones(16, 16, dtype=torch.bool)
    loss = tone_loss(err, rest, 8, tone_projection(path, 3.0))
    assert torch.isclose(
        loss, torch.tensor(1.5**2 + 0.5**2 + (3 * 2.0) ** 2), rtol=1e-4
    )


def test_tone_loss_skips_cells_under_half_coverage():
    err = torch.zeros(1, 4, 16, 16)
    err[:, :, :8, :8] = 2.0  # cell 0
    err[:, :, :8, 8:] = 3.0  # cell 1, a quarter covered: skipped
    rest = torch.zeros(16, 16, dtype=torch.bool)
    rest[:8, :8] = True
    rest[:4, 8:12] = True
    assert torch.isclose(tone_loss(err, rest, 8), torch.tensor(4.0))
    assert tone_loss(err, torch.zeros(16, 16, dtype=torch.bool), 8) == 0


def test_ema_step_seeds_then_lerps():
    p = torch.full((2, 3), 1.0)
    shadows, staging = [[], []], [torch.empty(2, 3)]
    ema_step(shadows, staging, [p], [0.9, 0.5])
    for shadow in shadows:
        assert torch.equal(shadow[0], p) and shadow[0] is not staging[0]
    p.fill_(2.0)
    ema_step(shadows, staging, [p], [0.9, 0.5])
    torch.testing.assert_close(shadows[0][0], torch.full((2, 3), 1.1))
    torch.testing.assert_close(shadows[1][0], torch.full((2, 3), 1.5))


def test_ema_decay_from_span():
    assert abs(ema_decay(0.14, 240) - 0.97) < 1e-3
    assert abs(ema_decay(0.055, 600) - 0.97) < 1e-3
    assert ema_decay(0.1, 5) == 0.0  # a span under one update is no average


def test_training_loss_logs_output_space_grad_norms():
    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(1))

        def forward(self, x, t, e, return_dict=False, ref=None):
            return (x * self.w,)

    latent = torch.randn(1, 4, 8, 8)
    rest = torch.ones(8, 8, dtype=torch.bool)
    parts = {}
    tone = dict(rest=rest, pool=0, proj=None, weight=0.0, log_weight=0.5)
    loss = training_loss(Fake(), latent, None, 0.5, parts=parts, tone=tone)
    assert {"g_fm", "g_tone", "g_cos"} <= parts.keys()
    assert parts["g_fm"] > 0 and parts["g_tone"] > 0 and -1 <= parts["g_cos"] <= 1
    loss.backward()  # the logged grads leave the graph intact


def test_delta_row_ids_and_row_mask():
    from library.moss.train import delta_row_ids, row_mask

    assert delta_row_ids("text", 12, []) == list(range(12))
    assert delta_row_ids("caption", 12, []) == [3, 4]
    assert delta_row_ids("caption", 9, []) == []
    assert delta_row_ids("trigger", 12, [5, 6, 30]) == [5, 6]
    mask = row_mask(4, 12, [5, 6], n_ref=2)
    assert mask.shape == (1, 18, 1) and mask.dtype == torch.bfloat16
    assert mask[0, :, 0].nonzero().flatten().tolist() == [9, 10]
    assert row_mask(4, 12, []).sum() == 0


def test_sample_captions_and_load_neg(tmp_path):
    from safetensors.torch import save_file

    from library.moss.train import NEG_SUFFIX, load_neg, sample_captions

    (tmp_path / "a.png").touch()
    (tmp_path / "a.txt").write_text("1girl, @x (y), smile. On the left, cat.\n")
    te = tmp_path / "a.moss_te.safetensors"
    save_file(
        {"embeds": _rows(5), "embeds_plain": _rows(4)},
        str(te),
        metadata={
            "caption": '{"summary":{"artists":["@x (y)"]}}',
            "caption_sha256": "abc",
            "source": str(tmp_path / "a.png"),
            "caption_v1": '{"summary":{}}',
        },
    )
    caps = sample_captions(te)
    assert caps == {
        "full": '{"summary":{"artists":["@x (y)"]}}',
        "plain": "1girl, @x (y), smile. On the left, cat.",
        "v1": '{"summary":{}}',
        "empty": "",
    }
    import pytest

    with pytest.raises(SystemExit, match="needs"):
        load_neg(te)
    neg = tmp_path / ("a" + NEG_SUFFIX)
    save_file(
        {
            "embeds_pos": _rows(5),
            "embeds": _rows(3),
            "embeds_plain_pos": _rows(4),
            "embeds_plain": _rows(2),
        },
        str(neg),
        metadata={"caption_sha256": "zzz"},
    )
    with pytest.raises(SystemExit, match="stale"):
        load_neg(te)
    save_file(
        {"embeds_pos": _rows(5), "embeds": _rows(3)},
        str(neg),
        metadata={"caption_sha256": "abc"},
    )
    pair = load_neg(te)
    assert set(pair) == {"full"} and pair["full"][1].shape[0] == 3


def test_trigger_rows_cover_the_trigger_tokens():
    import pytest

    from library.moss.loader import (
        load_tokenizer,
        prompt_ids,
        trigger_rows,
        variant_dir,
    )

    if not (variant_dir(None, "base") / "tokenizer").exists():
        pytest.skip("MOSS tokenizer not downloaded")
    tok = load_tokenizer(None)
    caption = '{"summary":{"artists":["@channel (caststation)"],"caption":"1girl"}}'
    rows = trigger_rows(tok, caption, "@channel (caststation)")
    ids = prompt_ids(tok, caption).input_ids[0]
    assert rows == list(range(rows[0], rows[-1] + 1)) and rows[0] >= 3
    assert "@channel (caststation)" in tok.decode(ids[rows])
    assert "@channel (caststation)" not in tok.decode(ids[: rows[0]])
    assert trigger_rows(tok, caption, "absent") == []
