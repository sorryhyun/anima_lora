"""Sample renders during MOSS training: prompt parsing, the epoch schedule, and the
skip hook block swap needs when the uncond branch skips a block (CPU)."""

import json
from types import SimpleNamespace

import torch

from library.moss.model import MOSSImageTransformer
from library.moss.training.sampling import Sampler, parse_prompts
from library.runtime.blockswap import Attached


def test_parse_prompts_inline_and_files(tmp_path):
    assert parse_prompts("") == []
    assert parse_prompts(" a girl \n\n a boy \n") == ["a girl", "a boy"]
    txt = tmp_path / "p.txt"
    txt.write_text("one\n\ntwo\n", encoding="utf-8")
    assert parse_prompts(str(txt)) == ["one", "two"]
    js = tmp_path / "p.json"
    js.write_text(json.dumps(["multi\nline", " ", "x"]), encoding="utf-8")
    assert parse_prompts(str(js)) == ["multi\nline", "x"]


def test_due_every_n_and_the_last_epoch():
    s = Sampler.__new__(Sampler)
    s.req = SimpleNamespace(sample_every_epochs=3)
    assert [e for e in range(1, 9) if s.due(e, 8)] == [3, 6, 8]
    s.req = SimpleNamespace(sample_every_epochs=0)
    assert not any(s.due(e, 8) for e in range(1, 9))


def _tiny(layers=4):
    torch.manual_seed(0)
    return MOSSImageTransformer(
        dim=96,
        ffn_dim=128,
        num_heads=4,
        num_layers=layers,
        text_dim=32,
        latent_channels=8,
    ).eval()


def test_skip_hook_sees_each_skipped_block():
    model = _tiny()
    seen = []
    model.skip_hook = seen.append
    g = torch.Generator().manual_seed(1)
    x, e = torch.randn(2, 8, 4, 6, generator=g), torch.randn(2, 5, 32, generator=g)
    mask = torch.ones(2, 5, dtype=torch.bool)
    with torch.no_grad():
        model(x[:1], torch.tensor([600.0]), e[:1], skip_layers=[2])
        assert seen == [2]
        # Batched: a block skipped by every row is skipped by the forward.
        model.forward_batched(x, 500.0, e, mask, skip_layers=[(1,), (1, 2)])
        assert seen == [2, 1]


def test_attached_skip_waits_and_moves_as_the_hooks_would():
    calls = []
    offloader = SimpleNamespace(
        wait_for_block=lambda i: calls.append(("wait", i)),
        _submit_move_blocks=lambda blocks, a, b: calls.append(("move", a, b)),
    )
    attached = Attached(offloader, blocks=[], model=None)
    attached.schedule = {22: (22, 28)}
    attached.loaded = {22, 28}
    attached.skip(22)
    attached.skip(5)  # no hooks on it
    assert calls == [("wait", 22), ("move", 22, 28)]
