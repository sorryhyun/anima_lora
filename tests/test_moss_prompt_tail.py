"""The assistant-turn prompt tail: what it appends, and that cache, LoRA and generate agree."""

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import library.moss.loader as loader
from library.moss.cache import Pair, _te_current, cache_text, caption_sha
from library.moss.generation.conditioning import cached_embeds
from library.moss.generation.jobs import fingerprint
from library.moss.prompt_tail import resolve_prompt_tail, with_prompt_tail
from library.moss.requests import GenerateRequest
from library.moss.token_split import with_token_split
from library.moss.training.data import cache_prompt_tail


@pytest.fixture
def tok():
    if not (loader.variant_dir(None, "base") / "tokenizer").exists():
        pytest.skip("MOSS tokenizer not downloaded")
    return loader.load_tokenizer(None)


def ids(tok, text):
    return loader.prompt_ids(tok, text).input_ids[0].tolist()


def test_sure_adds_four_rows_after_the_caption(tok):
    cap = "safe, 1girl, @channel (caststation)"
    shipped, tailed = ids(tok, cap), ids(with_prompt_tail(tok, "sure"), cap)
    assert len(tailed) == len(shipped) + 4
    assert tailed[: len(shipped) - 1] == shipped[:-1]
    assert tok.convert_ids_to_tokens(tailed[len(shipped) - 1 :]) == [
        "ĊĊ",
        "</think>",
        "ĊĊ",
        "Sure",
        ",",
    ]


def test_tail_composes_under_the_syllable_split(tok):
    cap = "분인가요"
    split = ids(with_token_split(tok, "syllable"), cap)
    both = ids(with_token_split(with_prompt_tail(tok, "sure"), "syllable"), cap)
    assert len(both) == len(split) + 4 and both[: len(split) - 1] == split[:-1]


def test_off_is_the_tokenizer_itself(tok):
    assert with_prompt_tail(tok, "off") is tok
    with pytest.raises(ValueError):
        with_prompt_tail(tok, "ok")


def _cache_on_cpu(monkeypatch, seen):
    monkeypatch.setattr(loader, "load_tokenizer", lambda _d: "tok")
    monkeypatch.setattr(loader, "load_text_encoder", lambda *a, **k: (k["tok"], None))
    monkeypatch.setattr(
        loader, "prompt_ids", lambda *_a: SimpleNamespace(input_ids=torch.zeros(1, 3))
    )

    def encode(tok, _te, text, *_a):
        seen[text] = tok
        return torch.randn(3, 4096)

    monkeypatch.setattr(loader, "encode_prompt", encode)
    monkeypatch.setattr(loader, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)


def test_switching_the_tail_reencodes_and_stamps(tmp_path, monkeypatch):
    seen = {}
    _cache_on_cpu(monkeypatch, seen)
    pair = Pair("a", tmp_path / "a.png", "CAP", caption_sha("CAP"))
    cache_text([pair], tmp_path, "", "cpu", 4096, False)
    te = tmp_path / "a.moss_te.safetensors"
    assert cache_prompt_tail([SimpleNamespace(te_path=te)]) == "off"
    seen.clear()

    report = cache_text([pair], tmp_path, "", "cpu", 4096, False, prompt_tail="sure")

    assert report["encoded"] == 1 and seen["CAP"]._tail == "\n</think>\n\nSure,"
    with safe_open(str(te), "pt") as f:
        assert f.metadata()["prompt_tail"] == "sure"
    assert _te_current(te, "CAP", 4096, prompt_tail="sure")
    assert not _te_current(te, "CAP", 4096)
    assert (
        cache_prompt_tail([SimpleNamespace(te_path=None), SimpleNamespace(te_path=te)])
        == "sure"
    )


def test_training_refuses_a_cache_mixing_tails(tmp_path):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    save_file({"x": torch.zeros(1)}, str(a), metadata={"prompt_tail": "sure"})
    save_file({"x": torch.zeros(1)}, str(b))
    with pytest.raises(SystemExit):
        cache_prompt_tail([SimpleNamespace(te_path=a), SimpleNamespace(te_path=b)])


def test_generate_follows_the_lora_stamp(tmp_path):
    stamped = tmp_path / "a.safetensors"
    plain = tmp_path / "b.safetensors"
    save_file({"x": torch.zeros(1)}, str(stamped), metadata={"prompt_tail": "sure"})
    save_file({"x": torch.zeros(1)}, str(plain))
    assert resolve_prompt_tail(GenerateRequest(lora=str(stamped))) == "sure"
    assert resolve_prompt_tail(GenerateRequest(lora=str(plain))) == "off"
    assert (
        resolve_prompt_tail(GenerateRequest(lora=str(stamped), prompt_tail="off"))
        == "off"
    )
    assert resolve_prompt_tail(GenerateRequest(prompt_tail="sure")) == "sure"
    with pytest.raises(SystemExit):
        resolve_prompt_tail(GenerateRequest(prompt_tail="sure", ref_vision=True))


def test_fingerprint_and_prompt_cache_key_by_tail(tmp_path, monkeypatch):
    job = dict(skip=(22,), prompt="p")
    assert "prompt_tail" not in fingerprint(GenerateRequest(), job)
    assert (
        fingerprint(GenerateRequest(prompt_tail="sure"), job)["prompt_tail"] == "sure"
    )

    calls = []

    def encode(_m, texts, precision, token_split, prompt_tail):
        calls.append(prompt_tail)
        return {t: torch.zeros(1, 2) for t in texts}

    monkeypatch.setattr("library.moss.generation.conditioning.encode_prompts", encode)
    cached_embeds(None, ["p"], tmp_path, "int8")
    cached_embeds(None, ["p"], tmp_path, "int8", prompt_tail="sure")
    cached_embeds(None, ["p"], tmp_path, "int8", prompt_tail="sure")
    assert calls == ["off", "sure"]
