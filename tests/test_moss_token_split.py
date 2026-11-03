"""The Hangul token split: what it tokenizes, and that cache, LoRA and generate agree."""

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import library.moss.loader as loader
from library.moss.cache import Pair, _te_current, cache_text, caption_sha
from library.moss.generation.conditioning import cached_embeds
from library.moss.generation.jobs import fingerprint
from library.moss.requests import GenerateRequest
from library.moss.token_split import (
    effective_split,
    resolve_token_split,
    syllable_ids,
    with_token_split,
)
from library.moss.training.data import cache_token_split


@pytest.fixture
def tok():
    if not (loader.variant_dir(None, "base") / "tokenizer").exists():
        pytest.skip("MOSS tokenizer not downloaded")
    return loader.load_tokenizer(None)


def pieces(tok, text):
    return [tok.decode([i]) for i in syllable_ids(tok, text)]


def test_composite_and_byte_split_tokens_come_apart(tok):
    assert pieces(tok, "분인가요") == ["분", "인", "가", "요"]
    assert pieces(tok, "이건 뭔가요") == ["이", "건", " ", "뭔", "가", "요"]
    assert pieces(tok, "싸… 싼다") == ["싸", "…", " ", "싼", "다"]


def test_one_syllable_keeps_one_form_in_every_context(tok):
    ids = with_token_split(tok, "syllable")
    for a, b in (("싼다", " 싼다"), ("이건", "…이건"), ("인가요", "분인가요")):
        syllable = a[0] if a != "인가요" else "인"
        one = tok(syllable, add_special_tokens=False).input_ids
        assert one[0] in ids(a, return_tensors="pt").input_ids[0].tolist()
        assert one[0] in ids(b, return_tensors="pt").input_ids[0].tolist()


def test_a_syllable_with_no_token_stays_its_byte_form(tok):
    alone = tok("뜩", add_special_tokens=False).input_ids
    assert len(alone) > 1
    assert syllable_ids(tok, "뜩") == alone


def test_text_round_trips_through_the_chat_template(tok):
    formatted = loader.prompt_text(tok, '{"text":"어머? 처음 오시는 분인가요?"}')
    assert tok.decode(syllable_ids(tok, formatted)) == formatted


def test_prompt_without_hangul_tokenizes_as_the_default(tok):
    text = '{"text":"UGH… YOU\'RE THE BEST","bbox":[1,2,3,4]}'
    plain = loader.prompt_ids(tok, text)
    split = loader.prompt_ids(with_token_split(tok, "syllable"), text)
    assert torch.equal(plain.input_ids, split.input_ids)
    assert torch.equal(plain.attention_mask, split.attention_mask)


def test_truncation_keeps_max_length(tok):
    split = with_token_split(tok, "syllable")
    assert loader.prompt_ids(split, "분인가요" * 20, 12).input_ids.shape[1] == 12


def test_split_tokenizer_refuses_what_it_cannot_map(tok):
    split = with_token_split(tok, "syllable")
    with pytest.raises(TypeError):
        split("분인가요", return_offsets_mapping=True)


def test_effective_split_is_off_without_hangul():
    assert effective_split("syllable", "분인가요") == "syllable"
    assert effective_split("syllable", "ENGLISH") == "off"
    assert effective_split("off", "분인가요") == "off"


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


def test_switching_reencodes_only_captions_with_hangul(tmp_path, monkeypatch):
    seen = {}
    _cache_on_cpu(monkeypatch, seen)
    ko = Pair("ko", tmp_path / "ko.png", "분인가요", caption_sha("분인가요"))
    en = Pair("en", tmp_path / "en.png", "ENGLISH", caption_sha("ENGLISH"))
    cache_text([ko, en], tmp_path, "", "cpu", 4096, False)
    seen.clear()

    report = cache_text(
        [ko, en], tmp_path, "", "cpu", 4096, False, token_split="syllable"
    )

    assert report["encoded"] == 1 and list(seen) == ["분인가요"]
    assert seen["분인가요"]._tok == "tok"
    ko_te, en_te = (
        tmp_path / "ko.moss_te.safetensors",
        tmp_path / "en.moss_te.safetensors",
    )
    with safe_open(str(ko_te), "pt") as f:
        assert f.metadata()["token_split"] == "syllable"
    with safe_open(str(en_te), "pt") as f:
        assert "token_split" not in f.metadata()
    assert _te_current(ko_te, "분인가요", 4096, token_split="syllable")
    assert not _te_current(ko_te, "분인가요", 4096)
    assert _te_current(en_te, "ENGLISH", 4096, token_split="syllable")

    samples = [
        SimpleNamespace(te_path=None),
        SimpleNamespace(te_path=en_te),
        SimpleNamespace(te_path=ko_te),
    ]
    assert cache_token_split(samples) == "syllable"
    assert cache_token_split(samples[:2]) == "off"


def _lora(path, **meta):
    save_file({"x": torch.zeros(1)}, str(path), metadata=meta)
    return str(path)


def test_generate_follows_the_lora_stamp(tmp_path):
    stamped = _lora(tmp_path / "a.safetensors", token_split="syllable")
    plain = _lora(tmp_path / "b.safetensors")
    assert resolve_token_split(GenerateRequest(lora=stamped)) == "syllable"
    assert resolve_token_split(GenerateRequest(lora=plain)) == "off"
    assert resolve_token_split(GenerateRequest()) == "off"
    assert (
        resolve_token_split(GenerateRequest(lora=plain, token_split="syllable"))
        == "syllable"
    )
    assert (
        resolve_token_split(GenerateRequest(lora=stamped, token_split="off")) == "off"
    )


def test_ref_vision_refuses_the_split(tmp_path):
    with pytest.raises(SystemExit):
        resolve_token_split(GenerateRequest(token_split="syllable", ref_vision=True))


def test_fingerprint_tracks_the_split_and_leaves_old_latents_alone():
    job = dict(skip=(22,), prompt="p")
    off = fingerprint(GenerateRequest(), job)
    split = fingerprint(GenerateRequest(token_split="syllable"), job)
    assert "token_split" not in off and split["token_split"] == "syllable"


def test_prompt_cache_keys_by_split(tmp_path, monkeypatch):
    calls = []

    def encode(_m, texts, precision, token_split):
        calls.append(token_split)
        return {t: torch.zeros(1, 2) for t in texts}

    monkeypatch.setattr(
        "library.moss.generation.conditioning.encode_prompts",
        lambda m, t, precision, token_split: encode(m, t, precision, token_split),
    )
    cached_embeds(None, ["분인가요"], tmp_path, "int8")
    cached_embeds(None, ["분인가요"], tmp_path, "int8", "syllable")
    cached_embeds(None, ["분인가요"], tmp_path, "int8", "syllable")
    assert calls == ["off", "syllable"]
