"""The sm_120 FlashAttention build loads on that card alone, and only once unpacked."""

import platform
import sys

import pytest
import torch

from library.runtime import fa2


@pytest.fixture
def unpacked(tmp_path, monkeypatch):
    monkeypatch.setattr(fa2, "_active", None)
    monkeypatch.setattr(fa2, "build_dir", lambda: tmp_path)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "flash_attn_2_cuda", raising=False)
    monkeypatch.setattr(torch, "__version__", "2.14.1+cu132")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (12, 0))
    (tmp_path / "flash_attn").mkdir()
    (tmp_path / "flash_attn_2_cuda.py").touch()
    yield tmp_path
    sys.modules.pop("flash_attn_2_cuda", None)


def test_sm120_goes_first(unpacked):
    assert fa2.use_sm120_build()
    assert sys.path[0] == str(unpacked)


def test_windows_goes_first(unpacked, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    assert fa2.use_sm120_build()


def test_failed_load_keeps_the_pin(unpacked):
    (unpacked / "flash_attn_2_cuda.py").write_text("raise ImportError('dll')")
    assert not fa2.use_sm120_build()
    assert str(unpacked) not in sys.path


def test_not_unpacked(unpacked):
    (unpacked / "flash_attn").rmdir()
    assert not fa2.use_sm120_build()
    assert str(unpacked) not in sys.path


@pytest.mark.parametrize("cc", [(8, 9), (9, 0), (10, 0)])
def test_other_cards_keep_the_pin(unpacked, monkeypatch, cc):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: cc)
    assert not fa2.use_sm120_build()
    assert str(unpacked) not in sys.path


def test_other_torch_minor_keeps_the_pin(unpacked, monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.15.0+cu132")
    assert not fa2.use_sm120_build()


def test_no_card_keeps_the_pin(unpacked, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not fa2.use_sm120_build()


def test_pinned_wheel_already_imported(unpacked, monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn_2_cuda", object())
    assert not fa2.use_sm120_build()
