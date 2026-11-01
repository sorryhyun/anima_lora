"""MOSS quantized text encoders: the export's key set and scale, the dequantising
Linear, the file round trip, the TE cache's encoder stamp."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from library.moss.cache import (
    ENCODER_ID,
    TE_SUFFIX,
    Pair,
    _te_current,
    cache_text,
    encoder_id,
)
from library.moss.fp8 import (
    FP8,
    LINEARS,
    N_LINEARS,
    FP8Linear,
    is_quantized_linear,
    quantize_te,
    quantize_weight,
)
from library.moss.scan import caption_sha


def test_only_decoder_linears_are_quantized():
    assert is_quantized_linear("language_model.layers.0.mlp.gate_proj.weight")
    assert is_quantized_linear("language_model.layers.31.linear_attn.in_proj_a.weight")
    for key in (
        "language_model.layers.0.linear_attn.conv1d.weight",
        "language_model.layers.0.input_layernorm.weight",
        "language_model.layers.3.self_attn.q_norm.weight",
        "language_model.embed_tokens.weight",
        "visual.blocks.0.attn.qkv.weight",
        "visual.merger.linear_fc1.weight",
    ):
        assert not is_quantized_linear(key), key
    assert len(LINEARS) == 12 and N_LINEARS == 248


def test_quantize_weight_scale_and_error():
    w = torch.randn(64, 128) * 0.02
    q, scale = quantize_weight(w)
    assert q.dtype == FP8 and scale.dim() == 0
    assert q.float().abs().max() <= torch.finfo(FP8).max
    rel = (q.float() * scale - w).norm() / w.norm()
    assert rel < 0.04  # e4m3: 3 mantissa bits


def test_fp8_linear_is_the_dequantised_matmul():
    w = torch.randn(32, 64)
    q, scale = quantize_weight(w)
    lin = FP8Linear(q, scale)
    x = torch.randn(5, 64, dtype=torch.bfloat16)
    ref = F.linear(x, q.to(x.dtype) * scale.to(x.dtype))
    assert torch.equal(lin(x), ref)
    assert lin.weight.dtype == FP8 and (lin.out_features, lin.in_features) == (32, 64)


def _tiny_te_dir(tmp_path):
    """A two-shard checkpoint with one quantizable and two kept tensors per shard."""
    src = tmp_path / "text_encoder"
    src.mkdir()
    shards = {
        "model-00001-of-00002.safetensors": {
            "language_model.layers.0.mlp.up_proj.weight": torch.randn(8, 16).bfloat16(),
            "language_model.layers.0.linear_attn.A_log": torch.randn(4),
        },
        "model-00002-of-00002.safetensors": {
            "language_model.embed_tokens.weight": torch.randn(10, 16).bfloat16(),
        },
    }
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, str(src / name))
        weight_map |= {k: name for k in tensors}
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return src, shards


def test_quantize_te_writes_comfy_mixed_precision(tmp_path, monkeypatch):
    import library.moss.fp8 as fp8

    monkeypatch.setattr(fp8, "N_LINEARS", 1)
    src, shards = _tiny_te_dir(tmp_path)
    out = tmp_path / "te_fp8.safetensors"
    quantize_te(src, out, log=lambda *_: None)
    state = load_file(str(out))
    prefix = "language_model.layers.0.mlp.up_proj."
    assert state[f"{prefix}weight"].dtype == FP8
    assert state[f"{prefix}weight_scale"].dtype == torch.float32
    conf = json.loads(state[f"{prefix}comfy_quant"].numpy().tobytes())
    assert conf == {"format": "float8_e4m3fn", "full_precision_matrix_mult": True}
    # Everything else keeps its checkpoint dtype and value.
    a_log = shards["model-00001-of-00002.safetensors"][
        "language_model.layers.0.linear_attn.A_log"
    ]
    assert torch.equal(state["language_model.layers.0.linear_attn.A_log"], a_log)
    assert state["language_model.embed_tokens.weight"].dtype == torch.bfloat16
    with safe_open(str(out), "pt") as f:
        assert json.loads(f.metadata()["quantization"]) == conf


@pytest.mark.core
def test_encoder_stamp_reencodes_on_a_switch(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import library.moss.loader as loader

    assert encoder_id("bf16") == ENCODER_ID != encoder_id("fp8")
    path = tmp_path / f"a{TE_SUFFIX}"
    meta = {"caption_sha256": caption_sha("tags"), "max_text_tokens": "4096"}
    save_file({"embeds": torch.zeros(1, 4)}, path, metadata=meta)
    # A cache written before the stamp counts as bf16.
    assert _te_current(path, "tags", 4096)
    assert not _te_current(path, "tags", 4096, encoder=encoder_id("fp8"))

    seen = {}
    monkeypatch.setattr(loader, "load_tokenizer", lambda _d: None)
    monkeypatch.setattr(
        loader,
        "load_text_encoder",
        lambda *a, **k: (seen.update(k), (None, None))[1],
    )
    monkeypatch.setattr(
        loader, "prompt_ids", lambda *_a: SimpleNamespace(input_ids=torch.zeros(1, 3))
    )
    monkeypatch.setattr(loader, "encode_prompt", lambda *_a: torch.randn(3, 4096))
    monkeypatch.setattr(loader, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    pair = Pair("a", tmp_path / "a.png", "tags", caption_sha("tags"))
    report = cache_text([pair], tmp_path, "", "cpu", 4096, False, precision="fp8")
    assert report["encoded"] == 1 and seen["precision"] == "fp8"
    with safe_open(str(path), "pt") as f:
        assert f.metadata()["encoder"] == encoder_id("fp8")
    assert _te_current(path, "tags", 4096, encoder=encoder_id("fp8"))
