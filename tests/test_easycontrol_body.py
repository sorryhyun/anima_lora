"""EasyControl body LoRA (plan_render S6) — build, step-0 equivalence, round-trip.

Contract (networks/methods/easycontrol.py, "Body LoRA"):
- off by default: no target_lora_* / adapter_lora.* / ext_lora_* keys,
- target-stream sublayers with zero-init LoRA are bit-exact to the frozen modules
  (self-attn qkv/out, cross-attn q/kv, mlp), and each delta moves only its own path,
- llm_adapter Linears are wrapped on apply and restored on remove,
- the ext-row delta lands only on ext-id positions, on top of the pack rows,
- body params get their own lr group when target_lr is set,
- checkpoint round-trip rebuilds every body module from the weights alone.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from library.anima.ext_vocab import T5_TABLE_SIZE
from library.anima.models import Attention, GPT2FeedForward
from library.anima.vocab_pack import attach_vocab_pack, detach_vocab_pack
from networks import attention_dispatch
from networks.methods.easycontrol import (
    _LoRAProj,
    _target_cross_attn,
    _target_mlp,
    _target_out_proj,
    _target_self_qkv,
    create_network,
    create_network_from_weights,
)


def _make_network(unet=None, **kwargs):
    return create_network(1.0, 16, 16, None, [], unet, **kwargs)


class _FakeAdapter(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.embed = nn.Embedding(T5_TABLE_SIZE, dim)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, 2 * dim), nn.Linear(2 * dim, dim))]
        )


def _fake_pack(rows=3, dim=4):
    return SimpleNamespace(
        table=torch.randn(rows, dim), digest="fake", name="fake", rows=rows
    )


def test_body_off_by_default():
    net = _make_network()
    assert net.train_target is False
    assert net.adapter_lora is None and net.ext_rows == 0
    assert not any(
        k.startswith(("target_lora_", "adapter_lora.", "ext_lora_"))
        for k in net.state_dict()
    )
    meta = net.metadata_fields()
    assert meta["ss_train_target"] == "0" and meta["ss_ext_rows"] == "0"
    params, names = net.prepare_optimizer_params_with_multiple_te_lrs(None, None, 1e-3)
    assert len(params) == 1 and len(names) == 1


def _loras(D=16, ctx=12, ffn=64, r=2):
    return tuple(
        _LoRAProj(i, o, r, float(r))
        for i, o in ((D, 3 * D), (D, D), (D, D), (ctx, 2 * D), (D, ffn), (ffn, D))
    )


def test_target_sublayers_zero_init_bit_exact_and_isolated():
    torch.manual_seed(0)
    D, ctx, ffn = 16, 12, 64
    self_attn = Attention(D, None, 2, 8)
    cross_attn = Attention(D, ctx, 2, 8)
    mlp = GPT2FeedForward(D, ffn)
    params = attention_dispatch.AttentionParams.create_attention_params("torch", None)
    x = torch.randn(1, 5, D)
    context = torch.randn(1, 7, ctx)
    loras = _loras(D, ctx, ffn)

    base_qkv = self_attn.compute_qkv(x, x)
    for b, z in zip(base_qkv, _target_self_qkv(self_attn, x, None, loras, 1.0)):
        assert torch.equal(b, z)
    assert torch.equal(
        self_attn.output_proj(x), _target_out_proj(self_attn, x, loras, 1.0)
    )
    base_cross = cross_attn(x, params, context)
    assert torch.equal(
        base_cross, _target_cross_attn(cross_attn, x, params, context, None, loras, 1.0)
    )
    assert torch.equal(mlp(x), _target_mlp(mlp, x, loras, 1.0))

    # Bump only the cross-attn kv delta: cross moves, self-attn and mlp don't.
    nn.init.ones_(loras[3].lora_up.weight)
    bumped = _target_cross_attn(cross_attn, x, params, context, None, loras, 1.0)
    assert not torch.equal(base_cross, bumped)
    for b, z in zip(base_qkv, _target_self_qkv(self_attn, x, None, loras, 1.0)):
        assert torch.equal(b, z)
    assert torch.equal(mlp(x), _target_mlp(mlp, x, loras, 1.0))
    # Multiplier 0 gates the delta off.
    assert torch.equal(
        base_cross, _target_cross_attn(cross_attn, x, params, context, None, loras, 0.0)
    )


def test_adapter_lora_wraps_and_restores_linears():
    torch.manual_seed(0)
    adapter = _FakeAdapter()
    unet = SimpleNamespace(llm_adapter=adapter)
    net = _make_network(unet, train_llm_adapter="true", target_rank="2")
    assert set(net.adapter_lora.keys()) == {"blocks_0_0", "blocks_0_1"}
    x = torch.randn(3, 4)
    base = adapter.blocks[0](x)

    net._patch_llm_adapter(unet)
    assert torch.equal(adapter.blocks[0](x), base)  # zero-init up
    assert net._adapter_lora_calls == 2
    nn.init.ones_(net.adapter_lora["blocks_0_1"].lora_up.weight)
    assert not torch.equal(adapter.blocks[0](x), base)

    net.remove_from()
    assert torch.equal(adapter.blocks[0](x), base)


def test_ext_rows_delta_only_on_ext_positions():
    torch.manual_seed(0)
    adapter = _FakeAdapter()
    unet = SimpleNamespace(llm_adapter=adapter)
    pack = _fake_pack()
    attach_vocab_pack(adapter, pack)
    try:
        net = _make_network(unet, train_ext_rows="true", target_rank="2")
        assert net.ext_lora_a.shape == (3, 2) and net.ext_lora_b.shape == (2, 4)
        ids = torch.tensor([[5, T5_TABLE_SIZE + 1, 7, T5_TABLE_SIZE + 2]])
        base = adapter.embed(ids)
        assert torch.equal(base[0, 1], pack.table[1])

        net._hook_ext_rows(unet)
        assert torch.equal(adapter.embed(ids), base)  # B zero-init
        nn.init.ones_(net.ext_lora_b)
        out = adapter.embed(ids)
        assert torch.equal(out[0, 0], base[0, 0]) and torch.equal(out[0, 2], base[0, 2])
        want = pack.table[1] + net.ext_lora_a[1].detach() @ net.ext_lora_b.detach()
        assert torch.allclose(out[0, 1], want)
        net.remove_from()
        assert torch.equal(adapter.embed(ids), base)
    finally:
        detach_vocab_pack(adapter)


def test_ext_rows_requires_attached_pack():
    unet = SimpleNamespace(llm_adapter=_FakeAdapter())
    with pytest.raises(ValueError, match="vocab pack"):
        _make_network(unet, train_ext_rows="true")


def test_body_lr_group_and_checkpoint_roundtrip(tmp_path):
    from safetensors.torch import save_file

    adapter = _FakeAdapter()
    unet = SimpleNamespace(llm_adapter=adapter)
    attach_vocab_pack(adapter, _fake_pack())
    try:
        net = _make_network(
            unet,
            train_target="true",
            target_rank="4",
            train_llm_adapter="true",
            train_ext_rows="true",
            target_lr="1e-4",
        )
    finally:
        detach_vocab_pack(adapter)

    params, names = net.prepare_optimizer_params_with_multiple_te_lrs(None, None, 2e-5)
    assert [g["lr"] for g in params] == [2e-5, 1e-4] and len(names) == 2
    body_ids = {id(p) for p in params[1]["params"]}
    assert id(net.target_lora_xkv[0].lora_up.weight) in body_ids
    assert id(net.ext_lora_b) in body_ids
    assert id(net.b_cond[0]) not in body_ids
    assert sum(len(g["params"]) for g in params) == len(net.get_trainable_params())

    path = tmp_path / "ec_body.safetensors"
    save_file(net.state_dict_for_save(torch.float32), str(path), net.metadata_fields())
    loaded, weights_sd = create_network_from_weights(1.0, str(path), None, [], None)
    assert loaded.train_target and loaded.target_rank == 4
    assert loaded.crossattn_dim == net.crossattn_dim
    assert set(loaded.adapter_lora.keys()) == set(net.adapter_lora.keys())
    assert loaded.ext_rows == 3
    missing, unexpected = loaded.load_state_dict(weights_sd, strict=True)
    assert not missing and not unexpected
