"""The int8 base's linears, ``wq_t`` stream, card EMA and memory plan."""

import dataclasses

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import flop_registry

from library import downloads
from library.moss.dit_int8.base import check_plan, forward_file
from library.moss.dit_int8.linear import (
    Int8TrainLinear,
    convert_,
    int8_mm,
    stream_wqt_,
)
from library.moss.requests import TrainRequest
from library.moss.train import ema_step, ema_step_card
from library.moss.training.memory import set_activation_memory_budget
from library.moss.training.muon import BatchedMuon
from library.runtime.blockswap import place

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

M, K, N, R = 256, 1024, 768, 16
DEV = "cuda"


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def linear(seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(K, N, bias=False, device=DEV, dtype=torch.bfloat16)
    lin.requires_grad_(False)
    return lin


def test_default_plan_is_stream_at_budget_03_swap_0():
    req = TrainRequest()
    assert (req.int8, req.ema_card, req.stream) == ("stream", True, "fp32")
    assert (req.blocks_to_swap, req.activation_memory_budget) == (0, 0.3)
    check_plan(req)


def test_bf16_config_is_the_swap_plan():
    req = TrainRequest.from_argv(["--config", "configs/moss/lora_bf16.toml"])
    assert (req.int8, req.ema_card, req.stream) == ("off", False, "bf16")
    assert (req.blocks_to_swap, req.activation_memory_budget) == (36, 0.5)


@needs_cuda
@pytest.mark.parametrize("mode", ["infer", "full"])
def test_forward_hands_the_kernel_fp32(mode):
    lin = linear()
    convert_(lin, mode)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    y = lin(x)
    assert y.dtype == torch.bfloat16
    assert torch.equal(y, int8_mm(x.float(), lin.wq, lin.s))
    assert not torch.equal(y, int8_mm(x, lin.wq, lin.s))


def test_int8_refuses_loha_and_runs_swap():
    check_plan(TrainRequest(blocks_to_swap=20))
    with pytest.raises(SystemExit, match="lora_bf16.toml"):
        check_plan(TrainRequest(algo="loha"))


def test_forward_file_order(monkeypatch, tmp_path):
    fetched = []
    target = tmp_path / downloads.MOSS_DIT_INT8_FILE
    asset = dataclasses.replace(downloads.by_id()["moss_dit_int8"], dest=tmp_path)
    monkeypatch.setattr(downloads, "by_id", lambda: {"moss_dit_int8": asset})
    monkeypatch.setattr(downloads, "fetch", lambda a: fetched.append(a))

    assert forward_file(TrainRequest(int8_file="rtn")) == ""
    assert forward_file(TrainRequest(int8_file="x.safetensors")) == "x.safetensors"
    assert forward_file(TrainRequest(model="sft")) == ""
    assert forward_file(TrainRequest()) == str(target) and fetched == [asset]
    target.touch()
    assert forward_file(TrainRequest()) == str(target) and len(fetched) == 1


@needs_cuda
@pytest.mark.parametrize("mode", ["fwd", "full"])
def test_forward_and_dx_match_bf16(mode):
    lin = linear()
    w = lin.weight.detach().clone()
    convert_(lin, mode)
    x = torch.randn(2, M, K, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    y = lin(x)
    assert y.shape == (2, M, N) and y.dtype == torch.bfloat16
    assert rel(y, x.detach() @ w.T) < 0.03
    dy = torch.randn_like(y)
    (dx,) = torch.autograd.grad(y, x, dy)
    assert rel(dx, dy @ w) < 0.03


@needs_cuda
def test_infer_mode_matches_bf16_and_drops_weight():
    lin = linear()
    w = lin.weight.detach().clone()
    convert_(lin, "infer")
    assert "weight" not in lin._parameters and not hasattr(lin, "wq_t")
    x = torch.randn(2, M, K, device=DEV, dtype=torch.bfloat16)
    with torch.no_grad():
        assert rel(lin(x), x @ w.T) < 0.03


@needs_cuda
@pytest.mark.parametrize("mode", ["fwd", "full"])
def test_lora_grads_match_eager_reference(mode):
    lin = linear()
    w = lin.weight.detach().clone()
    convert_(lin, mode)
    org_forward = lin.forward  # captured after the swap, as LoRA's apply_to does
    torch.manual_seed(1)
    A = (torch.randn(R, K, device=DEV) / K**0.5).bfloat16().requires_grad_()
    B = (torch.randn(N, R, device=DEV) / R**0.5).bfloat16().requires_grad_()
    scale = 0.5
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(M, N, device=DEV, dtype=torch.bfloat16)

    y = org_forward(x) + F.linear(F.linear(x, A), B) * scale
    dx, dA, dB = torch.autograd.grad(y, (x, A, B), g)

    with torch.no_grad():
        h = x @ A.T
        ref_dB = scale * g.T @ h
        dh = scale * g @ B
        ref_dA = dh.T @ x
        base_dx = int8_mm(g, lin.wq_t, lin.s_t) if mode == "full" else g @ w
        ref_dx = base_dx + dh @ A
    assert rel(dA, ref_dA) < 1e-2
    assert rel(dB, ref_dB) < 1e-2
    assert rel(dx, ref_dx) < 1e-2


@needs_cuda
def test_ema_on_card_matches_cpu_shadow():
    torch.manual_seed(0)
    params = [torch.randn(32, K, device=DEV), torch.randn(N, 32, device=DEV)]
    staging = [torch.empty(p.shape).pin_memory() for p in params]
    cpu, card = [[]], [[]]
    for _ in range(20):
        ema_step(cpu, staging, params, [0.958])
        ema_step_card(card, staging, params, [0.958])
        for p in params:
            p.add_(torch.randn_like(p) * 0.01)
    for c, g in zip(cpu[0], card[0]):
        assert g.device.type == "cuda"
        assert rel(g.cpu(), c) < 1e-6


@needs_cuda
def test_dtype_cast_keeps_fp32_scales():
    lin = linear()
    convert_(lin, "full")
    lin.to(dtype=torch.bfloat16)
    assert lin.wq.dtype == lin.wq_t.dtype == torch.int8
    assert lin.s.dtype == lin.s_t.dtype == torch.float32
    lin.to("cpu")
    assert lin.s.device.type == lin.s_t.device.type == lin.wq.device.type == "cpu"
    assert not hasattr(lin, "weight")
    assert isinstance(lin, Int8TrainLinear)


class MiniBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(K, elementwise_affine=False)
        self.layer1 = nn.Linear(K, 4 * K, bias=False)
        self.layer2 = nn.Linear(4 * K, K, bias=False)

    def forward(self, x):
        return x + self.layer2(F.gelu(self.layer1(self.norm(x)), approximate="tanh"))


def full_stack(n=4, mode="full"):
    torch.manual_seed(0)
    stack = nn.Sequential(*[MiniBlock() for _ in range(n)])
    stack.to(DEV, torch.bfloat16).requires_grad_(False)
    for block in stack:
        convert_(block.layer1, mode)
        convert_(block.layer2, mode)
    return stack


def stream_blocks(stack):
    return stream_wqt_([[b.layer1, b.layer2] for b in stack])


@needs_cuda
def test_stream_dx_is_full_dx_and_prefetches():
    torch.manual_seed(1)
    x0 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)

    def grad(stack):
        x = x0.clone().requires_grad_()
        stack(x).square().mean().backward()
        return x.grad

    stack = full_stack()
    ref = grad(stack)
    stream = stream_blocks(stack)
    assert not hasattr(stack[0].layer1, "wq_t")
    assert stack[3].layer2.sid.device.type == "cpu"
    assert torch.equal(grad(stack), ref)
    assert stream.entered == 4 and stream.cold == 1  # block 3; the rest prefetched
    assert torch.equal(grad(stack), ref)  # the next step restarts at block 3
    assert stream.entered == 8 and stream.cold == 2


@needs_cuda
def test_stream_out_of_order_block_is_copied_in():
    stack = full_stack(3)
    linears = [b.layer1 for b in stack]
    wq_t = [lin.wq_t.clone() for lin in linears]
    stream = stream_blocks(stack)
    for b in (0, 2, 1, 1, 0, 2):
        got = stream.get(int(linears[b].sid))
        assert torch.equal(got, wq_t[b])


class SwappedStack(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@needs_cuda
@pytest.mark.parametrize("h2d_only", [False, True])
@pytest.mark.parametrize("mode", ["fwd", "full", "stream"])
def test_swapped_dx_is_resident_dx(mode, h2d_only):
    torch.manual_seed(1)
    x0 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)

    def step(net):
        x = x0.clone().requires_grad_()
        y = net(x)
        y.square().mean().backward()
        return y.detach(), x.grad

    net = SwappedStack(full_stack(6, "full" if mode == "stream" else mode))
    stream = stream_blocks(net.layers) if mode == "stream" else None
    y_ref, dx_ref = step(net)
    attached = place(
        net,
        "layers",
        torch.device(DEV),
        blocks_to_swap=3,
        supports_backward=True,
        h2d_only=h2d_only,
    )
    tail = net.layers[5].layer1
    assert tail.wq.device.type == "cpu" and tail.s.device.type == "cuda"
    for _ in range(3):  # the masters are reused, not rebuilt
        y, dx = step(net)
        assert torch.equal(y, y_ref) and torch.equal(dx, dx_ref)
    if stream is not None:
        stream.close()
    attached.detach()


@needs_cuda
@pytest.mark.parametrize("mode", ["fwd", "full", "stream"])
def test_compiled_block_has_no_graph_break(mode):
    torch.manual_seed(0)
    block = MiniBlock().to(DEV, torch.bfloat16).requires_grad_(False)
    x = torch.randn(1, M, K, device=DEV, dtype=torch.bfloat16)
    ref = block(x)
    for lin in (block.layer1, block.layer2):
        convert_(lin, "full" if mode == "stream" else mode)
    if mode == "stream":
        stream_wqt_([[block.layer1, block.layer2]])
    torch._dynamo.reset()
    explained = torch._dynamo.explain(block)(x)
    assert explained.graph_break_count == 0, explained.break_reasons

    compiled = torch.compile(block, fullgraph=True)
    xg = x.clone().requires_grad_()
    y = compiled(xg)
    assert rel(y, ref) < 0.03
    y.square().mean().backward()
    assert xg.grad is not None and torch.isfinite(xg.grad).all()


def test_budget_prices_custom_ops(monkeypatch):
    import library.moss.loader  # noqa: F401  (registers FA2's formula)
    import torch._functorch.config as functorch_config

    monkeypatch.setattr(functorch_config, "activation_memory_budget", 1.0)
    monkeypatch.setattr(functorch_config, "is_non_builtin_to_include", False)
    set_activation_memory_budget(0.1)
    assert functorch_config.is_non_builtin_to_include
    priced = (
        torch.ops.flash_attn._flash_attn_forward,
        torch.ops.int8_train.linear_stream,
    )
    assert all(op in flop_registry for op in priced)


MUON_TOL = 0.02  # both run Newton-Schulz in bf16; the rounding differs


def lora_matrices(device):
    torch.manual_seed(3)
    shapes = [(32, 3072)] * 3 + [(3072, 32)] * 3 + [(16384, 32)] * 2 + [(32, 8192)] * 2
    return [torch.randn(sh, device=device) * 1e-3 for sh in shapes]


@needs_cuda
def test_batched_muon_matches_torch():
    kwargs = dict(lr=5e-5, weight_decay=0.01, adjust_lr_fn="match_rms_adamw")
    ref, got, start = lora_matrices(DEV), lora_matrices(DEV), lora_matrices(DEV)
    ref_opt, got_opt = torch.optim.Muon(ref, **kwargs), BatchedMuon(got, **kwargs)
    torch.manual_seed(4)
    for _ in range(5):
        for a, b in zip(ref, got):
            a.grad = torch.randn_like(a)
            b.grad = a.grad.clone()
        ref_opt.step()
        got_opt.step()
    for a, b, s in zip(ref, got, start):
        assert rel(b - s, a - s) < MUON_TOL
        assert torch.equal(
            ref_opt.state[a]["momentum_buffer"], got_opt.state[b]["momentum_buffer"]
        )


@needs_cuda
def test_batched_muon_skips_param_without_grad():
    a, b = lora_matrices(DEV)[:2]
    a0, b0 = a.clone(), b.clone()
    a.grad = torch.randn_like(a)
    kwargs = dict(lr=5e-5, weight_decay=0.01, adjust_lr_fn="match_rms_adamw")
    BatchedMuon([a, b], **kwargs).step()
    assert not torch.equal(a, a0) and torch.equal(b, b0)
