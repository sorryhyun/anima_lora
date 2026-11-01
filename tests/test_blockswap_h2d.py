"""Host-to-device-only block swap matches the classic swap and a resident model.

``h2d_only`` (``library/runtime/offloading.py::HostMasters``) repoints an evicted
block at a page-locked CPU master instead of copying it back. With frozen block
weights and an adapter held outside the blocks (MOSS / Qwen LoRA), the loss and
the adapter gradients must equal the resident model's bit for bit, over several
steps (the masters are reused, not rebuilt), for the training half-swap and the
inference round-trip schedule alike. CUDA only:
``ANIMA_TEST_GPU=1 pytest tests/test_blockswap_h2d.py``.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from library.runtime.blockswap import place  # noqa: E402

N_BLOCKS, DIM = 8, 64
DEVICE = torch.device("cuda")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.up = nn.Linear(DIM, 2 * DIM, bias=False)
        self.down = nn.Linear(2 * DIM, DIM, bias=False)

    def forward(self, x):
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm(x))))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(N_BLOCKS))

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


def _with_adapters(net: Net) -> list[torch.Tensor]:
    """A rank-4 adapter on every ``up``, its weights outside the blocks."""
    params = []
    g = torch.Generator().manual_seed(1)
    for block in net.layers:
        a = (0.1 * torch.randn(4, DIM, generator=g)).to(DEVICE).requires_grad_()
        b = (0.1 * torch.randn(2 * DIM, 4, generator=g)).to(DEVICE).requires_grad_()
        inner = block.up.forward
        block.up.forward = lambda x, _inner=inner, _a=a, _b=b: (
            _inner(x) + x @ _a.t() @ _b.t()
        )
        params += [a, b]
    return params


def _run(net: Net, swap: int, h2d_only: bool, backward: bool, steps: int = 3):
    net.requires_grad_(False)
    params = _with_adapters(net)
    attached = place(
        net,
        "layers",
        DEVICE,
        blocks_to_swap=swap,
        supports_backward=backward,
        h2d_only=h2d_only,
    )
    x = torch.randn(2, 16, DIM, generator=torch.Generator().manual_seed(2)).to(DEVICE)
    out = []
    try:
        for _ in range(steps):
            if backward:
                loss = net(x).square().mean()
                loss.backward()
                out.append([loss.detach()] + [p.grad.clone() for p in params])
                for p in params:
                    p.grad = None
            else:
                with torch.no_grad():
                    out.append([net(x)])
        torch.cuda.synchronize()
    finally:
        if attached is not None:
            attached.detach()
        else:
            net.to("cpu")
    return out


def _net() -> Net:
    torch.manual_seed(0)
    return Net()


@pytest.mark.core
@pytest.mark.parametrize("swap", [3, 6])
@pytest.mark.parametrize("backward", [True, False])
def test_h2d_only_matches_resident_and_classic(swap, backward):
    base = _net()
    resident = _run(copy.deepcopy(base), 0, False, backward)
    classic = _run(copy.deepcopy(base), swap, False, backward)
    h2d = _run(copy.deepcopy(base), swap, True, backward)
    for ref, a, b in zip(resident, classic, h2d):
        for r, c, h in zip(ref, a, b):
            assert torch.equal(r, c)
            assert torch.equal(r, h)


def test_h2d_only_leaves_weights_unchanged():
    base = _net()
    net = copy.deepcopy(base)
    _run(net, 6, True, backward=True)
    for (name, p), q in zip(net.state_dict().items(), base.state_dict().values()):
        assert p.device.type == "cpu", name
        assert torch.equal(p, q), name


def test_h2d_only_refuses_trainable_block_weights():
    net = _net()
    with pytest.raises(ValueError, match="frozen"):
        place(net, "layers", DEVICE, blocks_to_swap=3, h2d_only=True)


@pytest.mark.parametrize("pin_ok", [True, False], ids=["pin", "pageable"])
def test_h2d_only_falls_back_when_register_refused(monkeypatch, capsys, pin_ok):
    """A refused ``cudaHostRegister`` (Windows, swap 30) sends that master and
    every later one to ``pin_memory()``, and on its failure to pageable memory;
    the numerics stay the resident model's."""
    real = torch.cuda.cudart()
    real_empty = torch.empty

    class Refusing:
        calls = 0

        def cudaHostRegister(self, ptr, size, flags):
            Refusing.calls += 1
            if Refusing.calls > 2:
                return 2
            return real.cudaHostRegister(ptr, size, flags)

        def cudaGetErrorString(self, err):
            return "out of memory"

        def __getattr__(self, name):
            return getattr(real, name)

    def empty(*args, pin_memory=False, **kwargs):
        if pin_memory and not pin_ok:
            raise RuntimeError("pinned host allocation refused")
        return real_empty(*args, pin_memory=pin_memory, **kwargs)

    base = _net()
    resident = _run(copy.deepcopy(base), 0, False, backward=True)
    monkeypatch.setattr(torch.cuda, "cudart", Refusing)
    monkeypatch.setattr(torch, "empty", empty)
    h2d = _run(copy.deepcopy(base), 6, True, backward=True)
    out = capsys.readouterr().out
    assert "out of memory) for" in out and "after 2 pinned blocks" in out
    assert ("falling back to pageable" in out) != pin_ok
    assert Refusing.calls == 3
    for ref, h in zip(resident, h2d):
        for r, x in zip(ref, h):
            assert torch.equal(r, x)
