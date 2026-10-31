"""``grad_error.py``'s split arms against the kernel.

.venv/bin/python -m pytest project/int8_train/src/test_int8_linear.py
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from library.int8.linear import convert_, int8_mm  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

M, K, N, R = 256, 1024, 768, 16
DEV = "cuda"


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def linear(seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(K, N, bias=False, device=DEV, dtype=torch.bfloat16)
    lin.requires_grad_(False)
    return lin


def test_split_emulation_matches_kernel_error():
    """grad_error.py's w8a16 / w16a8 arms are halves of the kernel's error only if
    emulating both sides at once is the eager reference and lands at the kernel's
    distance from bf16. The CUDA kernel and the eager reference round differently
    (0.5% apart on these shapes) at the same distance, so the realisation is not
    compared."""
    from comfy_kitchen.backends.eager.quantization import int8_linear as eager_linear
    from grad_error import qdq_tokens

    from library.moss.int8 import rotate

    lin = linear()
    w = lin.weight.detach().clone()
    convert_(lin, "fwd")
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    ref = x.float() @ w.float().T
    y_kernel = int8_mm(x, lin.wq, lin.s)
    y_eager = eager_linear(x, lin.wq, lin.s, convrot=True, convrot_groupsize=256)
    y_emul = F.linear(qdq_tokens(x), rotate(lin.wq.float() * lin.s))
    e_kernel = rel(y_kernel, ref)
    assert rel(y_emul, y_eager) < 0.3 * e_kernel
    assert abs(rel(y_emul, ref) - e_kernel) < 0.1 * e_kernel
