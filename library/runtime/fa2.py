"""Release coordinates of the sm_120 FlashAttention build.

The build (sorryhyun/flash-attention-fp16) holds the 128×32 forward tile, the
fp16-accumulate backward and the fp16-accumulate PV forward, and kernels for
sm_120 alone, so it cannot be the default pin. The wheels are built on torch's
stable ABI (one ``cp310-abi3`` wheel per platform for torch 2.12 and later on
cu132). The opt-in ``fa2-sm120`` dependency group installs it into the venv; its
Linux x86_64 and Windows wheel URLs in ``pyproject.toml`` must match ``TAG``
(``tests/test_rocm_backend.py`` checks).
"""

TAG = "v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-cu132-torch2.12plus"
REPO = "sorryhyun/flash-attention-fp16"
