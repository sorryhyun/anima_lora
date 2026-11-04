"""Release coordinates of the sm_120 FlashAttention build.

The build (sorryhyun/flash-attention-fp16) holds the 128×32 forward tile, the
fp16-accumulate backward and, from ``TAG``, the fp16-accumulate PV forward, and
kernels for sm_120 alone, so it cannot be the default pin. The Linux wheel is built on torch's stable ABI (one
``cp310-abi3`` wheel for torch 2.12 and later on cu132). The opt-in ``fa2-sm120``
dependency group installs it into the venv; its Linux x86_64 wheel URL in
``pyproject.toml`` must match ``TAG`` and its Windows one ``WINDOWS_TAG``, which
has no forward wheel yet (``tests/test_rocm_backend.py`` checks).
"""

TAG = "v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-cu132-torch2.12plus"
WINDOWS_TAG = "v2.8.3-sm120-hdim64-128-fp16acc-cu132-torch2.14"
REPO = "sorryhyun/flash-attention-fp16"
