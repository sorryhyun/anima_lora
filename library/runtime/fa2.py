"""Release coordinates of the sm_120 FlashAttention build.

The build (sorryhyun/flash-attention-fp16) holds the 128×32 forward tile and the
fp16-accumulate backward, and kernels for sm_120 alone, so it cannot be the
default pin. The opt-in ``fa2-sm120`` dependency group installs it into the venv;
its wheel URLs in ``pyproject.toml`` must match ``TAG``
(``tests/test_rocm_backend.py`` checks).
"""

TAG = "v2.8.3-sm120-hdim64-128-fp16acc-cu132-torch2.14"
REPO = "sorryhyun/flash-attention-fp16"
