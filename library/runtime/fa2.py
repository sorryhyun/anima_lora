"""Release coordinates of the sm_120 FlashAttention build.

The build (sorryhyun/flash-attention-fp16: fp16-accumulate backward and PV
forward, ``flash_attn_f16in_func``) ships sm_120 kernels only, so it is the
opt-in ``fa2-sm120`` dependency group, not the default pin. Wheels are
``cp310-abi3`` on torch's stable ABI (torch 2.12+, cu132). The Linux wheel holds
head dims 64, 96 and 128 (96: PaddleOCR-VL's tower at 72); the Windows wheel is
the previous release's, 64 and 128. The wheel URLs in ``pyproject.toml`` must
come from ``TAG`` / ``WIN_TAG`` (``tests/test_rocm_backend.py`` checks).
"""

TAG = "v2.8.3-sm120-hdim64-96-128-fp16acc-fwdpv-f16in-cu132-torch2.12plus"
WIN_TAG = "v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-f16in-cu132-torch2.12plus"
REPO = "sorryhyun/flash-attention-fp16"
