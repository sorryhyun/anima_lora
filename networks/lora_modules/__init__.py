# LoRA module building blocks — package form.
#
# Split out of a monolithic lora_modules.py. Public API (module classes +
# _absorb_channel_scale / _sigma_sinusoidal_features) is preserved by the
# re-exports below so `from networks.lora_modules import LoRAModule, ...`
# keeps working unchanged.

from networks.lora_modules.base import BaseLoRAModule, _absorb_channel_scale
from networks.lora_modules.hydra import HydraLoRAModule, _sigma_sinusoidal_features
from networks.lora_modules.lora import LoRAModule
from networks.lora_modules.ortho import (
    OrthoHydraLoRAExpModule,
    OrthoLoRAExpModule,
)
from networks.lora_modules.reft import ReFTModule

__all__ = [
    "BaseLoRAModule",
    "HydraLoRAModule",
    "LoRAModule",
    "OrthoHydraLoRAExpModule",
    "OrthoLoRAExpModule",
    "ReFTModule",
    "_absorb_channel_scale",
    "_sigma_sinusoidal_features",
]
