# LoRA module building blocks; public API re-exported here.

from networks.lora_modules.base import BaseLoRAModule, _absorb_channel_scale
from networks.lora_modules.hydra import HydraLoRAModule, _sigma_sinusoidal_features
from networks.lora_modules.lora import LoRAModule

__all__ = [
    "BaseLoRAModule",
    "HydraLoRAModule",
    "LoRAModule",
    "_absorb_channel_scale",
    "_sigma_sinusoidal_features",
]
