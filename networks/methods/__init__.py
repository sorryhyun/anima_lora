"""Per-method network bolt-ons (postfix / ip_adapter / easycontrol / apex).

These attach to a frozen-DiT or LoRA-adapted DiT depending on the method:
- ``postfix`` — learned vectors appended to cross-attention.
- ``ip_adapter`` — image cross-attention via Perceiver resampler + ip_kv heads.
- ``easycontrol`` — extended self-attention image conditioning + per-block cond LoRA.
- ``apex`` — `c_fake = A·c + b` shift on top of a LoRA network for self-adversarial distillation.

The classic LoRA / OrthoLoRA / T-LoRA / HydraLoRA / ReFT family lives in
``networks.lora_anima`` because of its size and internal structure.
"""
