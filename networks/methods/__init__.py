"""Per-method network bolt-ons outside the LoRA family.

- ``easycontrol`` — extended self-attention image conditioning + per-block cond LoRA.
- ``soft_tokens`` — SoftREPA per-layer × per-t soft text tokens.
- ``byg`` — BYG unpaired instruction editing (plain LoRA + method adapter).
- ``turbo_dmd`` — DP-DMD distillation harness (student + fake LoRA stacks).
- ``ip_adapter_pe_lora`` — LoRA injection into the PE-Core vision tower.

The LoRA / T-LoRA / HydraLoRA family lives in
``networks.lora_anima``.
"""
