"""Training-free sampler-boundary corrections that plug into the denoise loop.

Each module is an optional technique applied at (or around) the sampler step,
not part of the core engine: SMC-CFG (``smc_cfg``), mod-guidance AdaLN
(``mod_guidance``), CNS (``cns``), DAVE (``dave``) and FSG (``fsg``). The
``*_core`` modules are the pure-compute halves shared with the ComfyUI nodes.
Imported by ``library.inference.generation``.
"""
