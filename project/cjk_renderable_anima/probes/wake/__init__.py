"""wake — the wake-probe instrument's shared plumbing (stages live in ``stage/``).

Entry point stays ``probes/wake_probe.py`` (same CLI, same output dirs).

  common       paths, inventories, prompt templates, CER, shapes, output dirs
  models       checkpoints, generation requests, VAE, text encoding, DiT forward
  hooks        ExtDelta (the ext-row delta) and AdapterLoRA
  readers      detector + two OCR readers, contact sheets
  render       font renders: layout sampling, string rendering, corpus crops
  inventory    Qwen pieces → pack rows; word / kanji inventories; clean strings
  encoder      W2d glyph encoder g(render) → Δ_row and its glyph bank
  trainables   what an arm trains and how it becomes the ExtDelta table
  cli          argparse, flags grouped by the stage that reads them

Torch is imported only by the modules that need it, so ``--stage data``
stays CPU-only.
"""
