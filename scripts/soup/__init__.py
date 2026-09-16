"""LoRA soup builders + the `make soup` pipeline (uncond-init soup training).

Design source: bench/memorization/report.md (the uncond-init ladder); user
docs: docs/experimental/soup.md. ``build.py`` holds the
ΔW-level soup math; ``pipeline.py`` orchestrates uncond inter-train → seeded
fine-tunes → SVD-truncated soup.
"""
