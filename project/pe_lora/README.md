# pe_lora — Qwen-Image-2.1-PE-T2I as a LoRA on the MOSS TE

- [report.md](report.md) — claims, the two reads (what the DiT reads of the PE delta, PE's
  thinking as context), the open items and the off-repo artifacts.
- `src/` — extraction, PE mode, the probes (`dit_read.py`, `think_inject.py`).

## Rules

- The user's interest is the representation side (what the TE change does to the DiT's read), not
  the rewriter as a product.
- A prose rewrite is a valid MOSS caption: MOSS-Image has a layout-JSON capability, it is not
  trained on layout JSON alone.
- Ask the user before deleting `/media/sorryhyun/new/qwen35_pe/lora/`. The calibration Grams are
  gone, so a re-extraction recomputes them (`src/calib.py`).
