# SR sidecar — finished (2026-08-22)

Standalone ResShift super-resolution for our art (×4 and ×2), deliberately
outside the Anima adapter system. Status: finished — both scale lines sit at
their measured ceilings. The whole working tree was moved here from repo-root
`sr/` when the line finished; the `make sr-*` targets were removed at the same
time (training-only surface, essentially unused — scripts run directly, see
[`README.md`](README.md), the ops surface).

## Why it's finished

- ×4: the released ResShift ×4 transferred to our art with no domain gap
  (Phase 0, 2026-06-29 — wins every metric vs bicubic, no hallucination/color
  shift), the art finetune (`x4ft`) feeds the RSD distiller, and the shipped
  1-step student is faithful to its teacher.
- ×2: exhaustively closed 2026-07. The 1-step student plateaus by ~10–12k
  steps (24k ≈ 2k, dead tie); a three-way teacher|2k|24k comparison shows the
  **ceiling lives in the teacher + the shared VQ-f4 recon floor**, not
  distillation. The teacher-side recipe levers (text crops, scale jitter,
  LPIPS, DC) were all tried and tied — the wired teacher
  (`weights/resshift_x2_final.pth`) already contains the text-fidelity
  knobs. Do not re-run long distills or another ×2 teacher retrain without a
  genuinely new lever.
- Tiling UX is solved: shared full-image noise fields + feathered blending
  landed in `distill_rsd/infer.py` and the ComfyUI node (seam median below
  the content-control floor, MUSIQ/VRAM parity).

## Shipped artifacts

- 1-step RSD students (×2 and ×4) + finetuned teachers under `weights/` /
  `output/sr/` (gitignored; students exported as safetensors).
- ComfyUI node `~/ComfyUI-Distilled-ResShift` (standalone repo — shared-noise
  + feathered blending in v0.3.0).
- Text-region tooling: `scripts/detect_text_boxes.py` (CTD) →
  `data/text_boxes.json`; `ArtSRDataset` text-crop/scale-jitter knobs.

## Open remainder

- Korean text: the text-fidelity work ran on the existing (JP-heavy)
  pool; a Korean-text training pass is the one data axis not yet trained.
  This is a *data* lever, so it does not contradict the "recipe levers
  exhausted" closure above.
- Teacher-path tiling (`scripts/sr_infer.py`, the multi-step teacher path)
  still uses independent per-tile noise — open only if teacher tiling seams
  ever matter.

## Canonical sources

- [`README.md`](README.md) — ops (direct-invocation commands), invariants
  (`lq` = LR pixels, not latent), Phase-0 table.
- [`distill_rsd/DESIGN.md`](distill_rsd/DESIGN.md) — distiller design +
  throughput bench.
- `_archive/proposals/resshift_sr_sidecar.md` — founding proposal (retired —
  shipped).
- Ceiling/plateau/tiling evidence with full numbers lives in project memory
  (`project_rsd_x2_24k_plateau_and_tile_noise`, `project_sr_x2_text_fidelity`,
  `project_rsd_distill_scale_and_bottleneck`).
