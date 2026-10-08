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
- Ceiling/plateau/tiling/text-fidelity mechanics: the sections below.

## Mechanics (the record)

Predecessor: the PiD decode/SR line was retired 2026-07-05 in favour of this
sidecar. Its surviving reads: the tile-vs-whole tone gap is mostly SDE variance
(not GroupNorm); a static 3×3 color calibration generalizes; a zero/null caption
is off-distribution for it.

### Training

- Train at native/4096 scale, never 1024 — deployment is 1024→4096, so HR
  patches must carry 4096-scale detail; the 1024-HR eval set is itself
  mis-scaled for this target.
- VAE latents are not disk-cacheable: `data.py::__getitem__` ignores `idx`
  (fresh file + crop + degradation per call), so a cache would freeze the
  augmentation.
- VAE/teacher `torch.compile` reverted (Blackwell inductor stall); only the
  student SwinUNet block-compile survives. The real throughput lever is bf16 +
  batch 4–6.
- Never infer while training: ~5.5 GB + ~12 GB OOMs a 16 GB card and kills the
  trainer (exit 247); `train.py` has no resume. EMA is teacher-heavy before
  ~1500 steps (eyeball early checkpoints with `--weights student`). The
  checkpoint picker uses mtime, not name.

### Tiling (×2)

- The boundary-gradient seam metric is content-confounded and non-monotonic —
  do not rank overlaps by it; compare excess gradient at known boundaries
  against random control positions.
- Seam cause: two independent unseeded noise draws per tile. Fix: one
  full-image latent noise field sliced per tile (`shared_noise=True`, default
  on). The residual floor is the VAE conv receptive field. Box-average blending
  → raised-cosine `FeatheredSpliter` (y-seam 0.605→0.031).
- `swin_align` is 256 regardless of sf (f_vq 4 · 2^(L−1) 8 · window 8) — the
  ×2 bug multiplied it by sf. Recipe: chop 512 / overlap 64 / shared_noise on;
  at true ×2 inference use CHOP=256 (512 OOMs the VQ quantize on 16 GB).

### Text fidelity (levers exhausted)

- The LPIPS+DC 45k teacher arm tied on MUSIQ and was worse on ref-LPIPS —
  never promoted; the wired teacher is the 2026-07-05 30k text-knob run.
  Untried levers only: DC-on / LPIPS ≤ 0.1.
- The CTD seg head alone is unusable (halo/hair false positives; thin strokes
  score lower than solid curves) — use the yolo `blk` head (conf 0.4 / NMS
  0.35) + a stroke-coverage cross-check.
- MUSIQ rewards glyph hallucination (70 vs 58) — never gate text fidelity on
  it.
- Pre-2026-07 masks under `post_image_dataset/masks/` are un-gated by CTD.

Launch runs through the daemon (`make daemon-run`); this loop bypasses
`train.py`, so none of its queue/compile-cache/`--deterministic` infra applies.
