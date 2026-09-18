# plan — the P line (superseded; kept for its verdicts)

> **Superseded 2026-09-15** by the S line (scene composites, rows arm):
> [`plan_synth3.md`](plan_synth3.md) and [`plan_synth4.md`](plan_synth4.md)
> are the live plans, [`synth.md`](synth.md) is how the S line is built,
> [`findings.md`](findings.md) holds the verdicts and
> [`reports/`](reports/README.md) the dated record. The P phases (P0b–P4,
> encoder arm, flat-canvas singles) did not survive the table-parts probe:
> a flat-trained row addresses the whole training canvas, and its identity
> does not transfer to a composite (`findings_seed.md`, transplant 0/64).
> What is left here is the canvas-pool result, the artefact form, and the
> two fallbacks that are still fallbacks.

## The target artefact (unchanged)

A vocab pack: a static table over ext rows, loaded by the existing
`llm_adapter.embed` hook, shipped as safetensors + mapping json under the
same `vocab_pack` key (training, TE caching, `inference.py`,
`GenerationRequest`, the register node). Rows are Qwen pieces; a new pack
changes the digest → `make preprocess-te ARGS=--overwrite` for CJK
captions. Anything that leaves this form (DiT/adapter LoRA, runtime gate)
is a fallback, not a phase. Packaging and the pre-upload gates are
[`deploy_plan.md`](deploy_plan.md).

## P0a — canvas shapes — PASSED 2026-09-14 (better at cost)

Run 3's recipe on a mixed 384–512 pool (`--shapes
"384,448,512:2,384x512,512x384"`, one-shape batches, per-shape latent
caches) against Run 3 at 512² (arm `encoder_wds_w120_s8k_fres_warm_shp`,
jobs `20260914-154123-df61b7` / `-8978a0`), same ruler (largest detector
box) for both:

| eval | A mixed | Run 3 |
|---|---|---|
| single 512², sfx / both readers | **36/36 / 29/36** | 33/36 / 26/36 |
| word 512², sfx / both | 10/32 / 7/32 | 9/32 / 9/32 |
| line / combo / corpus | 0/32 / 0/36 / 0/20 | 0/32 / 1/36 / 0/20 |
| EN 512² | 24/24 | 24/24 |
| single **384×512**, sfx / both | **34/36 / 29/36** | — |
| train wall / it/s | **47.8 min / 2.79** | 58 min / 2.30 |

Verdict: **better at cost** — singles above Run 3 at 512², 29/36 on a
canvas the 512²-only rows never saw, at 0.82× the wall (1.21× it/s, not
the 1.33× "cheaper at parity" wanted, so the matched-wall arm B was never
run). Every 512² miss is a vl misread of a correct render (ひ→U, ち→5,
ぬ→奴, キ→丰); at 384×512 two are real (み→る, ケ→タ). The identity band
does not move with canvas size: the same-noise classifier on the 512² band
rows at 384² peaks at σ 0.80 (top-1 0.73; 512²: 0.69), chance at 0.65 and
0.95 — one hard band serves the whole pool.

256² stays dead (the base cannot spell EN there, 11/24); 384² is alive (EN
24/24, 512² rows read 21/36). That is one tier, not "below 512²".

## Fallbacks (not phases)

- **Slot rows** — tokenizer routes the i-th piece of a quoted string to
  row (piece, i); train `Δ(piece, i) = Δ_piece + P_i`, bake the sum. Still
  a pack. Only if order and count cannot live in one table — the strings
  arm says they can.
- **W3 DiT-side ext-gated cross-attn LoRA** — only if slot rows also fail;
  carries the EN-safety list (ext gate on ext-free sequences, position
  mask, EN replay on mixed prompts — `reports/wake_plan_2026_09_13.md`)
  and leaves the native pack form.
