# Plain vs OCR captions on sincos — the shipped speech + SFX clauses (2026-09-08)

**Status: GRADED (2026-09-08). Verdict — the shipped clauses cost nothing;
the lean to OCR is real in direction but not significant on 24 pairs.** Two
arms, one training seed each, three render seeds.

## The question

`plan_det` D3 flipped the OCR defaults (AnimeText detector + VL reader) and
`plan_ocr` O4c–O4e changed what a caption says about a page four times over
(SFX clause, both dedupes, the glyph floor, the ellipsis-blind guard) — all
of it **unmeasured on a trained arm**: the D2 arm that would have measured
the detector flip was killed at step 715/2808, and every arm through C11
trained on the retired PP-OCRv6 3-layer stack's records. This is the plain
control the line has not had since 2026-09-01 (arm B, whose checkpoint and
renders are gone), re-run against what the package publishes **today**.

## Design

Shard `sincos` (351 pages), plain LoRA rank 32 / alpha 32 / lr 2e-5 / 8
epochs / batch 1 / `blocks_to_swap 0`, **masks off in both arms**
(`masked_loss = false`, the v2 default), latents shared bit-identically
(`post_image_dataset/lora/sincos`). One variable: what the caption says
about the text in the picture.

| arm | captions | TE encode | config |
|---|---|---|---|
| PLAIN | production captions, untouched | stock caches | `cjk_unmask_plain.toml` |
| OCR | + the **shipped** clauses — `Japanese text reads as "…", "…". Japanese SFX reads as "…".` | ext pack `synthjakozh1sym_r256_isoq` re-cache | `cjk_unmask_ocr.toml` |

The OCR arm's captions are composed by
`anime_tools.captions.ocr_sidecar.with_ocr_clause` straight off the
`post_image_dataset/ocr/sincos/{stem}.ocr.txt` tree (package 8ebaf58) — the
same function, floors and dedupes Export's `--combine_ocr` publishes, so the
arm trains on what any user of the package would get, not on a research
records file. `cache_te_ext.py --sidecars` (new this day) is the wiring;
`run_unmask_r2.py --sidecars` passes it.

Treatment size: **162 of 351 pages** carry a clause — 411 speech lines on 145
pages, 341 SFX lines on 125 pages, 5,810 characters. The mirror rewrote 164
captions (two of them whitespace only) and every variants row beside them.

A CJK-free caption encodes bit-identically through the ext pack, so on the
189 untouched pages the two arms' conditioning is the same tensor; inference
conditioning is identical in both arms (stock encoder, no CJK in the eval
prompts), so every render difference is weights-only.

## Readouts

- 8-row eval grid (`assets/unmask_eval_prompts.txt`, all text-free prompts)
  at seeds 42 / 7 / 1234 per arm, plus a no-LoRA `base_s*` reference.
- Spam tally (`probes/grid_spam_tally.py`): lenient PP-OCRv6 boxes per cell —
  the arm that spams text on a text-free prompt loses.
- `probes/unmask_grid_judge.py`: dbv4 adherence + PE-Spatial cos to base and
  to the sincos training mean.
- Blind pairs for the user (`probes/blind_pairs.py`, set `s16_OCR_vs_PLAIN`)
  — **pushed 2026-09-08**, 24 pairs (8 rows × seeds 42/7/1234), awaiting
  grading at `sets/s16_OCR_vs_PLAIN/verdicts.tsv` in the private repo.
  Both arm names are new to the grader, so the 42/7/1234 grids satisfy
  the re-blind rule.

## Results

### Blind pairs — OCR 14 / PLAIN 9 / 1 tie

All 24 graded (`reports/blind_s16_OCR_vs_PLAIN.md` in `cjk_aware_anima/`, the
script's home). 14–9 on 23 decisive pairs is **p = 0.20 one-sided** — a lean,
not a result. Per row it is not uniform: OCR takes r2 and r5 3–0 and r1/r3/r4
2–1, PLAIN takes r6/r7/r8 (2–1, 2–0–1, 2–1) — i.e. the OCR arm wins the rows
it wins on the *early* prompts and loses the late ones, which on 3 seeds a row
is well inside noise.

### Automated readouts — flat

`probes/unmask_grid_judge.py` (dbv4 adherence + PE-Spatial), 24 cells per arm:

| arm | adherence prob | recall | cos→base | cos→sincos |
|---|---:|---:|---:|---:|
| OCR | 0.7421 | 0.8861 | 0.9837 | 0.9079 |
| PLAIN | 0.7361 | 0.8809 | 0.9837 | 0.9065 |

`cos→base` matches to four decimals, but **read that as a null instrument, not
as evidence of equality**: across every arm this line has ever judged, `cos→base`
sits in 0.975–0.986 and `cos→sincos` in 0.903–0.912, and the spread between two
*training seeds of the same arm* (C10: 0.9840 / 0.9865 / 0.9793; C9 vs C9s2:
0.9857 / 0.9787) is larger than any between-arm gap ever measured. Adherence
likewise differs by less than the row-to-row spread. So the judge says only that
this pair is unremarkable by the line's standards — at α/r = 1 and lr 2e-5 the
adapter barely moves the base at all, which is the operating point every C arm
shares. (`reports/unmask_grid_judge_ocr_plain.json`.)

### Spam tally — neither arm spams

`probes/grid_spam_tally.py`, **ported this day to the AnimeText detect-only
engine** (D3 retired the PP-OCRv6 stack, so `load_ocr` lost `min_score` /
`min_chars` / `skip_en` and boxes carry no text): boxes only, `det_conf 0.15`
(lenient; default 0.25). **These counts do not compare to any tally before the
D3 flip** — including C10's "~2 on 3 seeds".

| arm | cells with a box | boxes | mean glyph % |
|---|---:|---:|---:|
| OCR | 10 / 24 | 15 | 1.11 |
| PLAIN | 11 / 24 | 26 | 0.55 |

Cells-with-a-box is a wash and PLAIN carries more boxes; the mean glyph % runs
the other way only because of **one** box covering 21.7 % of OCR r6/s42, near
certainly a false positive at this floor (PLAIN's largest single cell is 5.6 %
on r3/s7, 8 boxes). Read the counts, not the area.

## Verdict

**The shipped speech + SFX clauses are safe to keep as the default caption for
manga pages, and this is the plain control the line owed since 2026-09-01.**
Against a plain control on today's package (AnimeText detector + VL reader,
O4c–O4e caption rules), the OCR arm is flat on every automated readout and
+5 on blind pairs. Direction replicates 2026-09-01's unmask A/B/C (arm C, OCR
captions, cleanest); magnitude does not — that set's separation is not
reproduced here, and on this set **B's text spam does not reappear in the
PLAIN arm either**. The honest statement is *no cost*, not *a win*.

Two caveats the numbers carry:

- **24 pairs cannot resolve a 5-pair lean.** Confirming it needs a second
  training seed per arm (the s11 pattern: replication, not more render seeds),
  which is only worth spending if something downstream depends on OCR captions
  being *better* rather than merely harmless.
- **Treatment is 46 % of the shard** (162/351 pages). The per-page effect is
  diluted by the 189 pages whose conditioning is bit-identical between arms.
