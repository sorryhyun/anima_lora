# MOSS i2i: from four task LoRAs toward a free-prompt edit LoRA

Status: idea, 2026-10-29. Nothing below is built. Measurements it rests on:
[mod_rank_i2i.md](../findings/mod_rank_i2i.md) (§ Sanitize, § Shared across i2i tasks) and
[bbox_grounding.md](../findings/bbox_grounding.md).

The goal is one LoRA that takes a page and an instruction ("remove the watermark", "colour
this", "letter this balloon in English") and edits only what the instruction names. Today
each i2i task is its own LoRA with a fixed caption or a fixed caption format
([i2i.md](../methods/i2i.md)); the edit target is learned from the pairs, not read from the prompt.

## Four tasks, two axes

An edit has to **find** the region (detection) and **draw** what goes there. The four tasks
sit at different corners:

| task | detection | drawing | what we see |
|---|---|---|---|
| **sanitize** | hard: text, SFX and watermarks anywhere, faint or stamped over art | easy: the background behind them | Misses are misses of detection: a faint stamp or handle left as a ghost, `usual_case1`'s watermark removed at some seeds only. v3 edits in blocks 26–35, where Base reads a bbox, after marking the text in the reference stream |
| **uncensor** | easy: bars and mosaic are obvious | hard: anatomy under them | The edited area is 0–1.8 % of the page; quality is all in the drawing |
| **swap_text** | given: the layout JSON names each line and its box | hard even so: erase the Japanese, letter the new text to the balloon | English lands clean; Korean drops or swaps syllables. Knowing what and where does not make the glyphs |
| **colorize** | everywhere | everywhere | Many regions and a lot of drawing, but each is a local, low-ambiguity mapping; the v4 recipe works |

A free-prompt edit needs both axes driven by the prompt: the instruction picks the region
(detection) and the content (drawing), and everything else is copied.

## What is shared already

Measured across the existing runs (seed 0, same init; two t2i runs as the init-only control):

- **Copying the reference.** The block adapters of different i2i tasks share a direction in
  blocks 26–41 that t2i runs do not (cos up to 0.37, t2i ≤ 0.015). The overlap follows how
  much of the page is copied: removal–removal most, removal–swap_text less, colorize least.
- **A reference-only modulation.** With `mod_rank`, the adapter's t = 0 end (only the
  reference rows use it) points the same way in sanitize, uncensor and swap_text (attention
  gate cos 0.88–0.92); it marks the text in the reference stream from block 0 and moves the
  copied region's loss most.
- **A location path in Base.** The text refiner turns a layout bbox into a location code the
  DiT reads over blocks ~14–35. Sanitize, trained without boxes, edits in the same blocks.

So the copy part looks common, and detection has a hook in Base (the bbox). Drawing is the
task-specific part (swap_text's image-side FFN gate turns the other way).

## Steps

Each step is a quarter-data micro arm (~20 min train + renders) before any full run.

1. **An i2i prior.** Build an init from the shared part (the mean, or the top singular
   directions, of the sanitize / uncensor adapters in blocks 26–41 plus the reference-only
   Δmod) and train a held-out task (swap_text) from it against kaiming. Pass: the copied
   region's loss falls faster at equal steps. This is the test that the shared direction is
   reusable, which the cosines do not show.
2. **Two tasks, one LoRA.** sanitize + uncensor pairs, each with its own caption. Pass: each
   caption does its own edit and leaves the other's target alone (uncensor's caption does
   not remove text, and the reverse).
3. **Detection from a box.** Sanitize with the OCR boxes as layout items ("remove" + bbox),
   and pages where only some boxes are listed. Pass: the listed boxes go, the unlisted text
   stays. This hands sanitize's hard axis to Base's bbox path.
4. **Free instructions.** Varied captions over the pooled pairs (remove X / colour / letter),
   then pairs from new edit types. Data is the cost here, not the architecture.

Extraction ("keep only X") inverts the copy: the region kept is the named one and the rest
is drawn new. It comes after step 3.

## Risks

- Cosine similarity of weights is not transfer; step 1 is there to test it.
- Every run compared shares one recipe (isolated reference at `ref_timestep 0`, Muon,
  seed 0, anime pages). The t2i control removes init, not recipe or domain; colorize shares
  the recipe and overlaps least, which argues against recipe alone.
- `mod_rank` on sanitize renders warmer with the tone term on (Δb +0.29 → +0.87). Its
  no-tone arm holds tone (Δb −0.13), and the term's change to the adapter is a smaller,
  turned FFN shift; whether the adapter holds tone by itself or the run is too short to
  drift is open.
