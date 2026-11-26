# The reward reader on the card (2026-11-13)

What reads the policy's renders during P1 (proposal § 4, before P1). Probe
`../probes/reader_batch.py`; result `../results/20261113-1657-reader_batch/`.

Swapping the readers on and off each group costs a load each time, so the plan is a
reward reader light enough to stay resident beside the update.

**int8 / fp8 dropped (11-13, microbench).** Both readers are PaddleOCR-VL-1.6 (0.9 B, 1.8 GB
each). The tower's K (1152, 4304) is not a multiple of ConvRot's 256-column group, and at
decode M every quantized linear is slower than bf16 (3072 × 1024 at M = 4: bf16 14 µs,
dequant 30, ConvRot W8A8 72, fp8 85, `torch._weight_int8pack_mm` 45; only the lm_head is
faster on ConvRot). GPTQ would fix the error, not the speed.

**Batched read, no whole page (`Readers.read_images` in `../../cjk_anima_reseed/src/common/readers.py`).** A group's 8 renders read in shared batches, the whole page left
out. s a group (det / sfx / vl): one page a call with the whole page 7.3 (0.23 / 2.97 /
4.02), without it 4.7, batched 2.2 (0.17 / 1.10 / 0.83). Resident 3.5 GiB, peak 4.4. Gate
(fixed before the run: paired against P0's reads with the whole page stripped, p ≥ 0.05):
r_seq 0.3939 → 0.3937 (9 / 12, p 0.66), g_p 0.1632 → 0.1635 (50 / 58, p 0.5), **passes**.
Boxes are identical on all 640 pages; the strings differ on 11 % (SFX) / 15 % (VL) of
1,693 boxes from batch padding alone. Dropping the whole page lowers r_seq by 0.019
(0 up / 46 down), a level shift.

**Stock VL16 is out of every read (user, 11-13).** The line's reader is the tuned
PaddleOCR-VL-1.6 (`OcrReader`, trained on SFX and dialogue); stock VL16 was a second
reader carried over from the ruler, best-of-two in cer / r_seq, averaged into g_p and
required for `official`. `Readers` loads the tuned reader alone, `READERS = ("sfx",)`,
`official` is gone — the reward and the reads of record alike. Expected read ≈ 1.3 s a
group (det 0.17 + reader 1.1), readers ≈ 1.7 GiB.
