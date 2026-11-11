# swap_text on the re-crawled Korean: copy init against kaiming, 500 steps

Date: 2026-11-08. A smoke ahead of the swap_text main run on the relaxed root.

Result in one line: **copy init does no harm. At 500 steps it keeps the page outside the
text slightly better than kaiming (L1 out lower on 8 of 10 pages, less colour drift) and letters
slightly worse. Both arms fix e2's long-narration failures and lose short-dialogue syllables
e2 drew right.**

## Setup

- **Data:** the relaxed root restaged after the Korean re-crawl (5,627 ko pages, 175 works;
  `post_image_dataset/i2i/swap_text_relax`), cached as `i2i_swap_text_relax_ko_ksplit`
  (`_archive/configs/moss_i2i/swap_text/cache_relax_ko_ksplit.toml`). The arms train on the 65 new works'
  pages with text cover ≥ 10 % or ≥ 50 syllables: 1,509 pairs, 58 works, 187k syllables. `tk`
  (historical narration) is 802 of them. List: `staging/pairs_new_dense1509.jsonl`; cache: the
  symlink farm `i2i_swap_text_relax_ko_dense1509`.
- **Runs:** [`_archive/configs/lora_blocks/swap_dense_k_s500.toml`](../configs/swap_dense_k_s500.toml) (kaiming) and
  [`_archive/configs/lora_blocks/swap_dense_c_s500.toml`](../configs/swap_dense_c_s500.toml) (from C v2,
  `c_fill_v2_s800`).
  - Recipe: e2's (muon 5e-5, rank 32, `mod_rank` 16, `region_weight` 2, int8 plan, EMA on the
    card), 500 steps.
  - `init_lora` now loads a file without the modulation adapter; that adapter keeps its own init
    (`library/moss/training/setup.py`).
  - Memory: swap 36, budget 0.05. With captions of up to 2,088 tokens, the largest sample
    (4,104 + 4,104 reference + 2,088) peaks at 12.07 GB allocated / 12.88 GB reserved; budget
    0.15 at swap 36 ran out of memory. 2.65 s/step, about 22 min.
- **Eval:** `src/hangul_eval.sh <arm> <lora> ko10`, seed 0, against ksplit e2
  (`results/ko10_ksplit_e2`). Glyphs are read against each caption on the
  `src/hangul_read.py --ko10` sheets. Outside the text: signed ΔL / Δb in Lab and luma L1 against
  the JA source, outside the caption boxes padded 20 px.

## Lettering, wrong syllables or events per page (by eye)

| page | e2 | K | C | note |
|---|---|---|---|---|
| 589883_0002 | 2 | 2 | 2–3 | e2 drops part of a line; dense box: K misplaces a fragment, C garbles `사하ㄱ 과만`; e2 and K put coloured plates behind two boxes |
| 589883_0010 | 3 | 3 | 3 | `반영` for 반응 in all; a dropped or added syllable in `시작하겠습니다` in all |
| 589883_0013 | 2 | 2 | 3 | `했스니까` in all; K garbles `이제`, C `정릴로` and a doubled `르` |
| 589883_0015 | 3 | 2 | 3 | 소속 wrong in all; the dense paragraph garbles in all, C the most |
| 589883_0021 | half the narration dropped | 0 | 0 | e2 letters one of the two lines |
| 589883_0032 | 4 | 1 | 3 | e2 `존에` `서서` `용든` `번석`; K `용든`; C `존애` `아로바이트`, a doubled word, stray marks |
| 737486_0011 | 1 | 2 | 2 | `스윽스윽`: e2 right, K `스워스윽`, C `스위스육` |
| 737486_0012 | 1 | 2 | 3 | `둘러싸여`: e2 right, K `둘러씨여`, C `들러씨여` |
| 737486_0020 | 1 | 2 | 2 | `싸놓고도`: e2 right, K / C `쌰놓고도`; unseen `끜` → `끔` in all |
| 737486_0026 | 1 | 2 | 3 | `서있어서`: e2 right, K / C `서었어서`; C `아츰`; `변배` for 변태 in all |
| total | ~19 + a dropped half | ~18 | ~25 | |

## Outside the text

| | ΔL | Δb | L1 out (range) |
|---|---|---|---|
| ksplit e2 (4,218 steps) | +0.40 | −0.58 | 3.34 (1.93–5.58) |
| K | +0.45 | −0.86 | 3.78 (2.40–6.38) |
| C | +0.47 | −0.45 | 3.64 (1.98–6.09) |

- C's L1 out is below K's on 8 of 10 pages, by 0.1–0.8.
- C's Δb is the closest to zero.
- C's exception is 589883_0015: ΔL +1.54, L1 3.90. That is C v2's known brightening
  ([c_fill_v2.md](c_fill_v2.md)).
- The 737486 pages sit at L1 5–6 in every arm (screentone).

## Loss trajectory

Both arms draw the same σ and samples on every step (`progress.jsonl`), so the comparison is
paired. Means per window:

| steps | region K / C | rest K / C | grad norm K / C |
|---|---|---|---|
| 1–25 | 0.323 / 0.904 | 0.293 / 0.020 | 0.72 / 1.28 |
| 26–50 | 0.311 / 0.608 | 0.312 / 0.074 | 0.63 / 1.00 |
| 51–100 | 0.258 / 0.296 | 0.240 / 0.037 | 1.62 / 0.77 |
| 101–150 | 0.242 / 0.198 | 0.190 / 0.028 | 0.96 / 0.69 |
| 151–200 | 0.199 / 0.167 | 0.102 / 0.023 | 1.10 / 0.55 |
| 201–250 | 0.197 / 0.175 | 0.057 / 0.019 | 0.79 / 0.70 |
| 251–300 | 0.166 / 0.155 | 0.043 / 0.017 | 0.68 / 0.54 |
| 301–400 | 0.180 / 0.173 | 0.046 / 0.029 | 0.90 / 0.63 |
| 401–500 | 0.149 / 0.141 | 0.021 / 0.012 | 0.75 / 0.43 |

- **rest:** C copies from step 1 (0.02). K reaches the copy phase change at steps 150–250
  (0.19 → 0.06), as [copy_vs_step.md](copy_vs_step.md) found. Over steps 301–500, C's rest is
  lower on every step, by 0.013 on average. This matches its better L1 out.
- **region:** C starts high (0.90): C copies the JA lettering into the text region. It passes K
  by step 100. Over steps 301–500 it is lower on 81 % of steps, by 0.007. A lower region loss did
  not read as better lettering: C misspells more on ko10. The flow-matching loss in the text
  region does not track glyph correctness.
- Neither arm's region loss flattens by step 500 (0.17–0.18 over 301–400, 0.14–0.15 over
  401–500).
- C's gradient norm is lower and steadier from step 50 on.

## Claims

1. **Copy init is not a loss.** At 500 steps it trades a little lettering for a little copy,
   as P2 found on sanitize ([p2_sanitize_ft.md](p2_sanitize_ft.md)). Both arms are past the
   copy phase change (steps ~175–225), so most of C's head start is gone, as
   [findings/copy.md](../findings/copy.md) predicts.
2. **The dense pages fix long narration:** 0021 is lettered whole, 0032 has 1 error against
   e2's 4.
3. **The dense pages alone lose short dialogue.** Syllables e2 drew right (윽, 싸, 있, 둘) go
   wrong in both arms. The main run mixes the dense pages with dialogue pages.

## Not tested

- One seed. The outside-the-text gap (L1 0.14 on average) is smaller than the spread between
  pages.
- C v1 (`c_fill_s400`), which lacks v2's brightening.
- Init from e2.
