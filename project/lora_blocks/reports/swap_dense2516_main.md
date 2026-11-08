# swap_text v3: the dense Korean pages + en400, from C v2, two epochs

Date: 2026-11-08. The main run after [swap_dense_copy_init.md](swap_dense_copy_init.md); shipped
as swap_text v3 (Hub `sorryhyun/MOSS-I2I-LORAs`, `swap_text/moss_i2i_swap_text_v3.safetensors`).

Result in one line: **on the ten held-out Korean pages e2 misspells ~11 syllables against ksplit
e2's ~15 and drops no line (ksplit e2 drops half of 0021's narration); the short-dialogue
syllables the 500-step smoke lost come back. English is exact on 11 of 12 renders (ksplit e2: 12).**

## Setup

- **Data:** every `pairs_all` page of the relaxed Korean root (old and re-crawled works, minus
  the two eval works) with OCR line-box cover ≥ 10 % or ≥ 50 caption syllables, the smoke's
  filter: 2,516 pairs, 163 works (`staging/pairs_dense2516.jsonl`). Plus v3's en400: 2,916
  pairs. Cache: the symlink farm `i2i_swap_text_relax_ko_dense2516_en400` (Korean into
  `i2i_swap_text_relax_ko_ksplit`, English into `i2i_swap_text_v3_all`).
- **Run:** [`configs/swap_dense2516_en400_c_e2.toml`](../configs/swap_dense2516_en400_c_e2.toml):
  from C v2, e2's recipe (muon 5e-5, rank 32, `mod_rank` 16, `region_weight` 2, int8 plan, EMA
  on the card), 2 epochs = 5,832 steps, swap 36 / budget 0.05. 2.57–2.7 s/step, 259 min, peak
  12.58 GB. Region loss ~0.15 from step 500 on, rest 0.025 → 0.008; six grad-norm spikes over
  10 (max 56 at step 2,648), each clipped at 1.0 and gone the next step.
- **Eval:** `src/hangul_eval.sh` (one job per LoRA): ko10 seed 0 and eval4 (two English, two
  Korean pages) seeds 0–2, all int8. Glyphs read on the `src/hangul_read.py` zoom sheets
  against each caption; caption OCR misreads (둑실, 낦습니다, 에잊, 벌거) are not counted.

## ko10, wrong syllables or events per page (seed 0, by eye)

| page | ksplit e2 | smoke C | e1 | **e2** |
|---|---|---|---|---|
| 589883_0002 | 2 (생력, part of a line dropped) + coloured plates behind two boxes | 2 | two lines cut short | **0** |
| 589883_0010 | 2 (시작하만, 처이) | 3 | 1 (반영) | **0** |
| 589883_0013 | 1 (SFX) | 3 | 0 | **0** |
| 589883_0015 (small dense paragraph) | ~3 | ~3 | ~4 | ~5 |
| 589883_0021 (long narration) | half dropped | 0 | half dropped | 1 (카테 → 캐) |
| 589883_0032 | 4 (존에, 서서, 용든, 번석) | 4 | 3 | 2 (존에, 번적) |
| 737486_0011 | 0 | 2 (스위스육) | 2 (스읙스육) | **0** |
| 737486_0012 | 1 (에입) | 3 | 1 (둘러써여) | **0** |
| 737486_0020 | 1 (끜 → 끔) | 2 (쌰) | 2 (쌰) | 1 (끜 → 끔) |
| 737486_0026 | 1 (변배) | 2 | 2 | 2 (존 아침, 변배) |
| total | ~15 + a dropped half | ~24 | ~17 + two drops | **~11, no drop** |

- e2 letters the short dialogue the smoke lost (스윽, 둘러싸여, 싸, 반응) and keeps its long-text
  fix (0021 whole, 0032 down from 4).
- e1 → e2 matters: e1 still cuts lines (0002, 0021) and misses the short syllables as the smoke
  did.
- 0015's small dense paragraph garbles in its middle line in every arm, e2 included.
- `끜` (unseen) → `끔` and 변태 → 변배 in every arm.

## eval4 (seeds 0–2)

| page | ksplit e2 | e2 |
|---|---|---|
| 1483610_0007 (en) | 3/3 exact | 3/3 exact; s0 drops part of the `Uwah` balloon's outline |
| 3124789_0042 (en) | 3/3 exact | 2/3 (s2: YOU'RE → `YOU?`) |
| 589883_0004 (ko) | 3/3 exact | 3/3 exact |
| 737486_0019 (ko) | 3/3 exact | 3/3 exact |

English stays at en400 while Korean grew to 2,516 pages; one word on one seed is the cost seen.

## Not measured

- Outside-the-text tone (ΔL / Δb / L1 out) against ksplit e2.
- More seeds on ko10.
