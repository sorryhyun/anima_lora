# blind pairs — s12_ISO1_vs_HOT

arms ['ISO1', 'HOT'], 48 pairs, graded 48, skipped/blank 0

| arm | wins |
|---|---:|
| ISO1 | 18 |
| HOT | 22 |
| tie | 8 |

| row | ISO1 | HOT | tie |
|---|---:|---:|---:|
| r1 | 1 | 2 | 0 |
| r2 | 1 | 1 | 1 |
| r3 | 1 | 2 | 0 |
| r4 | 0 | 2 | 1 |
| r5 | 1 | 2 | 0 |
| r6 | 1 | 2 | 0 |
| r7 | 0 | 1 | 2 |
| r8 | 3 | 0 | 0 |
| r9 | 1 | 2 | 0 |
| r10 | 1 | 0 | 2 |
| r11 | 1 | 1 | 1 |
| r12 | 2 | 1 | 0 |
| r13 | 3 | 0 | 0 |
| r14 | 1 | 2 | 0 |
| r15 | 0 | 3 | 0 |
| r16 | 1 | 1 | 1 |

| pair | row | seed | A | B | verdict | result | note |
|---|---|---|---|---|---|---|---|
| p01 | r2 | s3 | ISO1 | HOT | A | **ISO1** |  |
| p02 | r2 | s5 | HOT | ISO1 | A | **HOT** |  |
| p03 | r5 | s5 | HOT | ISO1 | B | **ISO1** |  |
| p04 | r1 | s3 | HOT | ISO1 | B | **ISO1** |  |
| p05 | r5 | s4 | HOT | ISO1 | A | **HOT** |  |
| p06 | r8 | s4 | HOT | ISO1 | B | **ISO1** |  |
| p07 | r15 | s3 | HOT | ISO1 | A | **HOT** |  |
| p08 | r13 | s3 | ISO1 | HOT | A | **ISO1** |  |
| p09 | r8 | s3 | HOT | ISO1 | B | **ISO1** |  |
| p10 | r13 | s4 | HOT | ISO1 | B | **ISO1** |  |
| p11 | r13 | s5 | ISO1 | HOT | A | **ISO1** |  |
| p12 | r16 | s5 | HOT | ISO1 | B | **ISO1** |  |
| p13 | r15 | s5 | ISO1 | HOT | B | **HOT** |  |
| p14 | r6 | s3 | HOT | ISO1 | A | **HOT** |  |
| p15 | r11 | s5 | ISO1 | HOT | A | **ISO1** |  |
| p16 | r12 | s3 | HOT | ISO1 | B | **ISO1** |  |
| p17 | r6 | s4 | ISO1 | HOT | B | **HOT** |  |
| p18 | r12 | s4 | HOT | ISO1 | B | **ISO1** |  |
| p19 | r16 | s3 | HOT | ISO1 | TIE | **tie** |  |
| p20 | r11 | s3 | HOT | ISO1 | A | **HOT** |  |
| p21 | r14 | s5 | ISO1 | HOT | A | **ISO1** |  |
| p22 | r4 | s3 | ISO1 | HOT | B | **HOT** |  |
| p23 | r10 | s3 | ISO1 | HOT | TIE | **tie** |  |
| p24 | r5 | s3 | ISO1 | HOT | B | **HOT** |  |
| p25 | r8 | s5 | ISO1 | HOT | A | **ISO1** |  |
| p26 | r14 | s4 | HOT | ISO1 | A | **HOT** |  |
| p27 | r7 | s5 | ISO1 | HOT | TIE | **tie** |  |
| p28 | r2 | s4 | HOT | ISO1 | TIE | **tie** |  |
| p29 | r4 | s5 | HOT | ISO1 | TIE | **tie** |  |
| p30 | r10 | s5 | ISO1 | HOT | TIE | **tie** |  |
| p31 | r9 | s5 | ISO1 | HOT | B | **HOT** |  |
| p32 | r7 | s3 | HOT | ISO1 | A | **HOT** |  |
| p33 | r3 | s5 | ISO1 | HOT | A | **ISO1** |  |
| p34 | r3 | s4 | ISO1 | HOT | B | **HOT** |  |
| p35 | r9 | s3 | ISO1 | HOT | B | **HOT** |  |
| p36 | r12 | s5 | HOT | ISO1 | A | **HOT** |  |
| p37 | r14 | s3 | ISO1 | HOT | B | **HOT** |  |
| p38 | r6 | s5 | HOT | ISO1 | B | **ISO1** |  |
| p39 | r7 | s4 | ISO1 | HOT | TIE | **tie** |  |
| p40 | r15 | s4 | ISO1 | HOT | B | **HOT** |  |
| p41 | r4 | s4 | HOT | ISO1 | A | **HOT** |  |
| p42 | r10 | s4 | ISO1 | HOT | A | **ISO1** |  |
| p43 | r16 | s4 | ISO1 | HOT | B | **HOT** |  |
| p44 | r1 | s5 | ISO1 | HOT | B | **HOT** |  |
| p45 | r1 | s4 | ISO1 | HOT | B | **HOT** |  |
| p46 | r3 | s3 | HOT | ISO1 | A | **HOT** |  |
| p47 | r11 | s4 | HOT | ISO1 | TIE | **tie** |  |
| p48 | r9 | s4 | HOT | ISO1 | B | **ISO1** |  |

## Read (v2 prompts × seeds 3/4/5, ties allowed: 8/48)

ISO1 vs HOT **18-22** (tie 8), pair sign p 0.64; rows 4-9 (tie 3), p 0.27; per seed 5-9 / 5-8 / 8-5.
Flat. ISO1 = the same seed-0 random table as HOT at ×1 the trained norm (near-orthogonal rows,
common-direction ratio 0.004, PR 1009), so the ×5 scale is NOT what makes HOT beat C9 (s05, s11).
Inferred by transitivity only (ISO1 ≈ HOT > C9); a direct ISO1 vs C9 set is owed before this is
claimed — transitivity already failed once (s03/s04).

What separates the winners {HOT, ISO1, COLLAPSE} from the losers {C9, R, ROTATE, INIT} is not
scale, not the common-direction ratio (0.004 / 1.0 win, 0.23–0.29 lose), and not content.
The losers are the packs with a **structured low-rank spread** (PR 18–236, spectrum top-2..5
= coherent directions shared across many rows); the winners have either no spread (COLLAPSE)
or isotropic spread (HOT/ISO1). Candidate mechanism: a coherent low-rank spread acts as a few
consistent pseudo-tags the LoRA can bind to, i.e. spurious content. Untested.
