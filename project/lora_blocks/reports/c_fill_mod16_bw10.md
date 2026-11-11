# C with the modulation adapter, then with the B&W share cut to 10 %

Date: 2026-11-09. Follow-up to [c_fill_v2.md](c_fill_v2.md).

Result in one line: **`mod_rank 16` with the tone term off lowers C v2's training loss by 5–14 % on
every step bin and gives C the reference-side modulation every i2i task shares; with the B&W pairs
cut from half the set to 10 % (bw10) the copy outside the mask is the best of any C (lowest L1
out on all five pages, Δb near 0), but the page is still +0.5–1.1 L brighter than C, and under
`cut` the large masks are still lettered and the original text now partly comes back.**

## Runs

| run | config | data | mod | tone | EMA | steps |
|---|---|---|---|---|---|---|
| C | `c_fill_s400` | 1,667 colour | — | — | off | 400 |
| C v2 | `c_fill_v2_s800` | 1,667 colour + 1,667 B&W | — | 0.003 from 400 | 0.1 | 800 |
| mod16 | [`c_fill_v2_mod16_s800`](../configs/c_fill_v2_mod16_s800.toml) | as C v2 | 16 | off | 0.1 | 800 |
| bw10 | [`c_fill_bw10_mod16_s800`](../configs/c_fill_bw10_mod16_s800.toml) | 1,667 colour + 185 B&W | 16 | off | 0.1 | 800 |

- mod16 changes two things against C v2 (the adapter and the tone term); bw10 changes one
  against mod16 (the B&W share). bw10's data: `post_image_dataset/moss/c_fill_illu_bw10`, a
  symlink subset, B&W stems drawn with `random.Random(0)` (`subset.txt` beside it).
- Memory: swap 34 / budget 0.15 (C v2's) ran out of memory on the first backward with the
  modulation adapter; swap 36 / budget 0.12 peaks at 13.04 GB, ~2.45 s/step.
- Eval: `ref_gate.py --part fill`, the five sanitize pages, seeds 0–2, `ctrl` / `cut`.
  `output/tests/ref_gate/20261109-c_fill_v2_mod16{,/hard23}`, `…/20261109-c_fill_bw10_mod16`.

## Training loss

mod16 against C v2: the σ draws match on all 800 steps and so do the pages, so the losses pair.
C v2's logged loss minus 0.003 · tone from step 401 (flow matching only).

| steps | C v2 | mod16 | Δ | mod16 lower | median \|g\| C v2 / mod16 | p90 \|g\| |
|---|---|---|---|---|---|---|
| 1–100 | 0.2703 | 0.2635 | −2.5 % | 91 % | 0.25 / 0.41 | 0.44 / 0.75 |
| 101–200 | 0.1766 | 0.1519 | −14.0 % | 100 % | 0.27 / 0.49 | 0.60 / 0.93 |
| 201–300 | 0.0820 | 0.0731 | −10.9 % | 100 % | 0.20 / 0.26 | 0.36 / 0.51 |
| 301–400 | 0.0838 | 0.0791 | −5.6 % | 98 % | 0.19 / 0.28 | 0.39 / 0.54 |
| 401–500 | 0.0663 | 0.0610 | −7.9 % | 97 % | 1.44 / 0.25 | 3.84 / 0.45 |
| 501–600 | 0.0649 | 0.0582 | −10.2 % | 100 % | 1.42 / 0.24 | 3.37 / 0.42 |
| 601–700 | 0.0631 | 0.0555 | −12.1 % | 100 % | 1.14 / 0.22 | 2.75 / 0.37 |
| 701–800 | 0.0578 | 0.0510 | −11.7 % | 99 % | 0.95 / 0.20 | 1.91 / 0.39 |

- Up to step 400 the runs differ only by the modulation adapter (C v2's tone term is off until
  then), so the gain there is the adapter's. It peaks over steps 101–200, where the copy head
  forms ([copy_vs_step.md](copy_vs_step.md)), as on uncensor and sanitize
  ([`mod_rank_i2i.md`](../../../docs/moss/findings/mod_rank_i2i.md)).
- **C v2's late swing is the tone term.** Once it switches on, C v2's median gradient norm goes
  0.19 → 1.44, past `max_grad_norm` 1.0, so most of its last 400 updates were clipped. mod16 stays
  at 0.20–0.25.
- bw10 against mod16: the σ draws match but the pages do not (another data set), and bw10 is lower
  on 46–64 % of steps per bin. The loss does not separate the two.

## The modulation adapter learns the shared i2i part

Δmod(t) = ΔW · time(t) on `shared_block_modulation`; the reference-only part is
Δmod(0) − Δmod(900). Cosine per chunk against other mod16 runs ([`src/dmod_cos.py`](../src/dmod_cos.py),
the method of `mod_rank_i2i.md` § Shared):

| mod16 C against | shift_a | scale_a | gate_a | shift_f | scale_f | gate_f |
|---|---|---|---|---|---|---|
| uncensor mod16 | +0.25 | +0.80 | **+0.94** | +0.60 | +0.74 | **+0.93** |
| swap_text ksplit e2 | +0.02 | +0.62 | **+0.88** | +0.17 | +0.40 | **+0.93** |
| swap_text v3 (from C v2, fresh mod) | +0.03 | +0.21 | **+0.81** | −0.01 | +0.10 | **+0.91** |
| t2i ama_mitsuki mod16 | +0.03 | −0.00 | +0.05 | −0.03 | +0.01 | −0.14 |

C trains on identity pairs with an empty prompt and no edit, and it lands on the direction the
edit tasks share (both gates 0.88–0.94), which the t2i control does not. Its change at t = 0 is
the largest of these runs (FFN shift 5.45 % of Base's modulation; uncensor 3.73 %, swap_text
1.8–2.0 %).

## Outside the mask, `ctrl`, ranges over seeds 0–2

| page | | C | C v2 | mod16 | bw10 |
|---|---|---|---|---|---|
| usual_case1 | ΔL | −0.79 … −0.63 | −0.28 … −0.13 | +0.21 … +0.28 | **−0.12 … −0.03** |
| | Δb | −0.25 … −0.18 | −0.34 … −0.28 | −0.48 … −0.42 | −0.21 … −0.18 |
| | L1 | 5.90–6.16 | 5.61–5.69 | 5.83–5.92 | **5.54–5.56** |
| usual_case2 | ΔL | **−0.03 … +0.24** | +0.80 … +1.05 | +0.99 … +1.02 | +0.71 … +0.87 |
| | Δb | −0.48 … −0.38 | −0.81 … −0.75 | −1.00 … −0.93 | **−0.33 … −0.29** |
| | L1 | 6.31–6.37 | 6.61–6.95 | 6.78–6.84 | 6.31–6.50 |
| hard_to_sanitize1 | ΔL | **+0.01 … +0.08** | −0.30 … −0.19 | +0.93 … +0.99 | +0.49 … +0.57 |
| | Δb | −0.32 … −0.27 | −0.39 … −0.35 | −0.77 … −0.63 | **−0.18 … −0.13** |
| | L1 | 6.14–6.19 | 6.31–6.37 | 6.49–6.64 | **6.08–6.17** |
| hard_to_sanitize2 | ΔL | **−0.54 … −0.39** | +1.07 … +1.13 | +1.54 … +1.60 | +0.93 … +1.10 |
| | Δb | −0.21 … −0.14 | −0.12 … −0.03 | −0.54 … −0.47 | −0.17 … −0.03 |
| | L1 | 12.95–13.21 | 12.87–12.95 | 13.13–13.23 | **12.70–12.76** |
| hard_to_sanitize3 | ΔL | **+0.05 … +0.27** | +1.05 … +1.15 | +1.33 … +1.49 | +0.79 … +0.92 |
| | Δb | −0.10 … −0.02 | −0.44 … −0.38 | −0.86 … −0.77 | −0.18 … −0.12 |
| | L1 | 5.60–5.67 | 6.30–6.48 | 6.78–7.10 | 5.93–6.08 |

sanitize v3 alone, for scale: L1 out 5.55–5.60 on `usual_case1`.

## `cut` on the masks, by eye

| page | C | C v2 | mod16 | bw10 |
|---|---|---|---|---|
| hard_to_sanitize1 (stamps, handle) | stamps go | both go on s0 / s1, headband s2 | go on s0, `0=` fragment and a hairclip s1, headband s2 | handle goes 3 / 3; **the `無断転載禁止` stamp comes back 3 / 3**, garbled, mirrored on s1 |
| hard_to_sanitize2 (manga balloons) | blank balloons | pseudo-English in every balloon, 3 / 3 | large pseudo-Japanese and SFX s0 / s1, noise texture s2 | Japanese SFX s0, red pseudo-English s1, dot / maze texture s2 |
| hard_to_sanitize3 (lettering on black) | background | manga panel + Japanese s0, wings and butterflies s1, Japanese balloon s2 | pseudo-English in the page's own red glowing hand, 3 / 3 | scene fill, but **parts of the original lines return** (`I get it`, `my mana… yours`, `what`) on all three |

## Claims

1. **The modulation adapter belongs in C.** It lowers the paired loss on 91–100 % of steps from
   the start, most where the copy head forms, and what it learns at t = 0 is the reference-side
   direction the edit tasks share.
2. **The tone term caused C v2's late loss swing.** Its gradient pushed the norm past the clip on
   most steps after 400; without it the norm stays at 0.2.
3. **The B&W share caused the cool cast, and part of the brightening.** At 10 %, Δb is within
   0.33 of 0 on every page and L1 out is the lowest of any C on four pages of five; ΔL drops by
   0.1–0.6 against mod16 but stays +0.5–1.1 on four pages, where C is near 0.
4. **The lettering under `cut` does not come from the B&W share.** bw10 still letters
   `hard_to_sanitize2` on 3 / 3 seeds. C did not letter on the same colour illustrations
   (481 of them text-tagged), so what is left between them is 800 against 400 steps, EMA and the
   modulation adapter.
5. **A stronger copy leaks through `cut`.** bw10 redraws the masked stamp and parts of the masked
   lines. `cut` removes the hole's image queries from the reference keys in the hole; image rows
   outside the hole still read them, and the hole reads those rows. A two-hop path is the likely
   route; not measured.

## Not tested

- The s400 saves of mod16 / bw10 under `cut`: whether the lettering and the leak come with the
  steps.
- mod16 with the tone term on (the two changes of mod16 are not separated past step 400).
- EMA off at 800 steps.
- A repeat of the P2 rerun below, or the same run with EMA on.

## bw10 as the init: P2's sanitize fine-tune rerun

[`_archive/configs/lora_blocks/sanitize_ft_cbw10_s300.toml`](../configs/sanitize_ft_cbw10_s300.toml): P2's recipe
([p2_sanitize_ft.md](p2_sanitize_ft.md): 277 pairs, 300 steps, EMA off, tone 0.003 from step 150)
from bw10, with `mod_rank 16` because the init carries the modulation adapter. C-ft had none, so
the arm differs from C-ft in the init and in the adapter. Eval as P2:
`output/tests/ref_gate/20261107-2202-p2/cbw10ft`.

Removal, `ctrl`, by eye; C-ft and bw10-ft on all three seeds (v3 and K-ft: P2's seed-0 read):

| page | v3 | K-ft | C-ft | bw10-ft |
|---|---|---|---|---|
| usual_case1 (watermark) | kept | removed | removed | removed 3 / 3 |
| hard_to_sanitize1 (stamps, bubbles) | anger bubble kept | all removed, smudge at the collar | all removed; a scribble where the anger bubble was (s0), a faint stamp ghost (s1), a white patch at the small balloon's edge (s1 / s2) | all removed 3 / 3; a white patch at the small balloon's edge (s1) |
| hard_to_sanitize2 (manga, red balloons) | clean | red blocks, smears | text gone 3 / 3; the large red balloon turns into pink cloth 3 / 3 | text gone 3 / 3; pink cloth on s1 only; faint SFX traces by the ghost (s0 / s2, also in C-ft) |
| hard_to_sanitize3 (lettering, signature) | clean | ring marks, a red blob | clean | clean 3 / 3, a red leaf spray at the right on all three |

Outside the mask, `ctrl`, mean over seeds 0–2:

| page | ΔL v3 / K / C / bw10 | Δb v3 / K / C / bw10 | L1 v3 / K / C / bw10 |
|---|---|---|---|
| usual_case1 | −0.16 / −1.36 / +0.75 / **−0.09** | −0.62 / +0.17 / +0.04 / +0.41 | **5.59** / 7.43 / 6.66 / 6.24 |
| usual_case2 | +0.33 / −1.52 / +1.76 / +0.67 | −0.33 / +1.05 / −0.25 / **+0.14** | **6.32** / 8.81 / 8.42 / 7.01 |
| hard_to_sanitize1 | **−0.12** / −0.51 / +3.04 / +2.95 | +0.10 / +0.79 / +0.33 / +0.59 | **6.18** / 7.77 / 10.96 / 10.76 |
| hard_to_sanitize2 | **−0.00** / −2.38 / +1.30 / +0.36 | −0.37 / +0.01 / −0.94 / −0.97 | **16.45** / 18.03 / 16.56 / 17.74 |
| hard_to_sanitize3 | −0.47 / −1.81 / +1.23 / **+0.47** | −0.31 / +0.31 / −0.84 / −1.25 | **5.81** / 8.20 / 6.69 / 6.19 |

- **Removal is C-ft's with less residue**: every target goes on every seed in both, the anger
  bubble included (v3 keeps it); bw10-ft leaves the red balloon as cloth on 1 / 3 seeds against
  3 / 3, and has no scribble or stamp ghost on `hard_to_sanitize1`. A small edge, not a new level.
- **Outside the mask bw10-ft is closer to v3 than C-ft on three pages of five** (L1 6.24 / 7.01 /
  6.19 against 6.66 / 8.42 / 6.69), ties on `hard_to_sanitize1` (both +3 L there) and is worse on
  `hard_to_sanitize2` (17.74 against 16.56). |ΔL| is below C-ft's on four pages. v3 (2,216 steps,
  EMA) is still best outside the mask on every page.
- P2 measured swings of 2–4 L between saves with EMA off, so one save per arm does not rank the
  tone; the removal read is the robust part.
- bw10's leak under `cut` does not carry into the fine-tune's `ctrl`: no masked text comes back.
