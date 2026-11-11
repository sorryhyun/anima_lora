# Reference-read runs: copy block C and character_ref

Date: 2026-11-11. Covers every MOSS i2i run up to this date in which a LoRA learns to read the
reference latent: the copy / fill block C (`project/lora_blocks`), the sanitize fine-tunes started
from C, and character_ref. It replaces the memory notes `project_copy_block.md` and
`project_copy_offset_kv.md`.

Ground truth for settings is each run's `output/ckpt_moss_i2i/<run>.json` → `request`, which
records the request the run actually used. Config files were edited after some runs (see § 7). The
loss numbers below are recomputed from the JSONs' `losses` lists (every step, not only the logged
10th step) unless a cell says otherwise. When a claim depends on the configuration, the
configuration is stated with it.

## 1. Axes

Definitions are checked against `library/moss/requests.py` (`TrainRequest`), `library/moss/model.py`
(`_RoPE.forward`, `_attention`, module docstring) and `library/moss/lora.py`.

| axis | values | what it does |
|---|---|---|
| `ref_pos` | `aligned` (default), `offset`, `canvas` | RoPE coordinates `(r, f, h, w)` of the reference tokens. **aligned**: `r = 1`, the reference grid stretched onto the image's own `(h, w)` grid, so each reference token sits over the image cell it covers. **offset**: as aligned, then `w + W` (one image width to the right), still `r = 1`. **canvas**: `r = 0`, the reference as the right panel of a two-panel picture at unit spacing; the image's own `w` moves left by half the reference width. For a reference the image's size, offset and canvas both put the reference +W from the image; they differ in `r` (1 vs 0) and in the image's `w` shift. Field added 2026-11-10 18:46 (commit d37d26bb), `canvas` 2026-11-11 (c6a4618e). JSONs written before that have no `ref_pos` key: those runs are aligned, the only placement the code had. |
| `ref_adapters` | `off` (default), `kv`, `all` | **off**: one LoRA bank on image, text and reference rows; the reference rows are in the graph with backward, and the modulation adapter (if any) reaches them. **kv**: the reference stream runs on Base weights at Base modulation (`model.py`: `base=True` for `ref_mod` once a block has a `ref_bank`), no backward; the image and text rows read it through export adapters on `proj.k` / `proj.v` at `ref_rank` (saved as `ref_layers_N_proj_{k,v}`). **all**: a full second bank at `ref_rank` on the reference rows, Base modulation, in the graph. kv and all need `ref_isolate` and `ref_timestep >= 0`. Not used by any run here: `all`. |
| `ref_rank` | 32 (default) | rank of the kv export adapters / the `all` bank. 32 in every kv run below. |
| `ref_isolate` | bool | reference queries attend only to reference keys; image and text still attend to the reference. True in every run below. |
| `ref_timestep` | 1000·σ; 0 = clean | the timestep whose block modulation the reference tokens get; negative = the image's timestep. 0 in every run below. |
| `mod_rank` | 0 = off | LoRA on `shared_block_modulation` (shift / scale / gate, one Linear shared by the 42 blocks); alpha defaults to alpha·√(mod_rank/rank) = 90.5 at rank 32 / alpha 128. Under kv it acts on image and text rows only. |
| `ref_gate cut` | `off` / `cut`, `gate_rate`, `gate_holes` | on `gate_rate` of the steps a hole (a text layout from `gate_holes`, flipped / shifted, plus random boxes; `training/holes.py`) is drawn; image queries inside the hole lose the reference keys inside the hole and read the rest. |
| `init_lora` | path | the adapters start from a saved LoRA; algo, rank, alpha, targets and state keys must match. A run with `mod_rank` may start from a file without the modulation adapter; a kv run cannot start from a shared-bank file (its `ref_layers_*` keys are missing: `training/setup.py::load_init_lora` refuses). |
| `lr` | Muon lr | 5e-5 default. |
| warmup | `warmup_ratio` 0.05 | linear warmup over 5 % of the run's steps: 20 at 400 steps, 40 at 800, 50 at 1,000, 80 at 1,600, 240 at 4,800. It scales with `max_steps`, so two runs at the same lr but different lengths have different warmups. |
| EMA | `ema_window` | span as a fraction of the run's updates, decay 1 − 1/(window·updates), starting at the end of warmup; saves are the EMA, the raw weights go to `<name>/<name>_raw`. 0 = off. |
| `distill` | `off`, `base`, `ema` | self-distillation teacher: the DiT with the target image in the TE ahead of the caption (`{stem}.moss_vis.safetensors`). `distill_sigma_min` limits it to σ ≥ that; `distill_add 1` keeps the flow target and adds the teacher MSE scaled to the same gradient norm; `distill_ema_decay` is the EMA teacher's decay. **add90 ema97** = `distill ema`, `distill_sigma_min 0.9`, `distill_add 1`, `distill_ema_decay 0.97`. |
| `fit_report` | bool | moves the largest sample to step 1 of epoch 1 (the rest keeps the shuffle). It changes which page each step sees; see the pairing note. |

**Pairing.** All runs use seed 0. The σ draw per step is the same in every run of this report
(checked: every C run against `c_fill_bw10_mod16_s800` on their common steps, and the three
character_ref runs against e1, 1,000 / 1,000). The page per step is the same only when cache and
`fit_report` both match. e1 and `1111_1413` (both `fit_report` true) log 0.3909 / 0.3902,
0.1945 / 0.1943 and 0.6694 / 0.6695 on steps 1–3. `c_fill_bw10_mod16_off_s800` (false) and
`c_fill_bw10_mod16_off_kv_s1600` (true), same cache, correlate 0.15 per step over steps 1–40. e1
and off_e1, same pages but different placement, correlate 0.88. `fit_report` per run is in the
table's notes.

## 2. Runs

Common to every run unless the row says otherwise: rank 32, alpha 128, Muon, int8 base
(`int8 = stream`), fp32 stream, `ref_isolate` true, `ref_timestep` 0, seed 0. C runs train on
identity pairs with the empty prompt (`caption_dropout 1`) under `ref_gate cut` at `gate_rate 0.9`.
The sanitize fine-tunes use the 277 sanitize pairs, the sanitize caption, no gate and tone 0.003
from step 150. character_ref runs use 1,000 pairs and no gate.

Caches: `illu` = `post_image_dataset/moss/c_fill_illu` (1,667 colour identity pairs); `illu_bw` =
`…/c_fill_illu_bw` (+1,667 B&W pairs); `illu_bw10` = `…/c_fill_illu_bw10` (1,667 + 185 B&W = 1,852);
`san` = `…/i2i_sanitize_idt`; `cref` = `…/i2i_character_ref`.

s/step and peak (allocated / reserved GB) come from the JSON (`s_per_step` = wall minutes / steps,
compile included). Swap is `blocks_to_swap / activation_memory_budget`.

| run | task | cache | ref_pos | ref_adapters | mod_rank | init_lora | lr | steps (warmup) | EMA | distill | swap | s/step | peak GB | result | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `c_fill_s400` | copy C v1 | illu | aligned | off | 0 | — | 5e-5 | 400 (20) | off | off | 38/0.10 | 2.40 | 10.55 / 11.06 | copy head by s200 (h19 0.63, 49 heads > 0.1), 67 heads at s400; with `cut` removes the boxed stamps / watermark 3/3 seeds; out-L1 `ctrl` 5.9–6.4 on 3 pages | `reports/p1_c_fill.md` |
| `c_fill_v2_s800` | copy C v2 | illu_bw | aligned | off | 0 | — | 5e-5 | 800 (40) | 0.1 | off | 34/0.15 | 2.34 | 13.28 / 13.87 | tone 0.003 from step 400; out-L1 `ctrl` 5.6–6.9 (+12.9 on hts2); usual_case1 darkening gone; letters the large masks under `cut` | `reports/c_fill_v2.md` |
| `c_fill_v2_mod16_s800` | copy C | illu_bw | aligned | off | 16 | — | 5e-5 | 800 (40) | 0.1 | off | 36/0.12 | 2.48 | 13.21 / 13.76 | tone off; paired loss −2.5 % to −14 % per 100-step bin vs C v2; Δmod gate cosine 0.81–0.94 with i2i task runs, 0.05 / −0.14 with t2i | `reports/c_fill_mod16_bw10.md` |
| `c_fill_bw10_mod16_s800` | copy C, **current init** | illu_bw10 | aligned | off | 16 | — | 5e-5 | 800 (40) | 0.1 | off | 36/0.12 | 2.43 | 13.22 / 13.76 | loss 701–800 0.049; out-L1 `ctrl` 5.5–12.8 over 5 pages; probe s800 80 / 79 heads > 0.1 at step 0, L1 5.7 / 6.2; `cut` letters large masks, masked text partly returns | `reports/c_fill_mod16_bw10.md`, `output/tests/ref_gate/20261109-c_fill_bw10_mod16` |
| `c_fill_bw10_mod16_off_s800` | copy C | illu_bw10 | offset | off | 16 | — | 5e-5 | 800 (40) | 0.1 | off | 36/0.12 | 2.46 | 13.22 / 14.34 | loss 0.38–0.42 per 100 steps to step 500, falls from ~650 (701–800 0.140); s800 save does not copy: out-L1 `ctrl` 35.5–67.0, new pictures | `output/tests/ref_gate/20261110-c_fill_bw10_mod16_off` |
| `c_fill_bw10_mod16_off_kv_s1600` | copy C | illu_bw10 | offset | kv | 16 | — | 5e-5 | 1,600 (80) | 0.1 | off | 38/0.08 | 1.76 | 13.42 / 14.43 | loss 0.35–0.42 per 100 steps to step 1,100, falls from ~1,100 (1501–1600 0.225); s1600 out-L1 `ctrl` 31.3–48.9: same character and palette, pose redrawn | `output/tests/ref_gate/20261110-c_fill_bw10_mod16_off_kv_s1600` |
| `c_fill_bw10_mod16_off_kv_lr2e4_s1600` | copy C | illu_bw10 | offset | kv | 16 | — | 2e-4 | 1,600 (80) | 0.1 | off | 34/0.05 | 1.86 | 13.20 / 14.26 | loss falls at 400–600 (301–400 0.387 → 501–600 0.195); s1600 out-L1 `ctrl` 14.7–39.9; probe s800 L1 32.5 / 37.4, 2 / 5 heads | `output/tests/ref_gate/20261111-c_fill_bw10_mod16_off_kv_lr2e4_s1600`, `output/tests/copy_vs_step/20261111-canvas/offset_twin` |
| `c_fill_bw10_mod16_off_kv_lr2e4_s4800` | copy C | illu_bw10 | offset | kv | 16 | — | 2e-4 | 4,800 (240) | 0.1 | off | 34/0.05 | 1.86 | 13.23 / 14.28 | epoch loss 0.221 / 0.107 / 0.095; s4800 out-L1 `ctrl` 7.1–26.6; probe s4800 37 / 41 heads, L1 9.5 / 9.9; no early-block own-cell copy at any save | `reports/offset_kv_copy.md`, `output/tests/ref_gate/20261111-…_s4800{,_at2400}` |
| `1111_1232` | copy C | illu_bw10 | canvas | kv | 16 | — | 2e-4 | 1,600 planned, stopped at 818 (80) | 0.1 | off | 34/0.05 | 1.89 † | 12.28 / 13.58 at step 1; 13.15 max in step log † | loss tracks the offset twin (301–400 0.379 vs 0.387; 701–800 0.151 vs 0.162); probe s800 L1 32.7 / 35.8, 1 / 2 heads; s800 out-L1 `ctrl` 25.2–44.8 | `reports/offset_kv_copy.md` § Canvas, `output/tests/ref_gate/20261111-1111_1232_at800` |
| `1111_1459` | copy C | illu | aligned | kv | 0 | — | 5e-5 | 400 (20) | off | off | 38/0.10 | 1.68 | 13.00 | Copy head forms between s200 and s400 (aligned shared bank: between s100 and s200). Bin means 0.298 / 0.209 / 0.154 / 0.114 against `c_fill_s400`'s 0.285 / 0.144 / 0.073 / 0.076. Probe in § 3.5. Twin of `c_fill_s400` with only `ref_adapters` changed (and `fit_report`) | `_archive/configs/moss_ref_read_20261111/1111_1459.toml` |
| `probe_sanitize_copy_s300` | sanitize, copy timing | san | aligned | off | 0 | — | 5e-5 | 300 (15) | off | off | 38/0.10 | 2.25 | 10.20 / 10.63 | no tone; copy head phase change at steps 175–225 (h19 0.06 → 0.42 → 0.60); L1 15.4 → 8.0 by step 100 | `reports/copy_vs_step.md` |
| `sanitize_ft_k_s300` | sanitize ft, control | san | aligned | off | 0 | — (kaiming) | 5e-5 | 300 (15) | off | off | 36/0.15 | 2.29 | 12.31 / 12.86 | manga pages left with red blocks / smears; out-L1 mean 7.4–18.0 | `reports/p2_sanitize_ft.md` |
| `sanitize_ft_c_s300` | sanitize ft | san | aligned | off | 0 | `c_fill_s400` | 5e-5 | 300 (15) | off | off | 36/0.15 | 2.23 | 12.31 / 12.86 | removal on the hard pages as v3; out-L1 mean 6.7–16.6; ΔL swings 2–4 between saves | `reports/p2_sanitize_ft.md` |
| `sanitize_ft_cbw10_s300` | sanitize ft | san | aligned | off | 16 | `c_fill_bw10_mod16_s800` | 5e-5 | 300 (15) | off | off | 36/0.12 | 2.33 | 12.46 / 13.31 | removal as C-ft with less residue; out-L1 mean 6.2–17.7 | `reports/c_fill_mod16_bw10.md` |
| `sanitize_ft_off_kv_s300` | sanitize ft | san | offset | kv | 16 | `…off_kv_lr2e4_s4800` | 5e-5 | 300 (15) | off | off | 34/0.05 | 1.75 | 12.32 / 12.76 | worse than cbw10-ft by eye on every page (stamp back on hts1 s1/s2, lettering back on hts3); out-L1 mean 8.1–22.1; moved the init 10 % in ‖ΔW‖ (aligned ft 63 %) | `reports/offset_kv_copy.md` |
| `moss_i2i_character_ref_e1` | character_ref | cref | aligned | off | 0 | — | 5e-5 | 1,000 (50) | 0.1 | off | 36/0.15 | 1.66 | 12.73 / 13.33 | epoch loss 0.382; neither character nor style; bubble text follows the prompt | `output/moss_i2i_eval/character_ref/README.md` |
| `moss_i2i_character_ref_off_e1` | character_ref | cref | offset | off | 0 | — | 5e-5 | 1,000 (50) | 0.1 | off | 36/0.15 | 1.65 | 12.73 / 13.33 | paired vs e1, steps 300–1000: −0.24 % (sem 0.04); neither character nor style; keeps much of e1's composition | same; `results/sheet_e1_vs_off_e1.jpg` |
| `moss_i2i_character_ref_offkv_cinit` | character_ref | cref | offset | kv | 16 | `…off_kv_lr2e4_s4800` | 2e-4 | 1,000 (50) | 0.1 | off | 34/0.05 | 1.56 | 12.80 / 13.45 | paired vs off_e1, steps 300–1000: +7.3 % (ratio of means); no character, style or pose | same; `results/sheet_off_e1_vs_offkv_cinit.jpg` |
| `1111_1413` | character_ref | cref | aligned | kv | 0 | — | 5e-5 | 1,000 (50) | 0.1 | add90 ema97 | 34/0.05 | 1.70 | 13.40 / 14.49 | paired FM vs e1, steps 300–1000: +0.51 % at σ < 0.9, +1.35 % at σ ≥ 0.9; teacher MSE 0.0159 → 0.0185; renders are e1's scenes | same; `results/sheet_1111_1413_*.jpg` |

† `1111_1232` has no JSON (stopped by the user). Settings are from its LoRA metadata (`ref_pos canvas`,
`ref_adapters kv`, lr 2e-4, `ref_gate cut`) and the job's stdout (`shared_block_modulation rank 16`,
warmup 80, EMA from update 80): `output/daemon/jobs/20261111-123253-97bf8d/`. s/step is from the
progress timestamps over steps 100–818.

`fit_report`: false for `c_fill_s400`, `c_fill_v2*`, `c_fill_bw10_mod16_s800`,
`c_fill_bw10_mod16_off_s800`, `sanitize_ft_{k,c,cbw10}_s300` and `probe_sanitize_copy_s300`; true for
every kv run, `1111_1232`, `1111_1459` and all four character_ref runs.

Other runs that start from a C file and are not covered here: `hangul_s500`, `hangul_v2_s500`
(from `c_fill_s400`), `swap_dense2516_en400_c_e2` (from `c_fill_v2_s800`; `reports/swap_dense2516_main.md`),
`swap_v6_cbw10_e2` (from `c_fill_bw10_mod16_s800`).

## 3. What is established

### 3.1 aligned, shared bank (`ref_adapters off`), lr 5e-5

- The copy head forms in a phase change at steps 175–225 on sanitize pairs from kaiming (no gate,
  EMA off; `probe_sanitize_copy_s300`), and by step 200 on C v1's holed identity pairs (`c_fill_s400`:
  h19 0.63 at s200 against 0.42 on the sanitize pairs). `reports/copy_vs_step.md`, `reports/p1_c_fill.md`.
- Most of the pixel copy comes before the head: L1 15.4 → 8.0 by step 100 with own-cell mass at 0
  (`probe_sanitize_copy_s300`). Base, reference at aligned, has early own-cell heads b5 h3 0.33,
  b14 h15 0.21, b8 h14 0.19 (10 heads over 0.1), a position match that training leaves in place.
- The loss of the aligned C runs falls in the first 200–300 steps: per-100 means 0.285 → 0.144 →
  0.073 (`c_fill_s400`), 0.289 → 0.135 → 0.072 (`c_fill_bw10_mod16_s800`).
- `mod_rank 16` lowers the paired loss by 2.5–14 % per bin against C v2 (`c_fill_v2_mod16_s800`
  vs `c_fill_v2_s800`, same pages and σ; up to step 400 the modulation adapter is the only
  difference, after it C v2's tone term is the second). `reports/c_fill_mod16_bw10.md`.
- A 300-step sanitize fine-tune at lr 5e-5 from an aligned shared-bank C removes text on the hard
  manga pages as the shipped v3 does; from kaiming it leaves blocks (`sanitize_ft_c_s300`,
  `sanitize_ft_cbw10_s300` vs `sanitize_ft_k_s300`). Neither run holds the tone. `reports/p2_sanitize_ft.md`.

### 3.2 offset, shared bank

- Base with the reference at offset has no own-cell heads: 0 over 0.1, largest 0.02 (probe Base row of
  `output/tests/copy_vs_step/20261111-offset/offset`; the file is `sanitize_ft_off_kv_s300` at multiplier 0).
- `c_fill_bw10_mod16_off_s800` (lr 5e-5, 800 steps): per-100 means 0.39–0.42 to step 500; per 50
  steps 0.368 (501–550), then 0.309 / 0.338 / 0.251 / 0.145 / 0.134 for steps 551–800. The fill eval of the s800 save (EMA)
  shows new pictures, out-L1 `ctrl` 35.5–67.0. The loss falling by step 700 did not give a copy at s800.
- character_ref at offset (`off_e1`) against aligned (`e1`), shared bank, lr 5e-5, 1,000 steps:
  § 4.

### 3.3 offset + kv

| claim | configuration | evidence |
|---|---|---|
| The loss falls from ~1,100 steps | offset, kv, lr 5e-5, warmup 80 (`…off_kv_s1600`) | per-50 means 0.35 at 1001–1100, 0.28–0.31 at 1101–1200; per-100 0.376 (901–1000) → 0.225 (1501–1600) |
| Both σ bands fall in that run, σ < 0.7 the most | same | per-200 means from 601–800 to 1401–1600: σ < 0.7 0.346 → 0.179, σ ≥ 0.7 0.422 → 0.276 |
| At lr 2e-4 the loss falls at 400–600 steps, both σ bands | offset, kv, lr 2e-4, warmup 80 (`…lr2e4_s1600`) | per-100 0.387 (301–400) → 0.267 → 0.195 (501–600); per-200 σ ≥ 0.7 0.416 → 0.261 |
| The same lr with a 240-step warmup is slower to 600 and level from 800 | offset, kv, lr 2e-4, warmup 240 (`…lr2e4_s4800`) | per-100 401–500 0.338 vs 0.267, 701–800 0.179 vs 0.162, 1501–1600 0.131 vs 0.129 |
| The copy forms between s800 and s4800 and stays narrower than aligned | `…lr2e4_s4800` saves | heads over 0.1 at step 0: 0 (s800), 5–6 (s1600), 21–22 (s2400), 37–41 (s4800) vs aligned C s800 79–80; L1 52.9–61.6 → 9.5–9.9 (aligned s800 5.7–6.2); blocks 0–13 own-cell mass ≤ 0.04 at every save. `reports/offset_kv_copy.md` |
| The s4800 save copies on two of five fill pages | `…lr2e4_s4800` | out-L1 `ctrl`: usual_case1 7.2–9.5, hts1 7.1–11.7, hts2 13.2–19.8, usual_case2 19.4–26.6 (posterized patches), hts3 24.3–25.9 (slight zoom); aligned C 5.5–12.8 |
| A 300-step lr 5e-5 fine-tune barely moves this init | sanitize ft from `…lr2e4_s4800`, offset, kv, mod16 | ‖ΔW‖ of offset C ~10× aligned C's (q, blocks 0–13: 49.5 vs 4.2); ft − init relative 0.10 vs 0.63. `reports/offset_kv_copy.md` |
| kv needs more swap than the shared bank and runs faster | offset, illu_bw10, mod16 | swap 36 / 0.12 OOMs on the first backward under kv (13.85 GB allocated; comment in `_archive/configs/lora_blocks/c_fill_bw10_mod16_off_kv_lr2e4_s1600.toml`, no JSON). Shared bank 36 / 0.12: 2.46 s/step, 13.22 GB; kv 34 / 0.05: 1.86 s/step, 13.20 GB; kv 38 / 0.08: 1.76 s/step, 13.42 GB. The kv runs swap more blocks and are still 24–29 % faster per step. On cref: offkv_cinit 1.56 s/step (34 / 0.05, mod16) vs off_e1 1.65 (36 / 0.15, shared bank) |

On offset at lr 5e-5, the shared-bank run's loss fell at ~650–750 and the kv run's at ~1,100
(`c_fill_bw10_mod16_off_s800` vs `…off_kv_s1600`). That is one run each. The two runs also differ
in warmup (40 vs 80), swap / budget, and page order (`fit_report` false vs true). The shared-bank
run did not copy at its s800 save, and no shared-bank offset run went past 800 steps.

### 3.4 canvas + kv

`1111_1232` (canvas, kv, lr 2e-4, warmup 80) against `…lr2e4_s1600` (offset, kv, lr 2e-4, warmup 80;
same cache, σ and pages): loss per 100 steps overlaps (301–400 0.379 / 0.387; 401–500 0.264 / 0.267;
701–800 0.151 / 0.162). Copy probe s800: L1 32.7 / 35.8 vs 32.5 / 37.4, top-8 copy mass 0.64 / 0.71
vs 0.80 / 0.85, heads over 0.1 at step 0 1 / 2 vs 2 / 5, no early own-cell heads in either. Fill eval
s800: out-L1 `ctrl` 25.2–44.8; with `cut` it draws a second figure on hts3 on all three seeds.
Base reads canvas and offset coordinates alike under isolation (`docs/moss/proposal/ref_canvas.md`,
teacher-forced: iso −0.0012 / offset −0.0009 at σ 0.9). Measured only at lr 2e-4 with kv.

### 3.5 aligned + kv

The only aligned + kv run that has finished is `1111_1413` (character_ref, with the D-OPSD teacher, lr 5e-5). `1111_1459` (below) is the
only aligned + kv copy run.

`1111_1459` is the twin of `c_fill_s400` with `ref_adapters kv` (same cache, lr, warmup 20, EMA off,
mod_rank 0, swap 38 / 0.10). It also has `fit_report` true where `c_fill_s400` had false, so step k
of the two runs shows different pages (step 1 losses 0.338 vs 0.287 at the same σ 0.747). Compare
the two by per-100-step means and by the copy probe on s100–s400, not by per-step ratios.

Result (2026-11-11). Copy probe `output/tests/copy_vs_step/20261111-alignkv` (job `20261111-145945-1492d8`,
`copy_vs_step.py`, sanitize prompt, seed 0, both arms through `RefCache`); in0 = usual_case1, in1 = hard_to_sanitize1:

| arm | b39 h19 copy s0 (in0 / in1) | top-8 copy mass | heads > 0.1 | luma L1 / ref |
|---|---|---|---|---|
| base (multiplier 0) | 0.000 / – | 1.51 / – | 10 / – | 15.4 / – |
| shared s100 | 0.000 / 0.000 | 1.51 / 1.38 | 12 / 16 | 7.4 / 9.8 |
| shared s200 | 0.628 / 0.791 | 3.27 / 4.23 | 49 / 49 | 5.9 / 7.5 |
| shared s400 | 0.769 / 0.953 | 5.42 / 6.73 | 67 / 63 | 6.1 / 6.4 |
| kv s100 | 0.000 / 0.000 | 1.49 / 1.35 | 10 / 9 | 10.0 / 12.9 |
| kv s200 | 0.000 / 0.000 | 1.54 / 1.48 | 13 / 14 | 8.1 / 10.8 |
| kv s400 | 0.607 / 0.888 | 2.69 / 3.45 | 30 / 31 | 6.6 / 7.9 |

| | shared bank | kv |
|---|---|---|
| s/step (swap 38 / 0.10) | 2.40 | 1.68 |
| peak allocated | 10.55 GB | 13.00 GB |
| loss, steps 1–100 / 101–200 / 201–300 / 301–400 | 0.285 / 0.144 / 0.073 / 0.076 | 0.298 / 0.209 / 0.154 / 0.114 |

- On aligned, kv alone slows the copy: the b39 h19 head appears between s200 and s400 under kv,
  between s100 and s200 under the shared bank. At s400 kv has the head at the shared bank's s200
  strength, with ~30 heads over 0.1 against 49.
- By wall time kv is behind too: shared s200 (8.0 min) has the head, kv s200 (5.6 min) does not;
  at 11.2 min (shared ≈ s280, kv s400) the loss is 0.060 (shared, steps 241–280) against 0.109
  (kv, steps 361–400).
- By memory kv is behind: at the same swap and budget it peaks 2.5 GB higher (the export adapters
  keep the per-block reference y / k / v).

### 3.6 Not run

- aligned + kv at lr 2e-4.
- aligned + kv character_ref without the teacher, so `1111_1413`'s two changes against e1 (kv,
  teacher) are not separated; the teacher on a shared-bank character_ref run.
- character_ref from the aligned C (`c_fill_bw10_mod16_s800` as `init_lora`, aligned, shared bank).
- offset shared bank past 800 steps, or at lr 2e-4.
- canvas with the shared bank, or canvas + kv at lr 5e-5.
- the offset sanitize ft at lr 2e-4; an offset C trained shorter so that its ‖ΔW‖ is closer to aligned.
- a kv run with and without `mod_rank` (under kv the reference rows get Base modulation, so the
  reference-side Δmod(0) that was the reason for choosing bw10 does not apply to them).
- `ref_adapters all`; the reference at `ref_timestep 300`; the joint reference noised at the step's σ
  without `RefCache` (route 1 of `ref_canvas.md`).

## 4. character_ref

### Method

Same seed, same cache, `fit_report` true in all four runs: every step draws the same σ (1,000 / 1,000
against e1) and the same page (step 1–3 losses of e1 and `1111_1413` agree to 0.2 %). Each run's
per-step loss is divided by e1's (or off_e1's) at the same step. Tables give the mean of these
ratios with its standard error, and the ratio of the summed losses where the two differ. On
`1111_1413`'s teacher steps (σ ≥ 0.9, 196 steps) the logged `loss` includes the teacher term, so
FM there is `loss_fm`. That field is in both the JSON's `losses` and the job's
`output/daemon/jobs/20261111-141406-683120/progress.jsonl`.

Scale: under isolation Base's teacher-forced gain from the matching reference is about −0.001 MSE
at σ 0.3–0.9 against a ~0.4 loss (`ref_canvas.md`), so a loss difference of a few tenths of a percent
is the size of the whole effect a read could have. The renders are the read.

### Results, paired against e1

| run | steps 1–100 | 101–200 | 201–300 | 300–1000 | 801–1000 |
|---|---|---|---|---|---|
| off_e1 | −10.1 % (sem 1.3) | −1.5 % | −0.66 % | **−0.24 % (sem 0.04)** | −0.11 % (sem 0.05) |
| offkv_cinit | +9.4 % | +9.6 % | +11.5 % | +8.3 % (sem 0.17) | +8.2 % |

offkv_cinit against off_e1 (its single-axis-closer baseline: same ref_pos; differs in kv, mod16,
init, lr): steps 1–100 +22.7 % mean of ratios / +19.7 % ratio of means / +16.5 % median ratio;
300–1000 +8.5 % / +7.3 % / +7.7 %. The README's "+17 % to step 100, flat +7.5 % from 300" matches
the median and ratio-of-means figures. It is at +6.8–7.7 % in every 100-step bin from 301 to 1000
(ratio of means).

`1111_1413`, FM only, against e1:

| steps | σ < 0.9: n, mean ratio (sem) | σ ≥ 0.9: n, mean ratio (sem) |
|---|---|---|
| 1–100 | 86, +3.70 % (0.63) | 14, +2.2 % (1.0) |
| 101–300 | 152, +2.23 % (0.26) | 48, +8.2 % (1.75) |
| **300–1000** | **566, +0.51 % (0.03)** | **135, +1.35 % (0.22)** |
| 300–600 | 244, +0.76 % (0.05) | 57, +2.08 % (0.33) |
| 601–1000 | 322, +0.33 % (0.04) | 78, +0.82 % (0.28) |

Teacher MSE (`loss_t`, 196 teacher steps): mean 0.0159 over steps 1–300 (62 steps), 0.0185 over
301–1000 (134); 0.0184 over 300–650, 0.0186 over 651–1000 (medians 0.0125 → 0.0153). It does not
fall. Its size matches Base's zero-shot gap ‖v_img_cap − v_cap‖² = 0.0187 at σ 0.95
(`docs/moss/proposal/self_distill_lora.md`, probe table). That probe is Base on 30 caststation images
at σ 0.95 exactly. Here the teacher is the LoRA's EMA on character_ref pages over σ 0.9–1, so the
match is in order of magnitude, from different data.

### Renders

Eval: `output/moss_i2i_eval/character_ref/`, two held-out OCs (current inputs `oc_akipeko_10808866`,
`oc_yamamoto_souichirou_10278837`), seeds 0–2, prompt "singularity is coming" in a speech bubble.

| run | renders | source |
|---|---|---|
| e1 | neither character nor style; bubble text follows the prompt | `results/e1` |
| off_e1 | the same; much of e1's composition per seed | `results/sheet_e1_vs_off_e1.jpg` |
| offkv_cinit | no character, style or pose; more saturated and varied than off_e1 | `results/sheet_off_e1_vs_offkv_cinit.jpg` |
| `1111_1413` | neither character nor style; each seed is e1's picture redrawn | `results/sheet_1111_1413_{akipeko,yamamoto}.jpg` |
| C bw10 zero-shot, aligned (`c_init`) | returns the reference on every seed, prompt ignored | `results/c_init`, `ref_canvas.md` |
| C bw10 read at offset, untrained there (`c_init_off`, a re-stamped copy of the aligned file) | follows the prompt, washed grey figures, the reference's scene and palette at the reference's heights | `results/c_init_off`, `ref_canvas.md` |

None of the four character_ref runs reads the reference by eye. By loss, the reference's placement
moves e1's loss by 0.24 % from step 300 (off_e1 is lower; 10 % lower over steps 1–100).

## 5. User decisions on record

| date | decision | source |
|---|---|---|
| 2026-11-09 | The copy init is `c_fill_bw10_mod16_s800`, chosen mainly because it carries the modulation adapter (its t = 0 delta matches the reference-side direction the i2i edit tasks learn). Fine-tunes from it run with `mod_rank 16`. C v1 (`c_fill_s400`) and C v2 (`c_fill_v2_s800`) are superseded. | memory `project_copy_block.md`; `project/lora_blocks/README.md` |
| 2026-11-10 | Plan: judge the offset C (`c_fill_bw10_mod16_off_s800`) first; if it works, the copy line continues as offset + RefCache-compatible (`ref_adapters kv`), and the r = 1 workaround gets replaced. The kv runs followed after the offset C did not copy at s800; the memory does not record why. | memory `project_copy_offset_kv.md` |
| 2026-11-10 | C init LoRAs are shared-bank, so a kv C is trained fresh and kv fine-tunes init from it. The user accepted this: C is only 800 steps. | same |
| 2026-11-11 | No fixed displacement between image and reference: keep +W (recorded after the lr 2e-4 s1600 eval, before the s4800 run). | same |
| 2026-11-11 | `1111_1232` (canvas + kv) stopped at s800 by the user. | same; job `20261111-123253-97bf8d` state `stopped` |
| 2026-11-11 | Run names are `mmdd_hhmm` (submit time), written as the output name in the run's config with a header comment saying what the run is. | memory `feedback_experiments.md` |

## 6. Open questions

1. Answered 2026-11-11 (§ 3.5): on aligned, kv alone slows the copy, by step, by wall time and by
   memory.
2. On offset at lr 5e-5 the shared-bank loss fell ~400 steps before the kv one. Is that kv, or the
   warmup / page-order differences? Would the shared-bank offset C copy past 800 steps?
3. Can the early positional stage be given back under offset (aligned RoPE for the reference in
   the early blocks only)?
4. character_ref: no run reads the reference. Which of these starts the read: aligned + kv without
   the teacher, character_ref from the aligned C, the reference at `ref_timestep 300`, or the joint
   noised reference without `RefCache`?
5. Is the teacher MSE a usable progress signal for the read, given that it does not fall and is
   the size of Base's zero-shot gap on other data?
6. Does the reference-side modulation matter once the reference runs on Base modulation (kv), i.e.
   should kv runs keep `mod_rank 16`?
7. Part of the offset copy at s1600 may land a few cells off the own cell (the fill eval shows a
   small shift). That needs copy mass at displaced cells, which has not been measured.

## 7. Where sources disagree, and corrections to the retired memory notes

### Between files

| topic | one source | the other |
|---|---|---|
| bw10's outside-mask copy | `reports/c_fill_mod16_bw10.md` one-line summary: lowest L1 out "on all five pages" | the same report's table: lowest on usual_case1, hts1, hts2; C v1 lower on hts3 (5.60–5.67 vs 5.93–6.08) and on usual_case2 (6.31–6.37 vs 6.31–6.50); its Claim 3 says "four pages of five" |
| bw10 / mod16 peak memory | `reports/c_fill_mod16_bw10.md`: 13.04 GB | JSONs: 13.21 (mod16), 13.22 (bw10) GB allocated |
| C v1 / C v2 speed | `reports/p1_c_fill.md` 2.36 s/step, 16.4 min; `reports/c_fill_v2.md` 2.36 s/step, 31.5 min | JSONs: 2.40 s/step, 16.0 min; 2.34 s/step, 31.1 min |
| offkv_cinit speed | `output/moss_i2i_eval/character_ref/README.md` and `docs/moss/methods/i2i.md`: 1.47 s/step | JSON 1.56 s/step (26.0 min / 1,000); step-log mean 1.56 (`output/daemon/jobs/20261111-081214-02eb78/stdout.log`) |
| character_ref eval inputs | README first paragraph: akipeko/8351281, koh_(minagi_kou)/dan_7661898 | those are now in `inputs_old/`; every run past e1 renders `oc_akipeko_10808866` and `oc_yamamoto_souichirou_10278837` (`inputs/`) |
| e1's recipe | `configs/moss/i2i/character_ref/train.toml` now: 3 epochs, swap 38 / 0.10, save per epoch | e1 JSON: 1 epoch, swap 36 / 0.15 (as `docs/moss/methods/i2i.md` says) |
| `1111_1413` peak | `_archive/configs/moss_ref_read_20261111/character_ref_train_distill.toml` header: kv plan "peak 12.8 GB on this cache" | JSON 13.40 GB (the teacher adds to it; 12.8 is offkv_cinit's) |
| offset kv lr 5e-5, steps 1401–1600 | `_archive/configs/lora_blocks/c_fill_bw10_mod16_off_kv_lr2e4_s1600.toml` header: σ < 0.7 0.17, σ ≥ 0.7 0.36; "canvas 0.05 at σ ≥ 0.7" | every step: 0.179 / 0.276 (0.17 / 0.36 are the 20 logged 10th steps). "canvas" there is the aligned C `c_fill_bw10_mod16_s800` (σ ≥ 0.7 mean 0.042 at 601–800), not `ref_pos canvas` |
| offset C on the character_ref eval | `docs/moss/proposal/ref_canvas.md`, "Read 1": "the renders show it (the reference returned, prompt ignored)" | no character_ref render of the offset-trained C exists; `results/c_init_off` is the aligned file re-stamped to offset, which follows the prompt |
| "forms ~10× later" | `reports/offset_kv_copy.md` claim 3, repeated in the memory without configuration | compares offset + kv at lr 2e-4 with a 240-step warmup (s4800 run) against aligned shared bank at lr 5e-5; placement, `ref_adapters`, lr and warmup all differ |

### Memory statements found wrong, ambiguous or unsupported

`project_copy_block.md`:

- "the copy outside the mask is the best of any C": best on three of five pages; C v1 is better on
  hts3 and level on usual_case2 (table above).
- The rest checks out against the JSON and the report: recipe, swap 36 / 0.12, EMA 0.1, tone off,
  +0.5–1.1 L, the `cut` lettering and leak, "beat C v1 only slightly, one run".

`project_copy_offset_kv.md`:

- "canvas" is used for the aligned C `c_fill_bw10_mod16_s800` ("same VRAM as the canvas run",
  "out-L1 35–67 vs canvas 6–13", "canvas 0.098", "canvas 0.05"). Since `ref_pos canvas` exists
  (`1111_1232`), every such use is ambiguous; all of them mean aligned.
- "(compile marks the main axis without the reference, so VRAM drops)" is wrong; the same file
  corrects it later ("kv does NOT save VRAM … it saves time").
- "loss runs higher than canvas (steps 100–170: ~0.2–0.65 vs 0.08–0.38)": per-step range over all
  steps is 0.11–0.84 vs 0.025–0.42; the quoted ranges look like the logged 10th steps.
- "offset s800 shared bank did not copy" is true of the s800 render. It leaves out that the
  loss fell from step ~650 (701–800 mean 0.140, lower than the lr 2e-4 kv twin's 0.162 at the same steps).
- "kv lr 5e-5 s1600: … L1 31–51": `ctrl` out-L1 is 31.3–48.9; with `cut` 30.6–61.4.
- "loss broke ~1,200, only at σ < 0.7": the fall starts at ~1,100 and is in both σ bands (σ ≥ 0.7
  0.422 → 0.276), larger at σ < 0.7.
- s4800 "late σ ≥ 0.7 0.07–0.09 (canvas 0.05)": all-step means per 200 steps from step 3,201 are
  0.098–0.117 (medians 0.06–0.07); aligned C σ ≥ 0.7 is 0.042 (mean, 601–800).
- `_archive/configs/lora_blocks/sanitize_ft_off_kv_s300.toml` is `_archive/configs/lora_blocks/sanitize_ft_off_kv_s300.toml`.
- "forms ~10× later" carries no configuration (see the table above).
- "e1 and off_e1 losses are identical from step 300": they differ by −0.24 % (sem 0.04), off_e1
  lower, shrinking to −0.11 % at 801–1000; by −10 % over steps 1–100. "No char_ref run reads the
  reference yet" is a reading of the renders, which the loss does not contradict.
- "loss +7.5 % over off_e1": +7.3 % (ratio of means) / +8.5 % (mean of ratios) / +7.7 % (median), steps 300–1000.
- 1111_1232 "Trap: the s800 / s1600 offset numbers quoted in the config header (L1 53–62, 0 heads)":
  stale. The header of `_archive/configs/lora_blocks/1111_1232.toml` now quotes the twin's numbers (32.5 / 37.4).
- 1111_1413 "teacher MSE flat at ~0.018 (= Base's zero-shot img-in-TE gap)": the 0.0187 is Base on
  other data at σ 0.95 only; the agreement is in size, not a like-for-like measurement.
- 1111_1413 "Variance reduction alone did not start the read": the run differs from e1 in two
  things, kv and the teacher, so it does not isolate the teacher.
- Checked and still true on 2026-11-11: neither `~/ComfyUI-MOSS-Toolkit` nor
  `../comfy/custom_nodes/moss-image` reads `ref_adapters` / `ref_pos`; kv swap 36 / 0.12 OOMs
  and 34 / 0.05 peaks at 13.2 GB, ~1.86 s/step; the s4800 run took 148 min with warmup 240 and EMA
  from 240; its epoch losses 0.221 / 0.107 / 0.095; all fill-eval L1 ranges other than the one above.
