# ΔFM Δ0 — paired-difference supervision on 12 dakuten rows, and the lr arm (2026-09-17)

> `plan_synth2.md`'s Δ0: plain FM vs ΔFM (`--pair_loss 1`, EN sibling under
> the JA frame) on the same 2 000-item data dir, 1 500 / 3 000 steps, then
> one ΔFM arm at `--lr_rows 2e-3`. **The plan's pass gate is missed** —
> ΔFM saves no draws at lr 1e-3 (10/24 vs 20/24 at 1 500 steps, 18 vs 21 at
> 3 000). **lr is a lever**: ΔFM at 2e-3 × 1 500 steps reads 19/24, the
> 3 000-step arm's number at half the steps — the ΔFM deficit was travel
> (lr integral), not information. On natives ΔFM removes the wipe tail
> (en cos 0.93 vs 0.855, images under 0.85: 6 vs 20 of 64) but halves the
> reader hits: the glyph lands as scene text beside **JA pseudo-text the
> plain arm suppresses**. Joint number (both readers hit ∧ en cos ≥ 0.85)
> still favours plain FM, 28 vs 18 of 64.

## What ran

Code as built (commit `2ffacc78`): `src/data/pair.py`, `scene.py`
`ref_text=`, `synth.py` pairing + `sheet_scene_pair.png`, `train/stage.py`
`pair_branch` / `pair_stats`; data flags `--pair_ref en --pair_ref_pool 4`,
train flags `--pair_loss 1`, `--pair_sigma_min`. Log fields `fm_plain`,
`pres`, `ref_bias` as the plan specified.

Data: `output/wake_probe/data_synth_pair_d0` — 2 000 composites, singles
only, `chars:がぎぐげござガギグゲゴザ`, every item paired, 1 020 reference
captions, three token families (shapes 384×640 … 640×448). `chars:` units
land in eval group **`single`** (the plan wrote `single_extra`).
`sheet_scene_pair.png` read: sibling = same scene, same erase, same box,
one Latin letter, romaji excluded — correct on all 20 pairs shown.

Train argv (all arms; only the marked flags move):

    project/cjk_renderable_anima/src/wake_probe.py --stage train eval --arm rows
    --data_tag synth_pair_d0 --units chars:がぎぐげござガギグゲゴザ --shapes 512
    --train_steps <1500|3000> --batch 4 --t_min 0.7 --t_max 0.9 --compile 1
    --grad_ckpt 0 --aggressive_recompute 0 --lr_rows <1e-3|2e-3> --lr_decay cosine
    --free_residual 1e-3 --box_weight 4 --seeds 2 --no_floor --c_flat 0
    --pair_loss <0|1> --arm_tag <arm>

Native argv: `--stage native --arm rows --data_tag synth_pair_d0 --arm_tag
<arm> --native_chars が,ガ,ご,ゴ --native_clauses en,swap --seeds 2
--delta_parts full` (scene-kept ruler off — no floor renders; the
EN-reference ruler is the preservation read).

| arm (`rows_synth_pair_d0_<arm>`) | job | wall (train + eval) |
|---|---|---|
| `pairEN_s1500` | `20260917-173051-8679dc` | 23.9 min (first compile) |
| `pair0_s1500` | `20260917-173623-9e5e33` | 17.3 min |
| `pairEN_s3000` | `…-173623-02f94e` | 37.7 min |
| `pair0_s3000` | `…-173623-eec2b5` | 28.0 min |
| native `pairEN_s3000` / `pair0_s3000` | `…-381ea6` / `…-ea404c` | 9.2 / 8.8 min |
| `pairEN_s1500_lr2e-3` | `20260917-194144-490b76` | 22.5 min |
| native `pairEN_s1500_lr2e-3` | `20260917-200512-29dcad` | 9.2 min |

Throughput: plain 2.35 it/s, ΔFM 1.63 it/s (0.69×; the plan priced 1.7).
The `no_grad` sibling forward compiled without trouble.

## Singles (12 chars × 2 seeds, flat eval prompt, exact on the sfx read)

| arm | lr | steps | `single` | CER sfx / vl | row norm end (peak) |
|---|---|---|---|---|---|
| `pair0_s1500` | 1e-3 | 1 500 | 20/24 | 0.167 / 0.083 | 120 (121) |
| `pair0_s3000` | 1e-3 | 3 000 | 21/24 | 0.125 / 0.167 | 149 (153) |
| `pairEN_s1500` | 1e-3 | 1 500 | 10/24 | 0.583 / 0.583 | 92 (96) |
| `pairEN_s3000` | 1e-3 | 3 000 | 18/24 | 0.250 / 0.292 | 115 (126) |
| `pairEN_s1500_lr2e-3` | 2e-3 | 1 500 | **19/24** | 0.208 / 0.125 | 144 (165) |

EN 24/24 on every arm; `line` / `combo` at the floor on every arm (rows
untrained for them).

- **Gate:** `pairEN_s1500 ≥ pair0_s3000` — no (10 vs 21); `pairEN_s3000 ≥
  pair0_s3000 + 4` — no (18 vs 21). With lr doubled ΔFM ties plain FM at
  equal draws (19 vs 20), still under the gate and at 1.45× the wall.
- **The control has no headroom.** Plain FM reads 83 % at 500 draws per row
  on 12 rows; the 36 % point of the exposure curve is a 53k-table number.
  `pairEN_s1500` would have needed 21/24 to pass. A 12-row inventory
  cannot show an exposure saving at these step counts.
- **Hits track row travel, not the loss.** `pairEN_s3000` (norm 115–126) ≈
  `pair0_s1500` (norm 120) at 18 vs 20; `pairEN_s1500` at norm 92 reads
  10; `pairEN_s1500_lr2e-3` at norm 144 reads 19. ΔFM travels slower per
  step at equal lr: the paired loss is 0.013–0.02 against `fm_plain` ≈ 0.10
  on the same items (≈ 80 % of the residual cancels), the cancelled part
  is the sign-consistent part Adam marches on, and `--free_residual 1e-3`
  is added unscaled to a 5–8× smaller loss (norm peaks at 126 then sags to
  115 where the plain arm holds 149–153). Doubling lr overshoots that:
  norm 147 by step 250 (78 at 1e-3).
- **`fm_plain` of the ΔFM arms ≈ `fm` of the control** at equal steps
  (0.08–0.13 both) — the variance reduction does not show as a lower plain
  residual.
- **`ref_bias`** settles at 0.26–0.28 by step 250 and stays there on all
  three ΔFM arms — under the 0.3 gate, never decaying. `pres` ≈ 3–8e-4.
  The plan's fallback `--pair_sigma_min 0.7` is a no-op on this recipe
  (`--t_min 0.7`).
- **Per glyph.** ご fails in every arm at both seeds (`ぶし` / `ござ` /
  `ふし` — reads as a word prior, not a loss effect). ΔFM at 1e-3 × 1 500
  misses ぎ げ ガ グ ゲ at both seeds; at 2e-3 ガ ゲ ザ hit at both
  seeds, グ s0 draws ダ. `pair0`'s misses are ご ×2 and ザ s1 (reads
  `チ`), plus ぎ s1 at 1 500 only.

## What the single sheets show (seed 0)

- `pair0_*`: one large glyph on the white canvas; a bubble in 1–2 of 12,
  holding the glyph again.
- `pairEN_s1500`: the glyph **inside** a round bubble on 5 of 12 (the
  training layout), identity unfinished (ガ → ズ, グ ゲ malformed).
- `pairEN_s3000`, `pairEN_s1500_lr2e-3`: large glyph on the canvas **plus a
  tall bubble with a four-glyph vertical pseudo-kanji column** (`唷だ掲か`,
  `ただ掴く`, `塀だ掴ナ`) on 10 / 8 of 12; the column's last glyph is often
  the target without its dakuten (か く さ イ ケ). lr changes identity, not
  the column.

## Natives (が ガ ご ゴ × 8 prompts × 2 seeds, 64 per clause)

| arm | clause | hit sfx | hit vl | both | any CJK | en cos | box IoU | en cos < 0.85 | both ∧ cos ≥ 0.85 |
|---|---|---|---|---|---|---|---|---|---|
| `pair0_s3000` | en | 45 | 49 | 40 | 64 | 0.855 | 0.12 | 20 | **28** |
| `pairEN_s3000` | en | 25 | 26 | 21 | 64 | 0.931 | 0.24 | 6 | 18 |
| `pairEN_s1500_lr2e-3` | en | 24 | 20 | 18 | 64 | 0.935 | 0.32 | 6 | 15 |
| `pair0_s3000` | swap | 34 | 29 | 22 | 53 | 0.889 | 0.17 | 16 | **16** |
| `pairEN_s3000` | swap | 16 | 12 | 12 | 44 | 0.975 | 0.45 | 0 | 12 |
| `pairEN_s1500_lr2e-3` | swap | 16 | 13 | 12 | 44 | 0.975 | 0.48 | 0 | 12 |

(last two columns computed from `native/native_reads.json`; 0.85 is a read
threshold, not a gate — at 0.9 the en clause reads 23 / 17 / 15.)

- **Plain FM, seed 0:** 4 of 8 が prompts are a white canvas with one large
  が (en cos 0.54–0.63) and two more keep only a ghost of the scene — the
  wipe. Seed 1 keeps the scene and hits.
- **ΔFM:** the scene is kept on all 16 が and all 16 ガ images; the glyph is
  visible as scene text (board, subtitle bar, panel corner) on ≈ 13 / ≈ 11
  of 16, next to lines of JA pseudo-text. The readers score the largest
  box or the whole image and read the pseudo-text, so reader hits
  under-count what the sheet shows — but the co-text is a real defect, not
  a reader artefact.
- **Equal norm, different wipe.** `pairEN_s1500_lr2e-3` (norm 144) vs
  `pair0_s3000` (norm 149): en cos 0.935 vs 0.855, box IoU 0.32 vs 0.12.
  The wipe is not the delta norm alone; the outside-box term holds the
  scene at the same norm.
- lr 2e-3 leaves the native read where `pairEN_s3000` had it (18 vs 21
  both, same en cos) — same table by a shorter road.
- ご / ゴ `swap`: 0–2 of 16 on every arm.

## Reading

1. **lr is a lever, and the budget is the lr integral.** 1 500 steps at
   2e-3 ≡ 3 000 steps at 1e-3 for ΔFM on singles and natives alike. Every
   exposure number of the S line (≈ 1 000 draws per row, the 1 330 / 670 /
   490 curve) was measured at one lr with a cosine, so draws and ∫lr were
   never separated. Plain FM at 2e-3 is unmeasured (3e-3 is closed:
   off-manifold at 2.4× row norm); the 12-row control is ceilinged at
   1 500 steps, so the test needs a shorter arm.
2. **ΔFM is not an exposure saver.** No point on the Δ0 curve beats plain
   FM at equal draws; at equal wall it is behind.
3. **ΔFM is a preservation term that also preserves the base's JA-frame
   text prior.** The residual both branches share is not only scene
   posterior + paste artefacts: under `japanese text … reads as "…"` the
   base wants JA pseudo-text, and that error is in `r_A` and `r_B` alike.
   Plain FM's gradient on it is what teaches the row "this glyph and
   nothing else" (and, overdone, the wipe); ΔFM cancels it, and outside
   the box trains the row to match the sibling's render, pseudo-text
   included. (Inference from the sheets + `ref_bias` 0.27; no floor
   renders were made on these prompts — the floor's JA CER 1.000 is the
   record.)

## Δ0b arm 1 — `--pair_sigma_min 0.8` (2026-09-17 20:22): moves toward plain FM along the same axis, gate not cleared

`pairEN_s1500_lr2e-3_smin0p8` = the lr 2e-3 argv + `--pair_sigma_min 0.8`
(paired loss on σ ≥ 0.8, plain FM on 0.7–0.8). Jobs `20260917-202206-52f1fb`
(train + eval, 23.3 min) / `…-57aa68` (native, 9.4 min). The plain samples
carry ≈ 0.1 loss against ≈ 0.015 paired, so the batch gradient leans plain;
logged loss 0.02–0.10, row norm 168 (peak 190 — above every Δ0 arm),
`ref_bias` 0.27 unchanged (it is computed on the sibling, not on the mix).

| arm | `single` | seed-0 singles without the column | native en both | joint en | tail en < 0.85 | en cos | swap both / joint / tail |
|---|---|---|---|---|---|---|---|
| `pair0_s3000` (plain) | 21/24 | 12/12 | 40 | 28 | 20 | 0.855 | 22 / 16 / 16 |
| `pairEN_s1500_lr2e-3` | 19/24 | 4/12 | 18 | 15 | 6 | 0.935 | 12 / 12 / 0 |
| `…_smin0p8` | 20/24 | 7/12 | 22 | 20 | 9 | 0.919 | 12 / 12 / 0 |

- Gate (joint en ≥ 28, swap ≥ 16, tail ≤ 10, `single` ≥ 19, column-free
  ≥ 8/12): singles and tail pass, joint and the column do not. `swap` did
  not move at all.
- On the (tail, joint) plane the arm sits between the two parents: a
  straight line from ΔFM (6, 15) to plain (20, 28) gives ≈ 18 at tail 9;
  measured 20. No evidence yet that the split leaves the wipe ↔ co-text
  axis — it is what the plan's first open risk looks like.
- Sheets: the scene is kept on all 16 が natives; the glyph is larger and
  cleaner than under pure ΔFM (p03 both seeds, p04 s0, p06, p07), lines of
  JA pseudo-text remain on p00 / p01 / p02 / p05. Singles: が s0 draws だ
  inside the bubble; げ ガ グ ゲ ザ keep the pseudo-kanji column.

## Δ0b arm 2 — `--pair_ref_frame en` (2026-09-17 20:58): flat; the co-text is not in the sibling's caption

Hypothesis tested: under the JA frame the sibling's residual `r_A` carries
the base's JA pseudo-text prior, the paired loss cancels it, the row never
learns to suppress it; put A's caption back under the scene's EN frame and
that part stays in the loss. Code: `data/pair.py::en_frame` (inverse of
`scene_caption`'s swap), train flag `--pair_ref_frame ja|en`, test
`test_pair_en_frame_inverts_scene_caption`, CLI golden regenerated. Of the
2 000 sibling captions all carry the `japanese text` tag, 558 the `Japanese
text reads as` clause; the rest are language-neutral frames (`She is saying
"…"`), so for 72 % of items the change is the tag alone.

`pairENf_s1500_lr2e-3` = the lr 2e-3 argv + `--pair_ref_frame en`, no
σ split. Jobs `20260917-205820-2db0e4` (23.5 min) / `…-0a0edd` (native).

| arm | `single` | column-free seed-0 singles | en both / joint / tail | en cos | swap both / joint / tail | mean paired loss | `ref_bias` |
|---|---|---|---|---|---|---|---|
| `pairEN_s1500_lr2e-3` (JA frame) | 19/24 | 4/12 | 18 / 15 / 6 | 0.935 | 12 / 12 / 0 | 0.0156 | 0.27 |
| `pairENf_s1500_lr2e-3` (EN frame) | 17/24 | 9/12 | 18 / 18 / 6 | 0.932 | 13 / 13 / 0 | 0.0154 | 0.24 |
| `pair0_s3000` (plain) | 21/24 | 12/12 | 40 / 28 / 20 | 0.855 | 22 / 16 / 16 | — | — |

- **The hypothesis is wrong where it matters.** The paired loss is the
  same to the third decimal from step 1 (0.0207 vs 0.0237) to the mean —
  the sibling's frame does not change how much of the residual cancels, so
  the pseudo-text error was never a measurable part of `r_A` or `r_B` at
  the training σ. Training is teacher-forced on a real scene with one
  glyph; the pseudo-text is a free-running inference behaviour of the base
  under `japanese text`, and no training residual of this recipe sees it.
- Natives unchanged: scene kept on all 16 ガ images, glyph as scene text,
  lines of JA pseudo-text on p00 / p01 / p03 / p05 / p07. The singles
  column fell 8 → 3 of 12 seed-0 images (12 images; not a native effect).
- **What suppresses the co-text under plain FM is therefore the wipe
  itself** — the row's "bare canvas, one glyph" component overriding the
  base's free-running scene, pseudo-text included. ΔFM cancels the scene
  residual that builds that component, so it removes both together. Two
  arms (σ split: interpolates; EN frame: flat) now agree with the plan's
  first open risk: wipe ↔ co-text is one axis, and mixing the two losses
  moves along it.

Decisions and next arms: `../plan_synth2.md`.
