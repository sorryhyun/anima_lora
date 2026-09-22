# plan_z8 — `step1_0921z` on a borrowed 96 GB box (2026-09-21)

Index: [`README.md`](README.md). The live plan is [`plan.md`](plan.md); this
file is the logistics of its second table, **`step1_0921z` — one cold table
over everything `step1_0921` did not train** (multi-glyph single-token pieces
first, kanji by the same ranking), on a machine that exists for one night.
Goal (user, 2026-09-21): **a complete JA table; KO / ZH are parked** (bottom).
The recipe is `plan.md` § 1's grid mix. What the table is sized
in: [`reports/piece_coverage_2026_09_21.md`](reports/piece_coverage_2026_09_21.md).

> Every launch states its pack, **on the Z8 too**:
> `ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack` (raw, sha
> `7b9fce0bb57b`) — `configs/base.toml` there defaults to the preview pack
> exactly as it does here. Read the sha off the job log before walking away.

## Status — 2026-09-21 21:40

**On the Z8 now** (`~/anima_lora`, GPU 1 only, venv = `~/z8bench/anima_lora/.venv`
with `PYTHONPATH=~/anima_lora`):

- Working tree, raw pack (sha `7b9fce0bb57b`), DiT / TE / VAE, `step1_0921`'s
  `trained.pt`; `data_step1_0921z` + `_p0..p9` extracted, paths rebased,
  102 000 / 102 000 files resolve. `.env` holds the three model paths only.
- **Smoke done** (`z921_smoke`, 21:04 – 21:36, exit 0): pack sha `7b9fce0bb57b`
  in the job log; `te_cache/` 98 991 captions / 1 901 ext rows in 631 s
  (98 GB); `latents_mixed_*/` 12 shapes, 102 000 items in 1 192 s (25 GB); 300
  steps at **5.27 it/s** steady (steps 250 → 300; `it_s` in `train_log.json` is
  the cumulative mean, 3.4 with compile in it). `delta_norm_mean` 27.3 at the
  end of the 300-step cosine, `in_box` ≈ 0.20 / `out_box` ≈ 0.09. The cgroup
  sat at its cap throughout — 69.7 GB of it page cache from the chunked
  writes, anon 3.7 GB, `oom_kill` 0. `MemoryMax=70G` is 70 GiB (75.16 GB).
- **Running: `z921_s152k`**, launched 21:38:25 — the smoke's unit line
  (`systemd-run --user --unit z921_s152k --collect -p MemoryMax=70G
  -p MemorySwapMax=0`, the same four `--setenv`, log
  `output/wake_probe/z921_s152k.log`) and the smoke's argv with
  `--train_steps 152000 --save_every 5000 --arm_tag s152k`. 152 k at 5.27 it/s
  = 8.0 h → ends ≈ 05:45. Arm dir `rows_step1_0921z_s152k/`.
- `Linger=no` for `dss`: the unit lives in the user manager, which has been up
  since 09-17 behind the desktop session on seat0. A logout of that desktop
  ends the run.

Captions and latents are written to disk chunk by chunk and mapped back
(`common/models.py`, `train/stage.py`); every job on this box runs under
`MemoryMax`.

**Next, in order:**

1. Pull `trained_partial.pt` back whenever it is rewritten (every 5 000 steps
   ≈ 16 min); the whole arm dir before 08:30; clean the box
   (`~/anima_lora/output`, 125 GB of caches).
2. Owed here meanwhile: the eval set (own units by glyph count, head / tail of
   the rank range) and progress lines in the two encoders.

## The split

| where | what | why |
|---|---|---|
| here (5070 Ti box) | code, `--stage data` (CPU: grid + scene composites on the existing pools); the eval of record if the Z8 cannot load the readers | the corpus `boxes.jsonl`, scene pools and fonts live here; the data stage needs no GPU |
| Z8 (`dss@100.123.86.95`, Tailscale SSH) | the **cold table**: text-encode + latent cache (first-train side effects), `--stage train`, `eval` / `native` if the readers install | 96 GB, 125 GB RAM, 250 GB disk |

The Z8 as found (2026-09-21 16:40): 2 × RTX PRO 6000 Blackwell Max-Q 96 GB,
driver 595 / CUDA 13.2; **GPU 0 is someone else's vLLM (84 GB) — everything
here is `CUDA_VISIBLE_DEVICES=1`**. `uv` at `~/.local/bin/uv` (not on the
non-interactive PATH), `git`, `rsync`, `zstd`; no `tmux`, no sudo; HF and
GitHub reachable; link ≈ 30 MB/s direct (16 MB in 0.5 s). Same sm_120 arch as
the 5070 Ti, so the lockfile's torch build is the right one.

The tailnet SSH rule was switched `check` → `accept` on 2026-09-21, so ssh needs
no browser approval and does not expire; switch it back when the machine is
gone. Jobs still go through the repo daemon on the Z8 (`--queue`) — no `tmux`,
and nothing should depend on a live ssh.

## The window (user, 2026-09-21)

**The Z8 is a shared company workstation. Nothing of this plan touches it —
no rsync, no `uv sync`, no smoke — until everyone there has left for the day**
(the user says when), and **everything is finished and pulled back before
09:00 on 2026-09-22**. GPU 0's vLLM tenants are left alone at any hour. So the
window is one night, and everything that can be done here beforehand is done
here beforehand: the code owed, every data build, the `args.json` files, the
transfer bundle, the exact argv of every job. (`~/z8bench/anima_lora` — a
public-commit clone with a synced `.venv` — is already on the box from the
user's benchmark run; bring-up reuses its venv.)

## Speed — measured

- **5.62 it/s mixed at batch 4** on the Z8's GPU 1 (user ran `z8_dit_bench`:
  the real 28-block Anima + llm_adapter, random weights, block compile,
  flash-attn, the run's four canvas shapes; 5.65–5.78 per shape, peak 12.6 GB).
  DiT forward + backward only, so the real loop lands a little under it:
  **plan on 5.3 it/s**, ≈ 2.3 × the 5070 Ti's 2.29. The first 300 steps of the
  first real job give the final number.
- **Batch stays 4.** An eager stand-in read 26.5 / 26.3 / 25.5 items/s at batch
  4 / 8 / 16: the card is compute-bound at batch 4, the 96 GB buys no
  throughput, and a bigger batch would be a recipe change for nothing. Tables
  run one after another through the daemon, not side by side.
- No parallel run here (user, 2026-09-21): the Z8's table comes back and is
  merged; whatever ranks it does not reach are an ordinary later run.

## What "complete" is, and what each GPU takes

`step1_0921` trains 374 rows (kana, `kana_ext` incl. the small kana, 13
punctuation units, `kanji:200`). **Everything else a dialogue line needs is the
cold list**: the joint frequency ranking of multi-glyph single-token pieces and
single kanji against `step1_0921`'s own row set
([`reports/piece_coverage_ranked_2026_09_21.tsv`](reports/piece_coverage_ranked_2026_09_21.tsv),
re-cut against `step1_0921`'s `ext_ids` — 85 kanji of the report's approximate
"today" were already warm — with the 10 units of 6 + glyphs dropped →
`assets/units/ja_cold_0001_1900.txt`, last source rank 1 983). Multi-glyph pieces are 35 % of dialogue piece tokens
against single kanji's 16 %, and a row of them buys ≈ 12 × the covered lines —
all 1 452 remaining corpus kanji take coverage 13.5 → 17.6 %, the joint list
takes it to 95 %.

| cold ranks | table | rows | steps at 80 per row | covered dialogue lines, cumulative with `step1_0921` | where | hours |
|---|---|---|---|---|---|---|
| — | `step1_0921` | 374 | 30 k | 13.5 % | here | done |
| **1 – 1 900** | **`step1_0921z`** (583 kanji, 1 317 multi-glyph) | 1 900 | 152 k | **≈ 85 %** | Z8 GPU 1, 5.3 it/s | 8.0 h |
| 1 901 – 3 000 | later, here (≈ 415 kanji, ≈ 685 multi-glyph) | 1 100 | 88 k | 95.1 % | — | — |
| 3 001 – 4 132 | later | 1 132 | — | 98.4 % | — | — |

**Disjoint-id** tables, unioned by `src/probe/merge_tables.py` with the
`row_scale` correction. A disjoint cold table is `plan.md` § V's own design
("a separate step-1 table on the step-1 recipe, merged by ext id"), and the
merge is mechanically sound (`ExtDelta` is per-row additive; krzh16). What it
costs when rows from different tables meet in one sentence is K0's question,
still unrun — the merged table's first read answers it.

**80 steps per row stands.** `step1_0921` (374 rows, 30 k = 80 per row) read
28 / 20 / 10 / 19 and native 34 / 28 against `step1_0920`'s 20 / 8 / 0 / 8 and
19 / 8 — the mean rates hold, so the table is 1 900 rows. The multi-glyph gate
V0 is **dropped** (user, 2026-09-21): the data was reviewed on sheets instead,
and whether a multi-glyph piece trains as one unit is read on the table itself.
One difference from `step1_0921` to keep in mind when reading it: the
length-aware deal moves the grid share to 2x2 (5.25 cells per item against
6.3), so a row gets ≈ 840 cell draws over 152 k steps against `step1_0921`'s
≈ 1 010.

Night budget for the Z8, working back from 09:00: results pulled and the box
cleaned by 08:30; bring-up + smoke ≈ 0.5 h (venv already there), the table
8.0 h → **the window has to open by ≈ 23:30**; every hour later costs
≈ 240 rows off the list's tail end (cut the list to the clock before the
build ships, not mid-run — a cosine run cannot be stopped early).

Disk / RAM on the Z8: 102 000 items → prompt embeds ≈ 107 GB (temp file,
`TMPDIR` on the data disk; 250 GB free), latents ≈ 27 GB RAM of 125, PNGs
**16 GB** to transfer (≈ 9 min at 30 MB/s).

## Inputs owed from the user

1. **When the window opens** (everyone gone). T_end is 09:00 on 09-22.

## Is the data mechanical? Yes for the grid, nearly for the scene half

- **Grid items** are font renders on a flat / rounded-box canvas with one
  position clause per cell: no model, no pool, CPU only, ≈ 9 min per 10 000
  (`data_m2_grid`). Any unit with a pack row and a covering font can be dealt;
  `draw_grid` already stacks a multi-glyph unit as a column or a line and
  shrinks it to the cell.
- **Scene singles** composite one unit into a bubble of an already-rendered
  scene (`scenes_s1 / s1w / sl1w / ja_comic`, 1.6 GB, reused per the standing
  rule). Also CPU, also here. No new pool is rendered for this plan.
- **Grid alone is never a seed table** (S0), so both tables are the 50 % mix;
  the build is one process per shard (`--seed k`), 10 shards side by side:
  102 k items in ≈ 20 min on 12 cores.
- Every cold piece is one Qwen piece with a pack row by construction of the
  ranking (no-row piece mass in the corpus: 0.0 %).

## Order

### Before the window — here

1. **Code** — done: `list:@<file>` (mixes with literals), the length-aware
   grid deal `--grid_unit_min_glyph 56`, `--grid_mark_horizontal 1`
   (`horizontal Japanese text reads as`), `--seed` moving every data stream;
   tests + CLI golden. All default-off.
2. **The cold list** — done: `assets/units/ja_cold_0001_1900.txt`.
3. **The build** — done: `output/wake_probe/data_step1_0921z` (`train.jsonl`
   joined from `data_step1_0921z_p0..p9`, which hold the images; `args.json`
   has the argv and the checks). Scene items per unit min 10 / median 27 /
   max 50, grid 140 – 145, no unit at zero; every unit's own ext id is in its
   caption and no other ext row is (102 000 items); review sheets in
   `data_step1_0921z_rev/`.
4. **Owed:** the eval set (own units as strings, by glyph count, head and tail
   of the rank range — `eval.json` is the list's first 18 today); the transfer
   bundle (working tree + `data_step1_0921z` + `data_step1_0921z_p*` + the raw
   pack), the path-rebase `sed`, and the train argv written out.

### Z0 — bring-up on the Z8 (when the window opens)

1. `rsync` the **working tree** (not a clone — `project/cjk_renderable_anima`
   has uncommitted changes and `main` is 5 commits ahead of the public repo),
   excluding `output/ models/ .venv/ .git/ bench/ _archive/ workspace/ docs/
   image_dataset/ post_image_dataset/ project/finished/ project/qwen21_lora/`.
   Venv: reuse `~/z8bench/anima_lora/.venv` if the lockfile matches, else
   `~/.local/bin/uv sync --frozen`.
2. Models: `make download-models` on the Z8 (HF reachable), **the raw pack
   copied from here** (`anima_cjk_vocab_pack.{json,safetensors}`, 285 MB) and
   its sha checked in the first job log.
3. Data: `rsync` `data_step1_0921z` and the ten `data_step1_0921z_p*` dirs,
   rebase the absolute paths in `train.jsonl` (`sed 's#/home/sorryhyun/anima/anima_lora#/home/dss/anima_lora#g'`).
4. Smoke = the table's own argv at `--train_steps 300`: the real it/s.
5. Readers: does `common.readers.Readers` load on the Z8 (anime_tools OCR
   weights, the frozen line's `ocr/pseudo_label`, VL16)? Yes → eval runs there.
   No → `trained.pt` (a few MB) comes back and the read runs here.

### The long run

`step1_0921`'s train argv with `--data_tag step1_0921z`, `--units 'list:@ja_cold_0001_1900.txt*1'` and `--train_steps 152000` — plain, `--lr_rows 1e-3`, cosine, σ 0.7–0.9,
`--box_share 0.25`, grid 50 %, cold, `--save_every 5000`.

- Early read: `delta_norm_mean` per draw against `step1_0921`'s curve over the
  first few thousand steps (a row appears in ≈ 0.2 % of batches at 1 900 rows;
  AdamW β₂ decays `v` between visits — K1's guard).
- Eval set: its own units as `single_extra`, **split by glyph
  count**, sampled across the rank range (head and tail of the list read
  separately — rank 1 800 sees the same draws as rank 1 but is a rarer piece).
- Gate: multi-glyph pieces exact **as strings** at not below
  `step1_0921`'s kana rate, by glyph count; kanji not below `step1_0921`'s
  `single_kanji`.
- `trained_partial.pt` is pulled back whenever it is rewritten — the box can
  vanish mid-run, and a cosine partial is only crash insurance, but it is the
  only copy.

### After 09:00 — here

1. Merge `step1_0921` + the Z8 table (`merge_tables.py`, `row_scale`
   corrected); the 374 shared-eval rows inside `step1_0921`'s rerun floor, a
   native read, and **a JA caption that mixes rows of both tables** —
   K0's first real instance.
2. The read this table exists for: `step2_0920b`'s data argv rebuilt on the
   merged inventory (the sentence pool opens from 538 sentences to ≈ 30 000),
   sized to the pool.
3. Piece / glyph-row conflict (って the piece vs っ + て the rows) — a free
   `table_geometry.py` read.

## Every build writes its argv

The data stage records none. Each `data_<tag>/` built for this plan gets an
`args.json` by hand, and the Z8 arm dir comes back whole (`report.md`,
`train_log.json`, sheets, `trained.pt`) into `output/wake_probe/` here before
08:30; the Z8's daemon `job.json` is the only argv record of its train and goes
with it. Reports: one `reports/z8_<what>_2026_09_22.md` per read.

## Parked: KO / ZH

Not tonight (user, 2026-09-21: finish JA first). What was settled while
planning it, for whoever reopens it:

- Premise holds (krzh16): the DiT holds Hangul and simplified units. The pack
  has 2 512 single-row Hangul syllables and 8 501 single-row han.
- **The clause is the bare `text reads as "…"`**, not `Korean` / `Chinese text
  reads as` (user: the base model knows `Japanese text reads as` and the bare
  form, not those). First gate = `probe/position_probe.py` on the base model
  over bare / `Japanese` / `English` clause forms.
- Code owed: a `--clause_lang` data flag (`grid_caption`'s `lang` parameter
  exists, nothing passes it) through every template; the `.ttc` face per script
  (Noto CJK index 0 JP, 1 KR, 2 SC) and `NotoSansCJK*` in `find_fonts` — Hangul
  and simplified-only forms exist in the 14 Noto weights only (font-diversity
  confound); `chars:@<file>`; a real frequency list (Qwen-id order as the
  zero-dependency fallback).
- KO reads on the VL reader (the SFX reader is JA-only). ZH drops every
  codepoint a JA table trains — shared han are one row.

## Not this plan

- Spending the night on `kanji:N` alone (17.6 % of lines at best).
- A bigger batch because the VRAM is there (items/s is flat from batch 4 to 16).
- A parallel table on the 5070 Ti the same night (user: merge later instead).
- Retraining `step1_0921`'s 374 rows on the Z8; warm-starting the cold tables
  from it (disjoint ids — there is nothing to warm).
- Hand-pinned deploy strings: the ranking is the inventory.
- Rendering a new scene pool; grid-only tables; a grid share above 50 %.
- ΔFM anywhere; random multi-glyph strings; step 2 on the Z8.
- Leaving the only copy of anything on the Z8.
- Touching the Z8 during working hours, or GPU 0 at any hour.
