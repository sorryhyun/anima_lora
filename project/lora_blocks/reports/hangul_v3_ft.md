# v3 data: ksplit e2 fine-tuned 600 steps on synthetic lettering + its own pairs

Date: 2026-11-08. Follow-up to [hangul_ko10.md](hangul_ko10.md). Sheets (`output/tests/lora_blocks/`):
`hangul_v3_ko10/zoom_<page>_s0.png` (parts in `parts/`), `hangul_v3_eval2/zoom_<page>_s<seed>.png`;
rows edition, ksplit e2, the fine-tune.

Result in one line: **mixed, not a win. The fine-tune removes e2's pseudo-glyphs on the long
narration of 589883_0015 and fixes a few syllables, but garbles a balloon and a line e2 wrote
right, draws marks in a balloon, and adds vowel errors; on the eval pages and most 737486
pages it renders as e2 does.**

## Setup

- **Data** (`src/build_hangul_v3.py`, `post_image_dataset/i2i/hangul_block_v3/`): v2's
  builder plus flat boxes (ring ≥ 92 % within ±20 luma of its median on both sides of the
  twin, no other OCR box in the margin; filled with the ring's ground, light ink on dark
  ground), strings up to 120 syllables, left alignment read off the edition, Regular /
  Medium faces first (Pretendard Regular 3, Nanum Gothic 2, Gangwon Edu All Bold 2,
  Pretendard Bold 1, KMU HaeongSans 1), and p 0.2 one syllable swapped for a KS X 1001
  syllable absent from swap_text v3's Korean captions (1,336 such; 267 used). 800 pages,
  1,435 lines: 1,214 balloon, 221 flat (33 dark, 48 left-aligned); 976 syllables. Ink height
  per line < 20 px: 987, 20–28: 283, 28–40: 66, 40+: 99 (staged-image pixels). 3,656 boxes
  still rejected as neither balloon nor flat.
- **Cache** `configs/hangul_v3_cache.toml`, plus 400 ko + 200 en twins of e2's ksplit cache
  (`src/link_hangul_twins.py --lang`): 1,400 samples.
- **Run** `configs/hangul_v3_ft_e2_s600.toml`: init from ksplit e2 (EMA file), e2's recipe
  (muon 5e-5, rank 32, mod_rank 16, region_weight 2, int8 plan), EMA window 0.1, 600 steps,
  24.7 min, peak 13.1 GB. `output/ckpt_moss_i2i/hangul_v3_ft_e2_s600[/…]`.
- **Eval** ko10 seed 0 (int8 generate default) and the two eval pages seeds 0–2 (bf16 + swap 10).

## Reads (per glyph)

Better than e2:

- 589883_0015, the long narration: no pseudo-glyphs, every line in place (e2: `잘못ᄐ`, the
  `마음에 들어서라는 이유로 계속` stretch garbled, the duplicated-caption box filled with
  `오랜 시간이' 일에 라을`).
- `정마알`, `처어` (589883_0010), `에잇` (737486_0012; e2 `에입`).

Worse than e2:

- 589883_0010: the `그러면, 실전 씻겨주기 시작하겠습니다` balloon is pseudo-glyphs (e2 had one
  inserted `만`); `커다란것`→`키다란것`, `귀여운데`→`귀어운데`.
- 589883_0002: one narration line garbled (`계실수몬 있보도는 난레나슨`; e2 clean).
- 737486_0011: two box-like marks drawn above `정말~~` in the balloon.
- 589883_0013: `괜찮은`→`펜찮은`, `이대로`→`이데로`.
- 589883_0015 (beside the gain): `문화`→`문홰`, `현재`→`헌제`, `먹히고`→`막히고`, `능숙`→`능속`; the
  strokes greyer than the edition's.
- 589883_0032: the dark narration box `재개`→`재게`; the light box centred where the edition
  is left-aligned.

Same as e2: the eval pages on all seeds (clean, `분인가요` right), 737486_0020 / 0026
(`미끈미끔`, `변배` in both), `용든` on 0032.

## What it means

- 600 steps on this mix move e2 about as much as they fix it. The errors that remain or
  appear are the same two kinds as before: a similar vowel in place of the right one, and
  pseudo-glyphs in long text. Neither tracks the faces or the narration share in a way these
  ten pages can show.
- The lettering on short balloons was clean in e2 already; the sheets do not show the
  fine-tune drawing crisper strokes there.

## Not tested

- The s200 / s400 saves; seeds 1–2 on ko10; a lower lr or fewer synthetic pages.
