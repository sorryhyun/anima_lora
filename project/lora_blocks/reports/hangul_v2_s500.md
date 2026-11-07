# Step 3, v2 data: the Hangul block, 500 steps from C

Date: 2026-11-08. Follow-up to [hangul_s500.md](hangul_s500.md). Sheets (`output/tests/lora_blocks/`): `hangul_v2_s500/zoom_<page>_s<seed>.png`
(rows edition, ksplit e2, v1 block, v2 block), `hangul_v2_s500/page_<page>_s<seed>.png` (full page:
source, e2, v1, v2), `hangul_v2_s500/grad_in.json`.

Result in one line: **with the data laid out as the edition letters it, the block alone letters
both Korean eval pages right on every seed, in a bold, clean gothic laid out as e2's, and leaves
the uncaptioned Japanese as it is. All three of v1's failures are gone. On these two pages e2
was already crisp, so the pages show the block reaching e2's quality and not passing it.**

## What changed from v1 (`src/build_hangul_v2.py`)

- **Balloons, not boxes.** A box is used when its closed white balloon is found: the region
  reached from the light pixels around the box, with holes filled, is 1.2x the box to 6 % of
  the page. The balloon is filled with its own ground colour, which removes the Japanese and
  Korean lettering alike. Text on artwork, open balloons and balloons holding more than one
  OCR line are left alone.
- **Base page**: the Japanese raw (299) or the Korean edition (301) of a swap_text v3 twin.
  The twins are aligned, so the edition's boxes locate the same balloons on both.
- **Layout**: line count, glyph height and pitch are measured from the edition's ink in the
  box. The string is wrapped into that many lines, balanced at word breaks, and centred on
  the edition's block.
- **Faces**: one per page, bold dialogue faces only (Pretendard Bold / Black, Gangwon Edu
  All Bold, KMU HaeongSans, Do Hyeon, Jua).
- **Partial captions**: each usable balloon is re-lettered with p 0.7. The rest keep their
  text on both sides of the pair and stay out of the caption.
- **Text**: the edition's own line (427) or another OCR line within ±25 % of its length
  (611): 1,038 lines, 681 distinct syllables. Condition `blank` 285 / `orig` 315.
- **Rejected boxes**: 3,489 not a closed balloon, 704 sharing a balloon, 666 failed the ink
  measurement, 77 spilled out of the balloon. A single-core build takes ~6 min.
- **Run**: as v1 (`configs/hangul_v2_cache.toml`, `configs/hangul_v2_s500.toml`), plus the
  same 150 ko twins: 750 samples, 19.6 min, peak 13.1 GB.

## Reads

`grad_in` (render / edition, inside the caption's boxes; on `737486` the edition is a soft scan):

| arm | 589883 s0 / s1 / s2 | 737486 s0 / s1 / s2 |
|---|---|---|
| ksplit e2 | 0.89 / 0.91 / 0.91 | 1.30 / 1.30 / 1.32 |
| v1 block | 0.78 / 0.87 / 0.79 | 0.93 / 0.81 / 0.90 |
| v2 block | 1.31 / 1.22 / 1.22 | 1.35 / 1.26 / 1.43 |

Per glyph, from the sheets:

- **Syllables**: every captioned syllable is right on seeds 0–2, `분인가요` included.
- **Strokes**: bold, even, clean-edged on every line. Heavier than e2's lettering (most of
  the `grad_in` gain) and than the editions'.
- **Layout**: the edition's and e2's line structure (`어머? / 처음 오시는 / 분인가요?`,
  `괭장해… / 질 속이 / 정액으로 / 가득해…`). No v1 one-word lines, size jumps, letter
  spacing or magenta.
- **Spacing**: two slips.
  - On `589883` s0 the line break falls inside a word (`저기… 이 / 건 뭔가요?`).
  - On `737486` s0 / s2 the space in `질 속이` is lost (`질속이`).

  e2 has neither.
- **Uncaptioned text stays**: the Japanese balloons the caption leaves out are kept as they
  were on both pages (`愛宕さんとセックス… もっと…`, `きー気になる…`), where v1 wrote
  pseudo-glyphs.

## What it means

- The v1 failures came from the data, not from too few steps. The same 500 steps on data
  laid out as the edition letters it give a working Korean letterer from C, with no
  swap_text pairs beyond 150 ko twins.
- These two pages cannot show "crisper than e2": e2 letters them cleanly. The soft strokes
  the plan targets show on other pages (colorize v5's `to_colorize4`, swap_text pages with
  small text). Step 0's per-glyph read is what finds them.
- The weight comes from the faces. It sits above the editions' and is a choice to revisit
  once there is a page where e2 is soft.

## Not tested

- The block on `to_colorize4` (as colorize's lettering, or stacked on colorize v5).
- The English eval pages (the block saw no English).
- v2 stacked on e2; the s100–s400 saves; a run from kaiming.
