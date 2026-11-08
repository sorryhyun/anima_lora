# OCR reader: keep line breaks

Status: proposal (2026-11-08), parked behind the swap_text main run.

Goal: the manga reader (PaddleOCR-VL v4, `anime_tools.ocr.sfx`) returns a text block's line
breaks, and the swap_text captions carry them into each layout-JSON item as `\n`. Then the DiT
is told where a balloon's lines break instead of choosing the breaks itself.

## Where the breaks are lost

- **Training labels:** `project/finished/ocr_reader/ocr/crop_dataset.py::normalize_target`
  (`textnorm.training_target`) collapses every whitespace run, newlines included, to one ASCII
  space. 3.80 % of the COO / Manga109 targets carried a newline or U+3000. v4 was trained to
  answer one line.
- **Reader output:** `anime_tools/ocr/sfx.py` normalises the decode with `" ".join(text.split())`
  ("a column boundary stays a space").
- **Detector:** AnimeText boxes text blocks (a balloon, a narration box), not lines, so one
  caption item is a whole block.

Measured on the swap_text trees: 0 of 177,677 stored reads in `swap_text_relax/ocr.json` hold a
newline. One item spans a median of 3.2 visual lines in Korean and 4.1 in English (estimated
from box size per glyph). 86–89 % of items span two lines or more.

## Why it is an improvement

- **Korean placement:** the user finds Korean line breaking and placement weaker than English or
  Chinese, though both languages get the same break-free captions. Given breaks, the DiT would
  only have to follow them.
- **Japanese and Chinese text content:** a column or line break inside a word is not a space.
  Collapsing it to one writes a space into the middle of a word. Korean breaks fall at word
  spaces (eojeol), so for Korean only the layout is lost.
- **Inference:** a caption helper or the user can insert breaks with a plain wrapper, or by hand.

## Approaches, cheapest first

1. **Gate, no training:** in a few `inputs_ko10/` captions, insert the Korean edition's
   breaks by hand as `\n` and render with ksplit e2. If the DiT follows them, the rest is worth
   doing. If it ignores them, captions with `\n` must be trained in as well.
2. **Geometry, no reader change:** the Korean and English targets are horizontal. A row
   projection of the ink inside each block gives the line count and line widths. Split the read
   string at spaces to fit those widths. CPU only, over the existing `ocr.json`.
3. **Reader:**
   - Keep newlines in `normalize_target` and in the `sfx.py` record. Comparison is already
     whitespace-blind (`exact_key`).
   - Rebuild the line-break labels: COO already has them on 3.8 % of rows; the rest get them
     from geometry, or from the source annotations where they exist.
   - Fine-tune v5.
   - Bump the `anime_tools` tag and the pin.

## Downstream changes

- `i2i_tasks/swap_text/caption.py` passes `\n` into the item text (today it only folds quotes).
- The colorize and t2i caption paths that also read OCR lines.
- The ComfyUI MOSS Toolkit caption helper.
- Check how Qwen's tokenizer and the `token_split` wrapper treat `\n` inside a layout-JSON
  string.
- Re-cache the TE after the captions change.
