# Proofread OCR labels on text crops (agent prompt)

You are proofreading first-pass OCR labels. Each row is one text crop from a manga page, an
anime-style illustration or a doujin page. The first-pass read comes from an OCR reader and is
often wrong where its confidence is low: the wrong script (a Japanese read of Korean lettering or
the reverse), a misread SFX, text read where there is no text, a mangled Latin word.

Input: `batch.tsv` in your folder (the path is in your task): `id`, `crop` (the PNG the label is
for), `page` (the full image), `box` (x0,y0,x1,y1 of the crop on the page), the reader's read
(`<reader>_read`, e.g. `v7_read`; `\n` = line break) and its confidence (`<reader>_score`).

## How to look

Crops are small (often 20-60 px). For every row make enlarged views before deciding, with
`/home/sorryhyun/anima/anima_lora/.venv/bin/python` and PIL, written into YOUR folder only (never
next to the inputs):

1. the crop upscaled 4-6x (LANCZOS);
2. a context view: the page region around the box, padded ~1.5x the box size on each side,
   upscaled so the box is ~300 px, the box outlined.

Read both with the Read tool. The context view is for understanding only (is it a heart, a
signature, part of the drawing?); the label is what the crop itself shows.

## What the label holds

- `main`: the text with the largest glyphs in the crop, exactly as lettered. Korean stays Korean,
  Japanese stays Japanese, Latin stays Latin (keep its case). Hearts: `♥` filled, `♡` outlined.
  Long-vowel and wave marks as drawn (`ー`, `~`, `〜`).
- Line breaks (`\n`) follow the printed lines: one per column of vertical text (right to left),
  one per row of horizontal text.
- Several items of the same kind with glyphs at least 0.7x the largest are one main, one per
  line, right to left then top to bottom.
- `sub`: every other, smaller text visible in the crop, one item per line in reading order; empty
  if none.
- Text cut by the crop edge is typed as far as it shows in the crop; what only the page shows is
  not typed.
- A crop with no readable lettering (drawing, screentone, a stroke you cannot tell from art) gets
  an EMPTY main. If you cannot tell whether a mark is a letter or drawing, do not type it.
- Hand-lettered Korean SFX is the hard case. Read it jamo by jamo before writing a syllable: a
  small tick or dot above ㅈ makes it ㅊ; count the short bars on the vowel (ㅏ / ㅑ, ㅓ / ㅕ,
  ㅗ / ㅛ, ㅜ / ㅠ); ㄹ over ㅜ is 루, ㄹ over ㅡ is 르 (look for the downward stroke). Repeated
  syllables often shrink as they trail off: count them. Still unsure of a syllable: conf `low`,
  not the commonest word.
- A heart (or `~`, `ー`) drawn as part of the lettering (beside or after the word, same colour
  and stroke) belongs in the main, in reading position (`♡좋아` when the heart comes first), not
  in the sub.
- Signatures, handles and watermarks are text: type them as written (`@name123`).
- Do not guess illegible words. If part of the lettering is genuinely unreadable, give an empty
  main and say why in the note.

## Output

Write `out.tsv` in your folder: columns `id`, `main`, `sub`, `conf`, `note`. `conf` is your own
certainty: `high` means you would bet on every character, then `mid`, `low`. `note` is short:
what the crop shows, why you changed the read. `\n` for a line break inside a field, no tabs
inside fields; one row per input row, in input order. Reply with the path of `out.tsv` and a
2-line summary.
