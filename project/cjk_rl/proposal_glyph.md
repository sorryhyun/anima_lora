# Glyph-first OCR reward — after the sixteen rows

Status (2026-11-25): the stage's next steps, G-D's path, G0d and G1, are in
[`plan.md`](plan.md); what ran is in [`reports/`](reports/) (`glyph_g0_2026_11_25.md`,
`glyph_scenes_2026_11_25.md`). This file proposes what follows G1 and nothing before it. The
first draft's G-R arm (GRPO on the glyph reward) is out on the user's call of 11-25; that text
is in the git history.

The idea (user, 2026-11-25): the JA rows exist only to be lettered, so an ideal row should draw
its glyph more reliably than the model draws EN text. A caption quotes a glyph with no text tag
around it (`… Text reads as "あ".`), the render is decoded by the pixel head and read by the OCR
reader, and the reader's loss on the row's own glyph goes back to the row. The sentence arms of
[`proposal.md`](proposal.md) (A, D on captions) wait behind this stage.

## 1. Where G1 leaves the stage

G1 trains 16 rows and reads them on scenes and seeds no update saw. What it hands on:

- the steps a row needs to its first gain, which sets the budget below;
- whether a row that starts at zero (を, 人, 行) can be started at all;
- whether rows that were not trained stay where they were.

Evidence the later stages rest on, measured:

- `reports/glyph_scenes_2026_11_25.md`: on the same four scenes, bubble frame, a one-token EN
  word reads alone on 0.57 of the renders, a hiragana on 0.27, a frequent kanji on 0.05; 22 of
  the 32 kanji never.
- `reports/turbo_read_2026_11_25.md`: in the ruler's frame (`speech bubble, japanese text`) the
  rows read near their null on `m15v11`, and the 225 kanji only jp_v1 trained read as untrained
  pack rows do.
- The finished CJK lines saw 2 – 3 glyph prompts draw the first glyph only.

## 2. G2 — the kana, the frequent kanji, then words

- Rows: the 163 kana, then the 1 410 kanji in frequency order, in G1's loss and frame.
- Budget: rows × G1's steps to a gain × the step's seconds from G0d. Both are unknown today; if
  the product does not fit this box for the kanji, the kana and the first few hundred kanji run
  and the rest wait on a cheaper step.
- Words of 2 – 4 glyphs join once singles pass, every glyph a trained row, real words only (the
  standing rule: no random strings). The reader's cross-entropy scores each position of a word
  by construction, so a word needs no separate credit rule.
- Read: G1's grid on every trained row, and the words' `solo` and per-position hits against the
  start rows, on held-out scenes and seeds. Bars: spread unknown until G1 has run twice.

## 3. G3 — transfer and ship reads

Each against jp_v1's rows:

- the ruler's frame on `m15v11` (`turbo_read.py`'s ruler part): what the rows do once
  `speech bubble, japanese text` are back around a sentence;
- the base and merged15 at 28 steps (`proposal.md` P3's rule: rows that help the stack and hurt
  the base ship as a pack for that stack);
- the ruler's short bin and the words, for composition.

The user's read of the sheets decides each.

## 4. Risks

- Singles may not compose into words. G2's words are the test, G3 reads it on the ruler.
- A row trained to letter only itself in a tagless caption may letter less inside dialogue.
  G3's first read is for this.
- Rows fitted on the turbo stack may not carry to the base (G3).
- Reader bait: a gradient through a reader can paint pixels only the reader reads. Rows act
  through the text conditioning alone, which leaves little room; the page term and the sheets
  stand at every gate, and the read of record is the sampled read, not the loss.
- This stage cannot give order and count over a long string. Those stay with the sentence arms;
  the frozen model does not do them for EN either (`reports/turbo_read_2026_11_25.md`).
