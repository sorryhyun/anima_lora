# Glyph-first OCR reward — a row draws its own glyph, and nothing else

Status (2026-11-25): proposal, nothing run. It puts a glyph and word stage in front of
[`proposal.md`](proposal.md)'s sentence arms: those arms (A, D on captions) wait behind it.

The idea (user, 2026-11-25): the JA rows exist only to be lettered, so an ideal row should draw
its glyph more reliably than the model draws EN text. Train toward that directly: render a
caption that quotes a glyph with no text tag around it (`… Text reads as "あ".`, no
`japanese text` / `english text`), decode with the pixel head, read the render with the OCR
reader, score the row on its own glyph, and score down any lettering that was not asked for.

## 1. Evidence

Measured, with the report each number comes from.

- [`reports/turbo_read_2026_11_25.md`](reports/turbo_read_2026_11_25.md), merged15 + the turbo
  v1.1 delta (`m15v11`), 4 steps, the ruler at four seeds:
  - jp_v1's rows: hit 0.14 against null 0.10, cer 0.93; long strings 0.11 against 0.09. The EN
    reference on the same stack: hit 0.28 against 0.12, cer 0.83.
  - In the frame `speech bubble, japanese text. Japanese text reads as "<key>".` a single kanji
    is never read alone (0 of 384 renders); every group letters 32 – 37 glyphs a page whatever
    is asked. The 225 kanji only jp_v1 trained read as the untrained pack rows do (0 and 1 of
    128 on the page).
  - Substitution kinds follow the null's; look-alike kinds are at or under 1 %.
- [`reports/position_read_2026_11_25.md`](reports/position_read_2026_11_25.md), the base, the
  old floor's rows: at 3 – 4 tokens the JA string reads as the EN line does (hit 0.76 against
  0.73, the EN null the higher one); behind it from 5 tokens on.
- [`reports/p1_2026_11_13.md`](reports/p1_2026_11_13.md): GRPO on the rows over caption
  dialogue moved nothing. A row was in a median of 2 groups, the reward sat near its null, and
  the row gradient was near Adam's ε (log π averaged over latent dimensions).
- `../cjk_anima_reseed/reports/xattn_box_2026_10_15.md` and `status.md` § 4: the text tags
  letter the page on their own; dropping `japanese text` half the time in supervised training
  lost words.
- Cost, `turbo_read` on `m15v11` at 768²: sampling 0.30 s an image (4 steps), the pixel head
  0.007 s a decode (job `20261125-093350-0b3107`, 58 renders), the reads 26 crops a second with
  detection. P0 on merged14 + v3 with the VAE: 14 s a group of 8 (rollout 6, read 4, update 4).
- The pixel head's turbo point was trained on this stack (`anima_turbo_4step_v4_longer`, σ 0.5;
  `docs/anima/inference/pixel_head.md`).

## 2. Why this shape

- With `speech bubble` and `japanese text` in the caption the stack fills bubbles with its own
  dialogue, so "letter nothing that was not asked" has no parameter of a row to act on. In a
  caption with no text tag, every glyph on the page is there because of the quote. The penalty
  then lands on the quoted rows.
- One glyph a group gives one row all of the group's credit. P1's three problems (a row seen
  twice, a reward at its null, neighbours sharing the credit) do not arise.
- A single large glyph is what the reader reads best, so less of the within-group spread is
  reader error than on P0's vertical bubbles.
- What this stage cannot give: order and count over a long string. Those stay with the sentence
  arms; § 1's EN reads say the frozen model does not do them for EN either.

## 3. Design

### 3.1 Policy, frame, pool

- Policy: `m15v11` (the ruler's `MODELS` entry), 4 steps, cfg 1, shift 3, pixel-head decode.
  Frozen: the DiT, the turbo delta, the pixel head, the reader. Trains: the rows (`ExtDelta`),
  warm from `seed_1008` (jp_v1).
- Frame: `<scene caption>. Text reads as "<string>".` The scene caption is a training caption
  with its text clauses and its text tags taken out (`japanese text`, `english text`, the other
  language tags; parsed with `anime_tools.captions.position_clauses`, never split by hand).
  Whether `speech bubble` stays is read in G0, not assumed.
- Scene pool: captions of the training set, `src/pool.py`'s hold-outs (the ruler's images,
  `$CHARACTER_REF_EXCLUDE`), plus a held-out scene set that no update sees, for the reads.
  A scene whose render letters text with no clause at all leaves the pool (G0 counts them).
- Strings by stage: G1 one glyph; G2 real words of 2 – 4 glyphs (the standing rule: no random
  strings), every glyph a trained row.

### 3.2 The score

One instrument for training and for the read, built from what the line has
(`src/reward.py`, `eval.ruler.score`):

```
r = r_seq + β · g_p − γ · (1 − m_init) − δ · max(0, fw_over_init)
```

- `r_seq`: the asked string against its best box, LCS / max(asked, read). For one glyph it is
  1 when a box reads exactly that glyph.
- `g_p`: the asked glyphs among everything the page letters. This is the "nothing that was not
  asked" term: on a tagless caption it is 1 only when the page letters the string alone.
- `m_init`, `fw_over_init`: the page anchor and the flat-white (paste) guard, as in P1.
- β, γ, δ: set on each term's within-group std from G0, as P0b set them; not tuned on a read.

### 3.3 Two estimators

| arm | estimator | what it should teach | to build |
|---|---|---|---|
| **G-D** | the reader's teacher-forced cross-entropy of the asked glyph on its box, backpropagated through the pixel head and the DiT to the rows | the glyph's shape, per position, from one sample | a tap that keeps the graph (`PixelTap` detaches), the reader's tensor path with labels (`proposal.md` § 3.7), checkpointed backward over k steps |
| **G-R** | GRPO on `r`, `src/grpo.py` with `proposal.md` § 3.6's fixes (log π summed, G 16 – 24) | drawing the glyph at all, and only it: `g_p` and "no box" have no gradient in G-D | pixel-head decode and seed batching in the loop (as `probes/turbo_read.py`), the tagless pool |

- G-D comes first where the glyph is already drawn on some seeds; G-R covers what G-D has no
  crop for. `proposal.md` held G-D back because a page-sized VAE decode backward does not fit
  beside the DiT and the reader; the pixel head removes that.
- G-D's page term: MSE to the anchor render outside the asked box, in pixel space. The anchor is
  the same caption without the clause, same noise; on a tagless scene it holds no text.
- Per-position credit in G2: a word's LCS hits give each glyph its own advantage
  (`proposal.md` § 3.4). G-D's cross-entropy has it by construction.

### 3.4 Budget (estimates from § 1's costs, not measured as a loop)

- G-R, G = 16 at 768²: about 5 s sampling, 1 s reads, and an update of about 8 s if it scales
  as P0's did: about 14 s a group, 260 groups an hour.
- Rows: 163 kana and 1 410 kanji. At 20 groups a row the kana take about 12 h and every row
  about 120 h, so G-R alone does not cover the kanji on this box. How many groups a row needs
  is what the micro arm measures.
- G-D's step cost is unknown until it is built (G0d).

## 4. Phases and gates

Criteria are fixed here, before any result (`experiment-planning` skill, Gate). The read of
record is the glyph grid: each glyph on held-out scenes at 4 seeds, in the tagless frame, with
three numbers a glyph — a box reads exactly it (`solo`), it is on the page (`on_page`), glyphs
lettered beside it (`extra`) — and the null of `turbo_read` (the glyph on another glyph's page).
The EN arm of the same grid (one-token EN words in the same frame, rows off) is the level the
user's premise names. Sheets go to the user at every gate; the user's read decides.

**G0 — forward probes** (about 10 min, no update)

- G0a, the frame: 24 hiragana, the 32 frequent kanji, 24 one-token EN words; 8 seeds; the
  tagless frame with and without `speech bubble`; the no-clause render of every scene. Reads:
  `solo`, `on_page`, `extra`, the letters the no-clause render draws.
- G0b, spread: the share of glyphs whose 8 seeds do not all score alike (P0b's
  non-degenerate count), and each term's within-group std. A reward that is constant inside
  nearly every group stops G-R before it is built.
- G0c, the eyes: 64 renders decoded by the pixel head and by the VAE from the same latents,
  read by the same reader. Paired `r`; the pixel head qualifies when it does not shift (sign
  test p ≥ 0.05), as `reward_reader_2026_11_13.md` qualified the batched read.
- G0d, after G-D's path is built: the gradient's size on a row against the optimizer's floor,
  peak VRAM and seconds a step for a backward through the last step and through all four.

**G1 — micro, 16 rows** (8 hiragana, 8 frequent kanji, picked from G0a across the range of
`solo`)

- G-D: 50 steps a row. G-R: 40 groups a row (about 2.5 h on § 3.4's estimate), beside a null
  run of the same length with the advantage permuted inside each group.
- Pass, per arm: `solo` up and `extra` down on the 16 glyphs against the start rows, paired
  over glyphs, on held-out scenes and seeds; the permuted run moves neither; the user reads the
  sheets as lettering, not as a paste or reader bait.
- Bars: spread unknown until G0a's seeds give one.
- Also reported: the 16 rows' neighbours on the grid (rows not trained must not move), and
  groups-to-first-gain per row, which sets G2's budget.

**G2 — the kana and the frequent kanji, then words**, on the arm G1 leaves standing (both, if
each moves its own term). Words of 2 – 4 trained glyphs join once singles pass, with
per-position credit.

**G3 — transfer and ship reads**, each against jp_v1:

- the ruler's frame on `m15v11` (`turbo_read`'s ruler part): what the rows do once
  `speech bubble, japanese text` are back;
- the base and merged15 at 28 steps (`proposal.md` P3's rule: rows that help the stack and
  hurt the base ship as a pack for that stack);
- words and the ruler's short bin, for composition.

## 5. Risks

- The clause may letter nothing without a text tag on this stack, or the reward may be
  constant inside a group. G0a and G0b read both before anything is built.
- Singles may not compose: the finished lines saw 2 – 3 glyph prompts draw the first glyph only.
  G2's words are the test, and G3 reads it on the ruler.
- Layout is set in the early steps. A gradient through the last step alone may reach the
  strokes and not whether a glyph is drawn; G0d measures the four-step backward, and G-R is the
  arm for that part.
- Reader bait. A gradient through a reader can paint pixels only the reader reads; rows act
  through the text conditioning alone, which leaves little room. The flat-white guard, the
  page term and the sheets stand at every gate.
- A row trained to letter only itself in a tagless caption may letter less inside dialogue.
  G3's first read is for this.
- Rows fitted on the turbo stack may not carry to the base (G3).
- The pixel head was trained at the Euler rollout's last step; G-R samples with the SDE. G0c
  reads the SDE renders too.

## 6. What is built and what is not

- Built: `probes/turbo_read.py` (the `m15v11` render path with the pixel head and seed batches,
  the reads, the tables and their null), `probes/position_read.py` (alignment), `src/grpo.py`,
  `src/reward.py`, `src/pool.py`.
- Not built: the tagless scene pool and the glyph grid, G-D's differentiable path, the GRPO
  loop on the pixel head with § 3.6's fixes.
