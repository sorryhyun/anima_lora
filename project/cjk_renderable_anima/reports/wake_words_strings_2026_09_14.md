# Word addresses, order probe, σ diagnostic, strings arm (2026-09-14 pm)

> Lifted out of the former `history.md` on 2026-09-15; text unchanged. Run 3
> (a row can be a word; a sequence renders one row), the base-model order probe
> (sequence reading is pretrained), the `classify_str` σ diagnostic (order at
> σ 0.5–0.8), and the strings arm (a static table carries order + count).
> Index: [`README.md`](README.md). Backticked paths (`probes/…`, `plan_synth.md`, `output/…`) are relative to the line root `project/cjk_renderable_anima/` or the repo root, as they were in `history.md`.

## Run 3 — word addresses (2026-09-14 pm): a row can be a word; a sequence is still one row

**Why it was owed.** Every combo / corpus eval so far scored strings whose
pieces were *not all trained rows* (corpus coverage 19/28), so "one glyph
per string" had never been measured on a sequence of trained addresses,
and every combo was a random kana salad. Two hypotheses, one run: real
words as addresses, and trained-address sequences.

**What a "word address" is.** The pack's ext rows are Qwen pieces and Qwen
already merges many JA words into one piece (ありがとう / 行く / 明日 / いい /
って → one row each; 大丈夫 → 大+丈夫, ドキドキ → ド+キ+ド+キ). A word address
is therefore an *existing* row put into the trained inventory with the
whole-word render as `g`'s input — no pack change, no cache digest change.
The corpus bubbles (`render/ja/resized/boxes.jsonl`, 3 607 lines) hold
1 417 distinct multi-char single-piece words (10 503 occurrences; the top
60 cover ≈ 44 %); only 17 lines are a single piece, typical short lines are
2–5 pieces.

**Data `wd`** (`--words 120 --held_out_words 8 --n_single 40`): kana 92 +
the 120 most frequent single-piece words (mostly grammar: って いい から
ちゃ して じゃ ない …; content words 気持ち 好き ちょっと), 8 held out by seed
(えて かい こと これ だけ なら にも ません); 40 renders per row (8 860 font
items) + the 132 / 641 corpus lines whose every piece is a trained row
(coverage 100 % by construction). Bubble renders now shrink to fit the
ellipse (可愛い overflowed at 400/n px; singles unchanged). Eval adds
`word` (16 trained words), `word_held` (8), `line` (16 held-out corpus lines
of 2–3 pieces, every piece trained: いやいい = いや+いい, なにそれ = なに+それ,
ほらそれ, さわって, フフフ …). Recipe = fres, 8 000 steps, `g` and the 134
overlapping `f` rows warm-started from Run 1d (`--init_free`), word `f`
from zero. Job `20260914-111212-d1542c`, 58 min + eval 15 min; arm
`encoder_wd_w120_s8k_fres_warm`.

| group | n | exact (sfx) | note |
|---|---|---|---|
| single (18 kana × 2) | 36 | **34/36** | all 92 kana trained — the kana pack exists; misses と→こし, れ→ね |
| word (trained) | 32 | **9/32** | します 2/2, してる 2/2, きた, こう, いや, もう, ッド; misses: first glyph only (こう→こ, ただ→た, した→し, には→に), repeats (でも→てもも, もう→もうう), one-glyph collapse (こんな/そんな→ん) |
| word_held | 16 | 0/16 | `g` alone on a word render draws text salad, not even a kana |
| **line** (2–3 trained pieces) | 32 | **0/32** | every render is *one* piece: そオニ→ニ, ほらそれ→ら, さわって→わ, なにそれ→な, いやいい→いい, ちゃく→く, めちゃ→め, フフフ→フ |
| combo / corpus | 36 / 20 | 1/36 / 0/20 | as before |
| EN | 24 | 24/24 | bit-exact |

Instruments: `rel` 2.39 → 1.45, `rel_spread_ref` 2.2 → 0.78, `free_norm`
0.23 → 0.70, `free_ratio` 0.90 (Run 1d: 0.17), `table_pr` 1.1 → 2.8, `g_pr`
1.09. With 246 rows the identity moved almost entirely into `f`; `g`
shrank to a prior.

**Reading.**

- *A row is a unit, and the unit can be a word.* します and してる (three
  glyphs) render from one ext row on both seeds; きた こう いや もう ッド on
  one. So the frozen DiT's "one glyph per address" is really "one *unit*
  per address" — an address can carry a short word's layout. That is new,
  and it is a per-row exposure question (words had 40 renders + a few
  corpus crops and `f` from zero; kana had a warm start). 9/32 misses the
  67 % gate; the miss modes (first glyph only, repeats) are the
  under-exposed row rounding to the nearest kana basin or the repeat mode
  Run 2 showed.
- *A sequence of trained addresses renders one of them.* `line` 0/32 with
  coverage 39/39 is the clean version of the string verdict: the DiT does
  not enumerate ext tokens in a clause, it draws one. Which one is not
  positional (last: そオニ→ニ; second: ほらそれ→ら, さわって→わ, いやいい→いい;
  first: なにそれ→な, めちゃ→め) and not obviously frequency. Random combos
  were not the reason; real words in real order fail the same way. **The
  sequence question is closed on the rows path: strings are W3.**
- *Held-out words are worse than held-out kana.* `g` on a word render
  produces salad, not a kana — the shared encoder has no word prior. Word
  addresses are lookup, one row per word, exposure per row.

**What this adds to the pack.** (1) The full kana inventory renders
(34/36) with words in the same table — the 92-kana pack is this run's
`trained.pt`. (2) A common-word pack is buildable the same way, at a cost
that is per-word exposure; a words-only continuation warm-started from
this run (`f` now non-zero for the 112 words) is the one cheap lever to
see whether words reach the kana bar. (3) Nothing here moves strings; a
2-piece bubble needs the DiT to read two addresses, which is W3's
ext-gated cross-attention LoRA.

**Decision owed.** Whether to spend ~1 h on the words-only continuation
(product value: single-word bubbles from a common-word pack) before W3,
or go to W3 on the kana+word table as is.

## Order probe (2026-09-14 pm): the frozen adapter + DiT read piece sequences in order

**Why it was owed.** Every "sequence" verdict above was drawn from ext
rows, while the EN control was read as a single-word sanity check. The
T5 tokenizer says otherwise: HELLO = ▁H·ELL·O, SORRY = ▁S·OR·RY, HELP =
▁·HE·LP — the EN 24/24 was already a multi-piece result. Real words could
still be recognised as wholes by the adapter's self-attn, so the
discriminating probe is *nonsense* multi-piece words, base model, no
delta: `probes/order_probe.py` (job `20260914-131417-18883e`, 42 renders
in 2.5 min; `output/wake_probe/order_probe/`).

| group | n | exact (sfx / vl) | misses |
|---|---|---|---|
| real (HELLO STOP SORRY HELP PLOVEN OSTREB) | 12 | 11 / 11 | OSTREB s0 (a stray "las" before it) |
| nonsense, 4–5 pieces (GLORPAX MIZUKANE SUMIMASEN TOBRINEK VASQUOLM PILDROME KANTOBRE ZEMURIAL DRAVOKIN NURPELTA OKTABRIS FELMUNDO) | 24 | 21 / 22 | KANTOBRE→KANTOBBFF, OKTABRIS→OKTAABIRS (pieces present, tail scrambled) |
| two words (ZORP KAV, BLIM TOK, WAY NO) | 6 | 6 / 6 | — (WAY NO renders in the *given* order, not as NO WAY) |

**Reading.** A sequence of four or five T5 addresses the DiT has never seen
as a word renders as that sequence, in order, at 512² with the same
template, cfg and steps as the wake evals. So "the DiT does not enumerate
tokens in a clause" and "token → position binding was never learned"
(W2 § what W2 settled, W3) are wrong as general statements: enumeration
and order are a pretrained capability of the frozen adapter (self-attn +
cross-attn, both RoPE'd) and the frozen DiT. What fails is *ext-row*
sequences, and the ext rows differ from EN pieces in exactly two ways —
they are off-manifold random directions the adapter was never trained to
contextualise, and every arm trained them on single-unit canvases
(Run 3: ≈ 1.5 % multi-piece items), so the delta carries "one centred
unit" and nothing ever asked it to be contextualisable. The string
question therefore reopens on the rows path, cheaper than W3:

- **Strings arm** (next, ~1 h): warm-start the Run 3 table (identity is in
  `f`), train on 2–4-piece random-order strings only — no singles — and
  score `line` plus order-flipped pairs (なに vs にな). Non-zero `line`
  means strings are reachable as a static table; still 0 means the frozen
  adapter cannot contextualise off-manifold rows, and *that* is the case
  for W3 (or for slot rows: Δ(piece, i) = Δ_piece + P_i, tokenizer-side,
  still a pack).
- `order_probe.py` is the standing EN control for order; every future arm
  should carry a nonsense multi-piece group so order reading is measured
  for free.

## σ diagnostic for strings (2026-09-14 pm): order is read at σ 0.5–0.8, nothing above 0.9

Two `--stage classify_str` runs (same-noise diffusion classifier, the true
caption S vs named alternatives, both conds; win = S has the lower error,
gap in spread units; jobs `-19743b` JA, `-4a8071` EN; reports under
`encoder_wd_w120_s8k_fres_warm/classify_str{,_en}/`).

**JA — 48 2-kana + 16 3-kana strings of Run 3's rows** (pairs chosen so
every permutation tokenizes to its own kana rows, never a word piece):

| contrast | σ 0.35 | 0.50 | 0.65 | **0.80** | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| identity (S < substituted) | 0.34 | 0.59 (+0.5) | 0.56 (+0.9) | **0.77 (+1.3)** | 0.48 (+0.5) | 0.34 | 0.33 |
| count (S < one piece) | 0.50 | 0.41 | 0.48 (+0.8) | **0.72 (+1.1)** | 0.58 (+0.8) | 0.42 | 0.19 (−1.3) |
| order (S < reversed) | 0.44 | 0.58 | 0.58 | 0.64 (+0.1) | 0.55 | 0.61 | 0.47 |

Floor (delta off) is chance or worse everywhere. Identity and count in
the trained rows live at σ 0.65–0.9 (the singles band). **Order is chance
at every σ** — the rows carry no order signal, so they cannot say where
the DiT would read it.

**EN — 24 nonsense two-word strings ("LIZOFUV RIZO"; a word swap keeps the
piece multiset), base model** (no ext id, so trained ≡ floor):

| contrast | σ 0.35 | 0.50 | **0.65** | 0.80 | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| order (S < swapped) | 0.69 (+0.4) | 0.94 (+1.9) | **0.98 (+2.5)** | 0.90 (+1.1) | 0.54 | 0.44 | 0.31 |
| count (S < one word) | 0.73 (+1.3) | 0.96 (+1.5) | **0.98 (+2.1)** | 0.90 (+2.0) | 0.69 (+1.1) | 0.25 (−0.5) | 0.06 (−1.2) |
| identity (S < substituted) | 0.73 (+1.4) | 0.98 (+1.8) | 0.98 (+1.8) | 0.83 (+0.8) | 0.38 | 0.23 | 0.27 |

**Reading.** In the base, order / count / identity of a piece sequence are
all decided at **σ 0.5–0.8** (peak 0.65) and there is no signal at σ ≥ 0.9.
The "one-piece caption wins at σ 0.95–0.99" pattern shows up for EN too, so
it is a near-pure-noise artefact (fewer tokens = lower error), not a
collapse decision at high σ. Consequences for the strings arm: the band
moves **down**, not up — `--t_min 0.5 --t_max 0.9` covers the order window
and the kana identity peak (0.8); widening above 0.9 buys nothing. The
sampler is a hard affine remap (sigmoid density inside, zero outside), so
the band *is* the weighting.

## Strings arm (2026-09-14 pm, running): `encoder_ws_w120_s8k_strings_warm`

Data `ws` (`--strings_only --n_strings 6000 --words 120 --held_out_words 8`):
no singles at all — 6 000 random-order 2–4-piece strings of trained rows
(kana, trained words at 25 % per slot; each string checked to tokenize to
exactly its intended pieces) + the 132 fully-covered corpus lines. Eval
adds `flip` (12 clean kana pairs × both orders) and `str3` (8 clean 3-kana
strings) to the Run 3 groups. Recipe = Run 3 with `g` frozen (`--lr_enc 0`),
`f` and `c` warm-started from Run 3, band `--t_min 0.5 --t_max 0.9`, 8 000
steps. Job `20260914-134255-fe1767`.

Read: `line` / `flip` / `str3` > 0 → strings are reachable as a static
table (then: does `flip` distinguish orders, or draw the same string for
both?). All 0 with singles intact → the frozen adapter cannot
contextualise these rows → W3 or slot rows. Singles collapsing → the rows
lost identity under string-only exposure (then mix singles back at a
small fraction before any verdict).

### Strings arm result (2026-09-14 evening): sequences render, order is read, singles inherit the multi-unit prior

67 min train (2.31 it/s) + 15 min eval; `output/wake_probe/encoder_ws_w120_s8k_strings_warm/`.
Delta stayed on-manifold (rel 1.42, max 2.2, Run 3 band); the warm-started
residual dipped 0.78 → 0.52 in the first 25 steps and re-settled at 0.71–0.78.

| group | n | CER sfx / vl | exact | Run 3 |
|---|---|---|---|---|
| single | 36 | 0.861 / 0.917 | **5/36** | 34/36 |
| word | 32 | 0.820 / 0.831 | 3/32 | 9/32 |
| line (held-out corpus, all rows trained) | 32 | 0.747 / 0.740 | **2/32** (ちゃく, フフフ) | 0/32 |
| flip (12 clean kana pairs × both orders) | 48 | 0.719 / 0.719 | **4/48** (たン, タひ, こユ, ユこ) | — |
| str3 (clean 3-kana, unseen) | 16 | 0.625 / 0.583 | **3/16** (んソヒ, にワル, ひくル) | — |
| combo / corpus | 36 / 20 | 0.667 / 0.667 | 1/36 / 2/20 | 1/36 / 0/20 |
| en | 24 | 0.000 | 24/24 | 24/24 |

- **Sequences no longer collapse to one unit.** Nearly every string prompt
  draws 2–4 glyphs; `flip` has both pieces present in 25/48. Every string
  group moved off zero, including *unseen* strings of trained rows (`str3`
  3/16, `flip` 4/48) — composition generalises across strings, no
  per-string exposure.
- **Order is read from the static table.** In `flip`, the first rendered
  glyph equals the caption's *first* piece 28/48 and its *last* piece 5/48
  (chance would be equal); the same two rows swapped change the picture
  (くル→くくル vs ルく→ルくル, うネ→ううう vs ネう→ネうう, こユ / ユこ both
  exact). The frozen adapter does contextualise these rows once the loss
  asks for it. The "cannot contextualise off-manifold rows" branch is
  dead; W3 is not needed for this.
- **The dominant failure is repetition, not identity**: ううう, ねねね,
  くくル, ををら, はめめ — the count is right, the second slot copies a
  neighbour. This is Run 2's repeat mode.
- **Singles inherited the multi-unit prior** (34 → 5/36): a one-kana
  caption now draws the glyph *plus* extras (ひ→ひひ, む→むむろ, ケ→ケケケ,
  テ→ズテテ, チ→チテチ). The identity is still there; the rows now carry
  "several units" the way they used to carry "one centred unit". Unit
  count is a data-distribution prior baked into the rows, symmetric with
  the singles-era artefact. Held-out words 0/16 as before.

**Reading.** Rows can carry order and count, and the DiT reads them, but
the rows also absorb whatever count the training distribution had. The
fix is not two hard steps but one distribution: singles and 2–4-piece
strings mixed so the number of units is only predictable from the number
of ext tokens in the caption. Next arm: same recipe from Run 3's table,
data = 30 % singles + 70 % strings (`--strings_only` gains a
`--single_frac`), band 0.5–0.9. Gate: singles back ≥ 30/36 with `flip` /
`str3` / `line` at or above this arm. Then the repeat mode is the
remaining lever (harder negatives: strings with a repeated piece, so a
repeat is only right when the caption says so).
