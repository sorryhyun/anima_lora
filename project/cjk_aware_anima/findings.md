# CJK-aware Anima — findings

Every settled verdict of the line in one place, with the evidence pointer.
Read this before proposing anything: most of the obvious levers are measured
and closed. Companion files: [`deliverables.md`](deliverables.md) (what exists
and where) · [`plan.md`](plan.md) (what remains). The dated reports carry the
full tables: [`reports/0816_phase2.md`](reports/0816_phase2.md) (Phase 2b gates +
2c first pass + corpus work), [`reports/0827_names_synth.md`](reports/0827_names_synth.md)
(name register, §8 JA-context, §9 attn term),
[`reports/0830_adapter_lora.md`](reports/0830_adapter_lora.md) (plan3 adapter
LoRA, closed). Dataset-side numbers: [`datasets/README.md`](datasets/README.md).

## 1. The problem and the opening (Phase 0 probe, 2026-08-15)

Anima conditions on two streams: Qwen3 hidden states (content) and a T5-id
query table that cross-attends into them through the 6-block LLM Adapter. Qwen
reads JA natively; the T5 side is a stock EN SentencePiece that collapses
native JA to `▁ <unk> </s>` — conditioning dies with it (cos ~0.02 vs the
all-EN reference, discrimination 0.91 = no prompt-specific signal). Feeding the
T5 side an **EN translation** of the same JA prompt recovers cos 0.69–0.76 with
healthy discrimination (~0.11). So the T5 stream needs to be *semantic*, not
Japanese, and the broken piece is small, isolated, and has a working reference
(`t5en`) to imitate. `bench/cjk_adapter/results/20260815-1836/`.

**Ruled out by the probe** (do not reopen): romanization (`t5rom` cos ~0.06 —
non-unk is not enough); reverse routing (EN on Qwen, JA on T5: cos ~0.07/0.15,
disc 0.92 — Qwen is the content channel); cleverer zero-shot maps (a 0.75
anchor-holdout cos still gave ~0.05 end-to-end — the gap is contextual, not
map quality, which is what makes training necessary); a fresh SentencePiece
over T5 (borrowing Qwen tokenization keeps source/query streams token-aligned
over CJK spans); mT5 / any vocab swap (ext rows are already Qwen BPE pieces;
rare names fall to char pieces in every tokenizer).

## 2. The design that holds (Phase 2b, settled 2026-08-16)

Ext vocab = Qwen's CJK tokens mapped into the T5 embedding space (anchor
init), appended as new rows; distill the new rows so the student
`adapter(qwen(ja), t5ext(ja))` matches the teacher `adapter(qwen(ja), t5(en_mt))`.
Shared Qwen side isolates the broken piece; trainable surface = new rows only,
so **EN prompts are bit-identical by construction** (G1, unit-tested).

Settled by gates G0b/G0/G1/G2/G3/G4 and confirmed at 2c scale:

- `param=global` — a shared low-rank + per-dim diagonal + scalar-gain
  correction over the ext rows; `global_row`'s 1,892 free rows buy nothing
  end-to-end.
- `loss=span` — segment-mean cosine per aligned span. Flat cosine is a
  **control, not a gate**: G5's oracle argmin of the span objective scores 0.13
  on the old `cos_vs_en ≥ 0.6` bar, and both flat probes showed buying flat
  points costs readout-space alignment (`flat`-trained near-disc 0.914).
- The readout that measures anything is the **real-query attention readout**
  (`build_query_bank.py`, DiT image-token queries at 2–3 σ, centered by
  `fit_centers`); random query directions are refused. `recovery_attn` is a
  **mix statistic** — the readout floor is register-dependent by 100× (G3).
- Teacher ceiling for `tags` is 0.823 readout recovery; the design reaches
  ~87% of the addressable signal on the corpus holdout.
- Trust hedging is a non-lever (G4b): dropping `mt_unverified` spans is
  *worse* on every column at 10⁴ pairs — noisy supervision beats none.
- Cost model: the surface saturates in ~20 GPU-min; the binding constraint is
  span-carrying ext rows, never compute.
- Two contracts: max-pad 512 with no `crossattn_seqlens` masking on both arms;
  GPU work through the daemon.

## 3. What the render grid taught (Phase 2c, 2026-08-16 → 08-17)

The 20-prompt same-seed grid (`en / ja_t5en / ja_ext`) splits exactly along
**supervision density**: the teacher is at EN parity everywhere; the student
transfers high-visit tag content (t1 school, t2 maid) and collapses on
thin/zero-visit content (t3 armor `鎧`:0, names `博`:2, prose function words
0 by construction). `gates/coverage.py` is the diagnostic; identity-carrying
tokens want O(100+) visits (`教室`:39 renders a classroom, `霊夢`:37 does not
render Reimu).

Corpus levers, each measured once and closed:

| lever | result |
|---|---|
| **D1-wide** (3,008 → 16,128 captions, 45,230 pairs) | buys **visits, not vocabulary** — 500+ band 381 → 756, rows visited flat at ~6,400, no `v=0` token moves. More captions multiply the same glossary. |
| **D1-pairs tail fill** (`danbooru-ja-tag-pair`, CC0) | buys vocabulary: 5,248 tags filled, unmapped segments 42,530 → 13,714. |
| **D1-pairs item 2** (community names as arbiter candidates + widened `--mt` rebuild) | 4,438 wordings moved, 0 pinned regressions, unmapped → 878 (−94%); grid moves on the coverage-bound prompts. The **polysemy class** (`bow`→蝶結び back-translates to the sense, not the string) is unwinnable by F1 → human review. |
| **D1-words** (katakana loanword chosen over native kanji: `armor`→アーマー not 鎧, 119 tags) | half are Chinese and correctly rejected (`bed`→床 = *floor*) → a human review axis, not automatic. |
| **D2** commentary (73k native JA, 9,068 paired) | span-less → **inert under `loss=span`**; +13% only in its own register under `attn`. A register/promo-filter problem, not access. |
| **manga_text** OCR corpus | rejected as distill material (duplicates D4, OCR noise arrives MT-laundered); kept for the glyph line's geometry. |

Invariants the glossary work established: **verified ≠ Japanese** (棕毛 /
藍眼睛 back-translate perfectly and are Chinese — keep both script filters and
kana-first ranking); **selection is pure post-processing** (`--reselect`, ~1 s);
**a glossary rebuild requires `--mt`** (the CPU-only path drops the
back-translation layer and regressed 1,991 wordings — tried and reverted);
leaving danbooru tags latin is strictly worse (routes to original spiece rows,
trains *no* ext rows); Wikidata covers 0/89 artists (handles are not entities).

**2026-08-31 corpus rebuild** ([`reports/0831_axis_joiner_rebuild.md`](reports/0831_axis_joiner_rebuild.md)):
two corpus-wide bugs found by reading `spotcheck.md` — (a) characters outside
`image_dataset` fell to the `general` axis and were **MT-rendered as words**
(`ame (mignon)` → 雨（可愛い）; 6,149/14,959 `names` pairs left the name EN);
(b) the `、` joiner was itself an ext row on every pair. Fixed (wiki-category
/ artist-OC axis fallback; `, ` 80 % / `、` 20 % joiner recorded per pair),
zero general-axis wordings changed, corpus + cache rebuilt, `synthja_v2`
retrained: readout 0.53 → 0.67, tags-register attn recovery 0.54 → 1.13,
grids gain on t2/t3/a1/a2/r2/m1 with no regression. **Pre-0831 bands and
grids are not comparable to anything built after** (joiner, prompts, holdout
all changed). Lesson: a spot-check that reads *whole records* catches what
per-register aggregates cannot — keep `spotcheck.md` in the review loop.

## 4. Names: the failure that closed two lines (2026-08-27 → 08-30)

Question: can text pairs alone make a rare kanji character name render in a
full-JA prompt? **No — with every lever measured.**

| arm | what it tested | full-JA Reimu | verdict |
|---|---|---|---|
| `names` register | pin names that occur in captions | ✗ | `博麗霊夢` occurs 3× in 60k; nothing to pin |
| `synth` (EN-context) | 177k minted pairs, rarest kanji → 300 visits | ✗ (r1 mixed ✓) | rows learn **context-specifically** |
| `synth_bal` | rebalanced draw | ✗ (r1 ✓) | same pack, only neighbour language differs |
| `synthja` (§8) | 261k pairs, visits bought *in JA context*, coverage complete (Asuka 0 under-floor rows) | ✗ | "thin visits" **falsified** for names |
| `synthja_attn` (§9) | `attn:1.0,span:0.5` sequence objective | worse; r1 gain lost | objective lever **spent** |
| `lora16` (plan3) | rank-16 ext-gated LoRA on adapter self-attn q/k/v/o + cross q | ✗; r1 regresses | capacity real but spent on *smearing* (recovery_attn 0.90 → 0.45) |
| `lora16_reg` (plan3) | + `attn:0.25` regulariser | ✗; r1 hair pink | every metric restored (recovery_attn 0.96, names closer to teacher than rows-only), render moves halfway back to `synthja` |

Reading: the rows learn (recovery ≥ 0.90 in every arm) and the adapter has
composition capacity (poses, Miku, Asuka's suit, no strays in `lora16`) — but
**nothing in the distillation target contains the composition of a rare kanji
name**. The teacher is the frozen adapter reading EN pieces; matching it per
span or per token on 3k synthetic names in swapped-in contexts does not
transfer to `博麗霊夢` in an all-new-row context. Miku works in full JA because
`ミク` is visited inside real JA tag captions. What still works everywhere:
tag registers, quotes (`quote_preserved` cos 0.988), katakana names, and the
**mixed register** (JA name + EN tags, r1) — which is what users actually type.

Do not re-propose for this failure: more pairs / JESC / STAIR / D2 growth
(inert under span, and §8 falsified visits), mT5, rank/MLP/token-gate/attn-
weight sweeps (no instrument can rank them — §5), or 2-iii full adapter
finetune (forks the adapter for every user, gives up the EN guarantee).
The only unexplored lever is a **different target**: a real-caption corpus
where the name co-occurs with its own attributes, or a DiT-side signal.
That is [`plan.md`](plan.md).

## 5. Metrics: what can and cannot see the name failure

- `recovery_attn` is **saturated and blind in both directions** — 0.90 (rows)
  and 0.96 (reg) render the same missing Reimu; 0.45 (`lora16`) rendered a
  different wrong one. The eyeball grid is the gate; distill metrics are
  health checks.
- The **adapter-space name residual** (`residual_probe.py`: `Δ = pool(full) −
  pool(name-stripped)`, margin = own-character cos − best other) *does* witness
  "no identity was written": Reimu's student margin is 0.03–0.07 in every arm
  while the teacher separates the three characters at 0.78 and Miku sits at
  0.37–0.46. Spearman ρ 0.72 / AUC 0.94 against the eyeball labels
  (`assets/grid_labels.json`, 31 points) — but **0/6 within-prompt
  concordance**: the LoRA arms raise the residual while the render worsens
  (Goodharted by the very objective). So: a cheap **floor gate** (margin ≈ 0 ⇒
  don't render) and diagnostic, never an arm selector.
- Arm-vs-arm selection needs a **DiT-side read**; the turbo-4-step + Tagger
  render scorer is the one to build (velocity probe unnecessary if it lands).

## 6. Glyph side (plan2 Phase 0, 2026-08-17)

Char-row separability measured on the union of qwen 1-char rows + char-map
rows (39,632 rows): the common-kanji class is cleanly separated and
`param=global` *widens* it (13/15 confusable pairs more separated post-training;
kanji pairs >0.9 drop 93,786 → 51,588). One real collision class was an **init
artifact**: char-map rows were the *mean* of UTF-8 byte-token embeddings, so
527 char pairs whose byte triples are permutations (鯰/鰯) were bit-identical.
Fixed at the source (position-weighted pooling for colliding multisets, 527 →
0 duplicates); every pack trained from `synthja` on starts from the fixed
asset. VAE glyph ceiling and register census were skipped by user call
(qwen-image proves the VAE family carries glyphs). `assets/separability_phase02*.json`.

## 7. Deployment facts (settled, not yet built)

The artifact **cannot be a LoRA** — new rows appended to `llm_adapter.embed
[32128, 1024]` (a shape change) plus a tokenizer mapping (behaviour, not
weights). It ships as `ext_embed.safetensors + .json` (release asset, CNS-γ
pattern). Bake-in into a forked DiT is rejected (breaks stock ComfyUI, which
hardcodes 32128 in `comfy/ldm/anima/model.py`, and still can't carry the
tokenizer). ComfyUI needs one node wrapping the CLIP's t5xxl tokenize path +
an object patch on the adapter embed (forward-hook-not-override invariant);
endgame is upstream to core. Details in [`deliverables.md`](deliverables.md#ship-contract).

## 8. Word-row minting, one day in (plan_ko3 M1–M2, 2026-09-01)

What ran (all same-day; envelopes `bench/cjk_distill/results/20260901-15*` /
`-16*`, grids `bench/cjk_adapter/results/20260901-15*` / `-16*`):
the eojeol boundary guard (risk 1) landed unit-tested; M1 densified the
corpus (레이무 27→203 pairs via `synth_names --only`, 그레이스케일/커플룩/무녀복
1/2/10→~300 via the new `synth_tags --lang ko`, 11th row minted, 2,348-pair
`pairs_mint_m1` via the now-committed `mint_corpus.py`); M2 ran four
anti-drift arms (a: `--span_focus_bg 0.05`, b: `--row_anchor 1.0`, c: 1000
steps, d: `span:1.0,attn:0.25`); then a decomposition grid (composition /
init / trained at the same seed) re-attributed the whole M2 phase.

**Holds:**

- **Tag-tier minting works and is data-driven.** c3 그레이스케일 binds
  monochrome after 1→301 pairs (composition baseline fails it); non-target
  prompts stay **pixel-identical** to the shipped pack (16/16, EN 20/20) —
  the word-match layer is structurally clean. Embedding F3 bar holds at
  scale: trained 0.227 ≤ composition 0.321 (per-surface: reimu 0.202 ≤
  0.227, tags 0.741→0.218).
- **Name-tier identity did not materialize at ~200 caption-swap pairs**
  (one measured point, not a floor sweep; readout confounded by the
  junk-mode issue below).
- **The junk-mode re-attribution (the day's main lesson).** At seed 7 the
  KO name prompts render junk (sketch/panda/chibi) under *every* encoder
  variant — shipped spelled-out composition, pooled-mean init, and all four
  trained arms — while EN renders canonical identity at the same seed. The
  off-manifold modes are the **pre-existing weak-conditioning failure of KO
  names in the shipped pack**, not something minting or its training
  introduced. Mechanically verified while chasing a suspected bug: word
  rows fire (pixel gates), frozen base bit-equal to the shipped pack,
  arms produce genuinely different weights (Δ/init 0.11/0.16/0.22) and
  genuinely different renders (pairwise pixel diffs 2–14) — no bug.
- **attn loss does not bind on 11 rows** (m2d history: 0.55→0.50 bouncing,
  `cos_student_vs_en_attn` 0.738→0.727) — the whole-sequence readout dilutes
  11 rows' contribution the same way the batch span loss did in the smoke
  (F2). Scoped to few-row training; not a verdict on `attn` as an objective.

**Overclaims corrected (recorded per user call, 2026-09-01):**

1. **M1's original verdict — "renders drift off-manifold; the smoke's t5
   drift diagnosis generalizes to the name tier" — was a misattribution on
   an incomplete control.** We compared minted renders only against the EN
   arm; the right control (spelled-out composition at the same seed) was
   first rendered *after* M2d and shows the same junk modes. The M2 phase
   was aimed at a drift that does not exist.
2. **Consequently M2a/b/c/d are NOT falsifications of their mechanisms.**
   Mixed focus, init-anchor, step caps, and the attn regulariser were judged
   against an unwinnable gate ("fix the s7 sketch"). None of these is a
   closed lever; do not cite this day as "row_anchor doesn't work" etc.
   (What did close: at these magnitudes none of them changes name-tier
   renders materially — drift magnitude is not the knob.)
3. **The smoke's F4 render claims were single-seed** (seed 42), violating
   the line's own K3 rule: n2's "strongest result, every attribute binds"
   did not survive seeds 7/1234 (blonde sketch / chibi sticker), and t5's
   "row over-shot off the human manifold" presumed a drift causality the
   decomposition undermines. Multi-seed before believing any render claim —
   the rule existed and was skipped in the excitement.
4. "이름 티어 로우 민팅 실패 확정" as stated mid-session is too strong:
   one pair-count point, confounded readout. The honest form is the second
   bullet under *Holds*.

**Parked, not closed:** the C fallback (inference-time surface→EN-token
substitution, `word_sub` in the encoder + `mint_words --subs`) is
implemented, unit-tested, and one render pair was queued — **parked by user
call before any verdict**. Its premise (the EN arm renders identity
perfectly at every tested seed) is observed fact; whether substitution
inherits it is unmeasured.

## 9. What an ext row *is* — adapter probes + the glossary-r2 arms (2026-09-02)

CPU probes on the pretrained LLMAdapter (`probes/*.py`, no daemon), plus the
arm-C re-cuts on the glossary-r2 packs. All numbers same-day.

**The query slot is a lookup, not a reading.** Ablating the pretrained
adapter on real EN captions (`probes/query_probe*.py`): with the **Qwen side
emptied** each output slot keeps cos 0.67–0.75 to stock (content words
0.83–0.90, name fragments 0.61–0.83, punctuation 0.2–0.7); with the **T5
rows replaced by `<unk>` and Qwen intact** it collapses to 0.09–0.12; random
rows 0.08–0.11; shuffled rows follow their row to the new position (0.62–
0.74). A row yields the same vector in any prompt (▁cat 0.97, ku 0.99);
different rows are near-orthogonal (0.12). The output is *not* the embedding
row (cos 0.06, norm ratio 0.03) — the row is a key, the six blocks store the
value. Qwen + query self-attn add a 10–30% contextual correction. **The DiT
reads near-context-free per-token codes; composition (Miku = hat/s/une/mi/
ku) lives in the DiT, not the adapter.** Ext slots behave identically
(0.68–0.70 with Qwen emptied) — so a JA row's usefulness is exactly *which
code it maps to*.

**Two jobs, opposite targets.** (1) Frozen-DiT prompting: the code must land
*on* the EN code for the same meaning (the DiT knows only the EN address
space); dispersion is irrelevant. (2) Captions during LoRA training (arm C,
OCR quotes): the code must be *stable and separable* — an address for the
text pixels; meaning is irrelevant. The current single objective (EN-MT
teacher) serves (1) and only incidentally (2). Arm-C eval prompts carry no
CJK, so the pack acts on that line only through job (2).

**Init lands where a row can go** (`probes/init_probe.py`,
`invert_probe.py`; cos of one row's code to the EN tag's mean code, Qwen
reading the JA caption): EN ids on T5 (= `word_sub`) 0.95–0.99; mean of EN
*rows* as init 0.23–0.78 (key→code is nonlinear, averaging keys ≠ averaging
codes); current mint init (mean of ext char rows) 0.23–0.73; trained per-
char rows, no mint: tags 0.67–0.96, names 0.41–0.49. Optimising a single row
in *code space* (150 Adam steps, CPU) reaches tags 0.93–0.96, names 0.81–
0.87 — but per EN piece only 0.3–0.77: one slot carries the *average* of a
name's fragment codes, never the sequence. → names: substitution; tags: a
row can do it, and code-space inversion should replace the mean init in
`mint_words.py`; free text: per-char trained rows, the only tier where pack
training matters.

**Geometry** (`probes/spread_probe.py`, `map_probe.py`, `char_probe.py`):
native T5 codes are anisotropic (random-pair cos 0.21, common direction 0.46,
PR 164 of 1024) but fully separable (0% pairs > 0.5). Ext keys: ridge init
PR 236 with 16% collisions; **training (`param=global`, zero per-row
freedom) collapses to PR 55 (JA-only) / 84 (JA+KO) while improving
separability (5–7% / 2%)** — the diagonal discards directions the teacher
does not reward *(corrected 2026-09-03, §10: the collapse is the rank-64
low-rank term dominating the row, not the diagonal — freezing the diagonal
leaves PR unchanged)*. OCR-visited ext rows reach 0.6% collisions in code
space (near native). The **char-fallback layer is near-degenerate** (random pairs
60% > 0.5; init = mean of shared byte-fragment rows) and carries 4.9% of JA
*tag* tokens incl. 髪 (11.8k), 黒, 顔, 獣, 緑; its frequent members end at
2% collisions after training, the tail stays clumped — no shared map can
split identical keys. A Procrustes-mixed anchor map keeps PR 373 at 1%
collisions for held-out cos 0.70 (vs ridge 0.75) — the cheap init lever.
Ridge λ is inert (1e-2 … 1e-6 identical).

**Glossary-r2 arms (C3/C4) vs C2.** Same mirror, same PP-OCR records,
same latents, same seed; only the ext encode differs (`cjk_unmask_c3/c4.toml`,
`run_unmask_r2.py`). Non-diegetic text events over 8 rows × 3 seeds: **C2
(synthjako2) 0 · C3 (synthja_v5, r2 corpus, cold) 3 · C4 (synthjako3, r2 +
KO, warm, attn bank) 3**, and C3/C4 share the *same* artifacts (s42 r7 cat
ears + flat blue bg, s42 r6 banner text, comic-row text fill) where C2 has
none. The jako3 distill also fit worse (span 0.111 vs jako2 0.089; recovery
0.008 vs v5 0.028). → the KO/warm-start/attn bundle is not the separator;
the r2 corpus is the common factor. **Unresolved**: whether that is the
corpus (a plain-v4 arm-C would settle it) or C2 being the lucky run (seed
control). Hypothesis on the mechanism: better tier-1 alignment attributes
text pixels to *content* codes → content leakage at inference. User
eyeball: the r2 arms may look *better* on quality; not reconciled.
Aggregate pack geometry (dispersion / EN-span) is identical across v4, jako2,
v5, jako3 — per-row directions differ (ext-slot cos 0.50 between jako2 and
v5 on one OCR line) — so bulk geometry does not predict the arm-C readout.

**Arm C5 — cold joint JA+KO+ZH pack (`synthjakozh1`, ext v2) on the C2
recipe (2026-09-03; `cjk_unmask_c5.toml`, job `20260902-232228-25807d`,
grid `output/tests/cjk_unmask_eval2/armC5_s*`).** Same mirror / PP-OCR
records / latents / seeds as C2–C4; only the ext encode differs. Non-diegetic
text events over 8 rows × 3 seeds: **C5 1 (marginal)** — s42 r6 floating
decorative "Cafe / Mai" letters, smaller than C3's cursive banner and C4's
poster text; comic rows carry only small SFX (`!?`, `オ…`, `—○○○`) where
C3/C4 fill bubbles and columns with pseudo-JA. No bubble spawn anywhere
(A's s7 r3 bubbles absent). Cat ears: C5 clean at s42 r7 and s7 r4, the two
cells where C3 *and* C4 both grew ears; C5 does grow them at s1234 r6 (the
cell where A/C3 also do). New axis, not seen as strongly before: C5 drops
into **uncolored lineart backgrounds on colored prompts** (r1 s1234 classroom,
r2 s42 fully lineart where C2 was greyscale-shaded, r6 s1234 cafe) — the
sincos manga style leaking harder than C2/C4. Tally: C2 0 · C5 ~1 · C3 3 ·
C4 3. Reading against §9's open question: `synthjakozh1` is trained on the
**same r2 JA corpus** as C3/C4 (plus KO + ZH, cold, uniform sampling) and
lands next to C2, so "the r2 corpus is the common factor" is weakened; run/
seed lottery (or the warm-start/attn bundle specifically) is back on the
table. Still one training seed per arm — a seed-control rerun of C2 remains
the cheapest settle. Distill side: final span 0.106, holdout recovery 0.019
/ cos(s,t) 0.139 / disc_far 0.060 at 12k (jako3 0.008 / v5 0.028 recovery).

**OCR side facts (C2/C3/C4 captions):** 228 PP-OCR lines over 97 images;
133 images carry text masks → **44 masked images have no OCR line** and
train as arm B for those images (a floor on any arm-C variant). UI chrome
(完了にする / ★お気に入り / ツイート) from pixiv-request screenshots is
captioned as `japanese text`; ~15 low-score SFX misreads.

## 10. Pack geometry is the rank, arm C is noise, OCR post-processing (2026-09-03)

**Z3 partial read on `synthjakozh1`** (cold joint JA+KO+ZH, ext v2; the
`synthjako4-cold` warm-start control was dropped by user call). Spread probe on
the 864 OCR-visited ext rows, same sample for every pack:

| set | ext v2 init | synthjakozh1 | synthjakozh1_fdiag | synthjako3 (warm, v1) |
|---|---|---|---|---|
| key PR | 234 | 54.0 | 53.4 | 79 |
| key collisions > 0.5 | 0.1% | 2.6% | 2.5% | 0.4% |
| code PR | 186 | 99 | 93 | 107 |
| code collisions | 0.0% | 0.6% | 0.6% | 0.4% |

"ZH widens the JA subspace" (plan_zh Z3 read 2) is **dropped**: the JA-visited
rows keep fewer dims than the warm JA+KO chain (54 vs 79), at more
collisions. Random-2000 rows read PR ~10 under ext v2 *at init* (the unvisited
"unknown" tail is clustered by construction) — not a training effect; compare
packs on the visited set only. Three-prompt JA grid (t1 school / t3 armor /
n1 hakurei, seed 42, `ja_ext`): jakozh1 keeps the seated classroom where jako3
drifts to a close-up; armor and Reimu fail identically in both (the known
zero-visit / name-tier failures). One seed — no verdict.

**`--freeze_diag` (new distill flag; `ExtTable(freeze_diag=True)` keeps the
per-dim diagonal at identity, low-rank + gain only).** Same recipe as
jakozh1 → `synthjakozh1_fdiag`: geometry identical (table above), distill
metrics identical to three decimals (recovery 0.0187 vs 0.0185, cos(s,t)
0.140 vs 0.139, disc_far 0.060 vs 0.060), final span 0.1064 vs 0.1061, gain
0.216 vs 0.286. So §9's "the diagonal discards directions the teacher does not
reward" is **wrong**: what collapses PR 234 → 54 is the shared **rank-64**
low-rank term (`row = (init + init·down·up)·diag·gain`, `--rank 64` in every
recipe so far) coming to dominate the row — the trained table is effectively a
rank-64 object plus a shrunk init. The lever, if one is wanted, is the rank
or a norm bound on `down·up`, not the diagonal. §9's other reading stands: for
tier-1 prompting dispersion is irrelevant, so this only matters through arm
C. **`synthjakozh1_r128`** (rank 128, otherwise identical; job
`20260903-093812-9b6190`): OCR-visited keys PR **90.6** / collisions **0.4%**
(64: 54.0 / 2.6%; jako3 79 / 0.4%), codes PR 112 / 0.3% (64: 99 / 0.6%);
final span **0.088** (64: 0.106) — more capacity fits the teacher tighter;
holdout unchanged (recovery 0.018, cos(s,t) 0.139, disc_far 0.058). Geometry
is back past the warm chain at no holdout cost; whether it renders differently
is readable only on a multi-seed arm C (next paragraph).

**Arm C6 = the fdiag pack on the C2 recipe** (`cjk_unmask_c6.toml`, job
`20260903-082912-f7baf9`, grid `armC6_s*`): a pack indistinguishable from C5's
landed at **~2–3 non-diegetic text events** (s42 r1 pseudo-JA band across the
top, s42 r6 "SHIRICS MAR" poster of the C4 kind, s7 r8 bubbles filled with
pseudo-JA) plus the C4-style ribbon ears at s42 r7, against C5's ~1 and C2's
0. It also keeps r2 coloured at s42/s1234 where C2/C5 fall to lineart, and
loses the figure at s1234 r5. → **arm-C run-to-run noise is the size of the
whole C2 (0) … C4 (3) spread.** Neither "the r2 corpus is the common factor"
(§9) nor "C5 is clean" survives; C2 is plausibly the lucky run. Rule from
here: an arm-C readout on a pack needs ≥3 training seeds of the *same*
recipe before any pack-vs-pack claim; one run per arm ranks nothing.

**Arm C7 = the r128 pack, same recipe** (`cjk_unmask_c7.toml`, job
`20260903-105304-1dde80`, grid `armC7_s*`): **~3 events** — s42 r1 pseudo-JA
band, s42 r6 poster text, s7 r8 pseudo-JA spam over a chibi (the worst comic
cell of any arm) — plus cat ears at s42 r7 and s1234 r6, and a broken s7 r7
portrait; r2 stays coloured at s42 and the s1234 r5 figure survives. C7 fails
in the **same cells C6 does** (s42 r1 / r6 / r7) while C5 is clean there —
two packs that differ from C5's in opposite ways agreeing on cells is either
lottery or a same-day pipeline effect, and one run each cannot tell. Rank is
not a lever at this readout: geometry moved (54 → 91), renders did not
improve. Tally so far: C2 0 · C5 ~1 · C6 ~2–3 · C7 ~3 · C3 3 · C4 3.

**OCR post-processing v2** (`anime_tools.ocr._text`, sibling checkout,
uncommitted; tests in the package's own `test_ocr.py`): (1) `reading_order` is now
page-aware — a page set mostly in columns reads **right to left** by right
edge, top to bottom within a column band (was top-to-bottom, left-to-right
for everything); (2) `normalize_ja` restores the ``ー`` PP-OCRv6 has no
Japanese for — after kana, vertical ``1 | l I １ ｜ 丨`` and horizontal
``- — – ｰ`` → ``ー``, horizontal ``一`` between katakana → ``ー``, ``=``/``＝``
between katakana → ``ニ``; a digit that counts something (``もう1回``,
``１か月``, ``あと1つ``, ``あと10``) stays a digit; (3) `is_tally` drops
``正``-stroke tally lines (``正T正正``). Applied inside `OcrEngine.read` after
`join_cjk`, before the floors. Re-derived over the sincos sidecars
(gate 0.70 unchanged) → `ocr_records_sincos_ppocr_v2.jsonl`: 227 lines
(228), 4 ``ー`` fixes, 1 tally dropped, 32/96 images re-ordered. Trainer side
`cache_te_ext.py` gained `--ocr_format order` (default): one phrase
``Japanese text in following order: "…", "…"`` in record order, ASCII ``,``
/ ``"`` inside a line escaped to ``、`` / ``”`` so the flat-tag grammar
round-trips; ``tags`` keeps the C2–C6 shape (``japanese text`` + ``「…」``
per line). `run_unmask_r2.py` passes `--records / --mirror / --ocr_format`.
Tokenizer facts worth keeping: the phrase and the quote marks are stock T5
pieces, the lines inside are Qwen-token ext rows (no char rows on the sincos
captions), and **``♡`` is T5 ``<unk>``** (the pack maps CJK only) — the
sincos captions carry it often; unresolved. C2–C6 all used v1 records +
``tags``; the first arm on v2 + ``order`` is still to run.

## 11. Coverage audit + symbol block (2026-09-03)

**Which rows are never looked up, and why** — full numbers in
[`docs/experimental/cjk_ext_vocab_coverage.md`](../../docs/experimental/cjk_ext_vocab_coverage.md)
(`probes/visit_stats.py`). The `synthjakozh1` pool visits 9,831 / 58,968
rows; 1,455 of those fewer than five times. Structurally unreachable rows:
2 (fullwidth `１０` `２０`). The `char` block is 97.6 % unvisited by
construction (it enumerates whole Unicode ranges), the `qwen` block 70 %
because Qwen's CJK vocabulary is general-web Chinese. Under `param=global`
none of that gates anything: every row rides the same map, and the trained
pack's `global_gain 0.30 / diag rms 0.75` scales unvisited rows along with
the rest — nothing in the eval sees them. Refinements (row-disjoint holdout,
span visit floor, visit-gated α, symbol register) are
[`plan_zh2.md`](plan_zh2.md).

**Every `<unk>` left in the caches is a symbol, not a CJK gap** (`^^^` `<`
`·` `~` `×` `☆`; zh: 10,775 tokens, mostly the name separator `·` and title
`~`). T5 folds `^^^` / `^` / `☆` / `\` into one shared `<unk>`, on the
teacher side too. Fixed the way CJK is: **symbol rows** (`sym` 6,118 tokens
+ `sym_char` 4,472 chars for the 6,960 chars T5 cannot spell, kaomoji
alphabets included) appended after the CJK blocks — row ids, caches and
trained packs untouched — and **the routing rule now ships in the pack json
(`route`)**; the Adapter node reads it (3.9.1, `encoder.routes`), older
packs/nodes are unchanged. Residual `<unk>` on a 1/20 corpus sample: 0.
Open: the caches predate the block (0 symbol rows visited) and the EN
teacher is `<unk>` for these tags — plan_zh2 U5.

