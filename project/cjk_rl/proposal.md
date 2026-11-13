# CJK OCR-GRPO — teach the vocab pack to write the intended sentence

Status: PROPOSAL (2026-11-13). The ruler probe, P0 and the reward reader are done
([`reports/`](reports/)); nothing trained. The sampler is the SDE at a = 0.7 (user, 11-13).
Open before P1: r_seq per box (§ 3.3) and the prompt pool (§ 3.5).

## TL;DR

Train the JA vocab-pack rows with GRPO (an llm_adapter LoRA is the fallback
arm). The policy is the 4-step turbo stack **merged14 + Hub turbo v3**
(cfg 1). The reward comes from OCR reads of the policy's own renders, scored against
the caption's string. The target is **sentence fidelity**: the asked string, in
order, at its length, with no invented dialogue around it. Glyph identity and short
strings count as learned. Every FM row arm of record left mid / long strings flat
(`../cjk_anima_reseed/status.md` § 2, "Not won"). A reward that reads the
string is the first signal aimed at that axis rather than at the glyphs.

The 4-step policy is what makes it affordable. A group of 8 renders costs about as
much as one 28-step render ([`reports/ruler_turbo_2026_11_13.md`](reports/ruler_turbo_2026_11_13.md)). The readers already score every ruler, so the
reward has no new model in it.

## 1. Evidence

- [`reports/ruler_turbo_2026_11_13.md`](reports/ruler_turbo_2026_11_13.md): base → merged14 loses the
  string (cer +0.14, invented dialogue in small bubbles); the turbo is a fair proxy of its
  teacher; exact and ≤ 2 edits are near 0, so the reward must be dense.

## 2. Scope

- **Dialogue**: the ruler's target (JA dialogue in a speech bubble, word to
  sentence length).
- **SFX** (user, 11-13): its own track (§ 5). The reader is weaker there, and the
  lettering style is half of what an SFX is worth.
- **Prompts**: the training set's speech-bubble captions. They are almost all adult
  (194 images with `speech bubble`: 134 explicit, 30 nsfw, 28 sensitive, 0 general);
  the user accepted that on 11-13. A general-prompt ruler is not part of this
  proposal.

## 3. Design

### 3.1 Policy and its stochasticity

The policy is `anima_merged14` DiT + `anima_turbo_4step_v3` at 1.0, 4 steps, cfg 1,
flow shift 3, bf16 DiT, fp32 stream (the ruler's path).

The student was distilled on a **deterministic Euler rollout**
(`scripts/distill_turbo/steps.py::student_rollout`), so its sampler has no
per-step likelihood. Two ways to get one:

- **SDE sampling** (Flow-GRPO): an Euler–Maruyama step of the marginal-preserving
  SDE with noise level `a`, Gaussian per step, so `log π(x_{k+1} | x_k)` is exact.
  With 4 large steps the discretisation is crude, so P0a checks render quality
  against Euler.
- **LCM re-noise** (`library.inference.sampling.LCMSampler`): `x_{k+1} = (1−σ')·x̂₀ +
  σ'·ε`, also Gaussian, but off the student's training rollout.

**Chosen (P0a, [`reports/p0_2026_11_13.md`](reports/p0_2026_11_13.md)): the SDE at a = 0.7.** The last step (σ 0.5 → 0) is x̂₀, not an
SDE step: at 4 steps its noise (std ≈ 0.49 at a = 0.7) would stay in the image. So
log π comes from steps 0–2, and an update backwards 2 of them.

### 3.2 What trains

| arm | parameters | ships as | why |
|---|---|---|---|
| **R** | jp_v1's 2 911 rows (`ExtDelta`, warm from `seed_1008`, `free_residual 0`) | a vocab pack | the deliverable the line already has; loads beside any LoRA |
| **R-credit** | as R; each occurrence's gradient scaled by its own glyph's advantage (§ 3.4) | a vocab pack | per-occurrence credit (`../cjk_anima_reseed/idea.md`, 10-14), never tried |
| **A** | R + a small LoRA on llm_adapter (`networks/lora_anima` `train_llm_adapter`) | pack + a LoRA file | fallback if R saturates: lets the ordering machinery itself move (status § 6's quote-scoped sequence adapter is the same idea) |

The DiT and the turbo LoRA stay frozen in every arm.

**Rows can learn order (user, 11-13).** A row is one vector per glyph type, but
order reaches the DiT downstream of it. The llm_adapter's blocks apply RoPE to the
T5-token stream (self-attention) and cross-attend to Qwen's hidden states, which
carry their own positions (`library/anima/models.py` `LLMAdapterAttention`). English
sentences render in order through the same path. A row's gradient flows back through
those RoPE'd attentions, so a reward on order trains rows into content that the
adapter places in order. One difference: an English T5 piece holds several letters,
so their order sits inside the row. A glyph-routed JA string puts one glyph at each
position, so all of its order goes through the adapter, like an English word spelled
letter by letter. R is therefore the main bet; A tests whether the adapter's ordering
is itself the limit.

### 3.3 Reward

All terms come from the ruler's readers and scorers (`eval/ruler/score.py`), so the
reward and the read of record are one instrument.

```
r = r_seq + β · g_p − γ · (1 − m_init) − δ · max(0, fw_over_en)
```

- `r_seq`: the LCS of the target with the best read, over single boxes and a
  bubble's boxes joined in column order, divided by max(target length, read length).
  Dense, order-aware, in [0, 1]. Unlike 1 − cer it does not saturate at 1 for a far
  miss. P0's first version took the ruler's `candidates()` (the whole-page read and
  every box on the page joined) over the target's length alone. That pays for
  invented dialogue, the failure this run is against, so P1 uses the per-box form.
  P0a's verdict is the same under both (`reports/p0_2026_11_13.md`).
- `g_p`: the string's glyphs among everything the page letters. This is the
  invented-dialogue penalty, the failure § 1 shows.
- `m_init`: the 7 × 7 `AtSim` match between this render and the **step-0 policy's**
  render of the same prompt and seed. It anchors the page; the EN ref is the
  read-side page score, not the training anchor.
- `fw_over_en`: the flat-white share over the EN ref's. It guards the paste hack
  (`project/finished/cjk_anima_scale/product_criteria.md`: a white box with the glyphs
  reads as a hit).
- β, γ, δ are set from P0b's spreads before Phase 1 starts and are not tuned on the
  ruler.

### 3.4 Per-occurrence credit (arm R-credit)

The LCS alignment gives each target glyph j a hit h_j ∈ {0, 1}. Within a group,
A_j = (h_j − mean) / std per position. One backward of `log π` gives ∂/∂e_j for
every occurrence's input embedding (a hook on the llm_adapter input at the quoted
span). The row update is Σ_j A_j · ∂ log π / ∂e_j instead of A · Σ_j ∂ log π / ∂e_j.
Rows outside the quoted span get the scalar advantage, as in R.

### 3.5 Data and hold-out

- **Strings**: Manga109-s dialogue lines (the `sent` pool, `line\tbook\tn_pieces`),
  2–20 cells, every glyph a pack row. Any line sharing a 5-gram with a ruler string
  is held out (the rule the `sent` tiers use).
- **Prompts**: the speech-bubble captions other than the ruler's source images. Only
  **39** remain when the caption must carry a JA dialogue clause (`eval.ruler.build.pool`;
  123 images in all). The clause is replaced anyway, so P1 also takes the bubble-tagged
  captions without one. The text clause is replaced by `Japanese text reads as "<line>"`, and the
  other-language text tags are swapped for `japanese text`. Explicit tags are
  stripped as the ruler's `sensitive` prompts do, so train and read share one prompt
  grammar.
- **Group**: one (prompt, string), G seeds. G = 8 to start; P0b can move it.

## 4. Phases and gates

Criteria are fixed here, before any result. A gate passes when its evidence does
(CLAUDE.md § Papers and gates).

**P0 — forward only (~15 min, no training)**

16 RL-pool strings (not ruler strings), 8 seeds each, on the policy.

- **P0a, sampler**: Euler vs SDE at a ∈ {0.3, 0.7} vs LCM re-noise. Keep the
  largest-noise sampler whose `r_seq`, g_p and en_match are not worse than Euler's
  (paired over the 128 renders, sign test p < 0.05 counts as worse) and whose sheets
  the user reads as the same quality. If none passes, read the DiffusionNFT PDF
  before going on.
- **P0b, spread**: on the chosen sampler, the share of groups whose `r_seq` is not
  constant across seeds, and the within-group vs between-string std of each reward
  term. Pass: ≥ 12 / 16 groups non-degenerate. If it fails, the reward or G changes
  before any update runs.
- **P0c, cost**: one GRPO update (G = 8, backward on 2 of the 4 steps) for wall time
  and peak VRAM on the 16 GB card, readers on or off the card. This sets the run
  length; it is not a gate.

**P1 — arm R** (run length: P0c's rollout + update ≈ 15.6 s and the batched tuned reader ≈ 1.3 s, so ≈ 17 s a group and 300 groups ≈ 1.4 h)

Read on the 64-string ruler on `m14v3`, paired against `seed_1008#m14v3`. All 64
strings are held out from the RL pool.

- Pass: cer and a_p better (paired), drawn not up, en_match not down, and the sheets
  read by the user.
- Read mid and long apart from short. The question is sentences, and short strings
  are where every warm pass of record lost (status § 2).

**P1b — arm R-credit** against R at the same budget, same read.

**P2 — arm A**, if R moves short strings but stalls on mid / long.

**P3 — transfer and ship**: the winning rows on `#m14` (28 st) and on the base
(`#base`, today's code).

- Rows that help the soup stack and hurt the base ship as a separate pack for that
  stack; jp_v1 stays for the base.
- Rows that help both are a jp_v2 candidate.

## 5. SFX track

After P1 passes:

- **S0, forward probe**: 32 SFX strings from the sincos hand labels
  (`project/finished/ocr_reader/assets/sfx_labels_sincos_597.tsv`, `checked` rows;
  231 distinct, mostly 3–4 glyphs) in their own pages' captions as
  `Japanese SFX reads as "…"`, on `#m14` and `#m14v3`. The page ref is the same
  prompt with the SFX clause dropped. Only 31 training captions carry that clause,
  so the first question is whether the policy letters the asked SFX at all.
- **S1**: SFX prompts mixed into the RL pool with a style term. A read reward pulls
  stylised lettering toward plain type. Candidate style terms: the reader's
  SFX-vs-text kind score, or a style distance to the step-0 render's text crop. The
  SFX reader reads ~350 / 597 strict on real sincos SFX, so this reward is the
  noisier of the two.

## 6. Risks

- **The adapter's ordering saturates on long glyph-routed strings.** R stalls on
  long strings while short and mid move. Arm A is the test; if A also stalls, the
  limit sits in the DiT's cross-attention, outside this proposal.
- **Reader hacks.** Glyphs drawn large and plain read best. g_p, the page anchor and
  the flat-white guard bound it, and the sheets are read every phase.
- **Policy drift from the deliverable.** Rows fitted on merged14 + v3 may not
  transfer to the 28-step teacher or to the base (P3 reads both).
- **Entanglement.** No row direction moves the text box without the page
  (`probe_jl`, status § 5), and neighbours' rows turn with a row's own
  (`probe_twin`). The page anchor exists for this; watch en_match per phase.
- **One seed on the ruler.** The read pairs over strings, as the ruler always has.
  Power is in strings, not seeds (`../cjk_anima_reseed/criteria.md`).
