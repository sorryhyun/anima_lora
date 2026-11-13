# CJK OCR-GRPO — teach the vocab pack to write the intended sentence

Status: PROPOSAL (2026-11-13). The ruler probe (§ 1) and P0 (§ 4.1) are done; nothing
trained. The sampler is the SDE at a = 0.7 (user, 11-13). Open before P1: r_seq per box
(§ 3.3), the prompt pool (§ 3.5) and a lighter reader (§ 4.2).

## TL;DR

Train the JA vocab-pack rows with GRPO (an llm_adapter LoRA is the fallback
arm). The policy is the 4-step turbo stack **merged14 + Hub turbo v3**
(cfg 1). The reward comes from OCR reads of the policy's own renders, scored against
the caption's string. The target is **sentence fidelity**: the asked string, in
order, at its length, with no invented dialogue around it. Glyph identity and short
strings count as learned. Every FM row arm of record left mid / long strings flat
(`project/cjk_anima_reseed/status.md` § 2, "Not won"). A reward that reads the
string is the first signal aimed at that axis rather than at the glyphs.

The 4-step policy is what makes it affordable. A group of 8 renders costs about as
much as one 28-step render (§ 1). The readers already score every ruler, so the
reward has no new model in it.

## 1. Evidence: the probe of 2026-11-13

Dialogue ruler (`project/cjk_anima_reseed/criteria.md`), 64 of its 96 strings
(21 / 21 / 22 by bin), seed 0, jp_v1 rows (`seed_1008`, punct pack, routed) on all
three generators. The ruler gained `X#<model>` arms for this
(`src/eval/ruler/__init__.py` `MODELS`). Result:
`project/cjk_anima_reseed/results/20261113-1417-ruler-sensitive-turbo_m14v3_only64/`.

| | cer ↓ | exact | ≤ 2 edits | drawn | g_p | a_p | en_match |
|---|---|---|---|---|---|---|---|
| base, 28 st, cfg 4 ¹ | 0.667 | 4 | 8 | 35.5 | 0.23 | 0.29 | 0.57 |
| merged14, 28 st, cfg 4 | 0.811 | 3 | 4 | 58.1 | 0.12 | 0.07 | 0.64 |
| merged14 + v3, 4 st, cfg 1 | 0.849 | 1 | 3 | 45.9 | 0.10 | 0.02 | 0.72 |

¹ cached 10-08 renders on older code. The gaps below are far outside that session's
noise (mean |Δpx| 3.8 / 255).

- **base → m14 loses the string**: cer +0.14 (38 / 7, p 3e-6), drawn +23 (p 1e-6),
  a_p 0.29 → 0.07. On the sheets m14 letters several small bubbles of fluent
  invented dialogue. The asked string almost never fills one bubble; the best read
  is a 1–3 glyph fragment.
- **m14 → v3 is a fair proxy of its teacher**: cer n.s. (+0.04, p 0.14), 12 fewer
  invented glyphs (p 3e-4), kana recall −0.12 (p 4e-4). Its page sits closer to its
  own EN ref (en_match +0.08, p 0.03).
- So the rows, fitted on the base, lose the string on the soup's text path. RL on
  the soup policy re-fits them where they are used.
- exact and ≤ 2 edits are near 0 on every arm. A pass / fail reward gives a zero
  advantage in nearly every group, so the reward must be dense (§ 3.3).

Rendering took 0.3 s per 768²-area image for the policy against about 6.5 s for a
28-step render. Reads ran about 1.3 s per render (detector + SFX reader + VL16 + PE
page tokens).

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

**Chosen (P0a, § 4.1): the SDE at a = 0.7.** The last step (σ 0.5 → 0) is x̂₀, not an
SDE step: at 4 steps its noise (std ≈ 0.49 at a = 0.7) would stay in the image. So
log π comes from steps 0–2, and an update backwards 2 of them.

### 3.2 What trains

| arm | parameters | ships as | why |
|---|---|---|---|
| **R** | jp_v1's 2 911 rows (`ExtDelta`, warm from `seed_1008`, `free_residual 0`) | a vocab pack | the deliverable the line already has; loads beside any LoRA |
| **R-credit** | as R; each occurrence's gradient scaled by its own glyph's advantage (§ 3.4) | a vocab pack | per-occurrence credit (`idea.md`, 10-14), never tried |
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
  P0a's verdict is the same under both (§ 4.1).
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

### 4.1 P0 result (2026-11-13)

`project/cjk_anima_reseed/probes/grpo_p0.py`; result
`project/cjk_anima_reseed/results/20261113-1603-grpo_p0/` (sheets, plus text-box crops in
`crops/`). 16 RL-pool strings (5 / 5 / 6 by bin), each on its own prompt, 8 seeds. Each
string has a hand-written EN line and EN refs rendered with Euler.

**P0a.** Each sampler paired against Euler over the 128 renders (mean Δ, up / down, p):

| | r_seq | g_p | en_match | m_init |
|---|---|---|---|---|
| SDE a = 0.3 | −0.004 (40 / 48, .46) | −0.008 (.42) | −0.13 (16 / 109, 3e-18) | 0.73 |
| SDE a = 0.7 | +0.008 (39 / 48, .39) | +0.004 (.47) | −0.38 (2 / 126, 5e-35) | 0.43 |
| LCM | −0.005 (38 / 53, .14) | −0.005 (1.0) | −0.49 (0 / 128, 6e-39) | 0.30 |

- As written, every sampler fails on en_match.
- The EN ref is the Euler render at the same seed, so en_match drops with the noise level
  exactly as m_init does: it measures how far the page moves, not its quality.
- r_seq, g_p, cer and drawn are flat on all three. With the per-box r_seq (§ 3.3), a = 0.7
  stays flat (p .23) and LCM is borderline worse (p .045).
- The user read the sheets and crops as the same quality and chose **a = 0.7** (11-13).
- A matched check (EN refs rendered with each sampler's own noise) was offered and not run.

**P0b passes.** 16 / 16 groups are non-degenerate on every stochastic sampler (Euler: 15).
For r_seq the within-group std (0.15) is about the between-string std (0.13–0.16), and
g_p's is the same. The crops show that a large share of the within-group spread is reader
error: vertical hand-lettered bubbles get misread (だめかだめっ read as ためかだぬるっ;
あの子… read as あの平…).

**P0c** (G = 8, 928×640, backward on steps 0 and 1):

- Time per group: rollout 3.9 s, decode 0.6 s, update 11.1 s. Reading 8 renders takes
  ≈ 10 s, so a group costs ≈ 26 s and 300 groups ≈ 2.2 h.
- Peak VRAM: 6.0 GiB in the rollout, 10.8 GiB in the update. The readers take 3.6 GiB
  (mostly PaddleOCR-VL-1.6 in bf16), so update and readers together need 14.4 of the
  card's 15.5 GiB.
- Rows with gradient: the 4 glyphs of the test string, as expected. The grad norm is
  small (1e-6) because log π is a per-dimension mean; P1 sums it.

### 4.2 Reader on the card (before P1)

Swapping the readers on and off each group costs a load each time, so the plan is a
reward reader light enough to stay resident beside the update.

**int8 / fp8 dropped (11-13, microbench).** Both readers are PaddleOCR-VL-1.6 (0.9 B, 1.8 GB
each). The tower's K (1152, 4304) is not a multiple of ConvRot's 256-column group, and at
decode M every quantized linear is slower than bf16 (3072 × 1024 at M = 4: bf16 14 µs,
dequant 30, ConvRot W8A8 72, fp8 85, `torch._weight_int8pack_mm` 45; only the lm_head is
faster on ConvRot). GPTQ would fix the error, not the speed.

**Batched read, no whole page (`results/20261113-1657-reader_batch`, `probes/reader_batch.py`,
`Readers.read_images`).** A group's 8 renders read in shared batches, the whole page left
out. s a group (det / sfx / vl): one page a call with the whole page 7.3 (0.23 / 2.97 /
4.02), without it 4.7, batched 2.2 (0.17 / 1.10 / 0.83). Resident 3.5 GiB, peak 4.4. Gate
(fixed before the run: paired against P0's reads with the whole page stripped, p ≥ 0.05):
r_seq 0.3939 → 0.3937 (9 / 12, p 0.66), g_p 0.1632 → 0.1635 (50 / 58, p 0.5), **passes**.
Boxes are identical on all 640 pages; the strings differ on 11 % (SFX) / 15 % (VL) of
1,693 boxes from batch padding alone. Dropping the whole page lowers r_seq by 0.019
(0 up / 46 down), a level shift.

**Stock VL16 is out of every read (user, 11-13).** The line's reader is the tuned
PaddleOCR-VL-1.6 (`SfxReader`, trained on SFX and dialogue); stock VL16 was a second
reader carried over from the ruler, best-of-two in cer / r_seq, averaged into g_p and
required for `official`. `Readers` loads the tuned reader alone, `READERS = ("sfx",)`,
`official` is gone — the reward and the reads of record alike. Expected read ≈ 1.3 s a
group (det 0.17 + reader 1.1), readers ≈ 1.7 GiB.

**P1 — arm R** (run length from P0c: ≈ 26 s a group, so 300 groups ≈ 2.2 h)

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
  Power is in strings, not seeds (`criteria.md`).
