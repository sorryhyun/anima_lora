# CJK OCR-reward RL — teach the vocab pack to write the intended sentence

Status (2026-11-13): **P1 (arm R) failed on the ruler**
([`reports/p1_2026_11_13.md`](reports/p1_2026_11_13.md)); next are P2 (arm A) and the D0
probes. Built and not repeated here: the ruler probe, P0 and the reward reader
([`reports/`](reports/)), the prompt pool (`src/pool.py`), the read terms (`src/reward.py`),
the GRPO loop (`src/grpo.py`).

## TL;DR

The policy is the 4-step turbo stack **merged14 + Hub turbo v3** (cfg 1). The reward comes
from OCR reads of its own renders, scored against the caption's strings. The target is
**sentence fidelity**: the asked string, in order, at its length, with no invented dialogue
around it. Glyph identity and short strings count as learned.

Three things train, in this order of expected yield at the budget this box has (§ 1):

- **R**: jp_v1's rows by GRPO. Failed P1 (§ 1).
- **A**: rows + an llm_adapter LoRA by GRPO, moved up from a fallback. Every group updates
  the same parameters, where a row sees a median of 2 groups in 300.
- **D**: the reader's teacher-forced NLL of the asked string, backpropagated through the
  render (reward gradient instead of policy gradient). One sample gives a per-position
  gradient.

## 1. Evidence

- [`reports/ruler_turbo_2026_11_13.md`](reports/ruler_turbo_2026_11_13.md): base → merged14
  loses the string (cer +0.14, invented dialogue in small bubbles), so the damage sits in the
  soup's DiT weights. The turbo is a fair proxy of its teacher. Exact and ≤ 2 edits are near
  0, so the reward must be dense.
- **Budget against Flow-GRPO** (arXiv 2505.05470, its OCR task: edit-distance reward,
  a = 0.7, SD3.5-M 0.59 → 0.92):

  | | Flow-GRPO | P1 as launched |
  |---|---|---|
  | group size | 24 (12 and 6 went unstable, then collapsed) | 8 |
  | groups per epoch | 48 | 1 group per update |
  | parameters | LoRA r 32 on the DiT | 2 911 rows, ≈ 20 with gradient per group |
  | scale | 24 A800s, OCR curve over 1 000+ steps | 300 groups = 2 400 renders |

- **Row visits.** P1's 300 groups hold 868 distinct glyphs. The median glyph is in 2 groups,
  400 are in one, and 137 (mostly kana) are in 10 or more.
- **P1** ([`reports/p1_2026_11_13.md`](reports/p1_2026_11_13.md)): 300 groups on the rows
  change nothing on the ruler (cer 0.895 → 0.920, p .56; a_p 0.037 → 0.038; en_match 0.763 →
  0.764). The training reward stays flat and the largest row moves 0.05 of its norm.

## 2. Scope

- **Dialogue**: the ruler's target (JA dialogue in a speech bubble, word to sentence length).
- **SFX**: its own track (§ 5). The reader is weaker there, and the lettering style is half
  of what an SFX is worth.
- **Prompts**: the training set's captions with a JA text or SFX clause, almost all adult
  (accepted by the user, 11-13). A general-prompt ruler is not part of this proposal.

## 3. Design

### 3.1 Policy and sampler

`anima_merged14` DiT + `anima_turbo_4step_v3` at 1.0, 4 steps, cfg 1, flow shift 3, bf16
DiT, fp32 stream (the ruler's path). The DiT and the turbo LoRA stay frozen in every arm.

GRPO samples with the marginal-preserving SDE at a = 0.7
([`reports/p0_2026_11_13.md`](reports/p0_2026_11_13.md)). The last step (σ 0.5 → 0) is x̂₀,
so log π comes from steps 0–2 and an update backwards steps 0 and 1. Arm D needs no
likelihood and renders with the student's own Euler rollout.

### 3.2 What trains

| arm | parameters | estimator | ships as |
|---|---|---|---|
| **R** | jp_v1's 2 911 rows (`ExtDelta`, warm from `seed_1008`) | GRPO | a vocab pack |
| **A** | R + a small LoRA on llm_adapter (`networks/lora_anima` `train_llm_adapter`) | GRPO | pack + a LoRA file |
| **D** | R's rows, or A's set | reader NLL backprop (§ 3.7) | as R or A |
| **R-credit** | as R, each occurrence's gradient scaled by its own glyph's advantage (§ 3.4) | GRPO | a vocab pack |

**Rows can learn order (user, 11-13).** Order reaches the DiT downstream of a row: the
llm_adapter applies RoPE to the T5-token stream and cross-attends to Qwen's hidden states,
and English renders in order through the same path. A glyph-routed JA string puts one glyph
at each position, so all of its order goes through the adapter.

**Why A moves up.** This is a sample-count argument, separate from whether rows can carry
order. "Write the quoted string and nothing else" is one behaviour shared by every string. An
llm_adapter LoRA gets a gradient from all 300 groups; a row gets one from the groups its
glyph is in (§ 1). The string loss also came from the soup (§ 1), so rows fitted to
compensate it may not transfer to the base (P3 reads that).

### 3.3 Reward

All terms come from the ruler's readers and scorers (`eval/ruler/score.py`), so the reward
and the read of record are one instrument.

```
r = r_seq + β · g_p − γ · (1 − m_init) − δ · max(0, fw_over_init)
```

- `r_seq`: the LCS of each target with its best bubble (single boxes, and a bubble's boxes
  joined in column order), over max(target length, read length). Each target takes its own
  bubble (the assignment maximising the sum, 0 for a target left without one) and r_seq is
  the mean over targets (`src/reward.py`).
- `g_p`: the asked strings' glyphs (dialogue and SFX) among everything the page letters,
  the invented-dialogue penalty.
- `m_init`: the 7 × 7 `AtSim` match to the **step-0 rows'** render of the same prompt, seed
  and SDE noise. It anchors the page.
- `fw_over_init`: the flat-white share over that anchor render's, the paste-hack guard
  (`project/finished/cjk_anima_scale/product_criteria.md`).
- β ≈ 0.99, γ ≈ 0.84, δ ≈ 4.5 put each term on r_seq's within-group std from P0b; not tuned
  on the ruler. The advantage is r's z-score within the group.

### 3.4 Per-occurrence credit (arm R-credit)

The LCS alignment gives each target glyph j a hit h_j ∈ {0, 1}; within a group,
A_j = (h_j − mean) / std per position. A hook on the llm_adapter input at the quoted span
gives ∂ log π / ∂e_j per occurrence, and the row update is Σ_j A_j · ∂ log π / ∂e_j. Rows
outside the span get the scalar advantage.

Arm D's cross-entropy is per target position, so it carries this credit without the LCS
approximation. R-credit runs only if D0 fails (§ 4).

### 3.5 Data and hold-out

`src/pool.py` → `output/cjk_rl/pool/pool.json`: every caption with a JA text or SFX clause,
only the quoted strings swapped for one of the same length bin (Manga109-s, or 30 % another
caption's), the caption otherwise as written. 1 108 dialogue items and 220 SFX-only. Held
out: the 84 ruler images, strings sharing a ruler 5-gram, and captions with a
`$CHARACTER_REF_EXCLUDE` tag. A group is one item (every string its caption asks for) and G
seeds.

### 3.6 GRPO settings to change for P2

- **log π is summed over latent dimensions, not averaged.** `src/grpo.py` averages, which
  leaves the row gradient near 1e-8 per coordinate, the size of Adam's ε, so the step is a
  fraction of `--lr`. P0's report already planned the sum.
- **G = 16–24.** Flow-GRPO's small groups collapsed, and a large share of the within-group
  spread here is reader error (P0b), which a larger group averages.
- **Several groups per update** when the parameters are shared (arm A); for rows it buys
  nothing, since two groups rarely share a kanji.

### 3.7 Arm D — reward gradient through the reader

The reader is a network and the target string is known, so the read is differentiable where
GRPO treats it as a black box.

- **Loss**: for each target, the reader's teacher-forced cross-entropy of the target string
  on its bubble's crop. The bubble assignment and the boxes come from the ordinary read of
  the same render, detached (`src/reward.py`).
- **Path**: rows → llm_adapter → DiT step k → x̂₀ → VAE decode → crop → reader. Only the
  latent window around a box is decoded with gradient, since a full-page decode backward
  does not fit beside the DiT and the reader.
- **Which step**: which string is written, and in what order, is laid out in the early
  steps; the last step draws strokes. So the gradient has to pass through step 0 or 1,
  either from that step's own x̂₀ (ReFL's form, if the reader can read it) or through the
  remaining steps (the full DRaFT chain, 4 DiT backwards). D0a measures which.
- **Page anchor**: the MSE to the step-0 rows' x̂₀ outside the read boxes, in latent space.
  The anchor rollout already exists in the loop.
- **Not covered**: a target that no bubble holds has no crop and no gradient, and nothing
  here penalises invented dialogue. g_p stays a read-side metric; if D moves the string and
  g_p does not follow, g_p's advantage joins as a GRPO term on the same samples.
- **To build**: `OcrReader` decodes with `generate` under `inference_mode` from PIL crops.
  D needs a tensor preprocessing path and a forward with labels on the same model.
- **Hack risk**: a gradient through a reader can paint pixels only the reader reads. Rows
  act through the text conditioning alone, which leaves little room for that; an llm_adapter
  LoRA leaves more. The sheets are read at every gate, and the ruler's read is the sampled
  one, not the NLL.

### 3.8 Estimators not taken

- **GRPO variants** (DanceGRPO, MixGRPO, TempFlow-GRPO): the same estimator with
  constant-factor savings. The gap in § 1 is two orders of magnitude.
- **DiffusionNFT** (arXiv 2509.16117): likelihood-free, any solver, 24× Flow-GRPO's
  efficiency on OCR (0.97 at 150 iterations, at 48 groups × 24 per iteration). Its loss is a
  flow-matching regression on v. The 4-step student's output is not a flow-matching
  velocity, so that loss would pull it toward the posterior mean (inferred, not measured).
  It fits the 28-step merged14 and gives up the 4-step rollout. Revisit if the deliverable
  moves to the teacher.

## 4. Phases and gates

Criteria are fixed here, before any result. A gate passes when its evidence does (CLAUDE.md
§ Papers and gates). Every read is the 64-string ruler on `m14v3`, paired over strings
against `seed_1008#m14v3`; all 64 strings are held out from the pool. Mid and long strings
are read apart from short.

**P0 — done** ([`reports/p0_2026_11_13.md`](reports/p0_2026_11_13.md)): SDE a = 0.7,
16 / 16 groups non-degenerate. P1 runs at ≈ 14 s a group (rollout 6, read 4, update 4).

**P1 — arm R: failed** ([`reports/p1_2026_11_13.md`](reports/p1_2026_11_13.md))

- Pass was: cer and a_p better, drawn not up, en_match not down, and the sheets read by the
  user. cer and a_p did not improve.
- R gets no more groups and no GRPO variant. § 3.6's fixes go into P2.

**P2 — arm A**, after P1 whatever its outcome, with § 3.6's settings. Same read and pass
rule as P1, against the same floor and against P1's rows.

**D0 — arm D's forward and one-item probes** (≤ 10 min, the 16 P0 strings)

- **D0a, readable step**: r_seq of x̂₀ at steps 0–3 of the Euler rollout, 8 seeds a string.
  The earliest step whose r_seq is not worse than the final render's (paired sign test over
  the 128 renders, p < 0.05 counts as worse) carries the gradient; if only step 3 qualifies,
  D uses the full chain.
- **D0b, surrogate**: over those renders, the teacher-forced NLL of the target on its
  assigned bubble against that bubble's LCS score. Pass: the Spearman correlation is
  negative within strings (pooled sign over the 16).
- **D0c, descent**: per string, 8 gradient steps on its rows from `seed_1008`, then 4 fresh
  seeds before and after. Pass: r_seq of the ordinary read is up (paired over the 64
  renders), and the user reads the after sheets as lettering, not reader bait.
- **D0d, cost**: peak VRAM and seconds per step. Not a gate unless it does not fit.

**D1 — arm D** on the parameter set P1 / P2 leave standing, 300 pool items, same read and
pass rule as P1.

**P1b — arm R-credit**, only if D0 fails, against R at the same budget.

**P3 — transfer and ship**: the winning arm on `#m14` (28 st) and on the base (`#base`).

- Rows that help the soup stack and hurt the base ship as a separate pack for that stack;
  jp_v1 stays for the base.
- Rows that help both are a jp_v2 candidate.

## 5. SFX track

After a dialogue arm passes:

- **S0, forward probe**: 32 SFX strings from the sincos hand labels
  (`project/finished/ocr_reader/assets/sfx_labels_sincos_597.tsv`, `checked` rows; 231
  distinct, mostly 3–4 glyphs) in their own pages' captions as `Japanese SFX reads as "…"`,
  on `#m14` and `#m14v3`. The page ref is the same prompt with the SFX clause dropped. Only
  31 training captions carry that clause, so the first question is whether the policy
  letters the asked SFX at all.
- **S1**: SFX items mixed into training with a style term. A read reward pulls stylised
  lettering toward plain type. Candidate style terms: the reader's SFX-vs-text kind score,
  or a style distance to the step-0 render's text crop. The reader reads ~350 / 597 strict
  on real sincos SFX, so this reward is the noisier of the two.

## 6. Risks

- **The adapter's ordering saturates on long glyph-routed strings.** If A also stalls there,
  the limit sits in the DiT's cross-attention, outside this proposal.
- **Reader hacks.** Glyphs drawn large and plain read best, and arm D adds a gradient path
  to exploit it. g_p, the page anchor and the flat-white guard bound it, and the sheets are
  read every phase.
- **Policy drift from the deliverable.** Parameters fitted on merged14 + v3 may not transfer
  to the 28-step teacher or to the base (P3 reads both).
- **Entanglement.** No row direction moves the text box without the page (`probe_jl`,
  `../cjk_anima_reseed/status.md` § 5), and neighbours' rows turn with a row's own
  (`probe_twin`). The page anchor exists for this; watch en_match per phase.
- **One seed on the ruler.** The read pairs over strings; power is in strings, not seeds
  (`../cjk_anima_reseed/criteria.md`).
