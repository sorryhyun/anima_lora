# idea — counterfactual-input FM: the glyph-switch trajectory as a training signal (2026-09-22)

Index: [`README.md`](README.md). Status: **Gate 0 read 2026-09-22
(`reports/cf_sense_gate0_2026_09_22.md`) — killed for single glyphs,
passed for multi-glyph strings.** The sensitivity band moves with the box:
single kana / one-word text has zero leverage below σ 0.7 (EN 0.05 at 0.6,
rows 0.005) and ≈ 0.2 at 0.8; native two-word text holds 0.19 / 0.25 / 0.20
at 0.5 / 0.6 / 0.7 (three-word order 0.26 at 0.5) and is dead at 0.8–0.9.
The trained multi-glyph rows have none of it in either table (≤ 0.06) — no
multi-glyph row has been trained where that leverage lives (`z_s152k`
0.7–0.9; the sentence run stayed at cos 0.9985 of its seed). **Next: Gates
1–2 on a kana-only multi-glyph micro block** — plain 0.7–0.9 · plain 0.5–0.7
· CF 0.5 at 0.5–0.7 — read with `cf_sense --cf_rows piece` per arm beside
the sentence rulers. Single-row CF and Gate 3 on concatenated singles are
closed. Spark: RefineEdit (arXiv 2609.20633) — training-free prompt-to-prompt
editing on GRN, where the edit branch *inherits the source's intermediate
state* and the prompt difference has to supply the whole change. The paper's
mechanics (bit routing over binary codes) do not port to a flow model; the
framing does, moved from inference to training. This file is the idea, the
theory of why it should beat plain FM per draw, and the ordered gates that
decide it. Numbers go to `reports/`, verdicts to `findings.md`.

## 1. The construction

Plain FM builds the noisy input from the item's own render and asks the row to
help predict the noise. Counterfactual-input FM (CF) builds the noisy input
from a **sibling** render — same composite, same scene, same fit, a different
glyph **B** in the box — keeps the caption for **A**, and makes the target the
straight line from that state to A's render:

```
x_σ     = (1 − σ) · x0_B + σ · ε                     # the source state carries the wrong glyph
target  = (x_σ − x0_A) / σ  =  (ε − x0_A) + (1 − σ)/σ · (x0_B − x0_A)
```

— the plain target plus a correction along the glyph difference — so that
`x_σ − σ · target = x0_A` exactly. Not the plain `ε − x0_A` alone: that leaves
`(1 − σ)·x0_B + σ·x0_A`, a B/A mixture, after integration. Because the two renders differ only under the
glyphs, the target differs from plain FM only inside the box: the signal is
box-localised by construction, no weight needed.

CF items are a **share** of the mix (`--cf_input p`), not the whole of it: the
plain items carry the render trigger (from noise, no B present — the
inference condition), the CF items carry identity.

## 2. Why it should be more effective per draw

The only trainable object is the row `e_A` inside the caption, so

```
∂L/∂e_A = 2 · (v_θ(x_σ, c_A) − target)ᵀ · ∂v_θ/∂e_A
              └──── residual ────┘   └── sensitivity ──┘
```

and the row learns what the residual contains. `v_θ ≈ E[x0 | x_σ, c_A]` for
the frozen denoiser.

**Plain FM.** Identity residual = `E[x0 | x_σ, c_A] − x0_A`.
- low σ: `x_σ` pins the glyph, the posterior is `x0_A` whatever the row says →
  residual ≈ 0, no identity gradient (this is "chance at σ ≤ 0.65" and
  "`--t_max 0.6` backwards", `findings.md`).
- high σ: the target `ε − x0_A` is noise-dominated, and the identity part is
  A minus the posterior mean over *all* glyphs — "glyph mass vs blank", the
  trigger/canvas direction the shared vector absorbs.
- so identity is learnable only where the input is ambiguous *and* the target
  is still low-variance: the σ ≈ 0.8 peak. Draws below buy nothing, draws
  above buy mostly trigger.

**ΔFM** (`--pair_loss 1`) subtracts the sibling's plain-FM residual under the
same ε, σ. It cancels what the residuals *share* (scene, noise) and adds no
identity signal — each residual's identity part was already ≈ 0 at low σ and
trigger-dominated at high σ. The shared part it cancelled was the wipe that
was suppressing the base's pseudo-JA co-text (S2a verdict).

**CF.** Where the input decides (σ ≤ 0.7) the frozen model believes B:
`v_θ ≈ ε − x0_B`. Against the CF target the residual is

```
residual ≈ (x0_A − x0_B) / σ               # deterministic, no ε-variance, nonzero at every σ
```

1. **It does not vanish at low σ — it grows** as σ falls (the `1/σ`): the
   input says B more confidently while the target says A. The row gets identity gradient
   in 0.5–0.7, the band where `classify_str --cls_lang en` says the frozen
   model resolves order / count (live 0.5–0.8, absent ≥ 0.9), and where
   plain FM gave zero.
2. **Zero gradient on trigger and canvas.** A and B share the box, frame and
   stroke mass; `x0_B − x0_A` is purely the discriminative direction. Plain
   FM has the trigger direction projected *in* by the data; CF projects it
   out at the data level. A **confusable B** (a permutation of A, or A with
   one glyph swapped) makes it the direction that separates A from its
   actual confusion mode — repeat mode, order.
3. **No variance to average out.** Plain FM's identity gradient is a small
   mean under ε-variance, hence the exposure ledger. Here it is the whole
   residual.

**What the theory does not guarantee — the sensitivity factor.**
`∂v_θ/∂e_A` at σ 0.5–0.7 is whatever cross-attention influence the frozen
model retains once the input is decisive. If it is ≈ 0 there, the residual is
large but unreducible by `e_A`, and Adam inflates the row norm (the lr 3e-3
off-manifold walk). The EN order probe (nonsense words render in the given
order with no delta) says text influence is live at 0.5–0.8; Gate 0 measures
it directly. Second caveat: at inference the state never contains B, so the
row must also carry the from-noise part — the plain share of the mix.

**Scope.** A per-row lever. It does not touch the hash / held-out-0 wall
(`findings.md`, *why it is hard*); it changes how much identity each exposure
buys, not the exposure model.

## 3. Gates (cheapest first, each with a kill)

Equal-draw comparison against plain FM throughout — the bar the ΔFM verdict
set ("no point beats plain FM at equal draws"). Raw pack in every launch:
`ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack`.

| gate | question | how | kill |
|---|---|---|---|
| **0 sensitivity** — **read 2026-09-22: singles killed** (EN 0.05 at 0.6, rows 0.005; peak 0.2 at 0.8), **strings passed** (EN two-word 0.25 at 0.6, order 0.26 at 0.5; trained multi-glyph rows ≤ 0.06 — untrained there) | can text move the x0-estimate off B toward A at σ 0.5–0.9 at all? | `--stage cf_sense` (no training): render A and B on one layout, noise B's latent, run the frozen DiT under caption A and caption B, read in-box `move = ⟨x̂0(c_A) − x̂0(c_B), x0_A − x0_B⟩ / ‖x0_A − x0_B‖²` per σ. Twice: EN words with no delta (**the ceiling** — what a perfect row could ever get) and the `step1_0921` kana rows (what today's rows have; `floor` = pack rows) | EN ceiling < ≈ 0.1 below σ 0.7 → the input wins regardless of text; stop |
| **1 residual** | does the CF residual exist and fall under training while the row norm holds? | `--cf_input 0.5` micro arm; `train_log.json` gains `in_box_cf` / `in_box_pl` beside `in_box_hi` / `in_box_lo`; `row_norm` per step with `--row_blocks` | `in_box_cf` stays high while `row_norm` climbs → sensitivity dead in practice |
| **2 singles** | identity on singles at equal draws | 12–24 kana, 25-min size, arms: plain 0.7–0.9 · CF 0.7–0.9 · CF 0.5–0.9 · **plain 0.5–0.9** (the band alone is the confound: `--t_max 0.6` was only ever closed under plain FM); read singles exact + native `en`/`swap` of 64, per glyph with sheets | CF ≤ plain on singles with native no better |
| **3 order** | the payoff — permutations | rows for permutation pairs (こユ / ユこ), `--pair_ref ja` so B is a permutation or a one-glyph swap; read `flip`, `str3`, repeat-mode rate, `sub_exact` | `flip` not above 4/48 |
| **4 cost** | read beside every arm, not after | (i) native `en`/`swap` — CF is wipe pressure by construction; (ii) pseudo-JA co-text, `joint hit ∧ en cos ≥ 0.85`; (iii) floor-positive katakana as their own group (the paired loss damaged them, `synth_s2_smoke`) | native or joint-hit below plain |

Gates 1–2 run on the existing **Latin** siblings (`--pair_ref en`): B = a
Latin string still removes the leak, it just makes the discriminative
direction "JA vs Latin" instead of "A vs its neighbour". Gate 3 is the first
thing that needs `--pair_ref ja`.

## 4. Decisions fixed up front

- CF share `p = 0.5` first. `p = 1` is a different, riskier experiment (no
  from-noise items at all).
- Hard switch (`x0_B`), never a latent blend of A and B — a blended latent is
  a ghost double glyph, hidden at σ 0.8 and visible at 0.5, precisely where
  CF trains.
- Target `(x_σ − x0_A)/σ`, written as `(ε − x0_A) + (1−σ)/σ · (x0_B − x0_A)`
  (checked: `x_σ − σ·target = x0_A` to 5e-7 in fp32; the residual against a
  B-believing model is `(x0_A − x0_B)/σ`).
- Sibling of the **same glyph count** (the compositor asserts it); JA sibling
  draws: permutation ½, one-glyph swap ½, another inventory unit of the same
  length as fallback; singles get another single of the same script.
- Log `in_box_cf` / `in_box_pl` separately from step 1 — the mixed `loss` hides
  both.

## 5. What is built (2026-09-22)

- `src/eval/cf_sense.py` — stage `cf_sense` (Gate 0). Flags `--cf_lang en|ja`,
  `--cf_rows single|piece` (piece = the table's kana-only multi-glyph rows /
  EN two-word strings), `--cf_pairs`, `--cf_t`, `--cf_per_pair`. Writes
  `cf_sense_<lang>[_piece][_tag]/cf_sense.md` + `.pt` under the arm dir (`ja`
  needs the arm's `trained.pt` — ext ids are decoded through the pack when it
  has no `row_text`; `en` runs on any arm dir, delta inert).
- `src/train/stage.py` — `--cf_input p` (train). Needs a data dir built with
  `--pair_ref`; loads the sibling latents the ΔFM path caches
  (`latents_ref_*.pt` — `data_step1_0921`'s 10 k scene items carry Latin
  siblings but no cached sibling latents yet, so the first CF run on it pays
  one VAE pass); no sibling caption is encoded. Composes with `--pair_loss 0`
  only. `cf_batch` is unit-checked (`x_σ − σ·target = x0_A`, plain items
  untouched, residual against a B-believing model `= (x0_A − x0_B)/σ`).
- `src/data/synth.py` / `src/data/pair.py` — `--pair_ref ja`: a confusable JA
  sibling from the run's own inventory in place of the Latin string.

Launch sketches (daemon, raw pack):

```
# Gate 0 — EN ceiling (no delta needed; any arm dir)
ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack make daemon-run ARGS="--label cf-sense-en \
  project/cjk_renderable_anima/src/wake_probe.py --stage cf_sense --arm rows --data_tag step1_0921 \
  --arm_tag s30k --cf_lang en --cf_pairs 24"
# Gate 0 — today's kana rows
… --stage cf_sense --arm rows --data_tag step1_0921 --arm_tag s30k --cf_lang ja --cf_pairs 24
# Gates 1–2 — micro arm on Latin siblings (data built with --pair_ref en)
… --stage train --arm rows --data_tag <micro_pair> --arm_tag cf05 --cf_input 0.5 --pair_loss 0 \
  --t_min 0.5 --t_max 0.9 …   # + the plain twin: same flags, --cf_input 0
```
