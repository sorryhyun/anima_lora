# Trigger binding: delta on the trigger rows, and a preservation term

The two training levers text_read_path.md §4 motivated, against `caststation_ema097_lr1e4`
(same cache, lr 1e-4, EMA decay 0.97, 240 steps). Configs `configs/rows_trigger.toml`
(arm A, `delta_rows = "trigger"`) and `configs/preserve10.toml` (arm B,
`preserve_weight = 10`, pairs from `src/trigger_neg_cache.py`). LoRAs
`output/ckpt_moss/caststation_{rows_trigger,preserve10}.safetensors`.

Reads: `src/lora_use.py` (`/media/sorryhyun/new/qwen35_sae/lora_use/
caststation_{rows_trigger_20261030-0917,preserve10_20261030-0923}/`; baseline
`caststation_ema097_lr1e4_20261027-1626`) and `bench/moss_ab/lora_ab.py` (Base, the 3
PROMPTS × seeds 1 / 2, ER-SDE 50, CFG 3, Spectrum; `output/tests/
ab_caststation_{rows_trigger,preserve10}_50st/`). The no-LoRA column is
`ab_caststation_lay08p02_at_muon_50st` x0.0 (same prompts and settings). Both scripts apply a
`delta_rows` LoRA's mask at inference (arm A renders eager, the others compiled).

## Claims

1. **Arm A binds by construction and carries content, not the palette.** No trigger, no
   delta. Through the trigger rows come outfit, eyes and composition; the baseline's pale,
   desaturated palette does not. One render of six writes `caststation` as a logo.
2. **Arm A's trigger rows carry a context-dependent effect, not a key.** Δ keeps its norm
   under caption edits but turns (cos 0.26–0.29 at σ 0.3, baseline 0.75–0.82); the
   trigger's own gradient × input share is 3.8 %.
3. **Arm B shrinks the delta everywhere instead of binding it.** Without the trigger Δ
   keeps its full norm; on another artist's captions it is as large as on its own. The
   renders are close to no LoRA.
4. **The preservation term was strong, not weak.** The main loss (~0.25) is almost all
   the irreducible noise floor; the part a LoRA can move is of the order of Δ² (~1e-3),
   the size of the preservation MSE, so ×10 outweighs it. With no feature telling the two
   captions apart, the cheapest joint solution is a smaller Δ.

For the palette: it is the part the baseline writes as the prompt-independent shift on
the image rows, which a trigger-row mask removes. A stronger trigger (or keying on `@`,
which in the causal TE precedes the name and is the same for every artist, and which
`text_embedder.proj` damps, sae_fit.md) would bring more of what the trigger rows carry,
the string included. Untested candidates: a delta on the image rows gated on the
trigger's presence; a learned trigger vector on the trigger rows (name_rows.md approach
B) with the row-confined LoRA.

## 1–2. `lora_use.py`

8 training captions + 4 `hews_layout` captions, σ 0.3 / 0.7, Δ = v_lora − v_base.
Edits as in text_read_path.md §4; `no_artist` drops the trigger (the only artist).

| | baseline | arm A | arm B |
|---|---|---|---|
| ‖Δ‖/‖v‖, own captions | 5.4 / 5.7 % | 2.6 / 2.3 % | 3.0 / 2.2 % |
| ‖Δ‖/‖v‖, other artist | 6.0 / 6.4 % | 0 (mask) | 3.1 / 2.2 % |
| `no_artist` norm / cos | 1.00 / 0.82–0.94 | 0 (mask) | 1.00 / 0.43–0.61 |
| `empty` cos | 0.74 / 0.83 | 0 (mask) | 0.29 / 0.38 |
| `no_chars` cos | 0.81 / 0.91 | 0.29 / 0.62 | 0.42 / 0.54 |
| `artist_only` cos | 0.75 / 0.83 | 0.26 / 0.55 | 0.30 / 0.38 |
| trigger (artist) share of grad × input | 2.2 % | 3.8 % | 2.9 % |

- Arm A: every caption edit keeps ‖Δ‖ (0.96–1.06) and turns it. The trigger sits at rows
  27–32 after `characters` / `series` (causal TE) and mixes with the caption in the joint
  blocks; the top gradient tokens are still commas and delimiters.
- Arm B: removing the trigger turns Δ about as much as removing the characters (0.43 vs
  0.42 at σ 0.3). The trigger is not special; Δ became more caption-dependent overall.

## 3–4. Renders and the loss scale

By eye, against no LoRA:

- Baseline: paler teal, thinner lines, a grey ground on Neru s1, composition changes.
- Arm A: Neru s2 gets the baseline's black vest and striped tie; NL s1 a dark uniform and
  another pose; heavy-lidded dark eyes on Miku s1 and NL s1. Miku's hair stays no-LoRA
  aqua. NL s1 has a `caststation` logo top left, NL s2 maybe a plate on the bench.
- Arm B: Neru s1, NL s1, Miku s2 match no LoRA nearly pixel for pixel in layout and
  colour.

Main loss, mean of the first / last 40 steps: baseline 0.254 / 0.277, arm A 0.255 / 0.283,
arm B 0.255 / 0.283; no run moves it. Arm B's preservation MSE (before ×10) rose
3.8e-4 → 1.5e-3 over training, the size a ~3 % trigger-free Δ gives.
