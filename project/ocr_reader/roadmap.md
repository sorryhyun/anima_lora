# ocr_reader — roadmap

## P0 — gates first

No KO/ZH claim can be made until this phase exists.

- **K3**: hand-label a KO and a ZH set in the sincos schema. Source the
  crops from plan2 K1's mined `kscreen` hits (screened by hayai, read by
  stock), not from a B′ screen. Hold the set out of every training manifest.
- Score stock VL-1.6, v3 (`vl16_b2_norm4`) and `vl16_pl_kozh` on K3. This
  turns "KO/ZH got worse" into a number, and tells us whether the stock
  model's KO/ZH ability is the floor that v3 lost.
- Size: big enough that the binomial SE is below the gaps we expect to see
  (n ≈ 300 per language gives SE ≈ 2.9 points).

## P1 — data intake

- Every source must have a licence that allows a shipped reader (no NC). Note
  the licence in the manifest.
- Keep the labels' spacing (KO 띄어쓰기). Check the target spellings against
  the tokenizer before training (the `♥`/`♡` lesson).
- Write manifests in the format `finetune_vl16_lora.py --extra_manifest`
  already reads, so no trainer change is needed to try a source.
- Dedupe against the K3 / sincos / COO test crops.

## P2 — parametrization

- **P2a (eval only)**: cut the v3 tower ΔW down to rank k ∈ {16, 64, 128}
  per matrix and score sincos + COO with the v3 LM adapter. If k=16 holds
  about 310 / 617, a tower LoRA is enough. If k=128 drops, the tower needs
  full FT and P2b's tower half is dead.
- **P2b (one arm)**: tower LoRA (rank taken from P2a) plus LM decoder layers
  in full FT (embed/lm_head frozen), against the v3 recipe on the same data.
  Read sincos, COO speech (to catch damage to the language prior) and peak
  VRAM.

## P3 — v4

- One mixed-language run on P1 data with the P2 parametrization.
- Gate: K3 KO and ZH both rise above v3, **and** sincos / COO SFX / COO
  speech do not drop by more than 1 SE against v3.
- Ship through `anime_tools.ocr.sfx` (with the decode guard) as Hub v4.

## Kill / stop

- If P0 shows v3 ≈ stock on KO/ZH (so nothing was lost) and P1 brings no
  licence-clean KO/ZH data, then KO/ZH has nothing to train on. Stop at P2.
- If P2b does not beat v3 by more than 1 SE on any gate, keep v3's
  parametrization.
