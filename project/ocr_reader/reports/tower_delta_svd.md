# Tower ΔW spectrum — v3 (`vl16_b2_norm4`) vs stock VL-1.6 (2026-10-10)

`probes/tower_delta_svd.py <tower.safetensors> <base model.safetensors> <out.json>`
runs on CPU in a few minutes. ΔW = fine-tuned tower − stock, per 2D weight;
the embeddings are skipped. The original B′ (`vl16_tower_lr1e-5`) tower was
pruned, so this measures v3, which uses the same recipe (tower lr 1e-5, 1 ep,
LM LoRA r16).

Median over the 27 layers. In "top-k" cells the first number is the energy
fraction in the top k singular values; the number after the slash is the same
fraction for an i.i.d. Gaussian matrix of the same shape.

| matrix | shape | ‖ΔW‖/‖W‖ | top-8 | top-16 | top-32 | top-64 | rank @ 90 % | eff. rank |
|---|---|---|---|---|---|---|---|---|
| q_proj | 1152² | 2.2e-2 | 0.39/0.03 | 0.52/0.05 | 0.66/0.10 | 0.79/0.19 | 137 | 91 |
| k_proj | 1152² | 2.1e-2 | 0.36/0.03 | 0.51/0.05 | 0.65/0.10 | 0.79/0.19 | 144 | 102 |
| v_proj | 1152² | 1.8e-2 | 0.29/0.03 | 0.41/0.05 | 0.57/0.10 | 0.73/0.19 | 172 | 138 |
| out_proj | 1152² | 2.0e-2 | 0.26/0.03 | 0.40/0.05 | 0.56/0.10 | 0.73/0.19 | 166 | 144 |
| mlp.fc1 | 4304×1152 | 1.9e-2 | 0.30/0.02 | 0.42/0.03 | 0.57/0.06 | 0.69/0.12 | 272 | 138 |
| mlp.fc2 | 1152×4304 | 1.8e-2 | 0.21/0.02 | 0.31/0.03 | 0.44/0.06 | 0.58/0.12 | 351 | 250 |
| projector linear_1 | 4608² | 1.9e-2 | 0.24/0.01 | 0.35/0.01 | 0.50/0.03 | 0.67/0.05 | 234 | 195 |
| projector linear_2 | 1024×4608 | 1.7e-2 | 0.41/0.02 | 0.52/0.03 | 0.65/0.06 | 0.78/0.12 | 150 | 92 |

- bf16 storage rounding accounts for about 1 % of ‖ΔW‖², so the spectrum is
  not quantization noise. 85 % of all tower elements changed.
- **The update has structure but is not low-rank.** The top 16 hold about
  10× the energy a random matrix would. Still, an r16 LoRA could express
  only 30–50 % of the full-FT update's energy, and the MLP is the most spread.
- **This does not decide the question.** Adam full FT leaves a high-rank tail
  that may not matter to the function. The functional test is P2a: truncate
  each ΔW to rank k, re-score, and read the gates.

Parameter budget, from the base checkpoint, for P2b sizing:

| part | params |
|---|---|
| tower | 466 M |
| projector | 26 M |
| LM decoder layers | 255 M |
| embed + lm_head | 212 M |

v3 trains 492 M in full at 12.1 GB. An LM-layers full FT trains about half
that.
