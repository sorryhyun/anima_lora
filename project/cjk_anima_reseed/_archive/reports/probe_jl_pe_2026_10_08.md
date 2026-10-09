# probe_jl, the page in PE's terms (2026-10-07/08)

`idea2.md` § The page in PE's terms. Probe 0 (`probe_jl_2026_10_07.md`)
read the page on the prediction's latent cells; here the page side is
PE-Spatial's out-box patch tokens z of the one-step x̂₀ = x_σ − σ·v
decoded: Gaussian probes on z, u = ∂z/∂x̂₀ᵀ v through the VAE decoder and
PE, then the DiT's Jᵀ(−σ u) to the rows → M_pe, in the "out" slot of the
same analysis (`read` / `check --page pe`). The box side stays M_in on
the cells.

## The blur gate (`probe_jl.py blur`, pres's rows, 48 items, forward only)

PE's out-box tokens of x̂₀ against the clean latent decoded and its
Gaussian blurs; top-1 is the item's own clean page among the 48 (chance
0.02):

| | cos tokens | top-1 CLS / tokens | tokens nearest |
|---|---|---|---|
| σ 0.5–0.7 | 0.78–0.88 | 0.94–1.00 | a 2–4 px blur |
| σ 0.8 | 0.70 | 0.81 / 0.77 | an 8 px blur |
| σ 0.9 | 0.54 | 0.46 / 0.48 | past a 16 px blur |
| σ 0.95 | 0.41 | 0.13 / 0.19 | — |

PE still reads the scene at σ 0.8, half of it at 0.9. CLS is no page
signal (0.93–0.99 at every σ, the clean pages 0.94 among themselves). At
0.9 x̂₀ is faded and dithered as well as blurred. Passed at 0.8,
borderline at 0.9; σ 0.95 dropped.

## Is u a stable quantity (`probe_jl.py pecheck`, 6 items)

u for one probe, on the out-box cells, fp32:

- **fp32 repeats exactly**; the leg's bf16 trouble is the VAE's (VAE in
  bf16: cos 0.06–0.34 to fp32; PE in bf16: 0.95–0.99). The fit runs the
  leg in fp32.
- **White-noise δ on x̂₀** (|δ| / |x̂₀|): at 1e-3 u's cos 0.71–0.77 (σ
  0.5–0.6) and 0.22–0.56 (0.8–0.9), at 1e-2 −0.02–0.29, while z moves only
  1–11 % at 1e-3 — the function smooth, its gradient not.
- **Low-pass does not save it.** 85–91 % of |u|² is gone after a 2×2
  pool, and at σ 0.8–0.9 the pooled u (up to 8×8) still turns: cos
  0.40–0.71 at 1e-3, no gain at 0.9.
- **Along the rows' own moves it holds locally.** x̂₀ at the rows + ε ·
  Δ(pres − f0) or ε · (f0 − start): a whole trained move changes x̂₀ by
  0.4–2.8 %; at 1e-3 (1/10–1/20 of a move) u's cos is 0.72–0.94 (page
  move) and 0.90–0.96 (text move), at a whole move's 1e-2 0.24–0.72.
  Pooling makes no difference there. White noise was the off-manifold
  worst case; the lens is local, not broken. Fit run unpooled.

## The fit (`fit --side pe --label start_pe`)

f0's start rows (stick080), `sent_kanji`'s items in the trainer's order,
σ 0.5–0.7 / 0.8 / 0.9 × A / B, 100 items each (600), 8 probes per mask —
in / out on the cells, drawn as probe 0 drew them, and on z from a
generator of their own. The page leg runs first for every item (the fp32
DiT to the CPU while the VAE and PE take u; together they ran out at
13.6 GiB on a 448×640 item, job `20261007-232807-7f9025`), then the DiT's
forward and backwards as probe 0; each item's x̂₀ asserted equal between
the two passes. Job `20261007-233230-e8d5a6`, 58 min (5.8 s / item, page
leg 2 s) → `output/cjk_anima_reseed/probe_jl/start_pe/` (`g.pt`;
`read{,_pair}_pe.json`, `check_pair_pe.json`). The in / out slots match
`start/g.pt` on the same steps (same items, σ, rows by ext id; G to
0–1e-4), so the cell numbers below are probe 0's.

Pair-weighted (none in brackets where it differs in kind):

| | M_out (cells) | M_pe | bar |
|---|---|---|---|
| A / B overlap @16, kana σ 0.5–0.7 / 0.8 / 0.9 | 0.53 / 0.69 / 0.76 | 0.50 / 0.57 / 0.58 (0.29 / 0.30 / 0.36) | 0.8 |
| … kanji | 0.30 / 0.44 / 0.48 | 0.27 / 0.32 / 0.40 | 0.8 |
| cross λ / r top-1 / top-16, kana σ 0.5–0.7 | 11.3 / 3.6 | 11.1 / 3.5 | 2 |
| … kana σ 0.8 | 3.5 / 1.9 | 3.8 / 2.0 | 2 |
| … kana σ 0.9 | 1.8 / 1.2 | 1.4 / 1.3 | 2 |
| … kanji σ 0.8 | 1.6 / 1.3 | 2.0 / 1.45 | 2 |
| high-λ subspace A / B @16, σ 0.8–0.9 | 0.02–0.04 | 0.02–0.04 | isotropic 0.016 |

- **Less stable than the cells**, by 0.1–0.2 at σ 0.8–0.9; it climbs with
  the sample (σ 0.8 kana 0.48 at 50 items, 0.57 at 100), far from 0.8.
- **No text-only subspace either way.** The cross-fit λ sits where the
  cells' did: σ 0.8 kana at the 2 r line, σ 0.9 none.
- **The high-λ subspace does not repeat** — at the isotropic floor, as on
  the cells. The projection arm has nothing to project onto.
- **`check --page pe`** (pres − f0 per string, n 96): the move as M_pe
  sees it against en_match ρ +0.22, iou_en +0.35 — the move's plain Σ|x|²
  gives +0.22 / +0.34, the ratio negative and n.s. As the cells.
- **What it does see:** Δ(pres − f0)'s shared vector puts 0.45 of its
  energy in M_pe's top 16 at σ 0.8 kana (28× isotropic; 15–20× on
  M_out) — L_pres moved the rows along what PE's page sees most.

## Verdict

The PE page lens stops by probe 0's rules and adds nothing over the
cells: at σ 0.8–0.9 the rows move the box and the page along the same
directions whether the page is read in cells or in PE's tokens. Both of
idea2's page lenses end before the projection arm. Not tested: a
PE-on-the-EN-render teacher (L_pres's PE twin in `idea2.md`) — a training
term, not a lens.
