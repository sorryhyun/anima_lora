# Issue: check each released FA2 sm_120 wheel on a real training step

**MOSS: resolved 2026-11-04.** precast was the regression, not the PV forward. MOSS
`configs/moss/lora.toml` on `hews_layout`, forward + backward summed over steps 20–40:

| wheel | PV forward | total | peak |
|---|---|---|---|
| mjun0812 stock (two runs) | – | 22.92 / 22.83 s | 12.85 GB |
| fp16acc | – | 21.57 s (−5.9%) | 12.85 GB |
| fwdpv, per-torch | on / off | 21.38 / 21.61 s | 12.85 GB |
| fwdpv, stable ABI | on | 21.42 s (−6.5%) | 12.85 GB |
| precast | on / off | 22.51 / 21.56 s | 13.01 / 12.85 GB |

With the PV forward on, precast's backward was back at stock and the peak 0.16 GB higher. The
precast release is deleted; the pin is now `v2.8.3-sm120-hdim64-128-fp16acc-fwdpv-cu132-torch2.12plus`
(fwdpv on the stable ABI, 21.44 s on the released wheel). Write-up: `precast.md` in
sorryhyun/flash-attention-fp16.

Open: the same check on Anima, and on generation (Spectrum is on by default, so a step-window
timer needs `--no-spectrum`).
