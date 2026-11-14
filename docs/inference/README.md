# Inference methods shared across lines

Training-free sampler methods that run on both Anima and MOSS-Image. Each doc covers Anima
first and has a MOSS-Image section. Anima-only stacks are in
[`../anima/inference/`](../anima/inference/README.md).

| Doc | What it is | Anima | MOSS-Image |
|---|---|---|---|
| [spectrum.md](spectrum.md) | Chebyshev feature forecasting: most steps skip the blocks and run only the head. | `--spectrum` / `make test SPECTRUM=1` | `generate.py` default on (2.6× at 50 steps) |
| [cns.md](cns.md) | Colored-noise ER-SDE: the injected noise goes to the frequency bands not yet resolved, RMS-renormalized. | `--sampler er_sde --cns auto` | γ files for the bench and the ComfyUI node; no `generate.py` flag |
| [dave.md](dave.md) | Same-prompt seed variety: attenuates the cross-seed DC of block features in the first steps. | `--dave auto` / `make test DAVE=1`, blocks 8–18 | `--dave auto`, blocks 19–30, image tokens only |
