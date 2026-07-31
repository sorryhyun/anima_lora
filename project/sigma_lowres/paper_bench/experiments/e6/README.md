# E6 [STRETCH] — one generalization arm each

| | |
|---|---|
| **Status** | **OPEN** — stretch; raises the acceptance ceiling, not required for correctness |
| **Why it exists** | Review triage: the 2-model × 2-adapter × 2-domain generalization matrix is the right ask for a strong venue but is not what makes the current claims true or false. One extra DiT + one full-FT probe arm is the 80/20. |
| **Partly discharged by** | [E7](../e7/) — its two controlled adapters cover the "second LoRA checkpoint on Anima" leg |

- **One extra DiT** (any open flow-matching DiT with a different VAE),
  endpoint + 3-bin grid, routes {×0.875, ×0.5}, N=12: turns the case
  study into a phenomenon.
- **Full-FT probe** (all-param grads, N small, grad-ckpt): does the floor
  live in LoRA geometry or the model? One run answers it.
- ~~Second LoRA checkpoint on Anima (different corpus)~~ — near-free,
  uses the existing instrument verbatim; **largely discharged by
  [E7](../e7/)'s two controlled adapters**.
