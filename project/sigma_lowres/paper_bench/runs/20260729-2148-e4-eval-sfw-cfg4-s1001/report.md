# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 1.3487 | 1.3893 | +0.0405 |
| sigma896 | 1.2012 | 1.2705 | +0.0694 |
| unsafe768 | 1.4765 | 1.5533 | +0.0768 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9774 / 0.9302 |
| native~unsafe768 | 0.9803 / 0.9613 |
| sigma896~unsafe768 | 0.9728 / 0.9375 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.5301 | 0.6866 | +0.1565 |
| sigma896 | 0.4050 | 0.5318 | +0.1268 |
| unsafe768 | 0.5426 | 0.6253 | +0.0826 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9752 / 0.9406 |
| native~unsafe768 | 0.9710 / 0.9179 |
| sigma896~unsafe768 | 0.9814 / 0.9599 |
