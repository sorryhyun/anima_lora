# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.4455 | 0.4944 | +0.0489 |
| combo | 0.4048 | 0.5113 | +0.1065 |
| rescond_c | 0.4658 | 0.5676 | +0.1018 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9554 / 0.8784 |
| native~rescond_c | 0.9599 / 0.9265 |
| combo~rescond_c | 0.9700 / 0.9475 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.1575 | 0.4002 | +0.2427 |
| combo | 0.2216 | 0.4662 | +0.2446 |
| rescond_c | 0.1751 | 0.4334 | +0.2583 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9735 / 0.9465 |
| native~rescond_c | 0.9767 / 0.9585 |
| combo~rescond_c | 0.9710 / 0.9326 |
