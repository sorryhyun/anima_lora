# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.3397 | 0.4344 | +0.0947 |
| combo | 0.4412 | 0.5553 | +0.1141 |
| rescond_c | 0.4107 | 0.5015 | +0.0908 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9675 / 0.9082 |
| native~rescond_c | 0.9608 / 0.9046 |
| combo~rescond_c | 0.9671 / 0.9082 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.3607 | 1.0513 | +0.6906 |
| combo | 0.2210 | 0.6249 | +0.4039 |
| rescond_c | 0.2046 | 0.6248 | +0.4202 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9623 / 0.8628 |
| native~rescond_c | 0.9595 / 0.8698 |
| combo~rescond_c | 0.9661 / 0.9095 |
