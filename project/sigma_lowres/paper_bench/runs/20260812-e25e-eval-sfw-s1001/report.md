# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.4768 | 0.5490 | +0.0721 |
| combo | 0.6590 | 0.7391 | +0.0801 |
| rescond_c | 0.4749 | 0.5279 | +0.0529 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9408 / 0.7646 |
| native~rescond_c | 0.9554 / 0.8958 |
| combo~rescond_c | 0.9373 / 0.8044 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.2460 | 0.5723 | +0.3263 |
| combo | 0.2533 | 0.6182 | +0.3649 |
| rescond_c | 0.1822 | 0.5836 | +0.4015 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9590 / 0.8350 |
| native~rescond_c | 0.9600 / 0.8676 |
| combo~rescond_c | 0.9678 / 0.9332 |
