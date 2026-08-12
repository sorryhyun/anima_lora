# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.3252 | 0.4169 | +0.0917 |
| combo | 0.5052 | 0.6411 | +0.1359 |
| rescond | 0.6599 | 0.7516 | +0.0917 |
| sigma768 | 0.4723 | 0.5780 | +0.1057 |
| rescond768 | 0.3716 | 0.4467 | +0.0751 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9689 / 0.9185 |
| native~rescond | 0.9518 / 0.9087 |
| native~sigma768 | 0.9596 / 0.8883 |
| native~rescond768 | 0.9581 / 0.9213 |
| combo~rescond | 0.9587 / 0.9041 |
| combo~sigma768 | 0.9679 / 0.9100 |
| combo~rescond768 | 0.9484 / 0.8195 |
| rescond~sigma768 | 0.9639 / 0.9021 |
| rescond~rescond768 | 0.9616 / 0.9069 |
| sigma768~rescond768 | 0.9485 / 0.8772 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.3607 | 1.0513 | +0.6906 |
| combo | 0.2210 | 0.6249 | +0.4039 |
| rescond | 0.2924 | 0.5050 | +0.2126 |
| sigma768 | 0.2513 | 0.6325 | +0.3812 |
| rescond768 | 0.3453 | 0.4478 | +0.1024 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9623 / 0.8628 |
| native~rescond | 0.9400 / 0.8699 |
| native~sigma768 | 0.9601 / 0.9005 |
| native~rescond768 | 0.9298 / 0.8529 |
| combo~rescond | 0.9383 / 0.8159 |
| combo~sigma768 | 0.9758 / 0.9349 |
| combo~rescond768 | 0.9424 / 0.7870 |
| rescond~sigma768 | 0.9311 / 0.7950 |
| rescond~rescond768 | 0.9647 / 0.9484 |
| sigma768~rescond768 | 0.9274 / 0.7637 |
