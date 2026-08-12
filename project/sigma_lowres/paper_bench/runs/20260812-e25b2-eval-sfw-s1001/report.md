# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.5064 | 0.5711 | +0.0647 |
| combo | 0.6382 | 0.7058 | +0.0676 |
| rescond | 0.4132 | 0.5474 | +0.1342 |
| sigma768 | 0.3685 | 0.4383 | +0.0699 |
| rescond768 | 0.5362 | 0.6036 | +0.0674 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9467 / 0.8029 |
| native~rescond | 0.9414 / 0.8530 |
| native~sigma768 | 0.9525 / 0.8866 |
| native~rescond768 | 0.9615 / 0.9248 |
| combo~rescond | 0.9180 / 0.5964 |
| combo~sigma768 | 0.9456 / 0.8317 |
| combo~rescond768 | 0.9386 / 0.7767 |
| rescond~sigma768 | 0.9548 / 0.8765 |
| rescond~rescond768 | 0.9438 / 0.8240 |
| sigma768~rescond768 | 0.9328 / 0.8184 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.2460 | 0.5723 | +0.3263 |
| combo | 0.2533 | 0.6182 | +0.3649 |
| rescond | 0.1744 | 0.5257 | +0.3513 |
| sigma768 | 0.2761 | 0.7427 | +0.4666 |
| rescond768 | 0.2395 | 0.5507 | +0.3113 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9590 / 0.8350 |
| native~rescond | 0.9513 / 0.8758 |
| native~sigma768 | 0.9723 / 0.9321 |
| native~rescond768 | 0.9517 / 0.8766 |
| combo~rescond | 0.9551 / 0.9272 |
| combo~sigma768 | 0.9635 / 0.8847 |
| combo~rescond768 | 0.9399 / 0.8731 |
| rescond~sigma768 | 0.9566 / 0.9158 |
| rescond~rescond768 | 0.9561 / 0.9286 |
| sigma768~rescond768 | 0.9564 / 0.8970 |
