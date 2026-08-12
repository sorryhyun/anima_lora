# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.4274 | 0.4814 | +0.0540 |
| combo | 0.3917 | 0.5056 | +0.1138 |
| rescond | 0.5502 | 0.6759 | +0.1258 |
| sigma768 | 0.4102 | 0.5511 | +0.1409 |
| rescond768 | 0.5345 | 0.6561 | +0.1216 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9568 / 0.8662 |
| native~rescond | 0.9419 / 0.8823 |
| native~sigma768 | 0.9474 / 0.8263 |
| native~rescond768 | 0.9530 / 0.9145 |
| combo~rescond | 0.9635 / 0.9275 |
| combo~sigma768 | 0.9710 / 0.9405 |
| combo~rescond768 | 0.9656 / 0.9392 |
| rescond~sigma768 | 0.9610 / 0.9311 |
| rescond~rescond768 | 0.9757 / 0.9616 |
| sigma768~rescond768 | 0.9607 / 0.9229 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.1575 | 0.4002 | +0.2427 |
| combo | 0.2216 | 0.4662 | +0.2446 |
| rescond | 0.2761 | 0.4311 | +0.1550 |
| sigma768 | 0.3569 | 0.7292 | +0.3723 |
| rescond768 | 0.3011 | 0.5373 | +0.2362 |

| pair | paired cos (mean / min) |
|---|---|
| native~combo | 0.9735 / 0.9465 |
| native~rescond | 0.9633 / 0.9175 |
| native~sigma768 | 0.9697 / 0.9261 |
| native~rescond768 | 0.9638 / 0.9090 |
| combo~rescond | 0.9539 / 0.8794 |
| combo~sigma768 | 0.9787 / 0.9587 |
| combo~rescond768 | 0.9483 / 0.8486 |
| rescond~sigma768 | 0.9442 / 0.8375 |
| rescond~rescond768 | 0.9681 / 0.9152 |
| sigma768~rescond768 | 0.9420 / 0.8495 |
