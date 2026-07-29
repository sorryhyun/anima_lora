# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.4427 | 0.4953 | +0.0526 |
| sigma896 | 0.4894 | 0.5685 | +0.0792 |
| unsafe768 | 0.4457 | 0.5493 | +0.1036 |
| 896only | 0.4327 | 0.5453 | +0.1125 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9532 / 0.9032 |
| native~unsafe768 | 0.9417 / 0.8635 |
| native~896only | 0.9312 / 0.7818 |
| sigma896~unsafe768 | 0.9395 / 0.8577 |
| sigma896~896only | 0.9534 / 0.8775 |
| unsafe768~896only | 0.9468 / 0.8695 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.1575 | 0.4002 | +0.2427 |
| sigma896 | 0.2186 | 0.4779 | +0.2593 |
| unsafe768 | 0.2160 | 0.4727 | +0.2567 |
| 896only | 0.2161 | 0.4377 | +0.2216 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9634 / 0.9216 |
| native~unsafe768 | 0.9692 / 0.9030 |
| native~896only | 0.9649 / 0.9390 |
| sigma896~unsafe768 | 0.9546 / 0.8752 |
| sigma896~896only | 0.9687 / 0.9147 |
| unsafe768~896only | 0.9606 / 0.9260 |
