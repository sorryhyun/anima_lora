# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.4557 | 0.5120 | +0.0563 |
| sigma896 | 0.5187 | 0.5972 | +0.0786 |
| sigma896late | 0.3735 | 0.5161 | +0.1426 |
| win768late | 0.4714 | 0.5865 | +0.1152 |
| combo | 0.5256 | 0.6379 | +0.1123 |
| 896only | 0.3819 | 0.4972 | +0.1153 |
| combolate | 0.4300 | 0.5420 | +0.1121 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9542 / 0.9032 |
| native~sigma896late | 0.9532 / 0.8627 |
| native~win768late | 0.9650 / 0.9162 |
| native~combo | 0.9577 / 0.9025 |
| native~896only | 0.9369 / 0.7891 |
| native~combolate | 0.9480 / 0.8841 |
| sigma896~sigma896late | 0.9375 / 0.8793 |
| sigma896~win768late | 0.9517 / 0.9081 |
| sigma896~combo | 0.9594 / 0.9232 |
| sigma896~896only | 0.9606 / 0.9393 |
| sigma896~combolate | 0.9346 / 0.8893 |
| sigma896late~win768late | 0.9593 / 0.9074 |
| sigma896late~combo | 0.9600 / 0.9200 |
| sigma896late~896only | 0.9543 / 0.9081 |
| sigma896late~combolate | 0.9656 / 0.9364 |
| win768late~combo | 0.9644 / 0.9009 |
| win768late~896only | 0.9413 / 0.8370 |
| win768late~combolate | 0.9634 / 0.9355 |
| combo~896only | 0.9535 / 0.8647 |
| combo~combolate | 0.9573 / 0.9201 |
| 896only~combolate | 0.9319 / 0.8755 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.1575 | 0.4002 | +0.2427 |
| sigma896 | 0.2186 | 0.4779 | +0.2593 |
| sigma896late | 0.2991 | 0.4866 | +0.1875 |
| win768late | 0.2502 | 0.4308 | +0.1806 |
| combo | 0.3175 | 0.6348 | +0.3173 |
| 896only | 0.2161 | 0.4377 | +0.2216 |
| combolate | 0.2983 | 0.4554 | +0.1571 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9634 / 0.9216 |
| native~sigma896late | 0.9713 / 0.9483 |
| native~win768late | 0.9747 / 0.9194 |
| native~combo | 0.9627 / 0.9141 |
| native~896only | 0.9649 / 0.9390 |
| native~combolate | 0.9791 / 0.9537 |
| sigma896~sigma896late | 0.9685 / 0.9164 |
| sigma896~win768late | 0.9650 / 0.9166 |
| sigma896~combo | 0.9580 / 0.8821 |
| sigma896~896only | 0.9687 / 0.9147 |
| sigma896~combolate | 0.9685 / 0.9070 |
| sigma896late~win768late | 0.9628 / 0.8895 |
| sigma896late~combo | 0.9574 / 0.8367 |
| sigma896late~896only | 0.9698 / 0.9315 |
| sigma896late~combolate | 0.9807 / 0.9457 |
| win768late~combo | 0.9547 / 0.9166 |
| win768late~896only | 0.9657 / 0.9284 |
| win768late~combolate | 0.9725 / 0.9124 |
| combo~896only | 0.9561 / 0.9046 |
| combo~combolate | 0.9698 / 0.9360 |
| 896only~combolate | 0.9676 / 0.9349 |
