# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.5994 | 0.6710 | +0.0716 |
| sigma896 | 0.5540 | 0.6646 | +0.1106 |
| sigma896late | 0.3924 | 0.4761 | +0.0837 |
| win768late | 0.5486 | 0.6288 | +0.0802 |
| combo | 0.5580 | 0.6448 | +0.0868 |
| 896only | 0.5671 | 0.6751 | +0.1080 |
| combolate | 0.4170 | 0.4852 | +0.0682 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9586 / 0.9204 |
| native~sigma896late | 0.9482 / 0.8512 |
| native~win768late | 0.9743 / 0.9440 |
| native~combo | 0.9624 / 0.9253 |
| native~896only | 0.9614 / 0.9378 |
| native~combolate | 0.9421 / 0.8814 |
| sigma896~sigma896late | 0.9585 / 0.9421 |
| sigma896~win768late | 0.9647 / 0.9348 |
| sigma896~combo | 0.9603 / 0.9054 |
| sigma896~896only | 0.9767 / 0.9466 |
| sigma896~combolate | 0.9500 / 0.9002 |
| sigma896late~win768late | 0.9563 / 0.8786 |
| sigma896late~combo | 0.9537 / 0.9060 |
| sigma896late~896only | 0.9474 / 0.8721 |
| sigma896late~combolate | 0.9795 / 0.9650 |
| win768late~combo | 0.9531 / 0.9281 |
| win768late~896only | 0.9597 / 0.9114 |
| win768late~combolate | 0.9507 / 0.8921 |
| combo~896only | 0.9708 / 0.9547 |
| combo~combolate | 0.9487 / 0.8719 |
| 896only~combolate | 0.9427 / 0.8696 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.2460 | 0.5723 | +0.3263 |
| sigma896 | 0.3283 | 0.6628 | +0.3345 |
| sigma896late | 0.2725 | 0.8245 | +0.5519 |
| win768late | 0.3009 | 0.7607 | +0.4598 |
| combo | 0.1695 | 0.5026 | +0.3331 |
| 896only | 0.3332 | 0.6933 | +0.3601 |
| combolate | 0.2228 | 0.7471 | +0.5243 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9667 / 0.8647 |
| native~sigma896late | 0.9629 / 0.8684 |
| native~win768late | 0.9790 / 0.9554 |
| native~combo | 0.9547 / 0.8392 |
| native~896only | 0.9367 / 0.7638 |
| native~combolate | 0.9618 / 0.8801 |
| sigma896~sigma896late | 0.9641 / 0.8279 |
| sigma896~win768late | 0.9660 / 0.9004 |
| sigma896~combo | 0.9550 / 0.8856 |
| sigma896~896only | 0.9529 / 0.8345 |
| sigma896~combolate | 0.9629 / 0.8402 |
| sigma896late~win768late | 0.9791 / 0.9575 |
| sigma896late~combo | 0.9559 / 0.8504 |
| sigma896late~896only | 0.9438 / 0.7983 |
| sigma896late~combolate | 0.9818 / 0.9507 |
| win768late~combo | 0.9617 / 0.9151 |
| win768late~896only | 0.9399 / 0.7167 |
| win768late~combolate | 0.9756 / 0.9478 |
| combo~896only | 0.9476 / 0.7643 |
| combo~combolate | 0.9618 / 0.9096 |
| 896only~combolate | 0.9417 / 0.7856 |
