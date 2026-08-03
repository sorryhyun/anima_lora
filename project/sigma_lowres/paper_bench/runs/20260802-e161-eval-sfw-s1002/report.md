# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.3541 | 0.4516 | +0.0975 |
| sigma896 | 0.5146 | 0.6338 | +0.1192 |
| sigma896late | 0.4668 | 0.5649 | +0.0981 |
| win768late | 0.5429 | 0.6261 | +0.0832 |
| combo | 0.4750 | 0.5745 | +0.0994 |
| 896only | 0.5013 | 0.5678 | +0.0665 |
| combolate | 0.6195 | 0.6896 | +0.0701 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9487 / 0.9107 |
| native~sigma896late | 0.9592 / 0.9350 |
| native~win768late | 0.9642 / 0.8994 |
| native~combo | 0.9526 / 0.9260 |
| native~896only | 0.9499 / 0.9332 |
| native~combolate | 0.9481 / 0.9154 |
| sigma896~sigma896late | 0.9565 / 0.8534 |
| sigma896~win768late | 0.9424 / 0.8705 |
| sigma896~combo | 0.9439 / 0.8752 |
| sigma896~896only | 0.9562 / 0.8835 |
| sigma896~combolate | 0.9600 / 0.8834 |
| sigma896late~win768late | 0.9681 / 0.9082 |
| sigma896late~combo | 0.9466 / 0.8676 |
| sigma896late~896only | 0.9532 / 0.8821 |
| sigma896late~combolate | 0.9773 / 0.9582 |
| win768late~combo | 0.9437 / 0.8630 |
| win768late~896only | 0.9437 / 0.8836 |
| win768late~combolate | 0.9685 / 0.9523 |
| combo~896only | 0.9529 / 0.8832 |
| combo~combolate | 0.9461 / 0.8883 |
| 896only~combolate | 0.9506 / 0.8759 |

## channel_(caststation)

- prompts 12 · refs 15/15 (holdout/member) · noise floor **1.6209**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.3607 | 1.0513 | +0.6906 |
| sigma896 | 0.3102 | 0.7540 | +0.4438 |
| sigma896late | 0.2902 | 0.7058 | +0.4157 |
| win768late | 0.2351 | 0.6759 | +0.4408 |
| combo | 0.2805 | 0.6989 | +0.4184 |
| 896only | 0.3533 | 0.6362 | +0.2829 |
| combolate | 0.1826 | 0.5848 | +0.4022 |

| pair | paired cos (mean / min) |
|---|---|
| native~sigma896 | 0.9622 / 0.8751 |
| native~sigma896late | 0.9565 / 0.8612 |
| native~win768late | 0.9647 / 0.9008 |
| native~combo | 0.9565 / 0.8775 |
| native~896only | 0.9483 / 0.8156 |
| native~combolate | 0.9583 / 0.8335 |
| sigma896~sigma896late | 0.9587 / 0.8650 |
| sigma896~win768late | 0.9571 / 0.8019 |
| sigma896~combo | 0.9530 / 0.8344 |
| sigma896~896only | 0.9619 / 0.8386 |
| sigma896~combolate | 0.9613 / 0.8365 |
| sigma896late~win768late | 0.9695 / 0.9036 |
| sigma896late~combo | 0.9516 / 0.8678 |
| sigma896late~896only | 0.9540 / 0.8331 |
| sigma896late~combolate | 0.9668 / 0.9043 |
| win768late~combo | 0.9578 / 0.8685 |
| win768late~896only | 0.9518 / 0.8695 |
| win768late~combolate | 0.9714 / 0.9449 |
| combo~896only | 0.9652 / 0.9071 |
| combo~combolate | 0.9567 / 0.8420 |
| 896only~combolate | 0.9569 / 0.8702 |
