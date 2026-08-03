# E4 eval (exercise pass)

## hews

- prompts 9 · refs 61/60 (holdout/member) · noise floor **0.4288**

| arm | cmmd_holdout ↓ | cmmd_member | Δ(member−holdout) |
|---|---|---|---|
| native | 0.5070 | 0.5807 | +0.0737 |
| early | 0.4786 | 0.5993 | +0.1206 |
| late | 0.5718 | 0.6381 | +0.0663 |
| spread | 0.4039 | 0.5305 | +0.1266 |
| win768 | 0.6282 | 0.7247 | +0.0964 |
| win768late | 0.5344 | 0.6067 | +0.0722 |
| sigma896late | 0.4200 | 0.5100 | +0.0900 |

| pair | paired cos (mean / min) |
|---|---|
| native~early | 0.9606 / 0.9313 |
| native~late | 0.9580 / 0.8854 |
| native~spread | 0.9577 / 0.9193 |
| native~win768 | 0.9605 / 0.9102 |
| native~win768late | 0.9755 / 0.9346 |
| native~sigma896late | 0.9429 / 0.8584 |
| early~late | 0.9476 / 0.8933 |
| early~spread | 0.9524 / 0.8959 |
| early~win768 | 0.9594 / 0.9157 |
| early~win768late | 0.9543 / 0.9098 |
| early~sigma896late | 0.9473 / 0.8988 |
| late~spread | 0.9515 / 0.8803 |
| late~win768 | 0.9656 / 0.9312 |
| late~win768late | 0.9724 / 0.9527 |
| late~sigma896late | 0.9632 / 0.9139 |
| spread~win768 | 0.9400 / 0.8693 |
| spread~win768late | 0.9500 / 0.9257 |
| spread~sigma896late | 0.9490 / 0.8870 |
| win768~win768late | 0.9682 / 0.9365 |
| win768~sigma896late | 0.9528 / 0.8939 |
| win768late~sigma896late | 0.9539 / 0.8837 |
