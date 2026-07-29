# E4 seed-noise yardstick

## hews

| within-seed arm pair | mean cos |
|---|---|
| native~896only|s1001 | 0.9666 |
| native~896only|s1002 | 0.9533 |
| native~896only|s1003 | 0.9312 |
| native~sigma896|s1001 | 0.9616 |
| native~sigma896|s1002 | 0.9506 |
| native~sigma896|s1003 | 0.9532 |
| native~unsafe768|s1001 | 0.9510 |
| native~unsafe768|s1002 | 0.9652 |
| native~unsafe768|s1003 | 0.9417 |
| sigma896~896only|s1001 | 0.9769 |
| sigma896~896only|s1002 | 0.9570 |
| sigma896~896only|s1003 | 0.9534 |
| sigma896~unsafe768|s1001 | 0.9512 |
| sigma896~unsafe768|s1002 | 0.9573 |
| sigma896~unsafe768|s1003 | 0.9395 |
| unsafe768~896only|s1001 | 0.9597 |
| unsafe768~896only|s1002 | 0.9624 |
| unsafe768~896only|s1003 | 0.9468 |

| cross-seed same arm | mean cos |
|---|---|
| 896only|s1001~s1002 | 0.9651 |
| 896only|s1001~s1003 | 0.9441 |
| 896only|s1002~s1003 | 0.9374 |
| native|s1001~s1002 | 0.9577 |
| native|s1001~s1003 | 0.9592 |
| native|s1002~s1003 | 0.9505 |
| sigma896|s1001~s1002 | 0.9638 |
| sigma896|s1001~s1003 | 0.9526 |
| sigma896|s1002~s1003 | 0.9419 |
| unsafe768|s1001~s1002 | 0.9596 |
| unsafe768|s1001~s1003 | 0.9443 |
| unsafe768|s1002~s1003 | 0.9630 |

**Verdict**: arm effect 0.9551 vs seed yardstick 0.9558 → **outside the seed lottery** (real footprint — quality case must rest on distribution metrics + eyeball)

## channel_(caststation)

| within-seed arm pair | mean cos |
|---|---|
| native~896only|s1001 | 0.9367 |
| native~896only|s1002 | 0.9483 |
| native~896only|s1003 | 0.9649 |
| native~sigma896|s1001 | 0.9667 |
| native~sigma896|s1002 | 0.9622 |
| native~sigma896|s1003 | 0.9634 |
| native~unsafe768|s1001 | 0.9471 |
| native~unsafe768|s1002 | 0.9472 |
| native~unsafe768|s1003 | 0.9692 |
| sigma896~896only|s1001 | 0.9529 |
| sigma896~896only|s1002 | 0.9619 |
| sigma896~896only|s1003 | 0.9687 |
| sigma896~unsafe768|s1001 | 0.9551 |
| sigma896~unsafe768|s1002 | 0.9550 |
| sigma896~unsafe768|s1003 | 0.9546 |
| unsafe768~896only|s1001 | 0.9422 |
| unsafe768~896only|s1002 | 0.9668 |
| unsafe768~896only|s1003 | 0.9606 |

| cross-seed same arm | mean cos |
|---|---|
| 896only|s1001~s1002 | 0.9581 |
| 896only|s1001~s1003 | 0.9346 |
| 896only|s1002~s1003 | 0.9615 |
| native|s1001~s1002 | 0.9454 |
| native|s1001~s1003 | 0.9639 |
| native|s1002~s1003 | 0.9531 |
| sigma896|s1001~s1002 | 0.9649 |
| sigma896|s1001~s1003 | 0.9641 |
| sigma896|s1002~s1003 | 0.9580 |
| unsafe768|s1001~s1002 | 0.9588 |
| unsafe768|s1001~s1003 | 0.9564 |
| unsafe768|s1002~s1003 | 0.9705 |

**Verdict**: arm effect 0.9641 vs seed yardstick 0.9541 → **inside the seed lottery** (defensible)
