# E4 seed-noise yardstick

## hews

| within-seed arm pair | mean cos |
|---|---|
| combo~896only|s1001 | 0.9708 |
| combo~896only|s1002 | 0.9529 |
| combo~896only|s1003 | 0.9535 |
| native~896only|s1001 | 0.9614 |
| native~896only|s1002 | 0.9499 |
| native~896only|s1003 | 0.9369 |
| native~combo|s1001 | 0.9624 |
| native~combo|s1002 | 0.9526 |
| native~combo|s1003 | 0.9577 |
| native~sigma896late|s1001 | 0.9482 |
| native~sigma896late|s1002 | 0.9592 |
| native~sigma896late|s1003 | 0.9532 |
| native~sigma896|s1001 | 0.9586 |
| native~sigma896|s1002 | 0.9487 |
| native~sigma896|s1003 | 0.9542 |
| native~win768late|s1001 | 0.9743 |
| native~win768late|s1002 | 0.9642 |
| native~win768late|s1003 | 0.9650 |
| sigma896late~896only|s1001 | 0.9474 |
| sigma896late~896only|s1002 | 0.9532 |
| sigma896late~896only|s1003 | 0.9543 |
| sigma896late~combo|s1001 | 0.9537 |
| sigma896late~combo|s1002 | 0.9466 |
| sigma896late~combo|s1003 | 0.9600 |
| sigma896late~win768late|s1001 | 0.9563 |
| sigma896late~win768late|s1002 | 0.9681 |
| sigma896late~win768late|s1003 | 0.9593 |
| sigma896~896only|s1001 | 0.9767 |
| sigma896~896only|s1002 | 0.9562 |
| sigma896~896only|s1003 | 0.9606 |
| sigma896~combo|s1001 | 0.9603 |
| sigma896~combo|s1002 | 0.9439 |
| sigma896~combo|s1003 | 0.9594 |
| sigma896~sigma896late|s1001 | 0.9585 |
| sigma896~sigma896late|s1002 | 0.9565 |
| sigma896~sigma896late|s1003 | 0.9375 |
| sigma896~win768late|s1001 | 0.9647 |
| sigma896~win768late|s1002 | 0.9424 |
| sigma896~win768late|s1003 | 0.9517 |
| win768late~896only|s1001 | 0.9597 |
| win768late~896only|s1002 | 0.9437 |
| win768late~896only|s1003 | 0.9413 |
| win768late~combo|s1001 | 0.9531 |
| win768late~combo|s1002 | 0.9437 |
| win768late~combo|s1003 | 0.9644 |

| cross-seed same arm | mean cos |
|---|---|
| 896only|s1001~s1002 | 0.9661 |
| 896only|s1001~s1003 | 0.9483 |
| 896only|s1002~s1003 | 0.9414 |
| combo|s1001~s1002 | 0.9443 |
| combo|s1001~s1003 | 0.9586 |
| combo|s1002~s1003 | 0.9527 |
| native|s1001~s1002 | 0.9537 |
| native|s1001~s1003 | 0.9611 |
| native|s1002~s1003 | 0.9493 |
| sigma896late|s1001~s1002 | 0.9540 |
| sigma896late|s1001~s1003 | 0.9663 |
| sigma896late|s1002~s1003 | 0.9500 |
| sigma896|s1001~s1002 | 0.9616 |
| sigma896|s1001~s1003 | 0.9573 |
| sigma896|s1002~s1003 | 0.9428 |
| win768late|s1001~s1002 | 0.9408 |
| win768late|s1001~s1003 | 0.9638 |
| win768late|s1002~s1003 | 0.9512 |

**Verdict**: arm effect 0.9537 vs seed yardstick 0.9547 → **outside the seed lottery** (real footprint — quality case must rest on distribution metrics + eyeball)

## channel_(caststation)

| within-seed arm pair | mean cos |
|---|---|
| combo~896only|s1001 | 0.9476 |
| combo~896only|s1002 | 0.9652 |
| combo~896only|s1003 | 0.9561 |
| native~896only|s1001 | 0.9367 |
| native~896only|s1002 | 0.9483 |
| native~896only|s1003 | 0.9649 |
| native~combo|s1001 | 0.9547 |
| native~combo|s1002 | 0.9565 |
| native~combo|s1003 | 0.9627 |
| native~sigma896late|s1001 | 0.9629 |
| native~sigma896late|s1002 | 0.9565 |
| native~sigma896late|s1003 | 0.9713 |
| native~sigma896|s1001 | 0.9667 |
| native~sigma896|s1002 | 0.9622 |
| native~sigma896|s1003 | 0.9634 |
| native~win768late|s1001 | 0.9790 |
| native~win768late|s1002 | 0.9647 |
| native~win768late|s1003 | 0.9747 |
| sigma896late~896only|s1001 | 0.9438 |
| sigma896late~896only|s1002 | 0.9540 |
| sigma896late~896only|s1003 | 0.9698 |
| sigma896late~combo|s1001 | 0.9559 |
| sigma896late~combo|s1002 | 0.9516 |
| sigma896late~combo|s1003 | 0.9574 |
| sigma896late~win768late|s1001 | 0.9791 |
| sigma896late~win768late|s1002 | 0.9695 |
| sigma896late~win768late|s1003 | 0.9628 |
| sigma896~896only|s1001 | 0.9529 |
| sigma896~896only|s1002 | 0.9619 |
| sigma896~896only|s1003 | 0.9687 |
| sigma896~combo|s1001 | 0.9550 |
| sigma896~combo|s1002 | 0.9530 |
| sigma896~combo|s1003 | 0.9580 |
| sigma896~sigma896late|s1001 | 0.9641 |
| sigma896~sigma896late|s1002 | 0.9587 |
| sigma896~sigma896late|s1003 | 0.9685 |
| sigma896~win768late|s1001 | 0.9660 |
| sigma896~win768late|s1002 | 0.9571 |
| sigma896~win768late|s1003 | 0.9650 |
| win768late~896only|s1001 | 0.9399 |
| win768late~896only|s1002 | 0.9518 |
| win768late~896only|s1003 | 0.9657 |
| win768late~combo|s1001 | 0.9617 |
| win768late~combo|s1002 | 0.9578 |
| win768late~combo|s1003 | 0.9547 |

| cross-seed same arm | mean cos |
|---|---|
| 896only|s1001~s1002 | 0.9581 |
| 896only|s1001~s1003 | 0.9346 |
| 896only|s1002~s1003 | 0.9615 |
| combo|s1001~s1002 | 0.9654 |
| combo|s1001~s1003 | 0.9513 |
| combo|s1002~s1003 | 0.9589 |
| native|s1001~s1002 | 0.9454 |
| native|s1001~s1003 | 0.9639 |
| native|s1002~s1003 | 0.9531 |
| sigma896late|s1001~s1002 | 0.9565 |
| sigma896late|s1001~s1003 | 0.9578 |
| sigma896late|s1002~s1003 | 0.9532 |
| sigma896|s1001~s1002 | 0.9649 |
| sigma896|s1001~s1003 | 0.9641 |
| sigma896|s1002~s1003 | 0.9580 |
| win768late|s1001~s1002 | 0.9612 |
| win768late|s1001~s1003 | 0.9633 |
| win768late|s1002~s1003 | 0.9520 |

**Verdict**: arm effect 0.9638 vs seed yardstick 0.9541 → **inside the seed lottery** (defensible)
