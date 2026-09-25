"""Canvas shapes: ``(W, H)`` pairs and the ``--shapes`` weighted list."""

from __future__ import annotations


def wh(size) -> tuple[int, int]:
    """Canvas ``(W, H)`` from an int side or a ``(W, H)`` pair."""
    if isinstance(size, int):
        return size, size
    W, H = size
    return int(W), int(H)


def parse_shape(tok: str) -> tuple[int, int]:
    """``'512'`` → (512, 512); ``'384x512'`` → (384, 512) as (W, H)."""
    tok = tok.strip().lower()
    if "x" in tok:
        W, H = tok.split("x")
        return int(W), int(H)
    return int(tok), int(tok)


def parse_shapes(spec: str) -> list[tuple[int, int, float]]:
    """``--shapes`` → ``[(W, H, weight)]``. ``'384,448,512:2,384x512'`` draws
    512² twice as often as each other entry. Sides must be multiples of 16
    (VAE 8× and a 2-patch), so every entry is one static token family."""
    out = []
    for tok in spec.split(","):
        if not tok.strip():
            continue
        shp, _, w = tok.partition(":")
        W, H = parse_shape(shp)
        assert W % 16 == 0 and H % 16 == 0, (
            f"shape {tok}: sides must be multiples of 16"
        )
        out.append((W, H, float(w) if w else 1.0))
    return out
