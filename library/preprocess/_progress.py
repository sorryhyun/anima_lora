"""Optional progress callback for the cache functions.

The cache loops live in ``library/`` and must run headless (daemon, tests,
embedding code), so they never create a progress bar themselves. A caller that
*does* want one passes a ``progress`` callback; the CLI wrappers pass
:func:`tqdm_progress`. The protocol:

    progress(advance, total=N, detail="…")

called once up front with ``total`` to size the bar, then once per processed
item with ``advance=1`` (and an optional ``detail`` postfix). Cache functions
guard on ``progress is None``, so omitting it is a clean no-op.
"""

from __future__ import annotations

from typing import Callable, Optional

ProgressFn = Callable[..., None]


def tqdm_progress(desc: str) -> ProgressFn:
    """Return a :data:`ProgressFn` that drives a lazily-created ``tqdm`` bar.

    The bar is created on the first call that supplies ``total`` so the
    function controls when (and at what size) the bar appears.
    """
    from tqdm import tqdm

    state: dict[str, object] = {"bar": None}

    def cb(advance: int = 0, *, total: Optional[int] = None, detail: str = "") -> None:
        bar = state["bar"]
        if total is not None and bar is None:
            bar = state["bar"] = tqdm(total=total, desc=desc)
        if bar is None:
            return
        if detail:
            bar.set_postfix_str(detail)
        if advance:
            bar.update(advance)

    return cb
