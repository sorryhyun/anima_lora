"""Backwards-compat re-export — the real code lives in library.anima.text_strategies.

External scripts (``scripts/``, ``bench/``, ``train.py``, ``inference.py``) still
import ``library.strategy_base``; keeping this shim means we can relocate the
definitions without touching every caller.
"""

from library.anima.text_strategies import *  # noqa: F401, F403
from library.anima.text_strategies import (  # noqa: F401
    LatentsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
    TextEncoderOutputsCachingStrategy,
)
