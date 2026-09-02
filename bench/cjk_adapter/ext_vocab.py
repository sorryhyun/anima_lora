"""Re-export shim — the module was promoted to ``library.anima.ext_vocab``.

Kept so the bench/gate/dataset scripts' ``from bench.cjk_adapter import
ext_vocab`` imports keep working; edit the canonical module, not this file.
"""

from library.anima.ext_vocab import (  # noqa: F401
    _CJK_RANGES,
    T5_EOS_ID,
    T5_PAD_ID,
    T5_TABLE_SIZE,
    T5_UNK_ID,
    MAP_METHODS,
    HybridT5Encoder,
    build_anchor_pairs,
    build_ext_table,
    char_row_surfaces,
    collect_clean_qwen_tokens,
    fit_anchor_map,
    is_cjk_char,
    is_hangul_char,
    load_ext_assets,
    segment_runs,
)
