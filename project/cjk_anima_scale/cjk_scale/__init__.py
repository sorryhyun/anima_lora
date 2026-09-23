"""cjk_scale — the JA vocab pack at scale: one loss, one trainer, a σ-band
schedule whose stages differ only in their data (``../design.md``).

    paths      where runs land (``output/cjk_anima_scale/{data,rows}_scale_<stage>_<tag>``)
    windows    the vocab band law as code: (kind, px, layout) → band
    config     ``configs/<stage>.toml`` → StageConfig
    recipes    the item generators over the probe line's render primitives
    builder    stage config → data dir, with the band gate
    rows       the ExtDelta table, warm chain, anchor
    loss       the box-share FM loss, log in the glyph count
    train      rows-only plain FM on one band
    boxprobe   the box-share gradient read (no training): ‖g_in‖ / ‖g_out‖ per glyph count
    eval       exact / native / cf_sense via the probe stages + the regression check
    bake       table → pack pair
    ledger     runs/ledger.jsonl

The probe line's ``src/`` packages are top-level names (``common``, ``data``,
``train``, ``eval``); ``paths.bootstrap()`` puts them on ``sys.path``, points
their ``common.paths.OUT`` at ``output/cjk_anima_scale/``, and this package's
name keeps the two apart.
"""
