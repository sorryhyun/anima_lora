"""cjk_scale — the JA vocab pack at scale: one loss, one trainer, a σ-band
schedule whose stages differ only in their data (``../design.md``).

    paths      where runs land (``output/wake_probe/{data,rows}_scale_<stage>_<tag>``)
    windows    the vocab band law as code: (kind, px, layout) → band
    config     ``configs/<stage>.toml`` → StageConfig
    recipes    the item generators over the probe line's render primitives
    builder    stage config → data dir, with the band gate
    rows       the ExtDelta table, warm chain, anchor
    train      rows-only plain FM on one band
    eval       exact / native / cf_sense via the probe stages + the regression check
    bake       table → pack pair
    ledger     runs/ledger.jsonl

The probe line's ``src/`` packages are top-level names (``common``, ``data``,
``train``, ``eval``); ``paths.bootstrap()`` puts them on ``sys.path`` and this
package's name keeps the two apart.
"""
