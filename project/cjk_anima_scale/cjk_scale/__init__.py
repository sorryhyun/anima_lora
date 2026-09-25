"""cjk_scale — the JA vocab pack at scale: a run is its vocabs + what to read;
one loss, one trainer, σ per item from the band law (``../plan.md``).

    paths      where runs land (``output/cjk_anima_scale/<run>/``), the seed table
    windows    the vocab band law as code: (kind, px, layout) → band
    config     ``configs/runs/<run>.toml`` = {vocabs, read} → RunConfig; the data pools
    recipes    the item drawers over the vendored render primitives
    builder    the recipe table by kind → ``<run>/data/``, every item stamped with its band
    rows       the ExtDelta table, seed warm start, frozen context
    loss       the box-share FM loss, log in the glyph count
    train      the fixed trainer: the vocabs' rows, σ per item
    eval       floor + trained arms on the automatic rulers → sheet.png + reads.json
    conflict   do the run's band groups pull a row the same way (no training)
    bake       table → pack pair
    ledger     runs/ledger.jsonl
    legacy     the pre-2026-09-25 stage configs, for the experiments that read old dirs

The stage packages (``common`` / ``data`` / ``train`` / ``eval`` /
``scenes``, ``cli``, ``stages``) are the line's own ``../src/`` — vendored
2026-09-25, top-level names; ``paths.bootstrap()`` puts them on ``sys.path``.
"""
