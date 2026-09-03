"""Dataset curation tasks — organize / select images, distinct from preprocess.

``preprocess-*`` makes data training-ready (resize → latent/text/PE caches);
``curate-*`` is about *curation* — grouping, dedup, coverage — over the native
source tree. Command bodies for the ``make curate-*`` targets live here. Each
runs an ``anime_tools`` stage as its request object (``GroupRequest``)
through ``_common.execute_stage``: in-process under a daemon job, a
``python -m`` child from a shell.
"""

from __future__ import annotations

from ._common import _path, execute_stage, request_with_args, stage_by_id


def _group_request(extra):
    """The ``GroupRequest`` for ``make curate-group ARGS=…``: the trainer's
    source tree as the base, ``ARGS`` applied through the request's parser
    (``--match-frac-min 0.4``, ``--embedder module:callable``, …)."""
    from anime_tools.grouping.requests import GroupRequest

    req = GroupRequest(source_dir=_path("source_image_dir", "image_dataset"))
    return request_with_args(req, extra, prog="make curate-group ARGS=")


def cmd_curate_group(extra):
    """Group dataset images by PE-Spatial visual similarity.

    Writes ``post_image_dataset/groups/groups.json`` (per-artist
    connected-components over the same near-twin grid gate the miner uses — two
    images group when ``match_frac >= --match-frac-min`` at per-cell floor
    ``--cell-match-min``). The GUI Dataset tab reads the manifest to filter the
    image list by group. Tune via ``ARGS="--match-frac-min 0.4 --cell-match-min
    0.9"`` (higher = tighter) / ``ARGS="--min-size 2"``. Reuses the shared PE
    feature cache, so re-runs and threshold sweeps are cheap. The embedder
    defaults to anime_tools' own PE-Spatial; ``ARGS="--embedder
    module:callable"`` overrides.
    """
    execute_stage(stage_by_id("groups"), _group_request(extra))
