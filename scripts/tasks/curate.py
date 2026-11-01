"""Command bodies for the ``make curate-*`` targets.

Grouping, dedup and coverage over the native source tree. Each runs an
``anime_tools`` stage as its request object (``GroupRequest``) through
``_common.execute_stage``.
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

    Writes ``workspace/groups/groups.json`` (``GroupRequest.out``; per-artist
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
