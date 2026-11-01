"""Anima trainer CLI: the argparse surface, the network-kwarg intake, the
startup crash reporter and :func:`main` (what ``train.py`` runs)."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from library.anima import training as anima_train_utils
from library.config import loader as config_util
from library.config import schema as _config_schema
from library.config.cli_args import (
    add_dataset_arguments,
    add_dit_training_arguments,
    add_masked_loss_arguments,
    add_network_arguments,
    add_optimizer_arguments,
    add_sd_models_arguments,
    add_train_misc_arguments,
    add_training_arguments,
    add_validation_arguments,
    verify_command_line_training_args,
)
from library.config.io import read_config_from_file
from library.datasets import base as _datasets_base
from library.log import add_logging_arguments
from library.models import sai_spec as sai_model_spec
from library.training import add_custom_train_arguments
from library.runtime.pause import TrainingPaused
from networks import all_network_kwargs as _all_network_kwargs

logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    add_sd_models_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)
    add_dataset_arguments(parser, True, True, True)
    add_training_arguments(parser, True)
    add_masked_loss_arguments(parser)
    add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)

    add_network_arguments(parser)
    add_validation_arguments(parser)
    add_train_misc_arguments(parser)
    return parser


# Network-module-consumed flags. Don't flow through argparse directly since
# `create_network` reads them from ``kwargs``; sourced from the flat
# ``NETWORK_KWARGS`` allowlist in ``networks/__init__.py``.
NETWORK_KWARG_ALLOWLIST: tuple[str, ...] = _all_network_kwargs()

# Top-level training args that flow through ``net_kwargs`` though they aren't
# network kwargs. Kept explicit; growth here should be reviewed.
_EXTRA_FORWARDED_TOP_LEVEL_ARGS: tuple[str, ...] = (
    # Postfix contrastive resets its intra-step reference set on step
    # boundary, so it needs the grad-accum window.
    "gradient_accumulation_steps",
)


def resolve_network_kwargs(args) -> dict[str, str]:
    """The single intake for network kwargs, merging both config paths
    (``--network_args k=v`` and allowlisted top-level config keys; CLI wins
    on overlap). Cached on ``args._network_kwargs`` — read from here rather
    than re-scanning ``args.network_args``. Values are strings.
    """
    cached = getattr(args, "_network_kwargs", None)
    if cached is not None:
        return cached

    net_kwargs: dict[str, str] = {}
    for net_arg in getattr(args, "network_args", None) or []:
        key, value = net_arg.split("=", 1)
        net_kwargs[key] = value

    # Forward known network-arg keys from top-level config (TOML), plus a
    # small tail of top-level training args network modules still want.
    for key in NETWORK_KWARG_ALLOWLIST + _EXTRA_FORWARDED_TOP_LEVEL_ARGS:
        if (
            key not in net_kwargs
            and hasattr(args, key)
            and getattr(args, key) is not None
        ):
            net_kwargs[key] = str(getattr(args, key))

    args._network_kwargs = net_kwargs
    return net_kwargs


def build_network_extras() -> dict[str, _config_schema.ConfigKey]:
    return {
        k: _config_schema.ConfigKey(name=k, type="str", source="network_module")
        for k in NETWORK_KWARG_ALLOWLIST
    }


def _install_crash_reporter(argv: list[str]) -> None:
    """Record a fatal startup/training exception into ``--progress_jsonl``.

    Under the daemon's windowless ``pythonw.exe`` launch stdout/stderr are
    dropped, so an uncaught traceback is otherwise lost. ``run_scope`` only
    covers failures after ``ProgressSink.run_start`` (late in ``train()``);
    this excepthook catches earlier ones (cache incomplete, config/dataset
    build, model load).
    """
    path = None
    for i, tok in enumerate(argv):
        if tok == "--progress_jsonl" and i + 1 < len(argv):
            path = argv[i + 1]
        elif tok.startswith("--progress_jsonl="):
            path = tok.split("=", 1)[1]
    if not path or path.strip().lower() in ("", "none", "off"):
        return

    import json as _json

    prev_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        # KeyboardInterrupt is a clean stop, handled by run_scope/the daemon's
        # stop_requested path — don't mislabel it an error.
        if not issubclass(exc_type, KeyboardInterrupt):
            try:
                # Dedupe: run_scope may already have written the terminal event
                # for an in-loop failure; don't append a second one.
                already_ended = False
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if line:
                                last = line
                    try:
                        already_ended = _json.loads(last).get("ev") == "run_end"
                    except (NameError, ValueError):
                        already_ended = False
                if not already_ended:
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    with open(path, "a", encoding="utf-8") as fh:
                        fh.write(
                            _json.dumps(
                                {
                                    "ev": "run_end",
                                    "status": "error",
                                    "final_step": -1,
                                    "error": f"{exc_type.__name__}: {exc}",
                                }
                            )
                            + "\n"
                        )
            except Exception:  # noqa: BLE001 — reporting must never mask the crash
                pass
        prev_hook(exc_type, exc, tb)

    sys.excepthook = _hook


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    _install_crash_reporter(argv)
    parser = setup_parser()
    _config_schema.populate_schema(parser, extras=build_network_extras())

    args = parser.parse_args(argv)
    verify_command_line_training_args(args)
    args = read_config_from_file(args, parser, argv=argv)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    artist = getattr(args, "artist_filter", None)
    if artist:
        _datasets_base.set_artist_filter(artist)
        slug = artist.lstrip("@")
        args.output_dir = "output/ckpt-artist"
        args.output_name = f"{args.output_name}_{slug}"
        logger.info(
            f"artist_filter active: '{artist}' → output_dir={args.output_dir}, "
            f"output_name={args.output_name}"
        )

    from library.anima.trainer.trainer import AnimaTrainer

    trainer = AnimaTrainer()
    try:
        trainer.train(args)
    except TrainingPaused:
        sys.exit(0)  # run_end(paused) already carries the state dir
