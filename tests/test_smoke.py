from __future__ import annotations

import argparse
import importlib

import pytest

from tests.conftest import iter_method_names

METHOD_NAMES = list(iter_method_names())


def test_train_imports():
    train = importlib.import_module("train")
    assert callable(train.setup_parser)
    assert hasattr(train, "AnimaTrainer")


def test_parser_builds():
    train = importlib.import_module("train")
    parser = train.setup_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    args = parser.parse_args([])
    assert hasattr(args, "method")


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_load_method_preset_resolves(method: str):
    from library.train_util import load_method_preset

    merged = load_method_preset(method, preset="default")
    assert isinstance(merged, dict)
    assert len(merged) > 0


def test_method_config_coverage():
    assert len(METHOD_NAMES) >= 9, f"expected ≥9 method configs, got {METHOD_NAMES}"
