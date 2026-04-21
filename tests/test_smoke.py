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
    from library.config.io import load_method_preset

    merged = load_method_preset(method, preset="default")
    assert isinstance(merged, dict)
    assert len(merged) > 0


def test_method_config_coverage():
    # lora/tlora/tlora_rf/hydralora collapsed into lora.toml; postfix/postfix_exp/
    # postfix_func/prefix collapsed into postfix.toml. Expected survivors: apex,
    # graft, lora, postfix.
    expected = {"apex", "graft", "lora", "postfix"}
    assert expected.issubset(set(METHOD_NAMES)), (
        f"expected {expected} in method configs, got {METHOD_NAMES}"
    )
