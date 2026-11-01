"""``--config`` TOML layering on the non-Anima lines' requests."""

import pytest

from library.moss.requests import CacheRequest, TrainRequest


@pytest.mark.core
def test_shipped_moss_config_is_the_defaults():
    assert TrainRequest.from_argv(["--config", "configs/moss/lora.toml"]) == (
        TrainRequest()
    )


@pytest.mark.core
def test_shipped_moss_cache_config_is_the_defaults():
    assert CacheRequest.from_argv(["--config", "configs/moss/cache.toml"]) == (
        CacheRequest()
    )


@pytest.mark.core
def test_cli_wins_over_config(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text("epochs = 3\nlr = 1\ncompile = false\n")
    req = TrainRequest.from_argv(["--config", str(cfg), "--epochs", "5"])
    assert (req.epochs, req.lr, req.compile) == (5, 1.0, False)
    assert TrainRequest.from_argv(["--config", str(cfg), "--compile"]).compile


@pytest.mark.parametrize("body", ["rnak = 3\n", 'epochs = "3"\n', 'algo = "lorx"\n'])
def test_bad_config_raises(tmp_path, body):
    cfg = tmp_path / "c.toml"
    cfg.write_text(body)
    with pytest.raises(ValueError):
        TrainRequest.from_argv(["--config", str(cfg)])
