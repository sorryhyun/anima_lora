"""`write_gen_manifest` — the inference-side half of the daemon result-envelope
lift (proposal Phase 1a). Under a daemon spawn it drops a generation manifest +
`result_path.json` pointer.
"""

from __future__ import annotations

import argparse
import json

import pytest

from library.inference import write_gen_manifest


def _args(**kw) -> argparse.Namespace:
    base = dict(
        lora_weight="output/ckpt/foo.safetensors",
        infer_steps=28,
        guidance_scale=4.0,
        sampler="euler",
        save_path="output/tests",
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.core
def test_daemon_writes_manifest_and_pointer(tmp_path, monkeypatch):
    monkeypatch.setenv("ANIMA_DAEMON_JOB_DIR", str(tmp_path))
    images = ["output/tests/x.png", "output/tests/y.png"]
    mp = write_gen_manifest(_args(), images)

    manifest = json.loads((tmp_path / "gen_manifest.json").read_text())
    assert manifest["label"] == "foo"  # derived from the adapter stem
    assert manifest["metrics"]["n_images"] == 2
    assert manifest["metrics"]["infer_steps"] == 28
    assert manifest["images"] == images

    pointer = json.loads((tmp_path / "result_path.json").read_text())
    assert pointer["path"] == mp
