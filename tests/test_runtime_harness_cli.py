"""Tests for the run-harness promotion (Phase 3 / item B) and the shared CLI
flag groups (item C) of ``docs/anima/proposal/tooling_architecture.md``.

All model-free: ``build_anima`` is only exercised on its no-DiT-path guard, and
``discover_bucketed_samples`` runs against tiny fixture sidecars. The harness +
flag helpers themselves never touch the encoder weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest


def test_harness_reexports_match_canonical_homes() -> None:
    """`bench._anima` must re-export the promoted symbols, not redefine them."""
    import bench._anima as a
    from library.io.cache import discover_bucketed_samples
    from library.runtime.harness import AnimaBundle, build_anima

    assert a.build_anima is build_anima
    assert a.AnimaBundle is AnimaBundle
    assert a.discover_bucketed_samples is discover_bucketed_samples


def test_build_anima_requires_dit_path() -> None:
    from library.runtime.harness import build_anima

    args = argparse.Namespace(device="cpu", dtype="bf16")
    with pytest.raises(SystemExit, match="no DiT path"):
        build_anima(args)  # no dit_path, no args.dit -> guard fires before load


def test_add_device_args_defaults() -> None:
    from library.runtime.argparse_groups import add_device_args

    p = argparse.ArgumentParser()
    add_device_args(p)
    args = p.parse_args([])
    assert args.dtype == "bf16"
    assert args.device in ("cuda", "cpu")
    # narrowed choices + custom default are honored
    p2 = argparse.ArgumentParser()
    add_device_args(
        p2,
        include_device=False,
        dtype_default="bfloat16",
        dtype_choices=("bfloat16", "float16", "float32"),
    )
    a2 = p2.parse_args([])
    assert a2.dtype == "bfloat16"
    assert not hasattr(a2, "device")
    with pytest.raises(SystemExit):
        p2.parse_args(["--dtype", "bf16"])  # not in narrowed choices


def test_add_io_args_required_and_optional() -> None:
    from library.runtime.argparse_groups import add_io_args

    p = argparse.ArgumentParser()
    add_io_args(p, include_batch_size=True, batch_size_default=4)
    args = p.parse_args(["--dir", "/tmp/x"])
    assert args.dir == "/tmp/x"
    assert args.cache_dir is None
    assert args.recursive is False
    assert args.batch_size == 4
    with pytest.raises(SystemExit):
        p.parse_args([])  # --dir required by default

    p2 = argparse.ArgumentParser()
    add_io_args(p2, dir_required=False, include_num_workers=True)
    a2 = p2.parse_args([])  # --dir optional here
    assert a2.dir is None
    assert a2.num_workers == 4


def test_add_common_args_delegates_device_dtype() -> None:
    from bench._anima import add_common_args

    p = argparse.ArgumentParser()
    add_common_args(p)
    args = p.parse_args([])
    assert args.dtype == "bf16"
    assert args.seed == 0
    assert args.compile is False
    # opt-out path drops the device/dtype flags entirely
    p2 = argparse.ArgumentParser()
    add_common_args(p2, include_device=False, include_dtype=False)
    a2 = p2.parse_args([])
    assert not hasattr(a2, "device")
    assert not hasattr(a2, "dtype")


def test_compile_signature_normalizes_mode() -> None:
    """train.py (mode=None) and distill_turbo (mode="") must serialize the same.

    A formatting drift between the two entry points would thrash-wipe the
    shared inductor cache on every lora <-> turbo switch.
    """
    from library.runtime.dynamo import compile_signature

    kw = dict(n_token_families=4, seq_range=(3000, 4200), dynamic_seq=True)
    assert compile_signature(**kw, mode="") == compile_signature(**kw, mode=None)
    # and the pre-promotion train.py marker format is preserved verbatim, so
    # markers written by older runs still compare equal (no spurious wipe)
    assert compile_signature(**kw, mode=None) == (
        "families=4;seq_range=(3000, 4200);dynamic_seq=True;backend=inductor;mode=None"
    )


def test_isolate_compile_cache(tmp_path: Path, monkeypatch) -> None:
    """Per-signature TORCHINDUCTOR_CACHE_DIR subdirs off a stable base."""
    import os

    from library.runtime import dynamo

    monkeypatch.setattr(dynamo, "_compile_cache_base", None)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))

    dir_a = dynamo.isolate_compile_cache("sig-a")
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == dir_a
    assert Path(dir_a).parent == tmp_path  # nested under the original base

    # deterministic: same signature -> same dir (warm cache reuse across runs)
    assert dynamo.isolate_compile_cache("sig-a") == dir_a

    # different signature -> sibling dir off the SAME base (no nesting under
    # the previous per-signature dir, even though the env var now points there)
    dir_b = dynamo.isolate_compile_cache("sig-b")
    assert dir_b != dir_a
    assert Path(dir_b).parent == tmp_path


def _make_sample(data_dir: Path, stem: str, bucket: str, *, with_te: bool) -> None:
    """Write a {stem}_{bucket}_anima.npz (+ optional TE sidecar) fixture."""
    w, h = bucket.split("x")
    npz = data_dir / f"{stem}_{int(w) * 8}x{int(h) * 8}_anima.npz"
    np.savez(
        npz, **{f"latents_{bucket}": np.zeros((4, int(h), int(w)), dtype=np.float32)}
    )
    if with_te:
        (data_dir / f"{stem}_anima_te.safetensors").write_bytes(b"")


def test_discover_bucketed_samples(tmp_path: Path) -> None:
    from library.io.cache import discover_bucketed_samples

    for i in range(3):
        _make_sample(tmp_path, f"a{i}", "16x24", with_te=True)
    _make_sample(tmp_path, "b0", "32x32", with_te=True)
    _make_sample(tmp_path, "orphan", "16x24", with_te=False)  # no TE -> skipped

    # most-populous bucket chosen when bucket=None
    chosen, picks = discover_bucketed_samples(tmp_path, None, 2, seed=0)
    assert chosen == "16x24"
    assert len(picks) == 2
    stem, key, npz_path, te_path = picks[0]
    assert key == "latents_16x24"
    assert Path(te_path).exists()
    assert "orphan" not in {p[0] for p in picks}

    # explicit bucket; too-small pool raises unless allow_replace
    with pytest.raises(SystemExit, match="has 1 samples"):
        discover_bucketed_samples(tmp_path, "32x32", 4, seed=0)
    _, picks2 = discover_bucketed_samples(
        tmp_path, "32x32", 4, seed=0, allow_replace=True
    )
    assert len(picks2) == 4

    with pytest.raises(SystemExit, match="not found"):
        discover_bucketed_samples(tmp_path, "999x999", 1, seed=0)
