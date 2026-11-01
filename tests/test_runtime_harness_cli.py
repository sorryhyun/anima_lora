"""Compile-cache identity in ``library.runtime.dynamo``: a signature drift or a
nested cache dir silently throws away the inductor cache. Model-free.
"""

from __future__ import annotations

from pathlib import Path


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
