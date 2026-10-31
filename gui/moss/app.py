"""MOSS-Image LoRA window — the ``gui.lines`` window over ``library.moss``.

Sidecars ``scripts/moss/{cache,train,generate}.py``. The generate sidecar takes
one ``lora_multiplier``, so a Test is two jobs into one run folder: ``lora/``
with the adapter, then ``base/`` without it, same prompts and seeds. The i2i tabs
after Train are ``gui.moss.i2i``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from gui.lines.window import LineSpec, run
from gui.moss.plans import generate_plans, model_items, train_plans
from gui.moss.strings import FIELDS, FIELDS_GENERATE, UI
from library.moss.requests import (
    CacheRequest,
    GenerateRequest,
    TrainRequest,
    resolve_model_dir,
    variant_dir,
)
from library.moss.scan import cache_counts, scan


def _extra_pages(panel) -> list:
    from gui.moss.i2i import pages

    return pages(panel)


def _setup_forms(train, test) -> None:
    model_items(train, train_plans(), "model_help")
    model_items(test, generate_plans(), "model_help_test")


def _test_runs(req: GenerateRequest) -> list[GenerateRequest]:
    out = Path(req.out_dir)
    return [
        dataclasses.replace(req, out_dir=str(out / "lora")),
        dataclasses.replace(req, lora="", out_dir=str(out / "base")),
    ]


def _records(manifest: Path) -> list[dict]:
    if not manifest.is_file():
        return []
    return json.loads(manifest.read_text(encoding="utf-8"))["images"]


def _test_images(run_dir: Path) -> list[tuple[Path, float, str]]:
    """LoRA and base image of each prompt side by side."""
    base = {r["key"]: r for r in _records(run_dir / "base" / "manifest.json")}
    out = []
    for r in _records(run_dir / "lora" / "manifest.json"):
        out.append((Path(r["image"]), r.get("lora_multiplier") or 0.0, r["prompt"]))
        if r["key"] in base:
            out.append((Path(base[r["key"]]["image"]), 0.0, r["prompt"]))
    return out


SPEC = LineSpec(
    key="moss",
    cache_cls=CacheRequest,
    train_cls=TrainRequest,
    test_cls=GenerateRequest,
    model_index=lambda model_dir, variant: (
        variant_dir(model_dir, variant) / "model_index.json"
    ),
    resolve_model_dir=resolve_model_dir,
    scan=scan,
    cache_counts=cache_counts,
    test_runs=_test_runs,
    test_images=_test_images,
    ui=UI,
    fields=FIELDS,
    fields_test=FIELDS_GENERATE,
    extra_pages=_extra_pages,
    setup_forms=_setup_forms,
)


def main() -> None:
    run(SPEC)
