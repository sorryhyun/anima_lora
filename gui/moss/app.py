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

from library.env import resolve_under_home

from gui.lines.window import LineJob, LineSpec, run, tr
from gui.moss.plans import generate_plans, model_items, recipe_knob, train_plans
from gui.moss.strings import FIELDS, FIELDS_GENERATE, UI
from library.moss.requests import (
    CacheRequest,
    DistillCacheRequest,
    GenerateRequest,
    TrainRequest,
    resolve_model_dir,
    variant_dir,
)
from library.moss.scan import cache_counts, header_metadata, scan, stale_teachers


def _extra_pages(panel) -> list:
    from gui.moss.i2i import pages

    return pages(panel)


def _setup_forms(train, test) -> None:
    model_items(train, train_plans(), "model_help")
    recipe_knob(train)
    model_items(test, generate_plans(), "model_help_test")


def _cache_encoder(cache: str) -> str:
    """The TE weights a cache was encoded with (its ``encoder`` stamp), int8 when
    no TE file is there yet."""
    folder = resolve_under_home(cache)
    te = next(folder.glob("*.moss_te.safetensors"), None) if folder.is_dir() else None
    if te is None:
        return "int8"
    stamp = header_metadata(te).get("encoder", "")
    return next((p for p in ("int8", "fp8") if stamp.endswith(f"{p} weights")), "bf16")


def _before_train(req: TrainRequest, cache_req: CacheRequest | None) -> list[LineJob]:
    """With distill on, the teacher embeds first when any is missing or older than
    its TE file, unless the chain's preprocessing writes them (distill_teacher)."""
    if req.distill == "off" or (cache_req is not None and cache_req.distill_teacher):
        return []
    if cache_req is None and not stale_teachers(resolve_under_home(req.cache)):
        return []
    teacher = DistillCacheRequest(
        cache=req.cache,
        text_encoder=(
            cache_req.text_encoder if cache_req else _cache_encoder(req.cache)
        ),
        model_dir=req.model_dir,
    )
    return [LineJob(teacher, "moss-cache", note=tr("distill_teacher"))]


def _test_runs(req: GenerateRequest) -> list[GenerateRequest]:
    """The LoRA render and the base render, both on the LoRA's prompt tail."""
    from library.moss.prompt_tail import resolve_prompt_tail

    if req.prompt_tail == "auto" and resolve_under_home(req.lora or ".x").is_file():
        req = dataclasses.replace(req, prompt_tail=resolve_prompt_tail(req))
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
    before_train=_before_train,
)


def main() -> None:
    run(SPEC)
