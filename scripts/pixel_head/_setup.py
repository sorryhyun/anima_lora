"""What the pixel-head trainers set up alike: the frozen models and the progress sink.

Torch is imported inside the functions: the entry points set the CUDA allocator's env
before it loads.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def load_models(req, device, seq_bands: list, adapter: str = ""):
    """→ (VAE, frozen DiT, adapter network or None), in the order the hooks need:
    build → stream change → adapter → compile."""
    import torch

    from library.anima.pixel_head_train import attach_adapter, compile_dit
    from library.models import qwen_vae
    from library.runtime.harness import build_anima

    vae = qwen_vae.load_vae(
        req.vae, device="cpu", dtype=torch.bfloat16, eval=True, vae_2d=True
    ).to(device)
    ns = SimpleNamespace(
        dit=req.dit,
        device=req.device,
        dtype="bf16",
        attn_mode=req.attn_mode,
        gradient_checkpointing=False,
        compile=False,  # compiled below, after the stream change
        compile_mode=None,
    )
    anima = build_anima(ns, adapter=None, train_mode=False).anima
    if req.stream == "fp32":
        from library.anima.stream import stream_fp32_

        stream_fp32_(anima)
    network = attach_adapter(anima, adapter, device) if adapter else None
    if req.compile:
        compile_dit(anima, seq_bands)
    return vae, anima, network


def open_sink(req, out_dir: Path, method: str):
    """The run's ``progress.jsonl`` sink, or None where progress is off."""
    import os

    from library.runtime.progress import ProgressSink, command_job_path

    path = command_job_path(
        ProgressSink.resolve_path(
            SimpleNamespace(output_dir=str(out_dir), output_name=req.output_name)
        )
    )
    if not path:
        return None
    sink = ProgressSink(path, run=req.output_name, method=method, preset=None)
    sink.run_start(total_steps=req.rollouts, total_epochs=1, pid=os.getpid())
    return sink
