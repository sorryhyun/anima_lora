"""Sample renders during training (``TrainRequest.sample_prompts``).

The prompts are encoded once, before the DiT loads, on the cache's text encoder,
template, prompt tail and caption form (``cached_embeds``, generate's TE cache).
A render runs the training transformer with its adapters as they stand: no
grad, ER-SDE with CFG and the uncond branch skipping block 22, as
``scripts/moss/generate.py`` renders by default. Under ``compile`` the token
range ``compile_blocks`` marks takes in the sample lengths (``seq_range``), so
the no-grad graph is one more dynamic graph per block, compiled once. The VAE decodes on the CPU in
a background thread (fp32, ~15 s per 1024² image), so training resumes while
the PNGs are written.

Under block swap each forward is followed by ``attached.prepare`` (the
training schedule leaves the head evicted after a forward no backward follows,
as the distill teacher's does), and the skipped block 22 runs its swap hooks
through ``transformer.skip_hook``. Not with reference latents (i2i), trigger
registers or ``delta_rows``.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from library.env import resolve_under_home
from library.moss.requests import (
    DEFAULT_SHIFT,
    DEFAULT_UNCOND_SKIP,
    TrainRequest,
    resolve_model_dir,
)


def parse_prompts(value: str) -> list[str]:
    """``sample_prompts``: a ``.txt`` / ``.json`` file path, or prompts one per line."""
    value = value.strip()
    if not value:
        return []
    if "\n" not in value and value.endswith((".txt", ".json")):
        path = resolve_under_home(value)
        if not path.is_file():
            raise SystemExit(f"sample_prompts: no file at {path}")
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            return [p.strip() for p in json.loads(text) if p.strip()]
        value = text
    return [ln.strip() for ln in value.splitlines() if ln.strip()]


def _cache_te(samples) -> tuple[str, str]:
    """``(text_encoder precision, caption_format)`` the cache's TE files stamp."""
    from safetensors import safe_open

    from library.moss.cache import ENCODER_ID

    for s in samples:
        if s.te_path is None:
            continue
        with safe_open(str(s.te_path), "pt") as f:
            meta = f.metadata() or {}
        encoder = meta.get("encoder", ENCODER_ID)
        precision = "bf16"
        if encoder.startswith(f"{ENCODER_ID}, ") and encoder.endswith(" weights"):
            precision = encoder[len(ENCODER_ID) + 2 : -len(" weights")]
        return precision, meta.get("caption_format", "text")
    raise SystemExit("sample_prompts: the cache has no TE files")


class Sampler:
    """Built before the DiT loads (it runs the TE); ``render`` after each epoch."""

    def __init__(
        self,
        req: TrainRequest,
        samples,
        *,
        token_split: str,
        prompt_tail: str,
        out_dir: Path,
        device,
    ):
        prompts = parse_prompts(req.sample_prompts)
        if not prompts:
            raise SystemExit(
                "sample_every_epochs / sample_at_first need sample_prompts"
            )
        if any(s.load_ref is not None for s in samples):
            raise SystemExit("sample_prompts does not run on an i2i cache")
        if req.trigger_registers or req.delta_rows != "all":
            raise SystemExit(
                "sample_prompts does not run with trigger_registers / delta_rows"
            )
        if req.sample_width % 16 or req.sample_height % 16:
            raise SystemExit("sample_width / sample_height: multiples of 16")
        from library.moss.generation.conditioning import cached_embeds

        precision, caption_format = _cache_te(samples)
        fmt = req.sample_prompt_format
        if fmt == "auto":
            fmt = (
                caption_format if caption_format in ("layout_json", "tags") else "text"
            )
        if fmt != "text":
            from library.moss.caching.sources import caption_converter

            convert = caption_converter(fmt)
            prompts = [convert(p) for p in prompts]
        self.req, self.device, self.out_dir = req, device, out_dir
        self.prompts = prompts
        model_dir = resolve_model_dir(req.model_dir)
        texts = prompts + ([""] if req.sample_cfg != 1.0 else [])
        print(
            f"sample prompts: {len(prompts)} ({fmt}), {req.sample_width}x"
            f"{req.sample_height}, {req.sample_steps} steps, cfg {req.sample_cfg:g}",
            flush=True,
        )
        self.embeds = cached_embeds(
            model_dir,
            texts,
            out_dir.parent.parent / "_sample_te_cache",
            precision,
            token_split,
            prompt_tail,
        )
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moss-vae")
        self.model_dir = model_dir
        self._vae = None

    def seq_range(self) -> tuple[int, int]:
        """The joint lengths (image + text tokens) a render runs at."""
        n_img = (self.req.sample_width // 16) * (self.req.sample_height // 16)
        lengths = [n_img + e.shape[0] for e in self.embeds.values()]
        return min(lengths), max(lengths)

    def due(self, epoch: int, last: int) -> bool:
        every = self.req.sample_every_epochs
        return bool(every) and (epoch % every == 0 or epoch == last)

    @torch.no_grad()
    def render(self, transformer, attached, tag: str) -> None:
        """Every prompt at the current weights → ``samples/<tag>_<k>.png`` (decoded
        in the background)."""
        from library.moss.generate import denoise, initial_noise
        from library.moss.loader import pad_embeds
        from library.moss.model import snapped_sigmas

        req, device = self.req, self.device
        self.out_dir.mkdir(parents=True, exist_ok=True)
        sigmas = snapped_sigmas(req.sample_steps, DEFAULT_SHIFT)
        uncond = uncond_mask = None
        if req.sample_cfg != 1.0:
            uncond, uncond_mask = pad_embeds([self.embeds[""]], device=device)
        hook, gate = transformer.layer_hook, transformer.ref_gate
        transformer.layer_hook = transformer.ref_gate = None
        restore = None
        if attached is not None:
            transformer.skip_hook = attached.skip
            restore = transformer.register_forward_hook(
                lambda *_: attached.prepare(free_cache=False)
            )
        t0 = time.time()
        try:
            for k, prompt in enumerate(self.prompts):
                cond, cond_mask = pad_embeds([self.embeds[prompt]], device=device)
                seed = req.sample_seed + k
                noise = initial_noise(seed, req.sample_height, req.sample_width)
                latent = denoise(
                    transformer,
                    noise.to(device),
                    cond,
                    cond_mask,
                    uncond,
                    uncond_mask,
                    sigmas,
                    cfg=req.sample_cfg,
                    uncond_skip=DEFAULT_UNCOND_SKIP,
                    sampler="er_sde",
                    seed=seed,
                ).cpu()
                path = self.out_dir / f"{tag}_{k:02d}.png"
                self.pool.submit(self._decode, latent, path)
        finally:
            transformer.layer_hook, transformer.ref_gate = hook, gate
            transformer.skip_hook = None
            if restore is not None:
                restore.remove()
            torch.cuda.empty_cache()
        print(
            f"  sampled {len(self.prompts)} prompts ({tag}) in {time.time() - t0:.0f}s "
            f"-> {self.out_dir}",
            flush=True,
        )

    def _decode(self, latent: torch.Tensor, path: Path) -> None:
        from diffusers.image_processor import VaeImageProcessor

        from library.moss.loader import load_vae
        from library.moss.model import decode

        try:
            if self._vae is None:
                self._vae = load_vae(self.model_dir)  # fp32, CPU
            with torch.no_grad():
                image = decode(latent, self._vae)
            processor = VaeImageProcessor(vae_scale_factor=16)
            processor.postprocess(image, output_type="pil")[0].save(path)
            print(f"  sample {path.name}", flush=True)
        except Exception as e:  # noqa: BLE001 — a sample never fails the run
            print(f"  sample {path.name}: decode failed: {e!r}", flush=True)

    def close(self) -> None:
        """Wait for the decodes still running."""
        self.pool.shutdown(wait=True)
        self._vae = None
