"""Train a LoRA on the precached folder — flow matching, batch 1, block swap.

Reads only what ``cache.py`` wrote, so neither encoder is ever loaded:
the transformer gets the whole card under the block swapper.

The sigma schedule matches inference rather than being uniform. The pipeline
shifts its sigmas by ``mu = calculate_shift(image_seq_len)`` off the scheduler
config, so training samples ``t`` logit-normally and pushes it through the
scheduler's own ``time_shift`` at the same mu; a uniform sample would spend most
of its steps where the sampler never looks. ``shift_terminal`` is a property of
the discrete schedule and has no training counterpart, so it is not applied.

The joint sequence length varies per sample on both axes — each image keeps its
native aspect at ~1 MP, and caption length runs 112–346 tokens on this folder —
so ``mu`` is per-sample, the pipeline deriving it from the image token count.

Block compile is **off** by default for the same reason: it measured ±0 at a
fixed 512² (``backward_smoke.py`` — 12 swaps × 0.41 GB × 2 directions is the
whole step time, so compute hides under PCIe) and a moving token count can only
make it worse. ``--compile`` turns it on dynamically if the swap count ever
drops far enough for compute to matter.

    make daemon-run ARGS="scripts/qwen21/train.py --epochs 8"
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from library.env import resolve_under_home
from library.qwen21 import blockswap
from library.qwen21.accel import compile_blocks, recompile_report
from library.qwen21.loader import (
    TRANSFORMER_BLOCKS,
    block_devices,
    empty_cache,
    free_vram_gb,
    load_transformer,
    place,
)
from library.qwen21.lora import LoRANetwork
from library.qwen21.requests import TrainRequest, resolve_model_dir

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load_cache(cache_dir: Path) -> list[dict]:
    """Every stem with both a latent and a text-embedding file.

    The latent's shape is read from the file's metadata — the packed tensor is
    a flat ``(T, 64)`` and the model needs the ``(h, w)`` it came from.
    """
    items = []
    for latent_path in sorted(cache_dir.glob("*.latent.safetensors")):
        stem = latent_path.name[: -len(".latent.safetensors")]
        te_path = cache_dir / f"{stem}.te.safetensors"
        if not te_path.exists():
            print(f"skip {stem}: no text cache", flush=True)
            continue
        with safe_open(latent_path, framework="pt") as handle:
            meta = handle.metadata() or {}
            latents = handle.get_tensor("latents")
        if "latent_h" not in meta:
            raise SystemExit(
                f"{latent_path.name} predates aspect-preserving caching — "
                "re-run caching with --overwrite"
            )
        latent_h, latent_w = int(meta["latent_h"]), int(meta["latent_w"])
        if latent_h * latent_w != latents.shape[0]:
            raise SystemExit(
                f"{latent_path.name}: {latent_h}x{latent_w} != {latents.shape[0]}"
            )
        text = load_file(te_path)
        items.append(
            {
                "stem": stem,
                "latents": latents,
                "latent_h": latent_h,
                "latent_w": latent_w,
                "size": meta.get("size", ""),
                "prompt_embeds": text["prompt_embeds"],
                "prompt_embeds_mask": text["prompt_embeds_mask"],
            }
        )
    return items


def calculate_shift(image_seq_len: int, config) -> float:
    """The pipeline's own dynamic shift, off the scheduler config."""
    base_seq_len = config.get("base_image_seq_len", 256)
    max_seq_len = config.get("max_image_seq_len", 4096)
    base_shift = config.get("base_shift", 0.5)
    max_shift = config.get("max_shift", 1.15)
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


def sample_sigma(
    scheduler, mu: float, logit_mean: float, logit_std: float, device: torch.device
) -> torch.Tensor:
    t = torch.sigmoid(torch.randn((1,), device=device) * logit_std + logit_mean)
    return scheduler.time_shift(mu, 1.0, t)


def training_step(transformer, item: dict, sigma: torch.Tensor):
    latents = item["latents"]
    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype)
    sigma_t = sigma.to(latents.dtype)
    noisy = (1.0 - sigma_t) * latents + sigma_t * noise
    target = noise - latents

    tokens = latents.shape[1]
    img_mask = torch.cat(
        [
            torch.zeros(
                item["prompt_embeds"].shape[:2], dtype=torch.bool, device=latents.device
            ),
            torch.ones((1, tokens // 4), dtype=torch.bool, device=latents.device),
        ],
        dim=1,
    )
    pred = transformer(
        hidden_states=noisy,
        encoder_hidden_states=item["prompt_embeds"],
        encoder_hidden_states_mask=item["prompt_embeds_mask"],
        timestep=sigma_t,
        img_shapes=[[(1, item["latent_h"], item["latent_w"])]],
        img_mask=img_mask,
        return_dict=False,
    )[0][:, -tokens:]
    return F.mse_loss(pred.float(), target.float())


def report_fit(blocks, attached, batch: dict) -> None:
    """After step 1: does it fit, and is the swap count paying for itself?

    Printing this at step 1 rather than at the end of the epoch is the
    difference between resizing the swap in five seconds and finding out two
    minutes in.

    Headroom is ``mem_get_info``'s free bytes, not ``total - max_allocated``.
    Two things live in that gap and both are real: the allocator's reserved-but
    -unallocated pool (~0.7 GB here), and whatever else has the card — a desktop
    session holds ~0.5 GB on this box. Sizing the swap against allocated bytes
    over total capacity overstates the room by more than a block and hands back
    a recommendation that OOMs.
    """
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_gb = free_vram_gb()
    per_block = blockswap.block_size_gb(blocks)
    swapped = attached.offloader.blocks_to_swap if attached else 0
    # Leave a block's worth of slack: the allocator grows for a few more steps.
    spare = max(0, int((free_gb - per_block) // per_block))
    print(
        f"fit @ step 1 ({batch['latents'].shape[1]} image + "
        f"{batch['prompt_embeds'].shape[1]} text tokens, the largest sample): "
        f"peak {allocated:.2f} GB allocated / {reserved:.2f} GB reserved of "
        f"{total_gb:.2f} GB, {free_gb:.2f} GB actually free, "
        f"swapping {swapped}/{len(blocks)} at {per_block:.2f} GB each",
        flush=True,
    )
    if swapped and spare:
        print(
            f"  -> {min(spare, swapped)} of those could stay resident: "
            f"rerun with --blocks_to_swap {max(0, swapped - spare)}",
            flush=True,
        )
    elif swapped:
        print("  -> swap count is at the edge of what fits; leave it", flush=True)


def run_train(req: TrainRequest) -> Path:
    """Train on ``req.cache``; returns the final LoRA path."""
    model_dir = resolve_model_dir(req.model_dir)
    cache_dir = resolve_under_home(req.cache)
    torch.manual_seed(req.seed)
    random.seed(req.seed)
    torch.cuda.init()
    device = torch.device("cuda")
    lora_dtype = DTYPES[req.lora_dtype]

    items = load_cache(cache_dir)
    if not items:
        raise SystemExit(f"no cached pairs under {cache_dir}")
    image_tokens = sorted(i["latents"].shape[0] for i in items)
    text_lens = sorted(i["prompt_embeds"].shape[0] for i in items)
    sizes = {i["size"] for i in items}
    print(
        f"{len(items)} samples, {len(sizes)} distinct sizes, "
        f"image {image_tokens[0]}-{image_tokens[-1]} tokens, "
        f"text {text_lens[0]}-{text_lens[-1]} tokens",
        flush=True,
    )

    from diffusers import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir / "scheduler")
    print(
        "sigma: logit-normal -> time_shift(mu), mu per sample from its image "
        f"token count ({calculate_shift(image_tokens[0], scheduler.config):.4f}"
        f"-{calculate_shift(image_tokens[-1], scheduler.config):.4f})",
        flush=True,
    )

    transformer = load_transformer(model_dir)
    transformer.requires_grad_(False)
    network = LoRANetwork(
        transformer,
        rank=req.rank,
        alpha=req.alpha,
        targets=req.targets,
        dtype=lora_dtype,
    )
    patched = network.apply_to()
    network.to(device)
    print(
        f"lora: rank {req.rank} alpha {network.alpha} on {patched} linears, "
        f"{network.num_parameters / 1e6:.1f}M params {req.lora_dtype}",
        flush=True,
    )

    if req.grad_checkpointing:

        def checkpointing_func(module, *inputs):
            return torch.utils.checkpoint.checkpoint(
                module.__call__,
                *inputs,
                use_reentrant=False,
                context_fn=blockswap.checkpoint_context_fn,
            )

        transformer.enable_gradient_checkpointing(checkpointing_func)

    attached = place(
        transformer,
        TRANSFORMER_BLOCKS,
        device,
        blocks_to_swap=req.blocks_to_swap,
        supports_backward=True,
        activation_reserve_gb=req.activation_reserve_gb,
        label="transformer",
    )
    blocks = transformer.transformer_blocks
    placement = block_devices(blocks)
    if req.compile:
        # dynamic: the joint sequence moves with both the image and caption size.
        compile_blocks(blocks, mode=req.compile_mode, dynamic=True, decode_only=False)
    empty_cache()
    print(f"free VRAM before training: {free_vram_gb():.2f} GB", flush=True)

    params = list(network.parameters())
    optimizer = torch.optim.AdamW(params, lr=req.lr)
    total_steps = req.epochs * len(items)
    warmup = max(1, int(total_steps * req.warmup_ratio))
    scheduler_lr = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    print(
        f"{req.epochs} epochs x {len(items)} = {total_steps} steps, warmup {warmup}",
        flush=True,
    )

    out_path = resolve_under_home(req.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": "Qwen-Image-2.1",
        "rank": str(req.rank),
        "alpha": str(network.alpha),
        "targets": req.targets,
        "lora_dtype": req.lora_dtype,
        "image_tokens": f"{image_tokens[0]}-{image_tokens[-1]}",
        "epochs": str(req.epochs),
        "samples": str(len(items)),
        "lr": str(req.lr),
    }

    def save(tag: str | None = None) -> Path:
        path = (
            out_path
            if tag is None
            else out_path.with_name(f"{out_path.stem}_{tag}{out_path.suffix}")
        )
        state = {
            k: v.detach().to("cpu").contiguous()
            for k, v in network.state_dict().items()
        }
        save_file(state, path, metadata=metadata)
        return path

    # The largest sample first, so step 1 is the worst case and the fit report
    # below is a bound rather than a sample of the middle of the band.
    order = sorted(
        range(len(items)),
        key=lambda i: items[i]["latents"].shape[0] + items[i]["prompt_embeds"].shape[0],
        reverse=True,
    )
    history = []
    step = 0
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(req.epochs):
        losses = []
        t_epoch = time.time()
        for index in order:
            item = items[index]
            batch = {
                "latents": item["latents"].unsqueeze(0).to(device),
                "prompt_embeds": item["prompt_embeds"].unsqueeze(0).to(device),
                "prompt_embeds_mask": item["prompt_embeds_mask"]
                .unsqueeze(0)
                .to(device),
                "latent_h": item["latent_h"],
                "latent_w": item["latent_w"],
            }
            mu = calculate_shift(item["latents"].shape[0], scheduler.config)
            sigma = sample_sigma(scheduler, mu, req.logit_mean, req.logit_std, device)
            loss = training_step(transformer, batch, sigma)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, req.max_grad_norm)
            optimizer.step()
            scheduler_lr.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
            step += 1
            peak = torch.cuda.max_memory_allocated() / 1024**3
            if step == 1:
                report_fit(blocks, attached, batch)
            # One short line per step for progress readers (the GUI bar); the
            # full line below stays at every 20th step for the log.
            per_step = (time.time() - t_start) / step
            eta = int(per_step * (total_steps - step))
            print(
                f"  progress {step}/{total_steps} epoch {epoch + 1}/{req.epochs} "
                f"loss {sum(losses) / len(losses):.4f} "
                f"eta {eta // 3600}:{eta % 3600 // 60:02d}:{eta % 60:02d}",
                flush=True,
            )
            if step % 20 == 0 or step == 1:
                print(
                    f"  step {step}/{total_steps} loss {loss.item():.4f} "
                    f"|g| {grad_norm.item():.3f} sigma {sigma.item():.3f} "
                    f"lr {scheduler_lr.get_last_lr()[0]:.2e} "
                    f"peak {peak:.2f} GB "
                    f"{(time.time() - t_start) / step:.2f}s/step",
                    flush=True,
                )
        mean = sum(losses) / len(losses)
        drift = block_devices(blocks) != placement
        history.append({"epoch": epoch, "loss": mean, "seconds": time.time() - t_epoch})
        print(
            f"epoch {epoch + 1}/{req.epochs}: loss {mean:.4f}  "
            f"{time.time() - t_epoch:.0f}s  "
            f"peak {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            f"{'  PLACEMENT DRIFTED' if drift else ''}",
            flush=True,
        )
        if req.save_every_epochs and (epoch + 1) % req.save_every_epochs == 0:
            print(f"  saved {save(f'e{epoch + 1}')}", flush=True)

    final = save()
    if req.compile:
        print(recompile_report(), flush=True)
    summary = {
        "output": str(final),
        "metadata": metadata,
        "total_steps": total_steps,
        "minutes": (time.time() - t_start) / 60,
        "history": history,
        "peak_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "blocks_to_swap": attached.offloader.blocks_to_swap if attached else 0,
    }
    report = out_path.with_suffix(".json")
    report.write_text(json.dumps(summary, indent=2))
    print(
        f"wrote {final} and {report} — {summary['minutes']:.1f} min, "
        f"final epoch loss {history[-1]['loss']:.4f}",
        flush=True,
    )

    if attached is not None:
        attached.detach()
    return final
