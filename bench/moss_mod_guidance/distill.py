"""Micro distill of a pooled-text head for MOSS-Image modulation guidance
(Starodubcev et al., ICLR 2026, §5 "CLIP-free models").

The head maps the mean of the caption's content-token TE rows to an offset on the
timestep embedding ``time``. ``shared_block_modulation`` and ``video_head_modulation``
are affine in ``time``, so the offset enters every block's modulation as
``W_mod · offset`` (``attach_head`` patches ``time_conditioning``; model.py is
untouched). Teacher: the frozen DiT on the real caption, no head. Student: the same
DiT on the empty prompt plus the head on the caption's pool. Loss: MSE between the
two velocity predictions.

Phase 1 runs every teacher forward with the DiT fully on the card (inference
placement) and keeps ``(noisy, σ, v_teacher)`` in RAM; phase 2 re-places the DiT with
the training block swapper and trains the head. One step per sample, no repeats.

    make daemon-run ARGS="--stall-timeout 900 bench/moss_mod_guidance/distill.py"
"""

import argparse
import json
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from bench._common import make_run_dir, write_result
from library.env import resolve_under_home

DEFAULT_CACHES = [
    "post_image_dataset/moss/ama_mitsuki_layout",
    "post_image_dataset/moss/channel_(caststation)_layout",
    "post_image_dataset/moss/group_wsw_layout",
    "post_image_dataset/moss/hews_layout",
    "post_image_dataset/moss/sincos_layout",
]
TEXT_DIM, DIM = 4096, 3072


class PooledHead(nn.Module):
    """``pool (B, 4096)`` → ``time`` offset ``(B, 3072)``; the output layer starts at
    zero, so an untrained head is the base model."""

    def __init__(self, text_dim=TEXT_DIM, dim=DIM):
        super().__init__()
        self.norm = nn.LayerNorm(text_dim, elementwise_affine=False)
        self.fc1 = nn.Linear(text_dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, pool):
        return self.fc2(F.silu(self.fc1(self.norm(pool.float()))))


def template_span(model_dir):
    """``(n_head, n_tail)``: chat-template tokens around the user text."""
    from library.moss.loader import load_tokenizer

    tok = load_tokenizer(model_dir)
    sentinel = "⁣SPAN⁣"
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    head, tail = formatted.split(sentinel)
    return (
        len(tok(head, add_special_tokens=False).input_ids),
        len(tok(tail, add_special_tokens=False).input_ids),
    )


def content_pool(embeds, span):
    """Mean of the user-text rows of one ``(S, 4096)`` TE output → ``(4096,)`` fp32."""
    a, b = span
    rows = embeds[a : embeds.shape[0] - b]
    return (rows if rows.shape[0] else embeds).float().mean(0)


def attach_head(transformer):
    """Patch ``time_conditioning`` to add ``W · transformer._time_offset`` (``(B, D)``
    or None) to the block and head modulation. ``transformer._block_steer`` (a
    ``(6, D)`` tensor and a block set, or None) is added to those blocks only."""
    orig = transformer.time_conditioning
    transformer._time_offset = None
    transformer._block_steer = None

    def time_conditioning(timestep, batch):
        mod, head = orig(timestep, batch)
        off = transformer._time_offset
        if off is None:
            return mod, head
        w_mod = transformer.shared_block_modulation.weight
        w_head = transformer.video_head_modulation.weight
        off = off.expand(batch, -1)
        mod = mod + F.linear(off.to(w_mod.dtype), w_mod).view(batch, 6, -1).to(
            mod.dtype
        )
        head = head + F.linear(off.to(w_head.dtype), w_head).view(batch, 2, -1).to(
            head.dtype
        )
        return mod, head

    transformer.time_conditioning = time_conditioning
    for index, layer in enumerate(transformer.layers):
        inner = layer.forward

        def forward(x, rope, modulation, *args, _inner=inner, _index=index, **kwargs):
            steer = transformer._block_steer
            if steer is not None and _index in steer[1]:
                d = steer[0].to(modulation.dtype)
                modulation = modulation + (d if modulation.ndim == 3 else d[:, None])
            return _inner(x, rope, modulation, *args, **kwargs)

        layer.forward = forward


def steer_mod(transformer, delta):
    """``(6, D)`` block-modulation shift of a ``time`` offset ``delta (D,)``."""
    w = transformer.shared_block_modulation.weight
    return F.linear(delta.to(w.dtype)[None], w).view(6, -1)


def pick_samples(caches, n, seed):
    from library.moss.train import samples_from_cache

    pool = []
    for c in caches:
        got = samples_from_cache(resolve_under_home(c))
        print(f"[data] {len(got):4d} samples in {c}", flush=True)
        pool += got
    random.Random(seed).shuffle(pool)
    return pool[:n]


def main():
    from library.moss.requests import TrainRequest

    d = TrainRequest()
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--caches", default=",".join(DEFAULT_CACHES))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--blocks_to_swap", type=int, default=36)
    p.add_argument("--activation_memory_budget", type=float, default=0.5)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="base")
    p.add_argument("--model_dir", default="")
    p.add_argument(
        "--output", default="output/ckpt_moss/mod_guidance/pooled_head.safetensors"
    )
    p.add_argument("--label", default=None)
    args = p.parse_args()

    from safetensors.torch import save_file

    from library.moss.loader import compile_blocks, empty_cache, load_transformer
    from library.moss.train import (
        NUM_TRAIN_TIMESTEPS,
        empty_prompt_embeds,
        place,
        sample_sigma,
        set_activation_memory_budget,
        step_reserve_gb,
    )

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda")
    samples = pick_samples(args.caches.split(","), args.steps, args.seed)
    span = template_span(args.model_dir)
    empty = empty_prompt_embeds(TrainRequest(model_dir=args.model_dir))
    print(
        f"[data] {len(samples)} steps, template span {span}, empty prompt "
        f"{empty.shape[0]} tokens",
        flush=True,
    )

    # Phase 1: teacher targets, DiT resident.
    transformer = load_transformer(args.model_dir, args.model, device=device)
    transformer.requires_grad_(False)
    items, t0 = [], time.time()
    with torch.no_grad():
        for i, s in enumerate(samples):
            latent, embeds = s.load()
            latent = latent[None].to(device, torch.float32)
            sigma = sample_sigma(
                gen, d.sigma_mu, d.sigma_s, d.sigma_cutoff, d.sigma_uniform
            )
            noise = torch.randn(latent.shape, generator=gen).to(device)
            noisy = (1.0 - sigma) * latent + sigma * noise
            ts = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
            v = transformer(noisy, ts, embeds[None].to(device), return_dict=False)[0]
            v_u = transformer(noisy, ts, empty[None].to(device), return_dict=False)[0]
            items.append(
                dict(
                    stem=s.stem,
                    noisy=noisy.cpu(),
                    sigma=sigma,
                    v=v.float().cpu(),
                    floor=F.mse_loss(v_u.float(), v.float()).item(),
                    pool=content_pool(embeds, span),
                    n_img=s.n_img,
                )
            )
            if i % 50 == 0:
                print(
                    f"[teacher] {i}/{len(samples)} {time.time() - t0:.0f}s", flush=True
                )
    print(f"[teacher] done in {time.time() - t0:.0f}s", flush=True)
    transformer.to("cpu")
    empty_cache()

    # Phase 2: student, training placement.
    attach_head(transformer)
    head = PooledHead().to(device)
    joint = max(it["n_img"] for it in items) + empty.shape[0]
    budgeted = not args.no_compile
    reserve = step_reserve_gb(
        joint, grad_checkpointing=False, budget=args.activation_memory_budget
    )
    attached = place(transformer, device, args.blocks_to_swap, reserve, True)
    if budgeted:
        set_activation_memory_budget(args.activation_memory_budget)
        lo = min(it["n_img"] for it in items) + empty.shape[0]
        compile_blocks(transformer.layers, (lo, joint))
    params = list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warm = max(1, int(args.steps * args.warmup))

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        f = (step - warm) / max(1, len(items) - warm)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * f))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    empty_dev = empty[None].to(device)
    losses, ratios, t0 = [], [], time.time()
    for step, it in enumerate(items):
        noisy = it["noisy"].to(device)
        ts = torch.tensor([NUM_TRAIN_TIMESTEPS * it["sigma"]], device=device)
        transformer._time_offset = head(it["pool"][None].to(device))
        pred = transformer(noisy, ts, empty_dev, return_dict=False)[0]
        loss = F.mse_loss(pred.float(), it["v"].to(device))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm).item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
        ratios.append(loss.item() / it["floor"])
        if step % 10 == 0 or step == len(items) - 1:
            el = time.time() - t0
            print(
                f"[student] {step}/{len(items)} loss {loss.item():.4f} "
                f"/ floor {it['floor']:.4f}, ratio avg10 "
                f"{sum(ratios[-10:]) / len(ratios[-10:]):.3f} gn {gn:.3f} "
                f"lr {sched.get_last_lr()[0]:.2e} "
                f"|off| {transformer._time_offset.norm().item():.3f} {el:.0f}s",
                flush=True,
            )
    transformer._time_offset = None
    if attached is not None:
        attached.detach()

    out = resolve_under_home(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "pool": "content_mean",
        "span": json.dumps(list(span)),
        "steps": str(len(items)),
        "lr": str(args.lr),
        "base_variant": args.model,
        "caches": args.caches,
    }
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in head.state_dict().items()},
        str(out),
        meta,
    )
    first = sum(ratios[:20]) / min(20, len(ratios))
    last = sum(ratios[-50:]) / min(50, len(ratios))
    metrics = {"ratio_first20": first, "ratio_last50": last}
    run_dir = make_run_dir("moss_mod_guidance", args.label or "distill")
    (run_dir / "losses.json").write_text(
        json.dumps(
            [
                {
                    "stem": it["stem"],
                    "sigma": it["sigma"],
                    "loss": lo,
                    "floor": it["floor"],
                }
                for it, lo in zip(items, losses)
            ]
        )
    )
    write_result(
        run_dir, script=__file__, args=args, metrics=metrics, artifacts=["losses.json"]
    )
    print(
        f"[done] head -> {out}; loss/floor {first:.3f} (first 20) -> {last:.3f} (last 50)"
    )
    print(f"[done] {run_dir}")


if __name__ == "__main__":
    main()
