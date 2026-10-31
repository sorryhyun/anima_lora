"""Time block swap against the activation budget on this machine.

Without checkpointing, every swapped block frees VRAM that the compiled
blocks' saved set can use, so the budget rises and the backward recomputes
less; past some swap count the PCIe transfers stop hiding behind compute and
the step slows again. Where that knee sits depends on the link (generation,
lanes) and on the GPU, so this measures it instead of assuming the 16 GB
figures. It times the bf16 base; the int8 plan's swap only trades VRAM.

One process loads the DiT once. Each arm places it with ``N`` blocks swapped,
reads the free VRAM, picks the largest budget :func:`train.budget_for_free`
allows, recompiles, and times ``steps`` training steps on the largest sample.
An OOM retries one budget step lower. The sweep stops ``patience`` arms past
the fastest so far. The recommendation is the least swap within
``tolerance`` of the fastest arm that still leaves ``margin_gb`` free.

    make daemon-run ARGS="scripts/moss/tune_swap.py --cache post_image_dataset/moss/<set>"
"""

from __future__ import annotations

import json
import subprocess
import time

import torch

from library.env import resolve_under_home
from library.moss.loader import (
    compile_blocks,
    empty_cache,
    free_vram_gb,
    load_transformer,
)
from library.moss.lora import LoRANetwork
from library.moss.requests import TrainRequest, TuneSwapRequest
from library.moss.train import (
    Sample,
    budget_for_free,
    checkpoint_blocks,
    place,
    samples_from_cache,
    set_activation_memory_budget,
    training_loss,
)


def synthetic_samples(sizes: str) -> list[Sample]:
    """``WxH:text_tokens`` comma list → random latents / embeddings."""
    samples = []
    for i, spec in enumerate(sizes.split(",")):
        size, n_text = spec.split(":")
        w, h = map(int, size.split("x"))
        n_text = int(n_text)

        def load(w=w, h=h, n_text=n_text, i=i):
            g = torch.Generator().manual_seed(i)
            latent = torch.randn(128, h // 16, w // 16, generator=g)
            embeds = torch.randn(n_text, 4096, generator=g).to(torch.bfloat16)
            return latent, embeds

        samples.append(
            Sample(f"synthetic{i}_{w}x{h}", (w // 16) * (h // 16), n_text, load)
        )
    return samples


def pick(arms: list[dict], margin_gb: float, tolerance: float = 0.02) -> dict | None:
    """Among finished arms that leave ``margin_gb`` free at their peak, the
    least swap within ``tolerance`` of the fastest: below the PCIe knee the
    curve is flat to within timing noise, and fewer swapped blocks keep the
    run further from the knee."""
    ok = [a for a in arms if a.get("s_per_step") and a["free_gb"] >= margin_gb]
    if not ok:
        return None
    fastest = min(a["s_per_step"] for a in ok)
    close = [a for a in ok if a["s_per_step"] <= fastest * (1 + tolerance)]
    return min(close, key=lambda a: (a["blocks_to_swap"], a["s_per_step"]))


def flags(arm: dict) -> str:
    """The arm as flags on the bf16 plan, which is what this times."""
    base = "--int8 off --stream bf16 --no-ema_card"
    if arm["grad_checkpointing"]:
        return f"{base} --grad_checkpointing --blocks_to_swap 0"
    return (
        f"{base} --blocks_to_swap {arm['blocks_to_swap']} "
        f"--activation_memory_budget {arm['budget']:g}"
        + (" --swap_h2d_only" if arm.get("swap_h2d_only") else "")
    )


def _pcie_link() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=pcie.link.gen.max,pcie.link.width.max",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    gen, _, width = out.partition(",")
    return f"gen {gen.strip()} x{width.strip()}" if width else out


def _unwrap(blocks) -> None:
    """Drop the compile / checkpoint wrappers (instance ``forward`` attributes)."""
    for block in blocks:
        block.__dict__.pop("forward", None)


class _Bench:
    def __init__(self, req: TuneSwapRequest, transformer, network, sample, device):
        self.req = req
        self.transformer = transformer
        self.params = list(network.parameters())
        # Every arm starts from these, so its first step is comparable across arms.
        self.init = [p.detach().clone() for p in self.params]
        latent, embeds = sample.load()
        self.latent = latent.unsqueeze(0).to(device, torch.float32)
        self.embeds = embeds.unsqueeze(0).to(device)
        ref = sample.load_ref() if sample.load_ref is not None else None
        self.ref = None if ref is None else ref.unsqueeze(0).to(device, torch.float32)
        self.joint = sample.n_img + sample.n_ref + sample.n_text
        self.device = device

    def arm(self, swap: int, *, grad_checkpointing: bool = False) -> dict:
        """Place, budget, compile, time; always leaves the DiT back on CPU."""
        req = self.req
        result = {
            "blocks_to_swap": swap,
            "grad_checkpointing": grad_checkpointing,
            "swap_h2d_only": req.swap_h2d_only and not grad_checkpointing and swap > 0,
        }
        attached = place(
            self.transformer,
            self.device,
            swap,
            reserve_gb=0.0,
            h2d_only=req.swap_h2d_only and not grad_checkpointing,
        )
        try:
            empty_cache()
            free = free_vram_gb()
            budget = (
                1.0
                if grad_checkpointing
                else budget_for_free(free, self.joint, req.budget_step)
            )
            result["free_placed_gb"] = round(free, 3)
            while budget > 0:
                result["budget"] = budget
                try:
                    result.update(self._time(budget, grad_checkpointing))
                    return result
                except torch.OutOfMemoryError:
                    pass
                # Out of the except block, so the traceback's tensors are gone.
                print(f"  OOM at budget {budget:g}", flush=True)
                result.setdefault("oom_budgets", []).append(budget)
                for p in self.params:
                    p.grad = None
                empty_cache()
                if grad_checkpointing:
                    break
                budget = round(budget - req.budget_step, 6)
            result["skipped"] = "does not fit"
            return result
        finally:
            self._restore(attached)

    def _time(self, budget: float, grad_checkpointing: bool) -> dict:
        req = self.req
        blocks = self.transformer.layers
        _unwrap(blocks)
        set_activation_memory_budget(budget)
        torch._dynamo.reset()
        compile_blocks(blocks)
        if grad_checkpointing:
            checkpoint_blocks(blocks)
        with torch.no_grad():
            for p, p0 in zip(self.params, self.init):
                p.copy_(p0)
        optimizer = torch.optim.AdamW(self.params, lr=1e-6)
        gen = torch.Generator().manual_seed(req.seed)
        torch.manual_seed(req.seed)
        torch.cuda.reset_peak_memory_stats()
        start = 0.0
        first = {}
        for i in range(req.warmup + req.steps):
            if i == req.warmup:
                torch.cuda.synchronize()
                start = time.perf_counter()
            sigma = 0.25 + 0.5 * torch.rand((), generator=gen).item()
            loss = training_loss(
                self.transformer, self.latent, self.embeds, sigma, ref=self.ref
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            if i == 0:
                # Same weights, noise and sigma in every arm: a swap mode that
                # changes the numerics shows up here.
                first = {"step0_loss": loss.item(), "step0_grad_norm": norm.item()}
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return {
            "s_per_step": round((time.perf_counter() - start) / req.steps, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            # The allocator keeps its peak reserved until empty_cache, so this
            # is what the step left on the card.
            "free_gb": round(free_vram_gb(), 3),
            **first,
        }

    def _restore(self, attached) -> None:
        _unwrap(self.transformer.layers)
        set_activation_memory_budget(1.0)
        torch._dynamo.reset()
        if attached is not None:
            attached.detach()
        else:
            self.transformer.to("cpu")
        empty_cache()


def _row(arm: dict) -> str:
    name = "ckpt" if arm["grad_checkpointing"] else f"swap {arm['blocks_to_swap']:>2}"
    if "s_per_step" not in arm:
        return f"  {name:8s} {arm.get('skipped', '')}"
    return (
        f"  {name:8s} budget {arm['budget']:<5g} {arm['s_per_step']:.3f} s/step  "
        f"peak {arm['peak_reserved_gb']:.2f} GB reserved  {arm['free_gb']:.2f} GB free  "
        f"step0 loss {arm['step0_loss']:.6f} |g| {arm['step0_grad_norm']:.6f}"
    )


def run_tune(req: TuneSwapRequest) -> dict:
    torch.manual_seed(req.seed)
    device = torch.device("cuda")
    if req.sizes:
        samples = synthetic_samples(req.sizes)
    else:
        samples = samples_from_cache(resolve_under_home(req.cache))
        if not samples:
            raise SystemExit(f"no cached pairs under {req.cache}")
    sample = max(samples, key=lambda s: s.n_img + s.n_ref + s.n_text)
    print(
        f"timing {sample.stem}: {sample.n_img} image + {sample.n_ref} ref + "
        f"{sample.n_text} text tokens "
        f"(largest of {len(samples)})",
        flush=True,
    )

    transformer = load_transformer(req.model_dir, req.model)
    transformer.requires_grad_(False)
    network = LoRANetwork(
        transformer, rank=req.rank, targets=TrainRequest.TARGETS, dtype=torch.float32
    )
    network.apply_to()
    network.to(device)
    max_swap = len(transformer.layers) - 2
    bench = _Bench(req, transformer, network, sample, device)

    arms: list[dict] = []
    if req.reference:
        arms.append(bench.arm(0, grad_checkpointing=True))
        print(_row(arms[-1]), flush=True)
    best, behind = None, 0
    for swap in sorted({min(int(n), max_swap) for n in req.swap_grid.split(",")}):
        arm = bench.arm(swap)
        arms.append(arm)
        print(_row(arm), flush=True)
        if "s_per_step" not in arm:
            continue  # too little swap to fit: more swap may
        if best is None or arm["s_per_step"] < best["s_per_step"]:
            best, behind = arm, 0
        else:
            behind += 1
            if behind >= req.patience:
                break

    chosen = pick(arms, req.margin_gb, req.tolerance)
    report = {
        "gpu": torch.cuda.get_device_name(device),
        "pcie": _pcie_link(),
        "swap_h2d_only": req.swap_h2d_only,
        "image_tokens": sample.n_img,
        "text_tokens": sample.n_text,
        "rank": req.rank,
        "margin_gb": req.margin_gb,
        "tolerance": req.tolerance,
        "arms": arms,
        "recommended": chosen,
        "flags": flags(chosen) if chosen else None,
    }
    out = resolve_under_home(req.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(
        f"\n{report['gpu']} ({report['pcie']}), {sample.n_img + sample.n_ref + sample.n_text} joint tokens"
    )
    for arm in arms:
        print(_row(arm) + ("   <- recommended" if arm is chosen else ""))
    if chosen:
        print(f"\nrecommended: {flags(chosen)}")
    else:
        print(
            f"\nno arm left {req.margin_gb} GB free; lower --margin_gb or train smaller"
        )
    print(f"wrote {out}", flush=True)
    return report
