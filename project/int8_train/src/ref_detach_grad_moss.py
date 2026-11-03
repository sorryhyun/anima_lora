"""swap_text's LoRA gradient with the isolated reference stream attached, under
``no_grad``, and with only its per-block keys and values kept in the graph
(_archive/proposals/step_headroom.md, candidate 4).

Every arm runs one split block: the reference rows through the block with
``kv_out``, then the main rows with ``ref_kv``. Arms:

- ``attached``: the split with nothing cut, the baseline.
- ``detached``: the reference part under ``no_grad``, ``ref_mod`` included.
- ``mod_kv``: the reference ``x`` cut at every block; ``ref_mod`` and the keys and
  values stay in the graph.
- ``joint``: the unpatched block, once, to check the split against it.

States: ``init`` (fresh adapters, only ``up`` has a gradient) and ``e1`` (the epoch
checkpoint). Same noise across arms. The noise floor is the cosine between the mean
gradients of two disjoint pair sets on ``attached``.

    make daemon-run ARGS="--stall-timeout 900 project/int8_train/src/ref_detach_grad_moss.py --lora <e1.safetensors>"

bf16 base, fp32 stream, eager, checkpointed per block: the base did not change
training (docs/moss/methods/i2i.md).
"""

from __future__ import annotations

from library.runtime.allocator import default_expandable_segments

default_expandable_segments()

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import torch  # noqa: E402

from bench._common import start_heartbeat  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.moss import model as moss_model  # noqa: E402
from library.moss.loader import load_transformer, stream_fp32_  # noqa: E402
from library.moss.lora import LoRANetwork  # noqa: E402
from library.moss.train import (  # noqa: E402
    checkpoint_blocks,
    place,
    samples_from_cache,
    step_reserve_gb,
    training_loss,
)
from library.moss.training.losses import edit_region  # noqa: E402

ARMS = ("attached", "detached", "mod_kv")
BLOCK = re.compile(r"layers[._](\d+)[._]proj[._](ffn_up_gate|ffn_down|q|k|v|o)")
_orig = moss_model._Block.forward
_arm = {"name": "attached"}


def split_forward(
    self,
    x,
    rope,
    modulation,
    key_mask=None,
    ref_start=None,
    ref_isolate=False,
    ref_kv=None,
    kv_out=False,
):
    arm = _arm["name"]
    if arm == "joint" or ref_start is None:
        return _orig(
            self, x, rope, modulation, key_mask, ref_start, ref_isolate, ref_kv, kv_out
        )
    assert key_mask is None and modulation.ndim == 4 and ref_isolate
    s = ref_start
    main_mod, ref_mod = modulation[:, :, 0], modulation[:, :, 1]
    xr, rr = x[:, s:], rope[:, s:]
    if arm == "attached":
        xo, k, v = _orig(self, xr, rr, ref_mod, kv_out=True)
    elif arm == "detached":
        with torch.no_grad():
            xo, k, v = _orig(self, xr, rr, ref_mod, kv_out=True)
    else:
        xo, k, v = _orig(self, xr.detach(), rr, ref_mod, kv_out=True)
        xo = xo.detach()
    main = _orig(self, x[:, :s], rope[:, :s], main_mod, ref_kv=(k, v, rr))
    return torch.cat((main, xo), 1)


def group_of(name: str) -> str:
    if "modulation" in name:
        return "mod"
    m = BLOCK.search(name)
    if m is None:
        raise ValueError(f"no group for {name}")
    return f"{m[2]}/{int(m[1]) // 14 * 14:02d}"


def rollups(group: str) -> list[str]:
    return ["all", group] + ([group.split("/")[0]] if "/" in group else [])


def dots(a: dict, b: dict, names: dict) -> dict:
    """Per roll-up ``(a·b, a·a, b·b)``, over the params both hold."""
    out = defaultdict(lambda: [0.0, 0.0, 0.0])
    for n, group in names.items():
        x, y = a[n].double(), b[n].double()
        vals = (float(x.mul(y).sum()), float(x.pow(2).sum()), float(y.pow(2).sum()))
        for g in rollups(group):
            for i in range(3):
                out[g][i] += vals[i]
    return out


def cosines(a: dict, b: dict, names: dict) -> dict:
    return {
        g: {
            "cos": ab / max((aa * bb) ** 0.5, 1e-40),
            "norm_ratio": (bb / max(aa, 1e-40)) ** 0.5,
        }
        for g, (ab, aa, bb) in dots(a, b, names).items()
    }


def pick_pairs(samples, n: int, seed: int):
    full = [s for s in samples if s.n_img >= 3500 and s.n_text <= 600]
    rng = random.Random(seed)
    rng.shuffle(full)
    ko = [s for s in full if s.stem.endswith("_ko")][: n // 2]
    en = [s for s in full if s.stem.endswith("_en")][: n - len(ko)]
    return [x for pair in zip(ko, en) for x in pair]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument(
        "--cache", default="post_image_dataset/moss/i2i_swap_text_v3_en400_ko1709"
    )
    ap.add_argument("--n_pairs", type=int, default=6)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.6, 0.9])
    ap.add_argument("--blocks_to_swap", type=int, default=20)
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default="project/int8_train/assets/moss_ref_detach_grad.json",
    )
    args = ap.parse_args()
    start_heartbeat(label="ref_detach")
    device = torch.device("cuda")

    pairs = pick_pairs(
        samples_from_cache(resolve_under_home(args.cache)), args.n_pairs, args.seed
    )
    print([(s.stem, s.n_img, s.n_text, s.n_ref) for s in pairs], flush=True)
    joint = max(s.n_img + s.n_ref + s.n_text for s in pairs)

    moss_model._Block.forward = split_forward
    tr = load_transformer(args.model_dir, "base")
    tr.requires_grad_(False)
    stream_fp32_(tr)
    tr.ref_timestep, tr.ref_isolate = 0.0, True
    lora = resolve_under_home(args.lora)
    network, meta = LoRANetwork.from_file(tr, lora, dtype=torch.float32)
    trained = {k: v.clone() for k, v in network.state_dict().items()}
    torch.manual_seed(args.seed)
    fresh = LoRANetwork(
        tr,
        rank=int(meta["rank"]),
        alpha=float(meta["alpha"]),
        targets=meta["targets"],
        dtype=torch.float32,
        rank_pattern=json.loads(meta["rank_pattern"]),
    ).state_dict()
    network.apply_to()
    network.to(device)
    network.requires_grad_(True)
    params = {n: p for n, p in network.named_parameters() if p.requires_grad}
    print(len(params), "adapter params, e.g.", list(params)[:2], flush=True)
    names = {n: group_of(n) for n in params}
    place(
        tr,
        device,
        args.blocks_to_swap,
        step_reserve_gb(joint, grad_checkpointing=True) + 1.5,
    )
    checkpoint_blocks(tr.layers)

    def one_pass(arm, s, latent, embeds, ref, region, sigma, noise_seed):
        _arm["name"] = arm
        network.zero_grad(set_to_none=True)
        gen = torch.Generator(device).manual_seed(noise_seed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = training_loss(
            tr,
            latent,
            embeds,
            sigma,
            gen=gen,
            ref=ref,
            region=region,
            region_weight=2.0,
            parts={},
        )
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        grads = {
            n: p.grad.detach().float().cpu()
            if p.grad is not None
            else torch.zeros_like(p, device="cpu")
            for n, p in params.items()
        }
        return grads, loss.item(), t1 - t0, t2 - t1

    report = {"args": vars(args), "pairs": [s.stem for s in pairs], "states": {}}
    for state, sd in (("init", fresh), ("e1", trained)):
        network.load_state_dict(sd)
        sums = {
            arm: [
                {n: torch.zeros_like(p, device="cpu") for n, p in params.items()}
                for _ in (0, 1)
            ]
            for arm in ARMS
        }
        per = defaultdict(lambda: defaultdict(list))
        times = defaultdict(list)
        check = None
        for i, s in enumerate(pairs):
            latent, embeds = s.load()
            ref = s.load_ref()
            region = edit_region(latent, ref, 0.3, 5, 1)
            latent = latent.unsqueeze(0).to(device, torch.float32)
            embeds = embeds.unsqueeze(0).to(device)
            ref = ref.unsqueeze(0).to(device, torch.float32)
            region = region.to(device)
            for j, sigma in enumerate(args.sigmas):
                seed = args.seed + 1000 * i + j
                base = None
                for arm in ARMS:
                    g, loss, fwd, bwd = one_pass(
                        arm, s, latent, embeds, ref, region, sigma, seed
                    )
                    times[arm].append((fwd, bwd))
                    half = sums[arm][i % 2]
                    for n in g:
                        half[n] += g[n]
                    if base is None:
                        base = g
                    else:
                        for k, v in cosines(base, g, names).items():
                            per[arm][k].append(v["cos"])
                    print(
                        f"{state} {s.stem} σ{sigma} {arm}: loss {loss:.5f} "
                        f"fwd {fwd:.2f} bwd {bwd:.2f} peak "
                        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
                        flush=True,
                    )
                if check is None:
                    g, *_ = one_pass(
                        "joint", s, latent, embeds, ref, region, sigma, seed
                    )
                    check = cosines(base, g, names)["all"]
                    print(f"{state}: split vs joint {check}", flush=True)
        mean = {
            arm: {n: sums[arm][0][n] + sums[arm][1][n] for n in params} for arm in ARMS
        }
        result = {
            "split_vs_joint": check,
            "noise_floor": cosines(sums["attached"][0], sums["attached"][1], names),
        }
        for arm in ARMS[1:]:
            result[arm] = {
                "mean_grad": cosines(mean["attached"], mean[arm], names),
                "per_sample_median_cos": {
                    k: statistics.median(v) for k, v in per[arm].items()
                },
            }
        result["median_s"] = {
            arm: {
                "fwd": statistics.median(t[0] for t in times[arm]),
                "bwd": statistics.median(t[1] for t in times[arm]),
            }
            for arm in ARMS
        }
        report["states"][state] = result
        out = resolve_under_home(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        for g in ("all", "mod", "q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            print(
                state,
                g,
                f"floor {result['noise_floor'][g]['cos']:.4f}",
                *(
                    f"{arm} cos {result[arm]['mean_grad'][g]['cos']:.4f} "
                    f"norm {result[arm]['mean_grad'][g]['norm_ratio']:.3f} "
                    f"sample {result[arm]['per_sample_median_cos'][g]:.4f}"
                    for arm in ARMS[1:]
                ),
                flush=True,
            )
        print(state, result["median_s"], flush=True)


if __name__ == "__main__":
    main()
