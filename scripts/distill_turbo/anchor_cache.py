"""Anchor target cache (``dpdmd.anchor_cache``): build it, and read it in the loop.

The anchor target ``v_target`` depends only on (prompt, ε, latent shape) and the frozen
teacher, so it is rolled once and reused. A cache holds ``draws`` pairs for each of a
fixed set of samples of ``data_dir``; pair ``(sample, k)`` uses an ε regenerated from a
seed hashed from ``(seed, sample key, k)``, so only ``v_target`` is stored.

Build (a daemon job; resumable, skips pairs already on disk). The builder loads the
frozen teacher alone and takes no run config::

    python -m scripts.distill_turbo.anchor_cache --out <dir> --prompts 64 --draws 2 \
        [--dit_path … --k_anchor 7 --teacher_anchor_steps 28 --teacher_cfg 4 …]

``--k_anchor`` of ``--teacher_anchor_steps`` sets the σ the chord ends on; it has to be
the first boundary σ_1 of the student grid the run trains (7 of 28 for 4 steps at
shift 3, 14 of 28 for 2 steps).

A run trains on it with ``dpdmd.anchor_cache = "<dir>"``: the loop keeps only the cached
samples of its ``data_dir``, ``AnchorDataset`` adds each sample's pair to the DataLoader
item, and epoch e reads draw e mod K.

``manifest.json`` records the teacher the targets were rolled with, the ε seed and the
sample keys.

The builder is no-grad, so an int8 teacher loads straight from the int8 checkpoint as
``infer`` linears (``library.anima.int8.infer_load_file``); every int8 mode runs the
same forward copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from library.runtime.proc import no_window_main
from library.datasets.cache import make_cached_collate
from library.env import resolve_under_home
from library.io.cache_names import TE_CACHE_SUFFIX

logger = logging.getLogger(__name__)

FORMAT = 2
MANIFEST = "manifest.json"
_base_collate = make_cached_collate()

# Teacher fields the targets depend on, recorded in the manifest.
TEACHER_FIELDS = (
    "dit_path",
    "dit_nonquant_path",
    "int8",
    "int8_file",
    "stream",
    "teacher_cfg",
    "flow_shift",
    "k_anchor",
    "teacher_anchor_steps",
)


def teacher_signature(src) -> dict:
    return {f: getattr(src, f) for f in TEACHER_FIELDS}


def sample_key(te_path: str, data_dir: str) -> str:
    """Stable per-sample key: the TE cache path under ``data_dir``, suffix dropped."""
    rel = os.path.relpath(te_path, data_dir).removesuffix(TE_CACHE_SUFFIX)
    return rel.replace(os.sep, "__")


def key_stem(key: str) -> str:
    return key.rsplit("__", 1)[-1]


def pair_seed(seed: int, key: str, k: int) -> int:
    h = hashlib.blake2b(f"{seed}:{key}:{k}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") & ((1 << 63) - 1)


def pair_eps(shape, seed: int, *, device, dtype) -> torch.Tensor:
    """ε for one pair: fp32 CPU draw, then cast (device-independent)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(tuple(shape), generator=g, dtype=torch.float32).to(
        device=device, dtype=dtype
    )


def pair_path(cache_dir: Path, key: str, k: int) -> Path:
    return cache_dir / f"{key}.d{k}.safetensors"


def read_manifest(cache_dir) -> dict:
    path = Path(resolve_under_home(cache_dir)) / MANIFEST
    if not path.exists():
        raise FileNotFoundError(
            f"dpdmd.anchor_cache: no {MANIFEST} under {path.parent}; build it with "
            "`python -m scripts.distill_turbo.anchor_cache --out <dir> …`."
        )
    man = json.loads(path.read_text())
    if man.get("format") != FORMAT:
        raise ValueError(
            f"dpdmd.anchor_cache {path.parent}: format {man.get('format')}, "
            f"this loop reads {FORMAT}."
        )
    return man


def keep_stems(cache_dir) -> set[str]:
    """Stems of the cached samples, for ``CachedDataset(keep_list=…)``."""
    return {key_stem(k) for k in read_manifest(cache_dir)["keys"]}


class AnchorDataset(torch.utils.data.Dataset):
    """``CachedDataset`` wrapper whose items also carry the sample's anchor pair.

    ``base`` holds exactly the cached samples (built with ``keep_stems``). Each item
    gains ``anchor_eps`` (fp32, regenerated from the pair seed) and ``anchor_v`` (the
    cached ``v_target``) for draw ``epoch % draws``. The loop calls ``next_epoch()``
    before it re-iterates the DataLoader; workers are re-created per ``iter()``, so
    they pick the new draw up.
    """

    def __init__(self, base, cache_dir, cfg, *, epoch: int = 0):
        self.base = base
        self.dir = Path(resolve_under_home(cache_dir))
        man = read_manifest(self.dir)
        # The targets were rolled to the cache's own chord; a run that states another
        # one would train on it and stamp the wrong depth.
        rolled = man.get("teacher", {})
        differs = {
            f: (rolled[f], getattr(cfg, f))
            for f in ("k_anchor", "teacher_anchor_steps", "flow_shift", "teacher_cfg")
            if f in rolled and rolled[f] != getattr(cfg, f)
        }
        if differs:
            raise ValueError(
                f"dpdmd.anchor_cache {self.dir}: the cache was rolled with other "
                "teacher settings than this run's (cache, run): "
                + ", ".join(f"{f} {a} / {b}" for f, (a, b) in differs.items())
            )
        self.draws = int(man["draws"])
        self.seed = int(man["seed"])
        self.keys = [sample_key(te, cfg.data_dir) for _, te in base.samples]
        cached = set(man["keys"])
        if set(self.keys) != cached:
            extra = sorted(set(self.keys) - cached)
            absent = sorted(cached - set(self.keys))
            raise ValueError(
                f"dpdmd.anchor_cache {self.dir}: the dataset under {cfg.data_dir} "
                f"does not match the cache's {len(cached)} samples "
                f"(not cached: {extra[:3]}, not in data_dir: {absent[:3]})."
            )
        missing = [
            (key, k)
            for key in self.keys
            for k in range(self.draws)
            if not pair_path(self.dir, key, k).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"dpdmd.anchor_cache {self.dir}: {len(missing)} of "
                f"{len(self.keys) * self.draws} pairs missing (first {missing[0]}); "
                "rerun the builder, it resumes."
            )
        self.set_epoch(epoch)
        logger.info(
            f"anchor cache: {self.dir}, {len(self.keys)} samples × {self.draws} draws, "
            f"starting at epoch {epoch} (draw {self.draw})"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.draw = epoch % self.draws

    def next_epoch(self) -> None:
        self.set_epoch(self.epoch + 1)

    @property
    def samples(self):
        return self.base.samples

    def make_batch_sampler(self, **kw):
        return self.base.make_batch_sampler(**kw)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]
        key, k = self.keys[idx], self.draw
        v = load_file(str(pair_path(self.dir, key, k)))["v_target"]
        if v.shape != out["latents"].shape:
            raise ValueError(
                f"anchor pair {key} d{k}: v_target {tuple(v.shape)} against latent "
                f"{tuple(out['latents'].shape)}; the cache is stale for this data_dir."
            )
        out["anchor_eps"] = pair_eps(
            v.shape, pair_seed(self.seed, key, k), device="cpu", dtype=torch.float32
        )
        out["anchor_v"] = v.float()
        return out


def anchor_collate(batch):
    """``make_cached_collate()``'s batch plus the stacked ``anchor_eps`` / ``anchor_v``."""
    out = _base_collate(batch)
    out["anchor_eps"] = torch.stack([b["anchor_eps"] for b in batch])
    out["anchor_v"] = torch.stack([b["anchor_v"] for b in batch])
    return out


# --- builder -------------------------------------------------------------------


def _choose_keys(data_dir: str, n: int, seed: int) -> list[str]:
    from library.io.cache import discover_cached_pairs

    keys = sorted(
        sample_key(c.te_path, data_dir) for c in discover_cached_pairs(data_dir)
    )
    if not keys:
        raise SystemExit(f"no latent + TE cache pairs under {data_dir}")
    if n <= 0 or n >= len(keys):
        return keys
    return sorted(random.Random(seed).sample(keys, n))


def _grow_keys(data_dir: str, held: list[str], n: int, seed: int) -> list[str]:
    """``held`` plus samples of ``data_dir`` not in it, up to ``n`` in all (0: all).
    Pair seeds depend only on (seed, key, draw), so the held pairs stay valid."""
    rest = sorted(set(_choose_keys(data_dir, 0, seed)) - set(held))
    if n > 0:
        if n < len(held):
            raise SystemExit(
                f"--grow --prompts {n}: the cache already holds {len(held)}"
            )
        rest = random.Random(seed).sample(rest, min(n - len(held), len(rest)))
    return sorted(set(held) | set(rest))


def _load_teacher(a, device, dtype, token_counts):
    """The frozen DiT the loop's teacher view runs: int8 base, fp32 stream, compiled."""
    from library.anima import weights as anima_utils
    from library.anima.int8 import convert_for_turbo, infer_load_file
    from library.anima.stream import stream_fp32_
    from library.runtime.dynamo import pin_dynamo_limit

    from .setup import compile_turbo_dit

    logger.info(f"loading teacher DiT: {a.dit_path}")
    infer_file = infer_load_file(a.int8_file, a.dit_path) if a.int8 != "off" else ""
    model = anima_utils.load_anima_model(
        device,
        a.dit_path,
        attn_mode=a.attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
        dit_nonquant_path=a.dit_nonquant_path,
        int8_infer_file=infer_file or None,
    )
    model.to(device)
    model.requires_grad_(False)
    if a.int8 != "off" and not infer_file:
        convert_for_turbo(model, a.int8, a.int8_file, a.dit_path)
    if a.stream == "fp32":
        stream_fp32_(model)
    model.prepare_block_swap_before_forward(free_cache=False)
    if a.compile:
        pin_dynamo_limit("recompile_limit", 64)
    compile_turbo_dit(
        model,
        enabled=a.compile,
        dynamic_seq=True,
        n_token_families=len(token_counts),
        seq_range=(min(token_counts), max(token_counts)),
        int8=a.int8,
        stream=a.stream,
    )
    return model


def _build(a) -> None:
    from library.anima.stream import latent_dtype
    from library.anima.uncond import (
        default_uncond_path,
        load_uncond_crossattn,
        uncond_for_batch,
    )
    from library.datasets.cache import CachedDataset
    from library.inference.sampling import get_timesteps_sigmas
    from library.runtime.progress import ProgressSink, command_job_path
    from library.training.forward import PadCache

    from .setup import latent_token_counts
    from .steps import anchor_target

    cache_dir = Path(resolve_under_home(a.out))
    cache_dir.mkdir(parents=True, exist_ok=True)
    man_path = cache_dir / MANIFEST
    teacher = teacher_signature(a)

    if man_path.exists():
        man = read_manifest(cache_dir)
        teacher = man["teacher"]
        if man["seed"] != a.seed or man["data_dir"] != a.data_dir:
            raise SystemExit(
                f"{cache_dir}: seed / data_dir ({man['seed']}, {man['data_dir']}) "
                f"against ({a.seed}, {a.data_dir})"
            )
        keys = man["keys"]
        if a.grow:
            keys = _grow_keys(a.data_dir, keys, a.prompts, a.seed)
        elif a.prompts and a.prompts != len(keys):
            raise SystemExit(
                f"{cache_dir} holds {len(keys)} samples; --prompts {a.prompts} "
                "would need a new cache dir, or --grow to add samples."
            )
        draws = max(int(man["draws"]), a.draws or 0)
    else:
        keys = _choose_keys(a.data_dir, a.prompts, a.seed)
        draws = a.draws or 1

    # Manifest first: a partial cache already says what it was built for. Written
    # through a temp file, so a hardlinked copy of a cache dir keeps its own manifest.
    tmp_man = man_path.with_suffix(".tmp")
    tmp_man.write_text(
        json.dumps(
            {
                "format": FORMAT,
                "teacher": teacher,
                "seed": a.seed,
                "data_dir": a.data_dir,
                "draws": draws,
                "keys": keys,
            },
            indent=1,
        )
    )
    os.replace(tmp_man, man_path)

    dataset = CachedDataset(
        a.data_dir, keep_list={key_stem(k) for k in keys}, need_pooled=False
    )
    index = {sample_key(te, a.data_dir): i for i, (_, te) in enumerate(dataset.samples)}
    absent = [k for k in keys if k not in index]
    if absent:
        raise SystemExit(
            f"{len(absent)} cached keys not under {a.data_dir}: {absent[:3]}"
        )
    todo = [
        (key, k)
        for k in range(draws)
        for key in keys
        if not pair_path(cache_dir, key, k).exists()
    ]
    if a.limit:
        todo = todo[: a.limit]
    logger.info(
        f"anchor cache {cache_dir}: {len(keys)} samples × {draws} draws, "
        f"{len(todo)} pairs to roll (k {a.k_anchor} of {a.teacher_anchor_steps}, "
        f"CFG {a.teacher_cfg}, shift {a.flow_shift})"
    )
    if not todo:
        return

    device, dtype = torch.device("cuda"), torch.bfloat16
    model = _load_teacher(
        a, device, dtype, latent_token_counts(npz for npz, _ in dataset.samples)
    )
    state_dtype = latent_dtype(model, dtype)
    uncond = load_uncond_crossattn(
        str(default_uncond_path()), device=device, dtype=dtype
    )
    sigmas = get_timesteps_sigmas(a.teacher_anchor_steps, a.flow_shift, "cpu")[1]
    sigmas = sigmas.tolist()
    pad_cache = PadCache(dtype)

    @torch.no_grad()
    def velocity(x, s, c):
        t_b = torch.full((x.shape[0],), s, device=device, dtype=state_dtype)
        with torch.autocast("cuda", dtype=dtype):
            out = model.forward_mini_train_dit(
                x.unsqueeze(2),
                t_b,
                c,
                padding_mask=pad_cache.get(x),
                skip_pooled_text_proj=True,
            )
        return out.squeeze(2).float()

    sink = None
    sink_path = command_job_path(None)
    if sink_path is not None:
        sink = ProgressSink(sink_path, run="anchor_cache", method="turbo", preset=None)
        sink.run_start(total_steps=len(todo), total_epochs=draws, pid=os.getpid())

    shown: list = []
    torch.cuda.synchronize()
    logger.info(
        f"before loop: allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
        f"load peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for n, (key, k) in enumerate(todo, 1):
        item = dataset[index[key]]
        c = item["crossattn_emb"].unsqueeze(0).to(device, dtype=dtype)
        c_null = uncond_for_batch(uncond, c)
        cc = torch.cat([c, c_null]) if a.cfg_batch else None
        eps = pair_eps(
            (1, *item["latents"].shape),
            pair_seed(a.seed, key, k),
            device=device,
            dtype=state_dtype,
        )

        def cfg_velocity(z, s):
            if a.cfg_batch:
                zz = torch.cat([z, z])
                if not shown:
                    logger.info(
                        f"cfg_batch forward: x {tuple(zz.shape)}, c {tuple(cc.shape)}"
                    )
                    shown.append(1)
                v_c, v_u = velocity(zz, s, cc).chunk(2)
            else:
                v_c = velocity(z, s, c)
                v_u = velocity(z, s, c_null)
            return v_u + a.teacher_cfg * (v_c - v_u)

        v_target = anchor_target(cfg_velocity, eps, sigmas, a.k_anchor)
        out = pair_path(cache_dir, key, k)
        tmp = out.with_suffix(".tmp")
        save_file({"v_target": v_target[0].cpu().contiguous()}, str(tmp))
        os.replace(tmp, out)
        if n % 10 == 0 or n == len(todo):
            rate = n / (time.time() - t0)
            logger.info(
                f"{n}/{len(todo)} pairs, {1 / rate:.2f} s/pair, "
                f"ETA {(len(todo) - n) / rate / 60:.1f} min, "
                f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
            )
            if sink is not None:
                sink.log({"s_per_pair": 1 / rate}, global_step=n, epoch=k)
    if sink is not None:
        sink.run_end(status="ok", final_step=len(todo))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="cache dir")
    p.add_argument("--data_dir", default="post_image_dataset/lora")
    p.add_argument(
        "--prompts",
        type=int,
        default=0,
        help="samples to cache, drawn from data_dir with --seed (0: all). Fixed "
        "by the first build of a cache dir.",
    )
    p.add_argument(
        "--draws",
        type=int,
        default=None,
        help="ε draws per sample (default 1, or what the cache already holds). "
        "Raising it on an existing cache adds the new draws.",
    )
    p.add_argument(
        "--grow",
        action="store_true",
        help="on an existing cache, add samples of data_dir it does not hold, up to "
        "--prompts in all (0: every sample). Held pairs are kept.",
    )
    p.add_argument("--seed", type=int, default=42, help="sample choice and ε seeds")
    p.add_argument(
        "--dit_path",
        default="models/diffusion_models/anima_merged15_int8_gptq.safetensors",
    )
    p.add_argument(
        "--dit_nonquant_path",
        default="models/diffusion_models/anima_merged15.safetensors",
    )
    p.add_argument("--int8", default="stream", choices=["off", "fwd", "full", "stream"])
    p.add_argument("--int8_file", default=None)
    p.add_argument("--stream", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--teacher_cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--k_anchor", type=int, default=7)
    p.add_argument("--teacher_anchor_steps", type=int, default=28)
    p.add_argument(
        "--cfg_batch",
        action="store_true",
        help="run cond and uncond as one B=2 forward (the training teacher runs two "
        "B=1 forwards).",
    )
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--no_compile", dest="compile", action="store_false")
    p.add_argument(
        "--limit", type=int, default=0, help="Roll at most this many pairs (smoke)."
    )
    return p


@no_window_main
def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    a = build_argparser().parse_args()
    if not 1 <= a.k_anchor < a.teacher_anchor_steps:
        raise SystemExit("--k_anchor in [1, --teacher_anchor_steps)")
    _build(a)


if __name__ == "__main__":
    main()
