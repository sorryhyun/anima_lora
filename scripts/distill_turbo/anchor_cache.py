"""Anchor target cache for ``dpdmd.anchor_only``: build it, and read it in the loop.

The anchor target ``v_target`` depends only on (prompt, ε, latent shape) and the frozen
teacher, so it can be rolled once and reused. A cache holds ``draws`` pairs per sample
of ``data_dir``; pair ``(sample, k)`` uses an ε regenerated from a seed hashed from
``(seed, sample key, k)``, so only ``v_target`` is stored.

Build (a daemon job; resumable, skips pairs already on disk)::

    python -m scripts.distill_turbo.anchor_cache --config <run>.toml [--draws K]

with ``dpdmd.anchor_cache = "<dir>"`` in the config. The same config then trains on
the cache: ``AnchorDataset`` adds each sample's pair to the DataLoader item, and
epoch e reads draw e mod K.

``manifest.json`` records the teacher, schedule and seed the targets were rolled with;
a loop whose config differs on any of them refuses the cache.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from library.datasets.cache import make_cached_collate
from library.env import resolve_under_home
from library.io.cache_names import TE_CACHE_SUFFIX

logger = logging.getLogger(__name__)

FORMAT = 1
MANIFEST = "manifest.json"
_base_collate = make_cached_collate()

# Config fields the targets depend on. A mismatch on any of them refuses the cache.
_SIGNATURE_FIELDS = (
    "dit_path",
    "dit_nonquant_path",
    "int8",
    "int8_file",
    "stream",
    "teacher_cfg",
    "flow_shift",
    "k_anchor",
    "teacher_anchor_steps",
    "seed",
    "data_dir",
)


def signature(cfg) -> dict:
    return {f: getattr(cfg, f) for f in _SIGNATURE_FIELDS}


def sample_key(te_path: str, data_dir: str) -> str:
    """Stable per-sample key: the TE cache path under ``data_dir``, suffix dropped."""
    rel = os.path.relpath(te_path, data_dir).removesuffix(TE_CACHE_SUFFIX)
    return rel.replace(os.sep, "__")


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


class AnchorDataset(torch.utils.data.Dataset):
    """``CachedDataset`` wrapper whose items also carry the sample's anchor pair.

    Each item gains ``anchor_eps`` (fp32, regenerated from the pair seed) and
    ``anchor_v`` (the cached ``v_target``) for draw ``epoch % draws``. The loop calls
    ``next_epoch()`` before it re-iterates the DataLoader; workers are re-created per
    ``iter()``, so they pick the new draw up.
    """

    def __init__(self, base, cache_dir, cfg, *, epoch: int = 0):
        self.base = base
        self.dir = Path(resolve_under_home(cache_dir))
        man_path = self.dir / MANIFEST
        if not man_path.exists():
            raise FileNotFoundError(
                f"dpdmd.anchor_cache: no {MANIFEST} under {self.dir}; build it with "
                "`python -m scripts.distill_turbo.anchor_cache --config <this config>`."
            )
        man = json.loads(man_path.read_text())
        want = signature(cfg)
        diff = {
            f: (man["signature"].get(f), want[f])
            for f in _SIGNATURE_FIELDS
            if man["signature"].get(f) != want[f]
        }
        if man.get("format") != FORMAT or diff:
            raise ValueError(
                f"dpdmd.anchor_cache {self.dir} was built for another run "
                f"(format {man.get('format')}, field: (cache, config) = {diff})."
            )
        self.draws = int(man["draws"])
        self.seed = int(cfg.seed)
        self.keys = [sample_key(te, cfg.data_dir) for _, te in base.samples]
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
    """``make_collate()``'s batch plus the stacked ``anchor_eps`` / ``anchor_v``."""
    out = _base_collate(batch)
    out["anchor_eps"] = torch.stack([b["anchor_eps"] for b in batch])
    out["anchor_v"] = torch.stack([b["anchor_v"] for b in batch])
    return out


def _build(args) -> None:
    from library.anima.uncond import uncond_for_batch
    from library.config.resolved import load_toml
    from library.runtime.progress import ProgressSink, command_job_path

    from .config import resolve_config
    from .setup import build_run
    from .steps import teacher_anchor

    cfg = resolve_config(args, load_toml(args.config))
    if not cfg.anchor_cache:
        raise SystemExit("set dpdmd.anchor_cache (the cache dir) in the config.")
    cache_dir = Path(resolve_under_home(cfg.anchor_cache))
    cache_dir.mkdir(parents=True, exist_ok=True)

    man_path = cache_dir / MANIFEST
    sig = signature(cfg)
    if man_path.exists():
        man = json.loads(man_path.read_text())
        if man.get("format") != FORMAT or man["signature"] != sig:
            raise SystemExit(
                f"{cache_dir} holds a cache for another signature: {man['signature']}"
            )
        draws = max(int(man["draws"]), args.draws or 0)
    else:
        draws = args.draws or 1

    # The teacher path of build_run; logs, resume, validation and the snapshot stay
    # out of the training run's names.
    bcfg = dataclasses.replace(
        cfg,
        output_dir=str(cache_dir),
        output_name="anchor_cache",
        no_log=True,
        resume="",
        validate_every_n_steps=0,
        single_prompt_idx=None,
        fake_warmup_steps=0,
        anchor_cache="",
    )
    ctx = build_run(args, bcfg)
    dataset = ctx.dataloader.dataset
    keys = [sample_key(te, cfg.data_dir) for _, te in dataset.samples]
    todo = [
        (i, k)
        for k in range(draws)
        for i in range(len(keys))
        if not pair_path(cache_dir, keys[i], k).exists()
    ]
    if args.limit:
        todo = todo[: args.limit]
    logger.info(
        f"anchor cache {cache_dir}: {len(keys)} samples × {draws} draws, "
        f"{len(todo)} pairs to roll (k {cfg.k_anchor} of {cfg.teacher_anchor_steps}, "
        f"CFG {cfg.teacher_cfg})"
    )

    sink = None
    sink_path = command_job_path(None)
    if sink_path is not None:
        sink = ProgressSink(sink_path, run="anchor_cache", method="turbo", preset=None)
        sink.run_start(total_steps=len(todo), total_epochs=draws, pid=os.getpid())

    # Manifest first: a partial cache already says what it was built for.
    man_path.write_text(
        json.dumps(
            {
                "format": FORMAT,
                "draws": draws,
                "signature": sig,
                "n_samples": len(keys),
            },
            indent=1,
        )
    )
    t0 = time.time()
    for n, (i, k) in enumerate(todo, 1):
        item = dataset[i]
        lat = item["latents"]
        c = item["crossattn_emb"].unsqueeze(0).to(ctx.device, dtype=ctx.cond_dtype)
        eps = pair_eps(
            (1, *lat.shape),
            pair_seed(cfg.seed, keys[i], k),
            device=ctx.device,
            dtype=ctx.dtype,
        )
        c_null = uncond_for_batch(ctx.uncond_base, c)
        v = teacher_anchor(ctx, cfg, eps, c, c_null, 1)[0]
        out = pair_path(cache_dir, keys[i], k)
        tmp = out.with_suffix(".tmp")
        save_file({"v_target": v.float().cpu().contiguous()}, str(tmp))
        os.replace(tmp, out)
        if n % 10 == 0 or n == len(todo):
            rate = n / (time.time() - t0)
            logger.info(
                f"{n}/{len(todo)} pairs, {1 / rate:.2f} s/pair, "
                f"ETA {(len(todo) - n) / rate / 60:.1f} min"
            )
            if sink is not None:
                sink.log({"s_per_pair": 1 / rate}, global_step=n, epoch=k)
    if sink is not None:
        sink.run_end(status="ok", final_step=len(todo))


def main() -> None:
    from .config import build_argparser

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = build_argparser()
    parser.add_argument(
        "--draws",
        type=int,
        default=None,
        help="ε draws per sample (default 1, or what the cache already holds). "
        "Raising it on an existing cache adds the new draws.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Roll at most this many pairs (smoke)."
    )
    _build(parser.parse_args())


if __name__ == "__main__":
    main()
