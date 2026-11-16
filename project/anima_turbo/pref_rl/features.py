"""Feature caches for the preference head (pref_rl/proposal.md § 3.2; P0 and P1).

    make daemon-run ARGS="--stall-timeout 600 project/anima_turbo/pref_rl/features.py --set gelcrawl"
    make daemon-run ARGS="--stall-timeout 600 project/anima_turbo/pref_rl/features.py --set <session>"

One pass over the images of a label set (``labels.py``) fills, per basis, the rows it does not
hold yet:

- ``dino``: DINOv3 CLS at 224² (the tower and preprocessing of gelcrawl's ``classify.py``)
- ``pe``: PE-Spatial-B16-512 tokens at the image's aspect bucket, mean-pooled
- ``tag_prob`` / ``tag_emb``: the Anima Tagger backbone's logits on the tagger vocabulary and
  its pooled hidden feature

The DiT-head basis is not here: it needs VAE latents and the caption.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import labels  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.vision.buckets import bucket_pixel_size, get_bucket_spec, pick_bucket  # noqa: E402

TOWERS = {"dino": ("dino",), "pe": ("pe",), "tagger": ("tag_prob", "tag_emb")}
DINO_MEAN, DINO_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SAVE_EVERY = 100  # batches


def prep_dino(im: Image.Image) -> torch.Tensor:
    x = TF.resize(TF.to_tensor(im), [224, 224], antialias=True)
    return TF.normalize(x, DINO_MEAN, DINO_STD)


def prep_pe(im: Image.Image) -> torch.Tensor:
    spec = get_bucket_spec("pe_spatial")
    bh, bw = bucket_pixel_size(pick_bucket(im.height, im.width, spec), spec)
    return TF.to_tensor(im.resize((bw, bh), Image.Resampling.BILINEAR)) * 2 - 1


class Images(Dataset):
    def __init__(self, items: list[labels.Item], preps: dict):
        self.items = items
        self.preps = preps

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        try:
            im = Image.open(self.items[i].path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"[pref-feat] unreadable {self.items[i].path}: {e}", flush=True)
            return None
        return {"i": i, **{name: prep(im) for name, prep in self.preps.items()}}


def collate(rows: list) -> dict:
    rows = [r for r in rows if r is not None]
    out = {"i": [r["i"] for r in rows]}
    for name in rows[0] if rows else ():
        if name == "pe":
            out[name] = [r[name] for r in rows]
        elif name != "i":
            out[name] = torch.stack([r[name] for r in rows])
    return out


def build(towers: list[str], device: torch.device, dino_dir: Path):
    """(per-image preprocessing, batch forward → {basis: [B, D] cpu float}) per tower."""
    preps, fwds = {}, {}
    if "dino" in towers:
        from transformers import AutoModel

        dino = AutoModel.from_pretrained(dino_dir).to(device, torch.bfloat16).eval()
        preps["dino"] = prep_dino

        def fwd_dino(x):
            h = dino(pixel_values=x.to(device, torch.bfloat16)).last_hidden_state[:, 0]
            return {"dino": h.float().cpu()}

        fwds["dino"] = fwd_dino
    if "pe" in towers:
        from library.vision.encoder import load_pe_encoder

        pe = load_pe_encoder(device, name="pe_spatial")
        preps["pe"] = prep_pe

        def fwd_pe(xs):
            by_shape = defaultdict(list)
            for j, x in enumerate(xs):
                by_shape[tuple(x.shape)].append(j)
            out = [None] * len(xs)
            for js in by_shape.values():
                x = torch.stack([xs[j] for j in js]).to(device, pe.dtype)
                h = pe.encoder(x).last_hidden_state.float().mean(1).cpu()
                for j, row in zip(js, h):
                    out[j] = row
            return {"pe": torch.stack(out)}

        fwds["pe"] = fwd_pe
    if "tagger" in towers:
        from anime_tools.tagger import DEFAULT_TAGGER_DIR, AnimaTagger
        from anime_tools.tagger.dbv4_backend import preprocess_dbv4
        from anime_tools.tagger.pixai_backend import preprocess_pixai

        tagger = AnimaTagger(resolve_under_home(DEFAULT_TAGGER_DIR), device=device)
        backend = tagger._backend
        prep = preprocess_pixai if tagger.backend_kind == "pixai" else preprocess_dbv4
        preps["tagger"] = partial(prep, size=backend.img_size)
        cols = torch.as_tensor(tagger._align_ext, dtype=torch.long)

        def fwd_tagger(x):
            o = backend.forward_tensor(x)
            p = o.probs[:, cols].clamp(1e-4, 1 - 1e-4)
            return {"tag_prob": torch.logit(p), "tag_emb": o.hidden}

        fwds["tagger"] = fwd_tagger
    return preps, fwds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--set", required=True, help="gelcrawl, or a session under output/turbo_pref/"
    )
    ap.add_argument("--tower", nargs="+", default=list(TOWERS), choices=list(TOWERS))
    ap.add_argument("--dino", default=str(labels.gelcrawl_root() / "dino"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--limit", type=int, default=0, help="at most this many items a group (smoke)"
    )
    args = ap.parse_args()

    items = labels.items_of(args.set)
    if args.limit:
        seen: dict = defaultdict(int)
        kept = []
        for it in items:
            if seen[(it.group, it.label)] < args.limit:
                seen[(it.group, it.label)] += 1
                kept.append(it)
        items = kept
    feats = {b: labels.load_feats(args.set, b) for t in args.tower for b in TOWERS[t]}
    todo = [it for it in items if any(it.key not in f for f in feats.values())]
    print(
        f"[pref-feat] {args.set}: {len(items)} items, {len(todo)} to extract "
        f"({', '.join(feats)})",
        flush=True,
    )
    if not todo:
        return

    device = torch.device("cuda")
    preps, fwds = build(args.tower, device, Path(args.dino))
    loader = DataLoader(
        Images(todo, preps),
        batch_size=args.batch,
        num_workers=args.workers,
        collate_fn=collate,
    )
    t0 = time.time()
    done = 0
    for bi, batch in enumerate(loader):
        if not batch["i"]:
            continue
        with torch.no_grad():
            for tower, fwd in fwds.items():
                for basis, rows in fwd(batch[tower]).items():
                    for i, row in zip(batch["i"], rows.numpy()):
                        feats[basis][todo[i].key] = row
        done += len(batch["i"])
        if bi % 10 == 0:
            print(
                f"[pref-feat] {done}/{len(todo)}  {done / (time.time() - t0):.1f} img/s",
                flush=True,
            )
        if (bi + 1) % SAVE_EVERY == 0:
            for basis, f in feats.items():
                labels.save_feats(args.set, basis, f)
    for basis, f in feats.items():
        labels.save_feats(args.set, basis, f)
    print(
        f"[pref-feat] done: {done} in {time.time() - t0:.0f}s -> {labels.feat_dir(args.set)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
