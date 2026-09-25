"""The training plumbing the line's trainer (``cjk_scale/train.py``) and
conflict probe reuse from the old ``train`` stage: the in / out-of-box
residual split, the TE cache, the per-shape latent cache and the batcher.
The stage itself (``stage_train``) is gone (pruned 2026-09-25).
"""

from __future__ import annotations

import json
import random
import time

import torch
from common.models import encode_captions, encode_images, ext_ids_of, load_vae
from common.shapes import parse_shape


class BoxSplit:
    """The plain FM residual split into its in-box (glyph) and out-of-box
    (scene) mean squares, per item, averaged over the steps between two log
    rows — a sentence box is ≈ 2 % of the canvas, so the logged ``loss`` cannot
    show an in-box change. ``in_box_hi`` / ``in_box_lo`` are the same in-box
    mean over the items drawn at σ ≥ / < ``SIGMA_SPLIT``."""

    SIGMA_SPLIT = 0.7
    KEYS = ("in_box", "out_box", "in_box_hi", "in_box_lo")

    def __init__(self):
        self.sum = {k: 0.0 for k in self.KEYS}
        self.n = {k: 0.0 for k in self.KEYS}

    @torch.no_grad()
    def add(self, pred, target, recs, ts):
        se = ((pred.float() - target.float()) ** 2).mean(dim=1, keepdim=True)
        m = _box_mask(se.shape, recs, se.device)
        n_in = m.sum(dim=(1, 2, 3))
        inb = (se * m).sum(dim=(1, 2, 3)) / n_in.clamp(min=1)
        outb = (se * (1 - m)).sum(dim=(1, 2, 3)) / (1 - m).sum(dim=(1, 2, 3)).clamp(
            min=1
        )
        has = n_in > 0
        hi = has & (ts.float().view(-1).to(has.device) >= self.SIGMA_SPLIT)
        rows = [
            ("in_box", inb, has),
            ("out_box", outb, has),
            ("in_box_hi", inb, hi),
            ("in_box_lo", inb, has & ~hi),
        ]
        for k, v, w in rows:
            self.sum[k] = self.sum[k] + (v * w).sum()
            self.n[k] = self.n[k] + w.sum()

    def pop(self) -> dict:
        out = {
            k: float(self.sum[k] / self.n[k]) for k in self.KEYS if float(self.n[k]) > 0
        }
        self.__init__()
        return out


def _box_mask(se_shape, recs, device):
    """``(B, 1, h, w)`` 1 under each item's ``box`` (latent cells), else 0."""
    B, _C, h, w = se_shape
    m = torch.zeros(B, 1, h, w, device=device)
    for b, r in enumerate(recs):
        x0, y0, x1, y1 = r["box"]
        m[b, :, y0 // 8 : -(-y1 // 8), x0 // 8 : -(-x1 // 8)] = 1.0
    return m


def _encode_text(recs, ev, device, out, te_cache=None):
    """Qwen side + pack-routed T5 ids, pre-adapter, for the training and eval
    captions. Writes ``eval_coverage.json`` (trained rows / rows per string)."""
    t0 = time.time()
    cache = encode_captions([r["caption"] for r in recs], device, te_cache)
    train_ext = ext_ids_of(cache)
    ev_cache = encode_captions([e["caption"] for e in ev], device)
    ev_ext = {
        e["text"]: sorted(ext_ids_of({e["caption"]: ev_cache[e["caption"]]}))
        for e in ev
    }
    cov = {
        t: (len([x for x in ids if x in train_ext]), len(ids))
        for t, ids in ev_ext.items()
    }
    print(
        f"text: {len(cache)} captions, {len(train_ext)} ext rows touched, {time.time() - t0:.0f}s",
        flush=True,
    )
    (out / "eval_coverage.json").write_text(
        json.dumps(cov, ensure_ascii=False, indent=1)
    )
    return cache, train_ext, ev_ext


def shape_index(recs) -> dict | None:
    """``{'WxH': [record indices]}`` for a data dir built with --shapes, else
    None (every record then trains at --train_size as before)."""
    if not recs or "shape" not in recs[0]:
        return None
    out: dict = {}
    for i, r in enumerate(recs):
        assert "shape" in r, f"record {i} has no shape in a --shapes data dir"
        out.setdefault("x".join(map(str, r["shape"])), []).append(i)
    return out


class LatentStore:
    """VAE latents cached in the data dir: one tensor at ``--train_size``
    (square data dirs), or one per canvas shape when the data stage drew
    ``--shapes`` (each item encoded at its own render size). Indexed by
    positions in the kept ``recs``; a batch must be one shape."""

    def __init__(self, a, data, recs_all, keep, device):
        t0 = time.time()
        self.keep = keep
        by_shape = shape_index(recs_all)
        self.row_of = None
        self.n_families = 1
        if by_shape is None:
            lat_file = data / f"latents_{a.train_size}.pt"
            if lat_file.exists():
                lat = torch.load(lat_file)
            else:
                vae = load_vae(device)
                size = (a.train_size, a.train_size)
                lat = encode_images(vae, [r["file"] for r in recs_all], device, size)
                torch.save(lat, lat_file)
                del vae
                torch.cuda.empty_cache()
            if len(keep) != len(recs_all):
                lat = lat[keep]
            print(f"latents: {tuple(lat.shape)} in {time.time() - t0:.0f}s", flush=True)
        else:
            lat_file = data / f"latents_mixed_{'_'.join(sorted(by_shape))}.pt"
            # one .npy per shape, written chunk by chunk and mapped back: a
            # 100 k-item dir neither collects its latents in RAM nor loses a
            # half-done encode (the single .pt is the pre-2026-09-21 cache)
            lat_dir = lat_file.with_suffix("")
            if lat_file.exists():
                lat = torch.load(lat_file)
            else:
                lat_dir.mkdir(exist_ok=True)
                vae = None

                def one(shp, idxs):
                    nonlocal vae
                    f = lat_dir / f"{shp}.npy"
                    mark = f.with_suffix(".npy.done")
                    if not (mark.exists() and int(mark.read_text()) == len(idxs)):
                        vae = vae or load_vae(device)
                    return encode_images(
                        vae,
                        [recs_all[i]["file"] for i in idxs],
                        device,
                        parse_shape(shp),
                        out_file=f,
                    )

                lat = {shp: one(shp, idxs) for shp, idxs in by_shape.items()}
                del vae
                torch.cuda.empty_cache()
            # recs_all index → (shape, row in that shape's tensor)
            self.row_of = {
                i: (shp, k)
                for shp, idxs in by_shape.items()
                for k, i in enumerate(idxs)
            }
            # a static block graph per distinct token count (384×512 and
            # 512×384 share one: same seq, rope comes in as a tensor)
            self.n_families = len(
                {(W // 16) * (H // 16) for W, H in map(parse_shape, by_shape)}
            )
            print(
                "latents: "
                + ", ".join(f"{shp} {tuple(lat[shp].shape)}" for shp in sorted(lat))
                + f" in {time.time() - t0:.0f}s",
                flush=True,
            )
        self.lat = lat

    def shape_of(self, i: int) -> str:
        return self.row_of[self.keep[i]][0]

    def __getitem__(self, idx):
        if self.row_of is None:
            return self.lat[idx]
        ent = [self.row_of[self.keep[i]] for i in idx]
        assert len({e[0] for e in ent}) == 1, f"mixed shapes in one batch: {ent}"
        return self.lat[ent[0][0]][[e[1] for e in ent]]


class Batcher:
    """Batch index lists over ``recs``:

    - balanced data (W2a): one layout group per step;
    - mixed shapes: an epoch is every shape's items shuffled and chunked into
      full batches, the batch list shuffled — a batch is one shape and each
      shape is drawn in proportion to its items (S line: one *(shape,
      source)* — flat vs scene composite — so ``c_flat`` toggles per batch);
    - else a shuffled walk in chunks of ``--batch``.

    Every epoch reshuffles with ``Random(seed + step)``."""

    def __init__(self, a, recs, lat: LatentStore):
        self.seed, self.batch = a.seed, a.batch
        self.groups = None
        if "layout_id" in recs[0]:
            by_lid: dict = {}
            for i, r in enumerate(recs):
                by_lid.setdefault(r["layout_id"], []).append(i)
            self.groups = list(by_lid.values())
            sizes = {len(x) for x in self.groups}
            assert sizes == {a.batch}, (
                f"balanced data has group sizes {sizes}; pass --batch to match"
            )
            print(
                f"batching: {len(self.groups)} layout groups of {a.batch}", flush=True
            )
        self.unit = 1 if self.groups else a.batch
        self.n = len(self.groups) if self.groups else len(recs)
        self.order = list(range(self.n))
        random.Random(a.seed).shuffle(self.order)
        self.ptr = 0
        self.shape_batches = None
        if lat.row_of is not None and self.groups is None:
            self.by_shape: dict = {}
            for r in range(len(recs)):
                key = lat.shape_of(r)
                if recs[r]["src"] == "scene":
                    key += "|scene"
                self.by_shape.setdefault(key, []).append(r)
            self.shape_batches = self._shape_epoch(random.Random(a.seed))
            self.bptr = 0
            print(
                "batching: mixed shapes "
                + ", ".join(f"{k} {len(v)}" for k, v in sorted(self.by_shape.items()))
                + f" → {len(self.shape_batches)} batches/epoch of {a.batch}",
                flush=True,
            )

    def _shape_epoch(self, brng: random.Random):
        out = []
        for shp in sorted(self.by_shape):
            ids = self.by_shape[shp][:]
            brng.shuffle(ids)
            out += [
                ids[j : j + self.batch]
                for j in range(0, len(ids) - self.batch + 1, self.batch)
            ]
        brng.shuffle(out)
        return out

    def next(self, step: int) -> list[int]:
        if self.shape_batches is not None:
            if self.bptr >= len(self.shape_batches):
                self.shape_batches = self._shape_epoch(random.Random(self.seed + step))
                self.bptr = 0
            idx = self.shape_batches[self.bptr]
            self.bptr += 1
            return idx
        if self.ptr + self.unit > self.n:
            random.Random(self.seed + step).shuffle(self.order)
            self.ptr = 0
        sel = self.order[self.ptr : self.ptr + self.unit]
        self.ptr += self.unit
        return self.groups[sel[0]] if self.groups else sel
