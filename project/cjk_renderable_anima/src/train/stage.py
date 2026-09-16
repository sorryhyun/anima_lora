"""Stage ``train`` — frozen DiT + frozen Qwen, rectified-flow loss on glyph crops.

Order (lazy loading): captions → text-encoder cache → VAE latents → DiT →
trainables → compile → loop → export the ExtDelta table to ``trained.pt``.
"""

from __future__ import annotations

import json
import math
import random
import time

import torch
from common.hooks import OutVec, load_out_vec
from common.models import (
    checkpoints,
    dit_forward,
    encode_captions,
    encode_images,
    ext_ids_of,
    gen_args,
    load_vae,
)
from common.paths import arm_dir, data_dir
from common.prompts import TPL_BUBBLE
from common.shapes import parse_shape

from .trainables import Trainables


def stage_train(a):
    from library.anima.vocab_pack import attached_pack_rows, strategy_pack
    from library.inference.generation import get_generation_settings
    from library.inference.models import load_dit_model
    from library.inference.text import ensure_text_strategies
    from library.runtime.noise import fm_training_batch

    torch.manual_seed(a.seed)
    data = data_dir(a)
    out = arm_dir(a)
    out.mkdir(parents=True, exist_ok=True)
    recs_all = [
        json.loads(ln) for ln in (data / "train.jsonl").read_text().splitlines()
    ]
    ev = json.loads((data / "eval.json").read_text())
    args = gen_args(a.train_size, a.steps, a.cfg, out)
    device = get_generation_settings(args).device

    held, keep = _held_out_split(a, recs_all, ev, out)
    recs = [recs_all[i] for i in keep]
    cache, train_ext, ev_ext = _encode_text(recs, ev, device, out)
    lat = LatentStore(a, data, recs_all, keep, device)

    anima = load_dit_model(args, device, torch.bfloat16)
    anima.requires_grad_(False)
    assert attached_pack_rows(anima), "no vocab pack attached to the DiT"
    # the pack table lives on the hook closure; read it via the strategy
    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    tr = Trainables(a, anima, device, train_ext, ev_ext, tok, strategy_pack(tok))

    outvec, q_vec = None, None
    if a.out_vec and a.out_vec_train > 0:
        # Q-in-training (plan_synth decision tree): the pretrained quoted-EN
        # output shift sits at the ext positions for every item, flat or
        # composite, so neither f nor c_flat has to build the trigger
        vhat, norm = load_out_vec(a.out_vec, a.out_vec_frame)
        q_vec = vhat * (a.out_vec_train * norm)
        outvec = OutVec(anima, device)
        outvec.set(q_vec)
        print(
            f"out_vec: Q fixed on, frame {a.out_vec_frame}, EN shift norm {norm:.2f}"
            f" × {a.out_vec_train:g} → ‖{float(q_vec.norm()):.2f}‖",
            flush=True,
        )
    opt = torch.optim.AdamW(tr.params, weight_decay=0.0, betas=(0.9, 0.99))
    sched = None
    if a.lr_decay == "cosine":
        # the identity has no parameter-side bound (attempt 8/9: linear norm
        # growth, 1.6× at 2000 steps); cosine to 0 stops it late
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda st: 0.5 * (1 + math.cos(math.pi * min(st / a.train_steps, 1.0)))
        )
    anima.train()
    if a.grad_ckpt:
        anima.enable_gradient_checkpointing(unsloth_offload=False)
    if a.compile:
        # Block compile is the repo's first OOM remedy (bit-exact, cuts
        # activation memory). Compile after the trainables are attached (the
        # ExtDelta hooks sit on the adapter's embed, the adapter LoRA patches
        # Linears the blocks never see).
        from library.runtime.harness import compile_blocks_for_training

        compile_blocks_for_training(
            anima,
            None,
            backend="inductor",
            n_token_families=lat.n_families,
            activation_memory_budget=a.activation_memory_budget,
            partitioner_aggressive_recomputation=bool(a.aggressive_recompute),
            grad_ckpt=bool(a.grad_ckpt),
        )

    batcher = Batcher(a, recs, lat)
    log = []
    aug_rng = random.Random(a.seed + 11)
    killed = ""
    t0 = time.time()
    for step in range(1, a.train_steps + 1):
        idx = batcher.next(step)
        latents = lat[idx].to(device)
        noise = torch.randn_like(latents)
        noisy, ts, target = fm_training_batch(
            latents,
            noise,
            dtype=torch.bfloat16,
            device=device,
            t_min=a.t_min,
            t_max=a.t_max,
        )
        tr.materialize(aug_rng)
        is_scene = recs[idx[0]]["src"] == "scene"
        tr.set_source(flat=not is_scene)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            captions = [recs[i]["caption"] for i in idx]
            pred = dit_forward(anima, noisy, ts, cache, captions, device)
        loss_fm = weighted_fm_loss(
            pred, target, [recs[i] for i in idx], a.box_weight if is_scene else 1.0
        )
        loss, decor_val = tr.regularized(loss_fm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        tr.after_step()
        if step % 25 == 0 or step == 1:
            rec = tr.log_record(step, loss_fm, loss, decor_val, t0)
            log.append(rec)
            print(json.dumps(rec), flush=True)
            killed = tr.kill_reason(rec, step)
            if killed:
                # save the table anyway — a killed run is still renderable
                # (attempt 8 was killed at 2.5× with nothing to eval)
                print(killed, flush=True)
                break
    tr.export_table(aug_rng)
    sd = tr.state_dict(held, killed)
    if q_vec is not None:
        sd["out_vec"] = q_vec.cpu()
    torch.save(sd, out / "trained.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    if killed:
        raise SystemExit(killed)
    print(
        f"train: {a.train_steps} steps in {(time.time() - t0) / 60:.1f} min → {out / 'trained.pt'}",
        flush=True,
    )
    del anima
    torch.cuda.empty_cache()


def weighted_fm_loss(pred, target, recs, box_weight: float):
    """MSE on the flow target, with the latent cells under a composite item's
    swapped text box (``rec['box']``, pixels at the item's own size, VAE 8×)
    weighted ``box_weight`` and the rest 1 — normalised by the weight sum so
    the loss scale matches the plain MSE (``box_weight`` 1 = plain MSE)."""
    se = (pred.float() - target.float()) ** 2
    if box_weight == 1.0:
        return se.mean()
    B, _C, h, w = se.shape
    wmap = torch.ones(B, 1, h, w, device=se.device)
    for b, r in enumerate(recs):
        x0, y0, x1, y1 = r["box"]
        wmap[b, :, y0 // 8 : -(-y1 // 8), x0 // 8 : -(-x1 // 8)] = box_weight
    return (se * wmap).sum() / (wmap.expand_as(se).sum())


def _held_out_split(a, recs_all, ev, out):
    """W2d held-out split: N single chars never appear in a training item (a
    single, a combo or a corpus line containing them); the eval set (edited
    in place, written to the arm dir) gains every held-out char as group
    ``single_held``. Returns ``(held chars, kept record indices)``."""
    if not (a.held_out or a.held_out_chars):
        return [], list(range(len(recs_all)))
    assert a.arm == "encoder", "--held_out is the encoder arm's generalisation test"
    inv = sorted(
        {r["text"] for r in recs_all if r["src"] == "font" and len(r["text"]) == 1}
    )
    if a.held_out_chars:
        # Run 2: an IDS-structured split — composites whose atoms stay trained
        # — is chosen by hand, not drawn
        held = list(a.held_out_chars)
        missing = [c for c in held if c not in inv]
        assert not missing, f"--held_out_chars not in the inventory: {missing}"
    else:
        held = random.Random(a.seed + 7).sample(inv, a.held_out)
    hs = set(held)
    keep = [i for i, r in enumerate(recs_all) if not (hs & set(r["text"]))]
    for e in ev:
        if e["group"] == "single" and e["text"] in hs:
            e["group"] = "single_held"
    present = {e["text"] for e in ev if e["group"] == "single_held"}
    ev += [
        {"group": "single_held", "text": c, "caption": TPL_BUBBLE.format(c)}
        for c in held
        if c not in present
    ]
    (out / "eval.json").write_text(json.dumps(ev, ensure_ascii=False, indent=1))
    (out / "held_out.json").write_text(json.dumps(held, ensure_ascii=False))
    print(
        f"held-out {len(held)} chars {''.join(held)}; train items {len(keep)}/{len(recs_all)}",
        flush=True,
    )
    return held, keep


def _encode_text(recs, ev, device, out):
    """Qwen side + pack-routed T5 ids, pre-adapter, for the training and eval
    captions. Writes ``eval_coverage.json`` (trained rows / rows per string)."""
    t0 = time.time()
    cache = encode_captions([r["caption"] for r in recs], device)
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
            if lat_file.exists():
                lat = torch.load(lat_file)
            else:
                vae = load_vae(device)
                lat = {
                    shp: encode_images(
                        vae,
                        [recs_all[i]["file"] for i in idxs],
                        device,
                        parse_shape(shp),
                    )
                    for shp, idxs in by_shape.items()
                }
                torch.save(lat, lat_file)
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
