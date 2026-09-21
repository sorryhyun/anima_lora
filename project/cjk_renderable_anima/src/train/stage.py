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
    if a.pair_loss:
        n_pair = sum("ref_file" in r for r in recs)
        assert n_pair, "--pair_loss needs a data dir built with --pair_ref"
        if a.pair_ref_frame == "en":
            from data.pair import en_frame

            for r in recs:
                if "ref_caption" in r:
                    r["ref_caption"] = en_frame(r["ref_caption"])
        print(
            f"pair loss: {n_pair}/{len(recs)} items have a sibling"
            + (
                f" ({sum('ref_file' in r and r['src'] != 'scene' for r in recs)} flat"
                f"{'' if a.pair_flat else ', left on plain FM'})"
            )
            + (f", σ ≥ {a.pair_sigma_min:g} only" if a.pair_sigma_min > 0 else "")
            + (
                ", sibling captions under the EN frame"
                if a.pair_ref_frame == "en"
                else ""
            ),
            flush=True,
        )
    cache, train_ext, ev_ext = _encode_text(recs, ev, device, out, bool(a.pair_loss))
    lat = LatentStore(a, data, recs_all, keep, device, ref=bool(a.pair_loss))

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
    if a.lr_decay == "cosine" or a.lr_warmup > 0:
        # the identity has no parameter-side bound (attempt 8/9: linear norm
        # growth, 1.6× at 2000 steps); cosine to 0 stops it late. Linear
        # warmup (--lr_warmup) multiplies in: a warm start's first Adam steps
        # otherwise move every coordinate ≈ lr and erase the rows.
        # --row_blocks: the schedule is per block — step and horizon are the
        # block's, so every row gets the same warmup + cosine regardless of
        # where in the run its block falls.
        horizon = a.row_blocks or a.train_steps

        def lr_mult(st):
            if a.row_blocks:
                st = st % a.row_blocks
            m = 1.0
            if a.lr_decay == "cosine":
                m = 0.5 * (1 + math.cos(math.pi * min(st / horizon, 1.0)))
            if a.lr_warmup > 0:
                m *= min((st + 1) / a.lr_warmup, 1.0)
            return m

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
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

    batcher = Batcher(a, recs, lat, cache=cache, delta=tr.delta)
    pair_ema: dict = {}
    flat_ema: dict = {}
    log = []
    split = BoxSplit()  # in-box / out-of-box residual, averaged between log rows
    block_log = []  # --row_blocks: every step, per row (row_blocks_log.jsonl)
    aug_rng = random.Random(a.seed + 11)
    killed = ""
    t0 = time.time()
    for step in range(1, a.train_steps + 1):
        idx = batcher.next(step)
        if batcher.blocks and (step - 1) % a.row_blocks == 0:
            # new row block: fresh optimizer state for this row only (the
            # other rows' slices are untouched; theirs is their own block's)
            st = opt.state.get(tr.delta.raw)
            if st:
                for k in ("exp_avg", "exp_avg_sq"):
                    st[k][batcher.cur_row].zero_()
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
        brecs = [recs[i] for i in idx]
        bw = a.box_weight if is_scene else 1.0
        bs = a.box_share if is_scene else 0.0
        cap = a.box_share_cap
        with torch.autocast("cuda", dtype=torch.bfloat16):
            captions = [r["caption"] for r in brecs]
            pred = dit_forward(anima, noisy, ts, cache, captions, device)
        extra = {}
        paired = (
            a.pair_loss
            and (is_scene or a.pair_flat)
            and all("ref_file" in r for r in brecs)
        )
        if paired:
            # ΔFM (plan_synth2): the sibling under the same ε and σ, its
            # residual r_A = v_θ(A) − v_A* subtracted as a control variate.
            # The A forward carries no gradient — nothing about the
            # reference reaches the rows.
            pred_a, target_a = pair_branch(
                anima, lat, idx, noise, ts, cache, brecs, device
            )
            keep_pair = (
                (ts.float() >= a.pair_sigma_min).view(-1, 1, 1, 1).to(pred.dtype)
            )
            loss_fm = weighted_fm_loss(
                pred - keep_pair * pred_a,
                target - keep_pair * target_a,
                brecs,
                bw,
                bs,
                cap,
            )
            if is_scene:
                extra = pair_stats(
                    pred, target, pred_a, target_a, brecs, bw, pair_ema, bs, cap
                )
            else:
                # flat siblings: own EMA and keys, the composite fields keep
                # their meaning
                st = pair_stats(
                    pred, target, pred_a, target_a, brecs, bw, flat_ema, bs, cap
                )
                extra = {f"{k}_flat": v for k, v in st.items()}
        else:
            loss_fm = weighted_fm_loss(pred, target, brecs, bw, bs, cap)
        loss, decor_val = tr.regularized(loss_fm)
        if is_scene:
            split.add(pred, target, brecs, ts)
        if batcher.blocks:
            # per-row trajectory: the paired (or plain) residual split into
            # its in-box (glyph) and out-of-box (scene) mean squares, every
            # step, with the row's norm *before* this step's update
            with torch.no_grad():
                if paired:
                    res = (pred.float() - keep_pair * pred_a.float()) - (
                        target.float() - keep_pair * target_a.float()
                    )
                else:
                    res = pred.float() - target.float()
                se = res**2
                m = _box_mask(se.shape, brecs, se.device)
                inb = float((se * m).sum() / m.expand_as(se).sum().clamp(min=1))
                outb = float(
                    (se * (1 - m)).sum() / (1 - m).expand_as(se).sum().clamp(min=1)
                )
                rn = float(tr.delta.raw[batcher.cur_row].norm() * tr.row_scale)
            block_log.append(
                {
                    "step": step,
                    "block": (step - 1) // a.row_blocks,
                    "local": (step - 1) % a.row_blocks + 1,
                    "ext": batcher.cur_ext,
                    "loss": float(loss_fm),
                    "in_box": inb,
                    "out_box": outb,
                    "row_norm": rn,
                    "lr": opt.param_groups[0]["lr"],
                }
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if batcher.blocks:
            # only the block's row moves: the others would otherwise keep
            # taking Adam steps from the μ‖f‖² pull and their stale momentum
            # for the rest of the run (smoke 2026-09-18: block-0 row ended at
            # norm 24 against block-11's 150)
            raw_before = tr.delta.raw.detach().clone()
        opt.step()
        if batcher.blocks:
            with torch.no_grad():
                keep = torch.ones(
                    raw_before.shape[0], dtype=torch.bool, device=raw_before.device
                )
                keep[batcher.cur_row] = False
                tr.delta.raw[keep] = raw_before[keep]
        if sched is not None:
            sched.step()
        tr.after_step()
        if step % 25 == 0 or step == 1:
            extra = {**extra, **split.pop()}
            rec = tr.log_record(step, loss_fm, loss, decor_val, t0, extra)
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
    if block_log:
        (out / "row_blocks_log.jsonl").write_text(
            "\n".join(json.dumps(r) for r in block_log) + "\n"
        )
    if killed:
        raise SystemExit(killed)
    print(
        f"train: {a.train_steps} steps in {(time.time() - t0) / 60:.1f} min → {out / 'trained.pt'}",
        flush=True,
    )
    del anima
    torch.cuda.empty_cache()


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
        for k, v, w in (
            ("in_box", inb, has),
            ("out_box", outb, has),
            ("in_box_hi", inb, hi),
            ("in_box_lo", inb, has & ~hi),
        ):
            self.sum[k] = self.sum[k] + (v * w).sum()
            self.n[k] = self.n[k] + w.sum()

    def pop(self) -> dict:
        out = {
            k: float(self.sum[k] / self.n[k]) for k in self.KEYS if float(self.n[k]) > 0
        }
        self.__init__()
        return out


BOX_SHARE_CAP = 0.75


def weighted_fm_loss(
    pred,
    target,
    recs,
    box_weight: float,
    box_share: float = 0.0,
    box_share_cap: float = BOX_SHARE_CAP,
):
    """MSE on the flow target, with the latent cells under a composite item's
    swapped text box (``rec['box']``, pixels at the item's own size, VAE 8×)
    weighted ``box_weight`` and the rest 1 — normalised by the weight sum so
    the loss scale matches the plain MSE (``box_weight`` 1 = plain MSE).

    ``box_share`` ρ_g > 0 replaces that with the area-independent form
    (plan_synth4 R4.5): per item ``s·mean_in + (1 − s)·mean_out`` with
    ``s = min(ρ_g · n_glyphs, box_share_cap)``, averaged over the batch — the
    in-box share of the loss no longer follows the box area, and a row's share
    does not fall with the item's glyph count. At ``d0``'s 64-cell box
    ρ_g 0.25 is ``box_weight`` 20."""
    se = (pred.float() - target.float()) ** 2
    if box_share > 0.0:
        m = _box_mask(se.shape, recs, se.device)
        per_cell = se.mean(dim=1, keepdim=True)  # (B, 1, h, w)
        n_in = m.sum(dim=(1, 2, 3))
        n_out = (1.0 - m).sum(dim=(1, 2, 3))
        mean_in = (per_cell * m).sum(dim=(1, 2, 3)) / n_in.clamp(min=1)
        mean_out = (per_cell * (1.0 - m)).sum(dim=(1, 2, 3)) / n_out.clamp(min=1)
        n_glyph = torch.tensor(
            [max(1, len("".join(r["text"].split()))) for r in recs],
            device=se.device,
            dtype=se.dtype,
        )
        s = (box_share * n_glyph).clamp(max=box_share_cap)
        s = torch.where(n_in > 0, s, torch.zeros_like(s))  # no box: plain mean
        s = torch.where(n_out > 0, s, torch.ones_like(s))
        return (s * mean_in + (1.0 - s) * mean_out).mean()
    if box_weight == 1.0:
        return se.mean()
    B, _C, h, w = se.shape
    wmap = torch.ones(B, 1, h, w, device=se.device)
    for b, r in enumerate(recs):
        x0, y0, x1, y1 = r["box"]
        wmap[b, :, y0 // 8 : -(-y1 // 8), x0 // 8 : -(-x1 // 8)] = box_weight
    return (se * wmap).sum() / (wmap.expand_as(se).sum())


def _box_mask(se_shape, recs, device):
    """``(B, 1, h, w)`` 1 under each item's ``box`` (latent cells), else 0."""
    B, _C, h, w = se_shape
    m = torch.zeros(B, 1, h, w, device=device)
    for b, r in enumerate(recs):
        x0, y0, x1, y1 = r["box"]
        m[b, :, y0 // 8 : -(-y1 // 8), x0 // 8 : -(-x1 // 8)] = 1.0
    return m


def pair_branch(anima, lat, idx, noise, ts, cache, recs, device):
    """The sibling's ``(pred_A, target_A)`` under the batch's own ``ε`` and
    ``σ`` (``ts`` is σ per sample), ``no_grad``."""
    lat_a = lat.ref(idx).to(device)
    sig = ts.float().view(-1, 1, 1, 1)
    noisy_a = ((1.0 - sig) * lat_a.float() + sig * noise.float()).to(torch.bfloat16)
    target_a = noise - lat_a
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred_a = dit_forward(
            anima, noisy_a, ts, cache, [r["ref_caption"] for r in recs], device
        )
    return pred_a, target_a


def pair_stats(
    pred,
    target,
    pred_a,
    target_a,
    recs,
    box_weight,
    ema: dict,
    box_share: float = 0.0,
    box_share_cap: float = BOX_SHARE_CAP,
) -> dict:
    """The ΔFM log fields (plan_synth2): ``fm_plain`` = plain ‖r_B‖²_w on
    the same items (the number comparable to a ``--pair_loss 0`` arm),
    ``pres`` = outside-box ‖v_θ(B) − v_θ(A)‖² (the scene-preservation term),
    ``ref_bias`` = the systematic part of the sibling's in-box residual:
    ‖EMA(mean_b m_b)‖ / EMA(mean_b ‖m_b‖) with ``m_b`` the in-box channel
    mean of ``r_A`` per item, EMA over ≈ 100 steps — ≈ 0 when the base's
    reference error is spread, → 1 when every item shares one in-box DC
    error (stroke-style bias is read on the sheets, not here)."""
    with torch.no_grad():
        out = {
            "fm_plain": float(
                weighted_fm_loss(
                    pred, target, recs, box_weight, box_share, box_share_cap
                )
            )
        }
        m = _box_mask(pred.shape, recs, pred.device)
        d = (pred.float() - pred_a.float()) ** 2
        outside = 1.0 - m
        out["pres"] = float(
            (d * outside).sum() / outside.expand_as(d).sum().clamp(min=1)
        )
        r_a = (pred_a.float() - target_a.float()) * m
        cells = m.sum(dim=(1, 2, 3)).clamp(min=1)  # (B,)
        m_b = r_a.sum(dim=(2, 3)) / cells.view(-1, 1)  # (B, C) in-box channel mean
        alpha = 0.01
        mean_vec = m_b.mean(0)
        mean_norm = m_b.norm(dim=1).mean()
        if "vec" not in ema:
            ema["vec"], ema["norm"] = mean_vec, mean_norm
        else:
            ema["vec"] = (1 - alpha) * ema["vec"] + alpha * mean_vec
            ema["norm"] = (1 - alpha) * ema["norm"] + alpha * mean_norm
        out["ref_bias"] = float(ema["vec"].norm() / ema["norm"].clamp(min=1e-8))
    return out


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


def _encode_text(recs, ev, device, out, refs: bool = False):
    """Qwen side + pack-routed T5 ids, pre-adapter, for the training and eval
    captions (and, ``refs``, the ΔFM sibling captions — Latin, so they touch
    no ext row and stay out of ``train_ext``). Writes ``eval_coverage.json``
    (trained rows / rows per string)."""
    t0 = time.time()
    cache = encode_captions([r["caption"] for r in recs], device)
    train_ext = ext_ids_of(cache)
    if refs:
        ref_caps = sorted({r["ref_caption"] for r in recs if "ref_caption" in r})
        cache.update(encode_captions(ref_caps, device))
        print(f"text: {len(ref_caps)} sibling captions", flush=True)
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

    def __init__(self, a, data, recs_all, keep, device, ref: bool = False):
        t0 = time.time()
        self.keep = keep
        by_shape = shape_index(recs_all)
        self.row_of = None
        self.n_families = 1
        self.ref_lat = None
        if ref:
            self._load_ref(a, data, recs_all, by_shape, device)
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

    def _load_ref(self, a, data, recs_all, by_shape, device):
        """ΔFM sibling latents (``ref_file``), one tensor per shape holding
        only the paired items, bf16 on disk; ``ref_row_of`` maps a recs_all
        index to (shape, row)."""
        paired = {
            shp: [i for i in idxs if "ref_file" in recs_all[i]]
            for shp, idxs in (by_shape or {"sq": range(len(recs_all))}).items()
        }
        paired = {k: v for k, v in paired.items() if v}
        lat_file = data / f"latents_ref_{'_'.join(sorted(paired))}.pt"
        if lat_file.exists():
            ref = torch.load(lat_file)
        else:
            vae = load_vae(device)
            ref = {}
            for shp, idxs in paired.items():
                size = parse_shape(shp) if by_shape else (a.train_size, a.train_size)
                ref[shp] = encode_images(
                    vae, [recs_all[i]["ref_file"] for i in idxs], device, size
                ).to(torch.bfloat16)
            torch.save(ref, lat_file)
            del vae
            torch.cuda.empty_cache()
        self.ref_lat = ref
        self.ref_row_of = {
            i: (shp, k) for shp, idxs in paired.items() for k, i in enumerate(idxs)
        }
        print(
            "sibling latents: "
            + ", ".join(f"{shp} {tuple(ref[shp].shape)}" for shp in sorted(ref)),
            flush=True,
        )

    def ref(self, idx):
        """Sibling latents (float32) for kept positions ``idx``, one shape."""
        ent = [self.ref_row_of[self.keep[i]] for i in idx]
        assert len({e[0] for e in ent}) == 1, f"mixed shapes in one batch: {ent}"
        return self.ref_lat[ent[0][0]][[e[1] for e in ent]].float()

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

    Every epoch reshuffles with ``Random(seed + step)``.

    ``--row_blocks N`` (rows arm, single-glyph scene items): one ext row at a
    time for N steps, rows in a shuffled cycle; every batch is ``--batch``
    items of that row from one shape, drawn with replacement (a row has
    ~10–30 items, a block ~4N draws). ``cur_row`` is the row's index in
    ``delta.raw`` for the current step."""

    def __init__(self, a, recs, lat: LatentStore, cache=None, delta=None):
        self.seed, self.batch = a.seed, a.batch
        self.groups = None
        self.blocks = None
        if a.row_blocks:
            self._init_row_blocks(a, recs, lat, cache, delta)
            return
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
            rep = self._row_boost_reps(a, recs, cache) if a.row_boost else None
            self.by_shape: dict = {}
            for r in range(len(recs)):
                key = lat.shape_of(r)
                if recs[r]["src"] == "scene":
                    key += "|scene"
                self.by_shape.setdefault(key, []).extend([r] * (rep[r] if rep else 1))
            self.shape_batches = self._shape_epoch(random.Random(a.seed))
            self.bptr = 0
            print(
                "batching: mixed shapes "
                + ", ".join(f"{k} {len(v)}" for k, v in sorted(self.by_shape.items()))
                + f" → {len(self.shape_batches)} batches/epoch of {a.batch}",
                flush=True,
            )

    @staticmethod
    def _row_boost_reps(a, recs, cache) -> list[int]:
        """``--row_boost``: slots per item in an epoch, so that every listed ext
        row expects ``--row_boost_draws`` item draws over the run. Fixed point —
        the added slots lengthen the epoch, which lowers every row's rate."""
        from library.anima.ext_vocab import T5_TABLE_SIZE

        assert a.arm == "rows", "--row_boost is a rows-arm batcher option"
        boost = [int(x) for x in a.row_boost.split(",") if x.strip()]
        ext_of = []
        for r in recs:
            t5 = cache[r["caption"]][2].tolist()
            ext_of.append({int(v) - T5_TABLE_SIZE for v in t5 if v >= T5_TABLE_SIZE})
        items = {e: [i for i, s in enumerate(ext_of) if e in s] for e in boost}
        missing = [e for e in boost if not items[e]]
        assert not missing, f"--row_boost: no item carries ext {missing}"
        rep = [1] * len(recs)
        total = a.train_steps * a.batch

        def draws(e):
            return total * sum(rep[i] for i in items[e]) / sum(rep)

        base = {e: draws(e) for e in boost}
        for _ in range(12):
            short = {e: a.row_boost_draws / draws(e) for e in boost}
            if max(short.values()) <= 1.0:
                break
            for i in {i for e in boost for i in items[e]}:
                need = max(short[e] for e in boost if e in ext_of[i])
                if need > 1.0:
                    rep[i] = math.ceil(rep[i] * need)
        print(
            f"row boost: {len(recs)} items → {sum(rep)} slots/epoch "
            f"(unboosted share ×{len(recs) / sum(rep):.2f}); ext: items, draws "
            + ", ".join(
                f"{e}: {len(items[e])}, {base[e]:.0f} → {draws(e):.0f}" for e in boost
            ),
            flush=True,
        )
        return rep

    def _init_row_blocks(self, a, recs, lat, cache, delta):
        from library.anima.ext_vocab import T5_TABLE_SIZE

        assert a.arm == "rows", "--row_blocks is a rows-arm batcher"
        self.n_block = a.row_blocks
        by_row: dict = {}
        skipped = 0
        for i, r in enumerate(recs):
            if r["src"] != "scene":
                skipped += 1
                continue
            t5 = cache[r["caption"]][2]
            ext = sorted(
                {int(v) - T5_TABLE_SIZE for v in t5.tolist() if v >= T5_TABLE_SIZE}
            )
            if len(ext) != 1:
                skipped += 1  # multi-row item: no single block owns it
                continue
            shp = lat.shape_of(i) if lat.row_of is not None else ""
            by_row.setdefault(ext[0], {}).setdefault(shp, []).append(i)
        assert by_row, "--row_blocks needs single-glyph scene items"
        self.rows = sorted(by_row)  # ext ids
        self.row_items = by_row  # ext id → {shape: [rec idx]}
        self.row_index = {e: delta.index[e] for e in self.rows}
        self.order = self.rows[:]
        random.Random(a.seed).shuffle(self.order)
        self.blocks = True
        self.cur_row = None
        self.cur_ext = None
        n_items = sum(len(v) for d in by_row.values() for v in d.values())
        print(
            f"batching: row blocks of {self.n_block} steps over {len(self.rows)} rows "
            f"({n_items} items, {skipped} skipped), {a.train_steps / self.n_block:.1f} "
            f"blocks/run = {a.train_steps / self.n_block / len(self.rows):.2f} "
            f"passes/row, {self.batch * self.n_block} draws/row/block",
            flush=True,
        )

    def _row_block_next(self, step: int) -> list[int]:
        blk = (step - 1) // self.n_block
        pos = blk % len(self.order)
        if pos == 0 and (step - 1) % self.n_block == 0:
            random.Random(self.seed + step).shuffle(self.order)
        self.cur_ext = self.order[pos]
        self.cur_row = self.row_index[self.cur_ext]
        rng = random.Random(self.seed * 7919 + step)
        shapes = self.row_items[self.cur_ext]
        # a shape drawn in proportion to its items, then the batch from it
        pool = [(shp, i) for shp, ids in shapes.items() for i in ids]
        shp = rng.choice(pool)[0]
        return [rng.choice(shapes[shp]) for _ in range(self.batch)]

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
        if self.blocks:
            return self._row_block_next(step)
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
