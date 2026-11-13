#!/usr/bin/env python
"""P1 — GRPO on jp_v1's rows, arm R (``project/cjk_rl/proposal.md`` § 3–4).

Policy: the ruler's ``m14v3`` (merged14 + Hub turbo v3, 4 steps, cfg 1) with
jp_v1's rows (``seed_1008``) trainable, everything else frozen. A group is one
dialogue item of ``output/cjk_rl/pool/pool.json`` and G seeds:

- rollout: G SDE (a = 0.7) samples, last step x̂₀ (P0a); the same seeds and
  SDE noise through the step-0 rows give the anchor renders;
- reward: r = r_seq + β·g_p − γ·(1 − m_init) − δ·max(0, fw_over_init), r_seq /
  g_p from the batched tuned reader (``src/reward.py``), m_init the ``AtSim``
  match to the anchor, fw the flat-white share over the anchor's. β, γ, δ put
  each term on r_seq's within-group std (P0b, ``results/20261113-1727-grpo_p0``);
- update: A = (r − mean) / std over the group; −A·log π of SDE steps 0 and 1
  backwarded through the rows (only seed_1008's), one AdamW step a group.

Rows are trained in effective units (raw × row_scale); ``--lr`` is in row-norm
units (the reseed trainer's), × row_scale on the parameter.

    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/src/grpo.py --run smoke --groups 8"
    make daemon-run ARGS="--queue --stall-timeout 900 project/cjk_rl/src/grpo.py --run p1_r --groups 300 --arm rl_p1"

Writes ``output/cjk_rl/grpo/<run>/`` (``log.jsonl``, ``state.pt`` every
``--save_every`` groups — a rerun resumes from it — and every ``--keep_every``-th
group's renders); with ``--arm`` the final rows also go to the reseed line's
``output/cjk_anima_reseed/<arm>/trained.pt`` for the ruler.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_rl
sys.path.insert(0, str(LINE / "probes"))

import grpo_p0 as p0  # noqa: E402  (sets the pack, routing and the reseed path)
import reward  # noqa: E402

from reseed import OUT as RESEED_OUT  # noqa: E402
from reseed import REPO  # noqa: E402

POOL = REPO / "output" / "cjk_rl" / "pool" / "pool.json"
ROOT = REPO / "output" / "cjk_rl" / "grpo"
A_NOISE = 0.7
STEPS = 4
GRAD_STEPS = (0, 1)
# P0b within-group stds on SDE a = 0.7: r_seq 0.109, g_p 0.110, m_init 0.130, fw 0.024
BETA, GAMMA, DELTA = 0.1089 / 0.1096, 0.1089 / 0.1303, 0.1089 / 0.0242


def load_items(seed: int) -> list:
    its = [
        m
        for m in json.loads(POOL.read_text("utf-8"))["items"]
        if m["track"] == "dialogue"
    ]
    random.Random(seed).shuffle(its)
    return its


def main() -> None:
    import torch

    from common.models import decode_image, dit_forward, encode_captions
    from eval.ruler.score import AtSim
    from library.inference.sampling import get_timesteps_sigmas

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--groups", type=int, default=300)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="row-norm units")
    ap.add_argument("--seed", type=int, default=1115)
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--keep_every", type=int, default=10)
    ap.add_argument("--arm", default="", help="final rows → reseed OUT/<arm>/")
    a = ap.parse_args()

    out = ROOT / a.run
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(a.seed)[: a.groups]

    from common import models

    models.TOKEN_COUNTS[:] = [(h // 16) * (w // 16) for w, h in (m["shape"] for m in items)]
    r, table = p0.make_renderer()
    dev = r.device
    r.shared["text_encoder"].to("cpu")  # dit_forward reads cached embeds
    torch.cuda.empty_cache()
    cache = encode_captions([m["caption"] for m in items], dev, out / "te")

    from common.models import load_trained

    seed_d = load_trained(RESEED_OUT / p0.ROWS)["delta"]
    row_scale = float(seed_d["row_scale"])
    ids = r.delta.ext_ids
    assert len(ids) == table.shape[0], (len(ids), table.shape)
    pos = {e: i for i, e in enumerate(ids)}
    train_idx = torch.tensor([pos[int(e)] for e in seed_d["ext_ids"]], device=dev)
    mask = torch.zeros(len(ids), 1, device=dev)
    mask[train_idx] = 1.0

    r.set_arm(table)
    raw = r.delta.raw
    init = raw.detach().clone()
    raw.requires_grad_(True)
    opt = torch.optim.AdamW(
        [raw], lr=a.lr * row_scale, weight_decay=0.0, betas=(0.9, 0.99)
    )
    k0 = 0
    if (out / "state.pt").exists():
        st = torch.load(out / "state.pt", map_location=dev, weights_only=False)
        raw.data.copy_(st["raw"])
        opt.load_state_dict(st["opt"])
        k0 = st["k"]
        print(f"resumed at group {k0}", flush=True)

    rd = p0.readers()
    at = AtSim("cuda")
    _, sig = get_timesteps_sigmas(STEPS, r.args.flow_shift, dev)
    anima = r.anima

    def v_of(x, s, caption, grad):
        b = x.shape[0]
        ts = torch.full((b,), s, device=dev, dtype=torch.float32)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            return dit_forward(anima, x, ts, cache, [caption] * b, dev).float()

    def rollout(m, seeds):
        # the G samples as one batch; each keeps its own start and SDE noise
        W, H = m["shape"]
        sdes = [p0.SDE(sig, A_NOISE, sd, dev) for sd in seeds]
        x = torch.cat(
            [
                torch.randn(
                    1,
                    16,
                    H // 8,
                    W // 8,
                    generator=torch.Generator(device=dev).manual_seed(sd),
                    device=dev,
                )
                for sd in seeds
            ]
        )
        xs = [x]
        for i in range(STEPS):
            s_i = sdes[0].s[i]
            den = x - s_i * v_of(x, s_i, m["caption"], False)
            x = torch.cat(
                [
                    sde.step(x[g : g + 1], den[g : g + 1], i)
                    for g, sde in enumerate(sdes)
                ]
            )
            xs.append(x)
        trajs = [[x[g : g + 1] for x in xs] for g in range(len(seeds))]
        return trajs, sdes

    def save_state(k):
        torch.save(
            {"k": k, "raw": raw.detach(), "opt": opt.state_dict(), "args": vars(a)},
            out / "state.pt.tmp",
        )
        (out / "state.pt.tmp").replace(out / "state.pt")

    log = open(out / "log.jsonl", "a")
    t_run = time.time()
    for k in range(k0, len(items)):
        m = items[k]
        seeds = [a.seed * 100_000 + k * a.G + g for g in range(a.G)]
        gdir = out / "renders" / f"g{k:04d}"
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        traj, sdes = rollout(m, seeds)
        with torch.no_grad():
            live = raw.data.clone()
            raw.data.copy_(init)
            anc, _ = rollout(m, seeds)
            raw.data.copy_(live)
        t_roll = time.time() - t0

        # everything in memory; PNGs only for the groups kept
        imgs = [decode_image(r.vae, xs[-1].unsqueeze(2), dev) for xs in traj]
        aimgs = [decode_image(r.vae, xs[-1].unsqueeze(2), dev) for xs in anc]
        del anc
        reads = rd.read_images([p0.to_bgr(im) for im in imgs], whole=False)
        m_init = match_init(at, imgs, aimgs)
        asked = m["texts"] + m["sfx"]
        terms = []
        for g in range(a.G):
            fw = flat_white_img(imgs[g]) - flat_white_img(aimgs[g])
            t = {
                "r_seq": reward.r_seq(m["texts"], reads[g]),
                "g_p": reward.g_p(asked, reads[g]),
                "m_init": m_init[g],
                "fw": fw,
            }
            t["r"] = (
                t["r_seq"]
                + BETA * t["g_p"]
                - GAMMA * (1 - t["m_init"])
                - DELTA * max(0.0, fw)
            )
            terms.append(t)
        t_read = time.time() - t0 - t_roll

        rs = torch.tensor([t["r"] for t in terms])
        std = float(rs.std())
        n_rows, gnorm = 0, 0.0
        if std > 1e-4:
            adv = ((rs - rs.mean()) / std).tolist()
            opt.zero_grad(set_to_none=True)
            for g in range(a.G):
                xs, sde = traj[g], sdes[g]
                for i in GRAD_STEPS:
                    v = v_of(xs[i], sde.s[i], m["caption"], True)
                    mu = sde.mean(xs[i], v, i)
                    sc = sde.scale(i)
                    lp = (-((xs[i + 1] - mu) ** 2) / (2 * sc**2)).mean()
                    (-(adv[g] * lp) / (a.G * len(GRAD_STEPS))).backward()
            if raw.grad is not None:
                raw.grad.mul_(mask)
                gnorm = float(raw.grad.norm())
                n_rows = int((raw.grad.abs().sum(1) > 0).sum())
                opt.step()
        del traj
        t_upd = time.time() - t0 - t_roll - t_read

        mean = {x: sum(t[x] for t in terms) / a.G for x in terms[0]}
        drift = float(((raw.detach() - init) * mask).norm(dim=1).max()) / row_scale
        rec = {
            "k": k,
            "i": m["i"],
            "n_texts": len(m["texts"]),
            "shape": m["shape"],
            **{x: round(v, 4) for x, v in mean.items()},
            "r_std": round(std, 4),
            "rows": n_rows,
            "gnorm": gnorm,
            "drift_max": round(drift, 5),
            "s": [round(t_roll, 1), round(t_read, 1), round(t_upd, 1)],
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(
            f"g{k:04d} r {mean['r']:+.3f}±{std:.3f} seq {mean['r_seq']:.3f} "
            f"gp {mean['g_p']:.3f} init {mean['m_init']:.3f} fw {mean['fw']:+.4f} "
            f"rows {n_rows} drift {drift:.4f} s {rec['s']} "
            f"peak {rec['peak_gib']} ({(time.time() - t_run) / 60:.1f} min)",
            flush=True,
        )
        if k % a.keep_every == 0:
            gdir.mkdir(parents=True, exist_ok=True)
            for g in range(a.G):
                imgs[g].save(gdir / f"s{g}.png", compress_level=1)
                aimgs[g].save(gdir / f"a{g}.png", compress_level=1)
            (gdir / "terms.json").write_text(
                json.dumps(
                    {"item": m, "terms": terms, "reads": reads},
                    ensure_ascii=False,
                    indent=1,
                ),
                "utf-8",
            )
        if (k + 1) % a.save_every == 0:
            save_state(k + 1)
    save_state(len(items))

    if a.arm:
        d = RESEED_OUT / a.arm
        d.mkdir(parents=True, exist_ok=True)
        rows = raw.detach()[train_idx].float().cpu() / row_scale
        torch.save(
            {
                "delta": {
                    "ext_ids": list(seed_d["ext_ids"]),
                    "raw": rows,
                    "row_scale": row_scale,
                },
                "record": {
                    "from": p0.ROWS,
                    "method": "grpo",
                    "run": str(out),
                    "args": vars(a),
                    "weights": {"beta": BETA, "gamma": GAMMA, "delta": DELTA},
                },
            },
            d / "trained.pt",
        )
        print(f"rows → {d / 'trained.pt'}", flush=True)


def flat_white_img(img) -> float:
    """``eval.ruler.score.flat_white`` on an image in memory."""
    import numpy as np

    im = np.asarray(img.convert("L"), np.float32)
    H, W = im.shape
    p = im[: H // 16 * 16, : W // 16 * 16].reshape(H // 16, 16, W // 16, 16)
    return float(((p.std(axis=(1, 3)) < 6) & (p.mean(axis=(1, 3)) > 225)).mean())


def match_init(at, imgs, aimgs) -> list:
    """``AtSim(f, anchor)["en_match"]`` per pair, in memory and one PE batch."""
    import numpy as np
    import torch
    from anime_tools.grouping.features import PE_NATIVE
    from anime_tools.grouping.matching import match_fracs, pool_cells_batch
    from PIL import Image

    def t512(im):
        im = im.convert("RGB").resize((PE_NATIVE, PE_NATIVE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0

    with torch.no_grad():
        _, g16 = at.emb(torch.stack([t512(im) for im in [*imgs, *aimgs]]))
        cells = pool_cells_batch(
            torch.from_numpy(g16.astype("float32")).to(at.emb.device), at.g
        )
    n = len(imgs)
    return [
        float(
            match_fracs(cells[g : g + 1], cells[n + g : n + g + 1], at.cell_min, at.ratio)[0]
        )
        for g in range(n)
    ]


if __name__ == "__main__":
    main()
