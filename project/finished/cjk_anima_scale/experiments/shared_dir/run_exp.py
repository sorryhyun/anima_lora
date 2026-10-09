#!/usr/bin/env python
"""shared_dir — is a trained arm's layout carried by the rows' shared direction? (hypothesis.md)

Two readings of ``b0305_reband``'s broken layout (the floor's banner → small
columns and sentence-length lines, kana captions only):

- **H1, band × px**: the items' 12–24 px glyphs do not resolve at
  0.75–0.93, so the rows learn the items' layout, and a row acts at every σ;
- **H2, shared credit**: every row of an item's caption gets the same
  region-wide signal (nothing ties ``a``'s gradient to ``a``'s place), so a
  large part of the update is one direction shared by the rows — "fill the
  text region" — and that is what moves the layout.

No training. The arm's update ``d_i`` = its row − the seed row (absolute
units, over the rows it moved) splits into the mean ``m`` and the residual
``d_i − m``. Row sets: ``seed``, ``full`` (the arm), ``nomean`` (seed +
residual), ``mean`` (seed + ``m`` on the moved rows). Arms = (rows above σ
``--switch``, rows below), the conditional switched as in ``sigma_split``
(``generate_body``'s ``context_alt``; the negative pass untouched), on the
``sent`` ruler's keys (``--seeds``, default seed 0: 92 of 184), read and paired against the seed's routed floor:

- ``full`` / ``nomean`` / ``mean``: one row set at every σ;
- ``s_full`` / ``s_nomean``: the seed rows above the switch, the arm's below.

H2 predicts ``mean`` breaks the layout and ``nomean`` keeps it; H1 predicts
``nomean`` breaks it too and ``s_full`` restores it.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="project/cjk_anima_scale/experiments/shared_dir/run_exp.py \\
      --label sd0"
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap, load_experiment  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "shared_dir"
ARM_ROWS = OUT / "experiments" / "b0305_reband_warm"
MOVED_REL = 1e-2  # a row moved: |d_i| > this × |seed row|
ARMS = {
    "s_full": ("seed", "full"),
    "s_nomean": ("seed", "nomean"),
    "mean": ("mean", "mean"),
    "nomean": ("nomean", "nomean"),
    "full": ("full", "full"),
    "full_s": ("full", "seed"),  # the arm's rows above the switch, the seed's below
}


def abs_rows(path: Path):
    import torch

    d = torch.load(path, weights_only=False)["delta"]
    return d["raw"].float() * float(d["row_scale"]), d


def decompose(arm_pt: Path) -> tuple[dict, dict]:
    """Absolute rows of each set (``(n_ext, dim)``) + the split's stats."""
    import torch

    seed, sd = abs_rows(SEED_ROWS)
    full, ad = abs_rows(arm_pt)
    assert list(sd["ext_ids"]) == list(ad["ext_ids"]), "ext ids differ"
    d = full - seed
    moved = d.norm(dim=1) > MOVED_REL * seed.norm(dim=1).clamp(min=1e-6)
    m = d[moved].mean(0)
    sets = {"seed": seed, "full": full}
    sets["nomean"] = full.clone()
    sets["nomean"][moved] -= m
    sets["mean"] = seed.clone()
    sets["mean"][moved] += m
    dm = d[moved]
    cos = torch.nn.functional.cosine_similarity(dm, m[None], dim=1)
    stats = {
        "moved": int(moved.sum()),
        "d_norm_mean": round(float(dm.norm(dim=1).mean()), 3),
        "seed_norm_mean": round(float(seed[moved].norm(dim=1).mean()), 3),
        "mean_norm": round(float(m.norm()), 3),
        "mean_energy_share": round(
            float(m.norm() ** 2 * len(dm) / (dm.norm(dim=1) ** 2).sum()), 4
        ),
        "cos_to_mean_median": round(float(cos.median()), 3),
    }
    return sets, stats


class Rows:
    """sigma_split's Splitter with the ext rows swapped per encode."""

    def __init__(self, SS, sets: dict):
        self.sp = SS.Splitter()
        dl = self.sp.delta
        self.raw = {
            k: (v / dl.row_scale).to(dl.raw.device, dl.raw.dtype)
            for k, v in sets.items()
        }
        err = float((dl.raw.detach() - self.raw["seed"]).abs().max())
        assert err < 1e-3, f"Splitter's rows are not the seed's: {err}"
        self.caches = {k: {} for k in sets}

    def encode(self, caption: str, rows: str):
        from library.inference.text import prepare_text_inputs

        sp = self.sp
        sp.delta.raw.data.copy_(self.raw[rows])
        sp.delta.scale = 1.0
        sp.shared["conds_cache"] = self.caches[rows]
        return prepare_text_inputs(sp._args(caption, 0), sp.device, sp.anima, sp.shared)

    def render(self, fn: Path, it: dict, above: str, below: str, switch: float):
        from common.models import decode_image
        from library.inference import generation as G

        if fn.exists():
            return
        sp = self.sp
        hi, null = self.encode(it["caption"], above)
        lo = self.encode(it["caption"], below)[0]
        body = G.generate_body
        G.generate_body = lambda *x, **k: body(
            *x, context_alt=lo, tag_drop_sigma=switch, **k
        )
        try:
            with sp.torch.no_grad():
                lat = G.generate(
                    sp._args(it["caption"], it["seed"]),
                    sp.gen,
                    sp.shared,
                    precomputed_text_data={"context": hi, "context_null": null},
                )
        finally:
            G.generate_body = body
        fn.parent.mkdir(parents=True, exist_ok=True)
        decode_image(sp.vae, lat, sp.device).save(fn)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--label", required=True)
    p.add_argument("--switch", type=float, default=0.8)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--arm_rows", default=str(ARM_ROWS))
    p.add_argument(
        "--seeds", default="0", help="the floor keys' seeds to render (of 0,1)"
    )
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    assert set(arms) <= set(ARMS), arms
    arm_pt = Path(args.arm_rows) / "trained.pt"
    sets, stats = decompose(arm_pt)
    print(f"{arm_pt}: {json.dumps(stats)}", flush=True)
    SS = load_experiment("sigma_split")
    seeds = {int(x) for x in args.seeds.split(",")}
    items = [it for it in SS.floor_items() if it["seed"] in seeds]
    print(
        f"{len(arms)} arms × {len(items)} renders, switch {args.switch}: "
        + ", ".join(f"{a} {ARMS[a]}" for a in arms),
        flush=True,
    )
    if args.dry_run:
        return
    root = OUT / "experiments" / f"{NAME}_{Path(args.arm_rows).name}"
    rw = Rows(SS, sets)
    # plumbing: the seed rows on both sides vs the floor cache of record
    chk = root / "check.png"
    chk.unlink(missing_ok=True)
    rw.render(chk, items[0], "seed", "seed", args.switch)
    import numpy as np
    from PIL import Image

    check_d = float(
        np.abs(
            np.asarray(Image.open(chk), np.float32)
            - np.asarray(Image.open(items[0]["floor_file"]), np.float32)
        ).mean()
    )
    print(f"check: mean |Δpx| {check_d:.3f}", flush=True)
    t0 = time.time()
    manifests = {}
    for a in arms:
        manifests[a] = []
        for n, it in enumerate(items):
            fn = (
                root
                / a
                / "img"
                / f"{a}_p{it['pi']:02d}_{it['text']}_{it['clause']}_s{it['seed']}.png"
            )
            rw.render(fn, it, *ARMS[a], args.switch)
            manifests[a].append(
                {
                    k: it[k]
                    for k in ("seed", "pi", "prompt", "text", "clause", "caption")
                }
                | {"file": str(fn), "cond": a, "switch": args.switch}
            )
            if n % 46 == 0:
                print(
                    f"  {a} {n}/{len(items)} · {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    device = rw.sp.device
    rw.sp.free()
    for a in arms:
        SS.read_arm(manifests[a], root / a, device)
        print(f"read {a}", flush=True)

    from cjk_scale import reads as R

    chars = sorted({it["text"] for it in items})
    keys = {(it["text"], it["clause"], it["pi"], it["seed"]) for it in items}
    floor_h = {
        k: v for k, v in R.hits(SS.FLOOR_READS, chars, SS.CLAUSE).items() if k in keys
    }
    recs = {
        "floor": [
            m
            for m in json.loads(SS.FLOOR_READS.read_text("utf-8"))
            if (m["text"], m["clause"], m["pi"], m["seed"]) in keys
        ]
    }
    metrics: dict = {
        "arm_rows": str(arm_pt),
        "decompose": stats,
        "switch": args.switch,
        "arms": {a: ARMS[a] for a in arms},
        "check_mean_abs_px": check_d,
        "floor_tally": R.tally(floor_h),
    }
    print("===== floor", flush=True)
    for a in arms:
        f = root / a / "native_reads.json"
        recs[a] = json.loads(f.read_text("utf-8"))
        h = R.hits(f, chars, SS.CLAUSE)
        print(f"===== {a} {ARMS[a]}", flush=True)
        metrics[a] = {"tally": R.tally(h), "vs_floor": R.paired(h, floor_h)}
        print(f"  vs floor {metrics[a]['vs_floor']}", flush=True)
    place = {}
    for name, ms in recs.items():
        ps = [SS.placement(m) for m in ms]
        place[name] = {
            k: round(st.mean(p[k] for p in ps), 4)
            for k in ("box", "box_h", "flat_white")
        } | {
            k: round(st.mean(m[k] for m in ms if m.get(k) is not None), 4)
            for k in ("en_cos", "en_cos_out", "box_iou")
        }
        print(f"  placement {name:<8} {place[name]}", flush=True)
    metrics["placement"] = place
    SS.sheets(items, arms, recs, root / "sheets")
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
