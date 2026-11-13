#!/usr/bin/env python
"""xattn_box — a training-free spatial prior on the quoted string's keys (10-15,
SpaG-DiT's CDPE carried over to cross-attn): does telling the string's tokens
*where* to draw cut the line habit / repetition / banner, with no row trained?

Anima's text keys carry no image position (1-D RoPE inside the adapter only),
so the prior is an additive cross-attn logit bias ``(N_q, L_ctx)`` on the
quoted string's tokens (first to last ext id), cond forward only, σ ≥ ``BAND``:

- ``xb<β>`` (box): every string token −β at the image tokens outside the
  target box (dilated ``PAD`` tokens), 0 inside;
- ``xs<β>`` (slot): glyph k −β outside its own slot — the box cut into
  tategaki cells (columns right to left, square-ish), k in reading order;
- ``xt<β>`` (box + tags, 10-15 after xb4 / xs4 drew ~19 more junk letters
  a page in extra bubbles): ``xb``, and the ``TAGS`` tokens (the lowercase
  tags, not the clause's ``Japanese text``) −β outside the box too.

The target box is the EN reference's (same prompt and seed, the base's own
choice of where this line goes): the EN read box whose text best matches the
EN string (``SIM_MIN``); an item with none is left out of the render and
the read (64 / 96 have one). Arm rows: ``seed_1008`` (jp_v1) on the punct pack; its cached
ruler renders are the control, so the read pairs ``seed_1008+x…`` against
``seed_1008`` on the same strings. Extra read: ``in_box`` — the share of
the page's text-box area inside the target box.

    .venv/bin/python project/cjk_anima_reseed/probes/xattn_box.py boxes
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/xattn_box.py run --arms xb4,xs4"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import bootstrap  # noqa: E402

bootstrap()

from eval.ruler import EN, FLOOR, VIEW, items, mode_dir, render_file  # noqa: E402

ROWS = "seed_1008"
BAND = 0.85  # σ at or above: the layout / slot-count band (status.md § 6)
PAD = 1.0  # box dilation, image tokens
SIM_MIN = 0.5
TOK_PX = 16  # VAE 8 × patch 2
TAGS = ("speech bubble", "japanese text")


def en_box(m: dict, en_reads: list):
    from common.text import lev, norm

    t = norm(m["en"])
    best, sim = None, 0.0
    for r in en_reads:
        if r.get("whole"):
            continue
        for rd in ("sfx",):
            c = norm(r.get(rd) or "")
            if c:
                s = 1.0 - lev(c, t) / max(len(c), len(t))
                if s > sim:
                    best, sim = r["box"], s
    return (best, sim) if sim >= SIM_MIN else (None, sim)


def slots(n: int, box_t) -> list:
    """``n`` tategaki cells in the box (token units ``x0, y0, x1, y1``): the
    column count that gives the largest square cell, columns right to left."""
    x0, y0, x1, y1 = box_t
    bw, bh = x1 - x0, y1 - y0
    nc = max(range(1, n + 1), key=lambda c: min(bw / c, bh / math.ceil(n / c)))
    nr = math.ceil(n / nc)
    cw, ch = bw / nc, bh / nr
    out = []
    for k in range(n):
        c, r = divmod(k, nr)
        xr = x1 - c * cw
        out.append((xr - cw, y0 + r * ch, xr, y0 + (r + 1) * ch))
    return out


def region(rect, gh: int, gw: int, pad: float):
    """Image tokens whose centre lies in ``rect`` dilated by ``pad``; at least
    the token under the rect's centre."""
    import torch

    x0, y0, x1, y1 = rect
    ys = torch.arange(gh, dtype=torch.float32)[:, None] + 0.5
    xs = torch.arange(gw, dtype=torch.float32)[None, :] + 0.5
    m = (xs >= x0 - pad) & (xs <= x1 + pad) & (ys >= y0 - pad) & (ys <= y1 + pad)
    cy = min(gh - 1, max(0, int((y0 + y1) / 2)))
    cx = min(gw - 1, max(0, int((x0 + x1) / 2)))
    m[cy, cx] = True
    return m.flatten()


def plan(its: list) -> dict:
    """``{i: {box, sim}}`` — the target box per ruler item (pixels or None)."""
    en = json.loads((mode_dir() / EN / "reads.json").read_text("utf-8"))
    out = {}
    for m in its:
        b, s = en_box(m, en[str(m["i"])])
        out[m["i"]] = {"box": b, "sim": round(s, 3)}
    return out


class Spatial:
    """The bias hooks: the string span from the cond caption's ids (pre-hook on
    ``llm_adapter.embed``, which runs at encode), σ and cond / uncond from the
    DiT call (pre-hook on the DiT), the bias set on every block's cross-attn
    before the blocks and cleared after."""

    def __init__(self, anima, mode: str, beta: float):
        from library.anima.ext_vocab import T5_TABLE_SIZE

        self.anima, self.mode, self.beta = anima, mode, beta
        self.T = T5_TABLE_SIZE
        self.span = None  # (first, last) ext position, L_ctx
        self.box = None  # pixels
        self.tags = False
        self.tag_seqs = [tag_ids(t) for t in TAGS]
        self.tag_pos: list = []
        self.cond_ptr = None
        self.log: dict = {"biased": 0, "calls": 0}
        # prepended: the pack's own pre-hook clamps the ext ids (vocab_pack.py)
        anima.llm_adapter.embed.register_forward_pre_hook(self._ids, prepend=True)
        anima.register_forward_pre_hook(self._pre)
        anima.register_forward_hook(self._post)

    def reset(self, box) -> None:
        self.box, self.cond_ptr, self.span = box, None, None

    def _ids(self, module, args):
        ids = args[0]
        pos = (ids[0] >= self.T).nonzero().flatten()
        if pos.numel():
            self.span = (int(pos[0]), int(pos[-1]), ids.shape[-1])
            row = ids[0].tolist()
            self.tag_pos = [
                j + k
                for q in self.tag_seqs
                for j in range(len(row) - len(q) + 1)
                if row[j : j + len(q)] == q
                for k in range(len(q))
            ]

    def _pre(self, module, args):
        from library.inference.adapters import set_xattn_kbias

        x, t, ctx = args[0], args[1], args[2]
        self.log["calls"] += 1
        if self.cond_ptr is None:
            self.cond_ptr = ctx.data_ptr()  # the sampler's first call is the cond
        if (
            self.box is None
            or self.span is None
            or ctx.data_ptr() != self.cond_ptr
            or float(t.flatten()[0]) < BAND
        ):
            return None
        set_xattn_kbias(self.anima, self.bias(x.shape[-2:], ctx).to(x.device))
        self.log["biased"] += 1
        return None

    def _post(self, module, args, out):
        from library.inference.adapters import set_xattn_kbias

        set_xattn_kbias(self.anima, None)

    def bias(self, lat_hw, ctx):
        import torch

        gh, gw = lat_hw[0] // 2, lat_hw[1] // 2
        a, b, L = self.span
        assert ctx.shape[1] == L, f"context {ctx.shape[1]} ≠ ids {L}"
        x0, y0, x1, y1 = (v / TOK_PX for v in self.box)
        out = torch.zeros(gh * gw, L)
        inside = region((x0, y0, x1, y1), gh, gw, PAD)
        if self.tags:
            out[~inside[:, None] & self._cols(L, self.tag_pos)] = -self.beta
        if self.mode == "box":
            out[~inside, a : b + 1] = -self.beta
        else:
            for k, rect in enumerate(slots(b - a + 1, (x0, y0, x1, y1))):
                inside = region(rect, gh, gw, 0.5)
                out[~inside, a + k] = -self.beta
        return out

    @staticmethod
    def _cols(L: int, pos: list):
        import torch

        m = torch.zeros(1, L, dtype=torch.bool)
        m[0, pos] = True
        return m


def tag_ids(tag: str) -> list:
    """The tag's T5 ids as it sits after ``, `` in a caption."""
    from library.anima.weights import load_t5_tokenizer

    return load_t5_tokenizer(None)(tag, add_special_tokens=False).input_ids


def arm_name(spec: str) -> tuple[str, str, bool, float]:
    mode, tags = {"xb": ("box", False), "xs": ("slot", False), "xt": ("box", True)}[
        spec[:2]
    ]
    return f"{ROWS}+{spec}", mode, tags, float(spec[2:])


def in_box(reads: list, box) -> float | None:
    from eval.ruler.score import box_mask

    if box is None:
        return None
    whole = next(r for r in reads if r.get("whole"))
    hw = (int(whole["box"][3]), int(whole["box"][2]))
    t = box_mask([r["box"] for r in reads if not r.get("whole")], hw)
    p = PAD * TOK_PX
    x0, y0, x1, y1 = box
    tb = box_mask([(x0 - p, y0 - p, x1 + p, y1 + p)], hw)
    return float((t & tb).sum() / t.sum()) if t.sum() else 0.0


def boxes() -> None:
    its = items()
    P = plan(its)
    got = [i for i, v in P.items() if v["box"]]
    print(f"target box on {len(got)} / {len(its)} items")
    for b in ("short", "mid", "long"):
        xs = [m for m in its if m["bin"] == b]
        print(f"  {b}: {sum(1 for m in xs if P[m['i']]['box'])} / {len(xs)}")
    for m in its[:6]:
        v = P[m["i"]]
        print(f"  r{m['i']:02d} {m['text']} | {m['en']} → {v}")
    m = next(m for m in its if m["bin"] == "long" and P[m["i"]]["box"])
    box_t = tuple(v / TOK_PX for v in P[m["i"]]["box"])
    print(f"  slots r{m['i']:02d} n={len(m['text'])} box_t={box_t}")
    for k, s in enumerate(slots(len(m["text"]), box_t)[:6]):
        print(f"    {k} {m['text'][k]} {tuple(round(v, 1) for v in s)}")


def run(specs: list, label: str) -> None:
    import numpy as np
    from PIL import Image

    from eval.ruler.arms import tables
    from eval.ruler.read import read
    from eval.ruler.render import Renderer

    P = plan(items())
    VIEW.only = {i for i, v in P.items() if v["box"]}  # render and read these only
    its = sorted(items(), key=lambda m: (m["shape"], m["i"]))
    _, tabs = tables([ROWS])
    r = Renderer()
    r.set_arm(tabs[ROWS])
    sp = Spatial(r.anima, "box", 0.0)
    # the hooks with no box reproduce the arm's cached render (session noise)
    m0 = its[0]
    fn = mode_dir() / "_tmp" / "xattn_check.png"
    fn.unlink(missing_ok=True)
    r.shared["conds_cache"].clear()
    sp.reset(None)
    r.render(fn, m0["prompts"][VIEW.mode]["caption"], 0, m0["shape"])
    d = float(
        np.abs(
            np.asarray(Image.open(fn), np.float32)
            - np.asarray(
                Image.open(render_file(ROWS, m0["i"])).convert("RGB"), np.float32
            )
        ).mean()
    )
    print(f"check {ROWS} r{m0['i']:02d} hooks idle: mean |Δpx| {d:.3f}", flush=True)
    assert d < 12.0, "the hooked path does not reproduce the cached render"
    names = []
    for spec in specs:
        a, sp.mode, sp.tags, sp.beta = arm_name(spec)
        names.append(a)
        for n, m in enumerate(its):
            f = render_file(a, m["i"])
            if f.exists():
                continue
            r.shared["conds_cache"].clear()  # the span is read at encode
            sp.reset(P[m["i"]]["box"])
            b0 = sp.log["biased"]
            r.render(f, m["prompts"][VIEW.mode]["caption"], 0, m["shape"])
            nb = sp.log["biased"] - b0
            if not nb:
                f.unlink()  # an unbiased render must not pass for the arm's
            assert nb > 0, f"r{m['i']:02d}: no biased forward (span {sp.span})"
            if sp.tags:
                assert sp.tag_pos, f"r{m['i']:02d}: no tag tokens"
            print(
                f"  {a}: {n + 1} / {len(its)} r{m['i']:02d} span {sp.span} "
                f"tags {len(sp.tag_pos)} biased {nb} fwd",
                flush=True,
            )
        (mode_dir() / a / "xattn_plan.json").write_text(
            json.dumps(
                {
                    "mode": sp.mode,
                    "tags": sp.tags,
                    "beta": sp.beta,
                    "band": BAND,
                    "pad": PAD,
                    "items": P,
                    "hook_log": sp.log,
                },
                indent=1,
            )
        )
    del r, sp
    import torch

    torch.cuda.empty_cache()
    run_dir = read(
        [x for x in dict.fromkeys([*FLOOR[1:], ROWS, *names])], label, __file__
    )
    recs = json.loads((run_dir / "renders.json").read_text("utf-8"))
    out = {}
    for a in [ROWS, *names]:
        by_bin: dict = {}
        for rec in recs[a]:
            v = in_box(rec["reads"], P[rec["i"]]["box"])
            if v is not None:
                by_bin.setdefault(rec["bin"], []).append(v)
                by_bin.setdefault("all", []).append(v)
        out[a] = {b: round(sum(v) / len(v), 3) for b, v in by_bin.items()}
        print(f"  in_box {a}: {out[a]}", flush=True)
    (run_dir / "in_box.json").write_text(json.dumps(out, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=["boxes", "run"])
    p.add_argument(
        "--arms",
        default="xb4,xs4",
        help="xb<β> (box) / xs<β> (slot) / xt<β> (box+tags)",
    )
    p.add_argument("--label", default="xattn_box")
    a = p.parse_args()
    VIEW.pack = "punct"
    from reseed.config import PACKS

    os.environ["ANIMA_VOCAB_PACK"] = PACKS[VIEW.pack]
    os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
    if a.verb == "boxes":
        boxes()
    else:
        run([s for s in a.arms.split(",") if s], a.label)


if __name__ == "__main__":
    main()
