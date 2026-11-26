#!/usr/bin/env python
"""Reader room — where the seconds of G1's reader side go, and what a change would buy.

G1's group spends its ``read`` and ``loss`` phases in the detector and the two readers
(``src/g1.py``: one page a call, since the last step's graph holds one render). This
probe runs that side alone, without the DiT, on pages a G1 run wrote
(``output/cjk_rl/g1/<run>/read/<arm>/``), a group being one glyph on one scene at its 4
seeds. Per group, in turn:

- ``serial``: G1's order. Per page the read, the pick, the loss (the tuned reader's
  cross-entropy plus hayai's at ``hayai_weight``, or the detector's score on a page
  with no box) and its backward to the image.
- ``split``: the same with a timer on each part: the tensor → BGR copy, the
  detector's prepare / forward / boxes, the reader's processor, ``generate`` and the
  forwards inside it, and in the loss the processor, the reader's forward, hayai's
  term, the detector's term, the backward.
- ``batch``: the 4 pages read in one call, one loss and one backward for the group.
- ``cap`` / ``batch_cap``: ``serial`` / ``batch`` with the reader's ``max_new_tokens``
  at ``cap`` (the asked string is one glyph).
- ``lone``: ``serial`` with a one-box page left unread (``gd.Reader.read(skip_lone=
  True)``, G1's ``skip_lone_read``).
- ``sdpa``: ``serial`` with the tuned reader on SDPA (``serial`` runs on the config's
  ``reader_attn``, ``flash_attention_2`` when absent).

A config's ``variants`` picks which of ``batch`` / ``cap`` / ``batch_cap`` / ``lone`` /
``sdpa`` run beside ``serial`` (absent: the first three), and ``split = false`` leaves
the timer pass out. ``reader_compile = true`` compiles the tuned reader's layers on
``reader_attn`` (``gd.Reader.compile``: the loss forward and backward; the read stays
eager), and ``warm_groups`` (1 when absent) groups run twice untimed before the clock.

Beside the seconds: how often a variant's pick (case, box, text) equals ``serial``'s,
the cosine of its image gradient with ``serial``'s where the box is the same, the
decode steps a ``generate`` call ran, and the GPU's kernel seconds inside the read and
the loss of the first ``prof_groups`` groups (``torch.profiler``) against their wall
seconds. For ``lone``, on the pages it left unread: the loss forward's argmax being the
glyph (``gd.Reader.ce``'s ``hit``) against ``serial``'s read being the glyph, and
``serial``'s case on those pages.

The run's settings, question and gate are its config's header
(``configs/<handle>_<mmdd>_<hhmm>.toml``). Result → ``results/<ts>-reader_room-<stem>/``.

    make daemon-run ARGS="--queue --stall-timeout 600 project/cjk_rl/probes/reader_room.py --config project/cjk_rl/configs/<name>.toml"
"""

from __future__ import annotations

import argparse
import functools
import statistics
import time
import tomllib
from pathlib import Path

import glyph_g0 as G
import turbo_read as T

import gd  # noqa: E402  (src/, on the path through turbo_read)

PAGES = T.REPO / "output" / "cjk_rl" / "g1"


class Clock:
    """Seconds by name, taken only while ``tag`` is set."""

    def __init__(self):
        self.tag: str | None = None
        self.s: dict = {}
        self.n: dict = {}
        self.steps: list | None = None  # (crops, decode steps) a generate call

    def add(self, name: str, dt: float) -> None:
        self.s[name] = self.s.get(name, 0.0) + dt
        self.n[name] = self.n.get(name, 0) + 1


CK = Clock()


def now() -> float:
    import torch

    torch.cuda.synchronize()
    return time.perf_counter()


def timed(f, name: str):
    # wraps: generate checks its kwargs against forward's signature
    @functools.wraps(f)
    def g(*a, **k):
        if CK.tag is None:
            return f(*a, **k)
        t = now()
        try:
            return f(*a, **k)
        finally:
            CK.add(f"{CK.tag}.{name}", now() - t)

    return g


class Proxy:
    """A processor whose call is timed; everything else is the processor's."""

    def __init__(self, inner, name: str):
        self._inner = inner
        self._call = timed(inner.__call__, name)

    def __call__(self, *a, **k):
        return self._call(*a, **k)

    def __getattr__(self, n):
        return getattr(self._inner, n)


def instrument(rdr) -> None:
    det, ocr = rdr.det, rdr.ocr
    for attr in ("prepare", "forward_batch", "detect_batch"):
        object.__setattr__(det, attr, timed(getattr(det, attr), f"det_{attr}"))
    object.__setattr__(ocr, "processor", Proxy(ocr.processor, "proc"))
    object.__setattr__(ocr.model, "forward", timed(ocr.model.forward, "fwd"))
    gen = timed(ocr.model.generate, "generate")

    def generate(*a, **k):
        out = gen(*a, **k)
        if CK.steps is not None:
            n = k["input_ids"].shape
            CK.steps.append((int(n[0]), int(out.shape[-1] - n[-1])))
        return out

    object.__setattr__(ocr.model, "generate", generate)


def load_groups(cfg: dict, smoke: bool) -> list:
    """``[(arm, key, scene, [uint8 RGB (H, W, 3)] × seeds)]``: a glyph of each arm on
    one scene, the scene cycling over the glyphs."""
    import numpy as np
    from PIL import Image

    root = PAGES / cfg["pages"] / "read"
    keys = list(cfg.get("hira", "") + cfg.get("kata", "") + cfg.get("kanji", ""))
    out = []
    for arm in cfg["arms"]:
        for n, key in enumerate(keys):
            si = cfg["scenes"][n % len(cfg["scenes"])]
            fs = [root / arm / f"{key}_sc{si:02d}_s{s}.webp" for s in cfg["seeds"]]
            out.append(
                (arm, key, si, [np.asarray(Image.open(f).convert("RGB")) for f in fs])
            )
    return out[:: len(out) // 4] if smoke else out


def to_px(pages: list):
    """``(B, 3, H, W)`` in [-1, 1] on the GPU, as the pixel head hands a render over."""
    import numpy as np
    import torch

    x = torch.from_numpy(np.stack(pages)).cuda().permute(0, 3, 1, 2).float()
    return x / 127.5 - 1.0


def picked(key: str, reads: list, conf_min: float) -> tuple:
    case, r = gd.pick(key, reads, conf_min)
    return case, r, (case, None if r is None else (tuple(r["box"]), r["sfx"]))


def loss_page(rdr, hy, cfg: dict, key: str, img, bgrs, r, laps: bool = False, hit=None):
    """G1's loss on one page and its gradient on the image ``(1, 3, H, W)``. ``hit``:
    ``gd.Reader.ce``'s."""
    import torch

    W, H = G.SHAPE
    t = now() if laps else 0.0

    def lap(name):
        nonlocal t
        if laps:
            CK.add(f"loss.{name}", now() - t)
            t = now()

    leaf = img.clone().requires_grad_(True)
    p01 = (gd.ste_clamp(leaf, -1.0, 1.0) + 1) / 2
    lap("leaf")
    if r is None:
        top = rdr.det_scores(p01).max(1).values
        total = top.clamp_min(1e-6).log().neg().sum()
        lap("det_term")
    else:
        box = gd.crop_box(r["box"], W, H)
        total = rdr.ce([p01[0]], bgrs, [box], [key], hit=hit).sum()
        lap("ce")
        if hy is not None:
            total = total + cfg["hayai_weight"] * hy.ce([p01[0]], [box], [key]).sum()
            lap("hayai")
    g = torch.autograd.grad(total, leaf)[0]
    lap("backward")
    return g


def loss_group(rdr, hy, cfg: dict, key: str, imgs, bgrs, picks: list):
    """The same losses for a group's pages in one graph: the no-box pages share a
    detector forward, the others a reader forward; one backward."""
    import torch

    W, H = G.SHAPE
    leaf = imgs.clone().requires_grad_(True)
    p01 = (gd.ste_clamp(leaf, -1.0, 1.0) + 1) / 2
    none = [b for b, r in enumerate(picks) if r is None]
    has = [b for b, r in enumerate(picks) if r is not None]
    total = 0.0
    if none:
        top = rdr.det_scores(p01[none]).max(1).values
        total = total + top.clamp_min(1e-6).log().neg().sum()
    if has:
        boxes = [gd.crop_box(picks[b]["box"], W, H) for b in has]
        pages = [p01[b] for b in has]
        total = (
            total + rdr.ce(pages, [bgrs[b] for b in has], boxes, [key] * len(has)).sum()
        )
        if hy is not None:
            aux = hy.ce(pages, boxes, [key] * len(has)).sum()
            total = total + cfg["hayai_weight"] * aux
    return torch.autograd.grad(total, leaf)[0]


def kernel_seconds(fn) -> float | None:
    """The GPU's kernel seconds inside ``fn`` (``torch.profiler``)."""
    import torch
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    try:
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            fn()
            torch.cuda.synchronize()
        us = 0.0
        for e in prof.key_averages():
            if e.device_type == DeviceType.CPU or getattr(e, "is_user_annotation", 0):
                continue
            us += getattr(e, "self_device_time_total", None) or getattr(
                e, "self_cuda_time_total", 0.0
            )
        return us / 1e6
    except Exception as err:  # the profiler is a side read
        print(f"  profiler: {type(err).__name__}: {err}", flush=True)
        return None


def summary(xs: list) -> dict:
    xs = sorted(xs)
    return {
        "mean": round(statistics.fmean(xs), 4),
        "median": round(statistics.median(xs), 4),
        "p90": round(xs[int(0.9 * (len(xs) - 1))], 4),
        "max": round(xs[-1], 4),
        "n": len(xs),
    }


def run(cfg: dict, smoke: bool) -> dict:
    import anime_tools.ocr.reader as R
    import numpy as np
    import torch

    from library.inference.output import pixels_to_pil

    groups = load_groups(cfg, smoke)
    attn = cfg.get("reader_attn", "flash_attention_2")
    rdr = gd.Reader("cuda", attn=attn)
    if cfg.get("reader_compile"):
        rdr.compile()
    hy = gd.Hayai("cuda") if cfg.get("hayai_weight") else None
    instrument(rdr)
    conf, full = cfg["conf_min"], R.MAX_NEW_TOKENS
    B = len(cfg["seeds"])
    extra = tuple(cfg.get("variants", ("batch", "cap", "batch_cap")))
    names = ("serial", *extra)
    sec = {v: {"read": [], "loss": []} for v in names}  # seconds a group
    page_read = {v: [] for v in ("serial", "cap", "lone", "sdpa")}  # seconds a page
    # lone's unread pages: serial's case, and the forward's argmax against its read
    lone = {"pages": 0, "serial_case": {"box": 0, "low": 0}}
    lone |= dict.fromkeys(("both", "read_only", "argmax_only", "neither"), 0)
    steps = {v: [] for v in names}
    agree = {v: {"case": 0, "box": 0, "text": 0, "n": 0} for v in names[1:]}
    gcos = {v: [] for v in names[1:]}
    cases = dict.fromkeys(("box", "low", "none"), 0)
    kept = []  # the first groups' serial reads, for the profiler

    def serial(v: str, key: str, px, skip_lone: bool = False):
        CK.steps = steps[v]
        out, tr, tl = [], 0.0, 0.0
        for b in range(B):
            img = px[b : b + 1]
            t = now()
            bgrs, reads = rdr.read(img, skip_lone=skip_lone)
            dt = now() - t
            page_read[v].append(dt)
            tr += dt
            t = now()
            _, r, sig = picked(key, reads[0], conf)
            hit: list = []
            g = loss_page(
                rdr, hy, cfg, key, img, bgrs, r, hit=hit if skip_lone else None
            )
            tl += now() - t
            out.append((sig, g, bgrs, r, hit[0] if hit else None))
        sec[v]["read"].append(tr)
        sec[v]["loss"].append(tl)
        return out

    def batch(v: str, key: str, px):
        CK.steps = steps[v]
        t = now()
        bgrs, reads = rdr.read(px)
        sec[v]["read"].append(now() - t)
        t = now()
        ps = [picked(key, reads[b], conf) for b in range(B)]
        g = loss_group(rdr, hy, cfg, key, px, bgrs, [p[1] for p in ps])
        sec[v]["loss"].append(now() - t)
        return [(ps[b][2], g[b : b + 1]) for b in range(B)]

    def compare(v: str, ref: list, got: list) -> None:
        for (s0, g0, *_), (s1, g1, *_) in zip(ref, got):
            a = agree[v]
            a["n"] += 1
            a["case"] += s0[0] == s1[0]
            same = s0[1] is not None and s1[1] is not None and s0[1][0] == s1[1][0]
            a["box"] += same or (s0[1] is None and s1[1] is None)
            a["text"] += s0[1] == s1[1]
            if same or (s0[1] is None and s1[1] is None):
                c = torch.nn.functional.cosine_similarity(
                    g0.flatten().double(), g1.flatten().double(), dim=0
                )
                gcos[v].append(float(c))

    def count_lone(key: str, ref: list, got: list) -> None:
        from common.text import norm

        for (s0, *_), (s1, *_, argmax) in zip(ref, got):
            if s1[0] != "lone":
                continue
            lone["pages"] += 1
            lone["serial_case"][s0[0]] += 1
            read = norm(s0[1][1] or "") == norm(key)
            lone[
                "both"
                if read and argmax
                else "read_only"
                if read
                else "argmax_only"
                if argmax
                else "neither"
            ] += 1

    # the first groups twice through every variant, untimed
    t_warm = time.time()
    for _ in range(2):
        for _, wkey, _, wpages in groups[: cfg.get("warm_groups", 1)]:
            warm = to_px(wpages)
            serial("serial", wkey, warm)
            if "lone" in names:
                serial("lone", wkey, warm, skip_lone=True)
            if {"batch", "batch_cap"} & set(names):
                batch("batch" if "batch" in names else "batch_cap", wkey, warm)
            if "sdpa" in names:
                rdr.set_attn("sdpa")
                serial("sdpa", wkey, warm)
                rdr.set_attn(attn)
    print(f"  warm {time.time() - t_warm:.0f} s", flush=True)
    for xs in (*page_read.values(), *steps.values()):
        xs.clear()
    for d in sec.values():
        d["read"].clear()
        d["loss"].clear()

    t_run = time.time()
    for n, (arm, key, si, pages) in enumerate(groups):
        px = to_px(pages)
        ref = serial("serial", key, px)
        for sig, *_ in ref:
            cases[sig[0]] += 1
        if n < cfg["prof_groups"]:
            kept.append((key, px, ref))

        # the same order with a timer on each part
        CK.steps, CK.tag = None, "read"
        for b in range(B if cfg.get("split", True) else 0):
            img = px[b : b + 1]
            t = now()
            bgrs = [
                np.ascontiguousarray(np.asarray(pixels_to_pil(p))[:, :, ::-1])
                for p in img.detach()
            ]
            CK.add("read.to_bgr", now() - t)
            t = now()
            reads = rdr.rd.read_images(bgrs)
            CK.add("read.read_images", now() - t)
            CK.tag = "loss"
            t = now()
            _, r, _ = picked(key, reads[0], conf)
            CK.add("loss.pick", now() - t)
            loss_page(rdr, hy, cfg, key, img, bgrs, r, laps=True)
            CK.tag = "read"
        CK.tag = None

        if "batch" in names:
            compare("batch", ref, batch("batch", key, px))
        if "lone" in names:
            got = serial("lone", key, px, skip_lone=True)
            compare("lone", ref, got)
            count_lone(key, ref, got)
        if "sdpa" in names:
            rdr.set_attn("sdpa")
            compare("sdpa", ref, serial("sdpa", key, px))
            rdr.set_attn(attn)
        R.MAX_NEW_TOKENS = cfg.get("cap", full)
        if "cap" in names:
            compare("cap", ref, serial("cap", key, px))
        if "batch_cap" in names:
            compare("batch_cap", ref, batch("batch_cap", key, px))
        R.MAX_NEW_TOKENS = full
        if n % 8 == 7:
            print(
                f"  group {n + 1} / {len(groups)} ({time.time() - t_run:.0f} s)",
                flush=True,
            )
    CK.steps = None

    # the GPU's kernel seconds inside the first groups' read and loss, G1's order
    def reads_again():
        for key, px, _ in kept:
            for b in range(B):
                rdr.read(px[b : b + 1])

    def losses_again():
        for key, px, ref in kept:
            for b in range(B):
                _, _, bgrs, r, _ = ref[b]
                loss_page(rdr, hy, cfg, key, px[b : b + 1], bgrs, r)

    k = len(kept)
    busy = {
        "groups": k,
        "read_kernel_s": kernel_seconds(reads_again),
        "read_wall_s": round(sum(sec["serial"]["read"][:k]), 3),
        "loss_kernel_s": kernel_seconds(losses_again),
        "loss_wall_s": round(sum(sec["serial"]["loss"][:k]), 3),
    }

    n_pages = len(groups) * B
    out = {
        "groups": len(groups),
        "pages": n_pages,
        "cases": cases,
        "s_per_group": {
            v: {p: round(statistics.fmean(xs), 4) for p, xs in d.items()}
            for v, d in sec.items()
        },
        "s_per_group_median": {
            v: {p: round(statistics.median(xs), 4) for p, xs in d.items()}
            for v, d in sec.items()
        },
        "page_read_s": {v: summary(xs) for v, xs in page_read.items() if xs},
        # a timer on each part, seconds a group (a part's mean a page × the pages)
        "split_s_per_group": {
            name: round(s / n_pages * B, 4) for name, s in sorted(CK.s.items())
        },
        "split_calls_per_page": {
            name: round(c / n_pages, 2) for name, c in sorted(CK.n.items())
        },
        "generate": {
            v: {
                "calls": len(xs),
                "crops": summary([c for c, _ in xs]),
                "steps": summary([s for _, s in xs]),
                "at_cap": sum(s >= cfg.get("cap", full) for _, s in xs),
            }
            for v, xs in steps.items()
            if xs
        },
        "agree_with_serial": agree,
        "lone": lone if "lone" in names else None,
        "grad_cos_with_serial": {v: summary(xs) for v, xs in gcos.items() if xs},
        "gpu_busy": busy,
        "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="its stem names the run")
    ap.add_argument("--smoke", action="store_true", help="4 groups")
    a = ap.parse_args()
    cfg = tomllib.loads(Path(a.config).read_text("utf-8"))
    name = ("smoke_" if a.smoke else "") + Path(a.config).stem
    m = run(cfg, a.smoke)
    for k, v in m.items():
        print(f"  {k}: {v}", flush=True)

    from bench._common import make_run_dir, write_result

    line = Path(__file__).resolve().parents[1]
    run_dir = make_run_dir("cjk_rl", label=f"reader_room-{name}", root=line / "results")
    write_result(
        run_dir,
        script="project/cjk_rl/probes/reader_room.py",
        args=vars(a) | {"cfg": cfg},
        label=name,
        metrics=m,
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
