#!/usr/bin/env python
"""grpo_p0 — OCR-GRPO's forward probes (``docs/anima/proposal/cjk_ocr_grpo.md`` § 4 P0).

The policy is the ruler's ``m14v3`` (merged14 + Hub turbo v3, 4 steps, cfg 1) with
jp_v1's rows (``seed_1008``, punct pack, routed). 16 RL-pool strings (Manga109-s
dialogue, 2–20 cells, every glyph a pack row, no 5-gram of a ruler string), each
on its own speech-bubble prompt (the ruler's ``sensitive`` grammar, the ruler's
source images left out), 8 seeds each.

    build   (CPU) the 16 (string, prompt) → ``output/cjk_anima_reseed/grpo_p0/pool.json``;
            the EN lines are written into it by hand (``en``), as the ruler's are
    render  (GPU) per string × seed: the EN ref (Euler) and the policy on four
            samplers — Euler, SDE a ∈ {0.3, 0.7} (Flow-GRPO), LCM re-noise
    cost    (GPU) P0c: one GRPO update, G = 8 SDE rollouts, backward on 2 of 4
            steps through the rows: wall time and peak VRAM
    read    readers + the reward terms + P0a pairs / P0b spreads + sheets
            → ``results/<ts>-grpo_p0/``
    run     render, cost, read

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/grpo_p0.py run"

The SDE (Flow-GRPO, eq. of ``sde_step_with_logprob``): at σ → σ' (dt = σ' − σ < 0),
std = a·√(σ / (1 − σ)) (σ = 1 takes σ₁ in the denominator), mean = x·(1 + std²·dt / 2σ)
+ v·(1 + std²(1 − σ) / 2σ)·dt, x' = mean + std·√(−dt)·ε. The last step (σ' = 0) is the
x̂₀ (as LCM's): 4 steps leave σ = 0.5 before it, and noise there stays in the image.
LCM: x' = (1 − σ')·x̂₀ + σ'·ε. The noise of both draws from seed + ``NOISE_OFF``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))

from reseed import HOME as RHOME, OUT, bootstrap  # noqa: E402

bootstrap()

from reseed.config import PACKS  # noqa: E402

os.environ["ANIMA_VOCAB_PACK"] = PACKS["punct"]
os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"

from eval.ruler import BINS, RULER, VIEW  # noqa: E402

VIEW.pack = "punct"
ROWS = "seed_1008"  # jp_v1
MODEL = "m14v3"
P0 = OUT / "grpo_p0"
SEED = 1113
PER_BIN = {"short": 5, "mid": 5, "long": 6}
SEEDS = range(8)
SAMPLERS = ("euler", "sde03", "sde07", "lcm")
NOISE = {"sde03": 0.3, "sde07": 0.7}
NOISE_OFF = 7919
LATIN = re.compile(r"[A-Za-z0-9Ａ-Ｚａ-ｚ０-９]")


# ---------------------------------------------------------------- build (CPU)


def pick_strings(ruler_texts: list) -> dict:
    """bin → strings: the ``sent`` rules (``reseed.pools.add_sentences``) on the
    Manga109-s pack lines, 2–20 cells, a ruler 5-gram held out."""
    from eval.ruler.build import length
    from reseed.config import LINES
    from reseed.pools import SENT_BASE, cells, ext_encoder, norm_ellipsis

    r5 = {r[i : i + 5] for r in ruler_texts for i in range(len(r) - 4)}
    ext = ext_encoder()
    src = Path(os.path.expandvars(LINES["m109_pack"])).expanduser()
    lines = [ln.split("\t")[0].strip() for ln in src.read_text("utf-8").splitlines()]
    by: dict = {b: [] for b in BINS}
    rng = random.Random(SEED)
    rng.shuffle(lines)
    for t in dict.fromkeys(lines):
        if all(len(v) >= 40 for v in by.values()):
            break
        s = norm_ellipsis(t)
        if s is None or LATIN.search(s) or not 2 <= cells(s) <= 20:
            continue
        if re.search(r"([^…])\1\1", s) or any(
            s[i : i + 5] in r5 for i in range(len(s) - 4)
        ):
            continue
        n = length(s)
        b = next((k for k, (lo, hi) in BINS.items() if lo <= n <= hi), None)
        if b is None or len(by[b]) >= 40:
            continue
        ids = [ext.glyph_row(c) for c in s]
        if any(i is None and c not in SENT_BASE for c, i in zip(s, ids)):
            continue
        if ext(True, s) != [i for i in ids if i is not None]:
            continue
        by[b].append(s)
    return {b: by[b][: PER_BIN[b]] for b in BINS}


def build() -> None:
    from anime_tools.captions.correction import find_tag_csv, load_tag_knowledge_base
    from anime_tools.captions.position_clauses import compose_caption, text_clause

    from eval.ruler.build import pool, prompt_of, shape_of
    from reseed import REPO

    ruler = json.loads((RULER / "ruler.json").read_text("utf-8"))["items"]
    stems = {m["stem"] for m in ruler}
    kb = load_tag_knowledge_base(find_tag_csv(REPO))
    imgs, _ = pool()
    free = sorted(st for st in imgs if st not in stems)
    print(f"{len(imgs)} bubble images, {len(free)} off the ruler", flush=True)
    strs = pick_strings([m["text"] for m in ruler])
    random.Random(SEED).shuffle(free)
    old = {}
    if (P0 / "pool.json").exists():
        old = {
            m["text"]: m.get("en")
            for m in json.loads((P0 / "pool.json").read_text("utf-8"))
        }
    items, k = [], 0
    for b, xs in strs.items():
        for s in xs:
            st = free[k]
            k += 1
            pr = prompt_of(imgs[st][3], kb)
            items.append(
                {
                    "i": len(items),
                    "bin": b,
                    "text": s,
                    "stem": st,
                    "shape": shape_of(imgs[st][2]),
                    "prompt": pr,
                    "caption": compose_caption([pr.rstrip(".")], [text_clause([s])]),
                    "en": old.get(s),
                }
            )
    P0.mkdir(parents=True, exist_ok=True)
    (P0 / "pool.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1), "utf-8"
    )
    for m in items:
        print(f"{m['i']:2d} {m['bin']:<5} {m['shape']} {m['text']}  | en={m['en']}")


def load_pool() -> list:
    its = json.loads((P0 / "pool.json").read_text("utf-8"))
    miss = [m["text"] for m in its if not m.get("en")]
    assert not miss, f"pool.json: no EN line for {miss}"
    for m in its:
        m["en_caption"] = (
            re.sub(r"\bjapanese text\b", "english text", m["prompt"]).rstrip(".")
            + f'. English text reads as "{m["en"]}".'
        )
    return its


def render_file(kind: str, i: int, seed: int) -> Path:
    return P0 / kind / f"r{i:02d}_s{seed}.png"


# ---------------------------------------------------------------- samplers


class SDE:
    """Flow-GRPO's marginal-preserving SDE step (``.step`` as ``LCMSampler``'s:
    ``latents``, x̂₀ = ``denoised``, step ``i``); the last step is x̂₀."""

    def __init__(self, sigmas, a: float, seed: int, device):
        import torch

        self.s = sigmas.float().tolist()
        self.a = a
        self.g = torch.Generator(device=device).manual_seed(seed + NOISE_OFF)
        self.device = device

    def std(self, i: int) -> float:
        s = self.s[i]
        den = 1 - (self.s[1] if s >= 1.0 else s)
        return self.a * (s / den) ** 0.5

    def mean(self, x, v, i: int):
        s, dt = self.s[i], self.s[i + 1] - self.s[i]
        sd = self.std(i)
        return x * (1 + sd**2 / (2 * s) * dt) + v * (1 + sd**2 * (1 - s) / (2 * s)) * dt

    def scale(self, i: int) -> float:
        return self.std(i) * (self.s[i] - self.s[i + 1]) ** 0.5

    def step(self, latents, denoised, i: int):
        import torch

        if self.s[i + 1] == 0.0:
            return denoised
        x = latents.float()
        v = (x - denoised) / self.s[i]
        eps = torch.randn(x.shape, generator=self.g, device=self.device, dtype=x.dtype)
        return self.mean(x, v, i) + self.scale(i) * eps


class LCM:
    def __init__(self, sigmas, seed: int, device):
        import torch

        self.s = sigmas.float().tolist()
        self.g = torch.Generator(device=device).manual_seed(seed + NOISE_OFF)
        self.device = device

    def step(self, latents, denoised, i: int):
        import torch

        sn = self.s[i + 1]
        d = denoised.float()
        if sn == 0.0:
            return d
        eps = torch.randn(d.shape, generator=self.g, device=self.device, dtype=d.dtype)
        return (1 - sn) * d + sn * eps


_KIND = {"name": "lcm"}


def _sampler_factory(sigmas, seed=None, device=None):
    k = _KIND["name"]
    if k in NOISE:
        return SDE(sigmas, NOISE[k], seed, device)
    return LCM(sigmas, seed, device)


# ---------------------------------------------------------------- render (GPU)


def make_renderer():
    from eval.ruler.arms import tables
    from eval.ruler.render import Renderer
    from library.inference import generation

    # generate's `lcm` slot builds the sampler named by _KIND
    generation.inference_utils.LCMSampler = _sampler_factory
    r = Renderer(MODEL)
    _, tabs = tables([ROWS])
    return r, tabs[ROWS]


def render(r, table) -> dict:
    import copy

    import torch

    from common.models import decode_image
    from library.inference.generation import generate

    its = sorted(load_pool(), key=lambda m: (m["shape"], m["i"]))
    t0, n, times = time.time(), 0, {}
    for m in its:
        for kind in ("en", *SAMPLERS):
            r.set_arm(None if kind == "en" else table)
            for seed in SEEDS:
                fn = render_file(kind, m["i"], seed)
                if fn.exists():
                    continue
                a2 = copy.deepcopy(r.args)
                a2.prompt = m["en_caption"] if kind == "en" else m["caption"]
                a2.seed = seed
                a2.image_size = (m["shape"][1], m["shape"][0])
                a2.sampler = "euler" if kind in ("en", "euler") else "lcm"
                _KIND["name"] = kind
                ts = time.time()
                with torch.no_grad():
                    lat = generate(a2, r.gen, r.shared)
                fn.parent.mkdir(parents=True, exist_ok=True)
                decode_image(r.vae, lat, r.device).save(fn)
                times.setdefault(kind, []).append(time.time() - ts)
                n += 1
        print(
            f"  r{m['i']:02d} {m['shape']} ({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )
    # the median drops each shape's compile
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    info = {
        "rendered": n,
        "minutes": round((time.time() - t0) / 60, 2),
        "median_s": med,
    }
    print(f"render: {info}", flush=True)
    return info


# ---------------------------------------------------------------- cost (GPU)


def cost(r, table, G: int = 8, grad_steps=(0, 1)) -> dict:
    """P0c: G SDE rollouts (a = 0.7) of one prompt, then one update: −A·log π
    of the steps ``grad_steps`` backwarded through the rows, A a stand-in
    (the time and memory do not depend on it)."""
    import torch

    from common.models import decode_image, dit_forward, encode_captions
    from library.inference.sampling import get_timesteps_sigmas

    m = load_pool()[0]
    dev = r.device
    r.set_arm(table)
    cache = encode_captions([m["caption"]], dev)
    W, H = m["shape"]
    _, sig = get_timesteps_sigmas(4, r.args.flow_shift, dev)
    sde = SDE(sig, 0.7, 0, dev)
    s = sde.s
    anima, raw = r.anima, r.delta.raw
    raw.requires_grad_(True)

    def v_of(x, i, grad):
        ts = torch.full((1,), s[i], device=dev, dtype=torch.float32)
        with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
            return dit_forward(anima, x, ts, cache, [m["caption"]], dev).float()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    traj = []
    for g in range(G):
        gen = torch.Generator(device=dev).manual_seed(g)
        x = torch.randn(1, 16, H // 8, W // 8, generator=gen, device=dev)
        xs = [x]
        for i in range(4):
            v = v_of(x, i, False)
            x0 = x - s[i] * v
            x = sde.step(x, x0, i)
            xs.append(x)
        traj.append(xs)
    torch.cuda.synchronize()
    t_roll = time.time() - t0
    t0 = time.time()
    for xs in traj:
        decode_image(r.vae, xs[-1].unsqueeze(2), dev)
    torch.cuda.synchronize()
    t_dec = time.time() - t0
    peak_roll = torch.cuda.max_memory_allocated() / 2**30
    adv = torch.linspace(-1, 1, G)
    raw.grad = None
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    lps = []
    for g, xs in enumerate(traj):
        for i in grad_steps:
            v = v_of(xs[i], i, True)
            mu = sde.mean(xs[i], v, i)
            sc = sde.scale(i)
            lp = (-((xs[i + 1] - mu) ** 2) / (2 * sc**2)).mean()
            (-(adv[g] * lp) / (G * len(grad_steps))).backward()
            lps.append(float(lp))
    torch.cuda.synchronize()
    t_upd = time.time() - t0
    gn = float(raw.grad.norm()) if raw.grad is not None else 0.0
    out = {
        "G": G,
        "grad_steps": list(grad_steps),
        "shape": m["shape"],
        "rollout_s": round(t_roll, 2),
        "decode_s": round(t_dec, 2),
        "update_s": round(t_upd, 2),
        "peak_rollout_gib": round(peak_roll, 2),
        "peak_update_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "card_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "row_grad_norm": gn,
        "rows_with_grad": int((raw.grad.abs().sum(1) > 0).sum())
        if raw.grad is not None
        else 0,
        "logp_mean": sum(lps) / len(lps),
    }
    raw.grad = None
    raw.requires_grad_(False)
    print(f"cost: {out}", flush=True)
    return out


# ---------------------------------------------------------------- read


def lcs(a: str, b: str) -> int:
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def r_seq(text: str, reads: list) -> float:
    from common.text import norm

    from eval.ruler.score import candidates

    t = norm(text)
    cs = candidates(reads, "sfx")
    return max((lcs(t, c) for c in cs), default=0) / max(1, len(t))


def read_all() -> tuple[dict, float]:
    """``{kind: {(i, seed): rec}}``; reads cached in ``<kind>/reads.json``."""
    import torch

    from common.readers import Readers, load_bgr
    from eval.ruler.score import AtSim, flat_white, score_page, score_text

    its = load_pool()
    raw, rd, vram = {}, None, 0.0
    for kind in ("en", *SAMPLERS):
        f = P0 / kind / "reads.json"
        got = json.loads(f.read_text("utf-8")) if f.exists() else {}
        todo = [(m["i"], s) for m in its for s in SEEDS if f"{m['i']}_{s}" not in got]
        if todo:
            if rd is None:
                torch.cuda.reset_peak_memory_stats()
                rd = Readers("cuda")
                vram = torch.cuda.memory_allocated() / 2**30
            t0 = time.time()
            for i, s in todo:
                got[f"{i}_{s}"] = rd.read_image(
                    load_bgr(render_file(kind, i, s)), whole=True
                )
            f.write_text(json.dumps(got, ensure_ascii=False, indent=1), "utf-8")
            print(f"read {kind}: {len(todo)} in {time.time() - t0:.0f} s", flush=True)
        raw[kind] = got
    del rd
    torch.cuda.empty_cache()
    at = AtSim("cuda")
    recs: dict = {}
    for kind in SAMPLERS:
        recs[kind] = {}
        for m in its:
            for s in SEEDS:
                k = f"{m['i']}_{s}"
                f, ref = render_file(kind, m["i"], s), render_file("en", m["i"], s)
                anchor = render_file("euler", m["i"], s)
                reads, en_reads = raw[kind][k], raw["en"][k]
                recs[kind][(m["i"], s)] = {
                    "i": m["i"],
                    "seed": s,
                    "bin": m["bin"],
                    "text": m["text"],
                    "file": str(f),
                    "r_seq": r_seq(m["text"], reads),
                    **score_text(m["text"], reads),
                    **score_page(m["text"], reads, en_reads),
                    **at(f, ref),
                    "m_init": 1.0 if kind == "euler" else at(f, anchor)["en_match"],
                    "fw_over_en": flat_white(str(f)) - flat_white(str(ref)),
                }
    return recs, vram


TERMS = ("r_seq", "g_p", "en_match", "cer", "drawn", "a_p", "m_init", "fw_over_en")


def sign_p(up: int, dn: int) -> float:
    from eval.ruler.stats import sign_p as sp

    return sp(up, dn)


def p0a(recs: dict) -> dict:
    """Each sampler against Euler, paired over the 128 (string, seed)."""
    out = {}
    e = recs["euler"]
    for kind in SAMPLERS[1:]:
        c = {}
        for t in TERMS:
            d = [recs[kind][k][t] - e[k][t] for k in e]
            up, dn = sum(x > 1e-9 for x in d), sum(x < -1e-9 for x in d)
            c[t] = {
                "mean_diff": round(sum(d) / len(d), 4),
                "up": up,
                "down": dn,
                "p": sign_p(up, dn),
            }
        worse = [
            t
            for t in ("r_seq", "g_p", "en_match")
            if c[t]["down"] > c[t]["up"] and c[t]["p"] < 0.05
        ]
        c["worse_on"] = worse
        out[kind] = c
    return out


def p0b(recs: dict) -> dict:
    """Per sampler: groups whose r_seq is not constant over the seeds; per
    term the mean within-group std and the std of the group means."""
    import statistics as st

    out = {}
    for kind, rs in recs.items():
        grp: dict = {}
        for (i, _), r in rs.items():
            grp.setdefault(i, []).append(r)
        c = {
            "groups": len(grp),
            "non_degenerate": sum(
                max(r["r_seq"] for r in g) - min(r["r_seq"] for r in g) > 1e-9
                for g in grp.values()
            ),
        }
        for t in ("r_seq", "g_p", "m_init", "fw_over_en", "en_match"):
            within = st.mean(st.pstdev([r[t] for r in g]) for g in grp.values())
            between = st.pstdev([st.mean(r[t] for r in g) for g in grp.values()])
            c[t] = {"within": round(within, 4), "between": round(between, 4)}
        out[kind] = c
    return out


def tally(recs: dict) -> dict:
    out = {}
    for kind, rs in recs.items():
        out[kind] = {}
        for g in ("all", *BINS):
            xs = [r for r in rs.values() if g == "all" or r["bin"] == g]
            out[kind][g] = {
                t: round(sum(r[t] for r in xs) / len(xs), 4) for t in TERMS
            } | {
                "exact": sum(r["exact"] for r in xs),
                "le2": sum(r["le2"] for r in xs),
                "n": len(xs),
            }
    return out


def sheets(recs: dict, out: Path, seeds=(0, 1), per: int = 4) -> None:
    from PIL import Image

    from common.readers import contact_sheet

    out.mkdir(parents=True, exist_ok=True)
    its = load_pool()
    for k in range(0, len(its), per):
        rows = []
        for m in its[k : k + per]:
            for s in seeds:
                rows.append(
                    (
                        Image.open(render_file("en", m["i"], s)).convert("RGB"),
                        [
                            f"r{m['i']:02d} s{s} {m['bin']}",
                            m["text"][:22],
                            f"EN {m['en'][:22]}",
                        ],
                    )
                )
                for kind in SAMPLERS:
                    r = recs[kind][(m["i"], s)]
                    rows.append(
                        (
                            Image.open(r["file"]).convert("RGB"),
                            [
                                kind,
                                r["best"][:22],
                                f"seq {r['r_seq']:.2f} gp {r['g_p']:.2f}",
                                f"enm {r['en_match']:.2f} init {r['m_init']:.2f}",
                            ],
                        )
                    )
        contact_sheet(
            rows, out / f"sheet_{k // per}.png", thumb=384, cols=1 + len(SAMPLERS)
        )


def read(info: dict) -> Path:
    from bench._common import make_run_dir, write_result

    recs, vram = read_all()
    t, a, b = tally(recs), p0a(recs), p0b(recs)
    for kind, g in t.items():
        print(f"  {kind:<6} {g['all']}", flush=True)
    for kind, c in a.items():
        print(
            f"  P0a {kind} vs euler: worse on {c['worse_on']}; "
            + ", ".join(
                f"{x} {c[x]['mean_diff']:+.3f} ({c[x]['up']}/{c[x]['down']}, p {c[x]['p']})"
                for x in TERMS
            ),
            flush=True,
        )
    for kind, c in b.items():
        print(f"  P0b {kind}: {c}", flush=True)
    run_dir = make_run_dir("cjk_anima_reseed", label="grpo_p0", root=RHOME / "results")
    sheets(recs, run_dir / "sheets")
    (run_dir / "renders.json").write_text(
        json.dumps(
            {k: list(v.values()) for k, v in recs.items()}, ensure_ascii=False, indent=1
        ),
        "utf-8",
    )
    info = info | {"readers_vram_gib": round(vram, 2)}
    write_result(
        run_dir,
        script=__file__,
        args={
            "rows": ROWS,
            "model": MODEL,
            "seeds": len(SEEDS),
            "samplers": SAMPLERS,
            "noise": NOISE,
        },
        label="grpo_p0",
        metrics={"tally": t, "p0a": a, "p0b": b, "info": info},
        artifacts=[str(P0)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)
    return run_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=["build", "render", "cost", "read", "run"])
    a = p.parse_args()
    if a.verb == "build":
        build()
        return
    info: dict = {}
    if a.verb in ("render", "cost", "run"):
        import gc

        import torch

        r, table = make_renderer()
        if a.verb in ("render", "run"):
            info["render"] = render(r, table)
        if a.verb in ("cost", "run"):
            info["cost"] = cost(r, table)
        (P0 / "info.json").write_text(json.dumps(info, indent=1))
        del r
        gc.collect()
        torch.cuda.empty_cache()
    if a.verb in ("read", "run"):
        if not info and (P0 / "info.json").exists():
            info = json.loads((P0 / "info.json").read_text())
        read(info)


if __name__ == "__main__":
    main()
