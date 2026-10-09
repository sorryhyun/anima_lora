#!/usr/bin/env python
"""grad_identity — how much of a row's step-0 gradient is glyph identity, per tier × σ

A training-free read of `hypothesis.md` H1: a row is one vector at every σ,
so the band only chooses *what gradient the row sees*. This splits that
gradient, at the grid_44 train leg's initial point, into what any row standing
in the glyph's slot receives (layout) and what only the drawn glyph's row
receives (identity).

Per sampled item and one slot whose glyph ``u`` (one of the 81 hiragana)
occurs exactly once in the item: the true caption ``C`` and ``m`` wrong
captions ``C^{v_j}`` — slot ``k``'s glyph replaced by a hiragana ``v_j``
absent from the item (and not a dakuten / small-kana sibling of ``u``). For
every σ of the grid one noise ε per (item, σ), shared by all ``1 + m``
captions and both caption sets. The DiT runs on the ``1 + m`` captions as one
batch over the repeated noisy latent; the per-sample trained losses
(`loss.box_share_fm_loss` with `train.BOX_SHARE*` / `GRID_BOX`, by the record's
``src``) are summed and backpropagated onto the rows. ``u`` sits in ``C``
only and ``v_j`` in ``C^{v_j}`` only (checked on the T5 ids: the sequences
differ at exactly one position, ``u``'s id against ``v_j``'s), so one
backward gives ``g_true`` = ∇row_u under ``C`` and ``g_wrong_j`` = ∇row_{v_j}
under ``C^{v_j}`` at once; the claim is checked against an unbatched run
on a few items before the loop (``verify`` in the result).

- ``layout`` = mean_j g_wrong_j, ``identity`` = g_true − layout;
  ``id_share`` = ‖identity‖ / ‖g_true‖;
- ``id2`` = the same against the mean of ``m − 1`` wrong (each leave-one-out,
  averaged) and ``null`` = ‖g_wrong_j − mean of the other wrong‖ / ‖g_wrong_j‖
  averaged over j — the same estimator on a row that does not match the
  image; ``excess`` = id2 − null is the readable zero;
- ``cos(g_true, layout)``, exposure ‖g_true‖, ``dloss`` = mean_j L(C^{v_j}) −
  L(C) (the teacher-forced caption leverage, cf_sense's quantity on JA rows).

The initial point is the grid_44 train leg's: the 82 rows (81 hiragana + ー)
cold at the pack rows (raw 0, `budget.COLD_KINDS`: singles start cold),
every other row a caption carries frozen at the old seed (`SEED_ROWS_0921`),
glyph routing on in-process, the raw pack (sha checked as `train.train`).

Pass 2 (``--render``): the same contrast with the row held fixed — row u,
the true caption, the image re-drawn (`data.grid.render_grid`, the record's
units / grid / canvas / bubble / fill, `grid_small`'s ``bubble_fit`` /
``cell_jitter``, one seeded rng per item) once as is (A) and once per wrong
glyph with slot k's glyph swapped (B_j; kept iff the pixel difference sits
inside slot k's cell). One batch of 1 + m latents under the one caption,
one ε; each sample's own gradient on row u comes from a zero tensor added
at u's position after the rows (`_SlotGrad`), checked against ∂raw_u. Pass 1
compares *different rows* (whose gradients are near-orthogonal whatever
the image: cos ≈ 0.15), pass 2 the *same row* under different glyphs;
`f_glyph` = 1 − cos between two wrong renders' gradients is the share of the
row's gradient energy that changes with the glyph drawn. `RENDER_TIERS` are
the grid / lone tiers.

Pass 2 on the bubble tiers (``--render_tiers bubble1_52 bubble1_32 bubbleN_34
bubbleN_18``, `SCENE_TIERS`): the record's scene, fill and orientation
re-drawn by `render_into_scene(ref_text=…)` — one fit, the sibling's glyphs in
place — with one font and one seeded rng for A and every B_j; a window is kept
iff each difference spans one glyph pitch at slot k. The loss box is the
union of the 1 + m ink boxes, the same for every sample. `bubble1_52` is not
in the grid_44 data: its records are `retrain_kana`'s (hiragana only). For a
window the same backward also reads **every other row of the window**
(`_SlotGrad` takes several rows): `f_cross` = the share of a neighbour row's
gradient that changes when slot k's glyph — not its own — is swapped, against
`f_glyph` on slot k's own row (`hypothesis.md` H2: the rows of an item share
one signal).

The magnitude estimator's median (``excess``) runs negative under
exchangeability: id2's three terms share one denominator ‖g_true‖, null's
three do not, so id2 is the more skewed (row gradient norms spread with
log-sd ≈ 0.6) — the mean excess is +0.02 where the median is −0.07. Read
``excess_dir`` / ``excess_norm``.

Data: ``OUT/run1002_grid_44/data_recap_b7593`` (plain captions — the arm of
record's) is the primary set; ``data`` (position-clause captions, same
images) the secondary; the band fields are ignored, σ is swept. Latents are
the recap dir's cache (the same images).

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="--label grad_identity --stall-timeout 900 \\
      project/cjk_anima_scale/experiments/grad_identity/run_exp.py --label r0 --queue"
    # CPU: the plan, the T5 checks, no model
    .venv/bin/python project/cjk_anima_scale/experiments/grad_identity/run_exp.py --label x --dry_run
    # pass 2 (renders under <results>/renders/)
    … --label render --render --queue
    # pass 2, the bubble tiers
    … --label scene --render --render_tiers bubble1_52 bubble1_32 bubbleN_34 bubbleN_18 --queue
    # CPU: re-read a finished results dir
    … --label x --analyze experiments/grad_identity/results/<stamp>-r0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale import paths  # noqa: E402
from cjk_scale.paths import OUT, SEED_ROWS_0921, bootstrap  # noqa: E402

bootstrap()
paths.pin_old_seed()  # the grid_44 train leg's context
from bench._common import make_run_dir, write_result  # noqa: E402

NAME = "grad_identity"
DATA_RUN = "run1002_grid_44"
DIRS = {
    "plain": OUT / DATA_RUN / "data_recap_b7593",  # the arm of record's captions
    "clause": OUT / DATA_RUN / "data",  # position-clause captions, per-px bands
}
SIGMAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
TIERS = (
    "grid_44",
    "grid_29",
    "grid_16",
    "lone_44",
    "lone_28",
    "lone_16",
    "bubbleN_34",
    "bubble1_32",
    "bubbleN_18",
)
# the bands grid_44 trained its tiers at (experiments/grid_44 SIZES + the bubble groups)
TRAINED = {
    "grid_44": (0.7, 0.9),
    "lone_44": (0.7, 0.9),
    "grid_29": (0.5, 0.7),
    "lone_28": (0.5, 0.7),
    "bubbleN_34": (0.5, 0.7),
    "bubble1_32": (0.5, 0.7),
    "grid_16": (0.3, 0.5),
    "lone_16": (0.3, 0.5),
    "bubbleN_18": (0.3, 0.5),
    "bubble1_52": (0.7, 0.9),  # builder.TABLE's band; not a grid_44 tier
}
# pass 2's bubble tiers; bubble1_52's records are retrain_kana's
SCENE_TIERS = ("bubble1_52", "bubble1_32", "bubbleN_34", "bubbleN_18")
SCENE_EXTRA = {"bubble1_52": OUT / "retrain_kana" / "data"}
EXTRA_I0 = 100_000  # item ids of the SCENE_EXTRA records
REPORT_TIERS = TIERS + ("bubble1_52",)
PX_BINS = (0, 18, 24, 32, 40, 56, 1e9)
_SMALL = dict(zip("ぁぃぅぇぉっゃゅょゎゕゖ", "あいうえおつやゆよわかけ"))


def family(c: str) -> str:
    """A glyph's base: dakuten / handakuten stripped, small kana → large."""
    b = unicodedata.normalize("NFD", c)[0]
    return _SMALL.get(b, b)


def form_of(tier: str) -> str:
    return tier.split("_")[0]


def px_bin(px: float) -> str:
    for lo, hi in zip(PX_BINS, PX_BINS[1:]):
        if lo <= px < hi:
            return f"{lo:g}–{hi:g}" if hi < 1e8 else f"≥{lo:g}"
    return "?"


# -- the plan (CPU) -------------------------------------------------------------


def load_recs() -> dict:
    out = {}
    for mode, d in DIRS.items():
        out[mode] = [
            json.loads(ln)
            for ln in (d / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if ln
        ]
    a, b = out["plain"], out["clause"]
    assert len(a) == len(b) and all(
        x["file"] == y["file"] and x["tier"] == y["tier"] for x, y in zip(a, b)
    ), "the two data dirs are not the same items in the same order"
    return out


def hiragana() -> list[str]:
    v = json.loads((DIRS["clause"] / "vocabs.json").read_text(encoding="utf-8"))
    hira = [c for c in v if len(c) == 1 and "ぁ" <= c <= "ゖ"]
    assert len(hira) == 81, len(hira)
    return hira


def slot_px(r: dict, u: str) -> float:
    """The slot's px: a grid cell's own box (one glyph per cell), else the
    item's ink px."""
    if r["src"] == "grid" and r.get("boxes") and len(r["boxes"]) == len(r["units"]):
        x0, y0, x1, y1 = r["boxes"][r["units"].index(u)]
        return round(math.sqrt(max(x1 - x0, 1) * max(y1 - y0, 1)), 1)
    return float(r["px"])


def make_plan(recs: dict, items: int, clause_items: int, wrong: int, seed: int):
    hira = hiragana()
    hset = set(hira)
    rng = random.Random(seed)
    by_tier: dict = defaultdict(list)
    for i, r in enumerate(recs["plain"]):
        by_tier[r["tier"]].append(i)
    plan, skipped = [], Counter()
    for tier in TIERS:
        pool = list(by_tier[tier])
        rng.shuffle(pool)
        got = 0
        for i in pool:
            if got >= items:
                break
            r = recs["plain"][i]
            caps = {m: recs[m][i]["caption"] for m in DIRS}
            cands = [
                c
                for c in sorted(set(r["text"]))
                if c in hset
                and r["text"].count(c) == 1
                and all(cap.count(c) == 1 for cap in caps.values())
            ]
            if not cands:
                skipped[tier] += 1
                continue
            u = rng.choice(cands)
            pool_v = [
                v
                for v in hira
                if v not in r["text"]
                and family(v) != family(u)
                and all(v not in cap for cap in caps.values())
            ]
            if len(pool_v) < wrong:
                skipped[tier] += 1
                continue
            vs = rng.sample(pool_v, wrong)
            plan.append(
                {
                    "i": i,
                    "tier": tier,
                    "form": form_of(tier),
                    "src": r["src"],
                    "px": float(r["px"]),
                    "slot_px": slot_px(r, u),
                    "glyphs": r.get("glyphs"),
                    "shape": r["shape"],
                    "u": u,
                    "wrong": vs,
                    "clause": got < clause_items,
                    "captions": {
                        m: [caps[m]] + [caps[m].replace(u, v) for v in vs] for m in DIRS
                    },
                }
            )
            got += 1
    return plan, skipped


def t5_check(plan: list, modes: list) -> dict:
    """Tokenize every caption (CPU, the pack-routed path) and check the
    batching claim: each wrong caption's T5 ids differ from the true one's at
    exactly one position, ``u``'s single row against ``v_j``'s. Returns
    {glyph: ext id}; asserts on any failure."""
    from common.models import checkpoints
    from data.inventory import pieces, qwen_pieces
    from library.anima.ext_vocab import T5_TABLE_SIZE
    from library.inference.text import ensure_text_strategies

    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    tq = qwen_pieces(char_rows=True)
    ext = {}

    def eid(c):
        if c not in ext:
            (p,) = pieces(*tq, c)
            assert p[1] is not None, c
            ext[c] = int(p[1])
        return ext[c]

    n = 0
    for it in plan:
        for c in it.get("cross", ()):  # a window's other rows (pass 2)
            eid(c)
        for m in modes:
            if m == "clause" and not it["clause"]:
                continue
            caps = it["captions"][m]
            t5 = tok.tokenize(caps)[2]
            ids = [[int(v) for v in row.tolist()] for row in t5]
            eu = eid(it["u"]) + T5_TABLE_SIZE
            assert ids[0].count(eu) == 1, (it["i"], m, "u not once in C")
            for v, row in zip(it["wrong"], ids[1:]):
                ev = eid(v) + T5_TABLE_SIZE
                diff = [k for k, (a, b) in enumerate(zip(ids[0], row)) if a != b]
                assert len(diff) == 1, (it["i"], m, v, "not one-position swap")
                k = diff[0]
                assert ids[0][k] == eu and row[k] == ev, (it["i"], m, v)
                assert row.count(ev) == 1 and eu not in row
                assert all(ev not in o for o in ids if o is not row)
            n += 1
    print(f"t5 check: {n} caption sets, every swap one position", flush=True)
    return ext


# -- the read (GPU) -------------------------------------------------------------


class Probe:
    def __init__(self, plan, modes, ext):
        import torch
        from common.models import checkpoints, encode_captions, ext_ids_of, gen_args
        from data.inventory import qwen_pieces
        from library.anima.ext_vocab import pack_digest
        from library.anima.vocab_pack import attached_pack_rows, strategy_pack
        from library.inference.generation import get_generation_settings
        from library.inference.models import load_dit_model
        from library.inference.text import ensure_text_strategies
        from train.stage import LatentStore

        from cjk_scale import train as T
        from cjk_scale.rows import Rows
        from cjk_scale.train import load_items, vocab_idx

        self.torch = torch
        self.T = T
        tmp = OUT / "experiments" / NAME
        tmp.mkdir(parents=True, exist_ok=True)
        args = gen_args(512, T.GEN_STEPS, T.GEN_CFG, tmp)
        self.device = device = get_generation_settings(args).device
        caps = sorted(
            {
                c
                for it in plan
                for m in modes
                if m != "clause" or it["clause"]
                for c in it["captions"][m]
            }
        )
        t0 = time.time()
        self.cache = encode_captions(caps, device)
        touched = ext_ids_of(self.cache)
        print(
            f"text: {len(caps)} captions in {time.time() - t0:.0f}s, {len(touched)} ext rows",
            flush=True,
        )
        recs, _ev, vocabs = load_items(DIRS["plain"])
        self.recs = recs
        idx = vocab_idx(vocabs, qwen_pieces(char_rows=True))
        assert len(idx) == 82, len(idx)
        assert {ext[c] for c in ext} <= idx
        from types import SimpleNamespace

        ns = SimpleNamespace(seed=0, batch=1, train_size=512)
        self.lat = LatentStore(ns, DIRS["plain"], recs, list(range(len(recs))), device)
        anima = load_dit_model(args, device, torch.bfloat16)
        anima.requires_grad_(False)
        assert attached_pack_rows(anima), "no vocab pack attached to the DiT"
        tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
        pack = strategy_pack(tok)
        raw_sha = pack_digest(
            pack.table, {k: v for k, v in pack.mapping.items() if k != "fold"}
        )
        assert raw_sha.startswith(paths.RAW_PACK_SHA), raw_sha
        print(
            f"pack {pack.name} raw sha {raw_sha[:12]}; ANIMA_VOCAB_GLYPH_ROUTE="
            f"{os.environ.get('ANIMA_VOCAB_GLYPH_ROUTE')}",
            flush=True,
        )
        # the grid_44 train leg's initial point: the 82 rows cold (raw 0 = the
        # pack rows), every other caption row frozen at the old seed
        self.rows = Rows(
            anima,
            device,
            idx,
            pack,
            warm=None,
            init_anchor=0.0,
            free_residual=0.0,
            lr=0.0,
            frozen=touched - idx,
            context=SEED_ROWS_0921,
        )
        self.raw = self.rows.delta.raw
        assert (
            float(
                self.raw.detach()[[self.rows.delta.index[e] for e in idx]].abs().max()
            )
            == 0
        )
        self.row = {c: self.rows.delta.index[e] for c, e in ext.items()}
        self.anima = anima
        anima.train()

    def compile(self):
        from library.runtime.harness import compile_blocks_for_training

        compile_blocks_for_training(
            self.anima, None, backend="inductor", n_token_families=self.lat.n_families
        )

    def noise(self, i: int, s: int, seed: int):
        torch = self.torch
        lat = self.lat[[i]]
        g = torch.Generator(device=self.device).manual_seed(
            seed * 1_000_003 + i * 101 + s
        )
        return torch.randn(lat.shape, generator=g, device=self.device, dtype=lat.dtype)

    def grad(self, it, mode, sigma, noise, which=None):
        """One forward + backward over the item's captions (``which``: a
        subset of their positions, default all) at σ; returns the full rows
        gradient and the per-sample losses."""
        from common.models import dit_forward
        from library.runtime.noise import fm_training_batch

        from cjk_scale.loss import box_share_fm_loss

        torch, T = self.torch, self.T
        caps = it["captions"][mode]
        which = list(range(len(caps))) if which is None else which
        B = len(which)
        r = self.recs[it["i"]]
        lat = self.lat[[it["i"]]].to(self.device)
        noisy, ts, target = fm_training_batch(
            lat.expand(B, *lat.shape[1:]).contiguous(),
            noise.expand(B, *noise.shape[1:]).contiguous(),
            dtype=torch.bfloat16,
            device=self.device,
            t_min=sigma,
            t_max=sigma,
        )
        # σ is exact up to the bf16 cast the trainer's call makes too
        assert float((ts.float() - sigma).abs().max()) < 4e-3, ts
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = dit_forward(
                self.anima, noisy, ts, self.cache, [caps[k] for k in which], self.device
            )
        bs = (
            T.BOX_SHARE
            if r["src"] == "scene" or (T.GRID_BOX and r["src"] == "grid")
            else 0.0
        )
        losses = [
            box_share_fm_loss(
                pred[b : b + 1],
                target[b : b + 1],
                [r],
                bs,
                T.BOX_SHARE_CAP,
                float(T.BOX_SHARE_GLYPHS),
                bool(T.GRID_BOX),
            )
            for b in range(B)
        ]
        (g,) = torch.autograd.grad(sum(losses), self.raw)
        return g, [float(x) for x in losses]

    def read(self, it, mode, sigma, noise):
        """(1 + m, D) gradient rows — u's under C, then each v_j's under
        C^{v_j} — and the 1 + m losses."""
        g, losses = self.grad(it, mode, sigma, noise)
        rows = [self.row[it["u"]]] + [self.row[v] for v in it["wrong"]]
        return g[rows].float().cpu(), losses

    def verify(self, items, mode, sigmas, seed) -> list:
        """The batching claim against one caption per backward, beside the
        numerical floor it has to be read against: the same batch run twice,
        the batch in reverse order, and each single run twice (bf16 autocast;
        flash-attention backward accumulates with atomics)."""
        torch = self.torch
        F = torch.nn.functional

        def cmp(a, b):
            return (
                float(F.cosine_similarity(a, b, dim=0)),
                float((a - b).norm() / b.norm().clamp(min=1e-30)),
            )

        out = []
        for it in items:
            for s_i, sigma in sigmas:
                nz = self.noise(it["i"], s_i, seed)
                m1 = 1 + len(it["wrong"])
                rows = [self.row[it["u"]]] + [self.row[v] for v in it["wrong"]]
                gb, lb = self.read(it, mode, sigma, nz)
                gb2, _ = self.read(it, mode, sigma, nz)
                rev = list(range(m1))[::-1]
                gr_full, lr = self.grad(it, mode, sigma, nz, which=rev)
                gr = gr_full[rows].float().cpu()
                # composition: the true caption four times — row u's per-sample
                # gradient at the same batch shape, none of the wrong captions
                gc_full, _ = self.grad(it, mode, sigma, nz, which=[0] * m1)
                gc = gc_full[rows[0]].float().cpu() / m1
                g1s, g1bs = [], []
                for k in range(m1):
                    g1, l1 = self.grad(it, mode, sigma, nz, which=[k])
                    g1b, _ = self.grad(it, mode, sigma, nz, which=[k])
                    a, b = gb[k], g1[rows[k]].float().cpu()
                    g1s.append(b)
                    g1bs.append(g1b[rows[k]].float().cpu())
                    # the rows the single-caption run must leave at zero
                    others = [rows[j] for j in range(m1) if j != k]
                    out.append(
                        {
                            "i": it["i"],
                            "tier": it["tier"],
                            "sigma": sigma,
                            "k": k,
                            "norm_batched": float(a.norm()),
                            "norm_single": float(b.norm()),
                            "single": cmp(a, b),
                            "repeat": cmp(gb2[k], a),
                            "reverse": cmp(gr[k], a),
                            "single_repeat": cmp(g1bs[-1], b),
                            "composition": cmp(gc, a) if k == 0 else None,
                            "loss_batched": lb[k],
                            "loss_single": l1[0],
                            "loss_reverse": lr[m1 - 1 - k],
                            "others_zero": float(g1[others].abs().max()) == 0.0,
                        }
                    )
                mets = {
                    "batched": metrics_of(gb, lb),
                    "repeat": metrics_of(gb2, lb),
                    "reverse": metrics_of(gr, lb),
                    "single": metrics_of(torch.stack(g1s), lb),
                    "single_repeat": metrics_of(torch.stack(g1bs), lb),
                }
                out[-1]["metrics"] = mets
                print(
                    f"verify {it['tier']} i={it['i']} σ {sigma}: cos/rel single "
                    + ", ".join("{:.3f}/{:.2f}".format(*x["single"]) for x in out[-m1:])
                    + " · repeat "
                    + ", ".join("{:.3f}/{:.2f}".format(*x["repeat"]) for x in out[-m1:])
                    + " · reverse "
                    + ", ".join(
                        "{:.3f}/{:.2f}".format(*x["reverse"]) for x in out[-m1:]
                    )
                    + " · single×2 "
                    + ", ".join(
                        "{:.3f}/{:.2f}".format(*x["single_repeat"]) for x in out[-m1:]
                    )
                    + " · excess "
                    + ", ".join(f"{k} {v['excess']:+.3f}" for k, v in mets.items()),
                    flush=True,
                )
        return out


def run_gpu(args, plan, modes, run_dir):
    import torch

    ext = t5_check(plan, modes)
    sigmas = list(args.sigmas)
    P = Probe(plan, modes, ext)
    P.compile()
    # the B = 1 verify graphs stay their own static graphs: no automatic
    # dynamic batch dim leaking into the B = 1 + m graphs of the loop
    torch._dynamo.config.automatic_dynamic_shapes = False
    # verify on 512×512 items of distinct tiers (one token family for the B = 1 graphs)
    ver_items, seen = [], set()
    for it in plan:
        if list(it["shape"]) == [512, 512] and it["tier"] not in seen:
            ver_items.append(it)
            seen.add(it["tier"])
        if len(ver_items) >= args.verify:
            break
    ver = P.verify(
        ver_items,
        "plain",
        [(sigmas.index(s), s) for s in (0.4, 0.8) if s in sigmas],
        args.seed,
    )
    (run_dir / "verify.json").write_text(json.dumps(ver, indent=1))

    def med(key, j):
        return st.median(v[key][j] for v in ver)

    floor = max(med("repeat", 1), med("reverse", 1), med("single_repeat", 1))
    worst = med("single", 1)
    comp = st.median(v["composition"][1] for v in ver if v["composition"])
    print(
        f"verify: {len(ver)} rows; median rel batched vs single {worst:.3f} (batch "
        f"shape numerics), vs the true caption ×{1 + args.wrong} {comp:.3f} "
        f"(composition); floor: repeat {med('repeat', 1):.3f}, reverse "
        f"{med('reverse', 1):.3f}, single×2 {med('single_repeat', 1):.3f}; others "
        f"zero {all(v['others_zero'] for v in ver)}",
        flush=True,
    )
    # the claim is that the other captions in the batch do not reach row u:
    # at one batch shape, swapping them out must stay inside the run-to-run
    # floor. B = 1 vs B = 4 is a different kernel shape (bf16), recorded only.
    assert all(v["others_zero"] for v in ver), "a single caption moved another row"
    assert comp <= max(2 * floor, 0.03), (
        "the batch's other captions reach row u — see verify.json"
    )
    if args.diag:
        return {"verify_rel_single": worst, "composition_rel": comp, "floor_rel": floor}
    D = P.raw.shape[1]
    m1 = 1 + args.wrong
    t0 = time.time()
    n_done, n_total = (
        0,
        sum(len(sigmas) for it in plan for m in modes if m != "clause" or it["clause"]),
    )
    for mode in modes:
        its = [it for it in plan if mode != "clause" or it["clause"]]
        G = torch.zeros(len(its), len(sigmas), m1, D)
        L = torch.zeros(len(its), len(sigmas), m1)
        for n, it in enumerate(its):
            for s_i, sigma in enumerate(sigmas):
                nz = P.noise(it["i"], s_i, args.seed)
                g, losses = P.read(it, mode, sigma, nz)
                G[n, s_i] = g
                L[n, s_i] = torch.tensor(losses)
                n_done += 1
            if (n + 1) % 10 == 0 or n + 1 == len(its):
                el = time.time() - t0
                print(
                    f"  {mode}: {n + 1}/{len(its)} items, {el / 60:.1f} min, "
                    f"eta {el / n_done * (n_total - n_done) / 60:.1f} min",
                    flush=True,
                )
        torch.save(
            {"G": G, "L": L, "items": [it["i"] for it in its], "sigmas": sigmas},
            run_dir / f"grads_{mode}.pt",
        )
    print(f"reads: {n_done} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return {
        "verify_rel_single": worst,
        "composition_rel": comp,
        "floor_rel": floor,
        "verify_n": len(ver),
        "reads": n_done,
    }


# -- pass 2: the render-side contrast (one row, one caption, the image swapped) --

RENDER_TIERS = ("grid_16", "grid_29", "grid_44", "lone_16", "lone_28", "lone_44")


def render_plan(plan: list, recs: dict, tiers, seed: int, out: Path) -> tuple:
    """Re-draw each grid / lone item of ``tiers`` as a fresh twin set: the
    record's units, grid, canvas, bubble and fill with `grid_small`'s
    ``bubble_fit`` / ``cell_jitter`` (the grid_44 build's), one seeded rng
    per item, drawn once as is (A) and once per wrong glyph with slot k's
    glyph swapped (B_j). The font list is cut to the fonts that cover every
    glyph involved, so ``pick_font`` draws the same font from the same list
    in every render. Kept iff each B_j differs from A only inside slot k's
    cell (``eval.cf_sense._diff_boxes``) and every other cell's ink box is
    A's. CPU; PNGs under ``out``."""
    from common.render.flat import find_fonts, font_covers
    from data.grid import render_grid
    from eval.cf_sense import _diff_boxes

    from cjk_scale.paths import load_experiment
    from cjk_scale.recipes import GRIDS

    GS = load_experiment("grid_small")
    fonts_all = find_fonts()
    out.mkdir(parents=True, exist_ok=True)
    kept, dropped = [], Counter()
    for it in plan:
        if it["tier"] not in tiers:
            continue
        r = recs["plain"][it["i"]]
        cols, rows, _ = GRIDS[r["grid"]]
        W, H = r["shape"]
        k = r["units"].index(it["u"])
        glyphs = set(r["units"]) | set(it["wrong"])
        fonts = [f for f in fonts_all if all(font_covers(f, c) for c in glyphs)]
        ims, boxes = [], []
        for v in [None, *it["wrong"]]:
            units = list(r["units"])
            if v is not None:
                units[k] = v
            rng = random.Random(seed * 7919 + it["i"])
            im, bx = render_grid(
                units,
                cols,
                rows,
                (W, H),
                fonts,
                rng,
                bool(r["bubble"]),
                (float(r["fill"]), float(r["fill"])),
                lines=[],
                horizontal_frac=0.3,
                bubble_fit=tuple(GS.BUBBLE_FIT),
                cell_jitter=GS.CELL_JITTER,
            )
            ims.append(im)
            boxes.append(bx)
        cw, ch = W / cols, H / rows
        rr, cc = divmod(k, cols)
        cell = (cc * cw, rr * ch, (cc + 1) * cw, (rr + 1) * ch)
        ok = True
        for j in range(1, len(ims)):
            (x0, y0, x1, y1), _ = _diff_boxes(ims[0], ims[j])
            inside = x0 >= cell[0] and y0 >= cell[1] and x1 <= cell[2] and y1 <= cell[3]
            same = all(
                boxes[0][q] == boxes[j][q] for q in range(len(boxes[0])) if q != k
            )
            if not (inside and same and (x1 > x0)):
                ok = False
        if not ok:
            dropped[it["tier"]] += 1
            continue
        files = []
        for j, im in enumerate(ims):
            f = out / f"{it['i']:05d}_{j}.png"
            im.save(f)
            files.append(str(f))
        x0, y0, x1, y1 = boxes[0][k]
        kept.append(
            {
                **it,
                "render": {
                    "files": files,
                    "boxes": boxes[0],
                    "slot_box": boxes[0][k],
                    "slot_px": round(math.sqrt(max(x1 - x0, 1) * max(y1 - y0, 1)), 1),
                    "n_fonts": len(fonts),
                },
            }
        )
    return kept, dropped


def scene_render_plan(
    recs: dict, tiers, items: int, window_items: int, wrong: int, seed: int, out: Path
) -> tuple:
    """Pass 2's items for the bubble tiers, drawn until each tier holds its
    count (``items`` for a ``bubble1`` tier, ``window_items`` for a
    ``bubbleN``): the record's scene, fill and orientation re-drawn with one
    font (of those covering every glyph involved) and one seeded rng, once
    per wrong glyph as `render_into_scene(ref_text=…)` — A is the text, B_j
    its sibling with slot k's glyph swapped, pixel-identical outside the two
    text boxes (the renderer's assert). A window is kept iff every A / B_j
    difference spans at most one glyph pitch along the text axis and sits at
    slot k. The loss box is the union of the 1 + m boxes. A window's other
    glyphs (each once in the caption) are its ``cross`` rows. CPU."""
    import numpy as np
    from common.render.flat import find_fonts, font_covers
    from common.render.scene import render_into_scene
    from data.synth import load_scenes
    from eval.cf_sense import _diff_boxes

    from cjk_scale.builder import tier_of, tiers as table_tiers
    from cjk_scale.config import DATA

    assert not DATA["stroke"], "the records were drawn without a stroke"
    scenes = {
        (s["pool"], s["i"]): s
        for s in load_scenes(DATA["scenes"], 0.0, 0, "", DATA["scene_one_bubble"])
    }
    hira = hiragana()
    hset = set(hira)
    fonts_all = find_fonts()
    out.mkdir(parents=True, exist_ok=True)
    pools: dict = defaultdict(list)
    for i, r in enumerate(recs["plain"]):
        if r["tier"] in tiers and r["tier"] not in SCENE_EXTRA:
            pools[r["tier"]].append((i, r))
    for tier, d in SCENE_EXTRA.items():
        if tier not in tiers:
            continue
        for j, ln in enumerate((d / "train.jsonl").read_text("utf-8").splitlines()):
            r = json.loads(ln) if ln else None
            if r and r["src"] == "scene" and tier_of(r) == tier:
                pools[tier].append((EXTRA_I0 + j, r))
    kept, dropped = [], Counter()
    for tier in tiers:
        (spec,) = table_tiers(tier)
        min_glyph = int(spec.params["min_glyph"])
        want = window_items if form_of(tier) == "bubbleN" else items
        rng = random.Random(seed * 104_729 + SCENE_TIERS.index(tier))
        pool = list(pools[tier])
        rng.shuffle(pool)
        got = 0
        for i, r in pool:
            if got >= want:
                break
            text, cap = r["text"], r["caption"]
            n = len(text)
            once = [c for c in text if text.count(c) == 1 and cap.count(c) == 1]
            cands = [c for c in once if c in hset]
            sc = scenes.get((r["scene_pool"], r["scene"]))
            if not cands or sc is None or cap.count(f'"{text}"') != 1:
                dropped[f"{tier}/plan"] += 1
                continue
            u = rng.choice(cands)
            k = text.index(u)
            pool_v = [
                v
                for v in hira
                if v not in text and v not in cap and family(v) != family(u)
            ]
            vs = rng.sample(pool_v, wrong)
            fonts = [
                f
                for f in fonts_all
                if all(font_covers(f, c) for c in set(text) | set(vs))
            ]
            font = rng.choice(fonts)
            horiz = bool(r["horizontal"])
            ims, boxes, ok = [], [], True
            for v in vs:
                drawn = render_into_scene(
                    scene=sc,
                    text=text,
                    font_path=font,
                    rng=random.Random(seed * 7919 + i),
                    min_glyph=min_glyph,
                    stroke=False,
                    fill_frac=float(r["fill"]),
                    max_lines=1,
                    vertical_only=not horiz,
                    horizontal=horiz,
                    ref_text=text[:k] + v + text[k + 1 :],
                )
                if drawn is None:
                    ok = False
                    break
                im_a, box_a, im_b, box_b = drawn
                if ims and (np.array(ims[0]) != np.array(im_a)).any():
                    ok = False  # A must be one image whatever the sibling
                    break
                if not ims:
                    ims.append(im_a)
                    boxes.append(box_a)
                ims.append(im_b)
                boxes.append(box_b)
            if not ok:
                dropped[f"{tier}/draw"] += 1
                continue
            x0, y0, x1, y1 = boxes[0]
            ax = (
                0 if horiz or n == 1 else 1
            )  # the text axis: x of a line, y of a column
            lo, ext_ = (x0, x1 - x0) if ax == 0 else (y0, y1 - y0)
            pitch = ext_ / n
            for j in range(1, len(ims)):
                d = _diff_boxes(ims[0], ims[j])[0]
                if d[2] <= d[0]:
                    ok = False
                    break
                if n > 1:
                    a, b = d[ax], d[ax + 2]
                    mid = ((a + b) / 2 - lo) / pitch
                    if b - a > 1.3 * pitch or not (k - 0.25 <= mid <= k + 1.25):
                        ok = False
                        break
            if not ok:
                dropped[f"{tier}/slot"] += 1
                continue
            files = []
            for j, im in enumerate(ims):
                f = out / f"{i:06d}_{j}.png"
                im.save(f)
                files.append(str(f))
            union = [
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            ]
            px = round(math.sqrt(max(x1 - x0, 1) * max(y1 - y0, 1) / n), 1)
            kept.append(
                {
                    "i": i,
                    "tier": tier,
                    "form": form_of(tier),
                    "src": "scene",
                    "px": float(r["px"]),
                    "slot_px": px,
                    "glyphs": n,
                    "shape": r["shape"],
                    "u": u,
                    "wrong": vs,
                    "clause": False,
                    "captions": {"plain": [cap] + [cap.replace(u, v) for v in vs]},
                    "cross": [c for c in once if c != u],
                    "slot": k,
                    "horizontal": horiz,
                    "render": {
                        "files": files,
                        "slot_px": px,
                        "font": Path(font).name,
                        "n_fonts": len(fonts),
                        "boxes": boxes,
                        "rec": {
                            "src": "scene",
                            "layout": "scene",
                            "text": text,
                            "box": union,
                        },
                    },
                }
            )
            got += 1
        assert got == want, f"{tier}: {got} of {want} items ({dict(dropped)})"
    return kept, dropped


class _SlotGrad:
    """A zero ``(R, B, D)`` tensor added to ``llm_adapter.embed``'s output at
    the one position per sample that holds each of the ``R`` rows ``exts``
    (after the pack and the rows' ``ExtDelta``): its gradient is each
    sample's own gradient on each row, from one backward over the batch (row
    units: × ``row_scale``)."""

    def __init__(self, anima, device, dim):
        import torch

        self.torch = torch
        self.exts = None
        self.z = None
        self.ids = None
        embed = anima.llm_adapter.embed

        def pre(module, args):
            self.ids = args[0] if args and torch.is_tensor(args[0]) else None

        def post(module, args, output):
            if self.z is None or self.ids is None:
                return None
            out = output.clone()
            for r, ext in enumerate(self.exts):
                mask = self.ids == ext
                assert bool((mask.sum(1) == 1).all()), "a row is not once per caption"
                out[mask] = out[mask] + self.z[r].to(out.dtype)
            return out

        self.handles = [
            embed.register_forward_pre_hook(pre, prepend=True),
            embed.register_forward_hook(post),
        ]


def run_render(args, plan_r: list, run_dir: Path) -> dict:
    """Pass 2: per item × σ, row u's gradient under the true caption on
    render A and on each B_j — one batch of 1 + m latents, one ε. A window's
    ``cross`` rows are read from the same backward (``X``)."""
    import torch

    from common.models import dit_forward, encode_images, load_vae
    from library.anima.ext_vocab import T5_TABLE_SIZE
    from library.inference.generation import get_generation_settings
    from library.runtime.noise import fm_training_batch

    from cjk_scale.loss import box_share_fm_loss

    ext = t5_check(plan_r, ["plain"])
    sigmas = list(args.sigmas)
    from common.models import gen_args

    device = get_generation_settings(gen_args(512, 28, 4.0, run_dir)).device
    t0 = time.time()
    vae = load_vae(device)
    lats = {}
    for it in plan_r:
        lats[it["i"]] = encode_images(vae, it["render"]["files"], device)
    del vae
    torch.cuda.empty_cache()
    print(f"renders: {len(lats)} items encoded in {time.time() - t0:.0f}s", flush=True)
    P = Probe(plan_r, ["plain"], ext)
    P.compile()
    T = P.T
    D = P.raw.shape[1]
    hook = _SlotGrad(P.anima, P.device, D)
    m1 = 1 + args.wrong
    G = torch.zeros(len(plan_r), len(sigmas), m1, D)
    L = torch.zeros(len(plan_r), len(sigmas), m1)
    x_items = [it["i"] for it in plan_r if it.get("cross")]
    x_at = {i: n for n, i in enumerate(x_items)}
    n_cross = max((len(it.get("cross", ())) for it in plan_r), default=0)
    X = torch.zeros(len(x_items), len(sigmas), n_cross, m1, D)
    checks = []
    t0 = time.time()
    for n, it in enumerate(plan_r):
        if it["src"] == "scene":
            r = it["render"]["rec"]  # the union of the 1 + m ink boxes
        else:
            r = dict(P.recs[it["i"]])
            r["boxes"] = it["render"]["boxes"]  # A's cells, the one loss box for all
        cap = it["captions"]["plain"][0]
        glyphs = [it["u"], *it.get("cross", ())]
        hook.exts = [T5_TABLE_SIZE + ext[c] for c in glyphs]
        lat = lats[it["i"]].to(P.device)
        # train.train's rule: a scene item takes the box share, a grid under GRID_BOX
        bs = (
            T.BOX_SHARE
            if r["src"] == "scene" or (T.GRID_BOX and r["src"] == "grid")
            else 0.0
        )
        for s_i, sigma in enumerate(sigmas):
            g = torch.Generator(device=P.device).manual_seed(
                args.seed * 1_000_003 + it["i"] * 101 + s_i
            )
            nz = torch.randn(
                (1, *lat.shape[1:]), generator=g, device=P.device, dtype=lat.dtype
            )
            noisy, ts, target = fm_training_batch(
                lat,
                nz.expand(m1, *nz.shape[1:]).contiguous(),
                dtype=torch.bfloat16,
                device=P.device,
                t_min=sigma,
                t_max=sigma,
            )
            hook.z = torch.zeros(
                len(glyphs), m1, D, device=P.device, requires_grad=True
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = dit_forward(P.anima, noisy, ts, P.cache, [cap] * m1, P.device)
            losses = [
                box_share_fm_loss(
                    pred[b : b + 1],
                    target[b : b + 1],
                    [r],
                    bs,
                    T.BOX_SHARE_CAP,
                    float(T.BOX_SHARE_GLYPHS),
                    bool(T.GRID_BOX),
                )
                for b in range(m1)
            ]
            gz, graw = torch.autograd.grad(sum(losses), [hook.z, P.raw])
            gz = gz.float() * P.rows.row_scale
            if len(checks) < 20 or (len(glyphs) > 1 and len(checks) < 40):
                for q, c in enumerate(glyphs):
                    a, b = gz[q].sum(0), graw[P.row[c]].float()
                    checks.append(float((a - b).norm() / b.norm().clamp(min=1e-30)))
            G[n, s_i] = gz[0].cpu()
            if len(glyphs) > 1:
                X[x_at[it["i"]], s_i, : len(glyphs) - 1] = gz[1:].cpu()
            L[n, s_i] = torch.tensor([float(x) for x in losses])
            hook.z = None
        if (n + 1) % 10 == 0 or n + 1 == len(plan_r):
            el = time.time() - t0
            print(
                f"  render: {n + 1}/{len(plan_r)} items, {el / 60:.1f} min, eta "
                f"{el / (n + 1) * (len(plan_r) - n - 1) / 60:.1f} min",
                flush=True,
            )
    print(
        f"slot check: Σ_b ∂z_b vs ∂raw_u rel max {max(checks):.2e} over {len(checks)}",
        flush=True,
    )
    assert max(checks) < 1e-2, checks
    save = {"G": G, "L": L, "items": [it["i"] for it in plan_r], "sigmas": sigmas}
    if x_items:
        save.update(X=X, x_items=x_items)  # (item, σ, cross row, 1 + m, D)
    torch.save(save, run_dir / "grads_render.pt")
    return {"render_items": len(plan_r), "slot_check_rel_max": max(checks)}


# -- analysis (CPU) -------------------------------------------------------------


def metrics_of(g, losses) -> dict:
    """``g``: (1 + m, D) — u's row under C, then each v_j's under C^{v_j}."""
    import torch
    import torch.nn.functional as F

    gt, gw = g[0].double(), g[1:].double()
    m = gw.shape[0]
    lay = gw.mean(0)
    nt = float(gt.norm())
    eps = 1e-30
    id_share = float((gt - lay).norm()) / max(nt, eps)
    id2, null = [], []
    for j in range(m):
        rest = torch.cat([gw[:j], gw[j + 1 :]]).mean(0)
        id2.append(float((gt - rest).norm()) / max(nt, eps))
        null.append(float((gw[j] - rest).norm()) / max(float(gw[j].norm()), eps))
    # the same on unit vectors (direction only: a row's own gradient scale —
    # its pack row, the caption — drops out) and the norm alone (log ratio)
    ut = gt / max(nt, eps)
    uw = gw / gw.norm(dim=1, keepdim=True).clamp(min=eps)
    nw = gw.norm(dim=1)
    id2d, nulld, lr_t, lr_w = [], [], [], []
    for j in range(m):
        rest = torch.cat([uw[:j], uw[j + 1 :]]).mean(0)
        id2d.append(float((ut - rest).norm()))
        nulld.append(float((uw[j] - rest).norm()))
        nrest = float(torch.cat([nw[:j], nw[j + 1 :]]).mean())
        lr_t.append(math.log(max(nt, eps) / max(nrest, eps)))
        lr_w.append(math.log(max(float(nw[j]), eps) / max(nrest, eps)))
    cos_tw = [float(F.cosine_similarity(gt, gw[j], dim=0)) for j in range(m)]
    cos_ww = [
        float(F.cosine_similarity(gw[a], gw[b], dim=0))
        for a in range(m)
        for b in range(a + 1, m)
    ]
    return {
        "id_share": id_share,
        "id2": st.mean(id2),
        "null": st.mean(null),
        "excess": st.mean(id2) - st.mean(null),
        "id2_dir": st.mean(id2d),
        "null_dir": st.mean(nulld),
        "excess_dir": st.mean(id2d) - st.mean(nulld),
        "lnorm": st.mean(lr_t),
        "lnorm_null": st.mean(lr_w),
        "excess_norm": st.mean(lr_t) - st.mean(lr_w),
        "cos_layout": float(F.cosine_similarity(gt, lay, dim=0)),
        "cos_tw": st.mean(cos_tw),
        "cos_ww": st.mean(cos_ww),
        # g = L + I(glyph): 1 − cos between two wrong rows' / renders' gradients
        # is I's share of the energy; ‖I‖ its size (pass 2 reads these)
        "f_glyph": 1.0 - st.mean(cos_ww),
        "g_ident": st.mean(float(x.norm()) for x in gw)
        * math.sqrt(max(0.0, 1.0 - st.mean(cos_ww))),
        "g_true": nt,
        "g_wrong": st.mean(float(x.norm()) for x in gw),
        "g_layout": float(lay.norm()),
        "dloss": st.mean(losses[1:]) - losses[0],
    }


def cross_of(it: dict, x, f_own: float) -> dict:
    """A window's other rows under the same swap of slot k: ``x`` is
    (cross row, 1 + m, D). Per row, f = 1 − cos between two wrong renders'
    gradients (``metrics_of``'s ``f_glyph``, on a row whose own glyph did not
    change) and ‖I‖ = ‖g‖·√f; means over the rows, and over the rows next to
    slot k / further off."""
    import torch.nn.functional as F

    text, k = it["render"]["rec"]["text"], it["slot"]
    fs, norms, dist = [], [], []
    for q, c in enumerate(it["cross"]):
        gw = x[q, 1:].double()
        m = gw.shape[0]
        cos = [
            float(F.cosine_similarity(gw[a], gw[b], dim=0))
            for a in range(m)
            for b in range(a + 1, m)
        ]
        fs.append(1.0 - st.mean(cos))
        norms.append(st.mean(float(v.norm()) for v in gw))
        dist.append(abs(text.index(c) - k))
    adj = [f for f, dd in zip(fs, dist) if dd == 1]
    far = [f for f, dd in zip(fs, dist) if dd > 1]
    return {
        "f_cross": st.mean(fs),
        "f_cross_adj": st.mean(adj) if adj else None,
        "f_cross_far": st.mean(far) if far else None,
        "g_cross": st.mean(norms),
        "i_cross": st.mean(g * math.sqrt(max(0.0, f)) for g, f in zip(norms, fs)),
        "own_minus_cross": f_own - st.mean(fs),
    }


CROSS = (
    "f_cross",
    "f_cross_adj",
    "f_cross_far",
    "g_cross",
    "i_cross",
    "own_minus_cross",
)


def _q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def sign_p(k: int, n: int) -> float:
    """Two-sided sign test: k of n positive."""
    if n == 0:
        return 1.0
    t = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(t + 1)) / 2**n
    return min(1.0, 2 * p)


def analyze(run_dir: Path) -> dict:
    import torch

    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    by_i = {it["i"]: it for it in plan["items"]}
    per = []
    for mode in ("plain", "clause", "render"):
        f = run_dir / f"grads_{mode}.pt"
        if not f.exists():
            continue
        d = torch.load(f)
        x_at = {i: n for n, i in enumerate(d.get("x_items", ()))}
        for n, i in enumerate(d["items"]):
            it = by_i[i]
            spx = it["render"]["slot_px"] if mode == "render" else it["slot_px"]
            for s_i, sigma in enumerate(d["sigmas"]):
                rec = metrics_of(d["G"][n, s_i], d["L"][n, s_i].tolist())
                if i in x_at:
                    rec.update(cross_of(it, d["X"][x_at[i], s_i], rec["f_glyph"]))
                    rec["splits"] = [
                        f"{it['tier']} · {'2–3' if it['glyphs'] <= 3 else '4–6'} glyphs",
                        f"{it['tier']} · {'line' if it['horizontal'] else 'column'}",
                    ]
                rec.update(
                    mode=mode,
                    i=i,
                    tier=it["tier"],
                    form=it["form"],
                    px=it["px"],
                    slot_px=spx,
                    pxbin=px_bin(spx),
                    sigma=sigma,
                    u=it["u"],
                )
                per.append(rec)
    with open(run_dir / "per_item.jsonl", "w", encoding="utf-8") as fh:
        for rec in per:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summ = summarize(per)
    (run_dir / "summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    report(run_dir, plan, per, summ)
    return summ


EXCESS = ("excess", "excess_dir", "excess_norm")


def cell(rs) -> dict:
    out = {"n": len(rs)}
    for key in (
        "id_share",
        "id2",
        "null",
        "excess",
        "id2_dir",
        "null_dir",
        "excess_dir",
        "lnorm",
        "lnorm_null",
        "excess_norm",
        "cos_layout",
        "cos_tw",
        "cos_ww",
        "f_glyph",
        "g_ident",
        "g_true",
        "g_layout",
        "dloss",
    ):
        xs = [r[key] for r in rs]
        out[key] = [_q(xs, 0.25), _q(xs, 0.5), _q(xs, 0.75)]
    for key in (*EXCESS, "dloss"):
        k = sum(r[key] > 0 for r in rs)
        out[f"{key}_pos"] = k
        out[f"{key}_p"] = sign_p(k, len(rs))
    for key in CROSS:  # a window's other rows (pass 2, bubbleN)
        xs = [r[key] for r in rs if r.get(key) is not None]
        if xs:
            out[key] = [_q(xs, 0.25), _q(xs, 0.5), _q(xs, 0.75)]
            out[f"{key}_n"] = len(xs)
    xs = [r["own_minus_cross"] for r in rs if r.get("own_minus_cross") is not None]
    if xs:
        k = sum(x > 0 for x in xs)
        out["own_minus_cross_pos"] = k
        out["own_minus_cross_p"] = sign_p(k, len(xs))
    return out


def summarize(per) -> dict:
    groups: dict = defaultdict(list)
    for r in per:
        groups[(r["mode"], "tier", r["tier"], r["sigma"])].append(r)
        groups[(r["mode"], "bin", f"{r['form']} {r['pxbin']}", r["sigma"])].append(r)
        for label in r.get("splits", ()):
            groups[(r["mode"], "split", label, r["sigma"])].append(r)
    out: dict = defaultdict(lambda: defaultdict(dict))
    for (mode, kind, key, sigma), rs in groups.items():
        out[f"{mode}/{kind}"].setdefault(key, {})[f"{sigma:g}"] = cell(rs)
    return {k: dict(v) for k, v in out.items()}


def separation(cells: dict, sigmas, key: str = "excess") -> list:
    """σ where the true row's ``key`` is positive by the sign test (p < 0.01,
    median > 0)."""
    return [
        s
        for s in sigmas
        if f"{s:g}" in cells
        and cells[f"{s:g}"][f"{key}_p"] < 0.01
        and cells[f"{s:g}"][key][1] > 0
    ]


def _mark(x: dict, key: str) -> str:
    if x[f"{key}_p"] >= 0.01:
        return ""
    return "*" if x[key][1] > 0 else "−"


def _tier_table(summ, per, mode, sigmas, fmt, key) -> list:
    lines = [
        "| tier | slot px | trained | "
        + " | ".join(f"σ {s:g}" for s in sigmas)
        + " | separates at |",
        "|---|---|---|" + "---|" * len(sigmas) + "---|",
    ]
    tab = summ[f"{mode}/tier"]
    for tier in REPORT_TIERS:
        c = tab.get(tier)
        if not c:
            continue
        pxs = [r["slot_px"] for r in per if r["mode"] == mode and r["tier"] == tier]
        cells = [fmt(c[f"{s:g}"]) if f"{s:g}" in c else "–" for s in sigmas]
        lo, hi = TRAINED[tier]
        sep = separation(c, sigmas, key)
        lines.append(
            f"| {tier} | {st.median(pxs):.0f} | {lo:g}–{hi:g} | "
            + " | ".join(cells)
            + f" | {', '.join(f'{s:g}' for s in sep) or 'none'} |"
        )
    return lines


def _window_tables(summ, mode, sigmas) -> list:
    """Pass 2 on windows: slot k's own row against the window's other rows,
    and the own row by window length / orientation."""
    tab = summ[f"{mode}/tier"]
    head = ["| " + " | ".join(f"σ {s:g}" for s in sigmas) + " |"]
    rule = "---|" * len(sigmas)
    win = [
        t for t in REPORT_TIERS if any("f_cross" in x for x in tab.get(t, {}).values())
    ]
    if not win:
        return []

    def opt(x, key, scale=1.0, fmt=".2f"):
        return format(x[key][1] * scale, fmt) if key in x else "–"

    def f_cross(x):
        if "f_cross" not in x:
            return "–"
        mark = "" if x["own_minus_cross_p"] >= 0.01 else "*"
        return (
            f"{x['f_glyph'][1]:.2f} · {x['f_cross'][1]:.2f} "
            f"({opt(x, 'f_cross_adj')} / {opt(x, 'f_cross_far')}) · "
            f"{x['own_minus_cross_pos']}/{x['f_cross_n']}{mark}"
        )

    def i_cross(x):
        if "f_cross" not in x:
            return "–"
        return (
            f"{x['g_ident'][1] * 1e3:.1f} · {x['i_cross'][1] * 1e3:.1f} · "
            f"{x['g_true'][1] * 1e3:.0f} · {x['g_cross'][1] * 1e3:.0f}"
        )

    lines = [
        f"## {mode} — a window's other rows under the same swap",
        "",
        "Slot k's glyph is swapped in the image; `own` is slot k's row, `cross` "
        "every other row of the window (its own glyph unchanged), read from "
        "the same backward. Cell: f_own · f_cross (rows next to slot k / "
        "further off) · items with f_own > f_cross (`*` = sign test p < 0.01).",
        "",
        "| tier " + head[0],
        "|---|" + rule,
    ]
    for t in win:
        c = tab[t]
        lines.append(
            f"| {t} | "
            + " | ".join(f_cross(c.get(f"{s:g}", {})) for s in sigmas)
            + " |"
        )
    lines += [
        "",
        "Cell: ‖I_own‖ · ‖I_cross‖ · ‖g_own‖ · ‖g_cross‖ (× 1e3, per row).",
        "",
        "| tier " + head[0],
        "|---|" + rule,
    ]
    for t in win:
        c = tab[t]
        lines.append(
            f"| {t} | "
            + " | ".join(i_cross(c.get(f"{s:g}", {})) for s in sigmas)
            + " |"
        )
    split = summ.get(f"{mode}/split", {})
    if split:
        lines += [
            "",
            f"## {mode} — windows by length and orientation",
            "",
            "Cell: f_own · f_cross · ‖I_own‖ × 1e3 (items).",
            "",
            "| tier · split " + head[0],
            "|---|" + rule,
        ]
        for label in sorted(split):
            c = split[label]
            cells = []
            for s in sigmas:
                x = c.get(f"{s:g}")
                cells.append(
                    "–"
                    if not x
                    else f"{x['f_glyph'][1]:.2f} · {opt(x, 'f_cross')} · "
                    f"{x['g_ident'][1] * 1e3:.1f} ({x['n']})"
                )
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def report(run_dir: Path, plan: dict, per: list, summ: dict) -> None:
    sigmas = plan["sigmas"]
    lines = [
        f"# grad_identity — {run_dir.name}",
        "",
        f"Items {plan['n_items']} (plain) / {plan['n_clause']} (clause), m = "
        f"{plan['wrong']}, σ {sigmas}, seed {plan['seed']}. Medians over items; "
        "`(k/n)` = items with the excess > 0; `*` / `−` = sign test p < 0.01 "
        "above / below zero; *separates at* = the σ with `*`.",
        "",
    ]

    def f_mag(x):
        return (
            f"{x['id2'][1]:.2f} · {x['null'][1]:.2f} · {x['excess'][1]:+.3f}"
            f"{_mark(x, 'excess')} ({x['excess_pos']}/{x['n']})"
        )

    def f_dir(x):
        return (
            f"{x['id2_dir'][1]:.2f} · {x['null_dir'][1]:.2f} · "
            f"{x['excess_dir'][1]:+.3f}{_mark(x, 'excess_dir')} "
            f"({x['excess_dir_pos']}/{x['n']})"
        )

    def f_norm(x):
        return (
            f"{x['lnorm'][1]:+.2f} · {x['excess_norm'][1]:+.2f}"
            f"{_mark(x, 'excess_norm')} ({x['excess_norm_pos']}/{x['n']})"
        )

    def f_glyph(x):
        return (
            f"{x['f_glyph'][1]:.3f} [{x['f_glyph'][0]:.2f}–{x['f_glyph'][2]:.2f}] · "
            f"{x['g_ident'][1] * 1e3:.1f}"
        )

    def f_lev(x):
        return (
            f"{x['dloss'][1] * 1e3:+.2f}{_mark(x, 'dloss')} ({x['dloss_pos']}) · "
            f"{x['g_true'][1]:.2g} · {x['cos_layout'][1]:.2f}"
        )

    for mode in ("render", "plain", "clause"):
        if f"{mode}/tier" not in summ:
            continue
        if mode == "render":
            lines += [
                "**render** = pass 2: row u, the true caption, the image re-drawn "
                "with slot k's glyph swapped (true = render A, wrong = B_j); "
                "`dloss` = mean L(B_j) − L(A) under the true caption.",
                "",
            ]
        lines += [
            f"## {mode} — id_share against the null (the brief's quantity)",
            "",
            "Cell: id2 (‖g_true − mean of 2 wrong‖ / ‖g_true‖, leave-one-out) · "
            "null (the same for a wrong row against the other two) · excess.",
            "",
            *_tier_table(summ, per, mode, sigmas, f_mag, "excess"),
            "",
            f"## {mode} — direction only (unit gradients)",
            "",
            "Cell: id2_dir · null_dir · excess_dir — the same estimator on "
            "g / ‖g‖, so a row's own gradient scale drops out.",
            "",
            *_tier_table(summ, per, mode, sigmas, f_dir, "excess_dir"),
            "",
            f"## {mode} — norm only",
            "",
            "Cell: ln(‖g_true‖ / mean ‖g_wrong‖) · excess over the same ratio "
            "for a wrong row (negative = the true row is pushed less).",
            "",
            *_tier_table(summ, per, mode, sigmas, f_norm, "excess_norm"),
            "",
            f"## {mode} — glyph-dependent share of the gradient",
            "",
            "Cell: f = 1 − cos(g_wrong_j, g_wrong_k) median [IQR] · ‖I‖ = "
            "‖g_wrong‖·√f × 1e3. With g = L + I(glyph), f is I's share of the "
            "energy (render: the share that changes with the glyph drawn; plain / "
            "clause: with the row standing in the slot). bf16 floor: f ≈ 0.01 "
            "(B = 1 vs B = 4 kernels, rel 0.12); run-to-run 2e-4. *separates at* "
            "is not used here.",
            "",
            *_tier_table(summ, per, mode, sigmas, f_glyph, "dloss"),
            "",
            f"## {mode} — caption leverage and exposure",
            "",
            "Cell: dloss = mean L(wrong) − L(true) × 1e3 (items > 0) · "
            "‖g_true‖ (raw units) · cos(g_true, layout).",
            "",
            *_tier_table(summ, per, mode, sigmas, f_lev, "dloss"),
            "",
            f"## {mode} — per form × slot px",
            "",
            "Cell: excess (mag) · excess_dir, items with excess_dir > 0.",
            "",
            "| form px | "
            + " | ".join(f"σ {s:g}" for s in sigmas)
            + " | dir separates at |",
            "|---|" + "---|" * len(sigmas) + "---|",
        ]
        tab = summ[f"{mode}/bin"]
        for key2 in sorted(
            tab,
            key=lambda k: (
                k.split()[0],
                float(k.split()[1].lstrip("≥").split("–")[0]),
            ),
        ):
            c = tab[key2]
            cells = []
            for s in sigmas:
                x = c.get(f"{s:g}")
                cells.append(
                    "–"
                    if not x
                    else f"{x['excess'][1]:+.2f}{_mark(x, 'excess')} · "
                    f"{x['excess_dir'][1]:+.3f}{_mark(x, 'excess_dir')} "
                    f"({x['excess_dir_pos']}/{x['n']})"
                )
            sep = separation(c, sigmas, "excess_dir")
            lines.append(
                f"| {key2} | "
                + " | ".join(cells)
                + f" | {', '.join(f'{s:g}' for s in sep) or 'none'} |"
            )
        lines.append("")
        lines += _window_tables(summ, mode, sigmas)
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def render_both(args, plan: list, recs: dict, out: Path) -> tuple:
    """Pass 2's items: the grid / lone tiers of ``--render_tiers`` from the
    pass-1 plan, the bubble tiers drawn by ``scene_render_plan``."""
    scene = [t for t in SCENE_TIERS if t in args.render_tiers]
    grid = [t for t in args.render_tiers if t not in SCENE_TIERS]
    kept, dropped = (
        render_plan(plan, recs, grid, args.seed, out) if grid else ([], Counter())
    )
    if scene:
        k2, d2 = scene_render_plan(
            recs, scene, args.items, args.window_items, args.wrong, args.seed, out
        )
        kept, dropped = kept + k2, dropped + d2
    return kept, dropped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--items", type=int, default=40, help="items per tier (plain)")
    p.add_argument(
        "--clause_items", type=int, default=16, help="of those, read under clause too"
    )
    p.add_argument("--sigmas", type=float, nargs="+", default=list(SIGMAS))
    p.add_argument("--wrong", type=int, default=3)
    p.add_argument("--captions", choices=["plain", "clause", "both"], default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verify", type=int, default=4, help="items in the batching check")
    p.add_argument("--diag", action="store_true", help="stop after the verify")
    p.add_argument(
        "--render", action="store_true", help="pass 2 (image swap) instead of pass 1"
    )
    p.add_argument("--render_tiers", nargs="+", default=list(RENDER_TIERS))
    p.add_argument(
        "--window_items", type=int, default=60, help="items per bubbleN tier (pass 2)"
    )
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--analyze", type=Path, help="re-read a finished results dir (CPU)")
    args = p.parse_args()
    if args.analyze:
        analyze(args.analyze if args.analyze.is_absolute() else LINE / args.analyze)
        return
    modes = ["plain", "clause"] if args.captions == "both" else [args.captions]
    recs = load_recs()
    clause_items = args.clause_items if "clause" in modes else 0
    if modes == ["clause"]:
        clause_items = args.items
    plan, skipped = make_plan(recs, args.items, clause_items, args.wrong, args.seed)
    n_clause = sum(it["clause"] for it in plan)
    per_tier = Counter(it["tier"] for it in plan)
    n_reads = (len(plan) * ("plain" in modes) + n_clause) * len(args.sigmas)
    print(
        f"{NAME}: {len(plan)} items ({dict(per_tier)}), skipped {dict(skipped)}; "
        f"clause {n_clause}; σ {args.sigmas}; m {args.wrong}; "
        f"{n_reads} batched reads of {1 + args.wrong}",
        flush=True,
    )
    print(
        "u per tier: "
        + "; ".join(
            f"{t}: {''.join(sorted(it['u'] for it in plan if it['tier'] == t))}"
            for t in TIERS
        ),
        flush=True,
    )
    if args.render:
        out = (
            Path(os.environ.get("TMPDIR", "/tmp")) / f"{NAME}_renders"
            if args.dry_run
            else None
        )
        if args.dry_run:
            plan_r, dropped = render_both(args, plan, recs, out)
            print(
                f"render: {len(plan_r)} items kept "
                f"({dict(Counter(it['tier'] for it in plan_r))}), dropped "
                f"{dict(dropped)}; PNGs in {out}",
                flush=True,
            )
            t5_check(plan_r, ["plain"])
            return
        run_dir = make_run_dir(
            NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
        )
        plan_r, dropped = render_both(args, plan, recs, run_dir / "renders")
        print(
            f"render: {len(plan_r)} items kept "
            f"({dict(Counter(it['tier'] for it in plan_r))}), dropped {dict(dropped)}",
            flush=True,
        )
        (run_dir / "plan.json").write_text(
            json.dumps(
                {
                    "pass": "render",
                    "sigmas": list(args.sigmas),
                    "wrong": args.wrong,
                    "seed": args.seed,
                    "n_items": len(plan_r),
                    "n_clause": 0,
                    "dropped": dict(dropped),
                    "items": plan_r,
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        metrics = run_render(args, plan_r, run_dir)
        summ = analyze(run_dir)
        metrics["separates"] = {
            key: {
                tier: separation(c, args.sigmas, key)
                for tier, c in summ.get("render/tier", {}).items()
            }
            for key in EXCESS
        }
        write_result(
            run_dir,
            script=__file__,
            args=args,
            label=args.label,
            metrics=metrics,
            artifacts=[str(run_dir / "report.md")],
        )
        print(f"→ {run_dir / 'result.json'}", flush=True)
        return
    if args.dry_run:
        t5_check(plan, modes)
        ex = plan[0]
        print(json.dumps(ex, ensure_ascii=False, indent=1), flush=True)
        return
    run_dir = make_run_dir(
        NAME, label=args.label, root=LINE / "experiments" / NAME / "results"
    )
    (run_dir / "plan.json").write_text(
        json.dumps(
            {
                "data": {k: str(v) for k, v in DIRS.items()},
                "modes": modes,
                "sigmas": list(args.sigmas),
                "wrong": args.wrong,
                "seed": args.seed,
                "n_items": len(plan),
                "n_clause": n_clause,
                "context": str(SEED_ROWS_0921),
                "items": plan,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    metrics = run_gpu(args, plan, modes, run_dir)
    if args.diag:
        write_result(
            run_dir, script=__file__, args=args, label=args.label, metrics=metrics
        )
        return
    summ = analyze(run_dir)
    metrics["separates"] = {
        key: {
            f"{mode}/{tier}": separation(c, args.sigmas, key)
            for mode in modes
            for tier, c in summ.get(f"{mode}/tier", {}).items()
        }
        for key in EXCESS
    }
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(run_dir / "report.md")],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
