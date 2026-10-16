#!/usr/bin/env python
"""freeswap — M2 of ``proposal_template.md``: swapword's M / S split, read where
the base still chooses the canvas (user 10-16: glyphs memorized, templates not).

M1 (``swapword.py``) read held-out training items, where the canvas is already
in x_σ, and was blind to the template. Here x₀ is the EN render of the identity
grid's prompt × seed (p00 / p01 × ``--seeds``, drawn again on identity's render
path to get the latents, checked against ``OUT/commit/EN/``), x_T that seed's
noise, and x_σ = (1 − σ)x₀ + σx_T at σ in ``SIGMAS``. At σ 1.0 / 0.95 the
canvas is free; at 0.9 / 0.85 the EN layout is pinned and what is free is style.

Captions: ``hira_cold_hp``'s 13 ``read`` words as ``<prompt>. Text reads as
"<w>".`` plus ``K`` swaps each (M1's rule: every glyph replaced by another of
its script, none of the word's own; the same swaps for every arm). Tables:
``FLOOR``'s with the 81 hiragana rows the arm's (arms differ in those only);
``ref`` the same with the hiragana and katakana rows at the pack. The katakana
words therefore read the floor's (``retrain_kana``-trained) katakana rows in
every arm: identical across arms by construction (a determinism check), and a
level for what retrain_kana-trained rows carry against the pack.

One forward a table on the 1 + K captions, one uncond forward shared. On the
generation path's x̂₀ = x_σ − σ(v_u + g(v_c − v_u)) (g = 4; the arms share v_u,
so arm − ref = −σg(v_arm − v_ref)), in the latent scaled by the VAE's
per-channel std:

- ``M`` = mean over the captions of x̂₀(arm) − x̂₀(ref) (holds whatever the
  word), ``S`` = the rest (follows the word); ``T`` / ``V`` their mean squared
  size over the image, in the EN render's text boxes (``_in``, the detector's
  boxes; none on some seeds) and outside them dilated (``_out``).
- pixel reads of decode(mean-over-captions x̂₀), arm − ref: ``white`` /
  ``ink`` / ``sat`` / ``lum`` over the image and in the box. Since the ref is
  subtracted, this is M's pixel shift: the pale band is white up, sat down.
- ``div`` = the variance over the seeds of the true caption's x̂₀ (arm) over
  the same for ref, per word × prompt × σ; ``collapse`` = 1 − div.

Template reads for the gate (per hiragana word, pooled over prompt × seed ×
σ ∈ {1.0, 0.95}): ``pale`` = −Δsat (image) and ``collapse`` (image). Gate,
per read: the floor above both ``hira_cold_sball_lr3_x2`` and
``hira_cold_sball_lr3`` on ≥ 8 / 9 words; ``hira_cold_up`` above
``hira_cold_hp`` (the positive control); the katakana words tie. M2 is the
template meter if a read passes all three.

    make daemon-run ARGS="project/cjk_anima_reseed/probes/freeswap.py --label smoke --words 2 --seeds 2"
    make daemon-run ARGS="project/cjk_anima_reseed/probes/freeswap.py --label all"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reseed import HOME, OUT, SCALE_OUT, bootstrap  # noqa: E402

bootstrap()
os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the runs trained routed, read routed

FLOOR = "seed_retrain_0930"
X2, LR3, UP, HP = (
    "hira_cold_sball_lr3_x2",
    "hira_cold_sball_lr3",
    "hira_cold_up",
    "hira_cold_hp",
)
ARMS = [FLOOR, X2, LR3, HP, UP]
SIGMAS = (1.0, 0.95, 0.9, 0.85)
FREE = (1.0, 0.95)  # the σ where the canvas is free: the gate's
K = 3
OUT_DIL = 2  # latent cells of slack around the box for the out-box reads
PROMPTS = (0, 1)
EN_WORD = "hi"
UNC = ""
ROOT = OUT / "freeswap"
COMMIT_EN = OUT / "commit" / "EN"
SRC = SCALE_OUT / "retrain_kana" / "native_r4_plain" / "native_reads.json"
HIRA = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわをんゔ"
KATA = "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロワヲンヴ"
PX_KEYS = ("white", "ink", "sat", "lum")
SHEET_SEEDS = (0, 1)


def script(w: str) -> str:
    return "hira" if all(c in HIRA for c in w) else "kata"


def swaps(w: str) -> list:
    """``K`` swaps of ``w``: the k-th takes the k-th of ``K`` distinct draws
    per position from its script, none of ``w``'s own glyphs."""
    al = HIRA if script(w) == "hira" else KATA
    rng = random.Random(f"freeswap-{w}")
    pool = sorted(set(al) - set(w))
    draws = [rng.sample(pool, K) for _ in w]
    out = ["".join(d[k] for d in draws) for k in range(K)]
    assert len({w, *out}) == K + 1, out
    return out


def grid(n_words: int) -> tuple[list, dict]:
    from reseed.config import load

    prompts = {
        m["pi"]: m["prompt"] for m in json.loads(SRC.read_text(encoding="utf-8"))
    }
    words = list(load(HP).read)
    if n_words:  # a smoke: the first n of each script
        words = [w for w in words if script(w) == "hira"][:n_words] + [
            w for w in words if script(w) == "kata"
        ][:1]
    out = []
    for w in words:
        sw = swaps(w)
        for pi in PROMPTS:
            out.append(
                {
                    "key": w,
                    "script": script(w),
                    "pi": pi,
                    "swaps": sw,
                    "caps": [f'{prompts[pi]}. Text reads as "{x}".' for x in [w, *sw]],
                }
            )
    return out, prompts


class _X0:
    """Stands in for ``TrajStatsRecorder``: keeps the step-0 latent (x_T)."""

    live = None

    def __init__(self):
        self.xT, self.sig0 = None, None

    def record(self, i, sigma, latents, noise_pred, uncond_noise_pred=None):
        if i == 0:
            self.sig0, self.xT = float(sigma), latents.detach().float().clone()

    def flush(self):
        pass

    @classmethod
    def from_args(cls, args, *, seed=None):
        return cls.live


def en_latents(r, prompts: dict, seeds: int) -> dict:
    """``{(pi, s): (x0, xT)}`` (1, C, H, W) model-space latents of the EN render,
    on disk under ``ROOT/EN``; each checked against commit's EN PNG."""
    import copy

    import numpy as np
    import torch
    from PIL import Image

    import library.inference.generation as gen_mod
    from common.models import decode_image
    from eval.enref import enref_caption

    out = {}
    r.set_arm(None)
    for pi in PROMPTS:
        for s in range(seeds):
            f = ROOT / "EN" / f"p{pi:02d}_s{s}.pt"
            if f.exists():
                out[pi, s] = torch.load(f)
                continue
            cap = _X0.live = _X0()
            old, gen_mod.TrajStatsRecorder = gen_mod.TrajStatsRecorder, _X0
            a2 = copy.deepcopy(r.args)
            a2.prompt, a2.seed, a2.image_size = (
                enref_caption(prompts[pi], EN_WORD),
                s,
                (512, 512),
            )
            a2.negative_prompt = ""
            with torch.no_grad():
                lat = gen_mod.generate(a2, r.gen, r.shared)
            gen_mod.TrajStatsRecorder, _X0.live = old, None
            assert abs(cap.sig0 - 1.0) < 1e-3, f"step-0 σ {cap.sig0}"
            x0 = lat.float()
            x0 = x0.squeeze(2) if x0.dim() == 5 else x0
            xT = cap.xT.squeeze(2) if cap.xT.dim() == 5 else cap.xT
            img = decode_image(r.vae, x0.unsqueeze(2), r.device)
            f.parent.mkdir(parents=True, exist_ok=True)
            img.save(f.with_suffix(".png"))
            ref = COMMIT_EN / f"p{pi:02d}_s{s}.png"
            if ref.is_file():
                d = np.abs(
                    np.asarray(img, np.float32)
                    - np.asarray(Image.open(ref).convert("RGB"), np.float32)
                ).mean()
                print(
                    f"check EN p{pi:02d} s{s} vs commit: mean |Δpx| {d:.3f}", flush=True
                )
                assert d < 12.0, "the EN render does not reproduce commit's"
            out[pi, s] = (x0.cpu(), xT.cpu())
            torch.save(out[pi, s], f)
    return out


def en_boxes(seeds: int) -> dict:
    """``{(pi, s): [box, …]}`` from the text detector on the EN renders (pixel
    coords, 512²; boxes over half the image dropped), cached."""
    f = ROOT / "EN" / "boxes.json"
    got = json.loads(f.read_text()) if f.exists() else {}
    todo = [(pi, s) for pi in PROMPTS for s in range(seeds) if f"{pi}_{s}" not in got]
    if todo:
        import torch

        from anime_tools.ocr.animetext import AnimeTextDetector
        from common.readers import load_bgr

        det = AnimeTextDetector.load(device="cuda")
        for pi, s in todo:
            bs = [
                [int(v) for v in b]
                for b in det.detect(load_bgr(ROOT / "EN" / f"p{pi:02d}_s{s}.png"))
            ]
            got[f"{pi}_{s}"] = [
                b for b in bs if (b[2] - b[0]) * (b[3] - b[1]) < 0.5 * 512 * 512
            ]
        del det
        torch.cuda.empty_cache()
        f.write_text(json.dumps(got, indent=1))
    return {(pi, s): got[f"{pi}_{s}"] for pi in PROMPTS for s in range(seeds)}


def box_mask(boxes: list, h: int, w: int, scale: int, device):
    """``(1, 1, h, w)`` mask of ``boxes`` at ``scale`` px a cell, or None."""
    import torch

    if not boxes:
        return None
    m = torch.zeros(1, 1, h, w, device=device)
    for x0, y0, x1, y1 in boxes:
        m[:, :, y0 // scale : -(-y1 // scale), x0 // scale : -(-x1 // scale)] = 1.0
    return m


def ref_tables(names: list):
    """The floor's table with the 81 hiragana rows each arm's; ``ref`` with the
    hiragana and katakana rows at the pack."""
    from data.inventory import qwen_pieces
    from eval.ruler.arms import tables
    from reseed.trainer import vocab_idx

    tokq = qwen_pieces(char_rows=True)
    hira_ids = vocab_idx(list(HIRA), tokq)
    kata_ids = vocab_idx(list(KATA), tokq)
    ids, tabs = tables(list(dict.fromkeys([FLOOR, *names])))
    pos = [i for i, e in enumerate(ids) if e in hira_ids]
    assert len(pos) == len(hira_ids) == len(HIRA), (len(pos), len(hira_ids))
    kpos = [i for i, e in enumerate(ids) if e in kata_ids]
    base = tabs[FLOOR]
    out = {"ref": base.clone()}
    out["ref"][pos + kpos] = 0.0
    for a in names:
        t = base.clone()
        t[pos] = tabs[a][pos]
        out[a] = t
    print(
        f"tables: {len(pos)} hiragana rows per arm, {len(kpos)} katakana rows "
        f"(the floor's; at the pack in ref)",
        flush=True,
    )
    return out


def px_reads(px, boxes) -> dict:
    """``px`` (3, H, W) in [0, 1] → the pixel reads over the image and in the
    box union."""
    hi, lo = px.max(0).values, px.min(0).values
    lum = 0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]
    maps = {
        "white": ((lo > 0.85) & (hi - lo < 0.08)).float(),
        "ink": (hi < 0.25).float(),
        "sat": hi - lo,
        "lum": lum,
    }
    out = {k: float(v.mean()) for k, v in maps.items()}
    if boxes is not None:
        n = boxes.sum().clamp(min=1)
        out.update({f"{k}_in": float((v * boxes).sum() / n) for k, v in maps.items()})
    return out


def run(a) -> None:
    import torch
    import torch.nn.functional as F

    from common.models import dit_forward, encode_captions
    from common.readers import contact_sheet
    from eval.ruler.arms import arm_dirs
    from eval.ruler.render import Renderer
    from library.inference.traj_stats import QWEN_LATENTS_STD
    from PIL import Image

    names = [x for x in ARMS if x in arm_dirs()]
    assert names == ARMS, f"arms not on disk: {sorted(set(ARMS) - set(names))}"
    g, prompts = grid(a.words)
    cache = encode_captions(
        [c for m in g for c in m["caps"]] + [UNC], "cuda", ROOT / "te_cache"
    )
    tabs = ref_tables(names)
    r = Renderer()
    dev = r.device
    gscale = float(r.args.guidance_scale)
    en = en_latents(r, prompts, a.seeds)
    bx = en_boxes(a.seeds)
    order = ["ref", *names]
    tabs = {k: v.to(dev) for k, v in tabs.items()}
    std = torch.tensor(QWEN_LATENTS_STD, device=dev).view(1, -1, 1, 1)

    def fwd(xs, sig, caps):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ts = torch.full((len(caps),), sig, device=dev)
            xb = xs.to(torch.bfloat16).repeat(len(caps), 1, 1, 1)
            return dit_forward(r.anima, xb, ts, cache, caps, dev).float()

    rows, divs = [], []
    t0 = time.time()
    n_tot = len(g) * a.seeds * len(SIGMAS)
    n_done = 0
    for m in g:
        caps, w, pi = m["caps"], m["key"], m["pi"]
        true_x0 = {(t, sg): [] for t in order for sg in SIGMAS}  # per-seed x̂₀
        union = None
        for s in range(a.seeds):
            x0, xT = (v.to(dev) for v in en[pi, s])
            _, _, h, wd = x0.shape
            b_lat = box_mask(bx[pi, s], h, wd, 8, dev)
            b_px = box_mask(bx[pi, s], 512, 512, 1, dev)
            if b_lat is not None:
                union = b_lat if union is None else torch.maximum(union, b_lat)
                b_out = 1.0 - F.max_pool2d(
                    b_lat, 2 * OUT_DIL + 1, stride=1, padding=OUT_DIL
                )
            for sig in SIGMAS:
                xs = (1 - sig) * x0 + sig * xT
                vu = fwd(xs, sig, [UNC])
                X = {}
                for t in order:
                    r.set_arm(tabs[t])
                    vc = fwd(xs, sig, caps)
                    X[t] = xs - sig * (vu + gscale * (vc - vu))
                    true_x0[t, sig].append((X[t][:1] / std).half())
                means = torch.stack([X[t].mean(0) for t in order]).unsqueeze(2)
                with torch.no_grad():
                    px = r.vae.decode_to_pixels(means.to(r.vae.dtype))
                px = (px.squeeze(2).float().clamp(-1, 1) + 1) / 2
                pr = {
                    t: px_reads(px[k], b_px[0, 0] if b_px is not None else None)
                    for k, t in enumerate(order)
                }
                if s in SHEET_SEEDS and sig in FREE:
                    for k, t in enumerate(order):
                        f = ROOT / "dec" / t / f"{w}_p{pi:02d}_s{s}_{sig:g}.png"
                        f.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(
                            (px[k].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
                        ).resize((256, 256)).save(f)
                for t in names:
                    d = (X[t] - X["ref"]) / std
                    M = d.mean(0, keepdim=True)
                    S = d - M
                    rec = {
                        "arm": t,
                        "key": w,
                        "script": m["script"],
                        "pi": pi,
                        "seed": s,
                        "sigma": sig,
                        "T": float(M.pow(2).mean()),
                        "V": float(S.pow(2).mean()),
                        **{f"d_{k}": v - pr["ref"][k] for k, v in pr[t].items()},
                    }
                    if b_lat is not None:
                        n_in = b_lat.sum().clamp(min=1)
                        n_out = b_out.sum().clamp(min=1)
                        e = lambda x, mk, n: float(  # noqa: E731
                            (x.pow(2).mean((0, 1), keepdim=True) * mk).sum() / n
                        )
                        rec.update(
                            T_in=e(M, b_lat, n_in),
                            V_in=e(S, b_lat, n_in),
                            T_out=e(M, b_out, n_out),
                            V_out=e(S, b_out, n_out),
                        )
                    rows.append(rec)
                n_done += 1
            el = time.time() - t0
            print(
                f"  {w} p{pi:02d} s{s}: {n_done} / {n_tot} ({el / 60:.1f} min, eta "
                f"{el / n_done * (n_tot - n_done) / 60:.1f})",
                flush=True,
            )
        for sig in SIGMAS:
            var = {}
            for t in order:
                st = torch.cat(true_x0[t, sig]).float()  # (seeds, C, h, w)
                v = st.var(0, keepdim=True).mean(1, keepdim=True)
                var[t] = (
                    float(v.mean()),
                    float((v * union).sum() / union.sum())
                    if union is not None
                    else None,
                )
            for t in names:
                divs.append(
                    {
                        "arm": t,
                        "key": w,
                        "script": m["script"],
                        "pi": pi,
                        "sigma": sig,
                        "div": var[t][0] / var["ref"][0],
                        "div_in": var[t][1] / var["ref"][1]
                        if var[t][1] is not None
                        else None,
                    }
                )

    res = summarize(rows, divs, names, [m["key"] for m in g if m["pi"] == 0])
    from bench._common import make_run_dir, write_result

    run_dir = make_run_dir(
        "cjk_anima_reseed", label=f"freeswap-{a.label}", root=HOME / "results"
    )
    (run_dir / "rows.json").write_text(
        json.dumps({"rows": rows, "divs": divs}, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "table.md").write_text(res.pop("md"), encoding="utf-8")
    for pi in PROMPTS:
        for sig in FREE:
            cells = []
            for w in res["words"]:
                for t in order:
                    f = ROOT / "dec" / t / f"{w}_p{pi:02d}_s0_{sig:g}.png"
                    cells.append((Image.open(f).convert("RGB"), [w, t[-18:], "", ""]))
            contact_sheet(
                cells,
                run_dir / f"sheet_p{pi:02d}_s0_{sig:g}.png",
                thumb=192,
                cols=len(order),
            )
    write_result(
        run_dir,
        script="project/cjk_anima_reseed/probes/freeswap.py",
        args={**vars(a), "K": K, "sigmas": SIGMAS, "free": FREE, "arms": ARMS},
        metrics=res,
    )
    print((run_dir / "table.md").read_text(encoding="utf-8"), flush=True)
    print(f"freeswap → {run_dir}", flush=True)


def summarize(rows: list, divs: list, names: list, words: list) -> dict:
    import statistics as stt

    from scipy.stats import binomtest, wilcoxon

    hira = [w for w in words if script(w) == "hira"]
    kata = [w for w in words if script(w) == "kata"]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return stt.fmean(xs) if xs else float("nan")

    def read(arm, w, key, sigmas=FREE):
        """A word's template read for ``arm``, pooled over prompt × seed × σ."""
        if key == "collapse":
            return mean(
                1 - d["div"]
                for d in divs
                if d["arm"] == arm and d["key"] == w and d["sigma"] in sigmas
            )
        if key == "pale":
            return mean(
                -x["d_sat"]
                for x in rows
                if x["arm"] == arm and x["key"] == w and x["sigma"] in sigmas
            )
        return mean(
            x.get(key)
            for x in rows
            if x["arm"] == arm and x["key"] == w and x["sigma"] in sigmas
        )

    res: dict = {
        "words": words,
        "by_sigma": {},
        "gate": {},
        "per_word": {},
        "paired": {},
    }
    cols = (
        "T",
        "V",
        "T_in",
        "V_in",
        "T_out",
        "d_white",
        "d_sat",
        "d_lum",
        "d_white_in",
        "d_sat_in",
        "div",
    )
    lines = []
    for scr, ws in (("hiragana", hira), ("katakana", kata)):
        lines += [
            f"## {scr} words ({len(ws)}), by σ (prompt × seed pooled)",
            "",
            "| arm | σ | " + " | ".join(cols) + " |",
            "|---|---" + "|---" * len(cols) + "|",
        ]
        for t in names:
            for sig in SIGMAS:
                rs = [
                    x
                    for x in rows
                    if x["arm"] == t and x["key"] in ws and x["sigma"] == sig
                ]
                ds = [
                    d
                    for d in divs
                    if d["arm"] == t and d["key"] in ws and d["sigma"] == sig
                ]
                v = {c: mean(x.get(c) for x in rs) for c in cols if c != "div"}
                v["div"] = mean(d["div"] for d in ds)
                res["by_sigma"].setdefault(scr, {}).setdefault(t, {})[str(sig)] = v
                lines.append(
                    f"| {t} | {sig:g} | "
                    + " | ".join(f"{v[c]:.4f}" for c in cols)
                    + " |"
                )
        lines.append("")
    for key in ("pale", "collapse", "T", "d_white"):
        pw = {t: {w: read(t, w, key) for w in words} for t in names}
        res["per_word"][key] = pw
        n = len(hira)

        def sign(a, b):  # words a reads above b, one-sided sign test at 1/2
            k = sum(pw[a][w] > pw[b][w] for w in hira)
            p = binomtest(k, n, 0.5, alternative="greater").pvalue if n else 1.0
            return k, float(p)

        (k_x2, p_x2), (k_lr3, p_lr3), (k_up, p_up) = (
            sign(FLOOR, X2),
            sign(FLOOR, LR3),
            sign(UP, HP),
        )
        kspread = max(
            (max(pw[t][w] for t in names) - min(pw[t][w] for t in names) for w in kata),
            default=0.0,
        )
        gate = {
            "floor_above_x2": k_x2,
            "p_x2": p_x2,
            "floor_above_lr3": k_lr3,
            "p_lr3": p_lr3,
            "n": n,
            "up_above_hp": k_up,
            "p_up": p_up,
            "up_minus_hp": mean(pw[UP][w] - pw[HP][w] for w in hira),
            "kata_spread": kspread,
            "kata_level": mean(pw[FLOOR][w] for w in kata),
        }
        gate["pass"] = (
            min(k_x2, k_lr3) >= min(8, n)
            and k_up > n / 2
            and gate["up_minus_hp"] > 0
            and kspread < 1e-3 * abs(gate["kata_level"]) + 1e-6
        )
        res["gate"][key] = gate
        lines += [
            f"## {key} per word (σ {FREE} pooled)",
            "",
            "| word | " + " | ".join(names) + " |",
            "|---" * (len(names) + 1) + "|",
        ]
        for w in words:
            lines.append(
                f"| {w} | " + " | ".join(f"{pw[t][w]:.4f}" for t in names) + " |"
            )
        lines += [
            "",
            f"gate {key}: floor > x2 on {k_x2} / {n} (p {p_x2:.2g}), floor > lr3 on "
            f"{k_lr3} / {n} (p {p_lr3:.2g}); up > hp on {k_up} / {n} "
            f"(p {p_up:.2g}, mean Δ {gate['up_minus_hp']:+.4f}); "
            f"katakana spread {gate['kata_spread']:.2g} (level {gate['kata_level']:.4f}) "
            f"→ {'PASS' if gate['pass'] else 'fail'}",
            "",
        ]
    # slot-paired (word × prompt × seed × σ in FREE) floor against x2 / lr3, hiragana
    for key, f in (("pale", lambda x: -x["d_sat"]), ("T", lambda x: x["T"])):
        by = {
            (x["arm"], x["key"], x["pi"], x["seed"], x["sigma"]): f(x)
            for x in rows
            if x["key"] in hira and x["sigma"] in FREE
        }
        for o in (X2, LR3):
            d = [v - by[(o, *k[1:])] for k, v in by.items() if k[0] == FLOOR]
            p = float(wilcoxon(d).pvalue) if any(d) else 1.0
            res["paired"][f"{key}:{FLOOR}-{o}"] = {
                "mean_d": mean(d),
                "up": sum(x > 0 for x in d),
                "down": sum(x < 0 for x in d),
                "p": p,
            }
            lines.append(
                f"paired {key} {FLOOR} − {o}: {mean(d):+.4f}, "
                f"{sum(x > 0 for x in d)} / {sum(x < 0 for x in d)}, p {p:.2g}"
            )
    res["md"] = "\n".join(lines) + "\n"
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument(
        "--words", type=int, default=0, help="smoke: the first N hiragana + 1 katakana"
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
