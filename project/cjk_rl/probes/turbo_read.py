#!/usr/bin/env python
"""turbo_read — the string read measured on the turbo itself: merged15 + the official
turbo v1.1 delta (``m15v11``, 4 Euler steps, cfg 1), fresh renders, today's reader.

Two parts, each rendered, read and tabulated:

    ruler   the dialogue ruler's 96 strings (``ruler.json`` as it stands, captioned
            prompts) × ``SEEDS``: the EN reference (the 66 strings with an EN line,
            rows off), jp_v1's rows (``seed_1008``) and the old floor's
            (``seed_retrain_0930``). Tables as ``position_read.py``: hits by position,
            length, class, substitutions, by token count; per seed, so the spread
            between seeds is on record.
    kanji   singles and words in a bubble (``<plain prompt>, speech bubble, japanese
            text. Japanese text reads as "<key>".``, 768², p00 / p01 × seeds 0 / 1):
            96 kanji (32 frequent and 32 rare of the 1 185 both tables hold, 32 of the
            225 only jp_v1 holds: on the old floor those are raw pack rows, the
            untrained control), the 24 hiragana singles of the identity grid, 32 kanji
            words (the scale line's six + 26 common ones). Per stratum: exact /
            contained / another kanji / kana only / nothing; the kanji read instead,
            and its bitmap overlap with the asked one against shuffled pairs.

Decode: the pixel head by default (``--decoder pixel_head``; its turbo point was
trained on this stack), the VAE with ``--decoder vae``. The 11-25 run (job
``20261125-084840-a2aa0d``) decoded with the VAE: 0.9 s a render, 0.30 s of it sampling.
Arms of one run share a decoder; a read across the two decoders needs the reader checked
on both first.

Renders: ``output/cjk_rl/turbo_read/<m15v11_px | m15v11>/<part>/<arm>/`` (pixel head |
VAE; a render on disk is not drawn again); reads cached per arm. Result →
``results/<ts>-turbo_read-<label>/``, with the render's seconds by stage.

    make daemon-run ARGS="--stall-timeout 900 project/cjk_rl/probes/turbo_read.py --label m15v11"
    .venv/bin/python project/cjk_rl/probes/turbo_read.py --label m15v11 --decoder vae --tabulate   # CPU, the 11-25 reads
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_rl
sys.path.insert(0, str(LINE.parent / "cjk_anima_reseed"))
sys.path.insert(0, str(LINE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reseed import REPO, SCALE_OUT, bootstrap  # noqa: E402

bootstrap()

from reseed.config import PACKS  # noqa: E402

os.environ["ANIMA_VOCAB_PACK"] = PACKS["punct"]
os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"

from eval.ruler import RULER, VIEW  # noqa: E402

VIEW.pack = "punct"

import position_read as P  # noqa: E402

MODEL = "m15v11"
SEEDS = (0, 1, 2, 3)
EN, ROWS = "en", ("seed_1008", "seed_retrain_0930")
OUT = REPO / "output" / "cjk_rl" / "turbo_read"
PLAIN = SCALE_OUT / "retrain_kana" / "native_r4_plain" / "native_reads.json"
K_PROMPTS, K_SEEDS, K_SHAPE = (0, 1), (0, 1), (768, 768)
KANA = "あうがくとひもりぬめねれわるろさちっゃぱぽぜゔを"  # probes/identity.py SINGLES
WORDS = (
    "山田太郎 小山田 日本人 大丈夫 何時間 愛してる "  # the scale line's six (C3)
    "今日 先生 学校 時間 自分 最後 本当 一緒 気持ち 大好き 友達 家族 世界 電話 約束 "
    "名前 結婚 仕事 勉強 言葉 部屋 写真 心配 天気 元気 大切"
).split()
FONT = (
    LINE.parent / "cjk_anima_reseed" / "assets" / "fonts" / "GenShinGothic-Medium.ttf"
)


def kanji_sets() -> dict:
    """The kanji strata: 32 evenly spaced over the first 256 of the 1 185 (their
    config lists them most frequent first), 32 over the last 256, 32 over the 225."""
    import tomllib

    cfg = tomllib.loads(
        (LINE.parent / "cjk_anima_reseed/configs/sent_kanji_225.toml").read_text(
            "utf-8"
        )
    )
    old, new = cfg["rows"][2][6:], cfg["rows"][3][6:]
    pick = lambda s: [s[round(k * (len(s) - 1) / 31)] for k in range(32)]  # noqa: E731
    return {
        "kanji frequent": pick(old[:256]),
        "kanji rare": pick(old[-256:]),
        "kanji new 225": pick(new),
    }


def jobs(parts: list, smoke: bool) -> list:
    """``[{part, arm, stem, caption, seed, shape, …}]``."""
    out = []
    if "ruler" in parts:
        its = json.loads((RULER / "ruler.json").read_text("utf-8"))["items"]
        if smoke:
            its = [m for m in its if m["en"]][:3]
        for m in its:
            pr = m["prompts"]["captioned"]
            for arm in (EN, *ROWS):
                if arm == EN and not m["en"]:
                    continue
                for seed in SEEDS[:2] if smoke else SEEDS:
                    out.append(
                        {
                            "part": "ruler",
                            "arm": arm,
                            "stem": f"r{m['i']:02d}_s{seed}",
                            "caption": pr["en_caption"] if arm == EN else pr["caption"],
                            "seed": seed,
                            "shape": tuple(m["shape"]),
                            "i": m["i"],
                            "bin": m["bin"],
                            "text": m["en"] if arm == EN else m["text"],
                        }
                    )
    if "kanji" in parts:
        prompts = {m["pi"]: m["prompt"] for m in json.loads(PLAIN.read_text("utf-8"))}
        groups = {**kanji_sets(), "kana": list(KANA), "word": list(WORDS)}
        for g, keys in groups.items():
            for k in keys[:2] if smoke else keys:
                for pi in K_PROMPTS[:1] if smoke else K_PROMPTS:
                    for seed in K_SEEDS:
                        for arm in ROWS:
                            out.append(
                                {
                                    "part": "kanji",
                                    "arm": arm,
                                    "stem": f"{k}_p{pi:02d}_s{seed}",
                                    "caption": f"{prompts[pi]}, speech bubble, japanese "
                                    f'text. Japanese text reads as "{k}".',
                                    "seed": seed,
                                    "shape": K_SHAPE,
                                    "group": g,
                                    "key": k,
                                    "pi": pi,
                                    "text": k,
                                }
                            )
    return out


def img(j: dict, root: Path) -> Path:
    """The render's file: WebP (quality 95); the 11-25 VAE run's PNG where that is
    what is on disk."""
    f = root / j["part"] / j["arm"] / f"{j['stem']}.webp"
    old = f.with_suffix(".png")
    return old if old.exists() and not f.exists() else f


def render(js: list, root: Path, decoder: str) -> dict:
    """``decoder``: ``pixel_head`` reads the DiT's last-step features (its turbo point
    was trained on this stack, 4 steps at σ 0.5) and the VAE is dropped; ``vae`` is the
    ruler's decode. A caption's seeds are sampled as one batch. Renders are written on
    worker threads as WebP, quality 95."""
    import copy
    import gc
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from common import models
    from common.models import decode_image
    from eval.ruler.arms import tables
    from eval.ruler.render import Renderer
    from library.anima.stream import latent_dtype
    from library.inference import generation
    from library.inference.generation import generate_body
    from library.inference.output import pixels_to_pil
    from library.inference.pixel_decoder import ensure_pixel_decoder, pixel_decode
    from library.inference.text import prepare_text_inputs

    todo = [j for j in js if not img(j, root).exists()]
    if not todo:
        return {"rendered": 0}
    models.TOKEN_COUNTS[:] = [
        (h // 16) * (w // 16) for w, h in {j["shape"] for j in js}
    ]
    _, tabs = tables(list(ROWS))
    r = Renderer(MODEL)
    head = decoder == "pixel_head"
    lat_ch = type(r.anima).LATENT_CHANNELS
    if head:
        r.args.decoder = "pixel_head"
        r.vae = None
        gc.collect()
        torch.cuda.empty_cache()
    t0, n = time.time(), 0
    secs = defaultdict(list)
    todo.sort(key=lambda j: (j["shape"], j["arm"], j["stem"]))
    arm, saves = None, []

    def save(im, fn: Path) -> None:
        fn.parent.mkdir(parents=True, exist_ok=True)
        tmp = fn.with_name(f"{fn.stem}.tmp{fn.suffix}")
        im.save(tmp, quality=95)
        tmp.rename(fn)  # a render on disk is a whole file

    # one forward per caption: its seeds are one batch (``generate_body`` takes the list)
    groups: dict = {}
    for j in todo:
        groups.setdefault((j["shape"], j["arm"], j["caption"]), []).append(j)
    with ThreadPoolExecutor(4) as pool:
        for (shape, g_arm, caption), g in groups.items():
            if g_arm != arm:
                arm = g_arm
                r.set_arm(None if arm == EN else tabs[arm])
            a2 = copy.deepcopy(r.args)
            a2.prompt, a2.seed = caption, g[0]["seed"]
            a2.image_size = (shape[1], shape[0])
            t = time.time()
            with torch.no_grad():
                context, context_null = prepare_text_inputs(
                    a2, r.device, r.anima, r.shared
                )
                r.anima.reset_mod_guidance()
                r.anima.reset_dave()
                ensure_pixel_decoder(a2, r.anima, r.device)
                torch.cuda.synchronize()
                secs["text"].append((time.time() - t) / len(g))
                t = time.time()
                # each seed's noise as the one-image path draws it (a CPU generator)
                noise = []
                for j in g:
                    gen = torch.Generator(device="cpu")
                    gen.manual_seed(j["seed"])
                    noise.append(
                        generation.randn_tensor(
                            (1, lat_ch, 1, shape[1] // 8, shape[0] // 8),
                            generator=gen,
                            device=r.device,
                            dtype=latent_dtype(r.anima),
                        )
                    )
                lat = generate_body(
                    a2,
                    r.anima,
                    context,
                    context_null,
                    r.device,
                    [j["seed"] for j in g],
                    latents=torch.cat(noise, dim=0),
                )
                torch.cuda.synchronize()
                secs["sample"].append((time.time() - t) / len(g))
                t = time.time()
                if head:
                    px = pixel_decode(a2, r.anima)
                    ims = [pixels_to_pil(px[k]) for k in range(len(g))]
                else:
                    ims = [
                        decode_image(r.vae, lat[k : k + 1], r.device)
                        for k in range(len(g))
                    ]
            secs["decode"].append((time.time() - t) / len(g))
            saves += [pool.submit(save, im, img(j, root)) for im, j in zip(ims, g)]
            n += len(g)
            if n // 100 != (n - len(g)) // 100 or n == len(todo):
                print(
                    f"  render {n} / {len(todo)} {shape} {arm} "
                    f"({(time.time() - t0) / 60:.1f} min)",
                    flush=True,
                )
        for f in saves:
            f.result()
    info = {
        "rendered": n,
        "decoder": decoder,
        "minutes": round((time.time() - t0) / 60, 2),
        # a render's median seconds by stage (the median drops each graph's compile;
        # the file write is off the loop)
        "s_per_render": {k: round(sorted(v)[len(v) // 2], 4) for k, v in secs.items()},
    }
    del r
    gc.collect()
    torch.cuda.empty_cache()
    return info


def read(js: list, root: Path, gpu: bool) -> dict:
    """``{(part, arm): {stem: reads}}``, cached in ``<part>/<arm>/reads.json``."""
    from common.readers import Readers, load_bgr

    rd, out = None, {}
    by = defaultdict(list)
    for j in js:
        by[(j["part"], j["arm"])].append(j)
    for (part, arm), xs in by.items():
        f = root / part / arm / "reads.json"
        got = json.loads(f.read_text("utf-8")) if f.exists() else {}
        todo = [j for j in xs if j["stem"] not in got]
        if todo:
            assert gpu, f"{part}/{arm}: {len(todo)} renders unread (run on the daemon)"
            rd = rd or Readers("cuda")
            t0 = time.time()
            for k in range(0, len(todo), 8):
                chunk = todo[k : k + 8]
                reads = rd.read_images([load_bgr(img(j, root)) for j in chunk])
                for j, x in zip(chunk, reads):
                    got[j["stem"]] = x
                if (k // 8) % 10 == 0:
                    print(
                        f"  read {part}/{arm} {k + len(chunk)} / {len(todo)} "
                        f"({(time.time() - t0) / 60:.1f} min)",
                        flush=True,
                    )
            f.write_text(json.dumps(got, ensure_ascii=False, indent=1), "utf-8")
        out[(part, arm)] = got
    return out


# ---------------------------------------------------------------- tables (CPU)


def ruler_tables(js: list, reads: dict) -> tuple[dict, dict]:
    from library.anima.weights import load_t5_tokenizer

    t5 = load_t5_tokenizer()
    res, align = {}, {}
    for arm in (EN, *ROWS):
        xs = [j for j in js if j["part"] == "ruler" and j["arm"] == arm]
        if not xs:
            continue
        real, null, per_seed = [], [], {}
        for seed in sorted({j["seed"] for j in xs}):
            recs = [
                {
                    "i": j["i"],
                    "bin": j["bin"],
                    "text": j["text"],
                    "reads": reads[("ruler", arm)][j["stem"]],
                }
                | (
                    {"ntok": len(t5(j["text"], add_special_tokens=False)["input_ids"])}
                    if arm == EN
                    else {}
                )
                for j in xs
                if j["seed"] == seed
            ]
            own = [r | {"seed": seed} for r in P.arm_recs(recs, "sfx")]
            real += own
            null += [r for s in P.NULL_SHIFTS for r in P.arm_recs(recs, "sfx", s)]
            n = sum(len(r["t"]) for r in own)
            per_seed[seed] = {
                "hit": round(sum(o[0] == "M" for r in own for o in r["ops"]) / n, 4),
                "cer": round(
                    sum(min(1.0, r["lev"] / max(1, len(r["t"]))) for r in own)
                    / len(own),
                    4,
                ),
                "exact": sum(r["lev"] == 0 for r in own),
                "le2": sum(r["lev"] <= 2 for r in own),
            }
        t_real, t_null = P.tables(real), P.tables(null)
        t_real.pop("glyph")
        t_null.pop("glyph")
        res[arm] = {
            "n": len(real),
            "real": t_real,
            "null": t_null,
            "by_tokens": P.by_tokens(real, null),
            "per_seed": per_seed,
        }
        align[arm] = real
    return res, align


def bitmap(c: str):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("L", (72, 72), 0)
    ImageDraw.Draw(im).text((4, 0), c, fill=255, font=ImageFont.truetype(str(FONT), 64))
    return np.asarray(im) > 96


def iou(a: str, b: str) -> float:
    x, y = bitmap(a), bitmap(b)
    u = (x | y).sum()
    return float((x & y).sum() / u) if u else 0.0


def kanji_tables(js: list, reads: dict) -> tuple[dict, list]:
    import random

    from common.text import norm

    rows, recs = defaultdict(Counter), []
    pairs = defaultdict(list)
    for j in js:
        if j["part"] != "kanji":
            continue
        rs = [r for r in reads[("kanji", j["arm"])][j["stem"]] if r.get("sfx")]
        cs = [norm(r["sfx"]) for r in rs]
        key, g = j["key"], j["group"]
        R = rows[(g, j["arm"])]
        R["n"] += 1
        rec = {k: j[k] for k in ("arm", "group", "key", "pi", "seed")} | {"reads": cs}
        if g == "word":
            x = P.read_string(key, rs, "sfx")
            hit = [o[0] == "M" for o in x["ops"]]
            R["exact"] += x["lev"] == 0
            R["glyphs"] += len(hit)
            R["hit"] += sum(hit)
            for c, h in zip(x["t"], hit):
                k = "kanji" if P.cls(c) == "kanji" else "kana"
                R[f"{k}_n"] += 1
                R[f"{k}_hit"] += h
            rec |= {"best": x["best"], "hit": sum(hit) / max(1, len(hit))}
        else:
            if key in cs:
                kind = "exact"
            elif any(key in c for c in cs):
                kind = "contained"
            elif any(P.cls(ch) == "kanji" for c in cs for ch in c):
                kind = "another kanji"
            elif cs:
                kind = "kana / other only"
            else:
                kind = "nothing"
            R[kind] += 1
            rec["kind"] = kind
            if kind == "another kanji" and g != "kana":
                # the most confident box that is one kanji
                one = [
                    (r.get("sfx_conf") or 0, norm(r["sfx"]))
                    for r in rs
                    if len(norm(r["sfx"])) == 1 and P.cls(norm(r["sfx"])) == "kanji"
                ]
                if one:
                    rec["instead"] = max(one)[1]
                    pairs[(g, j["arm"])].append((key, rec["instead"]))
        recs.append(rec)
    out = {f"{g} | {a}": dict(c) for (g, a), c in sorted(rows.items())}
    rng = random.Random(0)
    for (g, a), ps in pairs.items():
        real = [iou(x, y) for x, y in ps]
        ys = [y for _, y in ps]
        sh = []
        for _ in range(20):
            rng.shuffle(ys)
            sh += [iou(x, y) for (x, _), y in zip(ps, ys) if x != y]
        out[f"{g} | {a}"] |= {
            "instead_pairs": len(ps),
            "instead_iou": round(sum(real) / len(real), 3),
            "shuffled_iou": round(sum(sh) / max(1, len(sh)), 3),
            "instead_top": Counter(f"{x}→{y}" for x, y in ps).most_common(12),
        }
    return out, recs


def sheets(js: list, reads: dict, root: Path, out: Path) -> None:
    from PIL import Image

    from common.readers import contact_sheet
    from common.text import norm

    out.mkdir(parents=True, exist_ok=True)

    def label(j: dict) -> list:
        return [
            f"{j['text'][:14]}",
            f"{j['arm']}",
            " / ".join(
                norm(r["sfx"])[:10]
                for r in reads[(j["part"], j["arm"])][j["stem"]]
                if r.get("sfx")
            )[:26],
        ]

    ks = [j for j in js if j["part"] == "kanji" and j["pi"] == 0 and j["seed"] == 0]
    for g in dict.fromkeys(j["group"] for j in ks):
        for arm in ROWS:
            xs = [j for j in ks if j["group"] == g and j["arm"] == arm]
            contact_sheet(
                [(Image.open(img(j, root)).convert("RGB"), label(j)) for j in xs],
                out / f"kanji_{g.replace(' ', '_')}_{arm}.png",
                thumb=256,
                cols=8,
            )
    rs = [j for j in js if j["part"] == "ruler" and j["seed"] == 0]
    for b in ("short", "mid", "long"):
        ids = [i for i in dict.fromkeys(j["i"] for j in rs if j["bin"] == b)]
        ids = [i for i in ids if any(j["i"] == i and j["arm"] == EN for j in rs)][:8]
        rows = []
        for i in ids:
            for arm in (EN, *ROWS):
                j = next(j for j in rs if j["i"] == i and j["arm"] == arm)
                rows.append((Image.open(img(j, root)).convert("RGB"), label(j)))
        if rows:
            contact_sheet(rows, out / f"ruler_{b}.png", thumb=320, cols=3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--parts", default="ruler,kanji")
    p.add_argument("--smoke", action="store_true", help="a few items, its own dir")
    p.add_argument("--tabulate", action="store_true", help="CPU: the reads on disk")
    p.add_argument(
        "--decoder",
        default="pixel_head",
        choices=("pixel_head", "vae"),
        help="latent → pixels; each has its own render dir (vae: the 11-25 run's)",
    )
    a = p.parse_args()
    parts = a.parts.split(",")
    name = MODEL if a.decoder == "vae" else f"{MODEL}_px"
    root = OUT / (f"_smoke_{a.decoder}" if a.smoke else name)
    js = jobs(parts, a.smoke)
    info = {}
    if not a.tabulate:
        info = render(js, root, a.decoder)
        print(f"render: {info}", flush=True)
    reads = read(js, root, gpu=not a.tabulate)
    from bench._common import make_run_dir, write_result

    metrics, align = {"render": info}, {}
    if "ruler" in parts:
        metrics["ruler"], align["ruler"] = ruler_tables(js, reads)
        for arm, v in metrics["ruler"].items():
            print(f"  {arm:<18} {v['per_seed']}", flush=True)
    if "kanji" in parts:
        metrics["kanji"], align["kanji"] = kanji_tables(js, reads)
        for k, v in metrics["kanji"].items():
            print(f"  {k:<36} {v}", flush=True)
    run_dir = make_run_dir(
        "cjk_rl", label=f"turbo_read-{a.label}", root=LINE / "results"
    )
    sheets(js, reads, root, run_dir / "sheets")
    (run_dir / "alignments.json").write_text(
        json.dumps(align, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    write_result(
        run_dir,
        script="project/cjk_rl/probes/turbo_read.py",
        args=vars(a) | {"model": MODEL, "seeds": SEEDS, "renders": str(root)},
        label=a.label,
        metrics=metrics,
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
