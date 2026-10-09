#!/usr/bin/env python
"""sigma_split — the seed's rows gated by σ, no training (idea.md § 2b check 1 / § 3)

``idea.md``'s question before any data is built: can the rows act **below** σ
``--switch`` (default 0.5) on a layout the base laid out above it? The
sampler's commitment-σ switch (``generate_body``'s ``context_alt`` +
``tag_drop_sigma``: the conditional pass uses ``context`` while σ ≥ switch,
``context_alt`` below; CFG's negative pass untouched) does it with no new
sampler. The row Δ is added at encode (``ExtDelta`` on ``llm_adapter.embed``),
so each side is the same caption encoded with Δ scale 1 (**seed**) or 0
(**raw** = the pack rows exactly).

Arms, JA caption = the ``sent`` ruler's ``en`` clause
(``…, japanese text. Japanese text reads as "<k>".``):

- ``lo``  raw above, seed below — the product condition (§ 3 b): the base lays
  out with untrained pack rows, the rows only speak at σ < switch;
- ``hi``  seed above, raw below — the mirror;
- ``garble``  the garble caption above (``…, japanese text. She is saying
  something.``: no quote, so the base draws its own pseudo-Japanese — the
  caption idea.md's data would be rendered from), the JA caption with the
  seed rows below — the forced scaffold (§ 3 a): can the rows overwrite the
  base's own garble line?

Seed rows on both sides = the floor, read from the cache of record
(``seed_retrain_0930/routed/native_sent/``, the ``retrain_read`` grid: 4
prompts × 2 seeds × 23 strings = 184). ``--traj [--rows <arm>] [--traj_conds …] [--traj_sigmas …]`` decodes x̂0 per σ
(``--rows``: an arm's rows in place of the seed's, into ``traj_<arm>/``).
``--check`` renders one floor key
through the split path with seed on both sides and diffs it against the
cached file (the plumbing check). The seed trained singles at 0.7–0.9, so
``lo`` is a lower bound on what rows trained at 0.3–0.5 could do.

Reads per arm: official / ≤ 1 edit / dup (``cjk_scale.reads``, paired vs the
floor), EN-ref PE cos, and idea.md § 1's placement measures (``box`` = union
of the non-whole read boxes / canvas, ``box_h`` = tallest box / H,
``flat_white`` = share of 16² patches with std < 6 and mean > 225). One
sheet per string: EN ref | floor | lo | hi | garble.

    ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack \\
      make daemon-run ARGS="--label sigma_split \\
      project/cjk_anima_scale/experiments/sigma_split/run_exp.py --label s05"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["ANIMA_VOCAB_GLYPH_ROUTE"] = "1"  # the seed trained routed, reads routed
os.environ.setdefault("ANIMA_VOCAB_PACK", "models/vocab_packs/anima_cjk_vocab_pack")

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.builder import tier_of  # noqa: E402
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

FLOOR = SEED_ROWS.parent / "routed"  # the seed's routed floor cache
FLOOR_READS = FLOOR / "native_sent" / "native_reads.json"
PROMPTS = 4  # retrain_read's grid: the first 4 scene prompts × 2 seeds
CLAUSE = "en"
GARBLE = "{p}, japanese text. She is saying something."  # every grid prompt has a girl
EN_CAPTION = '{p}, english text. English text reads as "{k}".'  # enref_caption's shape
SIZE, STEPS, CFG = 512, 28, 4.0  # the sent ruler's (cjk_scale.eval)
ENREF = OUT / "native_enref" / f"{SIZE}_{STEPS}_{CFG:g}"
# arm → (above the switch, below it); "seed" / "raw" = JA caption at Δ 1 / 0,
# "garble" = GARBLE (no ext row: Δ irrelevant)
ARMS = {
    "lo": ("raw", "seed"),
    "hi": ("seed", "raw"),
    "garble": ("garble", "seed"),
    # below the switch, no string at all: "unk" = the JA caption under the
    # stock T5 tokenizer (the base with no pack: the quote is <unk>), "uncond"
    # = the negative embedding as the conditional (CFG collapses to uncond)
    "unk": ("seed", "unk"),
    "uncond": ("seed", "uncond"),
}


def floor_items() -> list[dict]:
    recs = json.loads(FLOOR_READS.read_text("utf-8"))
    items = [
        {k: m[k] for k in ("pi", "prompt", "text", "clause", "caption", "seed")}
        | {"floor_file": m["file"]}
        for m in recs
        if m["pi"] < PROMPTS and m["clause"] == CLAUSE
    ]
    assert len(items) == 184, f"{FLOOR_READS}: {len(items)} floor keys, not 184"
    return items


def arm_dir(switch: float, arm: str) -> Path:
    return OUT / "experiments" / f"sigma_split_s{switch:g}" / arm


def out_file(switch: float, arm: str, it: dict) -> Path:
    return (
        arm_dir(switch, arm)
        / "img"
        / f"{arm}_p{it['pi']:02d}_{it['text']}_{it['clause']}_s{it['seed']}.png"
    )


class Splitter:
    """The DiT + the seed Δ, rendering a caption pair split at σ."""

    def __init__(self, rows_dir: Path = SEED_ROWS.parent):
        import torch

        from common.hooks import ExtDelta
        from common.models import load_generator, load_trained, load_vae

        self.args, self.gen, self.device, self.shared = load_generator(
            SIZE, STEPS, CFG, OUT / "experiments" / "sigma_split_tmp"
        )
        self.anima = self.shared["model"]
        self.anima.eval()
        sd = load_trained(rows_dir)
        self.delta = ExtDelta.from_state(self.anima, sd["delta"], self.device)
        self.vae = load_vae(self.device)
        self.caches = {"seed": {}, "raw": {}, "unk": {}}  # conds_cache per side
        self.torch = torch

    def encode(self, caption: str, side: str):
        from library.inference.text import prepare_text_inputs

        from library.anima import vocab_pack as VP

        self.delta.scale = 0.0 if side in ("raw", "unk") else 1.0
        self.shared["conds_cache"] = self.caches[
            side if side in self.caches else "seed"
        ]
        a2 = self._args(caption, 0)
        if side != "unk":
            return prepare_text_inputs(a2, self.device, self.anima, self.shared)
        tok = VP.VocabPackTokenizeStrategy.tokenize  # stock T5 ids: no pack routing
        VP.VocabPackTokenizeStrategy.tokenize = VP.AnimaTokenizeStrategy.tokenize
        try:
            return prepare_text_inputs(a2, self.device, self.anima, self.shared)
        finally:
            VP.VocabPackTokenizeStrategy.tokenize = tok

    def _args(self, caption: str, seed: int):
        import copy

        a2 = copy.deepcopy(self.args)
        a2.prompt, a2.seed = caption, seed
        return a2

    def render(
        self,
        fn: Path,
        it: dict,
        above: str,
        below: str,
        switch: float,
        x0s: dict | None = None,
    ):
        """``x0s``: filled with ``{step: (σ, x̂0 latent on CPU, x_t on CPU)}`` —
        the (CFG-combined) prediction ``x_t − σ·v`` the Euler step is taken
        from, and the input it was taken on."""
        from common.models import decode_image
        from library.inference import generation as G
        from library.inference import sampling as S

        if fn.exists() and x0s is None:
            return
        cap = lambda side: (  # noqa: E731
            it.get("garble") or GARBLE.format(p=it["prompt"])
            if side == "garble"
            else EN_CAPTION.format(p=it["prompt"], k=it["en"])
            if side == "en"
            else it["caption"]
        )
        hi, null = self.encode(cap(above), above)
        lo = null if below == "uncond" else self.encode(cap(below), below)[0]
        body, step = G.generate_body, S.step
        G.generate_body = lambda *x, **k: body(
            *x, context_alt=lo, tag_drop_sigma=switch, **k
        )
        if x0s is not None:

            def rec(latents, noise_pred, sigmas, i):
                s = float(sigmas[i])
                x0s[i] = (
                    s,
                    (latents.float() - s * noise_pred.float()).cpu(),
                    latents.float().cpu(),
                )
                return step(latents, noise_pred, sigmas, i)

            S.step = rec
        a2 = self._args(it["caption"], it["seed"])
        if "shape" in it:  # an item's own (W, H)
            a2.image_size = (it["shape"][1], it["shape"][0])
        try:
            with self.torch.no_grad():
                lat = G.generate(
                    a2,
                    self.gen,
                    self.shared,
                    precomputed_text_data={"context": hi, "context_null": null},
                )
        finally:
            G.generate_body, S.step = body, step
        fn.parent.mkdir(parents=True, exist_ok=True)
        decode_image(self.vae, lat, self.device).save(fn)

    def decode(self, lat, fn: Path) -> None:
        from common.models import decode_image

        decode_image(self.vae, lat, self.device).save(fn)

    def free(self):
        del self.anima, self.vae, self.shared, self.delta
        self.torch.cuda.empty_cache()


def check(sp: Splitter, it: dict, switch: float) -> float:
    """Seed on both sides through the split path vs the cached floor render."""
    import numpy as np
    from PIL import Image

    fn = OUT / "experiments" / "sigma_split_tmp" / "check.png"
    fn.unlink(missing_ok=True)
    sp.render(fn, it, "seed", "seed", switch)
    a = np.asarray(Image.open(fn), np.float32)
    b = np.asarray(Image.open(it["floor_file"]), np.float32)
    d = float(np.abs(a - b).mean())
    print(f"check {Path(it['floor_file']).name}: mean |Δpx| {d:.3f}", flush=True)
    return d


def placement(m: dict) -> dict:
    """idea.md § 1: box (union of non-whole boxes / canvas), box_h (tallest /
    H), flat_white (16² patches, std < 6 and mean > 225)."""
    import numpy as np
    from PIL import Image

    im = np.asarray(Image.open(m["file"]).convert("L"), np.float32)
    H, W = im.shape
    mask = np.zeros((H, W), bool)
    hs = [0]
    for r in m.get("reads", []):
        if r.get("whole") or not r.get("box"):
            continue
        x0, y0, x1, y1 = (int(v) for v in r["box"])
        mask[max(0, y0) : y1, max(0, x0) : x1] = True
        hs.append(y1 - y0)
    p = im[: H // 16 * 16, : W // 16 * 16].reshape(H // 16, 16, W // 16, 16)
    std, mean = p.std(axis=(1, 3)), p.mean(axis=(1, 3))
    return {
        "box": float(mask.mean()),
        "box_h": max(hs) / H,
        "flat_white": float(((std < 6) & (mean > 225)).mean()),
    }


def read_arm(manifest: list, out: Path, device) -> None:
    from common.readers import Readers, hit, read_scored
    from common.text import CJK_RE
    from eval.enref import EnRef, enref_boxes

    rd = Readers(device)
    for m in manifest:
        reads = read_scored(rd, m)
        m["hit_sfx"], m["hit_vl"] = (
            hit(reads, m["text"], "sfx"),
            hit(reads, m["text"], "vl"),
        )
        m["exact"] = m["hit_sfx"] and m["hit_vl"]
        m["any_cjk"] = any(CJK_RE.search(r["sfx"] or "") for r in reads)
    enref = EnRef(device, ENREF, enref_boxes(ENREF, rd, device))
    del rd
    for m in manifest:
        sc = enref.score(m)
        m["en_cos"], m["en_cos_out"], m["box_iou"] = sc if sc else (None,) * 3
    (out / "native_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def sheets(items: list, arms: list, reads: dict, out: Path) -> None:
    """One sheet per string: rows = (prompt, seed), cols = EN ref | floor | arms."""
    from PIL import Image

    from common.readers import contact_sheet
    from eval.enref import enref_file

    out.mkdir(parents=True, exist_ok=True)
    by_key = {
        name: {(m["text"], m["pi"], m["seed"]): m for m in recs}
        for name, recs in reads.items()
    }
    for text in dict.fromkeys(it["text"] for it in items):
        rows = []
        for it in (i for i in items if i["text"] == text):
            k = (text, it["pi"], it["seed"])
            ref = enref_file(ENREF, it["pi"], it["seed"])
            rows.append((Image.open(ref).convert("RGB"), [f"EN ref p{k[1]} s{k[2]}"]))
            for name in ("floor", *arms):
                m = by_key[name].get(k)
                if m is None:
                    continue
                r0 = (m.get("reads") or [{}])[0]
                rows.append(
                    (
                        Image.open(m["file"]).convert("RGB"),
                        [
                            f"{name}{' ✓' if m.get('exact') else ''}",
                            f"sfx {(r0.get('sfx') or '')[:14]}",
                            f"vl {(r0.get('vl') or '')[:14]}",
                        ],
                    )
                )
        contact_sheet(rows, out / f"sheet_{text}.png", thumb=224, cols=2 + len(arms))


def summarize(
    items: list, arms: list, switch: float, label: str, check_d: float | None
) -> None:
    import statistics as st

    from cjk_scale import reads as R

    chars = sorted({it["text"] for it in items})
    keys = {(it["text"], it["clause"], it["pi"], it["seed"]) for it in items}
    floor_h = {k: v for k, v in R.hits(FLOOR_READS, chars, CLAUSE).items() if k in keys}
    recs = {
        "floor": [
            m
            for m in json.loads(FLOOR_READS.read_text("utf-8"))
            if (m["text"], m["clause"], m["pi"], m["seed"]) in keys
        ]
    }
    metrics: dict = {
        "switch": switch,
        "check_mean_abs_px": check_d,
        "floor": str(FLOOR_READS),
    }
    print("===== floor", flush=True)
    metrics["floor_tally"] = R.tally(floor_h)
    for arm in arms:
        f = arm_dir(switch, arm) / "native_reads.json"
        recs[arm] = json.loads(f.read_text("utf-8"))
        h = R.hits(f, chars, CLAUSE)
        print(f"===== {arm} {ARMS[arm]} @ σ {switch}", flush=True)
        metrics[arm] = {"tally": R.tally(h), "vs_floor": R.paired(h, floor_h)}
        print(f"  vs floor {metrics[arm]['vs_floor']}", flush=True)
    place = {}
    for name, ms in recs.items():
        ps = [placement(m) for m in ms]
        place[name] = {
            k: round(st.mean(p[k] for p in ps), 4)
            for k in ("box", "box_h", "flat_white")
        } | {
            k: round(st.mean(m[k] for m in ms if m.get(k) is not None), 4)
            for k in ("en_cos", "en_cos_out", "box_iou")
        }
        print(f"  placement {name:<6} {place[name]}", flush=True)
    metrics["placement"] = place
    sheet_dir = OUT / "experiments" / f"sigma_split_s{switch:g}" / "sheets"
    sheets(items, arms, recs, sheet_dir)
    run_dir = make_run_dir(
        "sigma_split",
        label=label,
        root=LINE / "experiments" / "sigma_split" / "results",
    )
    write_result(
        run_dir,
        script=__file__,
        args=argparse.Namespace(label=label, switch=switch),
        label=label,
        metrics=metrics,
        artifacts=[str(arm_dir(switch, a)) for a in arms] + [str(sheet_dir)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


# --traj: x̂0 per σ on a few samples — when layout / garble / identity appear
TRAJ_SIGMAS = (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.0)  # 0 = the final image
TRAJ_TEXTS = {"こんにちは": "hello", "やったネ": "we did it"}  # JA → its EN pair
TRAJ_SEED = 0
# condition → (above the switch, below it)
TRAJ = {
    "ja_seed": ("seed", "seed"),  # the floor condition
    "ja_raw": ("raw", "raw"),  # the untrained pack rows throughout
    "lo": ("raw", "seed"),
    "garble_ja": ("garble", "seed"),
    "garble": ("garble", "garble"),  # the base's own line, no quote
    "en": ("en", "en"),  # EN identity: TRAJ_TEXTS' EN side
}


def rows_dir(rows: str) -> Path:
    """``""`` / ``seed`` = the seed rows, else an arm under ``OUT/experiments``."""
    return SEED_ROWS.parent if rows in ("", "seed") else OUT / "experiments" / rows


def traj_dir(switch: float, rows: str = "") -> Path:
    """``rows``: ``seed`` or an arm under ``OUT/experiments`` (``rows_dir``);
    ``""`` = the record's own dir."""
    d = OUT / "experiments" / f"sigma_split_s{switch:g}"
    return d / (f"traj_{rows}" if rows else "traj")


def traj(
    label: str,
    switch: float,
    rows_arm: str = "",
    conds: tuple = tuple(TRAJ),
    sigmas: tuple = TRAJ_SIGMAS,
) -> None:
    """Render each TRAJ condition on the 4 grid prompts × TRAJ_TEXTS (seed
    ``TRAJ_SEED``; ``garble`` once per prompt: its caption has no string),
    decode x̂0 at the step nearest each of ``TRAJ_SIGMAS``, read every decode,
    one sheet per condition (rows = samples, cols = σ)."""
    import statistics as st

    from common.readers import Readers, contact_sheet, hit, read_scored
    from common.text import lev, norm
    from PIL import Image

    items = {
        (it["text"], it["pi"]): it
        for it in floor_items()
        if it["text"] in TRAJ_TEXTS and it["seed"] == TRAJ_SEED
    }
    root = traj_dir(switch, rows_arm)
    sp = Splitter(rows_dir(rows_arm))
    manifest = []
    for cond, (above, below) in ((c, TRAJ[c]) for c in conds):
        for (text, pi), it in sorted(
            items.items(), key=lambda kv: (kv[0][1], kv[0][0])
        ):
            if cond == "garble" and text != next(iter(TRAJ_TEXTS)):
                continue
            it = it | {"en": TRAJ_TEXTS[text]}
            d = root / cond / f"p{pi:02d}_{text}_s{TRAJ_SEED}"
            d.mkdir(parents=True, exist_ok=True)
            x0s: dict = {}
            sp.render(d / "sig0.00.png", it, above, below, switch, x0s)
            steps = sorted(x0s)
            for target in sigmas:
                if target == 0.0:
                    fn, s = d / "sig0.00.png", 0.0
                else:
                    i = min(steps, key=lambda j: abs(x0s[j][0] - target))
                    s = x0s[i][0]
                    fn = d / f"sig{s:.2f}.png"
                    sp.decode(x0s[i][1], fn)
                target_text = (
                    it["en"] if cond == "en" else None if cond == "garble" else text
                )
                manifest.append(
                    {
                        "file": str(fn),
                        "cond": cond,
                        "pi": pi,
                        "text": text,
                        "target": target_text,
                        "sigma_target": target,
                        "sigma": s,
                    }
                )
            print(
                f"  traj {cond} p{pi} {text}: σ {[round(x0s[j][0], 2) for j in steps[:1]]}…",
                flush=True,
            )
    device = sp.device
    sp.free()
    rd = Readers(device)
    for m in manifest:
        reads = read_scored(rd, m | {"text": m["target"] or ""})
        m["reads"] = reads
        t = m["target"]
        if t:
            reader = ("vl",) if m["cond"] == "en" else ("sfx", "vl")
            m["hit"] = all(hit(reads, t, r) for r in reader)
            m["best_edit"] = min(
                (
                    lev(norm(r.get(x) or ""), norm(t))
                    for r in reads
                    for x in reader
                    if r.get(x)
                ),
                default=len(t),
            )
        m.update(placement(m))
    (root / "traj_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    metrics: dict = {
        "switch": switch,
        "rows": rows_arm,
        "sigmas": sigmas,
        "per_cond": {},
    }
    for cond in conds:
        per = {}
        for target in sigmas:
            ms = [
                m for m in manifest if m["cond"] == cond and m["sigma_target"] == target
            ]
            per[f"{target:g}"] = {
                "n": len(ms),
                "sigma": round(ms[0]["sigma"], 3),
                "hit": sum(bool(m.get("hit")) for m in ms),
                "le1": sum(m.get("best_edit", 99) <= 1 for m in ms),
                "box": round(st.mean(m["box"] for m in ms), 4),
                "box_h": round(st.mean(m["box_h"] for m in ms), 4),
            }
        metrics["per_cond"][cond] = per
        print(f"===== traj {cond} {TRAJ[cond]}", flush=True)
        for k, v in per.items():
            print(
                f"  σ {k:>3} (step σ {v['sigma']}): hit {v['hit']}/{v['n']}  ≤1 {v['le1']}  "
                f"box {v['box']}  box_h {v['box_h']}",
                flush=True,
            )
        rows = []
        for m in (m for m in manifest if m["cond"] == cond):
            r0 = max(
                m["reads"] or [{}], key=lambda r: len(r.get("vl") or ""), default={}
            )
            rows.append(
                (
                    Image.open(m["file"]).convert("RGB"),
                    [
                        f"p{m['pi']} {m['target'] or 'garble'} σ {m['sigma']:.2f}{' ✓' if m.get('hit') else ''}",
                        f"sfx {(r0.get('sfx') or '')[:14]}",
                        f"vl {(r0.get('vl') or '')[:14]}",
                    ],
                )
            )
        contact_sheet(
            rows, root / f"sheet_traj_{cond}.png", thumb=224, cols=len(sigmas)
        )
    run_dir = make_run_dir(
        "sigma_split",
        label=label,
        root=LINE / "experiments" / "sigma_split" / "results",
    )
    write_result(
        run_dir,
        script=__file__,
        args=argparse.Namespace(
            label=label, switch=switch, traj=True, rows=rows_arm, conds=conds
        ),
        label=label,
        metrics=metrics,
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


# --span: proposal_length.md step 0 — does the caption's glyph count reach the
# text region's span at σ 0.95? Seed rows both sides; the same (prompt, seed)
# under captions of 2 / 4 / 5 / 5 / 6 glyphs. The OCR box reads 0 at σ 0.95 in
# every traj run (nothing boxes a blur), so the span is read as where the
# captions' x̂0 differ (`chg` = share of 16² patches with mean |Δpx| > CHG_PX,
# `chg_box` = their bounding box / canvas) beside the OCR box, and on the
# sheets. x_t at the step nearest σ 0.85 is kept for the count read.
SPAN_TEXTS = (
    "はい",
    "やったネ",
    "やったネネ",
    "やったネ！",
    "こんにちは",
    "こんにちはは",
    "こんにちは！",
)
# caption from; ！ folds to base T5 (no row) — the leftover slot given to the base
SPAN_BASE = {
    "やったネネ": "やったネ",
    "やったネ！": "やったネ",
    "こんにちはは": "こんにちは",
    "こんにちは！": "こんにちは",
}
SPAN_SIGMAS = (0.95, 0.9, 0.85, 0.0)
SPAN_PAIRS = (  # (a, b): the glyph-count contrasts, then a same-span control
    ("こんにちは", "こんにちはは"),
    ("やったネ", "やったネネ"),
    ("はい", "こんにちは"),
    ("はい", "こんにちはは"),
    ("やったネ", "こんにちは"),
    ("こんにちは！", "こんにちはは"),
    ("やったネ！", "やったネネ"),
)
CHG_PX = 12.0
XT_SIGMA = 0.85


# caption variants of the span leg: `ja` = the sent ruler's `en` clause;
# `notag` drops the `japanese text` tag and the language word from the clause
SPAN_CAPTIONS = {
    "ja": lambda c: c,
    "notag": lambda c: c.replace(
        ", japanese text. Japanese text reads as", ". Text reads as"
    ),
}


def span_items(variant: str = "ja") -> list[dict]:
    base = {
        (it["text"], it["pi"], it["seed"]): it
        for it in floor_items()
        if it["text"] in SPAN_TEXTS
    }
    out = []
    for text in SPAN_TEXTS:
        src = SPAN_BASE.get(text, text)
        for (t, pi, seed), it in sorted(base.items(), key=lambda kv: kv[0][1:]):
            if t != src:
                continue
            cap = it["caption"]
            assert f'"{src}"' in cap, cap
            cap = SPAN_CAPTIONS[variant](cap.replace(f'"{src}"', f'"{text}"'))
            assert variant == "ja" or "apanese" not in cap, cap
            out.append(it | {"text": text, "caption": cap})
    assert len(out) == len(SPAN_TEXTS) * PROMPTS * 2, len(out)
    return out


def pixel_diff(fa: str, fb: str) -> dict:
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(fa).convert("L"), np.float32)
    b = np.asarray(Image.open(fb).convert("L"), np.float32)
    d = np.abs(a - b)
    H, W = d.shape
    p = (
        d[: H // 16 * 16, : W // 16 * 16]
        .reshape(H // 16, 16, W // 16, 16)
        .mean(axis=(1, 3))
    )
    on = p > CHG_PX
    if on.any():
        ys, xs = np.nonzero(on)
        box = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) / on.size
    else:
        box = 0.0
    return {"dpx": float(d.mean()), "chg": float(on.mean()), "chg_box": float(box)}


def span(label: str, variant: str = "ja", sigmas: tuple = SPAN_SIGMAS) -> None:
    import statistics as st

    import torch
    from PIL import Image

    from common.readers import Readers, contact_sheet, hit, read_scored
    from common.text import lev, norm

    items = span_items(variant)
    root = (
        OUT
        / "experiments"
        / ("sigma_split_span" + ("" if variant == "ja" else f"_{variant}"))
    )
    sp = Splitter()
    manifest, xts = [], {}
    for it in items:
        key = f"p{it['pi']:02d}_s{it['seed']}"
        d = root / it["text"] / key
        d.mkdir(parents=True, exist_ok=True)
        x0s: dict = {}
        done = all((d / f"sig{t:.2f}.png").exists() for t in sigmas)
        if not done:  # a rerun with added captions renders only the new ones
            sp.render(d / "sig0.00.png", it, "seed", "seed", 1.0, x0s)
        steps = sorted(x0s)
        for target in sigmas:
            fn = d / f"sig{target:.2f}.png"
            if done or target == 0.0:
                s = target
            else:
                i = min(steps, key=lambda j: abs(x0s[j][0] - target))
                s = x0s[i][0]
                sp.decode(x0s[i][1], fn)
            manifest.append(
                {
                    "file": str(fn),
                    "text": it["text"],
                    "glyphs": len(it["text"]),
                    "pi": it["pi"],
                    "seed": it["seed"],
                    "caption": it["caption"],
                    "sigma_target": target,
                    "sigma": s,
                }
            )
        if not done:
            i = min(steps, key=lambda j: abs(x0s[j][0] - XT_SIGMA))
            xts[(it["text"], key)] = {"sigma": x0s[i][0], "x_t": x0s[i][2]}
        print(f"  span {it['text']} {key}{' (cached)' if done else ''}", flush=True)
    xt_file = root / f"xt_{XT_SIGMA:g}.pt"
    if xt_file.exists():
        xts = torch.load(xt_file) | xts
    torch.save(xts, xt_file)
    device = sp.device
    sp.free()
    rd = Readers(device)
    for m in manifest:
        reads = read_scored(rd, m)
        m["reads"] = reads
        m["hit"] = all(hit(reads, m["text"], r) for r in ("sfx", "vl"))
        m["best_edit"] = min(
            (
                lev(norm(r.get(x) or ""), norm(m["text"]))
                for r in reads
                for x in ("sfx", "vl")
                if r.get(x)
            ),
            default=len(m["text"]),
        )
        m.update(placement(m))
    by = {(m["text"], m["pi"], m["seed"], m["sigma_target"]): m for m in manifest}
    diffs = []
    for a, b in SPAN_PAIRS:
        for (t, pi, seed, sig), m in by.items():
            if t == a:
                diffs.append(
                    {"a": a, "b": b, "pi": pi, "seed": seed, "sigma_target": sig}
                    | pixel_diff(m["file"], by[(b, pi, seed, sig)]["file"])
                )
    (root / "span_reads.json").write_text(
        json.dumps(
            {"manifest": manifest, "diffs": diffs}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )
    metrics: dict = {"sigmas": sigmas, "chg_px": CHG_PX, "per_text": {}, "pairs": {}}
    print("===== span: per caption (OCR placement, reads)", flush=True)
    for text in SPAN_TEXTS:
        per = {}
        for target in sigmas:
            ms = [
                m for m in manifest if m["text"] == text and m["sigma_target"] == target
            ]
            per[f"{target:g}"] = {
                "n": len(ms),
                "hit": sum(m["hit"] for m in ms),
                "le1": sum(m["best_edit"] <= 1 for m in ms),
                "box": round(st.mean(m["box"] for m in ms), 4),
                "box_h": round(st.mean(m["box_h"] for m in ms), 4),
            }
        metrics["per_text"][text] = per
        print(
            f"  {text} ({len(text)}): "
            + "  ".join(
                f"σ{k} hit {v['hit']}/{v['n']} ≤1 {v['le1']} box {v['box']} h {v['box_h']}"
                for k, v in per.items()
            ),
            flush=True,
        )
    print("===== span: caption pairs, x̂0 pixel diff (mean over 8 samples)", flush=True)
    for a, b in SPAN_PAIRS:
        per = {}
        for target in sigmas:
            ds = [
                x
                for x in diffs
                if x["a"] == a and x["b"] == b and x["sigma_target"] == target
            ]
            per[f"{target:g}"] = {
                k: round(st.mean(x[k] for x in ds), 4)
                for k in ("dpx", "chg", "chg_box")
            }
        metrics["pairs"][f"{a}|{b}"] = per
        print(
            f"  {a} vs {b}: "
            + "  ".join(
                f"σ{k} dpx {v['dpx']:.2f} chg {v['chg']:.3f} box {v['chg_box']:.3f}"
                for k, v in per.items()
            ),
            flush=True,
        )
    # one sheet per (prompt, seed): rows = captions, cols = σ
    for pi in range(PROMPTS):
        for seed in (0, 1):
            rows = []
            for text in SPAN_TEXTS:
                for target in sigmas:
                    m = by[(text, pi, seed, target)]
                    r0 = max(
                        m["reads"] or [{}],
                        key=lambda r: len(r.get("vl") or ""),
                        default={},
                    )
                    rows.append(
                        (
                            Image.open(m["file"]).convert("RGB"),
                            [
                                f"{text} σ {m['sigma']:.2f}{' ✓' if m['hit'] else ''}",
                                f"sfx {(r0.get('sfx') or '')[:14]}",
                                f"vl {(r0.get('vl') or '')[:14]}",
                            ],
                        )
                    )
            contact_sheet(
                rows, root / f"sheet_p{pi}_s{seed}.png", thumb=224, cols=len(sigmas)
            )
    run_dir = make_run_dir(
        "sigma_split",
        label=label,
        root=LINE / "experiments" / "sigma_split" / "results",
    )
    write_result(
        run_dir,
        script=__file__,
        args=argparse.Namespace(
            label=label, span=True, variant=variant, texts=SPAN_TEXTS
        ),
        label=label,
        metrics=metrics,
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


# --en_small: the base alone (row Δ 0) writing EN at three lengths in a speech
# bubble — longer strings come out smaller. x̂0 per σ, read by VL: when does
# small text become legible on the trajectory, and from which σ does it stop
# changing? (cf_sense's 12–16 px ceiling 0.2–0.6 was read teacher-forced, on
# a noised clean render; this is the generation trajectory.)
EN_SMALL = {
    "hello": "hello",
    "s30": "I think it's going to rain today.",
    "s75": "I told you we should have left before midnight, but you never listen to me.",
}
EN_SMALL_CAPTION = '{p}, speech bubble, english text. English text reads as "{s}".'
EN_SMALL_SIGMAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0)


def _words(s: str) -> list[str]:
    import re

    return re.findall(r"[a-z']+", s.casefold().replace("’", "'"))


def en_small(label: str, sigmas: tuple = EN_SMALL_SIGMAS) -> None:
    import statistics as st

    from PIL import Image

    from common.readers import Readers, contact_sheet, read_scored

    keys = sorted({(it["pi"], it["prompt"], it["seed"]) for it in floor_items()})
    root = OUT / "experiments" / "sigma_split_en_small"
    sp = Splitter()
    manifest = []
    for cond, s in EN_SMALL.items():
        for pi, prompt, seed in keys:
            cap = EN_SMALL_CAPTION.format(p=prompt, s=s)
            it = {"pi": pi, "prompt": prompt, "seed": seed, "caption": cap}
            d = root / cond / f"p{pi:02d}_s{seed}"
            d.mkdir(parents=True, exist_ok=True)
            x0s: dict = {}
            sp.render(d / "sig0.00.png", it, "raw", "raw", 1.0, x0s)
            steps = sorted(x0s)
            for target in sigmas:
                fn = d / f"sig{target:.2f}.png"
                if target == 0.0:
                    sig = 0.0
                else:
                    i = min(steps, key=lambda j: abs(x0s[j][0] - target))
                    sig = x0s[i][0]
                    sp.decode(x0s[i][1], fn)
                manifest.append(
                    {
                        "file": str(fn),
                        "cond": cond,
                        "text": s,
                        "pi": pi,
                        "seed": seed,
                        "caption": cap,
                        "sigma_target": target,
                        "sigma": sig,
                    }
                )
            print(f"  en_small {cond} p{pi} s{seed}", flush=True)
    device = sp.device
    sp.free()
    rd = Readers(device)
    for m in manifest:
        reads = read_scored(rd, m)
        got = set()
        for r in reads:
            got.update(_words(r.get("vl") or ""))
        want = _words(m["text"])
        m["recall"] = sum(w in got for w in want) / len(want)
        m.update(placement(m))
    (root / "en_small_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    metrics: dict = {"sigmas": sigmas, "per_cond": {}}
    for cond in EN_SMALL:
        per = {}
        for target in sigmas:
            ms = [
                m for m in manifest if m["cond"] == cond and m["sigma_target"] == target
            ]
            per[f"{target:g}"] = {
                "n": len(ms),
                "sigma": round(ms[0]["sigma"], 3),
                "recall": round(st.mean(m["recall"] for m in ms), 3),
                "recall_ge_half": sum(m["recall"] >= 0.5 for m in ms),
                "cer_vl": round(st.mean(m["cer_vl"] for m in ms), 3),
                "box_h": round(st.mean(m["box_h"] for m in ms), 4),
            }
        metrics["per_cond"][cond] = per
        print(f"===== en_small {cond}: {EN_SMALL[cond]!r}", flush=True)
        for k, v in per.items():
            print(
                f"  σ {k:>4} (step σ {v['sigma']}): recall {v['recall']:.2f}  "
                f"≥½ {v['recall_ge_half']}/{v['n']}  cer_vl {v['cer_vl']:.2f}  "
                f"box_h {v['box_h']}",
                flush=True,
            )
        rows = []
        for m in (m for m in manifest if m["cond"] == cond):
            r0 = max(
                m["reads"] or [{}], key=lambda r: len(r.get("vl") or ""), default={}
            )
            rows.append(
                (
                    Image.open(m["file"]).convert("RGB"),
                    [
                        f"p{m['pi']} s{m['seed']} σ {m['sigma']:.2f}",
                        f"recall {m['recall']:.2f}",
                        f"vl {(r0.get('vl') or '')[:24]}",
                    ],
                )
            )
        contact_sheet(rows, root / f"sheet_{cond}.png", thumb=200, cols=len(sigmas))
    run_dir = make_run_dir(
        "sigma_split",
        label=label,
        root=LINE / "experiments" / "sigma_split" / "results",
    )
    write_result(
        run_dir,
        script=__file__,
        args=argparse.Namespace(label=label, en_small=True, texts=EN_SMALL),
        label=label,
        metrics=metrics,
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


# --en_grid: the base alone (row Δ 0) writing nine EN words in the 3 × 3 flat
# grid (the grid items' frame + one position clause per cell, the caption the
# base binds clause → cell on). x̂0 per σ down to 0.5: is the word decided at
# 0.85–0.7 here too, with the cells laid out by the caption?
EN_GRID_POOL = (
    "HELLO STOP YES WAIT SORRY WHAT OK RUN HELP GO HEY CAT DOG MOON STAR FIRE "
    "RAIN BLUE BOOK CAKE TREE FISH LOVE HOME SNOW KING GOLD BIRD MILK DOOR "
    "NIGHT APPLE WATER HAPPY MUSIC DREAM"
).split()
EN_GRID_SETS, EN_GRID_SEEDS = 8, (0, 1)
EN_GRID_SIGMAS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.0)


def en_grid_items() -> list[dict]:
    import random

    from common.prompts import grid_caption

    rng = random.Random(0)
    items = []
    for gi in range(EN_GRID_SETS):
        words = rng.sample(EN_GRID_POOL, 9)
        cap = grid_caption("flat", 3, 3, words, lang="english")
        for seed in EN_GRID_SEEDS:
            items.append({"gi": gi, "seed": seed, "words": words, "caption": cap})
    return items


def en_grid(label: str, sigmas: tuple = EN_GRID_SIGMAS) -> None:
    import statistics as st

    from PIL import Image

    from common.readers import Readers, contact_sheet, read_scored

    root = OUT / "experiments" / "sigma_split_en_grid"
    sp = Splitter()
    manifest = []
    for it in en_grid_items():
        d = root / f"g{it['gi']}_s{it['seed']}"
        d.mkdir(parents=True, exist_ok=True)
        x0s: dict = {}
        sp.render(d / "sig0.00.png", it, "raw", "raw", 1.0, x0s)
        steps = sorted(x0s)
        for target in sigmas:
            fn = d / f"sig{target:.2f}.png"
            if target == 0.0:
                sig = 0.0
            else:
                i = min(steps, key=lambda j: abs(x0s[j][0] - target))
                sig = x0s[i][0]
                sp.decode(x0s[i][1], fn)
            manifest.append(
                {
                    "file": str(fn),
                    "gi": it["gi"],
                    "seed": it["seed"],
                    "words": it["words"],
                    "text": " ".join(it["words"]),
                    "caption": it["caption"],
                    "sigma_target": target,
                    "sigma": sig,
                }
            )
        print(f"  en_grid g{it['gi']} s{it['seed']}", flush=True)
    device = sp.device
    sp.free()
    rd = Readers(device)
    for m in manifest:
        reads = read_scored(rd, m)
        got = set()
        for r in reads:
            got.update(_words(r.get("vl") or ""))
        want = [w.casefold() for w in m["words"]]
        m["hits"] = [w in got for w in want]
        m["recall"] = sum(m["hits"]) / len(want)
    (root / "en_grid_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    per = {}
    for target in sigmas:
        ms = [m for m in manifest if m["sigma_target"] == target]
        per[f"{target:g}"] = {
            "n": len(ms),
            "sigma": round(ms[0]["sigma"], 3),
            "recall": round(st.mean(m["recall"] for m in ms), 3),
            "all9": sum(m["recall"] == 1.0 for m in ms),
        }
    print("===== en_grid: 3×3 flat, 9 EN words", flush=True)
    for k, v in per.items():
        print(
            f"  σ {k:>4} (step σ {v['sigma']}): recall {v['recall']:.2f}  "
            f"all 9 {v['all9']}/{v['n']}",
            flush=True,
        )
    half = EN_GRID_SETS // 2
    for part, gis in (("a", range(half)), ("b", range(half, EN_GRID_SETS))):
        rows = []
        for m in (m for m in manifest if m["gi"] in gis):
            missed = [w for w, h in zip(m["words"], m["hits"]) if not h]
            rows.append(
                (
                    Image.open(m["file"]).convert("RGB"),
                    [
                        f"g{m['gi']} s{m['seed']} σ {m['sigma']:.2f}",
                        f"recall {m['recall']:.2f}",
                        f"miss {' '.join(missed)[:22]}",
                    ],
                )
            )
        contact_sheet(rows, root / f"sheet_{part}.png", thumb=200, cols=len(sigmas))
    run_dir = make_run_dir(
        "sigma_split",
        label=label,
        root=LINE / "experiments" / "sigma_split" / "results",
    )
    write_result(
        run_dir,
        script=__file__,
        args=argparse.Namespace(label=label, en_grid=True, pool=EN_GRID_POOL),
        label=label,
        metrics={"sigmas": sigmas, "per_sigma": per},
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


# --ja_grid: the seed rows on grid_single's own caption (b0709: 3 × 3, flat or
# bubble frame, one trained single per cell, 512²) — kana sets and kanji sets
# (retrain_kanji_b1's, the most frequent). Read per cell (the final image cut
# into its 3 × 3) and x̂0 per σ, as --en_grid.
JA_GRID_KANA = (
    "あいうえおかがきぎくぐけげこごさざしじすずせぜそぞただちぢつづてでとどなにぬねのはばぱ"
    "ひびぴふぶぷへべぺほぼぽまみむめもやゆよらりるれろわをん"
    "アイウエオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂツヅテデトドナニヌネノハバパ"
    "ヒビピフブプヘベペホボポマミムメモヤユヨラリルレロワヲン"
)
JA_GRID_SEED = 0
JA_GRID_SIGMAS = EN_GRID_SIGMAS


def ja_grid_items() -> list[dict]:
    import random

    from common.prompts import grid_caption

    kanji = [
        ln.split("\t")[0]
        for ln in (LINE / "assets/vocabs/ja_retrain_kanji_b1.txt")
        .read_text("utf-8")
        .splitlines()
        if ln and not ln.startswith("#")
    ]
    rng = random.Random(0)
    items = []
    for kind, pool in (("kana", list(JA_GRID_KANA)), ("kanji", kanji)):
        for gi in range(8):
            frame = ("flat", "bubble")[gi % 2]
            glyphs = rng.sample(pool, 9)
            items.append(
                {
                    "kind": kind,
                    "gi": gi,
                    "frame": frame,
                    "seed": JA_GRID_SEED,
                    "glyphs": glyphs,
                    "caption": grid_caption(frame, 3, 3, glyphs),
                }
            )
    return items


def ja_grid(
    label: str, below: str = "seed", switch: float = 1.0, sigmas=JA_GRID_SIGMAS
) -> None:
    """``below`` / ``switch``: the conditional below σ ``switch`` (``uncond`` =
    the negative embedding, as § 5's arm); the seed rows above it."""
    import numpy as np
    from PIL import Image

    from common.readers import Readers, contact_sheet, load_bgr

    base = OUT / "experiments" / "sigma_split_ja_grid"
    root = base if below == "seed" else base.with_name(f"{base.name}_{below}{switch:g}")
    sp = Splitter()
    manifest = []
    for it in ja_grid_items():
        d = root / f"{it['kind']}{it['gi']}_{it['frame']}"
        d.mkdir(parents=True, exist_ok=True)
        x0s: dict = {}
        sp.render(d / "sig0.00.png", it, "seed", below, switch, x0s)
        steps = sorted(x0s)
        for target in sigmas:
            fn = d / f"sig{target:.2f}.png"
            if target == 0.0:
                sig = 0.0
            else:
                i = min(steps, key=lambda j: abs(x0s[j][0] - target))
                sig = x0s[i][0]
                sp.decode(x0s[i][1], fn)
            manifest.append(
                {k: it[k] for k in ("kind", "gi", "frame", "seed", "glyphs", "caption")}
                | {"file": str(fn), "sigma_target": target, "sigma": sig}
            )
        print(f"  ja_grid {it['kind']}{it['gi']} {it['frame']}", flush=True)
    device = sp.device
    sp.free()
    rd = Readers(device)
    for m in manifest:
        bgr = load_bgr(Path(m["file"]))
        H, W = bgr.shape[:2]
        whole = rd.read_image(bgr, whole=True)
        text = "".join((r.get("sfx") or "") + (r.get("vl") or "") for r in whole)
        m["anywhere"] = [g in text for g in m["glyphs"]]
        cells = []
        for k, g in enumerate(m["glyphs"]):
            r, c = divmod(k, 3)
            crop = np.ascontiguousarray(
                bgr[r * H // 3 : (r + 1) * H // 3, c * W // 3 : (c + 1) * W // 3]
            )
            reads = rd.read_image(crop, whole=True)
            cells.append(
                {
                    "glyph": g,
                    "hit": any(
                        g in (x.get("sfx") or "") + (x.get("vl") or "") for x in reads
                    ),
                    "reads": [[x.get("sfx"), x.get("vl")] for x in reads],
                }
            )
        m["cells"] = cells
        m["cell_hits"] = sum(c["hit"] for c in cells)
        m["any_hits"] = sum(m["anywhere"])
    (root / "ja_grid_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    per = {}
    for kind in ("kana", "kanji"):
        for target in sigmas:
            ms = [
                m for m in manifest if m["kind"] == kind and m["sigma_target"] == target
            ]
            per[f"{kind} {target:g}"] = {
                "n": len(ms),
                "sigma": round(ms[0]["sigma"], 3),
                "cell": sum(m["cell_hits"] for m in ms),
                "anywhere": sum(m["any_hits"] for m in ms),
                "of": 9 * len(ms),
            }
    print("===== ja_grid: 3×3, one trained single per cell, seed rows", flush=True)
    for k, v in per.items():
        print(
            f"  {k:>11} (step σ {v['sigma']}): in cell {v['cell']}/{v['of']}  "
            f"anywhere {v['anywhere']}/{v['of']}",
            flush=True,
        )
    finals = []
    for kind in ("kana", "kanji"):
        rows = []
        for m in (m for m in manifest if m["kind"] == kind):
            rows.append(
                (
                    Image.open(m["file"]).convert("RGB"),
                    [
                        f"{kind}{m['gi']} {m['frame']} σ {m['sigma']:.2f}",
                        f"cell {m['cell_hits']}/9 any {m['any_hits']}/9",
                        "".join(m["glyphs"]),
                    ],
                )
            )
            if m["sigma_target"] == 0.0:
                finals.append(rows[-1])
        contact_sheet(rows, root / f"sheet_{kind}.png", thumb=200, cols=len(sigmas))
    contact_sheet(finals, root / "sheet_final.png", thumb=384, cols=4)
    if below != "seed":  # each final beside the seed-only render (same noise)
        pairs = []
        for m in (m for m in manifest if m["sigma_target"] == 0.0):
            ref = base / Path(m["file"]).parent.name / "sig0.00.png"
            name = f"{m['kind']}{m['gi']} {m['frame']}"
            if ref.exists():
                pairs.append((Image.open(ref).convert("RGB"), [f"{name} seed only"]))
            pairs.append(
                (
                    Image.open(m["file"]).convert("RGB"),
                    [f"{name} {below} < {switch:g}", f"cell {m['cell_hits']}/9"],
                )
            )
        contact_sheet(pairs, root / "sheet_vs_seed.png", thumb=300, cols=4)
    run_dir = make_run_dir(
        "sigma_split",
        label=label,
        root=LINE / "experiments" / "sigma_split" / "results",
    )
    write_result(
        run_dir,
        script=__file__,
        args=argparse.Namespace(label=label, ja_grid=True, below=below, switch=switch),
        label=label,
        metrics={"sigmas": sigmas, "per": per},
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


# --b0305: the seed's own b0305 training items (scene_window, 12–24 px dialogue
# windows at 0.3–0.5) as prompts — 4 from retrain_kana, 4 from
# retrain_kanji_b4, each at its own shape × 2 seeds; seed rows throughout vs
# seed rows above 0.8 and the negative embedding below (§ 5's `uncond`), and
# (10-02) the mirror: the base above 0.8, the seed rows below — the base as
# its own garble (the item's tags, no quote clause) or as the item's caption
# under the stock tokenizer (`unk`: no pack, the quote is <unk>).
B0305_RUNS = ("retrain_kana", "retrain_kanji_b4")
# arm → (above the switch, below it, switch)
B0305_ARMS = {
    "seed": ("seed", "seed", 1.0),
    "uncond0.8": ("seed", "uncond", 0.8),
    "garble0.8": ("garble", "seed", 0.8),
    "unk0.8": ("unk", "seed", 0.8),
}


def b0305_items() -> list[dict]:
    import random

    items = []
    for run in B0305_RUNS:
        recs = [
            json.loads(ln)
            for ln in (OUT / run / "data" / "train.jsonl").open(encoding="utf-8")
        ]
        recs = [r for r in recs if tier_of(r) == "bubbleN_18"]  # group b0305 of record
        for k, r in enumerate(random.Random(0).sample(recs, 4)):
            tags, clause = r["caption"].split(". ", 1)
            assert f'"{r["text"]}"' in clause, r["caption"]
            for seed in (0, 1):
                items.append(
                    {
                        "key": f"{run.split('_')[-1]}{k}",
                        "seed": seed,
                        "text": r["text"],
                        "caption": r["caption"],
                        "garble": tags + ".",
                        "shape": r["shape"],
                        "px": r["px"],
                        "train_file": r["file"],
                    }
                )
    return items


def b0305(label: str) -> None:
    from PIL import Image

    from common.readers import Readers, contact_sheet, hit, read_scored

    root = OUT / "experiments" / "sigma_split_b0305"
    items = b0305_items()
    sp = Splitter()
    manifest = []
    for it in items:
        for arm, (above, below, switch) in B0305_ARMS.items():
            fn = root / arm / f"{it['key']}_s{it['seed']}.png"
            sp.render(fn, it, above, below, switch)
            manifest.append(it | {"arm": arm, "file": str(fn)})
        print(f"  b0305 {it['key']} s{it['seed']} {it['text']}", flush=True)
    device = sp.device
    sp.free()
    rd = Readers(device)
    for m in manifest:
        reads = read_scored(rd, m)
        m["official"] = hit(reads, m["text"], "sfx") and hit(reads, m["text"], "vl")
        m["any_exact"] = hit(reads, m["text"], "sfx") or hit(reads, m["text"], "vl")
        m["vl"] = [r.get("vl") for r in reads if not r.get("whole")]
    (root / "b0305_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    metrics = {}
    for arm in B0305_ARMS:
        ms = [m for m in manifest if m["arm"] == arm]
        metrics[arm] = {
            "n": len(ms),
            "official": sum(m["official"] for m in ms),
            "any_exact": sum(m["any_exact"] for m in ms),
            "cer_vl": round(sum(m["cer_vl"] for m in ms) / len(ms), 3),
        }
        print(f"===== b0305 {arm}: {metrics[arm]}", flush=True)
    rows = []
    by = {(m["key"], m["seed"], m["arm"]): m for m in manifest}
    for it in items:
        tf = Path(it["train_file"])  # img/ may be pruned
        rows.append(
            (
                Image.open(tf).convert("RGB")
                if tf.exists()
                else Image.new("RGB", tuple(it["shape"]), "white"),
                [f"{it['key']} train item", it["text"], f"px {it['px']}"],
            )
        )
        for arm in B0305_ARMS:
            m = by[(it["key"], it["seed"], arm)]
            vl = max(m["vl"] or [""], key=lambda v: len(v or ""))
            rows.append(
                (
                    Image.open(m["file"]).convert("RGB"),
                    [
                        f"{it['key']} s{it['seed']} {arm}",
                        f"off {int(m['official'])} cer {m['cer_vl']:.2f}",
                        f"vl {(vl or '')[:20]}",
                    ],
                )
            )
    contact_sheet(rows, root / "sheet.png", thumb=300, cols=1 + len(B0305_ARMS))
    run_dir = make_run_dir(
        "sigma_split",
        label=label,
        root=LINE / "experiments" / "sigma_split" / "results",
    )
    write_result(
        run_dir,
        script=__file__,
        args=argparse.Namespace(label=label, b0305=True, arms=B0305_ARMS),
        label=label,
        metrics=metrics,
        artifacts=[str(root)],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--switch", type=float, default=0.5)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--traj", action="store_true", help="the x̂0-per-σ leg alone")
    ap.add_argument(
        "--rows",
        default="",
        help="traj: `seed` or an arm under OUT/experiments (its own traj_<rows> dir)",
    )
    ap.add_argument("--traj_conds", default=",".join(TRAJ))
    ap.add_argument("--traj_sigmas", nargs="+", type=float, default=list(TRAJ_SIGMAS))
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--span", action="store_true", help="proposal_length step 0: span vs caption"
    )
    ap.add_argument("--span_caption", default="ja", choices=tuple(SPAN_CAPTIONS))
    ap.add_argument(
        "--en_small", action="store_true", help="base EN text by length, x̂0 per σ"
    )
    ap.add_argument(
        "--en_grid", action="store_true", help="base EN 3×3 word grid, x̂0 per σ"
    )
    ap.add_argument(
        "--ja_grid", action="store_true", help="seed rows, 3×3 kana / kanji grid"
    )
    ap.add_argument(
        "--ja_below",
        default="seed",
        choices=("seed", "uncond", "raw", "unk"),
        help="ja_grid: the conditional below --switch",
    )
    ap.add_argument(
        "--b0305", action="store_true", help="b0305 training items: B0305_ARMS"
    )
    args = ap.parse_args()
    if args.b0305:
        items = b0305_items()
        print(f"b0305: {len(items)} renders × {len(B0305_ARMS)} arms", flush=True)
        for it in items[::2]:
            print(f"  {it['key']} {it['shape']} px {it['px']}: {it['caption'][-90:]}")
        if not args.dry_run:
            b0305(args.label)
        return
    if args.ja_grid:
        items = ja_grid_items()
        print(f"ja_grid: {len(items)} renders × {len(JA_GRID_SIGMAS)} σ", flush=True)
        for it in (items[0], items[9]):
            print(f"  {it['caption']}", flush=True)
        if not args.dry_run:
            ja_grid(
                args.label,
                args.ja_below,
                args.switch if args.ja_below != "seed" else 1.0,
            )
        return
    if args.en_grid:
        items = en_grid_items()
        print(f"en_grid: {len(items)} renders × {len(EN_GRID_SIGMAS)} σ", flush=True)
        print(f"  {items[0]['caption']}", flush=True)
        if not args.dry_run:
            en_grid(args.label)
        return
    if args.en_small:
        print(
            f"en_small: {len(EN_SMALL)} × 8 renders × {len(EN_SMALL_SIGMAS)} σ",
            flush=True,
        )
        if not args.dry_run:
            en_small(args.label)
        return
    if args.span:
        items = span_items(args.span_caption)
        print(
            f"span: {len(items)} renders × {len(SPAN_SIGMAS)} σ "
            f"({', '.join(SPAN_TEXTS)})",
            flush=True,
        )
        for it in items[:: PROMPTS * 2]:
            print(f"  {it['caption']}", flush=True)
        if not args.dry_run:
            span(args.label, args.span_caption)
        return
    if args.traj:
        conds = tuple(c for c in args.traj_conds.split(",") if c)
        assert set(conds) <= set(TRAJ), conds
        n = sum(1 if c == "garble" else len(TRAJ_TEXTS) for c in conds) * PROMPTS
        print(
            f"traj: {n} renders × {len(args.traj_sigmas)} σ → "
            f"{traj_dir(args.switch, args.rows)}",
            flush=True,
        )
        if not args.dry_run:
            traj(
                args.label,
                args.switch,
                args.rows,
                conds,
                tuple(args.traj_sigmas),
            )
        return
    arms = [a for a in args.arms.split(",") if a]
    assert set(arms) <= set(ARMS), arms
    items = floor_items()
    todo = {
        a: sum(not out_file(args.switch, a, it).exists() for it in items) for a in arms
    }
    print(
        f"{len(items)} floor keys ({len({i['text'] for i in items})} strings) · "
        f"switch σ {args.switch} · to render {todo}",
        flush=True,
    )
    for a in arms:
        print(f"  {a}: above {ARMS[a][0]} / below {ARMS[a][1]}", flush=True)
    if args.dry_run:
        return
    sp = Splitter()
    check_d = check(sp, items[0], args.switch)
    t0 = time.time()
    manifests = {}
    for a in arms:
        manifests[a] = []
        for n, it in enumerate(items):
            fn = out_file(args.switch, a, it)
            sp.render(fn, it, *ARMS[a], args.switch)
            manifests[a].append(
                {
                    "file": str(fn),
                    "cond": a,
                    "seed": it["seed"],
                    "pi": it["pi"],
                    "prompt": it["prompt"],
                    "text": it["text"],
                    "clause": it["clause"],
                    "caption": it["caption"],
                    "switch": args.switch,
                }
            )
            if n % 20 == 0:
                print(
                    f"  {a} {n}/{len(items)} · {(time.time() - t0) / 60:.1f} min",
                    flush=True,
                )
    device = sp.device
    sp.free()
    for a in arms:
        read_arm(manifests[a], arm_dir(args.switch, a), device)
        print(f"read {a}", flush=True)
    summarize(items, arms, args.switch, args.label, check_d)


if __name__ == "__main__":
    main()
