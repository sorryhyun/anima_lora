#!/usr/bin/env python
"""jamo_vl — the jamo arms scored by two VL readers, free and forced-choice
(user 10-11; the reads of ``jamo_phase1`` / ``jamo_curriculum`` were by eye).

Readers: ``stock`` (PaddleOCR-VL 1.6, ``common.readers.StockVl16``) and
``v4`` (anime_tools 0.7.10's ``SfxReader``: the manga reader, tower LoRA +
fine-tuned decoder, trained with KO / ZH SFX crops — ``project/ocr_reader``).
Every image is cut as the ruler cuts it (``ko_reader_cal.crops``: each
detector box padded 12 %, then the whole image). By default ``v4`` reads
free only (``--readers v4,stock`` / ``--forced`` add the others; ``--forced``
doubles the time); the two reads:

- **free**: the reader's greedy decode (``v4`` through its decode guard); a
  hit is any crop reading exactly the target under ``text.norm``; cho / jung
  / jong from the last crop read as one syllable (the whole image's, if it is).
- **forced**: a beam search whose every step may only spell a candidate from
  ``CANDS`` (KS X 1001 ∪ the jamo sets' syllables ∪ the 51 compatibility
  jamo) and then EOS, ``length_penalty`` 0 — a hypothesis' score is
  log p(read == that candidate) under the unconstrained model (the prefix
  processor masks after the log-softmax). The image's answer is the best
  (candidate, crop); top-1 exact and, for a syllable answer, which of cho /
  jung / jong match. Most syllables are three byte tokens in this
  tokenizer (276 of 2 350 are one), so the beam is not exhaustive: ``BEAMS``.

``generate`` copies an image's patches into every beam and would run the
vision tower once per copy; ``VisionCache`` computes each distinct image's
projector features once and serves the copies — and the forced read of a crop
the free read just saw — from a store cleared per chunk of ``CHUNK`` crops.

``cal``: font-drawn lone glyphs (``ko_reader_cal``'s faces and lone
renderer, 4 faces, 64 px, in a bubble) — the J128 ∪ H syllables and the 51 jamo — the
reader's own ceiling on the glyph. ``score``: the cached renders of the jamo
arms (``jamo_read.py`` / ``kozh_render.py``, seed 0): H and the words under
J64 / J96 / J128 / F64-reg / J64-lone, the trained 64 under J64 / F64, the 51
jamo under jamo_lone / seed_1008.

    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/jamo_vl.py cal"
    make daemon-run ARGS="--stall-timeout 900 project/cjk_anima_reseed/probes/jamo_vl.py score"

→ ``results/<ts>-jamo-vl-<mode>/``: ``reads.jsonl`` (every image: crops'
free reads, the forced top-5 per reader), ``result.json`` (the tables).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HOME))
sys.path.insert(0, str(HOME / "probes"))
from reseed import bootstrap  # noqa: E402

bootstrap()

READERS = ("v4", "stock")  # ``--readers``; v4 alone by default
BEAMS = 16
TOPK = 5
FORCED_BATCH = 8  # crops a beam search holds (× BEAMS sequences)
FREE_BATCH = 16
CHUNK = 64  # crops read free then forced on one feature store
CAL_PX = 64
# serif, sans, heavy, handwritten (of ``ko_reader_cal.FACES``)
CAL_FACES = (
    "NotoSerifCJK-Regular.ttc",
    "NanumGothic-Regular.ttf",
    "BlackHanSans-Regular.ttf",
    "NanumPenScript-Regular.ttf",
)
SEED = 0
# the arms' render dirs (seed 0) and what each holds
RENDER = {
    "jamo_j64": ("jamo_j64", ("J64", "H", "words")),
    "jamo_f64": ("jamo_j64", ("J64",)),
    "jamo_j96": ("jamo_j64", ("H", "words")),
    "jamo_j128": ("jamo_j64", ("H", "words")),
    "jamo_f64_reg": ("jamo_j64", ("H", "words")),
    "jamo_j64_lone": ("jamo_j64", ("H", "words")),
    "jamo_j64_112": ("jamo_j64", ("H", "words")),
    "jamo_lone": ("jamo_lone", ("jamo",)),
    "seed_1008": ("jamo_lone", ("jamo",)),
}


def sets() -> dict:
    from jamo_read import WORDS
    from reseed.jamo import COMPAT

    s = json.loads((HOME / "assets" / "jamo_sets.json").read_text(encoding="utf-8"))
    return {
        "J64": list(s["J64"]),
        "J128": list(s["J128"]),
        "H": list(s["H"]),
        "words": list(WORDS),
        "jamo": list(COMPAT),
    }


def cands() -> list:
    from reseed.jamo import COMPAT, KSX1001

    s = sets()
    return list(dict.fromkeys([*KSX1001, *s["J128"], *s["H"], *COMPAT]))


class VisionCache:
    """``model.model.get_image_features`` once per distinct image. An image
    is keyed by its grid and two float64 sums of its patches (plain and
    position-weighted); a call's unseen images go through the tower and the
    projector together, the rest come from ``store``."""

    def __init__(self, model):
        inner = model.model
        self.orig = inner.get_image_features
        self.store: dict = {}
        self.hits = self.misses = 0
        inner.get_image_features = self.features

    def clear(self) -> None:
        self.store.clear()

    @staticmethod
    def key(px, grid) -> tuple:
        import torch

        x = px.reshape(-1).double()
        w = torch.arange(1, x.numel() + 1, device=x.device, dtype=torch.float64)
        return (tuple(grid.tolist()), float(x.sum()), float((x * w).sum()))

    def features(self, pixel_values, image_grid_thw, **kw):
        import torch
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        parts = torch.split(pixel_values, image_grid_thw.prod(-1).tolist())
        keys = [self.key(p, g) for p, g in zip(parts, image_grid_thw)]
        todo = {}
        for k, p, g in zip(keys, parts, image_grid_thw):
            if k not in self.store and k not in todo:
                todo[k] = (p, g)
        self.misses += len(todo)
        self.hits += len(keys) - len(todo)
        if todo:
            got = self.orig(
                torch.cat([p for p, _ in todo.values()]),
                torch.stack([g for _, g in todo.values()]),
                **kw,
            ).pooler_output
            self.store.update(zip(todo, got))
        return BaseModelOutputWithPooling(
            pooler_output=tuple(self.store[k] for k in keys)
        )


class Reader:
    """One VL reader: ``free`` (greedy) and ``forced`` (constrained beam)."""

    def __init__(self, name: str, device: str):
        self.name = name
        if name == "stock":
            from common.readers import StockVl16

            r = StockVl16(device)
            self.model, self.proc, self.prompt = r.model, r.proc, r.text
            self._free = lambda cs: [t for t, _n in r.read(cs)]
        else:
            from anime_tools.ocr.sfx import SfxReader

            r = SfxReader.load(device=device, batch_size=FREE_BATCH)
            self.model, self.proc, self.prompt = r.model, r.processor, r.prompt
            self._free = lambda cs: [x[0] if x else "" for x in r.read_scored(cs)]
        self.vision = VisionCache(self.model)
        self.device = device
        self.min_edge = int(self.proc.image_processor.size["shortest_edge"])
        tok = self.proc.tokenizer
        self.eos = tok.eos_token_id
        self.trie = {"c": {}, "end": None}
        for c in cands():
            node = self.trie
            for t in tok.encode(c, add_special_tokens=False):
                node = node["c"].setdefault(t, {"c": {}, "end": None})
            node["end"] = c

    def free(self, crops: list) -> list[str]:
        out = []
        for i in range(0, len(crops), FREE_BATCH):
            out += self._free(crops[i : i + FREE_BATCH])
        return out

    def forced(self, crops: list) -> list[list]:
        """Top ``TOPK`` ``(candidate, log p)`` per crop."""
        import torch
        from PIL import Image

        out = []
        for i in range(0, len(crops), FORCED_BATCH):
            part = crops[i : i + FORCED_BATCH]
            inputs = self.proc(
                text=[self.prompt] * len(part),
                images=[Image.fromarray(c[:, :, ::-1]) for c in part],
                padding=True,
                padding_side="left",
                return_tensors="pt",
                images_kwargs={
                    "size": {
                        "shortest_edge": self.min_edge,
                        "longest_edge": 1280 * 28 * 28,
                    }
                },
            ).to(self.device)
            n = inputs["input_ids"].shape[-1]

            def allowed(_b, ids):
                node = self.trie
                for t in ids[n:].tolist():
                    node = node["c"].get(t)
                    if node is None:
                        return [self.eos]
                nxt = list(node["c"])
                return nxt + [self.eos] if node["end"] else nxt or [self.eos]

            with torch.inference_mode():
                g = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    num_beams=BEAMS,
                    num_return_sequences=BEAMS,
                    length_penalty=0.0,
                    early_stopping=True,
                    do_sample=False,
                    prefix_allowed_tokens_fn=allowed,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            seqs = g.sequences[:, n:].tolist()
            scores = g.sequences_scores.float().cpu().tolist()
            for k in range(len(part)):
                hyp = {}
                for j in range(k * BEAMS, (k + 1) * BEAMS):
                    node = self.trie
                    for t in seqs[j]:
                        if t == self.eos:
                            break
                        node = node["c"].get(t, {"c": {}, "end": None})
                    c = node["end"]
                    if c is not None and (c not in hyp or scores[j] > hyp[c]):
                        hyp[c] = scores[j]
                out.append(sorted(hyp.items(), key=lambda x: -x[1])[:TOPK])
        return out


def cal_items() -> list:
    """``(meta, PIL image)``: each J128 ∪ H syllable and each jamo alone, in
    ``CAL_FACES`` face, at ``CAL_PX``, in a bubble."""
    from data.grid import render_grid
    from ko_reader_cal import SIZE, faces

    s = sets()
    rng = random.Random(SEED)
    out = []
    for group, glyphs in (
        ("syllable", list(dict.fromkeys(s["J128"] + s["H"]))),
        ("jamo", s["jamo"]),
    ):
        for g in glyphs:
            for name, path in faces().items():
                if name not in CAL_FACES:
                    continue
                im, _ = render_grid(
                    [g], 1, 1, SIZE, [path], rng, True, (CAL_PX / SIZE[0],) * 2
                )
                out.append(({"text": g, "group": group, "face": name}, im))
    return out


def score_items(groups: set | None = None) -> list:
    from PIL import Image
    from reseed import OUT

    s = sets()
    out = []
    for arm, (base, groups_) in RENDER.items():
        for group in groups_:
            if groups and group not in groups:
                continue
            for t in s[group]:
                fn = OUT / base / "render" / arm / f"bubble_{t}_s0.png"
                assert fn.is_file(), fn
                out.append(({"text": t, "group": group, "arm": arm}, Image.open(fn)))
    return out


def judge(text: str, free: list, top: list) -> dict:
    from common.text import norm
    from reseed.jamo import decompose, is_syllable

    rec = {"free_hit": any(norm(r) == norm(text) for r in free)}
    if top is None:  # free only: the read's one syllable, if it is one
        one = [norm(r) for r in free if len(norm(r)) == 1 and is_syllable(norm(r))]
        best = one[-1] if one else ""
    else:
        best = top[0][0] if top else ""
        rec.update(
            forced=best,
            forced_hit=best == text,
            top5_hit=text in [c for c, _ in top],
        )
    if is_syllable(text) and is_syllable(best):
        a, b = decompose(best), decompose(text)
        rec.update(
            {f"{n}_hit": x == y for n, x, y in zip(("cho", "jung", "jong"), a, b)}
        )
    return rec


def table(recs: list, key) -> dict:
    g: dict = defaultdict(list)
    for r in recs:
        g[key(r)].append(r)
    out = {}
    for k, rs in sorted(g.items()):
        row = {"n": len(rs)}
        for rd in READERS:
            for m in (
                "free_hit",
                "forced_hit",
                "top5_hit",
                "cho_hit",
                "jung_hit",
                "jong_hit",
            ):
                v = [r[rd][m] for r in rs if rd in r and m in r[rd]]
                if v:
                    row[f"{rd}.{m.removesuffix('_hit')}"] = f"{sum(v)}/{len(v)}"
        out[k] = row
    return out


def main() -> None:
    import numpy as np
    import torch
    from anime_tools.ocr.animetext import AnimeTextDetector
    from ko_reader_cal import crops

    from bench._common import make_run_dir
    from reseed import HOME as RH

    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("cal", "score"))
    p.add_argument("--limit", type=int, default=0, help="smoke: the first N images")
    p.add_argument("--groups", default="", help="score: these groups only (H,words,…)")
    p.add_argument("--readers", default="v4", help="of READERS, comma-separated")
    p.add_argument(
        "--forced", action="store_true", help="the forced read too (×2 time)"
    )
    a = p.parse_args()
    t0 = time.time()
    groups = {g for g in a.groups.split(",") if g} or None
    items = cal_items() if a.mode == "cal" else score_items(groups)
    if a.limit:
        items = items[: a.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = AnimeTextDetector.load(device=device)
    flat, owner = [], []
    for k, (_m, im) in enumerate(items):
        bgr = np.ascontiguousarray(np.array(im.convert("RGB"))[:, :, ::-1])
        for c in crops(det, bgr):
            flat.append(c)
            owner.append(k)
    del det
    print(f"{len(items)} images, {len(flat)} crops", flush=True)
    recs = [dict(m) for m, _ in items]
    out = make_run_dir(
        "cjk_anima_reseed", label=f"jamo-vl-{a.mode}", root=RH / "results"
    )
    done = []
    for rd in a.readers.split(","):
        assert rd in READERS, rd
        r = Reader(rd, device)
        free, forced = [], []
        for i in range(0, len(flat), CHUNK):
            free += r.free(flat[i : i + CHUNK])
            if a.forced:
                forced += r.forced(flat[i : i + CHUNK])
            r.vision.clear()
            print(
                f"  {rd} {len(forced)} / {len(flat)} crops "
                f"({(time.time() - t0) / 60:.1f} min; vision {r.vision.misses} run, "
                f"{r.vision.hits} served)",
                flush=True,
            )
        per_free, per_top = defaultdict(list), defaultdict(list)
        for k, f in zip(owner, free):
            per_free[k].append(f)
        for k, t in zip(owner, forced):
            per_top[k] += t
        for k, rec in enumerate(recs):
            top = sorted(per_top[k], key=lambda x: -x[1])
            top = list(dict(reversed(top)).items())  # best score per candidate
            top = sorted(top, key=lambda x: -x[1])[:TOPK]
            rec[rd] = {
                "free": per_free[k],
                **({"top": [(c, round(s, 3)) for c, s in top]} if a.forced else {}),
                **judge(rec["text"], per_free[k], top if a.forced else None),
            }
        del r
        torch.cuda.empty_cache()
        done.append(rd)
        tabs = write(out, a, recs, len(flat), done, t0)
        print(json.dumps(tabs, ensure_ascii=False, indent=1), flush=True)
    print(f"→ {out}", flush=True)


def write(out, a, recs, n_crops, done, t0) -> dict:
    from bench._common import write_result

    if a.mode == "cal":
        tabs = {
            "per_group": table(recs, lambda r: r["group"]),
            "per_group_face": table(recs, lambda r: f"{r['group']} {r['face']}"),
        }
    else:
        tabs = {"per_arm": table(recs, lambda r: f"{r['arm']} {r['group']}")}
    with (out / "reads.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_result(
        out,
        script=__file__,
        args={
            "mode": a.mode,
            "beams": BEAMS,
            "cal_px": CAL_PX,
            "limit": a.limit,
            "groups": a.groups,
        },
        label=f"jamo-vl-{a.mode}",
        metrics={
            **tabs,
            "readers": done,
            "images": len(recs),
            "crops": n_crops,
            "minutes": round((time.time() - t0) / 60, 1),
        },
        artifacts=["reads.jsonl"],
    )
    return tabs


if __name__ == "__main__":
    main()
