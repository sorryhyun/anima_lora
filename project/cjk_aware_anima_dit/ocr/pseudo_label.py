#!/usr/bin/env python3
"""P1 (``plan2.md``): pseudo-labels for the AnimeText crops by cross-reader agreement.

    ANIMA_ANIMETEXT_ROOT=… ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="\\
        project/cjk_aware_anima_dit/ocr/pseudo_label.py sweep --reader vl16 --n 100000"
    …                                                  sweep --reader manga_ocr --n 100000
    …                                                  filter --name pl100k
    …                                                  manifest --name pl100k --rows 20000

Three subcommands, one per plan step:

* **sweep** — one reader over a seeded draw from ``manifest_all``, batched on
  :func:`crop_dataset.token_batches` (a token budget, **not** the eval readers'
  fixed ``bs 32``: the pool's large-crop tail reaches 122.6k tokens in a
  32-crop batch and OOMs 16 GB). Writes ``image_id, k, pred, n_tokens, score,
  runaway``, flushed every ``--flush`` batches so a crash resumes.
* **filter** — the plan's guards + ♡-blind agreement between the two readers,
  keeping **B′'s string** (so hearts and ``〜`` survive where manga-ocr drops
  them), then the kept-set stats against COO train.
* **manifest** — the kept rows in the COO manifest schema as ``kind = "sfx"``
  (``load_split`` only draws ``sfx``/``speech``, so an unknown kind is silently
  dropped), ``path`` absolute, and the ``--speech_ratio`` the SFT arm needs
  printed so the real speech draw does not inflate with N.

The draw is written once (``<name>_draw.parquet``) and both readers consume it,
so the two prediction tables are row-aligned by ``(image_id, k)``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_dataset as cd  # noqa: E402
import eval_manga109 as ev  # noqa: E402
import manga109 as m109  # noqa: E402
from animetext_crops import animetext_root  # noqa: E402

OUT = m109.REPO / "output/ocr/pseudo"
CKPT = {
    "vl16": m109.REPO / "output/ocr/vl16_tower_lr1e-5/best",
    "manga_ocr": m109.REPO / "output/ocr/mocr_lr5e-5/best",
}
# The processor's own bounds — mirrored from ssl_tower_simmim.py's wiring so the
# token estimate matches what the tower will actually be handed.
MIN_PX, MAX_PX = 112896, 1280 * 28 * 28


# --------------------------------------------------------------------------- draw


def load_draw(name: str, n: int | None, seed: int) -> pd.DataFrame:
    """The seeded draw, stratified over the three AnimeText splits. Written once."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}_draw.parquet"
    if path.is_file():
        df = pd.read_parquet(path)
        print(f"draw {path.name}: {len(df)} rows (existing)", flush=True)
        return df
    if n is None:
        raise SystemExit(f"{path} does not exist — pass --n to build the draw")
    root = animetext_root()
    pool = pd.read_parquet(
        root / "animetext_crops" / "manifest_all.parquet",
        columns=["image_id", "k", "w", "h", "box", "path", "split"],
    )
    frac = n / len(pool)
    parts = [
        g.sample(n=min(len(g), int(round(len(g) * frac))), random_state=seed)
        for _, g in pool.groupby("split")
    ]
    df = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_parquet(path, index=False)
    print(
        f"draw {path.name}: {len(df)} of {len(pool)} rows, "
        f"splits {df.split.value_counts().to_dict()}",
        flush=True,
    )
    return df


# --------------------------------------------------------------------------- sweep


class Vl16Sweeper:
    """``ev.Vl16Reader``'s model, read on a token budget and reporting token counts.

    Composition rather than a change to ``ev.Vl16Reader``: that class is the
    measured-eval path (``eval_manga109`` / ``eval_sfx``) and its batching rule
    is part of every number already on the table.
    """

    name = "vl16"

    def __init__(self, ckpt: str, device: str):
        self.r = ev.Vl16Reader(ckpt, device)
        self.proc, self.model, self.device = self.r.proc, self.r.model, device
        self.torch = self.r.torch
        msgs = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": "OCR:"}],
            }
        ]
        self.text = self.proc.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )

    def read(self, crops: list) -> list[tuple[str, int, float]]:
        from PIL import Image

        tok = self.proc.tokenizer
        images = [Image.fromarray(c[:, :, ::-1]) for c in crops]
        inputs = self.proc(
            text=[self.text] * len(images),
            images=images,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": self.r.min_edge,
                    "longest_edge": 1280 * 28 * 28,
                }
            },
        ).to(self.device)
        n = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            o = self.model.generate(
                **inputs,
                max_new_tokens=ev.MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        out = []
        for row in o:
            ids = [
                t
                for t in row[n:].tolist()
                if t not in (tok.eos_token_id, tok.pad_token_id)
            ]
            out.append((tok.decode(ids).strip(), len(ids), float("nan")))
        return out


class MangaOcrSweeper:
    """The manga-ocr fine-tune; its ``read`` already returns a mean token logprob."""

    name = "manga_ocr"

    def __init__(self, ckpt: str, device: str):
        self.r = ev.MangaOcrReader(ckpt, device)

    def read(self, crops: list) -> list[tuple[str, int, float]]:
        pairs = self.r.m.read(crops, len(crops), max_new_tokens=ev.MAX_NEW_TOKENS)
        return [(t, len(t), float(s)) for t, s in pairs]


SWEEPERS = {"vl16": Vl16Sweeper, "manga_ocr": MangaOcrSweeper}


def cmd_sweep(a: argparse.Namespace) -> None:
    draw = load_draw(a.name, a.n, a.seed)
    root = animetext_root()
    out_path = OUT / f"{a.name}_{a.reader}.parquet"

    done: set[tuple[int, int]] = set()
    prev: list[dict] = []
    if out_path.is_file() and not a.overwrite:
        old = pd.read_parquet(out_path)
        prev = old.to_dict("records")
        done = set(zip(old.image_id, old.k))
        print(f"resuming: {len(done)} rows already read", flush=True)

    todo = draw[[t not in done for t in zip(draw.image_id, draw.k)]].reset_index(
        drop=True
    )
    if todo.empty:
        print("nothing to do", flush=True)
        return

    if a.reader == "vl16":
        tokens = cd.vl_tokens(todo.w.to_numpy(), todo.h.to_numpy(), MIN_PX, MAX_PX)
        batches = cd.token_batches(tokens, a.token_budget, a.bs, random.Random(a.seed))
        print(
            f"{len(todo)} crops in {len(batches)} batches; per-crop tokens "
            f"median {int(np.median(tokens))} max {int(tokens.max())}, "
            f"budget {a.token_budget}/batch (bs {a.bs} max)",
            flush=True,
        )
    else:  # flat cost per crop — the ViT input is a fixed size
        idx = list(range(len(todo)))
        batches = [idx[s : s + a.bs] for s in range(0, len(idx), a.bs)]
        print(f"{len(todo)} crops in {len(batches)} batches of {a.bs}", flush=True)

    reader = SWEEPERS[a.reader](str(CKPT[a.reader]), a.device)
    rows: list[dict] = list(prev)
    rec = m109.pilot_records()
    t0 = time.perf_counter()
    n_done = 0
    for bi, idx in enumerate(batches):
        sub = todo.iloc[idx]
        crops, keep = [], []
        for r in sub.itertuples():
            img = cv2.imread(str(root / r.path), cv2.IMREAD_COLOR)
            if img is not None:
                crops.append(img)
                keep.append(r)
        if not crops:
            continue
        for r, (pred, ntok, score) in zip(keep, reader.read(crops)):
            rows.append(
                {
                    "image_id": int(r.image_id),
                    "k": int(r.k),
                    "pred": pred,
                    "n_tokens": int(ntok),
                    "score": score,
                    "runaway": bool(rec.is_runaway(pred)),
                }
            )
        n_done += len(crops)
        if (bi + 1) % a.flush == 0 or bi + 1 == len(batches):
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            el = time.perf_counter() - t0
            rate = n_done / max(el, 1e-9)
            left = (len(todo) - n_done) / max(rate, 1e-9)
            print(
                f"[{a.reader}] batch {bi + 1}/{len(batches)}  {n_done}/{len(todo)} "
                f"crops  {rate:.1f} crops/s  eta {left / 60:.0f} min",
                flush=True,
            )
    print(f"[{a.reader}] wrote {out_path} ({len(rows)} rows)", flush=True)


# --------------------------------------------------------------------------- filter

RUN4 = re.compile(r"(.)\1{4,}")
SMALL_KANA = set("っゃゅょぁぃぅぇぉ")


def ngram_dominated(s: str) -> bool:
    """A repeated 2- or 3-gram covering more than half the string (plan step 1.1).

    Stricter than ``build_ocr_records.is_runaway`` (run ≥ 8, 3-gram ≥ 3×) on
    purpose: a pseudo-label only has to be *safe*, and the pool is large enough
    that throwing away a borderline read costs nothing.
    """
    for n in (2, 3):
        if len(s) < 2 * n:
            continue
        counts = Counter(s[i : i + n] for i in range(len(s) - n + 1))
        gram, k = counts.most_common(1)[0]
        if k >= 2 and k * n > 0.5 * len(s):
            return True
    return False


def guard(pred: str, n_tokens: int) -> str:
    """``"ok"`` or the name of the guard that rejected the read."""
    if not pred.strip():
        return "empty"
    if n_tokens >= ev.MAX_NEW_TOKENS:
        return "truncated"
    if len(pred) > cd.MAX_TARGET_CHARS:
        return "too_long"
    if RUN4.search(pred):
        return "char_run"
    if ngram_dominated(pred):
        return "ngram"
    return "ok"


def script_mix(strings: list[str]) -> dict[str, float]:
    """Share of rows containing at least one character of each script."""

    def has(s: str, lo: int, hi: int) -> bool:
        return any(lo <= ord(c) <= hi for c in s)

    n = max(len(strings), 1)
    return {
        "hiragana": 100 * sum(has(s, 0x3040, 0x309F) for s in strings) / n,
        "katakana": 100 * sum(has(s, 0x30A0, 0x30FF) for s in strings) / n,
        "han": 100 * sum(has(s, 0x4E00, 0x9FFF) for s in strings) / n,
        "latin": 100 * sum(has(s, 0x0041, 0x007A) for s in strings) / n,
    }


def char_stats(strings: list[str]) -> dict[str, float]:
    n = max(len(strings), 1)
    keys = "".join(strings)
    return {
        "mean_len": sum(len(s) for s in strings) / n,
        "heart_%": 100 * sum(("♡" in s or "♥" in s) for s in strings) / n,
        "wave_%": 100 * sum(("〜" in s or "~" in s) for s in strings) / n,
        "chouon_%": 100 * sum("ー" in s for s in strings) / n,
        "smallkana_%": 100 * sum(bool(set(s) & SMALL_KANA) for s in strings) / n,
        "n_chars": len(keys),
    }


def coo_train_baseline() -> tuple[list[str], int]:
    df = pd.read_parquet(m109.derived_root() / "manifest.parquet")
    df = df[(df.split == "train")]
    return list(
        df.text.map(lambda s: "".join(unicodedata.normalize("NFKC", s).split()))
    ), len(df)


def cmd_filter(a: argparse.Namespace) -> None:
    draw = load_draw(a.name, None, a.seed)
    tabs = {}
    for reader in ("vl16", "manga_ocr"):
        p = OUT / f"{a.name}_{reader}.parquet"
        if not p.is_file():
            raise SystemExit(f"missing {p} — run `sweep --reader {reader}` first")
        tabs[reader] = pd.read_parquet(p).set_index(["image_id", "k"])

    df = draw.set_index(["image_id", "k"])
    df = df.join(tabs["vl16"].add_prefix("b_"), how="inner").join(
        tabs["manga_ocr"].add_prefix("m_"), how="inner"
    )
    df = df.reset_index()
    n = len(df)
    print(f"\n{n} crops read by both readers", flush=True)

    df["guard"] = [guard(p, t) for p, t in zip(df.b_pred, df.b_n_tokens)]
    ok = df[df.guard == "ok"].copy()
    ok["agree"] = [
        ev.exact_key(b) == ev.exact_key(m) for b, m in zip(ok.b_pred, ok.m_pred)
    ]
    kept = ok[ok.agree].copy()

    print("\n=== yield ===")
    counts = df.guard.value_counts().to_dict()
    for g in ("empty", "truncated", "too_long", "char_run", "ngram"):
        c = counts.get(g, 0)
        print(f"  guard {g:10s} {c:7d}  {100 * c / n:5.2f}%")
    print(f"  guard {'ok':10s} {len(ok):7d}  {100 * len(ok) / n:5.2f}%")
    print(
        f"  agree           {len(kept):7d}  {100 * len(kept) / len(ok):5.2f}% of survivors"
    )
    print(f"  KEPT            {len(kept):7d}  {100 * len(kept) / n:5.2f}% of the draw")

    # The plan asks for the SFX vs speech yields separately; AnimeText boxes are
    # not kind-tagged, so orientation is the only proxy the pool carries.
    ok["orient"] = np.where(ok.h > ok.w, "vertical", "horizontal")
    print("\n  by orientation (AnimeText has no kind labels — orient is the proxy):")
    for o, g in ok.groupby("orient"):
        print(
            f"    {o:11s} {int(g.agree.sum()):7d} / {len(g):7d} = {100 * g.agree.mean():5.2f}%"
        )

    coo_text, n_coo = coo_train_baseline()
    print(f"\n=== kept set vs COO train (n={len(kept)} vs {n_coo}) ===")
    ks, cs = char_stats(list(kept.b_pred)), char_stats(coo_text)
    print(f"  {'metric':14s} {'kept':>9s} {'COO train':>10s}")
    for k in ("mean_len", "heart_%", "wave_%", "chouon_%", "smallkana_%"):
        print(f"  {k:14s} {ks[k]:9.2f} {cs[k]:10.2f}")
    km, cm = script_mix(list(kept.b_pred)), script_mix(coo_text)
    for k in km:
        print(f"  {k + '_%':14s} {km[k]:9.2f} {cm[k]:10.2f}")

    verdict = (
        "PASS — the kept set carries more ♡ than COO train, so the arm can move "
        "the misses the labels were for."
        if ks["heart_%"] > cs["heart_%"]
        else "FAIL — the kept set carries no more ♡ than COO train; per plan2 § P1 "
        "step 1 the arm cannot fix the ♡ misses. Say so before any SFT."
    )
    print(f"\n{verdict}", flush=True)

    kept_path = OUT / f"{a.name}_kept.parquet"
    kept.to_parquet(kept_path, index=False)
    (OUT / f"{a.name}_filter.json").write_text(
        json.dumps(
            {
                "draw": n,
                "guards": counts,
                "kept": len(kept),
                "kept_frac": len(kept) / n,
                "kept_stats": ks,
                "coo_train_stats": cs,
                "kept_script": km,
                "coo_script": cm,
                "heart_precondition": "pass"
                if ks["heart_%"] > cs["heart_%"]
                else "fail",
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"wrote {kept_path} and {a.name}_filter.json", flush=True)


# --------------------------------------------------------------------------- manifest


def cmd_manifest(a: argparse.Namespace) -> None:
    kept = pd.read_parquet(OUT / f"{a.name}_kept.parquet")
    if a.rows and a.rows < len(kept):
        kept = kept.sample(a.rows, random_state=a.seed)
    root = animetext_root()

    def poly(box) -> str:
        x0, y0, x1, y1 = [float(v) for v in box]
        return json.dumps([x0, y0, x1, y0, x1, y1, x0, y1])

    out = pd.DataFrame(
        {
            "split": "train",
            # load_split only draws sfx / speech — an unknown kind is dropped silently.
            "kind": "sfx",
            "id": [f"{i}_{k}" for i, k in zip(kept.image_id, kept.k)],
            "book": [f"at{i}" for i in kept.image_id],
            "page": 0,
            "text": list(kept.b_pred),
            "joined": False,
            "orient": np.where(kept.h > kept.w, "vertical", "horizontal"),
            "w": kept.w.astype("int64").to_numpy(),
            "h": kept.h.astype("int64").to_numpy(),
            "poly": [poly(b) for b in kept.box],
            # absolute: CropDataset does `derived_root() / path`, which pathlib
            # resolves to the absolute path, so the other volume needs no symlink.
            "path": [str(root / p) for p in kept.path],
            "source": "pseudo",
        }
    )
    missing = sum(not Path(p).is_file() for p in out.path[:200])
    if missing:
        raise SystemExit(
            f"{missing}/200 sampled crop paths do not exist — check the root"
        )

    out_name = a.out_name or a.name
    path = m109.derived_root() / f"manifest_pseudo_{out_name}.parquet"
    out.to_parquet(path, index=False)
    # B′ ran ``--speech_ratio 1.0``, and the speech draw is capped at the SFX
    # count, so it trained on 38,582 speech rows — not the pool's full 38,634.
    # Targeting 38,582 makes the pseudo rows the *only* difference from B′;
    # plan2's 38634 would quietly hand the arm 52 extra speech rows.
    n_sfx, n_speech_bprime = 38582, 38582
    ratio = n_speech_bprime / (n_sfx + len(out))
    print(f"wrote {path} ({len(out)} rows, kind=sfx, source=pseudo)", flush=True)
    print(
        f"\nSFT arm:\n  --extra_manifest pseudo_{out_name} --speech_ratio {ratio:.6f}\n"
        f"  (holds speech at B′'s {n_speech_bprime} instead of inflating with N; "
        f"train becomes ~{n_sfx + len(out) + n_speech_bprime} rows)",
        flush=True,
    )


# --------------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="pl100k", help="draw / output name")
    ap.add_argument("--seed", type=int, default=0)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="one reader over the draw")
    s.add_argument("--reader", choices=sorted(SWEEPERS), required=True)
    s.add_argument("--n", type=int, help="draw size (only needed the first time)")
    s.add_argument("--bs", type=int, default=32, help="max crops per batch")
    s.add_argument("--token_budget", type=int, default=24000)
    s.add_argument("--flush", type=int, default=100, help="checkpoint every N batches")
    s.add_argument("--device", default="cuda")
    s.add_argument("--overwrite", action="store_true", help="ignore a partial sweep")
    s.set_defaults(fn=cmd_sweep)

    f = sub.add_parser("filter", help="guards + ♡-blind agreement + kept-set stats")
    f.set_defaults(fn=cmd_filter)

    m = sub.add_parser("manifest", help="kept rows → COO manifest schema")
    m.add_argument("--rows", type=int, help="subsample the kept set to N rows")
    m.add_argument(
        "--out_name",
        help="name the written manifest_pseudo_<out_name>.parquet (default: --name), "
        "so several row counts can be cut from one kept set",
    )
    m.set_defaults(fn=cmd_manifest)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
