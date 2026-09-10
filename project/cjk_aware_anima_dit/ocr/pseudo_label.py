#!/usr/bin/env python3
"""P1 (``plan2.md``): pseudo-labels for the AnimeText crops by cross-reader agreement.

    ANIMA_ANIMETEXT_ROOT=… ANIMA_MANGA109S_ROOT=… make daemon-run ARGS="\\
        project/cjk_aware_anima_dit/ocr/pseudo_label.py sweep --reader vl16 --n 100000"
    …                                                  sweep --reader manga_ocr --n 100000
    …                                                  filter --name pl100k
    …                                                  manifest --name pl100k --rows 20000

**Korean (plan2 § K1–K2, 2026-09-09).** manga-ocr cannot vote on hangul and B′
collapses on it (``기존에 쓰던 폰이…`` → ``프産の絲の緑の絶滅の…``), so the KO
route is a **screen** rather than a second full sweep: ``hayai`` reads the draw,
its hangul-dominant rows are the KO arm, and ``stock`` re-reads **only those**
(≈ 0.5 % of the pool). Teacher = ``stock``, voter = ``hayai``, label = stock's
string — including its spacing, which hayai drops and ``exact_key`` ignores.

    … sweep  --reader hayai
    … sweep  --reader stock --rows_from hayai --rows_script ko
    … filter --teacher stock --voter hayai --script ko --out_suffix ko

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
CKPT: dict[str, object] = {
    "vl16": m109.REPO / "output/ocr/vl16_tower_lr1e-5/best",
    "manga_ocr": m109.REPO / "output/ocr/mocr_lr5e-5/best",
    # ``None`` = the base weights (``Vl16Reader`` falls back to
    # ``models/paddleocr_vl_1.6``); hayai is pinned to the scored revision.
    "stock": None,
    "hayai": "JustANormalTinkerer/hayai-ocr-v2@v2.1.5",
}

# --------------------------------------------------------------------------- script

HANGUL = re.compile(r"[가-힣]")
KANA = re.compile(r"[぀-ヿ]")
# Simplified-only component blocks — the 讠/钅/贝/见/页/鸟/马/门/车/鱼/纠/饣
# series. Bounds stop one codepoint short of the traditional forms that follow
# each run (長 門 閃 谷 豆 角 骨 辛 鳥); validated at zero overlap against
# the 1.05M-char JA-only corpus of manga-ocr's ``pl100k`` reads.
SIMP_BLOCKS = (
    (0x8BA0, 0x8C36),
    (0x9485, 0x9576),
    (0x8D1D, 0x8D2D),
    (0x89C1, 0x89D1),
    (0x9875, 0x9891),
    (0x9E1F, 0x9E4F),
    (0x9A6C, 0x9AA7),
    (0x95E8, 0x9601),
    (0x8F66, 0x8F9A),
    (0x9C7C, 0x9CE4),
    (0x7EA0, 0x7F2F),
    (0x9971, 0x9980),
)
# Simplified forms outside those blocks whose Japanese counterpart is a
# different glyph (这/這, 发/発, 东/東 …). Shared shinjitai (国 来 当 体 数
# 点) is deliberately absent — it carries no signal. Same zero-overlap check.
SIMP_CHARS = set(
    "这么们说过还东长开亲书问题图产头儿务爱运习进无发经济应龙买卖岁离风飞灵丽药"
    "单边远连种类级时关对给让觉现动样个"
)


def hangul_dominant(s: str) -> bool:
    """Hangul is >= 60 % of the non-ASCII, non-space characters."""
    s = str(s)
    n = len([c for c in s if not c.isspace() and not c.isascii()])
    return bool(HANGUL.findall(s)) and len(HANGUL.findall(s)) >= 0.6 * n


def simplified_zh(s: str) -> bool:
    """Carries a simplified-only marker and neither kana nor hangul."""
    s = str(s)
    if KANA.search(s) or HANGUL.search(s):
        return False
    return any(
        c in SIMP_CHARS or any(a <= ord(c) <= b for a, b in SIMP_BLOCKS) for c in s
    )


SCRIPT = {"ko": hangul_dominant, "zh": simplified_zh}
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

    def __init__(self, ckpt: str | None, device: str):
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

    def __init__(self, ckpt: str | None, device: str):
        self.r = ev.MangaOcrReader(ckpt, device)

    def read(self, crops: list) -> list[tuple[str, int, float]]:
        pairs = self.r.m.read(crops, len(crops), max_new_tokens=ev.MAX_NEW_TOKENS)
        return [(t, len(t), float(s)) for t, s in pairs]


class HayaiSweeper:
    """hayai v2.1.5 — the KO screen. Its ``generate`` returns strings only, so
    ``n_tokens`` is reported as 0 and the ``truncated`` guard never fires on a
    hayai row; hayai is only ever the **voter**, and the label it votes on comes
    from a reader that does report a decode length."""

    name = "hayai"

    def __init__(self, ckpt: str | None, device: str):
        self.r = ev.HayaiReader(ckpt, device)

    def read(self, crops: list) -> list[tuple[str, int, float]]:
        orients = [
            "vertical" if c.shape[0] > c.shape[1] else "horizontal" for c in crops
        ]
        return [(t, 0, float("nan")) for t in self.r.read(crops, orients, len(crops))]


SWEEPERS = {
    "vl16": Vl16Sweeper,
    "manga_ocr": MangaOcrSweeper,
    "stock": Vl16Sweeper,
    "hayai": HayaiSweeper,
}


def _shard_path(out_path: Path, shard: int) -> Path:
    return out_path.with_suffix(f".shard{shard}.parquet")


def _fan_out(a: argparse.Namespace, out_path: Path) -> None:
    """Run ``--workers N`` copies of this sweep, one shard each, then merge.

    The decode loop is **single-threaded CPU-bound**, not GPU-bound: one worker
    pegs one core at 100 % while the GPU idles at ~25 % and 11 cores do nothing
    (measured 2026-09-09, hayai bs 32). Raising the batch makes it worse — the
    per-step beam bookkeeping is Python work over batch × beams — so the lever
    is more processes, each with its own interpreter and CUDA context.
    """
    import subprocess

    argv, skip = [], False
    for tok in sys.argv[1:]:  # drop --workers N / --workers=N, keep the rest
        if skip:
            skip = False
            continue
        if tok == "--workers":
            skip = True
            continue
        if tok.startswith("--workers="):
            continue
        argv.append(tok)
    procs = []
    for i in range(a.workers):
        cmd = [sys.executable, str(Path(__file__).resolve()), *argv]
        cmd += ["--shard", str(i), "--num_shards", str(a.workers)]
        print(f"[worker {i}] {' '.join(cmd[-4:])}", flush=True)
        procs.append(subprocess.Popen(cmd))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"worker exit codes {codes} — shards left on disk")

    frames = [pd.read_parquet(out_path)] if out_path.is_file() else []
    shards = [_shard_path(out_path, i) for i in range(a.workers)]
    frames += [pd.read_parquet(s) for s in shards if s.is_file()]
    if not frames:
        print("no shard produced any rows", flush=True)
        return
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["image_id", "k"], keep="last"
    )
    merged.to_parquet(out_path, index=False)
    for sp in shards:
        sp.unlink(missing_ok=True)
    print(f"merged {len(merged)} rows → {out_path}", flush=True)


def cmd_sweep(a: argparse.Namespace) -> None:
    draw = load_draw(a.name, a.n, a.seed)
    root = animetext_root()
    out_path = OUT / f"{a.name}_{a.reader}.parquet"

    if a.rows_from:
        # The screen (plan2 § K1): read only the rows a cheaper sweep already
        # called this script, so the expensive reader costs ~0.5 % of a pass.
        src = OUT / f"{a.name}_{a.rows_from}.parquet"
        if not src.is_file():
            raise SystemExit(
                f"missing {src} — run `sweep --reader {a.rows_from}` first"
            )
        t = pd.read_parquet(src)
        hit = t[[bool(SCRIPT[a.rows_script](p)) for p in t.pred.fillna("")]]
        keys = set(zip(hit.image_id, hit.k))
        draw = draw[[k in keys for k in zip(draw.image_id, draw.k)]].reset_index(
            drop=True
        )
        print(
            f"screen: {a.rows_from} calls {len(hit)} of {len(t)} rows "
            f"{a.rows_script} ({100 * len(hit) / max(len(t), 1):.3f}%) "
            f"→ {len(draw)} in the draw",
            flush=True,
        )

    if a.workers > 1 and a.shard is None:
        _fan_out(a, out_path)
        return

    # A shard writes its own file and reads the canonical one for `done`, so a
    # resumed run never re-reads a crop an earlier pass already has.
    shard_path = out_path if a.shard is None else _shard_path(out_path, a.shard)
    done: set[tuple[int, int]] = set()
    prev: list[dict] = []
    if not a.overwrite:
        # The canonical file is read-only here — it holds what earlier passes
        # merged, so a shard skips those crops but must not rewrite them.
        if out_path.is_file() and out_path != shard_path:
            done |= set(pd.read_parquet(out_path)[["image_id", "k"]].itertuples(False))
        if shard_path.is_file():
            mine = pd.read_parquet(shard_path)
            done |= set(zip(mine.image_id, mine.k))
            prev = mine.to_dict("records")
        if done:
            print(f"resuming: {len(done)} rows already read", flush=True)

    todo = draw[[t not in done for t in zip(draw.image_id, draw.k)]].reset_index(
        drop=True
    )
    if a.shard is not None:
        # Interleave, not slice: the draw is shuffled but crop size is not, and
        # a contiguous slice would hand one worker the large-crop tail.
        todo = todo.iloc[a.shard :: a.num_shards].reset_index(drop=True)
        print(f"shard {a.shard}/{a.num_shards}: {len(todo)} crops", flush=True)
    if todo.empty:
        print("nothing to do", flush=True)
        return
    out_path = shard_path

    if a.reader in ("vl16", "stock"):
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

    ckpt = CKPT[a.reader]
    reader = SWEEPERS[a.reader](str(ckpt) if ckpt is not None else None, a.device)
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
    for reader in (a.teacher, a.voter):
        p = OUT / f"{a.name}_{reader}.parquet"
        if not p.is_file():
            raise SystemExit(f"missing {p} — run `sweep --reader {reader}` first")
        tabs[reader] = pd.read_parquet(p).set_index(["image_id", "k"])

    df = draw.set_index(["image_id", "k"])
    # ``b_`` = the teacher (whose string becomes the label), ``m_`` = the voter.
    df = df.join(tabs[a.teacher].add_prefix("b_"), how="inner").join(
        tabs[a.voter].add_prefix("m_"), how="inner"
    )
    df = df.reset_index()
    if a.script:
        # The teacher only ever saw the screened rows, so the inner join above
        # has already cut the draw down; this re-asserts the call on the
        # teacher's own output, which is what K2 routes on.
        df = df[[bool(SCRIPT[a.script](p)) for p in df.b_pred.fillna("")]]
        df = df.reset_index(drop=True)
    n = len(df)
    print(
        f"\n{n} crops read by both {a.teacher} (teacher) and {a.voter} (voter)"
        + (f", {a.script}-called by the teacher" if a.script else ""),
        flush=True,
    )
    if not n:
        raise SystemExit("no rows survived the join — check the screen")

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

    space_pct = 100 * kept.b_pred.map(lambda t: " " in str(t)).mean()
    print(
        f"\n  label spacing: {space_pct:5.2f}% of kept labels carry a space "
        f"(the voter's spacing is ignored — exact_key is whitespace-blind)"
    )

    if a.script in ("ko", "zh"):
        # The ♡ precondition and the COO character profile are Japanese
        # questions; a KO/ZH arm is gated by K3's held-out set instead.
        kept_path = OUT / f"{a.name}_kept_{a.out_suffix or a.script}.parquet"
        kept.to_parquet(kept_path, index=False)
        (OUT / f"{a.name}_filter_{a.out_suffix or a.script}.json").write_text(
            json.dumps(
                {
                    "draw": n,
                    "teacher": a.teacher,
                    "voter": a.voter,
                    "script": a.script,
                    "guards": counts,
                    "kept": len(kept),
                    "kept_frac": len(kept) / n,
                    "kept_stats": char_stats(list(kept.b_pred)),
                    "kept_script": script_mix(list(kept.b_pred)),
                    "label_space_%": space_pct,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        print(f"wrote {kept_path}", flush=True)
        return

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
    suffix = f"_{a.in_suffix}" if a.in_suffix else ""
    kept = pd.read_parquet(OUT / f"{a.name}_kept{suffix}.parquet")
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
    s.add_argument(
        "--rows_from",
        choices=sorted(SWEEPERS),
        help="read only the rows this reader's sweep already called --rows_script",
    )
    s.add_argument("--rows_script", choices=sorted(SCRIPT), default="ko")
    s.add_argument(
        "--workers",
        type=int,
        default=1,
        help="run N sharded copies in parallel and merge (the decode loop is "
        "single-threaded CPU-bound, so this scales where --bs does not)",
    )
    s.add_argument("--shard", type=int, help="internal: this worker's shard index")
    s.add_argument("--num_shards", type=int, default=1, help="internal")
    s.set_defaults(fn=cmd_sweep)

    f = sub.add_parser("filter", help="guards + ♡-blind agreement + kept-set stats")
    f.add_argument("--teacher", choices=sorted(SWEEPERS), default="vl16")
    f.add_argument("--voter", choices=sorted(SWEEPERS), default="manga_ocr")
    f.add_argument(
        "--script",
        choices=sorted(SCRIPT),
        help="keep only rows the teacher calls this script (K2's route)",
    )
    f.add_argument("--out_suffix", help="name the kept parquet (default: --script)")
    f.set_defaults(fn=cmd_filter)

    m = sub.add_parser("manifest", help="kept rows → COO manifest schema")
    m.add_argument("--rows", type=int, help="subsample the kept set to N rows")
    m.add_argument(
        "--in_suffix", help="read <name>_kept_<suffix>.parquet (a script arm)"
    )
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
