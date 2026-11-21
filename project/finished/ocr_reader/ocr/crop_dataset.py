"""Shared training data for the O2 fine-tunes (``plan_ocr.md``): the O1 manifest as a
torch ``Dataset`` of (BGR crop, target string) with train-time augmentation.

* **Mix** — decision 2: COO ``sfx`` : Manga109 ``speech`` 1 : 1 *by count* (the
  manifest is already count-matched per book; ``--speech_ratio`` rescales the
  speech draw). Both kinds come from the same book split.
* **Target rule** (findings § O1, corrected 2026-09-09, folded 2026-09-12) —
  ``textnorm.fold_glyphs`` (dakuten, NFKC, hearts / wave / long dash / dot runs
  to one spelling each), then **collapse** whitespace runs to one ASCII space,
  not delete them: Manga109's ``<text>`` keeps line breaks and full-width
  punctuation that manga-ocr's vocab lacks. Deleting whitespace was harmless
  while every target was Japanese and is wrong the moment Korean enters —
  띄어쓰기 is lexical there. The scorer's ``exact`` is whitespace-blind and
  applies the same folds, so a target change moves no measured number by
  itself; every checkpoint trained before the whitespace fix is spacing-dirty
  (``whitespace_fixed.md``).
* **Augmentation** — ``augment.Augment`` on the train split only; per-worker
  seeding so DataLoader workers do not replay one RNG stream.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manga109 as m109  # noqa: E402
import textnorm  # noqa: E402
from augment import Augment  # noqa: E402

MAX_TARGET_CHARS = 150
"""Longest training target, in characters after :func:`normalize_target`. A longer label is
dropped, not cut (96 and sliced until 2026-11-20: 53 KO afterword / colophon paragraphs trained
cut mid-sentence; 14 KO and 3 Manga109 speech rows are over 150). Stays under the reader's
200-token decode cap."""


def cap_targets(out: pd.DataFrame, what: str) -> pd.DataFrame:
    """``target`` = the normalized label; rows over :data:`MAX_TARGET_CHARS` dropped."""
    out = out.copy()
    out["target"] = out.text.map(normalize_target)
    long = out.target.str.len() > MAX_TARGET_CHARS
    if long.any():
        print(
            f"[data] {what}: {int(long.sum())} rows over {MAX_TARGET_CHARS} chars dropped",
            flush=True,
        )
    return out[~long]


TARGET_NORM = 9
"""Bumped when :func:`normalize_target` changes. 1 = NFKC, whitespace deleted
(every run before 2026-09-09); 2 = whitespace collapsed to one ASCII space
(``vl16_b2_norm2``); 3 = ``textnorm.fold_glyphs`` first — hearts → ``♡``, dot
runs → ``…``, ``〜`` → ``~``, long dashes → ``―``, spacing dakuten → combining
(v2 taught ``あ ゙っ`` for ``あ゛っ`` because NFKC splits the spacing mark off
with a space) — ``vl16_b2_norm3``; 4 = the same folds with the heart flipped
to ``♥``, the single token (``♡`` is three bytes; norm3 lost every raw ``♥``
and 31 sincos rows — plan_vl_respace R5b); 5 = norm 4 with the label's line
breaks kept as ``\\n`` (v5; v4 trained on 4); 6 = norm 5 with curly double
quotes ``“`` / ``”`` → ``"`` and lone conjoining jamo back to the typed
compatibility jamo (``ᄏᄏ`` → ``ㅋㅋ``; norm 3–5 taught ``ᄏ``) (2026-11-17);
7 = norm 6 without the dot-run fold — ``...`` / ``・・・`` / ``…`` trained as
labelled, ``…`` kept from NFKC's ``...``, ``‥`` → ``..`` (2026-11-18; scoring keys
still fold); 8 = norm 7 with a Hangul string's dashes (``―`` / ``—`` / ``–`` …) → ``-``
(2026-11-20, ``textnorm`` fold 5); 9 = norm 8 with ``◯`` / ``〇`` → ``○`` and ``☓`` / ``✕`` … → ``×``
(2026-11-20, ``textnorm`` fold 4)."""


def normalize_target(s: str) -> str:
    """:func:`textnorm.training_target` — folds, whitespace runs inside a line
    collapsed to one ASCII space, line breaks kept as ``\\n``, edges stripped,
    heart spelled ``♥``.

    v1 joined the split with ``""``. Korean needs the spaces (``알고 있었어``
    vs ``알고있었어``), and 3.80 % of the COO/Manga109 targets carried U+3000 or
    a newline, so the versions differ on the Japanese rows as well —
    ``TARGET_NORM`` records which one a run trained on.
    """
    return textnorm.training_target(s)


def load_split(
    split: str,
    *,
    speech_ratio: float = 1.0,
    limit: int | None = None,
    seed: int = 0,
    extra: list[str] | None = None,
    extra_repeat: int = 1,
    extra_replace: bool = False,
) -> pd.DataFrame:
    """Manifest rows of one split; speech drawn at ``speech_ratio`` × the SFX count.

    ``extra`` = names of sibling manifests (``manifest_<name>.parquet`` — O3's
    colorized crops) whose rows of the same split are **appended** before the
    speech draw / ``limit``, tagged by their ``source`` column, each row
    ``extra_repeat`` times (oversampling a small colorized set).
    ``extra_replace`` instead **swaps** them in: every grey row whose polygon
    (``kind, book, page, id``) an extra row re-cuts is dropped first, so the
    colorized copy stands in for its original and the total stays put.
    """
    df = pd.read_parquet(m109.derived_root() / "manifest.parquet")
    if "source" not in df.columns:
        df["source"] = "grey"
    df = df[df.split == split]
    key = ["kind", "book", "page", "id"]
    for name in extra or []:
        ex = pd.read_parquet(m109.derived_root() / f"manifest_{name}.parquet")
        ex = ex[ex.split == split]
        if extra_replace:
            drop = set(map(tuple, ex[key].values))
            df = df[[t not in drop for t in map(tuple, df[key].values)]]
        df = pd.concat([df] + [ex] * max(1, extra_repeat), ignore_index=True)
    sfx = df[df.kind == "sfx"]
    sp = df[df.kind == "speech"]
    n_sp = min(len(sp), int(round(len(sfx) * speech_ratio)))
    sp = sp.sample(n_sp, random_state=seed) if n_sp < len(sp) else sp
    out = pd.concat([sfx, sp])
    if limit:
        out = pd.concat(
            [
                g.sample(min(limit, len(g)), random_state=seed)
                for _, g in out.groupby("kind")
            ]
        )
    out = cap_targets(out, split)
    out = out[out.target.str.len() > 0]
    return out.sort_values(["source", "kind", "book", "page", "id"]).reset_index(
        drop=True
    )


OCR_TASK, SFX_TASK = "OCR:", "SFX:"
KOZH_SFX = Path(__file__).resolve().parents[1] / "assets" / "kozh_sfx.tsv"


def load_sfxp(
    split: str,
    *,
    neg_share: float = 0.2,
    limit: int | None = None,
    seed: int = 0,
    det_manifest: str = "sfxp_det_clean",
    missed_manifest: str | None = None,
) -> pd.DataFrame:
    """The two-prompt mix (``docs/proposal/ocr_sfx_prefix.md``; crops from
    ``build_sfx_prefix.py``). Column ``task`` is the prompt, ``metric`` the val group.

    * ``OCR:`` — Manga109 speech + KO / ZH (train) + the speech of the detector's
      mixed boxes; negatives (target ``""``) = detector boxes holding only SFX.
    * ``SFX:`` — detector boxes with SFX (mixed boxes: the SFX alone); negatives =
      detector speech boxes and KO / ZH crops, half each where both exist.

    ``missed_manifest`` (``sfxp_missed``, ``build_sfx_prefix.py --missed``) adds the COO
    polygons the detector misses, cut from the polygon, as ``SFX:`` rows (source
    ``missed_<sub>``; rows without a crop skipped).

    ``assets/kozh_sfx.tsv`` (source, id, text): KO / ZH crops that also hold SFX
    lettering. Each gives an ``SFX:`` row with that text (source ``kozh_sfx``) and leaves
    the ``SFX:`` negative pool, which would otherwise train it to ``""``.

    Negatives are at most ``neg_share`` of each task's rows. ``limit`` caps every
    ``metric`` group (val / smoke).
    """
    d = m109.derived_root()
    rng = np.random.default_rng(seed)
    det = pd.read_parquet(d / f"manifest_{det_manifest}.parquet")
    det = det[(det.split == split) & det.path.notna()]
    speech = pd.read_parquet(d / "manifest_sfxp_speech.parquet")
    speech = speech[speech.split == split]
    kozh = pd.read_parquet(d / "manifest_sfxp_kozh.parquet")
    kozh = kozh[kozh.split == split]
    ks = (
        pd.read_csv(KOZH_SFX, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        if KOZH_SFX.exists()
        else pd.DataFrame(columns=["source", "id", "text"])
    )
    ks_text = {
        (s, i): t.replace("\\n", "\n") for s, i, t in zip(ks.source, ks.id, ks.text)
    }
    ks_mask = np.array([k in ks_text for k in zip(kozh.source, kozh.id)], dtype=bool)
    kozh_sfx = kozh[ks_mask]

    def rows(df, task, metric, kind, text, source):
        cols = ["book", "page", "id", "orient", "w", "h", "path"]
        out = df[cols].copy()
        out["text"] = list(text) if not isinstance(text, str) else text
        out["task"], out["metric"], out["kind"] = task, metric, kind
        out["source"] = source if isinstance(source, str) else list(source)
        return out

    def draw(df, n):
        n = int(min(len(df), n))
        return df.iloc[rng.permutation(len(df))[:n]]

    sfx_det = det[det.cat.isin(["pos", "pos_mixed"])]
    mixed_sp = det[(det.cat == "pos_mixed") & det.speech_text.notna()]
    ocr_pos = pd.concat(
        [
            rows(speech, OCR_TASK, "speech", "speech", speech.text, "m109_speech"),
            rows(kozh, OCR_TASK, "speech", "speech", kozh.text, kozh.source),
            rows(
                mixed_sp,
                OCR_TASK,
                "speech",
                "speech",
                mixed_sp.speech_text,
                "det_mixed",
            ),
        ]
    )
    sfx_pos = rows(
        sfx_det, SFX_TASK, "sfx", "sfx", sfx_det.sfx_text, "det_" + sfx_det.cat
    )
    if len(kozh_sfx):
        t = [ks_text[k] for k in zip(kozh_sfx.source, kozh_sfx.id)]
        sfx_pos = pd.concat(
            [sfx_pos, rows(kozh_sfx, SFX_TASK, "sfx", "sfx", t, "kozh_sfx")]
        )
    if missed_manifest:
        mi = pd.read_parquet(d / f"manifest_{missed_manifest}.parquet")
        mi = mi[(mi.split == split) & mi.path.notna()]
        if len(mi):
            sfx_pos = pd.concat(
                [sfx_pos, rows(mi, SFX_TASK, "sfx", "sfx", mi.text, mi.source)]
            )
    cap = neg_share / (1 - neg_share)
    only_sfx = det[det.cat == "pos"]
    ocr_neg = draw(
        rows(only_sfx, OCR_TASK, "ocr_neg", "sfx", "", "det_pos"), cap * len(ocr_pos)
    )
    n_sfx_neg = cap * len(sfx_pos)
    det_sp = det[det.cat == "neg_speech"]
    pools = [p for p in (det_sp, kozh[~ks_mask]) if len(p)]
    sfx_neg = []
    for i, p in enumerate(pools):
        share = n_sfx_neg / len(pools)
        src = "det_speech" if p is det_sp else p.source
        sfx_neg.append(draw(rows(p, SFX_TASK, "sfx_neg", "speech", "", src), share))
    out = pd.concat([ocr_pos, ocr_neg, sfx_pos, *sfx_neg], ignore_index=True)
    out = cap_targets(out, f"sfxp {split}")
    is_neg = out.metric.str.endswith("_neg")
    out = out[is_neg | (out.target.str.len() > 0)]
    if limit:
        out = pd.concat(
            [
                g.sample(min(limit, len(g)), random_state=seed)
                for _, g in out.groupby("metric")
            ]
        )
    return out.sort_values(
        ["task", "metric", "source", "book", "page", "id"]
    ).reset_index(drop=True)


OCR2_TASK = "OCR2:"
ASSETS = Path(__file__).resolve().parents[1] / "assets"
MAINSUB_JSONL = m109.REPO / "output/ocr/mainsub"
KANJI = __import__("re").compile(r"[一-鿿々〆ヶ]")


def load_mainsub(
    split: str,
    *,
    neg_share: float = 0.2,
    limit: int | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """The main / sub mix (``main-sub-approach.md``): ``OCR:`` reads the main text of a
    crop, ``OCR2:`` the rest (furigana, smaller lettering), one item per line.

    * Manga109: the detector crops (``manifest_sfxp_det_all``) with ``m109_mainsub.py``'s
      proposals (``output/ocr/mainsub/m109_mainsub[_val].jsonl``: ``main``, ``subp`` = ruby
      reads, the agents' fixes, other annotated items), a row reviewed on the page
      (``assets/m109_mainsub.tsv``) as reviewed, a dropped one out. An unreviewed ``trim``
      row (its main a box only partly inside) is out, and so is an unreviewed empty main
      (mainsub_1121_1806 read text on all 59 val ones: a box cut under half, a sign).
    * KO / ZH (``manifest_sfxp_kozh``, train only): ``OCR:`` = the label; ``OCR2:`` =
      ``assets/sub_labels.tsv``'s sub, else ``assets/kozh_sfx.tsv``'s SFX text, else ``""``.
    * ``OCR2:`` only where the sub is complete: reviewed, or a proposal on a crop with no
      partly visible text, or ``""`` on a crop whose main holds no kanji (no ruby
      possible). A kanji main with no ruby found is left out of ``OCR2:``.

    Negatives (target ``""``) are at most ``neg_share`` of each prompt's rows; ``metric``
    is ``main`` / ``main_kozh`` / ``main_neg`` / ``sub`` / ``sub_neg``. ``limit`` caps every
    ``metric`` group."""
    import json

    d = m109.derived_root()
    rng = np.random.default_rng(seed)
    rd = lambda p: (  # noqa: E731
        pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        if p.exists()
        else None
    )
    un = lambda t: t.replace("\\n", "\n")  # noqa: E731
    rev = {}
    if (t := rd(ASSETS / "m109_mainsub.tsv")) is not None:
        rev = {
            f"{r.source}:{r.id}": (r.status, un(r.main), un(r.sub))
            for r in t.itertuples()
        }
    subl = {}
    if (t := rd(ASSETS / "sub_labels.tsv")) is not None:
        subl = {(r.source, r.id): un(r.sub) for r in t.itertuples()}
    ks = {}
    if (t := rd(KOZH_SFX)) is not None:
        ks = {(r.source, r.id): un(r.text) for r in t.itertuples()}
    det = pd.read_parquet(d / "manifest_sfxp_det_all.parquet")
    det = det[det.split == split]
    geo = {p: (w, h, o) for p, w, h, o in zip(det.path, det.w, det.h, det.orient)}
    name = "m109_mainsub.jsonl" if split == "train" else f"m109_mainsub_{split}.jsonl"
    recs = []
    n_trim = n_drop = n_rev = n_empty = 0
    for line in open(MAINSUB_JSONL / name, encoding="utf-8"):
        x = json.loads(line)
        if x["path"] not in geo:
            continue
        w, h, o = geo[x["path"]]
        v = rev.get(f"{x['source']}:{x['id']}")
        if v is not None and v[0] == "drop":
            n_drop += 1
            continue
        if v is None and x["trim"]:
            n_trim += 1
            continue
        # an empty main proposal: no annotated text whole inside, yet text shows (a box
        # cut under half, a sign the annotation lacks): "" would teach ignoring it
        if v is None and not x["main"].strip():
            n_empty += 1
            continue
        if v is not None:
            n_rev += 1
            main, sub, done = v[1], v[2], True
        else:
            main, sub = x["main"], x["subp"]
            done = not x["part"] and (bool(sub) or not KANJI.search(main))
        base = dict(
            book=x["book"],
            page=x["page"],
            id=x["id"],
            orient=o,
            w=w,
            h=h,
            path=x["path"],
        )
        recs.append(
            dict(base, text=main, task=OCR_TASK, kind="main", source="m109_" + x["cat"])
        )
        if done:
            recs.append(
                dict(
                    base,
                    text=sub,
                    task=OCR2_TASK,
                    kind="sub",
                    source="m109_" + x["cat"],
                )
            )
    if split == "train":
        kozh = pd.read_parquet(d / "manifest_sfxp_kozh.parquet")
        kozh = kozh[kozh.split == split]
        for r in kozh.itertuples():
            base = dict(
                book=r.book,
                page=r.page,
                id=r.id,
                orient=r.orient,
                w=r.w,
                h=r.h,
                path=r.path,
            )
            k = (r.source, r.id)
            sub = subl.get(k, ks.get(k, ""))
            recs.append(
                dict(
                    base, text=r.text, task=OCR_TASK, kind="main_kozh", source=r.source
                )
            )
            recs.append(
                dict(base, text=sub, task=OCR2_TASK, kind="sub", source=r.source)
            )
    df = pd.DataFrame(recs)
    df = cap_targets(df, f"mainsub {split}")
    empty = df.target.str.len() == 0
    df["metric"] = np.where(
        df.task == OCR_TASK,
        np.where(
            empty, "main_neg", np.where(df.kind == "main_kozh", "main_kozh", "main")
        ),
        np.where(empty, "sub_neg", "sub"),
    )
    cap = neg_share / (1 - neg_share)
    parts = []
    for task in (OCR_TASK, OCR2_TASK):
        t = df[df.task == task]
        pos, neg = t[~t.metric.str.endswith("_neg")], t[t.metric.str.endswith("_neg")]
        n = int(min(len(neg), cap * len(pos)))
        parts += [pos, neg.iloc[rng.permutation(len(neg))[:n]]]
    out = pd.concat(parts, ignore_index=True)
    print(
        f"[data] mainsub {split}: reviewed {n_rev}, dropped {n_drop}, unreviewed trim "
        f"out {n_trim}, unreviewed empty main out {n_empty}; "
        f"{out.groupby(['task', 'metric']).size().to_dict()}",
        flush=True,
    )
    if limit:
        out = pd.concat(
            [
                g.sample(min(limit, len(g)), random_state=seed)
                for _, g in out.groupby("metric")
            ]
        )
    return out.sort_values(
        ["task", "metric", "source", "book", "page", "id"]
    ).reset_index(drop=True)


class CropDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, *, augment: bool, seed: int = 0, tight: bool = False
    ):
        """``tight``: the crops carry the 4 px margin (``load_sfxp``), see ``Augment.tight``."""
        self.df = df.reset_index(drop=True)
        self.tight = tight
        self.derived = m109.derived_root()
        self.paths = [str(self.derived / p) for p in self.df.path]
        self.targets = list(self.df.target)
        self.orients = list(self.df.orient)
        self.area = (self.df.w * self.df.h).to_numpy()
        self.augment = augment
        self.seed = seed
        self._aug: Augment | None = None

    def __len__(self):
        return len(self.df)

    def _get_aug(self) -> Augment:
        if self._aug is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info else 0
            self._aug = Augment(seed=self.seed * 1000 + wid, tight=self.tight)
        return self._aug

    def __getitem__(self, i: int):
        img = cv2.imread(self.paths[i])
        if img is None:
            raise FileNotFoundError(self.paths[i])
        if self.augment:
            img = self._get_aug()(img)
        return img, self.targets[i], i


def area_batches(
    area: np.ndarray, batch_size: int, rng: random.Random
) -> list[list[int]]:
    """Batches of similar crop area (the VL batching rule); batch order shuffled,
    membership jittered by a random tie-break so epochs differ."""
    key = area * np.exp(np.array([rng.gauss(0, 0.15) for _ in range(len(area))]))
    order = np.argsort(key)
    batches = [
        order[s : s + batch_size].tolist() for s in range(0, len(order), batch_size)
    ]
    rng.shuffle(batches)
    return batches


def vl_tokens(
    w: np.ndarray, h: np.ndarray, min_pixels: int, max_pixels: int, factor: int = 28
) -> np.ndarray:
    """Exact patch-token count each crop will cost the VL tower, from its (w, h) alone.

    Mirrors ``transformers…paddleocr_vl.smart_resize``: the processor rounds each
    edge to a multiple of ``factor`` (= patch 14 × merge 2) and then rescales so the
    pixel budget lands inside ``[min_pixels, max_pixels]``. Tokens = pixels / 14².

    The **min_pixels floor is why crop area is a bad batching key** — every crop
    below it is upscaled to the same cost, so sorting by area does not sort by cost.
    """
    w = np.asarray(w, dtype=np.float64).copy()
    h = np.asarray(h, dtype=np.float64).copy()
    small_h = h < factor
    w[small_h] = np.round(w[small_h] * factor / h[small_h])
    h[small_h] = factor
    small_w = w < factor
    h[small_w] = np.round(h[small_w] * factor / w[small_w])
    w[small_w] = factor
    hb = np.round(h / factor) * factor
    wb = np.round(w / factor) * factor
    px = hb * wb
    over = px > max_pixels
    if over.any():
        beta = np.sqrt(h[over] * w[over] / max_pixels)
        hb[over] = np.maximum(factor, np.floor(h[over] / beta / factor) * factor)
        wb[over] = np.maximum(factor, np.floor(w[over] / beta / factor) * factor)
    under = px < min_pixels
    if under.any():
        beta = np.sqrt(min_pixels / (h[under] * w[under]))
        hb[under] = np.ceil(h[under] * beta / factor) * factor
        wb[under] = np.ceil(w[under] * beta / factor) * factor
    return ((hb * wb) / 196).astype(np.int64)


def token_batches(
    tokens: np.ndarray,
    budget: int,
    batch_size: int,
    rng: random.Random,
    *,
    padded: bool = False,
) -> list[list[int]]:
    """Batches capped by **total packed tokens**, not crop count.

    ``padded=True`` caps the padded cost (largest crop × crops) instead of the sum:
    the LM input is padded to the batch's longest row, and the jittered order can
    put one large crop among small ones (sfxp train, budget 6144: sum-capped
    batches reach 10,560 padded, padded-capped ones stay under the budget).

    ``area_batches`` fixes the crop count, so a batch drawn from the large-crop tail
    can cost several times the median and OOM the tower on a spike (measured on
    ``manifest_all``: median 20.1k tokens, max 122.6k, 34 batches over 60k). Sorting
    by token cost and cutting on a budget bounds the peak; because ~99 % of crops sit
    at the processor's min_pixels floor, a 24k budget costs only +0.6 % more steps.

    ``batch_size`` stays the upper bound on crops per batch, so a budget wider than
    ``batch_size`` × median simply reproduces the old behaviour.
    """
    key = tokens * np.exp(np.array([rng.gauss(0, 0.15) for _ in range(len(tokens))]))
    order = np.argsort(key)
    batches: list[list[int]] = []
    cur: list[int] = []
    run = top = 0
    for i in order:
        t = int(tokens[i])
        cost = max(top, t) * (len(cur) + 1) if padded else run + t
        if cur and (cost > budget or len(cur) >= batch_size):
            batches.append(cur)
            cur, run, top = [], 0, 0
        cur.append(int(i))
        run += t
        top = max(top, t)
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def collate_raw(items):
    imgs, targets, idx = zip(*items)
    return list(imgs), list(targets), list(idx)
