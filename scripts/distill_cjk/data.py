"""Pairs, span alignment, and the on-disk cache reader (CPU only).

The corpus (``post_image_dataset/cjk_distill/pairs.jsonl``, built by
``scripts/distill_cjk/corpus/build_pairs.py``) is a list of
``{en, ja, register, spans}`` records. ``spans`` is what makes the span loss
possible: D1 captions are *composed* tag-by-tag from the glossary, so each EN
tag maps to exactly one JA tag in the same order. This module turns that free
alignment into token-index sets on both sides, and carries the per-span
provenance weight along with it.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

T5_TABLE_SIZE = 32128


@dataclass
class Span:
    """One aligned tag: teacher token indices, student token indices, weight."""

    teacher: list[int]
    student: list[int]
    weight: float
    via: str
    f1: float


@dataclass
class EncodedPair:
    pid: str
    register: str
    qwen_ids: torch.Tensor  # [L] JA, max-padded
    qwen_mask: torch.Tensor
    t_ids: torch.Tensor  # teacher target: stock spiece on the EN translation
    t_mask: torch.Tensor
    s_ids: torch.Tensor  # student target: hybrid ext encoder on the JA text
    s_mask: torch.Tensor
    spans: list[Span]
    # Eval-only arms (holdout split): the all-EN reference the Phase-0 probe
    # measures against, and today's unk-wall. Together with the teacher they
    # turn a bare cosine into a *recovery fraction* — how much of the
    # addressable gap the student actually closed.
    ref_qwen_ids: torch.Tensor | None = None
    ref_qwen_mask: torch.Tensor | None = None
    n_ids: torch.Tensor | None = None  # stock spiece on the JA text
    n_mask: torch.Tensor | None = None


def load_pairs(
    path: Path,
    *,
    registers: tuple[str, ...] = (),
    max_pairs: int = 0,
    holdout: int = 0,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """(train, held-out) pair records — deterministic split, **grouped by image**.

    The split is over images, not pairs, because every image contributes several
    pairs that share their tag content and differ only in wording (`tags` /
    `tags_alt`, and D6's two quote registers). Splitting per pair broke both
    things the holdout is for:

    * **Leakage** — a pair's sibling register lands in train ~91% of the time,
      so "held-out" measured generalization to new *wording of trained content*,
      not to new content.
    * **Near-pair starvation** — the `near` half of the stratified
      discrimination needs both registers of one image inside the eval slice. A
      per-pair split left exactly **one** such pair in a 256-record slice, so
      `discrimination_near` was a single-sample statistic (that is the provenance
      of the 0.71 zero-shot figure quoted in ``plan.md``).

    Grouping fixes both at once: whole images travel together, so no sibling
    crosses the boundary and every held-out image contributes its own near pair.
    ``holdout`` stays a pair budget — images are added until it is reached.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if registers and r.get("register") not in registers:
                continue
            rows.append(r)
    if not rows:
        raise ValueError(f"no pairs in {path} for registers={registers or 'ALL'}")

    by_image = _group(rows)
    keys = sorted(by_image)
    rng = random.Random(seed)
    rng.shuffle(keys)

    held: list[dict] = []
    train: list[dict] = []
    for k in keys:
        # Whole image to one side; `holdout` is a pair budget, so the last image
        # admitted may overshoot it slightly.
        (held if len(held) < holdout else train).extend(by_image[k])
    rng.shuffle(train)
    if max_pairs:
        train = train[:max_pairs]

    n_near = sum(
        {"tags", "tags_alt"} <= {r["register"] for r in grp}
        for grp in _group(held).values()
    )
    logger.info(
        "pairs: %d train / %d held-out over %d images (registers=%s); "
        "held-out images carrying a tags/tags_alt near pair: %d",
        len(train),
        len(held),
        len(keys),
        ",".join(registers) if registers else "ALL",
        n_near,
    )
    return train, held


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in rows:
        out.setdefault(r["id"].rsplit("/", 1)[0], []).append(r)
    return out


def _en_span_chars(text: str, segments: list[str]) -> list[tuple[int, int]]:
    """Char spans of each EN segment inside the raw caption (sequential scan)."""
    out, cursor = [], 0
    for seg in segments:
        i = text.find(seg, cursor)
        if i < 0:  # composition drifted from the caption — skip this span
            out.append((-1, -1))
            continue
        out.append((i, i + len(seg)))
        cursor = i + len(seg)
    return out


# Pre-2026-08-30 records carry no ``joiner`` field and were built with 、;
# newer builders (build_pairs / synth_names) record the joiner per pair.
LEGACY_JOINER = "、"


def _ja_span_chars(
    segments: list[str], joiner: str = LEGACY_JOINER
) -> list[tuple[int, int]]:
    """Char spans inside ``joiner.join(segments)`` — exact by construction."""
    out, pos = [], 0
    for seg in segments:
        out.append((pos, pos + len(seg)))
        pos += len(seg) + len(joiner)
    return out


def _tokens_in_span(
    offsets: list[tuple[int, int]], mask, span: tuple[int, int]
) -> list[int]:
    lo, hi = span
    if lo < 0 or hi <= lo:
        return []
    return [
        j
        for j, (a, b) in enumerate(offsets)
        if int(mask[j]) == 1 and b > a and a < hi and b > lo
    ]


class PairEncoder:
    """Tokenizes both arms and aligns their spans. No model weights involved."""

    def __init__(self, text_encoder_path: str, ext_prefix: Path, max_length: int = 512):
        import sys

        repo = Path(__file__).resolve().parents[2]
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from bench.cjk_adapter import ext_vocab
        from library.anima import strategy as strategy_anima

        self.max_length = max_length
        self.text_encoder_path = text_encoder_path
        self.ext_prefix = Path(ext_prefix)
        self.tok = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=text_encoder_path,
            qwen3_max_length=max_length,
            t5_max_length=max_length,
        )
        self.ext_init, self.mapping = ext_vocab.load_ext_assets(ext_prefix)
        self.ext = ext_vocab.HybridT5Encoder.from_mapping(
            self.tok.t5_tokenizer, self.tok.qwen3_tokenizer, self.mapping
        )

    def encode(
        self, pair: dict, trust: dict[str, float], *, eval_arms: bool = False
    ) -> EncodedPair:
        en, ja = pair["en"], pair["ja"]

        qwen = self.tok.qwen3_tokenizer(
            ja,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        teacher = self.tok.t5_tokenizer(
            en,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        t_ids = teacher["input_ids"][0]
        t_mask = teacher["attention_mask"][0]
        t_offsets = [(int(a), int(b)) for a, b in teacher["offset_mapping"][0].tolist()]

        s_ids_l, s_mask_l, s_offsets = self.ext.encode_aligned(ja, self.max_length)
        s_ids = torch.tensor(s_ids_l, dtype=torch.long)
        s_mask = torch.tensor(s_mask_l, dtype=torch.long)
        # `encode_aligned` returns offsets for real tokens only; pad the tail so
        # index lookups stay positional.
        s_offsets = s_offsets + [(-1, -1)] * (len(s_ids_l) - len(s_offsets))

        spans: list[Span] = []
        raw = pair.get("spans") or []
        if raw:
            en_chars = _en_span_chars(en, [s["en"] for s in raw])
            ja_chars = _ja_span_chars(
                [s["ja"] for s in raw], joiner=pair.get("joiner") or LEGACY_JOINER
            )
            for s, ec, jc in zip(raw, en_chars, ja_chars):
                t_idx = _tokens_in_span(t_offsets, t_mask, ec)
                s_idx = _tokens_in_span(s_offsets, s_mask, jc)
                if not t_idx or not s_idx:
                    continue
                via = str(s.get("via") or "unknown")
                w = float(trust.get(via, 1.0))
                if w <= 0.0:
                    continue
                spans.append(Span(t_idx, s_idx, w, via, float(s.get("f1") or 0.0)))

        out = EncodedPair(
            pid=str(pair.get("id") or ""),
            register=str(pair.get("register") or ""),
            qwen_ids=qwen["input_ids"][0],
            qwen_mask=qwen["attention_mask"][0],
            t_ids=t_ids,
            t_mask=t_mask,
            s_ids=s_ids,
            s_mask=s_mask,
            spans=spans,
        )
        if eval_arms:
            ref = self.tok.qwen3_tokenizer(
                en,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            native = self.tok.t5_tokenizer(
                ja,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            out.ref_qwen_ids = ref["input_ids"][0]
            out.ref_qwen_mask = ref["attention_mask"][0]
            out.n_ids = native["input_ids"][0]
            out.n_mask = native["attention_mask"][0]
        return out


# ---------------------------------------------------------------------------
# Cache reader
# ---------------------------------------------------------------------------


@dataclass
class CachedPair:
    pid: str
    register: str
    qwen: torch.Tensor  # [Lq, D] trimmed to non-pad
    teacher: torch.Tensor  # [Lt, D] trimmed to non-pad
    s_ids: torch.Tensor  # [Ls] trimmed to non-pad (EOS included)
    t_ids: torch.Tensor  # [Lt]
    spans: list[Span]
    ref: torch.Tensor | None = None  # all-EN reference arm (holdout only)
    native: torch.Tensor | None = None  # today's unk-wall arm (holdout only)


class CachedPairs:
    """Reader over the shards written by ``cache.py``.

    Everything is trimmed to non-pad on disk. That is safe *because* target
    pads are masked inside the adapter (``target_attention_mask`` gates both
    the self- and cross-attention), so a non-pad output does not depend on how
    many pads follow it — asserted by ``tests/test_cjk_distill.py``. The
    (512-N) zero rows the DiT actually sees are re-introduced analytically by
    the attention loss, which is the only place their count matters.
    """

    def __init__(self, cache_dir, split: str = "train"):
        from safetensors.torch import load_file

        # One dir, or a sequence of dirs staged separately (plan_ko K2: the
        # JA cache is untouched and KO lands in its own `cache_ko` — joint
        # training concatenates the records; each record remembers its dir).
        dirs = (
            [Path(cache_dir)]
            if isinstance(cache_dir, (str, Path))
            else [Path(d) for d in cache_dir]
        )
        self.dirs = [d / split for d in dirs]
        self.dir = self.dirs[0]  # primary, kept for messages/back-compat
        self.records = []
        self.meta: dict = {"pairs": self.records}
        for d in self.dirs:
            meta_path = d / "meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"no cache at {d} — run `python -m scripts.distill_cjk.cache`"
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for rec in meta["pairs"]:
                rec["_dir"] = str(d)
            self.records.extend(meta["pairs"])
            for k, v in meta.items():
                if k != "pairs":
                    self.meta.setdefault(k, v)
        self._shards: dict[str, dict] = {}
        self._span_cache: dict[int, dict] = {}
        self._load_file = load_file

    def __len__(self) -> int:
        return len(self.records)

    def apply_trust(self, weights: dict[str, float] | None) -> None:
        """Re-derive every span's weight from its ``via``, in place.

        The cache bakes one trust policy in at build time, which would make the
        G4 ablation a cache rebuild per arm. It does not have to be: ``via`` is
        stored alongside the weight, so the policy can be re-applied on load.

        **One asymmetry to know**: a policy that maps a `via` to 0 *drops* the
        span, and the build-time policy already dropped its own zeros (under
        `provenance`: `unresolved`, `unmapped`). Those cannot be resurrected
        here, so this re-weights the surviving spans only — which is exactly the
        G4 question (does demoting `mt_unverified` help?), but it means `all`
        means "every *mappable* span at 1.0", not "every span".
        """
        if weights is None:
            return
        for rec in self.records:
            spans = rec.get("spans")
            if not spans:
                continue
            kept = []
            for s in spans:
                w = float(weights.get(str(s[3]), 1.0))
                if w <= 0.0:
                    continue
                kept.append([s[0], s[1], w, s[3], s[4]])
            rec["spans"] = kept
        self._span_cache.clear()

    def _shard(self, name: str, dir_: str | None = None) -> dict:
        path = str(Path(dir_ or self.dir) / name)
        if path not in self._shards:
            self._shards[path] = self._load_file(path)
        return self._shards[path]

    def _span_index(self, i: int) -> dict:
        """Flat (token, span) index tensors for one pair — built once, reused.

        The span loss pools ~40 spans per pair. Doing that as a Python loop is
        ~1600 tiny CUDA launches per step and leaves the GPU 70% idle, so the
        pooling is expressed as two ``index_add``s instead; these are the
        gather indices they need. Static per pair, so they are memoized here
        rather than rebuilt per epoch.
        """
        if i in self._span_cache:
            return self._span_cache[i]
        raw = self.records[i].get("spans", [])
        s_idx, s_seg, t_idx, t_seg, w = [], [], [], [], []
        for seg, s in enumerate(raw):
            t_idx.extend(s[0])
            t_seg.extend([seg] * len(s[0]))
            s_idx.extend(s[1])
            s_seg.extend([seg] * len(s[1]))
            w.append(float(s[2]))
        out = {
            "s_idx": torch.tensor(s_idx, dtype=torch.long),
            "s_seg": torch.tensor(s_seg, dtype=torch.long),
            "t_idx": torch.tensor(t_idx, dtype=torch.long),
            "t_seg": torch.tensor(t_seg, dtype=torch.long),
            "w": torch.tensor(w, dtype=torch.float32),
        }
        self._span_cache[i] = out
        return out

    def get(self, i: int) -> CachedPair:
        rec = self.records[i]
        sd = self._shard(rec["shard"], rec.get("_dir"))
        k = rec["key"]
        spans = [
            Span(list(s[0]), list(s[1]), float(s[2]), str(s[3]), float(s[4]))
            for s in rec.get("spans", [])
        ]
        return CachedPair(
            pid=rec["id"],
            register=rec["register"],
            qwen=sd[f"{k}.qwen"],
            teacher=sd[f"{k}.teacher"],
            s_ids=sd[f"{k}.sids"].long(),
            t_ids=sd[f"{k}.tids"].long(),
            spans=spans,
            ref=sd.get(f"{k}.ref"),
            native=sd.get(f"{k}.native"),
        )

    def all_student_ids(self):
        for i in range(len(self)):
            yield self.get(i).s_ids

    def batch(self, indices, device, dtype) -> dict:
        """One collated batch, with the vectorized span pack attached."""
        return collate(
            [self.get(i) for i in indices],
            device,
            dtype,
            span_index=[self._span_index(i) for i in indices],
        )


def collate(batch: list[CachedPair], device, dtype, span_index=None) -> dict:
    """Right-pad a batch to its own max lengths (not to 512 — see CachedPairs)."""
    B = len(batch)
    lq = max(p.qwen.shape[0] for p in batch)
    lt = max(p.teacher.shape[0] for p in batch)
    ls = max(p.s_ids.shape[0] for p in batch)
    d = batch[0].qwen.shape[-1]

    qwen = torch.zeros(B, lq, d, dtype=dtype)
    qwen_mask = torch.zeros(B, lq, dtype=torch.long)
    teacher = torch.zeros(B, lt, d, dtype=dtype)
    teacher_mask = torch.zeros(B, lt, dtype=torch.long)
    s_ids = torch.zeros(B, ls, dtype=torch.long)
    s_mask = torch.zeros(B, ls, dtype=torch.long)
    t_ids = torch.zeros(B, lt, dtype=torch.long)

    for i, p in enumerate(batch):
        n = p.qwen.shape[0]
        qwen[i, :n] = p.qwen.to(dtype)
        qwen_mask[i, :n] = 1
        m = p.teacher.shape[0]
        teacher[i, :m] = p.teacher.to(dtype)
        teacher_mask[i, :m] = 1
        t_ids[i, :m] = p.t_ids[:m]
        k = p.s_ids.shape[0]
        s_ids[i, :k] = p.s_ids
        s_mask[i, :k] = 1

    out = {
        "qwen": qwen.to(device),
        "qwen_mask": qwen_mask.to(device),
        "teacher": teacher.to(device),
        "teacher_mask": teacher_mask.to(device),
        "s_ids": s_ids.to(device),
        "s_mask": s_mask.to(device),
        "t_ids": t_ids.to(device),
        "spans": [p.spans for p in batch],
        "registers": [p.register for p in batch],
    }
    if span_index is not None:
        # Flatten every span in the batch into one gather: token positions are
        # addressed in the flattened [B*L, D] view, and `seg` says which span
        # each entry pools into. Two index_adds replace ~40×B tiny kernels.
        s_flat, s_seg, t_flat, t_seg, w = [], [], [], [], []
        offset = 0
        for i, si in enumerate(span_index):
            n = si["w"].numel()
            if n == 0:
                continue
            s_flat.append(si["s_idx"] + i * ls)
            s_seg.append(si["s_seg"] + offset)
            t_flat.append(si["t_idx"] + i * lt)
            t_seg.append(si["t_seg"] + offset)
            w.append(si["w"])
            offset += n
        out["span_pack"] = (
            {
                "s_flat": torch.cat(s_flat).to(device),
                "s_seg": torch.cat(s_seg).to(device),
                "t_flat": torch.cat(t_flat).to(device),
                "t_seg": torch.cat(t_seg).to(device),
                "w": torch.cat(w).to(device),
                "n_spans": offset,
            }
            if offset
            else None
        )
    if batch[0].ref is not None:
        for name in ("ref", "native"):
            lengths = [getattr(p, name).shape[0] for p in batch]
            buf = torch.zeros(B, max(lengths), d, dtype=dtype)
            msk = torch.zeros(B, max(lengths), dtype=torch.long)
            for i, p in enumerate(batch):
                n = lengths[i]
                buf[i, :n] = getattr(p, name).to(dtype)
                msk[i, :n] = 1
            out[name] = buf.to(device)
            out[f"{name}_mask"] = msk.to(device)
    return out
