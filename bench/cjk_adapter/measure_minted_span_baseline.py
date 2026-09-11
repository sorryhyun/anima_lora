#!/usr/bin/env python3
"""Pre-mint baseline for the minted-word spans (ko2 O2 smoke diagnostics).

Measures the span loss of exactly the spans whose ja side is a minted word,
under the ORIGINAL spelled-out encoding + the joint synthjako2 pack (the
old cache), so smoke2's focused numbers (init 0.500 -> final 0.189) have a
reference: was 0.189 an improvement over composition, or a regression?
"""

from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from library.anima import weights as anima_weights  # noqa: E402
from scripts.distill_cjk import ext_table, losses as loss_mod  # noqa: E402
from scripts.distill_cjk.data import CachedPairs  # noqa: E402
from scripts.distill_cjk.distill import student_forward  # noqa: E402

MINTED = {
    "하쿠레이",
    "레이무",
    "동방프로젝트",
    "소류",
    "아스카",
    "랑그레이",
    "에반게리온",
    "쌍둥이",
    "무녀복",
    "커플룩",
}
BASE = REPO / "post_image_dataset" / "cjk_distill"

# label=cache_dir[,cache_dir]:pack_prefix — the smoke's three conditions.
SMOKE_ARMS = [
    "pre-mint (spelled-out, joint pack)="
    f"{BASE / 'cache_ko'},{BASE / 'cache_desc_ko'}:"
    f"{REPO / 'output' / 'ckpt' / 'cjk_vocab' / 'cjk_vocab_pack_synthjako2'}",
    f"mint init (pooled)={BASE / 'cache_mint_smoke'}:{HERE / 'assets' / 'ext_embed_mint'}",
    f"mint trained (smoke2)={BASE / 'cache_mint_smoke'}:"
    f"{REPO / 'output' / 'ckpt' / 'cjk_vocab' / 'cjk_vocab_pack_mint_smoke2'}",
]


def wanted_spans(corpus: Path, minted: set[str]) -> dict[str, list[int]]:
    """pair id -> indices of spans whose ja side is a minted word."""
    out: dict[str, list[int]] = {}
    for line in open(corpus, encoding="utf-8"):
        r = json.loads(line)
        idxs = [
            k
            for k, s in enumerate(r.get("spans") or [])
            if s.get("ja") in minted or s.get("ja", "").strip() in minted
        ]
        if idxs:
            out[r["id"]] = idxs
    return out


@torch.no_grad()
def masked_span_loss(cached, ids, adapter, device, dtype, span_sel, label):
    losses = []
    B = 32
    for start in range(0, len(ids), B):
        chunk = ids[start : start + B]
        b = cached.batch(chunk, device, dtype)
        pk = b["span_pack"]
        # keep only selected spans: rebuild per-batch keep mask by walking the
        # same order collate flattened them in (records in chunk order, spans
        # in record order).
        keep = []
        for i in chunk:
            rec = cached.records[i]
            sel = set(span_sel.get(rec["id"], []))
            n = len(rec.get("spans") or [])
            keep.extend(1.0 if k in sel else 0.0 for k in range(n))
        keep_t = torch.tensor(keep, device=device)
        if keep_t.numel() != pk["w"].numel():
            raise RuntimeError(
                f"span count mismatch {keep_t.numel()} vs {pk['w'].numel()}"
            )
        pk = {**pk, "w": pk["w"] * keep_t}
        if float(pk["w"].sum()) == 0:
            continue
        student = student_forward(adapter, b)
        losses.append(
            (
                float(loss_mod.loss_span(student, b["teacher"], pk)),
                float(pk["w"].sum()),
            )
        )
    total = sum(v * w for v, w in losses) / max(sum(w for _, w in losses), 1e-8)
    print(
        f"{label}: weighted span loss over minted-word spans = {total:.4f} ({len(ids)} pairs)"
    )
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus",
        type=Path,
        default=BASE / "pairs_mint_smoke.jsonl",
        help="pairs jsonl whose minted-word spans define the measured set",
    )
    ap.add_argument(
        "--mapping",
        type=Path,
        default=None,
        help="mint mapping json — minted surfaces come from its word map "
        "(default: the hardcoded smoke set)",
    )
    ap.add_argument(
        "--arm",
        action="append",
        default=None,
        metavar="LABEL=CACHE[,CACHE]:PACK",
        help="condition to measure (repeatable; default: the smoke's three)",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    minted = MINTED
    if args.mapping is not None:
        minted = set(json.loads(args.mapping.read_text(encoding="utf-8"))["word"])
    sel = wanted_spans(args.corpus, minted)
    print(f"{len(sel)} pairs carry a minted-word span ({len(minted)} surfaces)")

    from anima_lora import default_checkpoints

    adapter = anima_weights.load_llm_adapter(
        default_checkpoints().dit, dtype=dtype, device="cpu"
    )
    adapter.to(device).eval()

    def run(cache_dirs, pack, label):
        from bench.cjk_adapter import ext_vocab

        table_w, _ = ext_vocab.load_ext_assets(Path(pack))
        table = ext_table.ExtTable(
            table_w, mode="row", tunable_rows=torch.arange(0)
        ).to(device)
        if isinstance(adapter.embed, ext_table.SplitEmbedding):
            adapter.embed.table = table
        else:
            ext_table.attach(adapter, table)
        cached = CachedPairs([Path(d) for d in cache_dirs], "train")
        ids = [i for i, r in enumerate(cached.records) if r["id"] in sel]
        return masked_span_loss(cached, ids, adapter, device, dtype, sel, label)

    for arm in args.arm or SMOKE_ARMS:
        label, rest = arm.split("=", 1)
        caches, pack = rest.rsplit(":", 1)
        run(caches.split(","), pack, label)


if __name__ == "__main__":
    main()
