"""Char-row separability probe — glyph-line ceilings, see done2.md (CPU-only).

Glyph rendering needs visually distinct kanji (鎧 vs 錯) to arrive at
cross-attn as *distinct vectors*, not just distinct ids. The `param=global`
correction has no per-row freedom so it should not be able to collapse rows —
this measures that instead of assuming it. Measured verdicts + the
byte-permutation init collision it surfaced (fixed in
`ext_vocab.build_ext_table`) are recorded in
project/cjk_aware_anima/done2.md.

The measured set is every **single-character glyph row** in the ext table:
the union of qwen-map tokens whose decoded surface is one CJK char (where the
common kanji/kana live) and the whole char map (the 28,017-row byte-fragment
tail, whose plain-mean init could collide byte-permutation char pairs).

Reports, per table (Qwen-init and optionally a trained pack):
  * pairwise-cosine distribution over all glyph rows (full, chunked matmul);
  * nearest-neighbor cosine distribution (how close is each char's *closest*
    confusable), bucketed by script (kanji / kana / hangul / other);
  * near-duplicate pairs above thresholds, decoded + script-classed — the
    human-review list for "visually distinct but embedding-indistinguishable";
  * a handpicked visually-confusable kanji pair list (鎧/錯 class).

Usage (CPU)::

    python project/cjk_aware_anima/gates/separability.py \
        --pack output/ckpt/cjk_vocab_pack_item2.safetensors \
        --out project/cjk_aware_anima/assets/separability_phase02.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Visually-confusable kanji pairs (same stroke skeleton, different glyph) —
# the class the renderer must keep apart. Cos near 1.0 here would mean the
# text encoder cannot tell the DiT which of the two to draw.
CONFUSABLE_PAIRS = [
    ("鎧", "錯"),
    ("未", "末"),
    ("士", "土"),
    ("日", "曰"),
    ("人", "入"),
    ("力", "刀"),
    ("己", "已"),
    ("千", "干"),
    ("王", "玉"),
    ("大", "犬"),
    ("体", "休"),
    ("水", "氷"),
    ("右", "石"),
    ("名", "各"),
    ("会", "合"),
]

THRESHOLDS = (0.80, 0.90, 0.95, 0.99)


def script_of(ch: str) -> str:
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF:
        return "kanji"
    if 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:
        return "kana"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "hangul"
    return "other"


def load_glyph_rows(
    st_path: Path, json_path: Path, qwen_tok
) -> tuple[torch.Tensor, list[str], list[str]]:
    """All single-char glyph rows: qwen 1-char surfaces ∪ the char map.

    Returns (rows, chars, provenance) with provenance "qwen" or "char".
    A char present in both maps keeps its qwen row (the encoder resolves
    single-token chars through the qwen map first).
    """
    mapping = json.load(open(json_path))
    qwen_map = {int(k): v for k, v in mapping["qwen"].items()}
    ids = sorted(qwen_map)
    surfaces = qwen_tok.batch_decode([[i] for i in ids])
    chars: list[str] = []
    rows_idx: list[int] = []
    prov: list[str] = []
    seen: set[str] = set()
    for qid, s in zip(ids, surfaces):
        s = s.strip()
        if len(s) == 1 and s not in seen:
            seen.add(s)
            chars.append(s)
            rows_idx.append(qwen_map[qid])
            prov.append("qwen")
    for ch, row in mapping["char"].items():
        if ch not in seen:
            seen.add(ch)
            chars.append(ch)
            rows_idx.append(row)
            prov.append("char")
    table = load_file(str(st_path))["ext_embed"]
    return table[torch.tensor(rows_idx, dtype=torch.long)].float(), chars, prov


def analyze(rows: torch.Tensor, chars: list[str], chunk: int = 2048) -> dict:
    n = rows.shape[0]
    scripts = [script_of(c) for c in chars]
    unit = torch.nn.functional.normalize(rows, dim=1)
    hist = torch.zeros(200, dtype=torch.float64)
    nn_cos = torch.full((n,), -2.0)
    nn_idx = torch.zeros(n, dtype=torch.long)
    over: dict[float, int] = {t: 0 for t in THRESHOLDS}
    near_pairs: list[tuple[float, int, int]] = []
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        sim = unit[lo:hi] @ unit.T
        rows_idx = torch.arange(lo, hi)
        sim[rows_idx - lo, rows_idx] = -2.0  # mask self
        vals, idxs = sim.max(dim=1)
        nn_cos[lo:hi], nn_idx[lo:hi] = vals, idxs
        # upper triangle only for pair-level stats (j > i)
        cols = torch.arange(n)
        sim[cols.unsqueeze(0) <= rows_idx.unsqueeze(1)] = -2.0
        valid = sim > -1.5
        hist += torch.histc(sim[valid], bins=200, min=-1.0, max=1.0).double()
        for t in THRESHOLDS:
            over[t] += int((sim > t).sum())
        for r, c in (sim > 0.90).nonzero().tolist():
            near_pairs.append((float(sim[r, c]), lo + r, c))
    near_pairs.sort(reverse=True)
    q = torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])

    def pair_class(i: int, j: int) -> str:
        return "-".join(sorted((scripts[i], scripts[j])))

    by_class: dict[str, dict[str, int]] = {}
    for c, i, j in near_pairs:
        cls = by_class.setdefault(pair_class(i, j), {str(t): 0 for t in THRESHOLDS[1:]})
        for t in THRESHOLDS[1:]:
            if c > t:
                cls[str(t)] += 1

    def decode(pairs, k):
        return [
            {
                "cos": round(c, 4),
                "a": chars[i],
                "b": chars[j],
                "class": pair_class(i, j),
            }
            for c, i, j in pairs[:k]
        ]

    kanji_pairs = [p for p in near_pairs if pair_class(p[1], p[2]) == "kanji-kanji"]
    nn_by_script = {
        s: {
            "n": scripts.count(s),
            "nn_cos_median": float(
                torch.median(nn_cos[[i for i, x in enumerate(scripts) if x == s]])
            ),
        }
        for s in ("kanji", "kana", "hangul", "other")
        if s in scripts
    }
    return {
        "n_chars": n,
        "pairs": n * (n - 1) // 2,
        "pair_cos_hist_bins": 200,
        "pair_cos_hist": hist.tolist(),
        "pair_cos_mean": float(
            (hist * torch.linspace(-0.995, 0.995, 200).double()).sum() / hist.sum()
        ),
        "nn_cos_quantiles": dict(
            zip([f"p{int(x * 100)}" for x in q], torch.quantile(nn_cos, q).tolist())
        ),
        "nn_by_script": nn_by_script,
        "chars_with_nn_over": {str(t): int((nn_cos > t).sum()) for t in THRESHOLDS},
        "pairs_over": {str(t): c for t, c in over.items()},
        "pairs_over_by_class": by_class,
        "top_pairs": decode(near_pairs, 200),
        "top_kanji_pairs": decode(kanji_pairs, 200),
        "_nn_cos": nn_cos,
        "_nn_idx": nn_idx,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="bench/cjk_adapter/assets/ext_embed.safetensors")
    ap.add_argument("--init_json", default="bench/cjk_adapter/assets/ext_embed.json")
    ap.add_argument(
        "--pack", default=None, help="trained pack .safetensors (json sibling assumed)"
    )
    ap.add_argument("--out", default=None, help="write full JSON report here")
    args = ap.parse_args()

    from anima_lora import default_checkpoints  # noqa: PLC0415
    from library.anima.weights import load_qwen3_tokenizer  # noqa: PLC0415

    qwen_tok = load_qwen3_tokenizer(str(default_checkpoints().text_encoder))

    report: dict = {}
    init_rows, chars, prov = load_glyph_rows(
        REPO / args.init, REPO / args.init_json, qwen_tok
    )
    n_q = prov.count("qwen")
    print(
        f"[init] {args.init}: {len(chars)} glyph rows ({n_q} qwen 1-char + {len(chars) - n_q} char-map)"
    )
    init_stats = analyze(init_rows, chars)
    report["init"] = init_stats

    pack_stats = None
    pack_rows = None
    if args.pack:
        pack_path = REPO / args.pack
        pack_rows, pack_chars, _ = load_glyph_rows(
            pack_path, pack_path.with_suffix(".json"), qwen_tok
        )
        assert pack_chars == chars, "glyph row sets differ between init and pack"
        print(f"[pack] {args.pack}: {len(pack_chars)} glyph rows")
        pack_stats = analyze(pack_rows, chars)
        report["pack"] = pack_stats
        moved = torch.nn.functional.cosine_similarity(init_rows, pack_rows, dim=1)
        churn = int((init_stats["_nn_idx"] != pack_stats["_nn_idx"]).sum())
        report["init_vs_pack"] = {
            "row_cos_init_pack_min": float(moved.min()),
            "row_cos_init_pack_p1": float(torch.quantile(moved, 0.01)),
            "row_cos_init_pack_median": float(moved.median()),
            "nn_identity_churn": churn,
            "nn_identity_churn_frac": churn / len(chars),
        }

    cmap = {c: i for i, c in enumerate(chars)}
    conf = []
    for a, b in CONFUSABLE_PAIRS:
        if a not in cmap or b not in cmap:
            conf.append({"a": a, "b": b, "note": "no glyph row"})
            continue
        ia, ib = cmap[a], cmap[b]
        row = {
            "a": a,
            "b": b,
            "rows": f"{prov[ia]}/{prov[ib]}",
            "cos_init": round(
                float(
                    torch.nn.functional.cosine_similarity(
                        init_rows[ia], init_rows[ib], dim=0
                    )
                ),
                4,
            ),
        }
        if pack_rows is not None:
            row["cos_pack"] = round(
                float(
                    torch.nn.functional.cosine_similarity(
                        pack_rows[ia], pack_rows[ib], dim=0
                    )
                ),
                4,
            )
        conf.append(row)
    report["confusable_pairs"] = conf

    for name in ("init", "pack"):
        s = report.get(name)
        if not s:
            continue
        print(f"\n== {name} ==")
        print(
            f"  pair cos mean {s['pair_cos_mean']:.4f}; NN cos quantiles { ({k: round(v, 3) for k, v in s['nn_cos_quantiles'].items()}) }"
        )
        print(f"  NN by script: {s['nn_by_script']}")
        print(f"  chars with NN over {s['chars_with_nn_over']}")
        print(
            f"  pairs over threshold {s['pairs_over']} (of {s['pairs']:,}); by class {s['pairs_over_by_class']}"
        )
        print("  top near-duplicate pairs (any script):")
        for p in s["top_pairs"][:10]:
            print(f"    {p['cos']:.4f}  {p['a']} / {p['b']}  [{p['class']}]")
        print("  top kanji-kanji near-duplicates:")
        for p in s["top_kanji_pairs"][:20]:
            print(f"    {p['cos']:.4f}  {p['a']} / {p['b']}")
    if args.pack:
        print(f"\n== init vs pack == {report['init_vs_pack']}")
    print("\n== confusable pairs (visually-similar class) ==")
    for row in conf:
        print(f"  {row}")

    if args.out:
        for s in (init_stats, pack_stats):
            if s:
                s.pop("_nn_cos"), s.pop("_nn_idx")
        out = REPO / args.out
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
