#!/usr/bin/env python
"""merge_tables — union two or more rows-arm tables into one ``trained.pt``.

Why a file and not just ``--init_rows a.pt,b.pt``: the comma list merges at
*train* time, so the merged table cannot be rendered, read or shipped on its
own. This writes the union as an ordinary ``rows`` arm dir, so every stage that
takes an arm dir (``eval``, ``native``, ``classify``, the benches) reads it, and
``--init_rows`` can warm-start from the single file.

Two things the merge has to get right:

- **row-norm units.** ``ExtDelta`` adds ``raw × row_scale``, and ``row_scale``
  is the mean pack-row norm *as measured by the run that trained it* — it
  differs run to run (53k: 196.407, punct-only: 191.911). Rows coming from a
  source are rescaled by ``rs_src / rs_base`` so the delta each row applies is
  bit-for-bit what its own run applied.
- **ext-id order.** ``ExtDelta.__init__`` sorts ``ext_ids`` and ``load()``
  asserts the saved list equals that sorted list, so the union is written
  sorted with ``raw`` in the same order.

On an id present in more than one source, ``--on_overlap keep-base`` (the
default) keeps the base table's row: a short single-purpose run (punct-only,
3k steps, あ as its anchor) must not overwrite a row the long run trained on
the full inventory. ``override`` gives the last source the row instead.

    .venv/bin/python project/cjk_renderable_anima/probes/merge_tables.py \
        --base output/wake_probe/rows_synth_full_fm10k_full_s53k_qoff/trained.pt \
        --add  output/wake_probe/rows_synth_punct_only_punct_only_s3k/trained.pt \
        --out  rows_synth_full_fm10k_merge_punct

Writes ``<out>/trained.pt`` (delta only — no optimizer state, no encoder) and
``<out>/merge.json`` (provenance: every source, its row count, which ids it
contributed, which collided and who won).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from wake.common import OUT  # noqa: E402


def load_table(path: Path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    assert sd.get("arm") == "rows", f"{path}: arm {sd.get('arm')!r}, expected 'rows'"
    d = sd["delta"]
    ids = [int(e) for e in d["ext_ids"]]
    assert len(set(ids)) == len(ids), f"{path}: duplicate ext ids"
    return sd, ids, d["raw"].float(), float(d["row_scale"])


def row_text_map(ids: list[int]) -> dict:
    """ext id → piece text via the shipped pack. Reporting only — a failure
    here must not stop the merge."""
    try:
        from library.anima.vocab_pack import load_vocab_pack
        from library.env import default_checkpoints
        from library.inference.text import ensure_text_strategies

        from wake.encoder import row_texts

        ck = default_checkpoints()
        tok, _ = ensure_text_strategies(ck.text_encoder, vocab_pack=ck.vocab_pack)
        return row_texts(tok, load_vocab_pack(ck.vocab_pack), ids)
    except Exception as e:  # noqa: BLE001 — provenance is best effort
        print(f"(row text decode unavailable: {e})", flush=True)
        return {}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base", required=True, help="the table whose row_scale wins")
    p.add_argument(
        "--add", action="append", default=[], help="table to merge in (repeatable)"
    )
    p.add_argument(
        "--on_overlap",
        choices=["keep-base", "override"],
        default="keep-base",
        help="an ext id present in several sources: keep the base row (default) "
        "or let the last source win",
    )
    p.add_argument(
        "--out",
        required=True,
        help="arm dir: a name under output/wake_probe/ or an explicit path. "
        "Name it rows_<data_tag>_<arm_tag> so --stage native/eval finds it",
    )
    a = p.parse_args()

    base_path = Path(a.base)
    sd, base_ids, base_raw, rs = load_table(base_path)
    rows = {e: base_raw[i] for i, e in enumerate(base_ids)}
    origin = {e: base_path.parent.name for e in base_ids}
    print(
        f"base {base_path.parent.name}: {len(base_ids)} rows, row_scale {rs:.3f}, "
        f"norm mean {base_raw.norm(dim=1).mean():.3f}",
        flush=True,
    )

    prov, collisions = [], []
    for src in a.add:
        sp = Path(src)
        _, ids, raw, rs_src = load_table(sp)
        k = rs_src / rs
        hit = sorted(set(ids) & set(rows))
        added, replaced = [], []
        for i, e in enumerate(ids):
            if e in rows and a.on_overlap == "keep-base":
                continue
            rows[e] = raw[i] * k
            (replaced if e in origin else added).append(e)
            origin[e] = sp.parent.name
        print(
            f"+ {sp.parent.name}: {len(ids)} rows, row_scale {rs_src:.3f} "
            f"(× {k:.5f} into base units), norm mean {raw.norm(dim=1).mean():.3f} "
            f"→ {len(added)} new, {len(replaced)} replaced, "
            f"{len(hit) - len(replaced)} collisions kept from base",
            flush=True,
        )
        collisions += hit
        prov.append(
            {
                "path": str(sp),
                "arm_dir": sp.parent.name,
                "n_rows": len(ids),
                "row_scale": rs_src,
                "rescale": k,
                "added": added,
                "replaced": replaced,
                "collided": hit,
            }
        )

    ext_ids = sorted(rows)
    raw = torch.stack([rows[e] for e in ext_ids])
    out = Path(a.out) if "/" in a.out else OUT / a.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "delta": {"ext_ids": ext_ids, "raw": raw, "row_scale": rs},
            "arm": "rows",
            "merged_from": [str(base_path)] + [str(Path(s)) for s in a.add],
            "killed": "",
        },
        out / "trained.pt",
    )

    text = row_text_map(ext_ids)
    for rec in prov:
        rec["added_text"] = {str(e): text.get(e, "?") for e in rec["added"]}
        rec["collided_text"] = {str(e): text.get(e, "?") for e in rec["collided"]}
    json.dump(
        {
            "base": {
                "path": str(base_path),
                "arm_dir": base_path.parent.name,
                "n_rows": len(base_ids),
                "row_scale": rs,
            },
            "on_overlap": a.on_overlap,
            "n_rows_out": len(ext_ids),
            "sources": prov,
        },
        open(out / "merge.json", "w"),
        ensure_ascii=False,
        indent=1,
    )
    n = raw.norm(dim=1)
    print(
        f"\n{out.name}: {len(ext_ids)} rows "
        f"({len(ext_ids) - len(base_ids)} more than the base), row_scale {rs:.3f}, "
        f"norm mean {n.mean():.3f} max {n.max():.3f}",
        flush=True,
    )
    for rec in prov:
        got = [f"{text.get(e, '?')}" for e in rec["added"]]
        print(f"  from {rec['arm_dir']}: {' '.join(got) if got else '(nothing new)'}")
    if collisions:
        who = "base" if a.on_overlap == "keep-base" else "the last source"
        print(
            f"  collisions ({who} won): "
            + " ".join(f"{text.get(e, '?')}({e})" for e in sorted(set(collisions)))
        )


if __name__ == "__main__":
    main()
