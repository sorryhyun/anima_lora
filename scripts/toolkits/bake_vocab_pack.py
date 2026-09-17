#!/usr/bin/env python
"""bake_vocab_pack — fold a wake-line ext-row delta (``trained.pt``) into a
vocab pack, producing a new ``.safetensors`` + ``.json`` pair.

The render line (``project/cjk_renderable_anima/``) trains a delta on the
pack's ext rows through the ``ExtDelta`` hook. That hook adds
``raw[i] * row_scale * scale`` to the pack row of every ext id in
``delta.ext_ids`` at lookup time. Baking applies the same sum to the stored
table once, so the pair loads through every existing pack surface with no
hook: the ``vocab_pack`` config key, ``--vocab_pack`` on inference / TE
caching, ``GenerationRequest``, and the ComfyUI ``AnimaVocabPackLoader``
node (>= 3.9.1). Tokenization is untouched — the json keeps every routing
map — so EN prompts stay bit-exact and only the summed rows differ from the
base pack (``pack_digest`` changes, which is what makes stale TE caches and
LoRA stamps warn).

    .venv/bin/python scripts/toolkits/bake_vocab_pack.py \\
        output/wake_probe/rows_synth_sent_q_sent_s24k \\
        --out models/vocab_packs/anima_cjk_vocab_pack_sent_s24k

writes ``<out>/<stem>.safetensors`` + ``.json`` (``--out`` may be a directory,
in which case the stem is ``<base stem>_<arm_tag>``; each pack gets its own
directory because ``vocab_pack`` accepts a directory holding exactly one pair).
``--comfy_dir`` additionally symlinks the pair into a ComfyUI ``vocab_packs``
folder.

The json gains a ``render`` block (source arm, ext ids, row → piece text,
scale, base pack digest, git rev) and the safetensors header carries the
same summary under ``anima_render``; ``provenance`` marks the summed rows
``render``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from library.anima.ext_vocab import T5_TABLE_SIZE, pack_digest  # noqa: E402
from library.env import resolve_under_home  # noqa: E402

PROVENANCE_TIER = "render"


def load_delta(path: Path) -> dict:
    """``trained.pt`` (or the arm dir holding one) → its ``delta`` record plus
    the run's provenance fields."""
    path = Path(path)
    if path.is_dir():
        path = path / "trained.pt"
    sd = torch.load(path, map_location="cpu", weights_only=False)
    d = sd["delta"]
    ids = [int(e) for e in d["ext_ids"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ext ids in delta")
    raw = d["raw"].float()
    if raw.shape[0] != len(ids):
        raise ValueError(f"{path}: raw has {raw.shape[0]} rows for {len(ids)} ids")
    c_flat = sd.get("c_flat")
    if c_flat is not None and float(torch.as_tensor(c_flat).float().norm()) > 0:
        # c_flat is a train-time-only vector on flat-canvas items; eval and
        # native never add it (ExtDelta.common stays None), so the baked table
        # is the evaluated one without it. Say so rather than silently drop.
        print(
            f"note: {path} carries a non-zero c_flat (‖{float(torch.as_tensor(c_flat).float().norm()):.3f}‖); "
            "it is a training-batch vector and is NOT baked (matches eval / native).",
            flush=True,
        )
    args = sd.get("args") or {}
    return {
        "path": str(path),
        "arm": sd.get("arm"),
        "ext_ids": ids,
        "raw": raw,
        "row_scale": float(d["row_scale"]),
        "arm_tag": args.get("arm_tag"),
        "data_tag": args.get("data_tag"),
        "train_steps": args.get("train_steps"),
        "units": args.get("units"),
        "init_rows": args.get("init_rows"),
    }


def bake(
    table: torch.Tensor,
    mapping: dict,
    delta: dict,
    scale: float = 1.0,
) -> tuple[torch.Tensor, dict, dict]:
    """``(baked table, baked mapping, summary)``.

    ``table`` is the base pack's *stored* ``ext_embed`` (not iso-materialised:
    a regenerated iso block cannot carry a delta, and the stored row count is
    what the json's ``rows`` describes). Rows outside ``delta.ext_ids`` are
    returned byte-identical; listed rows get ``raw * row_scale * scale`` added
    in float32 and are cast back to the table's dtype.
    """
    ids = delta["ext_ids"]
    n_rows = int(table.shape[0])
    bad = [e for e in ids if e < 0 or e >= n_rows]
    if bad:
        raise ValueError(
            f"{len(bad)} delta ext ids fall outside the base pack's {n_rows} stored "
            f"rows (first: {bad[:5]}) — was the delta trained on a different pack?"
        )
    raw = delta["raw"]
    if raw.shape[1] != table.shape[1]:
        raise ValueError(f"delta dim {raw.shape[1]} != pack dim {table.shape[1]}")
    idx = torch.tensor(ids, dtype=torch.long)
    out = table.clone()
    add = raw * (delta["row_scale"] * float(scale))
    out[idx] = (table[idx].float() + add).to(table.dtype)

    m = json.loads(json.dumps(mapping, ensure_ascii=False))
    if int(m.get("rows", n_rows)) != n_rows:
        raise ValueError(f"base json says {m['rows']} rows, table has {n_rows}")
    prov = m.get("provenance")
    if isinstance(prov, list) and len(prov) == n_rows:
        for e in ids:
            prov[e] = PROVENANCE_TIER
    summary = {
        "source": delta["path"],
        "arm": delta["arm"],
        "arm_tag": delta["arm_tag"],
        "data_tag": delta["data_tag"],
        "train_steps": delta["train_steps"],
        "units": delta["units"],
        "init_rows": delta["init_rows"],
        "rows": len(ids),
        "row_scale": delta["row_scale"],
        "scale": float(scale),
        "delta_norm_mean": float(add.norm(dim=1).mean()),
        "delta_norm_max": float(add.norm(dim=1).max()),
        "t5_table_size": T5_TABLE_SIZE,
    }
    return out, m, summary


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:  # noqa: BLE001 — provenance is best effort
        return ""


def _row_texts(mapping: dict, ids: list[int]) -> dict[int, str]:
    """ext id → piece text, from the pack's own maps (char / symbol rows) and
    the Qwen tokenizer for Qwen-piece rows. Best effort: a missing tokenizer
    leaves Qwen rows unnamed rather than stopping the bake."""
    inv_c = {int(v): k for k, v in (mapping.get("char") or {}).items()}
    inv_s = {int(v): k for k, v in (mapping.get("sym_char") or {}).items()}
    inv_q = {int(v): int(k) for k, v in (mapping.get("qwen") or {}).items()}
    inv_q.update({int(v): int(k) for k, v in (mapping.get("sym") or {}).items()})
    qtok = None
    if any(e in inv_q for e in ids):
        try:
            from library.env import default_checkpoints
            from library.inference.text import ensure_text_strategies

            tok, _ = ensure_text_strategies(default_checkpoints().text_encoder)
            qtok = tok.qwen3_tokenizer
        except Exception as e:  # noqa: BLE001
            print(f"(qwen piece names unavailable: {e})", flush=True)
    out = {}
    for e in ids:
        t = ""
        if e in inv_q and qtok is not None:
            t = qtok.decode([inv_q[e]]).strip()
        elif e in inv_c:
            t = inv_c[e]
        elif e in inv_s:
            t = inv_s[e]
        if t:
            out[e] = t
    return out


def write_pack(
    prefix: Path,
    table: torch.Tensor,
    mapping: dict,
    base_meta: dict,
    summary: dict,
    label: str,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    from safetensors.torch import save_file

    st = prefix.with_suffix(".safetensors")
    js = prefix.with_suffix(".json")
    if (st.exists() or js.exists()) and not overwrite:
        raise FileExistsError(f"{prefix}.{{safetensors,json}} exists; pass --overwrite")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: str(v) for k, v in (base_meta or {}).items()}
    meta.update(
        {
            "format": "pt",
            "anima_pack_type": "cjk_vocab_pack",
            "anima_pack_label": label,
            "anima_pack_source": prefix.name,
            "anima_date": _dt.date.today().isoformat(),
            "anima_git_rev": _git_rev(),
            "anima_rows": str(int(table.shape[0])),
            "anima_dim": str(int(table.shape[1])),
            "anima_base_table_size": str(T5_TABLE_SIZE),
            "anima_render": json.dumps(summary, ensure_ascii=False),
        }
    )
    save_file({"ext_embed": table.contiguous()}, str(st), metadata=meta)
    js.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    return st, js


def link_into(comfy_dir: Path, st: Path, js: Path, overwrite: bool) -> None:
    comfy_dir.mkdir(parents=True, exist_ok=True)
    for src in (st, js):
        dst = comfy_dir / src.name
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                raise FileExistsError(f"{dst} exists; pass --overwrite")
            dst.unlink()
        os.symlink(src.resolve(), dst)
        print(f"linked {dst} -> {src.resolve()}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("table", help="trained.pt, or the arm dir holding one")
    p.add_argument(
        "--base",
        default=None,
        help="base pack prefix / file / dir (default: the configured vocab_pack)",
    )
    p.add_argument(
        "--out",
        required=True,
        help="output prefix, or a directory (stem = <base stem>_<arm_tag>)",
    )
    p.add_argument("--label", default=None, help="anima_pack_label (default: the stem)")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="ExtDelta scale to bake (1 = as evaluated)",
    )
    p.add_argument(
        "--comfy_dir",
        default=None,
        help="also symlink the pair into this ComfyUI vocab_packs folder",
    )
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    from safetensors import safe_open

    from library.anima.vocab_pack import default_vocab_pack, resolve_pack_prefix

    base = resolve_pack_prefix(a.base if a.base is not None else default_vocab_pack())
    if base is None:
        raise SystemExit("no base pack: pass --base or set vocab_pack")
    with safe_open(str(base.with_suffix(".safetensors")), "pt") as f:
        table = f.get_tensor("ext_embed")
        base_meta = f.metadata() or {}
    mapping = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    base_digest = pack_digest(table, mapping)

    delta = load_delta(resolve_under_home(a.table))
    baked, m, summary = bake(table, mapping, delta, a.scale)
    summary["base_pack"] = {"name": base.name, "sha": base_digest}
    summary["git_rev"] = _git_rev()
    summary["date"] = _dt.date.today().isoformat()
    row_text = _row_texts(mapping, delta["ext_ids"])
    m["render"] = dict(
        summary,
        ext_ids=delta["ext_ids"],
        row_text={str(k): v for k, v in row_text.items()},
    )

    out = resolve_under_home(a.out)
    if out.suffix in (".safetensors", ".json"):
        out = out.with_suffix("")
    if out.is_dir() or str(a.out).endswith(("/", os.sep)):
        stem = f"{base.name}_{delta['arm_tag'] or 'render'}"
        out = out / stem
    label = a.label or out.name
    st, js = write_pack(out, baked, m, base_meta, summary, label, a.overwrite)
    new_digest = pack_digest(baked, m)
    print(
        f"baked {summary['rows']} rows from {delta['path']} (arm {delta['arm']}, "
        f"scale {a.scale:g}, mean |Δ| {summary['delta_norm_mean']:.2f}) onto {base.name}\n"
        f"  -> {st}\n  -> {js}\n"
        f"  base sha {base_digest[:12]}…  baked sha {new_digest[:12]}…  "
        f"named rows {len(row_text)}/{summary['rows']}",
        flush=True,
    )
    if a.comfy_dir:
        link_into(Path(a.comfy_dir).expanduser(), st, js, a.overwrite)


if __name__ == "__main__":
    main()
