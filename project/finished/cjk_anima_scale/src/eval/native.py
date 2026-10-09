"""Stages ``native`` and ``target`` — a kana clause hung off ordinary scene
prompts (does the address survive outside the template? — the product
condition W3 needs; the EN-reference renders its ruler scores against are
drawn on the way), and the user's own captions rendered verbatim.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from common.hooks import ExtDelta
from common.models import (
    encode_captions,
    ext_ids_of,
    generate_to,
    load_generator,
    load_trained,
    load_vae,
)
from common.paths import arm_dir
from common.prompts import NATIVE_CLAUSES
from common.readers import Readers, contact_sheet, hit, read_scored
from common.shapes import parse_shape
from common.text import CJK_RE

from .enref import EnRef, enref_boxes, enref_dir, enref_file, render_enref
from .stage import blank_cell, sheet_row


def stage_native(a):
    """Render the blind-pairs scene prompts with a kana clause appended, delta
    off (floor) and on (trained, × ``--delta_scale``), same seeds; read; sheet
    per (prompt, kana)."""
    import torch

    sd = load_trained(arm_dir(a))
    assert "lora" not in sd, "native covers rows-only arms"
    out = arm_dir(a) / (f"native_{a.eval_tag}" if a.eval_tag else "native")
    (out / "img").mkdir(parents=True, exist_ok=True)
    prompts = [
        ln.strip()
        for ln in Path(a.native_prompts).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if a.native_limit:
        prompts = prompts[: a.native_limit]
    chars = [c for c in a.native_chars.split(",") if c]
    clauses = [c for c in a.native_clauses.split(",") if c]
    items = [
        {
            "pi": pi,
            "prompt": p,
            "text": k,
            "clause": cl,
            "caption": NATIVE_CLAUSES[cl].format(p=p, k=k),
        }
        for pi, p in enumerate(prompts)
        for k in chars
        for cl in clauses
    ]
    size = parse_shape(a.eval_shape) if a.eval_shape else a.eval_size
    args, gen, device, shared = load_generator(size, a.steps, a.cfg, out / "img")
    anima = shared["model"]
    anima.eval()
    delta = ExtDelta.from_state(anima, sd["delta"], device)
    trained = set(delta.ext_ids)
    # which ext rows each caption touches, and how many of them carry a delta:
    # a JA clause whose tokenizer merges 「あ」 into one piece misses the row
    cache = encode_captions([it["caption"] for it in items], device)
    for it in items:
        ids = sorted(ext_ids_of({it["caption"]: cache[it["caption"]]}))
        it["ext_rows"] = len(ids)
        it["trained_rows"] = len([x for x in ids if x in trained])
    for cl in clauses:
        xs = [it for it in items if it["clause"] == cl]
        print(
            f"clause {cl}: ext rows/caption {sum(x['ext_rows'] for x in xs) / len(xs):.1f}, "
            f"trained rows/caption {sum(x['trained_rows'] for x in xs) / len(xs):.2f}",
            flush=True,
        )
    del cache
    vae = load_vae(device)
    render_enref(a, prompts, a.seeds, args, gen, shared, vae, device)
    manifest = []
    t0 = time.time()
    parts = {"trained": sd["delta"]["raw"]}
    conds = list(parts)
    for cond in conds:
        delta.scale = a.delta_scale
        delta.raw.data.copy_(parts[cond].to(delta.raw.device))
        shared["conds_cache"].clear()
        for it in items:
            for seed in range(a.seeds):
                fn = (
                    out
                    / "img"
                    / f"{cond}_p{it['pi']:02d}_{it['text']}_{it['clause']}_s{seed}.png"
                )
                generate_to(fn, args, gen, shared, vae, device, it["caption"], seed)
                manifest.append({"file": str(fn), "cond": cond, "seed": seed, **it})
    print(
        f"native gen: {len(manifest)} images in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    del anima, vae, shared
    torch.cuda.empty_cache()
    _read_native(a, out, manifest, chars, clauses, conds)


TARGET_QUOTE = re.compile(r'["「]([^"」]+)["」]')


def target_items(path: Path) -> list[dict]:
    """One item per non-comment line of ``--target_prompts``: the caption
    verbatim, the expected text = its quoted span (``"…"`` or ``「…」``)."""
    items = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        m = TARGET_QUOTE.search(ln)
        if not m:
            print(f"target: no quoted text, skipped: {ln}", flush=True)
            continue
        items.append(
            {
                "pi": len(items),
                "prompt": ln,
                "text": m.group(1),
                "clause": "verbatim",
                "caption": ln,
            }
        )
    return items


def stage_target(a):
    """The user's own target captions (``--target_prompts``, one full caption
    per line, e.g. the ComfyUI prompts of 2026-09-17: hoshino ai by @akipeko
    saying はい), rendered verbatim — floor (delta off) unless ``--no_floor``,
    then the trained table — at ``--eval_shape`` (WxH) or
    ``--eval_size``², read by both readers. No EN-reference / scene-kept rulers:
    the question is only whether the picture says the quoted text."""
    import torch

    from common.shapes import parse_shape

    sd = load_trained(arm_dir(a))
    assert "lora" not in sd, "target covers rows-only arms"
    out = arm_dir(a) / (f"target_{a.eval_tag}" if a.eval_tag else "target")
    (out / "img").mkdir(parents=True, exist_ok=True)
    items = target_items(Path(a.target_prompts))
    if a.native_limit:
        items = items[: a.native_limit]
    assert items, f"no target captions in {a.target_prompts}"
    chars = sorted({it["text"] for it in items}, key=[it["text"] for it in items].index)
    clauses = ["verbatim"]
    size = parse_shape(a.eval_shape) if a.eval_shape else a.eval_size
    args, gen, device, shared = load_generator(
        size, a.steps, a.cfg, out / "img", a.negative
    )
    anima = shared["model"]
    anima.eval()
    delta = ExtDelta.from_state(anima, sd["delta"], device)
    trained = set(delta.ext_ids)
    cache = encode_captions([it["caption"] for it in items], device)
    for it in items:
        ids = sorted(ext_ids_of({it["caption"]: cache[it["caption"]]}))
        it["ext_rows"] = len(ids)
        it["trained_rows"] = len([x for x in ids if x in trained])
        print(
            f"target p{it['pi']:02d} {it['text']}: ext rows {it['ext_rows']}, "
            f"trained {it['trained_rows']} — {it['caption']}",
            flush=True,
        )
    del cache
    vae = load_vae(device)
    parts = {"trained": sd["delta"]["raw"]}
    conds = ([] if a.no_floor else ["floor"]) + list(parts)
    manifest = []
    t0 = time.time()
    for cond in conds:
        if cond == "floor":
            delta.scale = 0.0
        else:
            delta.scale = a.delta_scale
            delta.raw.data.copy_(parts[cond].to(delta.raw.device))
        shared["conds_cache"].clear()
        for it in items:
            for seed in range(a.seeds):
                fn = out / "img" / f"{cond}_p{it['pi']:02d}_{it['text']}_s{seed}.png"
                generate_to(fn, args, gen, shared, vae, device, it["caption"], seed)
                manifest.append({"file": str(fn), "cond": cond, "seed": seed, **it})
    print(
        f"target gen: {len(manifest)} images in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    del anima, vae, shared
    torch.cuda.empty_cache()
    _read_native(
        a,
        out,
        manifest,
        chars,
        clauses,
        conds,
        rulers=False,
        title="target (the user's captions, verbatim)",
        prompts_path=a.target_prompts,
    )


def _read_native(
    a,
    out: Path,
    manifest,
    chars,
    clauses,
    conds,
    *,
    rulers=True,
    title="native (scene prompts + kana clause)",
    prompts_path=None,
):
    """``rulers=False`` (the ``target`` stage): readers only — no EN-reference
    or scene-kept scoring (both are keyed by the native prompt index and would
    read another prompt set's refs)."""
    rd = Readers(a.device)
    for m in manifest:
        reads = read_scored(rd, m)
        m["hit_sfx"] = hit(reads, m["text"], "sfx")
        m["hit_vl"] = hit(reads, m["text"], "vl")
        m["exact"] = m["hit_sfx"] and m["hit_vl"]
        m["any_cjk"] = any(CJK_RE.search(r["sfx"] or "") for r in reads)
    # EN-reference ruler (eval/enref.py): refs rendered by stage_native;
    # boxes read once and cached beside them
    ed = enref_dir(a)
    enref = None
    if rulers and any(ed.glob("enref_p*_s*.png")):
        enref = EnRef(a.device, ed, enref_boxes(ed, rd, a.device))
    del rd
    for m in manifest:
        m["en_cos"] = m["en_cos_out"] = m["box_iou"] = None
        sc = enref.score(m) if enref is not None else None
        if sc is not None:
            m["en_cos"], m["en_cos_out"], m["box_iou"] = sc
    # the scene-kept ruler is retired (no floor renders beside a native run);
    # its fields stay in the reads, empty, so the files keep their shape
    for m in manifest:
        m["kept_cos"] = m["canvas_cos"] = m["kept"] = None
        m["hit_kept"] = False
    (out / "native_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1)
    )
    agg = defaultdict(list)
    for m in manifest:
        agg[(m["clause"], m["cond"])].append(m)
    trained_conds = [c for c in conds if c != "floor"]
    lines = [
        f"# wake_probe — arm `{a.arm}` {title}",
        "",
        f"prompts: `{prompts_path or a.native_prompts}`; chars {' '.join(chars)}; "
        f"{a.seeds} seed(s); {a.eval_shape or f'{a.eval_size}²'}; "
        f"delta scale {a.delta_scale}; parts {' '.join(trained_conds)}",
        "",
        "scene-kept ruler: off (no floor renders; --native_floor 1 to restore)",
        "",
        (
            f"EN-reference ruler: same prompt and seed rendered with `English text reads as "
            f"'{a.en_word}'` (`{ed}`); en cos = PE-Spatial cos to it, en cos out = the same over "
            "patch tokens outside both text boxes, box IoU = glyph box vs the EN word's box "
            "(means over the cond; floor row = the ruler's own floor)"
            if enref is not None
            else "EN-reference ruler: no refs rendered (run --stage enref)"
        ),
        "",
        "| clause | cond | n | CER sfx | CER vl16 | hit sfx | hit vl | both | any CJK read | floor cos | canvas cos | kept | hit & kept | en cos | en cos out | box IoU |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def _en_cols(ms):
        es = [m for m in ms if m["en_cos"] is not None]
        if not es:
            return "– | – | –"
        return (
            f"{sum(m['en_cos'] for m in es) / len(es):.3f} | "
            f"{sum(m['en_cos_out'] for m in es) / len(es):.3f} | "
            f"{sum(m['box_iou'] for m in es) / len(es):.2f}"
        )

    def _kept_cols(ms):
        return "– | – | – | –"

    for cl in clauses:
        for c in conds:
            ms = agg.get((cl, c), [])
            if not ms:
                continue
            lines.append(
                f"| {cl} | {c} | {len(ms)} | {sum(m['cer_sfx'] for m in ms) / len(ms):.3f} | "
                f"{sum(m['cer_vl'] for m in ms) / len(ms):.3f} | {sum(m['hit_sfx'] for m in ms)} | "
                f"{sum(m['hit_vl'] for m in ms)} | {sum(m['exact'] for m in ms)} | "
                f"{sum(m['any_cjk'] for m in ms)} | {_kept_cols(ms)} | {_en_cols(ms)} |"
            )
    lines += [
        "",
        "per kana (both readers hit / kept / hit & kept | en cos out / box IoU):",
        "",
    ]
    for c in trained_conds:
        for k in chars:
            for cl in clauses:
                ms = [
                    m
                    for m in manifest
                    if m["text"] == k and m["clause"] == cl and m["cond"] == c
                ]
                if ms:
                    lines.append(
                        f"- {c} {k} / {cl}: {sum(m['exact'] for m in ms)} / "
                        f"{sum(bool(m['kept']) for m in ms)} / {sum(m['hit_kept'] for m in ms)} "
                        f"of {len(ms)} | {_en_cols(ms)}"
                    )
    lines += [
        "",
        "per prompt (both readers hit / kept / hit & kept, all kana/clauses):",
        "",
    ]
    for c in trained_conds:
        by_p = defaultdict(list)
        for m in manifest:
            if m["cond"] == c:
                by_p[m["pi"]].append(m)
        for pi in sorted(by_p):
            ms = by_p[pi]
            lines.append(
                f"- {c} p{pi:02d} `{ms[0]['prompt']}`: {sum(m['exact'] for m in ms)} / "
                f"{sum(bool(m['kept']) for m in ms)} / {sum(m['hit_kept'] for m in ms)} of {len(ms)}"
            )
    # the EN reference (`English text reads as "<word>"`, same prompt + seed)
    # leads every seed's cells on the sheet, so the ruler's target is in view
    sheet_conds = (["enref"] if enref is not None else []) + list(conds)
    lines += [
        "",
        "Sheets: sheet_<kana>_<clause>.png — one row per prompt: "
        + ", ".join(f"{c} s{s}" for s in range(a.seeds) for c in sheet_conds)
        + "; label = prompt idx / sfx read / vl16 read"
        + (f" (enref = the `{a.en_word}` reference render)." if enref else "."),
    ]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    by_key = {
        (m["pi"], m["text"], m["clause"], m["seed"], m["cond"]): m for m in manifest
    }

    def _enref_row(pi: int, seed: int):
        ref = enref_file(ed, pi, seed)
        if not ref.exists():
            return None
        reads = enref_reads.get(ref.name) or []
        return sheet_row(
            {"file": str(ref), "reads": [r for r in reads if not r.get("whole")]},
            f"p{pi:02d} enref s{seed}: {a.en_word}",
        )

    enref_reads = {}
    if enref is not None and (ed / "enref_reads.json").exists():
        enref_reads = json.loads((ed / "enref_reads.json").read_text())
    for k in chars:
        for cl in clauses:
            rows = []
            for pi in sorted(by_p):
                for seed in range(a.seeds):
                    cells = [
                        sheet_row(
                            by_key[pi, k, cl, seed, c],
                            f"p{pi:02d} {c} s{seed}: {k}"
                            + (
                                f" e{by_key[pi, k, cl, seed, c]['en_cos_out']:.2f}"
                                if by_key[pi, k, cl, seed, c]["en_cos_out"] is not None
                                else ""
                            ),
                        )
                        for c in conds
                        if (pi, k, cl, seed, c) in by_key
                    ]
                    if not cells:
                        continue
                    if enref is not None:
                        cells.insert(
                            0,
                            _enref_row(pi, seed)
                            or blank_cell(
                                parse_shape(a.eval_shape)
                                if a.eval_shape
                                else a.eval_size
                            ),
                        )
                    rows += cells
            if rows:
                contact_sheet(
                    rows,
                    out / f"sheet_{k}_{cl}.png",
                    thumb=192,
                    cols=len(sheet_conds) * a.seeds,
                )
    if out.name in ("native", "target"):
        from .summary import summarize_quietly

        summarize_quietly(arm_dir(a))
