"""Stages ``native``, ``enref`` and ``native_rescore`` — a kana clause hung off
ordinary scene prompts (does the address survive outside the template? — the
product condition W3 needs), the EN-reference renders its ruler scores
against, and a re-read of an existing run.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path

from common.hooks import ExtDelta, OutVec, load_out_vec
from common.models import (
    encode_captions,
    ext_ids_of,
    generate_to,
    load_generator,
    load_trained,
    load_vae,
)
from common.paths import arm_dir, data_dir
from common.prompts import NATIVE_CLAUSES
from common.readers import Readers, contact_sheet, hit, read_scored
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
    args, gen, device, shared = load_generator(a.eval_size, a.steps, a.cfg, out / "img")
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
    parts = table_parts(sd, [x for x in a.delta_parts.split(",") if x])
    # floor (delta off) is opt-in since the EN-reference ruler (2026-09-15):
    # `en cos` against the "hi" render is the scene ruler now
    conds = (["floor"] if a.native_floor and not a.no_floor else []) + list(parts)
    outvec = None
    fq = {}
    saved_q = sd.get("out_vec")
    if saved_q is not None:
        # trained with Q fixed on: every trained cond renders with it
        outvec = OutVec(anima, device)
        print(
            f"native: + saved out_vec ‖{float(saved_q.norm()):.2f}‖ (Q on)", flush=True
        )
    if a.out_vec:
        # quote-probe cond: rows f + the pretrained quoted-EN output shift
        vhat, norm = load_out_vec(a.out_vec, a.out_vec_frame)
        f_rows = sd["delta"]["raw"]
        for sc in [float(x) for x in a.out_vec_scales.split(",") if x]:
            fq[f"fq{sc:g}"] = (f_rows, vhat * (sc * norm))
        outvec = outvec or OutVec(anima, device)
        conds += list(fq)
        print(
            f"out_vec: frame {a.out_vec_frame}, EN shift norm {norm:.2f}, scales {sorted(fq)}",
            flush=True,
        )
    for cond in conds:
        if outvec is not None:
            outvec.set(
                fq[cond][1] if cond in fq else (saved_q if cond != "floor" else None)
            )
        if cond == "floor":
            delta.scale = 0.0
        elif cond in fq:
            delta.scale = a.delta_scale
            delta.raw.data.copy_(fq[cond][0].to(delta.raw.device))
        else:
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
    then every ``--delta_parts`` cond — at ``--eval_shape`` (WxH) or
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
    args, gen, device, shared = load_generator(size, a.steps, a.cfg, out / "img")
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
    parts = table_parts(sd, [x for x in a.delta_parts.split(",") if x])
    conds = ([] if a.no_floor else ["floor"]) + list(parts)
    outvec = None
    saved_q = sd.get("out_vec")
    if saved_q is not None:
        outvec = OutVec(anima, device)
    manifest = []
    t0 = time.time()
    for cond in conds:
        if outvec is not None:
            outvec.set(None if cond == "floor" else saved_q)
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


def _native_prompts(a) -> list[str]:
    prompts = [
        ln.strip()
        for ln in Path(a.native_prompts).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    return prompts[: a.native_limit] if a.native_limit else prompts


def stage_enref(a):
    """Render the EN-reference images only (shared across arms; no delta)."""
    import torch

    prompts = _native_prompts(a)
    d = enref_dir(a)
    d.mkdir(parents=True, exist_ok=True)
    args, gen, device, shared = load_generator(a.eval_size, a.steps, a.cfg, d)
    shared["model"].eval()
    vae = load_vae(device)
    render_enref(a, prompts, a.seeds, args, gen, shared, vae, device)
    del shared, vae
    torch.cuda.empty_cache()
    boxes = enref_boxes(d, None, a.device)
    reads = json.loads((d / "enref_reads.json").read_text())
    n_hit = sum(
        hit(v, a.en_word, "sfx") or hit(v, a.en_word, "vl") for v in reads.values()
    )
    print(
        f"enref: {len(boxes)} refs, {sum(b is not None for b in boxes.values())} with a "
        f"detector box, {n_hit} read as `{a.en_word}`",
        flush=True,
    )


def stage_native_rescore(a):
    """Re-score an existing native run from its ``native_reads.json`` (no
    re-render, no re-OCR): kept + the EN-reference columns, report + sheets."""
    out = arm_dir(a) / (f"native_{a.eval_tag}" if a.eval_tag else "native")
    manifest = json.loads((out / "native_reads.json").read_text())
    assert manifest and all("reads" in m for m in manifest), f"{out}: no stored reads"
    chars = [c for c in a.native_chars.split(",") if c] or sorted(
        {m["text"] for m in manifest}, key=[m["text"] for m in manifest].index
    )
    clauses = sorted({m["clause"] for m in manifest})
    conds = list(dict.fromkeys(m["cond"] for m in manifest))
    _read_native(a, out, manifest, chars, clauses, conds, reread=False)


def table_parts(sd: dict, names: list[str]) -> dict:
    """``{cond name: raw table}`` for the requested parts of an encoder arm's
    saved table, ``raw = g + c + f`` (``g`` centred across rows, ``c`` the
    common vector broadcast, ``f`` the per-row residual). ``full`` keeps the
    name ``trained`` so older reports and file names stay comparable."""

    raw = sd["delta"]["raw"]
    out = {}
    comp = None
    for name in names:
        if name == "full":
            out["trained"] = raw
            continue
        if comp is None and "c_flat" in sd:
            # S-line rows arm: raw = f (the rows), c = c_flat (flat-only switch)
            c = sd["c_flat"].to(raw.dtype).expand_as(raw)
            comp = {"f": raw, "c": c}
        if comp is None:
            assert "free" in sd and "encoder" in sd, (
                f"--delta_parts {name}: the table has no g/c/f split (rows arm?)"
            )
            f = sd["free"].to(raw.dtype)
            c = sd["encoder"]["common"].to(raw.dtype).expand_as(raw)
            comp = {"f": f, "c": c, "g": raw - f - c}
        assert name and all(ch in comp for ch in name), f"--delta_parts: {name}"
        out[name] = sum(comp[ch] for ch in name)
    if comp is not None:
        n = lambda t: float(t.norm(dim=1).mean())  # noqa: E731
        print(
            "table parts (row-norm units, mean over rows): "
            + " ".join(f"{k} {n(v):.3f}" for k, v in comp.items())
            + f" full {n(raw):.3f}",
            flush=True,
        )
    return out


class SceneKept:
    """The scene-kept ruler (plan_synth item 4): an image is *kept* when its
    PE-Spatial pooled feature is closer to the floor image of the same
    (prompt, kana, clause, seed) than to the flat training canvas —
    ``margin = cos(img, floor) − cos(img, canvas prototype) ≥ τ`` with the
    prototype the mean feature of ``n_proto`` training renders. Floor images
    come from this run's manifest, else from ``ref_dir`` (an earlier native
    run of the same arm). A plain cos-to-floor cannot do it: a white bubble
    on black still scores 0.89 against a 2-koma scene while real scenes sit
    at 0.94+ (P0b calibration, 2026-09-14)."""

    def __init__(self, device, ref_dir: Path, canvas_dir: Path, tau: float, n_proto=64):
        import random

        import torch

        from library.training.cmmd import pool_and_normalize
        from library.vision.encoder import (
            encode_pe_from_imageminus1to1,
            load_pe_encoder,
        )

        self.device = torch.device(device)
        self.bundle = load_pe_encoder(self.device, name="pe_spatial")
        self._pool, self._enc = pool_and_normalize, encode_pe_from_imageminus1to1
        self.ref_dir = ref_dir
        self.tau = tau
        self.cache: dict = {}
        # the prototype is the arm's *flat* share: an S-line data dir also
        # holds scene composites (scene_*.png), which are not the canvas
        files = sorted(
            f for f in canvas_dir.glob("*.png") if not f.name.startswith("scene_")
        )
        assert files, f"scene-kept: no training canvases in {canvas_dir}"
        files = random.Random(0).sample(files, min(n_proto, len(files)))
        self.proto = torch.nn.functional.normalize(
            torch.stack([self.feat(f) for f in files]).mean(0), dim=0
        )
        self.n_proto = len(files)

    def feat(self, path: Path):
        import numpy as np
        import torch
        from PIL import Image

        key = str(path)
        if key not in self.cache:
            t = torch.from_numpy(np.asarray(Image.open(path).convert("RGB")))
            t = (t.permute(2, 0, 1).float() / 127.5 - 1.0).unsqueeze(0)
            with torch.no_grad():
                f = self._enc(self.bundle, t.to(self.device))[0]
            self.cache[key] = self._pool(f).cpu()
        return self.cache[key]

    def score(self, path: Path, ref: Path) -> tuple[float, float]:
        """``(cos to floor, cos to canvas prototype)``."""
        f = self.feat(path)
        return float((f * self.feat(ref)).sum()), float((f * self.proto).sum())

    def floor_file(self, m: dict, manifest) -> Path | None:
        for x in manifest:
            if x["cond"] == "floor" and all(
                x[k] == m[k] for k in ("pi", "text", "clause", "seed")
            ):
                return Path(x["file"])
        cand = (
            self.ref_dir
            / f"floor_p{m['pi']:02d}_{m['text']}_{m['clause']}_s{m['seed']}.png"
        )
        return cand if cand.exists() else None


def _read_native(
    a,
    out: Path,
    manifest,
    chars,
    clauses,
    conds,
    *,
    reread=True,
    rulers=True,
    title="native (scene prompts + kana clause)",
    prompts_path=None,
):
    """``rulers=False`` (the ``target`` stage): readers only — no EN-reference
    or scene-kept scoring (both are keyed by the native prompt index and would
    read another prompt set's refs)."""
    rd = None
    if reread or any("reads" not in m for m in manifest):
        rd = Readers(a.device)
        for m in manifest:
            reads = read_scored(rd, m)
            m["hit_sfx"] = hit(reads, m["text"], "sfx")
            m["hit_vl"] = hit(reads, m["text"], "vl")
            m["exact"] = m["hit_sfx"] and m["hit_vl"]
            m["any_cjk"] = any(CJK_RE.search(r["sfx"] or "") for r in reads)
    # EN-reference ruler (eval/enref.py): refs rendered by stage_native /
    # stage_enref; boxes read once and cached beside them
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
    ref_dir = Path(a.kept_ref) if a.kept_ref else arm_dir(a) / "native" / "img"
    has_floor = rulers and (
        any(m["cond"] == "floor" for m in manifest) or any(ref_dir.glob("floor_*.png"))
    )
    kept = (
        SceneKept(a.device, ref_dir, data_dir(a) / "img", a.kept_tau)
        if has_floor
        else None
    )
    for m in manifest:
        ref = (
            None
            if kept is None or m["cond"] == "floor"
            else kept.floor_file(m, manifest)
        )
        m["kept_cos"] = m["canvas_cos"] = m["kept"] = None
        if ref is not None:
            m["kept_cos"], m["canvas_cos"] = kept.score(Path(m["file"]), ref)
            m["kept"] = m["kept_cos"] - m["canvas_cos"] >= kept.tau
        m["hit_kept"] = bool(m["exact"] and m["kept"])
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
        (
            f"scene-kept ruler: kept ⇔ PE-Spatial cos(img, floor image of the same "
            f"prompt/kana/seed) − cos(img, flat training-canvas prototype of "
            f"{kept.n_proto} renders) ≥ {kept.tau:.2f} (floor ref `{ref_dir}`)"
            if kept is not None
            else "scene-kept ruler: off (no floor renders; --native_floor 1 to restore)"
        ),
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
        ks = [m for m in ms if m["kept_cos"] is not None]
        if not ks:
            return "– | – | – | –"
        return (
            f"{sum(m['kept_cos'] for m in ks) / len(ks):.3f} | "
            f"{sum(m['canvas_cos'] for m in ks) / len(ks):.3f} | "
            f"{sum(bool(m['kept']) for m in ms)} | {sum(m['hit_kept'] for m in ms)}"
        )

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
                                f" m{by_key[pi, k, cl, seed, c]['kept_cos'] - by_key[pi, k, cl, seed, c]['canvas_cos']:+.2f}"
                                if by_key[pi, k, cl, seed, c]["kept_cos"] is not None
                                else ""
                            )
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
                        cells.insert(0, _enref_row(pi, seed) or blank_cell(a.eval_size))
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
