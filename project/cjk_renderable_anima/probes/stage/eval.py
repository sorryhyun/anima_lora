"""Stages ``eval`` and ``native`` — T2I with the delta off (floor) and on
(trained), same seeds, read back by both OCR readers.

``eval`` renders the eval set on the bare template; ``native`` hangs a kana
clause off ordinary scene prompts (does the address survive outside the
template? — the product condition W3 needs).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from wake.common import (
    CJK_RE,
    EVAL_GROUPS,
    NATIVE_CLAUSES,
    arm_dir,
    data_dir,
    parse_shape,
)
from wake.hooks import AdapterLoRA, ExtDelta
from wake.models import encode_captions, ext_ids_of, generate_to, load_generator
from wake.models import load_trained, load_vae
from wake.readers import Readers, contact_sheet, hit, read_scored


def _sheet_row(m, first_line: str):
    from PIL import Image

    r0 = m["reads"][-1] if m["reads"] else {"sfx": "", "vl": ""}
    return (
        Image.open(m["file"]).convert("RGB"),
        [first_line, f"sfx {r0['sfx'] or ''}", f"vl {r0['vl'] or ''}"],
    )


# ----------------------------------------------------------------------------
# eval


def stage_eval(a):
    import torch

    train_dir = arm_dir(a)
    ev_file = train_dir / "eval.json"  # encoder arms: held-out singles added
    if not ev_file.exists():
        ev_file = data_dir(a) / "eval.json"
    ev = json.loads(ev_file.read_text())
    if a.eval_groups:
        keep = set(a.eval_groups.split(","))
        ev = [e for e in ev if e["group"] in keep]
    if a.eval_limit:
        seen: dict = {}
        ev = [
            e
            for e in ev
            if seen.setdefault(e["group"], []).append(1)
            or len(seen[e["group"]]) <= a.eval_limit
        ]
    sd = load_trained(train_dir)
    eval_dir = train_dir / f"eval_{a.eval_tag}" if a.eval_tag else train_dir
    (eval_dir / "img").mkdir(parents=True, exist_ok=True)
    eval_size = parse_shape(a.eval_shape) if a.eval_shape else a.eval_size
    args, gen, device, shared = load_generator(
        eval_size, a.steps, a.cfg, eval_dir / "img"
    )
    anima = shared["model"]
    anima.eval()
    delta = ExtDelta.from_state(anima, sd["delta"], device)
    lora = None
    if "lora" in sd:
        lora = AdapterLoRA(anima, sd["adapter_rank"], device)
        lora.load(sd["lora"])
    vae = load_vae(device)
    manifest = []
    t0 = time.time()
    conds = ("trained",) if a.no_floor else ("floor", "trained")
    for cond in conds:
        s = 0.0 if cond == "floor" else 1.0
        delta.scale = s
        if lora is not None:
            lora.scale = s
        shared["conds_cache"].clear()
        for ei, e in enumerate(ev):
            for seed in range(a.seeds):
                fn = eval_dir / "img" / f"{cond}_{e['group']}_{ei:03d}_s{seed}.png"
                generate_to(fn, args, gen, shared, vae, device, e["caption"], seed)
                manifest.append({"file": str(fn), "cond": cond, "seed": seed, **e})
    print(
        f"eval gen: {len(manifest)} images in {(time.time() - t0) / 60:.1f} min",
        flush=True,
    )
    del anima, vae, shared
    torch.cuda.empty_cache()
    _read_eval(a, eval_dir, manifest, train_dir)


def _read_eval(a, out: Path, manifest, train_dir: Path):
    """Reads / report / sheets land in ``out``; ``train_dir`` holds
    ``eval_coverage.json`` from the train stage."""
    rd = Readers(a.device)
    for m in manifest:
        m["exact"] = hit(read_scored(rd, m), m["text"], "sfx")
    (out / "eval_reads.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1)
    )
    agg = defaultdict(list)
    for m in manifest:
        agg[(m["group"], m["cond"])].append(m)
    lines = [
        f"# wake_probe — arm `{a.arm}` eval",
        "",
        "| group | cond | n | CER sfx | CER vl16 | exact (sfx) |",
        "|---|---|---|---|---|---|",
    ]
    for g in EVAL_GROUPS:
        for c in ("floor", "trained"):
            ms = agg.get((g, c), [])
            if not ms:
                continue
            lines.append(
                f"| {g} | {c} | {len(ms)} | {sum(m['cer_sfx'] for m in ms) / len(ms):.3f} | "
                f"{sum(m['cer_vl'] for m in ms) / len(ms):.3f} | {sum(m['exact'] for m in ms)}/{len(ms)} |"
            )
    cov_file = train_dir / "eval_coverage.json"
    cov = json.loads(cov_file.read_text()) if cov_file.exists() else {}
    if cov:
        lines += [
            "",
            "eval ext-row coverage (rows seen in training / rows in the string):",
        ]
        for g in EVAL_GROUPS:
            if g == "en":
                continue
            xs = [
                cov[m["text"]]
                for m in manifest
                if m["group"] == g
                and m["cond"] == "trained"
                and m["seed"] == 0
                and m["text"] in cov
            ]
            if xs:
                lines.append(f"- {g}: {sum(x[0] for x in xs)}/{sum(x[1] for x in xs)}")
    lines += [
        "",
        "Sheets: sheet_<group>.png — floor row then trained row per string, seed 0; label = ref / sfx read / vl16 read.",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    for g in EVAL_GROUPS:
        rows = [
            _sheet_row(m, f"{m['cond']}: {m['text']}")
            for m in manifest
            if m["group"] == g and m["seed"] == 0
        ]
        if rows:
            contact_sheet(rows, out / f"sheet_{g}.png", thumb=192, cols=6)


# ----------------------------------------------------------------------------
# native


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
    manifest = []
    t0 = time.time()
    parts = table_parts(sd, [x for x in a.delta_parts.split(",") if x])
    conds = ([] if a.no_floor else ["floor"]) + list(parts)
    for cond in conds:
        if cond == "floor":
            delta.scale = 0.0
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
            f"table parts (row-norm units, mean over rows): g {n(comp['g']):.3f} "
            f"c {n(comp['c']):.3f} f {n(comp['f']):.3f} full {n(raw):.3f}",
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
        files = sorted(canvas_dir.glob("*.png"))
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


def _read_native(a, out: Path, manifest, chars, clauses, conds):
    rd = Readers(a.device)
    for m in manifest:
        reads = read_scored(rd, m)
        m["hit_sfx"] = hit(reads, m["text"], "sfx")
        m["hit_vl"] = hit(reads, m["text"], "vl")
        m["exact"] = m["hit_sfx"] and m["hit_vl"]
        m["any_cjk"] = any(CJK_RE.search(r["sfx"] or "") for r in reads)
    del rd
    ref_dir = Path(a.kept_ref) if a.kept_ref else arm_dir(a) / "native" / "img"
    kept = SceneKept(a.device, ref_dir, data_dir(a) / "img", a.kept_tau)
    for m in manifest:
        ref = None if m["cond"] == "floor" else kept.floor_file(m, manifest)
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
        f"# wake_probe — arm `{a.arm}` native (scene prompts + kana clause)",
        "",
        f"prompts: `{a.native_prompts}`; chars {' '.join(chars)}; {a.seeds} seed(s); "
        f"{a.eval_size}²; delta scale {a.delta_scale}; parts {' '.join(trained_conds)}",
        "",
        f"scene-kept ruler: kept ⇔ PE-Spatial cos(img, floor image of the same "
        f"prompt/kana/seed) − cos(img, flat training-canvas prototype of "
        f"{kept.n_proto} renders) ≥ {kept.tau:.2f} (floor ref `{ref_dir}`)",
        "",
        "| clause | cond | n | CER sfx | CER vl16 | hit sfx | hit vl | both | any CJK read | floor cos | canvas cos | kept | hit & kept |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

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
                f"{sum(m['any_cjk'] for m in ms)} | {_kept_cols(ms)} |"
            )
    lines += ["", "per kana (both readers hit / kept / hit & kept):", ""]
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
                        f"of {len(ms)}"
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
    lines += [
        "",
        "Sheets: sheet_<kana>_<clause>.png — one row per prompt: "
        + ", ".join(f"{c} s{s}" for s in range(a.seeds) for c in conds)
        + "; label = prompt idx / sfx read / vl16 read.",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    by_key = {
        (m["pi"], m["text"], m["clause"], m["seed"], m["cond"]): m for m in manifest
    }
    for k in chars:
        for cl in clauses:
            rows = [
                _sheet_row(
                    by_key[pi, k, cl, seed, c],
                    f"p{pi:02d} {c} s{seed}: {k}"
                    + (
                        f" m{by_key[pi, k, cl, seed, c]['kept_cos'] - by_key[pi, k, cl, seed, c]['canvas_cos']:+.2f}"
                        if by_key[pi, k, cl, seed, c]["kept_cos"] is not None
                        else ""
                    ),
                )
                for pi in sorted(by_p)
                for seed in range(a.seeds)
                for c in conds
                if (pi, k, cl, seed, c) in by_key
            ]
            if rows:
                contact_sheet(
                    rows,
                    out / f"sheet_{k}_{cl}.png",
                    thumb=192,
                    cols=len(conds) * a.seeds,
                )
