"""Stage ``eval`` — T2I the eval set on the bare template with the delta off
(floor) and on (trained), same seeds, read back by both OCR readers.

``native`` (scene prompts + a kana clause) is ``native.py``; it shares
``sheet_row`` / ``blank_cell`` from here.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from common.hooks import AdapterLoRA, ExtDelta, OutVec
from common.models import generate_to, load_generator, load_trained, load_vae
from common.paths import arm_dir, data_dir
from common.prompts import EVAL_GROUPS
from common.readers import Readers, contact_sheet, hit, read_scored
from common.shapes import parse_shape


def sheet_row(m, first_line: str):
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
    if a.with_c_flat:
        # S0: the flat-template eval with the per-source switch on (native
        # never adds it — the scene is the composite source)
        assert "c_flat" in sd, (
            "--with_c_flat: the table has no c_flat (not an S-line rows arm)"
        )
        delta.raw.data.add_(sd["c_flat"].to(delta.raw))
        print(
            f"eval: + c_flat (norm {float(sd['c_flat'].norm()):.3f} row norms)",
            flush=True,
        )
    lora = None
    if "lora" in sd:
        lora = AdapterLoRA(anima, sd["adapter_rank"], device)
        lora.load(sd["lora"])
    outvec = None
    if "out_vec" in sd:
        # trained with Q fixed on: the deployed cond is rows + Q
        outvec = OutVec(anima, device)
        print(
            f"eval: + saved out_vec ‖{float(sd['out_vec'].norm()):.2f}‖ (Q on)",
            flush=True,
        )
    vae = load_vae(device)
    manifest = []
    t0 = time.time()
    conds = ("trained",) if a.no_floor else ("floor", "trained")
    for cond in conds:
        s = 0.0 if cond == "floor" else 1.0
        delta.scale = s
        if outvec is not None:
            outvec.set(sd["out_vec"] if s else None)
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
            sheet_row(m, f"{m['cond']}: {m['text']}")
            for m in manifest
            if m["group"] == g and m["seed"] == 0
        ]
        if rows:
            contact_sheet(rows, out / f"sheet_{g}.png", thumb=192, cols=6)
    if out == train_dir:
        from .summary import summarize_quietly

        summarize_quietly(train_dir)


def blank_cell(size: int):
    from PIL import Image

    return Image.new("RGB", (size, size), "lightgray"), ["enref: missing"]
