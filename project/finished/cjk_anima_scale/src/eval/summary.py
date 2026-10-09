"""``eval_summary.png`` — one glance at an arm: the headline numbers of every
result the arm has (train log, eval groups, native, target) and one or two
renders per group (a hit first when there is one, then a miss), each cell
framed green / red by the readers' verdict.

Written at the end of ``eval`` / ``native`` / ``target`` (best effort — a
summary failure never fails the run) and by ``--stage summary`` on its own
(CPU only; re-reads the json manifests, renders nothing).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from common.paths import arm_dir
from common.prompts import EVAL_GROUPS
from common.readers import _label_font

THUMB = 208
COLS = 6
LABEL_H = 22


def stage_summary(a):
    p = write_summary(arm_dir(a))
    print(f"summary: {p}" if p else "summary: nothing to summarise", flush=True)


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None


def _pick(ms: list[dict], key: str, n: int) -> list[dict]:
    """Up to ``n`` items: a hit first (if any), then misses, seed 0 preferred."""
    ms = sorted(ms, key=lambda m: (m["seed"], m.get("pi", 0)))
    hits = [m for m in ms if m.get(key)]
    misses = [m for m in ms if not m.get(key)]
    out = hits[:1] + misses
    return out[:n] if hits else misses[:n]


def _read_lines(m: dict) -> list[str]:
    r0 = m["reads"][-1] if m.get("reads") else {"sfx": "", "vl": ""}
    return [f"sfx {r0.get('sfx') or ''}", f"vl {r0.get('vl') or ''}"]


def collect(arm: Path, per_group: int = 2) -> tuple[list[str], list[dict]]:
    """``(header lines, cells)`` — cells are ``{file, hit, lines}``."""
    header: list[str] = [f"{arm.name}"]
    cells: list[dict] = []
    log = _load(arm / "train_log.json")
    if log:
        last = log[-1]
        ln = f"train: {last['step']} steps, delta norm {last['delta_norm_mean']:.1f}"
        if "warm_cos" in last:
            ln += f", warm_cos {last['warm_cos']:.3f}"
        header.append(ln)
    ev = _load(arm / "eval_reads.json")
    if ev:
        by = defaultdict(list)
        for m in ev:
            if m["cond"] == "trained":
                by[m["group"]].append(m)
        groups = [g for g in EVAL_GROUPS if g in by] + sorted(
            set(by) - set(EVAL_GROUPS)
        )
        header.append(
            "eval (trained, exact): "
            + "  ".join(
                f"{g} {sum(m['exact'] for m in by[g])}/{len(by[g])}" for g in groups
            )
        )
        for g in groups:
            for m in _pick(by[g], "exact", per_group):
                cells.append(
                    {
                        "file": m["file"],
                        "hit": bool(m["exact"]),
                        "lines": [f"eval {g}: {m['text']}", *_read_lines(m)],
                    }
                )
    for name, title in (("native", "native"), ("target", "target")):
        man = _load(arm / name / "native_reads.json")
        if not man:
            continue
        conds = []
        for m in man:
            if m["cond"] not in conds:
                conds.append(m["cond"])
        by = defaultdict(list)
        for m in man:
            by[(m["cond"], m["clause"], m["text"])].append(m)
        parts = []
        for c in conds:
            ms = [m for m in man if m["cond"] == c]
            s = f"{c} both {sum(m['exact'] for m in ms)}/{len(ms)}"
            if name == "native":
                es = [m["en_cos_out"] for m in ms if m.get("en_cos_out") is not None]
                if es:
                    s += f" en_out {sum(es) / len(es):.3f}"
            parts.append(s)
        header.append(f"{title}: " + "  ".join(parts))
        if name == "target":
            # per text, floor vs trained, both readers
            texts = []
            for m in man:
                if m["text"] not in texts:
                    texts.append(m["text"])
            header.append(
                "target per text: "
                + "  ".join(
                    f"{t} "
                    + "/".join(
                        f"{c} {sum(m['exact'] for m in man if m['cond'] == c and m['text'] == t)}"
                        for c in conds
                    )
                    + f" of {sum(1 for m in man if m['cond'] == conds[0] and m['text'] == t)}"
                    for t in texts
                )
            )
        trained_conds = [c for c in conds if c != "floor"] or conds
        for c in trained_conds:
            keys = [k for k in by if k[0] == c]
            for k in sorted(keys, key=lambda k: (k[2], k[1])):
                n = per_group if name == "target" else 1
                for m in _pick(by[k], "exact", n):
                    tag = f"{title} p{m['pi']:02d} {c}: {m['text']}"
                    if name == "native":
                        tag += f" /{m['clause']}"
                    cells.append(
                        {
                            "file": m["file"],
                            "hit": bool(m["exact"]),
                            "lines": [tag, *_read_lines(m)],
                        }
                    )
        if name == "target" and "floor" in conds:
            # the floor beside it, first prompt per text
            seen = set()
            for m in sorted(
                [m for m in man if m["cond"] == "floor"],
                key=lambda m: (m["pi"], m["seed"]),
            ):
                if m["text"] in seen:
                    continue
                seen.add(m["text"])
                cells.append(
                    {
                        "file": m["file"],
                        "hit": bool(m["exact"]),
                        "lines": [
                            f"target p{m['pi']:02d} floor: {m['text']}",
                            *_read_lines(m),
                        ],
                    }
                )
    return header, cells


def _width(s: str) -> int:
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in s)


def _clip(s: str, w: int) -> str:
    out, n = "", 0
    for ch in s:
        n += 2 if ord(ch) > 0x2E7F else 1
        if n > w:
            return out + "…"
        out += ch
    return out


def _wrap(s: str, w: int) -> list[str]:
    """Wrap a header line at the double-space separators ``collect`` emits."""
    parts, lines, cur = s.split("  "), [], ""
    for p in parts:
        nxt = p if not cur else f"{cur}  {p}"
        if cur and _width(nxt) > w:
            lines.append(cur)
            cur = "    " + p
        else:
            cur = nxt
    return lines + [cur]


def write_summary(
    arm: Path, per_group: int = 2, path: Path | None = None
) -> Path | None:
    from PIL import Image, ImageDraw

    header, cells = collect(arm, per_group)
    if len(header) <= 1:
        return None
    path = path or arm / "eval_summary.png"
    font = _label_font(16)
    hfont = _label_font(20)
    header = [w for ln in header for w in _wrap(ln, 110)]
    cell_h = THUMB + LABEL_H * 3 + 10
    head_h = 8 + 26 * len(header) + 8
    rows = max(1, -(-len(cells) // COLS))
    W = COLS * (THUMB + 8) + 8
    sheet = Image.new("RGB", (W, head_h + rows * cell_h + 8), "white")
    d = ImageDraw.Draw(sheet)
    for i, ln in enumerate(header):
        d.text((8, 8 + 26 * i), ln, fill="black", font=hfont)
    for i, c in enumerate(cells):
        x = (i % COLS) * (THUMB + 8) + 8
        y = head_h + (i // COLS) * cell_h + 4
        try:
            im = Image.open(c["file"]).convert("RGB")
        except Exception:
            im = Image.new("RGB", (THUMB, THUMB), "lightgray")
        im.thumbnail((THUMB, THUMB))
        sheet.paste(im, (x + 3, y + 3))
        d.rectangle(
            [x, y, x + THUMB + 5, y + THUMB + 5],
            outline="seagreen" if c["hit"] else "crimson",
            width=3,
        )
        for j, ln in enumerate(c["lines"][:3]):
            d.text(
                (x, y + THUMB + 8 + LABEL_H * j), _clip(ln, 22), fill="black", font=font
            )
    sheet.save(path)
    return path


def summarize_quietly(arm: Path) -> None:
    """End-of-stage hook: never raises."""
    try:
        p = write_summary(arm)
        if p:
            print(f"summary: {p}", flush=True)
    except Exception as e:  # noqa: BLE001 — a summary must not fail the run
        print(f"summary: skipped ({e})", flush=True)
