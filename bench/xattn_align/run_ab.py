#!/usr/bin/env python
"""The caststation A/B of the cross-attn alignment term
(docs/anima/proposal/xattn_align_loss.md § 5; adapter library/training/xattn_align.py).

Arms (all on the shipped recipe, configs/methods/lora.toml, channel caststation;
every arm carries the adapter so the term's loss and lift are logged alike):

- ``plain``    weight 0, calib on (the grad norms that fix λ), seed 42
- ``twin``     weight 0, seed 43
- ``box_lo``   λ at a gradient-norm ratio of 0.1 to FM, seed 42
- ``box_hi``   the same at 0.5
- ``contrast`` the contrast form, at the ratio box_lo / box_hi read better at

Verbs:

    .venv/bin/python bench/xattn_align/run_ab.py prompts            # the working set
    .venv/bin/python bench/xattn_align/run_ab.py train --arms plain,twin
    .venv/bin/python bench/xattn_align/run_ab.py calib              # λ from plain's log
    .venv/bin/python bench/xattn_align/run_ab.py train --arms box_lo,box_hi
    make daemon-run ARGS="--queue --stall-timeout 900 bench/xattn_align/run_ab.py render --arms …"
    make daemon-run ARGS="--queue --stall-timeout 600 bench/xattn_align/run_ab.py read --arms …"
    .venv/bin/python bench/xattn_align/run_ab.py sheets --arm box_lo

State (run names, λ) lives in ``output/xattn_align/state.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import toml

from library.env import resolve_under_home

CONFIGS = resolve_under_home("configs/xattn_align")
CKPT = "output/ckpt/xattn_align"
# --set: dataset dir, artist tag, output dir, run-name prefix, text prompts
# drawn (None = every caption with a text clause).
SETS = {
    "caststation": ("channel_(caststation)", "@channel (caststation)", "output/xattn_align", "xa", None),
    "sincos": ("sincos", "@sincos", "output/xattn_align/sincos", "xs", 12),
}  # fmt: skip
SET, TAG, OUT, PREFIX, N_TEXT = SETS["caststation"]
OUT = resolve_under_home(OUT)
STATE = OUT / "state.json"
RULER = "output/cjk_anima_reseed/ruler/ruler.json"
RESEED = "project/cjk_anima_reseed"
BINS = {"short": (0, 4), "mid": (5, 9), "long": (10, 20), "xlong": (21, 10**9)}
SEEDS = (0, 1, 2)
STEPS, CFG, SHIFT = 28, 4.0, 3.0
STYLE_T = [  # docs/anima/findings/lora_recipe.md's four prompts
    "safe, 1girl, solo, {tag}, blonde hair, school uniform, classroom, sitting at desk, smile, looking at viewer",
    "safe, 1boy, 1girl, {tag}, hetero, hug, indoors, curtains",
    "safe, 1girl, solo, {tag}, maid, holding a tray, cafe interior",
    "sensitive, 1girl, original, {tag}, black bikini, black hair, short hair, innertube, beach, day, blue sky, looking at viewer, standing",
]
STYLE = [p.format(tag=TAG) for p in STYLE_T]
STYLE_NEG = "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"
HANDLE = {"plain": "plain", "twin": "twin", "box_lo": "boxlo",
          "box_hi": "boxhi", "contrast": "con"}  # fmt: skip
TARGET = {"box_lo": 0.1, "box_hi": 0.5}


def configure(name: str) -> None:
    global SET, TAG, OUT, PREFIX, N_TEXT, STATE, STYLE
    SET, TAG, out, PREFIX, N_TEXT = SETS[name]
    OUT = resolve_under_home(out)
    STATE = OUT / "state.json"
    STYLE = [p.format(tag=TAG) for p in STYLE_T]


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"runs": {}}


def save_state(s: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1, ensure_ascii=False))


# ---------------------------------------------------------------- prompts


def glyph_len(s: str) -> int:
    from eval.ruler.build import length

    return length(s)


def bin_of(n: int) -> str:
    return next(b for b, (lo, hi) in BINS.items() if lo <= n <= hi)


def cmd_prompts(a) -> None:
    """The 12 caststation captions with a text clause (v0), each quoted text
    string swapped for a dialogue-ruler string of the same length bin (the
    cjk_rl pool's rule; exact glyph length first, no string twice)."""
    from anime_tools.captions.position_clauses import quote_text
    from PIL import Image

    from library.training.xattn_align import CLAUSE_RE

    ruler = json.loads(resolve_under_home(RULER).read_text("utf-8"))["items"]
    pool = [r["text"] for r in ruler if "\n" not in r["text"]]
    rng = random.Random(1120)
    rng.shuffle(pool)
    used: set = set()

    def draw(old: str) -> str:
        n = glyph_len(old)
        same = [s for s in pool if s not in used and glyph_len(s) == n]
        b = bin_of(n) if n <= 20 else "long"
        near = [s for s in pool if s not in used and bin_of(glyph_len(s)) == b]
        if b == "long" and n > 20:  # no xlong in the ruler: the longest left
            near = sorted(near, key=glyph_len, reverse=True)[:3]
        got = (same or near)[0]
        used.add(got)
        return got

    root = resolve_under_home("post_image_dataset/resized") / SET
    found = []
    for txt in sorted(root.glob("*.txt")):
        if txt.name.count(".") > 1:
            continue
        cap = txt.read_text("utf-8").strip()
        spans = []
        for m in CLAUSE_RE.finditer(cap):
            if m.group(2) != "text":
                continue
            i = m.end()
            while i < len(cap) and cap[i] == '"':
                j = cap.find('"', i + 1)
                spans.append((i + 1, j))
                i = j + 1
                if cap.startswith(", ", i):
                    i += 2
        if spans:
            found.append((txt, cap, spans))
    if N_TEXT and len(found) > N_TEXT:
        found = sorted(random.Random(f"{SET}-text").sample(found, N_TEXT))
    items = []
    for txt, cap, spans in found:
        new, out, last = [], [], 0
        for s0, s1 in spans:
            old = cap[s0:s1].replace("\\n", "")
            got = draw(old)
            out += [cap[last:s0], quote_text(got)]
            new.append({"old": cap[s0:s1], "new": got})
            last = s1
        out.append(cap[last:])
        W, H = Image.open(txt.with_suffix(".png")).size
        items.append({"stem": txt.stem, "caption": "".join(out), "strings": new,
                      "size": [W - W % 16, H - H % 16]})  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "prompts.json").write_text(json.dumps(items, indent=1, ensure_ascii=False))
    for m in items:
        print(m["stem"], m["size"], [(s["old"], s["new"]) for s in m["strings"]])
    print(f"{len(items)} prompts → {OUT / 'prompts.json'}")


# ---------------------------------------------------------------- train


def arm_overrides(arm: str, st: dict) -> dict:
    o = {"use_xattn_align": True, "xattn_align_weight": 0.0, "seed": 42}
    if arm == "plain":
        o["xattn_align_calib"] = True
    elif arm == "twin":
        o["seed"] = 43
    elif arm in ("box_lo", "box_hi"):
        o["xattn_align_weight"] = st["lambda"][arm]
    elif arm == "contrast":
        o["xattn_align_form"] = "contrast"
        o["xattn_align_weight"] = st["lambda"][st["contrast_from"]]
    return o


def cmd_train(a) -> None:
    st = state()
    base = toml.load(resolve_under_home("configs/methods/lora.toml"))
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for arm in a.arms.split(","):
        if arm in st["runs"]:
            print(f"{arm}: already {st['runs'][arm]}, skipped")
            continue
        name = f"{PREFIX}{HANDLE[arm]}_{time.strftime('%m%d_%H%M')}"
        cfg = dict(base) | arm_overrides(arm, st) | {
            "output_name": name, "output_dir": CKPT, "path_pattern": f"{SET}/*",
        }  # fmt: skip
        head = (
            f"# {name}: xattn_align A/B arm `{arm}` (bench/xattn_align/run_ab.py,\n"
            "# docs/anima/proposal/xattn_align_loss.md § 5). configs/methods/lora.toml\n"
            f"# on {SET} + the adapter: {json.dumps(arm_overrides(arm, st))}\n"
        )
        (CONFIGS / f"{name}.toml").write_text(head + toml.dumps(cfg), "utf-8")
        cmd = [
            "make",
            "lora",
            f"ARGS=--queue --methods_subdir xattn_align --method {name}",
        ]
        print(" ".join(cmd), flush=True)
        if not a.dry_run:
            r = subprocess.run(
                cmd, cwd=resolve_under_home("."), capture_output=True, text=True
            )
            job = re.search(r"queued job (\S+)", r.stdout)
            print(r.stdout.strip().splitlines()[0] if r.stdout else r.stderr)
            st["runs"][arm] = name
            st.setdefault("jobs", {})[arm] = job.group(1) if job else None
            save_state(st)
        time.sleep(61)  # one name per minute


def cmd_calib(a) -> None:
    st = state()
    log = resolve_under_home(CKPT) / f"{st['runs']['plain']}_xattn_align.jsonl"
    rows = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    first = [r for r in rows if "ratio" in r][: a.n]
    med = statistics.median(r["ratio"] for r in first)
    st["lambda"] = {k: round(v / med, 5) for k, v in TARGET.items()}
    st["calib"] = {"n": len(first), "median_ratio": med,
                   "g_fm_median": statistics.median(r["g_fm"] for r in first),
                   "g_al_median": statistics.median(r["g_al"] for r in first),
                   "term_steps_total": len(rows)}  # fmt: skip
    save_state(st)
    print(json.dumps({"lambda": st["lambda"], **st["calib"]}, indent=1))


# ---------------------------------------------------------------- render


def seed_of(kind: str, i: int, s: int) -> int:
    return (100_000 if kind == "style" else 0) + 1000 * (i + 1) + s


def cmd_render(a) -> None:
    """One inference process per arm (one compile): the text set at the
    ruler's settings (no negative), then the style prompts with the
    ``make test`` negative."""
    st = state()
    items = json.loads((OUT / "prompts.json").read_text("utf-8"))
    for arm in a.arms.split(","):
        name = st["runs"][arm]
        lora = resolve_under_home(CKPT) / f"{name}.safetensors"
        out = OUT / "renders" / name
        out.mkdir(parents=True, exist_ok=True)
        lines = [f"{m['caption']} --w {m['size'][0]} --h {m['size'][1]} "
                 f"--d {seed_of('text', i, s)}"
                 for i, m in enumerate(items) for s in SEEDS]  # fmt: skip
        lines += [f"{p} --w 1024 --h 1024 --d {seed_of('style', i, s)} --n {STYLE_NEG}"
                  for i, p in enumerate(STYLE) for s in SEEDS]  # fmt: skip
        if len(by_seed(out)) >= len(lines):
            print(f"{name}: done", flush=True)
            continue
        pf = out / "prompts.txt"
        pf.write_text("\n".join(lines) + "\n", encoding="utf-8")
        cmd = [sys.executable, "inference.py",  # catalog dit / te / vae
               "--vae_chunk_size", "64", "--vae_disable_cache",
               "--lora_weight", str(lora),
               "--lora_multiplier", "1.0", "--from_file", str(pf),
               "--save_path", str(out), "--negative_prompt", "",
               "--infer_steps", str(STEPS), "--guidance_scale", str(CFG),
               "--flow_shift", str(SHIFT), "--sampler", "euler",
               "--attn_mode", "flash", "--compile_blocks",
               "--compile_dynamic_seq"]  # fmt: skip
        print(f"{name}: {len(lines)} renders", flush=True)
        subprocess.run(cmd, check=True, cwd=resolve_under_home("."))


def by_seed(d: Path) -> dict[int, Path]:
    out = {}
    for f in d.glob("*.png"):
        m = re.search(r"_(\d+)_?$", f.stem)
        if m:
            out[int(m.group(1))] = f
    return out


# ---------------------------------------------------------------- read


def cmd_read(a) -> None:
    home = resolve_under_home(RESEED)
    sys.path += [str(home), str(home / "src")]
    from common.readers import Readers, load_bgr
    from eval.ruler.score import score_page, score_text

    st = state()
    items = json.loads((OUT / "prompts.json").read_text("utf-8"))
    rd = Readers("cuda")
    for arm in a.arms.split(","):
        name = st["runs"][arm]
        files = by_seed(OUT / "renders" / name)
        rows = []
        for i, m in enumerate(items):
            text = "".join(s["new"] for s in m["strings"])
            for s in SEEDS:
                f = files.get(seed_of("text", i, s))
                if f is None:
                    continue
                reads = rd.read_image(load_bgr(f), whole=True)
                page = score_page(text, reads, [])
                strs = [score_text(x["new"], reads) for x in m["strings"]]
                rows.append({
                    "arm": arm, "stem": m["stem"], "seed": s, "file": str(f),
                    **{k: page[k] for k in ("g_p", "g_r", "g_f1", "drawn", "a_p", "text_area")},
                    "exact": sum(bool(x["exact"]) for x in strs) / len(strs),
                    "le1": sum(bool(x["le1"]) for x in strs) / len(strs),
                })  # fmt: skip
        if not rows:
            print(f"{arm}: no renders, skipped", flush=True)
            continue
        (OUT / f"read_{arm}.json").write_text(
            json.dumps(rows, indent=1, ensure_ascii=False)
        )
        print(arm, summarize(rows), flush=True)


def summarize(rows: list) -> dict:
    keys = ("g_p", "g_r", "g_f1", "drawn", "a_p", "text_area", "exact", "le1")
    return {k: round(statistics.mean(r[k] for r in rows), 4) for k in keys} | {
        "n": len(rows)
    }


def cmd_table(a) -> None:
    for f in sorted(OUT.glob("read_*.json")):
        rows = json.loads(f.read_text())
        print(f.stem[5:], summarize(rows) if rows else "no rows")


# ---------------------------------------------------------------- sheets


def cmd_sheets(a) -> None:
    """``plain`` against one arm: two columns per (prompt, seed), one file
    each, no labels, sides shuffled per prompt; key in ``key.json``."""
    from PIL import Image

    st = state()
    items = json.loads((OUT / "prompts.json").read_text("utf-8"))
    ctl, trt = st["runs"]["plain"], st["runs"][a.arm]
    out = OUT / "sheets" / a.arm
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.shuffle_seed)
    key = {}
    for kind, prompts in (("text", [m["stem"] for m in items]), ("style", STYLE)):
        A = by_seed(OUT / "renders" / ctl)
        B = by_seed(OUT / "renders" / trt)
        seeds = SEEDS if kind == "style" or a.text_seeds is None else a.text_seeds
        for i, _ in enumerate(prompts):
            flip = rng.random() < 0.5
            for s in seeds:
                sd = seed_of(kind, i, s)
                if sd not in A or sd not in B:
                    continue
                left, right = (B[sd], A[sd]) if flip else (A[sd], B[sd])
                ia, ib = (
                    Image.open(left).convert("RGB"),
                    Image.open(right).convert("RGB"),
                )
                h = max(ia.height, ib.height)
                im = Image.new("RGB", (ia.width + ib.width + 16, h), "white")
                im.paste(ia, (0, 0))
                im.paste(ib, (ia.width + 16, 0))
                fn = f"{kind}_p{i:02d}_s{s}.jpg"
                im.save(out / fn, quality=90)
                key[fn] = {
                    "left": a.arm if flip else "plain",
                    "right": "plain" if flip else a.arm,
                }
    (OUT / f"sheets_key_{a.arm}.json").write_text(json.dumps(key, indent=1))
    caps = {"text": [m["caption"] for m in items], "style": STYLE}
    rows = [f"{fn}\t{caps[fn.split('_')[0]][int(fn.split('_')[1][1:])]}"
            for fn in sorted(key)]  # fmt: skip
    (out / "rows.txt").write_text("\n".join(rows) + "\n", "utf-8")
    print(f"{len(key)} sheets → {out}")


BLIND = resolve_under_home("..") / "anima-blind-pairs"
BLIND_KEYS = resolve_under_home("output/blind")
BLIND_ASK = (
    "Same prompt, same seed, two LoRAs trained with different recipes. Which side came out "
    "better? First: hands and body without distortion. Second: the quoted text drawn as "
    "written (text pairs). Fill `verdicts.tsv` (A / B / tie, or 0 = left / 1 = right / -) "
    "and commit, or send the picks as one string in pair order. Sides "
    "are shuffled per pair; the key is not in this repo.\n"
)


def blind_cells(items: list) -> list[tuple[str, int, int]]:
    """The 24 pairs: every text prompt at seed 0, every style prompt × 3 seeds."""
    return [("text", i, 0) for i in range(len(items))] + [
        ("style", i, s) for i in range(len(STYLE)) for s in SEEDS
    ]


def cmd_blind(a) -> None:
    """``plain`` vs ``--arm`` as a set in ../anima-blind-pairs (A / B labels,
    pair order and sides shuffled); key in output/blind/<set>/key.json."""
    from PIL import Image, ImageDraw

    st = state()
    items = json.loads((OUT / "prompts.json").read_text("utf-8"))
    arms = {"plain": st["runs"]["plain"], a.arm: st["runs"][a.arm]}
    files = {k: by_seed(OUT / "renders" / v) for k, v in arms.items()}
    cells = blind_cells(items)
    rng = random.Random(a.shuffle_seed)
    rng.shuffle(cells)
    out = BLIND / "sets" / a.name
    out.mkdir(parents=True, exist_ok=True)
    key, readme = {}, [f"# {a.name}\n", BLIND_ASK]
    for n, (kind, i, s) in enumerate(cells, 1):
        pair, sd = f"p{n:02d}", seed_of(kind, i, s)
        left, right = rng.sample(list(arms), 2)
        ia = Image.open(files[left][sd]).convert("RGB")
        ib = Image.open(files[right][sd]).convert("RGB")
        gap, h = 24, max(ia.height, ib.height)
        sheet = Image.new("RGB", (ia.width + ib.width + gap, h + 40), "white")
        sheet.paste(ia, (0, 40))
        sheet.paste(ib, (ia.width + gap, 40))
        d = ImageDraw.Draw(sheet)
        d.text((ia.width // 2 - 8, 12), "A", fill="black")
        d.text((ia.width + gap + ib.width // 2 - 8, 12), "B", fill="black")
        sheet.save(out / f"{pair}.webp", quality=90, method=6)
        key[pair] = {"A": left, "B": right, "kind": kind, "prompt": i, "seed": s}
        cap = items[i]["caption"] if kind == "text" else STYLE[i]
        readme.append(f"## {pair}\n\n`{cap}`\n\n![{pair}]({pair}.webp)\n")
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")
    (out / "verdicts.tsv").write_text(
        "pair\tverdict\tnote\n" + "".join(f"{p}\t\t\n" for p in key), encoding="utf-8"
    )
    (BLIND_KEYS / a.name).mkdir(parents=True, exist_ok=True)
    (BLIND_KEYS / a.name / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{a.name}: {len(key)} pairs → {out}")
    if a.push:
        sub = f"sets/{a.name}"
        subprocess.run(["git", "add", sub], cwd=BLIND, check=True)
        msg = f"blind set {a.name}: {len(key)} pairs (plain vs {a.arm}, xattn_align)"
        subprocess.run(["git", "commit", "-m", msg, "--", sub], cwd=BLIND, check=True)
        subprocess.run(["git", "push"], cwd=BLIND, check=True)


def cmd_score(a) -> None:
    """The set's verdicts against its key: wins per arm, overall and by kind."""
    key = json.loads((BLIND_KEYS / a.name / "key.json").read_text())
    if a.picks:  # one char per pair in order: 0 = left (A), 1 = right (B), - = tie
        verdicts = list(zip(sorted(key), a.picks))
    else:
        rows = (BLIND / "sets" / a.name / "verdicts.tsv").read_text("utf-8")
        verdicts = [(ln.split("\t") + [""])[:2] for ln in rows.splitlines()[1:]]
    norm = {"0": "A", "1": "B", "-": "TIE", "A": "A", "B": "B", "TIE": "TIE"}
    tally: dict = {}
    for pair, v in verdicts:
        v = norm.get(v.strip().upper())
        if pair not in key or v is None:
            continue
        k = key[pair]
        who = "tie" if v == "TIE" else k[v]
        for g in ("all", k["kind"]):
            tally.setdefault(g, {}).setdefault(who, 0)
            tally[g][who] += 1
    print(json.dumps(tally, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=sorted(SETS), default="caststation")
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("prompts")
    t = sub.add_parser("train")
    t.add_argument("--arms", required=True)
    t.add_argument("--dry_run", action="store_true")
    c = sub.add_parser("calib")
    c.add_argument("--n", type=int, default=20)
    r = sub.add_parser("render")
    r.add_argument("--arms", required=True)
    rd = sub.add_parser("read")
    rd.add_argument("--arms", required=True)
    sub.add_parser("table")
    s = sub.add_parser("sheets")
    s.add_argument("--arm", required=True)
    s.add_argument("--shuffle_seed", type=int, default=1120)
    s.add_argument("--text_seeds", type=lambda v: tuple(map(int, v.split(","))),
                   default=None, help="text sheets for these seeds only (e.g. 0)")
    b = sub.add_parser("blind")
    b.add_argument("--arm", required=True)
    b.add_argument("--name", required=True, help="set name, e.g. s44_XATTN_sincos")
    b.add_argument("--shuffle_seed", type=int, default=1120)
    b.add_argument("--push", action="store_true")
    sc = sub.add_parser("score")
    sc.add_argument("--name", required=True)
    sc.add_argument("--picks", help="p01… in order: 0 = left, 1 = right, - = tie")
    a = p.parse_args()
    configure(a.set)
    if a.verb == "prompts":
        home = resolve_under_home(RESEED)
        sys.path += [str(home), str(home / "src")]
    globals()[f"cmd_{a.verb}"](a)


if __name__ == "__main__":
    main()
