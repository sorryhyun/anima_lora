#!/usr/bin/env python3
"""Blind A/B pairs for the unmask grids, graded by the user in a private git repo.

``make``: for every (row, seed) of the 8-row eval grid and every arm pair,
compose one side-by-side image (A | B, arm order shuffled per pair with a
seeded RNG), write ``<repo>/sets/<set>/pNN.webp`` + a scrollable
``README.md`` + an empty ``verdicts.tsv``; the pair→arm key goes to
``output/blind/<set>/key.json`` **outside the repo**, so the repo alone
cannot reveal which side is which. ``--push`` commits and pushes.

``score``: ``git pull``, read ``verdicts.tsv`` (columns ``pair  verdict
note``; verdict ∈ A / B / tie / skip), join with the key, and write
``reports/blind_<set>.md`` — wins per arm, per row, and the pairs that were
skipped.

    P=project/cjk_aware_anima/probes/blind_pairs.py
    .venv/bin/python $P make  --set s01_C9_vs_P --arms C9 P --push
    .venv/bin/python $P score --set s01_C9_vs_P
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
EVAL = REPO / "output" / "tests" / "cjk_unmask_eval2"
KEYS = REPO / "output" / "blind"
DEFAULT_GIT = REPO.parent / "anima-blind-pairs"
PROMPTS = PROJ / "assets" / "unmask_eval_prompts.txt"


def rows_of(arm: str, seed: int, eval_dir: Path = EVAL) -> list[Path]:
    return sorted((eval_dir / f"arm{arm}_s{seed}").glob("*.png"))


def compose(a: Path, b: Path, out: Path, quality: int) -> None:
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    h = max(ia.height, ib.height)
    gap = 24
    sheet = Image.new("RGB", (ia.width + ib.width + gap, h + 40), "white")
    sheet.paste(ia, (0, 40))
    sheet.paste(ib, (ia.width + gap, 40))
    d = ImageDraw.Draw(sheet)
    d.text((ia.width // 2 - 8, 12), "A", fill="black")
    d.text((ia.width + gap + ib.width // 2 - 8, 12), "B", fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".png":
        sheet.save(out, optimize=True)
    else:
        sheet.save(out, quality=quality, method=6)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def cmd_make(o) -> None:
    if not o.eval_dir.is_absolute():
        o.eval_dir = (REPO / o.eval_dir).resolve()
    if not o.prompts.is_absolute():
        o.prompts = (REPO / o.prompts).resolve()
    prompts = [
        ln.strip()
        for ln in o.prompts.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    rng = random.Random(o.shuffle_seed)
    pairs = list(itertools.combinations(o.arms, 2))
    set_dir = o.git_dir / "sets" / o.set
    if set_dir.exists() and not o.overwrite:
        raise SystemExit(f"{set_dir} exists (use --overwrite)")
    key, readme = (
        [],
        [
            f"# {o.set}",
            "",
            "Same prompt, same seed, two different training arms. Which side is better on "
            "prompt adherence and image quality? Fill `verdicts.tsv` (A / B / tie / skip) "
            "and commit. Sides are shuffled per pair; the key is not in this repo.",
            "",
        ],
    )
    tsv = ["pair\tverdict\tnote"]
    items = []
    for arm_a, arm_b in pairs:
        for seed in o.seeds:
            ra, rb = rows_of(arm_a, seed, o.eval_dir), rows_of(arm_b, seed, o.eval_dir)
            assert len(ra) == len(rb) == len(prompts), (arm_a, arm_b, seed)
            for row in range(len(prompts)):
                if o.rows and (row + 1) not in o.rows:
                    continue
                items.append((arm_a, arm_b, seed, row, ra[row], rb[row]))
    if not o.keep_order:
        rng.shuffle(items)  # pair type / seed / row all hidden by position
    for n, (arm_a, arm_b, seed, row, fa, fb) in enumerate(items, 1):
        pid = f"p{n:02d}"
        left_is_a = rng.random() < 0.5
        left, right = (arm_a, arm_b) if left_is_a else (arm_b, arm_a)
        src = {arm_a: fa, arm_b: fb}
        fname = f"{pid}.{o.fmt}"
        compose(src[left], src[right], set_dir / fname, o.quality)
        key.append(
            {
                "pair": pid,
                "row": row + 1,
                "seed": seed,
                "A": left,
                "B": right,
                "A_src": str(src[left].relative_to(REPO)),
                "B_src": str(src[right].relative_to(REPO)),
            }
        )
        readme += [
            f"## {pid}" + ("" if not o.keep_order else f" — r{row + 1} s{seed}"),
            "",
            f"`{prompts[row]}`",
            "",
            f"![{pid}]({fname})",
            "",
        ]
        tsv.append(f"{pid}\t\t")
    (set_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    (set_dir / "verdicts.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    (set_dir / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    kdir = KEYS / o.set
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "key.json").write_text(
        json.dumps(
            {
                "set": o.set,
                "arms": o.arms,
                "seeds": o.seeds,
                "shuffle_seed": o.shuffle_seed,
                "pairs": key,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    total = sum(f.stat().st_size for f in set_dir.glob(f"*.{o.fmt}")) / 1e6
    print(f"{n} pairs -> {set_dir} ({total:.1f} MB); key -> {kdir / 'key.json'}")
    if o.push:
        git(o.git_dir, "add", "-A", f"sets/{o.set}")
        git(
            o.git_dir,
            "commit",
            "-m",
            f"blind set {o.set}: {n} pairs ({' vs '.join(o.arms)})",
        )
        git(o.git_dir, "push")
        url = git(o.git_dir, "remote", "get-url", "origin").strip()
        branch = git(o.git_dir, "branch", "--show-current").strip()
        print(f"pushed -> {url.removesuffix('.git')}/tree/{branch}/sets/{o.set}")


def cmd_score(o) -> None:
    if o.pull:
        git(o.git_dir, "pull", "--ff-only")
    key = json.loads((KEYS / o.set / "key.json").read_text(encoding="utf-8"))
    by_pair = {k["pair"]: k for k in key["pairs"]}
    verdicts = {}
    for ln in (
        (o.git_dir / "sets" / o.set / "verdicts.tsv")
        .read_text(encoding="utf-8")
        .splitlines()[1:]
    ):
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        pid, v = parts[0].strip(), parts[1].strip().upper()
        note = parts[2].strip() if len(parts) > 2 else ""
        if v:
            verdicts[pid] = (v, note)
    wins = defaultdict(int)
    per_row = defaultdict(lambda: defaultdict(int))
    ties = skipped = 0
    lines = []
    for pid, k in by_pair.items():
        if pid not in verdicts:
            skipped += 1
            continue
        v, note = verdicts[pid]
        if v in ("A", "B"):
            arm = k[v]
            wins[arm] += 1
            per_row[k["row"]][arm] += 1
            res = arm
        elif v == "TIE":
            ties += 1
            per_row[k["row"]]["tie"] += 1
            res = "tie"
        else:
            skipped += 1
            res = "skip"
        lines.append(
            f"| {pid} | r{k['row']} | s{k['seed']} | {k['A']} | {k['B']} | {v} | **{res}** | {note} |"
        )
    n_graded = sum(wins.values()) + ties
    md = [
        f"# blind pairs — {o.set}",
        "",
        f"arms {key['arms']}, {len(by_pair)} pairs, graded {n_graded}, skipped/blank {skipped}",
        "",
        "| arm | wins |",
        "|---|---:|",
    ]
    md += [f"| {a} | {wins[a]} |" for a in key["arms"]] + [
        f"| tie | {ties} |",
        "",
        "| row | " + " | ".join(key["arms"]) + " | tie |",
        "|---|" + "---:|" * (len(key["arms"]) + 1),
    ]
    for row in sorted(per_row):
        md.append(
            f"| r{row} | "
            + " | ".join(str(per_row[row][a]) for a in key["arms"])
            + f" | {per_row[row]['tie']} |"
        )
    md += [
        "",
        "| pair | row | seed | A | B | verdict | result | note |",
        "|---|---|---|---|---|---|---|---|",
    ] + lines
    out = PROJ / "reports" / f"blind_{o.set}.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[: 8 + len(key["arms"])]))
    print(f"-> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--git_dir", type=Path, default=DEFAULT_GIT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--set", required=True)
    m.add_argument("--arms", nargs="+", required=True)
    m.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 1234])
    m.add_argument(
        "--rows",
        nargs="*",
        type=int,
        default=None,
        help="1-based rows to include (default all)",
    )
    m.add_argument("--eval_dir", type=Path, default=EVAL)
    m.add_argument("--prompts", type=Path, default=PROMPTS)
    m.add_argument("--shuffle_seed", type=int, default=0)
    m.add_argument(
        "--keep_order",
        action="store_true",
        help="keep pair/seed/row order (s01–s10 style); default shuffles it",
    )
    m.add_argument("--fmt", choices=("webp", "png"), default="webp")
    m.add_argument("--quality", type=int, default=92)
    m.add_argument("--overwrite", action="store_true")
    m.add_argument("--push", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--set", required=True)
    s.add_argument("--no_pull", dest="pull", action="store_false")
    o = ap.parse_args()
    {"make": cmd_make, "score": cmd_score}[o.cmd](o)


if __name__ == "__main__":
    main()
