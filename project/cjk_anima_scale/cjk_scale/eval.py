"""eval — the stage's rulers, delegated to the probe line's stages so every
number is on the ruler the reads of record used.

  exact     ``eval``: the stage's own eval groups, floor-less, ``--seeds``
  native    ``native``: the fixed row sample on scene prompts, ``en`` / ``swap``
  cf_sense  ``cf_sense --cf_lang ja``: the trained rows' caption leverage by σ —
            does it land in the band the stage trained in

The probe stages open ``data_scale_<stage>_<tag>`` / ``rows_scale_<stage>_<tag>``
through their own ``--data_tag``, so nothing is copied. The warm-chain
regression check compares this table's ``eval_reads.json`` with the earlier
stages' on the groups they share (same units + seed → same eval strings).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .config import StageConfig
from .paths import arm_dir, run_tag


def probe_args(
    cfg: StageConfig, tag: str, stage_names: list, extra: list | None = None
):
    from cli import build_parser
    from stages import STAGES

    e = cfg.eval
    argv = [
        "--stage",
        *stage_names,
        "--arm",
        "rows",
        "--data_tag",
        run_tag(cfg.stage, tag),
        "--seeds",
        str(int(e["seeds"])),
        "--no_floor",
        "--eval_groups",
        str(e["groups"]),
        "--native_chars",
        str(e["native_chars"]),
        "--native_clauses",
        str(e["native_clauses"]),
        "--cf_lang",
        "ja",
        "--cf_rows",
        str(e["cf_rows"]),
        "--steps",
        str(int(cfg.train["steps"])),
        "--cfg",
        str(float(cfg.train["cfg"])),
        "--seed",
        str(int(cfg.train["seed"])),
        *(extra or []),
    ]
    return build_parser(STAGES).parse_args(argv)


def run(
    cfg: StageConfig, tag: str, which=("eval", "native", "cf_sense"), extra=None
) -> None:
    from stages import run as run_stage

    out = arm_dir(cfg.stage, tag)
    assert (out / "trained.pt").exists(), f"no table at {out / 'trained.pt'}"
    which = [w for w in which if w != "cf_sense" or cfg.eval["cf_sense"]]
    a = probe_args(cfg, tag, which, extra)
    for name in which:
        print(f"===== {cfg.stage} eval: {name}", flush=True)
        run_stage(name, a)
    regress(cfg, tag)


def exact_by_group(arm: Path) -> dict:
    """``eval_reads.json`` → group → (exact hits, n, texts) on the trained cond."""
    f = arm / "eval_reads.json"
    if not f.exists():
        return {}
    agg: dict = defaultdict(lambda: [0, 0, set()])
    for m in json.loads(f.read_text(encoding="utf-8")):
        if m.get("cond") != "trained":
            continue
        g = agg[m["group"]]
        g[0] += int(bool(m.get("exact")))
        g[1] += 1
        g[2].add(m["text"])
    return {k: (v[0], v[1], frozenset(v[2])) for k, v in agg.items()}


def regress(cfg: StageConfig, tag: str) -> dict:
    """The warm-chain check (design § 5): the earlier stages' exact groups,
    read on this table vs on theirs. Writes ``regress.json`` in the arm
    dir and prints one line per (stage, group)."""
    here = arm_dir(cfg.stage, tag)
    mine = exact_by_group(here)
    out: dict = {}
    for prev in cfg.eval.get("regress", []):
        theirs = exact_by_group(arm_dir(prev, tag))
        if not theirs:
            print(
                f"regress vs {prev}: no eval_reads.json under {arm_dir(prev, tag)}",
                flush=True,
            )
            continue
        for g, (hit0, n0, texts0) in sorted(theirs.items()):
            if g not in mine:
                continue
            hit1, n1, texts1 = mine[g]
            same = texts0 == texts1
            out[f"{prev}/{g}"] = {
                "prev": [hit0, n0],
                "now": [hit1, n1],
                "same_strings": same,
            }
            print(
                f"regress {prev} {g}: {hit0}/{n0} → {hit1}/{n1}"
                + ("" if same else "  (eval strings differ — not comparable)"),
                flush=True,
            )
    if out:
        (here / "regress.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=1)
        )
    return out
