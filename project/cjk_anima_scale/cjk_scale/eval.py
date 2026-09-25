"""eval — the stage's rulers, delegated to the probe line's stages so every
number is on the ruler the reads of record used.

  exact     ``eval``: the stage's own eval groups, floor-less, ``--seeds``
  native    ``native``: the fixed row sample on scene prompts, ``en`` / ``swap``
  cf_sense  ``cf_sense --cf_lang ja``: the trained rows' caption leverage by σ —
            does it land in the band the stage trained in
  sent      ``native --eval_tag sent``: ``sent_strings`` (multi-glyph) × ``en``
  target    ``target``: the user's verbatim captions (はい / こんにちは)

A run with ``context = "seed"`` reads every ruler on the overlay — the
seed table with this table's rows on top — under
``rows_scale_<stage>_<tag>_ctx/`` (``--arm_tag ctx``): a string's pieces
outside the inventory render as they rode in training, not as raw pack rows.

The probe stages open ``data_scale_<stage>_<tag>`` / ``rows_scale_<stage>_<tag>``
through their own ``--data_tag``, so nothing is copied. The warm-chain
regression check compares this table's ``eval_reads.json`` with the earlier
stages' on the groups they share (same units + seed → same eval strings).

``seed_only`` (micro_chain_result.md § 4) evaluates the run's ``seed_table``
instead of a trained one, under ``rows_scale_<stage>_<tag>_seed/`` (the
probe's ``--arm_tag seed``) with the stage's data dir: the baseline every
"vs seed" rule of the plan reads against. The wrapper table keeps only the
seed rows the stage's inventory names (``words.json``), so ``cf_sense`` draws
its pairs from the same rows a trained table would carry.
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


RULERS = ("eval", "native", "cf_sense", "sent", "target")


def run(
    cfg: StageConfig,
    tag: str,
    which=("eval", "native", "cf_sense", "sent", "target"),
    extra=None,
    seed_only: bool = False,
) -> None:
    from stages import run as run_stage

    extra = list(extra or [])
    if seed_only:
        out = seed_wrapper(cfg, tag)
        extra += ["--arm_tag", SEED_ARM_TAG]
    elif cfg.context_table() is not None:
        out = context_wrapper(cfg, tag)
        extra += ["--arm_tag", CTX_ARM_TAG]
    else:
        out = arm_dir(cfg.stage, tag)
    assert (out / "trained.pt").exists(), f"no table at {out / 'trained.pt'}"
    e = cfg.eval
    which = [
        w
        for w in which
        if (w != "cf_sense" or e["cf_sense"])
        and (w != "sent" or e["sent_strings"])
        and (w != "target" or e["target"])
    ]
    probe = [w for w in which if w not in ("sent",)]
    a = probe_args(cfg, tag, probe, extra)
    for name in which:
        print(
            f"===== {cfg.stage} eval{' (seed)' if seed_only else ''}: {name}",
            flush=True,
        )
        if name == "sent":
            s = probe_args(
                cfg,
                tag,
                ["native"],
                extra
                + [
                    "--eval_tag",
                    "sent",
                    "--native_chars",
                    str(e["sent_strings"]),
                    "--native_clauses",
                    "en",
                ],
            )
            run_stage("native", s)
        else:
            run_stage(name, a)
    if not seed_only and "eval" in which:
        regress(cfg, tag, out)


SEED_ARM_TAG = "seed"
CTX_ARM_TAG = "ctx"


def context_wrapper(cfg: StageConfig, tag: str) -> Path:
    """``rows_scale_<stage>_<tag>_ctx/trained.pt``: the context (seed) table
    with this stage's rows on top, in this table's ``row_scale`` (the
    ``merge_tables`` rescale). Rebuilt on every eval — the stage table may
    have been retrained since."""
    import torch

    ctx_path = cfg.context_table()
    own = torch.load(
        arm_dir(cfg.stage, tag) / "trained.pt", map_location="cpu", weights_only=False
    )
    ctx = torch.load(ctx_path, map_location="cpu", weights_only=False)
    rs = float(own["delta"]["row_scale"])
    k = float(ctx["delta"]["row_scale"]) / rs
    rows = {
        int(e): r.float() * k for e, r in zip(ctx["delta"]["ext_ids"], ctx["delta"]["raw"])
    }
    n_ctx = len(rows)
    for e, r in zip(own["delta"]["ext_ids"], own["delta"]["raw"]):
        rows[int(e)] = r.float()
    ids = sorted(rows)
    out = arm_dir(cfg.stage, f"{tag}_{CTX_ARM_TAG}")
    out.mkdir(parents=True, exist_ok=True)
    sd = {
        **{k2: v for k2, v in own.items() if k2 != "delta"},
        "delta": {
            **own["delta"],
            "ext_ids": ids,
            "raw": torch.stack([rows[e] for e in ids]),
            "row_scale": rs,
        },
        "context": str(ctx_path),
        "context_rows": n_ctx,
    }
    torch.save(sd, out / "trained.pt")
    print(
        f"context wrapper: {len(own['delta']['ext_ids'])} stage rows over "
        f"{n_ctx} context rows ({ctx_path}, × {k:.4f}) → {out / 'trained.pt'} "
        f"({len(ids)} rows)",
        flush=True,
    )
    return out


def seed_arm_dir(cfg: StageConfig, tag: str) -> Path:
    """``rows_scale_<stage>_<tag>_seed`` — what the probe's ``--arm_tag seed``
    resolves to next to the stage's data dir."""
    return arm_dir(cfg.stage, f"{tag}_{SEED_ARM_TAG}")


def seed_wrapper(cfg: StageConfig, tag: str, row_text=None) -> Path:
    """Write the run's ``seed_table`` as a rows-arm ``trained.pt`` under
    ``seed_arm_dir``, restricted to the rows whose text the stage's
    ``words.json`` names (all of them when no decoder is available). The
    merged seed table carries ``delta`` / ``arm`` / ``merged_from`` /
    ``killed`` and no ``args``; the wrapper adds a synthetic ``args`` so the
    probe readers that open it see the trained shape. Returns the arm dir."""
    import torch

    from .paths import data_dir

    assert cfg.run and cfg.run.seed_table, "--seed_only needs a run with a seed_table"
    src_path = cfg.__class__(**{**cfg.__dict__, "warm_from": ""}).warm_table(tag)
    assert src_path and src_path.exists(), f"seed table {src_path} does not exist"
    src = torch.load(src_path, map_location="cpu", weights_only=False)
    ids = [int(e) for e in src["delta"]["ext_ids"]]
    words = json.loads(
        (data_dir(cfg.stage, tag) / "words.json").read_text(encoding="utf-8")
    )
    inventory = {t for v in words.values() for t in v}
    if row_text is None:
        from probe.merge_tables import row_text_map

        row_text = row_text_map(ids)
    keep = [i for i, e in enumerate(ids) if row_text.get(e) in inventory]
    if row_text and keep:
        delta = {
            **src["delta"],
            "ext_ids": [ids[i] for i in keep],
            "raw": src["delta"]["raw"][keep].clone(),
        }
        note = f"{len(keep)} of {len(ids)} rows (the stage's inventory of {len(inventory)})"
    else:
        delta = src["delta"]
        note = f"all {len(ids)} rows (no row text — inventory filter skipped)"
    out = seed_arm_dir(cfg, tag)
    out.mkdir(parents=True, exist_ok=True)
    sd = {
        "delta": delta,
        "arm": "rows",
        "args": {
            "seed_only": True,
            "seed_table": str(src_path),
            "stage": cfg.stage,
            "run": cfg.run.name,
            "init_rows": str(src_path),
        },
        "killed": "",
        "warm_rows": len(delta["ext_ids"]),
        "merged_from": src.get("merged_from"),
    }
    torch.save(sd, out / "trained.pt")
    print(f"seed wrapper: {src_path} → {out / 'trained.pt'}: {note}", flush=True)
    return out


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


def regress(cfg: StageConfig, tag: str, here: Path | None = None) -> dict:
    """The warm-chain check (design § 5): the earlier stages' exact groups,
    read on this table vs on theirs. Writes ``regress.json`` in the arm
    dir the rulers wrote to (``here``; a context run's ``_ctx`` dir, read
    against the earlier stages' ``_ctx`` dirs when they have one) and prints
    one line per (stage, group)."""
    here = here or arm_dir(cfg.stage, tag)
    mine = exact_by_group(here)
    ctx = here.name.endswith(f"_{CTX_ARM_TAG}")
    out: dict = {}
    for prev in cfg.eval.get("regress", []):
        pdir = arm_dir(prev, tag)
        if ctx and (arm_dir(prev, f"{tag}_{CTX_ARM_TAG}") / "eval_reads.json").exists():
            pdir = arm_dir(prev, f"{tag}_{CTX_ARM_TAG}")
        theirs = exact_by_group(pdir)
        if not theirs:
            print(
                f"regress vs {prev}: no eval_reads.json under {pdir}",
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
