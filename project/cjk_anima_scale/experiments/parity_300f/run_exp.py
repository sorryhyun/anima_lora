#!/usr/bin/env python
"""parity_300f — does the one-file shape reproduce run0925_300f? (plan.md § 6-3)

The refactor's test without the 3.2 h retrain: every leg replays a piece of
the stage-shaped run (job ``20260925-144054-bcf21e``) through the new code
on the old run's own data and rows, and diffs it against what that run left
on disk. The data side needs no leg — the builder already reproduced the
stage builds' records and pixels (CLAUDE.md, verified 2026-09-25).

plan   (CPU; all ``--dry_run`` does) — the vocabs file vs the old
       ``words.json``; ``train.plan`` on the old data (rows split, schedule,
       every trainer constant) vs the old ``train_record.json``; the old
       vocabs-only ``trained.pt`` through ``rows.merge_seed`` vs the old ctx
       overlay (``rows_…_ctx/trained.pt``) row by row; the piece ruler's
       vocabs.
steps  (GPU) — ``train`` for the first ``--steps`` steps on the old data
       (full-length schedule), its log vs the old ``train_log.json`` at the
       shared log steps: same seed, same batches, same σ draws.
eval   (GPU) — the merged old rows as ``<name>/trained.pt``, ``eval.run``
       on them (floor + trained, every ruler), each ruler's reads vs the old
       ctx arm's (trained) and the seed rows' reads of record (floor).

The parity run is ``output/cjk_anima_scale/parity_300f_<label>/``: its
``data/`` is symlinks into the old data dir plus a ``vocabs.json`` (the old
dir predates it); nothing is written into a ``data_*`` dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap, legacy_arm_dir  # noqa: E402
from cjk_scale.paths import legacy_data_dir  # noqa: E402

bootstrap()

from bench._common import make_run_dir, write_result  # noqa: E402

RUN = "run0925_300f"
STAGE = "joint0507_0305"
OLD_DATA = legacy_data_dir(STAGE, RUN)
OLD_ROWS = legacy_arm_dir(STAGE, RUN)
OLD_CTX = legacy_arm_dir(STAGE, f"{RUN}_ctx")
SEED_DIR = SEED_ROWS.parent  # the floor reads of record (floor_score.md)
# the old record's keys that name the same thing under another key
RECORD_ALIASES = {"bands": "per_item_band"}
# keys that name the run, not the training (the parity run has its own name;
# the run file at the old path is now the one-file shape), and train_steps
# (the old record wrote 0 = steps_per_row × n_rows; compared on its own)
RECORD_SKIP = {"run", "run_config", "vocabs", "data", "train_steps"}
# the log fields compared in the steps leg; lr is exact, the rest bf16 + compile
LOG_KEYS = ("loss", "in_box", "out_box", "warm_drift", "warm_cos", "rel", "lr")
# reads.json totals may differ by the render noise floor (reports/next_2026_09_25.md § 4a: ±2)
NOISE = 2


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument(
        "--legs",
        nargs="+",
        default=["plan"],
        choices=["plan", "steps", "eval"],
        help="plan is CPU; steps / eval need the GPU (submit via make daemon-run)",
    )
    p.add_argument("--steps", type=int, default=50, help="steps leg: loop length")
    p.add_argument("--dry_run", action="store_true", help="the plan leg only")
    return p.parse_args()


def parity_rc(label: str):
    from cjk_scale.config import load_run

    rc = load_run(RUN)
    return type(rc)(
        name=f"parity_300f_{label}", path=rc.path, vocabs=rc.vocabs, read=rc.read
    )


def data_view(rc) -> Path:
    """``<name>/data/``: the old data dir's files as symlinks + ``vocabs.json``
    (the vocabs file's lines — the plan leg checks them against the old
    ``words.json``)."""
    from cjk_scale.paths import data_dir

    d = data_dir(rc.name)
    d.mkdir(parents=True, exist_ok=True)
    for src in OLD_DATA.iterdir():
        if src.name == "words.json":
            continue
        link = d / src.name
        if not link.exists():
            link.symlink_to(src)
    (d / "vocabs.json").write_text(
        json.dumps(vocab_lines(rc), ensure_ascii=False), encoding="utf-8"
    )
    return d


def vocab_lines(rc) -> list[str]:
    """The run's vocabs through the builder's own spec parser (a vocabs file
    is TSV with ``#`` lines; the first column is the vocab)."""
    from data.vocabs import parse_vocabs

    vs = [v for s in parse_vocabs(rc.vocab_specs()) for v in s.vocabs]
    return list(dict.fromkeys(vs))


# ---------------------------------------------------------------------------
# plan (CPU)


def leg_plan(rc) -> dict:
    from cjk_scale import train as T
    from cjk_scale.eval import piece_vocabs
    from cjk_scale.rows import merge_seed
    from common.models import encode_captions, ext_ids_of

    out: dict = {}
    data = data_view(rc)

    # vocabs: the run file's vocabs file vs what the old build called its inventory
    words = json.loads((OLD_DATA / "words.json").read_text(encoding="utf-8"))
    old_inv = {t for v in words.values() for t in v}
    new_inv = set(vocab_lines(rc))
    out["vocabs"] = {
        "n_new": len(new_inv),
        "n_old": len(old_inv),
        "only_new": sorted(new_inv - old_inv),
        "only_old": sorted(old_inv - new_inv),
        "pass": new_inv == old_inv,
    }

    # train.plan on the old data vs the old record
    recs, _ev, vocabs = T.load_items(data)
    te = data / "te_cache"
    meta = json.loads((te / "meta.json").read_text())
    assert meta["done"] == meta["n"], f"{te} is not a finished TE cache — no CPU plan"
    touched = ext_ids_of(encode_captions([r["caption"] for r in recs], "cpu", te))
    p = T.plan(rc, data, recs, vocabs, touched)
    old = json.loads((OLD_ROWS / "train_record.json").read_text(encoding="utf-8"))
    old["train_steps"] = old["train_steps"] or old["steps_per_row"] * old["n_rows"]
    diffs = {}
    for k, v in p.record.items():
        ok_key = RECORD_ALIASES.get(k, k)
        if k in RECORD_SKIP or ok_key not in old:
            continue
        a, b = _norm(v), _norm(old[ok_key])
        if a != b:
            diffs[k] = {"new": v, "old": old[ok_key]}
    steps_ok = p.steps == old["train_steps"]
    out["record"] = {
        "compared": sorted(
            k
            for k in p.record
            if k not in RECORD_SKIP and RECORD_ALIASES.get(k, k) in old
        ),
        "diffs": diffs,
        "train_steps": {"new": p.steps, "old": old["train_steps"]},
        "pass": not diffs and steps_ok,
    }

    # the merge at save vs the old ctx overlay
    import torch

    sd = torch.load(OLD_ROWS / "trained.pt", map_location="cpu", weights_only=False)
    merged, n_extra = merge_seed(sd["delta"], SEED_ROWS)
    ctx = torch.load(OLD_CTX / "trained.pt", map_location="cpu", weights_only=False)
    a = dict(zip(map(int, merged["ext_ids"]), merged["raw"].float()))
    b = dict(zip(map(int, ctx["delta"]["ext_ids"]), ctx["delta"]["raw"].float()))
    same_ids = set(a) == set(b)
    diff = max(float((a[i] - b[i]).abs().max()) for i in set(a) & set(b))
    out["merge"] = {
        "n_merged": len(a),
        "n_ctx": len(b),
        "n_appended": n_extra,
        "same_ids": same_ids,
        "row_scale": [float(merged["row_scale"]), float(ctx["delta"]["row_scale"])],
        "max_abs_diff": diff,
        "pass": same_ids and diff < 1e-5,
    }

    pv = list(piece_vocabs(rc))
    old_pv = sorted(
        {m["text"] for m in _load(OLD_CTX / "native_piece/native_reads.json")}
    )
    out["piece_vocabs"] = {
        "new": pv,
        "piece_report": old_pv,
        "overlap": sorted(set(pv) & set(old_pv)),
    }
    _print_plan(out)
    return out


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return round(float(v), 12)
    return v


def _print_plan(out: dict) -> None:
    v, r, m = out["vocabs"], out["record"], out["merge"]
    print(
        f"vocabs: {v['n_new']} vs old {v['n_old']} — "
        f"{'PASS' if v['pass'] else 'FAIL'} (only new {v['only_new'][:5]}, "
        f"only old {v['only_old'][:5]})",
        flush=True,
    )
    print(
        f"record: {len(r['compared'])} keys, train_steps {r['train_steps']} — "
        f"{'PASS' if r['pass'] else 'FAIL'} {r['diffs'] or ''}",
        flush=True,
    )
    print(
        f"merge: {m['n_merged']} rows vs ctx {m['n_ctx']} ({m['n_appended']} seed rows "
        f"appended), max |Δ| {m['max_abs_diff']:.2e} — {'PASS' if m['pass'] else 'FAIL'}",
        flush=True,
    )
    pv = out["piece_vocabs"]
    print(
        f"piece ruler: {pv['new']} (piece report's 8: {pv['piece_report']}; "
        f"overlap {len(pv['overlap'])})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# steps (GPU)


def leg_steps(rc, n: int) -> dict:
    from cjk_scale.paths import run_dir
    from cjk_scale.train import train

    out = run_dir(rc.name) / "steps"
    data_view(rc)
    train(rc, out=out, max_steps=n)
    new = {r["step"]: r for r in json.loads((out / "train_log.json").read_text())}
    old = {r["step"]: r for r in json.loads((OLD_ROWS / "train_log.json").read_text())}
    rows = []
    for st in sorted(set(new) & set(old)):
        row = {"step": st}
        for k in LOG_KEYS:
            if k in new[st] and k in old[st]:
                a, b = float(new[st][k]), float(old[st][k])
                row[k] = {"new": a, "old": b, "rel": abs(a - b) / max(abs(b), 1e-12)}
        rows.append(row)
    lr_ok = all(r["lr"]["rel"] < 1e-6 for r in rows if "lr" in r)
    s1 = next((r for r in rows if r["step"] == 1), None)
    loss1_ok = bool(s1) and s1["loss"]["rel"] < 1e-2
    for r in rows:
        print(
            f"step {r['step']:>4}: "
            + ", ".join(
                f"{k} {r[k]['new']:.5g}/{r[k]['old']:.5g}" for k in LOG_KEYS if k in r
            ),
            flush=True,
        )
    res = {
        "steps": n,
        "rows": rows,
        "lr_exact": lr_ok,
        "loss_step1_within_1pct": loss1_ok,
        "pass": lr_ok and loss1_ok,
    }
    print(
        f"steps: lr exact {lr_ok}, step-1 loss within 1 % {loss1_ok} — "
        f"{'PASS' if res['pass'] else 'FAIL'} (later steps: read the drift)",
        flush=True,
    )
    return res


# ---------------------------------------------------------------------------
# eval (GPU)


def leg_eval(rc) -> dict:
    import torch
    from cjk_scale import eval as E
    from cjk_scale.paths import trained_path
    from cjk_scale.rows import merge_seed

    data_view(rc)
    sd = torch.load(OLD_ROWS / "trained.pt", map_location="cpu", weights_only=False)
    merged, n_extra = merge_seed(sd["delta"], SEED_ROWS)
    torch.save(
        {
            **sd,
            "delta": merged,
            "seed_merged": str(SEED_ROWS),
            "seed_rows": int(sd.get("context_rows", 0)) + n_extra,
        },
        trained_path(rc.name),
    )
    E.run(rc)
    old_dirs = {E.TRAINED_ARM: OLD_CTX, E.FLOOR_ARM: SEED_DIR}
    res: dict = {}
    for ruler in E.rulers(rc):
        res[ruler] = {}
        for arm in E.ARMS:
            new_ms = E._reads(rc, arm, ruler)
            old_ms = [
                m
                for m in _load(old_dirs[arm] / E.READ_FILES[ruler])
                if m.get("cond", "trained") != "floor"
            ]
            res[ruler][arm] = compare_reads(ruler, new_ms, old_ms)
            c = res[ruler][arm]
            print(
                f"{ruler:<7} {arm:<8} renders {c['n_shared']} shared "
                f"({c['n_new']} new / {c['n_old']} old), same read {c['same_read']}; "
                f"official {c['new']['official']} vs {c['old']['official']}, "
                f"contained {c['new']['contained']} vs {c['old']['contained']}"
                + ("" if c["within_noise"] else "  ← beyond ±2"),
                flush=True,
            )
    return res


def _load(f: Path) -> list:
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _render_key(ruler: str, m: dict) -> tuple:
    if ruler == "eval":
        return (m["group"], m["text"], m["seed"])
    return (m.get("pi", 0), m["text"], m.get("clause", ""), m["seed"])


def _read_sig(m: dict) -> tuple:
    return tuple((r.get("sfx"), r.get("vl")) for r in m.get("reads", []))


def compare_reads(ruler: str, new_ms: list, old_ms: list) -> dict:
    """The two arms' reads on the renders both have: totals on the line's
    metric (``eval._metrics``) and how many renders read identically."""
    from cjk_scale.eval import _metrics

    new = {_render_key(ruler, m): m for m in new_ms}
    old = {_render_key(ruler, m): m for m in old_ms}
    keys = sorted(set(new) & set(old))
    a = _metrics(ruler, [new[k] for k in keys])
    b = _metrics(ruler, [old[k] for k in keys])
    return {
        "n_new": len(new),
        "n_old": len(old),
        "n_shared": len(keys),
        "same_read": sum(_read_sig(new[k]) == _read_sig(old[k]) for k in keys),
        "new": a,
        "old": b,
        "within_noise": all(abs(a[k] - b[k]) <= NOISE for k in a if k != "n"),
    }


# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    legs = ["plan"] if args.dry_run else args.legs
    rc = parity_rc(args.label)
    run_dir = make_run_dir(
        "parity_300f",
        label=args.label,
        root=LINE / "experiments" / "parity_300f" / "results",
    )
    metrics: dict = {}
    if "plan" in legs:
        metrics["plan"] = leg_plan(rc)
    if "steps" in legs:
        metrics["steps"] = leg_steps(rc, args.steps)
    if "eval" in legs:
        metrics["eval"] = leg_eval(rc)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(OUT / rc.name)],
        extra={
            "old_rows": str(OLD_ROWS),
            "old_ctx": str(OLD_CTX),
            "old_data": str(OLD_DATA),
        },
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
