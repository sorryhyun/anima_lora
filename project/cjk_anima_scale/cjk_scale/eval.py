"""eval — a run's rulers on two arms, one sheet: the seed floor and the
trained table, delegated to the vendored stages so every number is on the
ruler the reads of record used.

  eval      ``eval``: the automatic groups — ``word`` (18 of the piece vocabs),
            ``single`` (18 of the single vocabs, or the kana sample), ``en``
            (the EN control) — ``--seeds 2``, floor-less
  native    ``native``: あ / い on the scene prompts, ``en`` / ``swap`` — the
            frozen-row control
  sent      ``native --eval_tag sent``: the run's ``read`` strings × ``en``
  target    ``target``: the user's verbatim captions (``assets/target_prompts.txt``:
            はい / こんにちは)

Terminology (fixed, 2026-09-25): **vocab** = the token string (with its
kind — single / piece / multi), **idx** = its ext id, **row** = its trained
weight. The on-disk ``trained.pt`` schema keeps its keys (``ext_ids``,
``raw``) — every stage reader opens them.

A table an eval renders with is built from exactly two operations —
``load`` (a ``trained.pt`` → idx → row) and ``overwrite`` (one table's rows
over another's, by idx):

- **ctx arm** ``<run>/ctx/``: ``overwrite(seed, trained)`` — a vocab outside
  the run renders as it rode in training (at its seed row), never as a raw
  pack row.
- **floor arm** ``<run>/floor/``: ``load(seed)`` — the whole seed table, the
  baseline every "vs seed" read is against. Its rulers render once per
  string set (the seed never changes); the floor **of record** stays flat in
  the seed table's own dir (``floor_score.md``).

The stages open the run's dirs through ``--data_path`` / ``--arm_path``.
``compose`` then reads both arms' read files into ``<run>/reads.json``
(official = sfx ∧ VL exact, loose = sfx ∨ VL, contained = the string inside
any box read; the ``eval`` ruler's official is its sfx ``exact``) and
``<run>/sheet.png`` (floor | trained per string, seeds 0 / 1, the first
prompt of each).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .config import RunConfig
from .paths import SEED_TABLE, arm_dir, data_dir, run_dir, table_path

CTX_ARM = "ctx"
FLOOR_ARM = "floor"
ARMS = (FLOOR_ARM, CTX_ARM)
RULERS = ("eval", "native", "sent", "target")
EVAL_GROUPS = ("single", "single_kanji", "word", "en")  # the ones a run's eval.json has
NATIVE_CHARS = "あ,い"  # the frozen-row control (plan.md § 2)
NATIVE_CLAUSES = "en,swap"
SENT_CLAUSES = "en"
SEEDS = 2
GEN_STEPS, GEN_CFG, SEED = 28, 4.0, 0
# where each ruler writes its reads, under the arm dir
READ_FILES = {
    "eval": "eval_reads.json",
    "native": "native/native_reads.json",
    "sent": "native_sent/native_reads.json",
    "target": "target/native_reads.json",
}


def eval_groups(rc: RunConfig) -> list[str]:
    ev = json.loads((data_dir(rc.name) / "eval.json").read_text(encoding="utf-8"))
    have = {e["group"] for e in ev}
    return [g for g in EVAL_GROUPS if g in have]


def probe_args(rc: RunConfig, arm: str, stage_names: list, extra: list | None = None):
    """The stages' namespace for one arm of the run (the vendored ``cli``)."""
    from cli import build_parser
    from stages import STAGES

    argv = [
        "--stage",
        *stage_names,
        "--arm",
        "rows",
        "--data_path",
        str(data_dir(rc.name)),
        "--arm_path",
        str(arm_dir(rc.name, arm)),
        "--seeds",
        str(SEEDS),
        "--no_floor",
        "--native_chars",
        NATIVE_CHARS,
        "--native_clauses",
        NATIVE_CLAUSES,
        "--steps",
        str(GEN_STEPS),
        "--cfg",
        str(GEN_CFG),
        "--seed",
        str(SEED),
        *(extra or []),
    ]
    return build_parser(STAGES).parse_args(argv)


def ruler_args(rc: RunConfig, arm: str, ruler: str):
    if ruler == "eval":
        return probe_args(
            rc, arm, ["eval"], ["--eval_groups", ",".join(eval_groups(rc))]
        )
    if ruler == "sent":
        return probe_args(
            rc,
            arm,
            ["native"],
            [
                "--eval_tag",
                "sent",
                "--native_chars",
                ",".join(rc.read),
                "--native_clauses",
                SENT_CLAUSES,
            ],
        )
    return probe_args(rc, arm, [ruler])


def rulers(rc: RunConfig) -> list[str]:
    return [r for r in RULERS if r != "sent" or rc.read]


def run(rc: RunConfig) -> Path:
    """Both arms' rulers, then ``compose``. The ctx arm renders every time
    (the table may have been retrained); the floor arm renders a ruler only
    when its reads are missing or were read on other strings."""
    from stages import run as run_stage

    assert table_path(rc.name).exists(), (
        f"no table at {table_path(rc.name)} — run `scale.py {rc.name} train` first"
    )
    for arm in ARMS:
        out = floor_arm(rc) if arm == FLOOR_ARM else ctx_arm(rc)
        for ruler in rulers(rc):
            if arm == FLOOR_ARM and _fresh(rc, out, ruler):
                print(
                    f"===== {rc.name} {arm}: {ruler} — reads on hand, kept", flush=True
                )
                continue
            print(f"===== {rc.name} {arm}: {ruler}", flush=True)
            a = ruler_args(rc, arm, ruler)
            run_stage("native" if ruler == "sent" else ruler, a)
    return compose(rc)


def _fresh(rc: RunConfig, out: Path, ruler: str) -> bool:
    """The floor's reads of ``ruler`` exist and cover the strings this run
    would render (eval: the eval.json texts of its groups; sent: ``read``)."""
    f = out / READ_FILES[ruler]
    if not f.exists():
        return False
    have = {m["text"] for m in json.loads(f.read_text(encoding="utf-8"))}
    if ruler == "eval":
        ev = json.loads((data_dir(rc.name) / "eval.json").read_text(encoding="utf-8"))
        groups = set(eval_groups(rc))
        return have == {e["text"] for e in ev if e["group"] in groups}
    if ruler == "sent":
        return have == set(rc.read)
    return True


def load(path: Path) -> tuple[dict, dict]:
    """A ``trained.pt`` → (idx → row, the state dict). Rows come back in
    their table's own raw units; the table's ``row_scale`` is in the state
    dict, and composing two tables means bringing one into the other's
    scale first (see ``ctx_arm``)."""
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)
    d = sd["delta"]
    return {int(i): r.float() for i, r in zip(d["ext_ids"], d["raw"])}, sd


def overwrite(base: dict, top: dict) -> dict:
    """top's rows over base's, by idx. The only composition an eval table
    needs: the ctx arm is ``overwrite(seed, trained)``, the floor arm is
    the seed alone."""
    return {**base, **top}


def save_arm(out: Path, rows: dict, sd: dict) -> Path:
    """Serialize idx → row as a rows-arm ``trained.pt`` in ``out`` (an arm
    dir the stages open through ``--arm_path``). ``sd`` carries the metadata
    and the ``row_scale`` the rows are in; only ``ext_ids`` / ``raw`` are
    replaced."""
    import torch

    ids = sorted(rows)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **{k: v for k, v in sd.items() if k != "delta"},
            "delta": {
                **sd["delta"],
                "ext_ids": ids,
                "raw": torch.stack([rows[i] for i in ids]),
            },
        },
        out / "trained.pt",
    )
    return out


def ctx_arm(rc: RunConfig, seed: Path = SEED_TABLE) -> Path:
    """``overwrite(seed, trained)``: the seed rows with the run's trained rows
    on top, in the trained table's ``row_scale`` (the ``merge_tables``
    rescale). Rebuilt on every eval — the run may have been retrained."""
    trained, own = load(table_path(rc.name))
    seed_rows, sd = load(seed)
    k = float(sd["delta"]["row_scale"]) / float(own["delta"]["row_scale"])
    rows = overwrite({i: r * k for i, r in seed_rows.items()}, trained)
    out = save_arm(
        arm_dir(rc.name, CTX_ARM),
        rows,
        {**own, "context": str(seed), "context_rows": len(seed_rows)},
    )
    print(
        f"ctx arm: {len(trained)} trained rows over {len(seed_rows)} seed rows "
        f"({seed}, × {k:.4f}) → {out / 'trained.pt'} ({len(rows)} rows)",
        flush=True,
    )
    return out


def floor_arm(rc: RunConfig, seed: Path = SEED_TABLE) -> Path:
    """``load(seed)`` alone — the whole seed table as a rows-arm
    ``trained.pt``: the floor every "vs seed" read is against, carrying
    every row the ctx arm does (a vocab outside the run renders at its seed
    row, never as a raw pack row). The merged seed table has no ``args``; a
    synthetic one is added so the stage readers that open it see the
    trained shape."""
    assert seed.exists(), f"seed table {seed} does not exist"
    rows, src = load(seed)
    out = save_arm(
        arm_dir(rc.name, FLOOR_ARM),
        rows,
        {
            "delta": src["delta"],
            "arm": "rows",
            "args": {
                "floor": True,
                "seed_table": str(seed),
                "run": rc.name,
                "init_rows": str(seed),
            },
            "killed": "",
            "warm_rows": len(rows),
            "merged_from": src.get("merged_from"),
        },
    )
    print(f"floor arm: {seed} → {out / 'trained.pt'}: all {len(rows)} rows", flush=True)
    return out


# ---------------------------------------------------------------------------
# one sheet, one reads.json


def _metrics(ruler: str, ms: list) -> dict:
    from common.text import norm

    def contained(m):
        t = norm(m["text"])
        return any(
            t and (t in norm(r.get("sfx") or "") or t in norm(r.get("vl") or ""))
            for r in m.get("reads", [])
        )

    if ruler == "eval":
        off = [bool(m.get("exact")) for m in ms]
        loose = [bool(m.get("exact")) or m.get("cer_vl") == 0 for m in ms]
    else:
        off = [bool(m.get("hit_sfx")) and bool(m.get("hit_vl")) for m in ms]
        loose = [bool(m.get("hit_sfx")) or bool(m.get("hit_vl")) for m in ms]
    return {
        "n": len(ms),
        "official": sum(off),
        "loose": sum(loose),
        "contained": sum(contained(m) for m in ms),
    }


def _key(ruler: str, m: dict) -> str:
    if ruler == "eval":
        return f"{m['group']}|{m['text']}"
    if ruler == "target":
        return f"p{m['pi']:02d}|{m['text']}"
    return f"{m['text']}|{m['clause']}"


def _reads(rc: RunConfig, arm: str, ruler: str) -> list:
    f = arm_dir(rc.name, arm) / READ_FILES[ruler]
    if not f.exists():
        return []
    return [
        m
        for m in json.loads(f.read_text(encoding="utf-8"))
        if m.get("cond", "trained") != "floor"
    ]


def compose(rc: RunConfig) -> Path:
    """``<run>/reads.json`` + ``<run>/sheet.png`` from both arms' reads."""
    from common.readers import contact_sheet
    from PIL import Image

    reads: dict = {"run": rc.name, "arms": {a: str(arm_dir(rc.name, a)) for a in ARMS}}
    rows_out: dict = {}
    tiles = []
    for ruler in rulers(rc):
        per: dict = defaultdict(lambda: defaultdict(list))
        for arm in ARMS:
            for m in _reads(rc, arm, ruler):
                per[_key(ruler, m)][arm].append(m)
        table: dict = {}
        totals: dict = defaultdict(lambda: defaultdict(list))
        for key, by_arm in per.items():
            table[key] = {arm: _metrics(ruler, by_arm.get(arm, [])) for arm in ARMS}
            group = key.split("|")[0] if ruler == "eval" else ruler
            for arm in ARMS:
                totals[group][arm] += by_arm.get(arm, [])
            for arm in ARMS:  # floor s0 s1 | trained s0 s1, the first prompt
                ms = sorted(
                    by_arm.get(arm, []), key=lambda m: (m.get("pi", 0), m["seed"])
                )
                pick = [m for m in ms if m.get("pi", 0) == ms[0].get("pi", 0)][:2]
                for m in pick:
                    tiles.append((_tile(m, Image), _label(ruler, key, arm, m)))
                for _ in range(2 - len(pick)):
                    tiles.append((Image.new("RGB", (64, 64), "white"), [f"{arm}: –"]))
        rows_out[ruler] = {
            "totals": {
                g: {arm: _metrics(ruler, ms[arm]) for arm in ARMS}
                for g, ms in totals.items()
            },
            "strings": table,
        }
    reads["rulers"] = rows_out
    out = run_dir(rc.name)
    (out / "reads.json").write_text(
        json.dumps(reads, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if tiles:
        contact_sheet(tiles, out / "sheet.png", thumb=192, cols=4)
    for ruler, block in rows_out.items():
        for g, by_arm in block["totals"].items():
            f, t = by_arm[FLOOR_ARM], by_arm[CTX_ARM]
            print(
                f"{ruler:<7} {g:<14} floor {f['official']}/{f['n']} "
                f"(loose {f['loose']}, contained {f['contained']})  trained "
                f"{t['official']}/{t['n']} (loose {t['loose']}, contained {t['contained']})",
                flush=True,
            )
    print(f"→ {out / 'reads.json'}, {out / 'sheet.png'}", flush=True)
    return out


def _tile(m: dict, Image):
    try:
        return Image.open(m["file"]).convert("RGB")
    except OSError:
        return Image.new("RGB", (64, 64), "white")


def _label(ruler: str, key: str, arm: str, m: dict) -> list:
    boxes = [r for r in m.get("reads", []) if not r.get("whole")] or m.get("reads", [])
    r0 = boxes[0] if boxes else {}
    ok = m.get("exact") if ruler == "eval" else (m.get("hit_sfx") and m.get("hit_vl"))
    return [
        f"{ruler} {key}",
        f"{'trained' if arm == CTX_ARM else 'floor'} s{m['seed']} {'HIT' if ok else '-'}",
        f"sfx {r0.get('sfx') or ''}",
        f"vl {r0.get('vl') or ''}",
    ]
