"""eval — a run's rulers on two arms, one sheet: the seed floor and the
trained rows, delegated to the vendored stages so every number is on the
ruler the reads of record used.

  eval      ``eval``: the automatic groups — ``word`` (18 of the piece vocabs),
            ``single`` (18 of the single vocabs, or the kana sample), ``en``
            (the EN control) — ``--seeds 2``, floor-less
  native    ``native``: あ / い on the scene prompts, ``en`` / ``swap`` — the
            frozen-row control
  piece     ``native --eval_tag piece``: a trained piece alone in a native
            scene, ``en`` / ``swap`` — up to ``PIECE_N`` of the run's piece
            vocabs: the ones the ``read`` strings tokenize into, then the
            ``word`` group's (``piece_vocabs``). The one ruler that sees piece
            identity: ``word`` exact / ``sent`` / ``target`` read a run that
            bought piece natives as dead (reports/piece_2026_09_25.md)
  sent      ``native --eval_tag sent``: the run's ``read`` strings × ``en``
  target    ``target``: the user's verbatim captions (``assets/target_prompts.txt``:
            はい / こんにちは)

Terminology (fixed, 2026-09-25): **vocab** = the token string (with its
kind — single / piece / multi), **idx** = its ext id, **row** = its trained
weight. The on-disk ``trained.pt`` schema keeps its keys (``ext_ids``,
``raw``) — every stage reader opens them.

Two arms, no ``ctx`` sidecar (2026-09-25 — the merge lives in the save,
``rows.Rows.state_dict``):

- **trained**: the run dir itself. ``<run>/trained.pt`` is already the whole
  merged rows — the seed's rows with the run's on top, one ``row_scale`` —
  so a vocab outside the run renders as it rode in training (at its seed
  row), never as a raw pack row. Ruler outputs land at the run root
  (``<run>/native/``, ``<run>/eval_reads.json`` …). A pre-merge vocabs-only
  ``trained.pt`` (no ``seed_merged`` key) is refused — retrain.
- **floor** ``<run>/floor/``: ``load(seed)`` — the whole seed rows, the
  baseline every "vs seed" read is against. Its rulers render once per
  string set (the seed never changes); the floor **of record** stays flat in
  the seed rows' own dir (``floor_score.md``).

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
from functools import cache
from pathlib import Path

from .config import RunConfig
from .paths import SEED_ROWS, data_dir, floor_dir, run_dir, trained_path

FLOOR_ARM = "floor"
TRAINED_ARM = "trained"
ARMS = (FLOOR_ARM, TRAINED_ARM)
RULERS = ("eval", "native", "piece", "sent", "target")
EVAL_GROUPS = ("single", "single_kanji", "word", "en")  # the ones a run's eval.json has
NATIVE_CHARS = "あ,い"  # the frozen-row control (plan.md § 2)
NATIVE_CLAUSES = "en,swap"
PIECE_N = 8  # the piece ruler's vocabs (reports/piece_2026_09_25.md read 8)
PIECE_CLAUSES = "en,swap"
SENT_CLAUSES = "en"
SEEDS = 2
GEN_STEPS, GEN_CFG, SEED = 28, 4.0, 0
# where each ruler writes its reads, under the arm dir
READ_FILES = {
    "eval": "eval_reads.json",
    "native": "native/native_reads.json",
    "piece": "native_piece/native_reads.json",
    "sent": "native_sent/native_reads.json",
    "target": "target/native_reads.json",
}


def arm_out(rc: RunConfig, arm: str) -> Path:
    """The arm's dir: the run dir itself for ``trained``, the ``floor/``
    sidecar for the floor."""
    assert arm in ARMS, arm
    return run_dir(rc.name) if arm == TRAINED_ARM else floor_dir(rc.name)


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
        str(arm_out(rc, arm)),
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


@cache
def piece_vocabs(rc: RunConfig) -> tuple[str, ...]:
    """The piece ruler's vocabs, up to ``PIECE_N``: the run's piece vocabs
    the ``read`` strings tokenize into (one Qwen token, ≥ 2 glyphs), then the
    ``word`` group's texts in ``eval.json`` order (the build's sample of the
    piece vocabs). Empty for a run with no piece vocabs."""
    from data.inventory import pieces as qpieces
    from data.inventory import qwen_pieces

    from .windows import glyph_count

    data = data_dir(rc.name)
    vocabs = set(json.loads((data / "vocabs.json").read_text(encoding="utf-8")))
    tok, q = qwen_pieces()
    out: list[str] = []
    for s in rc.read:
        for p, e in qpieces(tok, q, s):
            if e is not None and p in vocabs and glyph_count(p) >= 2:
                out.append(p)
    ev = json.loads((data / "eval.json").read_text(encoding="utf-8"))
    out += [e["text"] for e in ev if e["group"] == "word"]
    return tuple(dict.fromkeys(out))[:PIECE_N]


def ruler_args(rc: RunConfig, arm: str, ruler: str):
    if ruler == "eval":
        return probe_args(
            rc, arm, ["eval"], ["--eval_groups", ",".join(eval_groups(rc))]
        )
    if ruler == "piece":
        return probe_args(
            rc,
            arm,
            ["native"],
            [
                "--eval_tag",
                "piece",
                "--native_chars",
                ",".join(piece_vocabs(rc)),
                "--native_clauses",
                PIECE_CLAUSES,
            ],
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


def has_pieces(rc: RunConfig) -> bool:
    """The run's ``vocabs.json`` holds a multi-glyph vocab (no tokenizer:
    ``compose`` asks this too; ``piece_vocabs`` does the real split)."""
    from .windows import glyph_count

    f = data_dir(rc.name) / "vocabs.json"
    return f.exists() and any(
        glyph_count(v) >= 2 for v in json.loads(f.read_text(encoding="utf-8"))
    )


def rulers(rc: RunConfig) -> list[str]:
    return [
        r
        for r in RULERS
        if (r != "sent" or rc.read) and (r != "piece" or has_pieces(rc))
    ]


def run(rc: RunConfig) -> Path:
    """Both arms' rulers, then ``compose``. The trained side reads the run's
    merged ``trained.pt`` in place and renders every time (the run may have
    been retrained); the floor arm renders a ruler only when its reads are
    missing or were read on other strings."""
    import torch

    from stages import run as run_stage

    tp = trained_path(rc.name)
    assert tp.exists(), f"no rows at {tp} — run `scale.py {rc.name} train` first"
    sd = torch.load(tp, map_location="cpu", weights_only=False)
    assert sd.get("seed_merged"), (
        f"{tp} is a pre-merge vocabs-only file (no seed_merged) — retrain: "
        f"since 2026-09-25 trained.pt is the whole merged rows"
    )
    for arm in ARMS:
        if arm == FLOOR_ARM:
            out = floor_arm(rc)
        for ruler in rulers(rc):
            if ruler == "piece" and not piece_vocabs(rc):
                print(f"===== {rc.name} {arm}: piece — no piece vocab, skipped")
                continue
            if arm == FLOOR_ARM and _fresh(rc, out, ruler):
                print(
                    f"===== {rc.name} {arm}: {ruler} — reads on hand, kept", flush=True
                )
                continue
            print(f"===== {rc.name} {arm}: {ruler}", flush=True)
            a = ruler_args(rc, arm, ruler)
            run_stage("native" if ruler in ("piece", "sent") else ruler, a)
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
    if ruler == "piece":
        return have == set(piece_vocabs(rc))
    return True


def load(path: Path) -> tuple[dict, dict]:
    """A ``trained.pt`` → (idx → row, the state dict). Rows come back in
    the file's own raw units (delta = raw × its ``row_scale``)."""
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)
    d = sd["delta"]
    return {int(i): r.float() for i, r in zip(d["ext_ids"], d["raw"])}, sd


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


def floor_arm(rc: RunConfig, seed: Path = SEED_ROWS) -> Path:
    """``load(seed)`` alone — the whole seed rows as a rows-arm
    ``trained.pt``: the floor every "vs seed" read is against, carrying
    every row the trained side does (a vocab outside the run renders at its
    seed row, never as a raw pack row). The merged seed file has no
    ``args``; a synthetic one is added so the stage readers that open it see
    the trained shape."""
    assert seed.exists(), f"seed rows {seed} do not exist"
    rows, src = load(seed)
    out = save_arm(
        floor_dir(rc.name),
        rows,
        {
            "delta": src["delta"],
            "arm": "rows",
            "args": {
                "floor": True,
                "seed_rows": str(seed),
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
    f = arm_out(rc, arm) / READ_FILES[ruler]
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

    reads: dict = {"run": rc.name, "arms": {a: str(arm_out(rc, a)) for a in ARMS}}
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
            f, t = by_arm[FLOOR_ARM], by_arm[TRAINED_ARM]
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
        f"{arm} s{m['seed']} {'HIT' if ok else '-'}",
        f"sfx {r0.get('sfx') or ''}",
        f"vl {r0.get('vl') or ''}",
    ]
