"""budget — steps and items per vocab by kind × glyph count × warm / cold
(plan_2900 § 5-1), and for cold kanji singles by ink (plan_retrain § 1).

The trainer's ``STEPS_PER_VOCAB`` (90) and the builder's ``ITEMS_PER_VOCAB``
are the base; a vocab's **factor** scales both, so items keep pace with the
steps (stage_i scaled both). A rule written as rows with provenance, like
``windows.py``: a new read changes a row and its source string. A vocab no
row matches gets factor 1.

**Singles may mix factors within a run** (``run_budget``): the builder
repeats each single in the draw pools by ``draw_weights`` (the factors'
integer ratio — the lone / count tiers' ``pools.singles``, the windows'
``pools.window_keys``), so a row's exposure keeps pace with its steps, and
the trainer sums each row's own steps. The piece / multi tiers draw vocabs
and corpus lines unweighted, so a piece run's vocabs must share one factor
— ``run_budget`` refuses it; split such a run by factor and join the rows
with ``scale.py <out> merge``.

Warm = every idx of the vocab has a row in the run's context rows
(``paths.SEED_ROWS``, or the run's ``context``) and its kind does not start
cold. **Singles start cold** (``COLD_KINDS``, plan_retrain: the singles
re-seed from the pack rows on lone + in-word data), so a single is never
warm here. Script (kana / kanji) splits the cold single row: the kana point
is P1b's; kanji split again by ink (``glyph_ink``, cells² at 48 px — C3 at
450: the gain over 225 went to the ink-dense glyphs).

``mix_factor``: the steps keep pace with the items a kind's groups draw
(Σ of ``builder.TABLE`` shares) — the single kind's lone 0.5 + in-word
0.5 + 0.5 is × 1.5, so the in-word items keep P1b's exposure (90 → 135).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from pathlib import Path

BASE_STEPS = 90  # train.STEPS_PER_VOCAB of record (run0925_300f; conflict_joint report)


@dataclass(frozen=True)
class Rule:
    kind: str  # windows.KINDS
    glyphs: tuple  # (lo, hi) glyph count, hi None = open
    warm: bool | None  # None = either
    steps: int  # per vocab, at the base budget
    source: str
    script: str | None = None  # "kana" / "kanji" (``script_of``); None = either
    ink: tuple | None = None  # [lo, hi) glyph ink (``glyph_ink``), hi None = open


COLD_KINDS = ("single",)  # retrain_experiments § 6: every single re-seeds cold


RULES = (
    Rule(
        "single",
        (1, 1),
        False,
        90,
        "cold kana: p1_mix (retrain_experiments § 4, hypothesis.md P1b), 36 hiragana "
        "cold at 90 × the in-word mix 1.5 = 135 / row: singles official 82 vs "
        "the floor's 91 / 144 (p 0.69), 8 held-in words ≤ 1 edit 80 / 128 "
        "(floor 1). Katakana at this row is unread (retrain_kana reads it)",
        script="kana",
    ),
    Rule(
        "single",
        (1, 1),
        False,
        150,
        "cold kanji, ink < 10: C3 (retrain_experiments § 4) at 150 × 1.5 = 225 / "
        "row: new kanji official 0 → 51 / 192; at 450 the ink < 10 glyphs did "
        "not gain (official 38 → 34, contained 78 → 81 of 7 × 16) — user "
        "2026-09-28 (plan_retrain § 1)",
        script="kanji",
        ink=(0.0, 10.0),
    ),
    Rule(
        "single",
        (1, 1),
        False,
        225,
        "cold kanji, ink ≥ 10: C3 at 450 / row brought the seed's dense kanji "
        "back to the floor (official 31 → 57 / 192, floor 65, p 0.38) and the "
        "ink ≥ 10.5 glyphs doubled (44 → 86 of 17 × 16); 225 × 1.5 = 337.5 / "
        "row is the midpoint of 225 and 450, unmeasured — user 2026-09-28 "
        "(plan_retrain § 1)",
        script="kanji",
        ink=(10.0, None),
    ),
    Rule(
        "piece",
        (4, 5),
        None,
        270,
        "long_b0 (reports/long_b0_2026_09_27.md): warm, piece table, 90 → 270 "
        "steps / row: 4-glyph contained 7 → 18 / 64 (4 / 4 pieces up), "
        "5-glyph 3 → 4 (near 8 → 13), 3-glyph 13 → 15 (stays at 90); "
        "user 2026-09-27: 270 for 4–5, cold pieces too (plan_2900 § 2 — B0 "
        "sets C-p's; cold is not read)",
    ),
)


def script_of(vocab: str) -> str:
    """``kanji`` when the vocab holds a CJK ideograph, else ``kana`` (kana,
    ー, punctuation)."""
    return "kanji" if any(0x3400 <= ord(c) <= 0x9FFF for c in vocab) else "kana"


# glyph → ink, pinned (``glyph_ink`` computes a glyph it lacks): the budget
# must not depend on which fonts a VM has
INK_TABLE = Path(__file__).resolve().parents[1] / "assets" / "glyph_ink.json"


@cache
def _ink_table() -> dict:
    if not INK_TABLE.is_file():
        return {}
    return json.loads(INK_TABLE.read_text(encoding="utf-8"))


def glyph_ink(ch: str) -> float:
    """Ink of one glyph drawn alone, in latent cells² (``cf_sense._glyph_ink``,
    the C3 read's measure): the pinned table, else rendered."""
    t = _ink_table()
    if ch in t:
        return float(t[ch])
    from eval.cf_sense import _glyph_ink

    return float(_glyph_ink(ch))


def starts_cold(kind: str) -> bool:
    return kind in COLD_KINDS


def _in(x: float, rng: tuple) -> bool:
    lo, hi = rng
    return x >= lo and (hi is None or x < hi)


def rule_for(
    kind: str, glyphs: int, warm: bool, script: str = "kana", ink: float | None = None
) -> Rule | None:
    for r in RULES:
        lo, hi = r.glyphs
        if not (
            r.kind == kind
            and glyphs >= lo
            and (hi is None or glyphs <= hi)
            and (r.warm is None or r.warm == warm)
            and (r.script is None or r.script == script)
        ):
            continue
        if r.ink is not None:
            assert ink is not None, f"rule {r.kind}/{r.script} splits by ink: pass it"
            if not _in(ink, r.ink):
                continue
        return r
    return None


def factor(
    kind: str, glyphs: int, warm: bool, script: str = "kana", ink: float | None = None
) -> float:
    r = rule_for(kind, glyphs, warm, script, ink)
    return 1.0 if r is None else r.steps / BASE_STEPS


def mix_factor(kinds, table=None) -> float:
    """Σ of the table's shares for the run's trained kinds (one value: a run
    whose kinds draw different totals is refused, like ``run_factor``)."""
    from .builder import TABLE

    table = TABLE if table is None else table
    by = {k: sum(g.share for g in table if g.kind == k) for k in set(kinds)}
    assert len(set(by.values())) <= 1, f"kinds draw different item totals {by}"
    return next(iter(by.values()), 1.0) or 1.0


@cache
def seed_ids(rows: Path | None = None) -> frozenset:
    """The idx a rows file holds (default ``paths.SEED_ROWS``)."""
    import torch

    from .paths import SEED_ROWS

    sd = torch.load(rows or SEED_ROWS, map_location="cpu", weights_only=False)
    return frozenset(int(e) for e in sd["delta"]["ext_ids"])


def _vocab_rules(vocabs, tokq, seeds=None) -> dict:
    """vocab → (kind, factor); a vocab with no pack row (nothing trains) is
    left out."""
    from data.inventory import pieces as qpieces

    from .windows import glyph_count, vocab_kind

    seeds = seed_ids() if seeds is None else seeds
    tok, q = tokq
    out = {}
    for v in vocabs:
        ps = qpieces(tok, q, v)
        idx = [e for _p, e in ps if e is not None]
        if not idx:
            continue
        kind = vocab_kind(v, len(ps))
        warm = not starts_cold(kind) and all(int(e) in seeds for e in idx)
        script = script_of(v)
        n = glyph_count(v)
        ink = glyph_ink(v) if kind == "single" and script == "kanji" else None
        out[v] = (kind, factor(kind, n, warm, script, ink))
    return out


def vocab_factors(vocabs, tokq, seeds=None) -> dict:
    """vocab → its factor; a vocab with no pack row (nothing trains) is left
    out."""
    return {v: f for v, (_k, f) in _vocab_rules(vocabs, tokq, seeds).items()}


def _by_factor(f: dict) -> str:
    by: dict = {}
    for v, x in f.items():
        by.setdefault(x, []).append(v)
    return "; ".join(
        f"× {x:g} ({len(vs)}: {' '.join(vs[:6])}{' …' if len(vs) > 6 else ''})"
        for x, vs in sorted(by.items())
    )


def run_budget(vocabs, tokq, seeds=None) -> dict:
    """vocab → factor for a run: singles may mix factors (their draws are
    weighted, ``draw_weights``); any other kind must share one."""
    r = _vocab_rules(vocabs, tokq, seeds)
    for kind in sorted({k for k, _f in r.values()} - {"single"}):
        f = {v: x for v, (k, x) in r.items() if k == kind}
        assert len(set(f.values())) <= 1, (
            f"the run's {kind} vocabs fall under different budgets {_by_factor(f)} "
            "— their tiers draw unweighted: split the run by budget and join the "
            "rows with `scale.py <out> merge`"
        )
    return {v: x for v, (_k, x) in r.items()}


def run_factor(vocabs, tokq, seeds=None) -> float:
    """The run's one factor; refuses a run whose vocabs fall under different
    factors (``run_budget`` is what the builder and trainer read)."""
    f = vocab_factors(vocabs, tokq, seeds)
    assert len(set(f.values())) <= 1, (
        f"the run's vocabs fall under different budgets {_by_factor(f)} — split "
        "the run by budget and join the rows with `scale.py <out> merge`"
    )
    return next(iter(f.values()), 1.0)


MAX_DRAW_WEIGHT = 16


def draw_weights(budget: dict) -> dict:
    """vocab → how many times it sits in a draw pool: the factors' ratio as
    small integers (225 : 337.5 → 2 : 3); all 1 when the factors agree, so
    a one-budget run draws exactly what it drew before."""
    if not budget:
        return {}
    lo = min(budget.values())
    r = {
        v: Fraction(x / lo).limit_denominator(MAX_DRAW_WEIGHT)
        for v, x in budget.items()
    }
    den = math.lcm(*(q.denominator for q in r.values()))
    w = {v: int(q * den) for v, q in r.items()}
    assert max(w.values()) <= MAX_DRAW_WEIGHT, (
        f"draw weights past {MAX_DRAW_WEIGHT}: {sorted(set(w.values()))} — the "
        "factors' ratio is not a small fraction"
    )
    return w
