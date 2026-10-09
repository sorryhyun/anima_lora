"""A reseed run is one file, ``configs/<run>.toml``::

    rows = ["chars:あい…"]       # the rows trained (data.vocabs specs, single glyphs);
                                 # a mark row (not a letter, not ``・``) trains in mark
                                 # windows around seed_retrain_0930's letters and in
                                 # ``sent`` lines holding it (hearts synthesised into
                                 # the lines), and is drawn alone only if its lone
                                 # spelling encodes to its row
    read = ["こんにちは", …]      # held out of the windows by trigram
    seed = "0921"                # the rows every other row rides frozen at: "0921" (the
                                 # old seed, the kana run's), "0930" (seed_retrain_0930)
                                 # or "1008" (seed_1008 = jp_v1's rows, on the punct pack)
    steps_per_row = 135
    shares = { grid_44 = 10, … }  # optional: % of the items per tier, every tier
                                  # named but the opt-in ones (table share 0: left
                                  # out = 0), Σ 100; else table.TABLE's shares
    upper_shift = 0.1             # optional: every tier's upper σ edge moved by this
                                  # (capped at 0.9)
    stick_from = "kana_up"        # optional: a stick run — that run's rows and data,
                                  # its rows' shared mean trained only (no data verb)
    rows_from = "retrain_kana"    # optional: warm from this scale-line run's merged rows
                                  # (``output/cjk_anima_scale/<it>/trained.pt``) — a stick
                                  # run's instead of ``stick_from``'s, the data stays its;
                                  # without ``stick_from`` / ``ball_on`` a plain warm run,
                                  # every row free (``sent_whole``)
    drop_tiers = ["lone_44", …]   # optional: tiers left out of the data at train
    band = [0.75, 0.95]           # optional, stick runs: every kept item's σ band at
                                  # train (the data's stamped bands replaced; not
                                  # the table, so UPPER_MAX does not apply)
    tag_drop = ["japanese text", 0.5]  # optional, stick runs: the tag out of an item's
                                  # caption at this p, drawn per item per step
    ball_on = "retrain_kana"      # optional: a ball run — the rows cold at this
                                  # scale-line run's mean over them, the mean held, the
                                  # rows less it trained; its merged rows the context
    warm = true                   # optional, ball runs: the rows warm at ``ball_on``'s
                                  # rows (not cold at their mean), that mean held
    data_from = "run1002_grid_small/data"  # optional, ball / plain warm runs: a scale-line data dir
                                  # (under ``output/cjk_anima_scale``) instead of a build;
                                  # a bare name (no ``/``) = that reseed run's ``data/``
    lr = 2e-4                     # optional: the rows' peak lr (``cjk_scale.train.LR``,
                                  # 1e-3, without it)
    pack = "punct"                # optional: the base pack (``PACKS``) instead of the raw
                                  # pack — its routing at build, train and read
    lines = "m109_pack"           # optional: the dialogue line file (``LINES``) the windows
                                  # and ``sent`` lines draw from, instead of the scale
                                  # line's ``PHRASE_FILE`` (dialogue_2_10)
    free_residual = 0.0           # optional: the trainer's norm pull μ‖f‖² (``cjk_scale.train
                                  # .FREE_RESIDUAL`` without it); 0 off — under AdamW it
                                  # walks a rare warm row to the pack row (``sent_kanji``)
    row_lr = [0.12, 1.0]          # optional: one factor per ``rows`` spec on its rows'
                                  # step (``cjk_scale.train(row_step_scale=)``: AdamW's
                                  # update scaled, a per-row lr); ``chars:`` specs only
    pres = { lam = 10, band = [0.8, 0.9], every = 2 }  # optional: λ · L_pres on every
                                  # ``every``-th step at σ ~ U(band) (``cjk_scale.train
                                  # (pres=)``, probes/probe_pres_train.py)
    focus = { rows = "chars:剣頼…", line_kanji = 3, window_kanji = 1 }  # optional: every
                                  # item holds one of these rows (a subset of ``rows``,
                                  # every row still trained): bubble1 draws them alone,
                                  # bubbleN their windows of ≤ window_kanji kanji, sent
                                  # the lines holding one with ≤ line_kanji kanji (the
                                  # kana : kanji share kept near sent_kanji's); items and
                                  # steps are ``steps_per_row`` × these rows
                                  # (``sent_kanji_225``)
    held = "$MANGA109S/derived/b5_held.tsv"  # optional: strings held out beside ``read``
                                  # — of the windows by trigram, of the ``sent`` lines by
                                  # 5-gram (the ruler's rule); a corpus file outside the
                                  # repo (first tsv column, ``#`` lines skipped)
    lang = { korean = "가힝…", chinese = "你这…" }  # optional: the rows lettered in
                                  # another language (glyph → language, named per row:
                                  # ``个`` is in Shift-JIS, so no encoding test tells): the
                                  # faces in ``../cjk_anima_scale/assets/fonts/kozh/`` join
                                  # the draw, a scene caption names the item's language
                                  # (``korean text`` / ``Korean text reads as``), and a
                                  # tier with nothing to draw for the rows (no window)
                                  # drops, the others scaled back to the table's Σ

Its outputs land in ``output/cjk_anima_reseed/<run>/`` (``data/``,
``trained.pt``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from . import CONFIGS, OUT

KEYS = (
    "rows",
    "read",
    "seed",
    "steps_per_row",
    "shares",
    "upper_shift",
    "stick_from",
    "rows_from",
    "drop_tiers",
    "band",
    "tag_drop",
    "ball_on",
    "warm",
    "data_from",
    "pack",
    "lr",
    "lines",
    "row_lr",
    "free_residual",
    "pres",
    "focus",
    "held",
    "lang",
)
# a run's ``seed``: the rows every other row rides frozen at
SEEDS = ("0921", "0930", "1008")
SEED_1008 = OUT / "seed_1008" / "trained.pt"  # transplant.py write (plan.md § 3)
LANGS = ("korean", "chinese")
# the base packs a run may sit on (``punct_pack.py``): the raw pack's rows and
# ids plus encode rules / appended rows, so the seed rows ride on it unchanged
PACKS = {"punct": "models/vocab_packs/anima_cjk_vocab_pack_punct"}
# the dialogue line files a run may draw from (``~/manga109s/derived/make_*.py``):
# ``m109_pack`` = every Manga109-s text on the pack's rows (10-05; dialogue_2_10
# holds the old cjk_renderable charset — no 応 / 転 / 壊)
# ``m109_b5`` = the same on the pack's rows + retrain_kanji_b5's 225 (10-08)
LINES = {
    "m109_pack": "$MANGA109S/derived/dialogue_pack.tsv",
    "m109_b5": "$MANGA109S/derived/dialogue_pack_b5.tsv",
}
UPPER_MAX = 0.9  # tests/test_boundary.py: no band past it


@dataclass(frozen=True)
class Run:
    name: str
    path: Path
    rows: tuple
    read: tuple
    seed: str
    steps_per_row: int
    shares: dict | None = None  # tier → % of the items
    upper_shift: float = 0.0  # added to every tier's upper σ edge
    stick_from: str = ""  # a stick run: warm from this run, its data, the mean trained
    rows_from: str = ""  # warm from this scale-line run (stick / plain warm runs)
    drop_tiers: tuple = ()  # left out of the data at train
    band: tuple | None = None  # every kept item's σ band at train (stick runs)
    tag_drop: tuple | None = None  # (tag, p): out of a caption at p (stick runs)
    ball_on: str = ""  # a ball run: the mean held at this scale-line run's
    warm: bool = False  # a ball run: the rows warm at ball_on's, not cold at its mean
    data_from: str = ""  # a ball / plain warm run: this data dir
    pack: str = ""  # the base pack (``PACKS``); "" = the raw pack
    lr: float = 0.0  # the rows' peak lr; 0 = cjk_scale.train.LR
    lines: str = ""  # the dialogue line file (``LINES``); "" = the scale line's
    row_lr: tuple = ()  # one step factor per ``rows`` spec; () = all 1
    free_residual: float | None = None  # the norm pull; None = the trainer's
    pres: tuple | None = None  # (λ, σ_lo, σ_hi, every): L_pres on; None = off
    focus: tuple = ()  # every item holds one of these glyphs; () = off
    focus_kanji: tuple = ()  # (line, window): the most kanji a focus item may hold
    held: str = ""  # strings held out beside ``read`` (a path, $MANGA109S expanded)
    lang: dict | None = None  # glyph → language for the rows not lettered as Japanese

    def phrase_file(self) -> str:
        """The dialogue line file: ``LINES[lines]``, else the scale line's
        ``phrase_file()``."""
        from cjk_scale.config import phrase_file

        if not self.lines:
            return phrase_file()
        return _expand(LINES[self.lines])

    def held_strings(self) -> tuple:
        """The ``held`` file's strings (first tsv column, blank / ``#`` lines
        skipped); () without one."""
        if not self.held:
            return ()
        lines = Path(_expand(self.held)).read_text(encoding="utf-8").splitlines()
        return tuple(
            dict.fromkeys(
                ln.split("\t")[0] for ln in lines if ln.strip() and ln[0] != "#"
            )
        )

    def steps(self, steps_per_row: int | None = None) -> int | None:
        """The whole step count a ``focus`` run trains (its rows ×
        ``steps_per_row``); ``None`` without one (every trained row's)."""
        if not self.focus:
            return None
        return (steps_per_row or self.steps_per_row) * len(self.focus)

    def row_step_scale(self) -> dict | None:
        """``{glyph: factor}`` for the rows ``row_lr`` scales (factor ≠ 1)."""
        out = {
            g: float(f)
            for spec, f in zip(self.rows, self.row_lr)
            if f != 1
            for g in spec.split(":", 1)[1]
        }
        return out or None

    def use_pack(self) -> None:
        """Name the run's base pack (``ANIMA_VOCAB_PACK``) before anything
        resolves the checkpoints."""
        import os

        if self.pack:
            os.environ["ANIMA_VOCAB_PACK"] = PACKS[self.pack]

    def table(self) -> tuple:
        """``table.TABLE``, its shares the run's when it gives them (Σ kept at
        the table's, so the items per row stay), every upper edge moved by
        ``upper_shift``."""
        from .table import TABLE

        tbl = TABLE
        if self.shares:
            total = sum(t.share for t in TABLE)
            tbl = tuple(
                replace(t, share=total * self.shares.get(t.name, 0) / 100) for t in tbl
            )
        if self.upper_shift:
            tbl = tuple(
                replace(
                    t,
                    band=(
                        t.band[0],
                        round(min(t.band[1] + self.upper_shift, UPPER_MAX), 4),
                    ),
                )
                for t in tbl
            )
        return tbl

    @property
    def dir(self) -> Path:
        return OUT / self.name

    @property
    def data(self) -> Path:
        if self.data_from:
            from cjk_scale import paths

            if "/" not in self.data_from:  # a reseed run's build
                return OUT / self.data_from / "data"
            return paths.OUT / self.data_from
        return (OUT / self.stick_from if self.stick_from else self.dir) / "data"

    def seed_rows(self) -> Path:
        """The rows the run sits on: a stick run's source rows (merged), else
        the seed's."""
        from cjk_scale import paths

        if self.rows_from:
            return paths.OUT / self.rows_from / "trained.pt"
        if self.ball_on:
            return paths.OUT / self.ball_on / "trained.pt"
        if self.stick_from:
            return OUT / self.stick_from / "trained.pt"

        return {
            "0921": paths.SEED_ROWS_0921,
            "0930": paths.SEED_ROWS,
            "1008": SEED_1008,
        }[self.seed]

    def scale_config(self):
        """The ``cjk_scale.train`` view of the run."""
        from cjk_scale.config import RunConfig

        return RunConfig(
            name=self.name, path=self.path, vocabs=self.rows, read=self.read
        )


def _expand(raw: str) -> str:
    """``raw`` with ``$MANGA109S`` / ``~`` expanded (``load_dotenv`` first, so a
    daemon child finds it); asserts the file is there."""
    import os

    from library.env import load_dotenv

    load_dotenv()
    p = os.path.expanduser(os.path.expandvars(raw))
    assert "$" not in p and Path(p).is_file(), f"{raw}: no file {p}"
    return p


def load(run: str) -> Run:
    path = CONFIGS / f"{run}.toml" if "/" not in run else Path(run)
    assert path.is_file(), f"no run config {path}"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    extra = sorted(set(raw) - set(KEYS))
    assert not extra, f"{path}: a run is {{{', '.join(KEYS)}}} — not {extra}"
    assert raw.get("seed") in SEEDS, f"{path}: seed is one of {SEEDS}"
    shares = raw.get("shares")
    if shares is not None:
        from .table import TABLE

        names = {t.name for t in TABLE}
        opt_in = {t.name for t in TABLE if not t.share}
        missing = names - set(shares) - opt_in
        assert not missing and set(shares) <= names, (
            f"{path}: shares names every tier (an opt-in one may be left out) — "
            f"missing {sorted(missing)}, unknown {sorted(set(shares) - names)}"
        )
        assert abs(sum(shares.values()) - 100) < 1e-9, f"{path}: shares sum to 100"
    from .table import TABLE

    drop = tuple(raw.get("drop_tiers", ()))
    assert set(drop) <= {t.name for t in TABLE}, f"{path}: drop_tiers {drop}"
    rows_from = raw.get("rows_from", "")
    if rows_from:
        assert not raw.get("ball_on"), f"{path}: rows_from or ball_on, not both"
    band = raw.get("band")
    if band is not None:
        assert raw.get("stick_from"), (
            f"{path}: band is a stick run's (its data is built)"
        )
        assert len(band) == 2 and 0 <= band[0] < band[1] < 1, f"{path}: band {band}"
    tag_drop = raw.get("tag_drop")
    if tag_drop is not None:
        assert raw.get("stick_from"), f"{path}: tag_drop is a stick run's"
        assert (
            len(tag_drop) == 2 and isinstance(tag_drop[0], str) and 0 < tag_drop[1] < 1
        ), f"{path}: tag_drop {tag_drop}"
    ball_on, data_from = raw.get("ball_on", ""), raw.get("data_from", "")
    if ball_on:
        assert not raw.get("stick_from"), f"{path}: ball_on or stick_from, not both"
    if data_from:
        assert ball_on or rows_from, f"{path}: data_from is a ball / plain warm run's"
        assert not raw.get("stick_from"), f"{path}: a stick run trains on its data"
    warm = bool(raw.get("warm", False))
    if warm:
        assert ball_on, f"{path}: warm is a ball run's"
    lr = float(raw.get("lr", 0.0))
    assert 0 <= lr < 1e-2, f"{path}: lr {lr}"
    pack = raw.get("pack", "")
    assert not pack or pack in PACKS, f"{path}: pack is one of {sorted(PACKS)}"
    lines = raw.get("lines", "")
    assert not lines or lines in LINES, f"{path}: lines is one of {sorted(LINES)}"
    fr = raw.get("free_residual")
    assert fr is None or 0 <= float(fr) < 1, f"{path}: free_residual {fr}"
    row_lr = tuple(float(f) for f in raw.get("row_lr", ()))
    if row_lr:
        assert len(row_lr) == len(raw["rows"]), f"{path}: row_lr is one per rows spec"
        assert all(0 < f <= 1 for f in row_lr), f"{path}: row_lr in (0, 1]"
        assert all(r.startswith("chars:") for r in raw["rows"]), (
            f"{path}: row_lr scales chars: specs only"
        )
    pres = raw.get("pres")
    if pres is not None:
        assert set(pres) == {"lam", "band", "every"}, (
            f"{path}: pres {{lam, band, every}}"
        )
        lo_p, hi_p = pres["band"]
        assert pres["lam"] > 0 and 0 <= lo_p < hi_p <= UPPER_MAX, f"{path}: pres {pres}"
        assert int(pres["every"]) >= 1, f"{path}: pres every {pres['every']}"
        pres = (float(pres["lam"]), float(lo_p), float(hi_p), int(pres["every"]))
    focus, focus_kanji = raw.get("focus"), ()
    if focus is not None:
        assert set(focus) == {"rows", "line_kanji", "window_kanji"}, (
            f"{path}: focus {{rows, line_kanji, window_kanji}}"
        )
        assert focus["rows"].startswith("chars:"), (
            f"{path}: focus rows is a chars: spec"
        )
        trained = {g for r in raw["rows"] for g in r.split(":", 1)[1]}
        focus_kanji = (int(focus["line_kanji"]), int(focus["window_kanji"]))
        focus = tuple(dict.fromkeys(focus["rows"].split(":", 1)[1]))
        assert set(focus) <= trained, (
            f"{path}: focus rows outside rows {sorted(set(focus) - trained)[:10]}"
        )
        assert min(focus_kanji) >= 1, f"{path}: focus kanji caps {focus_kanji}"
    held = raw.get("held", "")
    assert isinstance(held, str), f"{path}: held is a path"
    lang = raw.get("lang")
    if lang is not None:
        assert set(lang) <= set(LANGS), f"{path}: lang names {LANGS}"
        trained = {g for r in raw["rows"] for g in r.split(":", 1)[1]}
        lang = {g: name for name, gs in lang.items() for g in gs}
        assert set(lang) <= trained, (
            f"{path}: lang glyphs outside rows {sorted(set(lang) - trained)}"
        )
    return Run(
        name=path.stem,
        path=path,
        rows=tuple(raw["rows"]),
        read=tuple(raw.get("read", ())),
        seed=raw["seed"],
        steps_per_row=int(raw["steps_per_row"]),
        shares=dict(shares) if shares is not None else None,
        upper_shift=float(raw.get("upper_shift", 0.0)),
        stick_from=raw.get("stick_from", ""),
        rows_from=rows_from,
        drop_tiers=drop,
        band=tuple(band) if band is not None else None,
        tag_drop=tuple(tag_drop) if tag_drop is not None else None,
        ball_on=ball_on,
        warm=warm,
        data_from=data_from,
        pack=pack,
        lr=lr,
        lines=lines,
        row_lr=row_lr,
        free_residual=None if fr is None else float(fr),
        pres=pres,
        focus=tuple(focus or ()),
        focus_kanji=focus_kanji,
        held=held,
        lang=lang,
    )
