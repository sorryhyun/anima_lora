"""A reseed run is one file, ``configs/<run>.toml``::

    rows = ["chars:あい…"]       # the rows trained (data.vocabs specs, single glyphs);
                                 # a mark row (not a letter, not ``・``) trains in mark
                                 # windows around seed_retrain_0930's letters and in
                                 # ``sent`` lines holding it (hearts synthesised into
                                 # the lines), and is drawn alone only if its lone
                                 # spelling encodes to its row
    read = ["こんにちは", …]      # held out of the windows by trigram
    seed = "1008"                # the rows every other row rides frozen at: "0930"
                                 # (seed_retrain_0930) or "1008" (seed_1008 = jp_v1's
                                 # rows, on the punct pack); without ``rows_from`` only
    steps_per_row = 135
    steps = 14336                # optional: the whole step count in place of
                                 # steps_per_row × the rows (cjk_ko 10-12: a run held
                                 # at another's total); the items still count the rows
    shares = { grid_44 = 10, … }  # optional: % of the items per tier, every tier
                                  # named but the opt-in ones (table share 0: left
                                  # out = 0), Σ 100; else table.TABLE's shares
    rows_from = "seed_fixed_1005_stick080"  # instead of ``seed``: warm from this
                                  # scale-line run's merged rows
                                  # (``output/cjk_anima_scale/<it>/trained.pt``), every
                                  # row free (``sent_kanji``); ``<line>/<run>`` names
                                  # ``output/<line>/<run>/trained.pt`` instead
                                  # (``cjk_ko``: ``cjk_anima_reseed/jamo_j256_ko``)
    data_from = "sent_kanji"      # optional: that reseed run's ``data/`` instead of a
                                  # build (``sent_kanji_pres``; ``jamo_f64`` on
                                  # ``jamo_j64``'s)
    lr = 2e-4                     # optional: the rows' peak lr (``trainer.LR``,
                                  # 1e-3, without it)
    pack = "punct"                # optional: the base pack (``PACKS``) instead of the raw
                                  # pack — its routing at build, train and read
    lines = "m109_pack"           # optional: the dialogue line file (``LINES``) the windows
                                  # and ``sent`` lines draw from; ``dialogue_2_10`` (the
                                  # scale line's) without it. A KO file
                                  # (``LINE_LANG``) gives a ``lang`` run its windows and
                                  # ``sent`` lines, lettered horizontal, and its build
                                  # routes Hangul per syllable (``glyph_route_ko``)
    free_residual = 0.0           # optional: the trainer's norm pull μ‖f‖² (``trainer
                                  # .FREE_RESIDUAL`` without it); 0 off — under AdamW it
                                  # walks a rare warm row to the pack row (``sent_kanji``)
    row_lr = [0.12, 1.0]          # optional: one factor per ``rows`` spec on its rows'
                                  # step (``trainer.train(row_step_scale=)``: AdamW's
                                  # update scaled, a per-row lr); ``chars:`` specs only
    pres = { lam = 10, band = [0.8, 0.9], every = 2 }  # optional: λ · L_pres on every
                                  # ``every``-th step at σ ~ U(band) (``trainer.train
                                  # (pres=)``, _archive/probes/probe_pres_train.py)
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
                                  # faces in ``assets/fonts/kozh/`` join
                                  # the draw, a scene caption names the item's language
                                  # (``korean text`` / ``Korean text reads as``), and a
                                  # tier with nothing to draw for the rows (no window)
                                  # drops, the others scaled back to the table's Σ
    factor = "jamo"               # optional: the rows (every one a Hangul syllable)
                                  # composed each step from jamo factors, b + C[cho,
                                  # cls] + V[jung] + F[jong] (``jamo.Jamo``,
                                  # proposal_jamo § 2); cold, no ``row_lr``;
                                  # ``trained.pt`` holds every syllable's composed row.
                                  # A compat jamo row (ㄱ … ㅣ) stays free beside them;
                                  # a partial form (ᅡ … ᅵ, ᆨ … ᇂ: ``jamo.PART_SPELL``)
                                  # composes without its absent parts, drawn by the
                                  # opt-in ``partial`` tier only (cjk_ko, 10-12)
    factor_init = { from = "jamo_lone", scale = 0.577 }  # optional, with ``factor``:
                                  # the jamo curriculum's stage 2 (user 10-11) — C[cho,
                                  # every cls] / V[jung] / F[jong] start at scale × that
                                  # reseed run's compatibility-jamo rows (pack + Δ,
                                  # centred over the 51), ``b`` at 0;
                                  # or { glyph = "pe_spatial", radius = 211.0 }: C / V
                                  # / F start at a glyph prediction's least-squares
                                  # factors (``../cjk_ko/probes/hangul_init.py`` →
                                  # ``output/cjk_ko/hangul_init/<glyph>.pt``), scaled so
                                  # the composed rows' ball (centred over the run's
                                  # syllables) has that mean radius, ``b`` at 0
                                  # (``jamo_j256_ko_cf_pe``, user 10-15)
    factor_from = "cjk_anima_reseed/jamo_j256_ko"  # optional, with ``factor``: the
                                  # factors start at that factor run's (its
                                  # ``trained.pt`` ``jamo``; a name or ``<line>/<run>``
                                  # as ``rows_from``); with ``rows_from`` the factor
                                  # rows run warm (cjk_ko B / B′, 10-11)
    factor_residual = 0.0         # optional, with ``factor``: a per-syllable residual
                                  # on the composition, its norm pull this weight
                                  # (``jamo.Jamo`` ``residual``; cjk_ko B′)
    faces = { out = ["NanumPenScript-Regular.ttf"], px_min = { "BlackHanSans-Regular.ttf" = 32 } }
                                  # optional, with ``lang``: faces (file names) left
                                  # out of the draw, and the font px under which a
                                  # face is not drawn (the item's target px: grid,
                                  # bubble1 / sent ``glyph_px``) — the faces that
                                  # break a syllable small (``jamo_j128_ko``);
                                  # ``dirs = ["ko_extra"]``: every face in these
                                  # ``assets/fonts/`` subdirs joins the draw too
                                  # (``jamo_j128_ko``);
                                  # ``legible = { median = 1.5, p10 = 0.8,
                                  # from_px = 28 }``: from that font px, a face
                                  # is not drawn for a glyph whose stroke gaps it
                                  # fuses (``legible.py``; ``jamo_j1024_ko_cf``)
    upper_shift = 0.1             # optional: every item's upper σ edge moved up by
                                  # this at train (capped at UPPER_MAX), over the
                                  # bands its data stamped — with ``data_from``, an
                                  # arm on another run's items in other bands
                                  # (``hira_cold_up``; kana_up's knob, 10-04)
    init = "random"               # optional, cold free rows only: the trained rows
                                  # start at isotropic random directions at their
                                  # pack rows' norms, the norm pull centred there
                                  # (``hira_cold_rand``, user 10-15); "stick_ball":
                                  # pack row + the seed's stick over the trained rows
                                  # (their mean offset) + a random ball at the seed's
                                  # mean spike length (``hira_cold_sball``, user 10-15);
                                  # "stick_glyph:<set>": the ball each row's predicted
                                  # glyph component (``probes/glyph_init.py``,
                                  # ``hira_cold_sglyph``, user 10-15);
                                  # "stick_lean:<c>": the stick × c, the ball
                                  # leaning against the pack spike as the seed's
                                  # does (``hira_cold_slean``, user 10-16);
                                  # "" = the pack rows
    pose = true                   # optional: the glyph's ink centred in its scene
                                  # bubble / sign and tilted with it, a grid bubble's
                                  # ink on its centre (``render_into_scene(pose=)``,
                                  # ``render_grid(bubble_centre=)``)

Its outputs land in ``output/cjk_anima_reseed/<run>/`` (``data/``,
``trained.pt``).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from . import CONFIGS, HOME, OUT, SCALE_OUT, SEED_ROWS

KEYS = (
    "rows",
    "read",
    "seed",
    "steps_per_row",
    "shares",
    "rows_from",
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
    "factor",
    "factor_init",
    "factor_from",
    "factor_residual",
    "faces",
    "pose",
    "counterfactual",
    "steps",
    "upper_shift",
    "init",
)
# a run's ``seed``: the rows every other row rides frozen at
SEEDS = ("0930", "1008")
# seed_1008: transplant.py write (_archive/plan.md § 3)
SEED_1008 = OUT / "seed_1008" / "trained.pt"
LANGS = ("korean", "chinese")
FACTORS = ("jamo",)
# the base packs a run may sit on (``punct_pack.py``): the raw pack's rows and
# ids plus encode rules / appended rows, so the seed rows ride on it unchanged
PACKS = {"punct": "models/vocab_packs/anima_cjk_vocab_pack_punct"}
# the dialogue line files a run may draw from (``~/manga109s/derived/make_*.py``):
# ``dialogue_2_10`` = the scale line's (2–10 Qwen pieces on the old
# cjk_renderable charset; every run of record before ``lines``), the default
# ``m109_pack`` = every Manga109-s text on the pack's rows (10-05; dialogue_2_10
# holds the old cjk_renderable charset — no 応 / 転 / 壊)
# ``m109_b5`` = the same on the pack's rows + retrain_kanji_b5's 225 (10-08)
# ``ko_chat`` = songys Chatbot_data's lines, the ruler's strings held out
# (``project/cjk_ko/corpus.py``, 10-11)
LINES = {
    "dialogue_2_10": "$MANGA109S/derived/dialogue_2_10.tsv",
    "m109_pack": "$MANGA109S/derived/dialogue_pack.tsv",
    "m109_b5": "$MANGA109S/derived/dialogue_pack_b5.tsv",
    "ko_chat": "~/ko_corpus/derived/chatbot_lines.tsv",
    # ko_chat's lines on its 192 most frequent syllables (cjk_ko's micro arms)
    "ko_chat_192": "~/ko_corpus/derived/chatbot_lines_top192.tsv",
    # … on its 960 (cjk_ko 10-14: × 5 rows)
    "ko_chat_960": "~/ko_corpus/derived/chatbot_lines_top960.tsv",
}
# the line files not in Japanese: their language (``LANGS``)
LINE_LANG = {"ko_chat": "korean", "ko_chat_192": "korean", "ko_chat_960": "korean"}
UPPER_MAX = 0.9  # tests/test_boundary.py: no band past it


@dataclass(frozen=True)
class Run:
    name: str
    path: Path
    rows: tuple
    read: tuple
    steps_per_row: int
    seed: str = ""  # the seed rows (``SEEDS``); "" with ``rows_from``
    shares: dict | None = None  # tier → % of the items
    rows_from: str = ""  # warm from this scale-line run, every row free
    data_from: str = ""  # a ``rows_from`` run: this reseed run's data
    pack: str = ""  # the base pack (``PACKS``); "" = the raw pack
    lr: float = 0.0  # the rows' peak lr; 0 = trainer.LR
    lines: str = "dialogue_2_10"  # the dialogue line file (``LINES``)
    row_lr: tuple = ()  # one step factor per ``rows`` spec; () = all 1
    free_residual: float | None = None  # the norm pull; None = the trainer's
    pres: tuple | None = None  # (λ, σ_lo, σ_hi, every): L_pres on; None = off
    focus: tuple = ()  # every item holds one of these glyphs; () = off
    focus_kanji: tuple = ()  # (line, window): the most kanji a focus item may hold
    held: str = ""  # strings held out beside ``read`` (a path, $MANGA109S expanded)
    lang: dict | None = None  # glyph → language for the rows not lettered as Japanese
    factor: str = ""  # the rows composed from factors (``FACTORS``); "" = free rows
    factor_init: tuple = ()  # (reseed run, scale) or ("glyph", name, radius): the
    # factors' start; () = Δ 0
    factor_from: str = ""  # a factor run: the factors' start; "" = Δ 0
    factor_residual: float | None = None  # the pull on r; None = no residual
    faces: dict | None = (
        None  # {out, px_min: {face: px}, dirs, legible}; None = every face
    )
    pose: bool = False  # text centred / tilted with its bubble
    # counterfactual (cjk_ko, 10-12): {host: form} (``counterfactual.plan``) —
    # the hosts and the CV syllables a final's removal gives are rows too
    cf: dict | None = None
    total_steps: int = 0  # ``steps``: the whole step count; 0 = steps_per_row's
    upper_shift: float = 0.0  # added to every item's upper σ edge at train
    init: str = (
        ""  # "random" / "stick_ball": the trained rows' start; "" = the pack rows
    )

    def phrase_file(self) -> str:
        """The dialogue line file, ``LINES[lines]``."""
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
        ``steps_per_row``), or a ``counterfactual`` run (its rows less the
        hosts); ``steps`` when the config names it; ``None`` otherwise (every
        trained row's)."""
        if self.total_steps:
            return self.total_steps
        if self.cf:
            n = len({g for r in self.rows for g in r.split(":", 1)[1]}) - len(self.cf)
            return (steps_per_row or self.steps_per_row) * n
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
        the table's, so the items per row stay)."""
        from .table import TABLE

        if not self.shares:
            return TABLE
        total = sum(t.share for t in TABLE)
        return tuple(
            replace(t, share=total * self.shares.get(t.name, 0) / 100) for t in TABLE
        )

    def line_lang(self) -> str:
        """The language of the line file (``LINE_LANG``); "" = Japanese."""
        return LINE_LANG.get(self.lines, "")

    @property
    def out_root(self) -> Path:
        """``OUT``, or ``output/<line>`` for another line's config
        (``project/<line>/configs/<run>.toml``: ``cjk_ko``)."""
        p = self.path.resolve()
        return OUT if HOME in p.parents else OUT.parent / p.parents[1].name

    @property
    def dir(self) -> Path:
        return self.out_root / self.name

    @property
    def data(self) -> Path:
        root = self.out_root
        return (root / self.data_from if self.data_from else self.dir) / "data"

    def seed_rows(self) -> Path:
        """The rows the run sits on: ``rows_from``'s (merged), else the
        seed's."""
        if "/" in self.rows_from:
            return OUT.parent / self.rows_from / "trained.pt"
        if self.rows_from:
            return SCALE_OUT / self.rows_from / "trained.pt"
        return {"0930": SEED_ROWS, "1008": SEED_1008}[self.seed]

    def factor_from_rows(self) -> Path | None:
        """``factor_from``'s ``trained.pt``; ``None`` without it."""
        if not self.factor_from:
            return None
        name = self.factor_from
        return (OUT.parent / name if "/" in name else OUT / name) / "trained.pt"

    def factor_init_rows(self) -> tuple:
        """``(trained.pt, scale)`` of ``factor_init`` (a glyph prediction:
        ``(its .pt, radius, "glyph")``); () without it."""
        if not self.factor_init:
            return ()
        if self.factor_init[0] == "glyph":
            _, name, radius = self.factor_init
            return OUT.parent / "cjk_ko" / "hangul_init" / f"{name}.pt", radius, "glyph"
        name, scale = self.factor_init
        return OUT / name / "trained.pt", scale


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
    rows_from, seed = raw.get("rows_from", ""), raw.get("seed", "")
    assert bool(rows_from) != bool(seed), f"{path}: seed or rows_from, one of them"
    assert rows_from or seed in SEEDS, f"{path}: seed is one of {SEEDS}"
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
    data_from = raw.get("data_from", "")
    if data_from:
        assert "/" not in data_from, f"{path}: data_from names a reseed run"
    lr = float(raw.get("lr", 0.0))
    assert 0 <= lr < 1e-2, f"{path}: lr {lr}"
    pack = raw.get("pack", "")
    assert not pack or pack in PACKS, f"{path}: pack is one of {sorted(PACKS)}"
    lines = raw.get("lines", "dialogue_2_10")
    assert lines in LINES, f"{path}: lines is one of {sorted(LINES)}"
    if lines in LINE_LANG:
        assert LINE_LANG[lines] in (raw.get("lang") or {}), (
            f"{path}: lines {lines} letters {LINE_LANG[lines]} rows — a lang run"
        )
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
    factor = raw.get("factor", "")
    if factor:
        from .jamo import COMPAT, is_part, is_syllable

        assert factor in FACTORS, f"{path}: factor is one of {FACTORS}"
        assert (seed or raw.get("factor_from")) and not row_lr, (
            f"{path}: factor rows are cold or start at factor_from, without row_lr"
        )
        bad = [
            g
            for r in raw["rows"]
            for g in r.split(":", 1)[1]
            if not (is_syllable(g) or g in COMPAT or is_part(g))
        ]
        assert not bad and all(r.startswith("chars:") for r in raw["rows"]), (
            f"{path}: factor rows are chars: specs of Hangul syllables / compat jamo "
            f"/ partial forms "
            f"— not {bad[:10]}"
        )
    fi = raw.get("factor_init")
    if fi is not None:
        assert factor, f"{path}: factor_init needs factor"
        if set(fi) == {"glyph", "radius"}:
            assert 0 < float(fi["radius"]) < 1000, f"{path}: factor_init radius"
            fi = ("glyph", str(fi["glyph"]), float(fi["radius"]))
        else:
            assert set(fi) == {"from", "scale"}, (
                f"{path}: factor_init {{from, scale}} or {{glyph, radius}}"
            )
            assert "/" not in fi["from"], f"{path}: factor_init from names a reseed run"
            assert 0 < float(fi["scale"]) <= 2, (
                f"{path}: factor_init scale {fi['scale']}"
            )
            fi = (fi["from"], float(fi["scale"]))
    ff, fres = raw.get("factor_from", ""), raw.get("factor_residual")
    assert not ff or (factor and fi is None), (
        f"{path}: factor_from needs factor, not with factor_init"
    )
    assert fres is None or (factor and 0 <= float(fres) < 1), (
        f"{path}: factor_residual (a pull in [0, 1)) needs factor"
    )
    faces = raw.get("faces")
    if faces is not None:
        fonts = Path(__file__).resolve().parents[1] / "assets" / "fonts"
        assert lang, f"{path}: faces needs lang"
        assert set(faces) <= {"out", "px_min", "dirs", "legible"}, (
            f"{path}: faces {{out, px_min, dirs, legible}}"
        )
        leg = faces.get("legible")
        assert leg is None or set(leg) == {"median", "p10", "from_px"}, (
            f"{path}: faces.legible {{median, p10, from_px}}"
        )
        dirs = tuple(faces.get("dirs", ()))
        bad = [d for d in dirs if "/" in d or not any((fonts / d).glob("*.[ot]tf"))]
        assert not bad, f"{path}: faces dirs with no face under {fonts}: {bad}"
        named = set(faces.get("out", ())) | set(faces.get("px_min", {}))
        have = {f.name for f in fonts.rglob("*.[ot]tf")}
        have |= {f.name for f in Path("/usr/share/fonts/opentype/noto").glob("*.ttc")}
        assert named <= have, f"{path}: faces not in {fonts}: {sorted(named - have)}"
        faces = {
            "out": tuple(faces.get("out", ())),
            "px_min": {k: float(v) for k, v in faces.get("px_min", {}).items()},
            "dirs": dirs,
            "legible": {k: float(v) for k, v in leg.items()} if leg else None,
        }
    upper_shift = float(raw.get("upper_shift", 0.0))
    assert 0 <= upper_shift <= 0.3, f"{path}: upper_shift {upper_shift}"
    init = raw.get("init", "")
    assert (
        init in ("", "random", "stick_ball")
        or init.startswith("stick_glyph:")
        or (init.startswith("stick_lean:") and 0 < float(init[11:]) <= 1.5)
    ), f"{path}: init {init!r}"
    assert not init or not (rows_from or factor), (
        f"{path}: init random is cold free rows"
    )
    rows, cf = tuple(raw["rows"]), None
    if raw.get("counterfactual"):
        import json

        from .counterfactual import plan
        from .jamo import is_syllable

        assert factor == "jamo", f"{path}: counterfactual needs factor = jamo"
        syls = [g for r in rows for g in r.split(":", 1)[1] if is_syllable(g)]
        held_h = json.loads((HOME / "assets" / "jamo_sets.json").read_text())["H"]
        extra, cf = plan(syls, held_h)
        rows += ("chars:" + "".join(extra) + "".join(cf),)
        if lang:
            lang = {**lang, **dict.fromkeys(extra + list(cf), "korean")}
    return Run(
        name=path.stem,
        path=path,
        rows=rows,
        read=tuple(raw.get("read", ())),
        steps_per_row=int(raw["steps_per_row"]),
        seed=seed,
        shares=dict(shares) if shares is not None else None,
        rows_from=rows_from,
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
        factor=factor,
        factor_init=tuple(fi or ()),
        factor_from=ff,
        factor_residual=None if fres is None else float(fres),
        faces=faces,
        pose=bool(raw.get("pose", False)),
        cf=cf,
        total_steps=int(raw.get("steps", 0)),
        upper_shift=upper_shift,
        init=init,
    )
