"""A run is one file: ``configs/runs/<run>.toml`` = ``{vocabs, read}`` (plan.md § 1).

    vocabs = "ja_pieces_0925_300.txt"     # what trains: a units file, one vocab per line
                                          # (assets/units/, or a path) — or a list of unit
                                          # specs (["kana", "kanji:200"], the data.units grammar)
    read = ["はい", "おしい"]              # the strings native_sent reads (en clause)

Everything else is a rule in code (plan.md § 2): σ per item from the band law
(``windows.py``), the recipe table by kind (``builder.TABLE``), the volume
(``builder.ITEMS_PER_VOCAB``), the trainer (``train.py``), the seed table
(``paths.SEED_TABLE``), the automatic rulers (``eval.py``). A vocab outside
the file rides frozen at the seed; a vocab the seed lacks starts cold.

The constants below are the data pools every recipe shares — the old stage
files' ``[data]`` blocks, which were identical across the four stages.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .paths import RUN_CONFIGS, UNITS_DIR

RUN_KEYS = ("vocabs", "read")

SEED = 0  # the data draw and the trainer seed (every run of record used 0)

# corpus lines for scene_short / scene_sentence / grid_string "both" (piece
# vocabs only); MANGA109S comes from the repo's .env, the path never enters the repo
PHRASE_FILE = "$MANGA109S/derived/dialogue_2_10.tsv"

DATA = {
    "scenes": "s1,s1w,sl1w,ja_comic",
    "scene_one_bubble": "ja_comic",
    "single_scenes": "s1,s1w",
    "single_max_ar": 2.0,
    "horizontal_scenes": "sl1w",  # left-to-right scene items go to the EN-sentence pool only (user, 2026-09-24)
    "shapes": "448,512:2,448x512,512x448",
    "phrase_min_pieces": 2,
    "phrase_max_pieces": 10,
    "phrase_norm": True,
    "phrase_held_books": 2,
    "n_phrase_eval": 8,
    "n_piece_eval": 18,  # the `word` ruler: 18 of the piece vocabs
    "n_single_eval": 18,  # the `single` ruler of a units-file run: 18 of the single vocabs
    "vertical": True,  # no horizontal fit fallback: orientation is a draw (below)
    "stroke": 0.0,
    "horizontal_frac": 0.3,  # multi-glyph items (scene) / cells (grid) drawn as lines, marked in the caption
    "short_pieces": "2-5",
    "sentence_min_letters": 6,
}


@dataclass(frozen=True)
class RunConfig:
    name: str
    path: Path
    vocabs: str | tuple  # a units file, or unit specs
    read: tuple

    def units(self) -> list[str]:
        """The ``data.units`` specs the vocabs stand for: a file is one
        ``list:@<file>`` source (every line one Qwen piece with an ext row)."""
        if isinstance(self.vocabs, str):
            # a bare name resolves under data.units.UNITS_DIR (= UNITS_DIR), as
            # the stage builds of record spelled it
            f = self.vocabs if "/" not in self.vocabs else self.vocabs_file()
            return [f"list:@{f}"]
        return list(self.vocabs)

    def vocabs_file(self) -> Path | None:
        if not isinstance(self.vocabs, str):
            return None
        v = Path(os.path.expanduser(self.vocabs))
        return v if "/" in self.vocabs else UNITS_DIR / self.vocabs


def run_names() -> list[str]:
    return sorted(p.stem for p in RUN_CONFIGS.glob("*.toml"))


def phrase_file() -> str:
    """``PHRASE_FILE`` with ``$MANGA109S`` expanded (``load_dotenv`` first, so a
    daemon child finds it too)."""
    from library.env import load_dotenv

    load_dotenv()  # never overrides a real var
    p = os.path.expandvars(PHRASE_FILE)
    if "$" in p:
        raise SystemExit(f"{PHRASE_FILE}: env var unset — put MANGA109S=<root> in .env")
    return os.path.expanduser(p)


def load_run(run: str) -> RunConfig:
    path = RUN_CONFIGS / f"{run}.toml" if "/" not in run else Path(run)
    assert path.is_file(), f"no run config {path}"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    extra = sorted(set(raw) - set(RUN_KEYS))
    assert not extra, (
        f"{path}: a run is {{vocabs, read}} — {extra} are rules in code (plan.md § 2)"
    )
    assert "vocabs" in raw, f"{path}: no vocabs"
    v = raw["vocabs"]
    assert isinstance(v, str) or (
        isinstance(v, list) and v and all(isinstance(x, str) for x in v)
    ), f"{path}: vocabs is a units file or a list of unit specs"
    read = raw.get("read", [])
    assert isinstance(read, list) and all(isinstance(x, str) and x for x in read), (
        f"{path}: read is a list of strings"
    )
    rc = RunConfig(
        name=path.stem,
        path=path,
        vocabs=v if isinstance(v, str) else tuple(v),
        read=tuple(read),
    )
    f = rc.vocabs_file()
    assert f is None or f.is_file(), f"{path}: vocabs file {f} does not exist"
    return rc
