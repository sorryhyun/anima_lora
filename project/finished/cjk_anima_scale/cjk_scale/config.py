"""A run is one file: ``configs/runs/<run>.toml`` = ``{vocabs, read[, context]}`` (plan.md § 1).

    vocabs = "ja_pieces_0925_300.txt"     # what trains: a vocabs file, one vocab per line
                                          # (assets/vocabs/, or a path) — or a list of vocab
                                          # specs (["kana", "kanji:200"], the data.vocabs grammar)
    read = ["はい", "おしい"]              # the strings native_sent reads (en clause)
    context = "retrain_kana"              # optional: a run whose merged rows replace the
                                          # seed rows (plan_retrain § 2) — warm-from, frozen
                                          # context and merge base; its trained singles (and
                                          # its own context's, down the chain) may sit in
                                          # this run's windows. A rows dir under OUT with no
                                          # run config (seed_fixed_1005_stick080 = preview51's
                                          # rows) ends the chain; its singles are its
                                          # data/vocabs.json (merge.idx_source)
    phrases = "$MANGA109S/derived/dialogue_pack_b5.tsv"  # optional: the dialogue lines the
                                          # windows cut (default PHRASE_FILE, every run of
                                          # record; retrain_kanji_b5)
    held = "$MANGA109S/derived/b5_held.tsv"  # optional: strings held out of the windows
                                          # as ``read`` is (trigram), not read by eval — a
                                          # corpus file outside the repo (first tsv column)

Everything else is a rule in code (plan.md § 2): σ per item from the band law
(``windows.py``), the recipe table by kind (``builder.TABLE``), the volume
(``builder.ITEMS_PER_VOCAB``), the trainer (``train.py``), the seed table
(``paths.SEED_ROWS``, or the ``context`` run's rows), the automatic rulers
(``eval.py``). A vocab outside the file rides frozen at the seed (the
context); a vocab the seed lacks starts cold.

The pre-collapse configs are records: the stage-shaped run files stay in
``configs/runs/`` as they ran, and the stage / joint configs live split under
``configs/data_build/`` (the data recipe: band + pools + mix) and
``configs/train/`` (the trainer values); nothing reads them.

The constants below are the data pools every recipe shares — the old stage
files' ``[data]`` blocks, which were identical across the four stages.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .paths import RUN_CONFIGS, VOCABS_DIR, trained_path

RUN_KEYS = ("vocabs", "read", "context", "phrases", "held")

SEED = 0  # the data draw and the trainer seed (every run of record used 0)

# corpus lines for scene_short / scene_sentence / grid_string "both" (piece
# vocabs only); MANGA109S comes from the repo's .env, the path never enters the repo
PHRASE_FILE = "$MANGA109S/derived/dialogue_2_10.tsv"

# the training set's own text (plan_retrain § 2b, user 2026-09-30): the text
# clauses of the revised captions, JA only — the window pool's second source
# beside the dialogue lines (dialogue_2_10 is charset-filtered: 顔 姉 満 … occur
# 0 times in it). Read live; the dataset never enters the repo.
DATASET_CAPTIONS = "post_image_dataset/resized"
# a kana-less string with one of these is Chinese (你 是 很 …, plus the
# traditional forms the OCR text carries)
ZH_MARKERS = frozenset("你是很呢啊的了吗么这那们她他说吃吞得些讓點")

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
    "n_single_eval": 18,  # the `single` ruler of a vocabs-file run: 18 of the single vocabs
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
    vocabs: str | tuple  # a vocabs file, or vocab specs
    read: tuple
    context: str | None = None  # a run name: its merged rows replace the seed rows
    phrases: str = ""  # the dialogue lines (a path, $MANGA109S expanded); "" = PHRASE_FILE
    held: str = ""  # strings held out of the windows beside ``read`` (a path, as ``phrases``)

    def held_strings(self) -> tuple:
        """``read`` plus the ``held`` file's strings (first tsv column,
        blank / ``#`` lines skipped): what the windows hold out."""
        if not self.held:
            return self.read
        lines = Path(_expand(self.held)).read_text(encoding="utf-8").splitlines()
        extra = [ln.split("\t")[0] for ln in lines if ln.strip() and ln[0] != "#"]
        return tuple(dict.fromkeys((*self.read, *extra)))

    def vocab_specs(self) -> list[str]:
        """The ``data.vocabs`` specs the vocabs stand for: a file is one
        ``list:@<file>`` source (every line one Qwen piece with an ext row)."""
        if isinstance(self.vocabs, str):
            # a bare name resolves under data.vocabs.VOCABS_DIR (= VOCABS_DIR), as
            # the stage builds of record spelled it
            f = self.vocabs if "/" not in self.vocabs else self.vocabs_file()
            return [f"list:@{f}"]
        return list(self.vocabs)

    def vocabs_file(self) -> Path | None:
        if not isinstance(self.vocabs, str):
            return None
        v = Path(os.path.expanduser(self.vocabs))
        return v if "/" in self.vocabs else VOCABS_DIR / self.vocabs

    def context_rows(self) -> Path:
        """The rows this run sits on: the ``context`` run's merged
        ``trained.pt`` (finished, not a partial), else ``paths.SEED_ROWS``."""
        if not self.context:
            return paths.SEED_ROWS
        p = trained_path(self.context)
        assert p.is_file(), (
            f"{self.name}: context {self.context} has no rows at {p} — train it first"
        )
        import torch

        sd = torch.load(p, map_location="cpu", weights_only=False)
        assert sd.get("seed_merged") and sd.get("step") is None, (
            f"{p}: not a finished run's merged rows (step {sd.get('step')})"
        )
        return p

    def context_chain(self) -> list[str]:
        """The context runs, nearest first, down to the one on the seed."""
        out, rc = [], self
        while rc.context:
            assert rc.context not in out and rc.context != self.name, (
                f"{self.name}: context cycle through {rc.context}"
            )
            out.append(rc.context)
            if not (RUN_CONFIGS / f"{rc.context}.toml").is_file():
                break  # a rows-only context (no run config) ends the chain
            rc = load_run(rc.context)
        return out


def run_names() -> list[str]:
    return sorted(p.stem for p in RUN_CONFIGS.glob("*.toml"))


def phrase_file(rc: RunConfig | None = None) -> str:
    """The run's ``phrases`` (else ``PHRASE_FILE``) with ``$MANGA109S``
    expanded (``load_dotenv`` first, so a daemon child finds it too)."""
    return _expand((rc.phrases if rc else "") or PHRASE_FILE)


def _expand(raw: str) -> str:
    from library.env import load_dotenv

    load_dotenv()  # never overrides a real var
    p = os.path.expandvars(raw)
    if "$" in p:
        raise SystemExit(f"{raw}: env var unset — put MANGA109S=<root> in .env")
    return os.path.expanduser(p)


def is_ja_text(s: str) -> bool:
    """plan_retrain § 2b's rule: a string with a kana (ー aside) is Japanese;
    a kana-less one is not if it holds hangul, a glyph outside JIS X 0208,
    or a Chinese function word (``ZH_MARKERS``)."""
    import unicodedata

    def kana(c):
        return c != "ー" and "぀" <= c <= "ヿ"

    def jis(c):
        try:
            c.encode("iso2022_jp")
            return True
        except UnicodeEncodeError:
            return False

    if any(kana(c) for c in s):
        return True
    return not any(
        "가" <= c <= "힯"
        or c in ZH_MARKERS
        or (unicodedata.category(c) == "Lo" and not jis(c))
        for c in s
    )


def dataset_ja_lines() -> list[str]:
    """The JA strings of the training set's text clauses (``Japanese text /
    SFX reads as``), ``.variants.txt`` excluded, sorted and deduplicated."""
    from anime_tools.captions.position_clauses import TEXT_PREFIXES, parse_caption

    from library.env import resolve_under_home

    out: set = set()
    for f in resolve_under_home(DATASET_CAPTIONS).rglob("*.txt"):
        if f.name.endswith(".variants.txt"):
            continue
        for cl in parse_caption(f.read_text(encoding="utf-8")).clauses:
            if cl.prefix not in TEXT_PREFIXES:
                continue
            for t in cl.tags:
                s = t.strip().rstrip(".").strip('"')
                if s and is_ja_text(s):
                    out.add(s)
    return sorted(out)


def load_run(run: str) -> RunConfig:
    path = RUN_CONFIGS / f"{run}.toml" if "/" not in run else Path(run)
    assert path.is_file(), f"no run config {path}"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    extra = sorted(set(raw) - set(RUN_KEYS))
    assert not extra, (
        f"{path}: a run is {{vocabs, read[, context, phrases, held]}} — {extra} are rules in code "
        "(plan.md § 2)"
    )
    assert "vocabs" in raw, f"{path}: no vocabs"
    v = raw["vocabs"]
    assert isinstance(v, str) or (
        isinstance(v, list) and v and all(isinstance(x, str) for x in v)
    ), f"{path}: vocabs is a vocabs file or a list of vocab specs"
    read = raw.get("read", [])
    assert isinstance(read, list) and all(isinstance(x, str) and x for x in read), (
        f"{path}: read is a list of strings"
    )
    ctx = raw.get("context")
    assert ctx is None or (isinstance(ctx, str) and ctx and "/" not in ctx), (
        f"{path}: context is a run name"
    )
    assert ctx != path.stem, f"{path}: a run cannot be its own context"
    ph, held = raw.get("phrases", ""), raw.get("held", "")
    assert isinstance(ph, str) and isinstance(held, str), f"{path}: phrases / held are paths"
    rc = RunConfig(
        name=path.stem,
        path=path,
        vocabs=v if isinstance(v, str) else tuple(v),
        read=tuple(read),
        context=ctx,
        phrases=ph,
        held=held,
    )
    f = rc.vocabs_file()
    assert f is None or f.is_file(), f"{path}: vocabs file {f} does not exist"
    return rc
