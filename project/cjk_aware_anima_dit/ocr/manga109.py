"""Manga109-s + COO access for the SFX-reader line (``plan_ocr.md``).

Roots come from the environment, never from a literal path in a tracked file
(``plan.md`` principle 9): ``ANIMA_MANGA109S_ROOT`` is the unzipped
``Manga109s_released_*`` directory (``images/``, ``annotations/``,
``annotations_COO/``, ``books.txt``). Derived data (crops, manifests,
colorized pages) goes under ``$MANGA109S_ROOT/../derived/`` and never in-tree.

The book split is the **official COO split** (``ku21fan/COO-Comic-Onomatopoeia``,
``COO-data/books_{train,val,test}.txt``, 109 books) intersected with the 87
Manga109-s books — 74 / 7 / 6. ``assets/coo_split_manga109s.json`` is the
frozen artifact; :func:`write_split` regenerates it and :func:`load_split`
asserts it against ``books.txt``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

HERE = Path(__file__).resolve().parent
LINE = HERE.parent  # project/cjk_aware_anima_dit
REPO = LINE.parents[1]
ASSETS = LINE / "assets"
SPLIT_PATH = ASSETS / "coo_split_manga109s.json"
COO_SPLIT_URL = (
    "https://raw.githubusercontent.com/ku21fan/COO-Comic-Onomatopoeia/main/COO-data/"
)
SPLITS = ("train", "val", "test")


def manga109s_root() -> Path:
    """``ANIMA_MANGA109S_ROOT`` (the daemon forwards only ``ANIMA_``-prefixed env
    to its jobs — ``anima_daemon.config.CAPTURED_ENV_PREFIXES``); bare
    ``MANGA109S_ROOT`` accepted for inline runs."""
    env = os.environ.get("ANIMA_MANGA109S_ROOT") or os.environ.get("MANGA109S_ROOT")
    if not env:
        raise SystemExit(
            "ANIMA_MANGA109S_ROOT is not set — point it at the unzipped "
            "Manga109s_released_* directory (images/, annotations/, annotations_COO/)."
        )
    root = Path(env).expanduser()
    if not (root / "books.txt").is_file():
        raise SystemExit(f"MANGA109S_ROOT={root} has no books.txt")
    return root


def derived_root() -> Path:
    """``<root>/../derived`` — sibling of the release dir, outside the repo."""
    d = manga109s_root().parent / "derived"
    d.mkdir(parents=True, exist_ok=True)
    return d


def books() -> list[str]:
    return [
        ln.strip()
        for ln in (manga109s_root() / "books.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if ln.strip()
    ]


# --------------------------------------------------------------------------- split


def write_split(path: Path = SPLIT_PATH) -> dict[str, list[str]]:
    """Official COO lists ∩ ``books.txt`` → JSON. Network: three small text files."""
    import urllib.request

    ours = set(books())
    out: dict[str, object] = {
        "source": "ku21fan/COO-Comic-Onomatopoeia COO-data/books_{train,val,test}.txt "
        "∩ Manga109-s books.txt (book names only)",
    }
    seen: set[str] = set()
    for split in SPLITS:
        raw = urllib.request.urlopen(COO_SPLIT_URL + f"books_{split}.txt").read()
        names = [ln.strip() for ln in raw.decode("utf-8").splitlines()]  # CRLF-safe
        keep = sorted(n for n in names if n and n in ours)
        assert not (set(keep) & seen), f"book in two splits: {set(keep) & seen}"
        seen.update(keep)
        out[split] = keep
    out["unsplit"] = sorted(ours - seen)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return {s: out[s] for s in SPLITS}  # type: ignore[misc]


def load_split(path: Path = SPLIT_PATH) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    split = {s: list(data[s]) for s in SPLITS}
    ours = set(books())
    for s, names in split.items():
        missing = set(names) - ours
        if missing:
            raise SystemExit(
                f"split {s} names books not in books.txt: {sorted(missing)}"
            )
    return split


def split_of(split: dict[str, list[str]]) -> dict[str, str]:
    return {b: s for s, names in split.items() for b in names}


# --------------------------------------------------------------------------- annotations


@dataclass(frozen=True)
class Line:
    book: str
    page: int
    id: str
    kind: str  # "sfx" | "speech"
    poly: tuple[float, ...]  # x0,y0,x1,y1,...
    text: str
    joined: bool = False  # COO truncation link joined into one line


def _poly_of(el: ET.Element) -> tuple[float, ...]:
    pts: list[float] = []
    i = 0
    while el.get(f"x{i}") is not None:
        pts += [float(el.get(f"x{i}")), float(el.get(f"y{i}"))]
        i += 1
    return tuple(pts)


def iter_coo(book: str) -> Iterator[Line]:
    """COO onomatopoeia polygons of one book, truncation links joined.

    ``onomatopoeia_link1`` / ``link2`` pair two ids (``link0`` → ``link1``) on
    the same page (verified: 0 cross-page links); the pair becomes one line —
    text concatenated in link order, polygon = both point sets (the crop
    builder takes the ``minAreaRect`` of the union) — and the parts are dropped.
    """
    root = ET.parse(manga109s_root() / "annotations_COO" / f"{book}.xml").getroot()
    for pg in root.iter("page"):
        page = int(pg.get("index"))
        parts: dict[str, ET.Element] = {
            o.get("id"): o for o in pg.findall("onomatopoeia")
        }
        consumed: set[str] = set()
        for link in list(pg.findall("onomatopoeia_link1")) + list(
            pg.findall("onomatopoeia_link2")
        ):
            ids = [link.get(k) for k in ("link0", "link1", "link2") if link.get(k)]
            if not all(i in parts for i in ids) or any(i in consumed for i in ids):
                continue
            els = [parts[i] for i in ids]
            poly = tuple(v for e in els for v in _poly_of(e))
            text = "".join((e.text or "").strip() for e in els)
            consumed.update(ids)
            yield Line(book, page, "+".join(ids), "sfx", poly, text, joined=True)
        for oid, el in parts.items():
            if oid in consumed:
                continue
            yield Line(book, page, oid, "sfx", _poly_of(el), (el.text or "").strip())


def iter_text(book: str) -> Iterator[Line]:
    """Manga109-s ``<text>`` boxes (speech; axis-aligned) as 4-point polygons."""
    root = ET.parse(manga109s_root() / "annotations" / f"{book}.xml").getroot()
    for pg in root.iter("page"):
        page = int(pg.get("index"))
        for t in pg.findall("text"):
            x0, y0, x1, y1 = (float(t.get(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            yield Line(
                book,
                page,
                t.get("id"),
                "speech",
                (x0, y0, x1, y0, x1, y1, x0, y1),
                (t.text or "").strip(),
            )


def page_path(book: str, page: int) -> Path:
    return manga109s_root() / "images" / book / f"{page:03d}.jpg"


# --------------------------------------------------------------------------- shared code from the pilot


def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pilot_manga_text():
    """``project/cjk_aware_anima/datasets/manga_text.py`` — ``deskew_crop``, ``MangaOCR``."""
    return _load_by_path(
        "_pilot_manga_text", REPO / "project/cjk_aware_anima/datasets/manga_text.py"
    )


def pilot_records():
    """``build_ocr_records.py`` — ``norm`` / ``sim`` / ``is_runaway`` (the A/B's keys)."""
    return _load_by_path(
        "_pilot_records", REPO / "project/cjk_aware_anima/datasets/build_ocr_records.py"
    )
