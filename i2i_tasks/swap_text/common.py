"""What the swap_text stages share: output formats, atomic writers, the worker
setup and the index of built pages."""

from __future__ import annotations

import json
import os
import re
import tempfile
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

# kana / CJK ideographs: in a translated edition's region these are the reader
# taking hand-lettered Hangul for katakana, or JA lettering left in place
JA_SCRIPT = re.compile(
    r"[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]"
)
WEBP = [cv2.IMWRITE_WEBP_QUALITY, 95]
# The staged target is lossless: outside its paste mask it is the decoded source
# exactly, so target − reference is zero wherever the edit is not.
WEBP_LOSSLESS = [cv2.IMWRITE_WEBP_QUALITY, 101]


def worker_pool(n: int):
    """A pool of ``n`` forked workers on one OpenCV thread each. cv2 otherwise
    spawns a thread per core in every worker, and 12 workers ran at load ~100 on
    12 cores. The count drops in the parent around the fork: a parent that has
    already run cv2 (``det`` before ``pool``) forks its thread pool's state
    without the threads, and a worker's own ``cv2.setNumThreads`` or parallel
    call then waits on them forever."""
    threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        return Pool(n)
    finally:
        cv2.setNumThreads(threads)


def stable_seed(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def write_atomic(out: Path, write) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp" + out.suffix)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, out)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def imwrite(out: Path, img: np.ndarray, params=()) -> None:
    def write(tmp):
        if not cv2.imwrite(tmp, img, list(params)):
            raise OSError(f"cv2.imwrite failed: {out}")

    write_atomic(out, write)


def write_text(out: Path, text: str) -> None:
    write_atomic(out, lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))


def write_jsonl(out: Path, rows: list[dict]) -> None:
    write_text(out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def write_stats(out: Path, stats: dict) -> None:
    text = json.dumps(stats, indent=1)
    write_text(out, text)
    print(text, flush=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_cache(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(ln) for ln in f]


def ok_pages(a) -> list[dict]:
    """Every built page with regions: ``{key, lang, anchor, page, dir, size,
    regions}``, ``key = <lang>/<anchor>/<page>``."""
    out = []
    for lang in a.langs:
        for pj in sorted((a.root / "work" / lang).glob("*/pages.json")):
            anchor = pj.parent.name
            if a.works and anchor not in a.works:
                continue
            for p in read_json(pj):
                if p["status"] == "ok":
                    out.append(
                        {
                            "key": f"{lang}/{anchor}/{p['page']}",
                            "lang": lang,
                            "anchor": anchor,
                            "page": p["page"],
                            "dir": pj.parent,
                            "size": p["size"],
                            "resid": p["resid"],
                            "uncovered": p["uncovered"],
                            "regions": p["regions"],
                            "groups": p["groups"],
                        }
                    )
    return out
