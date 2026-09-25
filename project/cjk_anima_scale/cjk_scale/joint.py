"""joint — one data dir out of several stages' built dirs, band per item.

The conflict read (``conflict_result.md``) says the band stages pull a row
the same way, so the chain is one run with per-item bands. This is that
run's data step: for every stage in the joint stage's ``joint_from``, the
``train.jsonl`` built under the same tag is taken as is (renders and
latents stay where they are — ``file`` is absolute; latents re-encode into
the joint dir once), each record gaining ``band`` (its stage's) and
``stage``. ``eval.json`` is the stages' (asserted identical — same run,
same rows), ``words.json`` the union. ``train.py`` draws σ per item from
``band`` (``noisy_by_band``).
"""

from __future__ import annotations

import json
import time

from .config import StageConfig, load
from .paths import data_dir


def merge(cfg: StageConfig, tag: str):
    assert cfg.joint_from, f"{cfg.stage}: not a joint stage (no joint_from)"
    out = data_dir(cfg.stage, tag)
    out.mkdir(parents=True, exist_ok=True)
    (out / "te_cache").mkdir(exist_ok=True)
    t0 = time.time()
    recs: list = []
    words: dict = {}
    ev_ref = None
    counts: dict = {}
    for s in cfg.joint_from:
        sc = load(s, cfg.run)
        d = data_dir(s, tag)
        assert (d / "train.jsonl").exists(), f"no data dir {d} — build {s} first"
        ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
        if ev_ref is None:
            ev_ref = ev
        else:
            assert json.dumps(ev, sort_keys=True) == json.dumps(ev_ref, sort_keys=True), (
                f"{s}: eval.json differs from {cfg.joint_from[0]}'s — not the same run"
            )
        for k, v in json.loads((d / "words.json").read_text(encoding="utf-8")).items():
            seen = words.setdefault(k, [])
            seen.extend(x for x in v if x not in seen)
        n = 0
        for ln in (d / "train.jsonl").read_text(encoding="utf-8").splitlines():
            if not ln:
                continue
            r = json.loads(ln)
            r["band"] = [float(sc.band[0]), float(sc.band[1])]
            r["stage"] = s
            recs.append(r)
            n += 1
        counts[s] = {"items": n, "band": list(sc.band), "dir": str(d)}
        print(f"joint: {s} {n} items at σ {list(sc.band)} from {d}", flush=True)
    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "eval.json").write_text(json.dumps(ev_ref, ensure_ascii=False, indent=1))
    (out / "words.json").write_text(json.dumps(words, ensure_ascii=False, indent=1))
    (out / "build.json").write_text(
        json.dumps(
            {
                "stage": cfg.stage,
                "tag": tag,
                "joint_from": list(cfg.joint_from),
                "sources": counts,
                "n_items": len(recs),
                "run": cfg.run.name if cfg.run else None,
                "config": str(cfg.path),
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(
        f"joint: {len(recs)} items from {len(counts)} stages → {out} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    return out
