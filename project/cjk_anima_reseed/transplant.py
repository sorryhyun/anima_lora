#!/usr/bin/env python
"""transplant — a focus run's rows onto another run's (plan.md § 3, 10-08).

    .venv/bin/python project/cjk_anima_reseed/transplant.py stick   # CPU, the read
    .venv/bin/python project/cjk_anima_reseed/transplant.py write   # trained.pt + the pack

``FROM`` (sent_kanji_225, stage B) trained every row and keeps only its
``focus`` (the 225); they go onto ``ONTO``'s rows (sent_kanji_pres) with
nothing else changed → ``output/cjk_anima_reseed/<NAME>/trained.pt``, baked as
pres was (punct base, per-glyph routing) into
``models/vocab_packs/anima_cjk_vocab_pack_<NAME>/``.

``stick`` (effective units, raw × row_scale): the 225's mean in FROM against
ONTO's kanji stick (angle, length), beside their mean at FROM's start; FROM's
own kanji / kana Δstick against ONTO's (how close the context the 225 trained
in is to the one they land in).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reseed import bootstrap  # noqa: E402

bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from reseed import OUT, REPO  # noqa: E402

FROM, ONTO, NAME = "sent_kanji_225", "sent_kanji_pres", "seed_1008"
PACK_DIR = REPO / "models" / "vocab_packs"
COMFY_PACKS = REPO.parent / "comfy" / "models" / "vocab_packs"


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _eff(d: dict) -> dict:
    """ext id → effective row offset of a trained file."""
    rs = float(d["delta"]["row_scale"])
    return {
        int(e): (r.float() * rs).numpy()
        for e, r in zip(d["delta"]["ext_ids"], d["delta"]["raw"])
    }


def _cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _deg(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(_cos(a, b), -1, 1))))


def _setup():
    from cjk_scale.train import vocab_idx
    from data.inventory import qwen_pieces
    from reseed.config import load

    run = load(FROM)
    run.use_pack()
    tokq = qwen_pieces(char_rows=True)
    src, dst = _load(OUT / FROM / "trained.pt"), _load(OUT / ONTO / "trained.pt")
    focus = vocab_idx(["chars:" + "".join(run.focus)], tokq)
    onto_v = dst["args"]["vocabs"]
    # the family ids every file here carries (today's tokenizer yields a few
    # ids ONTO's file never held — 6 on sent_kanji_pres)
    held = {int(e) for e in dst["delta"]["ext_ids"]} & {
        int(e) for e in src["delta"]["ext_ids"]
    }
    fam = {
        "kana": vocab_idx(onto_v[:2], tokq) & held,
        "kanji": vocab_idx(onto_v[2:], tokq) & held,
    }
    assert focus <= {int(e) for e in src["delta"]["ext_ids"]}, "a focus id FROM lacks"
    assert not focus & fam["kanji"], "a focus row is one of ONTO's trained kanji"
    return run, src, dst, focus, fam


def stick() -> None:
    run, src, dst, focus, fam = _setup()
    S, D = _eff(src), _eff(dst)
    start = _eff(_load(Path(src["seed_merged"])))
    onto_start = _eff(_load(Path(dst["seed_merged"])))
    f = sorted(focus)
    print(
        f"{len(run.focus)} glyphs → {len(f)} ext ids; in {FROM}: "
        f"{sum(e in S for e in f)}, in {ONTO}: {sum(e in D for e in f)} (rows it "
        "carries untrained from its seed)"
    )
    m225 = np.stack([S[e] for e in f]).mean(0)
    m225_0 = np.stack([start[e] for e in f]).mean(0)
    k = sorted(fam["kanji"])
    pres = np.stack([D[e] for e in k]).mean(0)
    print(f"|{ONTO} kanji stick| {np.linalg.norm(pres):.1f} ({len(k)} ids)")
    for label, m in (("the 225 in " + FROM, m225), ("the 225 at its start", m225_0)):
        print(
            f"{label:28s} |m| {np.linalg.norm(m):6.1f}  "
            f"× {np.linalg.norm(m) / np.linalg.norm(pres):.3f} of pres's  "
            f"{_deg(m, pres):5.1f}° off it"
        )
    print("Δstick (the family's mean move), FROM vs ONTO from their own starts:")
    for name, ids in fam.items():
        ids = sorted(ids)
        dv_s = np.stack([S[e] - start[e] for e in ids]).mean(0)
        dv_d = np.stack([D[e] - onto_start[e] for e in ids]).mean(0)
        s0 = np.stack([onto_start[e] for e in ids]).mean(0)
        print(
            f"  {name:5s} |Δ| {FROM} {np.linalg.norm(dv_s):5.1f}  {ONTO} "
            f"{np.linalg.norm(dv_d):5.1f}  cos {_cos(dv_s, dv_d):+.3f}  "
            f"(|stick| at ONTO's start {np.linalg.norm(s0):.1f})"
        )
    mv = np.stack([S[e] - start[e] for e in f])
    print(
        f"the 225's own move in {FROM}: mean |Δrow| {np.linalg.norm(mv, axis=1).mean():.1f}, "
        f"Δstick {np.linalg.norm(mv.mean(0)):.1f}, cos(Δstick, ONTO kanji Δstick) "
        f"{_cos(mv.mean(0), np.stack([D[e] - onto_start[e] for e in k]).mean(0)):+.3f}"
    )


def write() -> None:
    import subprocess

    run, src, dst, focus, fam = _setup()
    k = float(src["delta"]["row_scale"]) / float(dst["delta"]["row_scale"])
    src_raw = {int(e): r for e, r in zip(src["delta"]["ext_ids"], src["delta"]["raw"])}
    rows = {int(e): r for e, r in zip(dst["delta"]["ext_ids"], dst["delta"]["raw"])}
    replaced = sum(e in rows for e in focus)
    for e in focus:
        rows[e] = src_raw[e].float() * k
    ids = sorted(rows)
    out = {
        **dst,
        "delta": {
            **dst["delta"],
            "ext_ids": ids,
            "raw": torch.stack([rows[e].float() for e in ids]).to(
                dst["delta"]["raw"].dtype
            ),
        },
        "transplant": {
            "onto": str(OUT / ONTO / "trained.pt"),
            "from": str(OUT / FROM / "trained.pt"),
            "focus": "".join(run.focus),
            "ext_ids": sorted(focus),
            "row_scale_ratio": k,
        },
    }
    d = OUT / NAME
    d.mkdir(parents=True, exist_ok=True)
    torch.save(out, d / "trained.pt")
    print(
        f"{NAME}: {ONTO}'s {len(dst['delta']['ext_ids'])} ids + {len(focus)} from "
        f"{FROM} ({replaced} replaced, {len(focus) - replaced} added) → "
        f"{d / 'trained.pt'}",
        flush=True,
    )
    stem = f"anima_cjk_vocab_pack_{NAME}"
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "toolkits" / "bake_vocab_pack.py"),
            str(d / "trained.pt"),
            "--base",
            str(PACK_DIR / "anima_cjk_vocab_pack_punct"),
            "--out",
            str(PACK_DIR / stem / stem),
            "--glyph_route",
            "--comfy_dir",
            str(COMFY_PACKS),
        ],
        check=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("verb", choices=["stick", "write"])
    {"stick": stick, "write": write}[p.parse_args().verb]()


if __name__ == "__main__":
    main()
