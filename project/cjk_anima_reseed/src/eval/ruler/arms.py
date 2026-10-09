"""The arms the ruler renders and the ext-id tables they render through."""

from __future__ import annotations

from pathlib import Path

from reseed import OUT

from eval.ruler import VIEW


def arm_dirs() -> dict:
    """The arms the ruler renders: name → run dir with a finished ``trained.pt``."""
    from reseed import SCALE_OUT

    dirs = {
        "retrain_kana": SCALE_OUT / "retrain_kana",
        "seed_retrain_0930": SCALE_OUT / "seed_retrain_0930",
        "kana_up": OUT / "kana_up",
        "ball_rk_bubble": OUT / "ball_rk_bubble",
        "stick_nlg_high": OUT / "stick_nlg_high",
        # preview51's rows (_archive/sent_plan.md's floor), read on the punct pack
        "seed_fixed_1005_stick080": SCALE_OUT / "seed_fixed_1005_stick080",
        **PACK_ARMS.get(VIEW.pack, {}),
    }
    return {a: d for a, d in dirs.items() if (d / "trained.pt").exists()}


PACK_ARMS = {
    "punct": {
        "punct": OUT / "punct",
        "sent_ball": OUT / "sent_ball",
        "sent_ball_lr2": OUT / "sent_ball_lr2",
        "sent_whole": OUT / "sent_whole",
        "sent_stick": OUT / "sent_stick",
        "sent_kanji": OUT / "sent_kanji",
        "sent_kanji_f0": OUT / "sent_kanji_f0",
        "sent_kanji_pres": OUT / "sent_kanji_pres",
        "seed_1008": OUT / "seed_1008",  # pres + sent_kanji_225's 225 (transplant.py)
        "kozh16": OUT
        / "kozh16",  # 8 Hangul + 8 hanzi cold on seed_1008 (probes/kozh_render.py)
    }
}


def base_arm(a: str) -> str:
    return a.split("@")[0]


def rows_pt(path: Path) -> dict:
    """A probe's ``rows.pt`` (``_archive/probes/probe_pres_train.py``: the live rows'
    ``start`` / ``raw`` at its ``row_scale``) on preview51's rows
    (``seed_fixed_1005_stick080``, the probe's start): those rows replaced, the
    rest as stick080 has them."""
    import torch
    from reseed import SCALE_OUT
    from common.models import load_trained

    base = load_trained(SCALE_OUT / "seed_fixed_1005_stick080")["delta"]
    R = torch.load(path, map_location="cpu", weights_only=False)
    scale, rs = float(base["row_scale"]), float(R["row_scale"])
    at = {int(e): i for i, e in enumerate(base["ext_ids"])}
    raw = base["raw"].float().clone()
    for j, e in enumerate(R["ext_ids"]):
        s = R["start"][j].float() * rs
        assert torch.allclose(raw[at[e]] * scale, s, atol=1e-3 * float(s.norm())), (
            f"{path}: row {e}'s start is not stick080's"
        )
        raw[at[e]] = R["raw"][j].float() * rs / scale
    return {"ext_ids": base["ext_ids"], "raw": raw, "row_scale": scale}


# arms built from other runs' rows, not trained: name → its delta state
# (``--rows_pt name=path`` adds a probe's rows.pt)
DERIVED: dict = {}


def tables(names) -> tuple[list, dict]:
    """The union of the arms' ext ids, and each arm's table over it in
    effective units (``raw × row_scale``; a row an arm lacks is 0 = the pack
    row, what that arm renders it as)."""
    import torch

    from common.models import load_trained

    deltas = {a: load_trained(d)["delta"] for a, d in arm_dirs().items()}
    for a in map(base_arm, names):
        if a in DERIVED:
            deltas[a] = DERIVED[a]()
    ids = sorted({int(e) for d in deltas.values() for e in d["ext_ids"]})
    pos = {e: i for i, e in enumerate(ids)}
    out = {}
    for a in names:
        d = deltas[base_arm(a)]
        t = torch.zeros(len(ids), d["raw"].shape[1])
        t[[pos[int(e)] for e in d["ext_ids"]]] = d["raw"].float() * float(
            d["row_scale"]
        )
        out[a] = t
    return ids, out
