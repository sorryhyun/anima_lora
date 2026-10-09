#!/usr/bin/env python
"""stick_fit — the stick runs' rows and renders, and the sticks around them (CPU).

The stick runs (``_archive/configs/stick_*.toml``, ``stick_from = "kana_up"``) train
kana_up's 166 rows' shared mean only: every row takes the sum of the rows'
gradients, so AdamW moves them by one vector and the rows less their mean
(the spikes) stay kana_up's (``cjk_scale.train(stick_only=True)``). Legs:

- ``geo``: the kana sticks (166 rows) — the stick runs against kana_up,
  kana_mix, retrain_kana: length, cos, the move and its direction; the
  hiragana sticks (81 rows) of the cold band arms (``grid_small`` /
  ``grid_lone``, ``_archive/reports/grid_small_lone_2026_10_02.md``).
- ``kanji``: the retrain's kanji family (b1–b4, each cold over the run
  before it) as a burr — stick, spikes, energy, nearest T5 — and the
  batches' sticks against each other, the kana stick and the 0921 seed's.
- ``ball``: the 81 hiragana rows of every cold arm as stick + ball against
  retrain_kana's; the ball / stick swap arms of ``probe_split`` (``gs_*``,
  ``rk_gsstick``, ``h0_*``) read and EN-ref cos'd against ``rk_self`` on the
  hiragana keys (``_archive/reports/ball_2026_10_04.md``).
- ``ball_sheets``: per prompt, the hiragana words / singles as EN ref |
  ``rk_self`` | ``gs_rkstick`` | ``gs_self`` → ``output/cjk_anima_reseed/<label>/sheets/``.
- ``scene``: probe_split's scene numbers (EN-ref cos out, flat white) on
  the cached plain renders at seed 0.
- ``sheets``: probe_split's per-key sheets at seed 0 (kana_mix | kana_up |
  the stick runs | retrain_kana) and overviews (singles p01, words p00)
  → ``output/cjk_anima_reseed/<label>/sheets/``.

Row = the delta ``raw × row_scale``; a run's rows = the ext rows that differ
from its context.

    .venv/bin/python project/cjk_anima_reseed/stick_fit.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "probes"))
import probe_split as PS  # noqa: E402  (bootstraps the scale line)

from reseed import HOME, OUT  # noqa: E402

STICK_RUNS = (
    "stick_full",
    "stick_nolone",
    "stick_nolonegrid",
    "stick_nlg_high",
    "stick_rk_nolonegrid",
    "stick_rk_full_band",
    "stick_rk_fb_jt50",
)
KANA_REF = ("kana_up", "kana_mix", "retrain_kana")
# the cold hiragana band arms (``cjk_anima_scale/experiments/<dir>``)
BAND_ARMS = {
    "grid_small_r0": "grid_small_cold_hira",
    "grid_small_b7593": "grid_small_cold_hira_b7593",
    "grid_lone_r0": "grid_lone_cold_hira",
    "grid_lone_recap": "grid_lone_cold_hira_recap",
    "grid_lone_recap_h0": "grid_lone_cold_hira_recap_h0",
    "grid_lone_recap_hp": "grid_lone_cold_hira_recap_hp",
}
KANJI_BATCHES = (1, 2, 3, 4)
PACK = "models/vocab_packs/anima_cjk_vocab_pack"


def run_dirs() -> dict:
    from cjk_scale.paths import OUT as SCALE_OUT

    return (
        {r: OUT / r for r in STICK_RUNS + ("kana_up", "kana_mix")}
        | {"retrain_kana": SCALE_OUT / "retrain_kana"}
        | {k: SCALE_OUT / "experiments" / v for k, v in BAND_ARMS.items()}
    )


class Tables:
    """The tokenizer, pack, T5 table and row loader the geometry legs share."""

    def __init__(self):
        import torch
        from safetensors import safe_open

        from cjk_scale.paths import load_experiment
        from library.anima.vocab_pack import load_vocab_pack
        from library.env import default_checkpoints
        from library.inference.text import ensure_text_strategies

        self.RG = load_experiment("row_geometry")
        ck = default_checkpoints()
        self.tok, _ = ensure_text_strategies(ck.text_encoder, vocab_pack=PACK)
        self.pack = load_vocab_pack(PACK)
        with safe_open(ck.dit, "pt") as f:
            t5 = f.get_tensor("net.llm_adapter.embed.weight").float()
        self.t5n = torch.nn.functional.normalize(t5, dim=1)

    def rows(self, path: Path) -> dict:
        return self.RG._rows(path)

    def moved(self, path: Path, keep) -> list:
        """The single-glyph ext ids ``path`` trained over its context, ``keep``(glyph)."""
        import torch
        from probe.merge_tables import row_texts

        sd = torch.load(path, map_location="cpu", weights_only=False)
        rows, ctx = self.rows(path), self.rows(sd["args"]["context"])
        moved = [
            e
            for e, v in rows.items()
            if e not in ctx or not torch.allclose(v, ctx[e], atol=1e-2)
        ]
        text = row_texts(self.tok, self.pack, moved)
        return sorted(
            e for e in moved if e in text and len(text[e]) == 1 and keep(text[e])
        )


def _r(x, n=3):
    return round(float(x), n)


def burr(T: Tables, ids: list, X) -> dict:
    import torch

    F = torch.nn.functional
    m = X.mean(0)
    sp = (X - m).norm(dim=1)
    eff = T.pack.table[ids].float() + X
    Xc = F.normalize(X - m, dim=1)
    off = ~torch.eye(len(X), dtype=bool)
    return {
        "n": len(ids),
        "norm": _r(X.norm(dim=1).mean(), 1),
        "stick": _r(m.norm(), 1),
        "spike": _r(sp.mean(), 1),
        "spike_cv": _r(sp.std() / sp.mean()),
        "mean_energy": _r(m.norm() ** 2 / (X.norm(dim=1) ** 2).mean()),
        "pr_centered": _r(T.RG._pr(X), 1),
        "abs_cos_centered_p95": _r((Xc @ Xc.T).abs()[off].quantile(0.95)),
        "nearest_t5": _r((F.normalize(eff, dim=1) @ T.t5n.T).max(1).values.median()),
    }


def geo(T: Tables) -> dict:
    import torch

    F = torch.nn.functional
    dirs = run_dirs()
    rows = {k: T.rows(d / "trained.pt") for k, d in dirs.items()}
    ids = T.moved(dirs["kana_up"] / "trained.pt", lambda c: True)
    from probe.merge_tables import row_texts

    text = row_texts(T.tok, T.pack, ids)
    hira = [e for e in ids if 0x3041 <= ord(text[e]) <= 0x309F]
    out: dict = {"kana_rows": len(ids), "hira_rows": len(hira)}

    def stack(k, sel):
        return torch.stack([rows[k][e] for e in sel])

    # the stick runs: spikes held, the stick moved
    names = STICK_RUNS + KANA_REF
    S = {k: stack(k, ids) for k in names}
    M = {k: x.mean(0) for k, x in S.items()}
    out["kana"] = {
        "stick": {k: _r(M[k].norm(), 1) for k in names},
        "cos": {
            a: {b: _r(F.cosine_similarity(M[a], M[b], dim=0)) for b in names}
            for a in names
        },
        "spikes_held_max": {
            k: float(((S[k] - M[k]) - (S["kana_up"] - M["kana_up"])).norm(dim=1).max())
            for k in STICK_RUNS
        },
    }
    rk_up = M["retrain_kana"] - M["kana_up"]
    moves = {k: M[k] - M["kana_up"] for k in STICK_RUNS}
    out["kana"]["move"] = {
        k: {
            "norm": _r(d.norm(), 1),
            "cos_rk_minus_up": _r(F.cosine_similarity(d, rk_up, dim=0)),
            "cos_mix_minus_up": _r(
                F.cosine_similarity(d, M["kana_mix"] - M["kana_up"], dim=0)
            ),
        }
        for k, d in moves.items()
    }
    ks = list(moves)
    out["kana"]["move"]["cos_between_runs"] = {
        f"{a}·{b}": _r(F.cosine_similarity(moves[a], moves[b], dim=0))
        for i, a in enumerate(ks)
        for b in ks[i + 1 :]
    }
    out["kana"]["rk_minus_up_norm"] = _r(rk_up.norm(), 1)
    # the hiragana sticks of the cold band arms
    H = {k: stack(k, hira) for k in dirs}
    MH = {k: x.mean(0) for k, x in H.items()}
    out["hira"] = {
        k: {
            "norm": _r(x.norm(dim=1).mean(), 1),
            "stick": _r(MH[k].norm(), 1),
            "spike": _r((x - MH[k]).norm(dim=1).mean(), 1),
            "mean_energy": _r(MH[k].norm() ** 2 / (x.norm(dim=1) ** 2).mean()),
            **{
                f"cos_{r}": _r(F.cosine_similarity(MH[k], MH[r], dim=0))
                for r in KANA_REF
            },
            "row_cos_retrain_kana": _r(
                F.cosine_similarity(x, H["retrain_kana"]).mean()
            ),
        }
        for k, x in H.items()
    }
    return out


# the ball / stick swap arms (``probe_split``), paired against ``rk_self``
BALL_ARMS = (
    "gs_self",
    "gs_rkstick",
    "rk_gsstick",
    "h0_self",
    "h0_rkstick",
    "ball_rk",
    "ball_rk_bubble",
    "ball_rkb_long",
)
# the ball arms trained on retrain_kana's stick, paired beyond ``rk_self``
BALL_VS = {
    "ball_rk": ("gs_rkstick",),
    "ball_rk_bubble": ("ball_rk", "gs_rkstick"),
    "ball_rkb_long": ("ball_rk_bubble",),
}
BALL_GEO = {
    "anchor": "experiments/reseed_anchor_cold_kana_anchor",
    "recap_hp_scene": "experiments/reseed_recap_cold_kana_hp",
}


def ball(T: Tables) -> dict:
    """The hiragana rows (81) of every cold arm as stick + ball against
    retrain_kana's, and the swap arms' reads and EN-ref cos against ``rk_self``
    on the hiragana keys (``_archive/reports/ball_2026_10_04.md``)."""
    import torch
    from scipy.stats import wilcoxon

    from cjk_scale import reads as R
    from cjk_scale.paths import OUT as SCALE_OUT
    from probe.merge_tables import row_texts

    F = torch.nn.functional
    dirs = {k: v for k, v in run_dirs().items() if k in KANA_REF or k in BAND_ARMS} | {
        k: SCALE_OUT / v for k, v in BALL_GEO.items()
    }
    for b in ("ball_rk", "ball_rk_bubble"):
        if (OUT / b / "trained.pt").exists():
            dirs[b] = OUT / b
    rows = {k: T.rows(d / "trained.pt") for k, d in dirs.items()}
    ids = T.moved(dirs["kana_up"] / "trained.pt", lambda c: True)
    text = row_texts(T.tok, T.pack, ids)
    hira = [e for e in ids if 0x3041 <= ord(text[e]) <= 0x309F]
    X = {k: torch.stack([rows[k][e] for e in hira]) for k in dirs}
    M = {k: x.mean(0) for k, x in X.items()}
    S = {k: X[k] - M[k] for k in X}
    geo = {
        k: {
            "stick": _r(M[k].norm(), 1),
            "spike": _r(S[k].norm(dim=1).mean(), 1),
            "mean_energy": _r(M[k].norm() ** 2 / (X[k].norm(dim=1) ** 2).mean()),
            "stick_cos_rk": _r(F.cosine_similarity(M[k], M["retrain_kana"], dim=0)),
            "stick_cos_up": _r(F.cosine_similarity(M[k], M["kana_up"], dim=0)),
            "ball_cos_rk": _r(F.cosine_similarity(S[k], S["retrain_kana"]).mean()),
            "ball_cos_up": _r(F.cosine_similarity(S[k], S["kana_up"]).mean()),
        }
        for k in X
    }
    out: dict = {"hira_rows": len(hira), "geo": geo}

    def is_hira(k):
        return all(0x3041 <= ord(c) <= 0x309F for c in k)

    keys = {it["text"] for it in PS.items()}
    arms = ["rk_self"] + [
        a for a in BALL_ARMS if (PS.arm_dir(a) / "native_reads.json").exists()
    ]
    recs, hits = {}, {}
    for a in arms + ["kana_up", "kana_mix"]:
        p = PS.arm_dir(a) / "native_reads.json" if a in arms else PS.reads_of(OUT / a)
        recs[a] = {
            (m["text"], m["pi"]): m
            for m in json.loads(p.read_text("utf-8"))
            if m["clause"] == PS.CLAUSE and m["pi"] < PS.PROMPTS and m["seed"] == 0
        }
        h = R.hits(p, keys, PS.CLAUSE)
        hits[a] = {k: v for k, v in h.items() if k[2] < PS.PROMPTS and k[3] < 1}
    reads, pairs = {}, {}
    for grp, sel in (("hira", is_hira), ("kata", lambda k: not is_hira(k))):
        for a in hits:
            t = R.tally({k: v for k, v in hits[a].items() if sel(k[0])})["total"]
            reads[f"{grp}: {a}"] = t
        for a in arms[1:]:
            for g in ("words", "singles"):
                f = lambda x: {  # noqa: E731
                    k: v
                    for k, v in hits[x].items()
                    if sel(k[0]) and (len(k[0]) > 1) == (g == "words")
                }
                pairs[f"{grp} {g}: {a} vs rk_self"] = R.paired(f(a), f("rk_self"))
                # a ball trained on the stick against the grid ball moved onto
                # it, and the bubble ball against the grid ball trained on it
                for b in BALL_VS.get(a, ()):
                    if b in arms:
                        pairs[f"{grp} {g}: {a} vs {b}"] = R.paired(f(a), f(b))
    en = {}
    for a in arms[1:] + ["kana_up", "kana_mix"]:
        for g in ("words", "singles"):
            ks = [
                k
                for k in recs["rk_self"]
                if is_hira(k[0]) and (len(k[0]) > 1) == (g == "words")
            ]
            row = {}
            for f in ("en_cos", "en_cos_out"):
                d = [recs[a][k][f] - recs["rk_self"][k][f] for k in ks]
                up = sum(x > 0 for x in d)
                row[f] = {
                    "mean": _r(sum(recs[a][k][f] for k in ks) / len(ks), 4),
                    "rk_self": _r(sum(recs["rk_self"][k][f] for k in ks) / len(ks), 4),
                    "delta": _r(sum(d) / len(d), 4),
                    "up_down": [up, sum(x < 0 for x in d)],
                    "p": _r(wilcoxon(d).pvalue if any(d) else 1.0, 4),
                }
                for b in BALL_VS.get(a, ()):
                    if b not in arms:
                        continue
                    d = [recs[a][k][f] - recs[b][k][f] for k in ks]
                    row[f][f"vs_{b}"] = {
                        "delta": _r(sum(d) / len(d), 4),
                        "up_down": [sum(x > 0 for x in d), sum(x < 0 for x in d)],
                        "p": _r(wilcoxon(d).pvalue if any(d) else 1.0, 4),
                    }
            en[f"{g}: {a}"] = row
    out |= {"reads": reads, "paired": pairs, "en_ref": en}
    return out


def kanji(T: Tables) -> dict:
    import torch

    from cjk_scale.paths import OUT as SCALE_OUT
    from cjk_scale.paths import SEED_ROWS_0921

    F = torch.nn.functional

    def han(c):
        return 0x3400 <= ord(c) <= 0x9FFF or 0xF900 <= ord(c) <= 0xFAFF

    def kana(c):
        return 0x3040 <= ord(c) < 0x3100

    sets = {}
    for name, keep in [("kana", kana)] + [(f"kanji_b{b}", han) for b in KANJI_BATCHES]:
        path = (
            SCALE_OUT
            / ("retrain_kana" if name == "kana" else f"retrain_{name}")
            / "trained.pt"
        )
        ids = T.moved(path, keep)
        r = T.rows(path)
        sets[name] = (ids, torch.stack([r[e] for e in ids]))
    kb = [f"kanji_b{b}" for b in KANJI_BATCHES]
    sets["kanji_all"] = (
        sum((sets[k][0] for k in kb), []),
        torch.cat([sets[k][1] for k in kb]),
    )
    s30 = T.rows(SCALE_OUT / "seed_retrain_0930" / "trained.pt")
    ids, X = sets["kanji_all"]
    out: dict = {
        "seed0930_eq_b1_b4_min_cos": _r(
            F.cosine_similarity(torch.stack([s30[e] for e in ids]), X).min(), 6
        ),
        "burr": {k: burr(T, *v) for k, v in sets.items()},
    }
    M = {k: v[1].mean(0) for k, v in sets.items()}
    seed = T.rows(SEED_ROWS_0921)
    for b in KANJI_BATCHES:
        have = [e for e in sets[f"kanji_b{b}"][0] if e in seed]
        if have:
            M[f"seed0921_b{b}"] = torch.stack([seed[e] for e in have]).mean(0)
            out.setdefault("seed0921_has", {})[f"b{b}"] = len(have)
    out["stick_cos"] = {
        a: {b: _r(F.cosine_similarity(M[a], M[b], dim=0)) for b in M} for a in M
    }
    return out


def scene() -> dict:
    from cjk_scale.paths import load_experiment
    from eval.enref import enref_file

    SS = load_experiment("sigma_split")
    en_fw = {
        (pi, 0): SS.placement({"file": str(enref_file(SS.ENREF, pi, 0))})["flat_white"]
        for pi in range(PS.PROMPTS)
    }
    dirs = {k: v for k, v in run_dirs().items() if k not in BAND_ARMS} | {
        k: v for k, v in PS.scale_arms().items()
    }
    out, recs = {}, {}
    for a, d in dirs.items():
        rs = [
            m
            for m in json.loads(PS.reads_of(d).read_text("utf-8"))
            if m["clause"] == PS.CLAUSE and m["pi"] < PS.PROMPTS and m["seed"] == 0
        ]
        assert len(rs) == PS.KEYS * PS.PROMPTS, (a, len(rs))
        for m in rs:
            m["flat_white"] = SS.placement(m)["flat_white"]
        recs[a] = rs
        out[a] = PS.scene(rs, en_fw)
    return out, recs


def sheets(recs: dict, label: str) -> Path:
    from PIL import Image

    from cjk_scale.paths import load_experiment
    from common.readers import contact_sheet
    from eval.enref import enref_file

    SS = load_experiment("sigma_split")
    out = OUT / label / "sheets"
    cols = ("kana_mix", "kana_up") + STICK_RUNS + ("retrain_kana",)
    PS.sheets(recs, [*STICK_RUNS, "retrain_kana"], out)
    by = {a: {(m["text"], m["pi"]): m for m in recs[a]} for a in cols}

    def overview(keys, pi, name):
        rows = []
        for t in keys:
            ref = Image.open(enref_file(SS.ENREF, pi, 0)).convert("RGB")
            rows.append((ref, [f"{t} EN ref p{pi:02d}"]))
            for a in cols:
                m = by[a][(t, pi)]
                r0 = next((r for r in m.get("reads", []) if not r.get("whole")), {})
                rows.append(
                    (
                        Image.open(m["file"]).convert("RGB"),
                        [
                            f"{a}{' ✓' if m.get('exact') else ''}",
                            f"sfx {(r0.get('sfx') or '')[:14]}",
                            f"out {m.get('en_cos_out') or 0:.3f} fw {m['flat_white']:.2f}",
                        ],
                    )
                )
        contact_sheet(rows, out / name, thumb=200, cols=1 + len(cols))

    keys = sorted({m["text"] for m in recs["kana_up"]})
    singles = [k for k in keys if len(k) == 1]
    words = [k for k in keys if len(k) > 1]
    for half, (a, b) in (("a", (0, 7)), ("b", (7, None))):
        overview(singles[a:b], 1, f"overview_singles_p01_{half}.png")
        overview(words[a:b], 0, f"overview_words_p00_{half}.png")
    return out


def ball_sheets(label: str, arms: tuple = ("rk_self", "gs_rkstick", "gs_self")) -> Path:
    """Per prompt, the hiragana words and singles: EN ref | ``arms`` (the
    ``probe_split`` swap arms) — read, EN-ref cos, flat white."""
    from PIL import Image

    from cjk_scale.paths import load_experiment
    from common.readers import contact_sheet
    from eval.enref import enref_file

    SS = load_experiment("sigma_split")
    out = OUT / label / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    by = {
        a: {
            (m["text"], m["pi"]): m
            for m in json.loads(
                (PS.arm_dir(a) / "native_reads.json").read_text("utf-8")
            )
            if m["clause"] == PS.CLAUSE and m["seed"] == 0
        }
        for a in arms
    }
    keys = sorted(
        {t for t, _ in by[arms[0]] if all(0x3041 <= ord(c) <= 0x309F for c in t)}
    )
    for pi in range(PS.PROMPTS):
        ref = Image.open(enref_file(SS.ENREF, pi, 0)).convert("RGB")
        for g, ks in (
            ("words", [k for k in keys if len(k) > 1]),
            ("singles", [k for k in keys if len(k) == 1]),
        ):
            rows = []
            for t in ks:
                rows.append((ref, [f"{t} EN ref p{pi:02d}"]))
                for a in arms:
                    m = by[a][(t, pi)]
                    r0 = next((r for r in m.get("reads", []) if not r.get("whole")), {})
                    fw = SS.placement(m)["flat_white"]
                    rows.append(
                        (
                            Image.open(m["file"]).convert("RGB"),
                            [
                                f"{a}{' ✓' if m.get('exact') else ''}",
                                f"sfx {(r0.get('sfx') or '')[:14]}",
                                f"cos {m['en_cos']:.3f} out {m['en_cos_out']:.3f} fw {fw:.2f}",
                            ],
                        )
                    )
            contact_sheet(
                rows, out / f"{g}_p{pi:02d}.png", thumb=240, cols=1 + len(arms)
            )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--legs", default="geo,kanji,scene,sheets")
    p.add_argument("--label", default="stick_fit")
    a = p.parse_args()
    legs = set(a.legs.split(","))
    from bench._common import make_run_dir, write_result

    metrics: dict = {}
    if legs & {"geo", "kanji", "ball"}:
        T = Tables()
        if "ball" in legs:
            metrics["ball"] = ball(T)
        if "geo" in legs:
            metrics["geo"] = geo(T)
        if "kanji" in legs:
            metrics["kanji"] = kanji(T)
    if "ball_sheets" in legs:
        metrics["ball_sheets"] = str(ball_sheets(a.label))
    if legs & {"scene", "sheets"}:
        metrics["scene"], recs = scene()
        if "sheets" in legs:
            metrics["sheets"] = str(sheets(recs, a.label))
    print(json.dumps(metrics, ensure_ascii=False, indent=1), flush=True)
    run_dir = make_run_dir("cjk_anima_reseed", label=a.label, root=HOME / "results")
    write_result(run_dir, script=__file__, args=vars(a), label=a.label, metrics=metrics)
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
