#!/usr/bin/env python3
"""merge_basis — does giving each artist its OWN input basis make the merge better?

Premise (two closed lines, joined): artist ΔWs are near-orthogonal except on the
input side, where the shared ``down_init="weight_svd"`` seed co-locates every
adapter's ``A`` (in-overlap 0.589, 33× null — ``artist_lora_merge_interference_phase0``);
and E1 (``bench/grad_init``) found ``kaiming`` render-equal to ``weight_svd`` per
artist, while a same-artist weight_svd-vs-kaiming pair has in-overlap exactly at
null — ``A`` stays where it was seeded. So a per-artist random seed moves the
input co-location to null at zero single-artist cost. Untested: whether that
shows up in the MERGE.

Arms (2-way ``merge_loras.py`` concat, default global normalize), X = ``aak``
(E1's checkpoints reused), Y = ``channel_(caststation)`` (trained here):

- ``SHAREDWSVD``  X weight_svd ⊕ Y weight_svd  — the shipped path
- ``SHAREDKAI``   X kaiming s42 ⊕ Y kaiming s42 — same random seed, isolates
  "shared seed" from "weight_svd" (checked post hoc: its in-overlap must be ≫
  null, else the seed did not reproduce and this arm is a second DISTINCT)
- ``DISTINCT``    X kaiming s42 ⊕ Y kaiming s7  — distinct random bases

Reads: (1) in/out-subspace overlap per (X, Y) pair — the mechanism gauge, from
the archived phase-0 probe's metric; (2) blind A/B on 24 tag-routed rows (12
``@aak`` + the same 12 as ``@channel (caststation)``), direct pairings
DISTINCT-vs-SHAREDWSVD (the shipping question) and DISTINCT-vs-SHAREDKAI (the
mechanism), each set at its own seed since DISTINCT recurs.

Caveat carried: one training seed per arm — the seed lottery
(``docs/experimental/soup.md`` Act 5) can swing a single pair; a lean here
earns a second-training-seed replicate, not a verdict.

``--variant n4`` (added after the N=2 flat) merges all four E0 held-out artists
— aak / channel / sweetonedollar / ootomo_takuji — as ``SHAREDWSVD4`` vs
``DISTINCT4`` (per-artist kaiming seeds 42/7/11/13), 6 base rows × 4 triggers
(``prompts_n4.txt``), one blind set. The shared-``A`` collapse ``(ΣB_i)·A`` is
N× harsher there: 32 input directions for four artists vs 128.

Usage (one daemon command job; ~4 min per train + renders)::

    make daemon-run ARGS="--stall-timeout 0 --queue bench/merge_basis/run.py --variant n4 --push"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
PY = sys.executable

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402
from bench.grad_init.e1_blind import BLIND, INFER_BASE, n_rows, rendered, run  # noqa: E402

CKPT = Path("output/ckpt/merge_basis")
# artist -> path_pattern, trigger tag, per-artist random seed for the DISTINCT
# arm, and (optional) explicit checkpoint paths for runs that already exist
# under another name (E1's aak arms, the N=2 channel runs).
ARTISTS = {
    "aak": {
        "pattern": "aak/*",
        "trigger": "@aak",
        "kai_seed": 42,
        "wsvd": "output/ckpt/e1_grad_init/e1_weightsvd.safetensors",
        "kai": "output/ckpt/e1_grad_init/e1_kaiming.safetensors",
    },
    "channel": {
        "pattern": "channel_(caststation)/*",
        "trigger": "@channel (caststation)",
        "kai_seed": 7,
        "wsvd": str(CKPT / "Y_wsvd_s42.safetensors"),
        "kai": str(CKPT / "Y_kai_s7.safetensors"),
        # same random seed as aak's kaiming run — the N=2 shared-seed control
        "kai_shared": str(CKPT / "Y_kai_s42.safetensors"),
    },
    "sweet": {
        "pattern": "sweetonedollar/*",
        "trigger": "@sweetonedollar",
        "kai_seed": 11,
    },
    "ootomo": {
        "pattern": "ootomo_takuji/*",
        "trigger": "@ootomo takuji",
        "kai_seed": 13,
    },
}
SHARED_SEED = 42

# variant -> arms {arm: [(artist, init_key), ...]} + blind sets (name, arms, seeds)
VARIANTS = {
    "n2": {
        "artists": ["aak", "channel"],
        "arms": {
            "SHAREDWSVD": [("aak", "wsvd"), ("channel", "wsvd")],
            "SHAREDKAI": [("aak", "kai"), ("channel", "kai_shared")],
            "DISTINCT": [("aak", "kai"), ("channel", "kai")],
        },
        "sets": [
            ("s26_MB_DISTINCT_vs_SHAREDWSVD", ["DISTINCT", "SHAREDWSVD"], [42]),
            ("s27_MB_DISTINCT_vs_SHAREDKAI", ["DISTINCT", "SHAREDKAI"], [7]),
        ],
        "prompts": "bench/merge_basis/prompts.txt",
        "eval_dir": "output/tests/merge_basis_eval",
    },
    "n4": {
        "artists": ["aak", "channel", "sweet", "ootomo"],
        "arms": {
            "SHAREDWSVD4": [(a, "wsvd") for a in ("aak", "channel", "sweet", "ootomo")],
            "DISTINCT4": [(a, "kai") for a in ("aak", "channel", "sweet", "ootomo")],
        },
        "sets": [
            ("s28_MB4_DISTINCT_vs_SHAREDWSVD", ["DISTINCT4", "SHAREDWSVD4"], [42])
        ],
        "prompts": "bench/merge_basis/prompts_n4.txt",
        "eval_dir": "output/tests/merge_basis_eval_n4",
    },
}


def artist_ckpt(artist: str, init_key: str) -> Path:
    a = ARTISTS[artist]
    if init_key in a:
        return Path(a[init_key])
    if init_key == "wsvd":
        return CKPT / f"{artist}_wsvd_s{SHARED_SEED}.safetensors"
    if init_key == "kai":
        return CKPT / f"{artist}_kai_s{a['kai_seed']}.safetensors"
    raise KeyError(init_key)


def artist_run_spec(artist: str, init_key: str) -> tuple[str, int]:
    """(down_init, seed) that produces ``artist_ckpt(artist, init_key)``."""
    if init_key == "wsvd":
        return "weight_svd", SHARED_SEED
    if init_key == "kai":
        return "kaiming", ARTISTS[artist]["kai_seed"]
    if init_key == "kai_shared":
        return "kaiming", SHARED_SEED
    raise KeyError(init_key)


def merged_ckpt(arm: str) -> Path:
    return CKPT / f"M_{arm}.safetensors"


# --- overlap metric (vendored from _archive/bench/lora_merge_interference/probe.py)


def load_lora(path: Path) -> dict[str, dict]:
    downs, ups, alphas = {}, {}, {}
    with safe_open(str(path), framework="pt") as f:
        for k in f.keys():
            if k.endswith(".lora_down.weight"):
                downs[k[: -len(".lora_down.weight")]] = f.get_tensor(k)
            elif k.endswith(".lora_up.weight"):
                ups[k[: -len(".lora_up.weight")]] = f.get_tensor(k)
            elif k.endswith(".alpha"):
                alphas[k[: -len(".alpha")]] = f.get_tensor(k)
    out = {}
    for stem, down in downs.items():
        r = down.shape[0]
        alpha = float(alphas[stem].item()) if stem in alphas else float(r)
        out[stem] = {"down": down.float(), "up": ups[stem].float(), "scale": alpha / r}
    return out


def _orth(mat: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q


def _overlap(qa: torch.Tensor, qb: torch.Tensor) -> float:
    """mean squared cosine of principal angles ∈ [0,1] (0 = orthogonal)."""
    return (torch.linalg.matrix_norm(qa.T @ qb) ** 2 / qa.shape[1]).item()


def pair_overlap(pa: Path, pb: Path) -> dict:
    la, lb = load_lora(pa), load_lora(pb)
    rows = []
    for stem in sorted(set(la) & set(lb)):
        a, b = la[stem], lb[stem]
        r, in_dim, out_dim = a["down"].shape[0], a["down"].shape[1], a["up"].shape[0]
        dwa = a["scale"] * (a["up"] @ a["down"])
        dwb = b["scale"] * (b["up"] @ b["down"])
        rows.append(
            {
                "in": _overlap(_orth(a["down"].T), _orth(b["down"].T)),
                "out": _overlap(_orth(a["up"]), _orth(b["up"])),
                "in_null": r / in_dim,
                "out_null": r / out_dim,
                "abs_cos": abs(
                    torch.nn.functional.cosine_similarity(
                        dwa.flatten(), dwb.flatten(), dim=0
                    ).item()
                ),
            }
        )
    mean = lambda k: sum(x[k] for x in rows) / len(rows)  # noqa: E731
    return {
        "n_modules": len(rows),
        "in_overlap": mean("in"),
        "in_null": mean("in_null"),
        "in_vs_null": mean("in") / mean("in_null"),
        "out_overlap": mean("out"),
        "out_null": mean("out_null"),
        "out_vs_null": mean("out") / mean("out_null"),
        "abs_cos_mean": mean("abs_cos"),
    }


# --- stages


def train_artist(artist: str, init_key: str) -> None:
    ckpt = artist_ckpt(artist, init_key)
    if ckpt.exists():
        print(f"=== {ckpt.name} exists, skip train", flush=True)
        return
    down_init, seed = artist_run_spec(artist, init_key)
    run(
        f"train {ckpt.stem}",
        [
            PY,
            "train.py",
            "--method",
            "lora",
            "--preset",
            "default",
            "--path_pattern",
            ARTISTS[artist]["pattern"],
            "--seed",
            str(seed),
            "--deterministic",
            "--paired_step_rng",
            "--output_dir",
            str(ckpt.parent),
            "--output_name",
            ckpt.stem,
            "--network_args",
            f"down_init={down_init}",
        ],
    )


def merge(arm: str, members: list[tuple[str, str]]) -> None:
    if merged_ckpt(arm).exists():
        print(f"=== M_{arm} exists, skip merge", flush=True)
        return
    run(
        f"merge {arm}",
        [
            PY,
            "scripts/toolkits/merge_loras.py",
            *(str(artist_ckpt(a, k)) for a, k in members),
            "--out",
            str(merged_ckpt(arm)),
        ],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=list(VARIANTS), default="n2")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--eval_dir", default=None)
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_render", action="store_true")
    ap.add_argument("--skip_sets", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--label", default=None)
    o = ap.parse_args()
    v = VARIANTS[o.variant]
    prompts = (REPO / (o.prompts or v["prompts"])).resolve()
    eval_dir = (REPO / (o.eval_dir or v["eval_dir"])).resolve()
    rows = n_rows(prompts)
    CKPT.mkdir(parents=True, exist_ok=True)
    members_needed = sorted({m for ms in v["arms"].values() for m in ms})

    if not o.skip_train:
        for artist, key in members_needed:
            train_artist(artist, key)
    for arm, members in v["arms"].items():
        merge(arm, members)

    # mechanism gauge: pairwise in/out-subspace overlap inside each arm, averaged
    overlap = {}
    for arm, members in v["arms"].items():
        pairs = [
            pair_overlap(REPO / artist_ckpt(*a), REPO / artist_ckpt(*b))
            for i, a in enumerate(members)
            for b in members[i + 1 :]
        ]
        keys = (
            "in_overlap",
            "in_vs_null",
            "out_overlap",
            "out_vs_null",
            "abs_cos_mean",
        )
        overlap[arm] = {k: sum(p[k] for p in pairs) / len(pairs) for k in keys}
        overlap[arm]["n_pairs"] = len(pairs)
        s = overlap[arm]
        print(
            f"=== overlap {arm} ({len(pairs)} pairs): in {s['in_overlap']:.3f}"
            f" ({s['in_vs_null']:.1f}x null)  out {s['out_overlap']:.3f}"
            f" ({s['out_vs_null']:.1f}x null)  |cos| {s['abs_cos_mean']:.3f}",
            flush=True,
        )
    if "SHAREDKAI" in overlap and overlap["SHAREDKAI"]["in_vs_null"] < 5:
        print(
            "!!! SHAREDKAI in-overlap is near null: the same --seed did NOT reproduce "
            "the kaiming basis across artists; read s27 as a second DISTINCT pairing.",
            flush=True,
        )

    needed = sorted(
        {(a, s) for _, arms, seeds in v["sets"] for a in arms for s in seeds}
    )
    if not o.skip_render:
        for arm, seed in needed:
            if rendered(eval_dir, arm, seed, rows):
                print(f"=== arm{arm} s{seed} already rendered, skip", flush=True)
                continue
            run(
                f"gen arm{arm} s{seed}",
                [
                    PY,
                    *INFER_BASE,
                    "--lora_weight",
                    str(merged_ckpt(arm)),
                    "--from_file",
                    str(prompts),
                    "--seed",
                    str(seed),
                    "--save_path",
                    str(eval_dir / f"arm{arm}_s{seed}"),
                ],
            )

    if not o.skip_sets:
        for name, arms, seeds in v["sets"]:
            run(
                f"blind {name}",
                [
                    PY,
                    str(BLIND),
                    "make",
                    "--set",
                    name,
                    "--arms",
                    *arms,
                    "--seeds",
                    *map(str, seeds),
                    "--eval_dir",
                    str(eval_dir),
                    "--prompts",
                    str(prompts),
                    "--overwrite",
                    *(["--push"] if o.push else []),
                ],
            )

    run_dir = make_run_dir("merge_basis", label=o.label or f"mb_{o.variant}")
    (run_dir / "overlap.json").write_text(json.dumps(overlap, indent=2))
    write_result(
        run_dir,
        script=__file__,
        args=o,
        metrics={
            "variant": o.variant,
            "arms": {
                arm: [str(artist_ckpt(a, k)) for a, k in ms]
                for arm, ms in v["arms"].items()
            },
            "merges": {k: str(merged_ckpt(k)) for k in v["arms"]},
            "overlap": overlap,
            "sets": [{"set": n, "arms": a, "seeds": s} for n, a, s in v["sets"]],
            "rows": rows,
        },
        artifacts=["overlap.json"],
    )
    print(f"wrote {run_dir}/result.json", flush=True)


if __name__ == "__main__":
    main()
