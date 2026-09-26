#!/usr/bin/env python
"""f0_interaction — proposal_factorizedrows.md § 2 (Stage F0): does the
adapter already modulate a row's change by its neighbours? (2026-09-26)

No DiT, no renders: Qwen3 + ``llm_adapter`` forward only, the pack hooked as
in training, an ``ExtDelta`` (row_scale 1) carrying effective rows. For a
glyph ``g``, the native ruler's captions (8 scene prompts × en + swap) with
``g`` **alone** and ``g`` **inside a spelled word**; per block boundary
``l`` (``L0`` = block 0's input, ``B1``–``B6`` = block outputs, ``out`` =
the crossattn embedding the DiT reads), at ``g``'s position:

    D_lone = H[seed + arm row on g]            − H[seed]   (g alone)
    D_self = H[seed + arm row on g]            − H[seed]   (g in the word)
    D_all  = H[seed + arm rows on every glyph] − H[seed]   (g in the word)

``D_self`` vs ``D_lone`` is the adapter's own context modulation of ``g``'s
change (same row change, different neighbours); ``D_all`` is what a render
sees (the neighbours changed too). Arms, all Stage B's, on disk:

    u1     the held-out 10's seed rows + step · u_S (tb_t1_u1) — held-out words
    rand1  the same step along a random ⟂ unit (tb_t1_rand1) — the baseline
           for a generic perturbation of that size
    donor  the Stage B donor's trained rows (run0926_stage_b) — こんにちは +
           12 donor words, on their own rows

Per pair (arm, word, position, prompt, clause) against the same (glyph,
prompt, clause) alone: cos(D_x, D_lone), ‖D_x − D_lone‖ / ‖D_lone‖ (the
interaction), gain = D_x · D_lone / ‖D_lone‖² (> 1: the change is larger in
the line), the seed's own context cos (H_seed line vs lone at g), the spill
of D to the other positions, and each block's branch (self-attn / cross-attn
/ MLP) share of the interaction.

Decision (proposal_factorizedrows.md § 2): ``out`` cos(D_self, D_lone) ≥ 0.95
→ the adapter passes the row change through blind to its neighbours, and
the gate is § 1's. Large interaction with gain > 1 (the change shrinks when
the row is alone) → the adapter already turns the mode down alone, and the
doubling is DiT-side or a data gap. Large but gain ≈ 1 → read the branches.

``--dry_run`` prints the captions and checks the encodings (tokenizers
only). GPU through the daemon:
``ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack make daemon-run
ARGS="project/cjk_anima_scale/experiments/f0_interaction/run_exp.py --label f0"``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[2]  # project/cjk_anima_scale
sys.path.insert(0, str(LINE))
from cjk_scale.paths import OUT, SEED_ROWS, bootstrap  # noqa: E402

bootstrap()
from bench._common import make_run_dir, write_result  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "stage_b_exp", LINE / "experiments" / "stage_b" / "run_exp.py"
)
SB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SB)

EXP = OUT / "experiments"
ARMS = {
    "u1": EXP / "tb_t1_u1" / "trained.pt",
    "rand1": EXP / "tb_t1_rand1" / "trained.pt",
    "donor": OUT / SB.NAME / "trained.pt",
}
N_DONOR_WORDS = 12
CLAUSES = ("en", "swap")
LAYERS = ("L0", "B1", "B2", "B3", "B4", "B5", "B6", "out")
BRANCHES = ("self_attn", "cross_attn", "mlp")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def words_by_arm() -> dict[str, list[str]]:
    dw = json.loads((OUT / SB.NAME / "donor_words.json").read_text("utf-8"))
    pool = [w for w in dw if 3 <= len(w) <= 5]
    donor = sorted(random.Random(0).sample(pool, N_DONOR_WORDS))
    held = list(SB.HELD_WORDS)
    return {"u1": held, "rand1": held, "donor": [SB.HELD_IN, *donor]}


def prompts() -> list[str]:
    from common.prompts import NATIVE_PROMPTS

    return [
        ln.strip()
        for ln in Path(NATIVE_PROMPTS).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


def caption(p: str, k: str, cl: str) -> str:
    from common.prompts import NATIVE_CLAUSES

    return NATIVE_CLAUSES[cl].format(p=p, k=k)


def plan():
    """``(prompts, words per arm, glyphs per arm, every caption)``."""
    ps = prompts()
    wa = words_by_arm()
    ga = {a: sorted({g for w in ws for g in w}) for a, ws in wa.items()}
    caps = set()
    for a in wa:
        for p in ps:
            for cl in CLAUSES:
                caps.update(caption(p, SB.spell(w), cl) for w in wa[a])
                caps.update(caption(p, g, cl) for g in ga[a])
    return ps, wa, ga, sorted(caps)


# ----------------------------------------------------------------------------
# capture


class Capture:
    """Block 0's input, every block's output and every branch's output."""

    def __init__(self, adapter):
        self.h: dict = {}
        blocks = adapter.blocks
        self.handles = [
            blocks[0].register_forward_pre_hook(
                lambda m, a: self.h.__setitem__("L0", a[0].float())
            )
        ]
        for b, blk in enumerate(blocks, 1):
            self.handles.append(
                blk.register_forward_hook(
                    lambda m, a, o, b=b: self.h.__setitem__(f"B{b}", o.float())
                )
            )
            for br in BRANCHES:
                self.handles.append(
                    getattr(blk, br).register_forward_hook(
                        lambda m, a, o, k=f"B{b}.{br}": self.h.__setitem__(k, o.float())
                    )
                )


def forward(adapter, cap, delta, base, over: dict, batch):
    """One adapter forward over ``batch`` (the encoded captions) with the
    delta = ``base`` (the seed's effective rows, in ``delta.ext_ids`` order)
    and ``over`` (``{ext id: effective row}``) on top → ``{key: (B, L, D)}``."""
    import torch

    raw = base.clone()
    for e, r in over.items():
        raw[delta.index[e]] = r.to(raw)
    delta.raw.data.copy_(raw)
    pe, am, t5, t5m = batch
    cap.h.clear()
    with torch.no_grad():
        out = adapter(
            source_hidden_states=pe,
            target_input_ids=t5,
            target_attention_mask=t5m,
            source_attention_mask=am,
        )
    h = dict(cap.h)
    h["out"] = out.float()
    return h


# ----------------------------------------------------------------------------
# stats


def _cos(a, b):
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def pair_stats(D_x: dict, D_lone: dict, Hs_line: dict, Hs_lone: dict) -> dict:
    """``D_x`` / ``D_lone``: ``{key: (dim,)}`` at g's position; ``Hs_*`` the
    seed's own states there."""
    out = {}
    for ly in LAYERS:
        x, lo = D_x[ly], D_lone[ly]
        n = lo.norm().clamp_min(1e-12)
        out[ly] = {
            "cos": _cos(x, lo),
            "inter": float((x - lo).norm() / n),
            "gain": float((x @ lo) / n**2),
            "ratio": float(x.norm() / n),
            "seed_ctx": _cos(Hs_line[ly], Hs_lone[ly]),
        }
    for b in range(1, 7):
        n = D_lone[f"B{b}"].norm().clamp_min(1e-12)
        for br in BRANCHES:
            k = f"B{b}.{br}"
            out[k] = {"inter": float((D_x[k] - D_lone[k]).norm() / n)}
    return out


def spill(D, valid, own: list[int], nbr: list[int]) -> dict:
    """RMS of D over the neighbour glyph positions and over the rest of the
    caption, relative to ‖D‖ at the own position(s)."""
    own_n = D[own].norm(dim=-1).mean().clamp_min(1e-12)
    rest = [i for i in valid if i not in own and i not in nbr]
    o = {"rest": float(D[rest].norm(dim=-1).pow(2).mean().sqrt() / own_n)}
    if nbr:
        o["nbr"] = float(D[nbr].norm(dim=-1).pow(2).mean().sqrt() / own_n)
    return o


def summarize(vals: list[dict]) -> dict:
    import statistics as st

    keys = vals[0].keys()
    out = {}
    for k in keys:
        xs = [v[k] for v in vals]
        out[k] = {"mean": round(st.fmean(xs), 4), "median": round(st.median(xs), 4)}
    return out


# ----------------------------------------------------------------------------


def main():
    args = parse_args()
    ps, wa, ga, caps = plan()
    ext = SB.encoder()
    all_words = sorted({w for ws in wa.values() for w in ws})
    ids = {}
    for w in all_words:
        got = ext(SB.spell(w))
        assert len(got) == len(w), (w, got)
        for c, e in zip(w, got):
            assert ids.setdefault(c, e) == e, (c, e)
            assert ext(c) == [e], (c, ext(c), e)
    print(
        f"prompts {len(ps)} × clauses {CLAUSES}; captions {len(caps)}; "
        + "; ".join(f"{a}: {len(wa[a])} words {len(ga[a])} glyphs" for a in wa),
        flush=True,
    )
    for a, ws in wa.items():
        print(f"  {a}: {' '.join(ws)}", flush=True)
    metrics: dict = {"prompts": len(ps), "words": wa, "glyph_ids": ids}
    if args.dry_run:
        print(caps[0], caps[-1], sep="\n", flush=True)
        return

    import torch

    from common.hooks import ExtDelta
    from common.models import checkpoints, encode_captions
    from library.anima.ext_vocab import T5_TABLE_SIZE
    from library.anima.weights import load_llm_adapter

    run_dir = make_run_dir(
        "f0_interaction",
        label=args.label,
        root=LINE / "experiments" / "f0_interaction" / "results",
    )
    dev = torch.device(args.device)
    ck = checkpoints()
    assert ck.vocab_pack, "set ANIMA_VOCAB_PACK"
    work = OUT / f"f0_interaction_{args.label}"
    enc = encode_captions(caps, dev, cache_dir=work / "te_cache")
    adapter = load_llm_adapter(
        ck.dit, dtype=torch.float32, device=dev, vocab_pack=ck.vocab_pack
    )
    seed, _ = SB.rows(SEED_ROWS)
    arm_rows = {a: SB.rows(p)[0] for a, p in ARMS.items()}
    delta = ExtDelta(
        type("A", (), {"llm_adapter": adapter})(),
        sorted(seed),
        next(iter(seed.values())).shape[0],
        dev,
        1.0,
    )
    base = torch.stack([seed[e] for e in delta.ext_ids]).to(dev)
    cap = Capture(adapter)
    for a, r in arm_rows.items():
        moved = [c for c in ga[a] if (r[ids[c]] - seed[ids[c]]).norm() > 1e-3]
        print(f"arm {a}: {len(moved)} / {len(ga[a])} glyph rows differ from the seed", flush=True)
        assert len(moved) == len(ga[a]), (a, set(ga[a]) - set(moved))

    def batch_of(k: str):
        cs = [caption(p, k, cl) for p in ps for cl in CLAUSES]
        keys = [(pi, cl) for pi in range(len(ps)) for cl in CLAUSES]
        parts = [enc[c] for c in cs]
        b = [
            torch.stack([torch.as_tensor(x[i]) for x in parts]).to(dev)
            for i in range(4)
        ]
        b[0] = b[0].float()
        b[2] = b[2].long()
        return keys, b

    def positions(t5_row, glyphs: str) -> list[int]:
        t = t5_row.tolist()
        pos = []
        for c in glyphs:
            hit = [i for i, v in enumerate(t) if v == T5_TABLE_SIZE + ids[c]]
            assert len(hit) == 1, (glyphs, c, hit)
            pos.append(hit[0])
        n_ext = sum(v >= T5_TABLE_SIZE for v in t)
        assert n_ext == len(glyphs), (glyphs, n_ext)  # no other pack row
        return pos

    def at(h: dict, i: int, p: int) -> dict:
        return {k: v[i, p].clone() for k, v in h.items()}

    results: dict = {}
    for a in ARMS:
        R = arm_rows[a]
        lone: dict = {}
        sp_lone = []
        for g in ga[a]:
            keys, b = batch_of(g)
            hs = forward(adapter, cap, delta, base, {}, b)
            ha = forward(adapter, cap, delta, base, {ids[g]: R[ids[g]]}, b)
            for i, key in enumerate(keys):
                (p,) = positions(b[2][i], g)
                D = {k: (ha[k][i, p] - hs[k][i, p]).clone() for k in ha}
                lone[(g, *key)] = (D, at(hs, i, p))
                valid = b[3][i].bool().nonzero().flatten().tolist()
                sp_lone.append(spill(ha["out"][i] - hs["out"][i], valid, [p], []))
        pairs = {"self": [], "all": []}
        sp_line = {"self": [], "all": []}
        by_pos: dict = {"first": [], "inner": [], "last": []}
        by_glyph: dict = {}
        for w in wa[a]:
            keys, b = batch_of(SB.spell(w))
            hs = forward(adapter, cap, delta, base, {}, b)
            h_all = forward(
                adapter,
                cap,
                delta,
                base,
                {ids[c]: R[ids[c]] for c in w},
                b,
            )
            for j, g in enumerate(w):
                h_self = None
                h_self = forward(
                    adapter, cap, delta, base, {ids[g]: R[ids[g]]}, b
                )
                for i, key in enumerate(keys):
                    pos = positions(b[2][i], w)
                    p = pos[j]
                    D_lone, Hs_lone = lone[(g, *key)]
                    Hs_line = at(hs, i, p)
                    valid = b[3][i].bool().nonzero().flatten().tolist()
                    for name, h in (("self", h_self), ("all", h_all)):
                        D = {k: h[k][i, p] - hs[k][i, p] for k in h}
                        s = pair_stats(D, D_lone, Hs_line, Hs_lone)
                        pairs[name].append(s)
                        if name == "self":
                            nbr = [q for q in pos if q != p]
                            sp_line["self"].append(
                                spill(h["out"][i] - hs["out"][i], valid, [p], nbr)
                            )
                            bucket = (
                                "first"
                                if j == 0
                                else "last"
                                if j == len(w) - 1
                                else "inner"
                            )
                            by_pos[bucket].append(s["out"])
                            by_glyph.setdefault(g, []).append(s["out"])
                if j == 0:
                    for i in range(len(keys)):
                        pos = positions(b[2][i], w)
                        valid = b[3][i].bool().nonzero().flatten().tolist()
                        sp_line["all"].append(
                            spill(h_all["out"][i] - hs["out"][i], valid, pos, [])
                        )
        res = {
            "n_pairs": len(pairs["self"]),
            **{
                name: {
                    k: summarize([v[k] for v in pairs[name]])
                    for k in pairs[name][0]
                }
                for name in pairs
            },
            "spill_out": {
                "lone": summarize(sp_lone),
                "self": summarize(sp_line["self"]),
                "all": summarize(sp_line["all"]),
            },
            "out_by_pos": {k: summarize(v) for k, v in by_pos.items() if v},
            "out_by_glyph": {k: summarize(v) for k, v in sorted(by_glyph.items())},
        }
        results[a] = res
        print(f"\n== {a}: {res['n_pairs']} pairs", flush=True)
        print(
            f"  {'layer':<6} {'cos self':>8} {'cos all':>8} {'int self':>8} "
            f"{'int all':>8} {'gain s':>7} {'gain a':>7} {'seed ctx':>8}",
            flush=True,
        )
        for ly in LAYERS:
            s, al = res["self"][ly], res["all"][ly]
            print(
                f"  {ly:<6} {s['cos']['mean']:>8.3f} {al['cos']['mean']:>8.3f} "
                f"{s['inter']['mean']:>8.3f} {al['inter']['mean']:>8.3f} "
                f"{s['gain']['mean']:>7.3f} {al['gain']['mean']:>7.3f} "
                f"{s['seed_ctx']['mean']:>8.3f}",
                flush=True,
            )
        print("  branch interaction (self), / ‖D_lone‖ at the block output:", flush=True)
        for bb in range(1, 7):
            print(
                f"    B{bb}: "
                + "  ".join(
                    f"{br} {res['self'][f'B{bb}.{br}']['inter']['mean']:.3f}"
                    for br in BRANCHES
                ),
                flush=True,
            )
        print(f"  spill (out): {json.dumps(res['spill_out'])}", flush=True)
        print(
            "  out by position: "
            + "  ".join(
                f"{k} cos {v['cos']['mean']:.3f} gain {v['gain']['mean']:.3f}"
                for k, v in res["out_by_pos"].items()
            ),
            flush=True,
        )
        print(
            "  out by glyph (cos · gain): "
            + " ".join(
                f"{g} {v['cos']['mean']:.2f}·{v['gain']['mean']:.2f}"
                for g, v in res["out_by_glyph"].items()
            ),
            flush=True,
        )
    metrics["arms"] = results
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        artifacts=[str(p) for p in ARMS.values()],
    )
    print(f"→ {run_dir / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
