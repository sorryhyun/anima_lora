"""G5 — is the Phase-2c `cos_vs_en` gate reachable *under the settled objective*?

Phase 2c is gated on two flat, position-wise cosines (`ja_ext cos_vs_en >= 0.6`
and held-out student-vs-teacher `>= 0.9`) — both measured in the space that G2
disqualified as an *objective*, for a reason that applies just as hard to a
metric: the teacher's T5 ids **are** the EN reference's ids, so its cosine is a
pure Qwen-side difference at perfectly aligned positions, while the student
segments the same content into a different number of JA tokens and is charged
position-by-position for the offset.

"The student scores 0.10 against a 0.6 gate" is therefore two claims wearing one
number, and they have opposite consequences:

  (a) the student is bad          → train harder / more corpus / escalate to 2-ii
  (b) the gate is nearly blind    → the gate is wrong, not the student

This gate separates them with an **oracle**: a synthetic student that achieves
the span objective *exactly*, scored by 2c's own metric. Nothing is trained and
no adapter runs — teacher / reference / native arms are all cached, and the
oracle is assembled out of the teacher's own rows.

  ``span_perfect``  every student span position := the teacher's mean over the
                    aligned EN span. This **is** the argmin of `L_span` (the
                    loss pools each side's span and cosines the two pools), so
                    its flat score is what a perfectly distilled span student
                    gets. Non-span positions are zero — the objective says
                    nothing about them.
  ``span_plus``     same, but non-span positions are filled with the teacher's
                    row at the same index instead of zero. The generous variant:
                    it hands the student, for free, the positions its objective
                    never supervised.
  ``ref_remap``     the reference itself, resampled onto the student's length.
                    Perfect content, wrong segmentation — isolates the length
                    penalty alone.
  ``prefix_bound``  ‖ref[:min(N_s,N_r)]‖ / ‖ref‖ — the unconstrained ceiling for
                    *any* tensor of the student's length. Reported to keep the
                    other three honest: if this is ~1.0, nothing about the gate
                    is impossible in principle, and the cap that matters is the
                    objective's, not arithmetic's.

Read the four against the teacher's own 0.823 and the 0.6 gate. Every row is
also reported in the cross-attention readout space, where the same oracles
should sit near the ceiling — that contrast *is* the finding.

Usage (CPU-viable; the readout columns want the DiT for K/V)::

    make daemon-run ARGS="project/finished/cjk_aware_anima/gates/g5.py"
"""

from __future__ import annotations

import collections
import json
import logging
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench._common import make_run_dir, write_result  # noqa: E402
from scripts.distill_cjk import config as cfg_mod  # noqa: E402
from scripts.distill_cjk import distill  # noqa: E402
from scripts.distill_cjk import losses as loss_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ORACLES = ("span_perfect", "span_plus", "ref_remap")


def build_oracles(b: dict, teacher: torch.Tensor, ref: torch.Tensor) -> dict:
    """The synthetic students, in the student's own [B, Ls, D] layout.

    Assembled from the *teacher's* rows, so the only thing being measured is
    what the objective's optimum looks like once it is laid out on the student's
    token positions.
    """
    B, Ls, D = b["s_mask"].shape[0], b["s_mask"].shape[1], teacher.shape[-1]
    dev, dt = teacher.device, teacher.dtype
    span_perfect = torch.zeros(B, Ls, D, device=dev, dtype=dt)
    covered = torch.zeros(B, Ls, dtype=torch.bool, device=dev)

    for j, spans in enumerate(b["spans"]):
        for sp in spans:
            t_idx = torch.tensor(sp.teacher, dtype=torch.long, device=dev)
            s_idx = torch.tensor(sp.student, dtype=torch.long, device=dev)
            # Guard: the cache is trimmed to non-pad, and a truncated pair can
            # leave a span index past the trimmed length.
            t_idx = t_idx[t_idx < teacher.shape[1]]
            s_idx = s_idx[s_idx < Ls]
            if t_idx.numel() == 0 or s_idx.numel() == 0:
                continue
            span_perfect[j, s_idx] = teacher[j, t_idx].mean(0)
            covered[j, s_idx] = True

    s_mask = b["s_mask"].bool()
    span_perfect = span_perfect * s_mask.unsqueeze(-1)
    covered = covered & s_mask

    # `span_plus`: the same, with the teacher's own row at index i wherever the
    # objective was silent.
    span_plus = span_perfect.clone()
    Lt = teacher.shape[1]
    fill = torch.zeros(B, Ls, D, device=dev, dtype=dt)
    n = min(Ls, Lt)
    fill[:, :n] = teacher[:, :n]
    free = (~covered) & s_mask
    span_plus = torch.where(free.unsqueeze(-1), fill, span_plus)

    # `ref_remap`: the reference resampled onto the student's length, per sample
    # (lengths differ inside a batch, so this is a per-row gather).
    ref_remap = torch.zeros(B, Ls, D, device=dev, dtype=dt)
    n_s = b["s_mask"].sum(1).tolist()
    n_r = b["ref_mask"].sum(1).tolist()
    for j in range(B):
        ns, nr = int(n_s[j]), int(n_r[j])
        if ns == 0 or nr == 0:
            continue
        src = (torch.arange(ns, device=dev).float() * nr / ns).long().clamp_max(nr - 1)
        ref_remap[j, :ns] = ref[j, src]

    return {
        "span_perfect": span_perfect,
        "span_plus": span_plus,
        "ref_remap": ref_remap,
        "_covered": covered,
    }


@torch.no_grad()
def run_g5(ctx, cfg, device, dtype, batch_size: int = 32) -> dict:
    hold = ctx["hold_cache"]
    probes = ctx["probes"]
    if hold is None:
        raise SystemExit("G5 needs a cached holdout split")

    acc: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for start in range(0, len(hold), batch_size):
        idx = list(range(start, min(start + batch_size, len(hold))))
        if len(idx) < 2:
            break
        b = hold.batch(idx, device, dtype)
        t = loss_mod.zero_pads(b["teacher"], b["teacher_mask"])
        ref = loss_mod.zero_pads(b["ref"], b["ref_mask"])
        nat = loss_mod.zero_pads(b["native"], b["native_mask"])
        oracles = build_oracles(b, t, ref)
        covered = oracles.pop("_covered")

        cols = {
            "teacher_vs_en": distill.flat_cos(t, ref),
            "native_vs_en": distill.flat_cos(nat, ref),
        }
        for name in ORACLES:
            cols[f"{name}_vs_en"] = distill.flat_cos(oracles[name], ref)
            cols[f"{name}_vs_teacher"] = distill.flat_cos(oracles[name], t)

        # The unconstrained ceiling for a tensor of the student's length: keep
        # the reference's first min(N_s, N_r) rows, drop the rest.
        r_norm = ref.float().reshape(ref.shape[0], -1).norm(dim=-1).clamp_min(1e-8)
        n_s = b["s_mask"].sum(1)
        n_r = b["ref_mask"].sum(1)
        keep = torch.minimum(n_s, n_r)
        pos = torch.arange(ref.shape[1], device=ref.device).unsqueeze(0)
        prefix = (
            (ref.float() * (pos < keep.unsqueeze(1)).unsqueeze(-1))
            .reshape(ref.shape[0], -1)
            .norm(dim=-1)
        )
        cols["prefix_bound"] = prefix / r_norm

        if probes:
            rr = distill.readout_vec(ref, b["ref_mask"], probes)
            rt = distill.readout_vec(t, b["teacher_mask"], probes)
            rn = distill.readout_vec(nat, b["native_mask"], probes)
            cols["teacher_vs_en_attn"] = distill.flat_cos(rt, rr)
            cols["native_vs_en_attn"] = distill.flat_cos(rn, rr)
            for name in ORACLES:
                ro = distill.readout_vec(oracles[name], b["s_mask"], probes)
                cols[f"{name}_vs_en_attn"] = distill.flat_cos(ro, rr)

        # Self-check: `span_perfect` must sit at ~0 on the objective it is the
        # argmin of. If it does not, the oracle is not an oracle.
        if b.get("span_pack"):
            cols["span_perfect_span_loss"] = torch.full(
                (len(idx),),
                float(loss_mod.loss_span(oracles["span_perfect"], t, b["span_pack"])),
                device=ref.device,
            )

        frac = (covered.sum(1).float() / b["s_mask"].sum(1).clamp_min(1)).tolist()
        for j, reg in enumerate(b["registers"]):
            for k, v in cols.items():
                acc[reg][k].append(float(v[j]))
            acc[reg]["span_covered_frac"].append(float(frac[j]))
            acc[reg]["n_student"].append(float(n_s[j]))
            acc[reg]["n_ref"].append(float(n_r[j]))

    def summarize(d):
        out = {k: sum(v) / len(v) for k, v in d.items() if v}
        out["n_pairs"] = len(d["teacher_vs_en"])
        return out

    per_reg = {r: summarize(d) for r, d in sorted(acc.items())}
    keys = next(iter(acc.values()))
    overall = summarize(
        {k: [x for d in acc.values() for x in d.get(k, [])] for k in keys}
    )
    return {"per_register": per_reg, "overall": overall}


def _table(g5: dict, gate: float) -> str:
    head = (
        "| register | n | span cov | teacher | native | span_perfect | "
        "span_plus | ref_remap | prefix_bound | span_perfect attn | "
        "teacher attn |\n"
    )
    head += "|" + "---|" * 11 + "\n"
    body = ""
    nan = float("nan")
    for reg, m in list(g5["per_register"].items()) + [("**all**", g5["overall"])]:
        body += (
            f"| {reg} | {int(m['n_pairs'])} | {m.get('span_covered_frac', nan):.2f} | "
            f"**{m['teacher_vs_en']:.3f}** | {m['native_vs_en']:.3f} | "
            f"**{m.get('span_perfect_vs_en', nan):.3f}** | "
            f"{m.get('span_plus_vs_en', nan):.3f} | "
            f"{m.get('ref_remap_vs_en', nan):.3f} | "
            f"{m.get('prefix_bound', nan):.3f} | "
            f"{m.get('span_perfect_vs_en_attn', nan):.3f} | "
            f"{m.get('teacher_vs_en_attn', nan):.3f} |\n"
        )
    body += (
        f"\nGate is `cos_vs_en >= {gate:.2f}`. Read the **span_perfect** column "
        "against it: that is what a student which has *exactly solved* the "
        "settled objective scores under 2c's own metric.\n"
    )
    return head + body


def main() -> None:
    parser = cfg_mod.build_argparser()
    parser.add_argument("--gate_cos", type=float, default=0.6)
    args = parser.parse_args()
    if args.eval_limit == 256:  # G5 wants the whole holdout, like G3
        args.eval_limit = 0
    cfg = cfg_mod.resolve_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    run_dir = make_run_dir("cjk_distill", cfg.label or "g5-flat-ceiling")
    logger.info("G5 %s", run_dir)
    t0 = time.time()

    ctx = distill.load_context(cfg, device, dtype)
    g5 = run_g5(ctx, cfg, device, dtype)
    tbl = _table(g5, args.gate_cos)
    (run_dir / "g5_flat_ceiling.md").write_text(tbl, encoding="utf-8")
    print("\n### G5 — what the objective's optimum scores on 2c's gate\n\n" + tbl)

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=cfg.label or "g5",
        metrics={
            "g5": g5,
            "gate_cos": args.gate_cos,
            "n_holdout": len(ctx["hold_cache"]) if ctx["hold_cache"] else 0,
            "wall_s": round(time.time() - t0, 1),
        },
        artifacts=["g5_flat_ceiling.md"],
    )
    logger.info("result.json → %s", run_dir)
    print(json.dumps({"run_dir": str(run_dir)}, indent=1))


if __name__ == "__main__":
    main()
