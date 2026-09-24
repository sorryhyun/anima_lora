"""G3 + G4 — the teacher ceiling per register, and corpus health (Phase 2b).

The last two Phase-2b gates. G0b/G0/G1/G2 answered *can the loop run* and
*which objective*; these two answer **what is the corpus actually worth**, and
they are the numbers a corpus-scaling decision (finish the D2 MT pass? widen
D1?) should be read off.

``G3`` **Teacher ceiling, per register.** The Phase-2c gate is written as
      `cos_vs_en >= 0.6` — a number lifted from the Phase-0 probe, where the
      *teacher* measured 0.69/0.77 on two hand-written prompts. That is one
      register (composed tags) and two samples. Distillation cannot pass its
      own teacher, so the honest gate is a fraction of the ceiling **in the
      register being scored**, and this measures that ceiling on all 900
      held-out pairs, split by register, in both metric spaces. It also reports
      the floor (`native`, today's unk-wall) — the *addressable* gap is the
      difference, and a register where the two nearly coincide has nothing to
      distil no matter how many pairs it contributes.

``G4a`` **Corpus health** (CPU only). Token-count ratio JA-student /
      EN-teacher per register — the L_flat position-alignment risk made
      concrete — plus occurrence-weighted span provenance and the ext-row
      visit histogram. `mt_unverified` at 34% of *occurrences* is the number
      `plan.md` flags as the open 2a blocker; this is where it is recomputed
      from the cache that training actually reads.

``G4b`` **Trust ablation** — `all` / `provenance` / `verified_only` at the
      settled `param=global, loss=span`. Does demoting the unverified
      translations help, or is the noise averaged out? Held-out scoring uses
      the **cached** (`provenance`) weighting for every arm, so no arm is
      graded on its own weighting — the same cross-tab discipline as G2.

Usage (GPU work goes through the daemon)::

    make daemon-run ARGS="project/finished/cjk_aware_anima/gates/g34.py"
    make daemon-run ARGS="project/finished/cjk_aware_anima/gates/g34.py --gates g3"
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import random
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bench._common import make_run_dir, write_result  # noqa: E402
from scripts.distill_cjk import config as cfg_mod, ext_table  # noqa: E402
from scripts.distill_cjk import distill  # noqa: E402
from scripts.distill_cjk import losses as loss_mod  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Trust policies compared by G4b, in escalating strictness. The ablation only
# has a handle on the span-carrying registers (`tags`/`tags_alt`, the default
# `--train_registers`) — prose has no tag alignment and so no trust weight.
TRUST_ARMS = ["all", "provenance", "verified_only"]


# ---------------------------------------------------------------------------
# G3 — teacher ceiling per register
# ---------------------------------------------------------------------------


@torch.no_grad()
def ceiling_by_register(ctx, cfg, device, dtype, batch_size: int = 32) -> dict:
    """Per-register ceiling / floor / zero-shot start, in both metric spaces.

    Nothing here is trained. Teacher, reference and native arms are all *cached*
    adapter outputs, so the only forward is the untrained student's — which is
    included because a ceiling without the starting point does not say how much
    of the gap is addressable *from here*.
    """
    hold = ctx["hold_cache"]
    probes = ctx["probes"]
    adapter = ctx["adapter"]
    if hold is None:
        raise SystemExit("G3 needs a cached holdout split")

    acc: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for start in range(0, len(hold), batch_size):
        idx = list(range(start, min(start + batch_size, len(hold))))
        if len(idx) < 2:  # flat_cos is per-sample, but discrimination is not
            break
        b = hold.batch(idx, device, dtype)
        s = distill.student_forward(adapter, b)
        t = loss_mod.zero_pads(b["teacher"], b["teacher_mask"])
        ref = loss_mod.zero_pads(b["ref"], b["ref_mask"])
        nat = loss_mod.zero_pads(b["native"], b["native_mask"])

        cols = {
            "teacher_vs_en": distill.flat_cos(t, ref),
            "native_vs_en": distill.flat_cos(nat, ref),
            "student0_vs_en": distill.flat_cos(s, ref),
            "student0_vs_teacher": distill.flat_cos(s, t),
        }
        if probes:
            rs, rt, rr, rn = (
                distill.readout_vec(x, m, probes)
                for x, m in (
                    (s, b["s_mask"]),
                    (t, b["teacher_mask"]),
                    (ref, b["ref_mask"]),
                    (nat, b["native_mask"]),
                )
            )
            cols.update(
                teacher_vs_en_attn=distill.flat_cos(rt, rr),
                native_vs_en_attn=distill.flat_cos(rn, rr),
                student0_vs_en_attn=distill.flat_cos(rs, rr),
            )
        n_s = b["s_mask"].sum(1).tolist()
        n_t = b["teacher_mask"].sum(1).tolist()
        for j, reg in enumerate(b["registers"]):
            for k, v in cols.items():
                acc[reg][k].append(float(v[j]))
            acc[reg]["n_student"].append(float(n_s[j]))
            acc[reg]["n_teacher"].append(float(n_t[j]))

    def summarize(d: dict[str, list[float]]) -> dict:
        m = {k: sum(v) / len(v) for k, v in d.items() if v}
        out = dict(m)
        out["n_pairs"] = len(d["teacher_vs_en"])
        # The gap distillation is even allowed to close, per space.
        out["addressable"] = m["teacher_vs_en"] - m["native_vs_en"]
        if "teacher_vs_en_attn" in m:
            out["addressable_attn"] = m["teacher_vs_en_attn"] - m["native_vs_en_attn"]
        return out

    per_reg = {r: summarize(d) for r, d in sorted(acc.items())}
    overall = summarize(
        {k: [x for d in acc.values() for x in d[k]] for k in next(iter(acc.values()))}
    )
    return {"per_register": per_reg, "overall": overall}


def _g3_table(g3: dict, gate: float) -> str:
    head = (
        "| register | n | teacher (ceiling) | native (floor) | addressable | "
        "zero-shot student | ceiling attn | floor attn | addressable attn | "
        f"gate {gate:.2f} / ceiling |\n"
    )
    head += "|" + "---|" * 10 + "\n"
    body = ""
    for reg, m in list(g3["per_register"].items()) + [("**all**", g3["overall"])]:
        ratio = gate / m["teacher_vs_en"] if m["teacher_vs_en"] > 1e-6 else float("nan")
        body += (
            f"| {reg} | {int(m['n_pairs'])} | **{m['teacher_vs_en']:.3f}** | "
            f"{m['native_vs_en']:.3f} | {m['addressable']:.3f} | "
            f"{m['student0_vs_en']:.3f} | "
            f"{m.get('teacher_vs_en_attn', float('nan')):.3f} | "
            f"{m.get('native_vs_en_attn', float('nan')):.3f} | "
            f"{m.get('addressable_attn', float('nan')):.3f} | "
            f"{ratio:.2f} |\n"
        )
    return head + body


# ---------------------------------------------------------------------------
# G4a — corpus health (no GPU)
# ---------------------------------------------------------------------------


def corpus_health(cfg, ctx) -> dict:
    """Token-count ratio, occurrence-weighted provenance, ext-row visit bands."""
    caches = {"train": ctx["train_cache"], "holdout": ctx["hold_cache"]}
    per_reg: dict[str, dict] = {}
    via_occ: collections.Counter = collections.Counter()
    via_spans: collections.Counter = collections.Counter()

    for split, cache in caches.items():
        if cache is None:
            continue
        for rec in cache.records:
            reg = rec["register"]
            d = per_reg.setdefault(
                reg,
                {
                    "n_train": 0,
                    "n_holdout": 0,
                    "ratios": [],
                    "n_with_spans": 0,
                    "n_span_tokens": 0,
                    "n_student": [],
                    "n_teacher": [],
                },
            )
            d[f"n_{split}"] += 1
            nt, ns = int(rec["n_teacher"]), int(rec["n_student"])
            d["n_student"].append(ns)
            d["n_teacher"].append(nt)
            if nt:
                d["ratios"].append(ns / nt)
            spans = rec.get("spans") or []
            if spans:
                d["n_with_spans"] += 1
            for s in spans:
                via_occ[str(s[3])] += len(s[1])
                via_spans[str(s[3])] += 1
                d["n_span_tokens"] += len(s[1])

    for d in per_reg.values():
        d["token_ratio_mean"] = statistics.fmean(d["ratios"]) if d["ratios"] else 0.0
        d["token_ratio_median"] = statistics.median(d["ratios"]) if d["ratios"] else 0.0
        d["n_student_mean"] = statistics.fmean(d["n_student"]) if d["n_student"] else 0
        d["n_teacher_mean"] = statistics.fmean(d["n_teacher"]) if d["n_teacher"] else 0
        for k in ("ratios", "n_student", "n_teacher"):
            d.pop(k)

    tot = sum(via_occ.values()) or 1
    provenance = {
        via: {
            "occurrences": n,
            "share": n / tot,
            "spans": via_spans[via],
            "weight_provenance": cfg_mod.TRUST_POLICIES["provenance"].get(via, 1.0),
        }
        for via, n in via_occ.most_common()
    }

    # Visit bands over the *training* pool actually used by the settled arm.
    visits = ctx["visits"]
    bands = {
        "0": int((visits == 0).sum()),
        "1-4": int(((visits >= 1) & (visits <= 4)).sum()),
        "5-49": int(((visits >= 5) & (visits <= 49)).sum()),
        "50+": int((visits >= 50).sum()),
    }
    rows_total = int(visits.numel())
    return {
        "per_register": dict(sorted(per_reg.items())),
        "provenance_occurrence_weighted": provenance,
        "unverified_occurrence_share": provenance.get("mt_unverified", {}).get(
            "share", 0.0
        ),
        "ext_rows": {
            "total": rows_total,
            "visited": rows_total - bands["0"],
            "visited_share": (rows_total - bands["0"]) / max(rows_total, 1),
            "bands": bands,
            "pool_registers": list(cfg.train_registers) or ["ALL"],
        },
    }


def _g4a_tables(g4a: dict) -> str:
    out = "| register | train | holdout | mean JA tok | mean EN tok | JA/EN ratio (mean/median) | pairs w/ spans | span tokens |\n"
    out += "|" + "---|" * 8 + "\n"
    for reg, d in g4a["per_register"].items():
        out += (
            f"| {reg} | {d['n_train']} | {d['n_holdout']} | {d['n_student_mean']:.1f} | "
            f"{d['n_teacher_mean']:.1f} | {d['token_ratio_mean']:.2f} / "
            f"{d['token_ratio_median']:.2f} | {d['n_with_spans']} | "
            f"{d['n_span_tokens']} |\n"
        )
    out += "\n| via | span tokens | share | spans | weight (provenance) |\n"
    out += "|" + "---|" * 5 + "\n"
    for via, d in g4a["provenance_occurrence_weighted"].items():
        out += (
            f"| {via} | {d['occurrences']} | {100 * d['share']:.1f}% | {d['spans']} | "
            f"{d['weight_provenance']} |\n"
        )
    e = g4a["ext_rows"]
    out += (
        f"\next rows: {e['visited']}/{e['total']} visited "
        f"({100 * e['visited_share']:.2f}%) over the "
        f"{','.join(e['pool_registers'])} pool — bands "
        + ", ".join(f"{k}: {v}" for k, v in e["bands"].items())
        + "\n"
    )
    return out


# ---------------------------------------------------------------------------
# G4b — trust ablation
# ---------------------------------------------------------------------------


def _g4b_table(rows: list[dict], baseline: dict) -> str:
    cols = ["span", "attn", "flat", "pool"]
    head = (
        "| trust | spans kept | recovery_attn | cos(s,en) tags | cos(s,en) all | "
        "disc far | disc near | " + " | ".join(f"held {c}" for c in cols) + " |\n"
    )
    head += "|" + "---|" * (7 + len(cols)) + "\n"
    body = ""

    def line(name: str, e: dict, kept: str) -> str:
        by_reg = e.get("cos_student_vs_en_by_register", {})
        return (
            f"| {name} | {kept} | {e['recovery_attn']:.3f} | "
            f"{by_reg.get('tags', float('nan')):.3f} | "
            f"{e['cos_student_vs_en']:.3f} | "
            f"{e['discrimination_far']:.3f} | {e['discrimination_near']:.3f} | "
            + " | ".join(f"{e['held_out_loss'].get(c, float('nan')):.4f}" for c in cols)
            + " |\n"
        )

    if baseline:
        body += line("_(zero-shot)_", baseline, "—")
    for r in rows:
        body += line(r["trust"], r["eval"], f"{r['n_spans_kept']}")
    return head + body


def count_spans(cache, pool) -> int:
    return sum(len(cache.records[i].get("spans") or []) for i in pool)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = cfg_mod.build_argparser()
    parser.add_argument(
        "--gates",
        default="g3,g4a,g4b",
        help="comma-separated subset of g3,g4a,g4b",
    )
    parser.add_argument(
        "--gate_cos",
        type=float,
        default=0.6,
        help="the Phase-2c cos_vs_en gate, reported against each register's ceiling",
    )
    args = parser.parse_args()
    # G3/G4 read the whole holdout: a 256-record prefix leaves `tags` at ~40
    # pairs and the register decomposition is the entire point.
    if args.eval_limit == 256:
        args.eval_limit = 0
    base_cfg = cfg_mod.resolve_config(args)
    gates = {g.strip() for g in args.gates.split(",") if g.strip()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    # Name the run dir after the *gates*, not just "g34". Two gate subsets are
    # usually launched back to back (g3/g4a take seconds, g4b takes 15 min), and
    # `make_run_dir` is minute-stamped with `exist_ok=True` — so a plain "g34"
    # label puts the second run in the first's directory and its `write_result`
    # silently replaces the first's `result.json`. Note `--label` cannot rescue
    # this from the daemon: `make daemon-run` consumes `--label` for the job
    # record, so it never reaches this argparser.
    run_dir = make_run_dir(
        "cjk_distill", base_cfg.label or "g34-" + "-".join(sorted(gates))
    )
    logger.info("G3/G4 %s — gates %s", run_dir, ",".join(sorted(gates)))
    t0 = time.time()

    ctx = distill.load_context(base_cfg, device, dtype)
    # A zero-shot table so the student ids resolve; `up` is zero-init, so this
    # is exactly the Phase-1 table (ExtTable docstring).
    table, _ = distill.build_table(
        base_cfg,
        ctx["train_cache"],
        ctx["pool"],
        ctx["ext_init"],
        device,
        visits=ctx["visits"],
    )
    ext_table.attach(ctx["adapter"], table)

    metrics: dict = {"gates": sorted(gates)}
    artifacts: list[str] = []

    if "g3" in gates:
        logger.info(
            "── G3: teacher ceiling per register (%d held-out pairs)",
            len(ctx["hold_cache"]),
        )
        g3 = ceiling_by_register(ctx, base_cfg, device, dtype)
        metrics["g3"] = g3
        tbl = _g3_table(g3, args.gate_cos)
        (run_dir / "g3_ceiling.md").write_text(tbl, encoding="utf-8")
        artifacts.append("g3_ceiling.md")
        print("\n### G3 — teacher ceiling per register\n\n" + tbl)

    if "g4a" in gates:
        logger.info("── G4a: corpus health")
        g4a = corpus_health(base_cfg, ctx)
        metrics["g4a"] = g4a
        tbl = _g4a_tables(g4a)
        (run_dir / "g4a_corpus.md").write_text(tbl, encoding="utf-8")
        artifacts.append("g4a_corpus.md")
        print("\n### G4a — corpus health\n\n" + tbl)

    if "g4b" in gates:
        rows: list[dict] = []
        baseline: dict = {}
        train_cache = ctx["train_cache"]
        for i, trust in enumerate(TRUST_ARMS, 1):
            # Same seed per arm: batch stream and global-map init identical, so
            # the trust policy is the only free variable.
            torch.manual_seed(base_cfg.seed)
            random.seed(base_cfg.seed)
            train_cache.apply_trust(cfg_mod.TRUST_POLICIES[trust])
            kept = count_spans(train_cache, ctx["pool"])
            logger.info(
                "── G4b arm %d/%d: trust=%s (%d spans kept in the pool)",
                i,
                len(TRUST_ARMS),
                trust,
                kept,
            )
            cfg = dataclasses.replace(
                base_cfg,
                param="global",
                losses={"span": 1.0},
                trust=trust,
                label=f"g4b-{trust}",
            )
            m, _tbl, _v = distill.train_arm(cfg, ctx, device, dtype)
            if not baseline and m["history"]:
                baseline = m["history"][0]
            rows.append(
                {
                    "trust": trust,
                    "n_spans_kept": kept,
                    "eval": m["history"][-1],
                    "final_loss": m["final_loss"],
                    "wall_s": m["wall_s"],
                    "history": m["history"],
                }
            )
            e = rows[-1]["eval"]
            logger.info(
                "   → recovery_attn %.3f  cos(s,en) tags %.3f  held span %.4f",
                e["recovery_attn"],
                e.get("cos_student_vs_en_by_register", {}).get("tags", float("nan")),
                e["held_out_loss"].get("span", float("nan")),
            )
        metrics["g4b"] = {
            "arms": rows,
            "baseline_zero_shot": baseline,
            "eval_weighting": "provenance (as cached) for every arm — no arm is "
            "graded on its own weighting",
            "note": "the cache was built under `provenance`, whose zeros "
            "(unresolved, unmapped) were dropped at build time and cannot be "
            "restored here — `all` means every *mappable* span at 1.0",
        }
        tbl = _g4b_table(rows, baseline)
        (run_dir / "g4b_trust.md").write_text(tbl, encoding="utf-8")
        artifacts.append("g4b_trust.md")
        print("\n### G4b — trust ablation\n\n" + tbl)
        # Restore the cached weighting so a later stage in the same process is
        # not silently reading the last arm's policy.
        train_cache.apply_trust(cfg_mod.TRUST_POLICIES["provenance"])

    metrics.update(
        n_train_pairs=len(ctx["pool"]),
        n_holdout=len(ctx["hold_cache"]) if ctx["hold_cache"] else 0,
        train_registers=list(base_cfg.train_registers),
        steps=base_cfg.steps,
        batch_size=base_cfg.batch_size,
        lr=base_cfg.lr,
        gate_cos=args.gate_cos,
        wall_s=round(time.time() - t0, 1),
    )
    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=base_cfg.label or "g34",
        metrics=metrics,
        artifacts=artifacts,
    )
    logger.info("result.json → %s", run_dir)
    print(json.dumps({"run_dir": str(run_dir)}, indent=1))


if __name__ == "__main__":
    main()
