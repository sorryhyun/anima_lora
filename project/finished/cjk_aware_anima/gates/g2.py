"""G2 — the loss × parameterization cross-tab (Phase 2b).

Runs every arm to the same step budget on the same data and scores each one on
**every** metric, not just the objective it was trained on. That cross-tab is
the point: an arm that wins on its own loss and nowhere else has told you
nothing, and a flat cosine is exactly the kind of metric a flat-cosine
objective can win by memorizing.

Arms are run in-process against a single loaded cache and adapter — the 4.3 GB
cache read and the adapter load dominate a single arm's cost, so re-launching
per arm would triple the wall clock.

Decision rule (reported, not enforced): highest held-out **recovery** subject
to discrimination ≤ 0.2. Recovery, not raw cosine, because teacher and student
share the Qwen context and an untrained student already scores well above zero.

Usage (GPU work goes through the daemon)::

    make daemon-run ARGS="project/finished/cjk_aware_anima/gates/g2.py --steps 1500 --batch_size 32"
"""

from __future__ import annotations

import dataclasses
import logging
import random
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Staged, not a full grid. A loss does not need to be a *row* to be measured —
# every arm is already scored on all four losses as columns, so training `pool`
# would only answer "does optimizing the weakest control raise recovery?", which
# nobody asks. And the parameterizations only need comparing at the *winning*
# loss; the loss × param interaction of a loss that lost is not a question.
#
#   Stage A — which loss?           param fixed at `global` (no per-row freedom,
#                                   so a loss cannot win by memorizing rows)
#   Stage B — which parameterization?  loss fixed at Stage A's winner
#
# `row` is in Stage B because it is the one mode that cannot touch a row the
# corpus never visited — it is the control for the 5%-coverage question.
STAGE_A_LOSSES = ["flat:1.0", "span:1.0", "attn:1.0", "attn:1.0,span:0.5"]
STAGE_A_PARAM = "global"
STAGE_B_PARAMS = ["global_row", "row"]

FULL_GRID = [
    (p, loss)
    for p in ("global", "global_row")
    for loss in (*STAGE_A_LOSSES, "pool:1.0")
] + [("row", "span:1.0"), ("row", "attn:1.0")]

COLUMNS = ["flat", "span", "attn", "pool"]


def _table(rows: list[dict], baseline: dict) -> str:
    head = (
        "| stage/param | loss | recovery_attn | recovery_flat | cos(s,en) | cos(s,t) | disc far | disc near | "
        + " | ".join(f"held {c}" for c in COLUMNS)
        + " | rows | s |\n"
    )
    head += "|" + "---|" * (9 + len(COLUMNS) + 1) + "\n"
    body = ""
    if baseline:
        body += (
            f"| _(zero-shot)_ | — | {baseline['recovery_attn']:.3f} | "
            f"{baseline['recovery']:.3f} | "
            f"{baseline['cos_student_vs_en']:.3f} | "
            f"{baseline['cos_student_vs_teacher']:.3f} | "
            f"{baseline['discrimination_far']:.3f} | "
            f"{baseline['discrimination_near']:.3f} | "
            + " | ".join(
                f"{baseline['held_out_loss'].get(c, float('nan')):.4f}" for c in COLUMNS
            )
            + " | — | — |\n"
        )
    for r in rows:
        e = r["eval"]
        body += (
            f"| {r.get('stage', '')} {r['param']} | {r['loss']} | "
            f"**{e['recovery_attn']:.3f}** | "
            f"{e['recovery']:.3f} | "
            f"{e['cos_student_vs_en']:.3f} | {e['cos_student_vs_teacher']:.3f} | "
            f"{e['discrimination_far']:.3f} | "
            f"{e['discrimination_near']:.3f} | "
            + " | ".join(
                f"{e['held_out_loss'].get(c, float('nan')):.4f}" for c in COLUMNS
            )
            + f" | {r['n_tunable_rows']} | {r['wall_s']:.0f} |\n"
        )
    return head + body


def main() -> None:
    parser = cfg_mod.build_argparser()
    parser.add_argument(
        "--arms",
        default="",
        help="';'-separated 'param:loss' subset, overriding the staged plan. "
        "Semicolons, because a loss spec is itself comma-separated "
        '("global:attn:1.0,span:0.5")',
    )
    parser.add_argument(
        "--full_grid",
        action="store_true",
        help="run every param × loss cell (10 arms) instead of the staged 6",
    )
    args = parser.parse_args()
    base_cfg = cfg_mod.resolve_config(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    run_dir = make_run_dir("cjk_distill", base_cfg.label or "g2")
    ctx = distill.load_context(base_cfg, device, dtype)

    rows: list[dict] = []
    baseline: dict = {}
    t0 = time.time()

    def run_arm(param: str, loss: str, tag: str) -> dict:
        # Same seed per arm so the batch stream and the global-map init are
        # identical — the arm is the only free variable.
        torch.manual_seed(base_cfg.seed)
        random.seed(base_cfg.seed)
        cfg = dataclasses.replace(
            base_cfg,
            param=param,
            losses=cfg_mod.parse_losses(loss),
            label=f"g2-{param}-{loss}",
        )
        logger.info("── %s: param=%s loss=%s", tag, param, loss)
        metrics, _table_obj, _visits = distill.train_arm(cfg, ctx, device, dtype)
        nonlocal baseline
        if not baseline and metrics["history"]:
            baseline = metrics["history"][0]  # step-0 row, identical across arms
        row = {
            "stage": tag.split()[0],
            "param": param,
            "loss": loss,
            "eval": metrics["history"][-1],
            "first_loss": metrics["first_loss"],
            "final_loss": metrics["final_loss"],
            "n_tunable_rows": metrics["n_tunable_rows"],
            "wall_s": metrics["wall_s"],
            "history": metrics["history"],
        }
        rows.append(row)
        e = row["eval"]
        logger.info(
            "   → recovery_attn %.3f  cos(s,en) %.3f  disc far/near %.3f/%.3f  held: %s",
            e["recovery_attn"],
            e["cos_student_vs_en"],
            e["discrimination_far"],
            e["discrimination_near"],
            " ".join(f"{k}={v:.4f}" for k, v in sorted(e["held_out_loss"].items())),
        )
        return row

    def best(candidates: list[dict]) -> dict:
        # Gate on the FAR statistic only: `near` pairs are supposed to be
        # similar, so including them would punish correct behavior.
        ok = [r for r in candidates if r["eval"]["discrimination_far"] <= 0.2]
        # Rank on the readout-space recovery: the flat one structurally
        # favours whichever arm happens to match the reference's tokenization.
        return max(ok or candidates, key=lambda r: r["eval"]["recovery_attn"])

    if args.arms or args.full_grid:
        grid = FULL_GRID
        if args.arms:
            wanted = {a.strip() for a in args.arms.split(";") if a.strip()}
            grid = [(p, loss) for p, loss in FULL_GRID if f"{p}:{loss}" in wanted]
            if not grid:
                raise SystemExit(
                    f"--arms matched none of: {[f'{p}:{loss}' for p, loss in FULL_GRID]}"
                )
        logger.info("G2 %s — %d arms × %d steps", run_dir, len(grid), base_cfg.steps)
        for i, (param, loss) in enumerate(grid, 1):
            run_arm(param, loss, f"grid arm {i}/{len(grid)}")
        winner = best(rows)
    else:
        n = len(STAGE_A_LOSSES) + len(STAGE_B_PARAMS)
        logger.info("G2 %s — staged, %d arms × %d steps", run_dir, n, base_cfg.steps)
        for i, loss in enumerate(STAGE_A_LOSSES, 1):
            run_arm(STAGE_A_PARAM, loss, f"A arm {i}/{len(STAGE_A_LOSSES)}")
        loss_winner = best(rows)
        logger.info(
            "stage A winner: loss=%s (recovery_attn %.3f) → carrying it into stage B",
            loss_winner["loss"],
            loss_winner["eval"]["recovery_attn"],
        )
        for i, param in enumerate(STAGE_B_PARAMS, 1):
            run_arm(param, loss_winner["loss"], f"B arm {i}/{len(STAGE_B_PARAMS)}")
        winner = best(rows)
    table_md = _table(rows, baseline)
    (run_dir / "g2_table.md").write_text(table_md, encoding="utf-8")
    print("\n" + table_md)
    logger.info(
        "winner: param=%s loss=%s (recovery_attn %.3f)",
        winner["param"],
        winner["loss"],
        winner["eval"]["recovery_attn"],
    )

    write_result(
        run_dir,
        script=__file__,
        args=args,
        label=base_cfg.label or "g2",
        metrics={
            "arms": rows,
            "baseline_zero_shot": baseline,
            "winner": {"param": winner["param"], "loss": winner["loss"]},
            "design": "full_grid" if (args.arms or args.full_grid) else "staged",
            "discrimination_gate": {
                "far_max": 0.2,
                "near": "expected high, must be < 1",
            },
            "n_train_pairs": len(ctx["pool"]),
            "steps": base_cfg.steps,
            "batch_size": base_cfg.batch_size,
            "lr": base_cfg.lr,
            "trust": base_cfg.trust,
            "wall_s": round(time.time() - t0, 1),
        },
        artifacts=["g2_table.md"],
    )
    logger.info("result.json → %s", run_dir)


if __name__ == "__main__":
    main()
