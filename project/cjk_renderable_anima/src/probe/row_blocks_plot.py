"""Per-row trajectories of a ``--row_blocks`` run — small multiples off
``row_blocks_log.jsonl`` (written every step by the train stage): one panel
per ext row, the in-box residual (glyph term) and the row norm against the
block-local step, the other rows in light grey behind for the spread. Up to
two arms overlay per panel (e.g. warmup 0 vs 8).

    python src/probe/row_blocks_plot.py <arm dir> [<arm dir 2>] [--ema 5] [--out x.png]

No GPU; the tokenizer is loaded only to name the rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SERIES = ["#2a78d6", "#eb6834"]  # palette slots 1–2 (dataviz reference instance)
GREY = "#c9c8c2"
TEXT = "#52514e"


def load(arm: Path) -> dict[int, list[dict]]:
    rows: dict[int, list[dict]] = {}
    for ln in (arm / "row_blocks_log.jsonl").read_text().splitlines():
        r = json.loads(ln)
        rows.setdefault(r["ext"], []).append(r)
    return rows


def ema(xs, k):
    if k <= 1:
        return list(xs)
    out, a = [], 2.0 / (k + 1)
    for x in xs:
        out.append(x if not out else (1 - a) * out[-1] + a * x)
    return out


def names(ext_ids):
    from common.models import checkpoints
    from library.anima.vocab_pack import strategy_pack
    from library.inference.text import ensure_text_strategies
    from train.encoder import row_texts

    tok, _ = ensure_text_strategies(checkpoints().text_encoder, vocab_pack=None)
    return row_texts(tok, strategy_pack(tok), ext_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+", type=Path)
    ap.add_argument("--ema", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no_names", action="store_true")
    a = ap.parse_args()
    assert 1 <= len(a.arms) <= 2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC", "DejaVu Sans"]
    logs = [load(p) for p in a.arms]
    ext = sorted(logs[0])
    name = {} if a.no_names else names(ext)
    n = len(ext)
    cols = min(n, 6)
    rws = -(-n // cols)
    fig, axes = plt.subplots(
        2 * rws, cols, figsize=(2.6 * cols, 2.2 * 2 * rws), squeeze=False, sharex=True
    )
    for i, e in enumerate(ext):
        ax_l = axes[2 * (i // cols)][i % cols]
        ax_n = axes[2 * (i // cols) + 1][i % cols]
        # the spread: every other row of arm 0, grey, unsmoothed
        for e2 in ext:
            if e2 == e:
                continue
            rs = logs[0][e2]
            ax_l.plot(
                [r["local"] for r in rs],
                ema([r["in_box"] for r in rs], a.ema),
                color=GREY,
                lw=0.8,
                zorder=1,
            )
            ax_n.plot(
                [r["local"] for r in rs], [r["row_norm"] for r in rs], color=GREY, lw=0.8, zorder=1
            )
        for k, lg in enumerate(logs):
            rs = lg.get(e, [])
            if not rs:
                continue
            x = [r["local"] for r in rs]
            ax_l.plot(x, [r["in_box"] for r in rs], color=SERIES[k], lw=0.8, alpha=0.35, zorder=2)
            ax_l.plot(x, ema([r["in_box"] for r in rs], a.ema), color=SERIES[k], lw=2, zorder=3)
            ax_n.plot(x, [r["row_norm"] for r in rs], color=SERIES[k], lw=2, zorder=3)
        label = name.get(e, str(e))
        ax_l.set_title(f"{label}  ({e})", fontsize=10, color=TEXT, loc="left")
        for ax in (ax_l, ax_n):
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7, colors=TEXT)
            ax.grid(axis="y", color="#eeede8", lw=0.6)
        ax_l.set_ylabel("in-box MSE", fontsize=8, color=TEXT)
        ax_n.set_ylabel("row norm", fontsize=8, color=TEXT)
        ax_n.set_xlabel("block step", fontsize=8, color=TEXT)
    for j in range(n, rws * cols):
        axes[2 * (j // cols)][j % cols].axis("off")
        axes[2 * (j // cols) + 1][j % cols].axis("off")
    if len(a.arms) == 2:
        fig.legend(
            handles=[
                plt.Line2D([], [], color=SERIES[k], lw=2, label=p.name) for k, p in enumerate(a.arms)
            ],
            loc="upper right",
            fontsize=8,
            frameon=False,
        )
    fig.suptitle(
        " vs ".join(p.name for p in a.arms) + f"  — per row, EMA {a.ema} on the in-box term",
        fontsize=10,
        color=TEXT,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    out = a.out or (a.arms[-1] / "row_blocks.png")
    fig.savefig(out, dpi=110)
    print(out)
    # the numbers behind the picture: per row, in-box at block start / end, norm at end
    for k, lg in enumerate(logs):
        print(f"== {a.arms[k].name}")
        for e in ext:
            rs = lg.get(e, [])
            if not rs:
                continue
            s = ema([r["in_box"] for r in rs], a.ema)
            print(
                f"  {name.get(e, e):>3}  in-box {s[min(len(s)-1, a.ema)]:.4f} → {s[-1]:.4f}"
                f"  min {min(s):.4f}@{s.index(min(s)) + 1}  norm {rs[-1]['row_norm']:.0f}"
            )


if __name__ == "__main__":
    main()
