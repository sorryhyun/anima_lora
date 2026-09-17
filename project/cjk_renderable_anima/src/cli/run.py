"""``--stage`` / ``--arm`` / tags, and the generation settings every sampler stage reads."""

from __future__ import annotations


def run_args(g, stages):
    g.add_argument("--stage", nargs="+", default=["all"], choices=["all", *stages])
    g.add_argument("--arm", default="rows", choices=["rows", "rows_adapter", "encoder"])
    g.add_argument("--device", default="cuda")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument(
        "--data_tag",
        default="",
        help="suffix for output/wake_probe/data_<tag> and <arm>_<tag>",
    )
    g.add_argument(
        "--arm_tag",
        default="",
        help="suffix for the arm dir only (<arm>_<data_tag>_<arm_tag>): a second "
        "train recipe on the same data without overwriting the first",
    )


def generation_args(g):
    g.add_argument("--steps", type=int, default=28, help="inference steps")
    g.add_argument("--cfg", type=float, default=4.0)
    g.add_argument(
        "--negative",
        default="",
        help="target: negative prompt for the verbatim-caption renders (pair with --eval_tag)",
    )
    g.add_argument(
        "--seeds", type=int, default=2, help="seeds per prompt (salad: 3 recommended)"
    )
    g.add_argument("--salad_size", type=int, default=768)
    g.add_argument("--train_size", type=int, default=512)
    g.add_argument("--eval_size", type=int, default=512)
    g.add_argument(
        "--eval_shape",
        default="",
        help="eval: WxH canvas instead of --eval_size² (e.g. 384x512); pair with --eval_tag",
    )
