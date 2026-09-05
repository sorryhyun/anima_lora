#!/usr/bin/env python3
"""Text-binding probe: can the DiT bind text pixels to an ext-row key sequence?

Train a plain LoRA on ONE sincos image for ~100 steps with its OCR line in the
caption (encoded through a CJK ext pack), then render prompt variants and ask
whether the text region survives when the *other* tags change. This measures
job (2) of findings §9 — the ext rows as stable, separable addresses — which
no readout-cos metric sees. Dreambooth-style: one image, vary the rest.

Conditions rendered per (arm, seed), all composed through the caption grammar
from the mirror caption (``v0``):

    same        the training caption verbatim (memorization sanity)
    swap        visual tags swapped (hair/eyes/clothes/background), text kept
    drop_quote  「…」 tags removed, ``japanese text`` kept (presence, no address)
    drop_all    ``japanese text`` removed too (= the original caption)
    other_text  「…」 replaced by a different JA line (address generalization)
    swap_drop   swap + drop_quote (leakage under distribution shift)

Render arms: ``ja_ext`` (T5 side through the pack — the trained-on encoding)
and ``ja_native`` (stock T5, 「…」 → <unk>: the no-address render of the same
LoRA). ``--with_base`` also renders the pack without the LoRA.

Stages run as direct subprocesses (never nested daemon jobs). Launch::

    make daemon-run ARGS="--label textbind-trained-9095721 --stall-timeout 0 \
        project/cjk_aware_anima/probes/text_bind_probe.py --stems 9095721 \
        --arm trained --ext_prefix output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256 \
        --queue"

Arms (``--arm`` is only a name; ``--ext_prefix`` / ``--ocr_format`` decide):
``trained`` = distilled pack, ``init`` = ``bench/cjk_adapter/assets/ext_embed``
(anchor init, no distill), ``presence`` = ``--ocr_format presence`` (no
address at all; rendered through the same pack for comparability).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
PY = sys.executable

BASE_DIR = Path("post_image_dataset/cjk_unmask/textbind")
METHODS_DIR = REPO / "configs" / "gui-methods" / "custom"
RESIZED = REPO / "post_image_dataset" / "resized" / "sincos"
LATENTS = REPO / "post_image_dataset" / "lora" / "sincos"
RECORDS = "post_image_dataset/cjk_unmask/ocr_records_sincos_ppocr_v2.jsonl"

# Per-stem visual swaps (training caption tag -> replacement) and the
# "other text" line. Both are plain flat-tag edits through the grammar.
SWAPS: dict[str, dict[str, str]] = {
    "9095721": {
        "brown hair": "blue hair",
        "purple eyes": "green eyes",
        "black jacket": "red jacket",
        "blue skirt": "black skirt",
        "simple background": "outdoors",
        "white background": "beach",
    },
    "11883943": {
        "pink shirt": "blue shirt",
        "twintails": "ponytail",
        "simple background": "indoors",
        "white background": "classroom",
        "gold bikini": "red bikini",
    },
    "6067089": {
        "light brown hair": "black hair",
        "blue skirt": "red skirt",
        "striped skirt": "plaid skirt",
        "yellow bow": "blue bow",
        "yellow bowtie": "blue bowtie",
        "simple background": "outdoors",
        "white background": "park",
    },
}
OTHER_TEXT: dict[str, str] = {
    "9095721": "明日も晴れるかな",
    "11883943": "本当にいいんですか！？",
    "6067089": "また明日ね",
}

QUOTE_RE = re.compile(r"^「.*」$")


def run(stage: str, argv: list[str]) -> None:
    print(f"\n=== [{stage}] {' '.join(argv)}", flush=True)
    subprocess.run(argv, cwd=REPO, check=True)


def bucket_hw(stem: str) -> tuple[int, int]:
    """(H, W) of the cached latent — the size the LoRA saw."""
    hits = sorted(LATENTS.glob(f"{stem}_*_anima.npz"))
    if not hits:
        sys.exit(f"no cached latent for {stem} under {LATENTS}")
    m = re.search(r"_(\d+)x(\d+)_anima\.npz$", hits[0].name)
    assert m, hits[0]
    w, h = int(m.group(1)), int(m.group(2))
    return h, w


def build_conditions(caption: str, stem: str, swaps: dict[str, str], other: str):
    from anime_tools.captions.position_clauses import compose_caption, parse_caption

    parsed = parse_caption(caption)
    tags = list(parsed.flat_tags)
    quotes = [t for t in tags if QUOTE_RE.match(t)]
    if not quotes:
        sys.exit(f"mirror caption for {stem} carries no 「…」 tag: {caption}")
    gt_lines = [q[1:-1] for q in quotes]
    gt = "".join(gt_lines)

    def compose(ts):
        return compose_caption(tuple(ts), parsed.clauses)

    def swap(ts):
        missing = [k for k in swaps if k not in ts]
        if missing:
            print(f"WARN swap tags absent from caption {stem}: {missing}")
        return [swaps.get(t, t) for t in ts]

    def drop_quote(ts):
        return [t for t in ts if not QUOTE_RE.match(t)]

    def drop_all(ts):
        return [t for t in drop_quote(ts) if t != "japanese text"]

    def other_text(ts):
        out, done = [], False
        for t in ts:
            if QUOTE_RE.match(t):
                if not done:
                    out.append(f"「{other}」")
                    done = True
                continue
            out.append(t)
        return out

    conds = {
        "same": compose(tags),
        "swap": compose(swap(tags)),
        "drop_quote": compose(drop_quote(tags)),
        "drop_all": compose(drop_all(tags)),
        "other_text": compose(other_text(tags)),
        "swap_drop": compose(drop_quote(swap(tags))),
    }
    prompts = {k: {"en": v, "ja": v} for k, v in conds.items()}
    prompts["_gt"] = gt
    prompts["_gt_lines"] = gt_lines
    prompts["_other"] = other
    prompts["_stem"] = stem
    # judge hint: which conditions should show the GT line vs nothing
    prompts["_expect"] = {
        "same": "gt",
        "swap": "gt",
        "drop_quote": "none",
        "drop_all": "none",
        "other_text": "other",
        "swap_drop": "none",
    }
    return prompts


def write_method_toml(
    name: str, mirror: Path, te_dir: Path, opts, repeats: int
) -> Path:
    path = METHODS_DIR / f"{name}.toml"
    path.write_text(
        f"""# text-binding probe arm — auto-generated by probes/text_bind_probe.py
# images {opts.stems}, ~{opts.steps} steps (num_repeats {repeats}), lr {opts.lr}, dim {opts.dim}
network_dim = {opts.dim}
network_alpha = {opts.dim}
learning_rate = {opts.lr}
max_train_epochs = 1
save_every_n_epochs = 1
checkpointing_epochs = 0
use_shuffled_caption_variants = true
cache_llm_adapter_outputs = true
caption_dropout_rate = 0.0
output_name = "{name}"
blocks_to_swap = 0
use_cmmd = false
masked_loss = false
sigma_lowres = false

[general]
caption_extension = '.txt'

[[datasets]]
batch_size = 1
validation_split_num = 0
validation_seed = 42

  [[datasets.subsets]]
  image_dir = '{mirror.as_posix()}'
  cache_dir = '{LATENTS.relative_to(REPO).as_posix()}'
  latent_cache_dir = '{LATENTS.relative_to(REPO).as_posix()}'
  text_cache_dir = '{te_dir.as_posix()}'
  num_repeats = {repeats}
""",
        encoding="utf-8",
    )
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stems",
        required=True,
        help="comma-separated image stems (one LoRA over all of them); "
        f"known: {sorted(SWAPS)}",
    )
    ap.add_argument(
        "--tag", default=None, help="run tag (default: stems joined by '_')"
    )
    ap.add_argument("--arm", required=True, help="arm name (trained/init/presence/…)")
    ap.add_argument(
        "--ext_prefix",
        default="output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256",
        help="pack prefix for BOTH the TE cache and the render",
    )
    ap.add_argument("--ocr_format", default="tags", choices=("tags", "presence"))
    ap.add_argument("--records", default=RECORDS)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1234])
    ap.add_argument("--infer_steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument(
        "--with_base", action="store_true", help="also render the pack without the LoRA"
    )
    ap.add_argument("--skip_cache", action="store_true")
    ap.add_argument("--skip_train", action="store_true")
    ap.add_argument("--skip_render", action="store_true")
    opts = ap.parse_args()
    stems = [x.strip() for x in opts.stems.split(",") if x.strip()]
    unknown = [x for x in stems if x not in SWAPS]
    if unknown:
        sys.exit(f"no SWAPS/OTHER_TEXT entry for {unknown}")
    tag = opts.tag or "_".join(stems)
    repeats = -(-opts.steps // len(stems))  # ceil: ~steps total optimizer steps

    name = f"textbind_{opts.arm}_{tag}"
    mirror = BASE_DIR / f"mirror_{opts.arm}_{tag}"
    te_dir = BASE_DIR / f"te_{opts.arm}_{tag}"
    pack = REPO / f"{opts.ext_prefix}.safetensors"
    if not pack.exists():
        sys.exit(f"ext pack missing: {pack}")

    if not opts.skip_cache:
        run(
            "cache",
            [
                PY,
                str(PROJ / "datasets" / "cache_te_ext.py"),
                "--shard",
                "sincos",
                "--stems",
                ",".join(stems),
                "--records",
                opts.records,
                "--mirror",
                str(mirror),
                "--ext_prefix",
                opts.ext_prefix,
                "--out",
                str(te_dir),
                "--ocr_format",
                opts.ocr_format,
                "--overwrite",
            ],
        )

    # Conditions come from the *tags*-format caption so the 「…」 exists even
    # for the presence arm (whose training caption has none).

    sys.path.insert(0, str(PROJ / "datasets"))
    from cache_te_ext import append_tags, ocr_lines_by_stem  # noqa: E402

    all_lines = ocr_lines_by_stem(REPO / opts.records, 8)
    assets = PROJ / "assets"
    prompt_paths: dict[str, Path] = {}
    for stem in stems:
        base_caption = (RESIZED / f"{stem}.txt").read_text(encoding="utf-8").strip()
        tagged = append_tags(base_caption, all_lines[stem], "tags")
        prompts = build_conditions(tagged, stem, SWAPS[stem], OTHER_TEXT[stem])
        prompts["_arm"] = opts.arm
        cap = REPO / mirror / f"{stem}.txt"
        prompts["_train_caption"] = (
            cap.read_text(encoding="utf-8").strip() if cap.exists() else None
        )
        prompt_paths[stem] = assets / f"text_bind_prompts_{stem}.json"
        prompt_paths[stem].write_text(
            json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"conditions -> {prompt_paths[stem]}\n  gt={prompts['_gt']!r}")

    if not opts.skip_train:
        toml = write_method_toml(name, mirror, te_dir, opts, repeats)
        print(f"method toml -> {toml}")
        run(
            "train",
            [
                PY,
                "train.py",
                "--method",
                name,
                "--preset",
                "default",
                "--methods_subdir",
                "gui-methods/custom",
                "--path_pattern",
                "|".join(f"{s}.*" for s in stems),
            ],
        )

    if opts.skip_render:
        return
    lora = f"output/ckpt/{name}.safetensors"
    for stem in stems:
        h, w = bucket_hw(stem)
        common = [
            PY,
            "bench/cjk_adapter/run_bench.py",
            "--ext",
            "--ext_prefix",
            opts.ext_prefix,
            "--arms",
            "ja_ext,ja_native",
            "--languages",
            "ja",
            "--prompts",
            str(prompt_paths[stem]),
            "--size",
            str(h),
            str(w),
            "--steps",
            str(opts.infer_steps),
            "--cfg",
            str(opts.cfg),
        ]
        for seed in opts.seeds:
            run(
                f"render {stem} s{seed}",
                common
                + [
                    "--lora",
                    lora,
                    "--seed",
                    str(seed),
                    "--label",
                    f"textbind-{opts.arm}-{tag}-{stem}-s{seed}",
                ],
            )
        if opts.with_base:
            for seed in opts.seeds:
                run(
                    f"render base {stem} s{seed}",
                    common
                    + ["--seed", str(seed), "--label", f"textbind-base-{stem}-s{seed}"],
                )
    print("\n=== done:", lora, flush=True)


if __name__ == "__main__":
    main()
