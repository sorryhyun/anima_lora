#!/usr/bin/env python
"""Tag-dropout eval harness — render prompt sets per arm, score with the tag judge.

The reusable render+score half of the tag-dropout proposal
(``_archive/proposals/tag_dropout_mechanism.md``). Consumes a ``prompt_sets.json``
(from :mod:`bench.tag_dropout.prompt_sets`) and one or more LoRA arms, renders
every (prompt, seed) cell with matched noise across arms, and scores three
group-relative content-adherence quantities via the frozen tag-readback judge
(``anime_tools.tagger.readback``):

* **richness** = readback(render, characteristic_set ∖ prompted) on the *sparse*
  and *no_trigger* prompts — did characteristic content appear unprompted?
* **adherence** = readback(render, prompted) on *detailed* prompts — did the
  render honor the tags it WAS given?
* **leakage** = readback(render, omitted) on *omission* prompts — did content
  survive removing its tag from the prompt (the controllability cost)?

Everything is matched-seed paired across arms (the turbo checkpoint-spread lesson
— never compare arms unpaired). Both aggregations (logsigmoid + calibrated
recall) are reported; the gate reads logsigmoid.

**Phase 0** (harness sanity, existing checkpoint) — a multi-artist LoRA vs base::

    uv run python bench/tag_dropout/run_eval.py \
        --prompt_sets bench/tag_dropout/prompt_sets_mignon.json \
        --lora artist_lora:output/ckpt/anima_soup_shard1of6.safetensors \
        --include_base --seeds 42 43 --label phase0

Gate (sanity only, cannot attribute the dropout axis — every existing ckpt is
dropout-trained): richness(lora) > richness(base) on sparse (paired win-rate),
and the readback instrument discriminates on generated images (adherence to the
true prompt beats a shuffled other prompt).

**Phase 1** (the A/B) — same harness, two dropout arms, drop ``--include_base``::

    uv run python bench/tag_dropout/run_eval.py \
        --prompt_sets bench/tag_dropout/prompt_sets_mignon.json \
        --lora shuffle_only:output/ckpt/anima_td_shuffle_only.safetensors \
        --lora dropout01:output/ckpt/anima_td_dropout01.safetensors \
        --seeds 42 43 44 45 --label phase1
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
VAE = "models/vae/qwen_image_vae.safetensors"
TEXT_ENCODER = "models/text_encoders/qwen_3_06b_base.safetensors"

# families that carry each score
RICHNESS_FAMILIES = ("sparse", "no_trigger")
ADHERENCE_FAMILIES = ("detailed",)
LEAKAGE_FAMILIES = ("omission",)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--prompt_sets", required=True, help="prompt_sets.json from prompt_sets.py."
    )
    p.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="NAME:PATH",
        help="An arm 'name:relpath'. Repeatable. Path relative to repo root.",
    )
    p.add_argument(
        "--include_base", action="store_true", help="Add a no-LoRA 'base' arm."
    )
    p.add_argument(
        "--model_dir",
        default="models/captioners/anima-tagger-dbv4",
        help="Tagger checkpoint dir (the frozen judge).",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--infer_steps", type=int, default=24)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument(
        "--gate_richness_winrate",
        type=float,
        default=0.60,
        help="Phase 0: paired richness win-rate (arm0 vs base) on sparse to pass sanity.",
    )
    p.add_argument(
        "--gate_shuffle_winrate",
        type=float,
        default=0.90,
        help="Readback instrument: adherence(true) > adherence(shuffled) win-rate.",
    )
    p.add_argument("--label", default=None)
    p.add_argument(
        "--reuse_renders",
        default=None,
        help="Skip rendering; score an existing render_manifest.json instead "
        "(e.g. results/<run>/renders/render_manifest.json).",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Rendering (mirrors bench/readback/render_turbo.py — matched noise across arms)
# --------------------------------------------------------------------------- #
def _flat_cells(prompt_sets: dict) -> List[Tuple[str, str, str]]:
    """Flatten prompt families → list of (family, prompt_id, text)."""
    cells: List[Tuple[str, str, str]] = []
    for fam, entries in prompt_sets["prompts"].items():
        for e in entries:
            cells.append((fam, e["id"], e["text"]))
    return cells


def render_arms(
    args, prompt_sets: dict, arms: List[Tuple[str, Optional[str]]], run_dir: Path
) -> Tuple[Path, dict]:
    from anima_lora import GenerationRequest, generate, load_vae
    from library.inference import get_generation_settings
    from library.inference.models import load_text_encoder
    from library.inference.output import decode_to_pil
    from library.runtime.device import clean_memory_on_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cells = _flat_cells(prompt_sets)
    seeds = list(args.seeds)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "rendering %d prompts x %d seeds x %d arms -> %s",
        len(cells),
        len(seeds),
        len(arms),
        run_dir,
    )

    vae = load_vae(
        VAE,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=64,
        disable_cache=True,
        dtype=torch.bfloat16,
        eval=True,
    )

    manifest: dict = {
        "prompt_sets": str(Path(args.prompt_sets).resolve()),
        "artist": prompt_sets["artist"],
        "seeds": seeds,
        "cells": [{"family": f, "id": pid, "text": txt} for f, pid, txt in cells],
        "arms": {},
    }

    for arm_name, lora in arms:
        arm_dir = run_dir / arm_name
        arm_dir.mkdir(exist_ok=True)
        req = GenerationRequest(
            dit=DIT,
            vae=VAE,
            text_encoder=TEXT_ENCODER,
            prompt=cells[0][2],
            image_size=(args.size[0], args.size[1]),
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg,
            sampler=args.sampler,
            lora_weight=[lora] if lora else None,
            lora_multiplier=[1.0] if lora else None,
            save_path=str(arm_dir),
            # All cells render at one fixed size → a single static block-compile
            # graph; warmup (once per arm) amortizes over len(cells)*len(seeds)
            # renders. No free-fit variation here, so compile_dynamic_seq is moot.
            extra_argv=["--compile_blocks"],
        )
        base_args = req.to_args()
        base_args.device = device
        gen_settings = get_generation_settings(base_args)

        # Preload the text encoder ONCE per arm (CPU-resident; prepare_text_inputs
        # moves it to device to encode and parks it back — OOM-safe alongside the
        # DiT). Without this, generate() reloads the TE from disk and frees it on
        # every one of the len(cells)*len(seeds) calls. `conds_cache` collapses the
        # repeated (identical) negative-prompt encodings to one.
        text_encoder = load_text_encoder(
            base_args, dtype=torch.bfloat16, device=torch.device("cpu")
        )
        text_encoder.eval()
        shared: dict = {"text_encoder": text_encoder, "conds_cache": {}}
        latents: Dict[str, torch.Tensor] = {}
        for fam, pid, text in cells:
            for si, seed in enumerate(seeds):
                a = copy.copy(base_args)
                a.prompt = text
                a.seed = seed
                latent = generate(a, gen_settings, shared_models=shared)
                latents[f"{pid}__s{si}"] = latent.detach().to("cpu")
        shared.pop("model", None)
        shared.pop("text_encoder", None)
        del text_encoder
        clean_memory_on_device(device)

        renders: Dict[str, str] = {}
        for key, latent in latents.items():
            pil = decode_to_pil(vae, latent, device)
            rel = f"{arm_name}/{key}.png"
            pil.save(run_dir / rel)
            renders[key] = str((run_dir / rel).resolve())
        clean_memory_on_device(device)

        manifest["arms"][arm_name] = {"lora": lora, "renders": renders}
        (run_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("[%s] %d renders saved", arm_name, len(renders))

    return run_dir, manifest


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def _winrate(pairs: List[Tuple[float, float]]) -> Tuple[float, int]:
    """Fraction of (a, b) pairs with a > b, dropping any nan pair. Returns (rate, n)."""
    valid = [(a, b) for a, b in pairs if a == a and b == b]
    if not valid:
        return float("nan"), 0
    wins = sum(1.0 for a, b in valid if a > b)
    return wins / len(valid), len(valid)


def score_manifest(args, prompt_sets: dict, manifest: dict) -> dict:
    """Score every render; return per-arm/per-family score tables + paired gates."""
    from PIL import Image

    from anime_tools.tagger.tagger import AnimaTagger
    from anime_tools.tagger.readback import AGG_LOGSIGMOID, AGG_RECALL, TagReadback

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # general-only judge — matches how the characteristic/prompted sets were typed.
    rb = TagReadback(
        AnimaTagger(args.model_dir, device=device), content_categories=("general",)
    )

    characteristic = set(prompt_sets["characteristic_set"])
    # id -> entry (carries prompted/omitted general-tag sets)
    entry_by_id: Dict[str, dict] = {}
    family_by_id: Dict[str, str] = {}
    for fam, entries in prompt_sets["prompts"].items():
        for e in entries:
            entry_by_id[e["id"]] = e
            family_by_id[e["id"]] = fam

    seeds = manifest["seeds"]
    aggs = (AGG_LOGSIGMOID, AGG_RECALL)

    # Precompute the three column masks per prompt id, per agg-independent (bool).
    def mask_for(tags: List[str]) -> torch.Tensor:
        return rb.caption_mask([t.replace("_", " ") for t in tags])

    richness_mask: Dict[str, torch.Tensor] = {}
    adherence_mask: Dict[str, torch.Tensor] = {}
    leakage_mask: Dict[str, torch.Tensor] = {}
    for pid, e in entry_by_id.items():
        prompted = set(e.get("prompted", []))
        fam = family_by_id[pid]
        if fam in RICHNESS_FAMILIES:
            richness_mask[pid] = mask_for(sorted(characteristic - prompted))
        if fam in ADHERENCE_FAMILIES:
            adherence_mask[pid] = mask_for(e.get("prompted", []))
        if fam in LEAKAGE_FAMILIES:
            leakage_mask[pid] = mask_for(e.get("omitted", []))

    # scores[arm][agg][kind][pid][si] = float
    scores: dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    # cache per-render logits so we score all masks in one forward
    for arm_name, arm in manifest["arms"].items():
        for key, path in arm["renders"].items():
            pid, s = key.split("__s")
            si = int(s)
            logits = rb.image_logits(Image.open(path)).unsqueeze(0)  # [1, n_tags]
            fam = family_by_id[pid]
            for agg in aggs:
                # richness is family-qualified (richness_sparse vs richness_no_trigger):
                # the sparse gate must not be diluted by the trigger-free H4 probe,
                # whose richness is EXPECTED to be low if migration is trigger-bound.
                if pid in richness_mask:
                    scores[arm_name][agg][f"richness_{fam}"][pid][si] = float(
                        rb.readback_from_logits(
                            logits, richness_mask[pid].unsqueeze(0), agg=agg
                        )[0, 0]
                    )
                if pid in adherence_mask:
                    scores[arm_name][agg]["adherence"][pid][si] = float(
                        rb.readback_from_logits(
                            logits, adherence_mask[pid].unsqueeze(0), agg=agg
                        )[0, 0]
                    )
                if pid in leakage_mask:
                    scores[arm_name][agg]["leakage"][pid][si] = float(
                        rb.readback_from_logits(
                            logits, leakage_mask[pid].unsqueeze(0), agg=agg
                        )[0, 0]
                    )

    arm_names = list(manifest["arms"].keys())
    kinds = [f"richness_{f}" for f in RICHNESS_FAMILIES] + ["adherence", "leakage"]

    # ---- per-arm means (report) ----
    per_arm: dict = {}
    for arm in arm_names:
        per_arm[arm] = {}
        for agg in aggs:
            per_arm[arm][agg] = {}
            for kind in kinds:
                vals = [
                    v
                    for pid in scores[arm][agg][kind]
                    for v in scores[arm][agg][kind][pid].values()
                ]
                per_arm[arm][agg][kind] = _mean(vals)

    # ---- paired win-rates arm_i vs reference (base if present else arm0) ----
    ref = "base" if "base" in arm_names else arm_names[0]
    paired: dict = {}
    for arm in arm_names:
        if arm == ref:
            continue
        paired[arm] = {}
        for agg in aggs:
            paired[arm][agg] = {}
            for kind in kinds:
                pairs: List[Tuple[float, float]] = []
                for pid in scores[arm][agg][kind]:
                    for si in scores[arm][agg][kind][pid]:
                        a = scores[arm][agg][kind][pid][si]
                        b = scores[ref][agg][kind].get(pid, {}).get(si, float("nan"))
                        pairs.append((a, b))
                rate, npair = _winrate(pairs)
                paired[arm][agg][kind] = {"winrate": rate, "n": npair}

    # ---- readback instrument check: adherence(true) vs adherence(shuffled) ----
    # For each arm's detailed renders, score its OWN prompted set vs a random OTHER
    # detailed prompt's prompted set (same render). Instrument-validity, per arm.
    import random

    det_ids = [pid for pid in entry_by_id if family_by_id[pid] in ADHERENCE_FAMILIES]
    shuffle_check: dict = {}
    from PIL import Image as _Image

    other_mask = {
        pid: mask_for(entry_by_id[pid].get("prompted", [])) for pid in det_ids
    }
    for arm_name, arm in manifest["arms"].items():
        rng = random.Random(1234)
        pairs: List[Tuple[float, float]] = []
        for key, path in arm["renders"].items():
            pid, _ = key.split("__s")
            if pid not in adherence_mask:
                continue
            others = [o for o in det_ids if o != pid]
            if not others:
                continue
            oid = rng.choice(others)
            logits = rb.image_logits(_Image.open(path)).unsqueeze(0)
            true_v = float(
                rb.readback_from_logits(
                    logits, adherence_mask[pid].unsqueeze(0), agg=AGG_LOGSIGMOID
                )[0, 0]
            )
            shuf_v = float(
                rb.readback_from_logits(
                    logits, other_mask[oid].unsqueeze(0), agg=AGG_LOGSIGMOID
                )[0, 0]
            )
            pairs.append((true_v, shuf_v))
        rate, npair = _winrate(pairs)
        shuffle_check[arm_name] = {"winrate": rate, "n": npair}

    return {
        "arms": arm_names,
        "reference": ref,
        "seeds": seeds,
        "kinds": kinds,
        "per_arm_mean": per_arm,
        "paired_vs_reference": paired,
        "shuffle_check": shuffle_check,
        "n_characteristic": len(characteristic),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    with open(args.prompt_sets) as f:
        prompt_sets = json.load(f)

    arms: List[Tuple[str, Optional[str]]] = []
    for spec in args.lora:
        name, _, path = spec.partition(":")
        if not path:
            raise SystemExit(f"--lora expects NAME:PATH, got {spec!r}")
        arms.append((name, path))
    if args.include_base:
        arms.append(("base", None))
    if not arms:
        raise SystemExit("no arms — pass --lora NAME:PATH and/or --include_base")

    # Single results dir for this run; renders live under it (results/<run>/renders/).
    run_dir = make_run_dir("tag_dropout", label=args.label)

    if args.reuse_renders:
        with open(args.reuse_renders) as f:
            manifest = json.load(f)
        render_dir = Path(args.reuse_renders).parent
    else:
        render_dir = run_dir / "renders"
        render_dir, manifest = render_arms(args, prompt_sets, arms, render_dir)

    res = score_manifest(args, prompt_sets, manifest)

    # -- gate (Phase 0 sanity; Phase 1 reads the same numbers, different verdict) --
    from anime_tools.tagger.readback import AGG_LOGSIGMOID

    kinds = res["kinds"]
    arm0 = (
        res["arms"][0]
        if res["arms"][0] != res["reference"]
        else (res["arms"][1] if len(res["arms"]) > 1 else res["arms"][0])
    )
    # Gate on SPARSE richness (trigger present) — the H1 direction; no_trigger is
    # the H4 locus probe, reported but not gated.
    rich_pair = (
        res["paired_vs_reference"]
        .get(arm0, {})
        .get(AGG_LOGSIGMOID, {})
        .get("richness_sparse", {})
    )
    rich_wr = rich_pair.get("winrate", float("nan"))
    shuf_wr = max(
        (
            sc["winrate"]
            for sc in res["shuffle_check"].values()
            if sc["winrate"] == sc["winrate"]
        ),
        default=float("nan"),
    )
    gate_rich = rich_wr == rich_wr and rich_wr >= args.gate_richness_winrate
    gate_shuf = shuf_wr == shuf_wr and shuf_wr >= args.gate_shuffle_winrate
    verdict = "PASS" if (gate_rich and gate_shuf) else "FAIL"

    def _cols(d: dict) -> str:
        return " | ".join(f"{d[k]:.3f}" for k in kinds)

    lines = [
        f"# tag-dropout eval — artist {prompt_sets['artist']}  (arms: {', '.join(res['arms'])})",
        "",
        f"verdict: **{verdict}**",
        f"- sparse-richness win-rate ({arm0} vs {res['reference']}, logsigmoid) = "
        f"{rich_wr:.3f} (n={rich_pair.get('n', 0)}) — gate >= {args.gate_richness_winrate}: "
        f"{'ok' if gate_rich else 'FAIL'}",
        f"- readback instrument: adherence(true) > adherence(shuffled) best win-rate = "
        f"{shuf_wr:.3f} — gate >= {args.gate_shuffle_winrate}: {'ok' if gate_shuf else 'FAIL'}",
        "",
        "## per-arm mean scores (logsigmoid)",
        "",
        "| arm | " + " | ".join(kinds) + " |",
        "|---|" + "---|" * len(kinds),
    ]
    for arm in res["arms"]:
        m = res["per_arm_mean"][arm][AGG_LOGSIGMOID]
        lines.append(f"| {arm} | " + _cols(m) + " |")
    lines += [
        "",
        "## paired win-rates vs " + res["reference"] + " (logsigmoid)",
        "",
        "| arm | " + " | ".join(kinds) + " |",
        "|---|" + "---|" * len(kinds),
    ]
    for arm, agg_d in res["paired_vs_reference"].items():
        d = agg_d[AGG_LOGSIGMOID]
        cells = " | ".join(f"{d[k]['winrate']:.3f} (n{d[k]['n']})" for k in kinds)
        lines.append(f"| {arm} | " + cells + " |")
    lines += [
        "",
        "## readback instrument check (adherence true vs shuffled, logsigmoid)",
        "",
    ]
    for arm, sc in res["shuffle_check"].items():
        lines.append(f"- {arm}: win-rate {sc['winrate']:.3f} (n={sc['n']})")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics={
            "verdict": verdict,
            "gate_richness": gate_rich,
            "gate_shuffle": gate_shuf,
            "richness_winrate": rich_wr,
            "shuffle_winrate": shuf_wr,
            "results": res,
            "render_dir": str(render_dir),
        },
        label=args.label,
        artifacts=[run_dir / "summary.md"],
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    )
    log.info("verdict: %s  ->  %s", verdict, run_dir)


if __name__ == "__main__":
    main()
