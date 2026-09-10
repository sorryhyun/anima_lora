#!/usr/bin/env python3
"""plan_render S3 — the judge: R0 reader floor, R1 held-out CER, R2 caption
swap (memorisation control), R3 contact sheet, for one adapter arm.

Three stages, run in one daemon job (``fill`` and ``read`` both want the GPU
but never at the same time — the DiT stack is freed before the reader loads)::

    make daemon-run ARGS="--stall-timeout 0 project/cjk_aware_anima_dit/render/judge.py fill read report \\
        --edition en --arm render_en_e2 --weight output/ckpt/anima_render_en/anima_render_en-000002.safetensors"

* ``fill`` — one loaded DiT + text encoder + VAE, the arm's EasyControl network
  applied once; per cell the holed panel (``heldout_staging/``) is VAE-encoded,
  ``set_cond`` + ``precompute_cond_kv`` re-prime the network, and
  ``library.inference.generate`` runs at the panel's own 768-tier size with the
  cell's caption. Cells: ``--n_heldout`` held-out panels (R1 / R3), the same
  panels for the two floors (``--floor_weight`` = an inpaint adapter that never
  saw a text clause; ``base`` = the network removed, text-to-image at the same
  size and caption), and ``--n_swap`` *training* panels whose clause is
  replaced by another training panel's lines of matched length (R2). Writes
  ``<out>/<arm>/fill/*.png`` + ``manifest.jsonl``.
* ``read`` — ``stock`` (PaddleOCR-VL-1.6, ``ocr/pseudo_label.py``) reads every
  holed bubble's box (padded 12 %, the S0 crop rule) off each fill, and off
  the *target* panel for R0. Writes ``reads.jsonl``.
* ``report`` — CER per bubble (whitespace-blind headline, spaced column), R0–R2
  numbers, ``report.json`` + ``report.md`` + ``sheet.png`` (R3: target | cond
  | fill, 24 held-out cells).

CER: ``levenshtein(norm(hyp), norm(ref)) / len(norm(ref))``, clipped to 1;
``norm`` is NFKC + casefold + the punctuation set of the old text-bind judge,
whitespace deleted (blind) or collapsed (spaced). Do not compare a number
here to any ``/ 617`` figure in findings.md — different units.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINE = HERE.parent
REPO = LINE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LINE / "ocr"))

import corpus_boxes as cb  # noqa: E402

DEFAULT_FLOOR = "output/ckpt/anima_inpaint_girl_preview_v1.safetensors"
STRIP = set("。、！？!?…・「」『』()（）,.~～ー-—\"'“”♡♥")
KIND_ORDER = ("heldout", "floor_inpaint", "base", "swap")


# --------------------------------------------------------------------------- text


def norm(s: str, spaced: bool = False) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(c for c in s if c not in STRIP)
    return " ".join(s.split()) if spaced else "".join(s.split())


def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb_ in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb_)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str, spaced: bool = False) -> float:
    h, r = norm(hyp, spaced), norm(ref, spaced)
    if not r:
        return 1.0
    return min(1.0, levenshtein(h, r) / len(r))


def swap_lines(lines: list[str], pool: list[str], rng: random.Random) -> list[str]:
    """For each line, a different pool line of the closest length (ties by rng)."""
    out = []
    for ln in lines:
        cands = [p for p in pool if p != ln]
        cands.sort(key=lambda p: (abs(len(p) - len(ln)), rng.random()))
        out.append(cands[0] if cands else ln)
    return out


# --------------------------------------------------------------------------- cells


def load_rows(base: Path, split: str) -> list[dict]:
    return [
        json.loads(line)
        for line in (base / split / "boxes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def build_cells(a, base: Path) -> list[dict]:
    from cut import caption_for

    rng = random.Random(a.seed)
    held = load_rows(base, "heldout")
    train = load_rows(base, "resized")
    held_pick = rng.sample(held, min(a.n_heldout, len(held)))
    cells = []
    for r in held_pick:
        for kind in ("heldout", "floor_inpaint", "base"):
            cells.append(
                {
                    "kind": kind,
                    "stem": r["stem"],
                    "target": str(base / "heldout" / r["rel"]),
                    "cond": str(base / "heldout_staging" / r["rel"]),
                    "size": r["size"],
                    "caption": r["caption"],
                    "bubbles": r["bubbles"],
                    "lines": [b["line"] for b in r["bubbles"]],
                }
            )
    pool = [b["line"] for r in train for b in r["bubbles"]]
    for r in rng.sample(train, min(a.n_swap, len(train))):
        lines = [b["line"] for b in r["bubbles"]]
        swapped = swap_lines(lines, pool, rng)
        cells.append(
            {
                "kind": "swap",
                "stem": r["stem"],
                "target": str(base / "resized" / r["rel"]),
                "cond": str(base / "staging" / r["rel"]),
                "size": r["size"],
                "caption": caption_for(a.edition, swapped),
                "bubbles": r["bubbles"],
                "lines": lines,
                "swapped": swapped,
            }
        )
    return cells


# --------------------------------------------------------------------------- fill


def _load_network(weight: str, anima, device):
    import torch

    from networks.methods.easycontrol import create_network_from_weights

    network, _ = create_network_from_weights(
        multiplier=1.0, file=weight, ae=None, text_encoders=None, unet=anima
    )
    network.load_weights(weight)
    network.to(device, dtype=torch.bfloat16)
    network.apply_to(text_encoders=None, unet=anima)
    anima._easycontrol_network = network
    return network


def _encode_cond(vae, path: str, size, device):
    import torch
    from PIL import Image
    from torchvision import transforms

    W, H = size
    img = Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    x = tfm(img).unsqueeze(0).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        lat = vae.encode_pixels_to_latents(x)
    return lat.squeeze(2) if lat.ndim == 5 else lat


def stage_fill(a, out: Path, cells: list[dict]) -> None:
    import torch

    from anima_lora.inference import GenerationRequest
    from anima_lora.models import default_checkpoints
    from library.inference.generation import generate, get_generation_settings
    from library.inference.models import load_dit_model, load_shared_models
    from library.inference.output import decode_latent, pixels_to_pil
    from library.models import qwen_vae

    ck = default_checkpoints()
    req = GenerationRequest(
        prompt="",
        image_size=(768, 768),
        infer_steps=a.steps,
        guidance_scale=a.cfg,
        seed=a.seed,
        dit=ck.dit,
        vae=ck.vae,
        text_encoder=ck.text_encoder,
        save_path=str(out / "fill"),
        vocab_pack=a.vocab_pack or None,
        no_vocab_pack=(a.vocab_pack == ""),
    )
    args = req.to_args()
    gen = get_generation_settings(args)
    device = gen.device
    shared = load_shared_models(args)
    shared["conds_cache"] = {}
    anima = load_dit_model(args, device, torch.bfloat16)
    shared["model"] = anima
    vae = qwen_vae.load_vae(
        args.vae, device="cpu", disable_mmap=True, disable_cache=True, vae_2d=True
    )
    vae.to(device, dtype=torch.bfloat16).eval()

    (out / "fill").mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done = (
        {json.loads(line)["id"] for line in manifest.read_text().splitlines()}
        if manifest.is_file() and not a.overwrite
        else set()
    )
    fh = manifest.open("a" if done else "w", encoding="utf-8")
    weights = {
        "heldout": a.weight,
        "swap": a.weight,
        "floor_inpaint": a.floor_weight,
        "base": None,
    }
    network, current = None, "__none__"
    t0, n = time.time(), 0
    for kind in KIND_ORDER:
        for c in [c for c in cells if c["kind"] == kind]:
            cid = f"{kind}__{c['stem']}"
            if cid in done:
                continue
            w = weights[kind]
            if w != current:
                if network is not None:
                    network.remove_from()
                    anima._easycontrol_network = None
                    network = None
                if w:
                    network = _load_network(w, anima, device)
                current = w
            if network is not None:
                network.set_cond(_encode_cond(vae, c["cond"], c["size"], device))
                network.precompute_cond_kv()
            a2 = copy.deepcopy(args)
            a2.prompt = c["caption"]
            a2.image_size = [c["size"][1], c["size"][0]]
            a2.seed = a.seed
            latent = generate(a2, gen, shared)
            img = pixels_to_pil(decode_latent(vae, latent, device))
            vae.to(device)
            png = out / "fill" / f"{cid}.png"
            img.save(png)
            row = {k: v for k, v in c.items()}
            row.update({"id": cid, "png": str(png)})
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n += 1
            if n % 10 == 0:
                print(
                    f"  [fill] {n} cells, {(time.time() - t0) / n:.1f} s/cell",
                    flush=True,
                )
    fh.close()
    if network is not None:
        network.remove_from()
    del anima, shared, vae
    torch.cuda.empty_cache()
    print(f"[fill] {n} cells in {time.time() - t0:.0f}s → {manifest}", flush=True)


# --------------------------------------------------------------------------- read


def stage_read(a, out: Path) -> None:
    import cv2
    import pseudo_label as pl

    rows = [
        json.loads(line)
        for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reader = pl.SWEEPERS["stock"](None, a.device)
    targets_done: set[str] = set()
    crops, keys = [], []
    for r in rows:
        fill = cv2.imread(r["png"])
        for j, b in enumerate(r["bubbles"]):
            c = cb.pad_crop(fill, b["box"])
            if c is not None:
                crops.append(c)
                keys.append((r["id"], j, "fill"))
        if r["stem"] not in targets_done and r["kind"] in ("heldout", "swap"):
            targets_done.add(r["stem"])
            tgt = cv2.imread(r["target"])
            for j, b in enumerate(r["bubbles"]):
                c = cb.pad_crop(tgt, b["box"])
                if c is not None:
                    crops.append(c)
                    keys.append((r["stem"], j, "target"))
    texts: list[str] = []
    t0 = time.time()
    for s in range(0, len(crops), a.bs):
        texts += [
            cb._flatten(a.edition, t) for t, _, _ in reader.read(crops[s : s + a.bs])
        ]
        print(f"  [read] {min(s + a.bs, len(crops))}/{len(crops)}", flush=True)
    with (out / "reads.jsonl").open("w", encoding="utf-8") as fh:
        for (rid, j, which), t in zip(keys, texts):
            fh.write(
                json.dumps(
                    {"id": rid, "j": j, "which": which, "text": t}, ensure_ascii=False
                )
                + "\n"
            )
    print(
        f"[read] {len(crops)} crops in {time.time() - t0:.0f}s → {out / 'reads.jsonl'}",
        flush=True,
    )


# --------------------------------------------------------------------------- report


def stage_report(a, out: Path) -> None:
    rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    reads: dict[tuple, str] = {}
    for line in (out / "reads.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            reads[(r["id"], r["j"], r["which"])] = r["text"]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    rep: dict = {"arm": a.arm, "edition": a.edition, "weight": a.weight, "kinds": {}}
    # R0: the reader on the targets themselves
    r0 = [
        cer(reads.get((r["stem"], j, "target"), ""), ln)
        for r in rows.values()
        if r["kind"] == "heldout"
        for j, ln in enumerate(r["lines"])
    ]
    rep["R0_reader_floor"] = {"cer_blind": mean(r0), "n_lines": len(r0)}
    # R1 + floors
    for kind in ("heldout", "floor_inpaint", "base"):
        blind, spaced, exact = [], [], 0
        for r in rows.values():
            if r["kind"] != kind:
                continue
            for j, ln in enumerate(r["lines"]):
                hyp = reads.get((r["id"], j, "fill"), "")
                blind.append(cer(hyp, ln))
                spaced.append(cer(hyp, ln, spaced=True))
                exact += norm(hyp) == norm(ln)
        rep["kinds"][kind] = {
            "cer_blind": mean(blind),
            "cer_spaced": mean(spaced),
            "exact": exact / max(1, len(blind)),
            "n_lines": len(blind),
            "n_cells": sum(1 for r in rows.values() if r["kind"] == kind),
        }
    # R2: caption swap on training panels
    follows, to_swapped, to_orig, n = 0, [], [], 0
    for r in rows.values():
        if r["kind"] != "swap":
            continue
        for j, (orig, sw) in enumerate(zip(r["lines"], r["swapped"])):
            if norm(orig) == norm(sw):
                continue
            hyp = reads.get((r["id"], j, "fill"), "")
            cs, co = cer(hyp, sw), cer(hyp, orig)
            to_swapped.append(cs)
            to_orig.append(co)
            follows += cs < co
            n += 1
    rep["R2_swap"] = {
        "follows_caption": follows / max(1, n),
        "cer_to_swapped": mean(to_swapped),
        "cer_to_original": mean(to_orig),
        "n_lines": n,
    }
    (out / "report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    k = rep["kinds"]
    md = [
        f"# judge — {a.arm} ({a.edition})",
        "",
        f"weight: `{a.weight}`  steps {a.steps}  cfg {a.cfg}  seed {a.seed}",
        "",
        "| ruler | CER blind | CER spaced | exact | n lines |",
        "|---|---|---|---|---|",
        f"| R0 reader floor (target) | {rep['R0_reader_floor']['cer_blind']:.3f} | — | — | {rep['R0_reader_floor']['n_lines']} |",
        f"| **R1 held-out fill** | **{k['heldout']['cer_blind']:.3f}** | {k['heldout']['cer_spaced']:.3f} | {k['heldout']['exact']:.2f} | {k['heldout']['n_lines']} |",
        f"| floor: inpaint (no text clause seen) | {k['floor_inpaint']['cer_blind']:.3f} | {k['floor_inpaint']['cer_spaced']:.3f} | {k['floor_inpaint']['exact']:.2f} | {k['floor_inpaint']['n_lines']} |",
        f"| floor: base (no adapter) | {k['base']['cer_blind']:.3f} | {k['base']['cer_spaced']:.3f} | {k['base']['exact']:.2f} | {k['base']['n_lines']} |",
        f"| R2 swap: follows caption | {rep['R2_swap']['follows_caption']:.2f} | to swapped {rep['R2_swap']['cer_to_swapped']:.3f} | to original {rep['R2_swap']['cer_to_original']:.3f} | {rep['R2_swap']['n_lines']} |",
        "",
        "Gates: G0 (EN) R1 ≤ 0.3 and R2 ≥ 0.8; G1 (JA-SHIP) R1 ≤ 0.5 and R2 ≥ 0.7.",
        "",
        "## per-line (held-out)",
        "",
        "| cell | j | caption line | read | CER |",
        "|---|---|---|---|---|",
    ]
    for r in rows.values():
        if r["kind"] != "heldout":
            continue
        for j, ln in enumerate(r["lines"]):
            hyp = reads.get((r["id"], j, "fill"), "")
            md.append(
                f"| {r['stem']} | {j} | {ln[:40]} | {hyp[:40]} | {cer(hyp, ln):.2f} |"
            )
    md += [
        "",
        "## swap (training panels)",
        "",
        "| cell | j | original | swapped | read | CER swapped / original |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows.values():
        if r["kind"] != "swap":
            continue
        for j, (orig, sw) in enumerate(zip(r["lines"], r["swapped"])):
            hyp = reads.get((r["id"], j, "fill"), "")
            md.append(
                f"| {r['stem']} | {j} | {orig[:30]} | {sw[:30]} | {hyp[:30]} | {cer(hyp, sw):.2f} / {cer(hyp, orig):.2f} |"
            )
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    _sheet(a, out, rows, reads)
    print("\n".join(md[:13]), flush=True)


def _sheet(a, out: Path, rows: dict, reads: dict) -> None:
    from PIL import Image, ImageDraw

    cells = [r for r in rows.values() if r["kind"] == "heldout"][: a.n_sheet]
    if not cells:
        return
    tw, th, cap = 300, 300, 60
    font = cb._font(13)
    sheet = Image.new("RGB", (3 * tw + 16, len(cells) * (th + cap)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(cells):
        y = i * (th + cap)
        for col, path in enumerate((r["target"], r["cond"], r["png"])):
            im = Image.open(path).convert("RGB")
            dr = ImageDraw.Draw(im)
            for b in r["bubbles"]:
                dr.rectangle(tuple(b["box"]), outline="red", width=3)
            im.thumbnail((tw - 4, th - 4))
            sheet.paste(im, (col * (tw + 8), y))
        text = " | ".join(
            f"[{j}] {ln[:22]} → {reads.get((r['id'], j, 'fill'), '')[:22]}"
            for j, ln in enumerate(r["lines"])
        )
        for li, seg in enumerate(cb._wrap(d, f"{r['stem']}: {text}", font, 3 * tw)[:3]):
            d.text((2, y + th + 2 + 18 * li), seg, "black", font)
    sheet.save(out / "sheet.png")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stages", nargs="+", choices=("fill", "read", "report"))
    ap.add_argument("--edition", default="en", choices=("en", "ja", "ko"))
    ap.add_argument(
        "--arm", required=True, help="output dir name under output/render/judge/"
    )
    ap.add_argument("--weight", default=None, help="the arm's EasyControl .safetensors")
    ap.add_argument("--floor_weight", default=DEFAULT_FLOOR)
    ap.add_argument("--root", default="post_image_dataset/render")
    ap.add_argument("--out", default="output/render/judge")
    ap.add_argument("--n_heldout", type=int, default=48)
    ap.add_argument("--n_swap", type=int, default=24)
    ap.add_argument("--n_sheet", type=int, default=24)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--vocab_pack", default=None, help="'' = stock (EN); a pack path (JA)"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=16, help="reader batch")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    base = REPO / a.root / a.edition
    out = REPO / a.out / a.arm
    out.mkdir(parents=True, exist_ok=True)
    if "fill" in a.stages:
        if not a.weight:
            sys.exit("fill needs --weight")
        stage_fill(a, out, build_cells(a, base))
    if "read" in a.stages:
        stage_read(a, out)
    if "report" in a.stages:
        stage_report(a, out)


if __name__ == "__main__":
    main()
