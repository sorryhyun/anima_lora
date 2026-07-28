"""In-place instruction editing — DirectEdit inversion x subject_edit adapter.

The untested middle ground between the line's two shipped outcomes:

    outcome 1 (in-place recipe)   preserves composition via inversion+anchor,
                                  but needs full ψ_src/ψ_tar captions + a mask.
    outcome 2 (EasyEdit)          takes an instruction prompt at 1x NFE,
                                  but is composition-free ("re-render").

This probe composes them: invert the source image A under ψ_src = A's own
caption (deployment source: the Anima Tagger — here the ground-truth .txt, an
upper bound), then run the DirectEdit edit pass with

    prompt  = the mined tag-delta instruction (subject_edit's training
              distribution — additions in target order + `-`-prefixed removals)
    cond    = A itself, through the subject_edit adapter at b_offset 0
    anchor  = global (NO mask, NO EC hole — the instruction, not a region box,
              is supposed to own what changes)

Arms per pair (all seed/size-matched; edit.py arms at t_inj 0):

    easyedit        feed-forward inference.py, cond=A, prompt=delta — the
                    shipped outcome-2 baseline: instruction lands, composition
                    free. flow_shift 3.0 (its shipped operating point).
    inv_noec        edit.py, no EC — control: base model + inversion + anchor
                    fed a bare delta prompt (expected: instruction fails; the
                    base TE reads `-tag` as an attractor).
    inv_ec_both     edit.py + subject_edit adapter active through BOTH passes
                    (honors the exact-recon invariant; the adapter sees an
                    off-distribution descriptive ψ_src during inversion).
    inv_ec_edit     edit.py + --easycontrol_edit_only — baseline inversion,
                    adapter only on the edit pass (clean trajectory; anchor
                    computed under a different effective model).

Judged on three axes: (1) is A's composition preserved (vs easyedit)?
(2) do the instructed changes land (vs inv_noec)? (3) does identity hold?
mse_vs_src in result.json is the quantitative composition proxy — easyedit
should be the high outlier if inversion is buying anything.

Pairs come from the subject_edit miner manifest, so cond A + the delta are
TRAIN-pair material (upper bound), but A here plays the *source* role the
adapter never saw framed this way: at train time cond=A/prompt=delta had
target B as ground truth; here the desired output is "A, edited".

Usage:
    uv run python project/directedit_ec/bench/run_inplace_probe.py \
        --n_pairs 3 --rating safe,sensitive
    uv run python project/directedit_ec/bench/run_inplace_probe.py \
        --b_offsets 0,2        # widen the EC arms if b0 turns out inert
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_edit_probe import (  # noqa: E402
    DEFAULT_EC_WEIGHT,
    RESIZED_DIR,
    make_sheet,
    pick_pairs,
    resolve,
)

from bench._common import make_run_dir, start_heartbeat, write_result  # noqa: E402
from library.datasets.buckets import (  # noqa: E402
    choose_edge,
    freefit_band_for_edge,
    freefit_bucket,
)
from library.env import default_checkpoints  # noqa: E402

logger = logging.getLogger(__name__)

NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, "
    "jpeg artifacts, sepia"
)


def caption_of(rel: str) -> str:
    txt = RESIZED_DIR / f"{rel}.txt"
    if not txt.is_file():
        raise SystemExit(f"missing caption sidecar for ψ_src: {txt}")
    return txt.read_text(encoding="utf-8").strip()


def pick_size(image: Path) -> tuple[int, int]:
    """(H, W): free-fit the source aspect into the 1024 tier, like edit.py."""
    from PIL import Image

    rw, rh = Image.open(image).size
    edge = choose_edge(rw, rh, [1024])
    bw, bh = freefit_bucket(rw, rh, freefit_band_for_edge(edge))
    return bh, bw


def mse_vs_src(out_png: Path, src_png: Path) -> float:
    import numpy as np
    from PIL import Image

    out = Image.open(out_png).convert("RGB")
    src = Image.open(src_png).convert("RGB").resize(out.size, Image.LANCZOS)
    a = np.asarray(out, dtype=np.float64) / 255.0
    b = np.asarray(src, dtype=np.float64) / 255.0
    return float(((a - b) ** 2).mean())


def easyedit_argv(
    prompt: str,
    cond: Path,
    size_hw: tuple[int, int],
    b_offset: float,
    out: Path,
    args,
    ck,
) -> list[str]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "inference.py"),
        "--dit",
        ck.dit,
        "--text_encoder",
        ck.text_encoder,
        "--vae",
        ck.vae,
        "--vae_chunk_size",
        "64",
        "--vae_disable_cache",
        "--attn_mode",
        "flash",
        "--prompt",
        prompt,
        "--negative_prompt",
        NEGATIVE,
        "--image_size",
        str(size_hw[0]),
        str(size_hw[1]),
        "--infer_steps",
        str(args.infer_steps),
        "--flow_shift",
        "3.0",
        "--sampler",
        "euler",
        "--guidance_scale",
        str(args.guidance_scale),
        "--seed",
        str(args.seed),
        "--save_path",
        str(out),
        "--easycontrol_weight",
        str(args.ec_weight),
        "--easycontrol_image",
        str(cond),
        "--easycontrol_image_match_size",
    ]
    if b_offset:
        argv += ["--easycontrol_b_offset", str(b_offset)]
    return argv


def invert_argv(
    psi_src: str,
    delta: str,
    image: Path,
    size_hw: tuple[int, int],
    ec: Optional[str],  # None | "both" | "edit"
    b_offset: float,
    anchor_scale: float,
    invert_b_offset: Optional[float],
    out: Path,
    args,
    ck,
) -> list[str]:
    argv = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "edit.py"),
        "--dit",
        ck.dit,
        "--text_encoder",
        ck.text_encoder,
        "--vae",
        ck.vae,
        "--image",
        str(image),
        "--prompt_src",
        psi_src,
        "--prompt_tar",
        delta,
        "--image_size",
        str(size_hw[0]),
        str(size_hw[1]),
        "--infer_steps",
        str(args.infer_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--t_inj",
        "0",
        "--seed",
        str(args.seed),
        "--save_path",
        str(out),
        "--no_compile_blocks",
        # ψ_src explanation probe: intrinsic FM error of the inversion-pass
        # conditioning against the source latent (cond-aware — runs under the
        # active effective model). Parsed from the arm log into result.json.
        "--fm_score",
    ]
    if anchor_scale != 1.0:
        argv += ["--anchor_scale", str(anchor_scale)]
    if ec is not None:
        argv += ["--easycontrol_weight", str(args.ec_weight)]
        if b_offset:
            argv += ["--easycontrol_b_offset", str(b_offset)]
        if ec == "edit":
            argv += ["--easycontrol_edit_only"]
        if invert_b_offset is not None:
            argv += ["--easycontrol_invert_b_offset", str(invert_b_offset)]
    return argv


def arm_name(
    kind: str,
    ec_mode: Optional[str],
    off: float,
    lam: float,
    src: str,
    ioff: Optional[float],
    tar: str = "delta",
) -> str:
    if kind == "ff":
        parts = ["easyedit", f"b{off:g}"]
    else:
        parts = ["inv", "noec" if ec_mode is None else f"ec_{ec_mode}"]
        if ec_mode is not None:
            parts.append(f"b{off:g}")
    if src == "empty":
        parts.append("src0")
    if ioff is not None:
        parts.append(f"ib{ioff:g}")
    if tar == "caption":
        parts.append("tarcap")
    if lam != 1.0:
        parts.append(f"l{lam:g}")
    return "_".join(parts)


Arm = tuple[str, str, Optional[str], float, float, str, Optional[float], str]


def parse_arms_spec(spec: str) -> list[Arm]:
    """Explicit arm list: ';'-separated arms, each 'key=val,key=val,...'.

    Keys: kind (ff|inv, default inv), ec (none|both|edit, default both),
    off (float, 0), lam (float, 1), src (caption|empty, caption),
    ioff (float, unset), tar (delta|caption|srccap, delta — the edit-pass
    prompt: the mined tag-delta instruction; the mined target's FULL caption;
    or the SOURCE caption + ", <delta>" appended — the base-model form for
    minimal edits, matching the phase-0 `caption + ", glasses"` convention),
    name (optional override).
    """
    arms: list[Arm] = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        kv = dict(part.split("=", 1) for part in chunk.split(",") if part.strip())
        kind = kv.get("kind", "inv")
        ec_raw = kv.get("ec", "both")
        ec_mode = None if ec_raw == "none" else ec_raw
        off = float(kv.get("off", 0))
        lam = float(kv.get("lam", 1))
        src = kv.get("src", "caption")
        ioff = float(kv["ioff"]) if "ioff" in kv else None
        tar = kv.get("tar", "delta")
        if kind not in ("ff", "inv") or src not in ("caption", "empty"):
            raise SystemExit(f"bad arm spec chunk: {chunk!r}")
        if ec_mode not in (None, "both", "edit"):
            raise SystemExit(f"bad ec mode in arm spec chunk: {chunk!r}")
        if tar not in ("delta", "caption", "srccap"):
            raise SystemExit(f"bad tar mode in arm spec chunk: {chunk!r}")
        name = kv.get("name") or arm_name(kind, ec_mode, off, lam, src, ioff, tar)
        arms.append((name, kind, ec_mode, off, lam, src, ioff, tar))
    if not arms:
        raise SystemExit("--arms_spec parsed to an empty arm list")
    return arms


def parse_arm_log(log_path: Path) -> dict:
    """Lift the instrumentation lines out of an edit.py arm log."""
    import re

    out: dict = {}
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    hits = re.findall(r"mean\|dz\|=([0-9.eE+-]+)", text)
    if hits:
        out["dz_mean_abs"] = float(hits[-1])
    hits = re.findall(r"FM-error = ([0-9.eE+-]+)", text)
    if hits:
        out["fm_error"] = float(hits[-1])
    return out


def main() -> None:
    from library.log import setup_logging

    setup_logging()
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--n_pairs", type=int, default=3)
    p.add_argument("--ec_weight", default=str(DEFAULT_EC_WEIGHT))
    p.add_argument(
        "--b_offsets",
        default="0",
        help="Comma-separated b_cond offsets for the EC arms. subject_edit is "
        "engaged at the trained point (0) and enters the copy regime at +2, "
        "so 0 is the expected operating point.",
    )
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=2400)
    p.add_argument(
        "--pair_scope",
        choices=("any", "same_artist", "cross_artist"),
        default="same_artist",
    )
    p.add_argument("--max_delta", type=int, default=20)
    p.add_argument(
        "--rating",
        default="any",
        help="comma-separated Anima rating bands both pair endpoints must "
        "fall in; 'any' disables the filter",
    )
    p.add_argument(
        "--anchor_scales",
        default="1",
        help="Comma-separated global Δz anchor multipliers λ for the "
        "inversion+EC arms (1 = full anchor, 0 = unanchored generation from "
        "the inverted init). Sweeping e.g. 1,0.5,0.25,0 traces the "
        "composition↔instruction frontier.",
    )
    p.add_argument(
        "--skip_easyedit",
        action="store_true",
        help="Drop the feed-forward baseline arm (e.g. when re-running only "
        "the inversion arms).",
    )
    p.add_argument(
        "--skip_noec",
        action="store_true",
        help="Drop the no-adapter inversion control arm (already judged in "
        "run 1: reconstruction or wash-out).",
    )
    p.add_argument(
        "--arms_spec",
        default=None,
        help="Explicit arm list overriding the default builder: ';'-separated "
        "arms, each 'key=val,...' with keys kind(ff|inv) ec(none|both|edit) "
        "off lam src(caption|empty) ioff name. E.g. "
        "'ec=both,src=empty;ec=both,src=empty,lam=0.5,ioff=2'.",
    )
    p.add_argument(
        "--delta_override",
        default=None,
        help="Replace the mined delta instruction with a fixed one (e.g. "
        "'glasses') for minimal-edit probes: delta arms get it verbatim, "
        "tar=srccap arms get 'source caption, <override>'. The mined target "
        "column then no longer depicts the desired output.",
    )
    p.add_argument("--label", default="inplace-edit-probe")
    args = p.parse_args()

    if not Path(args.ec_weight).is_file():
        raise SystemExit(f"EC checkpoint not found: {args.ec_weight}")
    start_heartbeat()
    ck = default_checkpoints()
    offsets = [float(s) for s in args.b_offsets.split(",") if s.strip()]
    ratings = (
        None
        if args.rating == "any"
        else {r.strip() for r in args.rating.split(",") if r.strip()}
    )
    pairs = pick_pairs(
        args.n_pairs, args.seed, args.pair_scope, args.max_delta, ratings
    )

    run_dir = make_run_dir(
        "directedit_ec",
        root=Path(__file__).resolve().parent / "results",
        label=args.label,
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger.info("Run dir: %s", run_dir)

    scales = [float(s) for s in args.anchor_scales.split(",") if s.strip()]

    # Arm = (name, kind, ec_mode, b_offset, anchor_scale, src, invert_b_offset);
    # kind: "ff" = inference.py, "inv" = edit.py. λ=1 caption arms keep the
    # run-1 names (no suffixes) so cross-run comparison stays greppable.
    if args.arms_spec:
        arms = parse_arms_spec(args.arms_spec)
    else:
        arms = []
        for off in offsets:
            if not args.skip_easyedit:
                arms.append(
                    (
                        f"easyedit_b{off:g}",
                        "ff",
                        None,
                        off,
                        1.0,
                        "caption",
                        None,
                        "delta",
                    )
                )
        if not args.skip_noec:
            arms.append(("inv_noec", "inv", None, 0.0, 1.0, "caption", None, "delta"))
        for off in offsets:
            for lam in scales:
                for mode in ("both", "edit"):
                    arms.append(
                        (
                            arm_name("inv", mode, off, lam, "caption", None),
                            "inv",
                            mode,
                            off,
                            lam,
                            "caption",
                            None,
                            "delta",
                        )
                    )

    per_pair = []
    sheet_rows = []
    for pair in pairs:
        src = resolve(pair["cond"])  # A — the image being edited in place
        target = resolve(pair["target"])  # B — what the delta was mined toward
        psi_src = caption_of(pair["cond"])
        delta = args.delta_override or pair["delta_caption"]
        size_hw = pick_size(src)
        key = pair["character"].replace(" ", "_").replace("/", "_")
        logger.info(
            "Pair %s: src=%s size=%s\n  ψ_src: %s\n  delta: %s",
            key,
            pair["cond"],
            size_hw,
            psi_src,
            delta,
        )
        rec = {
            "character": pair["character"],
            "source": str(src),
            "mined_target": str(target),
            "same_artist": pair["same_artist"],
            "n_additions": pair["n_additions"],
            "n_removals": pair["n_removals"],
            "psi_src": psi_src,
            "prompt": delta,
            "size_hw": list(size_hw),
            "arms": {},
        }
        cells: list[tuple[str, Optional[Path]]] = [
            ("source (A)", src),
            ("mined target (B)", target),
        ]
        for name, kind, ec_mode, off, lam, src_mode, ioff, tar_mode in arms:
            out_dir = run_dir / "renders" / key / name
            out_dir.mkdir(parents=True, exist_ok=True)
            # tar=caption: the mined target's full caption == ψ_src with the
            # delta applied (that is how the delta was mined). tar=srccap:
            # source caption + ", <delta>" — the minimal-edit form. Both are
            # descriptive ψ_tar for edit passes running the base model.
            if tar_mode == "delta":
                tar_prompt = delta
            elif tar_mode == "srccap":
                tar_prompt = f"{psi_src}, {delta}"
            else:
                tar_prompt = caption_of(pair["target"])
            if kind == "ff":
                argv = easyedit_argv(tar_prompt, src, size_hw, off, out_dir, args, ck)
            else:
                psi = psi_src if src_mode == "caption" else ""
                argv = invert_argv(
                    psi,
                    tar_prompt,
                    src,
                    size_hw,
                    ec_mode,
                    off,
                    lam,
                    ioff,
                    out_dir,
                    args,
                    ck,
                )
            log_path = logs_dir / f"{key}_{name}.log"
            logger.info("[%s/%s] running", key, name)
            t0 = time.time()
            try:
                with log_path.open("w") as lf:
                    lf.write(" ".join(argv) + "\n\n")
                    lf.flush()
                    proc = subprocess.run(
                        argv,
                        cwd=REPO_ROOT,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                    )
                ok = proc.returncode == 0
            except subprocess.TimeoutExpired:
                ok = False
                logger.error("[%s/%s] TIMEOUT", key, name)
            pngs = sorted(out_dir.glob("*.png"))
            out_png = pngs[-1] if (ok and pngs) else None
            rec["arms"][name] = {
                "ok": out_png is not None,
                "wall_s": round(time.time() - t0, 1),
                "png": str(out_png.relative_to(run_dir)) if out_png else None,
                "mse_vs_src": mse_vs_src(out_png, src) if out_png else None,
                **parse_arm_log(log_path),
            }
            cells.append((name, out_png))
            logger.info(
                "[%s/%s] %s wall=%.0fs",
                key,
                name,
                "ok" if out_png else "FAILED",
                time.time() - t0,
            )
        per_pair.append(rec)
        sheet_rows.append((key, cells))

    sheet = run_dir / "grid.png"
    make_sheet(sheet_rows, sheet)
    write_result(
        run_dir,
        script=str(Path(__file__).relative_to(REPO_ROOT)),
        args=args,
        metrics={
            "ec_weight": args.ec_weight,
            "arms": [a for a, *_ in arms],
            "per_pair": per_pair,
            "note": "judged on (1) composition of A preserved, (2) instructed "
            "changes land, (3) identity holds; mse_vs_src = composition "
            "proxy; ψ_src = ground-truth caption (tagger is the deployment "
            "source); train-pair delta = upper bound",
        },
        artifacts=["grid.png"],
    )
    logger.info("Human verdict artifact: %s", sheet)
    print(json.dumps({"run_dir": str(run_dir)}))


if __name__ == "__main__":
    main()
