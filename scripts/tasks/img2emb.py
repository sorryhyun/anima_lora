"""img2emb resampler training pipeline (preprocess / pretrain / finetune / infer).

TIPSv2-L/14 vision encoder by default; ENCODER env overrides. Aspect-preserving
bucketed preprocessing is always on — images are resized to the closest
patch-14 bucket (~1024 tokens, aspects 1:2..2:1); tokens zero-padded to a
single T_MAX so the cache stays a flat (N, T_MAX, D) tensor. See
``scripts/img2emb/{preprocess,pretrain,finetune,infer}.py``.
"""

from __future__ import annotations

import os
import sys

from ._common import PY, run


def cmd_img2emb(extra):
    """Usage: python tasks.py img2emb [preprocess|anchors|pretrain|finetune] [extra...]

    Bare `img2emb` runs pretrain + finetune (preprocess + anchors are a separate
    preprocessing step — `make preprocess-img2emb` / `python tasks.py preprocess-img2emb`).
    """
    stage = extra[0] if extra else None
    rest = extra[1:] if extra else []
    if stage is None:
        run([PY, "scripts/img2emb/pretrain.py"])
        run([PY, "scripts/img2emb/finetune.py"])
        return
    if stage == "anchors":
        run([PY, "scripts/img2emb/rebuild_anchor_artifacts.py", *rest])
        return
    if stage not in ("preprocess", "pretrain", "finetune"):
        print(
            f"Unknown img2emb stage '{stage}' "
            f"(use: preprocess | anchors | pretrain | finetune)",
            file=sys.stderr,
        )
        sys.exit(1)
    run([PY, f"scripts/img2emb/{stage}.py", *rest])


def _encoder_args() -> list[str]:
    """Return ``['--encoder', $ENCODER]`` if ENCODER env is set, else ``[]``.

    Forwarded to img2emb stage scripts so the cache, pretrain ckpt and finetune
    ckpt filenames stay coherent across stages.
    """
    enc = os.environ.get("ENCODER")
    return ["--encoder", enc] if enc else []


def cmd_img2emb_preprocess(extra):
    run([PY, "scripts/img2emb/preprocess.py", *_encoder_args(), *extra])


def cmd_img2emb_anchors(extra):
    run([PY, "scripts/img2emb/rebuild_anchor_artifacts.py", *extra])


def cmd_img2emb_align(extra):
    """One-shot Hungarian alignment of T5 variants v1..vN to v0.

    Runs implicitly inside ``img2emb-preprocess``; this entry point is for
    re-running standalone (idempotent).
    """
    run([PY, "scripts/img2emb/align_variants.py", *extra])


def cmd_img2emb_pretrain(extra):
    run([PY, "scripts/img2emb/pretrain.py", *_encoder_args(), *extra])


def cmd_img2emb_finetune(extra):
    run([PY, "scripts/img2emb/finetune.py", *_encoder_args(), *extra])


def cmd_img2emb_calibrate(extra):
    """Step-0 loss-weight calibration for img2emb finetune (no backward).

    Reports raw loss magnitudes so the ``loss.*`` weights in finetune.py can be
    tuned. Warm path defaults to the pretrain output for the selected ENCODER
    (``tipsv2`` unless overridden); set FINETUNE_WARM=... to point elsewhere.
    FINETUNE_BS, FINETUNE_SWAP env vars are forwarded as --batch_size /
    --blocks_to_swap (default ``1`` and ``-1`` — gradient checkpointing — for
    16 GB cards).
    """
    encoder = os.environ.get("ENCODER", "tipsv2")
    warm = os.environ.get(
        "FINETUNE_WARM",
        f"output/img2embs/pretrain/{encoder}_resampler_4layer_anchored.safetensors",
    )
    bs = os.environ.get("FINETUNE_BS", "1")
    swap = os.environ.get("FINETUNE_SWAP", "-1")
    run(
        [
            PY,
            "scripts/img2emb/finetune.py",
            "--dit",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--warm_start",
            warm,
            "--calibrate_only",
            "--batch_size",
            bs,
            "--blocks_to_swap",
            swap,
            *_encoder_args(),
            *extra,
        ]
    )


def cmd_preprocess_img2emb(extra):
    run([PY, "scripts/img2emb/preprocess.py", *_encoder_args(), *extra])
    run([PY, "scripts/img2emb/rebuild_anchor_artifacts.py"])


def cmd_test_img2emb(extra):
    """Generate from a ref image via the finetuned resampler.

    Reference image from REF_IMAGE env or first positional arg. ENCODER env is
    forwarded as ``--encoder``.
    """
    ref = os.environ.get("REF_IMAGE", "").strip()
    if not ref and extra and not extra[0].startswith("-"):
        ref = extra[0]
        extra = extra[1:]
    if not ref:
        print(
            "Usage: python tasks.py test-img2emb <ref_image> [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py test-img2emb [extra...]",
            file=sys.stderr,
        )
        sys.exit(1)
    run([PY, "scripts/img2emb/infer.py", "--ref_image", ref, *_encoder_args(), *extra])
