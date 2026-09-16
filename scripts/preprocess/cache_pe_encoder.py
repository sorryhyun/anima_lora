#!/usr/bin/env python3
"""Cache PE-Core (or other registered vision-encoder) features.

Mirrors the live PE encoding done at training time so callers can read
patch-token features off disk instead of running the encoder every step.
Loads each pre-resized image from ``--dir`` in [-1, 1], picks the
encoder's nearest-aspect bucket, runs a single forward, and saves
``{stem}_anima_{encoder}.safetensors`` into ``--cache_dir`` (or alongside
the image when omitted). Skips already-cached entries (idempotent).

Wrapped by ``make preprocess-pe`` (reads ``post_image_dataset/resized/``,
writes ``post_image_dataset/lora/``). The same sidecars are consumed by
IP-Adapter -- it shares the cache directory.

The cache key matches what the encoder produces at training time:
``encode_pe_from_imageminus1to1(bundle, x, same_bucket=True)`` -> ``[T_pe, d_enc]``.
Variable T per encoder bucket; per-image stored as a single tensor (no padding).

The walk → group → encode → save loop and the centroid pooling pass live in
``library/preprocess/pe.py``; this file is argparse + encoder load + reporting.

Centroid sidecar
----------------

Pass ``--centroid`` to also emit ``anima_pe_centroid_{encoder}.safetensors``
(dataset-mean of mean-over-patch-tokens pooled features, ``[D]`` fp32) after
the cache pass. Pass ``--centroid_only`` to skip encoding entirely and just
pool existing caches under ``--cache_dir``. Consumed by IP-Adapter
(``ip_centroid_path``) -- targets the participation-ratio-6 manifold collapse
on this dataset (see ``_archive/bench/ip_adapter/analysis.md``).
"""

import argparse
import sys
from pathlib import Path

import torch


from library.preprocess import (
    cache_pe_features,
    count_pending_pe,
    tqdm_progress,
    write_pe_centroid,
)
from library.runtime.argparse_groups import add_device_args, add_io_args
from library.vision.encoder import load_pe_encoder

ROOT = Path(__file__).resolve().parents[2]


def _default_centroid_out(encoder: str) -> Path:
    return (
        ROOT
        / "post_image_dataset"
        / "ip_adapter"
        / f"anima_pe_centroid_{encoder}.safetensors"
    )


def _report_centroid(n: int, centroid: torch.Tensor, out_path: Path) -> None:
    print(
        f"centroid shape: {tuple(centroid.shape)}  "
        f"‖centroid‖={float(centroid.norm()):.3f}  "
        f"mean={float(centroid.mean()):.4f}  std={float(centroid.std()):.4f}"
    )
    print(f"wrote {out_path}  (pooled {n} images)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_io_args(
        parser,
        dir_required=False,
        dir_help="Dataset directory. Required unless --centroid_only is set.",
        cache_noun="PE caches",
        include_batch_size=True,
        batch_size_default=8,
        include_num_workers=True,
        num_workers_default=4,
    )
    # PE features are consumed as fp32 everywhere (REPA upcasts the cache at
    # repa.py, CMMD pools in fp32), so cache at full precision by default.
    # --dtype drives both the encoder compute dtype and the on-disk save dtype.
    add_device_args(
        parser,
        include_device=False,
        dtype_default="float32",
        dtype_choices=("float32", "bfloat16", "float16"),
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="pe",
        help="Vision encoder registry name (default: pe). See library/vision/encoders.py.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Override the encoder's default model id / checkpoint path.",
    )
    parser.add_argument(
        "--centroid",
        action="store_true",
        help=(
            "After the cache pass, stream-pool all '_anima_{encoder}.safetensors' "
            "files under --cache_dir and emit a dataset-mean centroid sidecar "
            "consumed by IP-Adapter. Requires --cache_dir."
        ),
    )
    parser.add_argument(
        "--centroid_only",
        action="store_true",
        help=(
            "Skip encoding; just pool existing PE caches under --cache_dir and "
            "write the centroid sidecar. --cache_dir defaults to "
            "'post_image_dataset/lora' in this mode."
        ),
    )
    parser.add_argument(
        "--centroid_out",
        type=str,
        default=None,
        help=(
            "Output path for the centroid sidecar. Defaults to "
            "post_image_dataset/ip_adapter/anima_pe_centroid_{encoder}.safetensors "
            "(separate from the shared PE cache dir so LoRA stays untouched)."
        ),
    )
    parser.add_argument(
        "--centroid_limit",
        type=int,
        default=0,
        help="Cap the number of cache files pooled into the centroid (0 = all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Re-encode every PE sidecar even if it already exists. Use after an "
            "encoder / dtype change, which the existence skip can't detect."
        ),
    )
    args = parser.parse_args()

    if not args.centroid_only and args.dir is None:
        parser.error("--dir is required unless --centroid_only is set")
    if args.centroid and not args.cache_dir:
        parser.error(
            "--centroid needs --cache_dir (centroid pools files in a directory; "
            "alongside-image layout has no single dir to walk)"
        )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.centroid_only:
        centroid_cache_dir = cache_dir or (ROOT / "post_image_dataset" / "lora")
        if not centroid_cache_dir.is_absolute():
            centroid_cache_dir = (ROOT / centroid_cache_dir).resolve()
        if not centroid_cache_dir.is_dir():
            print(f"--cache_dir not found: {centroid_cache_dir}", file=sys.stderr)
            sys.exit(1)
        out_path = (
            Path(args.centroid_out)
            if args.centroid_out
            else _default_centroid_out(args.encoder)
        )
        try:
            n, centroid = write_pe_centroid(
                centroid_cache_dir,
                out_path,
                encoder=args.encoder,
                limit=args.centroid_limit,
            )
        except (FileNotFoundError, ValueError) as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        _report_centroid(n, centroid, out_path)
        return

    data_dir = Path(args.dir)
    if not data_dir.is_dir():
        print(f"--dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Pre-flight: skip the encoder load when every sidecar exists (--centroid
    # still runs — caches only). total == 0 falls through to the "no images" path.
    pending, total = count_pending_pe(
        data_dir,
        args.encoder,
        cache_dir=cache_dir,
        recursive=args.recursive,
        overwrite=args.overwrite,
    )
    if total > 0 and pending == 0:
        print(
            f"{args.encoder} feature caching: all {total} images already cached "
            "— skipping encoder load."
        )
        if args.centroid:
            out_path = (
                Path(args.centroid_out)
                if args.centroid_out
                else _default_centroid_out(args.encoder)
            )
            n, centroid = write_pe_centroid(
                cache_dir, out_path, encoder=args.encoder, limit=args.centroid_limit
            )
            _report_centroid(n, centroid, out_path)
        return

    from library.runtime.backend import warn_if_cuda_unavailable

    warn_if_cuda_unavailable(torch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(
        f"Loading vision encoder '{args.encoder}' on {device} (dtype: {save_dtype}) ..."
    )
    bundle = load_pe_encoder(
        device, name=args.encoder, model_id=args.model_id, dtype=save_dtype
    )
    print(
        f"  encoder={bundle.name} d_enc={bundle.d_enc} "
        f"patch={bundle.bucket_spec.patch} cls={bundle.bucket_spec.use_cls}"
    )

    stats = cache_pe_features(
        data_dir,
        bundle,
        cache_dir=cache_dir,
        recursive=args.recursive,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_dtype=save_dtype,
        progress=tqdm_progress(f"Caching {bundle.name} features"),
        overwrite=args.overwrite,
    )
    if stats.seen == 0:
        print(f"No images found in {data_dir}/", file=sys.stderr)
        sys.exit(1)
    print(
        f"\n{bundle.name} feature caching complete: "
        f"{stats.written} cached, {stats.skipped} skipped"
    )

    if args.centroid:
        out_path = (
            Path(args.centroid_out)
            if args.centroid_out
            else _default_centroid_out(bundle.name)
        )
        n, centroid = write_pe_centroid(
            cache_dir, out_path, encoder=bundle.name, limit=args.centroid_limit
        )
        _report_centroid(n, centroid, out_path)


if __name__ == "__main__":
    main()
