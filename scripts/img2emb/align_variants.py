#!/usr/bin/env python
"""Hungarian-align V T5 variants to v0 in cached ``_anima_te.safetensors``.

The caption pipeline produces V variants per image by *shuffling* booru
tags and re-encoding through T5. T5's positional encoding makes shuffles
produce wildly different per-token outputs even though the content
(set of tokens) is identical — measured ~0.33 inter-variant per-token
cosine without alignment, but ~0.975 *with* Hungarian alignment.

Position-wise MSE / cosine against ``variant_mean`` is therefore
pathological: averaging over random permutations smears the positional
structure to a noise centroid, and the resampler is graded on a target
it cannot match.

This script aligns variants v1..v{V-1} to v0 via Hungarian matching on
per-token cosine similarity. After alignment, all V variants share v0's
token ordering — ``variant_mean`` becomes meaningful and positional
losses become well-formed.

Idempotent: writes a marker tensor ``aligned_to_v0`` so re-running is a
no-op. The script permutes every per-variant tensor in lockstep
(``crossattn_emb``, ``prompt_embeds``, ``attn_mask``, ``t5_attn_mask``,
``t5_input_ids``) so caption text stays consistent with embeddings.

Usage:
    python scripts/img2emb/align_variants.py
    python scripts/img2emb/align_variants.py --image_dir post_image_dataset
    make img2emb-align
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


MARKER_KEY = "aligned_to_v0"
# ONLY crossattn_emb is permuted. The DiT consumes crossattn_emb directly via
# cross-attention, which has no K-side RoPE (see library/anima/models.py
# compute_qkv: rotary applied only when is_selfattn). So row permutation is a
# mathematical no-op for the LoRA training and inference paths that read
# `crossattn_emb_v*` (the cache_llm_adapter_outputs=True path).
#
# We deliberately do NOT permute `prompt_embeds_v*`, `t5_input_ids_v*`, or the
# `*_attn_mask_v*` keys: those feed the LLM adapter at training time
# (cache_llm_adapter_outputs=False path), and the adapter's internal cross-attn
# DOES apply K-side RoPE. Permuting them would change adapter outputs and
# break LoRA training that re-runs the adapter at runtime. Leaving them as-is
# means the cache becomes internally inconsistent (prompt_embeds_v1 no longer
# round-trips through the adapter to crossattn_emb_v1) but no code path reads
# both at once, so the inconsistency is harmless in practice.
PER_VARIANT_PREFIXES = ("crossattn_emb_v",)


def _active_length(state: dict[str, torch.Tensor], v: int) -> int:
    """Active token count for variant ``v``, derived from the attention mask
    when present and otherwise from the first all-zero row of crossattn_emb."""
    mask = state.get(f"attn_mask_v{v}")
    if mask is not None:
        return int(mask.sum().item())
    emb = state[f"crossattn_emb_v{v}"]
    nonzero = (emb.float().abs().sum(dim=-1) > 0).long()
    return int(nonzero.sum().item())


def align_te_state(
    state: dict[str, torch.Tensor], num_variants: int
) -> tuple[dict[str, torch.Tensor], dict]:
    """Align variants 1..V-1 to v0 in ``state``. Returns ``(new_state, diag)``.

    ``state`` is unchanged; the returned dict is a shallow copy with
    permuted per-variant tensors and a marker. Diagnostics include the
    pre/post per-token cosine averages so the caller can spot pathological
    cases (mostly: V mismatched lengths, or already-aligned files where
    cosine doesn't change).
    """
    new = dict(state)
    if MARKER_KEY in new:
        return new, {"already_aligned": True}

    L0 = _active_length(state, 0)
    if L0 == 0:
        # Nothing to align (empty caption?) — mark and skip.
        new[MARKER_KEY] = torch.tensor([1], dtype=torch.uint8)
        return new, {"empty": True}

    ref = state["crossattn_emb_v0"].float()[:L0]
    ref_n = F.normalize(ref, dim=-1, eps=1e-8)

    cos_before, cos_after = [], []
    for v in range(1, num_variants):
        emb_key = f"crossattn_emb_v{v}"
        if emb_key not in state:
            continue
        Lv = _active_length(state, v)
        L = min(L0, Lv)
        # Hungarian on the overlapping prefix; tail (if Lv > L0 or vice versa)
        # is left in original order. In practice L0 == Lv almost always for
        # caption-shuffle variants of the same image.
        e = state[emb_key].float()[:L]
        e_n = F.normalize(e, dim=-1, eps=1e-8)
        sim = (ref_n[:L] @ e_n.T).cpu().numpy()                     # (L, L)
        cos_before.append(float((ref_n[:L] * e_n).sum(-1).mean().item()))
        _, col = linear_sum_assignment(-sim)                        # maximize cos
        col_t = torch.as_tensor(col, dtype=torch.long)
        cos_after.append(
            float((ref_n[:L] * e_n[col_t]).sum(-1).mean().item())
        )

        # Apply the same permutation to every per-variant tensor.
        for prefix in PER_VARIANT_PREFIXES:
            key = f"{prefix}{v}"
            if key not in state:
                continue
            t = state[key]
            permuted = t.clone()
            permuted[:L] = t[:L][col_t]
            new[key] = permuted

    new[MARKER_KEY] = torch.tensor([1], dtype=torch.uint8)
    diag = {
        "L0": L0,
        "cos_before_mean": (sum(cos_before) / len(cos_before)) if cos_before else 1.0,
        "cos_after_mean": (sum(cos_after) / len(cos_after)) if cos_after else 1.0,
    }
    return new, diag


def _is_aligned(path: Path) -> bool:
    """Cheap marker check via safetensors header peek — no tensor data read."""
    try:
        with safe_open(str(path), framework="pt") as h:
            return MARKER_KEY in list(h.keys())
    except Exception:
        return False


def align_te_file(path: Path, num_variants: int) -> tuple[bool, dict]:
    """Align variants in ``path`` in place. Returns ``(modified, diag)``.

    Fast path: if the marker is already present, returns immediately without
    loading the file's tensor data.
    """
    if _is_aligned(path):
        return False, {"already_aligned": True}
    state = load_file(str(path))
    new, diag = align_te_state(state, num_variants)
    if diag.get("already_aligned"):
        return False, diag
    save_file(new, str(path))
    return True, diag


def _worker(args_tuple: tuple[str, int]) -> tuple[str, bool, dict]:
    path_str, V = args_tuple
    modified, diag = align_te_file(Path(path_str), V)
    return path_str, modified, diag


def align_te_dir(
    image_dir: Path,
    stems: list[str],
    num_variants: int,
    num_workers: int = 8,
) -> dict:
    """Align every ``{stem}_anima_te.safetensors`` under ``image_dir``.
    Returns aggregate diagnostics.

    Two-phase scan: a sequential header-peek pass partitions files into
    already-aligned (skip) vs. needs-work, then only the needs-work subset
    is fanned out to the process pool. Header peeks are metadata-only
    safetensors reads — orders of magnitude cheaper than ``load_file``,
    so caches written by the current TE encoder (which aligns on write)
    incur essentially no I/O cost when this is invoked defensively.
    """
    needs_work: list[tuple[str, int]] = []
    n_already = 0
    missing = 0
    for stem in tqdm(stems, desc="scan markers", dynamic_ncols=True, leave=False):
        p = image_dir / f"{stem}_anima_te.safetensors"
        if not p.exists():
            missing += 1
            continue
        if _is_aligned(p):
            n_already += 1
        else:
            needs_work.append((str(p), num_variants))

    n_modified = 0
    cos_before, cos_after = [], []

    if not needs_work:
        return {
            "n_total": len(stems),
            "n_modified": 0,
            "n_already_aligned": n_already,
            "n_missing": missing,
            "cos_before_mean": float("nan"),
            "cos_after_mean": float("nan"),
        }

    if num_workers <= 1:
        it = (_worker(t) for t in needs_work)
    else:
        # Process pool — Hungarian is CPU-bound and pure-numpy, GIL is fine
        # to release across workers.
        ex = ProcessPoolExecutor(max_workers=num_workers)
        futs = [ex.submit(_worker, t) for t in needs_work]
        it = (f.result() for f in as_completed(futs))

    for _path, modified, diag in tqdm(
        it, total=len(needs_work), desc="align variants", dynamic_ncols=True
    ):
        if diag.get("already_aligned"):
            n_already += 1
            continue
        if modified:
            n_modified += 1
        if "cos_before_mean" in diag:
            cos_before.append(diag["cos_before_mean"])
            cos_after.append(diag["cos_after_mean"])

    if num_workers > 1:
        ex.shutdown(wait=True)

    summary = {
        "n_total": len(stems),
        "n_modified": n_modified,
        "n_already_aligned": n_already,
        "n_missing": missing,
        "cos_before_mean": (
            float(sum(cos_before) / len(cos_before)) if cos_before else float("nan")
        ),
        "cos_after_mean": (
            float(sum(cos_after) / len(cos_after)) if cos_after else float("nan")
        ),
    }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--image_dir",
        default="post_image_dataset",
        help="Directory containing {stem}_anima_te.safetensors caches.",
    )
    p.add_argument(
        "--stems_json",
        default=str(REPO_ROOT / "output" / "img2embs" / "features" / "stems.json"),
        help="Optional ordered stems list. Defaults to the img2emb features "
             "stems.json. Falls back to scanning image_dir if absent.",
    )
    p.add_argument(
        "--num_variants",
        type=int,
        default=None,
        help="V — defaults to active_lengths.json[num_variants] when present, "
             "else 4.",
    )
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = REPO_ROOT / image_dir

    stems_path = Path(args.stems_json)
    if stems_path.exists():
        stems = json.loads(stems_path.read_text())
    else:
        # Fallback: scan the image_dir for cached TE files.
        stems = sorted(
            p.stem.replace("_anima_te", "")
            for p in image_dir.glob("*_anima_te.safetensors")
        )
        logger.info(f"  scanned {len(stems)} stems from {image_dir}")

    V = args.num_variants
    if V is None:
        act_path = REPO_ROOT / "output" / "img2embs" / "features" / "active_lengths.json"
        if act_path.exists():
            V = int(json.loads(act_path.read_text()).get("num_variants", 4))
        else:
            V = 4
    logger.info(f"  V={V} stems={len(stems)} image_dir={image_dir}")

    summary = align_te_dir(image_dir, stems, V, num_workers=args.num_workers)
    logger.info(
        f"done: modified={summary['n_modified']} "
        f"already_aligned={summary['n_already_aligned']} "
        f"missing={summary['n_missing']}"
    )
    logger.info(
        f"  per-token cos before={summary['cos_before_mean']:.4f} "
        f"after={summary['cos_after_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
