#!/usr/bin/env python
"""Tag-conditional slot subspace bench for Anima T5 crossattn embeddings.

Decomposes per-slot variance into "explained by known tag identity" + "residual"
so future inversion runs can anchor high-signal slots (rating / count-meta /
artist) to data-derived prototypes instead of optimizing them from scratch.

See ``bench/inversionv2/bench_plan.md`` for full design, motivation, and the
interpretation guide. Run:

    python bench/inversionv2/tag_slot_analysis.py
    python bench/inversionv2/tag_slot_analysis.py --max_images 50   # smoke
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from library.io.cache import discover_cached_images  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

# Rating tags that actually occur at tag-index 0 in this dataset. No
# `questionable` — zero occurrences in 1987 captions.
DEFAULT_RATINGS = ["explicit", "sensitive", "general", "absurdres"]
DEFAULT_COUNT_META = [
    "1girl", "1boy", "2girls", "2boys", "3girls", "1other",
    "solo", "multiple_girls", "multiple_boys",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image_dir", default="post_image_dataset")
    p.add_argument(
        "--tokenizer",
        default="library/anima/configs/t5_old",
        help="T5 tokenizer dir. Slots are T5-indexed, NOT Qwen3.",
    )
    p.add_argument(
        "--variants",
        default="all",
        help="'all' to iterate v0..v7, or a single integer (e.g. '0').",
    )
    p.add_argument("--min_artist_images", type=int, default=5)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument(
        "--top_suffix_tags",
        type=int,
        default=10,
        help="How many high-frequency post-@artist tags to include in phase 4.",
    )
    p.add_argument("--top_k_report", type=int, default=16)
    p.add_argument(
        "--cosine_sample_cap",
        type=int,
        default=64,
        help="Cap n for pairwise-cosine stats to avoid O(n^2) blowup.",
    )
    p.add_argument(
        "--output_dir",
        default=str(BENCH_DIR / "results" / "tag_slot"),
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device for SVDs / pairwise cosines (e.g. 'cuda'). File IO and "
        "tokenization stay CPU either way — GPU only helps analysis phases.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- helpers


def iter_variants(spec: str) -> list[int]:
    if spec == "all":
        return list(range(8))
    return [int(spec)]


def split_tags(caption: str) -> list[tuple[str, int, int]]:
    """Split caption into (tag, char_start, char_end) tuples.

    Robust against leading whitespace after commas, empty segments, and
    substrings (matches are comma-scoped, not raw ``find``).
    """
    out = []
    cursor = 0
    for raw in caption.split(","):
        tag = raw.strip()
        # skip forward past leading whitespace for the char start
        seg_start = cursor
        seg_end = cursor + len(raw)
        if tag:
            # Leading whitespace in raw means tag starts later in the caption.
            lws = len(raw) - len(raw.lstrip())
            char_start = seg_start + lws
            char_end = char_start + len(tag)
            out.append((tag, char_start, char_end))
        cursor = seg_end + 1  # +1 for the comma
    return out


def tokens_in_range(offset_mapping, char_start: int, char_end: int) -> list[int]:
    """Return token indices whose offset falls strictly inside [char_start, char_end).

    Skips special tokens (offsets == (0, 0)) and tokens that straddle the tag
    boundary. The first/last token of a tag are ``[slots[0], slots[-1]]``.
    """
    slots = []
    for i, (s, e) in enumerate(offset_mapping):
        if s == e == 0:  # special / padding
            continue
        if s >= char_start and e <= char_end:
            slots.append(i)
    return slots


def pairwise_cosine_stats(X: torch.Tensor, cap: int = 64) -> dict:
    """Percentile dict over off-diagonal cosine sims of rows of X."""
    n = X.shape[0]
    if n < 2:
        return {"median": float("nan"), "mean": float("nan"), "p10": float("nan"), "p90": float("nan"), "n": n}
    if n > cap:
        idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:cap]
        X = X[idx]
        n = cap
    Xn = X / (X.norm(dim=-1, keepdim=True) + 1e-9)
    S = Xn @ Xn.T
    mask = ~torch.eye(n, dtype=torch.bool, device=S.device)
    vals = S[mask]
    return {
        "median": float(vals.median().item()),
        "mean": float(vals.mean().item()),
        "p10": float(torch.quantile(vals, 0.10).item()),
        "p90": float(torch.quantile(vals, 0.90).item()),
        "n": n,
    }


def effective_rank(sv_sq: np.ndarray, thresholds=(0.80, 0.90, 0.95, 0.99)) -> dict:
    total = float(sv_sq.sum())
    if total <= 0:
        return {f"k_{int(t * 100)}": 0 for t in thresholds}
    cum = np.cumsum(sv_sq) / total
    return {
        f"k_{int(t * 100)}": int(np.searchsorted(cum, t) + 1)
        for t in thresholds
    }


def position_hist(positions: list[int]) -> dict:
    if not positions:
        return {}
    arr = np.asarray(positions, dtype=np.int64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": int(arr.min()),
        "max": int(arr.max()),
    }


# --------------------------------------------------------------------------- phase 1 + shared extraction


def process_variant(
    stem: str,
    vi: int,
    sd: dict[str, torch.Tensor],
    tokenizer,
    zero_eps: float = 1e-6,
):
    """Decode + re-tokenize one (image, variant); extract tag → slot map.

    Returns ``None`` if the round-trip assertion fails (tokenizer drift).
    """
    ids_key = f"t5_input_ids_v{vi}"
    emb_key = f"crossattn_emb_v{vi}"
    if ids_key not in sd or emb_key not in sd:
        return None
    ids = sd[ids_key].tolist()
    while ids and ids[-1] == 0:
        ids.pop()
    if not ids:
        return None

    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    # Re-tokenize with offsets. max_length = len(ids) + some slack so we never truncate.
    enc = tokenizer(
        decoded,
        return_offsets_mapping=True,
        truncation=True,
        max_length=len(ids) + 8,
        add_special_tokens=True,
    )
    if enc["input_ids"] != ids:
        # Known cause: cached ids contain <unk> (id 2) for chars the SentencePiece
        # model can't encode (e.g. Japanese subtitles in series names). decode()
        # with skip_special_tokens=True drops <unk>, so round-trip shortens. This
        # is a data property, not tokenizer drift — skip silently.
        # Anything else is a real drift problem.
        has_unk = 2 in ids
        return {"_skip": True, "has_unk": has_unk}

    emb = sd[emb_key].to(torch.float32)
    nonzero = emb.abs().amax(dim=-1) > zero_eps
    n_content = (
        int(nonzero.nonzero(as_tuple=False)[-1].item()) + 1 if nonzero.any() else 0
    )

    tags_with_slots = []
    for tag, cs, ce in split_tags(decoded):
        slots = tokens_in_range(enc["offset_mapping"], cs, ce)
        if not slots:
            continue
        tags_with_slots.append((tag, slots[0], slots[-1], len(slots)))

    return {
        "caption": decoded,
        "tags": tags_with_slots,
        "emb": emb,
        "n_content": n_content,
    }


def build_tag_data(images, tokenizer, variants, logger):
    """Single-pass extraction feeding phases 2–4.

    Returns dict with:
        positions_per_class: {class_name: [(stem, vi, slot_first, slot_last, n_tok), ...]}
        class_rows: {class_name: [(stem, vi, slot, vec_fp32), ...]}  # for phase 2
        artist_rows: {artist_name: [(stem, vi, slot, vec_fp32), ...]}  # for phase 3
        per_caption_class: {(class, stem): [(vi, slot, vec), ...]}  # for phase 4
        suffix_tag_rows: {tag: {stem: [(vi, slot, vec), ...]}}  # phase 4 suffix slice
        content_lengths: [n_content, ...]
        rt_failures: int
        total_ok: int
    """
    positions_per_class = defaultdict(list)
    class_rows = defaultdict(list)
    artist_rows = defaultdict(list)
    per_caption_class = defaultdict(list)
    suffix_tag_rows = defaultdict(lambda: defaultdict(list))
    content_lengths = []
    rt_failures = 0
    rt_fail_unk = 0
    total_ok = 0

    for idx, img in enumerate(images):
        try:
            sd = load_file(img.te_path)
        except Exception as e:
            logger.warning(f"cache load failed {img.te_path}: {e}")
            continue

        for vi in variants:
            result = process_variant(img.stem, vi, sd, tokenizer)
            if result is None:
                rt_failures += 1
                continue
            if result.get("_skip"):
                rt_failures += 1
                if result.get("has_unk"):
                    rt_fail_unk += 1
                continue
            total_ok += 1
            emb = result["emb"]
            tags_list = result["tags"]
            content_lengths.append(result["n_content"])

            # Find the @artist boundary (prefix ends here by the shuffle convention)
            artist_ti = -1
            for ti, (tag, *_rest) in enumerate(tags_list):
                if tag.startswith("@"):
                    artist_ti = ti
                    break

            for ti, (tag, s0, s1, nt) in enumerate(tags_list):
                vec = emb[s0].clone().cpu()  # (1024,) fp32
                entry = (img.stem, vi, s0, s1, nt)

                # rating: always tag-index 0
                if ti == 0 and tag in DEFAULT_RATINGS:
                    cname = f"rating={tag}"
                    positions_per_class[cname].append(entry)
                    class_rows[cname].append((img.stem, vi, s0, vec))
                    per_caption_class[(cname, img.stem)].append((vi, s0, vec))

                # count-meta: match on exact tag string anywhere in prefix
                if tag in DEFAULT_COUNT_META:
                    positions_per_class[tag].append(entry)
                    class_rows[tag].append((img.stem, vi, s0, vec))
                    per_caption_class[(tag, img.stem)].append((vi, s0, vec))

                # artist: any @-prefixed tag
                if tag.startswith("@"):
                    positions_per_class["@artist"].append(entry + (tag,))
                    artist_rows[tag].append((img.stem, vi, s0, vec))
                    per_caption_class[("@artist", img.stem)].append((vi, s0, vec))

                # suffix-tag tracking: anything after the @-tag
                if artist_ti >= 0 and ti > artist_ti:
                    suffix_tag_rows[tag][img.stem].append((vi, s0, vec))

        if (idx + 1) % 200 == 0:
            logger.info(
                f"  processed {idx + 1}/{len(images)}  "
                f"ok={total_ok}  rt_fail={rt_failures}"
            )

    return {
        "positions_per_class": positions_per_class,
        "class_rows": class_rows,
        "artist_rows": artist_rows,
        "per_caption_class": per_caption_class,
        "suffix_tag_rows": suffix_tag_rows,
        "content_lengths": content_lengths,
        "rt_failures": rt_failures,
        "rt_fail_unk": rt_fail_unk,
        "total_ok": total_ok,
    }


def phase1_summarize(positions_per_class: dict, content_lengths: list[int]) -> dict:
    """Position histograms per class + content-length distribution."""
    class_hists = {}
    for cname, entries in positions_per_class.items():
        # entry = (stem, vi, s0, s1, nt, [tag?])
        slots = [e[2] for e in entries]
        n_toks = [e[4] for e in entries]
        class_hists[cname] = {
            "occurrences": len(entries),
            "slot_position": position_hist(slots),
            "n_tokens_per_tag": position_hist(n_toks),
        }
    cl = np.asarray(content_lengths, dtype=np.int64)
    return {
        "per_class": class_hists,
        "content_length": {
            "n": int(cl.size),
            "mean": float(cl.mean()) if cl.size else 0.0,
            "p50": float(np.percentile(cl, 50)) if cl.size else 0.0,
            "p95": float(np.percentile(cl, 95)) if cl.size else 0.0,
            "p99": float(np.percentile(cl, 99)) if cl.size else 0.0,
            "max": int(cl.max()) if cl.size else 0,
        },
    }


# --------------------------------------------------------------------------- phase 2


def phase2_class_svd(class_rows: dict, args, logger, device) -> tuple[dict, dict]:
    spectra = {}
    prototypes = {}
    for cname, rows in class_rows.items():
        n = len(rows)
        if n < 4:
            spectra[cname] = {"n_samples": n, "skipped": True, "reason": "n<4"}
            continue
        X = torch.stack([r[3] for r in rows], dim=0).to(device)  # (n, 1024) fp32
        mean_vec = X.mean(dim=0)
        prototypes[cname] = mean_vec.detach().cpu().clone()

        try:
            _U, Sv, _Vh = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError as e:
            spectra[cname] = {"n_samples": n, "skipped": True, "error": str(e)}
            continue

        sv = Sv.detach().cpu().float().numpy()
        sv_sq = sv * sv
        er = effective_rank(sv_sq)
        row_norms = X.norm(dim=-1)
        cos_stats = pairwise_cosine_stats(X, cap=args.cosine_sample_cap)

        spectra[cname] = {
            "n_samples": n,
            "sv_top_k": sv[: args.top_k_report].tolist(),
            "total_energy": float(sv_sq.sum()),
            "row_norm_mean": float(row_norms.mean().item()),
            "row_norm_std": float(row_norms.std().item()),
            "mean_vec_norm": float(mean_vec.norm().item()),
            "mean_vec_norm_over_sv1": (
                float(mean_vec.norm().item() / sv[0]) if sv[0] > 0 else 0.0
            ),
            "pairwise_cosine": cos_stats,
            **er,
        }
        logger.info(
            f"  [{cname}] n={n}  k95={er['k_95']}  sv1={sv[0]:.1f}  "
            f"mean_norm={mean_vec.norm().item():.1f}  "
            f"cos_med={cos_stats['median']:.3f}"
        )
    return spectra, prototypes


# --------------------------------------------------------------------------- phase 3


def phase3_artists(
    artist_rows: dict, min_images: int, cosine_cap: int, logger, device
) -> tuple[dict, dict]:
    artist_stats = {}
    artist_means_raw: dict[str, torch.Tensor] = {}

    for artist, rows in artist_rows.items():
        unique_images = {r[0] for r in rows}
        if len(unique_images) < min_images:
            continue
        X = torch.stack([r[3] for r in rows], dim=0).to(device)
        mean_vec = X.mean(dim=0)
        within = pairwise_cosine_stats(X, cap=cosine_cap)
        artist_means_raw[artist] = mean_vec.detach().cpu()
        artist_stats[artist] = {
            "n_images": len(unique_images),
            "n_samples": len(rows),
            "mean_vec_norm": float(mean_vec.norm().item()),
            "within_cosine_median": within["median"],
            "within_cosine_mean": within["mean"],
            "within_cosine_p10": within["p10"],
        }

    if not artist_means_raw:
        logger.warning("No artists above min_artist_images; phase 3 skipped.")
        return {"artists": {}, "style_svd": None}, {}

    artists = list(artist_means_raw.keys())
    unit_means = torch.stack(
        [m / (m.norm() + 1e-9) for m in artist_means_raw.values()], dim=0
    ).to(device)  # (A, D)
    # between-cosine: each artist's mean against the mean of all OTHERS' means
    sim = unit_means @ unit_means.T  # (A, A)
    sim.fill_diagonal_(float("nan"))
    for i, a in enumerate(artists):
        row = sim[i]
        vals = row[~torch.isnan(row)]
        b_mean = float(vals.mean().item())
        b_max = float(vals.max().item())
        w = artist_stats[a]["within_cosine_median"]
        artist_stats[a]["between_cosine_mean"] = b_mean
        artist_stats[a]["between_cosine_max"] = b_max
        artist_stats[a]["within_over_between"] = (
            (w / b_mean) if b_mean not in (0.0,) and not np.isnan(b_mean) else None
        )

    # Style manifold SVD over unit means
    style_block = None
    try:
        _U, Sv_s, Vh_s = torch.linalg.svd(unit_means, full_matrices=False)
        svs = Sv_s.detach().cpu().float().numpy()
        # Project each artist's unit mean onto top components
        proj = unit_means @ Vh_s.T  # (A, k)
        top_components = []
        for k in range(min(3, proj.shape[1])):
            scores = proj[:, k]
            order = torch.argsort(scores.abs(), descending=True).tolist()
            top = [
                {"artist": artists[j], "score": float(scores[j].item())}
                for j in order[:5]
            ]
            top_components.append({
                "rank": k,
                "singular_value": float(svs[k]),
                "top_artists": top,
            })
        style_block = {
            "singular_values": svs.tolist(),
            "top_components": top_components,
            "n_artists": len(artists),
        }
    except RuntimeError as e:
        logger.warning(f"style SVD failed: {e}")

    # Ranking log
    ranked = sorted(
        artist_stats.items(),
        key=lambda kv: kv[1].get("within_over_between") or -1,
        reverse=True,
    )
    logger.info("  top-5 tightest artists (within/between):")
    for a, st in ranked[:5]:
        logger.info(
            f"    {a}: n={st['n_images']}  w={st['within_cosine_median']:.3f}  "
            f"b={st['between_cosine_mean']:.3f}  ratio={st['within_over_between']}"
        )

    return (
        {"artists": artist_stats, "style_svd": style_block},
        artist_means_raw,
    )


# --------------------------------------------------------------------------- phase 4


def phase4_position_invariance(
    per_caption_class: dict,
    suffix_tag_rows: dict,
    top_n: int,
    cosine_cap: int,
) -> dict:
    # Per tracked class (rating / count-meta / @artist)
    per_class_raw: dict[str, list[dict]] = defaultdict(list)
    for (cname, stem), rows in per_caption_class.items():
        # rows = [(vi, slot, vec), ...]
        if len(rows) < 2:
            continue
        vecs = torch.stack([r[2] for r in rows], dim=0)
        slots = [r[1] for r in rows]
        cs = pairwise_cosine_stats(vecs, cap=cosine_cap)
        per_class_raw[cname].append(
            {
                "stem": stem,
                "median_cos": cs["median"],
                "p10_cos": cs["p10"],
                "slot_range": int(max(slots) - min(slots)),
                "n_variants": len(rows),
            }
        )

    per_class_summary = {}
    for cname, items in per_class_raw.items():
        medians = np.asarray([it["median_cos"] for it in items])
        ranges = np.asarray([it["slot_range"] for it in items])
        per_class_summary[cname] = {
            "n_captions": len(items),
            "cos_median_of_medians": float(np.median(medians)),
            "cos_mean_of_medians": float(np.mean(medians)),
            "cos_p10": float(np.percentile(medians, 10)),
            "slot_range_median": float(np.median(ranges)),
            "slot_range_max": int(ranges.max()),
            "prefix_tag": True,  # defaults — overridden below for suffix
        }
    # @artist is the boundary, still fixed-position per the shuffle logic.

    # Top-N suffix tags by total occurrences
    totals = [
        (tag, sum(len(v) for v in per_stem.values()))
        for tag, per_stem in suffix_tag_rows.items()
    ]
    totals.sort(key=lambda x: -x[1])
    top_suffix = [t for t, _ in totals[:top_n]]

    suffix_summary = {}
    for tag in top_suffix:
        per_stem = suffix_tag_rows[tag]
        rows = []
        for stem, entries in per_stem.items():
            if len(entries) < 2:
                continue
            vecs = torch.stack([e[2] for e in entries], dim=0)
            slots = [e[1] for e in entries]
            cs = pairwise_cosine_stats(vecs, cap=cosine_cap)
            rows.append(
                {
                    "stem": stem,
                    "median_cos": cs["median"],
                    "slot_range": int(max(slots) - min(slots)),
                    "n_variants": len(entries),
                }
            )
        if not rows:
            continue
        medians = np.asarray([r["median_cos"] for r in rows])
        ranges = np.asarray([r["slot_range"] for r in rows])
        suffix_summary[tag] = {
            "n_captions": len(rows),
            "cos_median_of_medians": float(np.median(medians)),
            "cos_p10": float(np.percentile(medians, 10)),
            "slot_range_median": float(np.median(ranges)),
            "slot_range_max": int(ranges.max()),
            "prefix_tag": False,
        }

    return {
        "per_class": per_class_summary,
        "suffix_tags": suffix_summary,
    }


# --------------------------------------------------------------------------- output


def write_summary_md(
    out_dir: Path,
    phase1: dict,
    phase2: dict,
    phase3: dict,
    phase4: dict,
    args,
    n_images: int,
    n_variants: int,
    total_ok: int,
    rt_failures: int,
    rt_fail_unk: int,
):
    lines = []
    a = lines.append
    a("# Tag-Slot Subspace Bench — Summary")
    a("")
    a(f"- image_dir: `{args.image_dir}`  ({n_images} images × {n_variants} variants)")
    a(
        f"- round-trip OK: {total_ok}  failures: {rt_failures} "
        f"(of which {rt_fail_unk} caused by `<unk>` tokens — data property, not drift)"
    )
    cl = phase1["content_length"]
    a(
        f"- content length: n={cl['n']}  p50={cl['p50']:.0f}  "
        f"p95={cl['p95']:.0f}  max={cl['max']}"
    )
    a("")
    a("## Phase 2 — per-class subspace")
    a("")
    a("| class | n | k@95% | sv1 | mean_norm | mean/sv1 | cos_med |")
    a("|---|---|---|---|---|---|---|")
    for cname in sorted(phase2.keys()):
        r = phase2[cname]
        if r.get("skipped"):
            a(f"| {cname} | {r['n_samples']} | (skipped) | | | | |")
            continue
        a(
            f"| {cname} | {r['n_samples']} | {r['k_95']} | "
            f"{r['sv_top_k'][0]:.1f} | {r['mean_vec_norm']:.1f} | "
            f"{r['mean_vec_norm_over_sv1']:.2f} | "
            f"{r['pairwise_cosine']['median']:.3f} |"
        )
    a("")
    a("## Phase 3 — artist clustering")
    if phase3["style_svd"] is not None:
        a("")
        a(f"- artists with ≥{args.min_artist_images} images: {phase3['style_svd']['n_artists']}")
        svs = phase3["style_svd"]["singular_values"][:6]
        a(f"- style-manifold top singular values: {', '.join(f'{v:.2f}' for v in svs)}")
        a("")
        a("| artist | n_imgs | within_cos | between_cos | ratio |")
        a("|---|---|---|---|---|")
        ranked = sorted(
            phase3["artists"].items(),
            key=lambda kv: kv[1].get("within_over_between") or -1,
            reverse=True,
        )
        for name, st in ranked[:10]:
            ratio = st.get("within_over_between")
            ratio_s = f"{ratio:.2f}" if ratio is not None else "-"
            a(
                f"| {name} | {st['n_images']} | "
                f"{st['within_cosine_median']:.3f} | "
                f"{st['between_cosine_mean']:.3f} | {ratio_s} |"
            )
    a("")
    a("## Phase 4 — position invariance")
    a("")
    a("### Tracked prefix/boundary classes (expect cos≈1.0: prefix is shuffle-fixed)")
    a("")
    a("| class | n_capt | cos_med_of_med | p10 | slot_range_med |")
    a("|---|---|---|---|---|")
    for cname in sorted(phase4["per_class"].keys()):
        r = phase4["per_class"][cname]
        a(
            f"| {cname} | {r['n_captions']} | "
            f"{r['cos_median_of_medians']:.3f} | "
            f"{r['cos_p10']:.3f} | {r['slot_range_median']:.0f} |"
        )
    a("")
    a("### Top suffix tags (real position-invariance — slot varies across variants)")
    a("")
    a("| tag | n_capt | cos_med_of_med | slot_range_med | slot_range_max |")
    a("|---|---|---|---|---|")
    for tag in phase4["suffix_tags"]:
        r = phase4["suffix_tags"][tag]
        a(
            f"| {tag} | {r['n_captions']} | "
            f"{r['cos_median_of_medians']:.3f} | "
            f"{r['slot_range_median']:.0f} | {r['slot_range_max']} |"
        )
    a("")
    (out_dir / "summary.md").write_text("\n".join(lines))


# --------------------------------------------------------------------------- main


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading T5 tokenizer from {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    images = discover_cached_images(args.image_dir)
    images = [i for i in images if i.te_path is not None]
    if not images:
        raise SystemExit(f"No TE caches in {args.image_dir}")
    if args.max_images:
        images = images[: args.max_images]

    variants = iter_variants(args.variants)
    logger.info(
        f"Processing {len(images)} images × {len(variants)} variants "
        f"({len(images) * len(variants)} (image, variant) pairs total)"
    )

    logger.info("")
    logger.info("Phase 1 — tag→slot mapping + extraction pass")
    data = build_tag_data(images, tok, variants, logger)
    if data["rt_failures"] > 0:
        frac = data["rt_failures"] / (len(images) * len(variants))
        unk_frac = (
            data["rt_fail_unk"] / data["rt_failures"]
            if data["rt_failures"]
            else 0.0
        )
        logger.warning(
            f"round-trip failures: {data['rt_failures']} ({frac:.1%} of pairs); "
            f"{data['rt_fail_unk']} ({unk_frac:.1%}) caused by <unk> tokens in "
            f"cached ids (data property, not tokenizer drift). If a large "
            f"fraction of failures are NOT <unk>-attributable, investigate."
        )

    phase1 = phase1_summarize(data["positions_per_class"], data["content_lengths"])

    device = torch.device(args.device)
    logger.info(f"Analysis device: {device}")

    logger.info("")
    logger.info("Phase 2 — per-class subspace SVD")
    phase2, class_prototypes = phase2_class_svd(
        data["class_rows"], args, logger, device
    )

    logger.info("")
    logger.info("Phase 3 — per-artist clustering")
    phase3, artist_prototypes = phase3_artists(
        data["artist_rows"],
        args.min_artist_images,
        args.cosine_sample_cap,
        logger,
        device,
    )

    logger.info("")
    logger.info("Phase 4 — position invariance")
    phase4 = phase4_position_invariance(
        data["per_caption_class"],
        data["suffix_tag_rows"],
        args.top_suffix_tags,
        args.cosine_sample_cap,
    )

    # Save JSONs
    with open(out_dir / "phase1_positions.json", "w") as f:
        # Serialize the detailed occurrence lists plus the summary
        serializable = {
            "summary": phase1,
            "per_class_occurrences": {
                cname: [list(e) for e in entries]
                for cname, entries in data["positions_per_class"].items()
            },
        }
        json.dump(serializable, f, indent=2)
    logger.info(f"  → {out_dir / 'phase1_positions.json'}")

    with open(out_dir / "phase2_class_subspaces.json", "w") as f:
        json.dump(phase2, f, indent=2)
    logger.info(f"  → {out_dir / 'phase2_class_subspaces.json'}")

    if class_prototypes:
        save_file(
            {k: v.to(torch.bfloat16).contiguous() for k, v in class_prototypes.items()},
            str(out_dir / "phase2_class_prototypes.safetensors"),
        )
        logger.info(f"  → {out_dir / 'phase2_class_prototypes.safetensors'}")

    with open(out_dir / "phase3_artist_clustering.json", "w") as f:
        json.dump(phase3, f, indent=2)
    logger.info(f"  → {out_dir / 'phase3_artist_clustering.json'}")

    if artist_prototypes:
        save_file(
            {
                k: v.to(torch.bfloat16).contiguous()
                for k, v in artist_prototypes.items()
            },
            str(out_dir / "phase3_artist_prototypes.safetensors"),
        )
        logger.info(f"  → {out_dir / 'phase3_artist_prototypes.safetensors'}")

    with open(out_dir / "phase4_position_invariance.json", "w") as f:
        json.dump(phase4, f, indent=2)
    logger.info(f"  → {out_dir / 'phase4_position_invariance.json'}")

    write_summary_md(
        out_dir, phase1, phase2, phase3, phase4, args,
        n_images=len(images), n_variants=len(variants),
        total_ok=data["total_ok"], rt_failures=data["rt_failures"],
        rt_fail_unk=data["rt_fail_unk"],
    )
    logger.info(f"  → {out_dir / 'summary.md'}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
