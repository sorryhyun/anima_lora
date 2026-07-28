"""DirectEdit CLI — image editing via flow inversion + ψ_tar resampling.

Two prompts in (``--prompt_src``, ``--prompt_tar``), one edited image out.
The source prompt feeds the inversion pass; the target prompt drives the
edit forward pass anchored to per-step inversion residuals (DirectEdit,
Yang & Ye arXiv:2605.02417v1).

Usage:
    python scripts/edit.py \
        --image path/to/source.png \
        --prompt_src "1girl, smile, school_uniform" \
        --prompt_tar "1girl, smile, school_uniform, double peace" \
        --dit models/diffusion_models/anima-base-v1.0.safetensors \
        --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
        --vae models/vae/qwen_image_vae.safetensors \
        --save_path output/tests/directedit/

Wired by ``scripts/experimental_tasks/inference.py::cmd_test_directedit``
under ``make exp-test-directedit`` — that task picks a random source image,
runs the Anima Tagger to seed ``--prompt_src``, and forms ``--prompt_tar``
from ``PROMPT`` env (the user's edit instruction).

v1.1 status:
  * V-injection: WIRED. ``--t_inj N`` injects src self-attn V into the tar
    pass for the first N steps (paper Eq. 13). ``--t_inj_blocks`` selects
    the block subset (default = all but the final block, SD3.5-style).
  * Mask blending: ``--mask`` implements the anchor-side half of paper
    Eq. 12 — Δz dropped inside the edit region (the full background-lock
    latent blend remains future work). ``--easycontrol_mask`` composes the
    learned counterpart: gray-hole the EC cond over the same region so the
    inpaint prior clamps outside it (project/directedit_ec/bench Phase 1a).
  * Inversion runs at ``--invert_guidance 1.0`` (no CFG); the edit pass uses
    the user's ``--guidance_scale`` (default 4.0, Anima base-v1.0 standard).
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms
from typing import Optional

from library.anima import text_strategies  # noqa: E402
from library.datasets.buckets import (  # noqa: E402
    choose_edge,
    freefit_band_for_edge,
    freefit_bucket,
)
from library.inference import sampling as inference_utils  # noqa: E402
from library.inference.editing import directedit  # noqa: E402
from library.inference.editing.directedit_splice import splice_crossattn_emb  # noqa: E402
from library.inference.corrections.smc_cfg import SMCCFGState  # noqa: E402
from library.inference.editing.edit_dispatcher import (  # noqa: E402
    derive_target_caption,
    encode_last_pooled_via_anima_strategy,
)
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.output import save_images  # noqa: E402
from library.inference.text import (  # noqa: E402
    MAX_CROSSATTN_TOKENS,
    ensure_text_strategies,
    prepare_text_inputs,
)
from library.log import setup_logging  # noqa: E402
from library.models import qwen_vae as qwen_image_autoencoder_kl  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DirectEdit image editing for Anima")

    p.add_argument("--dit", required=True)
    p.add_argument("--text_encoder", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--attn_mode", default="flash")

    p.add_argument("--image", required=True, help="Source image path")
    p.add_argument(
        "--prompt_src",
        default="",
        help="Source caption (for inversion). Typically Anima Tagger output "
        "for external images, or the recorded prompt for self-generated "
        "images. Ignored when --cached_embed is set.",
    )
    p.add_argument(
        "--prompt_tar",
        default="",
        help="Target caption (for the edit pass). Usually `prompt_src + edit`. "
        "Ignored when --cached_embed is set. When --edit_instruction is given "
        "and --prompt_tar is empty, the dispatcher derives this automatically.",
    )
    p.add_argument(
        "--edit_instruction",
        default="",
        help="Short tag-phrase edit (e.g. 'large breasts', '-hair ornament', "
        "'no hair ornament'). When set, the dispatcher derives --prompt_tar "
        "from --prompt_src + this instruction: explicit '-X' or 'no X' "
        "(matching an existing tag) does REMOVE; Qwen3 last-pool cosine + "
        "threshold gate fires REPLACE on confident matches; otherwise APPEND. "
        "Ignored when --prompt_tar is set explicitly or when --cached_embed "
        "is set.",
    )
    p.add_argument(
        "--replace_threshold",
        type=float,
        default=0.92,
        help="Dispatcher: top-1 cosine must exceed this to fire REPLACE. "
        "Tuned against scripts/probes/edit_nearest_tag.py.",
    )
    p.add_argument(
        "--replace_gap",
        type=float,
        default=0.04,
        help="Dispatcher: top1−top2 cosine gap must exceed this to fire "
        "REPLACE. Probe ambiguous cases (huge+large both present, medium-vs-"
        "grey hair near-tie) sit at gap < 0.01 and abstain into APPEND.",
    )
    p.add_argument(
        "--use_slot_surgery",
        action="store_true",
        help="Build embed_tar by transplanting only the T5-diff-span slots of "
        "ψ_tar's crossattn_emb into ψ_src's encoding. Off by default (uses "
        "the full ψ_tar encoding as today). Requires --prompt_src non-empty. "
        "Untouched slots come from ψ_src — see library/inference/"
        "directedit_splice.py for the invariant.",
    )
    p.add_argument(
        "--cached_embed",
        default=None,
        help="Sanity-check mode: load a preprocessed `_anima_te.safetensors` "
        "cache (the file `cache_text_embeddings.py` writes — same format the "
        "trainer consumes) and run one invert + edit pass per stored variant "
        "with ψ_tar == ψ_src. With `--caption_shuffle_variants N` caches, "
        "this sweeps v0..v{N-1} (pristine + tag-shuffled re-encodings); "
        "single-variant caches collapse to one pass. Skips the text encoder "
        "entirely. Mismatched reconstruction across variants flags numeric "
        "drift in invert/edit_forward.",
    )
    p.add_argument(
        "--cached_embed_variants",
        default="all",
        help="Which variants to run from the --cached_embed cache. "
        "'all' (default) sweeps every stored variant. Otherwise pass a "
        "comma-separated list of indices, e.g. '0' for the pristine caption "
        "only, '0,2' for v0 + v2. Out-of-range indices fail loud. "
        "Ignored unless --cached_embed is set.",
    )
    p.add_argument(
        "--negative_prompt",
        default="",
        help="Negative prompt for CFG on the edit pass (default empty). In "
        "--cached_embed mode, an empty value is auto-replaced with 'worst "
        "quality' so CFG can still fire (the TE is loaded briefly to encode "
        "just the neg, then dropped).",
    )
    p.add_argument(
        "--mask",
        default=None,
        help="Edit-region mask path (white/nonzero = edit region): drops the "
        "Δz anchor inside the region (paper Eq. 12, anchor-side half) so the "
        "edit isn't pulled back to the source there. Downsampled to latent "
        "resolution. Combine with --easycontrol_mask (same file) so the EC "
        "prior also releases the region while clamping everything else.",
    )
    p.add_argument(
        "--fm_score",
        action="store_true",
        help="AGSM-style ψ_src probe: score each variant's source conditioning "
        "by its intrinsic flow-matching error against the source latent "
        "(lower = the model finds the caption a better explanation = more "
        "on-manifold). σ and noise are held FIXED across variants so the "
        "ranking reflects only the conditioning (the reward-premise contract: "
        "relative ranking on one image cancels per-sample noise). In "
        "--cached_embed mode also logs each variant's latent reconstruction "
        "MSE so you can check whether the lowest-FM variant reconstructs best.",
    )
    p.add_argument(
        "--fm_score_sigmas",
        default="0.25,0.5,0.7,0.9",
        help="Comma-separated σ grid for --fm_score (default biased mid/high, "
        "where the AGSM reward margin is largest). One forward per variant.",
    )
    p.add_argument(
        "--fm_score_seed",
        type=int,
        default=None,
        help="Seed for the fixed --fm_score noise draw (default: --seed). "
        "Same seed across variants is what makes the ranking comparable.",
    )

    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--flow_shift", type=float, default=1.0)
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="CFG scale for the edit (target) pass.",
    )
    p.add_argument(
        "--invert_guidance",
        type=float,
        default=1.0,
        help="CFG scale during inversion. Default 1.0 (no CFG); raise only if "
        "you need the inverted noise to match a high-CFG generation seed.",
    )
    p.add_argument(
        "--smc_cfg",
        action="store_true",
        help="α-adaptive Sliding-Mode Control on the edit pass's CFG combine "
        "(library/inference/smc_cfg.py). Clamps small/noisy CFG-residual "
        "voxels while preserving large semantic moves; composes with t_inj "
        "V-injection (SMC operates on the post-injection v_cond_tar / v_neg "
        "residual). No-op on the inversion pass.",
    )
    p.add_argument(
        "--smc_cfg_lambda",
        type=float,
        default=5.0,
        help="SMC sliding-manifold slope λ. Defaults match inference.py.",
    )
    p.add_argument(
        "--smc_cfg_alpha",
        type=float,
        default=0.1,
        help="SMC adaptive gain α ∈ (0, 1]. Defaults match inference.py.",
    )
    p.add_argument(
        "--t_inj",
        type=int,
        default=2,
        help="Number of early editing steps to inject src self-attn V into "
        "the tar pass (paper Eq. 13). Default 0 = pure ΔZ-anchored edit. "
        "Typical paper setting: t_inj ≈ T/10..T/3 (e.g. 3..9 at T=28). "
        "Higher = stronger source-feature preservation.",
    )
    p.add_argument(
        "--t_inj_blocks",
        default="all_but_last",
        help="Which DiT blocks V-injection targets. Accepts 'all', "
        "'all_but_last' (default, SD3.5-style), or a comma/range string like "
        "'8-22' or '8,9,12,14-18'. Ignored when --t_inj 0.",
    )
    p.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        help="Override image size (H W). Default: free-fit the source aspect "
        "ratio into the 1024 tier's token band (preserves native aspect).",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--save_path", required=True)

    # Passthroughs inference.py exposes that downstream code reads — keep so
    # generation-side accessors don't trip.
    p.add_argument("--vae_chunk_size", type=int, default=64)
    p.add_argument("--vae_disable_cache", action="store_true", default=True)
    p.add_argument("--text_encoder_cpu", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--no_metadata", action="store_true")
    p.add_argument("--lora_weight", nargs="*", default=None)
    p.add_argument("--lora_multiplier", nargs="*", type=float, default=1.0)
    p.add_argument(
        "--compile_blocks",
        action="store_true",
        default=True,
        help="torch.compile each transformer block's _forward individually "
        "(per-block compile, not full-model). Speeds up the inversion + edit "
        "loops; first call per shape pays a compile cost.",
    )
    p.add_argument(
        "--compile_inductor_mode",
        default=None,
        help="Inductor preset passed through to torch.compile(mode=...). "
        "e.g. 'reduce-overhead' for per-block CUDAGraphs.",
    )
    p.add_argument(
        "--no_compile_blocks",
        dest="compile_blocks",
        action="store_false",
        help="Disable per-block torch.compile (bench wall-time parity across "
        "arms, or debugging).",
    )
    p.add_argument(
        "--easycontrol_weight",
        default=None,
        help="EasyControl adapter checkpoint to load as a learned source-"
        "preservation prior on the edit trajectory (e.g. the inpaint adapter "
        "fed a hole-free cond). The cond stream is primed once from "
        "--easycontrol_image (default: the source image itself) and stays "
        "active through BOTH the inversion and edit passes, so ψ_tar == ψ_src "
        "still reconstructs the source under the same effective model.",
    )
    p.add_argument(
        "--easycontrol_scale",
        type=float,
        default=None,
        help="Override the checkpoint's cond_scale — the EC preservation-"
        "strength dial (analogue of --t_inj). Default: checkpoint metadata.",
    )
    p.add_argument(
        "--easycontrol_image",
        default=None,
        help="Reference image for the EC cond stream (default: --image).",
    )
    p.add_argument(
        "--easycontrol_b_offset",
        type=float,
        default=None,
        help="Additive offset on every block's learned b_cond gate — the "
        "continuous cond-softmax-mass dial (each -1 cuts cond attention mass "
        "~e×; cond_scale is near-binary on the inpaint prior, see "
        "project/directedit_ec/bench). Applied after load, read live per forward "
        "(NOT baked into the KV cache).",
    )
    p.add_argument(
        "--easycontrol_mask",
        default=None,
        help="Mask image path (white/nonzero = edit region). The masked "
        "region of the EC cond IMAGE is filled with flat mid-gray (128) "
        "before VAE encode — the inpaint prior's trained hole convention "
        "(easycontrol_adapters/inpainting/mask_image.py), so the prior "
        "clamps outside the hole and generates freely inside it, steered by "
        "ψ_tar. The fill happens in pixel space, never on the latent. "
        "Requires --easycontrol_weight.",
    )
    p.add_argument(
        "--easycontrol_edit_only",
        action="store_true",
        help="Prime the EC cond KV cache AFTER inversion instead of before, "
        "so the inversion pass runs the exact baseline DiT and only the edit "
        "pass sees the adapter. Breaks the ψ_tar == ψ_src exact-recon "
        "guarantee (the anchor residuals come from a different effective "
        "model than the edit pass). Requires --easycontrol_weight.",
    )
    p.add_argument(
        "--anchor_scale",
        type=float,
        default=1.0,
        help="Global multiplier λ on the Δz anchor residuals in the edit pass "
        "(1.0 = full anchor, 0 = unanchored generation from the inverted "
        "init). The continuous composition↔edit dial for whole-image "
        "instruction edits with no region mask; composes with --mask "
        "(regional release on top of the scaled residual).",
    )
    p.add_argument(
        "--easycontrol_invert_b_offset",
        type=float,
        default=None,
        help="Use this b_cond offset during the INVERSION pass only; the edit "
        "pass keeps --easycontrol_b_offset. b_cond is a live logit bias (not "
        "baked into the KV cache), so the per-pass swap needs no re-priming. "
        "E.g. invert in the copy regime (+2/+3 on subject_edit) so the cond "
        "stream explains the source, then edit at the trained point. "
        "Requires --easycontrol_weight; incompatible with "
        "--easycontrol_edit_only (inversion has no cond there).",
    )

    args = p.parse_args()
    args.compile = False
    return args


def _pick_bucket(img: Image.Image) -> tuple[int, int]:
    """Return (H, W): free-fit the source aspect into the canonical 1024 tier band.

    The old path snapped to the nearest discrete CONSTANT_TOKEN_BUCKETS; free-fit
    preserves the native aspect (sub-patch crop).
    """
    rw, rh = img.size
    edge = choose_edge(rw, rh, [1024])
    bw, bh = freefit_bucket(rw, rh, freefit_band_for_edge(edge))
    return bh, bw  # bucket is (W, H); we return (H, W)


def _parse_t_inj_blocks(spec: str, n_blocks: int) -> list[int] | None:
    """Parse `--t_inj_blocks` into a list of block indices.

    'all' → every block (0..n-1). 'all_but_last' → 0..n-2 (default; matches
    paper's SD3.5 placement). Otherwise parses comma-separated entries that
    are either a single int or a closed range 'A-B'. Returns None for the
    'all_but_last' default so the directedit module's own default applies
    (and the log message stays consistent across callers).
    """
    spec = spec.strip().lower()
    if spec in ("", "all_but_last"):
        return None  # → directedit default
    if spec == "all":
        return list(range(n_blocks))
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"--t_inj_blocks range {chunk!r}: lo > hi")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    if not out:
        raise ValueError(f"--t_inj_blocks={spec!r} parsed to empty set")
    bad = [i for i in out if i < 0 or i >= n_blocks]
    if bad:
        raise ValueError(
            f"--t_inj_blocks={spec!r}: indices {sorted(set(bad))} out of "
            f"range (model has {n_blocks} blocks; valid 0..{n_blocks - 1})"
        )
    return sorted(set(out))


def _parse_variant_selector(selector: str, n_available: int) -> list[int]:
    """Parse `--cached_embed_variants` into a list of variant indices.

    'all' yields [0..n_available-1]; comma-separated indices yield those.
    Out-of-range indices fail loud so a typo doesn't silently fall back to
    the full sweep.
    """
    if selector == "all":
        return list(range(n_available))
    try:
        wanted = [int(s.strip()) for s in selector.split(",") if s.strip()]
    except ValueError as e:
        raise ValueError(
            f"--cached_embed_variants={selector!r}: expected 'all' or a "
            "comma-separated list of integers"
        ) from e
    if not wanted:
        raise ValueError("--cached_embed_variants is empty")
    bad = [i for i in wanted if i < 0 or i >= n_available]
    if bad:
        raise ValueError(
            f"--cached_embed_variants={selector!r}: indices {bad} out of "
            f"range — cache has {n_available} variant(s) (0..{n_available - 1})"
        )
    return wanted


def _load_cached_embed_variants(
    cache_path: str,
    anima,
    device: torch.device,
    selector: str = "all",
) -> list[tuple[str, torch.Tensor]]:
    """Load preprocessed crossattn embeds from a `_anima_te.safetensors` cache.

    Returns a list of `(variant_label, crossattn_emb)` ready to feed
    DirectEdit. Mirrors `AnimaTextEncoderOutputsCachingStrategy.load_outputs_npz`
    but emits the variants requested by `selector` (default 'all') instead of
    stochastically sampling one — this is a sweep, not training.

    Behavior:
      * Multi-variant caches (`num_variants` key present): yields v_i for every
        i selected by `selector`.  v0 is the pristine caption; v1..v{N-1} are
        tag-shuffled re-encodings.
      * Single-variant caches: yields one pass.  `selector` must be 'all' or
        '0'.
      * Pre-baked `crossattn_emb*` (cached when training was preprocessed
        with `cache_llm_adapter_outputs=True`) is used directly. Otherwise
        we run `anima._preprocess_text_embeds` ourselves so the cache stays
        usable regardless of how it was preprocessed.

    Fails loud if the file is missing, shape-mismatched, or `selector` names a
    missing variant.
    """
    from safetensors import safe_open

    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"--cached_embed file not found: {cache_path}\n"
            "Run `make preprocess-te` (with --caption_shuffle_variants N to "
            "get a multi-variant cache) before running the dry test."
        )

    out: list[tuple[str, torch.Tensor]] = []
    with safe_open(cache_path, framework="pt") as f:
        keys = set(f.keys())
        has_variants = "num_variants" in keys
        if has_variants:
            n = int(f.get_tensor("num_variants"))
            wanted = _parse_variant_selector(selector, n)
            indices = [(f"v{i}", f"_v{i}") for i in wanted]
        else:
            # Single-variant cache: only v0 exists; reject anything else.
            _parse_variant_selector(selector, 1)
            indices = [("v0", "")]

        for label, suf in indices:
            crossattn_key = f"crossattn_emb{suf}"
            if crossattn_key in keys:
                crossattn_emb = f.get_tensor(crossattn_key).to(
                    device, dtype=torch.bfloat16
                )
                # Cache stores unbatched (N, D); DiT expects (B, N, D).
                if crossattn_emb.dim() == 2:
                    crossattn_emb = crossattn_emb.unsqueeze(0)
                # Pre-baked from training preprocess — already adapter-projected.
            else:
                # Run llm_adapter ourselves on the raw Qwen3 prompt_embeds.
                prompt_embeds = f.get_tensor(f"prompt_embeds{suf}").to(device)
                attn_mask = f.get_tensor(f"attn_mask{suf}").to(device)
                t5_input_ids = f.get_tensor(f"t5_input_ids{suf}").to(device)
                t5_attn_mask = f.get_tensor(f"t5_attn_mask{suf}").to(device)
                # Cached tensors are unbatched; the adapter expects a batch dim.
                if prompt_embeds.dim() == 2:
                    prompt_embeds = prompt_embeds.unsqueeze(0)
                if attn_mask.dim() == 1:
                    attn_mask = attn_mask.unsqueeze(0)
                if t5_input_ids.dim() == 1:
                    t5_input_ids = t5_input_ids.unsqueeze(0)
                if t5_attn_mask.dim() == 1:
                    t5_attn_mask = t5_attn_mask.unsqueeze(0)
                crossattn_emb, _ = anima._preprocess_text_embeds(
                    source_hidden_states=prompt_embeds,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=attn_mask,
                )
                crossattn_emb[~t5_attn_mask.bool()] = 0
                crossattn_emb = crossattn_emb.to(torch.bfloat16)
            out.append((label, crossattn_emb))
    return out


@torch.no_grad()
def _fm_error_score(
    anima,
    z_clean: torch.Tensor,
    emb: torch.Tensor,
    sv: torch.Tensor,
    noise: torch.Tensor,
) -> float:
    """AGSM intrinsic reward (negated) for one conditioning ``emb``.

    Returns the mean flow-matching error ``‖v_θ(x_σ, σ, emb) − (noise − x0)‖²``
    over the FIXED ``(sv, noise)`` grid. Lower = the model finds ``emb`` a
    better explanation of the source image = more on-manifold. The σ/noise
    grid is shared across every variant, so differences are attributable to
    the conditioning alone — this is the regime where the FM signal is
    informative (relative ranking on one image), unlike absolute FM val loss.

    ``z_clean`` is the 5D source latent ``[1, C, 1, H, W]``; ``sv`` is
    ``(n, 1, 1, 1)`` and ``noise`` is ``(n, C, H, W)``.
    """
    lat = z_clean.squeeze(2)  # [1, C, H, W]
    n = noise.shape[0]
    lat = lat.expand(n, -1, -1, -1)
    # Cast σ to the latent dtype (bf16) so the mix stays bf16 — sv arrives fp32
    # from torch.tensor(); a bf16*fp32 mix would promote the latent to fp32 and
    # mismatch the DiT's bf16 weights (mirrors fm_loss_step's cast).
    sv = sv.to(lat.dtype)
    noisy = ((1.0 - sv) * lat + sv * noise).unsqueeze(2)  # 5D for the DiT
    emb_e = emb.expand(n, -1, -1)
    pm = torch.zeros(
        n, 1, lat.shape[-2], lat.shape[-1], dtype=torch.bfloat16, device=lat.device
    )
    timesteps = sv.view(-1).to(torch.bfloat16)
    pred = anima(noisy, timesteps, emb_e, padding_mask=pm).squeeze(2)
    target = noise - lat
    return ((pred.float() - target.float()) ** 2).mean().item()


def _log_fm_score_table(rows: list[dict]) -> None:
    """Log the per-variant FM-error / reconstruction table + a probe verdict.

    ``rows`` carry ``label``, ``fm`` (always) and ``recon`` (cached_embed mode
    only). Sorted by FM error so the on-manifold ranking reads top-down. When
    reconstruction MSE is present the summary reports whether the lowest-FM
    variant is also the best-reconstructing one and, for n≥3, the Pearson r
    between the two — the core question the probe exists to answer.
    """
    have_recon = all(r.get("recon") is not None for r in rows)
    ordered = sorted(rows, key=lambda r: r["fm"])
    header = "  rank  variant            fm_error" + (
        "   recon_mse" if have_recon else ""
    )
    logger.info("ψ_src FM-error probe (lower fm_error = more on-manifold):")
    logger.info(header)
    for i, r in enumerate(ordered):
        line = f"  {i:>4}  {str(r['label']):<16}  {r['fm']:.6f}"
        if have_recon:
            line += f"   {r['recon']:.6f}"
        logger.info(line)
    if not have_recon or len(rows) < 2:
        return
    best_fm = min(rows, key=lambda r: r["fm"])
    best_recon = min(rows, key=lambda r: r["recon"])
    logger.info(
        "  lowest-fm variant=%s  |  best-reconstructing variant=%s  |  match=%s",
        best_fm["label"],
        best_recon["label"],
        best_fm["label"] == best_recon["label"],
    )
    if len(rows) >= 3:
        fm = torch.tensor([r["fm"] for r in rows], dtype=torch.float64)
        rc = torch.tensor([r["recon"] for r in rows], dtype=torch.float64)
        fm = fm - fm.mean()
        rc = rc - rc.mean()
        denom = (fm.norm() * rc.norm()).item()
        r = (fm @ rc).item() / denom if denom > 0 else float("nan")
        logger.info(
            "  Pearson r(fm_error, recon_mse) = %+.3f over n=%d variants "
            "(want > 0: low FM error predicts faithful reconstruction).",
            r,
            len(rows),
        )


def main() -> None:
    args = parse_args()

    if args.easycontrol_mask and not args.easycontrol_weight:
        raise SystemExit("--easycontrol_mask requires --easycontrol_weight")
    if args.easycontrol_edit_only and not args.easycontrol_weight:
        raise SystemExit("--easycontrol_edit_only requires --easycontrol_weight")
    if args.easycontrol_invert_b_offset is not None:
        if not args.easycontrol_weight:
            raise SystemExit(
                "--easycontrol_invert_b_offset requires --easycontrol_weight"
            )
        if args.easycontrol_edit_only:
            raise SystemExit(
                "--easycontrol_invert_b_offset is meaningless with "
                "--easycontrol_edit_only (the inversion pass runs without cond)"
            )
    if args.t_inj > 0 and args.compile_blocks:
        # V-injection monkey-patches Attention.forward at runtime, invalidating
        # dynamo's per-block graph; recompile cost > compile's speedup, so off
        # for editing (compile state is per-process — can't flip mid-run).
        logger.info(
            "--t_inj %d > 0: disabling --compile_blocks for V-injection "
            "(monkey-patch breaks dynamo graph cache).",
            args.t_inj,
        )
        args.compile_blocks = False
    if args.easycontrol_weight and args.compile_blocks:
        # EasyControl patches Block.forward and dispatches on cond state; the
        # inference engine runs it eager, so the edit CLI does too.
        logger.info(
            "--easycontrol_weight set: disabling --compile_blocks "
            "(patched Block.forward runs eager, matching inference.py)."
        )
        args.compile_blocks = False

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.device = device

    src_pil = Image.open(args.image).convert("RGB")
    if args.image_size is None:
        h_pix, w_pix = _pick_bucket(src_pil)
        args.image_size = [h_pix, w_pix]
        logger.info(
            "Image size auto-picked from source aspect %.3f -> %dx%d (HxW)",
            src_pil.size[0] / src_pil.size[1],
            h_pix,
            w_pix,
        )
    h_pix, w_pix = args.image_size
    src_pil = src_pil.resize((w_pix, h_pix), Image.LANCZOS)

    ensure_text_strategies(args.text_encoder, MAX_CROSSATTN_TOKENS)

    # Load DiT first — prepare_text_inputs's _preprocess_text_embeds needs it.
    logger.info("Loading DiT model...")
    anima = load_dit_model(args, device, dit_weight_dtype=torch.bfloat16)

    # EasyControl preservation prior: load + apply BEFORE any compile (the
    # compile-after-apply invariant), prime the cond KV cache after the source
    # latent exists below. Mirrors library/inference/generation.py's
    # _setup_easycontrol.
    ec_network = None
    if args.easycontrol_weight:
        from networks.methods.easycontrol import (
            create_network_from_weights as ec_create_from_weights,
        )

        ec_kwargs = {}
        if args.easycontrol_scale is not None:
            ec_kwargs["cond_scale"] = float(args.easycontrol_scale)
        ec_network, _ = ec_create_from_weights(
            multiplier=1.0,
            file=args.easycontrol_weight,
            ae=None,
            text_encoders=None,
            unet=anima,
            **ec_kwargs,
        )
        ec_network.load_weights(args.easycontrol_weight)
        if args.easycontrol_b_offset is not None:
            with torch.no_grad():
                for b in ec_network.b_cond:
                    b += args.easycontrol_b_offset
        ec_network.to(device, dtype=torch.bfloat16)
        ec_network.apply_to(text_encoders=None, unet=anima)
        logger.info(
            "EasyControl preservation prior: %s (cond_scale=%s, b_offset=%s)",
            args.easycontrol_weight,
            "ckpt" if args.easycontrol_scale is None else args.easycontrol_scale,
            args.easycontrol_b_offset,
        )

    if args.compile_blocks:
        anima.compile_blocks(mode=args.compile_inductor_mode)

    # Encode src + tar text — or in --cached_embed mode load preprocessed
    # crossattn variants from the TE cache (ψ_tar == ψ_src reconstructs source).
    cached_variants: list[tuple[str, torch.Tensor]] | None = None
    if args.cached_embed is not None:
        cached_variants = _load_cached_embed_variants(
            args.cached_embed, anima, device, args.cached_embed_variants
        )
        embed_src = embed_tar = None  # filled per-variant below

        # Cache file has no neg slot — encode one on the fly so CFG can fire.
        neg_prompt = args.negative_prompt or ""
        if not args.negative_prompt:
            logger.info(
                "DirectEdit dry: --negative_prompt empty; defaulting to '' for CFG."
            )

        # Reuse prepare_text_inputs with prompt == negative_prompt so only one
        # TE pass runs (positive hits the conds_cache); keep ctx_neg, drop pos.
        args_neg = SimpleNamespace(**vars(args))
        args_neg.prompt = neg_prompt
        args_neg.negative_prompt = neg_prompt

        te_dtype = torch.bfloat16
        te_device = torch.device("cpu") if args.text_encoder_cpu else device
        text_encoder = load_text_encoder(args, dtype=te_dtype, device=te_device)
        shared = {"text_encoder": text_encoder, "conds_cache": {}}
        _, ctx_neg = prepare_text_inputs(args_neg, device, anima, shared)
        text_encoder.to("cpu")
        del text_encoder, shared
        clean_memory_on_device(device)

        embed_neg = ctx_neg["embed"][0].to(device, dtype=torch.bfloat16)
        logger.info(
            "DirectEdit dry: loaded %d variant(s) from %s; CFG enabled (neg=%r).",
            len(cached_variants),
            args.cached_embed,
            neg_prompt,
        )
    else:
        # Load TE first — the dispatcher (--edit_instruction) needs Qwen3 hidden
        # states before we can build the prepare_text_inputs args.
        logger.info("Loading text encoder...")
        te_dtype = torch.bfloat16
        te_device = torch.device("cpu") if args.text_encoder_cpu else device
        text_encoder = load_text_encoder(args, dtype=te_dtype, device=te_device)
        text_encoder.eval()

        # Derive ψ_tar from (ψ_src + edit_instruction) only when --prompt_tar
        # wasn't given; explicit --prompt_tar always wins.
        if args.edit_instruction and not args.prompt_tar:
            tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
            encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()
            # Dispatcher needs TE on-device; move it (--text_encoder_cpu parks
            # it on CPU), then restore.
            te_was_on = text_encoder.device
            text_encoder.to(device)
            encode_fn = lambda phrases: encode_last_pooled_via_anima_strategy(  # noqa: E731
                phrases,
                text_encoder,
                tokenize_strategy,
                encoding_strategy,
                device,
            )
            plan = derive_target_caption(
                args.prompt_src,
                args.edit_instruction,
                encode_last_pooled=encode_fn,
                replace_threshold=args.replace_threshold,
                replace_gap=args.replace_gap,
            )
            text_encoder.to(te_was_on)
            args.prompt_tar = plan.tar_caption
            logger.info(plan.log_line())
            logger.info("DirectEdit dispatcher: ψ_tar=%r", plan.tar_caption)
        elif args.use_slot_surgery and not args.prompt_tar:
            raise SystemExit(
                "--use_slot_surgery requires a ψ_tar source: pass --prompt_tar "
                "explicitly or --edit_instruction to derive it."
            )

        if args.use_slot_surgery and not args.prompt_src:
            raise SystemExit(
                "--use_slot_surgery requires a non-empty --prompt_src "
                "(surgery transplants from ψ_src's encoding)."
            )

        args_src = SimpleNamespace(**vars(args))
        args_src.prompt = args.prompt_src
        args_src.negative_prompt = args.negative_prompt

        args_tar = SimpleNamespace(**vars(args))
        args_tar.prompt = args.prompt_tar
        args_tar.negative_prompt = args.negative_prompt

        logger.info("Encoding prompts...")
        # Share the TE instance across both prompt encodings.
        shared = {"text_encoder": text_encoder, "conds_cache": {}}

        ctx_src, ctx_neg = prepare_text_inputs(args_src, device, anima, shared)
        ctx_tar, _ = prepare_text_inputs(args_tar, device, anima, shared)

        embed_src = ctx_src["embed"][0].to(device, dtype=torch.bfloat16)
        embed_tar = ctx_tar["embed"][0].to(device, dtype=torch.bfloat16)
        embed_neg = ctx_neg["embed"][0].to(device, dtype=torch.bfloat16)

        if args.use_slot_surgery:
            # ctx["embed"] = [crossattn_emb_cpu, qwen3_attn_mask, t5_ids, t5_attn_mask];
            # T5 IDs stay on CPU (encode_tokens moves only qwen3 tensors).
            t5_ids_src = ctx_src["embed"][2]
            t5_ids_tar = ctx_tar["embed"][2]
            tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
            pad_id = tokenize_strategy.t5_tokenizer.pad_token_id
            embed_tar_full = embed_tar
            embed_tar, span = splice_crossattn_emb(
                crossattn_emb_src=embed_src,
                crossattn_emb_tar=embed_tar_full,
                t5_ids_src=t5_ids_src.to(device),
                t5_ids_tar=t5_ids_tar.to(device),
                pad_id=pad_id,
            )
            logger.info(
                "DirectEdit slot surgery: diff span src[%d:%d] -> tar[%d:%d] "
                "(src_len=%d tar_len=%d suffix_len=%d)",
                span.start,
                span.src_end,
                span.start,
                span.tar_end,
                span.src_len,
                span.tar_len,
                span.suffix_len,
            )

        # Drop TE; conds_cache hands us bare tensors and surgery is done.
        text_encoder.to("cpu")
        del text_encoder, shared
        clean_memory_on_device(device)

        with torch.no_grad():
            d_st = (embed_src.float() - embed_tar.float()).abs().mean().item()
            d_sn = (embed_src.float() - embed_neg.float()).abs().mean().item()
            d_tn = (embed_tar.float() - embed_neg.float()).abs().mean().item()
        logger.info(
            "DirectEdit embed diffs (abs mean): "
            "|src-tar|=%.6f  |src-neg|=%.6f  |tar-neg|=%.6f  "
            "(src.norm=%.3f tar.norm=%.3f shape=%s)",
            d_st,
            d_sn,
            d_tn,
            embed_src.float().norm().item(),
            embed_tar.float().norm().item(),
            tuple(embed_src.shape),
        )

    # VAE-encode the source image -> clean latent (5D, frame=1).
    logger.info("Loading VAE for source encode...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
    )
    vae.to(torch.bfloat16).eval().to(device)

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    # 5D [B, C, T=1, H, W] — qwen_vae preserves input rank, and the DiT
    # expects 5D latents (it concats a per-frame padding mask along dim=1).
    img_t = (
        tfm(src_pil).unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16)
    )  # [1, 3, 1, H, W] in [-1,1]

    with torch.no_grad():
        z_clean = vae.encode_pixels_to_latents(img_t)  # [1, C, 1, H/8, W/8]
    logger.info("Encoded source latent: %s", tuple(z_clean.shape))

    # Eq. 12 anchor mask: latent-resolution {0,1} map of the edit region,
    # broadcast over channels. delta_z is dropped where the mask is 1.
    anchor_mask = None
    if args.mask:
        import numpy as np

        h_lat, w_lat = int(z_clean.shape[-2]), int(z_clean.shape[-1])
        m_pil = (
            Image.open(args.mask).convert("L").resize((w_lat, h_lat), Image.BILINEAR)
        )
        m = (np.asarray(m_pil) > 127).astype("float32")
        anchor_mask = torch.from_numpy(m).view(1, 1, 1, h_lat, w_lat).to(device)
        logger.info(
            "Anchor mask from %s: %.1f%% of latent cells released.",
            args.mask,
            100.0 * float(m.mean()),
        )

    # Prime the EC cond stream while the VAE is still mounted. Default cond is
    # the source latent itself (hole-free "copy everything" reference); the
    # cache then serves every forward — inversion, edit, CFG branches — so the
    # effective model is identical across both passes and ψ_tar == ψ_src keeps
    # reconstructing exactly.
    if ec_network is not None:
        if args.easycontrol_image:
            cond_pil = (
                Image.open(args.easycontrol_image)
                .convert("RGB")
                .resize((w_pix, h_pix), Image.LANCZOS)
            )
        else:
            cond_pil = src_pil
        if args.easycontrol_mask:
            # Punch the trained exception channel back into the prior: fill
            # the edit region with flat mid-gray in PIXEL space (VAE-encodes
            # to the flat latent the inpaint gate reads as "regenerate here").
            import numpy as np

            mask_pil = (
                Image.open(args.easycontrol_mask)
                .convert("L")
                .resize((w_pix, h_pix), Image.NEAREST)
            )
            hole = np.asarray(mask_pil) > 127
            arr = np.asarray(cond_pil).copy()
            arr[hole] = 128  # mask_image.GRAY — the inpaint training fill
            cond_pil = Image.fromarray(arr)
            logger.info(
                "EasyControl: cond hole gray-filled from %s (%.1f%% of frame).",
                args.easycontrol_mask,
                100.0 * hole.mean(),
            )
        if args.easycontrol_image or args.easycontrol_mask:
            cond_t = (
                tfm(cond_pil).unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16)
            )
            with torch.no_grad():
                z_cond = vae.encode_pixels_to_latents(cond_t)
        else:
            z_cond = z_clean
        ec_z_cond = z_cond.to(device, dtype=torch.bfloat16)
        if args.easycontrol_edit_only:
            logger.info(
                "EasyControl: cond priming DEFERRED to post-inversion "
                "(--easycontrol_edit_only); inversion runs the baseline DiT."
            )
        else:
            ec_network.set_cond(ec_z_cond)
            ec_network.precompute_cond_kv()
            logger.info(
                "EasyControl: cond KV cache primed from %s (cond latent %s).",
                args.easycontrol_image or args.image,
                tuple(z_cond.shape),
            )

    # Move VAE off-device for the DiT loop, bring it back for decode.
    vae.to("cpu")
    clean_memory_on_device(device)

    # invert/edit_forward consume sigmas directly; the timesteps return is unused.
    _, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas.to(device)

    # Build the variant pass list: real-text mode = one src/tar pass;
    # --cached_embed mode = one pass per stored variant, each ψ_tar == ψ_src.
    if cached_variants is not None:
        variant_passes = [(label, e, e) for label, e in cached_variants]
    else:
        variant_passes = [(None, embed_src, embed_tar)]

    # AGSM ψ_src probe grid: one fixed (σ, noise) batch reused for every variant
    # so the FM-error ranking reflects only the conditioning (relative-ranking).
    fm_grid = None
    fm_rows: list[dict] = []
    if args.fm_score:
        sig_vals = [float(s) for s in args.fm_score_sigmas.split(",") if s.strip()]
        if not sig_vals:
            raise SystemExit("--fm_score_sigmas parsed to empty list")
        seed = args.fm_score_seed if args.fm_score_seed is not None else args.seed
        gen = torch.Generator(device=device).manual_seed(int(seed))
        sv = torch.tensor(sig_vals, device=device).view(-1, 1, 1, 1)
        lat4 = z_clean.squeeze(2)
        noise = torch.randn(
            len(sig_vals),
            lat4.shape[1],
            lat4.shape[2],
            lat4.shape[3],
            device=device,
            dtype=lat4.dtype,
            generator=gen,
        )
        fm_grid = (sv, noise)
        logger.info(
            "ψ_src FM-error probe enabled: σ grid=%s, seed=%d, %d variant(s).",
            sig_vals,
            int(seed),
            len(variant_passes),
        )

    # Inversion -> editing per variant. Hold all z_edits before re-mounting the
    # VAE so we do only one DiT-off / VAE-on swap.
    # Per-pass b_cond offset: shift into the inversion-pass operating point at
    # the top of each variant (before the FM probe, so it measures the
    # inversion-pass model), revert right after invert().
    ec_invert_shift = 0.0
    if ec_network is not None and args.easycontrol_invert_b_offset is not None:
        ec_invert_shift = float(args.easycontrol_invert_b_offset) - float(
            args.easycontrol_b_offset or 0.0
        )

    z_edits: list[tuple[Optional[str], torch.Tensor]] = []
    for variant, e_src, e_tar in variant_passes:
        tag = f"variant={variant}, " if variant else ""
        if ec_network is not None and ec_invert_shift:
            with torch.no_grad():
                for b in ec_network.b_cond:
                    b += ec_invert_shift
            logger.info(
                "EasyControl: inversion-pass b_offset=%s (shift %+.2f vs edit pass).",
                args.easycontrol_invert_b_offset,
                ec_invert_shift,
            )
        if fm_grid is not None:
            fm = _fm_error_score(anima, z_clean, e_src, fm_grid[0], fm_grid[1])
            fm_rows.append(
                {"label": variant if variant is not None else "src", "fm": fm}
            )
            logger.info("  %sψ_src FM-error = %.6f", tag, fm)
        # Fresh SMC state per variant so e_prev resets cleanly between passes.
        # SMC is no-op on the inversion path (single-forward, no residual).
        smc_state = (
            SMCCFGState(lam=args.smc_cfg_lambda, alpha=args.smc_cfg_alpha)
            if args.smc_cfg
            else None
        )
        logger.info(
            "DirectEdit: %sinversion (T=%d, src_guidance=%.2f) -> edit "
            "(tar_guidance=%.2f, t_inj=%d, smc_cfg=%s)",
            tag,
            args.infer_steps,
            args.invert_guidance,
            args.guidance_scale,
            args.t_inj,
            (
                f"λ={args.smc_cfg_lambda},α={args.smc_cfg_alpha}"
                if args.smc_cfg
                else "off"
            ),
        )
        if ec_network is not None and args.easycontrol_edit_only:
            # Baseline inversion: no cond state, patched Block.forward falls
            # through to original_forward. Idempotent across variants.
            ec_network.clear_cond()
        z_inv, delta_z = directedit.invert(
            anima=anima,
            z_clean=z_clean,
            embed_src=e_src,
            embed_neg=embed_neg if args.invert_guidance != 1.0 else None,
            sigmas=sigmas,
            guidance_scale=args.invert_guidance,
        )
        if ec_network is not None and ec_invert_shift:
            with torch.no_grad():
                for b in ec_network.b_cond:
                    b -= ec_invert_shift
        if ec_network is not None and args.easycontrol_edit_only:
            ec_network.set_cond(ec_z_cond)
            ec_network.precompute_cond_kv()
            logger.info(
                "EasyControl: cond KV cache primed post-inversion (edit pass only)."
            )
        t_inj_blocks = (
            _parse_t_inj_blocks(args.t_inj_blocks, len(anima.blocks))
            if args.t_inj > 0
            else None
        )
        z_edit = directedit.edit_forward(
            anima=anima,
            z_init=z_inv[0],
            delta_z=delta_z,
            embed_tar=e_tar,
            embed_neg=embed_neg,
            sigmas=sigmas,
            guidance_scale=args.guidance_scale,
            embed_src=e_src if args.t_inj > 0 else None,
            t_inj=args.t_inj,
            t_inj_blocks=t_inj_blocks,
            z_inv=z_inv if args.t_inj > 0 else None,
            mask=anchor_mask,
            anchor_scale=args.anchor_scale,
            smc_cfg_state=smc_state,
        )
        z_edits.append((variant, z_edit))
        # In cached_embed mode ψ_tar == ψ_src, so the edit pass is pure
        # reconstruction — its latent MSE is what we correlate FM-error against.
        if fm_grid is not None and cached_variants is not None:
            with torch.no_grad():
                recon = (
                    ((z_edit.reshape(-1).float() - z_clean.reshape(-1).float()) ** 2)
                    .mean()
                    .item()
                )
            fm_rows[-1]["recon"] = recon

    if fm_rows:
        _log_fm_score_table(fm_rows)

    # Decode + save (one VAE re-mount for all variants). Drop the EC network
    # first — it holds the DiT ref (_dit) and per-block cond tensors, which
    # would otherwise defeat the `del anima` VRAM free below.
    if ec_network is not None:
        ec_network.clear_cond()
        del ec_network
    del anima
    clean_memory_on_device(device)
    vae.to(device)
    os.makedirs(args.save_path, exist_ok=True)
    src_stem = Path(args.image).stem
    for variant, z_edit in z_edits:
        with torch.no_grad():
            pixels = vae.decode_to_pixels(z_edit.to(device, dtype=vae.dtype))
        if pixels.ndim == 5:
            pixels = pixels.squeeze(2)
        pixels = pixels[0].to("cpu", dtype=torch.float32)
        base = f"{src_stem}_{variant}" if variant else src_stem
        # save_images reads args.seed + args.save_path + args.no_metadata.
        saved = save_images(pixels, args, original_base_name=base)
        logger.info("DirectEdit done -> %s.png", saved)


if __name__ == "__main__":
    main()
