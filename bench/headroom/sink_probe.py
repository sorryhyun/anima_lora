"""RQ1 / Phase-0 sink probe for the *headroom (register) tokens* proposal.

Design: ``_archive/proposals/headroom_register_tokens.md`` (merged PR #68). This is the
falsifiable kill gate that runs BEFORE any training or new network code: does the
base DiT actually manufacture a *relocatable, border-located, self-attention* sink
at generation time? If not, unprompted borders are a data-prior / VAE-edge problem
and the register line closes.

The proposal's causal story ("the sink commits early at σ≈0.8 and is frozen into the
layout") directly contradicts the DSR paper (arXiv:2605.05206), which finds DiT
outliers *strengthen as noise decreases* (late σ / intermediate layers). Per the
review, that σ-direction question is the PRIMARY axis here, not a footnote — so this
probe sweeps **layer-depth × σ**, splits **self- vs cross-attn**, and reports which
cell the sink actually lives in rather than assuming high-σ.

What it measures, per generation, on the *conditional* pass only (CFG uncond skipped):

  - PRIMARY sink signature: per-token **residual activation norm ‖x‖** at each block
    output, un-flattened onto the patch grid — the ViT/DSR high-norm outlier token. We
    report its outlier ratio (max/median), border-vs-interior concentration, and whether
    the top-percentile ‖x‖ tokens sit on the border ring. The gated self- vs cross-attn
    **contribution** norms are the "who writes it" split (sink must be self-attn-written,
    image→image, not cross-attn, image→text).
  - Border-vs-interior concentration of that norm (a distance-from-edge profile, plus a
    ring ratio), resolved into a depth×σ table.
  - Pixel correlation: decode the final image, and correlate high self-norm tokens with
    low-information (near-uniform) / extreme (near white/black) patches — the thing that
    separates a *rendered border sink* from the base-owned data prior.
  - Step-0 gate: the unprompted border-artifact rate on the decoded images, to confirm
    there is even a border to chase (guards against a solution seeking a problem).
  - ``cropped``-tag knockout: the seed reproducer (below) carries the tag ``cropped``,
    a likely learned data-prior trigger. Each prompt with ``cropped`` is also run with
    it removed; if the border sink collapses without the tag it is tag/data-prior-driven
    (RQ1 kill → "wants crop-conditioning / border aug, not headroom"), if it persists it
    is a genuine relocatable sink. Same knockout discipline as ``cross_attn_drive``
    (archived: ``_archive/bench/cross_attn_drive/``).

Kill criteria (any ⇒ register line closes): no high-norm self-attn sink; sink not
border-co-located; sink lives in cross-attn; **or** the step-0 border rate is ~0.

Usage::

    uv run python bench/headroom/sink_probe.py --seeds 3 --label base
    # narrower/faster smoke:
    uv run python bench/headroom/sink_probe.py --seeds 1 --infer_steps 20 --label smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import GenerationRequest, load_vae  # noqa: E402
from bench._anima import add_common_args, add_model_args  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima.models import Block  # noqa: E402
from library.inference.generation import generate_body  # noqa: E402
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.output import decode_latent, pixels_to_pil  # noqa: E402
from library.inference.text import ensure_text_strategies, prepare_text_inputs  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

# Block modules are named exactly "blocks.<N>" (no trailing dot).
_BLOCK_RE = re.compile(r"blocks\.(\d+)$")

# The user-confirmed thick-border reproducer (2026-07-01). Carries the tag
# ``cropped`` (knockout target) and ``from below``.
_REPRO = (
    "sensitive, 2girls, gotoh hitori, chitanda eru, bocchi the rock!, hyouka, "
    "@channel (castsation), playboy bunny, bunny ears, from below, cropped, "
    "covering privates, pantyhose, leotard, fishnets, presenting, long hair, "
    "indoors, looking at viewer, blush, embarrassed. They are standing side by "
    "side at the bar. gotoh hitori is holding a present box."
)

# A clean prompt that does NOT request any border/crop — negative control. The
# sink's border-concentration should be markedly weaker here if the mechanism is
# border-specific rather than a global artifact of every generation.
_CONTROL = (
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top"
    " and denim shorts is standing outdoors, full body visible, centered composition."
    " She's looking at the viewer with a smile. The background features trees and blue"
    " sky with clouds."
)

# depth buckets (fraction of network depth) × σ bands — the layer×σ sweep cells.
_DEPTH_BUCKETS = [("early", 0.0, 1 / 3), ("mid", 1 / 3, 2 / 3), ("late", 2 / 3, 1.01)]
_SIGMA_BANDS = [("hi", 0.80, 1.01), ("mid", 0.45, 0.80), ("low", -0.01, 0.45)]


def _depth_bucket(frac: float) -> str:
    for name, lo, hi in _DEPTH_BUCKETS:
        if lo <= frac < hi:
            return name
    return "late"


def _sigma_band(sig: float) -> str:
    for name, lo, hi in _SIGMA_BANDS:
        if lo <= sig < hi:
            return name
    return "low"


# ---------------------------------------------------------------------------
# Patch-grid geometry helpers.
# ---------------------------------------------------------------------------


def _edge_distance(h: int, w: int) -> np.ndarray:
    """(H,W) int grid: min patch-distance to any edge (0 = outermost ring)."""
    rr = np.minimum(np.arange(h)[:, None], (h - 1) - np.arange(h)[:, None])
    cc = np.minimum(np.arange(w)[None, :], (w - 1) - np.arange(w)[None, :])
    return np.minimum(np.broadcast_to(rr, (h, w)), np.broadcast_to(cc, (h, w)))


def _ring_ratio(map_hw: np.ndarray, dist: np.ndarray, ring: int) -> tuple[float, float]:
    """(border_mean, interior_mean) for tokens with dist<ring vs dist>=ring."""
    border = map_hw[dist < ring]
    interior = map_hw[dist >= ring]
    bm = float(border.mean()) if border.size else 0.0
    im = float(interior.mean()) if interior.size else 0.0
    return bm, im


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without scipy."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / denom) if denom else 0.0


def _patch_luma_stats(img_chw: torch.Tensor, hp: int, wp: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch luma std (uniformity, low=flat) and mean (extremeness), on the
    patch grid (hp,wp). ``img_chw`` is [-1,1] CHW at pixel resolution."""
    luma = img_chw.float().mean(0)  # (H_px, W_px), [-1,1]
    hpx, wpx = luma.shape
    bh, bw = hpx // hp, wpx // wp
    luma = luma[: bh * hp, : bw * wp].reshape(hp, bh, wp, bw)
    std = luma.std(dim=(1, 3)).cpu().numpy()
    mean = luma.mean(dim=(1, 3)).cpu().numpy()
    return std, mean


def _border_artifact_score(img_chw: torch.Tensor, frac: float, flat_thr: float, extreme_thr: float) -> float:
    """Fraction of the 4 image edges that look like a rendered border: a flat
    (low-variance) strip that is near white or black. Detector-free step-0 gate."""
    luma = img_chw.float().mean(0)  # (H,W) in [-1,1]
    h, w = luma.shape
    sh, sw = max(1, int(h * frac)), max(1, int(w * frac))
    strips = [luma[:sh, :], luma[-sh:, :], luma[:, :sw], luma[:, -sw:]]
    flagged = 0
    for s in strips:
        if float(s.std()) < flat_thr and abs(float(s.mean())) > extreme_thr:
            flagged += 1
    return flagged / 4.0


# ---------------------------------------------------------------------------
# Recorder + block-forward patch (mirrors Block._forward, bit-identical compute).
# ---------------------------------------------------------------------------


class SinkRecorder:
    def __init__(self, anima, ring: int, topk_frac: float):
        self.ring = ring
        self.topk_frac = topk_frac
        self.rows: list[dict] = []
        # depth×σ accumulators of full (H,W) maps, per pathway.
        self._cell_sum: dict[tuple, np.ndarray] = {}
        self._cell_n: dict[tuple, int] = {}
        # per-generation context set by the driver before each generate_body.
        self.gen = -1
        self.variant = ""
        self.prompt_name = ""
        self._sigma = None
        self._last_sigma = None
        self._step = -1
        self._pass = 0
        self._dist_cache: dict[tuple, np.ndarray] = {}
        idxs = sorted(
            int(_BLOCK_RE.search(n).group(1))
            for n, m in anima.named_modules()
            if isinstance(m, Block) and _BLOCK_RE.search(n)
        )
        self._depth = {b: (i / max(1, len(idxs) - 1)) for i, b in enumerate(idxs)}
        self._n_blocks = len(idxs)

    def begin_generation(self, prompt_name: str, variant: str):
        self.gen += 1
        self.prompt_name = prompt_name
        self.variant = variant
        self._sigma = None
        self._last_sigma = None
        self._step = -1
        self._pass = 0

    def on_model_forward(self, timesteps: torch.Tensor):
        sigma = float(timesteps.flatten()[0])
        if self._last_sigma is None or sigma != self._last_sigma:
            self._step += 1
            self._pass = 0
            self._last_sigma = sigma
            self._sigma = sigma
        else:
            self._pass += 1

    def _dist(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if key not in self._dist_cache:
            self._dist_cache[key] = _edge_distance(h, w)
        return self._dist_cache[key]

    @torch.no_grad()
    def record(self, block: int, x_out: torch.Tensor, self_contrib: torch.Tensor, cross_contrib: torch.Tensor):
        if self._pass != 0:  # conditional pass only
            return
        # (B,T,H,W,D) -> per-token norm (H,W); B=1, T=1 for single-seed image gen.
        # PRIMARY sink signature: residual-stream activation norm ‖x‖ (ViT/DSR
        # high-norm outlier token). Contribution norms are the "who writes it" split.
        act_map = x_out[0, 0].float().norm(dim=-1).cpu().numpy()
        self_map = self_contrib[0, 0].float().norm(dim=-1).cpu().numpy()
        cross_map = cross_contrib[0, 0].float().norm(dim=-1).cpu().numpy()
        h, w = act_map.shape
        dist = self._dist(h, w)
        flat = act_map.ravel()
        med = float(np.median(flat)) or 1e-9
        a_b, a_i = _ring_ratio(act_map, dist, self.ring)
        s_b, s_i = _ring_ratio(self_map, dist, self.ring)
        c_b, c_i = _ring_ratio(cross_map, dist, self.ring)
        # is the sink a border phenomenon? fraction of the top-percentile
        # highest-‖x‖ tokens that sit on the border ring (chance ≈ ring area frac).
        thr = float(np.quantile(flat, 1.0 - self.topk_frac))
        topk_mask = flat >= thr
        topk_border_frac = float((dist.ravel()[topk_mask] < self.ring).mean()) if topk_mask.any() else 0.0
        amax = int(act_map.argmax())
        frac = self._depth.get(block, 0.0)
        self.rows.append(
            {
                "gen": self.gen,
                "prompt": self.prompt_name,
                "variant": self.variant,
                "block": block,
                "depth_frac": round(frac, 4),
                "step": self._step,
                "sigma": round(self._sigma, 5),
                # activation (‖x‖) sink signature — PRIMARY
                "act_outlier": round(float(flat.max()) / med, 4),
                "act_border": round(a_b, 5),
                "act_interior": round(a_i, 5),
                "act_ratio": round(a_b / a_i, 5) if a_i else 0.0,
                "act_topk_border_frac": round(topk_border_frac, 4),
                "act_peak_edge_dist": int(dist.ravel()[amax]),
                # contribution split — who writes the residual on the border
                "self_border": round(s_b, 5),
                "cross_border": round(c_b, 5),
                "self_ratio": round(s_b / s_i, 5) if s_i else 0.0,
                "cross_ratio": round(c_b / c_i, 5) if c_i else 0.0,
            }
        )
        # accumulate full maps into depth×σ cells (full variant only — the
        # knockout variant is compared via the row-level ratios, and mixing it
        # into the mean maps would blur the pixel correlation).
        if self.variant == "full":
            cell = (_depth_bucket(frac), _sigma_band(self._sigma))
            for path, m in (("act", act_map), ("self", self_map), ("cross", cross_map)):
                k = (path, *cell, h, w)
                if k not in self._cell_sum:
                    self._cell_sum[k] = np.zeros((h, w), dtype=np.float64)
                    self._cell_n[k] = 0
                self._cell_sum[k] += m
                self._cell_n[k] += 1

    def cell_mean(self, path: str, depth: str, band: str, h: int, w: int):
        k = (path, depth, band, h, w)
        n = self._cell_n.get(k, 0)
        if not n:
            return None
        return self._cell_sum[k] / n


def _patched_forward(block: Block, rec: SinkRecorder, block_idx: int):
    """Mirror of ``Block._forward`` that additionally hands the gated self- and
    cross-attn contributions to the recorder. Bit-identical to the original
    computation (same submodule calls, same residual), so generation is unchanged.
    Copied structurally from _archive/bench/cross_attn_drive/attn_contribution.py."""

    def _forward(
        x_B_T_H_W_D,
        emb_B_T_D,
        crossattn_emb,
        attn_params,
        rope_cos_sin=None,
        adaln_lora_B_T_3D=None,
    ):
        if block.use_adaln_lora:
            fused_down = block.adaln_fused_down(emb_B_T_D)
            down_self, down_cross, down_mlp = fused_down.chunk(3, dim=-1)
            sh_s, sc_s, g_s = (block.adaln_up_self_attn(down_self) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            sh_c, sc_c, g_c = (block.adaln_up_cross_attn(down_cross) + adaln_lora_B_T_3D).chunk(3, dim=-1)
            sh_m, sc_m, g_m = (block.adaln_up_mlp(down_mlp) + adaln_lora_B_T_3D).chunk(3, dim=-1)
        else:
            sh_s, sc_s, g_s = block.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            sh_c, sc_c, g_c = block.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            sh_m, sc_m, g_m = block.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        def _b(t):
            return t[:, :, None, None, :]

        sh_s, sc_s, g_s = _b(sh_s), _b(sc_s), _b(g_s)
        sh_c, sc_c, g_c = _b(sh_c), _b(sc_c), _b(g_c)
        sh_m, sc_m, g_m = _b(sh_m), _b(sc_m), _b(g_m)

        B, T, H, W, D = x_B_T_H_W_D.shape

        def adaln(_x, _ln, _scale, _shift):
            return _ln(_x) * (1 + _scale) + _shift

        nx = adaln(x_B_T_H_W_D, block.layer_norm_self_attn, sc_s, sh_s)
        xf = nx.flatten(1, 3)
        r_self = block.self_attn(xf, attn_params, xf, rope_cos_sin=rope_cos_sin).unflatten(1, (T, H, W))
        self_contrib = g_s * r_self
        x_B_T_H_W_D = x_B_T_H_W_D + self_contrib

        nx = adaln(x_B_T_H_W_D, block.layer_norm_cross_attn, sc_c, sh_c)
        r_cross = block.cross_attn(
            nx.flatten(1, 3), attn_params, crossattn_emb, rope_cos_sin=rope_cos_sin
        ).unflatten(1, (T, H, W))
        cross_contrib = r_cross * g_c
        x_B_T_H_W_D = cross_contrib + x_B_T_H_W_D

        nx = adaln(x_B_T_H_W_D, block.layer_norm_mlp, sc_m, sh_m)
        mlp_contrib = g_m * block.mlp(nx)
        x_B_T_H_W_D = x_B_T_H_W_D + mlp_contrib

        rec.record(block_idx, x_B_T_H_W_D, self_contrib, cross_contrib)
        return x_B_T_H_W_D

    return _forward


def install_hooks(anima, rec: SinkRecorder):
    orig_fwd = anima.forward

    def model_fwd(latents, timesteps, *a, **kw):
        rec.on_model_forward(timesteps)
        return orig_fwd(latents, timesteps, *a, **kw)

    anima.forward = model_fwd
    for name, mod in anima.named_modules():
        m = _BLOCK_RE.search(name)
        if isinstance(mod, Block) and m:
            mod._forward = _patched_forward(mod, rec, int(m.group(1)))


# ---------------------------------------------------------------------------
# Prompt corpus (+ cropped-knockout expansion).
# ---------------------------------------------------------------------------


def build_corpus(args) -> list[dict]:
    """Each entry: {name, text, variant, expects_border}. Prompts containing the
    ``cropped`` tag get an extra drop_cropped variant (the RQ1 knockout)."""
    base = [
        {"name": "repro", "text": _REPRO, "expects_border": True},
        {"name": "control", "text": _CONTROL, "expects_border": False},
    ]
    if args.prompts_json:
        base = json.loads(Path(args.prompts_json).read_text())
    out = []
    for p in base:
        out.append({**p, "variant": "full"})
        if _drop_cropped(p["text"]) != p["text"]:
            out.append({**p, "text": _drop_cropped(p["text"]), "variant": "drop_cropped"})
    return out


def _drop_cropped(text: str) -> str:
    # remove the standalone ``cropped`` tag (comma-delimited), keep the rest.
    parts = [t.strip() for t in text.split(",")]
    kept = [t for t in parts if t.lower() != "cropped"]
    return ", ".join(kept)


# ---------------------------------------------------------------------------
# Aggregation + verdict.
# ---------------------------------------------------------------------------


def _median(xs):
    xs = sorted(xs)
    return float(xs[len(xs) // 2]) if xs else 0.0


def _border_conditioned(rows, per_gen, chance_border, cell_pred, args) -> dict:
    """The core RQ1 test: within repro/full, is the self-attn sink MORE
    edge-concentrated on border-POSITIVE generations than border-NEGATIVE ones?
    Same prompt + same `cropped` tag on both sides → controls for the data prior;
    the only difference is whether the border actually rendered."""
    pos_gens = {p["gen"] for p in per_gen if p["prompt"] == "repro" and p["variant"] == "full" and p["border_artifact_score"] > 0}
    neg_gens = {p["gen"] for p in per_gen if p["prompt"] == "repro" and p["variant"] == "full" and p["border_artifact_score"] == 0}

    def stats(gens):
        rs = [r for r in rows if r["gen"] in gens and r["prompt"] == "repro" and r["variant"] == "full" and cell_pred(r)]
        if not rs:
            return None
        enr = [r["act_topk_border_frac"] / chance_border for r in rs] if chance_border else [0.0]
        return {
            "n_gens": len(gens),
            "n_rows": len(rs),
            "outlier": round(_median([r["act_outlier"] for r in rs]), 3),
            "border_enrichment": round(_median(enr), 3),
            "peak_edge_dist": round(_median([r["act_peak_edge_dist"] for r in rs]), 2),
            "self_border": round(_median([r["self_border"] for r in rs]), 2),
            "cross_border": round(_median([r["cross_border"] for r in rs]), 2),
        }

    pos, neg = stats(pos_gens), stats(neg_gens)
    out = {"n_border_pos": len(pos_gens), "n_border_neg": len(neg_gens), "pos": pos, "neg": neg}
    if pos and neg:
        out["enrichment_delta"] = round(pos["border_enrichment"] - neg["border_enrichment"], 3)
        out["outlier_delta"] = round(pos["outlier"] - neg["outlier"], 3)
    return out


def aggregate(rec: SinkRecorder, border_scores: dict, per_gen, patch_shape, args) -> dict:
    rows = rec.rows

    def sel(variant=None, prompt=None):
        return [
            r
            for r in rows
            if (variant is None or r["variant"] == variant)
            and (prompt is None or r["prompt"] == prompt)
        ]

    # layer×σ table over the ACTIVATION-norm sink signature (‖x‖), repro/full —
    # that's where a border-located high-norm sink should live if RQ1 holds.
    # `self_gt_cross` = does self-attn (not cross) write the border residual there.
    table = {}
    repro_full = sel(variant="full", prompt="repro")
    for dname, _, _ in _DEPTH_BUCKETS:
        for bname, _, _ in _SIGMA_BANDS:
            cellrows = [
                r
                for r in repro_full
                if _depth_bucket(r["depth_frac"]) == dname and _sigma_band(r["sigma"]) == bname
            ]
            if cellrows:
                table[f"{dname}/{bname}"] = {
                    "act_ratio": round(_median([r["act_ratio"] for r in cellrows]), 4),
                    "act_outlier": round(_median([r["act_outlier"] for r in cellrows]), 3),
                    "act_topk_border_frac": round(_median([r["act_topk_border_frac"] for r in cellrows]), 4),
                    "act_peak_edge_dist": round(_median([r["act_peak_edge_dist"] for r in cellrows]), 2),
                    "self_border": round(_median([r["self_border"] for r in cellrows]), 4),
                    "cross_border": round(_median([r["cross_border"] for r in cellrows]), 4),
                    "n": len(cellrows),
                }

    # peak cell = strongest border-located activation sink. A sink is a *sparse*
    # set of high-norm tokens, so the discriminator is top-percentile border
    # enrichment (not the border MEAN, which a handful of tokens can't move),
    # gated on the cell actually having outlier tokens at all.
    def _sink_strength(v):
        has_outlier = 1.0 if v["act_outlier"] >= args.outlier_min else 0.01
        return has_outlier * max(v["act_topk_border_frac"], 1e-6)

    peak_key, peak = None, None
    for k, v in table.items():
        if peak is None or _sink_strength(v) > _sink_strength(peak):
            peak_key, peak = k, v

    # pixel correlation at the peak cell: high ‖x‖ ↔ low luma-std (uniform patch)?
    # This is the DSR "outlier token sits on corrupted/low-info patch" claim.
    pixel_corr = {}
    if peak_key is not None and patch_shape is not None:
        hp, wp = patch_shape
        dname, bname = peak_key.split("/")
        amap = rec.cell_mean("act", dname, bname, hp, wp)
        ustd = border_scores.get("_uniformity_map")  # mean per-patch luma std (full/repro)
        if amap is not None and ustd is not None and ustd.shape == amap.shape:
            # positive corr(‖x‖, -std) ⇒ sink sits on flat/uniform patches.
            pixel_corr = {
                "spearman_actnorm_vs_neg_lumastd": round(_spearman(amap, -ustd), 4),
                "cell": peak_key,
            }

    chance_border = None  # fraction of tokens in the ring; enrichment baseline
    if patch_shape is not None:
        hp, wp = patch_shape
        d = _edge_distance(hp, wp)
        chance_border = round(float((d < rec.ring).mean()), 4)

    def _enrich(frac):
        return round(frac / chance_border, 3) if chance_border else None

    # knockout: repro border enrichment full vs drop_cropped in the peak depth×σ
    # cell — does the sink's edge-bias survive removing the `cropped` tag?
    knockout = {}
    if peak_key is not None:
        dname, bname = peak_key.split("/")
        for variant in ("full", "drop_cropped"):
            vr = [
                r
                for r in sel(variant=variant, prompt="repro")
                if _depth_bucket(r["depth_frac"]) == dname and _sigma_band(r["sigma"]) == bname
            ]
            if vr:
                knockout[variant] = _enrich(_median([r["act_topk_border_frac"] for r in vr]))

    # control contrast: does the sink concentrate less on a no-border prompt?
    control_enrich = None
    if peak_key is not None:
        dname, bname = peak_key.split("/")
        cr = [
            r
            for r in sel(variant="full", prompt="control")
            if _depth_bucket(r["depth_frac"]) == dname and _sigma_band(r["sigma"]) == bname
        ]
        if cr:
            control_enrich = _enrich(_median([r["act_topk_border_frac"] for r in cr]))

    step0_rate = border_scores.get("repro_full_border_rate", 0.0)
    peak_enrich = _enrich(peak["act_topk_border_frac"]) if peak else None

    # the core RQ1 test: sink signature on border-POS vs border-NEG repro gens,
    # evaluated at the peak depth×σ cell (where the sink is strongest).
    conditioned = {}
    if peak_key is not None and chance_border:
        dname, bname = peak_key.split("/")

        def _peakcell(r):
            return _depth_bucket(r["depth_frac"]) == dname and _sigma_band(r["sigma"]) == bname

        conditioned = _border_conditioned(rows, per_gen, chance_border, _peakcell, args)

    # ---- verdict (descriptive; the human decides) -------------------------
    # RQ1 is decided by the border-CONDITIONED contrast, not the prompt-average:
    # the border is a rare stochastic event, so a causal border sink must be
    # markedly stronger on border-positive gens than border-negative ones.
    reasons = []
    passed = True
    n_pos = conditioned.get("n_border_pos", 0)
    if n_pos == 0:
        passed = False
        reasons.append(f"no border-positive gens across {args.seeds} seeds (rate {step0_rate:.2f}) — raise --seeds; nothing to condition on")
    elif peak is None:
        passed = False
        reasons.append("no cells recorded")
    elif peak["act_outlier"] < args.outlier_min:
        passed = False
        reasons.append(f"no high-norm outlier tokens (peak act_outlier {peak['act_outlier']:.2f} < {args.outlier_min}) — no sink to relocate")
    elif peak["self_border"] <= peak["cross_border"]:
        passed = False
        reasons.append(f"sink not self-attn-written at {peak_key} (self {peak['self_border']:.1f} <= cross {peak['cross_border']:.1f})")
    elif conditioned.get("enrichment_delta") is None:
        passed = False
        reasons.append(f"only {n_pos} border-positive gens and no border-negative comparison — raise --seeds")
    elif conditioned["enrichment_delta"] < args.delta_min:
        passed = False
        reasons.append(
            f"sink NOT border-conditioned: edge-enrichment barely differs pos vs neg (Δ={conditioned['enrichment_delta']}× < {args.delta_min}×; "
            f"pos {conditioned['pos']['border_enrichment']}× vs neg {conditioned['neg']['border_enrichment']}×) — the sink exists but does not track the rendered border"
        )
    if knockout.get("full") and knockout.get("drop_cropped") is not None:
        if knockout["drop_cropped"] < args.enrich_min <= knockout["full"]:
            reasons.append(
                f"CAUTION: sink edge-bias collapses without `cropped` tag ({knockout['full']}×→{knockout['drop_cropped']}×) — data-prior contribution"
            )

    verdict = "PASS" if passed else "KILL"
    return {
        "verdict": verdict,
        "verdict_reasons": reasons or ["all gates cleared — see layer_sigma_table for onset σ/depth"],
        "peak_cell": peak_key,
        "peak": peak,
        "peak_border_enrichment": peak_enrich,
        "chance_border_frac": chance_border,
        "border_conditioned": conditioned,
        "layer_sigma_table": table,
        "pixel_corr": pixel_corr,
        "cropped_knockout_enrichment": knockout,
        "control_border_enrichment_at_peak": control_enrich,
        "step0_repro_border_rate": round(step0_rate, 4),
        "n_rows": len(rows),
        "n_blocks": rec._n_blocks,
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def build_args():
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    add_common_args(p)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--prompts_json", default=None, help="Optional [{name,text,expects_border}] override.")
    p.add_argument("--border_ring", type=int, default=2, help="Ring width (patches) counted as border.")
    p.add_argument("--border_px_frac", type=float, default=0.04, help="Outer image fraction scanned for border artifact.")
    p.add_argument("--flat_thr", type=float, default=0.05, help="Luma-std below this = flat strip.")
    p.add_argument("--extreme_thr", type=float, default=0.80, help="|luma-mean| above this = near white/black.")
    p.add_argument("--step0_min", type=float, default=0.15, help="Min repro border rate to have a border to chase.")
    p.add_argument("--outlier_min", type=float, default=3.0, help="Min max/median ‖x‖ to call it a high-norm outlier sink.")
    p.add_argument("--topk_frac", type=float, default=0.02, help="Top fraction of ‖x‖ tokens treated as the sink set.")
    p.add_argument("--enrich_min", type=float, default=1.50, help="Min border enrichment (topk_border_frac / chance) to call the sink border-located.")
    p.add_argument("--delta_min", type=float, default=0.50, help="Min pos-vs-neg enrichment gap (×chance) to call the sink border-CONDITIONED (the RQ1 gate).")
    p.add_argument("--save_images", action="store_true", help="Save every decoded image under <run>/images/ for eyeballing.")
    p.add_argument("--save_border_images", action="store_true", help="Save only border-positive decoded images (rare-event capture).")
    return p.parse_args()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    req = GenerationRequest(
        prompt="placeholder",
        dit=args.dit,
        vae=args.vae,
        text_encoder=args.text_encoder,
        negative_prompt=args.negative_prompt,
        image_size=(args.size, args.size),
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg,
        flow_shift=args.flow_shift,
        attn_mode=args.attn_mode,
        device=args.device,
        seed=0,
    )
    gen_args = req.to_args()
    gen_args.compile_blocks = False  # patched _forward must stay eager

    ensure_text_strategies(args.text_encoder)
    text_encoder = load_text_encoder(gen_args, dtype=torch.bfloat16, device=device)
    text_encoder.eval()
    shared = {"text_encoder": text_encoder, "conds_cache": {}}
    anima = load_dit_model(gen_args, device, torch.bfloat16)
    shared["model"] = anima

    vae = load_vae(args.vae, device="cpu", vae_2d=True)
    vae.to(torch.bfloat16).eval()

    corpus = build_corpus(args)
    rec = SinkRecorder(anima, ring=args.border_ring, topk_frac=args.topk_frac)
    install_hooks(anima, rec)

    out_dir = make_run_dir("headroom", args.label)
    img_dir = out_dir / "images"
    if args.save_images or args.save_border_images:
        img_dir.mkdir(exist_ok=True)

    # pixel-side accumulators (full/repro only, aligned to patch grid).
    uni_sum = None
    uni_n = 0
    patch_shape = None
    repro_border_scores: list[float] = []
    per_gen = []

    total = len(corpus) * args.seeds
    done = 0
    for entry in corpus:
        ctx, ctx_null = prepare_text_inputs(gen_args, device, anima, shared, prompt=entry["text"])
        for s in range(args.seeds):
            rec.begin_generation(entry["name"], entry["variant"])
            gen_id = rec.gen
            with torch.no_grad():
                latent = generate_body(gen_args, anima, ctx, ctx_null, device, seed=s)
            img = decode_latent(vae, latent, device)  # (3,H,W) [-1,1] on cpu
            clean_memory_on_device(device)

            bscore = _border_artifact_score(img, args.border_px_frac, args.flat_thr, args.extreme_thr)
            if args.save_images or (args.save_border_images and bscore > 0):
                pixels_to_pil(img).save(img_dir / f"{entry['name']}_{entry['variant']}_s{s}_b{bscore:.2f}.png")
            per_gen.append(
                {
                    "gen": gen_id,
                    "prompt": entry["name"],
                    "variant": entry["variant"],
                    "seed": s,
                    "expects_border": entry.get("expects_border", False),
                    "border_artifact_score": round(bscore, 4),
                }
            )
            if entry["name"] == "repro" and entry["variant"] == "full":
                repro_border_scores.append(bscore)
                # patch-grid luma uniformity, aligned to the self-norm maps.
                hp = wp = None
                # infer patch grid from any recorded row's map shape via last cell
                for k in rec._cell_sum:
                    if k[0] == "self":
                        hp, wp = k[-2], k[-1]
                        break
                if hp:
                    patch_shape = (hp, wp)
                    ustd, _ = _patch_luma_stats(img, hp, wp)
                    uni_sum = ustd if uni_sum is None else uni_sum + ustd
                    uni_n += 1
            done += 1
            print(f"[{done}/{total}] {entry['name']}/{entry['variant']} seed={s} border={bscore:.2f}", flush=True)

    border_scores = {
        "repro_full_border_rate": _median(repro_border_scores) if repro_border_scores else 0.0,
    }
    if uni_sum is not None and uni_n:
        border_scores["_uniformity_map"] = uni_sum / uni_n

    agg = aggregate(rec, border_scores, per_gen, patch_shape, args)

    with (out_dir / "rows.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rec.rows[0].keys()))
        w.writeheader()
        w.writerows(rec.rows)
    with (out_dir / "per_gen.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_gen[0].keys()))
        w.writeheader()
        w.writerows(per_gen)

    metrics = {
        "n_prompts": len(corpus),
        "n_seeds": args.seeds,
        "size": args.size,
        "infer_steps": args.infer_steps,
        "cfg": args.cfg,
        **{k: v for k, v in agg.items()},
    }
    write_result(
        out_dir,
        script="bench/headroom/sink_probe.py",
        label=args.label,
        args=args,
        metrics=metrics,
        artifacts=["rows.csv", "per_gen.csv"],
    )

    print("\n" + "=" * 70)
    print(f"VERDICT: {agg['verdict']}")
    for r in agg["verdict_reasons"]:
        print(f"  - {r}")
    print(f"peak cell (layer/σ): {agg['peak_cell']}  border_enrichment={agg['peak_border_enrichment']}×  peak={agg['peak']}")
    print(f"step-0 repro border rate: {agg['step0_repro_border_rate']}  (border ring chance frac ≈ {agg['chance_border_frac']})")
    bc = agg["border_conditioned"]
    print("\nBORDER-CONDITIONED contrast @ peak cell (the RQ1 gate):")
    print(f"  border-positive gens: {bc.get('n_border_pos')}  border-negative gens: {bc.get('n_border_neg')}")
    print(f"  POS: {bc.get('pos')}")
    print(f"  NEG: {bc.get('neg')}")
    print(f"  Δ enrichment (pos-neg): {bc.get('enrichment_delta')}×   Δ outlier: {bc.get('outlier_delta')}")
    print(f"cropped knockout (enrichment): {agg['cropped_knockout_enrichment']}")
    print(f"control enrichment @peak: {agg['control_border_enrichment_at_peak']}")
    print(f"pixel corr: {agg['pixel_corr']}")
    print("\nlayer×σ activation-sink table (repro/full):")
    print(f"  {'cell':12s} {'act_ratio':>9s} {'outlier':>8s} {'topk_bord':>9s} {'peak_dist':>9s} {'self_b':>7s} {'cross_b':>7s}  n")
    for k, v in agg["layer_sigma_table"].items():
        print(
            f"  {k:12s} {v['act_ratio']:9.3f} {v['act_outlier']:8.2f} "
            f"{v['act_topk_border_frac']:9.3f} {v['act_peak_edge_dist']:9.2f} "
            f"{v['self_border']:7.3f} {v['cross_border']:7.3f}  {v['n']}"
        )
    print(f"\nresults → {out_dir}")


if __name__ == "__main__":
    main()
