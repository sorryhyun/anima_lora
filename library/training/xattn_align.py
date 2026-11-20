"""Cross-attn alignment term for quoted OCR strings (Anima LoRA training).

Proposal and evidence: ``docs/anima/proposal/xattn_align_loss.md`` (probes in
``bench/xattn_loc/``). On a training item whose caption carries a text clause
(``Japanese text reads as "…"``) and whose OCR sidecar holds the line's box,
at a step with σ ≥ ``xattn_align_sigma_min``:

- ``A_b[q, c]``: block ``b``'s cross-attn probability from image token ``q``
  to T5 position ``c``, head mean, softmax over every key as the model
  computes it; ``b`` in ``xattn_align_blocks`` (8, 13, 14, 15).
- string ``s`` (positions ``C_s``, line box ``R_s``):
  ``p_s(q) ∝ Σ_b Σ_{c ∈ C_s} A_b[q, c]``, normalised over ``q``;
  ``box`` form ``L_s = −log Σ_{q ∈ R_s} p_s(q)``; ``contrast`` form
  ``L_s = −log own / (own + other text boxes)`` (strings on pages with
  another line only).
- the ``speech bubble`` / ``japanese text`` tag tokens: the ``box`` form
  against the union of the page's usable OCR boxes.
- ``L = mean_s L_s + L_tag``, weighted by ``xattn_align_weight`` in
  ``losses._xattn_align_loss``.

Mechanics. ``Block._forward`` is compiled, so the query cannot be read inside
it. A forward pre-hook on each listed block (it runs in ``block.__call__``,
outside the compiled frame) re-runs the block's self-attn half and the
cross-attn input norm eagerly with grad, then the cross-attn q / k. Block swap
is refused (the extra graph holds block weights for backward).

The caption variant the loader drew (``v0`` – ``v3``) is not in the batch:
the adapter identifies it by matching the step's context against each cached
variant's per-row sums. A dropped caption matches none and the item is left
out of the term for that step.

``xattn_align_calib = true`` with ``xattn_align_weight = 0`` is a measure-only
run: on every term-bearing step it logs the gradient norms of the FM loss (the
``.grad`` delta of the real backward) and of the term (a fresh forward to the
last listed block after that backward, one ``autograd.grad``) on the trainable
parameters (``<output_name>_xattn_align.jsonl``); neither touches the update.
Two backwards through one compiled graph (``retain_graph``) give NaN weights.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch.utils.checkpoint

from library.env import resolve_under_home
from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
)

logger = logging.getLogger(__name__)

OCR_DIR = "post_image_dataset/ocr"
TOK_PX = 16  # VAE 8 × patch 2
PAD = 0.5  # box dilation, image tokens
TAGS = ("speech bubble", "japanese text")
CLAUSE_RE = re.compile(r"([A-Za-z]+) (text|SFX) reads as ")
MATCH_TOL = 0.02  # relative L1 of row sums; a dropped caption sits far above
QCHUNK = 1024  # image queries per logits chunk


def clause_strings(cap: str) -> list[dict]:
    """Every quoted string of the caption's text clauses: ``[a, b)`` spans of
    the content in ``cap`` (same parse as ``bench/xattn_loc/probe.py``)."""
    out = []
    for m in CLAUSE_RE.finditer(cap):
        i = m.end()
        while i < len(cap) and cap[i] == '"':
            j = cap.find('"', i + 1)
            if j < 0:
                break
            out.append({"a": i + 1, "b": j})
            i = j + 1
            if cap.startswith(", ", i):
                i += 2
    return out


def region(box_px, gh: int, gw: int, pad: float = PAD) -> torch.Tensor:
    """Image tokens whose centre lies in the box dilated by ``pad`` tokens
    (the box's centre token always in); flat ``(gh·gw,)`` bool."""
    x0, y0, x1, y1 = (v / TOK_PX for v in box_px)
    ys = torch.arange(gh, dtype=torch.float32)[:, None] + 0.5
    xs = torch.arange(gw, dtype=torch.float32)[None, :] + 0.5
    m = (xs >= x0 - pad) & (xs <= x1 + pad) & (ys >= y0 - pad) & (ys <= y1 + pad)
    cy = min(gh - 1, max(0, int((y0 + y1) / 2)))
    cx = min(gw - 1, max(0, int((x0 + x1) / 2)))
    m[cy, cx] = True
    return m.flatten()


def read_variants(txt: Path) -> dict[int, str]:
    var = txt.with_suffix(".variants.txt")
    out: dict[int, str] = {}
    if var.exists():
        for line in var.read_text("utf-8").splitlines():
            m = re.match(r"v(\d+)\t(.*)$", line)
            if m:
                out[int(m.group(1))] = m.group(2)
    if not out and txt.exists():
        out[0] = txt.read_text("utf-8").strip()
    return out


def block_cross_probs(
    block,
    x: torch.Tensor,
    emb: torch.Tensor,
    crossattn_emb: torch.Tensor,
    attn_params,
    rope_cos_sin,
    adaln_lora: Optional[torch.Tensor],
    cols: torch.Tensor,
) -> torch.Tensor:
    """``(B, Nq, n)``: the block's head-mean cross-attn probability at the key
    columns ``cols``, softmax over every key. Mirrors ``Block._forward`` up to
    the cross-attn call (layout-agnostic: eager 5D or fake-5D flatten)."""
    if block.use_adaln_lora:
        down_self, down_cross, _ = block.adaln_fused_down(emb).chunk(3, dim=-1)
        sh_s, sc_s, g_s = (block.adaln_up_self_attn(down_self) + adaln_lora).chunk(
            3, dim=-1
        )
        sh_c, sc_c, _ = (block.adaln_up_cross_attn(down_cross) + adaln_lora).chunk(
            3, dim=-1
        )
    else:
        sh_s, sc_s, g_s = block.adaln_modulation_self_attn(emb).chunk(3, dim=-1)
        sh_c, sc_c, _ = block.adaln_modulation_cross_attn(emb).chunk(3, dim=-1)

    def e(t):
        return t[:, :, None, None, :]

    B, T, H, W, D = x.shape
    nx = block.layer_norm_self_attn(x) * (1 + e(sc_s)) + e(sh_s)
    flat = nx.flatten(1, 3)
    res = block.self_attn(flat, attn_params, flat, rope_cos_sin=rope_cos_sin)
    x = x + e(g_s) * res.unflatten(1, (T, H, W))
    nx = block.layer_norm_cross_attn(x) * (1 + e(sc_c)) + e(sh_c)
    ca = block.cross_attn
    q, k, _ = ca.compute_qkv(nx.flatten(1, 3), crossattn_emb)
    hd = q.shape[-1]
    scale = (
        attn_params.softmax_scale
        if attn_params is not None and attn_params.softmax_scale
        else hd**-0.5
    )
    out = []
    with torch.autocast(q.device.type, enabled=False):
        kf = k.float()
        for i in range(0, q.shape[1], QCHUNK):
            lg = torch.einsum("bqhd,bkhd->bhqk", q[:, i : i + QCHUNK].float(), kf)
            lg = lg * scale
            lse = torch.logsumexp(lg, dim=-1, keepdim=True)
            out.append(torch.exp(lg[..., cols] - lse).mean(1))  # (B, chunk, n)
    return torch.cat(out, dim=1)


class XattnAlignMethodAdapter(MethodAdapter):
    name = "xattn_align"
    sigma_demote_safe = False  # boxes are on the native grid

    def __init__(self) -> None:
        self._items: dict[str, Optional[dict]] = {}
        self._handles = []
        self._maps: dict[int, torch.Tensor] = {}
        self._cur: Optional[dict] = None  # item for this step
        self._var: Optional[dict] = None  # its matched variant
        self._live = False
        self._metrics: dict[str, float] = {}
        self._encoder = None
        self._micro = 0
        self._n_term = 0
        self._skips: dict[str, int] = {}
        self._log_path: Optional[str] = None

    # ------------------------------------------------------------ setup

    def on_network_built(self, ctx: SetupCtx) -> None:
        net, args = ctx.network, ctx.args
        self._weight = float(getattr(net, "_xattn_align_weight", 0.0) or 0.0)
        self._form = str(getattr(net, "_xattn_align_form", "box"))
        self._blocks = list(getattr(net, "_xattn_align_blocks", (8, 13, 14, 15)))
        self._sigma_min = float(getattr(net, "_xattn_align_sigma_min", 0.6))
        self._calib = bool(getattr(net, "_xattn_align_calib", False))
        self._tags = bool(getattr(net, "_xattn_align_tags", True))
        if self._form not in ("box", "contrast"):
            raise ValueError(
                f"xattn_align_form must be box | contrast, got {self._form!r}"
            )
        if int(getattr(args, "blocks_to_swap", 0) or 0) > 0:
            raise ValueError("xattn_align: blocks_to_swap must be 0")
        if getattr(args, "sigma_lowres", False):
            raise ValueError("xattn_align: sigma_lowres moves the grid off the boxes")
        if getattr(args, "flip_aug", False) or getattr(args, "random_crop", False):
            raise ValueError("xattn_align: flip / crop augmentation moves the boxes")
        if int(getattr(args, "train_batch_size", 1) or 1) != 1:
            logger.warning(
                "xattn_align: written for batch 1; extra samples are skipped"
            )
        self._cache_root = resolve_under_home(
            getattr(args, "lora_cache_dir", None) or "post_image_dataset/lora"
        )
        blocks = ctx.unet.blocks
        for b in self._blocks:
            if not (0 <= b < len(blocks)):
                raise ValueError(f"xattn_align block {b} out of range ({len(blocks)})")
            self._handles.append(
                blocks[b].register_forward_pre_hook(
                    self._make_hook(b), with_kwargs=True
                )
            )
        self._handles.append(
            ctx.unet.register_forward_pre_hook(self._model_hook, with_kwargs=True)
        )
        out_dir = str(getattr(args, "output_dir", "") or ".")
        name = str(getattr(args, "output_name", "") or "anima")
        os.makedirs(out_dir, exist_ok=True)
        self._log_path = os.path.join(out_dir, f"{name}_xattn_align.jsonl")
        open(self._log_path, "w").close()
        self._params = [p for p in net.parameters() if p.requires_grad]
        ctx.accelerator.print(
            f"xattn_align[{self._form}]: blocks {self._blocks}, σ ≥ {self._sigma_min}, "
            f"weight {self._weight:g}{', calib' if self._calib else ''}, "
            f"tags {'on' if self._tags else 'off'}; log {self._log_path}"
        )

    def _get_encoder(self):
        if self._encoder is None:
            from library.anima.text_strategies import TokenizeStrategy

            st = TokenizeStrategy.get_strategy()
            if st is None or not hasattr(st, "encoder"):
                raise RuntimeError("xattn_align: no tokenize strategy installed")
            self._encoder = st.encoder
        return self._encoder

    def _skip(self, why: str) -> None:
        self._skips[why] = self._skips.get(why, 0) + 1

    def _item(self, image_key: str) -> Optional[dict]:
        """Per image: every variant's string columns, boxes, tag columns and
        context fingerprint; ``None`` when the image has no matched string."""
        if image_key in self._items:
            return self._items[image_key]
        from anime_tools.captions.ocr_sidecar import read_ocr, text_boxes, usable_lines
        from safetensors import safe_open

        img = Path(image_key)
        st, stem = img.parent.name, img.stem
        item = None
        side = resolve_under_home(OCR_DIR) / st / f"{stem}.ocr.txt"
        te = self._cache_root / st / f"{stem}_anima_te.safetensors"
        caps = read_variants(img.with_suffix(".txt"))
        if (
            side.exists()
            and te.exists()
            and any(" reads as " in c for c in caps.values())
        ):
            lines = read_ocr(side)
            boxes = text_boxes(lines)
            all_boxes = [list(ln.box) for ln in usable_lines(lines)]
            enc = self._get_encoder()
            variants = []
            with safe_open(str(te), "pt") as f:
                keys = set(f.keys())
                for vi, cap in sorted(caps.items()):
                    ck, mk = f"crossattn_emb_v{vi}", f"t5_attn_mask_v{vi}"
                    if ck not in keys:
                        ck, mk = "crossattn_emb", "t5_attn_mask"
                        if vi != 0 or ck not in keys:
                            continue
                    _, mask, offs = enc.encode_aligned(cap, 512)
                    if int(f.get_tensor(mk).sum()) != int(sum(mask)):
                        self._skip("mask_mismatch")
                        continue
                    strings = []
                    for s in clause_strings(cap):
                        text = cap[s["a"] : s["b"]]
                        box = boxes.get(text.strip())
                        cols = [
                            j
                            for j, (x, y) in enumerate(offs[:-1])
                            if y > x and x >= s["a"] and y <= s["b"]
                        ]
                        if box is None or not cols:
                            self._skip(
                                "string_no_box" if box is None else "string_no_cols"
                            )
                            continue
                        strings.append({"text": text, "box": list(box), "cols": cols})
                    tag_cols = sorted(
                        {
                            j
                            for t in TAGS
                            for m in re.finditer(re.escape(t), cap)
                            for j, (x, y) in enumerate(offs[:-1])
                            if y > x and x >= m.start() and y <= m.end()
                        }
                    )
                    if not strings:
                        continue
                    fp = f.get_tensor(ck).float().sum(-1)
                    variants.append(
                        {"v": vi, "strings": strings, "tag_cols": tag_cols, "fp": fp}
                    )
            if variants:
                item = {
                    "key": f"{st}/{stem}",
                    "variants": variants,
                    "all_boxes": all_boxes,
                }
        self._items[image_key] = item
        return item

    # ------------------------------------------------------------ hooks

    def _model_hook(self, _mod, args, kwargs):
        """The DiT call: σ gate and variant match for this step."""
        self._live = False
        self._maps = {}
        if self._cur is None:
            return
        t = args[1] if len(args) > 1 else kwargs.get("timesteps")
        c = args[2] if len(args) > 2 else kwargs.get("context")
        if t is None or c is None:
            return
        sigma = float(t.reshape(-1)[0])
        self._sigma = sigma
        if sigma < self._sigma_min:
            return
        row = c[0].detach().float().sum(-1)
        best, dist = None, math.inf
        for v in self._cur["variants"]:
            fp = v["fp"].to(row.device)
            if fp.shape != row.shape:
                continue
            d = float((row - fp).abs().mean() / fp.abs().mean().clamp_min(1e-6))
            if d < dist:
                best, dist = v, d
        self._match_dist = dist
        if best is None or dist > MATCH_TOL:
            self._skip("no_variant_match")
            return
        self._var = best
        cols = sorted(
            {c_ for s in best["strings"] for c_ in s["cols"]}
            | (set(best["tag_cols"]) if self._tags else set())
        )
        self._cols = cols
        self._cols_t = torch.tensor(cols, device=c.device)
        self._live = True

    def _make_hook(self, b: int):
        def hook(block, args, kwargs):
            if not self._live:
                return
            x, emb, ctx_emb, attn_params = args[:4]
            rope = kwargs.get("rope_cos_sin", args[4] if len(args) > 4 else None)
            ada = kwargs.get("adaln_lora_B_T_3D", args[5] if len(args) > 5 else None)
            # Checkpointed: only the inputs (already held by the primary
            # graph) and the (Nq, n) output stay; backward recomputes.
            self._maps[b] = torch.utils.checkpoint.checkpoint(
                block_cross_probs,
                block, x, emb, ctx_emb, attn_params, rope, ada, self._cols_t,
                use_reentrant=False,
            )  # fmt: skip

        return hook

    # ------------------------------------------------------------ step

    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        self._cur, self._var, self._live, self._maps = None, None, False, {}
        if not is_train:
            return
        self._micro += 1
        keys = batch.get("image_keys") if isinstance(batch, dict) else None
        if not keys:
            return
        item = self._item(str(keys[0]))
        if item is None:
            return
        self._cur = item
        self._hw = (int(latents.shape[-2]), int(latents.shape[-1]))

    def _loss(self) -> tuple[torch.Tensor, dict]:
        h, w = self._hw
        gh, gw = h // 2, w // 2
        at = {c: i for i, c in enumerate(self._cols)}
        M = torch.stack([self._maps[b][0] for b in self._blocks]).sum(0)  # (Nq, n)
        if M.shape[0] != gh * gw:
            raise RuntimeError(f"xattn_align: {M.shape[0]} queries vs grid {gh}x{gw}")
        dev = M.device
        var = self._var
        regs = [region(s["box"], gh, gw).to(dev) for s in var["strings"]]
        union = torch.zeros(gh * gw, dtype=torch.bool, device=dev)
        for bx in self._cur["all_boxes"]:
            union |= region(bx, gh, gw).to(dev)
        terms, lifts, shares = [], [], []
        for s, r in zip(var["strings"], regs):
            P = M[:, [at[c] for c in s["cols"]]].sum(-1).float()
            tot = P.sum().clamp_min(1e-20)
            own = P[r].sum()
            lifts.append(float((own / tot / r.float().mean()).detach()))
            other = union & ~r
            if other.any():
                o = P[other].sum()
                shares.append(float((own / (own + o)).detach()))
            if self._form == "box":
                terms.append(-torch.log((own / tot).clamp_min(1e-8)))
            elif other.any():
                terms.append(-torch.log((own / (own + P[other].sum())).clamp_min(1e-8)))
        out = {"lift": sum(lifts) / len(lifts), "n_str": len(lifts)}
        if shares:
            out["own_share"] = sum(shares) / len(shares)
        loss = torch.stack(terms).mean() if terms else None
        if self._tags and var["tag_cols"] and union.any():
            P = M[:, [at[c] for c in var["tag_cols"]]].sum(-1).float()
            sh = P[union].sum() / P.sum().clamp_min(1e-20)
            out["tag_lift"] = float((sh / union.float().mean()).detach())
            lt = -torch.log(sh.clamp_min(1e-8))
            loss = lt if loss is None else loss + lt
        return loss, out

    def extra_forwards(self, ctx: StepCtx, primary: ForwardArtifacts) -> Optional[dict]:
        if not primary.is_train:
            return None
        if not self._live:
            self._metrics["xattn_align/active"] = 0.0
            return None
        self._live = False
        missing = [b for b in self._blocks if b not in self._maps]
        if missing:
            if not getattr(self, "_warned_no_capture", False):
                logger.warning("xattn_align: block hooks %s did not fire", missing)
                self._warned_no_capture = True
            return None
        loss, info = self._loss()
        self._maps = {}
        if loss is None:
            self._skip("contrast_one_line")
            return None
        self._n_term += 1
        rec = {
            "micro": self._micro, "item": self._cur["key"], "v": self._var["v"],
            "sigma": round(self._sigma, 4), "match": round(self._match_dist, 5),
            "loss": float(loss.detach()), **info,
        }  # fmt: skip
        if self._calib and self._weight <= 0.0:
            # Finished in after_backward: g_fm from the .grad delta of the
            # real backward (FM only at weight 0), g_al from a fresh forward.
            # Two backwards through one compiled graph give NaN.
            self._snap = [
                None if p.grad is None else p.grad.detach().clone()
                for p in self._params
            ]
            self._pending = {
                "rec": rec, "call": primary.anima_call,
                "inputs": (primary.noisy_model_input.detach(), primary.timesteps.detach(),
                           primary.crossattn_emb.detach()),
                "pm": primary.padding_mask, "kw": dict(primary.forward_kwargs),
            }  # fmt: skip
        else:
            self._write(rec)
        self._metrics.update(
            {
                "xattn_align/active": 1.0,
                "xattn_align/loss": rec["loss"],
                "xattn_align/lift": info["lift"],
                "xattn_align/n_term": float(self._n_term),
            }
        )
        if self._weight <= 0.0:
            return None
        return {"xattn_align": loss}

    def _write(self, rec: dict) -> None:
        if self._log_path:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def after_backward(self, ctx: StepCtx) -> None:
        pend = getattr(self, "_pending", None)
        if pend is None:
            return
        self._pending = None
        accum = int(getattr(ctx.args, "gradient_accumulation_steps", 1) or 1)
        sq = 0.0
        for p, s in zip(self._params, self._snap):
            if p.grad is None:
                continue
            d = p.grad.float() if s is None else p.grad.float() - s.float()
            sq += float(d.pow(2).sum())
        self._snap = None
        g_fm = accum * math.sqrt(sq)  # accelerator.backward divides by accum
        x, t, c = pend["inputs"]
        with torch.enable_grad(), ctx.accelerator.autocast():
            pend["call"](
                x, t, c, padding_mask=pend["pm"], **pend["kw"],
                return_block_features={max(self._blocks)}, return_features_early=True,
            )  # fmt: skip
            if not self._live or any(b not in self._maps for b in self._blocks):
                logger.warning("xattn_align calib: the re-forward did not capture")
                self._live, self._maps = False, {}
                return
            self._live = False
            loss, _ = self._loss()
        self._maps = {}
        gs = torch.autograd.grad(loss, self._params, allow_unused=True)
        g_al = math.sqrt(
            sum(float(g.float().pow(2).sum()) for g in gs if g is not None)
        )
        del gs, loss
        rec = pend["rec"] | {
            "g_fm": g_fm, "g_al": g_al, "ratio": g_al / max(g_fm, 1e-20),
        }  # fmt: skip
        self._write(rec)

    def on_epoch_end(self, ctx: StepCtx) -> None:
        logger.info(
            "xattn_align: %d term steps so far over %d train micro-steps; skips %s; "
            "items with strings %d / %d seen",
            self._n_term,
            self._micro,
            self._skips,
            sum(1 for v in self._items.values() if v),
            len(self._items),
        )

    def metrics(self, ctx) -> dict[str, float]:
        del ctx
        return dict(self._metrics)
