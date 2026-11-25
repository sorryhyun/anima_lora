#!/usr/bin/env python
"""G-D — the OCR reader's loss carried back to the rows (``plan.md`` item 1).

The path: reader → crop pixels → pixel head → DiT step(s) → llm_adapter → rows. Two
objects hold it.

:class:`Policy` is ``m15v11`` (4 Euler steps, cfg 1, pixel-head decode) with jp_v1's
rows as the one trainable tensor (``Policy.raw``, effective units as the ruler's
``ExtDelta`` holds them). A render is sampled with no graph (:meth:`Policy.rollout`
keeps every step's input); the gradient then goes back one step at a time
(:meth:`Policy.vjp`): the step is run again with a graph on its saved input and the
cotangent of its output is pulled to its input and to the text embedding. The last
step's output is the pixel head's image (the head reads that step's feature and x_t;
the step's velocity is not used), an earlier step's is the next x_t. One step's graph
is alive at a time, so the four-step backward peaks where the one-step backward does.
The embedding's cotangent is per render (the DiT treats a batch's renders apart), and
:meth:`Policy.row_grads` pulls it through the LLM adapter to the rows.

:class:`Reader` is the detector and the tuned reader with tensors in:

- :meth:`Reader.read` — the ordinary read (``common.readers.Readers``) of pages held
  as a tensor; the crop of a render is picked from it (:func:`pick`).
- :meth:`Reader.ce` — the reader's teacher-forced cross-entropy of a string on a crop
  cut from the page tensor. Ids, mask and labels come from the reader's own processor
  on the same crop as an image (the training collate of
  ``project/ocr_reader/ocr/finetune_vl16_lora.py``: prompt + text + EOS, loss on the
  text and the EOS); only ``pixel_values`` is replaced by the tensor's.
- :meth:`Reader.det_scores` — AnimeText's per-anchor text score on the page tensor.

The pixel head's image is clamped to [-1, 1] for the reads; the gradient passes the
clamp as identity (:func:`ste_clamp`), since a white bubble sits on the clamp's edge.

Probe: ``probes/gd_g0d.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

LINE = Path(__file__).resolve().parents[1]  # project/cjk_rl
sys.path.insert(0, str(LINE / "probes"))

import turbo_read as T  # noqa: E402  (the pack, routing, the reseed path)

import reward  # noqa: E402

CONF_MIN = 0.6  # a box read under it counts as nothing drawn (the user, 11-25)
CROP_PAD = 0.12  # common.readers.crop
ROWS = T.ROWS[0]  # seed_1008 (jp_v1)


def ste_clamp(x, lo: float, hi: float):
    """``x`` clamped, with the gradient of ``x``."""
    return x + (x.clamp(lo, hi) - x).detach()


def crop_box(box, W: int, H: int) -> tuple:
    """``common.readers.crop``'s rectangle: the box padded 12 % a side, clipped."""
    x0, y0, x1, y1 = (int(v) for v in box)
    pw, ph = int(CROP_PAD * (x1 - x0)), int(CROP_PAD * (y1 - y0))
    return max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph)


def pick(key: str, reads: list, conf_min: float = CONF_MIN) -> tuple:
    """``(case, read)`` for a page's reads and the asked string.

    ``box``: a box read at ``conf_min`` or over; among those, the best sequence score
    to ``key``, then the highest confidence. ``low``: boxes, none at ``conf_min``; the
    best of them by the same order. ``none``: no box (``read`` is ``None``)."""
    from common.text import norm

    k = norm(key)
    bs = [r for r in reads if not r.get("whole")]
    if not bs:
        return "none", None

    def conf(r):
        return r["sfx_conf"] or 0.0

    def seq(r):
        return reward.seq_score(k, [norm(r["sfx"])]) if r["sfx"] else 0.0

    hi = [r for r in bs if r["sfx"] and conf(r) >= conf_min]
    best = max(hi or bs, key=lambda r: (seq(r), conf(r)))
    return ("box" if hi else "low"), best


def area_matrix(n_in: int, n_out: int, device):
    """``(n_out, n_in)``: the overlap of each output cell with each input pixel over
    the cell's width — cv2's ``INTER_AREA`` on one axis when shrinking."""
    import torch

    s = n_in / n_out
    i = torch.arange(n_out, dtype=torch.float64)[:, None]
    j = torch.arange(n_in, dtype=torch.float64)[None]
    w = (torch.minimum((i + 1) * s, j + 1) - torch.maximum(i * s, j)).clamp_min(0) / s
    return w.float().to(device)


class Policy:
    """``m15v11`` with the rows trainable; everything else frozen."""

    def __init__(self, shapes, rows: str = ROWS, grad_bs: int = T.BATCH):
        import gc

        import torch

        from common import models
        from common.models import load_trained
        from eval.ruler.arms import tables
        from eval.ruler.render import Renderer
        from library.anima.pixel_head import PixelHeadX2
        from library.anima.stream import latent_dtype
        from library.inference.pixel_decoder import ensure_pixel_decoder
        from library.inference.sampling import get_timesteps_sigmas
        from reseed import OUT as RESEED_OUT

        models.TOKEN_COUNTS[:] = [(h // 16) * (w // 16) for w, h in shapes]
        _, tabs = tables([rows])
        r = Renderer(T.MODEL)
        r.args.decoder = "pixel_head"
        r.vae = None
        gc.collect()
        torch.cuda.empty_cache()
        assert r.args.guidance_scale == 1.0, "one forward a step"
        assert r.args.sampler == "euler", r.args.sampler
        self.torch = torch
        self.r, self.anima, self.dev = r, r.anima, r.device
        self.grad_bs = grad_bs
        r.set_arm(tabs[rows])
        self.raw = r.delta.raw  # (ext ids, dim), effective units
        self.raw.requires_grad_(True)
        self.row_scale = float(load_trained(RESEED_OUT / rows)["delta"]["row_scale"])
        self.anima.reset_mod_guidance()
        self.anima.reset_dave()
        dec = ensure_pixel_decoder(r.args, self.anima, self.dev)
        self.head = dec.head.requires_grad_(False)
        if isinstance(self.head, PixelHeadX2):
            self._head = lambda f, lat: self.head(f, lat, scale=dec.scale)
        else:
            self._head = self.head
        self.feat = None  # the last forward's final_layer.linear input, graph kept
        self.anima.final_layer.linear.register_forward_pre_hook(self._tap)
        self.steps = r.args.infer_steps
        ts, self.sig = get_timesteps_sigmas(self.steps, r.args.flow_shift, self.dev)
        self.ts = ts.to(self.dev, dtype=latent_dtype(self.anima))
        self.hid: dict = {}

    def _tap(self, _m, inp):
        self.feat = inp[0]

    # ---- text

    def encode(self, captions) -> None:
        """The text encoder's output for every caption (it does not read the rows),
        as ``turbo_read.contexts`` encodes them; the encoder then leaves the GPU, so
        this is called once."""
        torch = self.torch
        from library.anima.vocab_pack import resolve_active_pack
        from library.inference.text import ensure_text_strategies, process_escape

        te = self.r.shared.pop("text_encoder")
        tok, enc = ensure_text_strategies(
            self.r.args.text_encoder, vocab_pack=resolve_active_pack(self.r.args) or ""
        )
        caps = sorted(set(captions))
        with torch.no_grad():
            for i in range(0, len(caps), 16):
                chunk = caps[i : i + 16]
                out = enc.encode_tokens(
                    tok, [te], tok.tokenize([process_escape(c) for c in chunk])
                )
                for k, c in enumerate(chunk):
                    self.hid[c] = [o[k].cpu() for o in out]
        te.to("cpu")
        self.r.shared["conds_cache"].clear()
        del te
        torch.cuda.empty_cache()

    def embed(self, caption: str, grad: bool = False):
        """``(1, 512, D)``: the LLM adapter over the caption with the rows in place,
        max-padded with zeros. With ``grad`` its graph reaches ``raw``."""
        torch = self.torch
        from library.inference.text import MAX_CROSSATTN_TOKENS

        pe, am, t5, t5m = (h[None].to(self.dev) for h in self.hid[caption])
        with torch.set_grad_enabled(grad):
            x, _ = self.anima._preprocess_text_embeds(
                source_hidden_states=pe,
                target_input_ids=t5,
                target_attention_mask=t5m,
                source_attention_mask=am,
            )
            x = x * t5m.bool()[..., None]
            if x.shape[1] < MAX_CROSSATTN_TOKENS:
                x = torch.nn.functional.pad(
                    x, (0, 0, 0, MAX_CROSSATTN_TOKENS - x.shape[1])
                )
        return x

    def row_of(self, caption: str) -> int:
        """The index in ``raw`` of the caption's one ext row."""
        from library.anima.ext_vocab import T5_TABLE_SIZE

        ext = [
            int(v) - T5_TABLE_SIZE
            for v in self.hid[caption][2].tolist()
            if v >= T5_TABLE_SIZE
        ]
        assert len(ext) == 1, (caption[-24:], ext)
        return self.r.delta.index[ext[0]]

    def row_grads(self, e_graph, G):
        """``G`` ``(B, 512, D)``, a cotangent of the embedding per render →
        ``(B, rows, dim)`` on ``raw`` (``e_graph``: :meth:`embed` with ``grad``)."""
        torch = self.torch
        out = [
            torch.autograd.grad(
                e_graph, self.raw, G[b : b + 1].to(e_graph.dtype), retain_graph=True
            )[0]
            for b in range(G.shape[0])
        ]
        return torch.stack(out).float()

    # ---- sampling

    def noise(self, seeds, shape):
        """Each seed's start as the one-image path draws it (a CPU generator)."""
        torch = self.torch
        from library.anima.stream import latent_dtype
        from library.inference.generation import randn_tensor

        W, H = shape
        ch = type(self.anima).LATENT_CHANNELS
        out = []
        for s in seeds:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(s)
            out.append(
                randn_tensor(
                    (1, ch, 1, H // 8, W // 8),
                    generator=gen,
                    device=self.dev,
                    dtype=latent_dtype(self.anima),
                )
            )
        return torch.cat(out)

    def forward(self, i: int, x, e):
        """Step ``i``'s DiT forward as ``generate_body`` calls it → the velocity; the
        feature for the pixel head lands in ``self.feat``."""
        torch = self.torch
        from library.inference.adapters import (
            compute_and_set_hydra_fei,
            set_hydra_crossattn,
            set_hydra_sigma,
            set_step_expert_index,
        )

        b = x.shape[0]
        t = self.ts[i].expand(b)
        e = e.expand(b, -1, -1).to(torch.bfloat16)
        set_hydra_sigma(self.anima, t)
        set_step_expert_index(self.anima, i)
        compute_and_set_hydra_fei(self.anima, x)
        set_hydra_crossattn(self.anima, e)
        pad = torch.zeros(b, 1, *x.shape[-2:], dtype=torch.bfloat16, device=self.dev)
        return self.anima(x, t, e, padding_mask=pad)

    def euler(self, i: int, x, v):
        return (x.float() - float(self.sig[i] - self.sig[i + 1]) * v.float()).to(
            x.dtype
        )

    def decode(self, x):
        """The pixel head on the last forward's feature and its x_t →
        ``(B, 3, H, W)`` fp32, not clamped (``PixelTap.decode`` without the clamp)."""
        torch = self.torch
        from library.anima.pixel_head import lat_patches

        f = self.feat.float()
        if f.ndim == 5:
            f = f[:, 0]
        lat = x.float()
        if lat.ndim == 5:
            lat = lat[:, :, 0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = [
                self._head(f[b], lat_patches(lat[b])).float() for b in range(f.shape[0])
            ]
        return torch.stack(out)

    def rollout(self, x0, e):
        """No graph: ``(xs, px)``, the input of every step and the pixel head's image
        of the last."""
        torch = self.torch
        with torch.no_grad():
            xs, x = [], x0
            for i in range(self.steps):
                xs.append(x)
                v = self.forward(i, x, e)
                if i < self.steps - 1:
                    x = self.euler(i, x, v)
            return xs, self.decode(x)

    def head_steps(self, x0, e):
        """No graph: every step but the last → the last step's input."""
        torch = self.torch
        with torch.no_grad():
            x = x0
            for i in range(self.steps - 1):
                x = self.euler(i, x, self.forward(i, x, e))
            return x

    def last(self, x, e):
        """The last step with a graph: ``(px, e_leaf)``, the pixel head's image (not
        clamped) and the embedding leaf ``(B, 512, D)`` its graph ends in. The step
        runs once: this forward is the render's own last step. ``x`` carries no
        gradient (a one-step backward treats the last step's input as given)."""
        torch = self.torch
        x = x.detach()
        ei = e.detach().float().expand(x.shape[0], -1, -1).clone().requires_grad_(True)
        with torch.enable_grad():
            self.forward(self.steps - 1, x, ei)
            px = self.decode(x)
        self.feat = None
        return px, ei

    def vjp(self, i: int, x, e, cot):
        """Step ``i`` again with a graph, ``grad_bs`` renders a forward. ``cot``: the
        cotangent of the step's output, the image for the last step and x_{i+1} for
        the others. → the cotangents of ``e`` ``(B, 512, D)`` and of ``x``. An
        out-of-memory forward halves ``grad_bs`` and goes again."""
        torch = self.torch
        last = i == self.steps - 1
        B = x.shape[0]
        e = e.detach().float().expand(B, -1, -1)
        while True:
            ges, gxs = [], []
            try:
                for s in range(0, B, self.grad_bs):
                    sl = slice(s, s + self.grad_bs)
                    xi = x[sl].detach().requires_grad_(True)
                    ei = e[sl].clone().requires_grad_(True)
                    with torch.enable_grad():
                        v = self.forward(i, xi, ei)
                        out = self.decode(xi) if last else self.euler(i, xi, v)
                        ge, gx = torch.autograd.grad((out * cot[sl]).sum(), (ei, xi))
                    ges.append(ge)
                    gxs.append(gx)
                    del v, out, xi, ei
                return torch.cat(ges), torch.cat(gxs)
            except torch.OutOfMemoryError:
                if self.grad_bs == 1:
                    raise
            # outside the handler, so the failed forward's frames are released
            self.feat = None
            self.grad_bs //= 2
            torch.cuda.empty_cache()
            print(f"  out of memory: grad_bs → {self.grad_bs}", flush=True)


class Reader:
    """The detector and the tuned reader, with tensors in."""

    def __init__(self, device: str = "cuda"):
        from common.readers import Readers

        self.rd = Readers(device)
        self.det, self.ocr = self.rd.det, self.rd.sfx
        self.ocr.model.requires_grad_(False)
        self.det.model.requires_grad_(False)
        self.device = device

    def read(self, px) -> tuple[list, list]:
        """The ordinary read of pages ``(B, 3, H, W)`` in [-1, 1]: the BGR arrays the
        readers saw and their records."""
        import numpy as np

        from library.inference.output import pixels_to_pil

        bgrs = [
            np.ascontiguousarray(np.asarray(pixels_to_pil(p))[:, :, ::-1])
            for p in px.detach()
        ]
        return bgrs, self.rd.read_images(bgrs)

    def ce(self, pages, bgrs, boxes, texts, check: dict | None = None):
        """The teacher-forced cross-entropy of ``texts[k]`` on crop ``boxes[k]`` of
        ``pages[k]`` → ``(n,)`` fp32, a mean over the text's tokens and the EOS.

        ``pages``: ``(3, H, W)`` RGB in [0, 1], graph kept; ``bgrs``: the same pages
        as the arrays the read saw; ``boxes``: :func:`crop_box` rectangles. ``check``
        takes the tensor path's distance from the processor's ``pixel_values``."""
        import numpy as np
        import torch
        import torch.nn.functional as F
        from anime_tools.ocr.reader import MAX_PIXELS
        from PIL import Image
        from transformers.models.paddleocr_vl.image_processing_paddleocr_vl import (
            smart_resize,
        )

        ocr = self.ocr
        proc, tok = ocr.processor, ocr.processor.tokenizer
        ip = proc.image_processor
        images = [
            Image.fromarray(np.ascontiguousarray(b[y0:y1, x0:x1, ::-1]))
            for b, (x0, y0, x1, y1) in zip(bgrs, boxes)
        ]
        enc = proc(
            text=[ocr.prompt + t + tok.eos_token for t in texts],
            images=images,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            images_kwargs={
                "size": {"shortest_edge": ocr.min_edge, "longest_edge": MAX_PIXELS}
            },
        )
        labels = torch.full_like(enc["input_ids"], -100)
        for r, t in enumerate(texts):
            tid = tok(t + tok.eos_token, add_special_tokens=False).input_ids
            assert enc["input_ids"][r, -len(tid) :].tolist() == tid, (t, "boundary")
            labels[r, -len(tid) :] = torch.tensor(tid)
        dev = pages[0].device
        mean = torch.tensor(ip.image_mean, device=dev).view(1, 3, 1, 1)
        std = torch.tensor(ip.image_std, device=dev).view(1, 3, 1, 1)
        p = ip.patch_size
        pv = []
        for page, (x0, y0, x1, y1) in zip(pages, boxes):
            c = page[:, y0:y1, x0:x1][None].float()
            H2, W2 = smart_resize(
                y1 - y0,
                x1 - x0,
                factor=p * ip.merge_size,
                min_pixels=ocr.min_edge,
                max_pixels=MAX_PIXELS,
            )
            y = F.interpolate(
                c, size=(H2, W2), mode="bicubic", antialias=True, align_corners=False
            )
            y = (ste_clamp(y, 0.0, 1.0) - mean) / std
            gh, gw = H2 // p, W2 // p
            pv.append(
                y.reshape(1, 3, gh, p, gw, p)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(gh * gw, 3, p, p)
            )
        pv = torch.cat(pv)
        ref = enc["pixel_values"]
        assert pv.shape == ref.shape, (pv.shape, ref.shape)
        if check is not None:
            d = (pv.detach().cpu() - ref.float()).abs()
            check["pv_max"] = max(check.get("pv_max", 0.0), float(d.max()))
            check["pv_mean"] = max(check.get("pv_mean", 0.0), float(d.mean()))
        enc = enc.to(dev)
        labels = labels.to(dev)
        K = int((labels != -100).sum(1).max()) + 1
        feed = {k: v for k, v in enc.items() if k != "pixel_values"}
        logits = ocr.model(**feed, pixel_values=pv, logits_to_keep=K).logits[:, :-1]
        tgt = labels[:, -(K - 1) :]
        ce = F.cross_entropy(
            logits.float().transpose(1, 2), tgt, ignore_index=-100, reduction="none"
        )
        loss = ce.sum(1) / (tgt != -100).sum(1)
        if check is not None:  # the same loss on the processor's own pixel values
            with torch.no_grad():
                lg = ocr.model(**enc, logits_to_keep=K).logits[:, :-1]
                c2 = F.cross_entropy(
                    lg.float().transpose(1, 2), tgt, ignore_index=-100, reduction="none"
                )
                d = (c2.sum(1) / (tgt != -100).sum(1) - loss.detach()).abs()
            check["ce_max"] = max(check.get("ce_max", 0.0), float(d.max()))
        return loss

    def det_scores(self, pages, check_bgrs=None, check: dict | None = None):
        """AnimeText's text score per anchor ``(B, N)`` on pages ``(B, 3, H, W)`` RGB
        in [0, 1], graph kept: the detector's letterbox (area resize onto the top
        left of an ``imgsz`` square, grey elsewhere) done on the tensor."""
        import torch
        import torch.nn.functional as F
        from anime_tools.ocr.animetext import PAD_VALUE

        det = self.det
        B, _, H, W = pages.shape
        s = det.imgsz
        r = min(s / H, s / W)
        assert r <= 1.0, "the area resize is the shrinking one"
        nh, nw = max(1, round(H * r)), max(1, round(W * r))
        x = pages.float()
        if r != 1.0:
            x = torch.einsum(
                "ih,bchw,jw->bcij",
                area_matrix(H, nh, x.device),
                x,
                area_matrix(W, nw, x.device),
            )
        x = F.pad(x, (0, s - nw, 0, s - nh), value=PAD_VALUE / 255)
        w = next(det.model.parameters())
        scores = det.model(x.to(w.dtype))[:, 4].float()
        if check is not None:
            raws = det.forward_batch([det.prepare(b) for b in check_bgrs])
            ref = torch.tensor([float(h[4].max()) for h in raws])
            d = (scores.detach().max(1).values.cpu() - ref).abs()
            check["det_top_max"] = max(check.get("det_top_max", 0.0), float(d.max()))
        return scores
