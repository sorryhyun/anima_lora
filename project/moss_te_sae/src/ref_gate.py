"""lora_blocks P0: the reference gate on trained edit LoRAs, inference only.

With ``ref_isolate`` the reference reaches the main rows only through main-query →
reference-key attention. ``RefGate`` adds a bias matrix ``(n_img, n_ref)`` on those
edges in every block, head and step, both CFG branches. Modes for the image queries
in a cell mask:

- ``hard``: −∞ on every reference key;
- ``diag<k>``: −∞ only on the reference keys within ``k`` cells (Chebyshev) of the
  query's own cell, so the masked queries still read the rest of the reference;
- ``soft<v>``: −v on every reference key;
- ``boost<v>``: +v for the queries *outside* the mask, nothing inside;
- ``inboost<v>``: +v for the queries *inside* the mask, nothing outside;
- ``cut``: the queries in the mask lose the reference keys in the mask and read the
  rest (``model._attention``'s ``ref_gate``, the gate ``TrainRequest.ref_gate`` trains
  the lora_blocks C block under). Needs the reference grid = the image grid.

``ctrl`` is the stock path. Kernels (``--kernel``, see ``RefGate``): ``fast`` (FA2 halves
merged by LSE; flex attention for ``diag``) or ``dense`` (SDPA with a dense mask, every
arm; the edit and sanitize runs of 2026-11-07 17:32-17:51 used it).

Parts (``--part``):

- ``edit``: chinoll/Moss-Image-Edit on ``ref_edge_mask.py``'s two box cases, ``diag``
  at k = 0 / 1 / 2, seed 0. The ``lora`` / ``hard`` / ``soft`` arms of
  ``output/tests/ref_edge_mask/20261107-1549`` are the comparison and go into the sheet.
- ``sanitize``: sanitize v3 on the five eval pages, seeds 0-2, mask from AnimeText
  (conf 0.05, 1024 canvas, boxes padded 8 px, any overlap marks a cell).
- ``swap_text``: swap_text ksplit e2 on the four eval pages, seeds 0-2, mask from
  the prompt's layout boxes (``copy_head.text_mask``).
- ``colorize``: colorize v5 on Korean pages of ComfyUI's ``input/to_colorize4``,
  seeds 0-2. The prompt is v5's bare control caption (``comic`` and a box-only
  ``text`` item per OCR line ``keep_line`` keeps, as the cache builds it); the mask
  is those boxes padded 4 px. ``inboost`` asks whether a stronger copy inside the
  boxes keeps the source's strokes instead of redrawing them.
- ``fill``: the lora_blocks C block (``--lora``, default ``c_fill_s400``) on the
  sanitize pages and masks, on the empty prompt it trained on, ``ctrl`` (no gate)
  against ``cut``.

Reads per render, against the reference: outside the mask (one cell ring left out)
signed ΔL and Δb in Lab, luma L1, and luma L1 against the same seed's ``ctrl``;
inside, luma L1. Sheets per input: rows = seeds, columns = ref (mask outlined) then
the arms.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import copy_head_ext as ext  # noqa: E402
import library.moss.generate as gen  # noqa: E402
import library.moss.model as mm  # noqa: E402
from copy_head import EVAL, log, text_mask  # noqa: E402
from copy_vs_step import SAN, SWP  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.moss.requests import GenerateRequest  # noqa: E402
from ref_edge_mask import CASES as EDIT_CASES  # noqa: E402

EDIT_PRIOR = "output/tests/ref_edge_mask/20261107-1549"
SANITIZE_PAGES = [
    "usual_case1",
    "usual_case2",
    "hard_to_sanitize1",
    "hard_to_sanitize2",
    "hard_to_sanitize3",
]
SWAP_PAGES = ["1483610_0007_en", "3124789_0042_en", "589883_0004_ko", "737486_0019_ko"]
PARTS = {
    "edit": dict(arms="diag0,diag1,diag2", seeds="0"),
    "sanitize": dict(arms="ctrl,hard,diag1", seeds="0,1,2"),
    "swap_text": dict(arms="ctrl,diag1,boost2", seeds="0,1,2"),
    "colorize": dict(arms="ctrl,inboost1,inboost2", seeds="0,1,2"),
    "fill": dict(arms="ctrl,cut", seeds="0,1,2"),
}
FILL_LORA = "output/ckpt_moss_i2i/c_fill_s400"
# A prompts-file line stands for the empty prompt (empty lines are dropped).
EMPTY = "<empty>"
COLORIZE_DIR = "../comfy/input/to_colorize4"
COLORIZE_PAGES = ["03", "05", "09", "14"]
COLORIZE_LORA = "output/ckpt_moss_i2i/moss_colorize_v5"


def gate_matrix(cells: torch.Tensor, arm: str, n_ref: int) -> torch.Tensor:
    """``(n_img, n_ref)`` bias for ``arm`` over the ``(gh, gw)`` bool ``cells``."""
    gh, gw = cells.shape
    rh = round(math.sqrt(n_ref * gh / gw))
    rw = n_ref // rh
    if rh * rw != n_ref:
        raise ValueError(f"reference grid for {n_ref} tokens at {gh}x{gw}")
    q = cells.flatten()
    bias = torch.zeros(gh * gw, n_ref)
    if arm == "ctrl":
        return bias
    if arm == "hard":
        bias[q] = float("-inf")
    elif arm.startswith("soft"):
        bias[q] = -float(arm[4:])
    elif arm.startswith("boost"):
        bias[~q] = float(arm[5:])
    elif arm.startswith("inboost"):
        bias[q] = float(arm[7:])
    elif arm == "cut":
        if n_ref != q.numel():
            raise ValueError("cut needs the reference grid = the image grid")
        bias[q[:, None] & q[None, :]] = float("-inf")
    elif arm.startswith("diag"):
        k = int(arm[4:])
        qy, qx = torch.meshgrid(
            torch.arange(gh).float(), torch.arange(gw).float(), indexing="ij"
        )
        ry, rx = torch.meshgrid(
            (torch.arange(rh) + 0.5) * gh / rh - 0.5,
            (torch.arange(rw) + 0.5) * gw / rw - 0.5,
            indexing="ij",
        )
        near = ((qy.flatten()[:, None] - ry.flatten()[None]).abs() <= k + 0.5) & (
            (qx.flatten()[:, None] - rx.flatten()[None]).abs() <= k + 0.5
        )
        bias[q[:, None] & near] = float("-inf")
    else:
        raise ValueError(arm)
    return bias


def query_bias(cells: torch.Tensor, arm: str) -> torch.Tensor:
    """``(n_img,)`` bias on every reference key per image query, for the arms
    whose bias is constant over the reference (``hard`` / ``soft`` / ``boost`` /
    ``inboost``)."""
    q = cells.flatten()
    b = torch.zeros(q.numel())
    if arm == "hard":
        b[q] = float("-inf")
    elif arm.startswith("soft"):
        b[q] = -float(arm[4:])
    elif arm.startswith("boost"):
        b[~q] = float(arm[5:])
    elif arm.startswith("inboost"):
        b[q] = float(arm[7:])
    else:
        raise ValueError(arm)
    return b


def merge_lse(o_m, l_m, o_r, l_r, b):
    """Softmax over main ∪ reference keys from the two halves' outputs and LSEs
    (``(B, L, H, D)``, ``(B, H, L)``), the reference half shifted by ``b`` ``(L,)``."""
    l_r = l_r + b
    m = torch.maximum(l_m, l_r)
    w_m = (l_m - m).exp().transpose(1, 2).unsqueeze(-1)
    w_r = (l_r - m).exp().transpose(1, 2).unsqueeze(-1)
    out = (w_m * o_m.float() + w_r * o_r.float()) / (w_m + w_r)
    return out.to(o_m.dtype)


_flex = None


def flex():
    global _flex
    if _flex is None:
        from torch.nn.attention.flex_attention import flex_attention

        _flex = torch.compile(flex_attention, dynamic=False)
    return _flex


def diag_score_mod(cells: torch.Tensor, k: int):
    """Keys ordered reference first; an image query in ``cells`` loses the
    reference keys within ``k`` cells of its own (reference grid = image grid)."""
    gw = cells.shape[1]
    flat = cells.flatten()
    n = flat.numel()

    def score_mod(score, b, h, qi, ki):
        qc = torch.clamp(qi, max=n - 1)
        rc = torch.clamp(ki, max=n - 1)
        cut = (
            (ki < n)
            & (qi < n)
            & flat[qc]
            & ((qc // gw - rc // gw).abs() <= k)
            & ((qc % gw - rc % gw).abs() <= k)
        )
        return torch.where(cut, float("-inf"), score)

    return score_mod


class RefGate:
    """Patches ``mm._attention`` for the main rows' call against the cached
    reference keys.

    ``kernel = "fast"``: ``ctrl`` leaves the stock path (FA2); ``hard`` / ``soft`` /
    ``boost`` run FA2 over the main keys and over the reference keys and merge the
    two by LSE with the per-query bias; ``diag`` runs flex attention with the cut in
    a score_mod, compiled at static shapes outside the block graph (the blocks'
    dynamic compile cannot lower flex); the FA2 arms stay inside the block graph. ``kernel = "dense"``: every
    arm, ``ctrl`` included, through SDPA with a dense ``(L, L + n_ref)`` mask, out
    of the graph (the runs before 2026-11-07 17:50)."""

    def __init__(self, cells, arm, kernel="fast"):
        self.cells, self.arm, self.kernel = cells, arm, kernel
        self.orig = mm._attention
        self.bias = None

    def __enter__(self):
        if self.kernel == "dense":
            return self._enter_dense()
        if self.arm == "ctrl":
            return self
        if self.arm == "cut":
            return self._enter_cut()
        orig, cells = self.orig, self.cells
        n_img = cells.numel()
        diag = self.arm.startswith("diag")
        if diag:
            score_mod = diag_score_mod(cells.cuda(), int(self.arm[4:]))

            # flex under the blocks' dynamic=True compile fails to lower
            # (torch 2.14: FloatTrueDiv) and falls back to the dense math path;
            # it runs outside the block graph at static shapes instead.
            @torch.compiler.disable
            def diag_call(q_, k_, v_):
                scale = 1.0 / math.sqrt(q_.shape[-1])
                return flex()(q_, k_, v_, score_mod=score_mod, scale=scale)
        else:
            b_img = query_bias(cells, self.arm).cuda()
        dense = self._dense_attention()

        def attention(q, k, v, rope, heads, key_mask=None, ref_start=None,
                      ref_isolate=False, ref_kv=None):  # fmt: skip
            if ref_kv is None or ref_start is not None:
                return orig(q, k, v, rope, heads, key_mask, ref_start, ref_isolate, ref_kv)  # fmt: skip
            if key_mask is not None:
                return dense(q, k, v, rope, heads, key_mask, ref_kv)
            shape = (*q.shape[:2], heads, q.shape[-1] // heads)
            qr = mm.rotate(q.reshape(shape), rope).to(v.dtype)
            kr = mm.rotate(k.reshape(shape), rope).to(v.dtype)
            vr = v.reshape(shape)
            ref_k, ref_v, ref_rope = ref_kv
            ref_shape = (*ref_k.shape[:2], heads, ref_k.shape[-1] // heads)
            ref_k = mm.rotate(ref_k.reshape(ref_shape), ref_rope).to(v.dtype)
            ref_v = ref_v.reshape(ref_shape)
            L = q.shape[1]
            if diag:
                kk = torch.cat((ref_k, kr), 1).transpose(1, 2)
                vv = torch.cat((ref_v, vr), 1).transpose(1, 2)
                return diag_call(qr.transpose(1, 2), kk, vv).transpose(1, 2).flatten(2)
            o_m, l_m, _ = mm.flash_attn_func(qr, kr, vr, return_attn_probs=True)
            o_r, l_r, _ = mm.flash_attn_func(qr, ref_k, ref_v, return_attn_probs=True)
            b = torch.nn.functional.pad(b_img, (0, L - n_img))
            return merge_lse(o_m, l_m, o_r, l_r, b).flatten(2)

        mm._attention = attention
        return self

    def _dense_attention(self):
        gate = self

        def attention(q, k, v, rope, heads, key_mask, ref_kv):
            shape = (*q.shape[:2], heads, q.shape[-1] // heads)
            qr = mm.rotate(q.reshape(shape), rope).to(v.dtype)
            kr = mm.rotate(k.reshape(shape), rope).to(v.dtype)
            vr = v.reshape(shape)
            ref_k, ref_v, ref_rope = ref_kv
            ref_shape = (*ref_k.shape[:2], heads, ref_k.shape[-1] // heads)
            ref_k = mm.rotate(ref_k.reshape(ref_shape), ref_rope).to(v.dtype)
            kr = torch.cat((kr, ref_k), 1)
            vr = torch.cat((vr, ref_v.reshape(ref_shape)), 1)
            B, L, n_ref = q.shape[0], q.shape[1], ref_k.shape[1]
            if gate.bias is None or gate.bias.shape[1] != n_ref:
                gate.bias = gate_matrix(gate.cells, gate.arm, n_ref).to(q.device)
            n_img = gate.bias.shape[0]
            m = torch.zeros(B, 1, L, L + n_ref, device=q.device, dtype=v.dtype)
            if key_mask is not None:
                km = torch.cat((key_mask, key_mask.new_ones(B, n_ref)), 1)
                m.masked_fill_(~km[:, None, None, :], float("-inf"))
            m[:, 0, :n_img, L:] += gate.bias.to(v.dtype)
            return mm._sdpa(qr, kr, vr, m).flatten(2)

        return torch.compiler.disable(attention)

    def _enter_dense(self):
        orig, dense = self.orig, self._dense_attention()

        def attention(q, k, v, rope, heads, key_mask=None, ref_start=None,
                      ref_isolate=False, ref_kv=None):  # fmt: skip
            if ref_kv is None or ref_start is not None:
                return orig(q, k, v, rope, heads, key_mask, ref_start, ref_isolate, ref_kv)  # fmt: skip
            return dense(q, k, v, rope, heads, key_mask, ref_kv)

        mm._attention = attention
        return self

    def _enter_cut(self):
        orig, flat = self.orig, self.cells.flatten()
        gate = (flat.nonzero().flatten().cuda(), (~flat).nonzero().flatten().cuda())

        def attention(q, k, v, rope, heads, key_mask=None, ref_start=None,
                      ref_isolate=False, ref_kv=None):  # fmt: skip
            n_ref = None
            if ref_kv is not None:
                n_ref = ref_kv[0].shape[1]
            elif ref_start is not None:
                n_ref = q.shape[1] - ref_start
            if n_ref is None:
                return orig(q, k, v, rope, heads, key_mask, ref_start, ref_isolate, ref_kv)  # fmt: skip
            if n_ref != flat.numel():
                raise ValueError(f"cut needs the reference grid = the image grid ({n_ref} != {flat.numel()})")  # fmt: skip
            return orig(q, k, v, rope, heads, key_mask, ref_start, ref_isolate, ref_kv, ref_gate=gate)  # fmt: skip

        mm._attention = attention
        return self

    def __exit__(self, *exc):
        mm._attention = self.orig


class SharedTE:
    """One TE cache for every render of a part."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.orig = gen.cached_embeds

    def __enter__(self):
        orig, d = self.orig, self.cache_dir

        def cached(model_dir, texts, _c, *a, **kw):
            real = ["" if t == EMPTY else t for t in texts]
            out = orig(model_dir, real, d, *a, **kw)
            if EMPTY in texts:
                out[EMPTY] = out[""]
            return out

        gen.cached_embeds = cached
        return self

    def __exit__(self, *exc):
        gen.cached_embeds = self.orig


def cells_from_mask(mask: np.ndarray) -> torch.Tensor:
    """Pixel mask ``(H, W)`` → cell mask ``(H/16, W/16)``, any pixel marks a cell."""
    H, W = mask.shape
    return torch.from_numpy(mask.reshape(H // 16, 16, W // 16, 16).any((1, 3)))


def lab(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(
        np.asarray(im.convert("RGB"), np.float32) / 255, cv2.COLOR_RGB2Lab
    )


def luma(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), np.float32)


def grad(y: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude of a luma plane."""
    return np.hypot(cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1))


def reads(image, ref, cells, ctrl=None):
    im = Image.open(image)
    W, H = im.size
    R = Image.open(ref).convert("RGB").resize((W, H), Image.LANCZOS)
    c = cells.numpy()
    ring = cv2.dilate(c.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    inside = np.kron(c, np.ones((16, 16), bool))
    outside = ~np.kron(ring, np.ones((16, 16), bool))
    li, lr = lab(im), lab(R)
    d = li - lr
    gi, gr = grad(luma(im)), grad(luma(R))
    e = dict(
        dL_out=float(d[..., 0][outside].mean()),
        db_out=float(d[..., 2][outside].mean()),
        l1_out=float(np.abs(luma(im) - luma(R))[outside].mean()),
        l1_in=float(np.abs(luma(im) - luma(R))[inside].mean()) if c.any() else None,
        grad_in=float(gi[inside].mean() / gr[inside].mean()) if c.any() else None,
    )
    if ctrl is not None:
        e["l1_out_vs_ctrl"] = float(
            np.abs(luma(im) - luma(Image.open(ctrl)))[outside].mean()
        )
    return {k: (round(v, 3) if v is not None else None) for k, v in e.items()}


def sheet(path, ref, cells, grid, size):
    """``grid``: rows of image paths (None = blank)."""
    W, H = size
    tw = 384
    th = int(H * tw / W)
    R = Image.open(ref).convert("RGB").resize((W, H))
    draw = ImageDraw.Draw(R)
    for y, x in zip(*np.nonzero(cells.numpy())):
        draw.rectangle((x * 16, y * 16, x * 16 + 15, y * 16 + 15), outline=(0, 255, 0))
    cols = 1 + max(len(r) for r in grid)
    out = Image.new("RGB", (tw * cols, th * len(grid)), "white")
    for y, row in enumerate(grid):
        out.paste(R.resize((tw, th)), (0, y * th))
        for x, p in enumerate(row):
            if p:
                out.paste(Image.open(p).convert("RGB").resize((tw, th)), ((x + 1) * tw, y * th))  # fmt: skip
    out.save(path)


def zoom(path, ref, boxes, row_images, size, n=6, scale=2):
    """The ``n`` largest ``boxes`` (output pixels, padded 8 px) at ``scale``x:
    rows = boxes, columns = ref then ``row_images``."""
    W, H = size
    R = Image.open(ref).convert("RGB").resize((W, H), Image.LANCZOS)
    ims = [R] + [Image.open(p).convert("RGB") for p in row_images]
    big = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:n]
    crops = []
    for x0, y0, x1, y1 in big:
        b = (max(0, int(x0) - 8), max(0, int(y0) - 8), min(W, int(x1) + 8), min(H, int(y1) + 8))  # fmt: skip
        crops.append([im.crop(b).resize(((b[2] - b[0]) * scale, (b[3] - b[1]) * scale), Image.LANCZOS) for im in ims])  # fmt: skip
    if not crops:
        return
    cw = max(r[0].width for r in crops)
    out = Image.new("RGB", (cw * len(ims), sum(r[0].height for r in crops)), "white")
    y = 0
    for r in crops:
        for x, c in enumerate(r):
            out.paste(c, (x * cw, y))
        y += r[0].height
    out.save(path)


def output_size(ref: str) -> tuple[int, int]:
    """generate's size for a reference at the default 1024² area."""
    from library.moss.caching.images import target_size

    return target_size(*Image.open(ref).size, 1024)


def sanitize_inputs(pages=SANITIZE_PAGES):
    from anime_tools.ocr.animetext import AnimeTextDetector

    det = AnimeTextDetector.load(device="cpu", conf=0.05, imgsz=1024)
    out = []
    for stem in pages:
        ref = str(resolve_under_home(f"{EVAL}/sanitize/inputs/{stem}.png"))
        bgr = cv2.imread(ref)
        h, w = bgr.shape[:2]
        W, H = output_size(ref)
        mask = np.zeros((H, W), bool)
        boxes = [list(map(int, b[:4])) for b in det.detect(bgr)]
        for x0, y0, x1, y1 in boxes:
            x0, x1 = (max(0, int((x0 - 8) * W / w)), min(W, int((x1 + 8) * W / w) + 1))
            y0, y1 = (max(0, int((y0 - 8) * H / h)), min(H, int((y1 + 8) * H / h) + 1))
            mask[y0:y1, x0:x1] = True
        out.append(dict(key=stem, ref=ref, prompt=None, boxes=boxes, cells=cells_from_mask(mask)))  # fmt: skip
    return out


def swap_inputs(pages=SWAP_PAGES):
    out = []
    for stem in pages:
        ref = str(resolve_under_home(f"{EVAL}/swap_text/inputs/{stem}.src.webp"))
        prompt = resolve_under_home(f"{EVAL}/swap_text/inputs/{stem}.txt").read_text().strip()  # fmt: skip
        W, H = output_size(ref)
        out.append(dict(key=stem, ref=ref, prompt=prompt, cells=cells_from_mask(text_mask(prompt, H, W))))  # fmt: skip
    return out


def colorize_inputs(pages=COLORIZE_PAGES):
    from library.moss.colorize import keep_line, manga_layout, read_pages

    refs = [str(resolve_under_home(f"{COLORIZE_DIR}/{p}.webp")) for p in pages]
    lines = read_pages([Path(r) for r in refs], "cuda", keep=keep_line)
    out = []
    for stem, ref, ls in zip(pages, refs, lines):
        w, h = Image.open(ref).size
        W, H = output_size(ref)
        boxes = [
            [b[0] * W / w, b[1] * H / h, b[2] * W / w, b[3] * H / h] for _, b in ls
        ]
        mask = np.zeros((H, W), bool)
        for x0, y0, x1, y1 in boxes:
            mask[max(0, int(y0) - 4) : min(H, int(y1) + 5), max(0, int(x0) - 4) : min(W, int(x1) + 5)] = True  # fmt: skip
        log(f"colorize {stem}: {len(ls)} lines: {' / '.join(t for t, _ in ls)}")
        prompt = manga_layout([], boxes, (W, H))
        out.append(dict(key=stem, ref=ref, prompt=prompt, boxes=boxes, lines=[t for t, _ in ls], cells=cells_from_mask(mask)))  # fmt: skip
    return out


def run_task(part, out, arms, seeds, pages=None, kernel="fast", lora=None):
    if part == "fill":
        inputs = [dict(i, prompt=EMPTY) for i in sanitize_inputs(pages or SANITIZE_PAGES[:3])]  # fmt: skip
        base = ["--config", "configs/moss/i2i/sanitize/generate.toml", "--lora", f"{lora or FILL_LORA}.safetensors"]  # fmt: skip
    elif part == "colorize":
        inputs = colorize_inputs(pages or COLORIZE_PAGES)
        base = ["--config", "configs/moss/i2i/colorize/generate.toml", "--lora", f"{COLORIZE_LORA}.safetensors"]  # fmt: skip
    elif part == "sanitize":
        inputs = sanitize_inputs(pages or SANITIZE_PAGES)
        base = ["--config", "configs/moss/i2i/sanitize/generate.toml", "--lora", f"{SAN}.safetensors"]  # fmt: skip
    else:
        inputs = swap_inputs(pages or SWAP_PAGES)
        base = ["--config", "configs/moss/i2i/swap_text/generate.toml", "--lora", f"{SWP}.safetensors"]  # fmt: skip
    sp = out / "summary.json"
    summary = json.loads(sp.read_text()) if sp.is_file() else {}
    with SharedTE(out / "_te_cache"):
        for inp in inputs:
            rows = {}
            # One generate call per arm renders every seed (the prompt repeated, seed
            # +1 per line), so the DiT load, int8 conversion and compile run once.
            prompt = inp["prompt"] or GenerateRequest.from_argv(base).prompt
            pfile = out / inp["key"] / "prompts.txt"
            pfile.parent.mkdir(parents=True, exist_ok=True)
            pfile.write_text("\n".join([prompt] * len(seeds)) + "\n")
            for arm in arms:
                argv = base + [
                    "--ref_image", inp["ref"],
                    "--prompts_file", str(pfile),
                    "--seed", str(seeds[0]),
                    "--out_dir", str(out / inp["key"] / arm),
                ]  # fmt: skip
                t0 = time.time()
                with RefGate(inp["cells"], arm, kernel):
                    d = gen.run_generate(GenerateRequest.from_argv(argv))
                recs = json.loads((d / "manifest.json").read_text())["images"]
                gh, gw = inp["cells"].shape
                for rec in recs:
                    seed = rec["seed"]
                    if (rec["width"], rec["height"]) != (gw * 16, gh * 16):
                        raise SystemExit(f"{inp['key']}: size {rec['width']}x{rec['height']} != mask")  # fmt: skip
                    ctrl = rows.get(("ctrl", seed), {}).get("image")
                    e = dict(image=rec["image"], **reads(rec["image"], inp["ref"], inp["cells"], ctrl))  # fmt: skip
                    rows[(arm, seed)] = e
                    log(f"{part} {inp['key']} {arm} s{seed}: {json.dumps({k: v for k, v in e.items() if k != 'image'})}")  # fmt: skip
                log(f"{part} {inp['key']} {arm}: {len(recs)} seeds in {time.time() - t0:.0f}s")  # fmt: skip
            prev = summary.get(inp["key"], {}).get("arms", {})
            summary[inp["key"]] = dict(
                ref=inp["ref"],
                boxes=inp.get("boxes"),
                lines=inp.get("lines"),
                prompt=inp["prompt"],
                cells=int(inp["cells"].sum()),
                arms={**prev, **{f"{a}_s{s}": e for (a, s), e in rows.items()}},
            )
            (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))  # fmt: skip
            grid = [[rows[(a, s)]["image"] for a in arms] for s in seeds]
            W, H = Image.open(grid[0][0]).size
            sheet(
                out / f"sheet_{inp['key']}.png", inp["ref"], inp["cells"], grid, (W, H)
            )
            log(f"{part} {inp['key']}: sheet rows seeds {seeds}, columns ref, {', '.join(arms)}")  # fmt: skip
            if part == "colorize":
                for s, row in zip(seeds, grid):
                    zoom(out / f"zoom_{inp['key']}_s{s}.png", inp["ref"], inp["boxes"], row, (W, H))  # fmt: skip


def run_edit(out, arms, kernel="fast"):
    prior = json.loads(resolve_under_home(f"{EDIT_PRIOR}/summary.json").read_text())
    ext.SAMPLER.extend(["--width", "576", "--height", "1024"])
    summary = {}
    for n, case in enumerate(EDIT_CASES):
        p = prior[f"case{n}"]
        W, H = p["arms"]["lora"]["size"]
        ref = ext.find_image(case["ref"])
        x0, y0, x1, y1 = case["box"]
        mask = np.zeros((H, W), bool)
        mask[y0:y1, x0:x1] = True
        cells = cells_from_mask(mask)
        rows = {a: dict(image=e["image"]) for a, e in p["arms"].items()}
        for arm in arms:
            t0 = time.time()
            with RefGate(cells, arm, kernel):
                e = ext.render(
                    ref, case["prompt"], out / f"case{n}" / arm, mult=1, cut=None,
                    capture=False, frame0=True, te_cache=out / "_te_cache",
                )  # fmt: skip
            rows[arm] = dict(image=e["image"])
            log(f"edit case{n} {arm}: {time.time() - t0:.0f}s")
        for arm, e in rows.items():
            e.update(reads(e["image"], ref, cells, rows["lora"]["image"]))
            log(f"  case{n} {arm:6} {json.dumps({k: v for k, v in e.items() if k != 'image'})}")  # fmt: skip
        summary[f"case{n}"] = dict(ref=ref, prompt=case["prompt"], box=case["box"], arms=rows)  # fmt: skip
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        order = ["lora", "soft2", "soft4", "hard", *arms]
        sheet(out / f"sheet_case{n}.png", ref, cells, [[rows[a]["image"] for a in order if a in rows]], (W, H))  # fmt: skip
        log(f"edit case{n} sheet: ref, {', '.join(a for a in order if a in rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=list(PARTS), required=True)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--kernel", choices=["fast", "dense"], default="fast")
    ap.add_argument("--pages", default=None, help="comma list of input stems")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--lora", default=None, help="fill: the C checkpoint, no suffix")
    args = ap.parse_args()
    arms = (args.arms or PARTS[args.part]["arms"]).split(",")
    seeds = [int(s) for s in (args.seeds or PARTS[args.part]["seeds"]).split(",")]
    out = args.out or resolve_under_home(
        f"output/tests/ref_gate/{time.strftime('%Y%m%d-%H%M')}-{args.part}"
    )
    out.mkdir(parents=True, exist_ok=True)
    if args.part == "edit":
        run_edit(out, arms, args.kernel)
    else:
        run_task(
            args.part,
            out,
            arms,
            seeds,
            args.pages and args.pages.split(","),
            args.kernel,
            args.lora,
        )
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
