"""Where the swap_text LoRA reads each item's string, and whether the t2i string heads
carry it.

The four held-out pages of ``output/tests/moss_i2i_swap_text_region/`` (2 en, 2 ko;
11 text items) with their staged prompts and JA references, rendered as
``scripts/moss/generate.py`` renders them (ER-SDE 50 steps, Spectrum, CFG 3, uncond
skip 22, seed 0, INT8 TE, eager) with ``--lora``. Arms, cond branch only:

- ``lora``: the reference render. On every real forward, image queries in each item's
  box (and 1024 sampled queries outside every box) are scored against every key: mass
  on the item's own string (its ``"text"`` value tokens), the other items' strings,
  its own and the other boxes' coordinate tokens, the template rows, the rest of the
  caption, the reference token at the query's own cell (``copy``) and the whole
  reference; per block and head, summed per step third;
- ``no_string_keys``: image queries cannot attend to any item's string tokens;
- ``heads_off``: only the heads that read a quoted string in t2i
  (``quote_attn``: b13 h20, b21 h20, b24 h11, b26 h12) lose the string keys.

``crops.png``: each item's box in every arm, for reading glyph by glyph.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

import library.moss.model as mm
from bench.moss_bbox_grounding.probe import box_cells
from library.env import resolve_under_home
from library.moss import loader
from library.moss.generate import (
    parse_skip,
    build_spectrum,
    cached_embeds,
    denoise,
    initial_noise,
    ref_latents,
)
from library.moss.lora import load_for_inference
from library.moss.model import decode, snapped_sigmas
from library.moss.requests import GenerateRequest, resolve_model_dir
from probe import tokenize
from template_causal import log

REGION_DIR = "output/tests/moss_i2i_swap_text_region"
PAGES = ("1483610_0007_en", "3124789_0042_en", "589883_0004_ko", "737486_0019_ko")
REF_ARM = "en200ko800_rw2_mod16_ema"
ARMS = ("lora", "no_string_keys", "heads_off")
T2I_HEADS = {13: [20], 21: [20], 24: [11], 26: [12]}
STATS = (
    "own_string",
    "other_string",
    "own_coord",
    "other_coord",
    "template",
    "caption_rest",
    "image",
    "ref",
    "copy",
)
N_REST = 1024
HEAD_CHUNK = 6


def read_page(name):
    args = json.load(
        open(resolve_under_home(f"{REGION_DIR}/{name}/{REF_ARM}/manifest.json"))
    )
    rec = args["images"][0]
    return dict(
        name=name,
        key=rec["key"],
        prompt=rec["prompt"],
        width=rec["width"],
        height=rec["height"],
        ref=rec["ref"],
        ref_scale=rec["ref_scale"],
        ref_png=rec["image"],
    )


def item_spans(tok, caption):
    """Per item, the joint-text token indices of its string and its bbox list."""
    formatted = tok.apply_chat_template(
        [{"role": "user", "content": caption}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    base = formatted.index(caption)
    enc = tok(formatted, return_offsets_mapping=True)
    off = enc["offset_mapping"]

    def pick(a, b):
        a, b = base + a, base + b
        return [j for j, (s, e) in enumerate(off) if s < b and e > a]

    spans, pos = [], caption.index('"items"')
    for item in json.loads(caption)["items"]:
        key = '"text":"'
        lo = caption.index(key, pos) + len(key)
        hi = lo + len(json.dumps(item["text"], ensure_ascii=False)) - 2
        assert caption[lo:hi] == json.dumps(item["text"], ensure_ascii=False)[1:-1]
        blo = caption.index('"bbox":[', hi) + len('"bbox":[')
        bhi = caption.index("]", blo)
        spans.append(dict(string=pick(lo, hi), coord=pick(blo, bhi)))
        pos = bhi
    return spans


class Capture:
    """Patches ``mm._attention``. On the cond branch's main call (image + text
    queries, cached reference keys appended) it scores the probe queries; with
    ``cut`` set it masks the string keys for image queries (all heads, or the
    heads in ``cut_heads`` per block)."""

    def __init__(self, n_img, n_text, seg, items, regions, heads, n_blocks):
        self.n_img, self.n_text = n_img, n_text
        self.L = n_img + n_text
        self.heads, self.n_items = heads, len(items)
        dev = "cuda"
        # Key categories over image | text | reference: 0 image, 1 template,
        # 2 caption rest, 3 reference, 4 + j string j, 4 + J + j coord j.
        J = len(items)
        cat = torch.cat(
            [
                torch.zeros(n_img, dtype=torch.long),
                torch.where(seg == 1, 2, 1).long(),
                torch.full((n_img,), 3, dtype=torch.long),
            ]
        )
        self.string_keys = torch.zeros(self.L, dtype=torch.bool)
        for j, it in enumerate(items):
            for t in it["string"]:
                cat[n_img + t] = 4 + j
                self.string_keys[n_img + t] = True
            for t in it["coord"]:
                cat[n_img + t] = 4 + J + j
        self.onehot = torch.nn.functional.one_hot(cat, 4 + 2 * J).float().to(dev)
        self.groups = [r.nonzero()[:, 0].to(dev) for r in regions]
        outside = ~torch.stack(regions).any(0)
        g = torch.Generator().manual_seed(0)
        rest = outside.nonzero()[:, 0]
        self.groups.append(
            rest[torch.randperm(len(rest), generator=g)[:N_REST]].to(dev)
        )
        self.acc = torch.zeros(
            3, n_blocks, heads, J + 1, len(STATS), dtype=torch.float64
        )
        self.count = torch.zeros(3, n_blocks)
        self.capture, self.cut, self.cut_heads = False, False, None
        self.block = self.step = 0
        self.third = 1
        self._orig = mm._attention
        self.mask = None

    def __enter__(self):
        mm._attention = self._wrapped
        return self

    def __exit__(self, *exc):
        mm._attention = self._orig

    def _wrapped(self, q, k, v, rope, heads, key_mask=None, ref_start=None,
                 ref_isolate=False, ref_kv=None):  # fmt: skip
        main = q.shape[1] == self.L and ref_kv is not None and ref_start is None
        if not main:
            return self._orig(
                q, k, v, rope, heads, key_mask, ref_start, ref_isolate, ref_kv
            )
        assert key_mask is None
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        qr = mm.rotate(q.reshape(shape), rope).to(v.dtype)
        kr = mm.rotate(k.reshape(shape), rope).to(v.dtype)
        vr = v.reshape(shape)
        ref_k, ref_v, ref_rope = ref_kv
        ref_shape = (*ref_k.shape[:2], heads, ref_k.shape[-1] // heads)
        ref_k = mm.rotate(ref_k.reshape(ref_shape), ref_rope).to(v.dtype)
        kr = torch.cat((kr, ref_k), 1)
        vr = torch.cat((vr, ref_v.reshape(ref_shape)), 1)
        if self.capture:
            self._score(qr, kr)
        out = self._orig(q, k, v, rope, heads, None, None, False, ref_kv)
        if not self.cut:
            return out
        Lk = kr.shape[1]
        if self.mask is None or self.mask.shape[-1] != Lk:
            m = torch.ones(self.L, Lk, dtype=torch.bool)
            sk = torch.cat(
                [self.string_keys, torch.zeros(Lk - self.L, dtype=torch.bool)]
            )
            m[: self.n_img, sk] = False
            self.mask = m[None, None].cuda()
        hs = (
            list(range(heads))
            if self.cut_heads is None
            else self.cut_heads.get(self.block, [])
        )
        if not hs:
            return out
        hs_t = torch.tensor(hs, device=q.device)
        sub = mm._sdpa(qr[:, :, hs_t], kr[:, :, hs_t], vr[:, :, hs_t], self.mask)
        out = out.unflatten(-1, (heads, -1))
        out[:, :, hs_t] = sub.to(out.dtype)
        return out.flatten(2)

    @torch.no_grad()
    def _score(self, q, k):
        q, k = q[0].float(), k[0].float()
        n = self.n_img
        scale = q.shape[-1] ** -0.5
        assert k.shape[0] == self.L + n, "reference not at the image's size"
        for gi, idx in enumerate(self.groups):
            for h0 in range(0, self.heads, HEAD_CHUNK):
                self._group(q[idx, h0 : h0 + HEAD_CHUNK], k[:, h0 : h0 + HEAD_CHUNK],
                            idx, gi, slice(h0, h0 + HEAD_CHUNK), scale)  # fmt: skip
        self.count[self.third, self.block] += 1

    def _group(self, q, k, idx, gi, hs, scale):
        n, J, t = self.n_img, self.n_items, self.third
        p = torch.einsum("qhd,lhd->hql", q, k).mul_(scale).softmax(-1)
        m = p @ self.onehot  # (h, Q, C)
        strings, coords = m[..., 4 : 4 + J], m[..., 4 + J :]
        if gi < J:
            own_s, own_c = strings[..., gi], coords[..., gi]
        else:
            own_s = own_c = torch.zeros_like(m[..., 0])
        row = torch.stack(
            [
                own_s,
                strings.sum(-1) - own_s,
                own_c,
                coords.sum(-1) - own_c,
                m[..., 1],
                m[..., 2],
                m[..., 0],
                m[..., 3],
                p[:, torch.arange(len(idx), device=p.device), n + self.n_text + idx],
            ],
            -1,
        ).mean(1)  # (h, S)
        self.acc[t, self.block, hs, gi] += row.double().cpu()
        del p, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lora",
        default="output/ckpt_moss_i2i/moss_i2i_swap_text_en200_ko800_rw2_mod16_ema5e5.safetensors",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or resolve_under_home(
        f"output/tests/swap_text_attn/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    req = GenerateRequest()
    model_dir = resolve_model_dir("")
    pages = [read_page(p) for p in PAGES]
    embeds = cached_embeds(
        model_dir,
        [p["prompt"] for p in pages] + [""],
        out / "_te_cache",
        req.text_encoder,
    )
    refs = ref_latents(model_dir, pages, device)
    tok = loader.load_tokenizer(model_dir)
    for p in pages:
        p["items"] = item_spans(tok, p["prompt"])
        _, seg = tokenize(tok, p["prompt"], True)
        p["seg"] = seg
        assert embeds[p["prompt"]].shape[0] == seg.shape[0]
        layout = json.loads(p["prompt"])["items"]
        p["texts"] = [it["text"] for it in layout]
        p["boxes"] = [it["bbox"] for it in layout]
        p["regions"] = [box_cells(b, p["height"], p["width"]) for b in p["boxes"]]
        for it, txt in zip(p["items"], p["texts"]):
            log(
                f"{p['name']}: {txt!r} -> {len(it['string'])} string / {len(it['coord'])} coord tokens"
            )

    lat_dir = out / "latents"
    lat_dir.mkdir(exist_ok=True)
    tr = loader.load_transformer(model_dir, req.model, device=device)
    load_for_inference(tr, resolve_under_home(args.lora))
    log(
        f"lora {args.lora}: ref_timestep {tr.ref_timestep} ref_isolate {tr.ref_isolate}"
    )
    block_ref = []
    for b, layer in enumerate(tr.layers):
        layer.register_forward_pre_hook(
            lambda m, a, b=b: setattr(block_ref[0], "block", b) if block_ref else None
        )
    uncond, uncond_mask = loader.pad_embeds([embeds[""]], device)
    sigmas = snapped_sigmas(req.steps, req.shift)
    caps, latents = {}, {}
    for arm in ARMS:
        for p in pages:
            path = lat_dir / f"{arm}_{p['name']}.safetensors"
            if path.exists() and arm != "lora":
                latents[(arm, p["name"])] = load_file(path)["latent"]
                continue
            cond = embeds[p["prompt"]]
            n_img = (p["height"] // 16) * (p["width"] // 16)
            cap = Capture(n_img, cond.shape[0], p["seg"], p["items"], p["regions"],
                          tr.layers[0].heads, len(tr.layers))  # fmt: skip
            cap.capture = arm == "lora"
            cap.cut = arm != "lora"
            cap.cut_heads = T2I_HEADS if arm == "heads_off" else None
            block_ref[:] = [cap]
            cond, cm = loader.pad_embeds([cond], device)

            def on_step(i, _x0, cap=cap):
                cap.third = min(3 * (i + 1) // req.steps, 2)

            cap.third = 0
            with cap:
                z = denoise(
                    tr,
                    initial_noise(0, p["height"], p["width"]).to(device),
                    cond,
                    cm,
                    uncond,
                    uncond_mask,
                    sigmas,
                    cfg=req.cfg,
                    uncond_skip=parse_skip(req.uncond_skip),
                    spectrum=build_spectrum(req),
                    sampler=req.sampler,
                    seed=0,
                    on_step=on_step,
                    ref=refs[p["key"]].to(device),
                ).cpu()
            block_ref.clear()
            if arm == "lora":
                caps[p["name"]] = cap
            latents[(arm, p["name"])] = z
            save_file({"latent": z.contiguous()}, str(path))
            log(f"{arm} {p['name']} done")
    del tr
    loader.empty_cache()

    vae = loader.load_vae(model_dir, device=device)
    images = {}
    for (arm, name), z in latents.items():
        x = decode(z.to(device), vae)[0].float().add(1).mul(127.5).round()
        x = x.clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
        Image.fromarray(x).save(out / f"{arm}_{name}.png")
        images[(arm, name)] = Image.fromarray(x)
    del vae
    loader.empty_cache()

    results = []
    crops = []
    for p in pages:
        cap = caps[p["name"]]
        acc = (cap.acc / cap.count[:, :, None, None, None].clamp(min=1)).float()
        ref_png = np.asarray(Image.open(p["ref_png"]).convert("RGB"), np.float32)
        mine = np.asarray(images[("lora", p["name"])], np.float32)
        rec = dict(
            page=p["name"],
            texts=p["texts"],
            n_string_tokens=[len(it["string"]) for it in p["items"]],
            n_region_tokens=[int(r.sum()) for r in p["regions"]],
            px_lora_vs_shipped_render=float(np.abs(mine - ref_png).mean()),
            stats=list(STATS),
            # (third, block, head, group, stat); group J = outside every box
            by_block=acc.mean((0, 2)).tolist(),  # (block, group, stat)
            by_block_maxhead=acc.mean(0).max(1).values.tolist(),
            by_block_argmaxhead=acc.mean(0).argmax(1).tolist(),
        )
        results.append(rec)
        torch.save(acc.half(), out / f"acc_{p['name']}.pt")
        H, W = p["height"], p["width"]
        for j, box in enumerate(p["boxes"]):
            y0, x0, y1, x1 = box
            x0, x1 = int(x0 + W / 2) - 24, int(x1 + W / 2) + 24
            y0, y1 = int(y0 + H / 2) - 24, int(y1 + H / 2) + 24
            row = [Image.open(p["ref"]).convert("RGB").resize((W, H))] + [
                images[(a, p["name"])] for a in ARMS
            ]
            crops.append(
                [
                    im.crop((max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)))
                    for im in row
                ]
            )
        b = np.array(rec["by_block"])  # (B, G, S)
        J = len(p["items"])
        own = b[:, :J, 0].mean(1)
        other = b[:, :J, 1].mean(1)
        log(
            f"{p['name']}: px vs shipped {rec['px_lora_vs_shipped_render']:.1f} | box queries own string "
            f"{own.mean():.4f} (peak b{own.argmax()} {own.max():.3f}) other strings {other.mean():.4f} "
            f"| copy {b[:, :J, 8].mean():.3f} ref {b[:, :J, 7].mean():.3f} template {b[:, :J, 4].mean():.3f}"
        )

    cw = 320
    heights = [min(cw, int(r[0].height * cw / max(r[0].width, 1))) for r in crops]
    sheet = Image.new("RGB", (cw * 4, sum(heights) + 8 * len(crops)), "white")
    y = 0
    for row, hgt in zip(crops, heights):
        for j, c in enumerate(row):
            c = c.copy()
            c.thumbnail((cw, hgt))
            sheet.paste(c, (j * cw, y))
        y += hgt + 8
    sheet.save(out / "crops.png")
    (out / "summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    log(f"wrote {out}/summary.json, crops.png (columns: reference, {', '.join(ARMS)})")


if __name__ == "__main__":
    main()
