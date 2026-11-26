"""A glyph's place on a generated page, from the quote tokens' cross-attention.

``Recorder`` hooks blocks 8, 13, 14, 15 (a pre-hook on the block, outside the compiled
``_forward``, as ``library/training/xattn_align.py`` reads them) and keeps, a forward at
a time, the cross-attention probability from each image token to each quote token that
has a row. ``centres`` turns a glyph's map into a point inside the picked box, and
``window`` into the square the readers are given. Read on kmix's rows by
``probes/attn_loc.py`` (``aloc_1126_1501``)."""

from __future__ import annotations

BLOCKS = (8, 13, 14, 15)
HEADS4 = ((15, 1), (14, 15), (13, 7), (8, 14))  # ``bench/xattn_loc``'s
SETS = ("b13_15", "heads4")
TOK_PX = 16  # VAE 8 × patch 2
QCHUNK = 1024  # image queries a logits chunk
SIDE_MIN = 40.0  # px


def cross_probs(block, x, emb, ctx, attn_params, rope, ada, cols):
    """``(B, heads, Nq, n)``: the block's cross-attention probability at the key
    columns ``cols``, softmax over every key. ``xattn_align.block_cross_probs`` with
    the heads kept (it mirrors ``Block._forward`` up to the cross-attention)."""
    import torch

    if block.use_adaln_lora:
        down_self, down_cross, _ = block.adaln_fused_down(emb).chunk(3, dim=-1)
        sh_s, sc_s, g_s = (block.adaln_up_self_attn(down_self) + ada).chunk(3, dim=-1)
        sh_c, sc_c, _ = (block.adaln_up_cross_attn(down_cross) + ada).chunk(3, dim=-1)
    else:
        sh_s, sc_s, g_s = block.adaln_modulation_self_attn(emb).chunk(3, dim=-1)
        sh_c, sc_c, _ = block.adaln_modulation_cross_attn(emb).chunk(3, dim=-1)

    def e(t):
        return t[:, :, None, None, :]

    _, Tn, H, W, _ = x.shape
    nx = block.layer_norm_self_attn(x) * (1 + e(sc_s)) + e(sh_s)
    flat = nx.flatten(1, 3)
    res = block.self_attn(flat, attn_params, flat, rope_cos_sin=rope)
    x = x + e(g_s) * res.unflatten(1, (Tn, H, W))
    nx = block.layer_norm_cross_attn(x) * (1 + e(sc_c)) + e(sh_c)
    q, k, _ = block.cross_attn.compute_qkv(nx.flatten(1, 3), ctx)
    scale = (
        attn_params.softmax_scale
        if attn_params is not None and attn_params.softmax_scale
        else q.shape[-1] ** -0.5
    )
    out = []
    with torch.autocast(q.device.type, enabled=False):
        kf = k.float()
        for i in range(0, q.shape[1], QCHUNK):
            lg = torch.einsum("bqhd,bkhd->bhqk", q[:, i : i + QCHUNK].float(), kf)
            lg = lg * scale
            lse = torch.logsumexp(lg, dim=-1, keepdim=True)
            out.append(torch.exp(lg[..., cols] - lse))
    return torch.cat(out, dim=2)


class Recorder:
    """The quote tokens' maps of the listed blocks, a forward at a time."""

    def __init__(self, anima):
        self.cols = None  # the key columns; None: off
        self.maps: dict = {}
        for b in BLOCKS:
            anima.blocks[b].register_forward_pre_hook(self._hook(b), with_kwargs=True)

    def _hook(self, b: int):
        import torch

        def hook(block, args, kwargs):
            if self.cols is None:
                return
            x, emb, ctx, ap = args[:4]
            rope = kwargs.get("rope_cos_sin", args[4] if len(args) > 4 else None)
            ada = kwargs.get("adaln_lora_B_T_3D", args[5] if len(args) > 5 else None)
            with torch.no_grad():
                self.maps[b] = cross_probs(block, x, emb, ctx, ap, rope, ada, self.cols)

        return hook

    def sets(self):
        """``(B, 2, n, Nq)``: the forward's maps as ``SETS``."""
        import torch

        m = self.maps
        b13 = torch.stack([m[b].mean(1) for b in (13, 14, 15)]).mean(0)
        h4 = torch.stack([m[b][:, h] for b, h in HEADS4]).mean(0)
        return torch.stack([b13, h4], 1).transpose(2, 3)


def token_cols(pol, caption: str, ak: list, at: dict) -> tuple:
    """``(cols, groups, pos)``: the context position of each quote token that has a
    row, the tokens (indices into ``cols``) of each glyph that has one, and that
    glyph's index in ``ak``. A dot run is one glyph of one token or more."""
    from library.anima.ext_vocab import T5_TABLE_SIZE

    ids, mask = (v.tolist() for v in pol.hid[caption][2:4])
    index = pol.r.delta.index
    cols = [
        j
        for j, (v, m) in enumerate(zip(ids, mask))
        if m and v >= T5_TABLE_SIZE and v - T5_TABLE_SIZE in index
    ]
    pos = [p for p, k in enumerate(ak) if k in at]
    rows = [index[ids[j] - T5_TABLE_SIZE] for j in cols]
    groups, t = [], 0
    for p in pos:
        g = []
        while t < len(rows) and rows[t] == at[ak[p]] and (not g or ak[p] == "…"):
            g.append(t)
            t += 1
        assert g, (caption[-30:], rows, p)
        groups.append(g)
    assert t == len(rows), (caption[-30:], rows)
    return cols, groups, pos


def centres(maps, box, gh: int, gw: int) -> dict:
    """``maps`` ``(n, gh·gw)`` of one set → ``{centre kind: [(x, y) px a token]}``."""
    import torch
    import torch.nn.functional as F

    x0, y0, x1, y1 = (v / TOK_PX for v in box)
    ys = torch.arange(gh, dtype=torch.float32)[:, None] + 0.5
    xs = torch.arange(gw, dtype=torch.float32)[None, :] + 0.5
    inside = (xs >= x0 - 1) & (xs <= x1 + 1) & (ys >= y0 - 1) & (ys <= y1 + 1)
    m = maps.float().cpu().reshape(-1, gh, gw)
    tot = m.sum(0)
    live = inside & (tot >= tot[inside].median())
    kinds = {
        "raw": m * inside,
        "share": m / tot.clamp_min(1e-12) * live,
    }
    out = {}
    for kind, v in kinds.items():
        sm = F.avg_pool2d(v[:, None], 3, 1, 1, count_include_pad=False)[:, 0]
        pts = []
        for p in range(v.shape[0]):
            iy, ix = divmod(int(sm[p].argmax()), gw)
            ya, yb = max(0, iy - 1), min(gh, iy + 2)
            xa, xb = max(0, ix - 1), min(gw, ix + 2)
            w = v[p, ya:yb, xa:xb]
            s = float(w.sum())
            if s <= 0:
                pts.append(((ix + 0.5) * TOK_PX, (iy + 0.5) * TOK_PX))
                continue
            cy = float((w * ys[ya:yb]).sum()) / s
            cx = float((w * xs[:, xa:xb]).sum()) / s
            pts.append((cx * TOK_PX, cy * TOK_PX))
        out[kind] = pts
    return out


def window(c, side: float, W: int, H: int) -> list:
    x, y = c
    h = side / 2
    return [
        int(max(0, x - h)),
        int(max(0, y - h)),
        int(min(W, x + h)),
        int(min(H, y + h)),
    ]


def glyph_side(box, glyphs: int, scale: float) -> float:
    """A window's side in px: ``scale`` glyph sides, a glyph side being the root of
    the box's area over the string's glyphs."""
    x0, y0, x1, y1 = box
    return max(SIDE_MIN, scale * ((x1 - x0) * (y1 - y0) / glyphs) ** 0.5)
