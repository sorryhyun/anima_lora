#!/usr/bin/env python
"""xattn_loc — where the quoted OCR strings' cross-attn lands on real training pages.

Question (2026-11-20): before an attention-alignment loss goes into Anima LoRA
training, is there anything for it to teach? On the pages of the training set
whose caption carries a text clause and whose OCR sidecar has the line's box,
the base DiT (no LoRA, shipped vocab pack rows) is run once per σ on the real
``x_t = (1 − σ)·x0 + σ·ε`` with the cached context the trainer feeds it, and
each block's eager cross-attn map is read at the string's token columns:

- ``own``   — the string's map over its own line box: in-box share / the box's
  area share (lift; 1 = no preference) and the rank AUROC of in-box against
  out-of-box image tokens;
- ``other`` — the same string over the other lines' boxes on the page (does the
  string find *its* line, or any text?);
- ``ctrl``  — as many random caption tokens (outside every string and text
  tag) over the same box: the meter's null;
- ``tag``   — the ``speech bubble`` / ``japanese text`` tag tokens over the
  union of the page's text boxes;
- order     — on one-row / one-column lines of ≥ 4 tokens, the Spearman ρ of
  token order against each token's attention centroid along the line, and the
  share of a token's in-box attention inside its own cells against chance.

Token columns come from ``HybridT5Encoder.encode_aligned`` on the cached v0
caption (char offsets per T5 position; checked against the cache's
``t5_attn_mask_v0``). A forward is cond-only at batch 1, as a training step.

Compile is off: the per-block hooks read q / k in Python on every forward.

    .venv/bin/python bench/xattn_loc/probe.py plan
    make daemon-run ARGS="--queue --stall-timeout 300 bench/xattn_loc/probe.py run --pages 200"
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

import numpy as np

from library.env import resolve_under_home

RESIZED = "post_image_dataset/resized"
CACHE = "post_image_dataset/lora"
OCR = "post_image_dataset/ocr"
TOK_PX = 16  # VAE 8 × patch 2
PAD = 0.5  # box dilation, image tokens
TAGS = ("speech bubble", "japanese text")
CLAUSE_RE = re.compile(r"([A-Za-z]+) (text|SFX) reads as ")
ALWAYS = ("channel_(caststation)",)


# ---------------------------------------------------------------- items (CPU)


def v0_caption(txt: Path) -> str:
    var = txt.with_suffix(".variants.txt")
    if var.exists():
        for line in var.read_text("utf-8").splitlines():
            if line.startswith("v0\t"):
                return line.split("\t", 1)[1]
    return txt.read_text("utf-8").strip()


def clause_strings(cap: str) -> list[dict]:
    """Every quoted string of the caption's text clauses: ``{kind, lang, a, b}``
    with ``[a, b)`` the string's content in ``cap``."""
    out = []
    for m in CLAUSE_RE.finditer(cap):
        i = m.end()
        while i < len(cap) and cap[i] == '"':
            j = cap.find('"', i + 1)
            if j < 0:
                break
            out.append({"lang": m.group(1), "kind": m.group(2), "a": i + 1, "b": j})
            i = j + 1
            if cap.startswith(", ", i):
                i += 2
    return out


def discover(encoder) -> tuple[list[dict], dict]:
    from anime_tools.captions.ocr_sidecar import read_ocr, text_boxes, usable_lines

    root, cache, ocr = (resolve_under_home(p) for p in (RESIZED, CACHE, OCR))
    items, why = [], {}

    def skip(k):
        why[k] = why.get(k, 0) + 1

    for txt in sorted(root.glob("*/*.txt")):
        if txt.name.count(".") > 1:
            continue
        cap = v0_caption(txt)
        if " reads as " not in cap:
            continue
        st, stem = txt.parent.name, txt.stem
        side = ocr / st / f"{stem}.ocr.txt"
        if not side.exists():
            skip("no_sidecar")
            continue
        npz = sorted((cache / st).glob(f"{stem}_*_anima.npz"))
        te = cache / st / f"{stem}_anima_te.safetensors"
        if not npz or not te.exists():
            skip("no_cache")
            continue
        lines = read_ocr(side)
        boxes = text_boxes(lines)
        ids, mask, offs = encoder.encode_aligned(cap, 512)
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
                skip("string_no_box" if box is None else "string_no_tokens")
                continue
            strings.append(
                {**s, "text": text, "box": list(box), "cols": cols,
                 "offs": [list(offs[j]) for j in cols]}
            )  # fmt: skip
        if not strings:
            skip("page_no_matched_string")
            continue
        tag_cols = sorted(
            {
                j
                for t in TAGS
                for m in re.finditer(re.escape(t), cap)
                for j, (x, y) in enumerate(offs[:-1])
                if y > x and x >= m.start() and y <= m.end()
            }
        )
        items.append(
            {
                "set": st, "stem": stem, "caption": cap, "npz": str(npz[0]),
                "te": str(te), "n_real": int(sum(mask)), "strings": strings,
                "tag_cols": tag_cols,
                "all_boxes": [list(ln.box) for ln in usable_lines(lines)],
            }
        )  # fmt: skip
    return items, why


def pick(items: list[dict], pages: int, seed: int) -> list[dict]:
    keep = [m for m in items if m["set"] in ALWAYS]
    rest = [m for m in items if m["set"] not in ALWAYS]
    random.Random(seed).shuffle(rest)
    return keep + rest[: max(0, pages - len(keep))]


def load_encoder():
    from bench._anima import DEFAULT_TEXT_ENCODER
    from library.anima.text_strategies import TokenizeStrategy
    from library.anima.vocab_pack import default_vocab_pack, load_vocab_pack
    from library.inference.text import ensure_text_strategies

    pack = load_vocab_pack(default_vocab_pack())
    assert pack is not None, "no vocab pack configured"
    ensure_text_strategies(DEFAULT_TEXT_ENCODER, vocab_pack=pack)
    return TokenizeStrategy.get_strategy().encoder, pack


# ---------------------------------------------------------------- geometry


def region(box_px, gh: int, gw: int, pad: float = PAD):
    import torch

    x0, y0, x1, y1 = (v / TOK_PX for v in box_px)
    ys = torch.arange(gh, dtype=torch.float32)[:, None] + 0.5
    xs = torch.arange(gw, dtype=torch.float32)[None, :] + 0.5
    m = (xs >= x0 - pad) & (xs <= x1 + pad) & (ys >= y0 - pad) & (ys <= y1 + pad)
    cy = min(gh - 1, max(0, int((y0 + y1) / 2)))
    cx = min(gw - 1, max(0, int((x0 + x1) / 2)))
    m[cy, cx] = True
    return m.flatten()


def auroc(score, pos):
    """Rank AUROC of ``score`` (…, N) for the boolean ``pos`` (N)."""
    import torch

    n_pos = int(pos.sum())
    n_neg = pos.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.full(score.shape[:-1], float("nan"), device=score.device)
    r = score.argsort(-1).argsort(-1).float() + 1.0
    return (r[..., pos].sum(-1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def line_axis(s: dict):
    """``(axis, n_chars)`` for a one-row / one-column line, else ``None``:
    axis 1 = top to bottom (tategaki column), 0 = left to right."""
    x0, y0, x1, y1 = s["box"]
    w, h = x1 - x0, y1 - y0
    n = s["b"] - s["a"]
    if "\\n" in s["text"] or "\n" in s["text"] or len(s["cols"]) < 4:
        return None
    if h >= 2.5 * w:
        return 1, n
    if w >= 2.5 * h:
        return 0, n
    return None


# ---------------------------------------------------------------- the read


class Reader:
    """Per forward: the head-mean and per-head softmax map of every block's
    cross-attn at the item's columns of interest."""

    def __init__(self, anima):
        from library.anima.models import Attention

        self.cols = None
        self.maps: dict[int, object] = {}
        self.heads: dict[int, object] = {}
        self.blocks = []
        for name, mod in anima.named_modules():
            m = re.search(r"blocks\.(\d+)\.", name)
            if not (isinstance(mod, Attention) and not mod.is_selfattn and m):
                continue
            b = int(m.group(1))
            self.blocks.append(b)
            mod.forward = self._wrap(mod, b, mod.forward)

    def _wrap(self, mod, b, orig):
        import torch

        def fwd(x, attn_params, context, rope_cos_sin=None):
            if self.cols is not None:
                with torch.no_grad():
                    q, k, _ = mod.compute_qkv(x, context, rope_cos_sin=None)
                    H, D = q.shape[-2:]
                    q = q.reshape(-1, H, D).float()  # (Nq, H, D), batch 1
                    k = k.reshape(-1, H, D).float()
                    p = torch.einsum("qhd,khd->hqk", q, k) * D**-0.5
                    p = torch.softmax(p, -1)[:, :, self.cols]  # (H, Nq, n)
                    self.heads[b] = p
                    self.maps[b] = p.mean(0)  # (Nq, n)
            return orig(x, attn_params, context, rope_cos_sin=rope_cos_sin)

        return fwd


def run(a) -> None:
    import torch
    from safetensors import safe_open

    from anima_lora import GenerationRequest
    from bench._anima import DEFAULT_DIT
    from bench._common import make_run_dir, write_result
    from library.anima.stream import latent_dtype
    from library.inference.models import load_dit_model

    encoder, pack = load_encoder()
    items, why = discover(encoder)
    items = pick(items, a.pages, a.seed)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    print(f"pack {pack.name}; pages {len(items)}; skipped {why}", flush=True)

    device = torch.device("cuda")
    req = GenerationRequest(prompt="x", dit=a.dit or DEFAULT_DIT, device="cuda", seed=0)
    g = req.to_args()
    g.compile_blocks = False
    anima = load_dit_model(g, device, torch.bfloat16)
    anima.eval()
    rd = Reader(anima)
    blocks = sorted(rd.blocks)
    nb = len(blocks)
    print(f"blocks {nb}", flush=True)

    rows, keep_maps, fm = [], {}, {s: [] for s in sigmas}
    head_sum, head_n = None, 0
    n_skip = {"mask_mismatch": 0, "size_mismatch": 0}
    rng = random.Random(a.seed)
    for n, m in enumerate(items):
        z = np.load(m["npz"])
        lk = next(k for k in z.files if k.startswith("latents_"))
        x0 = torch.from_numpy(z[lk]).float()
        h, w = x0.shape[-2:]
        gh, gw = h // 2, w // 2
        with safe_open(m["te"], "pt") as f:
            ctx = f.get_tensor("crossattn_emb_v0")
            cmask = int(f.get_tensor("t5_attn_mask_v0").sum())
        if cmask != m["n_real"]:
            n_skip["mask_mismatch"] += 1
            continue
        if any(b[2] > w * 8 + 8 or b[3] > h * 8 + 8 for b in m["all_boxes"]):
            n_skip["size_mismatch"] += 1
            continue
        strs = m["strings"]
        taken = {c for s in strs for c in s["cols"]} | set(m["tag_cols"])
        free = [j for j in range(m["n_real"] - 1) if j not in taken]
        ctrl = [rng.sample(free, min(len(s["cols"]), len(free))) for s in strs]
        cols = sorted(taken | {c for cs in ctrl for c in cs})
        at = {c: i for i, c in enumerate(cols)}
        rd.cols = torch.tensor(cols, device=device)
        reg = [region(s["box"], gh, gw).to(device) for s in strs]
        union = torch.zeros(gh * gw, dtype=torch.bool, device=device)
        for b in m["all_boxes"]:
            union |= region(b, gh, gw).to(device)
        gen = torch.Generator().manual_seed(a.seed * 100003 + n)
        eps = torch.randn(x0.shape, generator=gen)
        for sg in sigmas:
            xt = ((1 - sg) * x0 + sg * eps)[None, :, None].to(
                device, latent_dtype(anima)
            )
            t = torch.tensor([sg], device=device, dtype=xt.dtype)
            pm = torch.zeros(1, 1, h, w, dtype=torch.bfloat16, device=device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                v = anima(xt, t, ctx[None].to(device, torch.bfloat16), padding_mask=pm)
            v = v.float().squeeze(2)[0].cpu()
            fm[sg].append(float(((v - (eps - x0)) ** 2).mean()))
            A = torch.stack([rd.maps[b] for b in blocks])  # (nb, Nq, n)
            assert A.shape[1] == gh * gw, f"{A.shape} vs grid {gh}x{gw}"
            Hd = torch.stack([rd.heads[b] for b in blocks])  # (nb, H, Nq, n)

            def stat(mp, box):  # mp (…, Nq) → lift, auroc
                share = mp[..., box].sum(-1) / mp.sum(-1).clamp_min(1e-20)
                return share / box.float().mean(), auroc(mp, box)

            for i, s in enumerate(strs):
                ci = [at[c] for c in s["cols"]]
                mp = A[:, :, ci].sum(-1)
                own_l, own_a = stat(mp, reg[i])
                cm = A[:, :, [at[c] for c in ctrl[i]]].sum(-1)
                ctl_l, ctl_a = stat(cm, reg[i])
                others = [reg[j] & ~reg[i] for j in range(len(strs)) if j != i]
                others = [o for o in others if o.any()]
                if others:
                    oth = torch.stack([stat(mp, o)[0] for o in others]).mean(0)
                    oth_a = torch.stack([stat(mp, o)[1] for o in others]).mean(0)
                else:
                    oth = oth_a = torch.full((nb,), float("nan"), device=device)
                mass = mp.mean(-1) / (len(ci) / 512.0)
                hl, _ = stat(Hd[:, :, :, ci].sum(-1), reg[i])  # (nb, H)
                head_sum = hl.double() if head_sum is None else head_sum + hl.double()
                head_n += 1
                # order along a one-row / one-column line
                rho = cell = [float("nan")] * nb
                ax = line_axis(s)
                if ax is not None and int(reg[i].sum()) >= 4:
                    axis, nch = ax
                    x0b, y0b, x1b, y1b = (v_ / TOK_PX for v_ in s["box"])
                    idx = reg[i].nonzero().flatten()
                    coord = (idx // gw if axis else idx % gw).float() + 0.5
                    lo, hi = (y0b, y1b) if axis else (x0b, x1b)
                    u = ((coord - lo) / max(hi - lo, 1e-6)).clamp(0, 1 - 1e-6)
                    P = A[:, idx][:, :, ci]  # (nb, n_in, n_tok)
                    P = P / P.sum(1, keepdim=True).clamp_min(1e-20)
                    cen = (P * u[None, :, None]).sum(1).cpu().numpy()  # (nb, n_tok)
                    k = np.arange(len(ci))
                    rho = [spearman(k, cen[b_]) for b_ in range(nb)]
                    cells = (u * nch).long()  # char cell of each in-box token
                    own = torch.zeros(len(idx), len(ci), device=device)
                    for kk, (ca, cb) in enumerate(s["offs"]):
                        own[:, kk] = (
                            (cells >= ca - s["a"]) & (cells < cb - s["a"])
                        ).float()
                    frac = own.mean(0).clamp_min(1e-6)  # chance share per token
                    ok = own.sum(0) > 0
                    if ok.any():
                        cl = ((P * own[None]).sum(1) / frac[None])[:, ok].mean(-1)
                        cell = cl.cpu().tolist()
                for bi, b_ in enumerate(blocks):
                    rows.append(
                        {
                            "set": m["set"], "stem": m["stem"], "str": i,
                            "kind": s["kind"], "lang": s["lang"],
                            "n_tok": len(ci), "n_char": s["b"] - s["a"],
                            "area": float(reg[i].float().mean()),
                            "sigma": sg, "block": b_,
                            "own": float(own_l[bi]), "own_auc": float(own_a[bi]),
                            "ctrl": float(ctl_l[bi]), "ctrl_auc": float(ctl_a[bi]),
                            "other": float(oth[bi]), "other_auc": float(oth_a[bi]),
                            "mass": float(mass[bi]), "rho": rho[bi], "cell": cell[bi],
                        }
                    )  # fmt: skip
                if m["set"] in ALWAYS:
                    keep_maps[f"{m['stem']}|{i}|{sg}"] = (
                        mp.reshape(nb, gh, gw).cpu().numpy().astype(np.float16)
                    )
            if m["tag_cols"] and union.any():
                tm = A[:, :, [at[c] for c in m["tag_cols"]]].sum(-1)
                tl, ta = stat(tm, union)
                for bi, b_ in enumerate(blocks):
                    rows.append(
                        {
                            "set": m["set"], "stem": m["stem"], "str": -1,
                            "kind": "tag", "sigma": sg, "block": b_,
                            "area": float(union.float().mean()),
                            "own": float(tl[bi]), "own_auc": float(ta[bi]),
                        }
                    )  # fmt: skip
        if n % 20 == 0:
            print(f"  {n + 1} / {len(items)} {m['set']}/{m['stem']}", flush=True)

    out = make_run_dir("xattn_loc", a.label)
    fields = ["set", "stem", "str", "kind", "lang", "n_tok", "n_char", "area", "sigma",
              "block", "own", "own_auc", "ctrl", "ctrl_auc", "other", "other_auc",
              "mass", "rho", "cell"]  # fmt: skip
    with (out / "rows.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    np.savez_compressed(out / "caststation_maps.npz", **keep_maps)
    (out / "items.json").write_text(
        json.dumps(
            [m for m in items if m["set"] in ALWAYS], ensure_ascii=False, indent=1
        )
    )
    heads = (head_sum / max(head_n, 1)).cpu().numpy() if head_sum is not None else None
    if heads is not None:
        np.save(out / "head_lift.npy", heads)
    metrics = {
        "pages": len(items), "skipped_discover": why, "skipped_run": n_skip,
        "strings": len({(r["stem"], r["str"]) for r in rows if r["kind"] != "tag"}),
        "fm_mse": {str(s): float(np.mean(v)) for s, v in fm.items() if v},
        "summary": summarize(rows),
    }  # fmt: skip
    write_result(out, script=__file__, args=a, metrics=metrics, label=a.label)
    print(json.dumps(metrics["fm_mse"]), flush=True)
    print_summary(metrics["summary"])
    print(f"→ {out}", flush=True)


def summarize(rows: list[dict]) -> dict:
    """Per σ: the across-string mean of each read at every block, and the
    depth mean; strings only (the tag rows on their own)."""
    out: dict = {}
    for sg in sorted({r["sigma"] for r in rows}, reverse=True):
        rs = [r for r in rows if r["sigma"] == sg]
        per: dict = {}
        for key, sel in (
            ("str", lambda r: r["kind"] != "tag"),
            ("tag", lambda r: r["kind"] == "tag"),
        ):
            for b in sorted({r["block"] for r in rs}):
                rb = [r for r in rs if r["block"] == b and sel(r)]
                cols = ("own", "own_auc") if key == "tag" else (
                    "own", "own_auc", "ctrl", "ctrl_auc", "other", "other_auc",
                    "mass", "rho", "cell")  # fmt: skip
                per.setdefault(key, {})[b] = {
                    c: _nanmean([r.get(c, float("nan")) for r in rb]) for c in cols
                } | {"n": len(rb)}
        out[str(sg)] = per
    return out


def _nanmean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return round(float(np.mean(xs)), 4) if xs else None


def print_summary(S: dict) -> None:
    for sg, per in S.items():
        print(f"σ {sg}", flush=True)
        print(
            "  blk   own  auc | ctrl  auc | other auc |  mass |  rho  cell | tag  auc"
        )
        for b, r in per["str"].items():
            t = per.get("tag", {}).get(b, {})
            f = lambda v, w=5: f"{v:{w}.2f}" if v is not None else " " * (w - 1) + "-"  # noqa: E731
            print(
                f"  {b:>3} {f(r['own'])} {f(r['own_auc'], 4)} |{f(r['ctrl'])} "
                f"{f(r['ctrl_auc'], 4)} |{f(r['other'])} {f(r['other_auc'], 4)} |"
                f"{f(r['mass'])} |{f(r['rho'])} {f(r['cell'])} |"
                f"{f(t.get('own'))} {f(t.get('own_auc'), 4)}",
                flush=True,
            )


def plan(a) -> None:
    encoder, pack = load_encoder()
    items, why = discover(encoder)
    ns = sum(len(m["strings"]) for m in items)
    axes = [line_axis(s) for m in items for s in m["strings"]]
    print(f"pack {pack.name}")
    print(f"pages {len(items)}, matched strings {ns}, skipped {why}")
    print(
        f"order lines: {sum(1 for x in axes if x)} "
        f"(columns {sum(1 for x in axes if x and x[0] == 1)}, "
        f"rows {sum(1 for x in axes if x and x[0] == 0)})"
    )
    print(f"always: {sum(1 for m in items if m['set'] in ALWAYS)} pages of {ALWAYS}")
    for m in pick(items, 3, a.seed)[:3]:
        print(
            f"  {m['set']}/{m['stem']} real tokens {m['n_real']} tags {m['tag_cols']}"
        )
        for s in m["strings"]:
            print(
                f"    {s['lang']} {s['kind']} {s['text']!r} box {s['box']} cols {s['cols']}"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("verb", choices=["plan", "run"])
    p.add_argument("--pages", type=int, default=200)
    p.add_argument("--sigmas", default="0.9,0.75,0.6,0.45")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dit", default=None)
    p.add_argument("--label", default="base")
    a = p.parse_args()
    plan(a) if a.verb == "plan" else run(a)


if __name__ == "__main__":
    main()
