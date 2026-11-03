"""How much the MOSS DiT reads the chat-template rows of the eval prompts.

The A/B eval prompts (``bench/moss_ab/lora_ab.py`` PROMPTS + the composition set,
``{artist}`` = the caststation trigger, layout JSON, INT8 TE) are rendered with
Base, Euler, CFG 3, no LoRA, no Spectrum. On the cond branch, every block and step,
a fixed sample of image queries (and every caption-body query) is scored against all
keys; per key segment (image / template prefix / caption body / template suffix):

- attention mass;
- value contribution ``‖Σ_seg p·v‖ / ‖Σ_all p·v‖`` (a sink takes mass and returns
  little);
- per template token, the mass on that token alone.

``--lora`` attaches an adapter (eager, as ``load_for_inference`` leaves it).
``raw.pt`` keeps the per-head records, the value rows of the template tokens and the
caption mean at ``--save_steps``, and each prompt's final latent, for cross-prompt and
cross-arm comparisons.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import library.moss.model as mm
from bench.moss_ab._render import uncond_embeds
from bench.moss_ab.lora_ab import PROMPTS
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader
from library.moss.cache import caption_converter
from library.moss.generate import denoise, initial_noise
from library.moss.model import snapped_sigmas
from probe import tokenize

COMPOSITION = "project/finished/moss_lora/composition_prompts.txt"
SEGS = ("image", "prefix", "body", "suffix")


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


class Capture:
    """Patches ``mm._attention``; scores only calls whose sequence is the cond
    branch's joint length (the text blocks and the uncond branch differ)."""

    def __init__(self, n_img, seg, n_q, heads, save_steps=()):
        self.n_img, self.heads = n_img, heads
        self.save_steps = set(save_steps)
        self.vals = {}  # (step, block) -> (template rows (T, H·D), caption mean (H·D,))
        self.seg = seg.cuda()
        self.L = n_img + len(seg)
        g = torch.Generator().manual_seed(0)
        self.q_img = torch.randperm(n_img, generator=g)[:n_q].cuda()
        self.q_body = n_img + (seg == 1).nonzero()[:, 0].cuda()
        self.tmpl = n_img + (seg != 1).nonzero()[:, 0].cuda()
        key_seg = torch.cat([torch.zeros(n_img, dtype=torch.long), seg.long() + 1])
        self.n_seg = int(seg.max()) + 2  # image + the text segments
        self.onehot = torch.nn.functional.one_hot(key_seg, self.n_seg).float().cuda()
        self.block = 0
        self.records = []  # per (step, block): dict of tensors
        self.step = 0
        self._orig = mm._attention

    def __enter__(self):
        mm._attention = self._wrapped
        return self

    def __exit__(self, *exc):
        mm._attention = self._orig

    def _wrapped(self, q, k, v, rope, heads, key_mask=None, *args, **kw):
        if q.shape[1] == self.L and key_mask is None:
            self._score(q, k, v, rope, heads)
        return self._orig(q, k, v, rope, heads, key_mask, *args, **kw)

    @torch.no_grad()
    def _score(self, q, k, v, rope, heads):
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        q = mm.rotate(q.reshape(shape), rope)[0].float()
        k = mm.rotate(k.reshape(shape), rope)[0].float()
        v = v.reshape(shape)[0].float()
        rec = {"step": self.step, "block": self.block}
        if self.step in self.save_steps:
            self.vals[(self.step, self.block)] = (
                v[self.tmpl].flatten(1).bfloat16().cpu(),
                v[self.q_body].mean(0).flatten().bfloat16().cpu(),
            )
        for name, idx in (("img", self.q_img), ("body", self.q_body)):
            p = torch.einsum("qhd,lhd->hql", q[idx], k).mul_(shape[-1] ** -0.5)
            p = p.softmax(-1)  # (H, Q, L)
            rec[f"{name}_mass"] = (p @ self.onehot).mean(1).cpu()  # (H, 4)
            total = torch.einsum("hql,lhd->hqd", p, v).norm(dim=-1)
            contrib = []
            for s in range(self.n_seg):
                m = self.onehot[:, s].bool()
                o = torch.einsum("hql,lhd->hqd", p[..., m], v[m]).norm(dim=-1)
                contrib.append((o / total).mean(1))
            rec[f"{name}_contrib"] = torch.stack(contrib, -1).cpu()  # (H, n_seg)
            rec[f"{name}_tmpl_tok"] = p[..., self.tmpl].mean(1).cpu()  # (H, T)
            del p
        self.records.append(rec)


def summarise(records, n_steps, tmpl_labels, n_tokens, segs=SEGS):
    """Mean over heads / blocks per step-third, plus per-block and per-token views."""
    out = {"segment_tokens": n_tokens}
    thirds = {
        "early": range(0, n_steps // 3),
        "mid": range(n_steps // 3, 2 * n_steps // 3),
        "late": range(2 * n_steps // 3, n_steps),
    }
    for q in ("img", "body"):
        for metric in ("mass", "contrib"):
            key = f"{q}_{metric}"
            table = {}
            for name, rng in thirds.items():
                rs = [r[key] for r in records if r["step"] in rng]
                t = torch.stack(rs)  # (N, H, n_seg)
                table[name] = {
                    s: {
                        "mean": float(t[..., i].mean()),
                        "max_head_block": float(t[..., i].mean(0).max()),
                    }
                    for i, s in enumerate(segs)
                }
            out[key] = table
            by_block = torch.stack(
                [
                    torch.stack([r[key] for r in records if r["block"] == b]).mean(0)
                    for b in range(42)
                ]
            )  # (42, H, n_seg)
            out[f"{key}_by_block"] = by_block.mean(1).tolist()
            out[f"{key}_by_block_maxhead"] = by_block.max(1).values.tolist()
        tok = torch.stack([r[f"{q}_tmpl_tok"] for r in records]).mean((0, 1))
        out[f"{q}_tmpl_token_mass"] = dict(zip(tmpl_labels, tok.tolist()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="@channel (caststation)")
    ap.add_argument("--te_precision", default="int8")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_q", type=int, default=1024)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--lora", default="")
    ap.add_argument("--save_steps", default="0,10,20,29")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="default: output/tests/template_attn/<stamp>",
    )
    args = ap.parse_args()
    args.out = args.out or resolve_under_home(
        f"output/tests/template_attn/{time.strftime('%Y%m%d-%H%M')}"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    w, h = map(int, args.size.split("x"))
    device = torch.device("cuda")

    templates = list(PROMPTS) + [
        line
        for line in resolve_under_home(COMPOSITION).read_text().splitlines()
        if line.strip()
    ]
    conv = caption_converter("layout_json")
    prompts = [conv(p.format(artist=args.artist)) for p in templates]
    embeds, uncond, uncond_mask = uncond_embeds(device, prompts, args.te_precision)
    tok = loader.load_tokenizer(default_moss_dir())
    tr = loader.load_transformer("", args.variant, device=device)
    if args.lora:
        from library.moss.lora import load_for_inference

        load_for_inference(tr, resolve_under_home(args.lora))
    heads = tr.layers[0].heads
    save_steps = [int(x) for x in args.save_steps.split(",")]
    cap_ref = []
    for i, layer in enumerate(tr.layers):
        layer.register_forward_pre_hook(
            lambda m, a, i=i: setattr(cap_ref[0], "block", i) if cap_ref else None
        )
    sigmas = snapped_sigmas(args.steps)
    n_img = (h // 16) * (w // 16)
    results, raw = [], []
    for pi, text in enumerate(prompts):
        ids, seg = tokenize(tok, text, True)
        cond = embeds[text]
        assert cond.shape[-2] == seg.shape[0], (cond.shape, seg.shape)
        labels = [
            f"{'pre' if s == 0 else 'suf'}{j}:{tok.decode([int(t)])!r}"
            for j, (t, s) in enumerate(zip(ids[0].tolist(), seg.tolist()))
            if s != 1
        ]
        cap = Capture(n_img, seg, args.n_q, heads, save_steps)
        cap_ref[:] = [cap]
        cond, cm = loader.pad_embeds([cond], device)
        with cap:

            def on_step(i, _x0, cap=cap):
                cap.step = i + 1

            latent = denoise(
                tr,
                initial_noise(args.seed, h, w).to(device),
                cond,
                cm,
                uncond,
                uncond_mask,
                sigmas,
                cfg=args.cfg,
                sampler="euler",
                on_step=on_step,
            )
        n_tokens = {s: int((seg == i).sum()) for i, s in enumerate(SEGS[1:])}
        n_tokens["image"] = n_img
        summ = summarise(cap.records, args.steps, labels, n_tokens)
        summ["prompt"] = templates[pi]
        results.append(summ)
        raw.append(
            {
                "prompt": templates[pi],
                "labels": labels,
                "records": cap.records,
                "vals": cap.vals,
                "latent": latent.cpu(),
            }
        )
        m = summ["img_mass"]
        c = summ["img_contrib"]
        log(
            f"p{pi} S={cond.shape[1]} (pre {n_tokens['prefix']} body {n_tokens['body']} "
            f"suf {n_tokens['suffix']})  img→ mass "
            + " ".join(
                f"{t}: " + "/".join(f"{m[t][s]['mean']:.3f}" for s in SEGS[1:])
                for t in m
            )
            + "  contrib "
            + " ".join(
                f"{t}: " + "/".join(f"{c[t][s]['mean']:.3f}" for s in SEGS[1:])
                for t in c
            )
        )
        cap_ref.clear()
    (args.out / "summary.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False)
    )
    torch.save({"lora": args.lora, "prompts": raw}, args.out / "raw.pt")
    log(f"wrote {args.out}/summary.json")


if __name__ == "__main__":
    main()
