"""The reference copy head across i2i LoRAs: does Base already copy, and does cutting it
break the copy.

Weight analysis (see reports/i2i_common.md) put every i2i LoRA's largest per-head
change on b39 h19, which in the swap_text run attends 0.98 to the reference token at the
query's own cell. Per task (swap_text, sanitize, colorize; two held-out inputs each)
the shipped generate config renders five arms:

- ``base``: the LoRA at multiplier 0 (Base, with the LoRA's ref_timestep / ref_isolate);
- ``lora``: the LoRA;
- ``cut_h19``: image queries of b39 h19 cannot attend to reference keys;
- ``cut_ctrl``: the same cut on b39 h5 (reads the reference 0.67, copy 0.007);
- ``cut_top8``: the cut on the eight heads with the most copy mass in the swap_text run.

``base`` and ``lora`` score every real forward: per block and head, the mass of 1024
sampled image queries on image / text / reference keys and on the reference token at
their own cell (``copy``), by step bucket (step 0, then thirds); b39 h19's copy mass per
image token is kept as a map. Step 0 sees the same x_T, prompt and reference in both
arms, so it compares Base and LoRA on identical input.

Pixel reads: luma L1 against the reference (resized to the output) and against the
``lora`` render; swap_text also outside its text boxes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import library.moss.generate as gen
import library.moss.model as mm
from library.env import resolve_under_home
from library.moss.requests import GenerateRequest

EVAL = "output/moss_i2i_eval"
TASKS = {
    "swap_text": dict(
        lora="output/ckpt_moss_i2i/moss_i2i_swap_text_v3_en400_ko1709_ksplit_rw2_mod16_ema5e5_int8_e2.safetensors",
        config="configs/moss/i2i/swap_text/generate.toml",
        inputs=[
            (
                f"{EVAL}/swap_text/inputs/{s}.src.webp",
                f"{EVAL}/swap_text/inputs/{s}.txt",
            )
            for s in ("1483610_0007_en", "589883_0004_ko")
        ],
    ),
    "sanitize": dict(
        lora="output/ckpt_moss_i2i/moss_i2i_sanitize_idt_w0003_late_lr5e5_e8.safetensors",
        config="configs/moss/i2i/sanitize/generate.toml",
        inputs=[
            (f"{EVAL}/sanitize/inputs/{s}.png", None)
            for s in ("hard_to_sanitize1", "usual_case1")
        ],
    ),
    "colorize": dict(
        lora="output/ckpt_moss_i2i/moss_colorize_v5.safetensors",
        config="configs/moss/i2i/colorize/generate.toml",
        inputs=None,  # first two of wild2/
    ),
}
COPY_TOP8 = {39: [19, 11], 41: [2], 40: [20], 36: [3, 21], 37: [20], 35: [3]}
ARMS = {
    "base": None,
    "lora": None,
    "cut_h19": {39: [19]},
    "cut_ctrl": {39: [5]},
    "cut_top8": COPY_TOP8,
}
STATS = ("image", "text", "ref", "copy")
N_Q = 1024
HEAD_CHUNK = 6


def log(msg):
    print(f"[copy_head {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Probe:
    """Patches ``mm._attention`` (main calls: target queries with cached reference
    keys) and ``gen.denoise`` / ``gen.load_transformer`` (job, step and block index)."""

    def __init__(self, te_cache):
        self.capture = False
        self.cut = None
        self.block = 0
        self.job = -1
        self.step = 0
        self.n_steps = 1
        self.records = {}  # job -> dict
        self.te_cache = te_cache
        self._orig = (
            mm._attention,
            gen.denoise,
            gen.load_transformer,
            gen.cached_embeds,
        )

    def __enter__(self):
        mm._attention = self._attention
        gen.denoise = self._denoise
        gen.load_transformer = self._load
        gen.cached_embeds = self._embeds
        return self

    def __exit__(self, *exc):
        mm._attention, gen.denoise, gen.load_transformer, gen.cached_embeds = self._orig

    def _embeds(self, model_dir, texts, _cache_dir, *a, **kw):
        # One TE cache for every arm: the arms differ only past the TE.
        return self._orig[3](model_dir, texts, self.te_cache, *a, **kw)

    def _load(self, *a, **kw):
        tr = self._orig[2](*a, **kw)
        for b, layer in enumerate(tr.layers):
            layer.register_forward_pre_hook(
                lambda m, args, b=b: setattr(self, "block", b)
            )
        self.heads = tr.layers[0].heads
        self.n_blocks = len(tr.layers)
        return tr

    def _denoise(self, *a, on_step=None, **kw):
        self.job += 1
        self.step = 0
        self.n_steps = len(a[6])

        def step_cb(i, x0):
            self.step = i + 1
            if on_step is not None:
                on_step(i, x0)

        return self._orig[1](*a, on_step=step_cb, **kw)

    def _bucket(self):
        if self.step == 0:
            return 0
        return 1 + min(3 * (self.step - 1) // max(self.n_steps - 1, 1), 2)

    def _attention(self, q, k, v, rope, heads, key_mask=None, ref_start=None,
                   ref_isolate=False, ref_kv=None):  # fmt: skip
        orig = self._orig[0]
        if ref_kv is None or ref_start is not None:
            return orig(q, k, v, rope, heads, key_mask, ref_start, ref_isolate, ref_kv)
        out = orig(q, k, v, rope, heads, key_mask, ref_start, ref_isolate, ref_kv)
        hs = (self.cut or {}).get(self.block, [])
        if not self.capture and not hs:
            return out
        shape = (*q.shape[:2], heads, q.shape[-1] // heads)
        qr = mm.rotate(q.reshape(shape), rope).to(v.dtype)
        kr = mm.rotate(k.reshape(shape), rope).to(v.dtype)
        vr = v.reshape(shape)
        ref_k, ref_v, ref_rope = ref_kv
        n_ref = ref_k.shape[1]
        ref_shape = (*ref_k.shape[:2], heads, ref_k.shape[-1] // heads)
        ref_k = mm.rotate(ref_k.reshape(ref_shape), ref_rope).to(v.dtype)
        kr = torch.cat((kr, ref_k), 1)
        vr = torch.cat((vr, ref_v.reshape(ref_shape)), 1)
        L = q.shape[1]
        n_img = n_ref  # ref_scale 1: one reference token per image token
        if self.capture:
            self._score(qr[0], kr[0], L, n_img, key_mask)
        if not hs:
            return out
        m = torch.ones(L, L + n_ref, dtype=torch.bool, device=q.device)
        m[:n_img, L:] = False
        if key_mask is not None:
            km = torch.cat((key_mask, key_mask.new_ones(key_mask.shape[0], n_ref)), 1)
            m = m[None] & km[:, None, :]
        m = m.reshape(-1, 1, L, L + n_ref) if m.dim() == 3 else m[None, None]
        h_t = torch.tensor(hs, device=q.device)
        sub = mm._sdpa(qr[:, :, h_t], kr[:, :, h_t], vr[:, :, h_t], m)
        out = out.unflatten(-1, (heads, -1)).clone()
        out[:, :, h_t] = sub.to(out.dtype)
        return out.flatten(2)

    @torch.no_grad()
    def _score(self, q, k, L, n_img, key_mask):
        n_text = L - n_img
        rec = self.records.setdefault(self.job, {})
        br = rec.setdefault(n_text, dict(
            acc=torch.zeros(4, self.n_blocks, self.heads, len(STATS), dtype=torch.float64),
            count=torch.zeros(4, self.n_blocks),
            h19=torch.zeros(n_img, dtype=torch.float64),
            h19_step0=None,
            h19_count=0,
            n_img=n_img,
        ))  # fmt: skip
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(n_img, generator=g)[: min(N_Q, n_img)].to(q.device)
        cat = torch.cat([
            torch.zeros(n_img, dtype=torch.long),
            torch.ones(n_text, dtype=torch.long),
            torch.full((k.shape[0] - L,), 2, dtype=torch.long),
        ]).to(q.device)  # fmt: skip
        onehot = torch.nn.functional.one_hot(cat, 3).float()
        valid = None
        if key_mask is not None:
            valid = torch.cat((key_mask[0], key_mask.new_ones(k.shape[0] - L)))
        q, k = q.float(), k.float()
        scale = q.shape[-1] ** -0.5
        t = self._bucket()
        for h0 in range(0, self.heads, HEAD_CHUNK):
            hsl = slice(h0, h0 + HEAD_CHUNK)
            s = torch.einsum("qhd,lhd->hql", q[idx, hsl], k[:, hsl]).mul_(scale)
            if valid is not None:
                s.masked_fill_(~valid, float("-inf"))
            p = s.softmax(-1)
            m = p @ onehot
            copy = p[:, torch.arange(len(idx), device=p.device), L + idx]
            row = torch.cat([m, copy[..., None]], -1).mean(1)
            br["acc"][t, self.block, hsl] += row.double().cpu()
            del s, p, m
        br["count"][t, self.block] += 1
        if self.block == 39:
            s = torch.einsum("qd,ld->ql", q[:n_img, 19], k[:, 19]).mul_(scale)
            if valid is not None:
                s.masked_fill_(~valid, float("-inf"))
            p = s.softmax(-1)
            c = p[
                torch.arange(n_img, device=p.device),
                L + torch.arange(n_img, device=p.device),
            ]
            br["h19"] += c.double().cpu()
            br["h19_count"] += 1
            if self.step == 0 and br["h19_step0"] is None:
                br["h19_step0"] = c.float().cpu()
            del s, p


def luma(im):
    a = np.asarray(im.convert("RGB"), np.float32)
    return a @ np.array([0.299, 0.587, 0.114], np.float32)


def text_mask(prompt, H, W):
    mask = np.zeros((H, W), bool)
    for it in json.loads(prompt)["items"]:
        y0, x0, y1, x1 = it["bbox"]
        y0, y1 = int(y0 + H / 2) - 16, int(y1 + H / 2) + 16
        x0, x1 = int(x0 + W / 2) - 16, int(x1 + W / 2) + 16
        mask[max(y0, 0) : min(y1, H), max(x0, 0) : min(x1, W)] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or resolve_under_home(
        f"output/tests/copy_head/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    wild2 = sorted(resolve_under_home(f"{EVAL}/colorize/inputs/wild2").iterdir())
    TASKS["colorize"]["inputs"] = [(str(p), None) for p in wild2[:2]]
    summary = {}
    for task in args.tasks.split(","):
        spec = TASKS[task]
        for arm in args.arms.split(","):
            arm_dir = out / task / arm
            for n, (ref, prompt_file) in enumerate(spec["inputs"]):
                argv = [
                    "--config", spec["config"],
                    "--lora", spec["lora"],
                    "--lora_multiplier", "0" if arm == "base" else "1",
                    "--ref_image", ref,
                    "--seed", "0",
                    "--out_dir", str(arm_dir / f"in{n}"),
                ]  # fmt: skip
                if prompt_file:
                    prompt = resolve_under_home(prompt_file).read_text().strip()
                    argv += ["--prompt", prompt]
                req = GenerateRequest.from_argv(argv)
                t0 = time.time()
                with Probe(out / "_te_cache") as probe:
                    probe.capture = arm in ("base", "lora")
                    probe.cut = ARMS[arm]
                    gen.run_generate(req)
                log(f"{task} {arm} in{n}: {time.time() - t0:.0f}s")
                man = json.loads((arm_dir / f"in{n}" / "manifest.json").read_text())
                rec = man["images"][0]
                entry = dict(image=rec["image"], ref=ref, prompt=req.prompt,
                             width=rec["width"], height=rec["height"])  # fmt: skip
                if probe.capture and probe.records:
                    br = probe.records[0]
                    cond = br[max(br)]
                    acc = cond["acc"] / cond["count"][:, :, None, None].clamp(min=1)
                    torch.save(dict(acc=acc.float(), stats=STATS,
                                    h19=(cond["h19"] / max(cond["h19_count"], 1)).float(),
                                    h19_step0=cond["h19_step0"], n_img=cond["n_img"]),
                               arm_dir / f"in{n}" / "attn.pt")  # fmt: skip
                    a = acc.float()
                    entry["b39h19"] = {
                        s: a[:, 39, 19, i].tolist() for i, s in enumerate(STATS)
                    }
                    top = a[0, :, :, 3].flatten().topk(8)
                    entry["top_copy_step0"] = [
                        (int(i) // 24, int(i) % 24, round(float(v), 3))
                        for v, i in zip(top.values, top.indices)
                    ]
                summary.setdefault(task, {}).setdefault(arm, []).append(entry)
                (out / "summary.json").write_text(
                    json.dumps(summary, indent=1, ensure_ascii=False)
                )

    # pixel reads and sheets
    for task, arms in summary.items():
        rows = []
        for n in range(len(next(iter(arms.values())))):
            lora = arms.get("lora", [None] * 9)[n]
            ref_im = (
                Image.open(resolve_under_home(arms["lora"][n]["ref"])) if lora else None
            )
            row = []
            for arm, entries in arms.items():
                e = entries[n]
                im = Image.open(e["image"]).convert("RGB")
                H, W = im.height, im.width
                L = luma(im)
                ref_l = luma(Image.open(resolve_under_home(e["ref"])).resize((W, H)))
                e["luma_vs_ref"] = float(np.abs(L - ref_l).mean())
                if lora:
                    e["luma_vs_lora"] = float(
                        np.abs(L - luma(Image.open(lora["image"]))).mean()
                    )
                if task == "swap_text":
                    keep = ~text_mask(e["prompt"], H, W)
                    e["luma_vs_ref_outside_text"] = float(
                        np.abs(L - ref_l)[keep].mean()
                    )
                row.append(im)
            if ref_im is not None:
                row.insert(0, ref_im.convert("RGB").resize(row[0].size))
            rows.append(row)
        tw = 384
        th = max(int(r[0].height * tw / r[0].width) for r in rows)
        sheet = Image.new("RGB", (tw * len(rows[0]), th * len(rows)), "white")
        for y, row in enumerate(rows):
            for x, im in enumerate(row):
                im = im.copy()
                im.thumbnail((tw, th))
                sheet.paste(im, (x * tw, y * th))
        sheet.save(out / f"sheet_{task}.png")
        log(f"{task}: sheet columns ref, {', '.join(arms)}")
        for arm, entries in arms.items():
            msg = " | ".join(
                f"L/ref {e['luma_vs_ref']:.1f}"
                + (f" L/lora {e['luma_vs_lora']:.1f}" if "luma_vs_lora" in e else "")
                + (
                    f" out-text {e['luma_vs_ref_outside_text']:.1f}"
                    if "luma_vs_ref_outside_text" in e
                    else ""
                )
                + (
                    f" h19 copy s0 {e['b39h19']['copy'][0]:.3f} late {e['b39h19']['copy'][3]:.3f}"
                    if "b39h19" in e
                    else ""
                )
                for e in entries
            )
            log(f"  {task} {arm:9}: {msg}")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
