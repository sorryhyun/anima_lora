"""Attention mass on an assistant-turn tail appended after the shipped template.

``template_attn.py``'s read (Base, Euler 30 steps, CFG 3, no Spectrum, seed 1, 1024²,
layout JSON, INT8 TE, cond branch, a fixed sample of image queries plus every caption
query) with a fifth key segment: the rows the tail adds. Cond and uncond both carry the
tail (``docs/moss/findings/assistant_tail.md``). Tails, appended to the formatted prompt
text (``ok`` / ``sure`` merge the template's last ``\\n`` into ``\\n\\n``, so that
suffix row changes token):

- ``none``: shipped;
- ``ok``: ``\\n</think>\\n\\nOK``;
- ``sure``: ``\\n</think>\\n\\nSure,``;
- ``think``: ``Okay,``;
- ``pad4``: four ``\\n`` token ids (198) appended to the ids.

    make daemon-run ARGS="--stall-timeout 900 project/moss_te_sae/src/tail_attn.py"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bench.moss_ab.lora_ab import PROMPTS
from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader
from library.moss.cache import caption_converter
from library.moss.generate import denoise, initial_noise
from library.moss.model import snapped_sigmas
from probe import tokenize
from template_attn import Capture, summarise

TAILS = {
    "none": "",
    "ok": "\n</think>\n\nOK",
    "sure": "\n</think>\n\nSure,",
    "think": "Okay,",
    "pad4": None,
}
SEGS5 = ("image", "prefix", "body", "suffix", "tail")


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def tail_ids(tok, text: str, tail: str):
    """``(ids (1, S), seg (S,))``: seg 0 prefix / 1 body / 2 suffix / 3 tail rows."""
    ids0, seg0 = tokenize(tok, text, True) if text else _empty(tok)
    if tail == "pad4":
        nl = tok.convert_tokens_to_ids("Ċ")
        ids = torch.cat((ids0, torch.full((1, 4), nl, dtype=ids0.dtype)), 1)
    else:
        formatted = loader.prompt_text(tok, text) + TAILS[tail]
        ids = tok(formatted, return_tensors="pt").input_ids
    n0 = ids0.shape[1]
    assert torch.equal(ids[0, : n0 - 1], ids0[0, : n0 - 1])
    seg = torch.cat((seg0, torch.full((ids.shape[1] - n0,), 3, dtype=seg0.dtype)))
    return ids, seg


def _empty(tok):
    ids = loader.prompt_ids(tok, "").input_ids
    seg = torch.tensor([0, 0, 0] + [2] * (ids.shape[1] - 3), dtype=torch.int8)
    return ids, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artist", default="@channel (caststation)")
    ap.add_argument("--tails", default="none,ok,sure,pad4")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_q", type=int, default=1024)
    ap.add_argument("--save_steps", default="0,10,20,29")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or resolve_under_home(
        f"output/tests/tail_attn/{time.strftime('%Y%m%d-%H%M')}"
    )
    out.mkdir(parents=True, exist_ok=True)
    w, h = map(int, args.size.split("x"))
    device = torch.device("cuda")
    tails = args.tails.split(",")

    conv = caption_converter("layout_json")
    prompts = [conv(p.format(artist=args.artist)) for p in PROMPTS]
    tok = loader.load_tokenizer(default_moss_dir())
    seqs = {
        (t, pi): tail_ids(tok, text, t)
        for t in tails
        for pi, text in [(-1, ""), *enumerate(prompts)]
    }
    longest = max(ids.shape[1] for ids, _ in seqs.values())
    _, te = loader.load_text_encoder(
        default_moss_dir(), device, longest, tok=tok, precision="int8"
    )
    embeds = {}
    with torch.no_grad():
        for k, (ids, _) in seqs.items():
            ids = ids.to(device)
            o = te(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            embeds[k] = o.last_hidden_state[0].cpu()
    del te
    loader.empty_cache()

    tr = loader.load_transformer("", "base", device=device)
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
    for t in tails:
        uncond, um = loader.pad_embeds([embeds[t, -1]], device)
        for pi in range(len(prompts)):
            ids, seg = seqs[t, pi]
            labels = [
                f"{('pre', '', 'suf', 'tail')[s]}{j}:{tok.decode([int(x)])!r}"
                for j, (x, s) in enumerate(zip(ids[0].tolist(), seg.tolist()))
                if s != 1
            ]
            cap = Capture(n_img, seg, args.n_q, heads, save_steps)
            cap_ref[:] = [cap]
            cond, cm = loader.pad_embeds([embeds[t, pi]], device)
            with cap:

                def on_step(i, _x0, cap=cap):
                    cap.step = i + 1

                latent = denoise(
                    tr,
                    initial_noise(args.seed, h, w).to(device),
                    cond,
                    cm,
                    uncond,
                    um,
                    sigmas,
                    cfg=args.cfg,
                    sampler="euler",
                    on_step=on_step,
                )
            segs = SEGS5[: cap.n_seg]
            n_tokens = {s: int((seg == i).sum()) for i, s in enumerate(segs[1:])}
            n_tokens["image"] = n_img
            summ = summarise(cap.records, args.steps, labels, n_tokens, segs)
            summ.update(tail=t, prompt=pi)
            results.append(summ)
            raw.append(
                {
                    "tail": t,
                    "prompt": pi,
                    "labels": labels,
                    "records": cap.records,
                    "vals": cap.vals,
                    "latent": latent.cpu(),
                }
            )
            m, c = summ["img_mass"], summ["img_contrib"]
            log(
                f"{t} p{pi} S={cond.shape[1]} {n_tokens}  img mass "
                + " ".join(
                    f"{th}: " + "/".join(f"{m[th][s]['mean']:.3f}" for s in segs[1:])
                    for th in m
                )
                + "  contrib "
                + " ".join(
                    f"{th}: " + "/".join(f"{c[th][s]['mean']:.3f}" for s in segs[1:])
                    for th in c
                )
            )
            tm = summ["img_tmpl_token_mass"]
            log("  per token: " + " ".join(f"{k}={v:.4f}" for k, v in tm.items()))
            cap_ref.clear()
    (out / "summary.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    torch.save({"prompts": raw}, out / "raw.pt")
    log(f"wrote {out}/summary.json")


if __name__ == "__main__":
    main()
