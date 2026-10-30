"""Which parts of the prompt a MOSS LoRA responds to.

The LoRA's effect at one ``(x_t, σ)`` is ``Δ = v_lora − v_base`` (multiplier 1 vs 0,
same noise). Three reads:

- ``variants``: Δ under edits of the layout-JSON caption (artist / characters / tag bag /
  items removed, artist only, the plain tag caption the run also trained on, empty),
  as ``‖Δ_variant‖ / ‖Δ_full‖`` and ``cos(Δ_variant, Δ_full)``; captions from another
  artist's cache as an off-domain control;
- ``tokens``: gradient × input of ``D = ½‖Δ‖²`` on the TE rows of the full caption,
  shared out over caption fields;
- ``features``: the same gradient split over layer-31 SAE features, linearised through
  the final norm (``a/rms · (g⊙d)·∂D/∂e``).

TE: bf16 Qwen3.5-9B (what the channel cache was encoded with); the full re-encode is
checked against the cached ``embeds``.

A LoRA stamped with ``delta_rows`` (``library/moss/train.py``) runs with the row mask it
trained under, built per conditioning from its ``trigger``; a caption without the
trigger then has no rows and a zero delta.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
import zlib
from pathlib import Path

import torch
from safetensors import safe_open

from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss import loader
from library.moss.lora import load_for_inference
from library.moss.loader import trigger_rows
from library.moss.train import (
    NUM_TRAIN_TIMESTEPS,
    checkpoint_blocks,
    delta_row_ids,
    place,
    row_mask,
    sample_captions,
    step_reserve_gb,
)

GROUPS = ("template", "meta", "characters", "series", "artist", "tags", "items", "json")


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def variants(caption: str) -> dict[str, str]:
    full = json.loads(caption)
    assert dumps(full) == caption, "caption is not compact layout JSON"
    s = full.get("summary", {})

    def edit(fn):
        obj = json.loads(caption)
        fn(obj)
        return dumps(obj)

    def drop_chars(o):
        o["summary"].pop("characters", None)
        o["summary"].pop("series", None)
        for it in o.get("items", []):
            it.pop("characters", None)

    out = {
        "full": caption,
        "no_artist": edit(lambda o: o["summary"].pop("artists", None)),
        "no_chars": edit(drop_chars),
        "no_tags": edit(lambda o: o["summary"].pop("caption", None)),
        "no_items": edit(lambda o: o.pop("items", None)),
        "artist_only": dumps({"summary": {"artists": s.get("artists", [])}}),
        "empty": "",
    }
    return {k: v for k, v in out.items() if k == "full" or v != caption}


def field_spans(caption: str) -> list[tuple[int, int, str]]:
    """``(start, end, group)`` char spans inside ``caption``; later entries win."""
    spans = []
    for key, group in (
        ("meta", "meta"),
        ("characters", "characters"),
        ("series", "series"),
        ("artists", "artist"),
    ):
        for m in re.finditer(rf'"{key}":\[[^\]]*\]', caption):
            spans.append((m.start(), m.end(), group))
    m = re.search(r'"caption":"(?:[^"\\]|\\.)*"', caption)
    if m:
        spans.append((m.start(), m.end(), "tags"))
    i = caption.find('"items":')
    if i >= 0:
        spans.append((i, len(caption), "items"))
    return spans


def token_groups(formatted: str, caption: str, offsets) -> list[str]:
    base = formatted.find(caption)
    spans = [(a + base, b + base, g) for a, b, g in field_spans(caption)]
    out = []
    for a, b in offsets:
        if b <= base or a >= base + len(caption):
            out.append("template")
            continue
        g = "json"
        for s, e, name in spans:
            if a >= s and b <= e:
                g = name
        out.append(g)
    return out


@torch.no_grad()
def encode_all(inst_dir, texts: list[str], last_layer: int):
    """``{text: (embeds (S,4096) bf16, resid (S,4096) fp32, ids, offsets, formatted)}``."""
    from transformers import AutoModel

    tok = loader.load_tokenizer(default_moss_dir())
    te = AutoModel.from_pretrained(inst_dir, dtype=torch.bfloat16).eval()
    loader.offload_by_layer(te, torch.device("cuda"), loader.te_reserve_gb(2048))
    grabbed = {}
    hook = te.language_model.layers[last_layer].register_forward_hook(
        lambda m, i, o: grabbed.__setitem__(
            "x", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()
        )
    )
    out = {}
    for text in dict.fromkeys(texts):
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        enc = tok(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=loader.MAX_SEQUENCE_LENGTH,
            return_offsets_mapping=True,
        )
        assert torch.equal(enc.input_ids, loader.prompt_ids(tok, text).input_ids)
        e = te(
            input_ids=enc.input_ids.cuda(),
            attention_mask=torch.ones_like(enc.input_ids).cuda(),
            use_cache=False,
        )
        out[text] = (
            e.last_hidden_state[0].cpu(),
            grabbed["x"],
            enc.input_ids[0],
            enc.offset_mapping[0].tolist(),
            formatted,
        )
    hook.remove()
    norm_w = te.language_model.norm.weight.detach().float().cpu()
    del te
    torch.cuda.empty_cache()
    return tok, out, norm_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lora", default="output/ckpt_moss/caststation_ema097_lr1e4.safetensors"
    )
    ap.add_argument(
        "--cache", default="post_image_dataset/moss/channel_(caststation)_layout"
    )
    ap.add_argument("--control", default="post_image_dataset/moss/hews_layout")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--n_control", type=int, default=4)
    ap.add_argument("--sigmas", default="0.3,0.7")
    ap.add_argument(
        "--inst_dir", type=Path, default=Path("/media/sorryhyun/new/qwen35_pe/inst")
    )
    ap.add_argument(
        "--sae_dir", type=Path, default=Path("/media/sorryhyun/new/qwen35_sae")
    )
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]
    lora_path = resolve_under_home(args.lora)
    out_path = (
        args.out
        or args.sae_dir
        / "lora_use"
        / f"{lora_path.stem}_{time.strftime('%Y%m%d-%H%M')}"
    )
    out_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    def pick(cache, n):
        items = moss_cache.list_items(resolve_under_home(cache))
        return rng.sample(items, min(n, len(items)))

    main_items, ctrl_items = (
        pick(args.cache, args.n),
        pick(args.control, args.n_control),
    )
    jobs = []
    for kind, items in (("train", main_items), ("control", ctrl_items)):
        for it in items:
            with safe_open(it.te_path, "pt") as g:
                caption = g.metadata()["caption"]
            latent, cached = moss_cache.load_item(it)
            plain = moss_cache.load_plain(it)
            vs = variants(caption) if kind == "train" else {"full": caption}
            jobs.append(
                {
                    "kind": kind,
                    "stem": it.stem,
                    "latent": latent,
                    "cached": cached,
                    "plain": plain,
                    "plain_text": sample_captions(it.te_path).get("plain"),
                    "variants": vs,
                }
            )
    log(
        f"{len(main_items)} training captions, {len(ctrl_items)} control; sigmas {sigmas}"
    )

    tok, enc, norm_w = encode_all(
        args.inst_dir, [t for j in jobs for t in j["variants"].values()], args.layer
    )
    for j in jobs:
        e = enc[j["variants"]["full"]][0].float()
        c = j["cached"].float()
        j["cache_rel_err"] = (
            float((e - c).norm() / c.norm()) if e.shape == c.shape else float("nan")
        )
    log(
        "re-encode vs cache rel err: "
        + " ".join(f"{j['cache_rel_err']:.1e}" for j in jobs)
    )

    device = torch.device("cuda")
    transformer = loader.load_transformer(default_moss_dir(), "base")
    transformer.requires_grad_(False)
    network = load_for_inference(transformer, lora_path, 1.0)
    network.to(device)
    with safe_open(str(lora_path), "pt") as f:
        lmeta = f.metadata() or {}
    delta_rows, trigger = lmeta.get("delta_rows", "all"), lmeta.get("trigger", "")
    log(f"delta_rows={delta_rows} trigger={trigger!r}")

    def rows_for(text, n_text):
        if delta_rows == "all":
            return None
        hits = trigger_rows(tok, text, trigger) if delta_rows == "trigger" else []
        return delta_row_ids(delta_rows, n_text, hits)

    joint = max(
        j["latent"].shape[1] * j["latent"].shape[2]
        + enc[j["variants"]["full"]][0].shape[0]
        for j in jobs
    )
    place(transformer, device, -1, step_reserve_gb(joint, grad_checkpointing=True))
    checkpoint_blocks(transformer.layers)

    def velocity(noisy, sigma, embeds, mult, rows=None):
        network.set_multiplier(mult)
        network.set_row_mask(
            None
            if rows is None
            else row_mask(
                noisy.shape[2] * noisy.shape[3], embeds.shape[0], rows, 0, device
            )
        )
        t = torch.tensor([NUM_TRAIN_TIMESTEPS * sigma], device=device)
        return transformer(
            noisy, t, embeds[None].to(device, torch.bfloat16), return_dict=False
        )[0].float()

    report = {
        "lora": str(lora_path),
        "delta_rows": delta_rows,
        "trigger": trigger,
        "sigmas": sigmas,
        "samples": [],
    }
    for j in jobs:
        latent = j["latent"][None].to(device)
        rec = {
            "kind": j["kind"],
            "stem": j["stem"],
            "cache_rel_err": j["cache_rel_err"],
            "sigma": {},
        }
        texts = dict(j["variants"])
        for si, sigma in enumerate(sigmas):
            gen = torch.Generator(device="cpu").manual_seed(
                zlib.crc32(f"{j['stem']}/{si}".encode())
            )
            noise = torch.randn(latent.shape, generator=gen).to(device)
            noisy = (1.0 - sigma) * latent + sigma * noise
            deltas, base_norm = {}, None
            with torch.no_grad():
                conds = {k: (enc[t][0], t) for k, t in texts.items()}
                if j["plain"] is not None and j["kind"] == "train":
                    conds["plain"] = (j["plain"], j["plain_text"])
                for name, (emb, text) in conds.items():
                    rows = rows_for(text, emb.shape[0])
                    vb = velocity(noisy, sigma, emb, 0.0)
                    deltas[name] = velocity(noisy, sigma, emb, 1.0, rows) - vb
                    if name == "full":
                        base_norm = float(vb.norm())
            full = deltas["full"]
            row = {"delta_over_base": float(full.norm()) / base_norm, "variants": {}}
            for name, d in deltas.items():
                row["variants"][name] = {
                    "norm_ratio": float(d.norm() / full.norm()),
                    "cos": float(
                        torch.nn.functional.cosine_similarity(
                            d.flatten(), full.flatten(), dim=0
                        )
                    ),
                }
            if j["kind"] == "train":
                e = enc[texts["full"]][0].float().to(device).requires_grad_(True)
                rows = rows_for(texts["full"], e.shape[0])
                for mult, sign in ((1.0, 1.0), (0.0, -1.0)):
                    v = velocity(noisy, sigma, e, mult, rows)
                    (sign * (v * full).sum()).backward()
                g = e.grad.detach().cpu()
                rec.setdefault("grads", []).append(g)
            rec["sigma"][str(sigma)] = row
            log(
                f"{j['kind']:7s} {j['stem']} σ={sigma}: ‖Δ‖/‖v‖={row['delta_over_base']:.3f}  "
                + "  ".join(
                    f"{k}={v['norm_ratio']:.2f}/{v['cos']:.2f}"
                    for k, v in row["variants"].items()
                    if k != "full"
                )
            )
        report["samples"].append(rec)
    transformer = network = None
    torch.cuda.empty_cache()

    # token groups + SAE features on the full caption
    sae = {
        k: v.float().cuda()
        for k, v in torch.load(
            args.sae_dir / f"layer{args.layer}.sae.pt",
            map_location="cpu",
            weights_only=True,
        ).items()
    }
    g_gain = (1.0 + norm_w).cuda()
    gd = g_gain[:, None] * sae["W_dec"]
    shares = {g: 0.0 for g in GROUPS}
    tok_rows, feat_abs, feat_sum, feat_ctx = (
        [],
        torch.zeros(gd.shape[1], device="cuda"),
        torch.zeros(gd.shape[1], device="cuda"),
        {},
    )
    for j, rec in zip(jobs, report["samples"]):
        if j["kind"] != "train":
            continue
        caption = j["variants"]["full"]
        e, x, ids, offs, formatted = enc[caption]
        grad = torch.stack(rec.pop("grads")).sum(0)
        gi = (e.float() * grad).sum(-1)
        groups = token_groups(formatted, caption, offs)
        total = float(gi.abs().sum())
        for g, v in zip(groups, gi.abs().tolist()):
            shares[g] += v / total
        rec["group_share"] = {
            g: sum(v for gg, v in zip(groups, gi.abs().tolist()) if gg == g) / total
            for g in GROUPS
        }
        for t in gi.abs().argsort(descending=True)[:12].tolist():
            tok_rows.append(
                (
                    float(gi[t]) / total,
                    j["stem"],
                    groups[t],
                    tok.decode(ids[max(0, t - 6) : t]),
                    tok.decode([int(ids[t])]),
                )
            )
        xc = x.cuda()
        pre = xc @ sae["W_enc"].T + sae["b_enc"]
        vals, idx = pre.topk(100, dim=-1)
        rms = xc.pow(2).mean(-1, keepdim=True).sqrt()
        contrib = (
            vals / rms * torch.einsum("dsk,sd->sk", gd[:, idx], grad.cuda()) / total
        )
        feat_abs.index_add_(0, idx.flatten(), contrib.abs().flatten())
        feat_sum.index_add_(0, idx.flatten(), contrib.flatten())
        best = contrib.abs().flatten().topk(200)
        for flat in best.indices.tolist():
            t, k = divmod(flat, 100)
            f = int(idx[t, k])
            feat_ctx.setdefault(f, []).append(
                f"{float(contrib[t, k]):+.3f} {groups[t]}: …{tok.decode(ids[max(0, t - 6) : t])}⟦{tok.decode([int(ids[t])])}⟧"
            )
    n_train = sum(j["kind"] == "train" for j in jobs)
    report["group_share_mean"] = {g: v / n_train for g, v in shares.items()}
    tok_rows.sort(key=lambda r: -abs(r[0]))
    report["top_tokens"] = [
        {"share": round(s, 4), "stem": st, "group": g, "ctx": c, "token": t}
        for s, st, g, c, t in tok_rows[:40]
    ]

    lm = json.loads((args.inst_dir / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(args.inst_dir / lm["lm_head.weight"], "pt") as h:
        W_U = h.get_tensor("lm_head.weight").cuda()
    feats = []
    for f in feat_abs.argsort(descending=True)[:30].tolist():
        lens = (W_U @ gd[:, f].to(W_U.dtype)).topk(8).indices.tolist()
        feats.append(
            {
                "feature": f,
                "abs_share": float(feat_abs[f]) / n_train,
                "signed": float(feat_sum[f]) / n_train,
                "lens": [tok.decode([t]) for t in lens],
                "contexts": feat_ctx.get(f, [])[:4],
            }
        )
    report["top_features"] = feats
    report["sae_abs_share_total"] = float(feat_abs.sum()) / n_train

    (out_path / "summary.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False)
    )
    log(
        "group share (|grad·input|, mean over captions): "
        + "  ".join(f"{g}={v:.3f}" for g, v in report["group_share_mean"].items())
    )
    for r in report["top_tokens"][:25]:
        log(f"  tok {r['share']:+.4f} {r['group']:10s} …{r['ctx']!r}⟦{r['token']}⟧")
    log(
        f"SAE features carry Σ|c| = {report['sae_abs_share_total']:.3f} of Σ|grad·input|"
    )
    for r in feats[:20]:
        log(
            f"  f{r['feature']:5d} |c|={r['abs_share']:.4f} signed={r['signed']:+.4f} lens={r['lens']}"
        )
        for c in r["contexts"][:2]:
            log(f"        {c}")
    log(f"wrote {out_path}/summary.json")


if __name__ == "__main__":
    main()
