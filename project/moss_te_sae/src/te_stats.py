"""Spectrum statistics of the text conditioning: shipped instruct TE (INT8) vs Qwen3.5-9B Base.

The probe's caption set (``collect_prompts``: layout-JSON caches + NL prompts) in the
shipped chat template, each prompt alone at its true length. Three stages: ``out``
(``last_hidden_state``), ``proj`` (``text_embedder.proj``), ``emb`` (the whole
``text_embedder``: proj + 2 refiner blocks, what the DiT blocks read). Rows split into
caption body and template.

Per TE × stage × segment: mean row norm, effective rank (exp of the singular-value
entropy) and participation ratio of the centred rows, top-1 / top-10 PC variance share,
mean cosine of random row pairs (raw and centred), the loudest channel's share of the
squared norm; per-prompt effective rank of the body rows (token diversity within a
caption). Across TEs: linear CKA on the body rows and the per-token relative change.

    make daemon-run ARGS="--stall-timeout 900 project/moss_te_sae/src/te_stats.py"
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModel

from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader
from probe import collect_prompts, tokenize

BASE_DIR = "/media/sorryhyun/data/qwen35_base/Qwen3.5-9B-Base"


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


@torch.no_grad()
def encode_all(te_name, tok, rows, device, base_dir):
    longest = max(loader.prompt_ids(tok, r["text"]).input_ids.shape[1] for r in rows)
    if te_name == "instruct":
        _, te = loader.load_text_encoder(
            default_moss_dir(), device, longest, tok=tok, precision="int8"
        )
    else:
        te = AutoModel.from_pretrained(base_dir, dtype=torch.bfloat16).eval()
        loader.offload_by_layer(te, device, loader.te_reserve_gb(longest))
    out = []
    for n, r in enumerate(rows):
        out.append(loader.encode_prompt(tok, te, r["text"], device).cpu())
        if n % 50 == 0:
            log(f"[{te_name}] {n + 1}/{len(rows)}")
    del te
    loader.empty_cache()
    return out


def spectrum(X: torch.Tensor) -> dict:
    """``X (N, D)`` fp32 on the GPU."""
    norms = X.norm(dim=1)
    Xc = X - X.mean(0)
    lam = torch.linalg.eigvalsh(Xc.T.double() @ Xc.double() / X.shape[0]).clamp_min(0)
    lam = lam.flip(0)
    s = lam.sqrt()
    p = s / s.sum()
    erank = torch.exp(-(p[p > 0] * p[p > 0].log()).sum()).item()
    pr = (lam.sum() ** 2 / (lam**2).sum()).item()
    g = torch.Generator(device="cpu").manual_seed(0)
    i = torch.randint(0, X.shape[0], (20000,), generator=g).to(X.device)
    j = torch.randint(0, X.shape[0], (20000,), generator=g).to(X.device)

    def mcos(A):
        return torch.nn.functional.cosine_similarity(A[i], A[j], dim=1).mean().item()

    ch = X.pow(2).mean(0)
    return {
        "rows": X.shape[0],
        "norm": norms.mean().item(),
        "erank": erank,
        "pr": pr,
        "pc1": (lam[0] / lam.sum()).item(),
        "pc10": (lam[:10].sum() / lam.sum()).item(),
        "cos": mcos(X),
        "cos_c": mcos(Xc),
        "loud_ch": int(ch.argmax()),
        "loud_share": (ch.max() / ch.sum()).item(),
    }


def per_prompt_erank(seqs) -> float:
    vals = []
    for X in seqs:
        if X.shape[0] < 8:
            continue
        s = torch.linalg.svdvals(X - X.mean(0))
        p = s / s.sum()
        vals.append(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()).item())
    return sum(vals) / len(vals)


def cka(A, B) -> float:
    A = (A - A.mean(0)).double()
    B = (B - B.mean(0)).double()
    return ((A.T @ B).norm() ** 2 / ((A.T @ A).norm() * (B.T @ B).norm())).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=BASE_DIR)
    ap.add_argument("--n_per_set", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output/tests/te_stats")
    args = ap.parse_args()
    out = resolve_under_home(args.out) / time.strftime("%Y%m%d-%H%M")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    rows = collect_prompts(
        resolve_under_home("post_image_dataset/moss"), args.n_per_set, args.seed
    )
    tok = loader.load_tokenizer(default_moss_dir())
    segs = [tokenize(tok, r["text"], True)[1] == 1 for r in rows]
    tes = ("instruct", "base")
    raw = {t: encode_all(t, tok, rows, device, args.base_dir) for t in tes}

    tr = loader.load_transformer("", "base", device=device)
    te_mod = tr.text_embedder
    stages = {t: {"out": [], "proj": [], "emb": []} for t in tes}
    with torch.no_grad():
        for t in tes:
            for x in raw[t]:
                x = x.to(device)
                rope = tr.rope(64, 64, x.shape[0], device).to(torch.bfloat16)
                rope = rope[:, 64 * 64 : 64 * 64 + x.shape[0]]
                stages[t]["out"].append(x.float().cpu())
                stages[t]["proj"].append(te_mod.proj(x[None])[0].float().cpu())
                stages[t]["emb"].append(te_mod(x[None], rope)[0].float().cpu())
    del tr, te_mod
    loader.empty_cache()

    res = {"prompts": len(rows), "stats": {}, "cross": {}}
    for st in ("out", "proj", "emb"):
        for t in tes:
            for seg_name, want in (("body", True), ("template", False)):
                X = torch.cat(
                    [x[m == want] for x, m in zip(stages[t][st], segs)]
                ).cuda()
                r = spectrum(X)
                if want:
                    r["erank_per_prompt"] = per_prompt_erank(
                        [x[m].cuda() for x, m in zip(stages[t][st], segs)]
                    )
                res["stats"][f"{st}|{t}|{seg_name}"] = r
                log(f"{st:4s} {t:8s} {seg_name:8s} {json.dumps(r)}")
        A = torch.cat([x[m] for x, m in zip(stages["instruct"][st], segs)]).cuda()
        B = torch.cat([x[m] for x, m in zip(stages["base"][st], segs)]).cuda()
        res["cross"][st] = {
            "cka_body": cka(A, B),
            "rel_body": ((B - A).norm(dim=1) / A.norm(dim=1)).mean().item(),
        }
        log(f"{st:4s} cross {json.dumps(res['cross'][st])}")
    (out / "te_stats.json").write_text(json.dumps(res, indent=2))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
