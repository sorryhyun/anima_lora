"""What the MOSS DiT reads of the PE delta: TE-mode vs PE-mode encodes through ``P``.

Each prompt (``moss_te_sae/src/probe.py``'s set: layout captions from five MOSS caches +
the 80 pe_lora NL prompts) is encoded on the catalog INT8 TE twice, TE mode and PE mode
(``pe_mode_int8_r128d256``), with the shipped ``sure`` tail. ``Δ = h_pe − h_te`` is the
change in the DiT's text input; ``P`` is ``text_embedder.proj.weight`` (its bias cancels
in ``Δ``; the two text blocks after it are not modelled). Per row segment (prefix, caption
body, suffix incl. tail, the ``</think>`` row):

  rel_te    ‖Δ‖ / ‖h_te‖
  rel_dit   ‖PΔ‖ / ‖P h_te‖
  read      ‖PΔ‖ / ‖Δ‖, against ‖P h‖ / ‖h‖ and random directions
  top_sub   energy share in P's top-256 right singular vectors (random: 256 / 4096)

and over the pooled body rows: the prompt-constant share ``‖mean Δ‖² / mean ‖Δ‖²``, the
PCA spectrum of Δ and the read gain of its top components, and the tokens whose rows move
most as the DiT sees them.

    make daemon-run ARGS="--stall-timeout 900 project/pe_lora/src/dit_read.py"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from library.downloads import default_moss_dir
from library.env import resolve_under_home
from library.moss import loader
from library.moss.prompt_tail import with_prompt_tail
from pe_mode import PEMode

sys.path.insert(0, str(resolve_under_home("project/moss_te_sae/src")))
from probe import collect_prompts, dit_projection, tokenize  # noqa: E402

SEGS = ("prefix", "body", "suffix", "think_close")


def log(s):
    print(s, flush=True)


@torch.no_grad()
def encode(te, ids):
    ids = ids.cuda()
    return (
        te(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
        .last_hidden_state[0]
        .float()
    )


def seg_masks(seg, ids, close_id):
    m = {
        "prefix": seg == 0,
        "body": seg == 1,
        "suffix": seg == 2,
        "think_close": ids == close_id,
    }
    return {k: v.cuda() for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pe_tensors",
        type=Path,
        default=Path(
            "/media/sorryhyun/new/qwen35_pe/lora/pe_mode_int8_r128d256.safetensors"
        ),
    )
    ap.add_argument("--cache_root", type=Path, default=Path("post_image_dataset/moss"))
    ap.add_argument("--n_per_set", type=int, default=30)
    ap.add_argument("--tail", default="sure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top_sub", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("output/tests/pe_dit_read"))
    args = ap.parse_args()
    out_dir = resolve_under_home(args.out) / time.strftime("%Y%m%d-%H%M")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_prompts(
        resolve_under_home(args.cache_root), args.n_per_set, args.seed
    )
    log(
        f"[prompts] {len(rows)} ({sum(r['source'] == 'layout' for r in rows)} layout, "
        f"{sum(r['source'] == 'nl' for r in rows)} nl)"
    )
    moss_dir = default_moss_dir()
    tok = with_prompt_tail(loader.load_tokenizer(moss_dir), args.tail)
    close_id = tok.convert_tokens_to_ids("</think>")
    toks = [tokenize(tok, r["text"], True) for r in rows]
    longest = max(ids.shape[1] for ids, _ in toks)
    tok, te = loader.load_text_encoder(
        None, precision="int8", max_tokens=longest, log=log, tok=tok
    )
    tensors = load_file(str(args.pe_tensors))
    tensors["lm_head.weight"] = torch.zeros(1, 4096, dtype=torch.bfloat16)
    pm = PEMode(te, tensors)
    del tensors
    log(f"[pe] {len(pm.branches)} branches, {len(pm.swaps)} swaps")

    P = dit_projection(moss_dir)  # (3072, 4096) fp32 cuda
    _, _, Vh = torch.linalg.svd(P, full_matrices=False)
    V_top = Vh[: args.top_sub].T  # (4096, k)

    def stats(x, d):
        """Row-wise sums for one segment: x = h_te rows, d = Δ rows."""
        Px, Pd = x @ P.T, d @ P.T
        return {
            "n": x.shape[0],
            "rel_te": (d.norm(dim=-1) / x.norm(dim=-1)).sum().item(),
            "rel_dit": (Pd.norm(dim=-1) / Px.norm(dim=-1)).sum().item(),
            "read_d": (Pd.norm(dim=-1) / d.norm(dim=-1)).sum().item(),
            "read_h": (Px.norm(dim=-1) / x.norm(dim=-1)).sum().item(),
            "top_d": ((d @ V_top).square().sum(-1) / d.square().sum(-1)).sum().item(),
            "top_h": ((x @ V_top).square().sum(-1) / x.square().sum(-1)).sum().item(),
        }

    acc = {src: {s: {} for s in SEGS} for src in ("layout", "nl")}
    body_d, body_h, body_meta = [], [], []
    roundtrip = 0.0
    for n, (row, (ids, seg)) in enumerate(zip(rows, toks)):
        pm.set_mode("te")
        h_te = encode(te, ids)
        pm.set_mode("pe")
        h_pe = encode(te, ids)
        if n == 0:
            pm.set_mode("te")
            roundtrip = float((encode(te, ids) - h_te).abs().max())
        d = h_pe - h_te
        masks = seg_masks(seg, ids[0], close_id)
        for s, m in masks.items():
            if not m.any():
                continue
            st = stats(h_te[m], d[m])
            a = acc[row["source"]][s]
            for k, v in st.items():
                a[k] = a.get(k, 0) + v
        mb = masks["body"]
        body_d.append(d[mb].cpu())
        body_h.append(h_te[mb].cpu())
        Pd = (d[mb] @ P.T).norm(dim=-1)
        Px = (h_te[mb] @ P.T).norm(dim=-1)
        body_ids = ids[0][mb.cpu()]
        for j in range(body_ids.shape[0]):
            body_meta.append((n, int(body_ids[j]), float(Pd[j] / Px[j])))
        if n % 25 == 0:
            log(f"[enc] {n + 1}/{len(rows)}  S={ids.shape[1]}")

    pm.set_mode("te")
    rand = torch.randn(4096, 4096, device="cuda")
    read_rand = ((P @ rand).norm(dim=0) / rand.norm(dim=0)).median().item()

    seg_table = {}
    for src, segs in acc.items():
        seg_table[src] = {}
        for s, a in segs.items():
            if not a:
                continue
            n_rows = a.pop("n")
            seg_table[src][s] = {"rows": n_rows} | {
                k: round(v / n_rows, 4) for k, v in a.items()
            }

    D = torch.cat(body_d).cuda()
    H = torch.cat(body_h).cuda()
    mean_d = D.mean(0)
    const_share = float(mean_d.square().sum() / D.square().sum(-1).mean())
    Dc = D - mean_d
    evals, evecs = torch.linalg.eigh(Dc.T @ Dc / Dc.shape[0])
    evals, evecs = evals.flip(0), evecs.flip(1)
    share = evals / evals.sum()
    pcs = evecs[:, :16]
    read_pc = ((P @ pcs).norm(dim=0)).tolist()
    pooled = {
        "body_rows": D.shape[0],
        "const_share": round(const_share, 4),
        "mean_d_read": round(float((P @ mean_d).norm() / mean_d.norm()), 3),
        "mean_d_cos_mean_h": round(
            float(torch.nn.functional.cosine_similarity(mean_d, H.mean(0), dim=0)), 4
        ),
        "pca_share_top": {
            k: round(float(share[:k].sum()), 4) for k in (1, 4, 16, 64, 256)
        },
        "pc_read": [round(x, 3) for x in read_pc],
        "pc_top_sub": [
            round(float((pcs[:, i] @ V_top).square().sum()), 4) for i in range(16)
        ],
        "read_rand_median": round(read_rand, 3),
        "top_sub_rand": round(args.top_sub / 4096, 4),
    }
    body_meta.sort(key=lambda t: -t[2])
    top_tokens = [
        {"prompt": n, "token": tok.decode([t]), "rel_dit": round(r, 3)}
        for n, t, r in body_meta[:60]
    ]
    result = {
        "pe_tensors": str(args.pe_tensors),
        "tail": args.tail,
        "n_prompts": len(rows),
        "te_roundtrip_maxabs": roundtrip,
        "segments": seg_table,
        "pooled_body": pooled,
        "top_tokens": top_tokens,
        "prompts": [
            {"source": r["source"], "set": r["set"], "text": r["text"]} for r in rows
        ],
    }
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False)
    )
    log(json.dumps({"segments": seg_table, "pooled_body": pooled}, indent=1))
    log(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
