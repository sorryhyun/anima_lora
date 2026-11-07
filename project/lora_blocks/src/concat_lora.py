"""Stack two MOSS LoRAs exactly as one file: the rank-r adapters of both concatenated to rank
2r (downs stacked, ups side by side), alpha doubled so alpha / rank, and each update, is
unchanged. Keys only in the first file (the mod_rank modulation adapter) pass through.
Metadata is the first file's with rank / alpha replaced, so its stamps (token_split,
ref flags) drive generate.

    .venv/bin/python project/lora_blocks/src/concat_lora.py A.safetensors B.safetensors OUT.safetensors
"""

import argparse

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load(path):
    with safe_open(path, "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys()}, dict(f.metadata() or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    args = ap.parse_args()
    sa, ma = load(args.a)
    sb, mb = load(args.b)
    for k in ("rank", "alpha", "targets"):
        if ma.get(k) != mb.get(k):
            raise SystemExit(f"{k} differs: {ma.get(k)} vs {mb.get(k)}")
    extra = set(sb) - set(sa)
    if extra:
        raise SystemExit(f"keys only in {args.b}: {sorted(extra)[:4]}")
    rank, alpha = int(ma["rank"]), float(ma["alpha"])
    out = {}
    for k, v in sa.items():
        if k not in sb:
            out[k] = v
        elif k.endswith("lora_down.weight"):
            out[k] = torch.cat([v, sb[k]], 0).contiguous()
        elif k.endswith("lora_up.weight"):
            out[k] = torch.cat([v, sb[k]], 1).contiguous()
        elif k.endswith(".alpha"):
            out[k] = torch.tensor(2 * alpha)
        else:
            raise SystemExit(f"unhandled key {k}")
    meta = dict(ma, rank=str(2 * rank), alpha=str(2 * alpha))
    meta["concat_of"] = f"{args.a} + {args.b}"
    save_file(out, args.out, metadata=meta)
    print(f"{len(out)} tensors, rank {2 * rank}, alpha {2 * alpha:g} -> {args.out}")


if __name__ == "__main__":
    main()
