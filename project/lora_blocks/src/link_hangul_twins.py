"""Link edition-twin pairs of swap_text's ksplit cache into the Hangul block's cache, so
the block keeps a share of real manga lettering beside the synthetic lines. Run after the
cache job (hangul_cache.toml); symlinks only, the twins' TE files are already
syllable-split.

    .venv/bin/python project/lora_blocks/src/link_hangul_twins.py [--n 150] [--lang ko|en] [--out <cache>]
"""

import argparse
import os
import random

from library.env import resolve_under_home

SRC = resolve_under_home("post_image_dataset/moss/i2i_swap_text_v3_en400_ko1709_ksplit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang", default="ko", choices=("ko", "en"))
    ap.add_argument("--out", default="post_image_dataset/moss/hangul_block")
    args = ap.parse_args()
    out = resolve_under_home(args.out)
    stems = sorted(
        p.name.split(".moss_ref")[0]
        for p in SRC.glob(f"*_{args.lang}.moss_ref.safetensors")
    )
    random.Random(args.seed).shuffle(stems)
    n = 0
    for stem in stems[: args.n]:
        files = [
            SRC / f"{stem}.moss_ref.safetensors",
            SRC / f"{stem}.moss_te.safetensors",
        ]
        files += list(SRC.glob(f"{stem}_*.moss_latent.safetensors"))
        if len(files) != 3 or not all(f.exists() for f in files):
            continue
        for f in files:
            dst = out / f.name
            if not dst.is_symlink() and not dst.exists():
                os.symlink(f.resolve(), dst)
        n += 1
    print(f"{n} {args.lang} twins linked into {out}")


if __name__ == "__main__":
    main()
