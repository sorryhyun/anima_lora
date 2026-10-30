"""Trigger-stripped caption pairs for ``preserve_weight`` (``library/moss/train.py``).

For every TE file of a layout-JSON cache, writes ``{stem}.moss_te_neg.safetensors``:

    embeds_pos / embeds              (S, 4096) / (S', 4096) bf16: the full caption and
                                     the same caption with the trigger removed from
                                     ``summary.artists``
    embeds_plain_pos / embeds_plain  the plain Anima caption and it without the
                                     trigger tag (``position_clauses``, never a split)
    metadata: trigger, caption_sha256 (the TE file's, the trainer's staleness
              check), caption_neg, plain_neg, encoder

Both sides of each pair come from one encode, so their difference is the trigger
alone and not the cache's encoder vs this one. The precision defaults to the cache's
``encoder`` stamp; the relative error of ``embeds_pos`` against the cached ``embeds``
is printed per precision as a check on that stamp.

    make daemon-run ARGS="--stall-timeout 900 project/moss_te_sae/src/trigger_neg_cache.py \\
        --cache 'post_image_dataset/moss/channel_(caststation)_layout' \\
        --trigger '@channel (caststation)'"
"""

from __future__ import annotations

import argparse
import json
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from library.env import resolve_under_home
from library.moss import cache as moss_cache
from library.moss.loader import encode_prompts
from library.moss.train import NEG_SUFFIX, sample_captions


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def strip_layout(caption: str, trigger: str) -> str:
    obj = json.loads(caption)
    artists = obj.get("summary", {}).get("artists", [])
    if trigger not in artists:
        raise ValueError(f"{trigger!r} is not among the artists {artists}")
    artists = [a for a in artists if a != trigger]
    if artists:
        obj["summary"]["artists"] = artists
    else:
        obj["summary"].pop("artists")
    return dumps(obj)


def strip_plain(caption: str, trigger: str) -> str:
    from anime_tools.captions.position_clauses import compose_caption, parse_caption

    parsed = parse_caption(caption)
    tags = [t for t in parsed.flat_tags if t != trigger]
    if len(tags) == len(parsed.flat_tags):
        raise ValueError(f"{trigger!r} is not a tag of {caption[:60]!r}")
    return compose_caption(tags, parsed.clauses)


def stamped_precision(encoder: str) -> str:
    m = re.search(r", (\w+) weights$", encoder)
    return m.group(1) if m else "bf16"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument(
        "--precision", default="", help="bf16 / fp8 / int8; '' = the cache's stamp"
    )
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cache_dir = resolve_under_home(args.cache)
    items = moss_cache.list_items(cache_dir)
    if not items:
        raise SystemExit(f"no cached pairs under {cache_dir}")
    with safe_open(str(items[0].te_path), "pt") as f:
        encoder = (f.metadata() or {}).get("encoder", "")
    precision = args.precision or stamped_precision(encoder)

    jobs = []
    for item in items:
        out = item.te_path.with_name(item.stem + NEG_SUFFIX)
        if out.exists() and not args.overwrite:
            continue
        caps = sample_captions(item.te_path)
        pair = {"full": (caps["full"], strip_layout(caps["full"], args.trigger))}
        if "plain" in caps:
            pair["plain"] = (caps["plain"], strip_plain(caps["plain"], args.trigger))
        jobs.append((item, out, pair))
    print(f"{len(jobs)}/{len(items)} TE files to pair, TE {precision}", flush=True)
    if not jobs:
        return

    texts = [t for _, _, pair in jobs for both in pair.values() for t in both]
    embeds = encode_prompts(args.model_dir, texts, precision=precision)

    errs = []
    for item, out, pair in jobs:
        with safe_open(str(item.te_path), "pt") as f:
            meta = f.metadata() or {}
            cached = f.get_tensor("embeds").float()
        pos = embeds[pair["full"][0]].float()
        errs.append(((pos - cached).norm(dim=1) / cached.norm(dim=1)).mean().item())
        tensors = {
            "embeds_pos": embeds[pair["full"][0]].to(torch.bfloat16).contiguous(),
            "embeds": embeds[pair["full"][1]].to(torch.bfloat16).contiguous(),
        }
        out_meta = {
            "trigger": args.trigger,
            "caption_sha256": meta["caption_sha256"],
            "caption_neg": pair["full"][1],
            "encoder": moss_cache.encoder_id(precision),
        }
        if "plain" in pair:
            tensors["embeds_plain_pos"] = (
                embeds[pair["plain"][0]].to(torch.bfloat16).contiguous()
            )
            tensors["embeds_plain"] = (
                embeds[pair["plain"][1]].to(torch.bfloat16).contiguous()
            )
            out_meta["plain_neg"] = pair["plain"][1]
        save_file(tensors, str(out), metadata=out_meta)
        print(
            f"  {out.name}: full {tensors['embeds_pos'].shape[0]} -> {tensors['embeds'].shape[0]}"
            + (
                f", plain {tensors['embeds_plain_pos'].shape[0]} -> {tensors['embeds_plain'].shape[0]}"
                if "plain" in pair
                else ""
            ),
            flush=True,
        )
    print(
        f"embeds_pos vs cached embeds ({precision}): mean per-token rel err "
        f"{sum(errs) / len(errs):.4f} (min {min(errs):.4f}, max {max(errs):.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
