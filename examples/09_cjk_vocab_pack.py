#!/usr/bin/env python3
"""Prompt in Japanese / Korean / Chinese through the Anima CJK vocab pack.

The pack (https://huggingface.co/sorryhyun/anima-vocab-pack-cjk) is **not a
LoRA**: it is a table of extra text-embedding rows (``ext_embed [rows, 1024]``,
ids ≥ 32128) plus a JSON sidecar with the segmentation / row maps. Applying it
means patching two places — the T5-side tokenizer and the LLM adapter's
embedding table — and ``library.anima.vocab_pack`` owns both, so on the front
door it is one field:

    GenerationRequest(prompt="1girl, 猫耳, 銀髪", vocab_pack="models/vocab_packs/anima_cjk_vocab_pack")

``generate()`` installs the pack-routing tokenize strategy and ``load_dit_model``
hooks the rows onto ``llm_adapter.embed`` (the module's state dict stays at the
stock 32128 rows, so the pack composes with any checkpoint or DiT LoRA). Leave
``vocab_pack`` unset to follow the ``vocab_pack`` key in ``configs/base.toml``
(``make download-vocab-pack`` fetches the shipped pack; ``""`` = off);
``no_vocab_pack=True`` forces the stock tokenizer. Prompts with no routed
character are bit-identical with or without the pack.

Run from the repo root after ``make download-models`` and ``make
download-vocab-pack``:

    python examples/09_cjk_vocab_pack.py --prompt "1girl, 猫耳, 銀髪, セーラー服, 笑顔, 教室"
    python examples/09_cjk_vocab_pack.py --prompt "1girl, 고양이귀, 은발, 세일러복, 미소"
    python examples/09_cjk_vocab_pack.py --prompt "1girl, 猫耳, 银发, 水手服, 微笑"
    # a local pack build: path prefix of the .safetensors/.json pair
    python examples/09_cjk_vocab_pack.py --pack output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256 --prompt …
    # tokenizer-only dry run (no DiT / VAE): prints the routed id stream
    python examples/09_cjk_vocab_pack.py --dry_run --prompt "1girl, 猫耳, 銀髪"

What to expect: danbooru-style tags in JA work like their English spelling
in same-seed grids; KO / ZH tag rows are trained but were not grid-validated
the same way; full-CJK rare-kanji character names do not compose (type
names in latin, ``hakurei reimu``). See ``docs/methods/cjk_vocab_pack.md``.
"""

from __future__ import annotations

import argparse

import torch

from anima_lora import (
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_vae,
    load_vocab_pack,
    save_output,
)
from library.anima.ext_vocab import T5_TABLE_SIZE
from library.anima.vocab_pack import DEFAULT_PACK_PREFIX, VocabPackTokenizeStrategy
from library.inference.text import MAX_CROSSATTN_TOKENS
from library.runtime.device import clean_memory_on_device

_ckpt = default_checkpoints()
DIT = _ckpt.dit
VAE = _ckpt.vae
TEXT_ENCODER = _ckpt.text_encoder


def report_routing(pack, prompt: str) -> None:
    """Show what the pack does to the T5 stream (ext rows vs stock vs <unk>)."""
    tok = VocabPackTokenizeStrategy(
        pack,
        qwen3_path=TEXT_ENCODER,
        qwen3_max_length=MAX_CROSSATTN_TOKENS,
        t5_max_length=MAX_CROSSATTN_TOKENS,
    )
    _, _, t5_ids, t5_mask = tok.tokenize(prompt)
    n = int(t5_mask[0].sum())
    live = t5_ids[0, :n]
    n_ext = int((live >= T5_TABLE_SIZE).sum())
    n_unk = int((live == tok.t5_tokenizer.unk_token_id).sum())
    print(
        f"T5 stream: {n} tokens, {n_ext} on pack rows, {n_unk} <unk>"
        f"  (routed={tok.encoder.routes(prompt)})"
    )


def build_request(opts: argparse.Namespace, pack_prefix: str) -> GenerationRequest:
    return GenerationRequest(
        dit=DIT,
        vae=VAE,
        text_encoder=TEXT_ENCODER,
        vocab_pack=pack_prefix,
        prompt=opts.prompt,
        save_path=opts.save_path,
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=tuple(opts.size),
        seed=opts.seed,
        lora_weight=opts.lora_weight,
        lora_multiplier=opts.multiplier,
    )


def generate_image(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    gen_settings = get_generation_settings(args)

    # generate() loads the DiT (hooking the pack rows onto llm_adapter.embed)
    # and installs the pack-routing tokenizer from args.vocab_pack — the same
    # path `inference.py --vocab_pack …` and `make test` take.
    latent = generate(args, gen_settings)

    clean_memory_on_device(device)
    vae = load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
        dtype=torch.bfloat16,
        eval=True,
    )
    save_output(args, vae, latent, device)
    print(f"saved → {args.save_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompt", default="1girl, 猫耳, 銀髪, セーラー服, 笑顔, 教室, 上半身"
    )
    p.add_argument(
        "--pack",
        default=_ckpt.vocab_pack or DEFAULT_PACK_PREFIX,
        help="path prefix of the .safetensors/.json pair (default: base.toml "
        "`vocab_pack`, else the `make download-vocab-pack` location)",
    )
    p.add_argument("--save_path", default="output/tests/example_09.png")
    p.add_argument("--lora_weight", nargs="+", default=[])
    p.add_argument("--multiplier", type=float, nargs="+", default=[1.0])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry_run", action="store_true", help="tokenize + report only, no weights"
    )
    opts = p.parse_args()
    if len(opts.multiplier) == 1 and len(opts.lora_weight) > 1:
        opts.multiplier = opts.multiplier * len(opts.lora_weight)

    pack = load_vocab_pack(opts.pack)  # FileNotFoundError names the download target
    print(f"pack {pack.name}: {pack.rows} rows, route={'route' in pack.mapping}")

    report_routing(pack, opts.prompt)
    if opts.dry_run:
        return
    generate_image(build_request(opts, str(pack.prefix)).to_args())


if __name__ == "__main__":
    main()
