#!/usr/bin/env python3
"""Prompt in Japanese / Korean / Chinese through the Anima CJK vocab pack.

The pack (https://huggingface.co/sorryhyun/anima-vocab-pack-cjk) is **not a
LoRA**: it is a table of extra text-embedding rows (``ext_embed [rows, 1024]``,
ids ≥ 32128) plus a JSON sidecar with the segmentation / row maps. Applying it
means patching two places, and this script shows both on the ``anima_lora``
front door:

1. **Tokenizer** — the T5-side id stream that feeds the DiT's ``llm_adapter``
   is produced by :class:`HybridT5Encoder` instead of the stock T5 tokenizer,
   so CJK (and, in packs that carry a ``route`` block, the symbol tail T5
   cannot spell) lands on pack rows instead of collapsing into ``<unk>``.
   Prompts with no routed character take the stock path untouched —
   pure-English prompts are bit-identical with or without the pack.
2. **Embedding table** — the pack rows are appended to
   ``anima.llm_adapter.embed`` after the DiT loads, so those ids resolve.

Everything else (Qwen3 text encoder, sampler, VAE) is the ordinary
``01_generate.py`` flow; the pack composes with any checkpoint or DiT LoRA
(disjoint parameters).

Run from the repo root after ``make download-models`` (the pack itself is
fetched from the Hub on first use, ~285 MB):

    python examples/09_cjk_vocab_pack.py --prompt "1girl, 猫耳, 銀髪, セーラー服, 笑顔, 教室"
    python examples/09_cjk_vocab_pack.py --prompt "1girl, 고양이귀, 은발, 세일러복, 미소"
    python examples/09_cjk_vocab_pack.py --prompt "1girl, 猫耳, 银发, 水手服, 微笑"
    # a local pack: path prefix of the .safetensors/.json pair
    python examples/09_cjk_vocab_pack.py --pack output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256 --prompt …
    # tokenizer-only dry run (no weights): prints the routed id stream
    python examples/09_cjk_vocab_pack.py --dry_run --prompt "1girl, 猫耳, 銀髪"

What to expect: danbooru-style tags in JA work like their English spelling
in same-seed grids; KO / ZH tag rows are trained but were not grid-validated
the same way; full-CJK rare-kanji character names do not compose (type
names in latin, ``hakurei reimu``). See the pack's model card.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from anima_lora import (
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_dit_model,
    load_vae,
    save_output,
)
from library.anima import strategy as strategy_anima
from library.anima import text_strategies
from library.anima.ext_vocab import T5_TABLE_SIZE, HybridT5Encoder, load_ext_assets
from library.inference.text import MAX_CROSSATTN_TOKENS
from library.runtime.device import clean_memory_on_device

PACK_REPO = "sorryhyun/anima-vocab-pack-cjk"
PACK_STEM = "anima_cjk_vocab_pack"

_ckpt = default_checkpoints()
DIT = _ckpt.dit
VAE = _ckpt.vae
TEXT_ENCODER = _ckpt.text_encoder


def fetch_pack(local_prefix: str | None) -> Path:
    """Path prefix of the ``.safetensors`` + ``.json`` pair (same stem).

    Without ``--pack`` both files come from the Hub cache (public repo, no
    token needed). The pack also ships ``tokenizer_qwen3/`` for pipelines that
    have no Qwen3 tokenizer of their own — this script reuses the text
    encoder's, which is the same vocabulary.
    """
    if local_prefix:
        return Path(local_prefix)
    from huggingface_hub import hf_hub_download

    st = hf_hub_download(PACK_REPO, f"{PACK_STEM}.safetensors")
    hf_hub_download(PACK_REPO, f"{PACK_STEM}.json")
    return Path(st).with_suffix("")


class VocabPackTokenizeStrategy(strategy_anima.AnimaTokenizeStrategy):
    """Patch point 1: the stock dual tokenizer with the T5 stream re-routed.

    ``super().tokenize`` still produces the Qwen3 ids (the text encoder side is
    untouched by the pack) and the stock T5 ids; rows whose prompt carries a
    routed character get their T5 ids / mask replaced by the hybrid encoding
    (eos-terminated, padded to ``t5_max_length`` — the same max-padding the
    pretrained model expects).
    """

    def __init__(self, mapping: dict, **kwargs) -> None:
        super().__init__(**kwargs)
        self.encoder = HybridT5Encoder.from_mapping(
            self.t5_tokenizer, self.qwen3_tokenizer, mapping
        )

    def tokenize(self, text):
        texts = [text] if isinstance(text, str) else list(text)
        q_ids, q_mask, t5_ids, t5_mask = super().tokenize(texts)
        for i, t in enumerate(texts):
            if self.encoder.routes(t):
                ids, mask = self.encoder.encode(t, self.t5_max_length)
                t5_ids[i] = torch.tensor(ids, dtype=t5_ids.dtype)
                t5_mask[i] = torch.tensor(mask, dtype=t5_mask.dtype)
        return [q_ids, q_mask, t5_ids, t5_mask]


def install_pack_strategies(
    text_encoder_path: str, mapping: dict
) -> VocabPackTokenizeStrategy:
    """Install the process-global strategy singletons with the pack in place.

    Must run **before** ``generate()`` / ``prepare_text_inputs()``: those call
    ``ensure_text_strategies``, which is a no-op once both singletons exist,
    so whatever is installed first wins.
    """
    tok = VocabPackTokenizeStrategy(
        mapping,
        qwen3_path=text_encoder_path,
        qwen3_max_length=MAX_CROSSATTN_TOKENS,
        t5_max_length=MAX_CROSSATTN_TOKENS,
    )
    text_strategies.TokenizeStrategy.set_strategy(tok)
    text_strategies.TextEncodingStrategy.set_strategy(
        strategy_anima.AnimaTextEncodingStrategy()
    )
    return tok


def extend_adapter_table(anima, table: torch.Tensor) -> None:
    """Patch point 2: append the pack rows to the adapter's frozen id table."""
    emb = anima.llm_adapter.embed
    if emb.weight.shape[0] != T5_TABLE_SIZE:
        raise RuntimeError(
            f"llm_adapter.embed has {emb.weight.shape[0]} rows, expected "
            f"{T5_TABLE_SIZE} — a pack is already applied?"
        )
    new_w = torch.cat(
        [emb.weight.data, table.to(emb.weight.dtype).to(emb.weight.device)]
    )
    anima.llm_adapter.embed = torch.nn.Embedding.from_pretrained(new_w)


def report_routing(tok: VocabPackTokenizeStrategy, prompt: str) -> None:
    """Show what the pack did to the T5 stream (ext rows vs stock vs <unk>)."""
    t5_ids, t5_mask = tok.tokenize(prompt)[2][0], tok.tokenize(prompt)[3][0]
    n = int(t5_mask.sum())
    live = t5_ids[:n]
    n_ext = int((live >= T5_TABLE_SIZE).sum())
    n_unk = int((live == tok.t5_tokenizer.unk_token_id).sum())
    print(
        f"T5 stream: {n} tokens, {n_ext} on pack rows, {n_unk} <unk>"
        f"  (routed={tok.encoder.routes(prompt)})"
    )


def build_request(opts: argparse.Namespace) -> GenerationRequest:
    return GenerationRequest(
        dit=DIT,
        vae=VAE,
        text_encoder=TEXT_ENCODER,
        prompt=opts.prompt,
        save_path=opts.save_path,
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=tuple(opts.size),
        seed=opts.seed,
        lora_weight=opts.lora_weight,
        lora_multiplier=opts.multiplier,
    )


def generate_image(args: argparse.Namespace, table: torch.Tensor) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    gen_settings = get_generation_settings(args)

    # Load the DiT ourselves (adapters from args.lora_weight attach here, as in
    # 01) so the table can be extended before generate() encodes the prompt;
    # handing it over via shared_models["model"] skips generate()'s own load.
    anima = load_dit_model(args, device, torch.bfloat16)
    extend_adapter_table(anima, table)
    latent = generate(args, gen_settings, shared_models={"model": anima})

    del anima
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
    p.add_argument("--pack", help="local path prefix of the .safetensors/.json pair")
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

    prefix = fetch_pack(opts.pack)
    table, mapping = load_ext_assets(prefix)
    print(f"pack {prefix.name}: {table.shape[0]} rows, route={'route' in mapping}")

    tok = install_pack_strategies(TEXT_ENCODER, mapping)
    report_routing(tok, opts.prompt)
    if opts.dry_run:
        return
    generate_image(build_request(opts).to_args(), table)


if __name__ == "__main__":
    main()
