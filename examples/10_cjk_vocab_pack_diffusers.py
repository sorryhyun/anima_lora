#!/usr/bin/env python3
"""The CJK vocab pack on the diffusers Anima pipeline (``ModularPipeline``, diffusers ≥ 0.39).

diffusers ships Anima as a *modular* pipeline
(``circlestone-labs/Anima-Base-v1.0-Diffusers``: Qwen3 text encoder + the
``AnimaTextConditioner`` that turns T5 token ids into the DiT's cross-attn
context + ``CosmosTransformer3DModel`` + the Qwen-Image VAE). The vocab pack
(https://huggingface.co/sorryhyun/anima-vocab-pack-cjk) is **not a LoRA** —
it is a table of extra text-embedding rows (ids ≥ 32128) plus a JSON sidecar
with the segmentation / row maps — and applying it to that pipeline means
patching the same two places ``09_cjk_vocab_pack.py`` patches in this repo:

1. **Tokenizer** — the ``text_encoder`` block's T5-side id stream comes from
   :class:`HybridT5Encoder` instead of ``pipe.t5_tokenizer`` (CJK and, in packs
   with a ``route`` block, the symbol tail T5 cannot spell land on pack rows).
   Done by swapping the block for a subclass that overrides one static method;
   prompts with no routed character take the stock path and return the same
   ids the stock block does.
2. **Embedding table** — the pack rows are appended to
   ``pipe.text_conditioner.embed`` once after ``load_components``.

The pipeline's own tokenizers are reused (its Qwen tokenizer is fast and
vocabulary-identical to the pack's bundled ``tokenizer_qwen3/``; its T5
tokenizer is the stock Anima one). LoRAs loaded through
``pipe.load_lora_weights`` compose with the pack — disjoint parameters.

    python examples/10_cjk_vocab_pack_diffusers.py --prompt "1girl, 猫耳, 銀髪, セーラー服, 笑顔, 教室"
    # a local pack: path prefix of the .safetensors/.json pair
    python examples/10_cjk_vocab_pack_diffusers.py --pack output/ckpt/cjk_vocab_pack_synthjakozh1sym_r256 …
    # CPU check of both patch points without the transformer / VAE (~1.5 GB download)
    python examples/10_cjk_vocab_pack_diffusers.py --dry_run

Only ``library.anima.ext_vocab`` (pure Python + torch, installed with this
repo) is used from the repo; nothing else from the ``anima_lora`` engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers.modular_pipelines.anima import AnimaAutoBlocks
from diffusers.modular_pipelines.anima.encoders import AnimaTextEncoderStep

from library.anima.ext_vocab import T5_TABLE_SIZE, HybridT5Encoder, load_ext_assets

PIPE_REPO = "circlestone-labs/Anima-Base-v1.0-Diffusers"
PACK_REPO = "sorryhyun/anima-vocab-pack-cjk"
PACK_STEM = "anima_cjk_vocab_pack"


def fetch_pack(local_prefix: str | None) -> Path:
    if local_prefix:
        return Path(local_prefix)
    from huggingface_hub import hf_hub_download

    st = hf_hub_download(PACK_REPO, f"{PACK_STEM}.safetensors")
    hf_hub_download(PACK_REPO, f"{PACK_STEM}.json")
    return Path(st).with_suffix("")


# --- patch point 1: the text-encoder block ----------------------------------
def make_pack_text_encoder_step(mapping: dict) -> type[AnimaTextEncoderStep]:
    """A ``text_encoder`` block whose T5 ids go through the pack.

    The stock block computes the T5 stream in one static method, so the
    subclass overrides just that; the Qwen side (the actual text encoder) is
    untouched. The hybrid encoder is built lazily from the pipeline's own
    tokenizers on first use, so the block needs nothing but the pack json.
    """

    class PackTextEncoderStep(AnimaTextEncoderStep):
        _mapping = mapping
        _encoder: HybridT5Encoder | None = None

        @classmethod
        def encoder(cls, components) -> HybridT5Encoder:
            if cls._encoder is None:
                cls._encoder = HybridT5Encoder.from_mapping(
                    components.t5_tokenizer, components.tokenizer, cls._mapping
                )
            return cls._encoder

        @classmethod
        def _get_t5_prompt_ids(cls, components, prompt, max_sequence_length, device):
            prompt = [prompt] if isinstance(prompt, str) else prompt
            enc = cls.encoder(components)
            rows = []
            for text in prompt:
                if enc.routes(text):
                    ids, mask = enc.encode(text, max_sequence_length)
                    n = sum(mask)
                    rows.append(ids[:n])  # eos-terminated; re-padded below
                else:  # stock path — identical to the parent block's ids
                    rows.append(
                        components.t5_tokenizer(
                            text, max_length=max_sequence_length, truncation=True
                        )["input_ids"]
                    )
            # The stock block pads "longest" (right, pad id 0); match it.
            longest = max(len(r) for r in rows)
            pad = components.t5_tokenizer.pad_token_id
            ids = torch.tensor([r + [pad] * (longest - len(r)) for r in rows])
            mask = torch.tensor([[1] * len(r) + [0] * (longest - len(r)) for r in rows])
            return ids.to(device), mask.to(device)

    return PackTextEncoderStep


# --- patch point 2: the conditioner's id table ------------------------------
def extend_conditioner_table(text_conditioner, table: torch.Tensor) -> None:
    emb = text_conditioner.embed
    if emb.weight.shape[0] != T5_TABLE_SIZE:
        raise RuntimeError(
            f"text_conditioner.embed has {emb.weight.shape[0]} rows, expected "
            f"{T5_TABLE_SIZE} — a pack is already applied?"
        )
    new_w = torch.cat(
        [emb.weight.data, table.to(emb.weight.dtype).to(emb.weight.device)]
    )
    text_conditioner.embed = torch.nn.Embedding.from_pretrained(new_w)


def build_pipeline(mapping: dict):
    """AnimaAutoBlocks with the text-encoder block swapped, as a pipeline."""
    blocks = AnimaAutoBlocks()
    blocks.sub_blocks["text_encoder"] = make_pack_text_encoder_step(mapping)()
    return blocks.init_pipeline(PIPE_REPO)


def report(pipe, prompt: str) -> None:
    step = pipe.blocks.sub_blocks["text_encoder"]
    out = step.encode_prompt(
        components=pipe,
        prompt=prompt,
        prepare_unconditional_embeds=False,
        device=torch.device("cpu"),
    )
    ids, mask = out["t5_input_ids"][0], out["t5_attention_mask"][0]
    live = ids[: int(mask.sum())]
    n_ext = int((live >= T5_TABLE_SIZE).sum())
    n_unk = int((live == pipe.t5_tokenizer.unk_token_id).sum())
    stock = pipe.t5_tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
    print(
        f"{prompt!r}\n  routed={step.encoder(pipe).routes(prompt)}  tokens={len(live)}"
        f"  pack_rows={n_ext}  unk={n_unk}  identical_to_stock={live.tolist() == stock}"
    )
    vec = pipe.text_conditioner.embed(
        live
    )  # would index out of range on the stock table
    print(f"  conditioner lookup {tuple(vec.shape)} ok")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--prompt", default="1girl, 猫耳, 銀髪, セーラー服, 笑顔, 教室, 上半身"
    )
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--pack", help="local path prefix of the .safetensors/.json pair")
    p.add_argument("--save_path", default="output/tests/example_10.png")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument(
        "--size", type=int, nargs=2, default=[1024, 1024], metavar=("H", "W")
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="CPU: load only the text side, show both patch points, no image",
    )
    opts = p.parse_args()

    prefix = fetch_pack(opts.pack)
    table, mapping = load_ext_assets(prefix)
    print(f"pack {prefix.name}: {table.shape[0]} rows, route={'route' in mapping}")

    pipe = build_pipeline(mapping)
    if opts.dry_run:
        pipe.load_components(
            names=["tokenizer", "t5_tokenizer", "text_encoder", "text_conditioner"],
            torch_dtype=torch.float32,
        )
        extend_conditioner_table(pipe.text_conditioner, table)
        for prompt in [
            "1girl, cat ears, silver hair, serafuku, smile, classroom",
            opts.prompt,
            "1girl, 고양이귀, 은발, 세일러복, 미소",
            "1girl, 猫耳, 银发, 水手服, 微笑, ☆",
        ]:
            report(pipe, prompt)
        return

    pipe.load_components(torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    extend_conditioner_table(pipe.text_conditioner, table)  # once, after loading
    pipe.guider.guidance_scale = opts.cfg

    image = pipe(
        prompt=opts.prompt,
        negative_prompt=opts.negative_prompt,
        height=opts.size[0],
        width=opts.size[1],
        num_inference_steps=opts.steps,
        generator=torch.Generator("cuda").manual_seed(opts.seed),
    ).images[0]
    Path(opts.save_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(opts.save_path)
    print(f"saved → {opts.save_path}")


if __name__ == "__main__":
    main()
