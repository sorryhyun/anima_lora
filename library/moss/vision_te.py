"""Qwen3.5-9B with its vision tower: the MOSS TE call with an image in the user turn.

The pipeline's encode (chat template, add_generation_prompt, enable_thinking,
last_hidden_state, one prompt per forward, no padding) with an image item ahead of
the text, preprocessed with Qwen3.5-9B's own processor settings (patch 16, merge 2,
mean/std 0.5; MOSS ships no processor config). A text-only call keeps the
pipeline's string content, byte-identical to its prompts.
"""

import torch

from library.moss.loader import offload_by_layer, te_reserve_gb
from library.moss.requests import variant_dir


def processor(tok, max_pixels):
    from transformers import (
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    ip = Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        size={"shortest_edge": 65536, "longest_edge": max_pixels},
    )
    return Qwen3VLProcessor(
        image_processor=ip,
        tokenizer=tok,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=tok.chat_template,
    )


def load_te(tok, device, model_dir="", precision="bf16", fp8_path=None):
    """``load_text_encoder`` with the vision tower on the card too (placed before the
    layer budget is taken, so the paging plan accounts for it). ``precision="fp8"``
    / ``"int8"`` reads a quantized file (``fp8_path``, else the catalog's row):
    decoder linears quantized, the vision tower bf16."""
    from transformers import AutoModel

    if precision != "bf16":
        from library.moss.fp8 import load_fp8_text_encoder

        te = load_fp8_text_encoder(fp8_path, precision=precision)
    else:
        te = AutoModel.from_pretrained(
            variant_dir(model_dir, "base") / "text_encoder", dtype=torch.bfloat16
        ).eval()
    te.visual.to(device)
    paged = offload_by_layer(te, device, te_reserve_gb(4096))
    print(
        f"[te] {precision}, vision resident, {paged} decoder layers paged", flush=True
    )
    return te


def te_inputs(tok, image, text, max_pixels, device, tail=""):
    """The TE's model inputs for ``text`` after ``image`` (``None``: text only),
    ``tail`` appended after the generation prompt (``prompt_tail.TAILS``)."""
    content = text
    if image is not None:
        content = [{"type": "image"}] + (
            [{"type": "text", "text": text}] if text else []
        )
    formatted = (
        tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        + tail
    )
    if image is None:
        return tok(formatted, return_tensors="pt").to(device)
    return processor(tok, max_pixels)(
        text=[formatted], images=[image], return_tensors="pt"
    ).to(device)


@torch.no_grad()
def encode(
    tok, te, image, text, max_pixels, device, drop_rows=False, prefix=False, tail=""
):
    """``(last_hidden_state (S, 4096), image token count)``; ``image=None`` is the
    pipeline's text-only encode. ``drop_rows`` removes the image-token rows.
    ``prefix`` appends the row count through ``<|vision_end|>``: the rows every
    text after the same page shares."""
    inputs = te_inputs(tok, image, text, max_pixels, device, tail)
    if image is None:
        return te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0], 0
    hidden = te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]
    is_image = inputs["mm_token_type_ids"][0] == 1
    if drop_rows:
        hidden = hidden[~is_image]
    if not prefix:
        return hidden, int(is_image.sum())
    end = int(is_image.nonzero().max()) + 1
    assert inputs["input_ids"][0, end] == tok.convert_tokens_to_ids("<|vision_end|>")
    return hidden, int(is_image.sum()), end + 1


@torch.no_grad()
def encode_variants(tok, te, image, texts, max_pixels, device, tail=""):
    """``encode(..., prefix=True)`` of each of ``texts`` after the same ``image``, the
    page run once: the rows through ``<|vision_end|>`` are prefilled into a cache
    (attention KV + GDN conv/recurrent state) and each text continues from a copy
    of it, with its M-RoPE positions sliced from its own full-sequence index.
    ``[(last_hidden_state, image token count, prefix)]``, one per text."""
    import copy

    inputs = [te_inputs(tok, image, t, max_pixels, device, tail) for t in texts]
    first = inputs[0]
    is_image = first["mm_token_type_ids"][0] == 1
    end = int(is_image.nonzero().max()) + 1
    assert first["input_ids"][0, end] == tok.convert_tokens_to_ids("<|vision_end|>")
    p = end + 1
    for x in inputs[1:]:
        assert torch.equal(x["input_ids"][0, :p], first["input_ids"][0, :p])

    def positions(x):
        return te.get_rope_index(
            x["input_ids"], x["mm_token_type_ids"], image_grid_thw=x["image_grid_thw"]
        )[0]

    out = te(
        input_ids=first["input_ids"][:, :p],
        pixel_values=first["pixel_values"],
        image_grid_thw=first["image_grid_thw"],
        mm_token_type_ids=first["mm_token_type_ids"][:, :p],
        position_ids=positions(first)[..., :p],
        use_cache=True,
        return_dict=True,
    )
    shared, cache = out.last_hidden_state[0], out.past_key_values
    n_img = int(is_image.sum())
    results = []
    for x in inputs:
        tail = te(
            input_ids=x["input_ids"][:, p:],
            position_ids=positions(x)[..., p:],
            past_key_values=copy.deepcopy(cache),
            use_cache=True,
            return_dict=True,
        ).last_hidden_state[0]
        results.append((torch.cat([shared, tail]), n_img, p))
    return results
