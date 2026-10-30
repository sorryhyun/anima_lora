"""MOSS-Image loading under a 16 GB budget: TE → encode → free → DiT → (DiT off) → VAE.

    text_encoder  Qwen3.5-9B (+ unused vision tower)  16.8 GB  <- larger than the card
    transformer   5B, 42 single-stream blocks         10.2 GB  (9.7 GiB resident, bf16)
    vae           FLUX.2 AutoencoderKLFlux2            0.25 GB (decoded in fp32)

The TE's 248k-row token embedding (1.9 GB) stays on the CPU, and the activation
reserve is sized to the longest prompt, so at caption lengths (< ~1k tokens) all
32 decoder layers (12.9 GB) fit on the card. Layers that still do not fit are
paged in whole, one at a time, from pinned memory; the encode is PCIe-bound while
any are. Leaf-level offload hooks break on Qwen3.5's gated-DeltaNet layers, which read
``conv1d.weight`` functionally instead of calling the module. The quantized TEs
(``precision="int8"``, the default request setting, W8A8 ``library/moss/int8.py``;
``precision="fp8"``, ``library/moss/fp8.py``) put the decoder layers at 6.5 GiB,
so none page at any prompt length.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from library.moss.model import MOSSImageTransformer
from library.moss.requests import variant_dir
from library.runtime.device import clean_memory as empty_cache
from library.runtime.device import free_vram_gb

MAX_SEQUENCE_LENGTH = 4096
TE_PRECISIONS = ("bf16", "fp8", "int8")
BLOCK_LINEAR = re.compile(r"layers\.\d+\.proj\.\w+")


def te_reserve_gb(max_tokens: int) -> float:
    """VRAM left free for the TE's activations: 2.5 GB at 4096 tokens, linear in
    length above a fixed floor."""
    return 0.75 + 1.75 * max_tokens / MAX_SEQUENCE_LENGTH


def _embed_on_cpu(embed: torch.nn.Module, device) -> None:
    """Look tokens up on the CPU and hand the rows to ``device``."""
    embed.register_forward_pre_hook(lambda m, a: (a[0].cpu(),))
    embed.register_forward_hook(lambda m, a, o: o.to(device))


def _page_layer(layer: torch.nn.Module, device) -> None:
    """Copy ``layer`` onto ``device`` for its own forward from a pinned CPU copy,
    and drop the device copy after it (no copy back: the weights are unchanged)."""
    tensors = [*layer.parameters(), *layer.buffers()]
    host = [t.data.pin_memory() for t in tensors]
    for t, h in zip(tensors, host):
        t.data = h

    def load(_m, _a):
        for t, h in zip(tensors, host):
            t.data = h.to(device, non_blocking=True)

    def drop(_m, _a, _o):
        for t, h in zip(tensors, host):
            t.data = h

    layer.register_forward_pre_hook(load)
    layer.register_forward_hook(drop)


def offload_by_layer(te, device, reserve_gb):
    """Non-layer modules but the token embedding, and as many leading decoder layers
    as fit in free VRAM minus ``reserve_gb``, stay resident; the rest page in for
    their own forward. The vision tower stays on the CPU (unused for T2I)."""
    lm = te.language_model
    for name, child in lm.named_children():
        if name == "embed_tokens":
            _embed_on_cpu(child, device)
        elif name != "layers":
            child.to(device)
    budget = free_vram_gb(device.index or 0) - reserve_gb
    paged = 0
    for layer in lm.layers:
        size = sum(p.numel() * p.element_size() for p in layer.parameters()) / 1024**3
        if size <= budget:
            layer.to(device)
            budget -= size
            continue
        paged += 1
        _page_layer(layer, device)
    return paged


def load_tokenizer(model_dir: str | Path | None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(variant_dir(model_dir, "base") / "tokenizer")


def load_text_encoder(
    model_dir: str | Path | None,
    device="cuda",
    max_tokens=MAX_SEQUENCE_LENGTH,
    log=print,
    tok=None,
    precision="bf16",
):
    """``(tokenizer, text_encoder)`` on ``device``, layers that do not fit paged in
    per forward. ``max_tokens`` is the longest prompt the caller will encode (it
    sizes the activation reserve). TE and tokenizer are byte-identical between
    Base and SFT; Base's are used. ``precision="int8"`` / ``"fp8"`` loads the
    catalog's quantized file (``library/moss/fp8.py``, fetched on first use)
    instead of the bf16 shards; the tokenizer still comes from Base."""
    from transformers import AutoModel

    if precision not in TE_PRECISIONS:
        raise ValueError(
            f"text encoder precision {precision!r}, not one of {TE_PRECISIONS}"
        )
    tok = tok or load_tokenizer(model_dir)
    if precision != "bf16":
        from library.moss.fp8 import load_fp8_text_encoder

        te = load_fp8_text_encoder(log=log, precision=precision)
    else:
        root = variant_dir(model_dir, "base")
        te = AutoModel.from_pretrained(
            root / "text_encoder", dtype=torch.bfloat16
        ).eval()
    reserve = te_reserve_gb(max_tokens)
    paged = offload_by_layer(te, torch.device(device), reserve)
    if log:
        log(
            f"[te] {precision}: {len(te.language_model.layers) - paged} decoder layers resident, "
            f"{paged} paged (reserve {reserve:.2f} GB for {max_tokens} tokens)"
        )
    return tok, te


def prompt_text(tok, text: str) -> str:
    """The prompt inside the chat template, as the TE reads it."""
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def prompt_ids(tok, text: str, max_length=MAX_SEQUENCE_LENGTH):
    """The pipeline's tokenisation of one prompt (chat template, no padding)."""
    formatted = prompt_text(tok, text)
    return tok(formatted, return_tensors="pt", truncation=True, max_length=max_length)


def trigger_rows(tok, text: str, trigger: str) -> list[int]:
    """The token rows of ``text``'s prompt whose span overlaps ``trigger`` (the
    first occurrence); empty when the caption lacks it."""
    formatted = prompt_text(tok, text)
    start = formatted.find(trigger) if trigger else -1
    if start < 0:
        return []
    end = start + len(trigger)
    offsets = tok(formatted, return_offsets_mapping=True).offset_mapping
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end]


@torch.no_grad()
def encode_prompt(tok, te, text: str, device="cuda", max_length=MAX_SEQUENCE_LENGTH):
    """One prompt → ``last_hidden_state`` ``(S, 4096)`` bf16 at its true length.
    Exactly ``MOSSImagePipeline.encode_prompt``: chat template with
    ``add_generation_prompt=True, enable_thinking=True``, one prompt per forward
    (no padding: the GDN layers misbehave on it)."""
    inputs = prompt_ids(tok, text, max_length).to(device)
    return te(**inputs, use_cache=False, return_dict=True).last_hidden_state[0]


def encode_prompts(
    model_dir,
    texts,
    device="cuda",
    max_length=MAX_SEQUENCE_LENGTH,
    log=print,
    precision="bf16",
):
    """Load the TE, encode each distinct text once, free it → ``{text: (S,4096) cpu}``."""
    texts = list(dict.fromkeys(texts))
    tok = load_tokenizer(model_dir)
    longest = max(prompt_ids(tok, t, max_length).input_ids.shape[1] for t in texts)
    tok, te = load_text_encoder(
        model_dir, device, longest, log=log, tok=tok, precision=precision
    )
    out = {}
    for text in texts:
        out[text] = encode_prompt(tok, te, text, device, max_length).cpu()
        if log:
            log(f"[te] {out[text].shape[0]:5d} tokens | {text[:60]!r}")
    del tok, te
    empty_cache()
    return out


def pad_embeds(rows, device=None):
    """List of ``(S_i, D)`` → ``embeds (B, S_max, D)``, bool ``mask (B, S_max)``."""
    length = max(r.shape[0] for r in rows)
    embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
    mask = torch.zeros(len(rows), length, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r
        mask[i, : r.shape[0]] = True
    if device is not None:
        embeds, mask = embeds.to(device), mask.to(device)
    return embeds, mask


def load_transformer(model_dir, variant="base", dtype=torch.bfloat16, device=None):
    tr = MOSSImageTransformer.from_pretrained(
        variant_dir(model_dir, variant) / "transformer", torch_dtype=dtype
    ).eval()
    return tr.to(device) if device is not None else tr


def stream_fp32_(tr) -> None:
    """Every parameter but the block linears' weights to fp32: the residual stream,
    modulation, gates, norms, embedders and head run fp32 around the bf16 block
    GEMMs (FA2 and the LoRA stay bf16). Prediction error against an fp32 forward
    2.4-3.5x lower, +1.5% time, +0.6 GB. Call after the LoRA is applied and before
    ``compile_blocks``."""
    keep = {id(m.weight) for n, m in tr.named_modules() if BLOCK_LINEAR.fullmatch(n)}
    for p in tr.parameters():
        if id(p) not in keep:
            p.data = p.data.float()


def compile_blocks(blocks, seq_range=None) -> None:
    """``torch.compile`` on each block's ``forward`` (instance attribute, so swap
    hooks stay outside the graph). One graph for every joint length; must run
    after ``LoRANetwork.apply_to`` / ``load_for_inference``.

    Without ``seq_range``: ``dynamic=True``. ``seq_range = (lo, hi)``:
    ``dynamic=False``, the joint token axis (``x`` / ``rope`` dim 1) marked
    strictly to that range on every call inside it (call-time, so the checkpoint
    recompute re-marks) and the ``ref_start`` int dynamic by source name
    (``dynamic_sources``); every other dim is static. A length outside the range
    runs unmarked (one warning per length for the whole stack). The
    compile cache moves to a per-range directory (persistent entries do not key
    on the mark's bounds). Under grad, an ``x`` without grad (block 0: the
    embedders are frozen) is made a grad leaf so every block shares one graph."""
    from library.runtime.dynamo import (
        pin_compiler_flag,
        pin_dynamo_flag,
        pin_dynamo_limit,
        pin_inductor_flag,
    )
    from library.runtime.harness import compile_signature, isolate_compile_cache

    pin_dynamo_limit("recompile_limit", 16)
    # The LoRA row mask (``LoRANetwork.set_row_mask``) reaches the graph as an
    # nn.Module attribute, whose shapes this flag forces static whatever
    # ``mark_dynamic`` says; its token axis would then guard the block inputs'
    # to one length. Under dynamic=False nothing else turns dynamic by it.
    pin_dynamo_flag("force_nn_module_property_static_shapes", False)
    # dynamic=True otherwise lifts float attributes (the LoRA's scale ·
    # multiplier) into 0-dim CPU float64 inputs, and Inductor emits a C++ CPU
    # kernel for them, which needs cl.exe on Windows (InvalidCxxCompiler).
    torch._dynamo.config.specialize_float = True
    if seq_range is not None:
        lo, hi = seq_range
        if not 1 < lo <= hi:
            raise ValueError(f"seq_range must satisfy 1 < lo <= hi, got {seq_range}")
        if lo < 4096 <= hi:
            # Inductor's mix-order reduction derives a `seq >= 4096` guard from
            # the trace-time hint, which a strict range across 4096 contradicts.
            pin_inductor_flag("triton.mix_order_reduction", False)
        isolate_compile_cache(
            compile_signature(
                n_token_families=None, seq_range=(lo, hi), dynamic_seq=True
            )
            + ";line=moss;static"
        )
        # A Python int is a constant under dynamic=False: ref_start would specialise
        # per sample and contradict the strict mark.
        pin_compiler_flag("dynamic_sources", "L['ref_start']")
    dynamic = seq_range is None
    warned = set()
    for block in blocks:
        block.forward = _marked(
            torch.compile(block.forward, dynamic=dynamic), seq_range, warned
        )


def _marked(compiled, seq_range, warned):
    lo, hi = seq_range or (0, 0)

    def forward(x, rope, *args, **kwargs):
        if torch.is_grad_enabled() and not x.requires_grad:
            x = x.detach().requires_grad_()
        if seq_range is not None:
            n = x.shape[1]
            if lo <= n <= hi:
                torch._dynamo.mark_dynamic(x, 1, min=lo, max=hi)
                torch._dynamo.mark_dynamic(rope, 1, min=lo, max=hi)
            elif n not in warned:
                warned.add(n)
                print(
                    f"compile: {n} tokens outside the marked range {lo}-{hi}",
                    flush=True,
                )
        return compiled(x, rope, *args, **kwargs)

    return forward


def load_vae(model_dir, dtype=torch.float32, device=None):
    from diffusers import AutoencoderKLFlux2

    vae = AutoencoderKLFlux2.from_pretrained(
        variant_dir(model_dir, "base") / "vae", torch_dtype=dtype
    ).eval()
    return vae.to(device) if device is not None else vae
