"""Tests for ``GenerationRequest`` — the typed front door for ``generate()``.

Covers:

* the always-emitted core defaults match ``inference.parse_args`` (drift guard:
  the dataclass mirrors the parser in two places, this pins them together)
* scalar / path / LoRA fields thread through to the namespace
* ``extra_argv`` is appended and can override a structured field
"""

from __future__ import annotations

import inference
import pytest

from library.inference.request import GenerationRequest

# Minimal argv covering the parser's two required args + a prompt, used as the
# "bare parser defaults" baseline.
_BASELINE_ARGV = ["--text_encoder", "te", "--save_path", "out.png", "--prompt", "p"]

# Fields GenerationRequest always emits with a parser-mirroring default.
_CORE_DEFAULT_FIELDS = [
    "negative_prompt",
    "image_size",
    "infer_steps",
    "guidance_scale",
    "flow_shift",
    "sampler",
    "attn_mode",
    "output_type",
]


@pytest.mark.core
def test_core_defaults_match_parser():
    """A default request must reproduce the parser's defaults verbatim."""
    baseline = inference.parse_args(_BASELINE_ARGV)
    args = GenerationRequest(prompt="p").to_args()
    for fld in _CORE_DEFAULT_FIELDS:
        assert getattr(args, fld) == getattr(baseline, fld), f"drift on {fld!r}"


def test_scalar_and_path_fields_thread_through():
    args = GenerationRequest(
        prompt="a fox",
        negative_prompt="blurry",
        image_size=(832, 1248),
        infer_steps=30,
        guidance_scale=4.0,
        flow_shift=2.5,
        sampler="er_sde",
        seed=123,
        dit="d.safetensors",
        vae="v.safetensors",
        text_encoder="te.safetensors",
        save_path="out/x.png",
        vae_chunk_size=64,
        pooled_text_proj="mod.safetensors",
    ).to_args()

    assert args.prompt == "a fox"
    assert args.negative_prompt == "blurry"
    assert args.image_size == [832, 1248]  # parser yields a list
    assert args.infer_steps == 30
    assert args.guidance_scale == 4.0
    assert args.flow_shift == 2.5
    assert args.sampler == "er_sde"
    assert args.seed == 123
    assert args.dit == "d.safetensors"
    assert args.vae == "v.safetensors"
    assert args.text_encoder == "te.safetensors"
    assert args.save_path == "out/x.png"
    assert args.vae_chunk_size == 64
    assert args.pooled_text_proj == "mod.safetensors"


def test_lora_list_and_multiplier_list():
    args = GenerationRequest(
        prompt="p",
        lora_weight=["a.safetensors", "b.safetensors"],
        lora_multiplier=[0.8, 0.5],
    ).to_args()
    assert args.lora_weight == ["a.safetensors", "b.safetensors"]
    assert args.lora_multiplier == [0.8, 0.5]


def test_extra_argv_overrides_structured_field():
    """extra_argv is appended last, so it wins on conflict."""
    args = GenerationRequest(
        prompt="p", infer_steps=30, extra_argv=["--infer_steps", "12"]
    ).to_args()
    assert args.infer_steps == 12
