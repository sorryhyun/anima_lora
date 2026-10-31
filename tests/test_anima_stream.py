"""The Anima fp32 stream's invariants: ``bf16`` is today's forward and sampler bit for bit,
``fp32`` keeps every parameter's dtype, runs the stream in fp32 and the block linears in
bf16, and survives what generation and the trainer do around it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from library.anima.int8 import CONVERT_SUFFIXES
from library.anima.models import Anima
from library.anima.stream import (
    Fp32Linear,
    Fp32RMSNorm,
    check_plan,
    fp32_linears,
    latent_dtype,
    stream_fp32_,
)
from library.anima.trainer.cli import setup_parser
from library.anima.trainer.step import AnimaStep
from library.anima.trainer.trainer import AnimaTrainer
from library.inference import generation
from library.inference.args import build_default_args
from library.inference.models import load_dit_model
from library.inference.request import GenerationRequest
from networks import attention_dispatch
from networks.lora_modules.lora import LoRAModule


def tiny(*, concat_padding_mask: bool = False) -> Anima:
    torch.manual_seed(0)
    model = Anima(
        max_img_h=256,
        max_img_w=256,
        max_frames=4,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=concat_padding_mask,
        model_channels=64,
        num_blocks=2,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    )
    for block in model.blocks:
        block.reset_parameters()
    for p in model.parameters():
        nn.init.normal_(p, std=0.05)
    return model.eval().to(torch.bfloat16)


def inputs():
    torch.manual_seed(1)
    return (
        torch.randn(1, 16, 1, 8, 8),
        torch.tensor([0.5]),
        torch.randn(1, 8, 64).to(torch.bfloat16),
    )


def test_conversion_keeps_parameters_and_block_linears():
    model = tiny()
    before = {n: (p.dtype, p.detach().clone()) for n, p in model.named_parameters()}
    stream_fp32_(model)

    for n, p in model.named_parameters():
        assert p.dtype == before[n][0] and torch.equal(p, before[n][1])
    assert model.dtype == torch.bfloat16
    swapped = {id(m) for m in model.modules() if isinstance(m, Fp32Linear)}
    assert swapped == {id(m) for m in fp32_linears(model)}
    for name, m in model.named_modules():
        if name.endswith(CONVERT_SUFFIXES) or name.endswith("kv_proj"):
            assert type(m) is nn.Linear, name
    assert len(swapped) == 1 + 1 + 2 + 2 + 2 * 4  # embed, head, t, final, 2 × adaln


def test_fp32_linear_leaves_autocast():
    m = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
    m.__class__ = Fp32Linear
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert m(torch.ones(1, 4)).dtype == torch.float32


def test_block_gemms_stay_bf16_and_the_stream_stays_fp32():
    model = tiny()
    stream_fp32_(model)
    seen = {}

    def tap(name):
        def hook(_mod, inputs, output):
            seen[name] = (inputs[0].dtype, output.dtype)

        return hook

    block = model.blocks[-1]
    for name, m in (
        ("qkv", block.self_attn.qkv_proj),
        ("kv", block.cross_attn.kv_proj),
        ("mlp", block.mlp.layer1),
        ("adaln", block.adaln_up_mlp),
        ("head", model.final_layer.linear),
        ("embed", model.x_embedder.proj[1]),
    ):
        m.register_forward_hook(tap(name))
    block.register_forward_hook(tap("block"))
    with torch.no_grad():
        out = model(*inputs())

    assert seen["qkv"][1] == seen["kv"][1] == seen["mlp"][1] == torch.bfloat16
    for name in ("adaln", "head", "embed"):
        assert seen[name][1] == torch.float32, name
    assert seen["block"] == (torch.float32, torch.float32)
    assert out.dtype == torch.float32


def test_rope_keeps_fp32_up_to_attention():
    model = tiny()
    stream_fp32_(model)
    q_norm = model.blocks[0].self_attn.q_norm
    assert type(q_norm) is Fp32RMSNorm
    x = torch.randn(1, 2, 4, 16).to(torch.bfloat16)
    assert q_norm(x).dtype == torch.float32

    plain = tiny()
    stream_fp32_(plain, rope=False)
    assert type(plain.blocks[0].self_attn.q_norm) is not Fp32RMSNorm


def test_stream_moves_the_prediction_only_a_little():
    x, t, c = inputs()
    model = tiny()
    with torch.no_grad():
        narrow = model(x.to(torch.bfloat16), t.to(torch.bfloat16), c).float()
    stream_fp32_(model)
    with torch.no_grad():
        wide = model(x, t, c)
    assert torch.isfinite(wide).all()
    assert (wide - narrow).norm() / narrow.norm() < 0.05


def test_lora_on_a_stream_linear_runs_fp32():
    model = tiny()
    stream_fp32_(model)
    target = model.blocks[0].adaln_up_self_attn
    lora = LoRAModule("lora_unet_t", target, 1.0, 4, 1)
    lora.to(torch.bfloat16)
    lora.apply_to()
    with torch.no_grad():
        out = model(*inputs())
    assert torch.isfinite(out).all()
    assert out.dtype == torch.float32


def gen_args(**over):
    argv = ["--text_encoder", "x", "--save_path", "x", "--prompt", "x"]
    argv += ["--image_size", "32", "32", "--infer_steps", "3", "--seed", "3"]
    argv += ["--guidance_scale", "1.0", "--flow_shift", "3.0", "--sampler", "euler"]
    argv += ["--device", "cpu"]
    args = build_default_args(argv)
    return SimpleNamespace(**{**vars(args), **over})


def context():
    torch.manual_seed(2)
    return {"embed": [torch.randn(1, 8, 64)], "prompt": "x"}


def today(model, args, ctx):
    """The loop as it stood before ``--stream``."""
    seed_g = torch.Generator(device="cpu").manual_seed(args.seed)
    from diffusers.utils.torch_utils import randn_tensor

    device = torch.device("cpu")
    latents = randn_tensor(
        (1, 16, 1, 4, 4), generator=seed_g, device=device, dtype=torch.bfloat16
    )
    embed = ctx["embed"][0].to(torch.bfloat16)
    pad = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    timesteps, sigmas = generation.inference_utils.get_timesteps_sigmas(3, 3.0, device)
    timesteps = timesteps.to(device, dtype=torch.bfloat16)
    for i, t in enumerate(timesteps):
        with torch.no_grad():
            v = model(latents, t.expand(1), embed, padding_mask=pad)
        latents = (latents.float() - (sigmas[i] - sigmas[i + 1]) * v.float()).to(
            latents.dtype
        )
    return latents


def test_bf16_generation_is_bit_identical_to_today():
    model = tiny(concat_padding_mask=True)
    args, ctx = gen_args(), context()
    expected = today(model, args, ctx)
    got = generation.generate_body(args, model, ctx, None, torch.device("cpu"), 3)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected)


def test_fp32_generation_keeps_the_sampler_state():
    model = tiny(concat_padding_mask=True)
    stream_fp32_(model)
    assert latent_dtype(model) == torch.float32
    got = generation.generate_body(
        gen_args(), model, context(), None, torch.device("cpu"), 3
    )
    assert got.dtype == torch.float32
    assert torch.isfinite(got).all()


def test_same_seed_is_the_same_noise_on_both_streams():
    from diffusers.utils.torch_utils import randn_tensor

    device = torch.device("cpu")
    draw = [
        randn_tensor(
            (1, 16, 1, 32, 32),
            generator=torch.Generator("cpu").manual_seed(5),
            device=device,
            dtype=dtype,
        )
        for dtype in (torch.bfloat16, torch.float32)
    ]
    assert torch.equal(draw[0], draw[1].to(torch.bfloat16))


def test_fp32_is_the_cli_default_and_a_namespace_without_it_is_bf16():
    assert gen_args().stream == "fp32"
    assert latent_dtype(SimpleNamespace()) == torch.bfloat16
    assert "--stream" not in GenerationRequest().to_argv()
    argv = GenerationRequest(stream="bf16").to_argv()
    assert argv[argv.index("--stream") + 1] == "bf16"


@pytest.mark.parametrize(
    ("stream", "easycontrol", "swapped"),
    [("fp32", None, True), ("bf16", None, False), ("fp32", "ec.safetensors", False)],
)
def test_load_gates_the_conversion(monkeypatch, stream, easycontrol, swapped):
    from library import anima as anima_pkg
    from library.inference import models as im

    model = tiny()
    monkeypatch.setattr(
        im.anima_utils, "load_anima_model", lambda *a, **k: model, raising=True
    )
    monkeypatch.setattr(im, "attach_adapters", lambda *a, **k: None)
    monkeypatch.setattr(im, "clean_memory_on_device", lambda *a: None)
    assert anima_pkg is not None
    args = gen_args(
        dit="/nowhere",
        lora_weight=None,
        stream=stream,
        easycontrol_weight=easycontrol,
    )
    out = load_dit_model(args, torch.device("cpu"), torch.bfloat16)
    assert any(isinstance(m, Fp32Linear) for m in out.modules()) is swapped
    assert latent_dtype(out) == (torch.float32 if swapped else torch.bfloat16)


def test_training_flag_defaults_to_bf16():
    assert setup_parser().parse_args([]).stream == "bf16"
    assert setup_parser().parse_args(["--stream", "fp32"]).stream == "fp32"
    with pytest.raises(SystemExit):
        setup_parser().parse_args(["--stream", "fp16"])


def test_training_refuses_easycontrol():
    with pytest.raises(SystemExit):
        check_plan(SimpleNamespace(network_module="networks.methods.easycontrol"))
    check_plan(SimpleNamespace(network_module="networks.lora_anima"))


@pytest.mark.parametrize(("stream", "stamped"), [("bf16", False), ("fp32", True)])
def test_lora_records_the_stream(stream, stamped):
    meta = {}
    args = SimpleNamespace(
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        timestep_sampling="sigma",
        sigmoid_scale=1.0,
        discrete_flow_shift=1.0,
        stream=stream,
    )
    AnimaTrainer._update_metadata(meta, args)
    assert ("ss_stream" in meta) is stamped
    if stamped:
        assert meta["ss_stream"] == "fp32"


@pytest.mark.parametrize(
    ("stream", "weight_dtype", "expected"),
    [
        ("bf16", torch.bfloat16, torch.bfloat16),
        ("bf16", torch.float32, torch.float32),
        ("fp32", torch.bfloat16, torch.float32),
    ],
)
def test_training_input_and_sigma_follow_the_stream(stream, weight_dtype, expected):
    args = SimpleNamespace(
        sampler="default",
        ip_noise_gamma=None,
        stream=stream,
    )
    ctx = SimpleNamespace(
        args=args,
        noise_scheduler=SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=1000)
        ),
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        weight_dtype=weight_dtype,
        network=SimpleNamespace(),
    )
    latents = torch.randn(1, 16, 8, 8)
    noisy, timesteps, _ = AnimaStep._sample_noisy_input(
        SimpleNamespace(_hydra_warmup_step=0),
        ctx,
        latents,
        torch.randn_like(latents),
        is_train=True,
        sigmas=torch.tensor([0.3]),
    )
    assert noisy.dtype == timesteps.dtype == expected


def test_preview_keeps_the_dits_dtype_off_the_stream():
    assert latent_dtype(SimpleNamespace(), torch.float32) == torch.float32
    assert latent_dtype(SimpleNamespace(stream="fp32")) == torch.float32


def test_training_gradient_reaches_a_lora_past_the_stream():
    model = tiny().train()
    model.requires_grad_(False)
    stream_fp32_(model)
    block = model.blocks[0]
    loras = [
        LoRAModule(f"lora_unet_{i}", target, 1.0, 4, 1)
        for i, target in enumerate((block.adaln_up_mlp, block.self_attn.output_proj))
    ]
    for lora in loras:
        lora.to(torch.bfloat16)
        lora.apply_to()
        lora.train()
    x, t, c = inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x, t, c)
    assert out.dtype == torch.float32
    out.square().mean().backward()
    for lora in loras:
        grad = lora.lora_up.weight.grad
        assert grad.dtype == torch.bfloat16
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_a_compiled_block_is_one_graph_with_fa2_rounding(monkeypatch):
    """The modulation's autocast exit and ``Attention.forward``'s FA2 rounding stay in the
    block's graph, so ``compile_dynamic_seq`` keeps one graph per block."""
    model = tiny()
    stream_fp32_(model)
    model.requires_grad_(False)
    params = attention_dispatch.AttentionParams.create_attention_params("torch", None)
    monkeypatch.setattr(
        type(params), "supports_fp32", property(lambda self: False), raising=True
    )
    x = torch.randn(1, 1, 16, 1, 64).requires_grad_()
    args = (
        x,
        torch.randn(1, 1, 64),
        torch.randn(1, 8, 64).to(torch.bfloat16),
        params,
        (torch.randn(1, 16, 16), torch.randn(1, 16, 16)),
        torch.randn(1, 1, 3 * 64),
    )
    torch.set_autocast_enabled("cuda", True)  # the state FA2's branch reads
    try:
        assert torch.is_autocast_enabled()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            explained = torch._dynamo.explain(model.blocks[0]._forward)(*args)
    finally:
        torch.set_autocast_enabled("cuda", False)
    assert explained.graph_count == 1
    assert explained.graph_break_count == 0
