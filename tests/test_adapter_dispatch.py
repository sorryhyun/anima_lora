"""Trainer wiring test for the method-adapter dispatch (issues.md P1.2).

``tests/test_repa.py`` exercises the adapter in isolation and structurally
cannot see trainer wiring — the silent-REPA bug lived exactly in that gap:
``extra_forwards`` dispatched only on the cached-crossattn branch of
``get_noise_pred_and_target``, so the in-model text path (crossattn_emb=None,
EasyControl's default) trained as a baseline with REPA configured ON.

This test calls the real ``AnimaStep.get_noise_pred_and_target`` with a
stubbed ``TrainCtx`` (fake DiT returning a correctly-shaped 5D tensor, a
recording adapter) once per text-conditioning branch and asserts the adapter
ran in BOTH and its aux key landed in ``extras_for_step``. CPU-only, no real
model.
"""

from __future__ import annotations

import argparse
import contextlib
from types import SimpleNamespace

import pytest
import torch

from library.anima.trainer.step import AnimaStep
from library.training import TrainCtx
from library.training.method_adapter import ForwardArtifacts, MethodAdapter

B, C, H, W = 2, 16, 8, 8
SEQ, DIM = 32, 64


class _RecordingAdapter(MethodAdapter):
    """Records every lifecycle dispatch and returns a tagged aux key."""

    name = "recording"

    def __init__(self):
        self.primed: list[bool] = []
        self.extra_calls: list[ForwardArtifacts] = []

    def prime_for_forward(self, ctx, batch, latents, *, is_train):
        self.primed.append(is_train)

    def extra_forwards(self, ctx, primary):
        self.extra_calls.append(primary)
        return {"recording_aux": torch.tensor(0.25)}


class _FakeAnima:
    """Stands in for the DiT: records call kwargs, returns a 5D prediction."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, noisy_model_input, timesteps, cond, *, padding_mask, **kw):
        assert noisy_model_input.dim() == 5, "DiT input must be 5D (B,C,1,H,W)"
        assert cond is not None, "positional conditioning must be supplied"
        self.calls.append({"cond": cond, "padding_mask": padding_mask, "kw": kw})
        return torch.randn_like(noisy_model_input)


class _StubAccelerator:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()


def _make_args() -> argparse.Namespace:
    return argparse.Namespace(
        sampler="default",
        timestep_sampling="uniform",
        sigmoid_bias=0.0,
        t_min=None,
        t_max=None,
        ip_noise_gamma=None,
        gradient_checkpointing=False,
        weighting_scheme="none",
        vr_loss_weight=0.0,
        max_train_steps=10,
        gradient_accumulation_steps=1,
    )


def _make_ctx(anima) -> TrainCtx:
    return TrainCtx(
        args=_make_args(),
        accelerator=_StubAccelerator(),
        network=SimpleNamespace(),  # no routers, no postfix
        unet=anima,
        vae=None,
        text_encoders=[],
        noise_scheduler=SimpleNamespace(
            config=SimpleNamespace(num_train_timesteps=1000)
        ),
        text_encoding_strategy=None,
        tokenize_strategy=None,
        vae_dtype=torch.float32,
        weight_dtype=torch.float32,
        optimizer_eval_fn=lambda: None,
        optimizer_train_fn=lambda: None,
        is_tracking=False,
    )


def _cached_crossattn_conds():
    # 5-tuple: cache includes crossattn_emb (T5-target space) — the cached path.
    return [
        None,  # prompt_embeds (unused on this path)
        None,  # attn_mask
        None,  # t5_input_ids
        torch.ones(B, SEQ),  # t5_attn_mask
        torch.randn(B, SEQ, DIM),  # crossattn_emb
    ]


def _in_model_conds():
    # 4-tuple: no cached crossattn_emb — the in-model text path
    # (EasyControl's default).
    return [
        torch.randn(B, SEQ, DIM),  # prompt_embeds
        torch.ones(B, SEQ),  # attn_mask
        torch.zeros(B, SEQ, dtype=torch.long),  # t5_input_ids
        torch.ones(B, SEQ),  # t5_attn_mask
    ]


def _run_one(text_encoder_conds):
    torch.manual_seed(0)
    anima = _FakeAnima()
    adapter = _RecordingAdapter()
    trainer = AnimaStep()
    trainer.adapters = [adapter]
    ctx = _make_ctx(anima)
    latents = torch.randn(B, C, H, W)

    pred, target, timesteps, weighting = trainer.get_noise_pred_and_target(
        ctx, latents, {}, text_encoder_conds, is_train=True
    )
    return trainer, anima, adapter, (pred, target, timesteps, weighting)


@pytest.mark.core
def test_dispatch_runs_on_cached_crossattn_path():
    trainer, anima, adapter, (pred, target, timesteps, _) = _run_one(
        _cached_crossattn_conds()
    )
    assert adapter.primed == [True]
    assert len(adapter.extra_calls) == 1
    primary = adapter.extra_calls[0]
    assert primary.crossattn_emb is not None
    assert primary.model_pred.dim() == 5
    # The adapter's aux key must land in extras_for_step for the composer.
    assert "recording_aux" in trainer.state.extras_for_step
    assert len(anima.calls) == 1
    assert pred.shape == target.shape == (B, C, H, W)
    assert timesteps.shape == (B,)


@pytest.mark.core
def test_dispatch_runs_on_in_model_text_path():
    # The regression: crossattn_emb=None used to skip extra_forwards entirely.
    trainer, anima, adapter, (pred, target, _, _) = _run_one(_in_model_conds())
    assert adapter.primed == [True]
    assert len(adapter.extra_calls) == 1
    primary = adapter.extra_calls[0]
    assert primary.crossattn_emb is None
    assert primary.model_pred.dim() == 5
    assert "recording_aux" in trainer.state.extras_for_step
    # In-model path threads the T5 ids/masks through the forward kwargs —
    # the same kw dict the adapter receives.
    assert set(anima.calls[0]["kw"]) == {
        "target_input_ids",
        "target_attention_mask",
        "source_attention_mask",
    }
    # Same tensors in the artifacts' kw as in the actual forward call
    # (``**kw`` re-packs the dict, so compare per-key identity).
    assert set(primary.forward_kwargs) == set(anima.calls[0]["kw"])
    for key, value in anima.calls[0]["kw"].items():
        assert primary.forward_kwargs[key] is value
    assert pred.shape == target.shape == (B, C, H, W)


def test_postfix_path_normalizes_into_uniform_conditioning():
    # build_forward_conditioning (issues.md P2.3) must keep the postfix
    # splice semantics: positional cond IS the extended crossattn_emb, and
    # the pooled override sees only the real (pre-splice) text.
    from library.training.forward import build_forward_conditioning
    from library.training.forward.text_conds import PreparedTextConds

    class _PostfixNet:
        def append_postfix(self, emb, seqlens, *, timesteps):
            return torch.cat([emb, torch.zeros(emb.shape[0], 2, emb.shape[2])], dim=1)

    emb = torch.randn(B, SEQ, DIM)
    tc = PreparedTextConds(
        crossattn_emb=emb,
        prompt_embeds=None,
        attn_mask=None,
        t5_input_ids=None,
        t5_attn_mask=torch.ones(B, SEQ),
    )
    cond = build_forward_conditioning(
        network=_PostfixNet(), tc=tc, timesteps=torch.rand(B)
    )
    assert cond.has_postfix
    assert cond.cond is cond.crossattn_emb
    assert cond.cond.shape == (B, SEQ + 2, DIM)
    torch.testing.assert_close(cond.kw["pooled_text_override"], emb.max(dim=1).values)
