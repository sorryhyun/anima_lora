"""REPA v2 adapter + wiring tests (docs/experimental/repa.md).

Covers the load-bearing invariants:
  - native_flatten layout-agnosticism: the captured (B,1,seq,1,D) and eager
    (B,1,H,W,D) block outputs produce a bit-identical alignment loss.
  - grad flows back through the captured feature (into the LoRA blocks).
  - PE-grid orientation disambiguation (aspect-symmetric token counts).
  - the composer gate is off by default (flag-off ⇒ "repa" never active).
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from library.training.repa import (
    REPAHead,
    REPAMethodAdapter,
    dog_standardize,
)
from library.vision.buckets import get_bucket_spec


def _make_adapter(mode: str, *, patch: int = 2) -> REPAMethodAdapter:
    a = REPAMethodAdapter()
    a._mode = mode
    a._patch = patch
    a._spec = get_bucket_spec("pe_spatial")
    return a


def _primary(latents: torch.Tensor, *, is_train: bool = True, timesteps=None):
    # extra_forwards reads .is_train, .latents, and (only when timestep
    # reweighting is on) .timesteps — per-sample σ∈[0,1].
    return types.SimpleNamespace(
        is_train=is_train, latents=latents, timesteps=timesteps
    )


def _ctx(network=None):
    return types.SimpleNamespace(network=network)


def _square_inputs(b=2, d_enc=768):
    """Square 32x32 encoder grid → 1024 patches + CLS. Latent 64x64, patch 2."""
    spec = get_bucket_spec("pe_spatial")
    # (32,32) bucket → 1024 patches, T_pe = 1025.
    pe = torch.randn(b, 32 * 32 + 1, d_enc)
    latents = torch.zeros(b, 16, 64, 64)  # h_dit=w_dit=32 → 1024 DiT tokens
    return spec, pe, latents


def test_native_flatten_layout_agnostic():
    """Eager (B,1,H,W,D) and native-flatten (B,1,seq,1,D) → identical loss."""
    _spec, pe, latents = _square_inputs()
    b, d = 2, 64
    # Shared underlying token data in row-major (B, 1024, D).
    tokens = torch.randn(b, 1024, d)
    eager = tokens.reshape(b, 1, 32, 32, d).clone()
    flat = tokens.reshape(b, 1, 1024, 1, d).clone()

    for mode in ("relational", "absolute"):
        net = None
        if mode == "absolute":
            net = types.SimpleNamespace(repa_head=REPAHead(d, d, 768))
        a = _make_adapter(mode)

        a._captured, a._pe_features, a._latent_hw = eager, pe, (64, 64)
        loss_eager = a.extra_forwards(_ctx(net), _primary(latents))["repa"]

        a._captured, a._pe_features, a._latent_hw = flat, pe, (64, 64)
        loss_flat = a.extra_forwards(_ctx(net), _primary(latents))["repa"]

        assert torch.allclose(loss_eager, loss_flat, atol=0, rtol=0), mode
        assert torch.isfinite(loss_eager)


def test_grad_flows_to_captured():
    _spec, pe, latents = _square_inputs()
    b, d = 2, 64
    cap = torch.randn(b, 1, 32, 32, d, requires_grad=True)
    a = _make_adapter("relational")
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    loss = a.extra_forwards(_ctx(), _primary(latents))["repa"]
    loss.backward()
    assert cap.grad is not None and torch.isfinite(cap.grad).all()
    assert cap.grad.abs().sum() > 0


def test_relational_zero_when_identical_structure():
    """Same per-token directions on both sides → Gram match → ~0 loss."""
    a = _make_adapter("relational")
    b, n, d = 2, 1024, 768
    feats = torch.randn(b, n, d)
    # DiT side D == PE d here so the same directions give identical Grams.
    cap = feats.reshape(b, 1, 32, 32, d)
    pe = torch.cat([torch.zeros(b, 1, d), feats], dim=1)  # prepend CLS
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    loss = a.extra_forwards(_ctx(), _primary(torch.zeros(b, 16, 64, 64)))["repa"]
    assert loss.item() < 1e-6


def test_metrics_surface_align_loss():
    """extra_forwards stashes the unweighted scalar for the loggers."""
    a = _make_adapter("relational")
    _spec, pe, latents = _square_inputs()
    a._captured, a._pe_features, a._latent_hw = (
        torch.randn(2, 1, 32, 32, 64),
        pe,
        (64, 64),
    )
    loss = a.extra_forwards(_ctx(), _primary(latents))["repa"]
    m = a.metrics(_ctx())
    assert m["repa/align_loss"] == pytest.approx(float(loss.detach()))
    # metrics() returns a copy — mutating it must not poison the adapter stash.
    m["repa/align_loss"] = -1.0
    assert a.metrics(_ctx())["repa/align_loss"] != -1.0


def test_timestep_weighting_off_is_bit_identical():
    """g=0 (default) never reads timesteps and is bit-exact to the legacy loss."""
    a = _make_adapter("relational")
    _spec, pe, latents = _square_inputs()
    cap = torch.randn(2, 1, 32, 32, 64)
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    base = a.extra_forwards(_ctx(), _primary(latents, timesteps=None))["repa"]

    a._timestep_weighting = 0.0  # explicit
    a._captured = cap
    same = a.extra_forwards(_ctx(), _primary(latents, timesteps=None))["repa"]
    assert torch.equal(base, same)
    assert "repa/tsw_w_mean" not in a.metrics(_ctx())


def test_timestep_weighting_high_noise_emphasis():
    """g>0 up-weights high-σ samples; an all-high-σ batch loss > all-low-σ batch."""
    _spec, pe, latents = _square_inputs(b=2)
    cap = torch.randn(2, 1, 32, 32, 64)

    def _loss(g, sigma):
        a = _make_adapter("relational")
        a._timestep_weighting = g
        a._captured, a._pe_features, a._latent_hw = cap.clone(), pe, (64, 64)
        t = torch.full((2,), sigma)
        return a.extra_forwards(_ctx(), _primary(latents, timesteps=t))["repa"]

    # Same Gram error per sample; only the σ weight differs. (g+1)·σ^g: high σ
    # gets a larger multiplier → larger weighted mean.
    hi = _loss(2.0, 0.9)
    lo = _loss(2.0, 0.1)
    assert hi > lo
    # g<0 inverts the ordering (low-noise emphasis).
    hi_n = _loss(-2.0, 0.9)
    lo_n = _loss(-2.0, 0.1)
    assert lo_n > hi_n


def test_pe_grid_orientation_disambiguation():
    a = _make_adapter("relational")
    # 1058 patches is shared by (46,23) portrait and (23,46) landscape.
    assert 46 * 23 == 23 * 46 == 1058
    gh, gw = a._pe_grid(1058, h_lat=144, w_lat=72)  # portrait
    assert (gh, gw) == (46, 23)
    gh, gw = a._pe_grid(1058, h_lat=72, w_lat=144)  # landscape
    assert (gh, gw) == (23, 46)
    # Square is unambiguous.
    assert a._pe_grid(1024, 64, 64) == (32, 32)


def test_skips_when_not_train_or_missing():
    a = _make_adapter("relational")
    _spec, pe, latents = _square_inputs()
    a._captured, a._pe_features, a._latent_hw = (
        torch.randn(2, 1, 32, 32, 64),
        pe,
        (64, 64),
    )
    # Validation pass → no REPA term.
    assert a.extra_forwards(_ctx(), _primary(latents, is_train=False)) is None
    # Missing PE features → skip.
    a._pe_features = None
    assert a.extra_forwards(_ctx(), _primary(latents)) is None


def test_composer_gate_off_by_default():
    from library.training.losses import build_loss_composer

    args = types.SimpleNamespace(
        vr_loss_weight=0.0, functional_loss_weight=0.0, multiscale_loss_weight=0.0
    )
    # A network with no _repa_weight attribute → repa must not be active.
    net = types.SimpleNamespace()
    comp = build_loss_composer(args, net)
    assert "repa" not in comp.active_losses

    net._repa_weight = 0.05
    comp = build_loss_composer(args, net)
    assert "repa" in comp.active_losses


def test_easycontrol_create_network_stamps_repa():
    """EasyControl's factory mirrors the LoRA factory's _repa_* stamping.

    Everything downstream (build_method_adapters, _repa_loss, the dataset PE
    loader gate in train.py) keys off network._repa_weight, so the stamp is
    the whole integration on the network side.
    """
    from networks.methods.easycontrol import create_network

    common = dict(vae=None, text_encoders=[], unet=None)
    net = create_network(1.0, 8, 8.0, **common, use_repa="true")
    assert net._repa_weight == pytest.approx(0.05)
    assert net._repa_mode == "relational"
    assert net._repa_layer == 8
    assert net._repa_encoder == "pe_spatial"

    # Off by default — and the adapter-attach predicate stays false.
    net_off = create_network(1.0, 8, 8.0, **common)
    assert net_off._repa_weight == 0.0

    # The absolute arm needs a repa_head EasyControlNetwork doesn't carry.
    with pytest.raises(ValueError, match="relational"):
        create_network(1.0, 8, 8.0, **common, use_repa="true", repa_mode="absolute")


def _train_ctx(max_train_steps=10, accum=1):
    args = types.SimpleNamespace(
        max_train_steps=max_train_steps, gradient_accumulation_steps=accum
    )
    return types.SimpleNamespace(
        args=args, accelerator=types.SimpleNamespace(device="cpu"), network=None
    )


def test_anneal_hard_cutoff_fraction():
    """Lever 1: fraction-of-run cutoff — term active before, skipped after."""
    a = _make_adapter("relational")
    a._anneal_steps = 0.5
    _spec, pe, latents = _square_inputs()
    batch = {"repa_pe_features": pe}
    cap = torch.randn(2, 1, 32, 32, 64)
    ctx = _train_ctx(max_train_steps=10)

    for step in range(10):
        a.prime_for_forward(ctx, batch, latents, is_train=True)
        a._captured = cap  # the block hook would fire during the forward
        out = a.extra_forwards(ctx, _primary(latents))
        if step < 5:
            assert out is not None and torch.isfinite(out["repa"]), step
        else:
            assert out is None, step


def test_anneal_absolute_steps_with_accumulation():
    """>1 = absolute optimizer steps; micro-batches convert via accum."""
    a = _make_adapter("relational")
    a._anneal_steps = 2  # optimizer steps
    _spec, pe, latents = _square_inputs()
    batch = {"repa_pe_features": pe}
    cap = torch.randn(2, 1, 32, 32, 64)
    ctx = _train_ctx(max_train_steps=100, accum=2)

    # accum=2 → micro-batches 0..3 are optimizer steps 0–1 (active), 4+ off.
    for micro in range(6):
        a.prime_for_forward(ctx, batch, latents, is_train=True)
        a._captured = cap
        out = a.extra_forwards(ctx, _primary(latents))
        assert (out is not None) == (micro < 4), micro


def test_anneal_clock_ignores_validation():
    """Validation passes must not advance the optimizer-step clock."""
    a = _make_adapter("relational")
    a._anneal_steps = 2.0  # 2 optimizer steps
    _spec, pe, latents = _square_inputs()
    batch = {"repa_pe_features": pe}
    ctx = _train_ctx(max_train_steps=100)

    a.prime_for_forward(ctx, batch, latents, is_train=True)  # step 0
    for _ in range(5):
        a.prime_for_forward(ctx, batch, latents, is_train=False)
    assert a._train_micro_steps == 1
    a.prime_for_forward(ctx, batch, latents, is_train=True)  # step 1 — last active
    assert a._pe_features is not None
    a.prime_for_forward(ctx, batch, latents, is_train=True)  # step 2 — cut off
    assert a._pe_features is None


def test_anneal_off_by_default():
    a = _make_adapter("relational")
    assert a._anneal_steps == 0.0
    assert not a._past_anneal_cutoff(_train_ctx().args, micro_step=10**6)


def test_spatial_norm_cancels_global_offset():
    """Lever 2: with spatial_norm a shared additive token direction is exactly
    removed from the target; without it the loss shifts."""
    _spec, pe, latents = _square_inputs()
    cap = torch.randn(2, 1, 32, 32, 64)
    # Large common direction added to every PE token (CLS row irrelevant).
    pe_shifted = pe + 5.0 * torch.randn(1, 1, pe.shape[-1])

    def _loss(adapter, target):
        adapter._captured, adapter._pe_features, adapter._latent_hw = (
            cap,
            target,
            (64, 64),
        )
        return adapter.extra_forwards(_ctx(), _primary(latents))["repa"]

    a_on = _make_adapter("relational")
    a_on._spatial_norm = True
    assert torch.allclose(_loss(a_on, pe), _loss(a_on, pe_shifted), atol=1e-5)

    a_off = _make_adapter("relational")
    assert not torch.allclose(_loss(a_off, pe), _loss(a_off, pe_shifted), atol=1e-3)


def test_spatial_norm_off_is_bit_identical_to_legacy():
    """Default-off flag must not perturb the existing relational loss."""
    _spec, pe, latents = _square_inputs()
    cap = torch.randn(2, 1, 32, 32, 64)
    a = _make_adapter("relational")
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    loss_default = a.extra_forwards(_ctx(), _primary(latents))["repa"]

    # Recompute the legacy formula by hand.
    tokens = cap.reshape(2, -1, 64)
    dit_grid = tokens.reshape(2, 32, 32, 64).permute(0, 3, 1, 2)
    dit_tok = (
        F.adaptive_avg_pool2d(dit_grid.float(), (32, 32)).flatten(2).transpose(1, 2)
    )
    dit_hat = F.normalize(dit_tok, dim=-1)
    pe_hat = F.normalize(pe[:, 1:, :].float(), dim=-1)
    g_dit = torch.bmm(dit_hat, dit_hat.transpose(1, 2))
    g_pe = torch.bmm(pe_hat, pe_hat.transpose(1, 2))
    expected = F.mse_loss(g_dit, g_pe)
    assert torch.equal(loss_default, expected)


def test_factory_stamps_phase1_levers():
    """Both factories stamp the lever kwargs (default-off)."""
    from networks.methods.easycontrol import create_network

    common = dict(vae=None, text_encoders=[], unet=None)
    net = create_network(
        1.0,
        8,
        8.0,
        **common,
        use_repa="true",
        repa_anneal_steps="0.5",
        repa_spatial_norm="true",
    )
    assert net._repa_anneal_steps == pytest.approx(0.5)
    assert net._repa_spatial_norm is True

    net_default = create_network(1.0, 8, 8.0, **common, use_repa="true")
    assert net_default._repa_anneal_steps == 0.0
    assert net_default._repa_spatial_norm is False


def test_grad_heatmap_accumulates_and_loss_unchanged():
    """Lever-3 diagnostic: probing must not perturb the loss, and counts
    accumulate exactly k per sample (top-10% of the canonical 32x32 grid)."""
    _spec, pe, latents = _square_inputs()
    cap = torch.randn(2, 1, 32, 32, 64)

    a_off = _make_adapter("relational")
    a_off._captured, a_off._pe_features, a_off._latent_hw = cap, pe, (64, 64)
    loss_off = a_off.extra_forwards(_ctx(), _primary(latents))["repa"]

    a_on = _make_adapter("relational")
    a_on._grad_heatmap_every = 1
    cap_g = cap.clone().requires_grad_(True)
    a_on._captured, a_on._pe_features, a_on._latent_hw = cap_g, pe, (64, 64)
    loss_on = a_on.extra_forwards(_ctx(), _primary(latents))["repa"]

    assert torch.equal(loss_on, loss_off)
    k = round(0.10 * 32 * 32)  # 102 per sample
    assert a_on._heat_counts is not None
    assert a_on._heat_counts.sum().item() == pytest.approx(2 * k)
    assert a_on._heat_samples == 2
    conc = a_on.metrics(_ctx())["repa/heatmap_conc"]
    assert conc >= 0.99  # max freq is at least the uniform expectation
    # retain_graph kept the main backward alive.
    loss_on.backward()
    assert cap_g.grad is not None and torch.isfinite(cap_g.grad).all()


def test_grad_heatmap_off_by_default():
    _spec, pe, latents = _square_inputs()
    a = _make_adapter("relational")
    assert a._grad_heatmap_every == 0
    cap = torch.randn(2, 1, 32, 32, 64, requires_grad=True)
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    a.extra_forwards(_ctx(), _primary(latents))
    assert a._heat_counts is None
    assert "repa/heatmap_conc" not in a.metrics(_ctx())


def test_grad_heatmap_every_n_cadence():
    _spec, pe, latents = _square_inputs()
    a = _make_adapter("relational")
    a._grad_heatmap_every = 2
    for _ in range(3):  # probes fire on runs 0 and 2
        cap = torch.randn(2, 1, 32, 32, 64, requires_grad=True)
        a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
        a.extra_forwards(_ctx(), _primary(latents))
    assert a._heat_samples == 4


def test_grad_heatmap_no_grad_path_is_safe():
    """A detached capture must warn-and-skip, never crash the step."""
    _spec, pe, latents = _square_inputs()
    a = _make_adapter("relational")
    a._grad_heatmap_every = 1
    cap = torch.randn(2, 1, 32, 32, 64)  # no requires_grad
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    out = a.extra_forwards(_ctx(), _primary(latents))
    assert out is not None and torch.isfinite(out["repa"])
    assert a._heat_counts is None


def test_grad_heatmap_epoch_dump(tmp_path):
    import numpy as np

    _spec, pe, latents = _square_inputs()
    a = _make_adapter("relational")
    a._grad_heatmap_every = 1
    cap = torch.randn(2, 1, 32, 32, 64, requires_grad=True)
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    a.extra_forwards(_ctx(), _primary(latents))

    ctx = types.SimpleNamespace(
        args=types.SimpleNamespace(output_dir=str(tmp_path), output_name="probe"),
        accelerator=types.SimpleNamespace(is_main_process=True),
        network=None,
        weight_dtype=torch.float32,
    )
    a.on_epoch_end(ctx)
    data = np.load(tmp_path / "probe_repa_grad_heatmap.npz")
    assert data["counts"].shape == (32, 32)
    assert int(data["n_samples"]) == 2
    assert data["counts"].sum() == pytest.approx(2 * round(0.10 * 32 * 32))
    assert float(data["concentration"]) >= 0.99


def test_factory_stamps_grad_heatmap():
    from networks.methods.easycontrol import create_network

    common = dict(vae=None, text_encoders=[], unet=None)
    net = create_network(1.0, 8, 8.0, **common, use_repa="true", repa_grad_heatmap="1")
    assert net._repa_grad_heatmap == pytest.approx(1.0)
    net_default = create_network(1.0, 8, 8.0, **common, use_repa="true")
    assert net_default._repa_grad_heatmap == 0.0


def test_repa_loss_handler_weighting():
    from library.training.losses import LossContext, _repa_loss

    pred = torch.zeros(2, 16, 1, 8, 8)
    base = dict(
        model_pred=pred,
        target=pred,
        timesteps=None,
        weighting=None,
        huber_c=None,
        loss_weights=None,
        batch={},
        args=None,
        is_train=True,
    )
    net = types.SimpleNamespace(_repa_weight=0.05)
    ctx = LossContext(network=net, aux={"repa": torch.tensor(2.0)}, **base)
    assert _repa_loss(ctx).item() == pytest.approx(0.1)
    # weight 0 → zero
    net0 = types.SimpleNamespace(_repa_weight=0.0)
    ctx0 = LossContext(network=net0, aux={"repa": torch.tensor(2.0)}, **base)
    assert _repa_loss(ctx0).item() == 0.0


# ------------------------------------------------------------------- REPA-DoG


def test_dog_reduces_to_spatial_norm_at_small_sigma1():
    """DoG at σ₁→0 (huge sigma1_div) is the DC-removal spatial_norm corner.

    ``Z − LP(Z, σ₁)`` with σ₁→0 → ``Z − Z = 0``; but with norm by spatial std
    the *direction field* matches spatial_norm's DC-removed field once the low
    band collapses to the DC mean. We check the looser invariant the proposal
    leans on: an aggressive low band-strip (small div) differs from DC-only.
    """
    b, n, d, gh, gw = 2, 1024, 768, 32, 32
    pe = torch.randn(b, n, d)
    # spatial_norm = DC removal: subtract per-channel token mean, /std.
    dc = (pe - pe.mean(dim=1, keepdim=True)) / (pe.std(dim=1, keepdim=True) + 1e-6)
    dog = dog_standardize(pe, gh, gw, sigma1_div=16.0)
    assert dog.shape == pe.shape
    assert torch.isfinite(dog).all()
    # A broad low-band strip removes more than DC alone → must differ.
    assert not torch.allclose(dog, dc, atol=1e-3)


def test_dog_bandpass_differs_from_highpass():
    """σ₂ on (band-pass) ≠ σ₂ off (high-pass): the inner kernel rolls off the
    very-high tail, so the two operators are distinct."""
    b, n, d, gh, gw = 2, 1024, 768, 32, 32
    pe = torch.randn(b, n, d)
    hp = dog_standardize(pe, gh, gw, sigma1_div=16.0, sigma2_div=0.0)
    bp = dog_standardize(pe, gh, gw, sigma1_div=16.0, sigma2_div=64.0)
    assert torch.isfinite(bp).all()
    assert not torch.allclose(hp, bp, atol=1e-3)


def test_dog_norm_std_fixed_vs_empirical():
    """norm_std>0 divides by a fixed constant (the paper's regime); norm_std=0
    by the empirical spatial std (matches spatial_norm). The two scale the field
    differently per-channel, so a fixed const changes the per-token directions."""
    b, n, d, gh, gw = 2, 1024, 768, 32, 32
    pe = torch.randn(b, n, d)
    emp = dog_standardize(pe, gh, gw, sigma1_div=16.0, norm_std=0.0)
    fixed = dog_standardize(pe, gh, gw, sigma1_div=16.0, norm_std=1.0)
    assert torch.isfinite(fixed).all()
    assert not torch.allclose(emp, fixed, atol=1e-3)


def test_dog_replaces_spatial_norm_in_adapter():
    """With _dog on the adapter band-passes the target then Gram-matches — it
    must equal hand-computed dog_standardize + relational_gram_loss(spatial_norm
    off), proving DoG slots in *instead of* the spatial_norm block."""
    from library.training.repa import (
        pool_dit_tokens_to_grid,
        relational_gram_loss,
    )

    _spec, pe, latents = _square_inputs()
    cap = torch.randn(2, 1, 32, 32, 64)

    a_dog = _make_adapter("relational")
    a_dog._dog = True
    a_dog._dog_sigma1_div = 16.0
    a_dog._captured, a_dog._pe_features, a_dog._latent_hw = cap, pe, (64, 64)
    loss_dog = a_dog.extra_forwards(_ctx(), _primary(latents))["repa"]
    assert torch.isfinite(loss_dog)

    # Hand-built reference: pool DiT to the 32×32 grid, DoG the CLS-dropped PE.
    dit_tok = pool_dit_tokens_to_grid(cap, (64, 64), 2, 32, 32)
    pe_dog = dog_standardize(pe[:, 1:, :].float(), 32, 32, 16.0)
    expected = relational_gram_loss(dit_tok, pe_dog, spatial_norm=False)
    assert torch.equal(loss_dog, expected)


def test_dog_off_is_bit_identical_to_relational():
    """Default-off DoG must not perturb the existing relational loss."""
    _spec, pe, latents = _square_inputs()
    cap = torch.randn(2, 1, 32, 32, 64)
    a = _make_adapter("relational")  # _dog defaults to False
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    loss_default = a.extra_forwards(_ctx(), _primary(latents))["repa"]

    tokens = cap.reshape(2, -1, 64)
    dit_grid = tokens.reshape(2, 32, 32, 64).permute(0, 3, 1, 2)
    dit_tok = (
        F.adaptive_avg_pool2d(dit_grid.float(), (32, 32)).flatten(2).transpose(1, 2)
    )
    dit_hat = F.normalize(dit_tok, dim=-1)
    pe_hat = F.normalize(pe[:, 1:, :].float(), dim=-1)
    g_dit = torch.bmm(dit_hat, dit_hat.transpose(1, 2))
    g_pe = torch.bmm(pe_hat, pe_hat.transpose(1, 2))
    expected = F.mse_loss(g_dit, g_pe)
    assert torch.equal(loss_default, expected)


def test_dog_grad_flows_to_captured():
    """The band-passed target is detached data, but grad still flows through the
    DiT-side pooled tokens (into the LoRA blocks)."""
    _spec, pe, latents = _square_inputs()
    cap = torch.randn(2, 1, 32, 32, 64, requires_grad=True)
    a = _make_adapter("relational")
    a._dog = True
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    loss = a.extra_forwards(_ctx(), _primary(latents))["repa"]
    loss.backward()
    assert cap.grad is not None and torch.isfinite(cap.grad).all()
    assert cap.grad.abs().sum() > 0


def test_factory_stamps_dog_levers():
    """Both factories stamp the DoG kwargs (default-off)."""
    import torch.nn as nn

    from networks.lora_anima.factory import create_network as lora_create
    from networks.methods.easycontrol import create_network as ec_create

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8, bias=False)

    class _DiT(nn.Module):
        model_channels = 8
        patch_spatial = 2

        def __init__(self):
            super().__init__()
            self.block = _Block()

    lora_common = dict(vae=None, text_encoders=[], unet=_DiT())
    net = lora_create(
        1.0,
        4,
        4.0,
        **lora_common,
        use_repa="true",
        repa_target_dog="true",
        repa_dog_sigma1_div="16",
    )
    assert net._repa_target_dog is True
    assert net._repa_dog_sigma1_div == pytest.approx(16.0)
    assert net._repa_dog_sigma2_div == 0.0

    net_default = lora_create(1.0, 4, 4.0, **lora_common, use_repa="true")
    assert net_default._repa_target_dog is False

    ec_common = dict(vae=None, text_encoders=[], unet=None)
    ec = ec_create(1.0, 8, 8.0, **ec_common, use_repa="true", repa_target_dog="true")
    assert ec._repa_target_dog is True
    ec_default = ec_create(1.0, 8, 8.0, **ec_common, use_repa="true")
    assert ec_default._repa_target_dog is False
