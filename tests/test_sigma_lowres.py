"""sigma_lowres Phase 1b invariants (σ>0.5 → 896 sibling latent).

Pins the load-bearing contracts of the trainer wiring:
  - the demote grid is a pure function shared by preprocess emit and trainer
    fetch (same inputs → same bucket), and off-route shapes return None;
  - the demoted npz key can never collide with the ``latents_*`` namespace
    (several readers grab the FIRST ``latents_*`` key);
  - ``draw_flat_sigmas`` is bit-identical to the in-body draw it was split
    from, and the σ-first two-step path reproduces the draw-inside path
    exactly (same seed → same noisy input / timesteps);
  - the preprocess emit appends the demoted key in-place, preserves every
    native key, is idempotent, and the dataset-side loader reads it back.
"""

from pathlib import Path
from types import SimpleNamespace

import random

import numpy as np
import pytest
import torch

from library.datasets.buckets import (
    SIGMA_DEMOTE_ROUTE,
    demote_bucket_for,
    demoted_token_counts,
    freefit_band_for_edge,
)
from library.io.cache_names import demoted_latents_key
from library.runtime.noise import draw_flat_sigmas, get_noisy_model_input_and_timesteps


def _args(**kw):
    base = dict(
        timestep_sampling="sigmoid",
        sigmoid_scale=1.0,
        discrete_flow_shift=3.0,
        ip_noise_gamma=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


_SCHED = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))


class TestDemoteBucket:
    def test_route_shapes_land_in_demote_band(self):
        lo, hi = freefit_band_for_edge(896)
        # The frozen top-5 1024-tier shapes (all 4032/4200-token).
        for w, h in [(896, 1200), (800, 1344), (1200, 896), (768, 1344), (896, 1152)]:
            bucket = demote_bucket_for(w, h, *SIGMA_DEMOTE_ROUTE)
            assert bucket is not None
            bw, bh = bucket
            tok = (bw // 16) * (bh // 16)
            assert lo <= tok <= hi
            # Aspect preserved to free-fit tolerance (sub-patch residual).
            assert abs(bw / bh - w / h) < 0.1

    def test_off_route_returns_none(self):
        # A native-896 shape (3024 tokens): trains as-is, no demote.
        assert demote_bucket_for(1008, 768, *SIGMA_DEMOTE_ROUTE) is None
        # 768-tier and 1280-tier shapes are off this route too.
        assert demote_bucket_for(768, 720, *SIGMA_DEMOTE_ROUTE) is None
        assert demote_bucket_for(1280, 1260, *SIGMA_DEMOTE_ROUTE) is None

    def test_deterministic(self):
        a = demote_bucket_for(896, 1200, *SIGMA_DEMOTE_ROUTE)
        b = demote_bucket_for(896, 1200, *SIGMA_DEMOTE_ROUTE)
        assert a == b

    def test_demoted_token_counts_only_route_members(self):
        resos = {(896, 1200), (1008, 768)}  # one 1024-tier, one 896-tier
        counts = demoted_token_counts(resos, *SIGMA_DEMOTE_ROUTE)
        lo, hi = freefit_band_for_edge(896)
        assert counts  # the 1024-tier member contributes
        assert all(lo <= c <= hi for c in counts)


class TestSigmaRoute:
    def test_default_and_custom(self):
        from train import AnimaTrainer

        assert AnimaTrainer._sigma_route(SimpleNamespace()) == (1024, 896)
        assert AnimaTrainer._sigma_route(SimpleNamespace(sigma_lowres_route=None)) == (
            1024,
            896,
        )
        assert AnimaTrainer._sigma_route(
            SimpleNamespace(sigma_lowres_route="1024:768")
        ) == (1024, 768)

    def test_bad_routes_rejected(self):
        from train import AnimaTrainer

        for bad in ("1024", "896:1024", "1024:0", "a:b", "1024:896:768"):
            with pytest.raises(ValueError):
                AnimaTrainer._sigma_route(SimpleNamespace(sigma_lowres_route=bad))

    def test_unsafe_768_bucket_derivable(self):
        # The E4 negative-control route must derive a 768-band sibling grid
        # for 1024-tier shapes (same pure function the emit uses).
        lo, hi = freefit_band_for_edge(768)
        bucket = demote_bucket_for(896, 1200, 1024, 768)
        assert bucket is not None
        bw, bh = bucket
        assert lo <= (bw // 16) * (bh // 16) <= hi


class TestDemotedKey:
    def test_never_in_latents_namespace(self):
        key = demoted_latents_key(880, 1184)
        assert not key.startswith("latents")
        assert key == "demoted_148x110"  # H//8 x W//8, native key convention


class TestSigmaDraw:
    @pytest.mark.parametrize("mode", ["sigmoid", "uniform", "shift"])
    def test_flat_draw_matches_inline_formula(self, mode):
        args = _args(timestep_sampling=mode)
        torch.manual_seed(7)
        got = draw_flat_sigmas(args, 4, 148, 110, torch.device("cpu"))
        torch.manual_seed(7)
        if mode == "sigmoid":
            want = torch.sigmoid(1.0 * torch.randn((4,)) + 0.0)
        elif mode == "uniform":
            want = torch.rand((4,))
        else:
            s = torch.sigmoid(torch.randn(4) * 1.0 + 0.0)
            want = (s * 3.0) / (1 + (3.0 - 1) * s)
        assert torch.equal(got, want)

    def test_density_modes_return_none(self):
        args = _args(timestep_sampling="something_else")
        assert draw_flat_sigmas(args, 4, 148, 110, torch.device("cpu")) is None

    def test_sigma_first_path_is_bit_exact(self):
        """draw σ → pass in ≡ draw-inside, given the same seed (RNG order is
        preserved because the split-out helper is the body's first RNG use)."""
        args = _args()
        latents = torch.randn(2, 16, 148, 110)
        noise = torch.randn_like(latents)

        torch.manual_seed(11)
        noisy_a, t_a, sig_a = get_noisy_model_input_and_timesteps(
            args, _SCHED, latents, noise, torch.device("cpu"), torch.float32
        )
        torch.manual_seed(11)
        pre = draw_flat_sigmas(args, 2, 148, 110, torch.device("cpu"))
        noisy_b, t_b, sig_b = get_noisy_model_input_and_timesteps(
            args,
            _SCHED,
            latents,
            noise,
            torch.device("cpu"),
            torch.float32,
            sigmas=pre,
        )
        assert torch.equal(noisy_a, noisy_b)
        assert torch.equal(t_a, t_b)
        assert torch.equal(sig_a, sig_b)


class TestEmitAndLoad:
    @pytest.fixture()
    def corpus(self, tmp_path: Path):
        """One 1024-tier resized PNG + its native npz (with extra keys)."""
        from PIL import Image

        w, h = 896, 1200  # 4200 tokens — 1024 tier
        img_dir = tmp_path / "resized" / "artist"
        img_dir.mkdir(parents=True)
        Image.new("RGB", (w, h), (128, 64, 32)).save(img_dir / "img1.png")

        cache_dir = tmp_path / "lora"
        npz_dir = cache_dir / "artist"
        npz_dir.mkdir(parents=True)
        npz_path = npz_dir / f"img1_{w:04d}x{h:04d}_anima.npz"
        native = {
            f"latents_{h // 8}x{w // 8}": np.zeros((16, h // 8, w // 8), np.float32),
            f"original_size_{h // 8}x{w // 8}": np.array([w, h]),
            f"crop_ltrb_{h // 8}x{w // 8}": np.array([0, 0, w, h]),
        }
        np.savez(npz_path, **native)
        return SimpleNamespace(
            data_dir=tmp_path / "resized",
            cache_dir=cache_dir,
            npz_path=npz_path,
            native_keys=set(native),
            wh=(w, h),
        )

    @pytest.fixture()
    def stub_vae(self):
        class _V:
            device = torch.device("cpu")
            dtype = torch.float32

            def encode_pixels_to_latents(self, px):
                return torch.ones(px.shape[0], 16, px.shape[-2] // 8, px.shape[-1] // 8)

        return _V()

    def test_emit_appends_preserves_and_idempotent(self, corpus, stub_vae):
        from library.preprocess.latents import (
            cache_demoted_latents,
            count_pending_demoted,
        )

        pending, eligible = count_pending_demoted(
            corpus.data_dir,
            native_edge=SIGMA_DEMOTE_ROUTE[0],
            demote_edge=SIGMA_DEMOTE_ROUTE[1],
            cache_dir=corpus.cache_dir,
            recursive=True,
        )
        assert (pending, eligible) == (1, 1)

        stats = cache_demoted_latents(
            corpus.data_dir,
            stub_vae,
            native_edge=SIGMA_DEMOTE_ROUTE[0],
            demote_edge=SIGMA_DEMOTE_ROUTE[1],
            cache_dir=corpus.cache_dir,
            recursive=True,
        )
        assert stats.written == 1 and stats.failed == 0

        bucket = demote_bucket_for(*corpus.wh, *SIGMA_DEMOTE_ROUTE)
        key = demoted_latents_key(*bucket)
        with np.load(corpus.npz_path) as npz:
            assert set(npz.files) == corpus.native_keys | {key}
            assert npz[key].shape == (16, bucket[1] // 8, bucket[0] // 8)

        # Idempotent: second pass skips.
        stats2 = cache_demoted_latents(
            corpus.data_dir,
            stub_vae,
            native_edge=SIGMA_DEMOTE_ROUTE[0],
            demote_edge=SIGMA_DEMOTE_ROUTE[1],
            cache_dir=corpus.cache_dir,
            recursive=True,
        )
        assert stats2.written == 0 and stats2.skipped == 1

    def test_dataset_loader_roundtrip(self, corpus, stub_vae):
        from library.datasets.base import BaseDataset
        from library.preprocess.latents import cache_demoted_latents

        cache_demoted_latents(
            corpus.data_dir,
            stub_vae,
            native_edge=SIGMA_DEMOTE_ROUTE[0],
            demote_edge=SIGMA_DEMOTE_ROUTE[1],
            cache_dir=corpus.cache_dir,
            recursive=True,
        )
        ds = BaseDataset(network_multiplier=1.0, debug_dataset=False)
        info = SimpleNamespace(latents_npz=str(corpus.npz_path), bucket_reso=corpus.wh)

        # Disabled → None (sidecar inert).
        assert ds._try_load_demoted_latent(info) is None

        ds.enable_sigma_demote(*SIGMA_DEMOTE_ROUTE)
        lat = ds._try_load_demoted_latent(info)
        bucket = demote_bucket_for(*corpus.wh, *SIGMA_DEMOTE_ROUTE)
        assert lat is not None and lat.dtype == torch.float32
        assert lat.shape == (16, bucket[1] // 8, bucket[0] // 8)

        # Off-route image → None even when enabled.
        off = SimpleNamespace(latents_npz=str(corpus.npz_path), bucket_reso=(1008, 768))
        assert ds._try_load_demoted_latent(off) is None


class TestPairedStepRng:
    """--paired_step_rng (CRN): σ/noise decoupled from the global stream so
    A/B arms with the same seed stay noise-locked."""

    def test_generator_draw_ignores_global_stream(self):
        args = _args()
        g1 = torch.Generator().manual_seed(123)
        torch.manual_seed(0)
        a = draw_flat_sigmas(args, 4, 148, 110, torch.device("cpu"), generator=g1)
        g2 = torch.Generator().manual_seed(123)
        torch.manual_seed(999)  # different global state must not matter
        torch.randn(1000)  # ...nor global consumption
        b = draw_flat_sigmas(args, 4, 148, 110, torch.device("cpu"), generator=g2)
        assert torch.equal(a, b)

    def test_two_arms_share_sigma_and_noise_sequences(self):
        """Simulate two arms: same (seed, counter) derivation → identical σ
        per step and identical native-shape noise, regardless of what each
        arm did to the global stream in between."""
        args = _args()

        def arm_step(counter, global_junk):
            torch.randn(global_junk)  # arm-specific global-stream consumption
            base = (42 * 1_000_003 + counter) * 2
            mask = (1 << 62) - 1
            g_s = torch.Generator().manual_seed(base & mask)
            g_n = torch.Generator().manual_seed((base + 1) & mask)
            sig = draw_flat_sigmas(
                args, 1, 148, 110, torch.device("cpu"), generator=g_s
            )
            noise = torch.randn((1, 16, 152, 108), generator=g_n)
            return sig, noise

        for step in (1, 2, 3):
            s_a, n_a = arm_step(step, global_junk=7)
            s_b, n_b = arm_step(step, global_junk=3001)
            assert torch.equal(s_a, s_b)
            assert torch.equal(n_a, n_b)


class TestYarnsigRope:
    """--sigma_lowres_yarnsig: σ-gated YaRN banded rope on demoted steps.

    Pins the two reduction identities the probe's read rests on (μ→0 ⇒
    native integer spacing, i.e. the intervention vanishes below the gate;
    all-bands-below-α ⇒ the uniform PI stretch) plus the μ gate formula and
    the flag's parse/guard behavior.
    """

    @staticmethod
    def _pe():
        from library.anima.models import VideoRopePosition3DEmb

        return VideoRopePosition3DEmb(
            model_channels=1024, len_h=128, len_w=128, len_t=8, head_dim=128
        )

    # The live route's top free-fit bucket: (152,108) native latent →
    # (130,92) demoted, patch grids (76,54) → (65,46).
    _SHAPE = torch.Size([1, 1, 65, 46, 1024])
    _HS, _WS = 76 / 65, 54 / 46

    def test_mu_zero_reduces_to_native(self):
        pe = self._pe()
        native = pe.generate_embeddings(self._SHAPE)
        yarn = pe.generate_embeddings_yarn(self._SHAPE, self._HS, self._WS, 1, 4, 0.0)
        for a, b in zip(yarn, native):
            assert torch.equal(a, b)

    def test_all_bands_full_stretch_reduces_to_uniform_pi(self):
        pe = self._pe()
        scaled = pe.generate_embeddings_scaled(
            self._SHAPE, h_scale=self._HS, w_scale=self._WS
        )
        yarn = pe.generate_embeddings_yarn(
            self._SHAPE, self._HS, self._WS, 1e9, 2e9, 1.0
        )
        for a, b in zip(yarn, scaled):
            # equal up to float association: (seq·s)⊗f vs seq⊗(f·s)
            assert torch.allclose(a, b, atol=1e-5)

    def test_static_yarn_between_native_and_pi(self):
        pe = self._pe()
        native = pe.generate_embeddings(self._SHAPE)
        yarn = pe.generate_embeddings_yarn(self._SHAPE, self._HS, self._WS, 1, 4, 1.0)
        assert not torch.allclose(yarn[0], native[0], atol=1e-4)

    def test_mu_gate_formula(self):
        from train import AnimaTrainer

        args = SimpleNamespace(sigma_lowres_yarnsig="1,4,0.35,2")
        alpha, beta, center, gamma = AnimaTrainer._yarnsig_params(args)
        assert (alpha, beta, center, gamma) == (1.0, 4.0, 0.35, 2.0)

        def mu(s):
            import math

            return 1.0 / (
                1.0
                + math.exp(
                    -gamma
                    * (math.log(s / (1.0 - s)) - math.log(center / (1.0 - center)))
                )
            )

        assert abs(mu(0.35) - 0.5) < 1e-12  # center is the half-gate point
        assert abs(mu(0.21) - 0.20) < 0.02  # probe's reported bin values
        assert abs(mu(0.59) - 0.88) < 0.02
        assert mu(0.999) > 0.99

    def test_bad_params_rejected(self):
        from train import AnimaTrainer

        for bad in ("1,4", "4,1,0.35,2", "1,4,1.5,2", "1,4,0.35,0"):
            with pytest.raises(ValueError):
                AnimaTrainer._yarnsig_params(SimpleNamespace(sigma_lowres_yarnsig=bad))
        assert (
            AnimaTrainer._yarnsig_params(SimpleNamespace(sigma_lowres_yarnsig=None))
            is None
        )


class TestSigmaSpan:
    """--sigma_lowres_span (E16 placement probe): step-span gate on top of
    the σ gate. Pins the parse/guard behavior, the exact early/late
    partition of the train-forward range (identical demoted mass ⇒ any ΔW
    ordering is pure placement signal), and the spread coin's determinism
    in (--seed, step) alone — it must touch no RNG stream.
    """

    @staticmethod
    def _args(span, total=480, seed=1001, accum=1):
        return SimpleNamespace(
            sigma_lowres_span=span,
            max_train_steps=total,
            gradient_accumulation_steps=accum,
            seed=seed,
        )

    def test_parse_defaults_and_custom(self):
        from train import AnimaTrainer

        assert AnimaTrainer._sigma_span_params(SimpleNamespace()) is None
        assert (
            AnimaTrainer._sigma_span_params(SimpleNamespace(sigma_lowres_span=None))
            is None
        )
        assert AnimaTrainer._sigma_span_params(self._args("early")) == ("early", 0.5)
        assert AnimaTrainer._sigma_span_params(self._args("late:0.25")) == (
            "late",
            0.25,
        )
        assert AnimaTrainer._sigma_span_params(self._args("spread:0.5")) == (
            "spread",
            0.5,
        )

    def test_bad_specs_rejected(self):
        from train import AnimaTrainer

        for bad in ("first", "early:0", "late:1", "spread:x", "early:0.5:0.5"):
            with pytest.raises(ValueError):
                AnimaTrainer._sigma_span_params(self._args(bad))

    def test_needs_finalized_total(self):
        from train import AnimaTrainer

        with pytest.raises(ValueError):
            AnimaTrainer._sigma_span_allows(self._args("early", total=0), 1)

    def test_none_allows_all(self):
        from train import AnimaTrainer

        args = self._args(None)
        assert all(AnimaTrainer._sigma_span_allows(args, i) for i in range(1, 481))

    def test_early_late_partition_exact(self):
        """early:0.5 and late:0.5 partition [1, T] exactly — same mass,
        disjoint, exhaustive — for even and odd T."""
        from train import AnimaTrainer

        for total in (480, 481):
            early = self._args("early", total=total)
            late = self._args("late", total=total)
            e = {
                i
                for i in range(1, total + 1)
                if AnimaTrainer._sigma_span_allows(early, i)
            }
            lt = {
                i
                for i in range(1, total + 1)
                if AnimaTrainer._sigma_span_allows(late, i)
            }
            b = round(0.5 * total)
            assert e == set(range(1, b + 1))
            assert lt == set(range(b + 1, total + 1))
            assert e | lt == set(range(1, total + 1))
            assert not (e & lt)
            assert abs(len(e) - len(lt)) <= 1  # identical mass (±1 at odd T)

    def test_grad_accum_scales_total(self):
        from train import AnimaTrainer

        args = self._args("early", total=480, accum=2)  # 960 forwards
        assert AnimaTrainer._sigma_span_allows(args, 480)
        assert not AnimaTrainer._sigma_span_allows(args, 481)

    def test_spread_coin_seed_keyed_and_stream_free(self):
        """Same seed → identical coin sequence regardless of global RNG
        state/consumption (CRN pairing); different seed → different set;
        empirical rate ≈ p."""
        from train import AnimaTrainer

        args_a = self._args("spread")
        torch.manual_seed(0)
        random.seed(0)
        seq_a = [AnimaTrainer._sigma_span_allows(args_a, i) for i in range(1, 481)]

        args_b = self._args("spread")
        torch.manual_seed(7)
        random.seed(999)
        random.random()  # global consumption must not matter
        seq_b = [AnimaTrainer._sigma_span_allows(args_b, i) for i in range(1, 481)]
        assert seq_a == seq_b

        args_c = self._args("spread", seed=1002)
        seq_c = [AnimaTrainer._sigma_span_allows(args_c, i) for i in range(1, 481)]
        assert seq_c != seq_a

        rate = sum(seq_a) / len(seq_a)
        assert 0.4 < rate < 0.6


class TestSigmaWindow:
    """--sigma_lowres_threshold_max: the half-line σ gate becomes a window.
    Pins the gate arithmetic the E16 win768 arm (0.65 < σ < 0.95) rests on."""

    @staticmethod
    def _gate(sigmas, lo, hi):
        from train import AnimaTrainer

        args = SimpleNamespace(sigma_lowres_threshold=lo, sigma_lowres_threshold_max=hi)
        return AnimaTrainer._sigma_gate_allows(args, sigmas)

    def test_window_gate(self):
        lo, hi = 0.65, 0.95
        assert self._gate(torch.tensor([0.7]), lo, hi)
        assert self._gate(torch.tensor([0.94]), lo, hi)
        assert not self._gate(torch.tensor([0.5]), lo, hi)  # below window
        assert not self._gate(torch.tensor([0.96]), lo, hi)  # above window
        assert not self._gate(torch.tensor([1.0]), lo, hi)  # endpoint excluded
        # all-samples rule: one out-of-window sample blocks the batch
        assert not self._gate(torch.tensor([0.7, 0.96]), lo, hi)
        # no upper bound = the shipped half-line gate
        assert self._gate(torch.tensor([0.96]), lo, None)


class TestSigmaStackedRouter:
    """--sigma_lowres_route2 (E16 combo arm): 768 if sigma in its window,
    elif sigma>0.5 -> 896, else native. Pins rule-2 priority, per-rule
    gate/span independence, and that rule-1-only behavior is unchanged."""

    @staticmethod
    def _combo_args(**kw):
        base = dict(
            sigma_lowres_threshold=0.5,
            sigma_lowres_route2="1024:768",
            sigma_lowres_threshold2=0.65,
            sigma_lowres_threshold2_max=0.95,
            max_train_steps=480,
            gradient_accumulation_steps=1,
            seed=1001,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_route2_parse(self):
        from train import AnimaTrainer

        assert AnimaTrainer._sigma_route2(SimpleNamespace()) is None
        assert (
            AnimaTrainer._sigma_route2(SimpleNamespace(sigma_lowres_route2=None))
            is None
        )
        assert AnimaTrainer._sigma_route2(
            SimpleNamespace(sigma_lowres_route2="1024:768")
        ) == (1024, 768)
        for bad in ("1024", "768:1024", "a:b"):
            with pytest.raises(ValueError):
                AnimaTrainer._sigma_route2(SimpleNamespace(sigma_lowres_route2=bad))

    def test_combo_choice_matches_spec(self):
        """768 if sigma in (0.65, 0.95); elif sigma > 0.5 -> 896; else native."""
        from train import AnimaTrainer

        args = self._combo_args()
        cases = {
            0.70: 2,  # in window -> deep route
            0.94: 2,
            0.55: 1,  # above primary threshold, outside window -> 896
            0.96: 1,  # above window top -> falls back to primary
            0.99: 1,
            0.30: None,  # below both -> native
            0.50: None,  # primary gate is strict >
        }
        for sigma, want in cases.items():
            got = AnimaTrainer._sigma_demote_choice(args, torch.tensor([sigma]), 1)
            assert got == want, (sigma, got, want)

    def test_rule1_only_unchanged(self):
        from train import AnimaTrainer

        args = self._combo_args(sigma_lowres_route2=None)
        assert AnimaTrainer._sigma_demote_choice(args, torch.tensor([0.7]), 1) == 1
        assert AnimaTrainer._sigma_demote_choice(args, torch.tensor([0.3]), 1) is None

    def test_per_rule_spans_independent(self):
        """rule 2 late-gated while rule 1 has no span: an in-window sigma in
        the first half falls back to the primary rule."""
        from train import AnimaTrainer

        args = self._combo_args(sigma_lowres_span2="late")
        sig = torch.tensor([0.7])
        assert AnimaTrainer._sigma_demote_choice(args, sig, 100) == 1
        assert AnimaTrainer._sigma_demote_choice(args, sig, 300) == 2
