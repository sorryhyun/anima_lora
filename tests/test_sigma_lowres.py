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

import math
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

    def test_default_on_with_sigma_lowres(self):
        """yarnsig is part of the shipped combo recipe, so an unset flag under
        --sigma_lowres resolves to the operating point — a plain --sigma_lowres
        run must not silently be a degraded (no-yarnsig) arm."""
        from train import AnimaTrainer

        args = SimpleNamespace(sigma_lowres=True, sigma_lowres_yarnsig=None)
        assert AnimaTrainer._yarnsig_params(args) == (1.0, 4.0, 0.35, 2.0)
        # …and the effective value is written back, so the snapshot records it.
        assert args.sigma_lowres_yarnsig == AnimaTrainer.YARNSIG_OPERATING_POINT

    def test_default_off_without_sigma_lowres(self):
        """Nothing to demote → nothing to realign; the implicit default must not
        leak into runs that never enabled sigma_lowres (train.py errors on
        yarnsig-without-sigma_lowres)."""
        from train import AnimaTrainer

        args = SimpleNamespace(sigma_lowres=False, sigma_lowres_yarnsig=None)
        assert AnimaTrainer._yarnsig_params(args) is None
        assert args.sigma_lowres_yarnsig is None

    def test_explicit_off_words_disable_it(self):
        """The escape hatch from default-on, and the reason a config may carry
        `sigma_lowres_yarnsig = "off"` without tripping the requires-sigma_lowres
        guard."""
        from train import AnimaTrainer

        for off in ("off", "OFF", "none", "false", "0", ""):
            args = SimpleNamespace(sigma_lowres=True, sigma_lowres_yarnsig=off)
            assert AnimaTrainer._yarnsig_params(args) is None, off

    def test_explicit_value_still_wins(self):
        from train import AnimaTrainer

        args = SimpleNamespace(sigma_lowres=True, sigma_lowres_yarnsig="2,6,0.4,3")
        assert AnimaTrainer._yarnsig_params(args) == (2.0, 6.0, 0.4, 3.0)


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


class TestSigmaRuleValidation:
    """--sigma_lowres_route2 must be given an explicit sigma WINDOW, and every
    span spec must parse, at SETUP — not on the first train forward.

    Rule 2 has priority, so an unbounded rule 2 fires wherever it passes and
    the primary rule never runs. With threshold2 defaulting to 0.5 that made a
    lone --sigma_lowres_route2 1024:768 demote the deep route across the whole
    sigma>0.5 half-line (including the off-map stretch below its certified
    window) while silently disabling yarnsig, which is primary-rule-only.
    """

    @staticmethod
    def _args(**kw):
        base = dict(
            sigma_lowres=True,
            sigma_lowres_route="1024:896",
            sigma_lowres_threshold=0.5,
            sigma_lowres_threshold_max=None,
            sigma_lowres_span=None,
            sigma_lowres_route2=None,
            sigma_lowres_threshold2=None,
            sigma_lowres_threshold2_max=None,
            sigma_lowres_span2=None,
            sigma_lowres_yarnsig=None,
            max_train_steps=480,
            gradient_accumulation_steps=1,
            seed=1001,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_route2_without_window_rejected(self):
        from train import AnimaTrainer

        with pytest.raises(ValueError) as e:
            AnimaTrainer._validate_sigma_rules(
                self._args(sigma_lowres_route2="1024:768")
            )
        msg = str(e.value)
        assert "--sigma_lowres_threshold2" in msg
        assert "--sigma_lowres_threshold2_max" in msg

    @pytest.mark.parametrize(
        "lo,hi",
        [(0.65, None), (None, 0.95)],
    )
    def test_route2_with_half_a_window_rejected(self, lo, hi):
        from train import AnimaTrainer

        with pytest.raises(ValueError):
            AnimaTrainer._validate_sigma_rules(
                self._args(
                    sigma_lowres_route2="1024:768",
                    sigma_lowres_threshold2=lo,
                    sigma_lowres_threshold2_max=hi,
                )
            )

    @pytest.mark.parametrize("lo,hi", [(0.95, 0.65), (0.7, 0.7), (-0.1, 0.9)])
    def test_route2_inverted_window_rejected(self, lo, hi):
        from train import AnimaTrainer

        with pytest.raises(ValueError):
            AnimaTrainer._validate_sigma_rules(
                self._args(
                    sigma_lowres_route2="1024:768",
                    sigma_lowres_threshold2=lo,
                    sigma_lowres_threshold2_max=hi,
                )
            )

    def test_shipped_combo_recipe_validates(self):
        from train import AnimaTrainer

        route, route2 = AnimaTrainer._validate_sigma_rules(
            self._args(
                sigma_lowres_route2="1024:768",
                sigma_lowres_threshold2=0.65,
                sigma_lowres_threshold2_max=0.95,
                sigma_lowres_yarnsig="",
            )
        )
        assert route == (1024, 896) and route2 == (1024, 768)

    def test_rule1_alone_needs_no_window(self):
        from train import AnimaTrainer

        assert AnimaTrainer._validate_sigma_rules(self._args()) == ((1024, 896), None)

    def test_primary_inverted_window_rejected(self):
        from train import AnimaTrainer

        with pytest.raises(ValueError):
            AnimaTrainer._validate_sigma_rules(
                self._args(sigma_lowres_threshold=0.8, sigma_lowres_threshold_max=0.6)
            )

    @pytest.mark.parametrize("attr", ["sigma_lowres_span", "sigma_lowres_span2"])
    def test_bad_span_raises_at_setup_not_first_step(self, attr):
        """A typo'd span used to raise inside _maybe_sigma_demote, i.e. after
        model load, adapter attach and caching."""
        from train import AnimaTrainer

        kw = {attr: "latte"}
        if attr == "sigma_lowres_span2":
            kw.update(
                sigma_lowres_route2="1024:768",
                sigma_lowres_threshold2=0.65,
                sigma_lowres_threshold2_max=0.95,
            )
        with pytest.raises(ValueError, match="early|late|spread"):
            AnimaTrainer._validate_sigma_rules(self._args(**kw))

    def test_rule_cfg_reads_each_rules_own_flags(self):
        from train import AnimaTrainer

        args = self._args(
            sigma_lowres_threshold=0.5,
            sigma_lowres_threshold_max=None,
            sigma_lowres_span="late",
            sigma_lowres_threshold2=0.65,
            sigma_lowres_threshold2_max=0.95,
            sigma_lowres_span2="early:0.3",
        )
        assert AnimaTrainer._sigma_rule_cfg(args, 1) == (0.5, None, "late")
        assert AnimaTrainer._sigma_rule_cfg(args, 2) == (0.65, 0.95, "early:0.3")
        with pytest.raises(ValueError):
            AnimaTrainer._sigma_rule_cfg(args, 3)


class TestSigmaSpanResumeOffset:
    """The span gate counts train FORWARDS, so a resumed run must start the
    counter at the resume offset — otherwise late:f is measured from the resume
    point against an absolute T and demotes over a shrunken tail."""

    def test_late_span_boundary_is_absolute_after_resume(self):
        from train import AnimaTrainer

        args = SimpleNamespace(
            sigma_lowres_span="late:0.5",
            max_train_steps=480,
            gradient_accumulation_steps=1,
            seed=0,
        )
        # T=480, late:0.5 -> demote for step_idx > 240, whatever the offset.
        assert AnimaTrainer._sigma_span_allows(args, 240) is False
        assert AnimaTrainer._sigma_span_allows(args, 241) is True

    @pytest.mark.parametrize("resume_at", [0, 120, 300])
    def test_resumed_run_demotes_the_same_absolute_tail(self, resume_at):
        """Replay the trainer's counter for a run resumed at ``resume_at``
        forwards and assert the demoted set is the same absolute tail a fresh
        run would produce. With a 0-based counter, resume_at=300 under
        late:0.5 demoted from global 540 (past the end) instead of 241.
        """
        from train import AnimaTrainer

        args = SimpleNamespace(
            sigma_lowres_span="late:0.5",
            max_train_steps=480,
            gradient_accumulation_steps=1,
            seed=0,
        )
        # The loop pre-seeds the counter with the resume offset.
        counter = resume_at
        demoted = []
        for _ in range(480 - resume_at):
            counter += 1
            if AnimaTrainer._sigma_span_allows(args, counter):
                demoted.append(counter)

        expected = [i for i in range(resume_at + 1, 481) if i > 240]
        assert demoted == expected


class TestPerRouteWarnAndNpzMemo:
    """A single shared warn flag let a stale image on one route swallow the
    warning for a wholly un-emitted second route — the stacked router would
    then quietly degrade to the primary rule, whose only symptom is a few
    points of wall-clock."""

    def test_missing_key_warns_once_per_route(self, caplog):
        import logging

        from library.datasets.base import BaseDataset

        ds = BaseDataset(network_multiplier=1.0, debug_dataset=False)
        ds.enable_sigma_demote(1024, 896)
        ds.enable_sigma_demote2(1024, 768)

        # An npz with NEITHER route's sibling key.
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            npz_path = Path(td) / "img1_0896x1200_anima.npz"
            np.savez(npz_path, latents_150x112=np.zeros((16, 150, 112), np.float32))
            info = SimpleNamespace(latents_npz=str(npz_path), bucket_reso=(896, 1200))

            with caplog.at_level(logging.WARNING):
                assert ds._try_load_demoted_latent(info) is None
                assert ds._try_load_demoted_latent2(info) is None
                # Second visit: no new warnings.
                assert ds._try_load_demoted_latent(info) is None
                assert ds._try_load_demoted_latent2(info) is None

        warns = [r.message for r in caplog.records if "sigma_lowres" in r.message]
        assert len(warns) == 2, f"expected one warning per route, got {warns}"
        assert any("896" in w for w in warns) and any("768" in w for w in warns)

    def test_npz_opened_once_for_both_routes(self, monkeypatch):
        """Both sidecar loaders run back-to-back for the same image; the memo
        keeps that to one archive open instead of one per route."""
        import numpy as _np

        from library.datasets.base import BaseDataset

        ds = BaseDataset(network_multiplier=1.0, debug_dataset=False)
        ds.enable_sigma_demote(1024, 896)
        ds.enable_sigma_demote2(1024, 768)

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            npz_path = Path(td) / "img1_0896x1200_anima.npz"
            b896 = demote_bucket_for(896, 1200, 1024, 896)
            b768 = demote_bucket_for(896, 1200, 1024, 768)
            payload = {
                "latents_150x112": _np.zeros((16, 150, 112), _np.float32),
                demoted_latents_key(*b896): _np.zeros(
                    (16, b896[1] // 8, b896[0] // 8), _np.float32
                ),
                demoted_latents_key(*b768): _np.zeros(
                    (16, b768[1] // 8, b768[0] // 8), _np.float32
                ),
            }
            _np.savez(npz_path, **payload)
            info = SimpleNamespace(latents_npz=str(npz_path), bucket_reso=(896, 1200))

            opens = []
            real_load = _np.load
            monkeypatch.setattr(
                _np,
                "load",
                lambda p, *a, **k: (opens.append(p), real_load(p, *a, **k))[1],
            )
            assert ds._try_load_demoted_latent(info) is not None
            assert ds._try_load_demoted_latent2(info) is not None
            assert len(opens) == 1, (
                f"expected one npz open for both routes, got {opens}"
            )

    def test_memo_dropped_on_pickle(self):
        """An open NpzFile is unpicklable and the dataset is shipped to
        DataLoader workers under Windows/spawn."""
        import pickle
        import tempfile

        from library.datasets.base import BaseDataset

        ds = BaseDataset(network_multiplier=1.0, debug_dataset=False)
        with tempfile.TemporaryDirectory() as td:
            npz_path = Path(td) / "x.npz"
            np.savez(npz_path, a=np.zeros(1, np.float32))
            ds._open_demote_npz(str(npz_path))
            assert ds._demote_npz_cache is not None
            assert pickle.loads(pickle.dumps(ds))._demote_npz_cache is None


class TestResCond:
    """E25b --sigma_lowres_res_cond Stage-0 invariants (frozen design,
    e25b README): zero-init ⇒ exactly-zero t-embedding delta (the identity
    invariant), non-degeneracy once trained (so the identity test can't
    pass vacuously), the known-input route scalar, merge refusal (the
    projection is not a ΔW), and the probe's control-checkpoint hard fail
    + always-frozen contract (flat grad-vector layout parity with a
    control twin)."""

    _T = torch.zeros(2, 1)  # timesteps_B_T stand-in: batch 2, T=1

    def test_zero_init_delta_is_exactly_zero(self):
        from library.anima.models import sigma_lowres_res_cond_delta

        proj = torch.zeros(64, 16)
        for s in (0.0, math.log2(896 / 1024), math.log2(768 / 1024)):
            delta = sigma_lowres_res_cond_delta(proj, s, self._T)
            assert delta.shape == (2, 1, 64)
            assert torch.equal(delta, torch.zeros_like(delta))
        # bit-exact identity on the summation point itself
        t_emb = torch.randn(2, 1, 64)
        delta = sigma_lowres_res_cond_delta(proj, -0.415, self._T)
        assert torch.equal(t_emb + delta, t_emb)

    def test_trained_proj_is_not_degenerate(self):
        """Companion guard (yarnsig test-suite convention): with a nonzero
        projection the delta is nonzero — including at s = 0 (native steps
        run the module too) — and the two shipped routes are distinguished."""
        from library.anima.models import sigma_lowres_res_cond_delta

        torch.manual_seed(0)
        proj = torch.randn(64, 16)
        d_native = sigma_lowres_res_cond_delta(proj, 0.0, self._T)
        d_896 = sigma_lowres_res_cond_delta(proj, math.log2(896 / 1024), self._T)
        d_768 = sigma_lowres_res_cond_delta(proj, math.log2(768 / 1024), self._T)
        assert not torch.allclose(d_native, torch.zeros_like(d_native))
        assert not torch.allclose(d_896, d_768)
        assert not torch.allclose(d_native, d_896)

    def test_route_scalar_is_the_config_fact(self):
        from train import AnimaTrainer

        args = _args(sigma_lowres_route="1024:896", sigma_lowres_route2="1024:768")
        assert AnimaTrainer._res_cond_scalar(args, None) == 0.0
        assert AnimaTrainer._res_cond_scalar(args, 1) == pytest.approx(
            math.log2(896 / 1024)
        )
        assert AnimaTrainer._res_cond_scalar(args, 2) == pytest.approx(
            math.log2(768 / 1024)
        )

    def test_merge_refuses_res_cond_proj(self):
        from library.anima.merge import scan_non_bakeable_keys

        found = scan_non_bakeable_keys(
            {"sigma_lowres_res_cond_proj": torch.zeros(64, 16)}
        )
        assert any("res-cond" in kind for kind in found)

    def test_probe_control_checkpoint_hard_fails(self):
        from project.sigma_lowres.bench.sigma_probe.kernel import resolve_res_cond

        control = SimpleNamespace()  # no sigma_lowres_res_cond_proj attribute
        with pytest.raises(ValueError, match="control checkpoint"):
            resolve_res_cond(control, enabled=True)

    def test_probe_freezes_proj_regardless_of_flag(self):
        from project.sigma_lowres.bench.sigma_probe.kernel import resolve_res_cond

        for enabled in (False, True):
            nw = SimpleNamespace(
                sigma_lowres_res_cond_proj=torch.nn.Parameter(torch.zeros(64, 16))
            )
            out = resolve_res_cond(nw, enabled=enabled)
            assert nw.sigma_lowres_res_cond_proj.requires_grad is False
            assert (out is nw.sigma_lowres_res_cond_proj) if enabled else (out is None)

    # --- Stage 2.0: inference-side attach (every keep-live/static-merge route
    # filters the state dict to lora_unet_*, so the loader installs the
    # projection separately; e25b README "Stage 2.0") ---

    @staticmethod
    def _ckpt(tmp_path, name, tensors, metadata=None):
        from safetensors.torch import save_file

        p = tmp_path / name
        save_file(tensors, str(p), metadata=metadata)
        return str(p)

    def test_inference_attach_installs_proj_at_s0(self, tmp_path):
        from library.anima.models import sigma_lowres_res_cond_delta
        from library.inference.models import _attach_res_cond

        torch.manual_seed(0)
        proj = torch.randn(64, 16)
        path = self._ckpt(
            tmp_path,
            "rescond.safetensors",
            {"sigma_lowres_res_cond_proj": proj, "lora_unet_x": torch.zeros(1)},
            metadata={"ss_sigma_lowres_res_cond": "true"},
        )
        model = SimpleNamespace()
        _attach_res_cond(model, [path], torch.device("cpu"))
        got, s, centered = model._sigma_lowres_res_cond
        assert s == 0.0 and got.dtype == torch.float32 and centered is False
        assert torch.equal(got, proj)
        # inference delta == the trainer's native-step (s=0) delta, bit-exact
        assert torch.equal(
            sigma_lowres_res_cond_delta(got, s, self._T),
            sigma_lowres_res_cond_delta(proj, 0.0, self._T),
        )
        # and it is a nonzero offset — dropping it would change the forward
        assert not torch.equal(
            sigma_lowres_res_cond_delta(got, s, self._T),
            torch.zeros(2, 1, 64),
        )

    def test_inference_attach_noop_on_control(self, tmp_path):
        from library.inference.models import _attach_res_cond

        path = self._ckpt(
            tmp_path, "control.safetensors", {"lora_unet_x": torch.zeros(1)}
        )
        model = SimpleNamespace()
        _attach_res_cond(model, [path], torch.device("cpu"))
        assert not hasattr(model, "_sigma_lowres_res_cond")

    def test_inference_attach_stamp_without_key_hard_fails(self, tmp_path):
        from library.inference.models import _attach_res_cond

        path = self._ckpt(
            tmp_path,
            "stripped.safetensors",
            {"lora_unet_x": torch.zeros(1)},
            metadata={"ss_sigma_lowres_res_cond": "true"},
        )
        with pytest.raises(RuntimeError, match="stripped"):
            _attach_res_cond(SimpleNamespace(), [path], torch.device("cpu"))

    def test_inference_knob_propagates_s_and_default_is_zero(self, tmp_path):
        """25c.0: --res_cond_s reaches the attached tuple; the default (0.0)
        is the trained native point, bit-identical to the Stage-2.0 attach."""
        from library.inference.models import _attach_res_cond

        path = self._ckpt(
            tmp_path,
            "rescond.safetensors",
            {"sigma_lowres_res_cond_proj": torch.zeros(64, 16)},
            metadata={"ss_sigma_lowres_res_cond": "true"},
        )
        default = SimpleNamespace()
        _attach_res_cond(default, [path], torch.device("cpu"))
        assert default._sigma_lowres_res_cond[1] == 0.0
        knob = SimpleNamespace()
        _attach_res_cond(knob, [path], torch.device("cpu"), s=0.193)
        assert knob._sigma_lowres_res_cond[1] == 0.193

    def test_inference_knob_hard_fails_without_carrier(self, tmp_path):
        from library.inference.models import _attach_res_cond

        path = self._ckpt(
            tmp_path, "control.safetensors", {"lora_unet_x": torch.zeros(1)}
        )
        with pytest.raises(RuntimeError, match="res_cond_s"):
            _attach_res_cond(SimpleNamespace(), [path], torch.device("cpu"), s=0.193)

    def test_inference_attach_refuses_multiple_carriers(self, tmp_path):
        from library.inference.models import _attach_res_cond

        paths = [
            self._ckpt(
                tmp_path,
                f"rc{i}.safetensors",
                {"sigma_lowres_res_cond_proj": torch.zeros(64, 16)},
                metadata={"ss_sigma_lowres_res_cond": "true"},
            )
            for i in (0, 1)
        ]
        with pytest.raises(RuntimeError, match="untested"):
            _attach_res_cond(SimpleNamespace(), paths, torch.device("cpu"))


class TestResCondCentered:
    """E25e --sigma_lowres_res_cond_centered invariants (registration,
    e25e README): with the re-centered delta W·(φ(s) − φ(0)) the s = 0
    forward is exactly zero — and contributes exactly zero gradient to the
    projection — for ANY projection value (native steps are bit-identical
    to a control arm throughout training: the common-mode-offset channel
    the E25b post-close decomposition measured is dead at the
    parameterization). Demoted deltas stay live and route-distinguishing;
    the variant round-trips through the "centered" stamp / cfg string."""

    _T = torch.zeros(2, 1)

    def test_centered_s0_delta_exactly_zero_for_any_proj(self):
        from library.anima.models import sigma_lowres_res_cond_delta

        torch.manual_seed(0)
        proj = torch.randn(64, 16)  # NOT zero-init — the E25b invariant's gap
        delta = sigma_lowres_res_cond_delta(proj, 0.0, self._T, centered=True)
        assert torch.equal(delta, torch.zeros_like(delta))
        # bit-exact on the summation point, the same convention the E25b
        # zero-init identity pins
        t_emb = torch.randn(2, 1, 64)
        assert torch.equal(t_emb + delta, t_emb)

    def test_centered_s0_zero_grad_to_proj(self):
        """Native steps train NOTHING through the centered module — the
        sinusoid difference is exactly the zero vector, so the projection's
        gradient is exactly zero (demoted-only training, structurally)."""
        from library.anima.models import sigma_lowres_res_cond_delta

        torch.manual_seed(0)
        proj = torch.nn.Parameter(torch.randn(64, 16))
        sigma_lowres_res_cond_delta(proj, 0.0, self._T, centered=True).sum().backward()
        assert torch.equal(proj.grad, torch.zeros_like(proj))
        # ...while a demoted step DOES move it (non-vacuity)
        proj2 = torch.nn.Parameter(torch.randn(64, 16))
        sigma_lowres_res_cond_delta(
            proj2, math.log2(896 / 1024), self._T, centered=True
        ).sum().backward()
        assert not torch.equal(proj2.grad, torch.zeros_like(proj2))

    def test_centered_demote_is_uncentered_minus_native_offset(self):
        """delta_c(s) == delta(s) − delta(0): centering removes exactly the
        trained native offset W·φ(0), nothing else — and the two shipped
        routes stay distinguished."""
        from library.anima.models import sigma_lowres_res_cond_delta

        torch.manual_seed(0)
        proj = torch.randn(64, 16)
        d0 = sigma_lowres_res_cond_delta(proj, 0.0, self._T)
        deltas = {}
        for s in (math.log2(896 / 1024), math.log2(768 / 1024)):
            dc = sigma_lowres_res_cond_delta(proj, s, self._T, centered=True)
            du = sigma_lowres_res_cond_delta(proj, s, self._T)
            torch.testing.assert_close(dc, du - d0)
            assert not torch.allclose(dc, torch.zeros_like(dc))
            deltas[s] = dc
        assert not torch.allclose(*deltas.values())

    def test_zero_init_identity_still_holds_centered(self):
        from library.anima.models import sigma_lowres_res_cond_delta

        proj = torch.zeros(64, 16)
        for s in (0.0, math.log2(896 / 1024), math.log2(768 / 1024)):
            delta = sigma_lowres_res_cond_delta(proj, s, self._T, centered=True)
            assert torch.equal(delta, torch.zeros_like(delta))

    @staticmethod
    def _cfg(value):
        from networks.lora_modules.lora import LoRAModule
        from networks.lora_anima.config import LoRANetworkCfg

        return LoRANetworkCfg.from_kwargs(
            {"sigma_lowres_res_cond": value},
            network_dim=8,
            network_alpha=8.0,
            neuron_dropout=None,
            module_class=LoRAModule,
        )

    def test_cfg_string_centered_sets_both_bools(self):
        cfg = self._cfg("centered")
        assert cfg.sigma_lowres_res_cond is True
        assert cfg.sigma_lowres_res_cond_centered is True
        cfg = self._cfg("true")
        assert cfg.sigma_lowres_res_cond is True
        assert cfg.sigma_lowres_res_cond_centered is False

    def test_centered_flag_requires_base_flag(self):
        from train import AnimaTrainer

        args = _args(
            sigma_lowres=True,
            sigma_lowres_res_cond=False,
            sigma_lowres_res_cond_centered=True,
        )
        with pytest.raises(ValueError, match="res_cond_centered"):
            AnimaTrainer._validate_res_cond_flags(args)
        # base pair passes; res_cond without the router still fails
        AnimaTrainer._validate_res_cond_flags(
            _args(
                sigma_lowres=True,
                sigma_lowres_res_cond=True,
                sigma_lowres_res_cond_centered=True,
            )
        )
        with pytest.raises(ValueError, match="requires --sigma_lowres"):
            AnimaTrainer._validate_res_cond_flags(
                _args(sigma_lowres=False, sigma_lowres_res_cond=True)
            )

    def test_inference_attach_centered_stamp(self, tmp_path):
        """A "centered"-stamped carrier attaches with centered=True, and the
        default s = 0 attach is a mathematical no-op on the forward."""
        from library.anima.models import sigma_lowres_res_cond_delta
        from library.inference.models import _attach_res_cond

        torch.manual_seed(0)
        proj = torch.randn(64, 16)
        path = TestResCond._ckpt(
            tmp_path,
            "rescond_c.safetensors",
            {"sigma_lowres_res_cond_proj": proj, "lora_unet_x": torch.zeros(1)},
            metadata={"ss_sigma_lowres_res_cond": "centered"},
        )
        model = SimpleNamespace()
        _attach_res_cond(model, [path], torch.device("cpu"))
        got, s, centered = model._sigma_lowres_res_cond
        assert s == 0.0 and centered is True
        delta = sigma_lowres_res_cond_delta(got, s, self._T, centered=centered)
        assert torch.equal(delta, torch.zeros_like(delta))

    def test_inference_attach_centered_stamp_without_key_hard_fails(self, tmp_path):
        from library.inference.models import _attach_res_cond

        path = TestResCond._ckpt(
            tmp_path,
            "stripped_c.safetensors",
            {"lora_unet_x": torch.zeros(1)},
            metadata={"ss_sigma_lowres_res_cond": "centered"},
        )
        with pytest.raises(RuntimeError, match="stripped"):
            _attach_res_cond(SimpleNamespace(), [path], torch.device("cpu"))
