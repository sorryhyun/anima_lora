"""sigma_lowres Phase 1b invariants (σ>0.5 → 896 sibling latent).

Pins the load-bearing contracts of the trainer wiring:
  - the demote grid lands route shapes in the demote band, and off-route
    shapes return None;
  - the demoted npz key can never collide with the ``latents_*`` namespace
    (several readers grab the FIRST ``latents_*`` key);
  - the σ-first two-step path reproduces the draw-inside path exactly (same
    seed → same noisy input / timesteps);
  - the preprocess emit appends the demoted key in-place, preserves every
    native key, is idempotent, and the dataset-side loader reads it back.
"""

from pathlib import Path
from types import SimpleNamespace


import numpy as np
import pytest
import torch

from library.datasets.buckets import (
    SIGMA_DEMOTE_ROUTE,
    demote_bucket_for,
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


class TestDemotedKey:
    @pytest.mark.core
    def test_never_in_latents_namespace(self):
        key = demoted_latents_key(880, 1184)
        assert not key.startswith("latents")
        assert key == "demoted_148x110"  # H//8 x W//8, native key convention


class TestSigmaDraw:
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

    @pytest.mark.core
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

    @pytest.mark.core
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


class TestYarnsigRope:
    """--sigma_lowres_yarnsig: σ-gated YaRN banded rope on demoted steps.

    Pins the two reduction identities the probe's read rests on (μ→0 ⇒
    native integer spacing, i.e. the intervention vanishes below the gate;
    all-bands-below-α ⇒ the uniform PI stretch) and that the shipped recipe
    is on by default.
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

    def test_default_on_with_sigma_lowres(self):
        """yarnsig is part of the shipped recipe, so an unset flag under
        --sigma_lowres resolves to the operating point — a plain --sigma_lowres
        run must not silently be a degraded (no-yarnsig) arm."""
        from library.anima.trainer import sigma_route as sr

        args = SimpleNamespace(sigma_lowres=True, sigma_lowres_yarnsig=None)
        assert sr.yarnsig_params(args) == (1.0, 4.0, 0.35, 2.0)
        # …and the effective value is written back, so the snapshot records it.
        assert args.sigma_lowres_yarnsig == sr.YARNSIG_OPERATING_POINT


class TestSigmaSpan:
    """--sigma_lowres_span (E16 placement probe): step-span gate on top of
    the σ gate. Pins the exact early/late partition of the train-forward
    range (identical demoted mass ⇒ any ΔW ordering is pure placement
    signal).
    """

    @staticmethod
    def _args(span, total=480, seed=1001, accum=1):
        return SimpleNamespace(
            sigma_lowres_span=span,
            max_train_steps=total,
            gradient_accumulation_steps=accum,
            seed=seed,
        )

    def test_early_late_partition_exact(self):
        """early:0.5 and late:0.5 partition [1, T] exactly — same mass,
        disjoint, exhaustive — for even and odd T."""
        from library.anima.trainer import sigma_route as sr

        for total in (480, 481):
            early = self._args("early", total=total)
            late = self._args("late", total=total)
            e = {i for i in range(1, total + 1) if sr.span_allows(early, i)}
            lt = {i for i in range(1, total + 1) if sr.span_allows(late, i)}
            b = round(0.5 * total)
            assert e == set(range(1, b + 1))
            assert lt == set(range(b + 1, total + 1))
            assert e | lt == set(range(1, total + 1))
            assert not (e & lt)
            assert abs(len(e) - len(lt)) <= 1  # identical mass (±1 at odd T)

    def test_grad_accum_scales_total(self):
        from library.anima.trainer import sigma_route as sr

        args = self._args("early", total=480, accum=2)  # 960 forwards
        assert sr.span_allows(args, 480)
        assert not sr.span_allows(args, 481)


class TestSigmaWindow:
    """--sigma_lowres_threshold_max: the half-line σ gate becomes a window.
    Pins the gate arithmetic the E16 win768 arm (0.65 < σ < 0.95) rests on."""

    @staticmethod
    def _gate(sigmas, lo, hi):
        from library.anima.trainer import sigma_route as sr

        args = SimpleNamespace(sigma_lowres_threshold=lo, sigma_lowres_threshold_max=hi)
        return sr.gate_allows(args, sigmas)

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
    gate/span independence."""

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

    def test_combo_choice_matches_spec(self):
        """768 if sigma in (0.65, 0.95); elif sigma > 0.5 -> 896; else native."""
        from library.anima.trainer import sigma_route as sr

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
            got = sr.demote_choice(args, torch.tensor([sigma]), 1)
            assert got == want, (sigma, got, want)

    def test_per_rule_spans_independent(self):
        """rule 2 late-gated while rule 1 has no span: an in-window sigma in
        the first half falls back to the primary rule."""
        from library.anima.trainer import sigma_route as sr

        args = self._combo_args(sigma_lowres_span2="late")
        sig = torch.tensor([0.7])
        assert sr.demote_choice(args, sig, 100) == 1
        assert sr.demote_choice(args, sig, 300) == 2


class TestSigmaRuleValidation:
    """--sigma_lowres_route2 must be given an explicit sigma WINDOW, at SETUP —
    not on the first train forward.

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
        from library.anima.trainer import sigma_route as sr

        with pytest.raises(ValueError) as e:
            sr.validate_rules(self._args(sigma_lowres_route2="1024:768"))
        msg = str(e.value)
        assert "--sigma_lowres_threshold2" in msg
        assert "--sigma_lowres_threshold2_max" in msg

    def test_shipped_combo_recipe_validates(self):
        from library.anima.trainer import sigma_route as sr

        route, route2 = sr.validate_rules(
            self._args(
                sigma_lowres_route2="1024:768",
                sigma_lowres_threshold2=0.65,
                sigma_lowres_threshold2_max=0.95,
                sigma_lowres_yarnsig="",
            )
        )
        assert route == (1024, 896) and route2 == (1024, 768)


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
