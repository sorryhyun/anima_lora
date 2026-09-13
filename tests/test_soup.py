"""Unit tests for the ΔW LoRA soup builders + `make soup` pattern helpers.

CPU-only, synthetic state dicts. Invariants pinned:
- the soup is exact at the ΔW level (alpha scaling + inv_scale folding included)
- SVD truncation is lossless when ingredients share a rank-r subspace
- non-plain checkpoints (hydra/registers/mismatched recipes) are refused loudly
- the uncond pool glob unions the fine-tune selection in (dedup) and the output
  slug derives from the fine-tune path_pattern
"""

from __future__ import annotations

import pytest
import torch

from scripts.soup.build import soup_state_dicts, truncated_soup
from scripts.soup.pipeline import (
    check_init_mode,
    configured_svd_slice,
    init_tag,
    pool_glob,
    resolve_lrs,
    sigma_argv,
    sigma_overrides,
    sigma_settings,
    slug_for_pattern,
    uncond_name,
)

OUT, IN, RANK = 12, 10, 4


def _delta_w(sd: dict, module: str) -> torch.Tensor:
    """ΔW as a loaded LoRAModule would apply it (raw-input path, alpha/r scale)."""
    down = sd[f"{module}.lora_down.weight"].to(torch.float32)
    up = sd[f"{module}.lora_up.weight"].to(torch.float32)
    scale = float(sd[f"{module}.alpha"]) / down.shape[0]
    inv = sd.get(f"{module}.inv_scale")
    if inv is not None:
        down = down * inv.to(torch.float32).unsqueeze(0)
    return up @ down * scale


def _make_sd(
    seed: int,
    modules=("lora_unet_blocks_0_attn_q_proj", "lora_unet_blocks_1_mlp_layer2"),
    alpha: float = 90.0,
    inv_scale: bool = False,
    down: torch.Tensor | None = None,
) -> dict:
    g = torch.Generator().manual_seed(seed)
    sd: dict = {}
    for m in modules:
        sd[f"{m}.lora_down.weight"] = (
            down.clone() if down is not None else torch.randn(RANK, IN, generator=g)
        )
        sd[f"{m}.lora_up.weight"] = torch.randn(OUT, RANK, generator=g)
        sd[f"{m}.alpha"] = torch.tensor(alpha)
        if inv_scale:
            sd[f"{m}.inv_scale"] = torch.rand(IN, generator=g) + 0.5
    return sd


class TestSoupStateDicts:
    def test_delta_w_is_exact_average(self):
        sds = [_make_sd(0), _make_sd(1, inv_scale=True), _make_sd(2, alpha=32.0)]
        soup = soup_state_dicts(sds)
        for module in (
            "lora_unet_blocks_0_attn_q_proj",
            "lora_unet_blocks_1_mlp_layer2",
        ):
            want = torch.stack([_delta_w(sd, module) for sd in sds]).mean(0)
            torch.testing.assert_close(_delta_w(soup, module), want)
            # rank-concat: soup rank = sum of ingredient ranks, alpha = rank
            assert soup[f"{module}.lora_down.weight"].shape[0] == 3 * RANK
            assert float(soup[f"{module}.alpha"]) == 3 * RANK
            assert f"{module}.inv_scale" not in soup

    def test_weights_used_as_is(self):
        sds = [_make_sd(0), _make_sd(1)]
        soup = soup_state_dicts(sds, weights=[0.75, 0.25])
        module = "lora_unet_blocks_0_attn_q_proj"
        want = 0.75 * _delta_w(sds[0], module) + 0.25 * _delta_w(sds[1], module)
        torch.testing.assert_close(_delta_w(soup, module), want)

    def test_refuses_single_ingredient(self):
        with pytest.raises(ValueError, match=">= 2"):
            soup_state_dicts([_make_sd(0)])

    def test_refuses_hydra_suffix(self):
        sd = _make_sd(0)
        sd["lora_unet_blocks_0_attn_q_proj.lora_up_experts.0.weight"] = torch.zeros(1)
        with pytest.raises(ValueError, match="refused"):
            soup_state_dicts([sd, _make_sd(1)])

    def test_refuses_register_tokens(self):
        sd = _make_sd(0)
        sd["register_tokens"] = torch.zeros(4, 8)
        with pytest.raises(ValueError, match="not a plain-LoRA module key"):
            soup_state_dicts([sd, _make_sd(1)])

    def test_refuses_mismatched_module_sets(self):
        with pytest.raises(ValueError, match="module sets differ"):
            soup_state_dicts(
                [_make_sd(0), _make_sd(1, modules=("lora_unet_blocks_0_attn_q_proj",))]
            )


class TestTruncatedSoup:
    def test_shapes_alpha_and_rank(self):
        soup, energy = truncated_soup([_make_sd(0), _make_sd(1)], rank=RANK)
        module = "lora_unet_blocks_0_attn_q_proj"
        assert soup[f"{module}.lora_down.weight"].shape == (RANK, IN)
        assert soup[f"{module}.lora_up.weight"].shape == (OUT, RANK)
        assert float(soup[f"{module}.alpha"]) == RANK
        assert set(energy) == {
            "lora_unet_blocks_0_attn_q_proj",
            "lora_unet_blocks_1_mlp_layer2",
        }

    def test_lossless_on_shared_subspace(self):
        # Ingredients share the same down basis (the weight_svd-pinned-init
        # regime) -> the averaged ΔW is at most rank RANK -> truncation exact.
        down = torch.randn(RANK, IN, generator=torch.Generator().manual_seed(99))
        sds = [_make_sd(i, down=down) for i in range(3)]
        soup, energy = truncated_soup(sds, rank=RANK)
        assert all(e == pytest.approx(1.0, abs=1e-5) for e in energy.values())
        module = "lora_unet_blocks_0_attn_q_proj"
        want = torch.stack([_delta_w(sd, module) for sd in sds]).mean(0)
        torch.testing.assert_close(_delta_w(soup, module), want, rtol=1e-4, atol=1e-4)

    def test_truncation_is_best_rank_r(self):
        sds = [_make_sd(0), _make_sd(1), _make_sd(2)]
        soup, energy = truncated_soup(sds, rank=RANK)
        module = "lora_unet_blocks_0_attn_q_proj"
        avg = torch.stack([_delta_w(sd, module) for sd in sds]).mean(0)
        # Eckart-Young: residual energy == dropped singular-value energy.
        s = torch.linalg.svdvals(avg)
        want_energy = float((s[:RANK] ** 2).sum() / (s**2).sum())
        assert energy[module] == pytest.approx(want_energy, abs=1e-5)
        resid = torch.linalg.norm(avg - _delta_w(soup, module)) ** 2
        assert float(resid) == pytest.approx(float((s[RANK:] ** 2).sum()), rel=1e-4)


class TestPoolGlob:
    def test_star_pool_covers_everything(self):
        # "*" already matches the fine-tune set, so it stands alone.
        assert pool_glob("*", "sincos/*") == "*"
        assert pool_glob("", "sincos/*") == "*"

    def test_finetune_union_and_dedup(self):
        # The fine-tune selection is always folded in; overlapping alternatives
        # are not duplicated (boolean glob matching keeps an image once anyway).
        assert pool_glob("shard/*", "sincos/*") == "shard/*|sincos/*"
        assert pool_glob("a/*", "a/*|b/*") == "a/*|b/*"
        assert pool_glob("a/*|b/*", "b/*|c/*") == "a/*|b/*|c/*"


class TestSlugForPattern:
    def test_plain_dir_glob_is_basename(self):
        assert slug_for_pattern("sincos/*") == "sincos"
        assert slug_for_pattern("nested/artist/*") == "artist"

    def test_rich_pattern_is_sanitized_and_hashed(self):
        slug = slug_for_pattern("a/*|b/*")
        assert slug.startswith("a_b_")  # sanitized prefix + short hash
        # Deterministic and distinct from a different pattern.
        assert slug == slug_for_pattern("a/*|b/*")
        assert slug != slug_for_pattern("a/*|c/*")


class TestNoUncondInit:
    def test_slice_from_config_then_args(self):
        assert configured_svd_slice({}, []) == 0
        assert configured_svd_slice({"svd_slice": 3}, []) == 3
        # ARGS override the merged config, in either argparse spelling.
        assert (
            configured_svd_slice({"svd_slice": 3}, ["--network_args", "svd_slice=5"])
            == 5
        )
        assert configured_svd_slice({}, ["--network_args=svd_slice=2"]) == 2
        assert (
            configured_svd_slice({}, ["--network_args", "foo=1", "svd_slice=4", "--x"])
            == 4
        )

    def test_tag_only_in_no_uncond_mode(self):
        assert init_tag(False, 3) == ""  # uncond path keeps the legacy slug
        assert init_tag(True) == "_nouncond"
        assert init_tag(True, 3) == "_nouncond_k3"

    def test_slice_under_uncond_path_is_refused(self):
        # --network_weights would overwrite the sliced A with the uncond's rows.
        with pytest.raises(SystemExit, match="svd_slice=3 under the uncond-init"):
            check_init_mode(False, None, 3, [])
        check_init_mode(True, None, 3, [])  # the supported combination
        check_init_mode(False, None, 0, [])  # shipped path untouched

    def test_contradictory_flags_are_refused(self):
        with pytest.raises(SystemExit, match="mutually exclusive"):
            check_init_mode(True, "anima_uncond_x", 0, [])
        with pytest.raises(SystemExit, match="network_weights"):
            check_init_mode(True, None, 0, ["--network_weights", "a.safetensors"])
        with pytest.raises(SystemExit, match="network_weights"):
            check_init_mode(True, None, 0, ["--network_weights=a.safetensors"])


class TestResolveLrs:
    def test_unset_leaves_the_config_lr_alone(self):
        # The default soup is seed-only: no --learning_rate reaches the fine-tunes.
        assert resolve_lrs(3, None, None) is None
        assert resolve_lrs(3, "", "") is None

    def test_pool_is_cycled_to_num_soup(self):
        assert resolve_lrs(3, "1e-5,2e-5,4e-5", None) == [1e-5, 2e-5, 4e-5]
        assert resolve_lrs(4, "1e-5,2e-5", None) == [1e-5, 2e-5, 1e-5, 2e-5]
        assert resolve_lrs(2, "1e-5, 2e-5", None) == [1e-5, 2e-5]  # spaces tolerated

    def test_interval_is_geometric_and_hits_both_bounds(self):
        lrs = resolve_lrs(3, None, "1e-5:4e-5")
        assert lrs[0] == pytest.approx(1e-5)
        assert lrs[-1] == pytest.approx(4e-5)
        assert lrs[1] == pytest.approx(2e-5)  # log-midpoint, not 2.5e-5

    def test_conflicts_and_bad_input_are_refused(self):
        with pytest.raises(SystemExit):
            resolve_lrs(3, "1e-5,2e-5", "1e-5:4e-5")  # mutually exclusive
        with pytest.raises(SystemExit):
            resolve_lrs(3, None, "1e-5")  # not a lo:hi pair
        with pytest.raises(SystemExit):
            resolve_lrs(3, None, "0:4e-5")  # geometric needs positive bounds
        with pytest.raises(SystemExit):
            resolve_lrs(3, "fast", None)


class TestSigmaSettings:
    OFF = {"sigma_lowres": False, "sigma_lowres_route": "1024:896"}
    ON = {
        "sigma_lowres": True,
        "sigma_lowres_route": "1024:896",
        "sigma_lowres_threshold": 0.5,
        "sigma_lowres_route2": "1024:768",
        "sigma_lowres_threshold2": 0.65,
        "sigma_lowres_threshold2_max": 0.95,
    }

    def test_off_config_yields_nothing(self):
        assert sigma_settings(self.OFF) == {}
        assert sigma_argv(sigma_settings(self.OFF)) == []

    def test_args_can_turn_it_on(self):
        # ARGS reaches only the fine-tunes, so the pipeline has to lift the flag
        # out of it to replay onto Phase 1.
        cfg = sigma_settings(self.OFF, sigma_overrides(["--sigma_lowres"]))
        assert cfg["sigma_lowres"] is True
        assert "--sigma_lowres" in sigma_argv(cfg)

    def test_args_override_the_config(self):
        cfg = sigma_settings(
            self.ON, sigma_overrides(["--foo", "1", "--sigma_lowres_threshold", "0.7"])
        )
        assert cfg["sigma_lowres_threshold"] == 0.7
        assert cfg["sigma_lowres_route"] == "1024:896"  # untouched keys survive

    def test_unset_flags_do_not_clobber(self):
        # argparse.SUPPRESS: absent flags must be omitted, not defaulted.
        assert sigma_overrides(["--network_dim", "32"]) == {}
        assert sigma_settings(self.ON, sigma_overrides([])) == sigma_settings(self.ON)

    def test_yarnsig_default_is_normalized_in(self):
        # train.py turns yarnsig on at its operating point with --sigma_lowres,
        # so an explicit config copy of that value must not fork the digest.
        implicit = sigma_settings(self.ON)
        explicit = sigma_settings({**self.ON, "sigma_lowres_yarnsig": "1,4,0.35,2"})
        assert implicit["sigma_lowres_yarnsig"] == "1,4,0.35,2"
        assert implicit == explicit
        assert uncond_name("*", 1.0, 4, implicit) == uncond_name("*", 1.0, 4, explicit)

    def test_empty_string_reads_as_absent(self):
        # "" is how a span is switched off, and train.py reads it as absent.
        assert sigma_settings({**self.ON, "sigma_lowres_span": ""}) == sigma_settings(
            self.ON
        )

    def test_args_switching_a_key_off_is_forwarded_explicitly(self):
        # Phase 1 merges the same config, so an ARGS "off" must be CLEARED on the
        # command line — silently dropping the flag leaves base.toml's span live.
        cfg_on = {**self.ON, "sigma_lowres_span": "late:0.75"}
        cfg = sigma_settings(cfg_on, sigma_overrides(["--sigma_lowres_span", ""]))
        assert "sigma_lowres_span" not in cfg
        argv = sigma_argv(cfg, cfg_on)
        assert argv[argv.index("--sigma_lowres_span") + 1] == ""
        # …and with nothing to clear, no stray empty flag is emitted.
        assert "--sigma_lowres_span" not in sigma_argv(sigma_settings(self.ON), self.ON)

    def test_int_and_float_thresholds_digest_the_same(self):
        assert sigma_settings(
            {**self.ON, "sigma_lowres_threshold": 1}
        ) == sigma_settings({**self.ON, "sigma_lowres_threshold": 1.0})

    def test_argv_round_trips_every_key(self):
        cfg = sigma_settings(self.ON)
        argv = sigma_argv(cfg)
        assert argv[0] == "--sigma_lowres"
        assert sigma_settings(self.OFF, sigma_overrides(argv)) == cfg


class TestUncondName:
    def test_determinism(self):
        n1 = uncond_name("a/*|b/*", 0.5, 2)
        assert n1 == uncond_name("a/*|b/*", 0.5, 2)  # same pool glob → same name
        assert n1.endswith("_r0p5_e2")
        assert n1 != uncond_name("a/*|c/*", 0.5, 2)
        assert uncond_name("*", 0.5, 2) != uncond_name("*", 0.5, 4)  # dose in name

    def test_sigma_off_keeps_the_legacy_name(self):
        # Pre-existing uncond checkpoints must not be orphaned by this knob.
        base = uncond_name("*", 1.0, 4)
        assert uncond_name("*", 1.0, 4, None) == base
        assert uncond_name("*", 1.0, 4, {}) == base

    def test_sigma_forks_the_name(self):
        base = uncond_name("*", 1.0, 4)
        on = uncond_name("*", 1.0, 4, {"sigma_lowres": True, "x": "1024:896"})
        assert on != base and on.startswith(base + "_sl")
        # Routing differences fork it too — a 896-route init is not a 768 one.
        other = uncond_name("*", 1.0, 4, {"sigma_lowres": True, "x": "1024:768"})
        assert other != on
