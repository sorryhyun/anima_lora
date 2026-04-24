from types import SimpleNamespace

import torch

from library.inference.adapters import clear_hydra_sigma, set_hydra_sigma
from library.inference.generation import generate_body, generate_body_tiled
from networks.spectrum import spectrum_denoise


class FakeHydraNetwork:
    use_sigma_router = True

    def __init__(self):
        self.sigmas = []
        self.clear_count = 0
        self.enabled_calls = []

    def set_sigma(self, sigmas):
        self.sigmas.append(sigmas.detach().cpu().clone())

    def clear_sigma(self):
        self.clear_count += 1

    def set_enabled(self, enabled):
        self.enabled_calls.append(enabled)


class FakeAnima:
    patch_spatial = 1
    blocks_to_swap = False

    def __init__(self, network):
        self._hydra_network = network
        self._pgraft_network = network

    def prepare_block_swap_before_forward(self):
        pass

    def __call__(self, latents, timesteps, embed, padding_mask=None, **kwargs):
        return torch.zeros_like(latents)


class FakeFinalLayer(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class FakeSpectrumAnima(FakeAnima):
    def __init__(self, network):
        super().__init__(network)
        self.final_layer = FakeFinalLayer()

    def __call__(self, latents, timesteps, embed, padding_mask=None, **kwargs):
        feature = torch.zeros(
            latents.shape[0], 2, 3, dtype=latents.dtype, device=latents.device
        )
        self.final_layer(feature)
        return torch.zeros_like(latents)


def _args(**overrides):
    values = dict(
        image_size=(32, 32),
        infer_steps=3,
        flow_shift=1.0,
        sampler="euler",
        seed=0,
        guidance_scale=1.0,
        fp8=False,
        tiled_diffusion=False,
        tile_size=4,
        tile_overlap=0,
        prefix_weight=None,
        postfix_weight=None,
        spectrum=False,
        lora_cutoff_step=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _context():
    return {
        "prompt": "hydra sigma test",
        "embed": [
            torch.zeros(1, 4, 8, dtype=torch.bfloat16),
            None,
            None,
            torch.ones(1, 4, dtype=torch.bool),
        ],
    }


def _assert_sigmas(observed, expected):
    assert len(observed) == len(expected)
    actual = torch.stack([s[0] for s in observed])
    assert torch.allclose(actual, torch.tensor(expected), atol=2e-3)


def test_adapter_sigma_helper_updates_each_hydra_network_once():
    first = FakeHydraNetwork()
    second = FakeHydraNetwork()
    original = SimpleNamespace(
        _hydra_networks=[first, second],
        _hydra_network=second,
        _pgraft_network=first,
    )
    model = SimpleNamespace(_orig_mod=original)

    set_hydra_sigma(model, torch.tensor([0.25, 0.5], dtype=torch.bfloat16))
    clear_hydra_sigma(model)

    assert len(first.sigmas) == 1
    assert len(second.sigmas) == 1
    assert first.clear_count == 1
    assert second.clear_count == 1
    assert first.sigmas[0].dtype == torch.float32
    assert torch.allclose(first.sigmas[0], torch.tensor([0.25, 0.5]))


def test_generate_body_sets_hydra_sigma_each_denoising_step():
    network = FakeHydraNetwork()
    anima = FakeAnima(network)
    latents = torch.ones(2, 3, 1, 4, 4, dtype=torch.bfloat16)

    generate_body(
        _args(),
        anima,
        _context(),
        None,
        torch.device("cpu"),
        seed=0,
        latents=latents,
    )

    _assert_sigmas(network.sigmas, [1.0, 2.0 / 3.0, 1.0 / 3.0])
    assert network.clear_count == 1


def test_generate_body_tiled_sets_hydra_sigma_per_denoising_step_not_per_tile():
    network = FakeHydraNetwork()
    anima = FakeAnima(network)

    generate_body_tiled(
        _args(image_size=(64, 64), infer_steps=2),
        anima,
        _context(),
        None,
        torch.device("cpu"),
        seed=0,
    )

    _assert_sigmas(network.sigmas, [1.0, 0.5])
    assert network.clear_count == 1


def test_spectrum_sets_hydra_sigma_each_step():
    network = FakeHydraNetwork()
    anima = FakeSpectrumAnima(network)
    latents = torch.ones(2, 3, 1, 4, 4, dtype=torch.bfloat16)
    timesteps = torch.tensor([1.0, 0.5, 0.25], dtype=torch.bfloat16)
    sigmas = torch.tensor([1.0, 0.5, 0.25, 0.0], dtype=torch.float32)
    embed = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    padding_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bfloat16)

    spectrum_denoise(
        anima,
        latents,
        timesteps,
        sigmas,
        embed,
        embed,
        padding_mask,
        guidance_scale=1.0,
        sampler=None,
        device=torch.device("cpu"),
        warmup_steps=3,
    )

    _assert_sigmas(network.sigmas, [1.0, 0.5, 0.25])
    assert network.clear_count == 1
