"""``library/lines/adapters.py`` as the qwen21 line binds it (CPU).

The MOSS binding is covered by ``tests/test_moss_lora.py``. Here: a file written by
qwen21's trainer before the shared module (no ``algo`` / ``ss_num_blocks`` /
``down_init`` metadata, ``alpha`` as an int string) still loads, and the
LyCORIS algos round-trip on the qwen21 binding.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import save_file  # noqa: E402

from library.qwen21.lora import LoRANetwork, load_network  # noqa: E402

TARGETS = r"transformer_blocks\.\d+\.attn\.to_q"


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.to_q = torch.nn.Linear(16, 16, bias=False)


class _Tiny(torch.nn.Module):
    def __init__(self, blocks=2):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(_Block() for _ in range(blocks))


def _randomize(network):
    torch.manual_seed(3)
    for p in network.parameters():
        torch.nn.init.normal_(p)


@pytest.mark.core
def test_pre_shared_qwen21_file_loads(tmp_path):
    torch.manual_seed(0)
    model = _Tiny()
    old = LoRANetwork(model, rank=4, targets=TARGETS)
    _randomize(old)
    path = tmp_path / "old.safetensors"
    # Exactly the metadata qwen21/train.py wrote before library/lines/.
    save_file(
        {k: v.contiguous() for k, v in old.state_dict().items()},
        str(path),
        metadata={
            "base_model": "Qwen-Image-2.1",
            "rank": "4",
            "alpha": "4",
            "targets": TARGETS,
            "lora_dtype": "fp32",
            "epochs": "8",
        },
    )
    network, meta = load_network(model, path)
    assert network.algo == "lora" and meta["epochs"] == "8"
    for key, value in old.state_dict().items():
        assert torch.equal(network.state_dict()[key], value), key


@pytest.mark.core
@pytest.mark.parametrize("algo", ["lora", "lokr", "loha"])
def test_algos_round_trip(tmp_path, algo):
    torch.manual_seed(0)
    model = _Tiny()
    network = LoRANetwork(model, rank=4, alpha=2.0, targets=TARGETS, algo=algo)
    _randomize(network)
    path = tmp_path / f"{algo}.safetensors"
    network.save(path, "fp32", epochs=1)
    loaded, meta = load_network(_Tiny(), path)
    assert meta["algo"] == algo and meta["base_model"] == "Qwen-Image-2.1"
    assert meta["ss_num_blocks"] == "2"
    for key, value in network.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key
    with pytest.raises(ValueError, match="2-block"):
        load_network(_Tiny(blocks=3), path)
