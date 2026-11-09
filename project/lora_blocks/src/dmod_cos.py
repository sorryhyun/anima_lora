"""Cosine of the modulation adapter's reference-only part, Δmod(0) − Δmod(900), between LoRAs, per chunk.

.venv/bin/python project/lora_blocks/src/dmod_cos.py name=path.safetensors [name=path …]
"""

import json
import sys
import torch
from safetensors import safe_open
from library.env import resolve_under_home

B = str(resolve_under_home("models/moss_image/MOSS-Image-Base/transformer")) + "/"
cfg = json.load(open(B + "config.json"))
with safe_open(B + "diffusion_pytorch_model-00001-of-00003.safetensors", "pt") as f:
    W = {
        k: f.get_tensor(k).double()
        for k in [
            "time_embedding.0.weight",
            "time_embedding.0.bias",
            "time_embedding.2.weight",
            "time_embedding.2.bias",
            "shared_block_modulation.weight",
            "shared_block_modulation.bias",
        ]
    }
half = cfg["freq_dim"] // 2
freq = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)


def time(t):
    ph = torch.tensor([float(t)], dtype=torch.float64)[:, None] * freq
    e = torch.cat((ph.cos(), ph.sin()), -1)
    h = torch.nn.functional.silu(
        e @ W["time_embedding.0.weight"].T + W["time_embedding.0.bias"]
    )
    return (h @ W["time_embedding.2.weight"].T + W["time_embedding.2.bias"])[0]


T = {t: time(t) for t in (0, 300, 900)}
base0 = W["shared_block_modulation.weight"] @ T[0] + W["shared_block_modulation.bias"]


def dmod(path):
    with safe_open(path, "pt") as f:
        p = "lora_unet_shared_block_modulation."
        d, u = (
            f.get_tensor(p + "lora_down.weight").double(),
            f.get_tensor(p + "lora_up.weight").double(),
        )
        a = float(f.get_tensor(p + "alpha"))
    s = a / d.shape[0]
    return {t: (s * u @ (d @ v)).view(6, -1) for t, v in T.items()}


names = ["shift_a", "scale_a", "gate_a", "shift_f", "scale_f", "gate_f"]
runs = dict(a.split("=", 1) for a in sys.argv[1:])
D = {k: dmod(v) for k, v in runs.items()}


def cos(x, y):
    return float(torch.nn.functional.cosine_similarity(x, y, dim=0))


print("‖Δmod(0)‖ / ‖base mod(0)‖ per chunk, and ‖Δmod(0)‖/‖Δmod(900)‖")
b = base0.view(6, -1)
for k, m in D.items():
    print(
        f"  {k:10}",
        " ".join(
            f"{n} {100 * m[0][i].norm() / b[i].norm():.2f}%/{m[0][i].norm() / m[900][i].norm():.1f}x"
            for i, n in enumerate(names)
        ),
    )
print("\ncos of reference-only part Δmod(0) − Δmod(900), per chunk")
ks = list(D)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        r1 = D[ks[i]][0] - D[ks[i]][900]
        r2 = D[ks[j]][0] - D[ks[j]][900]
        print(
            f"  {ks[i]:>10} – {ks[j]:<10}",
            " ".join(f"{cos(r1[c], r2[c]):+.2f}" for c in range(6)),
        )
