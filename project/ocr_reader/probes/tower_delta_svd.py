"""Spectrum of the fine-tuned PaddleOCR-VL tower delta: is ΔW low-rank enough for a LoRA?"""

import collections
import json
import re
import sys

import torch
from safetensors import safe_open

torch.set_num_threads(8)
TOWER, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
KS = (4, 8, 16, 32, 64, 128)

t = safe_open(TOWER, "pt")
b = safe_open(BASE, "pt")
print(
    "base dtype",
    b.get_tensor("visual.vision_model.encoder.layers.0.mlp.fc1.weight").dtype,
)


def base_key(k):
    if k.startswith("model.visual."):
        return k[len("model.") :]
    if k.startswith("model.projector."):
        return "mlp_AR." + k[len("model.projector.") :]
    raise KeyError(k)


def bf16_ulp(w):
    # spacing of bf16 values around |w| (8 mantissa bits incl. implicit → 2^(e-7))
    e = torch.floor(torch.log2(w.abs().clamp_min(1e-30)))
    return torch.pow(2.0, e - 7)


gauss_cache = {}


def gauss_energy(shape):
    if shape not in gauss_cache:
        g = torch.randn(shape, generator=torch.Generator().manual_seed(0))
        s2 = torch.linalg.svdvals(g) ** 2
        c = s2.cumsum(0) / s2.sum()
        gauss_cache[shape] = {k: c[k - 1].item() for k in KS}
    return gauss_cache[shape]


rows = []
other = {"n_params": 0, "n_changed": 0}
for k in t.keys():
    wf = t.get_tensor(k).float()
    w0 = b.get_tensor(base_key(k)).float()
    d = wf - w0
    other["n_params"] += d.numel()
    other["n_changed"] += int((d != 0).sum())
    if wf.ndim != 2 or "embedding" in k:
        continue
    s2 = torch.linalg.svdvals(d) ** 2
    tot = s2.sum().item()
    c = (s2.cumsum(0) / s2.sum()).tolist()
    p = s2 / s2.sum()
    eff_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()).item()
    r90 = next(i + 1 for i, v in enumerate(c) if v >= 0.9)
    # bf16 storage rounding of W_ft: uniform in ±ulp/2 → var ulp²/12
    noise = (bf16_ulp(wf) ** 2 / 12).sum().item()
    m = re.search(r"layers\.(\d+)\.(.+)\.weight", k)
    layer, kind = (int(m.group(1)), m.group(2)) if m else (-1, k.split(".")[-2])
    rows.append(
        dict(
            key=k,
            layer=layer,
            kind=kind,
            shape=list(d.shape),
            rel_norm=(tot**0.5) / w0.norm().item(),
            frac_zero=float((d == 0).float().mean()),
            energy={kk: c[kk - 1] for kk in KS},
            gauss={kk: v for kk, v in gauss_energy(tuple(d.shape)).items()},
            eff_rank=eff_rank,
            r90=r90,
            bf16_noise_frac=noise / tot,
        )
    )
    print(
        f"{k:70s} rel {rows[-1]['rel_norm']:.2e} top16 {c[15]:.3f} r90 {r90:5d} effr {eff_rank:7.1f} noise {noise / tot:.2f}",
        flush=True,
    )

json.dump(dict(rows=rows, other=other), open(OUT, "w"), indent=1)

print("\n== by kind (median over layers) ==")
by = collections.defaultdict(list)
for r in rows:
    by[r["kind"]].append(r)
for kind, rs in by.items():

    def med(f, rs=rs):
        return sorted(f(r) for r in rs)[len(rs) // 2]

    print(
        f"{kind:22s} n {len(rs):2d} shape {rs[0]['shape']} rel {med(lambda r: r[
                'rel_norm'
            ]):.2e} "
        + " ".join(
            f"top{k} {med(lambda r: r['energy'][k]):.2f}/{med(lambda r: r['gauss'][
                    k
                ]):.2f}"
            for k in (8, 16, 32, 64)
        )
        + f" r90 {med(lambda r: r['r90'])} effr {med(lambda r: r[
                'eff_rank'
            ]):.0f} noise {med(lambda r: r[
                'bf16_noise_frac'
            ]):.2f} zero {med(lambda r: r['frac_zero']):.2f}"
    )
print("\nall tensors: changed", other["n_changed"], "/", other["n_params"])
