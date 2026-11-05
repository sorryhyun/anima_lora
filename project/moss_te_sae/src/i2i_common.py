"""Weight-space common components of the MOSS i2i LoRAs (CPU only).

Per module, ΔW = scale · up · down for four i2i LoRAs (swap_text, colorize, sanitize,
uncensor), a same-task twin of each, and a t2i control (ama_mitsuki, same seed /
Muon / rank). Every run starts from the same seed-0 kaiming ``down``, rebuilt here in
LoRANetwork's construction order, so the input side is compared on ``down - init``
and the output side (up, zero-init) carries the fair overlaps. Writes one JSON:
per module the output-subspace overlaps (top 8 and full), the learned-down overlaps,
singular values, the 4-task shared-projector eigenvalues and per-head energy shares
of q / k / v / o.

    .venv/bin/python project/moss_te_sae/src/i2i_common.py output/tests/i2i_common/<stamp>/result.json
"""

import json
import sys

import torch
from safetensors.torch import load_file

from library.lines.adapters import LoRAAdapter

D = "output/ckpt_moss_i2i/"
L = {
    "swap": "moss_i2i_swap_text_v3_en400_ko1709_ksplit_rw2_mod16_ema5e5_int8_e2",
    "color": "moss_colorize_v5",
    "sanit": "moss_i2i_sanitize_idt_w0003_late_lr5e5_e8",
    "uncen": "moss_i2i_uncensor_combined_muon_lr5e5_rw10to1_mod16",
    "swap2": "moss_i2i_swap_text_v3_en400_ko1709_rw2_mod16_ema5e5_int8_e2",
    "color2": "moss_colorize_3k_textref1_iso_muon_ema",
    "sanit2": "moss_i2i_sanitize_idt_mod16_notone_lr5e5_e8",
    "uncen2": "moss_i2i_uncensor_combined_muon_lr5e5_rw10",
    "ama": "../ckpt_moss/ama_mitsuki_muon_ema_fp32s",
    "ama2": "../ckpt_moss/ama_mitsuki_muon_ema",
}
TASKS = ["swap", "color", "sanit", "uncen"]
PROJ = ["q", "k", "v", "o", "ffn_up_gate", "ffn_down"]
DIMS = {p: (3072, 3072) for p in "qkvo"} | {
    "ffn_up_gate": (16384, 3072),
    "ffn_down": (3072, 8192),
}
K = 8
torch.set_grad_enabled(False)

# seed-0 init, same construction order as LoRANetwork (layers, then proj order)
torch.manual_seed(0)
D0 = {}
for b in range(42):
    for p in PROJ:
        o, i = DIMS[p]
        a = LoRAAdapter(i, o, 32, 128.0, torch.float32)
        D0[(b, p)] = a.down.weight.clone()
        del a

W = {k: load_file(D + v + ".safetensors") for k, v in L.items()}


def get(n, b, p):
    key = f"lora_unet_layers_{b}_proj_{p}"
    up = W[n][key + ".lora_up.weight"].float()
    dn = W[n][key + ".lora_down.weight"].float()
    s = float(W[n][key + ".alpha"]) / 32
    return up, dn, s


def svd_factored(up, dn, s):
    qu, ru = torch.linalg.qr(up)
    u, sv, vh = torch.linalg.svd(s * ru @ dn, full_matrices=False)
    return qu @ u, sv, vh  # ΔW = U diag(sv) Vh


res = {
    "full_ov": {},
    "sv": {},
    "init_cos": {},
    "energy": {},
    "out_ov": {},
    "din_ov": {},
    "shared": {},
    "heads": {},
}
names = list(L)
F = {}
for b in range(42):
    for p in PROJ:
        for n in names:
            up, dn, s = get(n, b, p)
            U, sv, Vh = svd_factored(up, dn, s)
            d0 = D0[(b, p)]
            delta = dn - d0
            _, dsv, dVh = torch.linalg.svd(delta, full_matrices=False)
            F[(n, b, p)] = dict(
                U=U,
                sv=sv,
                dVh=dVh[:K],
                dsv=dsv,
                init_cos=float(
                    torch.nn.functional.cosine_similarity(dn.flatten(), d0.flatten(), 0)
                ),
                dfrac=float(delta.norm() / dn.norm()),
                up=up,
                dn=dn,
                s=s,
            )
    print("block", b, file=sys.stderr, flush=True)


def ew_overlap(a, bb):
    """energy of ΔW_a inside the top-K output subspace of ΔW_b"""
    Ua, sa = F[a]["U"], F[a]["sv"]
    Ub = F[bb]["U"][:, :K]
    proj = (Ub.T @ Ua).square().sum(0)  # per singular dir of a
    return float((proj * sa.square()).sum() / sa.square().sum())


def din_overlap(a, bb):
    Va, Vb = F[a]["dVh"], F[bb]["dVh"]
    return float((Va @ Vb.T).square().sum() / K)


def full_overlap(a, bb):
    Ua, sa = F[a]["U"], F[a]["sv"]
    Ub = F[bb]["U"]
    proj = (Ub.T @ Ua).square().sum(0)
    return float((proj * sa.square()).sum() / sa.square().sum())


pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1 :]]
for b, p in [(b, p) for b in range(42) for p in PROJ]:
    k = f"{b}.{p}"
    res["init_cos"][k] = {n: F[(n, b, p)]["init_cos"] for n in names}
    res["energy"][k] = {n: float(F[(n, b, p)]["sv"].norm()) for n in names}
    res["out_ov"][k] = {
        f"{x}-{y}": 0.5
        * (ew_overlap((x, b, p), (y, b, p)) + ew_overlap((y, b, p), (x, b, p)))
        for x, y in pairs
    }
    res["full_ov"][k] = {
        f"{x}-{y}": 0.5
        * (full_overlap((x, b, p), (y, b, p)) + full_overlap((y, b, p), (x, b, p)))
        for x, y in pairs
    }
    res["sv"][k] = {n: F[(n, b, p)]["sv"].tolist() for n in names}
    res["din_ov"][k] = {f"{x}-{y}": din_overlap((x, b, p), (y, b, p)) for x, y in pairs}
    # shared output directions across the 4 primary tasks: eigvals of mean projector
    Ucat = torch.cat([F[(n, b, p)]["U"][:, :K] for n in TASKS], 1)
    ev = torch.linalg.svdvals(Ucat).square() / len(TASKS)
    res["shared"][k] = ev[:12].tolist()
    if p in ("q", "k", "v", "o"):
        hs = {}
        for n in TASKS:
            up, dn, s = F[(n, b, p)]["up"], F[(n, b, p)]["dn"], F[(n, b, p)]["s"]
            if p == "o":  # heads index the input columns
                M = s * up @ dn  # 3072x3072
                e = M.square().sum(0).view(24, 128).sum(1)
            else:
                e = (s * up).square().view(24, 128, 32)
                # ‖ΔW rows of head h‖² = Σ_rows ‖up_row · dn‖² ; use exact via dn gram
                G = dn @ dn.T
                upv = (s * up).view(24, 128, 32)
                e = torch.einsum("hra,ab,hrb->h", upv, G, upv)
            hs[n] = (e / e.sum()).tolist()
        res["heads"][k] = hs

json.dump(res, open(sys.argv[1], "w"))
