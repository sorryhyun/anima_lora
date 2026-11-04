"""Peak values fp16acc's fp16 fragments reach on real MOSS steps (eager, sampled calls).

Rebuilds dP, dS and the per-tile dQ and dK from each sampled call's Q, K, V and dO in fp32:
every 11th call, every 6th head, tiles of 128 keys and 64 queries.

    make daemon-run ARGS="--stall-timeout 300 project/finished/int8_train/src/moss_fp16_range.py <out.json> \\
      --config configs/moss/lora.toml --cache '<cache>' --output <tmp.safetensors> \\
      --no-compile --grad_checkpointing --max_steps 24"
"""

import json
import math
import sys
import torch
import library.moss.model as m

LIMIT = 65504.0
OUT = sys.argv[1]
orig = m.flash_attn_func
peak = dict(q=0.0, k=0.0, v=0.0, sum_v=0.0, dP=0.0, dS=0.0, dQ_tile=0.0, dK_tile=0.0)
calls = {"n": 0, "sampled": 0, "shapes": set()}
scale = 1 / math.sqrt(128)


def bump(key, value):
    peak[key] = max(peak[key], float(value))


@torch.no_grad()
def measure(q, k, v, do):
    for h in range(0, q.shape[2], 6):
        Q, K, V, D = (t[0, :, h].float() for t in (q, k, v, do))
        P = (Q @ K.T * scale).softmax(-1)
        Dn = D / D.abs().amax().clamp_min(1e-30)
        dP = Dn @ V.T
        dS = P * (dP - (P * dP).sum(-1, keepdim=True))
        bump("dP", dP.abs().max())
        bump("dS", dS.abs().max())
        for j in range(0, K.shape[0], 128):
            bump("dQ_tile", (dS[:, j : j + 128] @ K[j : j + 128]).abs().max())
        for i in range(0, Q.shape[0], 64):
            bump("dK_tile", (dS[i : i + 64].T @ Q[i : i + 64]).abs().max())
        bump("q", Q.abs().max())
        bump("k", K.abs().max())
        bump("v", V.abs().max())
        bump("sum_v", V.abs().sum(-1).max())
    calls["sampled"] += 1


def wrapped(q, k, v):
    out = orig(q, k, v)
    if torch.is_grad_enabled() and out.requires_grad and q.shape[3] == 128:
        calls["n"] += 1
        if calls["n"] % 11 == 0:
            calls["shapes"].add(tuple(q.shape))
            qd, kd, vd = q.detach(), k.detach(), v.detach()
            out.register_hook(lambda d: measure(qd, kd, vd, d.detach()) or None)
    return out


m.flash_attn_func = wrapped
from scripts.moss.train import main  # noqa: E402

sys.argv = ["train.py", *sys.argv[2:]]
main()
res = dict(
    peak=peak,
    margin={
        k: LIMIT / max(peak[k], 1e-30) for k in ("sum_v", "dP", "dQ_tile", "dK_tile")
    },
    calls=calls["n"],
    sampled=calls["sampled"],
    shapes=sorted(calls["shapes"])[:6],
)
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res))
