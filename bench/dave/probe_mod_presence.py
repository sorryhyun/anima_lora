"""Same-seed MOD-on vs MOD-off block-output DC diff — is the Phase-4 MOD arm
(quality-prompt delta, w=3) actually perturbing the forward, or a numeric no-op?

Sanity companion to the Phase-4 compose probe (README § Phase 4). Cross-seed
lock stats cannot tell a common-mode MOD shift from no shift at all; this
same-seed cross-arm diff of each block's output DC can.

Mirrors the probe arms: OFF = no head injection at all (enable flag off, no
steering); ON = head loaded (proj(main) into t_emb) + quality steering delta.
"""

import copy
from pathlib import Path

import torch

from anima_lora import (
    GenerationRequest,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_dit_model,
    prepare_text_inputs,
)
from bench._anima import DEFAULT_NEG, DEFAULT_PROMPT
from library.inference.models import load_text_encoder

_ROOT = Path(__file__).resolve().parents[2]
STEPS, CFG, SEED = 8, 4.0, 1000
LORA = str(_ROOT / "output/ckpt/anima_channel_ortho.safetensors")
HEAD = str(_ROOT / "output/ckpt/pooled_text_proj-0611.safetensors")

device = torch.device("cuda")
ckpts = default_checkpoints()
req = GenerationRequest(
    dit=ckpts.dit,
    vae=ckpts.vae,
    text_encoder=ckpts.text_encoder,
    prompt=DEFAULT_PROMPT,
    negative_prompt=DEFAULT_NEG,
    save_path="output/tests/_mod_presence.png",
    infer_steps=STEPS,
    guidance_scale=CFG,
    image_size=(1024, 1024),
    seed=SEED,
    lora_weight=[LORA],
    pooled_text_proj=HEAD,
)
args = req.to_args()
args.device = device
args.compile = False
args.compile_blocks = False
gen_settings = get_generation_settings(args)

anima = load_dit_model(args, device, torch.bfloat16)
L = len(anima.blocks)

shared = {
    "model": anima,
    "text_encoder": load_text_encoder(args, dtype=torch.bfloat16, device=device),
}
shared["text_encoder"].eval()
context, context_null = prepare_text_inputs(args, device, anima, shared)
text_data = {"context": context, "context_null": context_null}

# capture per (step, block) cond-forward DC, probe-style parity counting
state = {"fi": -1, "store": None}


def mk(bidx):
    def hook(_m, _i, out):
        if bidx == 0:
            state["fi"] += 1
        if state["store"] is None:
            return
        fi = state["fi"]
        if fi % 2 != 0:  # uncond
            return
        step = fi // 2
        if step >= STEPS:
            return
        state["store"][(step, bidx)] = (
            out.detach().float().mean(dim=(1, 2, 3)).squeeze(0).cpu()
        )

    return hook


handles = [blk.register_forward_hook(mk(i)) for i, blk in enumerate(anima.blocks)]


def run(mod_on):
    a = copy.copy(args)
    if not mod_on:
        a.pooled_text_proj = None
        anima.enable_pooled_text_modulation = False
    else:
        anima.enable_pooled_text_modulation = True
    state["fi"] = -1
    state["store"] = {}
    generate(a, gen_settings, shared_models=shared, precomputed_text_data=text_data)
    s = state["store"]
    state["store"] = None
    return s


base = run(False)
mod = run(True)
for h in handles:
    h.remove()

print(f"\n{'blk':>3} | {'relΔDC (mean over steps)':>25} | {'cos(DC_mod, DC_base)':>20}")
for bl in range(L):
    rels, coss = [], []
    for s in range(STEPS):
        b, m = base[(s, bl)], mod[(s, bl)]
        rels.append(float((m - b).norm() / b.norm().clamp_min(1e-12)))
        coss.append(float(torch.nn.functional.cosine_similarity(m, b, dim=0)))
    flag = " <- MOD window" if 8 <= bl <= 26 else ""
    print(
        f"{bl:3d} | {sum(rels) / len(rels):25.4f} | {sum(coss) / len(coss):20.6f}{flag}"
    )

w_all = [
    sum((mod[(s, bl)] - base[(s, bl)]).norm() for s in range(STEPS)) / STEPS
    for bl in range(L)
]
print(
    f"\nmean |ΔDC| blocks 0-7: {sum(w_all[:8]) / 8:.4f}  blocks 8-26: {sum(w_all[8:27]) / 19:.4f}"
)
