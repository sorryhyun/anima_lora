"""PE mode on the shipped MOSS INT8 TE: LoRA branches + swapped small tensors.

The catalog TE (``moss_te_int8``) is ``te_gptq.py --smooth 0.5 --act_aware``: each
smoothed linear group sees ``x̃ = x / s`` (``s`` folded into the RMSNorm feeding it,
or into ``up_proj``'s rows for ``down_proj``). An extracted factor pair on the bf16,
unsmoothed basis (``extract.py``) is moved onto it by

  input-smoothed linears   lora_down · diag(s)
  up_proj                  diag(1/s_down) · lora_up   (its output is divided)
  smoothed RMSNorms        (1 + w_pe) / s − 1

``s`` is not stored in the file. For a norm-fed group it is exact,
``(1 + w_inst) / (1 + w_file)``. For ``down_proj`` it is ``‖r‖² / ⟨r, r̂⟩`` per
``up_proj`` row, where ``r`` is the bf16 row and ``r̂`` the file's dequantized row, both
in the ConvRot basis, so the rounding noise averages out of the inner product.

PE mode turns the branches on and swaps in PE's norms, GDN ``conv1d``, and the other
small tensors. Its ``lm_head`` is PE's own bf16 matrix: the TE file has no lm_head,
so any PE head costs 2 GB, and the exact one costs no more than inst + LoRA. The
token embedding stays inst's (``eval.py``: PE's table moves KL by < 0.003). TE mode
restores every swapped tensor.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from library.moss import int8

ATTN_IN = {
    "linear_attn": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
    "self_attn": ("q_proj", "k_proj", "v_proj"),
}
CKPT = "model.language_model."  # extract.py / HF checkpoint prefix
TE = "language_model."  # MOSS TE file / Qwen3_5Model prefix


def _deq_rot(f, key):
    """File linear ``key`` (``….weight``) dequantized, left in the ConvRot basis."""
    q = f.get_tensor(key).cuda().float()
    return q * f.get_tensor(key[: -len("weight")] + "weight_scale").cuda().float()


@torch.no_grad()
def convert(lora_path, inst_dir, pe_dir, te_path, rank=128, down_rank=256, log=print):
    """Extracted factors → ``{te_key: tensor}`` for PE mode on the INT8 TE file:
    ``<module>.lora_up`` / ``.lora_down`` (smoothing applied), ``pe.<param key>``
    for each swapped tensor, and ``lm_head.weight``."""
    from analyze import Ckpt

    lora = safe_open(str(lora_path), "pt", device="cpu")
    inst = Ckpt(Path(inst_dir))
    pe_small = {k[len("small.") :]: k for k in lora.keys() if k.startswith("small.")}
    out = {}
    f = safe_open(str(te_path), "pt", device="cpu")
    n_layers = 1 + max(
        int(k.split(".")[2]) for k in f.keys() if k.startswith(TE + "layers.")
    )
    checks = []
    for i in range(n_layers):
        p_te, p_ck = f"{TE}layers.{i}.", f"{CKPT}layers.{i}."
        attn = (
            "self_attn"
            if f"{p_te}self_attn.q_proj.weight" in f.keys()
            else "linear_attn"
        )

        def s_norm(norm):
            w_inst = inst.get(f"{p_ck}{norm}.weight").float()
            w_file = f.get_tensor(f"{p_te}{norm}.weight").float()
            return (1 + w_inst) / (1 + w_file)

        s_attn, s_mlp = s_norm("input_layernorm"), s_norm("post_attention_layernorm")
        r_up = int8.rotate(inst.get(f"{p_ck}mlp.up_proj.weight").cuda().float())
        r_hat = _deq_rot(f, f"{p_te}mlp.up_proj.weight")
        s_down = (r_up.square().sum(1) / (r_up * r_hat).sum(1)).cpu()
        if i in (0, 3):
            checks.append(_check(f, inst, p_te, p_ck, attn, s_attn, s_mlp, s_down, log))
        del r_up, r_hat
        scale_in = {f"{attn}.{m}": s_attn for m in ATTN_IN[attn]}
        scale_in |= {
            "mlp.gate_proj": s_mlp,
            "mlp.up_proj": s_mlp,
            "mlp.down_proj": s_down,
        }
        mods = [f"{attn}.{m}" for m in ATTN_IN[attn]]
        mods += [f"{attn}.{'o_proj' if attn == 'self_attn' else 'out_proj'}"]
        mods += ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        for m in mods:
            r = down_rank if m == "mlp.down_proj" else rank
            up = lora.get_tensor(f"{p_ck}{m}.lora_up")[:, :r].float()
            down = lora.get_tensor(f"{p_ck}{m}.lora_down")[:r].float()
            if m in scale_in:
                down = down * scale_in[m][None, :]
            if m == "mlp.up_proj":
                up = up / s_down[:, None]
            out[f"{p_te}{m}.lora_up"] = up.bfloat16()
            out[f"{p_te}{m}.lora_down"] = down.bfloat16()
        for norm, s in (
            ("input_layernorm", s_attn),
            ("post_attention_layernorm", s_mlp),
        ):
            w_pe = lora.get_tensor(pe_small[f"{p_ck}{norm}.weight"]).float()
            out[f"pe.{p_te}{norm}.weight"] = (1 + w_pe) / s - 1
    for ck, sk in pe_small.items():
        te_key = TE + ck[len(CKPT) :]
        if f"pe.{te_key}" not in out:
            out[f"pe.{te_key}"] = lora.get_tensor(sk)
    out["lm_head.weight"] = Ckpt(Path(pe_dir)).get("lm_head.weight")
    log(f"[convert] {n_layers} layers, rank {rank} / down {down_rank}; checks {checks}")
    return out


def _check(f, inst, p_te, p_ck, attn, s_attn, s_mlp, s_down, log):
    """Relative error of the file's dequantized weight against the bf16 weight with
    the recovered smoothing applied (and, as a control, without it)."""
    res = {}
    for m, apply in (
        (f"{attn}.{ATTN_IN[attn][0]}", lambda w: w * s_attn.cuda()[None, :]),
        ("mlp.gate_proj", lambda w: w * s_mlp.cuda()[None, :]),
        ("mlp.down_proj", lambda w: w * s_down.cuda()[None, :]),
    ):
        w = inst.get(f"{p_ck}{m}.weight").cuda().float()
        w_hat = int8.rotate(_deq_rot(f, f"{p_te}{m}.weight"))
        ref = apply(w)
        res[m] = (
            round(float((w_hat - ref).norm() / ref.norm()), 4),
            round(float((w_hat - w).norm() / w.norm()), 4),
        )
    return res


class LoRABranch(torch.nn.Module):
    """``base(x) + on·(x Dᵀ) Uᵀ`` around a quantized linear; ``x`` is what the base
    sees (the smoothed input), which ``convert`` already folded into ``D``."""

    def __init__(self, base, up, down):
        super().__init__()
        self.base = base
        self.up = torch.nn.Parameter(up, requires_grad=False)
        self.down = torch.nn.Parameter(down, requires_grad=False)
        self.on = False

    def forward(self, x):
        y = self.base(x)
        if self.on:
            y = y + (x.to(self.down.dtype) @ self.down.T) @ self.up.T
        return y


class PEMode:
    """A MOSS TE (``Qwen3_5Model``) plus PE-mode tensors. ``causal`` is a
    ``Qwen3_5ForCausalLM`` sharing the TE's language model, for ``generate``."""

    def __init__(self, te, tensors: dict, device="cuda"):
        from accelerate import init_empty_weights
        from transformers import Qwen3_5ForCausalLM

        lm = te.language_model
        self.branches = []
        for key in [k for k in tensors if k.endswith(".lora_up")]:
            path = key[len(TE) : -len(".lora_up")]
            parent, name = path.rsplit(".", 1)
            owner = lm.get_submodule(parent)
            base = getattr(owner, name)
            branch = LoRABranch(
                base,
                tensors[key].to(device),
                tensors[key[: -len("up")] + "down"].to(device),
            )
            setattr(owner, name, branch)
            self.branches.append((owner, name, branch))
        self.swaps = []
        for key in [k for k in tensors if k.startswith("pe.")]:
            pname = key[len("pe.") + len(TE) :]
            param = lm.get_parameter(pname)
            te_val = param.data.clone()
            pe_val = tensors[key].to(param.device, param.dtype)
            self.swaps.append((param, te_val, pe_val))

        class _Causal(Qwen3_5ForCausalLM):
            # the first parameter is the CPU-resident embedding; generate() puts its
            # position / cache tensors on ``self.device``
            @property
            def device(self):
                return torch.device(device)

        with init_empty_weights():
            causal = _Causal(te.config.text_config)
        causal.model = lm
        head = tensors["lm_head.weight"]
        causal.lm_head = torch.nn.Linear(head.shape[1], head.shape[0], bias=False)
        causal.lm_head.weight = torch.nn.Parameter(head.to(device), requires_grad=False)
        self.causal = causal.eval()
        self.mode = "te"

    @torch.no_grad()
    def set_mode(self, mode: str):
        if mode not in ("te", "pe"):
            raise ValueError(mode)
        for _, _, b in self.branches:
            b.on = mode == "pe"
        for param, te_val, pe_val in self.swaps:
            param.data.copy_(pe_val if mode == "pe" else te_val)
        self.mode = mode

    def detach(self):
        """TE mode, then the original modules back in place."""
        self.set_mode("te")
        for owner, name, b in self.branches:
            setattr(owner, name, b.base)
        self.branches, self.swaps = [], []
