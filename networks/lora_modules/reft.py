# LoReFT: Low-Rank Representation Fine-Tuning.
# Wu et al., "ReFT: Representation Finetuning for Language Models" (NeurIPS 2024)

from typing import Optional

import torch


class ReFTModule(torch.nn.Module):
    """
    LoReFT: Low-Rank Representation Fine-Tuning.
    Applies a learned low-rank subspace edit to the output representation:
        h_new = h + R^T(ΔW·h + b) * scale * multiplier
    where R is an orthogonal rotation selecting the intervention subspace and
    ΔW (``learned_source``) is the learned delta within that subspace. The
    paper's form ``(Wh + b) − Rh`` is algebraically identical under
    ``ΔW = W − R``; parameterizing ΔW directly avoids the activation-level
    cancellation, so the module runs in the ambient dtype (bf16 under mixed
    precision) without fp32 upcasts.

    Intervention target: the paper defines ReFT on the residual stream at
    specific layers (Wu et al., 2024 §3.3). ``org_module`` here is usually a
    DiT Block whose output is the block-level residual-stream hidden state;
    wrapping Blocks (not each internal Linear) keeps the parameter and
    activation budget aligned with the paper.

    Zero-init: learned_source is zero-initialized so delta=0 at init.
    Reference: Wu et al., "ReFT: Representation Finetuning for Language Models" (NeurIPS 2024)
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        embed_dim: Optional[int] = None,
        multiplier=1.0,
        reft_dim=4,
        alpha=1,
        dropout=None,
        module_dropout=None,
    ):
        super().__init__()
        self.lora_name = lora_name

        if embed_dim is None:
            if hasattr(org_module, "out_features"):
                embed_dim = org_module.out_features
            else:
                raise ValueError(
                    "embed_dim must be provided when wrapping a non-Linear module "
                    f"(got {type(org_module).__name__})"
                )
        self.reft_dim = reft_dim

        # R: orthogonal rotation (projects to intervention subspace)
        self.rotate_layer = torch.nn.Linear(embed_dim, reft_dim, bias=False)
        init_device = "cuda" if torch.cuda.is_available() else "cpu"
        r_rand = torch.randn(embed_dim, reft_dim, device=init_device)
        r_orth, _ = torch.linalg.qr(r_rand)  # (embed_dim, reft_dim)
        self.rotate_layer.weight.data = r_orth.T.cpu().clone().contiguous()
        del r_rand, r_orth

        # ΔW: learned delta in R's subspace — zero-init gives delta=0 at step 0.
        self.learned_source = torch.nn.Linear(embed_dim, reft_dim)
        torch.nn.init.zeros_(self.learned_source.weight)
        torch.nn.init.zeros_(self.learned_source.bias)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = reft_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / reft_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.module_dropout = module_dropout

        self._timestep_mask = None

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, *args, **kwargs):
        # Works for wrapped Linear (forward(x)) and wrapped DiT Block
        # (forward(x_B_T_H_W_D, emb, crossattn, attn_params, rope, adaln_lora_3D)).
        h = self.org_forward(*args, **kwargs)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return h

        # ΔW·h + b in ambient dtype — no cancellation, no fp32 copy of h.
        # Last-dim linear broadcasts over any leading shape (B,L,D) or (B,T,H,W,D).
        delta = torch.nn.functional.linear(
            h, self.learned_source.weight, self.learned_source.bias
        )

        if self._timestep_mask is not None and self.training:
            delta = delta * self._timestep_mask

        if self.dropout is not None and self.training:
            delta = torch.nn.functional.dropout(delta, p=self.dropout)

        edit = torch.nn.functional.linear(delta, self.rotate_layer.weight.T)
        return h + edit * (self.multiplier * self.scale)

    def regularization(self):
        """Orthogonality regularization: ||R R^T - I||^2"""
        R = self.rotate_layer.weight  # (reft_dim, embed_dim)
        reg = torch.sum((R @ R.T - torch.eye(self.reft_dim, device=R.device)) ** 2)
        return reg
